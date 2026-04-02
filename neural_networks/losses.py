
"""Custom losses for training."""

from __future__ import annotations

import tensorflow as tf


def _split_prediction_and_uncertainty(y_pred: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
	channels = tf.shape(y_pred)[-1]
	if_channels = channels // 2
	return y_pred[..., :if_channels], y_pred[..., if_channels:]


def _split_components(
	tensor: tf.Tensor,
	n_frames: tf.Tensor,
	*,
	fit_im: bool,
	fit_psf: bool,
	fit_noise: bool,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
	start = 0
	image = tf.zeros_like(tensor[..., :1])
	psf = tf.zeros_like(tensor[..., :n_frames])
	noise = tf.zeros_like(tensor[..., :n_frames])
	if fit_im:
		image = tensor[..., start : start + 1]
		start += 1
	if fit_psf:
		psf = tensor[..., start : start + n_frames]
		start += n_frames
	if fit_noise:
		noise = tensor[..., start : start + n_frames]
	return image, psf, noise


def _gaussian_nll(truth: tf.Tensor, pred: tf.Tensor, log_sigma2: tf.Tensor) -> tf.Tensor:
	err2 = tf.square(pred - truth)
	return tf.reduce_mean(err2 * tf.exp(-log_sigma2) + log_sigma2)


def _soft_clip(x: tf.Tensor, lo: float, hi: float, softness: float = 0.5) -> tf.Tensor:
	"""Smoothly clamp values to [lo, hi] with non-zero gradients near bounds."""
	x = lo + softness * tf.nn.softplus((x - lo) / softness)
	x = hi - softness * tf.nn.softplus((hi - x) / softness)
	return x


def _log10_clip_to_ln(x: tf.Tensor, log10_min: float, log10_max: float) -> tf.Tensor:
	ln10 = tf.constant(2.302585092994046, dtype=x.dtype)
	lo = tf.cast(log10_min, x.dtype) * ln10
	hi = tf.cast(log10_max, x.dtype) * ln10
	return _soft_clip(x, lo, hi)


def _convolve_image_with_psfs(
	image: tf.Tensor,
	psf: tf.Tensor,
) -> tf.Tensor:
	"""Convolve image with PSF cube using FFT (channels_last)."""
	# image: (B, H, W, 1), psf: (B, H, W, F)
	psf_t = tf.transpose(psf, perm=(0, 3, 1, 2))
	image_t = tf.transpose(image, perm=(0, 3, 1, 2))
	psf_t = tf.signal.ifftshift(psf_t, axes=(-2, -1))

	image_fft = tf.signal.fft2d(tf.cast(image_t, tf.complex64))
	psf_fft = tf.signal.fft2d(tf.cast(psf_t, tf.complex64))
	conv = tf.signal.ifft2d(image_fft * psf_fft)
	conv = tf.math.real(conv)
	conv = tf.transpose(conv, perm=(0, 2, 3, 1))
	return conv


def make_loss(
	observation: tf.Tensor | None = None,
	*,
	loss: str = "nll",
	log_sigma: bool = False,
	log_min: float = -20.0,
	log_max: float = 10.0,
	sigma2_eps: float = 1e-6,
	charb_eps: float = 1e-3,
	half_n_pix_crop: int = 12,
	use_pinn: bool = True,
	fit_im: bool = True,
	fit_psf: bool = True,
	fit_noise: bool = True,
	norm_psf: str | float | None = "npix2",
	norm_noise: str | float | None = "npix2",
) -> tf.keras.losses.Loss:
	"""Create the composite loss.

	Parameters
	----------
	observation : tf.Tensor | None, optional
		Input observation with shape (B, H, W, n_frames). If None, reconstruct
		observation from y_true components using the same noise convention as
		the PINN term.
	log_min : float, optional
		Minimum soft-clip value for log10-variance. Default -20.
	log_max : float, optional
		Maximum soft-clip value for log10-variance. Default 10.
	sigma2_eps : float, optional
		Small value added to sigma^2 for numerical stability. Default 1e-6.
	charb_eps : float, optional
		Charbonnier epsilon for PINN term. Default 1e-3.
	half_n_pix_crop : int, optional
		Crop this many pixels from each side for PINN term. Default 12.
	use_pinn : bool, optional
		Whether to include the PINN/Charbonnier term. Default True.
	"""
	loss = loss.lower().strip()
	if loss not in {"nll", "r2"}:
		raise ValueError("loss must be one of: nll, r2")
	log_sigma = bool(log_sigma)
	fit_im = bool(fit_im)
	fit_psf = bool(fit_psf)
	fit_noise = bool(fit_noise)
	if not (fit_im or fit_psf or fit_noise):
		raise ValueError("At least one of fit_im/fit_psf/fit_noise must be True")
	if observation is not None and tf.keras.backend.is_keras_tensor(observation):
		raise ValueError(
			"`observation` cannot be a symbolic KerasTensor in Keras 3 losses. "
			"Pass None to build observation from y_true, or pass a regular Tensor."
		)

	def _loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
		components = loss_components(y_true, y_pred)
		if loss == "nll":
			base = tf.constant(0.0, dtype=y_pred.dtype)
			for key in ("nll_im", "nll_psf", "nll_noise"):
				if key in components:
					base = base + components[key]
		else:
			base = tf.constant(0.0, dtype=y_pred.dtype)
			for key in ("r2_im", "r2_psf", "r2_noise"):
				if key in components:
					base = base + components[key]
		if use_pinn:
			return base + components["pinn_charb"]
		return base

	def loss_components(y_true: tf.Tensor, y_pred: tf.Tensor) -> dict[str, tf.Tensor]:
		channels = tf.shape(y_true)[-1]
		base = tf.cast(fit_im, channels.dtype)
		denom = tf.cast(fit_psf, channels.dtype) + tf.cast(fit_noise, channels.dtype)
		n_frames = tf.where(
			denom > 0,
			(channels - base) // denom,
			tf.constant(0, dtype=channels.dtype),
		)

		pred, log_sigma2 = _split_prediction_and_uncertainty(y_pred)
		truth_im, truth_psf, truth_noise = _split_components(
			y_true,
			n_frames,
			fit_im=fit_im,
			fit_psf=fit_psf,
			fit_noise=fit_noise,
		)
		pred_im, pred_psf, pred_noise = _split_components(
			pred,
			n_frames,
			fit_im=fit_im,
			fit_psf=fit_psf,
			fit_noise=fit_noise,
		)
		log_sigma2_im, log_sigma2_psf, log_sigma2_noise = _split_components(
			log_sigma2,
			n_frames,
			fit_im=fit_im,
			fit_psf=fit_psf,
			fit_noise=fit_noise,
		)

		if log_sigma:
			log_sigma2_im = _log10_clip_to_ln(log_sigma2_im, log_min, log_max)
			log_sigma2_psf = _log10_clip_to_ln(log_sigma2_psf, log_min, log_max)
			log_sigma2_noise = _log10_clip_to_ln(log_sigma2_noise, log_min, log_max)
			sigma2_im = tf.exp(log_sigma2_im) + tf.cast(sigma2_eps, log_sigma2_im.dtype)
			sigma2_psf = tf.exp(log_sigma2_psf) + tf.cast(sigma2_eps, log_sigma2_psf.dtype)
			sigma2_noise = tf.exp(log_sigma2_noise) + tf.cast(sigma2_eps, log_sigma2_noise.dtype)
		else:
			sigma2_im = tf.nn.softplus(log_sigma2_im) + tf.cast(sigma2_eps, log_sigma2_im.dtype)
			sigma2_psf = tf.nn.softplus(log_sigma2_psf) + tf.cast(sigma2_eps, log_sigma2_psf.dtype)
			sigma2_noise = tf.nn.softplus(log_sigma2_noise) + tf.cast(sigma2_eps, log_sigma2_noise.dtype)
			log_sigma2_im = _log10_clip_to_ln(tf.math.log(sigma2_im), log_min, log_max)
			log_sigma2_psf = _log10_clip_to_ln(tf.math.log(sigma2_psf), log_min, log_max)
			log_sigma2_noise = _log10_clip_to_ln(tf.math.log(sigma2_noise), log_min, log_max)
			sigma2_im = tf.exp(log_sigma2_im) + tf.cast(sigma2_eps, log_sigma2_im.dtype)
			sigma2_psf = tf.exp(log_sigma2_psf) + tf.cast(sigma2_eps, log_sigma2_psf.dtype)
			sigma2_noise = tf.exp(log_sigma2_noise) + tf.cast(sigma2_eps, log_sigma2_noise.dtype)

		nll_im = tf.constant(0.0, dtype=pred_im.dtype)
		nll_psf = tf.constant(0.0, dtype=pred_im.dtype)
		nll_noise = tf.constant(0.0, dtype=pred_im.dtype)
		r2_im = tf.constant(0.0, dtype=pred_im.dtype)
		r2_psf = tf.constant(0.0, dtype=pred_im.dtype)
		r2_noise = tf.constant(0.0, dtype=pred_im.dtype)

		if loss == "nll":
			if fit_im:
				nll_im = _gaussian_nll(truth_im, pred_im, log_sigma2_im)
			if fit_psf:
				nll_psf = _gaussian_nll(truth_psf, pred_psf, log_sigma2_psf)
			if fit_noise:
				nll_noise = _gaussian_nll(truth_noise, pred_noise, log_sigma2_noise)
		else:
			def _std_ratio(truth: tf.Tensor, pred: tf.Tensor, sigma2: tf.Tensor) -> tf.Tensor:
				den = tf.math.reduce_variance(truth) + tf.cast(1e-12, truth.dtype)
				err = tf.reduce_mean(tf.square(truth - pred)) / den
				unc = tf.reduce_mean(tf.square(truth - sigma2)) / den
				return err + unc

			if fit_im:
				r2_im = _std_ratio(truth_im, pred_im, sigma2_im)
			if fit_psf:
				r2_psf = _std_ratio(truth_psf, pred_psf, sigma2_psf)
			if fit_noise:
				r2_noise = _std_ratio(truth_noise, pred_noise, sigma2_noise)

		pinn_charb = tf.constant(0.0, dtype=pred_im.dtype)
		if use_pinn and fit_im and fit_psf and fit_noise:
			# Compute denormalization factors
			if norm_psf is None:
				_psf_df = tf.constant(1.0, dtype=pred_psf.dtype)
			elif norm_psf == "npix2":
				_n = tf.cast(tf.shape(pred_psf)[1], pred_psf.dtype)
				_psf_df = _n * _n
			else:
				_psf_df = tf.constant(float(norm_psf), dtype=pred_psf.dtype)
			if norm_noise is None:
				_noise_df = tf.constant(1.0, dtype=pred_noise.dtype)
			elif norm_noise == "npix2":
				_n = tf.cast(tf.shape(pred_noise)[1], pred_noise.dtype)
				_noise_df = _n * _n
			else:
				_noise_df = tf.constant(float(norm_noise), dtype=pred_noise.dtype)

			pred_psf_d = pred_psf / _psf_df
			pred_noise_d = pred_noise / _noise_df
			psf_sum = tf.reduce_sum(pred_psf_d, axis=(1, 2), keepdims=True) + 1e-12
			pred_psf_norm = pred_psf_d / psf_sum
			conv_model = _convolve_image_with_psfs(pred_im, pred_psf_norm)
			# TFRecords store noise as: noise = conv(image, psf) - obs
			# so reconstruction must be: obs = conv(image, psf) - noise.
			pred_obs = conv_model - pred_noise_d
			if observation is None:
				truth_psf_d = truth_psf / _psf_df
				truth_noise_d = truth_noise / _noise_df
				truth_psf_sum = tf.reduce_sum(truth_psf_d, axis=(1, 2), keepdims=True) + 1e-12
				truth_psf_norm = truth_psf_d / truth_psf_sum
				conv_truth = _convolve_image_with_psfs(truth_im, truth_psf_norm)
				obs_used = conv_truth - truth_noise_d
			else:
				obs_used = observation
			if half_n_pix_crop > 0:
				c = int(half_n_pix_crop)
				obs_used = obs_used[:, c:-c, c:-c, :]
				pred_obs = pred_obs[:, c:-c, c:-c, :]
			r_obs = obs_used - pred_obs
			charb_eps_t = tf.cast(charb_eps, r_obs.dtype)
			pinn_charb = tf.reduce_mean(
				tf.sqrt(tf.square(r_obs) + tf.square(charb_eps_t))
			)

		components = {
			"nll_im": nll_im,
			"nll_psf": nll_psf,
			"nll_noise": nll_noise,
			"r2_im": r2_im,
			"r2_psf": r2_psf,
			"r2_noise": r2_noise,
			"pinn_charb": pinn_charb,
			"log_sigma2_im": log_sigma2_im,
			"log_sigma2_psf": log_sigma2_psf,
			"log_sigma2_noise": log_sigma2_noise,
		}

		if loss == "nll":
			result = {}
			if fit_im:
				result["nll_im"] = components["nll_im"]
				result["log_sigma2_im"] = components["log_sigma2_im"]
			if fit_psf:
				result["nll_psf"] = components["nll_psf"]
				result["log_sigma2_psf"] = components["log_sigma2_psf"]
			if fit_noise:
				result["nll_noise"] = components["nll_noise"]
				result["log_sigma2_noise"] = components["log_sigma2_noise"]
			result["pinn_charb"] = components["pinn_charb"]
			return result
		result = {}
		if fit_im:
			result["r2_im"] = components["r2_im"]
		if fit_psf:
			result["r2_psf"] = components["r2_psf"]
		if fit_noise:
			result["r2_noise"] = components["r2_noise"]
		result["pinn_charb"] = components["pinn_charb"]
		return result

	if loss == "nll":
		component_names = []
		if fit_im:
			component_names.extend(["nll_im", "log_sigma2_im"])
		if fit_psf:
			component_names.extend(["nll_psf", "log_sigma2_psf"])
		if fit_noise:
			component_names.extend(["nll_noise", "log_sigma2_noise"])
	else:
		component_names = []
		if fit_im:
			component_names.append("r2_im")
		if fit_psf:
			component_names.append("r2_psf")
		if fit_noise:
			component_names.append("r2_noise")
	_loss.component_names = component_names
	_loss.components = loss_components
	return _loss


def make_loss_components(
	observation: tf.Tensor | None = None,
	*,
	loss: str = "nll",
	log_sigma: bool = False,
	log_min: float = -10.0,
	log_max: float = 10.0,
	sigma2_eps: float = 1e-6,
	charb_eps: float = 1e-3,
	half_n_pix_crop: int = 12,
	use_pinn: bool = True,
	fit_im: bool = True,
	fit_psf: bool = True,
	fit_res: bool = True,
) -> callable:
	"""Return a callable that computes sub-loss components."""
	return make_loss(
		observation,
		loss=loss,
		log_sigma=log_sigma,
		log_min=log_min,
		log_max=log_max,
		sigma2_eps=sigma2_eps,
		charb_eps=charb_eps,
		half_n_pix_crop=half_n_pix_crop,
		use_pinn=use_pinn,
		fit_im=fit_im,
		fit_psf=fit_psf,
		fit_noise=fit_res,
	).components
