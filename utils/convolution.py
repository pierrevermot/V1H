"""Convolution, deconvolution, and PSF reconstruction utilities."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.signal import convolve, correlate, fftconvolve
import tensorflow as tf


def _convolve_image_with_psfs_numpy(ao_instru, image, psf_cube):
	"""Convolve a single image with a PSF cube (per-frame).

	Uses ao_instru.xp attribute (CuPy or NumPy).
	"""
	xp = getattr(ao_instru, "xp", np)
	image_xp = xp.asarray(image)
	psf_xp = xp.asarray(psf_cube)

	if psf_xp.ndim != 3:
		raise ValueError("psf_cube must have shape (n_psf, n_pix, n_pix)")
	if image_xp.shape != psf_xp.shape[1:]:
		raise ValueError("image shape must match psf spatial shape")

	image_fft = xp.fft.fft2(image_xp)
	obs = xp.zeros_like(psf_xp)
	for idx in range(psf_xp.shape[0]):
		psf_shift = xp.fft.ifftshift(psf_xp[idx])
		psf_fft = xp.fft.fft2(psf_shift)
		conv = xp.fft.ifft2(image_fft * psf_fft)
		obs[idx] = xp.real(conv)
	return obs


def _convolve_image_with_psf_cube(image_hw: np.ndarray, psf_hwf: np.ndarray) -> np.ndarray:
	if image_hw.ndim != 2 or psf_hwf.ndim != 3:
		raise ValueError(f"Bad shapes image={image_hw.shape}, psf={psf_hwf.shape}")
	image_fft = np.fft.fft2(np.asarray(image_hw, dtype=np.float32))
	psf_shift = np.fft.ifftshift(np.asarray(psf_hwf, dtype=np.float32), axes=(0, 1))
	psf_fft = np.fft.fft2(psf_shift, axes=(0, 1))
	conv = np.fft.ifft2(image_fft[..., np.newaxis] * psf_fft, axes=(0, 1))
	return np.real(conv).astype(np.float32)


def _convolve_image_with_psf_cube_zero_padded_backend(image_hw: np.ndarray, psf_hwf: np.ndarray, *, backend: str) -> np.ndarray:
	backend_key = str(backend).strip().lower()
	if image_hw.ndim != 2 or psf_hwf.ndim != 3:
		raise ValueError(f"Bad shapes image={image_hw.shape}, psf={psf_hwf.shape}")
	image_hw = np.asarray(image_hw, dtype=np.float32)
	psf_hwf = np.asarray(psf_hwf, dtype=np.float32)
	def _convolve_same_2d(img: np.ndarray, ker: np.ndarray) -> np.ndarray:
		if backend_key == "fft":
			return fftconvolve(img, ker, mode="same")
		if backend_key == "direct":
			return convolve(img, ker, mode="same", method="direct")
		raise ValueError(f"Unknown convolution backend: {backend}")
	return np.stack([_convolve_same_2d(image_hw, psf_hwf[:, :, frame_idx]).astype(np.float32) for frame_idx in range(psf_hwf.shape[-1])], axis=-1)


def _deconvolve_obs_with_psfs(obs_hwf: np.ndarray, psf_hwf: np.ndarray, *, eps: float) -> np.ndarray:
	obs_t = np.transpose(np.asarray(obs_hwf, dtype=np.float32)[np.newaxis, ...], (0, 3, 1, 2))
	psf_t = np.transpose(np.asarray(psf_hwf, dtype=np.float32)[np.newaxis, ...], (0, 3, 1, 2))
	psf_t = np.fft.ifftshift(psf_t, axes=(-2, -1))
	obs_fft = np.fft.fft2(obs_t, axes=(-2, -1))
	psf_fft = np.fft.fft2(psf_t, axes=(-2, -1))
	den = np.conj(psf_fft) * psf_fft
	img_fft = obs_fft * (np.conj(psf_fft) / (den + float(eps)))
	img = np.fft.ifft2(img_fft, axes=(-2, -1))
	img = np.real(img)
	return np.transpose(img[0], (1, 2, 0)).astype(np.float32)


def _deconvolve_obs_with_image(obs_hwf: np.ndarray, image_hw1: np.ndarray, *, eps: float, noise_std: float | None = None) -> np.ndarray:
	obs_t = np.transpose(np.asarray(obs_hwf, dtype=np.float32)[np.newaxis, ...], (0, 3, 1, 2))
	img_t = np.transpose(np.asarray(image_hw1, dtype=np.float32)[np.newaxis, ...], (0, 3, 1, 2))
	obs_fft = np.fft.fft2(obs_t, axes=(-2, -1))
	img_fft = np.fft.fft2(img_t, axes=(-2, -1))
	den = np.conj(img_fft) * img_fft
	noise_var = float(eps) if noise_std is None else max(float(noise_std) ** 2, float(eps))
	psf_fft = obs_fft * (np.conj(img_fft) / (den + noise_var))
	psf = np.fft.ifft2(psf_fft, axes=(-2, -1))
	psf = np.real(psf)
	psf = np.fft.fftshift(psf, axes=(-2, -1))
	return np.transpose(psf[0], (1, 2, 0)).astype(np.float32)


def _convolve_image_with_psfs_tf(image: tf.Tensor, psf: tf.Tensor) -> tf.Tensor:
	psf_t = tf.transpose(psf, perm=(0, 3, 1, 2))
	image_t = tf.transpose(image, perm=(0, 3, 1, 2))
	psf_t = tf.signal.ifftshift(psf_t, axes=(-2, -1))
	image_fft = tf.signal.fft2d(tf.cast(image_t, tf.complex64))
	psf_fft = tf.signal.fft2d(tf.cast(psf_t, tf.complex64))
	conv = tf.signal.ifft2d(image_fft * psf_fft)
	conv = tf.math.real(conv)
	return tf.transpose(conv, perm=(0, 2, 3, 1))


def _estimate_psf_gradient_descent(
	observation_hw: np.ndarray,
	deconvolved_image_hw: np.ndarray,
	noise_hw: np.ndarray,
	psf_init_hw: np.ndarray,
	*,
	sigma_im_hw: np.ndarray | None = None,
	sigma_noise_hw: np.ndarray | None = None,
	optimizer: str = "adam",
	convolution_backend: str = "direct",
	data_loss: str = "relative_l1",
	relative_loss_eps: float = 1e-3,
	variance_eps: float = 1e-12,
	maxiter: int = 2000,
	n_crop_pix: int = 16,
	compactness_weight: float = 2e-3,
	l2_weight: float = 1e-2,
	tv_weight: float = 5e-3,
	verbose: bool = False,
	print_prefix: str = "",
) -> tuple[np.ndarray, Any]:
	observation_hw = np.asarray(observation_hw, dtype=np.float64)
	deconvolved_image_hw = np.asarray(deconvolved_image_hw, dtype=np.float64)
	noise_hw = np.asarray(noise_hw, dtype=np.float64)
	psf_init_hw = np.asarray(psf_init_hw, dtype=np.float64)
	sigma_im_hw = None if sigma_im_hw is None else np.asarray(sigma_im_hw, dtype=np.float64)
	sigma_noise_hw = None if sigma_noise_hw is None else np.asarray(sigma_noise_hw, dtype=np.float64)
	psf0 = np.clip(psf_init_hw, 0.0, None)
	psf0_sum = float(np.sum(psf0))
	if not np.isfinite(psf0_sum) or psf0_sum <= 0.0:
		psf0 = np.ones_like(psf0, dtype=np.float64)
		psf0_sum = float(psf0.size)
	psf0 = psf0 / psf0_sum
	target_hw = observation_hw - noise_hw
	shape = observation_hw.shape
	loss_mask = np.zeros(shape, dtype=np.float64)
	if n_crop_pix == 0:
		loss_mask[...] = 1.0
	else:
		loss_mask[n_crop_pix:-n_crop_pix, n_crop_pix:-n_crop_pix] = 1.0
	mask_weight = float(np.sum(loss_mask))
	grid_y, grid_x = np.indices(shape, dtype=np.float64)
	center_y = 0.5 * (shape[0] - 1)
	center_x = 0.5 * (shape[1] - 1)
	radius2_hw = (grid_y - center_y) ** 2 + (grid_x - center_x) ** 2
	max_radius2 = float(np.max(radius2_hw))
	if max_radius2 > 0.0:
		radius2_hw = radius2_hw / max_radius2
	sigma_im2_hw = None if sigma_im_hw is None else np.square(sigma_im_hw)
	sigma_noise2_hw = None if sigma_noise_hw is None else np.square(sigma_noise_hw)
	data_loss_key = str(data_loss).strip().lower()

	def _softmax(flat_z: np.ndarray) -> np.ndarray:
		shifted = flat_z - np.max(flat_z)
		exp_z = np.exp(shifted)
		return exp_z / np.sum(exp_z)

	def _crop_full_correlation_to_kernel(full_corr_hw: np.ndarray) -> np.ndarray:
		starts = tuple(size // 2 for size in deconvolved_image_hw.shape)
		slices = tuple(slice(starts[axis], starts[axis] + shape[axis]) for axis in range(len(shape)))
		return full_corr_hw[slices]

	def _tv_value_and_grad(psf_hw: np.ndarray) -> tuple[float, np.ndarray]:
		tv_eps = 1e-8
		dx = np.zeros_like(psf_hw)
		dy = np.zeros_like(psf_hw)
		dx[:, :-1] = psf_hw[:, 1:] - psf_hw[:, :-1]
		dy[:-1, :] = psf_hw[1:, :] - psf_hw[:-1, :]
		denom = np.sqrt(np.square(dx) + np.square(dy) + tv_eps)
		tv_value = float(np.sum(denom))
		px = dx / denom
		py = dy / denom
		grad = np.zeros_like(psf_hw)
		grad[:, :-1] -= px[:, :-1]
		grad[:, 1:] += px[:, :-1]
		grad[:-1, :] -= py[:-1, :]
		grad[1:, :] += py[:-1, :]
		return tv_value, grad

	def _convolve_same_2d(img: np.ndarray, ker: np.ndarray) -> np.ndarray:
		backend_key = str(convolution_backend).strip().lower()
		if backend_key == "fft":
			return fftconvolve(img, ker, mode="same")
		if backend_key == "direct":
			return convolve(img, ker, mode="same", method="direct")
		raise ValueError(f"Unknown convolution backend: {convolution_backend}")

	def _correlate_full_2d(lhs_hw: np.ndarray, rhs_hw: np.ndarray) -> np.ndarray:
		backend_key = str(convolution_backend).strip().lower()
		if backend_key == "fft":
			return correlate(lhs_hw, rhs_hw, mode="full", method="fft")
		if backend_key == "direct":
			return correlate(lhs_hw, rhs_hw, mode="full", method="direct")
		raise ValueError(f"Unknown convolution backend: {convolution_backend}")

	def _objective_and_grad(flat_z: np.ndarray) -> tuple[float, np.ndarray]:
		psf_flat = _softmax(flat_z)
		psf_hw = psf_flat.reshape(shape)
		pred_hw = _convolve_same_2d(deconvolved_image_hw, psf_hw)
		raw_residual_hw = pred_hw - target_hw
		if data_loss_key == "squared":
			data_term_grad_hw = raw_residual_hw * loss_mask
			data_loss_value = 0.5 * float(np.sum(np.square(data_term_grad_hw)))
			data_var_grad_hw = None
		elif data_loss_key == "gaussian_nll":
			sigma_eff2_hw = np.zeros_like(raw_residual_hw)
			if sigma_noise2_hw is not None:
				sigma_eff2_hw = sigma_eff2_hw + sigma_noise2_hw
			if sigma_im2_hw is not None:
				sigma_eff2_hw = sigma_eff2_hw + _convolve_same_2d(sigma_im2_hw, np.square(psf_hw))
			sigma_eff2_hw = np.maximum(sigma_eff2_hw, float(variance_eps))
			weighted_mask_hw = loss_mask / sigma_eff2_hw
			data_term_grad_hw = raw_residual_hw * weighted_mask_hw
			data_var_grad_hw = 0.5 * loss_mask * ((1.0 / sigma_eff2_hw) - (np.square(raw_residual_hw) / np.square(sigma_eff2_hw)))
			data_loss_value = 0.5 * float(np.sum(loss_mask * ((np.square(raw_residual_hw) / sigma_eff2_hw) + np.log(sigma_eff2_hw))))
		else:
			denom_hw = np.abs(target_hw) + float(relative_loss_eps)
			data_loss_value = float(np.sum((np.abs(raw_residual_hw) / denom_hw) * loss_mask) / mask_weight)
			data_term_grad_hw = (np.sign(raw_residual_hw) / denom_hw) * (loss_mask / mask_weight)
			data_var_grad_hw = None
		compactness_loss = float(compactness_weight * np.sum(psf_hw * radius2_hw))
		l2_loss = 0.5 * float(l2_weight * np.sum(np.square(psf_hw)))
		if tv_weight > 0.0:
			tv_value, tv_grad_hw = _tv_value_and_grad(psf_hw)
			tv_loss = float(tv_weight * tv_value)
		else:
			tv_grad_hw = np.zeros_like(psf_hw)
			tv_loss = 0.0
		loss = data_loss_value + compactness_loss + l2_loss + tv_loss
		grad_psf_hw = _crop_full_correlation_to_kernel(_correlate_full_2d(data_term_grad_hw, deconvolved_image_hw))
		if data_var_grad_hw is not None and sigma_im2_hw is not None:
			grad_sigma2_hw = _crop_full_correlation_to_kernel(_correlate_full_2d(data_var_grad_hw, sigma_im2_hw))
			grad_psf_hw = grad_psf_hw + (2.0 * psf_hw * grad_sigma2_hw)
		if compactness_weight > 0.0:
			grad_psf_hw = grad_psf_hw + compactness_weight * radius2_hw
		if l2_weight > 0.0:
			grad_psf_hw = grad_psf_hw + l2_weight * psf_hw
		if tv_weight > 0.0:
			grad_psf_hw = grad_psf_hw + tv_weight * tv_grad_hw
		grad_psf_flat = grad_psf_hw.ravel()
		dot = float(np.dot(grad_psf_flat, psf_flat))
		grad_z = psf_flat * (grad_psf_flat - dot)
		return loss, grad_z

	flat_z0 = np.log(np.clip(psf0, 1e-12, None)).ravel()
	if str(optimizer).strip().lower() == "lbfgs":
		result = minimize(fun=lambda z: _objective_and_grad(z)[0], x0=flat_z0, jac=lambda z: _objective_and_grad(z)[1], method="L-BFGS-B", options={"maxiter": int(maxiter), "disp": bool(verbose)})
	else:
		flat_z = flat_z0.copy()
		first_moment = np.zeros_like(flat_z)
		second_moment = np.zeros_like(flat_z)
		learning_rate = 5e-2
		beta1 = 0.9
		beta2 = 0.999
		epsilon_adam = 1e-8
		best_x = flat_z.copy()
		best_loss = np.inf
		last_loss = np.inf
		status = 1
		message = f"STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT ({int(maxiter)})"
		for iter_idx in range(1, int(maxiter) + 1):
			loss, grad = _objective_and_grad(flat_z)
			grad_norm = float(np.linalg.norm(grad))
			rel_loss_change = abs(last_loss - float(loss)) / max(abs(last_loss), 1.0) if np.isfinite(last_loss) else np.inf
			if loss < best_loss:
				best_loss = float(loss)
				best_x = flat_z.copy()
			if verbose and (iter_idx == 1 or iter_idx % 10 == 0 or iter_idx == int(maxiter)):
				print(f"{print_prefix}iter={iter_idx:04d} loss={loss:.6e} grad_norm={grad_norm:.6e}")
			if grad_norm <= 1e-6:
				status = 0
				message = "CONVERGENCE: GRADIENT NORM <= GTOL"
				break
			if rel_loss_change <= 1e-10:
				status = 0
				message = "CONVERGENCE: RELATIVE LOSS CHANGE <= FTOL"
				break
			first_moment = beta1 * first_moment + (1.0 - beta1) * grad
			second_moment = beta2 * second_moment + (1.0 - beta2) * np.square(grad)
			first_unbias = first_moment / (1.0 - beta1 ** iter_idx)
			second_unbias = second_moment / (1.0 - beta2 ** iter_idx)
			flat_z = flat_z - learning_rate * first_unbias / (np.sqrt(second_unbias) + epsilon_adam)
			last_loss = float(loss)
		result = OptimizeResult(x=best_x, success=(status == 0), status=status, message=message, nit=iter_idx, fun=best_loss, jac=_objective_and_grad(best_x)[1])
	psf_hat = np.exp(result.x - np.max(result.x))
	psf_hat = (psf_hat / np.sum(psf_hat)).reshape(shape)
	return psf_hat.astype(np.float32), result


def _compute_recovered_quantities(
	obs_hwf: np.ndarray,
	pred_im_hw1: np.ndarray,
	pred_psf_phys_hwf: np.ndarray,
	pred_noise_phys_hwf: np.ndarray,
	pred_sigma_im_hw1: np.ndarray | None = None,
	pred_sigma_noise_phys_hwf: np.ndarray | None = None,
	*,
	eps: float,
	psf_reconstruction_method: str = "wiener",
	psf_reconstruction_optimizer: str = "adam",
	psf_reconstruction_convolution_backend: str = "direct",
	psf_reconstruction_data_loss: str = "relative_l1",
	psf_reconstruction_relative_loss_eps: float = 1e-3,
	psf_reconstruction_variance_eps: float = 1e-12,
	psf_reconstruction_maxiter: int = 200,
	psf_reconstruction_n_crop_pix: int = 16,
	psf_reconstruction_compactness_weight: float = 2e-3,
	psf_reconstruction_l2_weight: float = 1e-2,
	psf_reconstruction_tv_weight: float = 5e-3,
	psf_reconstruction_verbose: bool = False,
) -> dict[str, np.ndarray]:
	recovered_im = np.empty_like(pred_im_hw1, dtype=np.float32)
	recovered_psf = np.empty_like(pred_psf_phys_hwf, dtype=np.float32)
	recovered_noise = np.empty_like(pred_noise_phys_hwf, dtype=np.float32)
	recovered_obs = np.empty_like(obs_hwf, dtype=np.float32)
	for idx in range(obs_hwf.shape[0]):
		obs_plus_noise = obs_hwf[idx] + pred_noise_phys_hwf[idx]
		im_per_frame = _deconvolve_obs_with_psfs(obs_plus_noise, pred_psf_phys_hwf[idx], eps=eps)
		im_recovered = np.mean(im_per_frame, axis=-1, keepdims=True).astype(np.float32)
		if psf_reconstruction_method == "optimize":
			psf_recovered = np.empty_like(pred_psf_phys_hwf[idx], dtype=np.float32)
			for frame_idx in range(obs_hwf.shape[-1]):
				psf_frame, _ = _estimate_psf_gradient_descent(
					obs_hwf[idx, :, :, frame_idx],
					pred_im_hw1[idx, :, :, 0],
					-pred_noise_phys_hwf[idx, :, :, frame_idx],
					pred_psf_phys_hwf[idx, :, :, frame_idx],
					sigma_im_hw=None if pred_sigma_im_hw1 is None else pred_sigma_im_hw1[idx, :, :, 0],
					sigma_noise_hw=None if pred_sigma_noise_phys_hwf is None else pred_sigma_noise_phys_hwf[idx, :, :, frame_idx],
					optimizer=psf_reconstruction_optimizer,
					convolution_backend=psf_reconstruction_convolution_backend,
					data_loss=psf_reconstruction_data_loss,
					relative_loss_eps=psf_reconstruction_relative_loss_eps,
					variance_eps=psf_reconstruction_variance_eps,
					maxiter=psf_reconstruction_maxiter,
					n_crop_pix=psf_reconstruction_n_crop_pix,
					compactness_weight=psf_reconstruction_compactness_weight,
					l2_weight=psf_reconstruction_l2_weight,
					tv_weight=psf_reconstruction_tv_weight,
					verbose=psf_reconstruction_verbose,
					print_prefix=f"[reconstruct_psf ex={idx} frame={frame_idx}] ",
				)
				psf_recovered[:, :, frame_idx] = psf_frame
		else:
			noise_std = float(np.std(pred_noise_phys_hwf[idx], dtype=np.float64))
			psf_recovered = _deconvolve_obs_with_image(obs_plus_noise, pred_im_hw1[idx], eps=eps, noise_std=noise_std)
		noise_recovered = obs_hwf[idx] - _convolve_image_with_psf_cube(pred_im_hw1[idx, :, :, 0], pred_psf_phys_hwf[idx])
		if psf_reconstruction_method == "optimize":
			obs_recovered = _convolve_image_with_psf_cube_zero_padded_backend(im_recovered[:, :, 0], psf_recovered, backend=psf_reconstruction_convolution_backend) - noise_recovered
		else:
			obs_recovered = _convolve_image_with_psf_cube(im_recovered[:, :, 0], psf_recovered) - noise_recovered
		recovered_im[idx] = im_recovered
		recovered_psf[idx] = psf_recovered
		recovered_noise[idx] = noise_recovered.astype(np.float32)
		recovered_obs[idx] = obs_recovered.astype(np.float32)
	return {"recovered_im": recovered_im, "recovered_psf": recovered_psf, "recovered_noise": recovered_noise, "recovered_obs": recovered_obs}
