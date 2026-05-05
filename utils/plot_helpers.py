"""Plotting helper functions for visualizing predictions and residuals."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, PowerNorm
from matplotlib.patches import Rectangle
import numpy as np

from utils.metrics import _normalized_residual


def _save_figure_png_and_pdf(fig: plt.Figure, out_path: Path, **savefig_kwargs: Any) -> None:
	out_path = Path(out_path)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	base_path = out_path if out_path.suffix == "" else out_path.with_suffix("")
	for suffix in (".png", ".pdf"):
		fig.savefig(base_path.with_suffix(suffix), **savefig_kwargs)


def _power_norm(data_list: list[np.ndarray]) -> PowerNorm | Normalize:
	arr = np.concatenate([np.ravel(np.asarray(x, dtype=np.float64)) for x in data_list])
	finite = arr[np.isfinite(arr)]
	if finite.size == 0:
		return Normalize(vmin=0.0, vmax=1.0)
	vmin = float(np.min(finite))
	vmax = float(np.max(finite))
	if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
		vmax = vmin + 1e-6
	return PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)


def _log_norm_no_clip(data_list: list[np.ndarray]) -> LogNorm | Normalize:
	arr = np.concatenate([np.ravel(np.asarray(x, dtype=np.float64)) for x in data_list])
	finite_positive = arr[np.isfinite(arr) & (arr > 0.0)]
	if finite_positive.size == 0:
		return Normalize(vmin=0.0, vmax=1.0)
	vmax = float(np.max(finite_positive))
	vmin = vmax / 1e4
	if not np.isfinite(vmax) or vmax <= 0.0:
		return Normalize(vmin=0.0, vmax=1.0)
	if not np.isfinite(vmin) or vmin <= 0.0 or vmin == vmax:
		vmax = max(vmax, 1e-6)
		vmin = max(vmax / 1e4, 1e-12)
	return LogNorm(vmin=vmin, vmax=vmax)


def _clip_data_to_norm_range(data: np.ndarray, norm: Normalize) -> np.ndarray:
	array = np.asarray(data, dtype=np.float32)
	if not hasattr(norm, "vmin") or not hasattr(norm, "vmax"):
		return array
	vmin = getattr(norm, "vmin", None)
	vmax = getattr(norm, "vmax", None)
	if vmin is None or vmax is None:
		return array
	return np.clip(array, float(vmin), float(vmax))


def _linear_norm(data_list: list[np.ndarray], *, symmetric: bool = False) -> Normalize:
	arr = np.concatenate([np.ravel(np.asarray(x, dtype=np.float64)) for x in data_list])
	finite = arr[np.isfinite(arr)]
	if finite.size == 0:
		return Normalize(vmin=-1.0 if symmetric else 0.0, vmax=1.0)
	if symmetric:
		m = float(np.max(np.abs(finite)))
		if not np.isfinite(m) or m == 0.0:
			m = 1e-6
		return Normalize(vmin=-m, vmax=m)
	vmin = float(np.min(finite))
	vmax = float(np.max(finite))
	if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
		vmax = vmin + 1e-6
	return Normalize(vmin=vmin, vmax=vmax)


def _truth_power_norm(truth: np.ndarray) -> PowerNorm | Normalize:
	return _power_norm([truth])


def _truth_linear_norm(truth: np.ndarray) -> Normalize:
	return _linear_norm([truth], symmetric=False)


def _clipped_power_norm(data_list: list[np.ndarray]) -> PowerNorm | Normalize:
	return _power_norm(data_list)


def _clipped_linear_norm(data_list: list[np.ndarray], *, symmetric: bool = False) -> Normalize:
	return _linear_norm(data_list, symmetric=symmetric)


def _zero_outer_pixels(image: np.ndarray, n_pix_zero: int) -> np.ndarray:
	image = np.array(image, copy=True)
	if n_pix_zero <= 0 or image.ndim < 2:
		return image
	height, width = image.shape[:2]
	if 2 * n_pix_zero >= height or 2 * n_pix_zero >= width:
		return np.zeros_like(image)
	image[:n_pix_zero, ...] = 0
	image[-n_pix_zero:, ...] = 0
	image[:, :n_pix_zero, ...] = 0
	image[:, -n_pix_zero:, ...] = 0
	return image


def _crop_outer_pixels(image: np.ndarray, n_pix_crop: int) -> np.ndarray:
	image = np.asarray(image)
	if n_pix_crop <= 0 or image.ndim < 2:
		return image
	height, width = image.shape[:2]
	if 2 * n_pix_crop >= height or 2 * n_pix_crop >= width:
		return image[0:0, 0:0, ...]
	return image[n_pix_crop:-n_pix_crop, n_pix_crop:-n_pix_crop, ...]


def _add_crop_box(ax, image: np.ndarray, n_pix_crop: int, *, color: str = "0.5") -> None:
	image = np.asarray(image)
	if n_pix_crop <= 0 or image.ndim < 2:
		return
	height, width = image.shape[:2]
	box_width = width - 2 * n_pix_crop
	box_height = height - 2 * n_pix_crop
	if box_width <= 0 or box_height <= 0:
		return
	ax.add_patch(
		Rectangle(
			(n_pix_crop - 0.5, n_pix_crop - 0.5),
			box_width,
			box_height,
			fill=False,
			edgecolor=color,
			linestyle=":",
			linewidth=1.5,
		)
	)


def _plot_normalized_residual_histogram(ax, values: np.ndarray, title: str) -> None:
	values = np.asarray(values, dtype=np.float32)
	finite = values[np.isfinite(values)]
	if finite.size > 0:
		finite = np.clip(finite, -6.0, 6.0)
		ax.hist(finite, bins=100, range=(-6.0, 6.0), color="tab:blue", alpha=0.85)
	ax.set_xlim(-6.0, 6.0)
	ax.set_title(title)
	ax.set_xlabel("Normalized residual [$\\sigma$]")
	ax.set_ylabel("Count")
	ax.grid(True, alpha=0.25)


def _imshow(ax, image: np.ndarray, title: str, *, norm, cmap: str = "viridis"):
	im = ax.imshow(np.asarray(image), origin="lower", norm=norm, cmap=cmap)
	ax.set_title(title)
	ax.set_xticks([])
	ax.set_yticks([])
	return im


def _plot_truth_vs_prediction(
	*,
	obs_true: np.ndarray,
	im_true: np.ndarray,
	psf_true: np.ndarray,
	noise_true: np.ndarray,
	obs_pred: np.ndarray,
	im_pred: np.ndarray,
	psf_pred: np.ndarray,
	noise_pred: np.ndarray,
	sigma_obs: np.ndarray | None,
	sigma_im: np.ndarray | None,
	sigma_psf: np.ndarray | None,
	sigma_noise: np.ndarray | None,
	frame: int,
	obs_panel_n_pix_zero: int,
	out_path: Path,
	dpi: int,
) -> None:
	has_unc = any(x is not None for x in (sigma_obs, sigma_im, sigma_psf, sigma_noise))
	n_rows = 6 if has_unc else 3
	fig, axes = plt.subplots(n_rows, 4, figsize=(18, 4.2 * n_rows), squeeze=False)
	obs_norm = _truth_power_norm(obs_true)
	im_norm = _truth_power_norm(im_true)
	psf_norm = _truth_power_norm(psf_true)
	noise_norm = _truth_linear_norm(noise_true)
	obs_res_norm = _linear_norm([obs_true - obs_pred], symmetric=True)
	im_res_norm = _linear_norm([im_true - im_pred], symmetric=True)
	psf_res_norm = _linear_norm([psf_true - psf_pred], symmetric=True)
	noise_res_norm = _linear_norm([noise_true - noise_pred], symmetric=True)
	obs_norm_res = _normalized_residual(obs_true, obs_pred, sigma_obs)
	im_norm_res = _normalized_residual(im_true, im_pred, sigma_im)
	psf_norm_res = _normalized_residual(psf_true, psf_pred, sigma_psf)
	noise_norm_res = _normalized_residual(noise_true, noise_pred, sigma_noise)
	obs_residual_display = obs_true - obs_pred
	norm_res_fixed_norm = Normalize(vmin=-6.0, vmax=6.0)
	obs_unc_display = None if sigma_obs is None else sigma_obs
	obs_norm_res_display = obs_norm_res
	obs_norm_res_hist = obs_norm_res
	_imshow(axes[0, 0], obs_true, "True obs", norm=obs_norm)
	_imshow(axes[0, 1], im_true, "True im", norm=im_norm)
	_imshow(axes[0, 2], psf_true, f"True psf [{frame}]", norm=psf_norm)
	_imshow(axes[0, 3], noise_true, f"True res [{frame}]", norm=noise_norm)
	for col in range(4):
		fig.colorbar(axes[0, col].images[0], ax=axes[0, col], fraction=0.046, pad=0.04)
	_imshow(axes[1, 0], obs_pred, "Pred obs", norm=obs_norm)
	_add_crop_box(axes[1, 0], obs_pred, obs_panel_n_pix_zero)
	_imshow(axes[1, 1], im_pred, "Pred im", norm=im_norm)
	_imshow(axes[1, 2], psf_pred, f"Pred psf [{frame}]", norm=psf_norm)
	_imshow(axes[1, 3], noise_pred, f"Pred res [{frame}]", norm=noise_norm)
	for col in range(4):
		fig.colorbar(axes[1, col].images[0], ax=axes[1, col], fraction=0.046, pad=0.04)
	if has_unc:
		obs_unc = sigma_obs if sigma_obs is not None else np.zeros_like(obs_true)
		im_unc = sigma_im if sigma_im is not None else np.zeros_like(im_true)
		psf_unc = sigma_psf if sigma_psf is not None else np.zeros_like(psf_true)
		noise_unc = sigma_noise if sigma_noise is not None else np.zeros_like(noise_true)
		obs_unc_norm = _power_norm([obs_unc_display if obs_unc_display is not None else obs_unc])
		im_unc_norm = _power_norm([im_unc])
		psf_unc_norm = _power_norm([psf_unc])
		noise_unc_norm = _power_norm([noise_unc])
		_imshow(axes[2, 0], obs_unc_display if obs_unc_display is not None else obs_unc, "Pred unc obs", norm=obs_unc_norm)
		_add_crop_box(axes[2, 0], obs_unc_display if obs_unc_display is not None else obs_unc, obs_panel_n_pix_zero)
		_imshow(axes[2, 1], im_unc, "Pred unc im", norm=im_unc_norm)
		_imshow(axes[2, 2], psf_unc, f"Pred unc psf [{frame}]", norm=psf_unc_norm)
		_imshow(axes[2, 3], noise_unc, f"Pred unc res [{frame}]", norm=noise_unc_norm)
		for col in range(4):
			fig.colorbar(axes[2, col].images[0], ax=axes[2, col], fraction=0.046, pad=0.04)
		_imshow(axes[3, 0], obs_residual_display, "Obs residual", norm=obs_res_norm, cmap="coolwarm")
		_imshow(axes[3, 1], im_true - im_pred, "Im residual", norm=im_res_norm, cmap="coolwarm")
		_imshow(axes[3, 2], psf_true - psf_pred, "PSF residual", norm=psf_res_norm, cmap="coolwarm")
		_imshow(axes[3, 3], noise_true - noise_pred, "Res residual", norm=noise_res_norm, cmap="coolwarm")
		for col in range(4):
			fig.colorbar(axes[3, col].images[0], ax=axes[3, col], fraction=0.046, pad=0.04)
		_imshow(axes[4, 0], obs_norm_res_display, "Obs normalized residual", norm=norm_res_fixed_norm, cmap="coolwarm")
		_imshow(axes[4, 1], im_norm_res, "Im normalized residual", norm=norm_res_fixed_norm, cmap="coolwarm")
		_imshow(axes[4, 2], psf_norm_res, "PSF normalized residual", norm=norm_res_fixed_norm, cmap="coolwarm")
		_imshow(axes[4, 3], noise_norm_res, "Res normalized residual", norm=norm_res_fixed_norm, cmap="coolwarm")
		for col in range(4):
			fig.colorbar(axes[4, col].images[0], ax=axes[4, col], fraction=0.046, pad=0.04)
		_plot_normalized_residual_histogram(axes[5, 0], obs_norm_res_hist, "Obs norm-res histogram")
		_plot_normalized_residual_histogram(axes[5, 1], im_norm_res, "Im norm-res histogram")
		_plot_normalized_residual_histogram(axes[5, 2], psf_norm_res, "PSF norm-res histogram")
		_plot_normalized_residual_histogram(axes[5, 3], noise_norm_res, "Res norm-res histogram")
	else:
		_imshow(axes[2, 0], obs_residual_display, "Obs residual", norm=obs_res_norm, cmap="coolwarm")
		_imshow(axes[2, 1], im_true - im_pred, "Im residual", norm=im_res_norm, cmap="coolwarm")
		_imshow(axes[2, 2], psf_true - psf_pred, "PSF residual", norm=psf_res_norm, cmap="coolwarm")
		_imshow(axes[2, 3], noise_true - noise_pred, "Res residual", norm=noise_res_norm, cmap="coolwarm")
		for col in range(4):
			fig.colorbar(axes[2, col].images[0], ax=axes[2, col], fraction=0.046, pad=0.04)
	_save_figure_png_and_pdf(fig, out_path, dpi=dpi)
	plt.close(fig)


def _plot_truth_vs_recovered(
	*,
	obs_true: np.ndarray,
	im_true: np.ndarray,
	psf_true: np.ndarray,
	noise_true: np.ndarray,
	obs_rec: np.ndarray,
	im_rec: np.ndarray,
	psf_rec: np.ndarray,
	noise_rec: np.ndarray,
	frame: int,
	out_path: Path,
	dpi: int,
) -> None:
	fig, axes = plt.subplots(3, 4, figsize=(18, 12.5), squeeze=False)
	obs_norm = _truth_power_norm(obs_true)
	im_norm = _truth_power_norm(im_true)
	psf_norm = _truth_power_norm(psf_true)
	noise_norm = _truth_linear_norm(noise_true)
	obs_res_norm = _linear_norm([obs_true - obs_rec], symmetric=True)
	im_res_norm = _linear_norm([im_true - im_rec], symmetric=True)
	psf_res_norm = _linear_norm([psf_true - psf_rec], symmetric=True)
	noise_res_norm = _linear_norm([noise_true - noise_rec], symmetric=True)
	_imshow(axes[0, 0], obs_true, "True obs", norm=obs_norm)
	_imshow(axes[0, 1], im_true, "True im", norm=im_norm)
	_imshow(axes[0, 2], psf_true, f"True psf [{frame}]", norm=psf_norm)
	_imshow(axes[0, 3], noise_true, f"True res [{frame}]", norm=noise_norm)
	for col in range(4):
		fig.colorbar(axes[0, col].images[0], ax=axes[0, col], fraction=0.046, pad=0.04)
	_imshow(axes[1, 0], obs_rec, "Recovered obs", norm=obs_norm)
	_imshow(axes[1, 1], im_rec, "Recovered im", norm=im_norm)
	_imshow(axes[1, 2], psf_rec, f"Recovered psf [{frame}]", norm=psf_norm)
	_imshow(axes[1, 3], noise_rec, f"Recovered res [{frame}]", norm=noise_norm)
	for col in range(4):
		fig.colorbar(axes[1, col].images[0], ax=axes[1, col], fraction=0.046, pad=0.04)
	_imshow(axes[2, 0], obs_true - obs_rec, "Obs residual", norm=obs_res_norm, cmap="coolwarm")
	_imshow(axes[2, 1], im_true - im_rec, "Im residual", norm=im_res_norm, cmap="coolwarm")
	_imshow(axes[2, 2], psf_true - psf_rec, "PSF residual", norm=psf_res_norm, cmap="coolwarm")
	_imshow(axes[2, 3], noise_true - noise_rec, "Res residual", norm=noise_res_norm, cmap="coolwarm")
	for col in range(4):
		fig.colorbar(axes[2, col].images[0], ax=axes[2, col], fraction=0.046, pad=0.04)
	_save_figure_png_and_pdf(fig, out_path, dpi=dpi)
	plt.close(fig)


def _plot_inference_example(
	*,
	obs_true: np.ndarray,
	pred_im: np.ndarray,
	pred_psf: np.ndarray,
	pred_noise: np.ndarray,
	pred_obs: np.ndarray,
	sigma_obs: np.ndarray | None,
	sigma_im: np.ndarray | None,
	sigma_psf: np.ndarray | None,
	sigma_noise: np.ndarray | None,
	rec_im: np.ndarray,
	rec_psf: np.ndarray,
	rec_noise: np.ndarray,
	frame: int | str,
	out_path: Path,
	dpi: int,
) -> None:
	_ = rec_im
	_ = rec_psf
	_ = rec_noise
	fig, axes = plt.subplots(4, 3, figsize=(13.5, 16.5), squeeze=False)
	obs_norm = _truth_power_norm(obs_true)
	im_norm = _log_norm_no_clip([pred_im])
	pred_im_display = _clip_data_to_norm_range(pred_im, im_norm)
	psf_norm = _clipped_power_norm([pred_psf])
	noise_norm = _clipped_linear_norm([pred_noise], symmetric=False)
	obs_unc_norm = _clipped_power_norm([np.zeros_like(obs_true) if sigma_obs is None else sigma_obs])
	im_unc_norm = _clipped_power_norm([np.zeros_like(pred_im) if sigma_im is None else sigma_im])
	psf_unc_norm = _clipped_power_norm([np.zeros_like(pred_psf) if sigma_psf is None else sigma_psf])
	noise_unc_norm = _clipped_power_norm([np.zeros_like(pred_noise) if sigma_noise is None else sigma_noise])
	obs_true_arr = np.asarray(obs_true, dtype=np.float32)
	pred_obs_arr = np.asarray(pred_obs, dtype=np.float32)
	obs_residual = obs_true_arr - pred_obs_arr
	obs_residual_norm = _linear_norm([obs_residual], symmetric=True)
	obs_norm_res = _normalized_residual(obs_true_arr, pred_obs_arr, sigma_obs)
	norm_res_fixed_norm = Normalize(vmin=-6.0, vmax=6.0)
	_imshow(axes[0, 0], obs_true, "True obs", norm=obs_norm)
	_imshow(axes[0, 1], pred_obs, "Reconstructed obs", norm=obs_norm)
	_imshow(
		axes[0, 2],
		np.zeros_like(obs_true) if sigma_obs is None else sigma_obs,
		"Pred obs unc",
		norm=obs_unc_norm,
	)
	for col in range(3):
		fig.colorbar(axes[0, col].images[0], ax=axes[0, col], fraction=0.046, pad=0.04)
	_imshow(axes[1, 0], obs_residual, "Obs residual", norm=obs_residual_norm, cmap="coolwarm")
	_imshow(axes[1, 1], obs_norm_res, "Obs normalized residual", norm=norm_res_fixed_norm, cmap="coolwarm")
	_plot_normalized_residual_histogram(axes[1, 2], obs_norm_res, "Obs norm-res histogram")
	for col in range(3):
		if col < 2:
			fig.colorbar(axes[1, col].images[0], ax=axes[1, col], fraction=0.046, pad=0.04)
	_imshow(axes[2, 0], pred_im_display, "Pred im", norm=im_norm)
	_imshow(axes[2, 1], pred_psf, f"Pred psf [{frame}]", norm=psf_norm)
	_imshow(axes[2, 2], pred_noise, f"Pred res [{frame}]", norm=noise_norm)
	for col in range(3):
		fig.colorbar(axes[2, col].images[0], ax=axes[2, col], fraction=0.046, pad=0.04)
	_imshow(
		axes[3, 0],
		np.zeros_like(pred_im) if sigma_im is None else sigma_im,
		"Pred im unc",
		norm=im_unc_norm,
	)
	_imshow(
		axes[3, 1],
		np.zeros_like(pred_psf) if sigma_psf is None else sigma_psf,
		f"Pred psf unc [{frame}]",
		norm=psf_unc_norm,
	)
	_imshow(
		axes[3, 2],
		np.zeros_like(pred_noise) if sigma_noise is None else sigma_noise,
		f"Pred res unc [{frame}]",
		norm=noise_unc_norm,
	)
	for col in range(3):
		fig.colorbar(axes[3, col].images[0], ax=axes[3, col], fraction=0.046, pad=0.04)
	fig.tight_layout()
	_save_figure_png_and_pdf(fig, out_path, dpi=dpi)
	plt.close(fig)
