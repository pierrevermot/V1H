#!/usr/bin/env python3
"""Plot predictions from a joint four-head PINN run.

This script is self-contained and mirrors the figure layout used for
independent-head plotting, but for saved four-head joint PINN runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")

from matplotlib.colors import Normalize
import numpy as np
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from utils.convolution import (
	_convolve_image_with_psf_cube,
	_convolve_image_with_psf_cube_zero_padded_backend,
	_convolve_image_with_psfs_tf as _convolve_image_with_psfs,
	_compute_recovered_quantities,
	_deconvolve_obs_with_image,
	_deconvolve_obs_with_psfs,
)
from utils.data_utils import _crop_data_to_model, _prepare_truth_arrays
from utils.io import (
	_load_joint_run_config,
	_load_snapshot_config,
	_write_fits_image,
)
from utils.metrics import _pred_to_sigma2
from utils.model_io import (
	_load_independent_head_model,
	_load_stage2_head_model,
	_load_weights_into_rebuilt_model,
	_resolve_joint_model_paths,
)
from utils.model_utils import (
	_extract_mean_output,
	_extract_uncertainty_output,
	_resolve_model_input_shape,
	_split_nll_output,
)
from utils.normalization import (
	_compute_norm_factor,
	_convert_normed_tensor,
	_normalize_psf_for_observation,
)
from utils.plot_helpers import (
	_plot_inference_example,
	_plot_truth_vs_prediction,
	_plot_truth_vs_recovered,
)
from utils.tfrecord_io import (
	_decode_raw_example,
	_iter_tfrecord_records,
	_load_selected_tfrecord_arrays,
	_resolve_joint_tfrecord_path,
	_resolve_tfrecord_path,
)
from workflow.joint_pinn_fourhead_training import FourHeadJointPinnModel


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Plot joint four-head PINN training results.")
	parser.add_argument("--run-dir", type=Path, required=True)
	parser.add_argument("--checkpoint", choices=("best", "final", "all"), default="best")
	parser.add_argument("--image-head-model", type=Path, default=None)
	parser.add_argument("--residual-head-model", type=Path, default=None)
	parser.add_argument("--psf-mean-head-model", type=Path, default=None)
	parser.add_argument("--psf-unc-head-model", type=Path, default=None)
	parser.add_argument("--tfrecord", type=Path, default=None)
	parser.add_argument("--data-path", type=Path, default=None)
	parser.add_argument("--out-dir", type=Path, default=None)
	parser.add_argument("--n-examples", type=int, default=15)
	parser.add_argument("--frame", type=int, default=0)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--shuffle", action="store_true")
	parser.add_argument("--deconv-eps", type=float, default=1e-6)
	parser.add_argument("--psf-reconstruction-method", choices=("wiener", "optimize"), default="wiener")
	parser.add_argument("--psf-reconstruction-maxiter", type=int, default=2000)
	parser.add_argument("--psf-reconstruction-optimizer", choices=("adam", "lbfgs"), default="adam")
	parser.add_argument("--psf-reconstruction-convolution-backend", choices=("direct", "fft"), default="direct")
	parser.add_argument("--psf-reconstruction-data-loss", choices=("relative_l1", "squared", "gaussian_nll"), default="relative_l1")
	parser.add_argument("--psf-reconstruction-relative-loss-eps", type=float, default=1e-3)
	parser.add_argument("--psf-reconstruction-variance-eps", type=float, default=1e-12)
	parser.add_argument("--psf-reconstruction-compactness-weight", type=float, default=2e-3)
	parser.add_argument("--psf-reconstruction-l2-weight", type=float, default=1e-2)
	parser.add_argument("--psf-reconstruction-tv-weight", type=float, default=5e-3)
	parser.add_argument("--psf-reconstruction-verbose", action="store_true")
	parser.add_argument("--psf-reconstruction-n-crop-pix", type=int, default=16)
	parser.add_argument("--obs-panel-n-pix-zero", type=int, default=16)
	parser.add_argument("--dpi", type=int, default=150)
	parser.add_argument("--inference-only", action="store_true")
	parser.add_argument("--skip-data-inference", action="store_true")
	return parser.parse_args()


def _merge_cli_source_models(training_cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
	source_models = dict(training_cfg.get("source_models", {}))
	cli_paths = {
		"image": args.image_head_model,
		"noise": args.residual_head_model,
		"psf_mean": args.psf_mean_head_model,
		"psf_unc": args.psf_unc_head_model,
	}
	for key, value in cli_paths.items():
		if value is None:
			continue
		path = value.expanduser().resolve()
		if not path.exists():
			raise FileNotFoundError(path)
		entry = dict(source_models.get(key, {}))
		entry["model_path"] = str(path)
		source_models[key] = entry
	training_cfg = dict(training_cfg)
	training_cfg["source_models"] = source_models
	return training_cfg


def _build_joint_model(checkpoint_path: Path, training_cfg: dict[str, Any], preview_obs: np.ndarray | None = None) -> FourHeadJointPinnModel:
	source_models = dict(training_cfg.get("source_models", {}))
	image_cfg = dict(source_models.get("image", {}))
	noise_cfg = dict(source_models.get("noise", source_models.get("residual", {})))
	psf_mean_cfg = dict(source_models.get("psf_mean", {}))
	psf_unc_cfg = dict(source_models.get("psf_unc", {}))
	image_model_path = Path(str(image_cfg.get("model_path", ""))).expanduser().resolve()
	noise_model_path = Path(str(noise_cfg.get("model_path", ""))).expanduser().resolve()
	psf_mean_model_path = Path(str(psf_mean_cfg.get("model_path", ""))).expanduser().resolve()
	psf_unc_model_path = Path(str(psf_unc_cfg.get("model_path", ""))).expanduser().resolve()
	missing = [
		name
		for name, path in {
			"image": image_model_path,
			"noise": noise_model_path,
			"psf_mean": psf_mean_model_path,
			"psf_unc": psf_unc_model_path,
		}.items()
		if str(path) == "." or not path.exists()
	]
	if missing:
		raise FileNotFoundError(
			"Missing source head checkpoint path(s) for "
			+ ", ".join(missing)
			+ ". Provide training_config.json in the run dir or pass --image-head-model, --residual-head-model, --psf-mean-head-model, and --psf-unc-head-model."
		)
	if preview_obs is None:
		raise ValueError("preview_obs is required to infer source head shapes when local metadata is missing")
	preview_input_shape = tuple(int(v) for v in preview_obs.shape[1:])
	n_frames = int(preview_input_shape[-1])
	spatial_shape = preview_input_shape[:2]
	image_output_shape = (spatial_shape[0], spatial_shape[1], 1)
	frame_output_shape = (spatial_shape[0], spatial_shape[1], n_frames)
	stage2_input_shape = (spatial_shape[0], spatial_shape[1], 2 * n_frames)
	image_model, _ = _load_independent_head_model(image_model_path, fallback_input_shape=preview_input_shape, fallback_output_shape=image_output_shape)
	noise_model, _ = _load_independent_head_model(noise_model_path, fallback_input_shape=preview_input_shape, fallback_output_shape=frame_output_shape)
	psf_mean_model, psf_mean_head_cfg = _load_independent_head_model(psf_mean_model_path, fallback_input_shape=preview_input_shape, fallback_output_shape=frame_output_shape)
	psf_unc_model, psf_unc_head_cfg = _load_stage2_head_model(psf_unc_model_path, fallback_input_shape=stage2_input_shape, fallback_output_shape=frame_output_shape)
	loss_cfg = dict(training_cfg.get("loss", {}))
	dataset_cfg = dict(training_cfg.get("dataset", {}))
	weights_cfg = dict(loss_cfg.get("weights", {}))
	joint_model = FourHeadJointPinnModel(
		image_model,
		noise_model,
		psf_mean_model,
		psf_unc_model,
		pinn_weight=float(weights_cfg.get("pinn", 1.0)),
		im_weight=float(weights_cfg.get("im", 1.0)),
		psf_weight=float(weights_cfg.get("psf", 1.0)),
		noise_weight=float(weights_cfg.get("noise", weights_cfg.get("res", 1.0))),
		log_sigma=bool(loss_cfg.get("log_sigma", False)),
		log_min=float(loss_cfg.get("log_min", -6.0)),
		log_max=float(loss_cfg.get("log_max", 20.0)),
		sigma2_eps=float(loss_cfg.get("sigma2_eps", 1e-12)),
		psf_mean_source_norm_psf=dict(psf_mean_head_cfg.get("dataset", {})).get("norm_psf"),
		psf_unc_input_norm_psf=dict(psf_unc_head_cfg.get("dataset", {})).get("norm_psf"),
		norm_psf=dataset_cfg.get("norm_psf"),
		norm_noise=dataset_cfg.get("norm_noise", dataset_cfg.get("norm_res")),
		reconstruction_crop=int(loss_cfg.get("reconstruction_crop", 16)),
		name="joint_pinn_fourhead_model",
	)
	if preview_obs is None:
		input_shape = _resolve_model_input_shape(image_model)
		preview_obs = np.zeros((1, int(input_shape[1]), int(input_shape[2]), int(input_shape[3])), dtype=np.float32)
	_ = joint_model(tf.convert_to_tensor(preview_obs.astype(np.float32)), training=False)
	return _load_weights_into_rebuilt_model(joint_model, checkpoint_path)


def _predict_joint_batch(model: FourHeadJointPinnModel, obs_hwf: np.ndarray) -> dict[str, np.ndarray | bool]:
	obs_tf = tf.convert_to_tensor(obs_hwf.astype(np.float32))
	n_frames = int(obs_hwf.shape[-1])
	y_pred = tf.convert_to_tensor(model(obs_tf, training=False))
	main_channels = 1 + 2 * n_frames
	pred_main = y_pred[..., :main_channels]
	pred_unc = y_pred[..., main_channels:]
	pred_im = pred_main[..., :1]
	pred_psf = pred_main[..., 1 : 1 + n_frames]
	pred_noise = pred_main[..., 1 + n_frames : 1 + 2 * n_frames]
	unc_im = pred_unc[..., :1]
	unc_psf = pred_unc[..., 1 : 1 + n_frames]
	unc_noise = pred_unc[..., 1 + n_frames : 1 + 2 * n_frames]
	sigma2_im, _ = _pred_to_sigma2(unc_im, log_sigma=model.log_sigma, log_min=model.log_min, log_max=model.log_max, sigma2_eps=model.sigma2_eps)
	sigma2_psf, _ = _pred_to_sigma2(unc_psf, log_sigma=model.log_sigma, log_min=model.log_min, log_max=model.log_max, sigma2_eps=model.sigma2_eps)
	sigma2_noise, _ = _pred_to_sigma2(unc_noise, log_sigma=model.log_sigma, log_min=model.log_min, log_max=model.log_max, sigma2_eps=model.sigma2_eps)
	psf_df = tf.cast(model._psf_denorm_factor, pred_psf.dtype)
	noise_df = tf.cast(model._noise_denorm_factor, pred_noise.dtype)
	pred_psf_phys = pred_psf / psf_df
	pred_noise_phys = pred_noise / noise_df
	sigma2_psf_phys = sigma2_psf / tf.square(psf_df)
	sigma2_noise_phys = sigma2_noise / tf.square(noise_df)
	pred_psf_recon, sigma2_psf_recon, _ = _normalize_psf_for_observation(pred_psf_phys, sigma2_psf=sigma2_psf_phys)
	pred_obs = _convolve_image_with_psfs(pred_im, pred_psf_recon) - pred_noise_phys
	sigma2_obs = _convolve_image_with_psfs(sigma2_im, tf.square(pred_psf_recon))
	sigma2_obs += _convolve_image_with_psfs(tf.square(pred_im), sigma2_psf_recon)
	sigma2_obs += sigma2_noise_phys
	sigma2_obs = tf.maximum(sigma2_obs, tf.cast(1e-12, sigma2_obs.dtype))
	return {
		"has_uncertainty": True,
		"pred_im": pred_im.numpy().astype(np.float32),
		"pred_psf": pred_psf.numpy().astype(np.float32),
		"pred_noise": pred_noise.numpy().astype(np.float32),
		"pred_psf_phys": pred_psf_recon.numpy().astype(np.float32),
		"pred_noise_phys": pred_noise_phys.numpy().astype(np.float32),
		"pred_obs": pred_obs.numpy().astype(np.float32),
		"sigma_im": tf.sqrt(tf.maximum(sigma2_im, 1e-12)).numpy().astype(np.float32),
		"sigma_psf": tf.sqrt(tf.maximum(sigma2_psf, 1e-12)).numpy().astype(np.float32),
		"sigma_noise": tf.sqrt(tf.maximum(sigma2_noise, 1e-12)).numpy().astype(np.float32),
		"sigma_psf_phys": tf.sqrt(tf.maximum(sigma2_psf_recon, 1e-12)).numpy().astype(np.float32),
		"sigma_noise_phys": tf.sqrt(tf.maximum(sigma2_noise_phys, 1e-12)).numpy().astype(np.float32),
		"sigma_obs": tf.sqrt(sigma2_obs).numpy().astype(np.float32),
	}


def _run_tfrecord_plots(
	*,
	run_dir: Path,
	training_cfg: dict[str, Any],
	tfrecord_path: Path,
	out_dir: Path,
	model_label: str,
	joint_model_path: Path,
	n_examples: int,
	frame: int,
	shuffle: bool,
	seed: int,
	deconv_eps: float,
	psf_reconstruction_method: str,
	psf_reconstruction_optimizer: str,
	psf_reconstruction_convolution_backend: str,
	psf_reconstruction_data_loss: str,
	psf_reconstruction_relative_loss_eps: float,
	psf_reconstruction_variance_eps: float,
	psf_reconstruction_maxiter: int,
	psf_reconstruction_n_crop_pix: int,
	psf_reconstruction_compactness_weight: float,
	psf_reconstruction_l2_weight: float,
	psf_reconstruction_tv_weight: float,
	psf_reconstruction_verbose: bool,
	obs_panel_n_pix_zero: int,
	dpi: int,
) -> None:
	_ = run_dir
	dataset_cfg = dict(training_cfg.get("dataset", {}))
	raw, indices = _load_selected_tfrecord_arrays(tfrecord_path, n_examples=n_examples, shuffle=shuffle, seed=seed)
	truth = _prepare_truth_arrays(raw, dataset_cfg)
	obs_hwf = truth["obs_hwf"]
	image_hw1 = truth["image_hw1"]
	psf_hwf = truth["psf_hwf"]
	noise_hwf = truth["noise_hwf"]
	if frame < 0 or frame >= obs_hwf.shape[-1]:
		raise ValueError(f"--frame must be in [0, {obs_hwf.shape[-1] - 1}], got {frame}")
	joint_model = _build_joint_model(joint_model_path, training_cfg, obs_hwf[:1])
	pred = _predict_joint_batch(joint_model, obs_hwf)
	recovered = _compute_recovered_quantities(
		obs_hwf,
		pred_im_hw1=np.asarray(pred["pred_im"]),
		pred_psf_phys_hwf=np.asarray(pred["pred_psf_phys"]),
		pred_noise_phys_hwf=np.asarray(pred["pred_noise_phys"]),
		pred_sigma_im_hw1=np.asarray(pred["sigma_im"]),
		pred_sigma_noise_phys_hwf=np.asarray(pred["sigma_noise_phys"]),
		eps=deconv_eps,
		psf_reconstruction_method=psf_reconstruction_method,
		psf_reconstruction_optimizer=psf_reconstruction_optimizer,
		psf_reconstruction_convolution_backend=psf_reconstruction_convolution_backend,
		psf_reconstruction_data_loss=psf_reconstruction_data_loss,
		psf_reconstruction_relative_loss_eps=psf_reconstruction_relative_loss_eps,
		psf_reconstruction_variance_eps=psf_reconstruction_variance_eps,
		psf_reconstruction_maxiter=psf_reconstruction_maxiter,
		psf_reconstruction_n_crop_pix=psf_reconstruction_n_crop_pix,
		psf_reconstruction_compactness_weight=psf_reconstruction_compactness_weight,
		psf_reconstruction_l2_weight=psf_reconstruction_l2_weight,
		psf_reconstruction_tv_weight=psf_reconstruction_tv_weight,
		psf_reconstruction_verbose=psf_reconstruction_verbose,
	)
	model_dir = out_dir / model_label / f"tfrecord_{tfrecord_path.stem}"
	pred_dir = model_dir / "predictions"
	rec_dir = model_dir / "recovered"
	pred_dir.mkdir(parents=True, exist_ok=True)
	rec_dir.mkdir(parents=True, exist_ok=True)
	for out_idx, ex_idx in enumerate(indices):
		obs_true = obs_hwf[out_idx, :, :, frame]
		im_true = image_hw1[out_idx, :, :, 0]
		psf_true = psf_hwf[out_idx, :, :, frame]
		noise_true = noise_hwf[out_idx, :, :, frame]
		obs_pred = np.asarray(pred["pred_obs"])[out_idx, :, :, frame]
		im_pred = np.asarray(pred["pred_im"])[out_idx, :, :, 0]
		psf_pred = np.asarray(pred["pred_psf_phys"])[out_idx, :, :, frame]
		noise_pred = np.asarray(pred["pred_noise_phys"])[out_idx, :, :, frame]
		sigma_obs = np.asarray(pred["sigma_obs"])[out_idx, :, :, frame]
		sigma_im = np.asarray(pred["sigma_im"])[out_idx, :, :, 0]
		sigma_psf = np.asarray(pred["sigma_psf_phys"])[out_idx, :, :, frame]
		sigma_noise = np.asarray(pred["sigma_noise_phys"])[out_idx, :, :, frame]
		_plot_truth_vs_prediction(obs_true=obs_true, im_true=im_true, psf_true=psf_true, noise_true=noise_true, obs_pred=obs_pred, im_pred=im_pred, psf_pred=psf_pred, noise_pred=noise_pred, sigma_obs=sigma_obs, sigma_im=sigma_im, sigma_psf=sigma_psf, sigma_noise=sigma_noise, frame=frame, obs_panel_n_pix_zero=obs_panel_n_pix_zero, out_path=pred_dir / f"example_{ex_idx:03d}.png", dpi=dpi)
		_plot_truth_vs_recovered(obs_true=obs_true, im_true=im_true, psf_true=psf_true, noise_true=noise_true, obs_rec=recovered["recovered_obs"][out_idx, :, :, frame], im_rec=recovered["recovered_im"][out_idx, :, :, 0], psf_rec=recovered["recovered_psf"][out_idx, :, :, frame], noise_rec=recovered["recovered_noise"][out_idx, :, :, frame], frame=frame, out_path=rec_dir / f"example_{ex_idx:03d}.png", dpi=dpi)


def _run_data_inference_plots(
	*,
	run_dir: Path,
	training_cfg: dict[str, Any],
	tfrecord_path: Path | None,
	data_path: Path,
	out_dir: Path,
	model_label: str,
	joint_model_path: Path,
	deconv_eps: float,
	psf_reconstruction_method: str,
	psf_reconstruction_optimizer: str,
	psf_reconstruction_convolution_backend: str,
	psf_reconstruction_data_loss: str,
	psf_reconstruction_relative_loss_eps: float,
	psf_reconstruction_variance_eps: float,
	psf_reconstruction_maxiter: int,
	psf_reconstruction_n_crop_pix: int,
	psf_reconstruction_compactness_weight: float,
	psf_reconstruction_l2_weight: float,
	psf_reconstruction_tv_weight: float,
	psf_reconstruction_verbose: bool,
	dpi: int,
) -> None:
	_ = run_dir
	cube = np.asarray(np.load(data_path), dtype=np.float32)
	if cube.ndim != 3:
		raise ValueError(f"Expected data.npy to have shape (H,W,F), got {cube.shape}")
	preview_obs: np.ndarray
	if tfrecord_path is not None:
		for serialized in _iter_tfrecord_records(tfrecord_path):
			image, obs, psf, res = _decode_raw_example(serialized)
			raw_preview = {
				"image_hh": image[np.newaxis, ...],
				"obs_fhh": obs[np.newaxis, ...],
				"psf_fhh": psf[np.newaxis, ...],
				"noise_fhh": res[np.newaxis, ...],
			}
			preview_truth = _prepare_truth_arrays(raw_preview, dict(training_cfg.get("dataset", {})))
			preview_obs = preview_truth["obs_hwf"][:1].astype(np.float32)
			break
		else:
			preview_obs = cube[None, ...]
	else:
		preview_obs = cube[None, ...]
	joint_model = _build_joint_model(joint_model_path, training_cfg, preview_obs=preview_obs)
	input_shape = _resolve_model_input_shape(joint_model.image_model)
	cube = _crop_data_to_model(cube, input_shape, keep_all_frames=True)
	f_total = cube.shape[-1]
	f_in = int(input_shape[3])
	infer_dir = out_dir / model_label / "data_inference"
	fits_dir = infer_dir / "fits"
	infer_dir.mkdir(parents=True, exist_ok=True)
	fits_dir.mkdir(parents=True, exist_ok=True)
	pred_im_frames = np.zeros((f_total, cube.shape[0], cube.shape[1]), dtype=np.float32)
	pred_psf_frames = np.zeros((f_total, cube.shape[0], cube.shape[1]), dtype=np.float32)
	pred_noise_frames = np.zeros((f_total, cube.shape[0], cube.shape[1]), dtype=np.float32)
	pred_obs_frames = np.zeros((f_total, cube.shape[0], cube.shape[1]), dtype=np.float32)
	pred_sigma_obs_frames = np.zeros((f_total, cube.shape[0], cube.shape[1]), dtype=np.float32)
	pred_sigma_im_frames = np.zeros((f_total, cube.shape[0], cube.shape[1]), dtype=np.float32)
	pred_sigma_psf_frames = np.zeros((f_total, cube.shape[0], cube.shape[1]), dtype=np.float32)
	pred_sigma_noise_frames = np.zeros((f_total, cube.shape[0], cube.shape[1]), dtype=np.float32)
	rec_im_frames = np.zeros((f_total, cube.shape[0], cube.shape[1]), dtype=np.float32)
	rec_psf_frames = np.zeros((f_total, cube.shape[0], cube.shape[1]), dtype=np.float32)
	rec_noise_frames = np.zeros((f_total, cube.shape[0], cube.shape[1]), dtype=np.float32)
	for chunk_idx in range(int(np.ceil(f_total / f_in))):
		start = chunk_idx * f_in
		end = min((chunk_idx + 1) * f_in, f_total)
		chunk = cube[:, :, start:end]
		if chunk.shape[-1] < f_in:
			pad = f_in - chunk.shape[-1]
			chunk = np.pad(chunk, pad_width=((0, 0), (0, 0), (0, pad)), mode="constant", constant_values=0.0)
		obs_batch = chunk[np.newaxis, ...].astype(np.float32)
		pred = _predict_joint_batch(joint_model, obs_batch)
		recovered = _compute_recovered_quantities(
			obs_batch,
			pred_im_hw1=np.asarray(pred["pred_im"]),
			pred_psf_phys_hwf=np.asarray(pred["pred_psf_phys"]),
			pred_noise_phys_hwf=np.asarray(pred["pred_noise_phys"]),
			pred_sigma_im_hw1=np.asarray(pred["sigma_im"]),
			pred_sigma_noise_phys_hwf=np.asarray(pred["sigma_noise_phys"]),
			eps=deconv_eps,
			psf_reconstruction_method=psf_reconstruction_method,
			psf_reconstruction_optimizer=psf_reconstruction_optimizer,
			psf_reconstruction_convolution_backend=psf_reconstruction_convolution_backend,
			psf_reconstruction_data_loss=psf_reconstruction_data_loss,
			psf_reconstruction_relative_loss_eps=psf_reconstruction_relative_loss_eps,
			psf_reconstruction_variance_eps=psf_reconstruction_variance_eps,
			psf_reconstruction_maxiter=psf_reconstruction_maxiter,
			psf_reconstruction_n_crop_pix=psf_reconstruction_n_crop_pix,
			psf_reconstruction_compactness_weight=psf_reconstruction_compactness_weight,
			psf_reconstruction_l2_weight=psf_reconstruction_l2_weight,
			psf_reconstruction_tv_weight=psf_reconstruction_tv_weight,
			psf_reconstruction_verbose=psf_reconstruction_verbose,
		)
		pred_im_chunk = np.asarray(pred["pred_im"])[0, :, :, 0]
		pred_psf_chunk = np.asarray(pred["pred_psf_phys"])[0]
		pred_noise_chunk = np.asarray(pred["pred_noise_phys"])[0]
		pred_obs_chunk = np.asarray(pred["pred_obs"])[0]
		pred_sigma_obs_chunk = np.asarray(pred["sigma_obs"])[0]
		pred_sigma_im_chunk = np.asarray(pred["sigma_im"])[0, :, :, 0]
		pred_sigma_psf_chunk = np.asarray(pred["sigma_psf_phys"])[0]
		pred_sigma_noise_chunk = np.asarray(pred["sigma_noise_phys"])[0]
		rec_im_chunk = np.asarray(recovered["recovered_im"])[0, :, :, 0]
		rec_psf_chunk = np.asarray(recovered["recovered_psf"])[0]
		rec_noise_chunk = np.asarray(recovered["recovered_noise"])[0]
		for local_frame, global_frame in enumerate(range(start, end)):
			pred_im_frames[global_frame] = pred_im_chunk.astype(np.float32)
			pred_psf_frames[global_frame] = pred_psf_chunk[:, :, local_frame].astype(np.float32)
			pred_noise_frames[global_frame] = pred_noise_chunk[:, :, local_frame].astype(np.float32)
			pred_obs_frames[global_frame] = pred_obs_chunk[:, :, local_frame].astype(np.float32)
			pred_sigma_obs_frames[global_frame] = pred_sigma_obs_chunk[:, :, local_frame].astype(np.float32)
			pred_sigma_im_frames[global_frame] = pred_sigma_im_chunk.astype(np.float32)
			pred_sigma_psf_frames[global_frame] = pred_sigma_psf_chunk[:, :, local_frame].astype(np.float32)
			pred_sigma_noise_frames[global_frame] = pred_sigma_noise_chunk[:, :, local_frame].astype(np.float32)
			rec_im_frames[global_frame] = rec_im_chunk.astype(np.float32)
			rec_psf_frames[global_frame] = rec_psf_chunk[:, :, local_frame].astype(np.float32)
			rec_noise_frames[global_frame] = rec_noise_chunk[:, :, local_frame].astype(np.float32)

	def _write_inference_panel_set(panel_dir: Path, *, obs_true: np.ndarray, pred_im: np.ndarray, pred_psf: np.ndarray, pred_noise: np.ndarray, pred_obs: np.ndarray, sigma_obs: np.ndarray | None, sigma_im: np.ndarray | None, sigma_psf: np.ndarray | None, sigma_noise: np.ndarray | None, rec_im: np.ndarray, rec_psf: np.ndarray, rec_noise: np.ndarray, frame_label: str | int, frame_header: int) -> None:
		_plot_inference_example(obs_true=obs_true, pred_im=pred_im, pred_psf=pred_psf, pred_noise=pred_noise, pred_obs=pred_obs, sigma_obs=sigma_obs, sigma_im=sigma_im, sigma_psf=sigma_psf, sigma_noise=sigma_noise, rec_im=rec_im, rec_psf=rec_psf, rec_noise=rec_noise, frame=frame_label, out_path=infer_dir / f"{panel_dir.name}.png", dpi=dpi)
		header = {"MODEL": model_label, "FRAME": int(frame_header), "DATA": str(data_path.name)}
		_write_fits_image(image_hw=obs_true, path=panel_dir / "true_obs.fits", header={**header, "PANEL": "true_obs"})
		_write_fits_image(image_hw=pred_im, path=panel_dir / "pred_im.fits", header={**header, "PANEL": "pred_im"})
		_write_fits_image(image_hw=pred_psf, path=panel_dir / "pred_psf.fits", header={**header, "PANEL": "pred_psf"})
		_write_fits_image(image_hw=pred_noise, path=panel_dir / "pred_noise.fits", header={**header, "PANEL": "pred_noise"})
		_write_fits_image(image_hw=pred_obs, path=panel_dir / "pred_obs.fits", header={**header, "PANEL": "pred_obs"})
		if sigma_obs is not None:
			_write_fits_image(image_hw=sigma_obs, path=panel_dir / "pred_obs_uncertainty.fits", header={**header, "PANEL": "obs_unc"})
		if sigma_im is not None:
			_write_fits_image(image_hw=sigma_im, path=panel_dir / "pred_im_uncertainty.fits", header={**header, "PANEL": "im_unc"})
		if sigma_psf is not None:
			_write_fits_image(image_hw=sigma_psf, path=panel_dir / "pred_psf_uncertainty.fits", header={**header, "PANEL": "psf_unc"})
		if sigma_noise is not None:
			_write_fits_image(image_hw=sigma_noise, path=panel_dir / "pred_noise_uncertainty.fits", header={**header, "PANEL": "noise_unc"})
		_write_fits_image(image_hw=rec_im, path=panel_dir / "recovered_im.fits", header={**header, "PANEL": "rec_im"})
		_write_fits_image(image_hw=rec_psf, path=panel_dir / "recovered_psf.fits", header={**header, "PANEL": "rec_psf"})
		_write_fits_image(image_hw=rec_noise, path=panel_dir / "recovered_noise.fits", header={**header, "PANEL": "rec_noise"})

	for frame_idx in range(f_total):
		frame_fits_dir = fits_dir / f"frame_{frame_idx:03d}"
		frame_fits_dir.mkdir(parents=True, exist_ok=True)
		_write_inference_panel_set(frame_fits_dir, obs_true=cube[:, :, frame_idx], pred_im=pred_im_frames[frame_idx], pred_psf=pred_psf_frames[frame_idx], pred_noise=pred_noise_frames[frame_idx], pred_obs=pred_obs_frames[frame_idx], sigma_obs=pred_sigma_obs_frames[frame_idx], sigma_im=pred_sigma_im_frames[frame_idx], sigma_psf=pred_sigma_psf_frames[frame_idx], sigma_noise=pred_sigma_noise_frames[frame_idx], rec_im=rec_im_frames[frame_idx], rec_psf=rec_psf_frames[frame_idx], rec_noise=rec_noise_frames[frame_idx], frame_label=frame_idx, frame_header=frame_idx)

	for label, panels in {
		"mean": {
			"obs_true": np.mean(cube, axis=-1),
			"pred_im": np.mean(pred_im_frames, axis=0),
			"pred_psf": np.mean(pred_psf_frames, axis=0),
			"pred_noise": np.mean(pred_noise_frames, axis=0),
			"pred_obs": np.mean(pred_obs_frames, axis=0),
			"sigma_obs": np.sqrt(np.mean(np.square(pred_sigma_obs_frames), axis=0)).astype(np.float32),
			"sigma_im": np.mean(pred_sigma_im_frames, axis=0),
			"sigma_psf": np.mean(pred_sigma_psf_frames, axis=0),
			"sigma_noise": np.mean(pred_sigma_noise_frames, axis=0),
			"rec_im": np.mean(rec_im_frames, axis=0),
			"rec_psf": np.mean(rec_psf_frames, axis=0),
			"rec_noise": np.mean(rec_noise_frames, axis=0),
		},
		"median": {
			"obs_true": np.median(cube, axis=-1),
			"pred_im": np.median(pred_im_frames, axis=0),
			"pred_psf": np.median(pred_psf_frames, axis=0),
			"pred_noise": np.median(pred_noise_frames, axis=0),
			"pred_obs": np.median(pred_obs_frames, axis=0),
			"sigma_obs": np.median(pred_sigma_obs_frames, axis=0).astype(np.float32),
			"sigma_im": np.median(pred_sigma_im_frames, axis=0).astype(np.float32),
			"sigma_psf": np.median(pred_sigma_psf_frames, axis=0).astype(np.float32),
			"sigma_noise": np.median(pred_sigma_noise_frames, axis=0).astype(np.float32),
			"rec_im": np.median(rec_im_frames, axis=0),
			"rec_psf": np.median(rec_psf_frames, axis=0),
			"rec_noise": np.median(rec_noise_frames, axis=0),
		},
	}.items():
		agg_dir = fits_dir / label
		agg_dir.mkdir(parents=True, exist_ok=True)
		_write_inference_panel_set(agg_dir, obs_true=panels["obs_true"], pred_im=panels["pred_im"], pred_psf=panels["pred_psf"], pred_noise=panels["pred_noise"], pred_obs=panels["pred_obs"], sigma_obs=panels["sigma_obs"], sigma_im=panels["sigma_im"], sigma_psf=panels["sigma_psf"], sigma_noise=panels["sigma_noise"], rec_im=panels["rec_im"], rec_psf=panels["rec_psf"], rec_noise=panels["rec_noise"], frame_label=label, frame_header=-1)


def main() -> None:
	args = parse_args()
	if args.n_examples <= 0:
		raise ValueError("--n-examples must be > 0")
	run_dir = args.run_dir.expanduser().resolve()
	training_cfg = _merge_cli_source_models(_load_joint_run_config(run_dir), args)
	available_joint_models = _resolve_joint_model_paths(run_dir, training_cfg)
	if args.checkpoint == "all":
		joint_models = available_joint_models
	else:
		checkpoint_key = f"{args.checkpoint}_model"
		joint_model_path = available_joint_models.get(checkpoint_key)
		if joint_model_path is None:
			available = ", ".join(sorted(available_joint_models))
			raise FileNotFoundError(
				f"Requested checkpoint '{args.checkpoint}' is not available under {run_dir}. Available: {available}"
			)
		joint_models = {checkpoint_key: joint_model_path}
	out_dir = args.out_dir.expanduser().resolve() if args.out_dir is not None else (run_dir / "plots_joint_pinn_fourhead")
	out_dir.mkdir(parents=True, exist_ok=True)
	dataset_cfg = dict(training_cfg.get("dataset", {}))
	tfrecord_path = _resolve_joint_tfrecord_path(run_dir, args.tfrecord, dataset_cfg)
	data_path = None if args.skip_data_inference else _resolve_data_path(run_dir, args.data_path)
	if tfrecord_path is None and data_path is None:
		raise FileNotFoundError("No TFRecord found and no data.npy available")
	for model_label, joint_model_path in joint_models.items():
		if tfrecord_path is not None and not args.inference_only:
			_run_tfrecord_plots(
				run_dir=run_dir,
				training_cfg=training_cfg,
				tfrecord_path=tfrecord_path,
				out_dir=out_dir,
				model_label=model_label,
				joint_model_path=joint_model_path,
				n_examples=args.n_examples,
				frame=args.frame,
				shuffle=args.shuffle,
				seed=args.seed,
				deconv_eps=args.deconv_eps,
				psf_reconstruction_method=args.psf_reconstruction_method,
				psf_reconstruction_optimizer=args.psf_reconstruction_optimizer,
				psf_reconstruction_convolution_backend=args.psf_reconstruction_convolution_backend,
				psf_reconstruction_data_loss=args.psf_reconstruction_data_loss,
				psf_reconstruction_relative_loss_eps=args.psf_reconstruction_relative_loss_eps,
				psf_reconstruction_variance_eps=args.psf_reconstruction_variance_eps,
				psf_reconstruction_maxiter=args.psf_reconstruction_maxiter,
				psf_reconstruction_n_crop_pix=args.psf_reconstruction_n_crop_pix,
				psf_reconstruction_compactness_weight=args.psf_reconstruction_compactness_weight,
				psf_reconstruction_l2_weight=args.psf_reconstruction_l2_weight,
				psf_reconstruction_tv_weight=args.psf_reconstruction_tv_weight,
				psf_reconstruction_verbose=args.psf_reconstruction_verbose,
				obs_panel_n_pix_zero=args.obs_panel_n_pix_zero,
				dpi=args.dpi,
			)
		if data_path is not None:
			_run_data_inference_plots(
				run_dir=run_dir,
				training_cfg=training_cfg,
				tfrecord_path=tfrecord_path,
				data_path=data_path,
				out_dir=out_dir,
				model_label=model_label,
				joint_model_path=joint_model_path,
				deconv_eps=args.deconv_eps,
				psf_reconstruction_method=args.psf_reconstruction_method,
				psf_reconstruction_optimizer=args.psf_reconstruction_optimizer,
				psf_reconstruction_convolution_backend=args.psf_reconstruction_convolution_backend,
				psf_reconstruction_data_loss=args.psf_reconstruction_data_loss,
				psf_reconstruction_relative_loss_eps=args.psf_reconstruction_relative_loss_eps,
				psf_reconstruction_variance_eps=args.psf_reconstruction_variance_eps,
				psf_reconstruction_maxiter=args.psf_reconstruction_maxiter,
				psf_reconstruction_n_crop_pix=args.psf_reconstruction_n_crop_pix,
				psf_reconstruction_compactness_weight=args.psf_reconstruction_compactness_weight,
				psf_reconstruction_l2_weight=args.psf_reconstruction_l2_weight,
				psf_reconstruction_tv_weight=args.psf_reconstruction_tv_weight,
				psf_reconstruction_verbose=args.psf_reconstruction_verbose,
				dpi=args.dpi,
			)
	print(f"[plot_joint_pinn_fourhead_results] Wrote plots to: {out_dir}")


if __name__ == "__main__":
	main()
