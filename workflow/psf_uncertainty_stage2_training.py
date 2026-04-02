"""Stage-2 PSF uncertainty training workflow.

This workflow trains a second-stage model that estimates the uncertainty of a
pretrained PSF predictor.

Usage
-----
    python psf_uncertainty_stage2_training.py --config <experiment.py>

Input
-----
- observation cube       : (H, W, F)
- frozen stage-1 PSF mean: (H, W, F)

These are concatenated along the channel axis, so the stage-2 model sees an
input tensor of shape (H, W, 2F).

Target / loss
-------------
The trainable stage-2 model predicts *only* a PSF uncertainty parameterization
with shape (H, W, F). The frozen stage-1 PSF prediction is treated as the fixed
Gaussian mean, and the optimized loss is a Gaussian NLL against the true PSF.

Important design choices
------------------------
- The stage-1 PSF model is frozen and used on the fly in the tf.data pipeline.
- The resulting stage-2 features are cached to disk.
- No explicit output normalization is applied to the stage-2 model output.
- Previous training workflows are left untouched; this is a standalone entrypoint.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from configs.load_config import load_experiment_config, extract_arch_config, extract_training_config
from neural_networks.dataset import list_tfrecord_files
from neural_networks.dense_psf import build_dense_psf
from neural_networks.gpkh import build_gpkh
from neural_networks.gpkh_convdecoder import build_gpkh_convdecoder
from neural_networks.plot_training import plot_training_outputs
from neural_networks.skh import build_skh
from neural_networks.training import train_unet
from neural_networks.layers import GroupNormalization
from neural_networks.unet import build_unet
from utils.model_utils import _wrap_model_output_activation
from neural_networks.layers import _upsample_bilinear
from utils.io import _load_json, _load_run_config, _clear_cache_prefix
from utils.metrics import _var_normalized_mse, _log10_clip_to_ln
from utils.model_io import (
    _keras_load_model, _load_weights_into_rebuilt_model, _infer_model_spec_from_keras_archive,
    _resolve_model_paths,
)
from utils.model_utils import _resolve_model_input_shape_3d as _resolve_model_input_shape, _extract_prediction_mean
from utils.normalization import _apply_norm_tf
from utils.tfrecord_io import _decode_raw_example, _load_preview_raw_from_first_tfrecord


DEFAULT_MODEL_NAME = "gpkh"
ALLOWED_MODEL_NAMES = ("gpkh", "gpkh_convdecoder", "skh", "dense", "unet")
_TEMP_ROOT: Path | None = None

_DEFAULT_ARCH: dict[str, object] = {
	"layers_per_block": 3,
	"base_filters": 32,
	"latent_dim": 512,
	"normalization": "none",
	"group_norm_groups": 8,
	"weight_decay": 0,
	"inner_activation_function": "relu",
	"output_activation_function": "linear",
	"normalize_output_sum": True,
	"normalize_with_first": True,
	"normalize_first_only": False,
	"normalize_by_mean": False,
}


def _resolve_temp_root(run_dir: Path) -> Path:
	global _TEMP_ROOT
	if _TEMP_ROOT is not None:
		root = _TEMP_ROOT
	else:
		root = run_dir / "tmp"
	root.mkdir(parents=True, exist_ok=True)
	return root


def _temporary_directory(*, prefix: str, run_dir: Path) -> tempfile.TemporaryDirectory[str]:
	return tempfile.TemporaryDirectory(prefix=prefix, dir=str(_resolve_temp_root(run_dir)))


def _resolve_source_psf_model_path(
	*,
	output_base_dir: Path,
	psf_head_config: dict,
	psf_unc_config: dict,
) -> Path:
	psf_run_name = str(psf_head_config.get("run_name", "psf_only"))
	model_label = str(psf_unc_config.get("source_psf_model_label", "best_model")).strip().lower()
	psf_run_dir = output_base_dir / psf_run_name

	if not psf_run_dir.exists():
		raise FileNotFoundError(f"Source PSF run directory not found: {psf_run_dir}")

	model_paths = _resolve_model_paths(psf_run_dir)
	if model_label not in model_paths:
		raise ValueError(
			f"source_psf_model_label must be one of {tuple(model_paths)}, got {model_label!r}"
		)
	return model_paths[model_label]


def _infer_psf_shapes_from_preview(
	run_cfg: dict[str, Any],
	*,
	preview_raw: dict[str, np.ndarray],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
	dataset_cfg = dict(run_cfg.get("dataset", {}))
	obs_fhh = np.asarray(preview_raw["obs_fhh"], dtype=np.float32)
	psf_fhh = np.asarray(preview_raw["psf_fhh"], dtype=np.float32)
	crop = int(dataset_cfg.get("half_n_pix_crop", 0))
	if crop > 0:
		obs_fhh = obs_fhh[:, :, crop:-crop, crop:-crop]
		psf_fhh = psf_fhh[:, :, crop:-crop, crop:-crop]
	input_shape = (
		int(obs_fhh.shape[2]),
		int(obs_fhh.shape[3]),
		int(obs_fhh.shape[1]),
	)
	output_shape = (
		int(psf_fhh.shape[2]),
		int(psf_fhh.shape[3]),
		int(psf_fhh.shape[1]),
	)
	return input_shape, output_shape


def _rebuild_psf_head_model(
	head_dir: Path,
	*,
	checkpoint_path: Path | None = None,
	preview_raw: dict[str, np.ndarray],
) -> tf.keras.Model:
	run_cfg = _load_run_config(head_dir)
	input_shape, output_shape = _infer_psf_shapes_from_preview(run_cfg, preview_raw=preview_raw)
	archive_spec = None
	if checkpoint_path is not None and checkpoint_path.suffix == ".keras":
		archive_spec = _infer_model_spec_from_keras_archive(checkpoint_path)

	model_name = str((archive_spec or {}).get("model_name") or run_cfg.get("model", {}).get("name", "gpkh")).strip().lower()
	archive_output_shape = None if archive_spec is None else archive_spec.get("archive_output_shape")
	if archive_output_shape is not None:
		archive_channels = int(archive_output_shape[-1])
		if archive_channels == int(output_shape[-1]):
			nll = False
		elif archive_channels == 2 * int(output_shape[-1]):
			nll = True
		else:
			loss_mode = str(run_cfg.get("loss", {}).get("loss", "nll")).strip().lower()
			nll = bool(run_cfg.get("nll", loss_mode == "nll"))
	else:
		loss_mode = str(run_cfg.get("loss", {}).get("loss", "nll")).strip().lower()
		nll = bool(run_cfg.get("nll", loss_mode == "nll"))

	output_activation = str(run_cfg.get("output_activation_function", "linear")).strip().lower() or "linear"
	output_channels = int(output_shape[-1])
	model_output_shape = (output_shape[0], output_shape[1], output_channels * (2 if nll else 1))

	if model_name == "gpkh":
		model_kwargs = dict(run_cfg.get("gpkh", _DEFAULT_ARCH))
		if archive_spec is not None:
			model_kwargs.update(
				{
					"layers_per_block": archive_spec.get("layers_per_block", model_kwargs.get("layers_per_block")),
					"base_filters": archive_spec.get("base_filters", model_kwargs.get("base_filters")),
					"normalization": archive_spec.get("normalization", model_kwargs.get("normalization")),
					"group_norm_groups": archive_spec.get("group_norm_groups", model_kwargs.get("group_norm_groups")),
					"weight_decay": archive_spec.get("weight_decay", model_kwargs.get("weight_decay")),
					"inner_activation_function": archive_spec.get("inner_activation_function", model_kwargs.get("inner_activation_function")),
					"latent_dim": archive_spec.get("latent_dim", model_kwargs.get("latent_dim")),
					"normalize_output_sum": archive_spec.get("normalize_output_sum", model_kwargs.get("normalize_output_sum")),
					"normalize_with_first": archive_spec.get("normalize_with_first", model_kwargs.get("normalize_with_first", True)),
				}
			)
		model_kwargs["input_shape"] = input_shape
		model_kwargs["output_shape"] = model_output_shape
		model_kwargs["output_activation_function"] = "linear"
		model = build_gpkh(**model_kwargs)
	elif model_name == "gpkh_convdecoder":
		model_kwargs = dict(run_cfg.get("gpkh_convdecoder", _DEFAULT_ARCH))
		if archive_spec is not None:
			model_kwargs.update(
				{
					"layers_per_block": archive_spec.get("layers_per_block", model_kwargs.get("layers_per_block")),
					"base_filters": archive_spec.get("base_filters", model_kwargs.get("base_filters")),
					"normalization": archive_spec.get("normalization", model_kwargs.get("normalization")),
					"group_norm_groups": archive_spec.get("group_norm_groups", model_kwargs.get("group_norm_groups")),
					"weight_decay": archive_spec.get("weight_decay", model_kwargs.get("weight_decay")),
					"inner_activation_function": archive_spec.get("inner_activation_function", model_kwargs.get("inner_activation_function")),
					"latent_dim": archive_spec.get("latent_dim", model_kwargs.get("latent_dim")),
					"normalize_output_sum": archive_spec.get("normalize_output_sum", model_kwargs.get("normalize_output_sum")),
					"normalize_with_first": archive_spec.get("normalize_with_first", model_kwargs.get("normalize_with_first", True)),
				}
			)
		model_kwargs["input_shape"] = input_shape
		model_kwargs["output_shape"] = model_output_shape
		model_kwargs["output_activation_function"] = "linear"
		model = build_gpkh_convdecoder(**model_kwargs)
	elif model_name == "skh":
		model_kwargs = dict(run_cfg.get("skh", _DEFAULT_ARCH))
		if archive_spec is not None:
			model_kwargs.update(
				{
					"layers_per_block": archive_spec.get("layers_per_block", model_kwargs.get("layers_per_block")),
					"base_filters": archive_spec.get("base_filters", model_kwargs.get("base_filters")),
					"normalization": archive_spec.get("normalization", model_kwargs.get("normalization")),
					"group_norm_groups": archive_spec.get("group_norm_groups", model_kwargs.get("group_norm_groups")),
					"weight_decay": archive_spec.get("weight_decay", model_kwargs.get("weight_decay")),
					"inner_activation_function": archive_spec.get("inner_activation_function", model_kwargs.get("inner_activation_function")),
					"latent_dim": archive_spec.get("latent_dim", model_kwargs.get("latent_dim")),
					"normalize_output_sum": archive_spec.get("normalize_output_sum", model_kwargs.get("normalize_output_sum")),
					"normalize_with_first": archive_spec.get("normalize_with_first", model_kwargs.get("normalize_with_first", True)),
				}
			)
		model_kwargs["input_shape"] = input_shape
		model_kwargs["output_shape"] = model_output_shape
		model_kwargs["output_activation_function"] = "linear"
		model = build_skh(**model_kwargs)
	elif model_name == "dense":
		model_kwargs = dict(run_cfg.get("dense", _DEFAULT_ARCH))
		model_kwargs["input_shape"] = input_shape
		model_kwargs["output_shape"] = model_output_shape
		model_kwargs["output_activation_function"] = "linear"
		model = build_dense_psf(**model_kwargs)
	else:
		model_kwargs = dict(run_cfg.get("unet", _DEFAULT_ARCH))
		model_kwargs["input_shape"] = input_shape
		model_kwargs["output_shape"] = model_output_shape
		model_kwargs["output_activation_function"] = "linear"
		model = build_unet(**model_kwargs)

	return _wrap_model_output_activation(
		model,
		activation_name=output_activation,
		output_channels=output_channels,
		nll=nll,
	)


def _keras_load_psf_model(
	path: Path,
	*,
	preview_raw: dict[str, np.ndarray] | None = None,
) -> tf.keras.Model:
	try:
		return _keras_load_model(path)
	except Exception:
		if preview_raw is None:
			raise
		head_dir = path.parent.parent if path.parent.name == "checkpoints" else path.parent
		model = _rebuild_psf_head_model(
			head_dir,
			checkpoint_path=path,
			preview_raw=preview_raw,
		)
		return _load_weights_into_rebuilt_model(model, path)


def _prepare_stage2_example(
	features: dict[str, tf.Tensor],
	*,
	source_psf_model: tf.keras.Model,
	half_n_pix_crop: int,
	norm_psf: str | float | None,
	source_input_shape: tuple[int, int, int],
) -> tuple[tf.Tensor, tf.Tensor]:
	obs = features["obs"]
	psf = features["psf"]

	if half_n_pix_crop > 0:
		c = int(half_n_pix_crop)
		obs = obs[:, c:-c, c:-c]
		psf = psf[:, c:-c, c:-c]

	psf = _apply_norm_tf(psf, norm_psf, spatial_axis=1)
	obs = tf.transpose(obs, perm=(1, 2, 0))
	psf = tf.transpose(psf, perm=(1, 2, 0))

	expected_h, expected_w, expected_f = source_input_shape
	with tf.control_dependencies(
		[
			tf.debugging.assert_equal(tf.shape(obs)[0], expected_h, message="Obs height does not match source PSF model input"),
			tf.debugging.assert_equal(tf.shape(obs)[1], expected_w, message="Obs width does not match source PSF model input"),
			tf.debugging.assert_equal(tf.shape(obs)[2], expected_f, message="Obs frame count does not match source PSF model input"),
		]
	):
		obs = tf.identity(obs)

	pred = source_psf_model(obs[tf.newaxis, ...], training=False)
	if isinstance(pred, (list, tuple)):
		pred = pred[0]
	pred = tf.convert_to_tensor(pred)
	pred_mean = _extract_prediction_mean(pred, tf.shape(psf)[-1])
	pred_mean = _apply_norm_tf(pred_mean, norm_psf, spatial_axis=1)
	pred_mean = tf.stop_gradient(tf.squeeze(pred_mean, axis=0))
	stage2_input = tf.concat([obs, pred_mean], axis=-1)
	target = tf.concat([psf, pred_mean], axis=-1)
	return stage2_input, target


def _prepare_stage2_arrays_numpy(
	*,
	obs_fhh: np.ndarray,
	psf_fhh: np.ndarray,
	source_psf_model: tf.keras.Model,
	half_n_pix_crop: int,
	norm_psf: str | float | None,
	source_input_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
	obs = np.asarray(obs_fhh, dtype=np.float32)
	psf = np.asarray(psf_fhh, dtype=np.float32)

	if half_n_pix_crop > 0:
		c = int(half_n_pix_crop)
		obs = obs[:, c:-c, c:-c]
		psf = psf[:, c:-c, c:-c]

	psf_tf = _apply_norm_tf(tf.convert_to_tensor(psf), norm_psf, spatial_axis=1)
	psf = np.asarray(psf_tf.numpy(), dtype=np.float32)
	obs_hwf = np.transpose(obs, (1, 2, 0)).astype(np.float32)
	psf_hwf = np.transpose(psf, (1, 2, 0)).astype(np.float32)

	expected_h, expected_w, expected_f = source_input_shape
	if obs_hwf.shape != (expected_h, expected_w, expected_f):
		raise ValueError(
			"Observation shape does not match source PSF model input shape: "
			f"got {obs_hwf.shape}, expected {(expected_h, expected_w, expected_f)}"
		)

	pred = source_psf_model(obs_hwf[np.newaxis, ...], training=False)
	if isinstance(pred, (list, tuple)):
		pred = pred[0]
	pred = tf.convert_to_tensor(pred)
	pred_mean = _extract_prediction_mean(pred, tf.constant(psf_hwf.shape[-1], dtype=tf.int32))
	pred_mean = _apply_norm_tf(pred_mean, norm_psf, spatial_axis=1)
	pred_mean_hwf = np.asarray(tf.stop_gradient(tf.squeeze(pred_mean, axis=0)).numpy(), dtype=np.float32)

	stage2_input = np.concatenate([obs_hwf, pred_mean_hwf], axis=-1).astype(np.float32)
	target = np.concatenate([psf_hwf, pred_mean_hwf], axis=-1).astype(np.float32)
	return stage2_input, target


def _prepare_stage2_batch_arrays_numpy(
	*,
	obs_bfhh: np.ndarray,
	psf_bfhh: np.ndarray,
	source_psf_model: tf.keras.Model,
	half_n_pix_crop: int,
	norm_psf: str | float | None,
	source_input_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
	obs = np.asarray(obs_bfhh, dtype=np.float32)
	psf = np.asarray(psf_bfhh, dtype=np.float32)

	if half_n_pix_crop > 0:
		c = int(half_n_pix_crop)
		obs = obs[:, :, c:-c, c:-c]
		psf = psf[:, :, c:-c, c:-c]

	psf_tf = _apply_norm_tf(tf.convert_to_tensor(psf), norm_psf, spatial_axis=2)
	psf = np.asarray(psf_tf.numpy(), dtype=np.float32)
	obs_hwf = np.transpose(obs, (0, 2, 3, 1)).astype(np.float32)
	psf_hwf = np.transpose(psf, (0, 2, 3, 1)).astype(np.float32)

	expected_h, expected_w, expected_f = source_input_shape
	if obs_hwf.shape[1:] != (expected_h, expected_w, expected_f):
		raise ValueError(
			"Observation batch shape does not match source PSF model input shape: "
			f"got {obs_hwf.shape[1:]}, expected {(expected_h, expected_w, expected_f)}"
		)

	pred = source_psf_model(obs_hwf, training=False)
	if isinstance(pred, (list, tuple)):
		pred = pred[0]
	pred = tf.convert_to_tensor(pred)
	pred_mean = _extract_prediction_mean(pred, tf.constant(psf_hwf.shape[-1], dtype=tf.int32))
	pred_mean = _apply_norm_tf(pred_mean, norm_psf, spatial_axis=1)
	pred_mean_hwf = np.asarray(tf.stop_gradient(pred_mean).numpy(), dtype=np.float32)

	stage2_input_batch = np.concatenate([obs_hwf, pred_mean_hwf], axis=-1).astype(np.float32)
	target_batch = np.concatenate([psf_hwf, pred_mean_hwf], axis=-1).astype(np.float32)
	return stage2_input_batch, target_batch


def _stage2_generator(
	*,
	data_dir: Path,
	source_psf_model: tf.keras.Model,
	source_inference_batch_size: int,
	half_n_pix_crop: int,
	norm_psf: str | float | None,
	source_input_shape: tuple[int, int, int],
):
	obs_batch: list[np.ndarray] = []
	psf_batch: list[np.ndarray] = []

	def _flush_pending():
		if not obs_batch:
			return
		stage2_input_batch, target_batch = _prepare_stage2_batch_arrays_numpy(
			obs_bfhh=np.stack(obs_batch, axis=0),
			psf_bfhh=np.stack(psf_batch, axis=0),
			source_psf_model=source_psf_model,
			half_n_pix_crop=half_n_pix_crop,
			norm_psf=norm_psf,
			source_input_shape=source_input_shape,
		)
		for idx in range(stage2_input_batch.shape[0]):
			yield stage2_input_batch[idx], target_batch[idx]
		obs_batch.clear()
		psf_batch.clear()

	files = list_tfrecord_files(data_dir)
	for tfrecord_path in files:
		for serialized in tf.data.TFRecordDataset([str(tfrecord_path)]):
			_image, obs, psf, _res = _decode_raw_example(bytes(serialized.numpy()))
			obs_batch.append(obs)
			psf_batch.append(psf)
			if len(obs_batch) >= int(source_inference_batch_size):
				yield from _flush_pending()
	if obs_batch:
		yield from _flush_pending()


def _infer_stage2_shapes_from_preview(
	*,
	preview_raw: dict[str, np.ndarray],
	half_n_pix_crop: int,
	norm_psf: str | float | None,
	source_input_shape: tuple[int, int, int],
) -> tuple[tuple[int, int, int], tuple[int, int, int], int]:
	obs = np.asarray(preview_raw["obs_fhh"][0], dtype=np.float32)
	psf = np.asarray(preview_raw["psf_fhh"][0], dtype=np.float32)
	if half_n_pix_crop > 0:
		c = int(half_n_pix_crop)
		obs = obs[:, c:-c, c:-c]
		psf = psf[:, c:-c, c:-c]
	psf_tf = _apply_norm_tf(tf.convert_to_tensor(psf), norm_psf, spatial_axis=1)
	psf = np.asarray(psf_tf.numpy(), dtype=np.float32)
	output_channels = int(psf.shape[0])
	input_shape = (int(obs.shape[1]), int(obs.shape[2]), int(obs.shape[0] + output_channels))
	output_shape = (int(psf.shape[1]), int(psf.shape[2]), output_channels)
	if (int(obs.shape[1]), int(obs.shape[2]), int(obs.shape[0])) != tuple(int(v) for v in source_input_shape):
		raise ValueError(
			"Preview observation shape does not match source PSF model input shape: "
			f"got {(int(obs.shape[1]), int(obs.shape[2]), int(obs.shape[0]))}, expected {source_input_shape}"
		)
	return input_shape, output_shape, output_channels


def make_stage2_dataset(
	data_dir: str | Path,
	*,
	source_psf_model: tf.keras.Model,
	stage2_input_shape: tuple[int, int, int],
	stage2_target_shape: tuple[int, int, int],
	batch_size: int,
	source_inference_batch_size: int,
	shuffle: bool,
	repeat: bool,
	seed: int | None,
	half_n_pix_crop: int,
	norm_psf: str | float | None,
	cache_path: str | None,
	num_parallel_calls: int | None,
	prefetch: bool,
) -> tf.data.Dataset:
	source_input_shape = _resolve_model_input_shape(source_psf_model)
	output_signature = (
		tf.TensorSpec(shape=stage2_input_shape, dtype=tf.float32),
		tf.TensorSpec(shape=(stage2_target_shape[0], stage2_target_shape[1], 2 * stage2_target_shape[2]), dtype=tf.float32),
	)
	ds = tf.data.Dataset.from_generator(
		lambda: _stage2_generator(
			data_dir=Path(data_dir),
			source_psf_model=source_psf_model,
			source_inference_batch_size=source_inference_batch_size,
			half_n_pix_crop=half_n_pix_crop,
			norm_psf=norm_psf,
			source_input_shape=source_input_shape,
		),
		output_signature=output_signature,
	)
	if cache_path:
		Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
		ds = ds.cache(cache_path)
	if shuffle:
		ds = ds.shuffle(buffer_size=1024, seed=seed, reshuffle_each_iteration=True)
	if repeat:
		ds = ds.repeat()
	ds = ds.batch(batch_size, drop_remainder=False)
	ds = ds.filter(
		lambda x, y: tf.reduce_all(tf.math.is_finite(x)) & tf.reduce_all(tf.math.is_finite(y))
	)
	if prefetch:
		ds = ds.prefetch(tf.data.AUTOTUNE)
	return ds


def _infer_shapes_from_batch(batch) -> tuple[tuple[int, int, int], tuple[int, int, int], int]:
	x, packed_target = batch
	input_shape = tuple(x.shape[1:])
	packed_shape = tuple(packed_target.shape[1:])
	if packed_shape[-1] % 2 != 0:
		raise ValueError(f"Expected packed target channels to be even, got {packed_shape[-1]}")
	output_channels = int(packed_shape[-1] // 2)
	output_shape = (int(packed_shape[0]), int(packed_shape[1]), output_channels)
	return input_shape, output_shape, output_channels


def _build_stage2_model(
	*,
	model_name: str,
	arch_config: dict,
	input_shape: tuple[int, int, int],
	output_shape: tuple[int, int, int],
) -> tf.keras.Model:
	_BUILDERS = {
		"gpkh": build_gpkh,
		"gpkh_convdecoder": build_gpkh_convdecoder,
		"skh": build_skh,
		"dense": build_dense_psf,
		"unet": build_unet,
	}
	builder = _BUILDERS.get(model_name)
	if builder is None:
		raise ValueError(f"Unsupported MODEL_NAME={model_name!r}")
	kwargs = dict(arch_config)
	kwargs.update({
		"input_shape": input_shape,
		"output_shape": output_shape,
		"output_activation_function": "linear",
	})
	return builder(**kwargs)


def make_stage2_psf_uncertainty_loss(
	*,
	log_sigma: bool,
	log_min: float,
	log_max: float,
	sigma2_eps: float,
) -> tf.keras.losses.Loss:
	log_sigma = bool(log_sigma)
	log_min = float(log_min)
	log_max = float(log_max)
	sigma2_eps = float(sigma2_eps)

	def _unpack(y_true: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
		channels = tf.shape(y_true)[-1]
		n_frames = channels // 2
		truth = y_true[..., :n_frames]
		mean = y_true[..., n_frames:]
		return truth, mean

	def loss_components(y_true: tf.Tensor, y_pred: tf.Tensor) -> dict[str, tf.Tensor]:
		truth_psf, fixed_mean_psf = _unpack(y_true)
		raw_unc = tf.convert_to_tensor(y_pred)
		with tf.control_dependencies(
			[
				tf.debugging.assert_equal(
					tf.shape(raw_unc)[-1],
					tf.shape(truth_psf)[-1],
					message="Stage-2 model output channels must match PSF channels",
				)
			]
		):
			raw_unc = tf.identity(raw_unc)

		if log_sigma:
			log_sigma2_psf = _log10_clip_to_ln(raw_unc, log_min, log_max)
			sigma2_psf = tf.exp(log_sigma2_psf) + tf.cast(sigma2_eps, raw_unc.dtype)
		else:
			sigma2_psf = tf.nn.softplus(raw_unc) + tf.cast(sigma2_eps, raw_unc.dtype)
			log_sigma2_psf = _log10_clip_to_ln(tf.math.log(sigma2_psf), log_min, log_max)
			sigma2_psf = tf.exp(log_sigma2_psf) + tf.cast(sigma2_eps, raw_unc.dtype)

		err2_psf = tf.square(truth_psf - fixed_mean_psf)
		nll_residual_term = tf.reduce_mean(err2_psf / sigma2_psf)
		nll_logsigma_term = tf.reduce_mean(log_sigma2_psf)
		nll_psf_unc = nll_residual_term + nll_logsigma_term
		psf_var_normalized_mse = _var_normalized_mse(truth_psf, fixed_mean_psf)

		return {
			"nll_psf_unc": nll_psf_unc,
			"nll_residual_term": nll_residual_term,
			"nll_logsigma_term": nll_logsigma_term,
			"psf_var_normalized_mse": psf_var_normalized_mse,
		}

	def _loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
		return loss_components(y_true, y_pred)["nll_psf_unc"]

	_loss.components = loss_components
	_loss.component_names = [
		"nll_residual_term",
		"nll_logsigma_term",
		"psf_var_normalized_mse",
	]
	return _loss


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Stage-2 PSF uncertainty training.")
	parser.add_argument("--config", type=Path, required=True, help="Path to experiment config .py file")
	return parser.parse_args()


def main() -> None:
	global _TEMP_ROOT
	args = _parse_args()
	cfg = load_experiment_config(args.config)

	psf_unc_config = dict(cfg.PSF_UNC_CONFIG)
	psf_head_config = dict(cfg.PSF_HEAD_CONFIG)
	dataset_config = dict(cfg.DATASET_LOAD_CONFIG)
	loss_config = dict(cfg.LOSS_CONFIG)
	output_base_dir = Path(cfg.OUTPUT_BASE_DIR)

	run_name = str(psf_unc_config.get("run_name", "psf_uncertainty_stage2"))
	model_name = str(psf_unc_config.get("model_name", DEFAULT_MODEL_NAME)).strip().lower()
	if model_name not in ALLOWED_MODEL_NAMES:
		raise ValueError(f"model_name must be one of {ALLOWED_MODEL_NAMES}, got {model_name!r}")
	arch_config = extract_arch_config(psf_unc_config)
	training_hparams = extract_training_config(psf_unc_config)

	run_dir = output_base_dir / run_name
	run_dir.mkdir(parents=True, exist_ok=True)
	_TEMP_ROOT = run_dir / "tmp"
	_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
	os.environ.setdefault("TMPDIR", str(_TEMP_ROOT))
	os.environ.setdefault("TEMP", str(_TEMP_ROOT))
	os.environ.setdefault("TMP", str(_TEMP_ROOT))
	print(f"[psf_uncertainty_stage2] Output dir: {run_dir}")

	source_model_path = _resolve_source_psf_model_path(
		output_base_dir=output_base_dir,
		psf_head_config=psf_head_config,
		psf_unc_config=psf_unc_config,
	)
	source_run_dir = source_model_path.parent.parent if source_model_path.parent.name == "checkpoints" else source_model_path.parent
	print(f"[psf_uncertainty_stage2] Source PSF model: {source_model_path}")
	cache_dir = run_dir / "cache"
	train_cache_path = str(cache_dir / "train_stage2.cache")
	val_cache_path = str(cache_dir / "val_stage2.cache")
	_clear_cache_prefix(train_cache_path)
	_clear_cache_prefix(val_cache_path)
	source_inference_batch_size = max(1, int(dataset_config["batch_size"]))

	data_root = Path(str(dataset_config["data_dir"])).expanduser().resolve()
	print(f"[psf_uncertainty_stage2] Dataset root: {data_root}")
	preview_raw = _load_preview_raw_from_first_tfrecord(data_root / "train")
	source_psf_model = _keras_load_psf_model(source_model_path, preview_raw=preview_raw)
	source_psf_model.trainable = False
	for layer in source_psf_model.layers:
		layer.trainable = False
	source_input_shape = _resolve_model_input_shape(source_psf_model)
	input_shape, output_shape, output_channels = _infer_stage2_shapes_from_preview(
		preview_raw=preview_raw,
		half_n_pix_crop=int(dataset_config["half_n_pix_crop"]),
		norm_psf=dataset_config["norm_psf"],
		source_input_shape=source_input_shape,
	)
	train_ds = make_stage2_dataset(
		data_root / "train",
		source_psf_model=source_psf_model,
		stage2_input_shape=input_shape,
		stage2_target_shape=output_shape,
		batch_size=int(dataset_config["batch_size"]),
		source_inference_batch_size=source_inference_batch_size,
		shuffle=bool(dataset_config["shuffle"]),
		repeat=bool(dataset_config["repeat"]),
		seed=None if dataset_config["seed"] is None else int(dataset_config["seed"]),
		half_n_pix_crop=int(dataset_config["half_n_pix_crop"]),
		norm_psf=dataset_config["norm_psf"],
		cache_path=train_cache_path,
		num_parallel_calls=dataset_config.get("num_parallel_calls", None),
		prefetch=bool(dataset_config["prefetch"]),
	)
	val_ds = make_stage2_dataset(
		data_root / "val",
		source_psf_model=source_psf_model,
		stage2_input_shape=input_shape,
		stage2_target_shape=output_shape,
		batch_size=int(dataset_config.get("val_batch_size", dataset_config["batch_size"])),
		source_inference_batch_size=source_inference_batch_size,
		shuffle=False,
		repeat=False,
		seed=None,
		half_n_pix_crop=int(dataset_config["half_n_pix_crop"]),
		norm_psf=dataset_config["norm_psf"],
		cache_path=val_cache_path,
		num_parallel_calls=dataset_config.get("num_parallel_calls", None),
		prefetch=bool(dataset_config["prefetch"]),
	)

	print(
		f"[psf_uncertainty_stage2] Input shape: {input_shape} | Output shape: {output_shape} | output_channels={output_channels} | model={model_name} | source_inference_batch_size={source_inference_batch_size}"
	)

	print("[psf_uncertainty_stage2] Building model")
	model = _build_stage2_model(
		model_name=model_name,
		arch_config=arch_config,
		input_shape=input_shape,
		output_shape=output_shape,
	)

	print("[psf_uncertainty_stage2] Creating loss")
	loss = make_stage2_psf_uncertainty_loss(
		log_sigma=bool(loss_config.get("log_sigma", False)),
		log_min=float(loss_config.get("log_min", -20.0)),
		log_max=float(loss_config.get("log_max", 10.0)),
		sigma2_eps=float(loss_config.get("sigma2_eps", 1e-6)),
	)

	checkpoint_path = run_dir / "checkpoints" / (
		"gpkh_best.keras" if model_name == "gpkh" else
		"gpkh_convdecoder_best.keras" if model_name == "gpkh_convdecoder" else
		"skh_best.keras" if model_name == "skh" else
		"best.keras" if model_name == "dense" else
		"unet_best.keras"
	)
	checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
	train_cfg = dict(training_hparams)
	train_cfg["checkpoint_path"] = str(checkpoint_path)

	print("[psf_uncertainty_stage2] Training")
	result = train_unet(
		model,
		loss,
		train_ds,
		val_dataset=val_ds,
		use_pinn=False,
		**train_cfg,
	)

	print("[psf_uncertainty_stage2] Saving final model")
	model_path = run_dir / "model_final.keras"
	model.save(model_path)

	print("[psf_uncertainty_stage2] Saving metrics")
	metrics_dir = run_dir / "metrics"
	metrics_dir.mkdir(parents=True, exist_ok=True)
	for key, values in result["history"].history.items():
		np.save(metrics_dir / f"history_{key}.npy", np.asarray(values))
	for key, values in result.get("subloss_history", {}).items():
		np.save(metrics_dir / f"subloss_{key}.npy", np.asarray(values))
	for key, values in result.get("batch_history", {}).items():
		np.save(metrics_dir / f"history_batch_{key}.npy", np.asarray(values))
	np.save(metrics_dir / "best_metric.npy", np.asarray(result["best_metric"]))
	np.save(metrics_dir / "best_value.npy", np.asarray(result["best_value"]))
	np.save(metrics_dir / "best_epoch.npy", np.asarray(result["best_epoch"]))
	np.save(metrics_dir / "lr_history.npy", np.asarray(result["lr_history"]))
	np.save(metrics_dir / "duration_s.npy", np.asarray(result["duration_s"]))
	np.save(metrics_dir / "checkpoint_path.npy", np.asarray(result["checkpoint_path"]))
	np.save(metrics_dir / "model_path.npy", np.asarray(str(model_path)))

	print("[psf_uncertainty_stage2] Saving run config")
	source_run_config = _load_json(source_run_dir / "training_config.json")
	source_psf_model_label = str(psf_unc_config.get("source_psf_model_label", "best_model"))
	config_path = run_dir / "training_config.json"
	with config_path.open("w", encoding="utf-8") as handle:
		json.dump(
			{
				"run": run_name,
				"workflow": "psf_uncertainty_stage2_training",
				"dataset": dataset_config,
				"loss": {
					"type": "gaussian_nll_fixed_mean_psf",
					"log_sigma": bool(loss_config.get("log_sigma", False)),
					"log_min": float(loss_config.get("log_min", -20.0)),
					"log_max": float(loss_config.get("log_max", 10.0)),
					"sigma2_eps": float(loss_config.get("sigma2_eps", 1e-6)),
				},
				"model": {
					"name": model_name,
					"input_shape": input_shape,
					"output_shape": output_shape,
					"output_channels": output_channels,
					"explicit_output_normalization": False,
				},
				model_name: arch_config,
				"training": train_cfg,
				"source_psf_model": {
					"path": str(source_model_path),
					"run_dir": str(source_run_dir),
					"model_label": source_psf_model_label,
					"frozen": True,
					"training_config": source_run_config,
				},
				"stage2_features": {
					"layout": "concat_channels(obs, pred_psf_mean)",
					"target_layout": "concat_channels(true_psf, pred_psf_mean)",
					"cache_mode": "on_the_fly_file_cache",
					"source_inference_batch_size": source_inference_batch_size,
					"train_cache_path": train_cache_path,
					"val_cache_path": val_cache_path,
				},
			},
			handle,
			indent=2,
		)

	print("[psf_uncertainty_stage2] Generating training plots")
	plots_dir = run_dir / "plots"
	plot_training_outputs(metrics_dir, plots_dir)

	print(f"[psf_uncertainty_stage2] Done. Wrote run to: {run_dir}")


if __name__ == "__main__":
	main()
