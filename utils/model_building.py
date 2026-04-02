from __future__ import annotations

from pathlib import Path
from typing import Any
from zipfile import ZipFile

import tensorflow as tf

from neural_networks.dense_psf import build_dense_psf
from neural_networks.gpkh import build_gpkh
from neural_networks.gpkh_convdecoder import build_gpkh_convdecoder
from neural_networks.skh import build_skh
from neural_networks.unet import build_unet
from utils.io import _resolve_dataset_root
from utils.tfrecord_io import _first_example_raw_from_dataset
from utils.data_utils import _prepare_truth_arrays
from utils.model_utils import _wrap_model_output_activation
from utils.model_io import _infer_model_spec_from_keras_archive

# Default architecture configs used as fallback when saved training_config.json
# does not contain architecture info.  In practice these are rarely hit because
# independent_training always serializes the arch config under the model name key.
_DEFAULT_ARCH: dict[str, object] = {
	"layers_per_block": 3,
	"base_filters": 32,
	"normalization": "none",
	"group_norm_groups": 8,
	"weight_decay": 0,
	"inner_activation_function": "relu",
	"output_activation_function": "linear",
	"latent_dim": 512,
	"normalize_output_sum": True,
	"normalize_with_first": True,
	"normalize_first_only": False,
	"normalize_by_mean": False,
}


# ---------------------------------------------------------------------------
# Target / shape inference for independent heads
# ---------------------------------------------------------------------------

def _infer_independent_head_target(head_dir: Path, head_cfg: dict[str, Any]) -> str:
	head_target = str(head_cfg.get("head_target", "")).strip().lower()
	if head_target in {"im", "psf", "noise", "res"}:
		if head_target == "res":
			return "noise"
		return head_target
	dir_name = head_dir.name.strip().lower()
	if dir_name in {"image_only", "im", "image"}:
		return "im"
	if dir_name in {"psf_only", "psf"}:
		return "psf"
	if dir_name in {"noise_only", "noise", "residuals_only", "res", "residual"}:
		return "noise"
	raise ValueError(f"Could not infer head target for {head_dir}")


def _infer_independent_head_shapes(head_dir: Path, head_cfg: dict[str, Any]) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
	data_root = _resolve_dataset_root(head_cfg)
	train_dir = data_root / "train"
	val_dir = data_root / "val"
	if train_dir.exists():
		raw = _first_example_raw_from_dataset(train_dir)
	elif val_dir.exists():
		raw = _first_example_raw_from_dataset(val_dir)
	else:
		raw = _first_example_raw_from_dataset(data_root)

	truth = _prepare_truth_arrays(raw, dict(head_cfg.get("dataset", {})))
	input_shape = tuple(int(v) for v in truth["obs_hwf"].shape[1:])
	head_target = _infer_independent_head_target(head_dir, head_cfg)
	if head_target == "im":
		output_shape = tuple(int(v) for v in truth["image_hw1"].shape[1:])
	elif head_target == "psf":
		output_shape = tuple(int(v) for v in truth["psf_hwf"].shape[1:])
	else:
		output_shape = tuple(int(v) for v in truth["noise_hwf"].shape[1:])
	return input_shape, output_shape


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _build_model_by_name(
	model_name: str,
	*,
	input_shape: tuple[int, int, int],
	output_shape: tuple[int, int, int],
	head_cfg: dict[str, Any],
) -> tf.keras.Model:
	model_name = str(model_name).strip().lower()
	if model_name == "gpkh":
		model_kwargs = dict(head_cfg.get("gpkh", _DEFAULT_ARCH))
		model_kwargs["input_shape"] = input_shape
		model_kwargs["output_shape"] = output_shape
		model_kwargs["output_activation_function"] = "linear"
		return build_gpkh(**model_kwargs)
	if model_name == "gpkh_convdecoder":
		model_kwargs = dict(head_cfg.get("gpkh_convdecoder", _DEFAULT_ARCH))
		model_kwargs["input_shape"] = input_shape
		model_kwargs["output_shape"] = output_shape
		model_kwargs["output_activation_function"] = "linear"
		return build_gpkh_convdecoder(**model_kwargs)
	if model_name == "skh":
		model_kwargs = dict(head_cfg.get("skh", _DEFAULT_ARCH))
		model_kwargs["input_shape"] = input_shape
		model_kwargs["output_shape"] = output_shape
		model_kwargs["output_activation_function"] = "linear"
		return build_skh(**model_kwargs)
	if model_name == "dense":
		model_kwargs = dict(head_cfg.get("dense", _DEFAULT_ARCH))
		model_kwargs["input_shape"] = input_shape
		model_kwargs["output_shape"] = output_shape
		model_kwargs["output_activation_function"] = "linear"
		return build_dense_psf(**model_kwargs)
	model_kwargs = dict(head_cfg.get("unet", _DEFAULT_ARCH))
	model_kwargs["input_shape"] = input_shape
	model_kwargs["output_shape"] = output_shape
	model_kwargs["output_activation_function"] = "linear"
	return build_unet(**model_kwargs)


def _build_independent_head_model(
	head_dir: Path,
	head_cfg: dict[str, Any],
	*,
	archive_spec: dict[str, Any] | None = None,
	fallback_input_shape: tuple[int, int, int] | None = None,
	fallback_output_shape: tuple[int, int, int] | None = None,
) -> tf.keras.Model:
	archive_spec = dict(archive_spec or {})
	loss_cfg = dict(head_cfg.get("loss", {}))
	model_name = str(head_cfg.get("model", {}).get("name", archive_spec.get("model_name", "unet"))).strip().lower() or "unet"
	nll = bool(head_cfg.get("nll", loss_cfg.get("loss", "").strip().lower() == "nll"))
	output_activation = str(head_cfg.get("output_activation_function", "linear")).strip().lower() or "linear"
	if dict(head_cfg.get("dataset", {})).get("data_dir"):
		input_shape, output_shape = _infer_independent_head_shapes(head_dir, head_cfg)
	else:
		archive_input_shape = archive_spec.get("input_shape")
		archive_output_shape = archive_spec.get("archive_output_shape")
		input_shape = tuple(int(v) for v in archive_input_shape) if archive_input_shape else None
		output_shape = tuple(int(v) for v in archive_output_shape) if archive_output_shape else None
		if input_shape is None:
			if fallback_input_shape is None:
				raise ValueError(f"Cannot infer input shape for checkpoint-only head at {head_dir}")
			input_shape = fallback_input_shape
		if output_shape is None:
			if fallback_output_shape is None:
				raise ValueError(f"Cannot infer output shape for checkpoint-only head at {head_dir}")
			output_shape = fallback_output_shape
	output_channels = int(output_shape[-1])
	last_conv_filters = archive_spec.get("last_conv_filters")
	if last_conv_filters is not None:
		if int(last_conv_filters) == 2 * output_channels:
			nll = True
		elif int(last_conv_filters) == output_channels:
			nll = False
	model_output_shape = (output_shape[0], output_shape[1], output_channels * (2 if nll else 1))
	model = _build_model_by_name(model_name, input_shape=input_shape, output_shape=model_output_shape, head_cfg=head_cfg)
	return _wrap_model_output_activation(model, activation_name=output_activation, output_channels=output_channels, nll=nll)


# ---------------------------------------------------------------------------
# Stage-2 uncertainty model rebuild
# ---------------------------------------------------------------------------

def _infer_stage2_uncertainty_shapes(
	head_cfg: dict[str, Any],
	*,
	fallback_input_shape: tuple[int, int, int] | None = None,
	fallback_output_shape: tuple[int, int, int] | None = None,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
	model_cfg = dict(head_cfg.get("model", {}))
	input_shape = tuple(int(v) for v in model_cfg.get("input_shape", ()))
	output_shape = tuple(int(v) for v in model_cfg.get("output_shape", ()))
	if len(input_shape) == 3 and len(output_shape) == 3:
		return input_shape, output_shape
	if not dict(head_cfg.get("dataset", {})).get("data_dir"):
		if fallback_input_shape is None or fallback_output_shape is None:
			raise ValueError("Cannot infer stage-2 shapes without dataset metadata or fallback shapes")
		return fallback_input_shape, fallback_output_shape
	data_root = _resolve_dataset_root(head_cfg)
	train_dir = data_root / "train"
	val_dir = data_root / "val"
	if train_dir.exists():
		raw = _first_example_raw_from_dataset(train_dir)
	elif val_dir.exists():
		raw = _first_example_raw_from_dataset(val_dir)
	else:
		raw = _first_example_raw_from_dataset(data_root)
	truth = _prepare_truth_arrays(raw, dict(head_cfg.get("dataset", {})))
	obs_shape = tuple(int(v) for v in truth["obs_hwf"].shape[1:])
	psf_shape = tuple(int(v) for v in truth["psf_hwf"].shape[1:])
	return (obs_shape[0], obs_shape[1], obs_shape[2] + psf_shape[2]), psf_shape


def _rebuild_stage2_uncertainty_model(
	path: Path,
	head_cfg: dict[str, Any],
	*,
	fallback_input_shape: tuple[int, int, int] | None = None,
	fallback_output_shape: tuple[int, int, int] | None = None,
) -> tf.keras.Model:
	archive_spec = _infer_model_spec_from_keras_archive(path) if path.suffix == ".keras" else {}
	if dict(head_cfg.get("dataset", {})).get("data_dir"):
		input_shape, output_shape = _infer_stage2_uncertainty_shapes(
			head_cfg,
			fallback_input_shape=fallback_input_shape,
			fallback_output_shape=fallback_output_shape,
		)
	else:
		archive_input_shape = archive_spec.get("input_shape")
		archive_output_shape = archive_spec.get("archive_output_shape")
		input_shape = tuple(int(v) for v in archive_input_shape) if archive_input_shape else None
		output_shape = tuple(int(v) for v in archive_output_shape) if archive_output_shape else None
		if input_shape is None:
			if fallback_input_shape is None:
				raise ValueError(f"Cannot infer input shape for checkpoint-only stage-2 head at {path}")
			input_shape = fallback_input_shape
		if output_shape is None:
			if fallback_output_shape is None:
				raise ValueError(f"Cannot infer output shape for checkpoint-only stage-2 head at {path}")
			output_shape = fallback_output_shape
	model_name = str(head_cfg.get("model", {}).get("name", "gpkh")).strip().lower() or "gpkh"
	archive_model_name = str(archive_spec.get("model_name", "")).strip().lower()
	if archive_model_name in {"gpkh", "gpkh_convdecoder", "skh", "dense", "unet"}:
		model_name = archive_model_name

	if model_name == "gpkh":
		model_kwargs = dict(head_cfg.get("gpkh", _DEFAULT_ARCH))
		model_kwargs.update(
			{
				"layers_per_block": archive_spec.get("layers_per_block", model_kwargs.get("layers_per_block")),
				"base_filters": archive_spec.get("base_filters", model_kwargs.get("base_filters")),
				"normalization": archive_spec.get("normalization", model_kwargs.get("normalization")),
				"group_norm_groups": archive_spec.get("group_norm_groups", model_kwargs.get("group_norm_groups")),
				"weight_decay": archive_spec.get("weight_decay", model_kwargs.get("weight_decay")),
				"inner_activation_function": archive_spec.get("inner_activation_function", model_kwargs.get("inner_activation_function")),
				"latent_dim": archive_spec.get("latent_dim", model_kwargs.get("latent_dim")),
				"normalize_output_sum": archive_spec.get("normalize_output_sum", model_kwargs.get("normalize_output_sum", False)),
				"normalize_with_first": archive_spec.get("normalize_with_first", model_kwargs.get("normalize_with_first", False)),
				"normalize_first_only": archive_spec.get("normalize_first_only", model_kwargs.get("normalize_first_only", False)),
				"normalize_by_mean": archive_spec.get("normalize_by_mean", model_kwargs.get("normalize_by_mean", False)),
			}
		)
		model_kwargs["input_shape"] = input_shape
		model_kwargs["output_shape"] = output_shape
		model_kwargs["output_activation_function"] = "linear"
		return build_gpkh(**model_kwargs)
	if model_name == "gpkh_convdecoder":
		model_kwargs = dict(head_cfg.get("gpkh_convdecoder", _DEFAULT_ARCH))
		model_kwargs.update(
			{
				"layers_per_block": archive_spec.get("layers_per_block", model_kwargs.get("layers_per_block")),
				"base_filters": archive_spec.get("base_filters", model_kwargs.get("base_filters")),
				"normalization": archive_spec.get("normalization", model_kwargs.get("normalization")),
				"group_norm_groups": archive_spec.get("group_norm_groups", model_kwargs.get("group_norm_groups")),
				"weight_decay": archive_spec.get("weight_decay", model_kwargs.get("weight_decay")),
				"inner_activation_function": archive_spec.get("inner_activation_function", model_kwargs.get("inner_activation_function")),
				"latent_dim": archive_spec.get("latent_dim", model_kwargs.get("latent_dim")),
				"normalize_output_sum": archive_spec.get("normalize_output_sum", model_kwargs.get("normalize_output_sum", False)),
				"normalize_with_first": archive_spec.get("normalize_with_first", model_kwargs.get("normalize_with_first", False)),
			}
		)
		model_kwargs["input_shape"] = input_shape
		model_kwargs["output_shape"] = output_shape
		model_kwargs["output_activation_function"] = "linear"
		return build_gpkh_convdecoder(**model_kwargs)
	if model_name == "skh":
		model_kwargs = dict(head_cfg.get("skh", _DEFAULT_ARCH))
		model_kwargs.update(
			{
				"layers_per_block": archive_spec.get("layers_per_block", model_kwargs.get("layers_per_block")),
				"base_filters": archive_spec.get("base_filters", model_kwargs.get("base_filters")),
				"normalization": archive_spec.get("normalization", model_kwargs.get("normalization")),
				"group_norm_groups": archive_spec.get("group_norm_groups", model_kwargs.get("group_norm_groups")),
				"weight_decay": archive_spec.get("weight_decay", model_kwargs.get("weight_decay")),
				"inner_activation_function": archive_spec.get("inner_activation_function", model_kwargs.get("inner_activation_function")),
				"latent_dim": archive_spec.get("latent_dim", model_kwargs.get("latent_dim")),
				"normalize_output_sum": archive_spec.get("normalize_output_sum", model_kwargs.get("normalize_output_sum", False)),
				"normalize_with_first": archive_spec.get("normalize_with_first", model_kwargs.get("normalize_with_first", False)),
				"normalize_first_only": archive_spec.get("normalize_first_only", model_kwargs.get("normalize_first_only", False)),
				"normalize_by_mean": archive_spec.get("normalize_by_mean", model_kwargs.get("normalize_by_mean", False)),
			}
		)
		model_kwargs["input_shape"] = input_shape
		model_kwargs["output_shape"] = output_shape
		model_kwargs["output_activation_function"] = "linear"
		return build_skh(**model_kwargs)
	if model_name == "dense":
		model_kwargs = dict(head_cfg.get("dense", _DEFAULT_ARCH))
		model_kwargs["input_shape"] = input_shape
		model_kwargs["output_shape"] = output_shape
		model_kwargs["output_activation_function"] = "linear"
		return build_dense_psf(**model_kwargs)
	model_kwargs = dict(head_cfg.get("unet", _DEFAULT_ARCH))
	model_kwargs["input_shape"] = input_shape
	model_kwargs["output_shape"] = output_shape
	model_kwargs["output_activation_function"] = "linear"
	return build_unet(**model_kwargs)
