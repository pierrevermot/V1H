from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import tensorflow as tf

from neural_networks.layers import GroupNormalization, _upsample_bilinear
from utils.model_utils import _wrap_model_output_activation
from utils.temp_paths import _resolve_temp_root


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HEAD_MODEL_BEST_CANDIDATES = (
	"unet_best.keras", "gpkh_best.keras", "gpkh_convdecoder_best.keras",
	"skh_best.keras", "best.keras",
)

ALLOWED_MODEL_LABELS = {"best_model", "final_model"}


# ---------------------------------------------------------------------------
# Checkpoint / model path helpers
# ---------------------------------------------------------------------------

def _checkpoint_filename(model_name: str) -> str:
	if model_name == "gpkh": return "gpkh_best.keras"
	if model_name == "gpkh_convdecoder": return "gpkh_convdecoder_best.keras"
	if model_name == "skh": return "skh_best.keras"
	if model_name == "dense": return "best.keras"
	return "unet_best.keras"


def _resolve_model_paths(run_dir: Path) -> dict[str, Path]:
	best_path = None
	for filename in HEAD_MODEL_BEST_CANDIDATES:
		candidate = run_dir / "checkpoints" / filename
		if candidate.exists():
			best_path = candidate
			break
	final_path = run_dir / "model_final.keras"
	result: dict[str, Path] = {}
	if best_path is not None:
		result["best_model"] = best_path
	if final_path.exists():
		result["final_model"] = final_path
	if not result:
		raise FileNotFoundError(f"Could not find best/final model under {run_dir}")
	return result


# ---------------------------------------------------------------------------
# Archive introspection / sanitization
# ---------------------------------------------------------------------------

def _infer_wrapper_info(config: dict[str, Any]) -> dict[str, Any] | None:
	layers = config.get("config", {}).get("layers", [])
	for layer in layers:
		if layer.get("class_name") != "Activation":
			continue
		name = str(layer.get("config", {}).get("name", ""))
		if not name.startswith("pred_main_"):
			continue
		inbound_nodes = layer.get("inbound_nodes", [])
		if not inbound_nodes:
			continue
		args = inbound_nodes[0].get("args", [])
		if not args:
			continue
		shape = args[0].get("config", {}).get("shape", [])
		if not shape:
			continue
		activation_name = str(layer.get("config", {}).get("activation") or name.removeprefix("pred_main_"))
		return {
			"activation_name": activation_name,
			"nll": True,
			"output_channels": int(shape[-1]),
		}
	return None


def _sanitize_archive_config(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
	config = json.loads(json.dumps(config))
	config["compile_config"] = None
	wrapper_info = _infer_wrapper_info(config)
	if wrapper_info is None:
		return config, None

	layers = config.get("config", {}).get("layers", [])
	kept_layers: list[dict[str, Any]] = []
	base_output_name: str | None = None
	for layer in layers:
		name = str(layer.get("config", {}).get("name", ""))
		if name == "pred_main_slice":
			inbound_nodes = layer.get("inbound_nodes", [])
			if inbound_nodes and inbound_nodes[0].get("args"):
				base_output_name = inbound_nodes[0]["args"][0]["config"]["keras_history"][0]
			continue
		if name in {"pred_unc_slice", "pred_concat_with_unc"} or name.startswith("pred_main_"):
			continue
		kept_layers.append(layer)

	if base_output_name is None:
		raise ValueError("Could not determine base output layer while sanitizing model archive")

	config["config"]["layers"] = kept_layers
	config["config"]["output_layers"] = [base_output_name, 0, 0]
	return config, wrapper_info


# ---------------------------------------------------------------------------
# Model loading from Keras archives
# ---------------------------------------------------------------------------

def _load_model_from_keras_archive(
	path: Path,
	*,
	sanitize_wrapped_outputs: bool = True,
) -> tf.keras.Model:
	custom_objects = {
		"_upsample_bilinear": _upsample_bilinear,
		"GroupNormalization": GroupNormalization,
	}
	with ZipFile(path, "r") as archive:
		config = json.loads(archive.read("config.json"))
		if sanitize_wrapped_outputs:
			archive_config, wrapper_info = _sanitize_archive_config(config)
		else:
			archive_config = json.loads(json.dumps(config))
			archive_config["compile_config"] = None
			wrapper_info = None
		config_json = json.dumps(archive_config)
		with tempfile.TemporaryDirectory(prefix="joint_pinn_fourhead_archive_", dir=str(_resolve_temp_root())) as tmp_dir:
			weights_path = Path(tmp_dir) / "model.weights.h5"
			weights_path.write_bytes(archive.read("model.weights.h5"))
			try:
				model = tf.keras.models.model_from_json(
					config_json,
					custom_objects=custom_objects,
					safe_mode=False,
				)
			except TypeError:
				config_api = getattr(tf.keras, "config", None)
				if config_api is not None and hasattr(config_api, "enable_unsafe_deserialization"):
					config_api.enable_unsafe_deserialization()
				model = tf.keras.models.model_from_json(config_json, custom_objects=custom_objects)
			model.load_weights(weights_path)

	if wrapper_info is not None:
		model = _wrap_model_output_activation(model, **wrapper_info)
	return model


def _keras_load_model(path: Path, *, sanitize_archive_wrappers: bool = True) -> tf.keras.Model:
	custom_objects = {
		"_upsample_bilinear": _upsample_bilinear,
		"GroupNormalization": GroupNormalization,
	}
	if path.suffix == ".keras":
		try:
			return tf.keras.models.load_model(
				path,
				compile=False,
				custom_objects=custom_objects,
				safe_mode=False,
			)
		except Exception:
			return _load_model_from_keras_archive(
				path,
				sanitize_wrapped_outputs=sanitize_archive_wrappers,
			)
	try:
		return tf.keras.models.load_model(
			path,
			compile=False,
			custom_objects=custom_objects,
			safe_mode=False,
		)
	except TypeError:
		return tf.keras.models.load_model(
			path,
			compile=False,
			custom_objects=custom_objects,
		)


# ---------------------------------------------------------------------------
# Archive spec inference
# ---------------------------------------------------------------------------

def _infer_model_spec_from_keras_archive(path: Path) -> dict[str, Any]:
	with ZipFile(path, "r") as archive:
		config = json.loads(archive.read("config.json"))

	layers = config.get("config", {}).get("layers", [])
	input_shape: tuple[int, int, int] | None = None
	reshape_targets: list[tuple[int, int, int]] = []
	conv_layers: list[dict[str, Any]] = []
	dense_layers: list[dict[str, Any]] = []
	activation_name = "linear"
	normalization = "none"
	group_norm_groups = 8
	normalize_output_sum = False
	normalize_with_first = False
	normalize_first_only = False
	normalize_by_mean = False
	weight_decay = 0.0

	for layer in layers:
		class_name = str(layer.get("class_name", ""))
		layer_cfg = dict(layer.get("config", {}))
		name = str(layer_cfg.get("name", ""))

		if class_name == "InputLayer":
			batch_shape = layer_cfg.get("batch_shape") or layer_cfg.get("batch_input_shape")
			if batch_shape and len(batch_shape) == 4:
				input_shape = tuple(int(v) for v in batch_shape[1:])
		elif class_name == "Conv2D":
			conv_layers.append(layer_cfg)
			regularizer = layer_cfg.get("kernel_regularizer")
			if isinstance(regularizer, dict):
				reg_cfg = regularizer.get("config", {})
				if "l2" in reg_cfg and reg_cfg["l2"] is not None:
					weight_decay = float(reg_cfg["l2"])
		elif class_name == "Dense":
			dense_layers.append(layer_cfg)
		elif class_name == "Activation" and name.startswith("activation") and activation_name == "linear":
			activation_name = str(layer_cfg.get("activation", "linear")).strip().lower() or "linear"
		elif class_name == "LeakyReLU":
			activation_name = "leakyrelu"
		elif class_name == "BatchNormalization":
			normalization = "batch"
		elif class_name == "GroupNormalization":
			groups = int(layer_cfg.get("groups", 8))
			group_norm_groups = groups
			if conv_layers and int(conv_layers[-1].get("filters", 0)) == groups:
				normalization = "instance"
			else:
				normalization = "group"
		elif class_name == "Reshape":
			target_shape = layer_cfg.get("target_shape")
			if target_shape and len(target_shape) == 3:
				reshape_targets.append(tuple(int(v) for v in target_shape))
		elif class_name == "Lambda":
			if name == "output_softplus":
				normalize_output_sum = True
			if name == "output_normalized_first_sum1":
				normalize_output_sum = True
				normalize_with_first = True
			elif name == "output_normalized_sum1":
				normalize_output_sum = True
			elif name == "output_normalize_first_only":
				normalize_first_only = True
			elif name == "output_normalize_by_mean":
				normalize_by_mean = True

	model_name = str(config.get("config", {}).get("name", "")).strip().lower() or path.stem.replace("_best", "")
	base_filters = int(conv_layers[0].get("filters", 32)) if conv_layers else 32
	layers_per_block = 1
	if conv_layers:
		stride1_before_first_down = 0
		for conv_cfg in conv_layers:
			strides = tuple(int(v) for v in conv_cfg.get("strides", (1, 1)))
			if strides == (1, 1):
				stride1_before_first_down += 1
			else:
				break
		layers_per_block = max(1, stride1_before_first_down)

	latent_dim = None
	if len(dense_layers) >= 2:
		latent_dim = int(dense_layers[-2].get("units", dense_layers[-1].get("units", 128)))

	archive_output_shape = reshape_targets[-1] if reshape_targets else None
	last_conv_filters = int(conv_layers[-1].get("filters", 0)) if conv_layers else None
	return {
		"model_name": model_name,
		"input_shape": input_shape,
		"archive_output_shape": archive_output_shape,
		"last_conv_filters": last_conv_filters,
		"inner_activation_function": activation_name,
		"normalization": normalization,
		"group_norm_groups": group_norm_groups,
		"weight_decay": weight_decay,
		"base_filters": base_filters,
		"layers_per_block": layers_per_block,
		"latent_dim": latent_dim,
		"normalize_output_sum": normalize_output_sum,
		"normalize_with_first": normalize_with_first,
		"normalize_first_only": normalize_first_only,
		"normalize_by_mean": normalize_by_mean,
	}


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def _load_weights_into_rebuilt_model(model: tf.keras.Model, path: Path) -> tf.keras.Model:
	if path.suffix == ".keras":
		with ZipFile(path, "r") as archive:
			with tempfile.TemporaryDirectory(prefix="joint_pinn_fourhead_weights_", dir=str(_resolve_temp_root())) as tmp_dir:
				weights_path = Path(tmp_dir) / "model.weights.h5"
				weights_path.write_bytes(archive.read("model.weights.h5"))
				model.load_weights(weights_path)
		return model
	model.load_weights(path)
	return model


# ---------------------------------------------------------------------------
# High-level model loaders (plot_results versions with fallback params)
# ---------------------------------------------------------------------------

def _load_independent_head_model(
	path: Path,
	*,
	fallback_input_shape: tuple[int, int, int] | None = None,
	fallback_output_shape: tuple[int, int, int] | None = None,
) -> tuple[tf.keras.Model, dict[str, Any]]:
	from utils.model_building import _build_independent_head_model
	from utils.io import _load_head_config

	head_dir = path.parent.parent if path.parent.name == "checkpoints" else path.parent
	head_cfg = _load_head_config(head_dir)
	archive_spec = _infer_model_spec_from_keras_archive(path) if path.suffix == ".keras" else {}
	model = _build_independent_head_model(
		head_dir,
		head_cfg,
		archive_spec=archive_spec,
		fallback_input_shape=fallback_input_shape,
		fallback_output_shape=fallback_output_shape,
	)
	return _load_weights_into_rebuilt_model(model, path), head_cfg


def _load_stage2_head_model(
	path: Path,
	*,
	fallback_input_shape: tuple[int, int, int] | None = None,
	fallback_output_shape: tuple[int, int, int] | None = None,
) -> tuple[tf.keras.Model, dict[str, Any]]:
	from utils.model_building import _rebuild_stage2_uncertainty_model
	from utils.io import _load_head_config

	head_dir = path.parent.parent if path.parent.name == "checkpoints" else path.parent
	head_cfg = _load_head_config(head_dir)
	model = _rebuild_stage2_uncertainty_model(
		path,
		head_cfg,
		fallback_input_shape=fallback_input_shape,
		fallback_output_shape=fallback_output_shape,
	)
	return _load_weights_into_rebuilt_model(model, path), head_cfg


# ---------------------------------------------------------------------------
# Joint model path resolution (from plot_results)
# ---------------------------------------------------------------------------

def _resolve_joint_model_paths(run_dir: Path, training_cfg: dict[str, Any]) -> dict[str, Path]:
	training_block = dict(training_cfg.get("training", {}))
	result: dict[str, Path] = {}
	checkpoint_path = str(training_block.get("checkpoint_path", "")).strip()
	if checkpoint_path:
		checkpoint = Path(checkpoint_path).expanduser().resolve()
		if checkpoint.exists():
			result["best_model"] = checkpoint
	default_best = run_dir / "checkpoints" / "joint_pinn_fourhead_best.keras"
	if default_best.exists():
		result.setdefault("best_model", default_best)
	final_path = run_dir / "model_final.keras"
	if final_path.exists():
		result["final_model"] = final_path
	if not result:
		raise FileNotFoundError(f"Could not find best/final joint checkpoints under {run_dir}")
	return result
