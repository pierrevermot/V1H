"""Independent training workflow for image/psf/noise heads.

Usage
-----
    python independent_training.py --config <experiment.py> --head-target im
    python independent_training.py --config <experiment.py> --head-target psf
    python independent_training.py --config <experiment.py> --head-target noise
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from configs.load_config import load_experiment_config, get_head_config, extract_arch_config, extract_training_config
from neural_networks.dataset import make_train_val_datasets
from neural_networks.losses import make_loss
from neural_networks.training import train_unet
from neural_networks.plot_training import plot_training_outputs
from neural_networks.dense_psf import build_dense_psf
from neural_networks.gpkh import build_gpkh
from neural_networks.gpkh_convdecoder import build_gpkh_convdecoder
from neural_networks.skh import build_skh
from neural_networks.unet import build_unet

from utils.metrics import _make_prediction_only_loss, _filter_make_loss_kwargs
from utils.model_io import _checkpoint_filename
from utils.model_utils import _infer_shapes_from_batch, _wrap_model_output_activation


HEAD_SPECS: dict[str, dict[str, Any]] = {
	"im": {
		"fit_im": True,
		"fit_psf": False,
		"fit_noise": False,
		"metric_name": "r2_im",
	},
	"psf": {
		"fit_im": False,
		"fit_psf": True,
		"fit_noise": False,
		"allowed_model_names": ("gpkh", "gpkh_convdecoder", "skh", "dense"),
		"metric_name": "r2_psf",
	},
	"noise": {
		"fit_im": False,
		"fit_psf": False,
		"fit_noise": True,
		"metric_name": "r2_noise",
	},
}


_BUILDERS = {
	"gpkh": build_gpkh,
	"gpkh_convdecoder": build_gpkh_convdecoder,
	"skh": build_skh,
	"dense": build_dense_psf,
}


def _build_model(model_name: str, arch_config: dict, *, input_shape, model_output_shape) -> tf.keras.Model:
	kwargs = dict(arch_config)
	kwargs["input_shape"] = input_shape
	kwargs["output_shape"] = model_output_shape
	kwargs["output_activation_function"] = "linear"
	builder = _BUILDERS.get(model_name, build_unet)
	return builder(**kwargs)


def _run_training(
	*,
	head_target: str,
	head_config: dict,
	dataset_config: dict,
	loss_config: dict,
	output_dir: Path,
) -> dict[str, object]:
	run_name = str(head_config.get("run_name", head_target))
	model_name = str(head_config.get("model_name", "unet")).strip().lower()
	nll = str(head_config.get("loss_mode", "nll")).strip().lower() == "nll"
	output_activation = str(head_config.get("output_activation_function", "linear")).strip().lower()
	arch_config = extract_arch_config(head_config)
	training_hparams = extract_training_config(head_config)
	spec = HEAD_SPECS[head_target]
	fit_im = bool(spec["fit_im"])
	fit_psf = bool(spec["fit_psf"])
	fit_noise = bool(spec["fit_noise"])
	metric_name = str(spec["metric_name"])

	run_dir = output_dir / run_name
	run_dir.mkdir(parents=True, exist_ok=True)
	checkpoint_path = run_dir / "checkpoints" / _checkpoint_filename(model_name)
	checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

	print(f"[independent_training] {run_name}: Loading datasets")
	train_ds, val_ds = make_train_val_datasets(
		dataset_config["data_dir"],
		batch_size=dataset_config["batch_size"],
		val_batch_size=int(dataset_config.get("val_batch_size", dataset_config["batch_size"])),
		val_shuffle=False,
		val_repeat=False,
		shuffle=dataset_config["shuffle"],
		repeat=dataset_config["repeat"],
		seed=dataset_config["seed"],
		channels_last=dataset_config["channels_last"],
		half_n_pix_crop=dataset_config["half_n_pix_crop"],
		fit_im=fit_im,
		fit_psf=fit_psf,
		fit_noise=fit_noise,
		norm_psf=dataset_config["norm_psf"],
		norm_noise=dataset_config["norm_noise"],
		num_parallel_calls=dataset_config["num_parallel_calls"],
		prefetch=dataset_config["prefetch"],
	)

	print(f"[independent_training] {run_name}: Inferring input/output shapes")
	first_batch = next(iter(train_ds.take(1)))
	input_shape, output_shape = _infer_shapes_from_batch(first_batch)
	output_channels = int(output_shape[2])
	model_output_shape = (output_shape[0], output_shape[1], output_channels * (2 if nll else 1))
	print(
		f"[independent_training] {run_name}: Input shape: {input_shape}, Output shape: {output_shape}, nll={nll}, model={model_name}, output_activation={output_activation}"
	)

	print(f"[independent_training] {run_name}: Building model")
	model = _build_model(model_name, arch_config, input_shape=input_shape, model_output_shape=model_output_shape)
	model = _wrap_model_output_activation(
		model,
		activation_name=output_activation,
		output_channels=output_channels,
		nll=nll,
	)

	print(f"[independent_training] {run_name}: Creating loss")
	if nll:
		loss_cfg = dict(loss_config)
		loss_cfg["loss"] = "nll"
		loss_cfg["use_pinn"] = False
		loss_cfg["fit_im"] = fit_im
		loss_cfg["fit_psf"] = fit_psf
		loss_cfg["fit_noise"] = fit_noise
		loss_cfg["norm_psf"] = dataset_config["norm_psf"]
		loss_cfg["norm_noise"] = dataset_config["norm_noise"]
		loss = make_loss(**_filter_make_loss_kwargs(loss_cfg))
	else:
		loss_cfg = {
			"loss": "var_normalized_mse",
			"use_pinn": False,
			"fit_im": fit_im,
			"fit_psf": fit_psf,
			"fit_noise": fit_noise,
			"norm_psf": dataset_config["norm_psf"],
			"norm_noise": dataset_config["norm_noise"],
			"metric_name": metric_name,
		}
		loss = _make_prediction_only_loss(metric_name)

	print(f"[independent_training] {run_name}: Training model")
	train_cfg = dict(training_hparams)
	train_cfg["checkpoint_path"] = str(checkpoint_path)
	result = train_unet(
		model,
		loss,
		train_ds,
		val_dataset=val_ds,
		use_pinn=False,
		best_examples_target_layout=head_target,
		**train_cfg,
	)

	print(f"[independent_training] {run_name}: Saving final model")
	model_path = run_dir / "model_final.keras"
	model.save(model_path)

	print(f"[independent_training] {run_name}: Saving metrics")
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

	print(f"[independent_training] {run_name}: Saving config")
	config_path = run_dir / "training_config.json"
	with config_path.open("w", encoding="utf-8") as handle:
		json.dump(
			{
				"dataset": dataset_config,
				"loss": loss_cfg,
				"model": {"name": model_name},
				model_name: arch_config,
				"training": train_cfg,
				"run": run_name,
				"head_target": head_target,
				"nll": nll,
				"output_activation_function": output_activation,
			},
			handle,
			indent=2,
		)

	print(f"[independent_training] {run_name}: Generating plots")
	plots_dir = run_dir / "plots"
	plot_training_outputs(metrics_dir, plots_dir)

	return result


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Independent training for image/psf/noise heads.")
	parser.add_argument("--config", type=Path, required=True, help="Path to experiment config .py file")
	parser.add_argument("--head-target", required=True, choices=sorted(HEAD_SPECS), help="Head to train")
	return parser.parse_args()


def main() -> None:
	print("[independent_training] Starting workflow")
	args = _parse_args()
	cfg = load_experiment_config(args.config)
	head_target = args.head_target
	dataset_config = dict(cfg.DATASET_LOAD_CONFIG)
	loss_config = dict(cfg.LOSS_CONFIG)
	output_dir = Path(cfg.OUTPUT_BASE_DIR)
	head_config = get_head_config(cfg, head_target)
	output_dir.mkdir(parents=True, exist_ok=True)

	run_name = str(head_config.get("run_name", head_target))
	print(f"[independent_training] Output dir: {output_dir} | head={head_target} | run_name={run_name}")

	_run_training(
		head_target=head_target,
		head_config=head_config,
		dataset_config=dataset_config,
		loss_config=loss_config,
		output_dir=output_dir,
	)

	print("[independent_training] Done")


if __name__ == "__main__":
	main()
