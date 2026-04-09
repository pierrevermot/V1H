#!/usr/bin/env python3
"""Evaluate a deconvolution backend on val and GalSim datasets.

The script is backend-driven so additional algorithms can be added later by
registering a new EvaluationBackend subclass.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from configs.load_config import load_experiment_config
from neural_networks.dataset import make_dataset
from utils.io import _load_joint_run_config
from utils.metrics import _pred_to_sigma2, _split_pred, _split_truth
from utils.model_io import (
	_load_independent_head_model,
	_load_stage2_head_model,
	_load_weights_into_rebuilt_model,
	_resolve_model_paths,
	_resolve_joint_model_paths,
)
from utils.normalization import _normalize_psf_for_observation
from workflow.joint_pinn_fourhead_training import FourHeadJointPinnModel
from neural_networks.losses import _convolve_image_with_psfs


BACKEND_REGISTRY: dict[str, type["EvaluationBackend"]] = {}


def register_backend(name: str):
	def decorator(cls: type["EvaluationBackend"]):
		BACKEND_REGISTRY[name] = cls
		cls.backend_name = name
		return cls

	return decorator


@dataclass(frozen=True)
class DatasetSpec:
	name: str
	path: Path


class EvaluationBackend:
	backend_name = "unknown"

	@classmethod
	def from_run(
		cls,
		*,
		cfg,
		run_dir: Path,
		model_label: str,
		preview_obs: tf.Tensor,
	) -> "EvaluationBackend":
		raise NotImplementedError

	def evaluate_batch(self, obs: tf.Tensor, y_true: tf.Tensor) -> dict[str, np.ndarray]:
		raise NotImplementedError

	def describe(self) -> dict[str, Any]:
		return {"backend": self.backend_name}


def _build_joint_training_cfg_from_experiment_config(cfg, *, run_dir: Path, model_label: str) -> dict[str, Any]:
	output_base_dir = Path(cfg.OUTPUT_BASE_DIR).expanduser().resolve()
	joint_config = dict(cfg.JOINT_PINN_CONFIG)
	loss_config = dict(cfg.LOSS_CONFIG)
	dataset_config = dict(cfg.DATASET_LOAD_CONFIG)

	image_run_dir = output_base_dir / str(cfg.IMAGE_HEAD_CONFIG.get("run_name", "image_only"))
	noise_run_dir = output_base_dir / str(cfg.NOISE_HEAD_CONFIG.get("run_name", "noise_only"))
	psf_mean_run_dir = output_base_dir / str(cfg.PSF_HEAD_CONFIG.get("run_name", "psf_only"))
	psf_unc_run_dir = output_base_dir / str(cfg.PSF_UNC_CONFIG.get("run_name", "psf_uncertainty_stage2"))

	image_model_path = _resolve_model_paths(image_run_dir)[model_label]
	noise_model_path = _resolve_model_paths(noise_run_dir)[model_label]
	psf_mean_model_path = _resolve_model_paths(psf_mean_run_dir)[model_label]
	psf_unc_model_path = _resolve_model_paths(psf_unc_run_dir)[model_label]
	joint_model_path = _resolve_joint_model_paths(run_dir, {})[model_label]

	return {
		"run": str(joint_config.get("run_name", run_dir.name)),
		"workflow": "joint_pinn_fourhead_training",
		"dataset": dataset_config,
		"loss": {
			"type": "weighted_supervised_nll_plus_pinn_r2",
			"log_sigma": bool(loss_config.get("log_sigma", False)),
			"log_min": float(loss_config.get("log_min", -6.0)),
			"log_max": float(loss_config.get("log_max", 20.0)),
			"sigma2_eps": float(loss_config.get("sigma2_eps", 1e-12)),
			"weights": {
				"pinn": float(joint_config.get("pinn_weight", 1.0)),
				"im": float(joint_config.get("im_weight", 1.0)),
				"psf": float(joint_config.get("psf_weight", 1.0)),
				"noise": float(joint_config.get("noise_weight", 1.0)),
			},
			"reconstruction_crop": int(joint_config.get("reconstruction_crop", 16)),
		},
		"training": {
			"checkpoint_path": str(joint_model_path),
		},
		"source_models": {
			"image": {
				"model_path": str(image_model_path),
				"run_dir": str(image_run_dir),
				"model_label": model_label,
			},
			"noise": {
				"model_path": str(noise_model_path),
				"run_dir": str(noise_run_dir),
				"model_label": model_label,
			},
			"psf_mean": {
				"model_path": str(psf_mean_model_path),
				"run_dir": str(psf_mean_run_dir),
				"model_label": model_label,
			},
			"psf_unc": {
				"model_path": str(psf_unc_model_path),
				"run_dir": str(psf_unc_run_dir),
				"model_label": model_label,
			},
		},
	}


def _per_example_mean(x: tf.Tensor) -> tf.Tensor:
	axes = tuple(range(1, len(x.shape)))
	return tf.reduce_mean(x, axis=axes)


def _per_example_variance(x: tf.Tensor, *, eps: float = 1e-12) -> tf.Tensor:
	axes = tuple(range(1, len(x.shape)))
	return tf.math.reduce_variance(x, axis=axes) + tf.cast(eps, x.dtype)


def _per_example_mse(truth: tf.Tensor, pred: tf.Tensor) -> tf.Tensor:
	return _per_example_mean(tf.square(truth - pred))


def _per_example_r2(truth: tf.Tensor, pred: tf.Tensor) -> tf.Tensor:
	return _per_example_mse(truth, pred) / _per_example_variance(truth)


def _per_example_nll(truth: tf.Tensor, pred: tf.Tensor, sigma2: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
	err2 = tf.square(pred - truth)
	residual = _per_example_mean(err2 / sigma2)
	logsigma2 = _per_example_mean(tf.math.log(sigma2))
	return residual + logsigma2, residual, logsigma2


def _per_example_weighted_residual(truth: tf.Tensor, pred: tf.Tensor, sigma2: tf.Tensor) -> tf.Tensor:
	err2 = tf.square(pred - truth)
	return _per_example_mean(err2 / sigma2)


def _propagate_reconstructed_observation_sigma2(
	*,
	pred_im: tf.Tensor,
	pred_psf_phys: tf.Tensor,
	sigma2_im: tf.Tensor,
	sigma2_psf_phys: tf.Tensor,
	sigma2_noise_phys: tf.Tensor,
	eps: float,
) -> tf.Tensor:
	term_im = _convolve_image_with_psfs(sigma2_im, tf.square(pred_psf_phys))
	term_psf = _convolve_image_with_psfs(tf.square(pred_im), sigma2_psf_phys)
	term_cross = _convolve_image_with_psfs(sigma2_im, sigma2_psf_phys)
	sigma2_obs = term_im + term_psf + term_cross + sigma2_noise_phys
	return tf.maximum(sigma2_obs, tf.cast(eps, sigma2_obs.dtype))


def _concat_components(*arrays: tf.Tensor) -> tf.Tensor:
	return tf.concat(list(arrays), axis=-1)


def _summarize_metric(values: np.ndarray) -> dict[str, float]:
	flat = np.asarray(values, dtype=np.float64).reshape(-1)
	return {
		"min": float(np.min(flat)),
		"max": float(np.max(flat)),
		"mean": float(np.mean(flat)),
		"median": float(np.median(flat)),
		"std": float(np.std(flat)),
	}


def _summarize_basic_stats(values: np.ndarray) -> dict[str, float]:
	flat = np.asarray(values, dtype=np.float64).reshape(-1)
	return {
		"min": float(np.min(flat)),
		"max": float(np.max(flat)),
		"mean": float(np.mean(flat)),
		"median": float(np.median(flat)),
		"std": float(np.std(flat)),
		"sum": float(np.sum(flat)),
	}


def _render_summary_table(title: str, summary: dict[str, dict[str, float]]) -> str:
	metric_width = max(len("metric"), *(len(name) for name in summary))
	header = f"{'metric':<{metric_width}}  {'min':>12}  {'max':>12}  {'mean':>12}  {'median':>12}  {'std':>12}"
	lines = [title, header, "-" * len(header)]
	for metric_name in sorted(summary):
		stats = summary[metric_name]
		lines.append(
			f"{metric_name:<{metric_width}}  "
			f"{stats['min']:>12.6g}  {stats['max']:>12.6g}  {stats['mean']:>12.6g}  {stats['median']:>12.6g}  {stats['std']:>12.6g}"
		)
	return "\n".join(lines)


def _render_component_stats_table(title: str, summary: dict[str, dict[str, float]]) -> str:
	metric_width = max(len("component"), *(len(name) for name in summary))
	header = (
		f"{'component':<{metric_width}}  {'min':>12}  {'max':>12}  {'mean':>12}  "
		f"{'median':>12}  {'std':>12}  {'sum':>12}"
	)
	lines = [title, header, "-" * len(header)]
	for component_name in sorted(summary):
		stats = summary[component_name]
		lines.append(
			f"{component_name:<{metric_width}}  "
			f"{stats['min']:>12.6g}  {stats['max']:>12.6g}  {stats['mean']:>12.6g}  "
			f"{stats['median']:>12.6g}  {stats['std']:>12.6g}  {stats['sum']:>12.6g}"
		)
	return "\n".join(lines)


def _summarize_dataset_components(
	dataset: tf.data.Dataset,
	*,
	max_examples: int,
) -> tuple[dict[str, dict[str, float]], int]:
	if max_examples <= 0:
		return {}, 0

	component_chunks: dict[str, list[np.ndarray]] = {
		"image": [],
		"noise": [],
		"obs": [],
		"psf": [],
	}
	seen_examples = 0
	for obs_batch, y_true_batch in dataset:
		batch_size = int(obs_batch.shape[0])
		if batch_size <= 0:
			continue
		remaining = max_examples - seen_examples
		if remaining <= 0:
			break
		take = min(batch_size, remaining)
		obs_slice = tf.convert_to_tensor(obs_batch[:take], dtype=tf.float32)
		y_true_slice = tf.convert_to_tensor(y_true_batch[:take], dtype=tf.float32)
		truth_im, truth_psf, truth_noise, _ = _split_truth(y_true_slice)
		component_chunks["obs"].append(np.asarray(obs_slice, dtype=np.float32).reshape(-1))
		component_chunks["image"].append(np.asarray(truth_im, dtype=np.float32).reshape(-1))
		component_chunks["psf"].append(np.asarray(truth_psf, dtype=np.float32).reshape(-1))
		component_chunks["noise"].append(np.asarray(truth_noise, dtype=np.float32).reshape(-1))
		seen_examples += take
		if seen_examples >= max_examples:
			break

	if seen_examples == 0:
		return {}, 0

	return {
		name: _summarize_basic_stats(np.concatenate(chunks, axis=0))
		for name, chunks in component_chunks.items()
		if chunks
	}, seen_examples


def _render_comparison_table(
	*,
	val_summary: dict[str, dict[str, float]],
	galsim_summary: dict[str, dict[str, float]],
) -> str:
	metric_names = sorted(set(val_summary) & set(galsim_summary))
	metric_width = max(len("metric"), *(len(name) for name in metric_names))
	header = (
		f"{'metric':<{metric_width}}  {'val_mean':>12}  {'galsim_mean':>12}  {'delta_mean':>12}  "
		f"{'val_median':>12}  {'galsim_median':>12}  {'delta_median':>12}"
	)
	lines = ["Comparison (Val vs GalSim)", header, "-" * len(header)]
	for metric_name in metric_names:
		val_stats = val_summary[metric_name]
		galsim_stats = galsim_summary[metric_name]
		lines.append(
			f"{metric_name:<{metric_width}}  "
			f"{val_stats['mean']:>12.6g}  {galsim_stats['mean']:>12.6g}  {galsim_stats['mean'] - val_stats['mean']:>12.6g}  "
			f"{val_stats['median']:>12.6g}  {galsim_stats['median']:>12.6g}  {galsim_stats['median'] - val_stats['median']:>12.6g}"
		)
	return "\n".join(lines)


@register_backend("joint_pinn")
class JointPinnBackend(EvaluationBackend):
	def __init__(
		self,
		*,
		model: FourHeadJointPinnModel,
		run_dir: Path,
		model_path: Path,
		training_cfg: dict[str, Any],
	):
		self.model = model
		self.run_dir = run_dir
		self.model_path = model_path
		self.training_cfg = training_cfg
		loss_cfg = dict(training_cfg.get("loss", {}))
		weights_cfg = dict(loss_cfg.get("weights", {}))
		self.im_weight = float(weights_cfg.get("im", 1.0))
		self.psf_weight = float(weights_cfg.get("psf", 1.0))
		self.noise_weight = float(weights_cfg.get("noise", 1.0))
		self.pinn_weight = float(weights_cfg.get("pinn", 1.0))

	@classmethod
	def from_run(
		cls,
		*,
		cfg,
		run_dir: Path,
		model_label: str,
		preview_obs: tf.Tensor,
	) -> "JointPinnBackend":
		training_cfg = _load_joint_run_config(run_dir)
		if not training_cfg:
			training_cfg = _build_joint_training_cfg_from_experiment_config(
				cfg,
				run_dir=run_dir,
				model_label=model_label,
			)
		joint_model_path = _resolve_joint_model_paths(run_dir, training_cfg)[model_label]

		source_models = dict(training_cfg.get("source_models", {}))
		image_cfg = dict(source_models.get("image", {}))
		noise_cfg = dict(source_models.get("noise", source_models.get("residual", {})))
		psf_mean_cfg = dict(source_models.get("psf_mean", {}))
		psf_unc_cfg = dict(source_models.get("psf_unc", {}))

		image_model_path = Path(str(image_cfg.get("model_path", ""))).expanduser().resolve()
		noise_model_path = Path(str(noise_cfg.get("model_path", ""))).expanduser().resolve()
		psf_mean_model_path = Path(str(psf_mean_cfg.get("model_path", ""))).expanduser().resolve()
		psf_unc_model_path = Path(str(psf_unc_cfg.get("model_path", ""))).expanduser().resolve()

		preview_input_shape = tuple(int(v) for v in preview_obs.shape[1:])
		n_frames = int(preview_input_shape[-1])
		spatial_shape = preview_input_shape[:2]
		image_output_shape = (spatial_shape[0], spatial_shape[1], 1)
		frame_output_shape = (spatial_shape[0], spatial_shape[1], n_frames)
		stage2_input_shape = (spatial_shape[0], spatial_shape[1], 2 * n_frames)

		image_model, _ = _load_independent_head_model(
			image_model_path,
			fallback_input_shape=preview_input_shape,
			fallback_output_shape=image_output_shape,
		)
		noise_model, _ = _load_independent_head_model(
			noise_model_path,
			fallback_input_shape=preview_input_shape,
			fallback_output_shape=frame_output_shape,
		)
		psf_mean_model, psf_mean_head_cfg = _load_independent_head_model(
			psf_mean_model_path,
			fallback_input_shape=preview_input_shape,
			fallback_output_shape=frame_output_shape,
		)
		psf_unc_model, psf_unc_head_cfg = _load_stage2_head_model(
			psf_unc_model_path,
			fallback_input_shape=stage2_input_shape,
			fallback_output_shape=frame_output_shape,
		)

		loss_cfg = dict(training_cfg.get("loss", {}))
		dataset_cfg = dict(training_cfg.get("dataset", {}))
		weights_cfg = dict(loss_cfg.get("weights", {}))
		model = FourHeadJointPinnModel(
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
		_ = model(tf.convert_to_tensor(preview_obs, dtype=tf.float32), training=False)
		model = _load_weights_into_rebuilt_model(model, joint_model_path)
		return cls(model=model, run_dir=run_dir, model_path=joint_model_path, training_cfg=training_cfg)

	def describe(self) -> dict[str, Any]:
		return {
			"backend": self.backend_name,
			"run_dir": str(self.run_dir),
			"model_path": str(self.model_path),
		}

	def _per_example_pinn_r2(
		self,
		obs: tf.Tensor,
		pred_im: tf.Tensor,
		pred_psf: tf.Tensor,
		pred_noise: tf.Tensor,
	) -> tf.Tensor:
		psf_df = tf.cast(self.model._psf_denorm_factor, pred_psf.dtype)
		noise_df = tf.cast(self.model._noise_denorm_factor, pred_noise.dtype)
		pred_psf_phys = pred_psf / psf_df
		pred_psf_phys, _, _ = _normalize_psf_for_observation(pred_psf_phys)
		pred_noise_phys = pred_noise / noise_df
		pred_obs = _convolve_image_with_psfs(pred_im, pred_psf_phys) - pred_noise_phys
		if self.model.reconstruction_crop > 0:
			c = int(self.model.reconstruction_crop)
			pred_obs = pred_obs[:, c:-c, c:-c, :]
			obs = obs[:, c:-c, c:-c, :]
		return _per_example_mean(tf.square(pred_obs - obs)) / _per_example_variance(obs)

	def _predicted_observation_and_sigma2(
		self,
		pred_im: tf.Tensor,
		pred_psf: tf.Tensor,
		pred_noise: tf.Tensor,
		sigma2_im: tf.Tensor,
		sigma2_psf: tf.Tensor,
		sigma2_noise: tf.Tensor,
	) -> tuple[tf.Tensor, tf.Tensor]:
		psf_df = tf.cast(self.model._psf_denorm_factor, pred_psf.dtype)
		noise_df = tf.cast(self.model._noise_denorm_factor, pred_noise.dtype)

		pred_psf_phys = pred_psf / psf_df
		sigma2_psf_phys = sigma2_psf / tf.square(psf_df)
		pred_psf_phys, sigma2_psf_phys, _ = _normalize_psf_for_observation(
			pred_psf_phys,
			sigma2_psf=sigma2_psf_phys,
		)
		pred_noise_phys = pred_noise / noise_df
		sigma2_noise_phys = sigma2_noise / tf.square(noise_df)
		pred_obs = _convolve_image_with_psfs(pred_im, pred_psf_phys) - pred_noise_phys
		sigma2_obs = _propagate_reconstructed_observation_sigma2(
			pred_im=pred_im,
			pred_psf_phys=pred_psf_phys,
			sigma2_im=sigma2_im,
			sigma2_psf_phys=sigma2_psf_phys,
			sigma2_noise_phys=sigma2_noise_phys,
			eps=self.model.sigma2_eps,
		)
		return pred_obs, sigma2_obs

	def evaluate_batch(self, obs: tf.Tensor, y_true: tf.Tensor) -> dict[str, np.ndarray]:
		obs = tf.convert_to_tensor(obs, dtype=tf.float32)
		y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
		y_pred = tf.convert_to_tensor(self.model(obs, training=False), dtype=tf.float32)

		truth_im, truth_psf, truth_noise, n_frames = _split_truth(y_true)
		main_channels = 1 + 2 * n_frames
		pred_main = y_pred[..., :main_channels]
		pred_unc = y_pred[..., main_channels:]
		pred_im, pred_psf, pred_noise = _split_pred(pred_main, n_frames)
		unc_im, unc_psf, unc_noise = _split_pred(pred_unc, n_frames)

		sigma2_im, _ = _pred_to_sigma2(
			unc_im,
			log_sigma=self.model.log_sigma,
			log_min=self.model.log_min,
			log_max=self.model.log_max,
			sigma2_eps=self.model.sigma2_eps,
		)
		sigma2_psf, _ = _pred_to_sigma2(
			unc_psf,
			log_sigma=self.model.log_sigma,
			log_min=self.model.log_min,
			log_max=self.model.log_max,
			sigma2_eps=self.model.sigma2_eps,
		)
		sigma2_noise, _ = _pred_to_sigma2(
			unc_noise,
			log_sigma=self.model.log_sigma,
			log_min=self.model.log_min,
			log_max=self.model.log_max,
			sigma2_eps=self.model.sigma2_eps,
		)

		nll_im, nll_im_residual, nll_im_logsigma2 = _per_example_nll(truth_im, pred_im, sigma2_im)
		nll_psf, nll_psf_residual, nll_psf_logsigma2 = _per_example_nll(truth_psf, pred_psf, sigma2_psf)
		nll_noise, nll_noise_residual, nll_noise_logsigma2 = _per_example_nll(truth_noise, pred_noise, sigma2_noise)

		truth_all = _concat_components(truth_im, truth_psf, truth_noise)
		pred_all = _concat_components(pred_im, pred_psf, pred_noise)
		sigma2_all = _concat_components(sigma2_im, sigma2_psf, sigma2_noise)

		mse_im = _per_example_mse(truth_im, pred_im)
		mse_psf = _per_example_mse(truth_psf, pred_psf)
		mse_noise = _per_example_mse(truth_noise, pred_noise)
		r2_im = _per_example_r2(truth_im, pred_im)
		r2_psf = _per_example_r2(truth_psf, pred_psf)
		r2_noise = _per_example_r2(truth_noise, pred_noise)

		mse = _per_example_mse(truth_all, pred_all)
		r2 = _per_example_r2(truth_all, pred_all)
		nll, _, _ = _per_example_nll(truth_all, pred_all, sigma2_all)
		r2_pinn = self._per_example_pinn_r2(obs, pred_im, pred_psf, pred_noise)
		pred_obs, sigma2_obs = self._predicted_observation_and_sigma2(
			pred_im,
			pred_psf,
			pred_noise,
			sigma2_im,
			sigma2_psf,
			sigma2_noise,
		)
		obs_for_pinn = obs
		if self.model.reconstruction_crop > 0:
			c = int(self.model.reconstruction_crop)
			pred_obs = pred_obs[:, c:-c, c:-c, :]
			sigma2_obs = sigma2_obs[:, c:-c, c:-c, :]
			obs_for_pinn = obs_for_pinn[:, c:-c, c:-c, :]
		pinn_loss_step4 = tf.cast(self.pinn_weight, r2_pinn.dtype) * r2_pinn
		pinn_loss_unc = _per_example_weighted_residual(obs_for_pinn, pred_obs, sigma2_obs)
		loss_supervised = self.im_weight * nll_im + self.psf_weight * nll_psf + self.noise_weight * nll_noise
		loss = loss_supervised + self.pinn_weight * r2_pinn

		result = {
			"loss": loss,
			"loss_supervised": loss_supervised,
			"pinn_loss_step4": pinn_loss_step4,
			"pinn_loss_unc": pinn_loss_unc,
			"mse": mse,
			"mse_im": mse_im,
			"mse_psf": mse_psf,
			"mse_noise": mse_noise,
			"r2": r2,
			"r2_im": r2_im,
			"r2_psf": r2_psf,
			"r2_noise": r2_noise,
			"nll": nll,
			"nll_im": nll_im,
			"nll_psf": nll_psf,
			"nll_noise": nll_noise,
			"nll_im_residual": nll_im_residual,
			"nll_im_logsigma2": nll_im_logsigma2,
			"nll_psf_residual": nll_psf_residual,
			"nll_psf_logsigma2": nll_psf_logsigma2,
			"nll_noise_residual": nll_noise_residual,
			"nll_noise_logsigma2": nll_noise_logsigma2,
			"r2_pinn": r2_pinn,
		}
		return {key: value.numpy().astype(np.float32) for key, value in result.items()}


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate a trained model on val and GalSim datasets.")
	parser.add_argument("--config", type=Path, required=True, help="Path to experiment config .py file")
	parser.add_argument("--algorithm", type=str, default=None, help="Evaluation backend name")
	parser.add_argument("--run-dir", type=Path, default=None, help="Override trained model run directory")
	parser.add_argument("--model-label", choices=("best_model", "final_model"), default=None)
	parser.add_argument("--output-dir", type=Path, default=None)
	parser.add_argument("--eval-batch-size", type=int, default=None)
	parser.add_argument(
		"--stats-examples",
		type=int,
		default=None,
		help="Number of examples per dataset to use for startup component statistics (0 disables)",
	)
	return parser.parse_args()


def _build_dataset_specs(cfg) -> list[DatasetSpec]:
	dataset_root = Path(str(cfg.DATASET_LOAD_CONFIG["data_dir"])).expanduser().resolve()
	galsim_root = Path(str(cfg.GALSIM_TEST_CONFIG["output_dir"])).expanduser().resolve()
	return [
		DatasetSpec(name="val", path=dataset_root / "val"),
		DatasetSpec(name="galsim", path=galsim_root),
	]


def _make_eval_dataset(cfg, data_dir: Path, batch_size: int) -> tf.data.Dataset:
	dataset_cfg = dict(cfg.DATASET_LOAD_CONFIG)
	return make_dataset(
		data_dir,
		batch_size=batch_size,
		shuffle=False,
		repeat=False,
		seed=dataset_cfg.get("seed"),
		channels_last=bool(dataset_cfg.get("channels_last", True)),
		half_n_pix_crop=int(dataset_cfg.get("half_n_pix_crop", 0)),
		fit_im=True,
		fit_psf=True,
		fit_noise=True,
		norm_psf=dataset_cfg.get("norm_psf"),
		norm_noise=dataset_cfg.get("norm_noise"),
		num_parallel_calls=dataset_cfg.get("num_parallel_calls"),
		prefetch=bool(dataset_cfg.get("prefetch", True)),
	)


def _evaluate_dataset(
	backend: EvaluationBackend,
	dataset: tf.data.Dataset,
	*,
	max_batches: int | None = None,
) -> tuple[dict[str, np.ndarray], int]:
	metric_history: dict[str, list[np.ndarray]] = {}
	batches_used = 0
	for batch_index, (obs_batch, y_true_batch) in enumerate(dataset):
		if max_batches is not None and batch_index >= max_batches:
			break
		batch_metrics = backend.evaluate_batch(obs_batch, y_true_batch)
		for name, values in batch_metrics.items():
			metric_history.setdefault(name, []).append(np.asarray(values, dtype=np.float32).reshape(-1))
		batches_used += 1
	if not metric_history:
		raise ValueError("Dataset evaluation produced no batches")
	return {name: np.concatenate(chunks, axis=0) for name, chunks in metric_history.items()}, batches_used


def _summarize_dataset(metrics: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
	return {name: _summarize_metric(values) for name, values in metrics.items()}


def _print_startup_dataset_diagnostics(
	*,
	dataset_specs: list[DatasetSpec],
	datasets: dict[str, tf.data.Dataset],
	stats_examples: int,
) -> None:
	if stats_examples <= 0:
		return
	for spec in dataset_specs:
		summary, used_examples = _summarize_dataset_components(
			datasets[spec.name],
			max_examples=stats_examples,
		)
		if not summary:
			print(f"[test_on_galsim] No startup component statistics available for {spec.name}")
			continue
		print(
			_render_component_stats_table(
				(
					f"Startup component stats for {spec.name} "
					f"(first {used_examples} examples from {spec.path})"
				),
				summary,
			)
		)


def main() -> None:
	args = _parse_args()
	cfg = load_experiment_config(args.config)
	test_cfg = dict(getattr(cfg, "TEST_ON_GALSIM_CONFIG", {}))
	algorithm = str(args.algorithm or test_cfg.get("algorithm", "joint_pinn"))
	if algorithm not in BACKEND_REGISTRY:
		raise ValueError(f"Unknown algorithm backend {algorithm!r}; available={tuple(sorted(BACKEND_REGISTRY))}")

	run_name = str(test_cfg.get("run_name", cfg.JOINT_PINN_CONFIG.get("run_name", "joint_pinn_fourhead")))
	run_dir = args.run_dir.expanduser().resolve() if args.run_dir is not None else (Path(cfg.OUTPUT_BASE_DIR) / run_name).resolve()
	model_label = str(args.model_label or test_cfg.get("model_label", "best_model"))
	eval_batch_size = int(args.eval_batch_size or test_cfg.get("eval_batch_size", cfg.DATASET_LOAD_CONFIG.get("val_batch_size", 64)))
	first_batch_only = bool(test_cfg.get("first_batch_only", True))
	max_eval_batches = 1 if first_batch_only else None
	stats_examples_value = args.stats_examples
	if stats_examples_value is None:
		stats_examples_value = test_cfg.get("stats_examples", min(eval_batch_size, 128))
	stats_examples = int(stats_examples_value)
	output_dir = args.output_dir.expanduser().resolve() if args.output_dir is not None else Path(str(test_cfg.get("output_dir", run_dir / "test_on_galsim"))).expanduser().resolve()
	output_dir.mkdir(parents=True, exist_ok=True)
	if first_batch_only:
		print("[test_on_galsim] first_batch_only=True, limiting evaluation to the first batch of each dataset")

	dataset_specs = _build_dataset_specs(cfg)
	datasets = {spec.name: _make_eval_dataset(cfg, spec.path, batch_size=eval_batch_size) for spec in dataset_specs}
	_print_startup_dataset_diagnostics(
		dataset_specs=dataset_specs,
		datasets=datasets,
		stats_examples=stats_examples,
	)
	preview_obs, _ = next(iter(datasets["val"].take(1)))
	backend = BACKEND_REGISTRY[algorithm].from_run(
		cfg=cfg,
		run_dir=run_dir,
		model_label=model_label,
		preview_obs=preview_obs,
	)

	all_metrics: dict[str, dict[str, np.ndarray]] = {}
	all_summaries: dict[str, dict[str, dict[str, float]]] = {}
	batches_evaluated: dict[str, int] = {}
	for spec in dataset_specs:
		print(f"[test_on_galsim] Evaluating {spec.name} dataset from {spec.path}")
		metrics, n_batches_used = _evaluate_dataset(
			backend,
			datasets[spec.name],
			max_batches=max_eval_batches,
		)
		summary = _summarize_dataset(metrics)
		all_metrics[spec.name] = metrics
		all_summaries[spec.name] = summary
		batches_evaluated[spec.name] = n_batches_used
		np.savez_compressed(output_dir / f"metrics_{spec.name}.npz", **metrics)
		if first_batch_only:
			print(f"[test_on_galsim] Used first {n_batches_used} batch for {spec.name}")
		print(_render_summary_table(f"Summary for {spec.name}", summary))

	comparison_text = _render_comparison_table(
		val_summary=all_summaries["val"],
		galsim_summary=all_summaries["galsim"],
	)
	print(comparison_text)

	report = {
		"algorithm": algorithm,
		"backend": backend.describe(),
		"run_dir": str(run_dir),
		"model_label": model_label,
		"eval_batch_size": eval_batch_size,
		"first_batch_only": first_batch_only,
		"batches_evaluated": batches_evaluated,
		"dataset_paths": {spec.name: str(spec.path) for spec in dataset_specs},
		"summaries": all_summaries,
	}
	with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
		json.dump(report, handle, indent=2)
	with (output_dir / "report.txt").open("w", encoding="utf-8") as handle:
		handle.write(_render_summary_table("Summary for val", all_summaries["val"]))
		handle.write("\n\n")
		handle.write(_render_summary_table("Summary for galsim", all_summaries["galsim"]))
		handle.write("\n\n")
		handle.write(comparison_text)
		handle.write("\n")

	print(f"[test_on_galsim] Wrote outputs to: {output_dir}")


if __name__ == "__main__":
	main()