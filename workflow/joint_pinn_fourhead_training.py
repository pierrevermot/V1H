"""Four-head joint PINN fine-tuning workflow.

Usage
-----
    python joint_pinn_fourhead_training.py --config <experiment.py>

This workflow combines four pretrained heads:
- image head         : obs -> [image_mean, image_unc]
- noise head         : obs -> [noise_mean, noise_unc]
- PSF mean head      : obs -> psf_mean
- PSF uncertainty    : concat(obs, stage2_prepared_psf_mean) -> psf_unc

The optimized loss is:

    w_pinn * r2_pinn + w_im * nll_im + w_psf * nll_psf + w_noise * nll_noise
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from configs.load_config import load_experiment_config
from neural_networks.dataset import make_train_val_datasets
from neural_networks.losses import _convolve_image_with_psfs
from neural_networks.plot_training import plot_training_outputs
from neural_networks.training import (
	_BatchHistory,
	_LrTracker,
	_SaveBestExamples,
	_TerminateOnNaNWithBatch,
)
from neural_networks.layers import GroupNormalization
from neural_networks.layers import _upsample_bilinear
from utils.metrics import _pred_to_sigma2, _gaussian_nll_parts, _split_truth, _split_pred
from utils.model_io import (
    _resolve_model_paths, _infer_model_spec_from_keras_archive,
    _load_independent_head_model, _load_stage2_head_model,
)
from utils.model_utils import _split_nll_output, _extract_mean_output, _extract_uncertainty_output
from utils.normalization import (
    _compute_norm_factor, _convert_normed_tensor, _normalize_psf_for_observation,
)


@tf.keras.utils.register_keras_serializable()
class FourHeadJointPinnModel(tf.keras.Model):
	def __init__(
		self,
		image_model: tf.keras.Model,
		noise_model: tf.keras.Model,
		psf_mean_model: tf.keras.Model,
		psf_unc_model: tf.keras.Model,
		*,
		pinn_weight: float,
		im_weight: float,
		psf_weight: float,
		noise_weight: float,
		log_sigma: bool,
		log_min: float,
		log_max: float,
		sigma2_eps: float,
		psf_mean_source_norm_psf,
		psf_unc_input_norm_psf,
		norm_psf: str | float | None,
		norm_noise: str | float | None,
		reconstruction_crop: int,
		**kwargs,
	):
		super().__init__(**kwargs)
		self.image_model = image_model
		self.noise_model = noise_model
		self.psf_mean_model = psf_mean_model
		self.psf_unc_model = psf_unc_model
		self.pinn_weight = tf.Variable(float(pinn_weight), trainable=False, dtype=tf.float32, name="pinn_weight")
		self.im_weight = tf.Variable(float(im_weight), trainable=False, dtype=tf.float32, name="im_weight")
		self.psf_weight = tf.Variable(float(psf_weight), trainable=False, dtype=tf.float32, name="psf_weight")
		self.noise_weight = tf.Variable(float(noise_weight), trainable=False, dtype=tf.float32, name="noise_weight")
		self.log_sigma = bool(log_sigma)
		self.log_min = float(log_min)
		self.log_max = float(log_max)
		self.sigma2_eps = float(sigma2_eps)
		self.psf_mean_source_norm_psf = psf_mean_source_norm_psf
		self.psf_unc_input_norm_psf = psf_unc_input_norm_psf
		self.norm_psf = norm_psf
		self.norm_noise = norm_noise
		self.reconstruction_crop = int(reconstruction_crop)

		in_shape = getattr(image_model, "input_shape", None)
		if isinstance(in_shape, list):
			in_shape = in_shape[0]
		n_pix_crop = int(in_shape[1]) if in_shape and len(in_shape) >= 3 and in_shape[1] is not None else None
		self._psf_denorm_factor = _compute_norm_factor(norm_psf, n_pix_crop)
		self._noise_denorm_factor = _compute_norm_factor(norm_noise, n_pix_crop)

		self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
		self.supervised_loss_tracker = tf.keras.metrics.Mean(name="loss_supervised")
		self.pinn_loss_tracker = tf.keras.metrics.Mean(name="r2_pinn")
		self.nll_im_tracker = tf.keras.metrics.Mean(name="nll_im")
		self.nll_psf_tracker = tf.keras.metrics.Mean(name="nll_psf")
		self.nll_noise_tracker = tf.keras.metrics.Mean(name="nll_noise")
		self.nll_im_residual_tracker = tf.keras.metrics.Mean(name="nll_im_residual")
		self.nll_im_logsigma_tracker = tf.keras.metrics.Mean(name="nll_im_logsigma2")
		self.nll_psf_residual_tracker = tf.keras.metrics.Mean(name="nll_psf_residual")
		self.nll_psf_logsigma_tracker = tf.keras.metrics.Mean(name="nll_psf_logsigma2")
		self.nll_noise_residual_tracker = tf.keras.metrics.Mean(name="nll_noise_residual")
		self.nll_noise_logsigma_tracker = tf.keras.metrics.Mean(name="nll_noise_logsigma2")

	def get_config(self):
		config = super().get_config()
		config.update(
			{
				"pinn_weight": float(tf.keras.backend.get_value(self.pinn_weight)),
				"im_weight": float(tf.keras.backend.get_value(self.im_weight)),
				"psf_weight": float(tf.keras.backend.get_value(self.psf_weight)),
				"noise_weight": float(tf.keras.backend.get_value(self.noise_weight)),
				"log_sigma": self.log_sigma,
				"log_min": self.log_min,
				"log_max": self.log_max,
				"sigma2_eps": self.sigma2_eps,
				"psf_mean_source_norm_psf": self.psf_mean_source_norm_psf,
				"psf_unc_input_norm_psf": self.psf_unc_input_norm_psf,
				"norm_psf": self.norm_psf,
				"norm_noise": self.norm_noise,
				"reconstruction_crop": self.reconstruction_crop,
			}
		)
		return config

	@classmethod
	def from_config(cls, config):
		raise ValueError(
			"FourHeadJointPinnModel must be instantiated with the four submodels. "
			"Use workflow.joint_pinn_fourhead_training.FourHeadJointPinnModel(...)."
		)

	@property
	def metrics(self):
		return [
			self.total_loss_tracker,
			self.supervised_loss_tracker,
			self.pinn_loss_tracker,
			self.nll_im_tracker,
			self.nll_psf_tracker,
			self.nll_noise_tracker,
			self.nll_im_residual_tracker,
			self.nll_im_logsigma_tracker,
			self.nll_psf_residual_tracker,
			self.nll_psf_logsigma_tracker,
			self.nll_noise_residual_tracker,
			self.nll_noise_logsigma_tracker,
		]

	def call(self, inputs, training=False):
		n_frames_static = inputs.shape[-1]
		n_frames = int(n_frames_static) if n_frames_static is not None else tf.shape(inputs)[-1]
		im_raw = self.image_model(inputs, training=training)
		noise_raw = self.noise_model(inputs, training=training)
		psf_mean_raw = self.psf_mean_model(inputs, training=training)
		psf_mean_source = _extract_mean_output(psf_mean_raw, expected_channels=n_frames, head_name="psf_mean_head")
		psf_mean = _convert_normed_tensor(
			psf_mean_source,
			source_norm=self.psf_mean_source_norm_psf,
			target_norm=self.norm_psf,
			spatial_axis=1,
		)
		psf_unc_tail = _convert_normed_tensor(
			psf_mean_source,
			source_norm=self.psf_mean_source_norm_psf,
			target_norm=self.psf_unc_input_norm_psf,
			spatial_axis=1,
		)
		psf_unc_inputs = tf.concat([inputs, psf_unc_tail], axis=-1)
		psf_unc_raw = self.psf_unc_model(psf_unc_inputs, training=training)

		im_main, im_unc = _split_nll_output(im_raw, expected_channels=1, head_name="image_head")
		noise_main, noise_unc = _split_nll_output(noise_raw, expected_channels=n_frames, head_name="noise_head")
		psf_unc = _extract_uncertainty_output(psf_unc_raw, expected_channels=n_frames, head_name="psf_unc_head")

		main = tf.concat([im_main, psf_mean, noise_main], axis=-1)
		unc = tf.concat([im_unc, psf_unc, noise_unc], axis=-1)
		return tf.concat([main, unc], axis=-1)

	def _supervised_losses_parts(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> dict[str, tf.Tensor]:
		truth_im, truth_psf, truth_noise, n_frames = _split_truth(y_true)
		main_channels = 1 + 2 * n_frames
		pred_main = y_pred[..., :main_channels]
		pred_unc = y_pred[..., main_channels:]
		pred_im, pred_psf, pred_noise = _split_pred(pred_main, n_frames)
		unc_im, unc_psf, unc_noise = _split_pred(pred_unc, n_frames)

		sigma2_im, _ = _pred_to_sigma2(
			unc_im,
			log_sigma=self.log_sigma,
			log_min=self.log_min,
			log_max=self.log_max,
			sigma2_eps=self.sigma2_eps,
		)
		sigma2_psf, _ = _pred_to_sigma2(
			unc_psf,
			log_sigma=self.log_sigma,
			log_min=self.log_min,
			log_max=self.log_max,
			sigma2_eps=self.sigma2_eps,
		)
		sigma2_noise, _ = _pred_to_sigma2(
			unc_noise,
			log_sigma=self.log_sigma,
			log_min=self.log_min,
			log_max=self.log_max,
			sigma2_eps=self.sigma2_eps,
		)

		nll_im, nll_im_residual, nll_im_logsigma2 = _gaussian_nll_parts(truth_im, pred_im, sigma2_im)
		nll_psf, nll_psf_residual, nll_psf_logsigma2 = _gaussian_nll_parts(truth_psf, pred_psf, sigma2_psf)
		nll_noise, nll_noise_residual, nll_noise_logsigma2 = _gaussian_nll_parts(truth_noise, pred_noise, sigma2_noise)

		return {
			"nll_im": nll_im,
			"nll_psf": nll_psf,
			"nll_noise": nll_noise,
			"nll_im_residual": nll_im_residual,
			"nll_im_logsigma2": nll_im_logsigma2,
			"nll_psf_residual": nll_psf_residual,
			"nll_psf_logsigma2": nll_psf_logsigma2,
			"nll_noise_residual": nll_noise_residual,
			"nll_noise_logsigma2": nll_noise_logsigma2,
		}

	def _pinn_r2_loss(self, obs: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
		channels = tf.shape(obs)[-1]
		main_channels = 1 + 2 * channels
		pred_main = y_pred[..., :main_channels]
		pred_im = pred_main[..., :1]
		pred_psf = pred_main[..., 1 : 1 + channels]
		pred_noise = pred_main[..., 1 + channels : 1 + 2 * channels]

		psf_df = tf.cast(self._psf_denorm_factor, pred_psf.dtype)
		noise_df = tf.cast(self._noise_denorm_factor, pred_noise.dtype)
		pred_psf_phys = pred_psf / psf_df
		pred_psf_phys, _, _ = _normalize_psf_for_observation(pred_psf_phys)
		pred_noise_phys = pred_noise / noise_df
		pred_obs = _convolve_image_with_psfs(pred_im, pred_psf_phys) - pred_noise_phys

		if self.reconstruction_crop > 0:
			c = int(self.reconstruction_crop)
			pred_obs = pred_obs[:, c:-c, c:-c, :]
			obs = obs[:, c:-c, c:-c, :]

		err2 = tf.reduce_mean(tf.square(pred_obs - obs))
		den = tf.math.reduce_variance(obs) + tf.cast(1e-12, obs.dtype)
		return err2 / den

	def train_step(self, data):
		obs, y_true = data
		with tf.GradientTape() as tape:
			y_pred = self(obs, training=True)
			sup_parts = self._supervised_losses_parts(y_true, y_pred)
			r2_pinn = self._pinn_r2_loss(obs, y_pred)
			supervised = (
				tf.cast(self.im_weight, sup_parts["nll_im"].dtype) * sup_parts["nll_im"]
				+ tf.cast(self.psf_weight, sup_parts["nll_psf"].dtype) * sup_parts["nll_psf"]
				+ tf.cast(self.noise_weight, sup_parts["nll_noise"].dtype) * sup_parts["nll_noise"]
			)
			loss = supervised + tf.cast(self.pinn_weight, r2_pinn.dtype) * r2_pinn
			if self.losses:
				loss += tf.add_n(self.losses)

		train_vars = self.trainable_variables
		grads = tape.gradient(loss, train_vars)
		self.optimizer.apply_gradients(zip(grads, train_vars))

		self.total_loss_tracker.update_state(loss)
		self.supervised_loss_tracker.update_state(supervised)
		self.pinn_loss_tracker.update_state(r2_pinn)
		self.nll_im_tracker.update_state(sup_parts["nll_im"])
		self.nll_psf_tracker.update_state(sup_parts["nll_psf"])
		self.nll_noise_tracker.update_state(sup_parts["nll_noise"])
		self.nll_im_residual_tracker.update_state(sup_parts["nll_im_residual"])
		self.nll_im_logsigma_tracker.update_state(sup_parts["nll_im_logsigma2"])
		self.nll_psf_residual_tracker.update_state(sup_parts["nll_psf_residual"])
		self.nll_psf_logsigma_tracker.update_state(sup_parts["nll_psf_logsigma2"])
		self.nll_noise_residual_tracker.update_state(sup_parts["nll_noise_residual"])
		self.nll_noise_logsigma_tracker.update_state(sup_parts["nll_noise_logsigma2"])

		logs = {m.name: m.result() for m in self.metrics}
		logs.update(
			{
				"inst_loss": loss,
				"inst_loss_supervised": supervised,
				"inst_r2_pinn": r2_pinn,
				"inst_nll_im": sup_parts["nll_im"],
				"inst_nll_psf": sup_parts["nll_psf"],
				"inst_nll_noise": sup_parts["nll_noise"],
				"inst_nll_im_residual": sup_parts["nll_im_residual"],
				"inst_nll_im_logsigma2": sup_parts["nll_im_logsigma2"],
				"inst_nll_psf_residual": sup_parts["nll_psf_residual"],
				"inst_nll_psf_logsigma2": sup_parts["nll_psf_logsigma2"],
				"inst_nll_noise_residual": sup_parts["nll_noise_residual"],
				"inst_nll_noise_logsigma2": sup_parts["nll_noise_logsigma2"],
				"pinn_weight": tf.cast(self.pinn_weight, loss.dtype),
				"im_weight": tf.cast(self.im_weight, loss.dtype),
				"psf_weight": tf.cast(self.psf_weight, loss.dtype),
				"noise_weight": tf.cast(self.noise_weight, loss.dtype),
			}
		)
		return logs

	def test_step(self, data):
		obs, y_true = data
		y_pred = self(obs, training=False)
		sup_parts = self._supervised_losses_parts(y_true, y_pred)
		r2_pinn = self._pinn_r2_loss(obs, y_pred)
		supervised = (
			tf.cast(self.im_weight, sup_parts["nll_im"].dtype) * sup_parts["nll_im"]
			+ tf.cast(self.psf_weight, sup_parts["nll_psf"].dtype) * sup_parts["nll_psf"]
			+ tf.cast(self.noise_weight, sup_parts["nll_noise"].dtype) * sup_parts["nll_noise"]
		)
		loss = supervised + tf.cast(self.pinn_weight, r2_pinn.dtype) * r2_pinn
		if self.losses:
			loss += tf.add_n(self.losses)

		self.total_loss_tracker.update_state(loss)
		self.supervised_loss_tracker.update_state(supervised)
		self.pinn_loss_tracker.update_state(r2_pinn)
		self.nll_im_tracker.update_state(sup_parts["nll_im"])
		self.nll_psf_tracker.update_state(sup_parts["nll_psf"])
		self.nll_noise_tracker.update_state(sup_parts["nll_noise"])
		self.nll_im_residual_tracker.update_state(sup_parts["nll_im_residual"])
		self.nll_im_logsigma_tracker.update_state(sup_parts["nll_im_logsigma2"])
		self.nll_psf_residual_tracker.update_state(sup_parts["nll_psf_residual"])
		self.nll_psf_logsigma_tracker.update_state(sup_parts["nll_psf_logsigma2"])
		self.nll_noise_residual_tracker.update_state(sup_parts["nll_noise_residual"])
		self.nll_noise_logsigma_tracker.update_state(sup_parts["nll_noise_logsigma2"])

		return {m.name: m.result() for m in self.metrics}


def _format_metrics(metrics: dict[str, float]) -> str:
	preferred = [
		"loss",
		"loss_supervised",
		"r2_pinn",
		"nll_im",
		"nll_psf",
		"nll_noise",
		"nll_im_residual",
		"nll_im_logsigma2",
		"nll_psf_residual",
		"nll_psf_logsigma2",
		"nll_noise_residual",
		"nll_noise_logsigma2",
	]
	parts = []
	for key in preferred:
		if key in metrics:
			parts.append(f"{key}={float(metrics[key]):.6f}")
	for key, value in metrics.items():
		if key not in preferred:
			parts.append(f"{key}={float(value):.6f}")
	return ", ".join(parts)


def _format_debug_values(values: dict[str, Any]) -> str:
	parts = []
	for key, value in values.items():
		if isinstance(value, bool):
			parts.append(f"{key}={str(value).lower()}")
		elif isinstance(value, str):
			parts.append(f"{key}={value}")
		else:
			parts.append(f"{key}={float(value):.6f}")
	return ", ".join(parts)


def _collect_batch_median_metrics(
	model: tf.keras.Model,
	dataset: tf.data.Dataset,
	*,
	metric_names: list[str],
	steps: int | None,
) -> dict[str, float]:
	metric_history: dict[str, list[float]] = {"loss": []}
	for name in metric_names:
		metric_history[name] = []

	for batch_idx, (obs, y_true) in enumerate(dataset):
		if steps is not None and batch_idx >= steps:
			break
		y_pred = model(obs, training=False)
		sup_parts = model._supervised_losses_parts(y_true, y_pred)
		r2_pinn = model._pinn_r2_loss(obs, y_pred)
		supervised = (
			tf.cast(model.im_weight, sup_parts["nll_im"].dtype) * sup_parts["nll_im"]
			+ tf.cast(model.psf_weight, sup_parts["nll_psf"].dtype) * sup_parts["nll_psf"]
			+ tf.cast(model.noise_weight, sup_parts["nll_noise"].dtype) * sup_parts["nll_noise"]
		)
		loss = supervised + tf.cast(model.pinn_weight, r2_pinn.dtype) * r2_pinn
		if model.losses:
			loss += tf.add_n(model.losses)

		batch_values = {
			"loss": loss,
			"loss_supervised": supervised,
			"r2_pinn": r2_pinn,
			"nll_im": sup_parts["nll_im"],
			"nll_psf": sup_parts["nll_psf"],
			"nll_noise": sup_parts["nll_noise"],
			"nll_im_residual": sup_parts["nll_im_residual"],
			"nll_im_logsigma2": sup_parts["nll_im_logsigma2"],
			"nll_psf_residual": sup_parts["nll_psf_residual"],
			"nll_psf_logsigma2": sup_parts["nll_psf_logsigma2"],
			"nll_noise_residual": sup_parts["nll_noise_residual"],
			"nll_noise_logsigma2": sup_parts["nll_noise_logsigma2"],
		}
		for name, value in batch_values.items():
			if name in metric_history:
				metric_history[name].append(float(value.numpy()))

	return {
		name: float(np.median(values))
		for name, values in metric_history.items()
		if values
	}


def _truth_reconstruction_audit(
	dataset: tf.data.Dataset,
	*,
	steps: int | None,
	norm_psf: str | float | None,
	norm_noise: str | float | None,
	reconstruction_crop: int,
) -> dict[str, float]:
	def _iterate_limited(ds: tf.data.Dataset, n_steps: int | None):
		if n_steps is None:
			yield from ds
			return
		for batch_idx, batch in enumerate(ds):
			if batch_idx >= n_steps:
				break
			yield batch

	r2_values: list[float] = []
	r2_values_psf_sum1: list[float] = []
	psf_sum_values: list[np.ndarray] = []
	noise_mean_values: list[float] = []

	for obs, y_true in _iterate_limited(dataset, steps):
		truth_im, truth_psf, truth_noise, _ = _split_truth(y_true)
		psf_df = tf.cast(_compute_norm_factor(norm_psf, int(truth_psf.shape[1])), truth_psf.dtype)
		noise_df = tf.cast(_compute_norm_factor(norm_noise, int(truth_noise.shape[1])), truth_noise.dtype)
		truth_psf_phys = truth_psf / psf_df
		truth_psf_phys, _, psf_sum = _normalize_psf_for_observation(truth_psf_phys)
		truth_noise_phys = truth_noise / noise_df

		obs_truth = _convolve_image_with_psfs(truth_im, truth_psf_phys) - truth_noise_phys
		obs_truth_psf_sum1 = obs_truth

		if reconstruction_crop > 0:
			c = int(reconstruction_crop)
			obs = obs[:, c:-c, c:-c, :]
			obs_truth = obs_truth[:, c:-c, c:-c, :]
			obs_truth_psf_sum1 = obs_truth_psf_sum1[:, c:-c, c:-c, :]

		den = tf.math.reduce_variance(obs) + tf.cast(1e-12, obs.dtype)
		r2_values.append(float((tf.reduce_mean(tf.square(obs_truth - obs)) / den).numpy()))
		r2_values_psf_sum1.append(float((tf.reduce_mean(tf.square(obs_truth_psf_sum1 - obs)) / den).numpy()))
		psf_sum_values.append(tf.reshape(psf_sum, (-1,)).numpy())
		noise_mean_values.append(float(tf.reduce_mean(truth_noise_phys).numpy()))

	if not r2_values:
		return {}

	psf_sum_concat = np.concatenate(psf_sum_values, axis=0).astype(np.float64)
	return {
		"truth_r2_pinn": float(np.mean(r2_values)),
		"truth_r2_pinn_psf_sum1": float(np.mean(r2_values_psf_sum1)),
		"truth_psf_sum_mean": float(np.mean(psf_sum_concat)),
		"truth_psf_sum_std": float(np.std(psf_sum_concat)),
		"truth_psf_sum_min": float(np.min(psf_sum_concat)),
		"truth_psf_sum_max": float(np.max(psf_sum_concat)),
		"truth_noise_mean": float(np.mean(np.asarray(noise_mean_values, dtype=np.float64))),
	}


def _prediction_reconstruction_audit(
	model: FourHeadJointPinnModel,
	dataset: tf.data.Dataset,
	*,
	steps: int | None,
) -> dict[str, float]:
	def _iterate_limited(ds: tf.data.Dataset, n_steps: int | None):
		if n_steps is None:
			yield from ds
			return
		for batch_idx, batch in enumerate(ds):
			if batch_idx >= n_steps:
				break
			yield batch

	def _crop_tensor(tensor: tf.Tensor, crop: int) -> tf.Tensor:
		if crop <= 0:
			return tensor
		return tensor[:, crop:-crop, crop:-crop, :]

	def _r2_to_obs(obs: tf.Tensor, candidate: tf.Tensor, crop: int) -> float:
		obs_used = _crop_tensor(obs, crop)
		candidate_used = _crop_tensor(candidate, crop)
		den = tf.math.reduce_variance(obs_used) + tf.cast(1e-12, obs_used.dtype)
		return float((tf.reduce_mean(tf.square(candidate_used - obs_used)) / den).numpy())

	r2_all_values: list[float] = []
	r2_psf_sum1_values: list[float] = []
	r2_pred_im_truth_values: list[float] = []
	r2_truth_im_pred_psf_values: list[float] = []
	r2_truth_im_pred_noise_values: list[float] = []
	r2_pred_im_pred_psf_truth_noise_values: list[float] = []
	r2_pred_im_truth_psf_pred_noise_values: list[float] = []
	r2_truth_im_pred_psf_pred_noise_values: list[float] = []
	pred_psf_sum_values: list[np.ndarray] = []
	pred_psf_sum_ratio_values: list[np.ndarray] = []
	pred_noise_mean_values: list[float] = []
	pred_im_mean_values: list[float] = []

	for obs, y_true in _iterate_limited(dataset, steps):
		y_pred = model(obs, training=False)
		truth_im, truth_psf, truth_noise, n_frames = _split_truth(y_true)
		main_channels = 1 + 2 * n_frames
		pred_main = y_pred[..., :main_channels]
		pred_im, pred_psf, pred_noise = _split_pred(pred_main, n_frames)

		psf_df = tf.cast(model._psf_denorm_factor, pred_psf.dtype)
		noise_df = tf.cast(model._noise_denorm_factor, pred_noise.dtype)
		truth_psf_phys = truth_psf / psf_df
		pred_psf_phys = pred_psf / psf_df
		truth_psf_phys, _, truth_psf_sum = _normalize_psf_for_observation(truth_psf_phys)
		pred_psf_phys, _, pred_psf_sum = _normalize_psf_for_observation(pred_psf_phys)
		truth_noise_phys = truth_noise / noise_df
		pred_noise_phys = pred_noise / noise_df

		pred_obs = _convolve_image_with_psfs(pred_im, pred_psf_phys) - pred_noise_phys
		pred_obs_psf_sum1 = pred_obs
		pred_im_truth = _convolve_image_with_psfs(pred_im, truth_psf_phys) - truth_noise_phys
		truth_im_pred_psf = _convolve_image_with_psfs(truth_im, pred_psf_phys) - truth_noise_phys
		truth_im_pred_noise = _convolve_image_with_psfs(truth_im, truth_psf_phys) - pred_noise_phys
		pred_im_pred_psf_truth_noise = _convolve_image_with_psfs(pred_im, pred_psf_phys) - truth_noise_phys
		pred_im_truth_psf_pred_noise = _convolve_image_with_psfs(pred_im, truth_psf_phys) - pred_noise_phys
		truth_im_pred_psf_pred_noise = _convolve_image_with_psfs(truth_im, pred_psf_phys) - pred_noise_phys

		crop = int(model.reconstruction_crop)
		r2_all_values.append(_r2_to_obs(obs, pred_obs, crop))
		r2_psf_sum1_values.append(_r2_to_obs(obs, pred_obs_psf_sum1, crop))
		r2_pred_im_truth_values.append(_r2_to_obs(obs, pred_im_truth, crop))
		r2_truth_im_pred_psf_values.append(_r2_to_obs(obs, truth_im_pred_psf, crop))
		r2_truth_im_pred_noise_values.append(_r2_to_obs(obs, truth_im_pred_noise, crop))
		r2_pred_im_pred_psf_truth_noise_values.append(_r2_to_obs(obs, pred_im_pred_psf_truth_noise, crop))
		r2_pred_im_truth_psf_pred_noise_values.append(_r2_to_obs(obs, pred_im_truth_psf_pred_noise, crop))
		r2_truth_im_pred_psf_pred_noise_values.append(_r2_to_obs(obs, truth_im_pred_psf_pred_noise, crop))
		truth_psf_sum_safe = tf.where(
			truth_psf_sum >= 0,
			tf.maximum(truth_psf_sum, tf.cast(1e-12, truth_psf_sum.dtype)),
			tf.minimum(truth_psf_sum, tf.cast(-1e-12, truth_psf_sum.dtype)),
		)
		pred_psf_sum_safe = tf.where(
			pred_psf_sum >= 0,
			tf.maximum(pred_psf_sum, tf.cast(1e-12, pred_psf_sum.dtype)),
			tf.minimum(pred_psf_sum, tf.cast(-1e-12, pred_psf_sum.dtype)),
		)
		pred_psf_sum_values.append(tf.reshape(pred_psf_sum, (-1,)).numpy())
		pred_psf_sum_ratio_values.append(tf.reshape(pred_psf_sum_safe / truth_psf_sum_safe, (-1,)).numpy())
		pred_noise_mean_values.append(float(tf.reduce_mean(pred_noise_phys).numpy()))
		pred_im_mean_values.append(float(tf.reduce_mean(pred_im).numpy()))

	if not r2_all_values:
		return {}

	pred_psf_sum_concat = np.concatenate(pred_psf_sum_values, axis=0).astype(np.float64)
	pred_psf_sum_ratio_concat = np.concatenate(pred_psf_sum_ratio_values, axis=0).astype(np.float64)
	return {
		"pred_r2_pinn": float(np.mean(r2_all_values)),
		"pred_r2_pinn_psf_sum1": float(np.mean(r2_psf_sum1_values)),
		"pred_r2_pinn_pred_im_truth_psf_res": float(np.mean(r2_pred_im_truth_values)),
		"pred_r2_pinn_truth_im_pred_psf_truth_noise": float(np.mean(r2_truth_im_pred_psf_values)),
		"pred_r2_pinn_truth_im_truth_psf_pred_noise": float(np.mean(r2_truth_im_pred_noise_values)),
		"pred_r2_pinn_pred_im_pred_psf_truth_noise": float(np.mean(r2_pred_im_pred_psf_truth_noise_values)),
		"pred_r2_pinn_pred_im_truth_psf_pred_noise": float(np.mean(r2_pred_im_truth_psf_pred_noise_values)),
		"pred_r2_pinn_truth_im_pred_psf_pred_noise": float(np.mean(r2_truth_im_pred_psf_pred_noise_values)),
		"pred_psf_sum_mean": float(np.mean(pred_psf_sum_concat)),
		"pred_psf_sum_std": float(np.std(pred_psf_sum_concat)),
		"pred_psf_sum_min": float(np.min(pred_psf_sum_concat)),
		"pred_psf_sum_max": float(np.max(pred_psf_sum_concat)),
		"pred_psf_sum_ratio_mean": float(np.mean(pred_psf_sum_ratio_concat)),
		"pred_psf_sum_ratio_std": float(np.std(pred_psf_sum_ratio_concat)),
		"pred_noise_mean": float(np.mean(np.asarray(pred_noise_mean_values, dtype=np.float64))),
		"pred_im_mean": float(np.mean(np.asarray(pred_im_mean_values, dtype=np.float64))),
	}


def _startup_psf_convention_audit(
	*,
	model: FourHeadJointPinnModel,
	psf_mean_model_path: Path,
	psf_mean_head_cfg: dict[str, Any],
	psf_unc_head_cfg: dict[str, Any],
	dataset_half_n_pix_crop: int,
	obs: tf.Tensor,
	y_true: tf.Tensor,
) -> dict[str, Any]:
	_, truth_psf, _, n_frames = _split_truth(y_true)
	pred_output = model.psf_mean_model(obs, training=False)
	pred_psf_source = _extract_mean_output(pred_output, n_frames, head_name="psf_mean_head")
	stage2_dataset_cfg = dict(psf_unc_head_cfg.get("dataset", {}))
	stage2_norm_psf = stage2_dataset_cfg.get("norm_psf")
	stage2_half_n_pix_crop = int(stage2_dataset_cfg.get("half_n_pix_crop", dataset_half_n_pix_crop))
	pred_psf = _convert_normed_tensor(
		pred_psf_source,
		source_norm=model.psf_mean_source_norm_psf,
		target_norm=model.norm_psf,
		spatial_axis=1,
	)
	joint_psf_unc_tail = _convert_normed_tensor(
		pred_psf_source,
		source_norm=model.psf_mean_source_norm_psf,
		target_norm=model.psf_unc_input_norm_psf,
		spatial_axis=1,
	)
	stage2_prepared_pred_psf = _convert_normed_tensor(
		pred_psf_source,
		source_norm=model.psf_mean_source_norm_psf,
		target_norm=stage2_norm_psf,
		spatial_axis=1,
	)

	joint_psf_unc_input = tf.concat([obs, joint_psf_unc_tail], axis=-1)
	stage2_psf_unc_input = tf.concat([obs, stage2_prepared_pred_psf], axis=-1)
	pred_unc_output = model.psf_unc_model(joint_psf_unc_input, training=False)
	pred_unc_output_stage2 = model.psf_unc_model(stage2_psf_unc_input, training=False)
	pred_psf_unc = _extract_uncertainty_output(pred_unc_output, n_frames, head_name="psf_unc_head")
	pred_psf_unc_stage2 = _extract_uncertainty_output(pred_unc_output_stage2, n_frames, head_name="psf_unc_head")

	psf_df = tf.cast(model._psf_denorm_factor, pred_psf.dtype)
	pred_psf_phys = pred_psf / psf_df
	pred_sigma2_psf, _ = _pred_to_sigma2(
		pred_psf_unc,
		log_sigma=model.log_sigma,
		log_min=model.log_min,
		log_max=model.log_max,
		sigma2_eps=model.sigma2_eps,
	)
	pred_sigma2_psf_stage2, _ = _pred_to_sigma2(
		pred_psf_unc_stage2,
		log_sigma=model.log_sigma,
		log_min=model.log_min,
		log_max=model.log_max,
		sigma2_eps=model.sigma2_eps,
	)
	pred_sigma2_psf_phys = pred_sigma2_psf / tf.square(psf_df)
	pred_sigma2_psf_stage2_phys = pred_sigma2_psf_stage2 / tf.square(psf_df)
	truth_psf_phys = truth_psf / tf.cast(model._psf_denorm_factor, truth_psf.dtype)
	pred_psf_recon, pred_sigma2_psf_recon, _ = _normalize_psf_for_observation(
		pred_psf_phys,
		sigma2_psf=pred_sigma2_psf_phys,
	)
	stage2_pred_psf_phys = stage2_prepared_pred_psf / psf_df
	stage2_pred_psf_recon, pred_sigma2_psf_stage2_recon, _ = _normalize_psf_for_observation(
		stage2_pred_psf_phys,
		sigma2_psf=pred_sigma2_psf_stage2_phys,
	)
	truth_psf_recon, _, _ = _normalize_psf_for_observation(truth_psf_phys)

	pred_psf_sum = tf.reduce_sum(pred_psf, axis=(1, 2), keepdims=True)
	stage2_prepared_pred_psf_sum = tf.reduce_sum(stage2_prepared_pred_psf, axis=(1, 2), keepdims=True)
	pred_psf_phys_sum = tf.reduce_sum(pred_psf_phys, axis=(1, 2), keepdims=True)
	stage2_pred_psf_phys_sum = tf.reduce_sum(stage2_pred_psf_phys, axis=(1, 2), keepdims=True)
	pred_psf_recon_sum = tf.reduce_sum(pred_psf_recon, axis=(1, 2), keepdims=True)
	stage2_pred_psf_recon_sum = tf.reduce_sum(stage2_pred_psf_recon, axis=(1, 2), keepdims=True)
	truth_psf_sum = tf.reduce_sum(truth_psf, axis=(1, 2), keepdims=True)
	truth_psf_phys_sum = tf.reduce_sum(truth_psf_phys, axis=(1, 2), keepdims=True)
	truth_psf_recon_sum = tf.reduce_sum(truth_psf_recon, axis=(1, 2), keepdims=True)
	pred_psf_max = tf.reduce_max(pred_psf, axis=(1, 2), keepdims=True)
	stage2_prepared_pred_psf_max = tf.reduce_max(stage2_prepared_pred_psf, axis=(1, 2), keepdims=True)
	pred_psf_phys_max = tf.reduce_max(pred_psf_phys, axis=(1, 2), keepdims=True)
	stage2_pred_psf_phys_max = tf.reduce_max(stage2_pred_psf_phys, axis=(1, 2), keepdims=True)
	pred_psf_recon_max = tf.reduce_max(pred_psf_recon, axis=(1, 2), keepdims=True)
	stage2_pred_psf_recon_max = tf.reduce_max(stage2_pred_psf_recon, axis=(1, 2), keepdims=True)

	joint_input_tail_diff = joint_psf_unc_tail - stage2_prepared_pred_psf
	joint_input_tail_rmse = tf.sqrt(tf.reduce_mean(tf.square(joint_input_tail_diff)))
	joint_input_tail_mae = tf.reduce_mean(tf.abs(joint_input_tail_diff))
	stage2_over_joint_sum_ratio = stage2_prepared_pred_psf_sum / tf.maximum(
		tf.reduce_sum(joint_psf_unc_tail, axis=(1, 2), keepdims=True),
		tf.cast(1e-12, pred_psf_sum.dtype),
	)
	nll_psf_joint_proxy, nll_psf_joint_residual, nll_psf_joint_logsigma = _gaussian_nll_parts(
		truth_psf,
		pred_psf,
		pred_sigma2_psf,
	)
	nll_psf_stage2_proxy, nll_psf_stage2_residual, nll_psf_stage2_logsigma = _gaussian_nll_parts(
		truth_psf,
		stage2_prepared_pred_psf,
		pred_sigma2_psf_stage2,
	)

	head_dataset_cfg = dict(psf_mean_head_cfg.get("dataset", {}))
	head_model_cfg = dict(psf_mean_head_cfg.get("model", {}))
	head_model_name = str(head_model_cfg.get("name", "unet")).strip().lower() or "unet"
	head_arch_cfg = dict(psf_mean_head_cfg.get(head_model_name, {}))
	archive_spec: dict[str, Any] = {}
	if psf_mean_model_path.suffix == ".keras":
		try:
			archive_spec = _infer_model_spec_from_keras_archive(psf_mean_model_path)
		except Exception as exc:
			archive_spec = {"error": f"{type(exc).__name__}: {exc}"}

	return {
		"joint_norm_psf": str(model.norm_psf),
		"joint_psf_unc_input_norm_psf": str(model.psf_unc_input_norm_psf),
		"psf_mean_source_norm_psf": str(model.psf_mean_source_norm_psf),
		"joint_psf_denorm_factor": float(model._psf_denorm_factor),
		"psf_unc_saved_norm_psf": str(stage2_norm_psf),
		"psf_unc_saved_half_n_pix_crop": float(stage2_half_n_pix_crop),
		"psf_head_saved_norm_psf": str(head_dataset_cfg.get("norm_psf")),
		"psf_head_saved_model_name": head_model_name,
		"psf_head_saved_normalize_output_sum": bool(head_arch_cfg.get("normalize_output_sum", False)),
		"psf_head_saved_normalize_with_first": bool(head_arch_cfg.get("normalize_with_first", False)),
		"psf_head_saved_normalize_first_only": bool(head_arch_cfg.get("normalize_first_only", False)),
		"psf_head_saved_normalize_by_mean": bool(head_arch_cfg.get("normalize_by_mean", False)),
		"psf_head_archive_model_name": str(archive_spec.get("model_name", "")),
		"psf_head_archive_normalize_output_sum": bool(archive_spec.get("normalize_output_sum", False)),
		"psf_head_archive_normalize_with_first": bool(archive_spec.get("normalize_with_first", False)),
		"psf_head_archive_normalize_first_only": bool(archive_spec.get("normalize_first_only", False)),
		"psf_head_archive_normalize_by_mean": bool(archive_spec.get("normalize_by_mean", False)),
		"pred_psf_supervised_sum_mean": float(tf.reduce_mean(pred_psf_sum).numpy()),
		"pred_psf_supervised_sum_std": float(tf.math.reduce_std(pred_psf_sum).numpy()),
		"pred_psf_supervised_max_mean": float(tf.reduce_mean(pred_psf_max).numpy()),
		"pred_psf_unc_input_sum_mean": float(tf.reduce_mean(tf.reduce_sum(joint_psf_unc_tail, axis=(1, 2), keepdims=True)).numpy()),
		"pred_psf_unc_input_sum_std": float(tf.math.reduce_std(tf.reduce_sum(joint_psf_unc_tail, axis=(1, 2), keepdims=True)).numpy()),
		"pred_psf_unc_input_max_mean": float(tf.reduce_mean(tf.reduce_max(joint_psf_unc_tail, axis=(1, 2), keepdims=True)).numpy()),
		"pred_psf_stage2prep_sum_mean": float(tf.reduce_mean(stage2_prepared_pred_psf_sum).numpy()),
		"pred_psf_stage2prep_sum_std": float(tf.math.reduce_std(stage2_prepared_pred_psf_sum).numpy()),
		"pred_psf_stage2prep_max_mean": float(tf.reduce_mean(stage2_prepared_pred_psf_max).numpy()),
		"pred_psf_stage2prep_over_joint_sum_ratio_mean": float(tf.reduce_mean(stage2_over_joint_sum_ratio).numpy()),
		"pred_psf_stage2prep_over_joint_sum_ratio_std": float(tf.math.reduce_std(stage2_over_joint_sum_ratio).numpy()),
		"psf_unc_input_tail_mae": float(joint_input_tail_mae.numpy()),
		"psf_unc_input_tail_rmse": float(joint_input_tail_rmse.numpy()),
		"pred_psf_physical_sum_mean": float(tf.reduce_mean(pred_psf_phys_sum).numpy()),
		"pred_psf_physical_sum_std": float(tf.math.reduce_std(pred_psf_phys_sum).numpy()),
		"pred_psf_physical_max_mean": float(tf.reduce_mean(pred_psf_phys_max).numpy()),
		"pred_psf_stage2prep_physical_sum_mean": float(tf.reduce_mean(stage2_pred_psf_phys_sum).numpy()),
		"pred_psf_stage2prep_physical_sum_std": float(tf.math.reduce_std(stage2_pred_psf_phys_sum).numpy()),
		"pred_psf_stage2prep_physical_max_mean": float(tf.reduce_mean(stage2_pred_psf_phys_max).numpy()),
		"pred_psf_recon_sum_mean": float(tf.reduce_mean(pred_psf_recon_sum).numpy()),
		"pred_psf_recon_sum_std": float(tf.math.reduce_std(pred_psf_recon_sum).numpy()),
		"pred_psf_recon_max_mean": float(tf.reduce_mean(pred_psf_recon_max).numpy()),
		"pred_psf_stage2prep_recon_sum_mean": float(tf.reduce_mean(stage2_pred_psf_recon_sum).numpy()),
		"pred_psf_stage2prep_recon_sum_std": float(tf.math.reduce_std(stage2_pred_psf_recon_sum).numpy()),
		"pred_psf_stage2prep_recon_max_mean": float(tf.reduce_mean(stage2_pred_psf_recon_max).numpy()),
		"pred_psf_sigma2_physical_mean": float(tf.reduce_mean(pred_sigma2_psf_phys).numpy()),
		"pred_psf_sigma2_recon_mean": float(tf.reduce_mean(pred_sigma2_psf_recon).numpy()),
		"pred_psf_stage2prep_sigma2_physical_mean": float(tf.reduce_mean(pred_sigma2_psf_stage2_phys).numpy()),
		"pred_psf_stage2prep_sigma2_recon_mean": float(tf.reduce_mean(pred_sigma2_psf_stage2_recon).numpy()),
		"proxy_nll_psf_joint_input": float(nll_psf_joint_proxy.numpy()),
		"proxy_nll_psf_joint_input_residual": float(nll_psf_joint_residual.numpy()),
		"proxy_nll_psf_joint_input_logsigma2": float(nll_psf_joint_logsigma.numpy()),
		"proxy_nll_psf_stage2prep_input": float(nll_psf_stage2_proxy.numpy()),
		"proxy_nll_psf_stage2prep_input_residual": float(nll_psf_stage2_residual.numpy()),
		"proxy_nll_psf_stage2prep_input_logsigma2": float(nll_psf_stage2_logsigma.numpy()),
		"truth_psf_supervised_sum_mean": float(tf.reduce_mean(truth_psf_sum).numpy()),
		"truth_psf_supervised_sum_std": float(tf.math.reduce_std(truth_psf_sum).numpy()),
		"truth_psf_physical_sum_mean": float(tf.reduce_mean(truth_psf_phys_sum).numpy()),
		"truth_psf_physical_sum_std": float(tf.math.reduce_std(truth_psf_phys_sum).numpy()),
		"truth_psf_recon_sum_mean": float(tf.reduce_mean(truth_psf_recon_sum).numpy()),
		"truth_psf_recon_sum_std": float(tf.math.reduce_std(truth_psf_recon_sum).numpy()),
	}


def _evaluate_and_print(model: tf.keras.Model, dataset: tf.data.Dataset, *, label: str, steps: int | None) -> dict[str, float]:
	metrics = model.evaluate(dataset, steps=steps, verbose=0, return_dict=True)
	result = {key: float(value) for key, value in metrics.items()}
	median_metrics = _collect_batch_median_metrics(
		model,
		dataset,
		metric_names=[m.name for m in model.metrics if m.name != "loss"],
		steps=steps,
	)
	parts = [_format_metrics(result)]
	if median_metrics:
		median_prefixed = {f"median_{key}": value for key, value in median_metrics.items()}
		parts.append(_format_metrics(median_prefixed))
	print(f"[joint_pinn_fourhead] Initial {label}: " + ", ".join(parts))
	return result


class _ValidationLossPrinter(tf.keras.callbacks.Callback):
	def __init__(self, metric_names: list[str], val_dataset: tf.data.Dataset, verbose: bool = True):
		super().__init__()
		self.metric_names = metric_names
		self.val_dataset = val_dataset
		self.verbose = verbose

	def _collect_validation_batch_metrics(self) -> dict[str, float]:
		return _collect_batch_median_metrics(
			self.model,
			self.val_dataset,
			metric_names=self.metric_names,
			steps=None,
		)

	def on_epoch_end(self, epoch, logs=None):
		if not self.verbose or not logs:
			return
		parts = []
		val_total = logs.get("val_loss")
		if val_total is not None:
			parts.append(f"val_loss={val_total:.6f}")
		for name in self.metric_names:
			val_name = f"val_{name}"
			if val_name in logs:
				parts.append(f"{val_name}={logs[val_name]:.6f}")
		median_metrics = self._collect_validation_batch_metrics()
		if median_metrics:
			parts.append("validation_batch_medians")
			if "loss" in median_metrics:
				parts.append(f"val_median_loss={median_metrics['loss']:.6f}")
			for name in self.metric_names:
				if name in median_metrics:
					parts.append(f"val_median_{name}={median_metrics[name]:.6f}")
		if parts:
			print(f"[joint_pinn_fourhead] Epoch {epoch + 1} validation: " + ", ".join(parts))


class _TrainingProgbar(tf.keras.callbacks.Callback):
	def __init__(self, *, verbose: bool = True):
		super().__init__()
		self.verbose = verbose
		self._metric_names = ["loss", "r2_pinn", "nll_im", "nll_psf", "nll_noise"]
		self._progbar: tf.keras.utils.Progbar | None = None
		self._target: int | None = None
		self._last_values: list[tuple[str, float]] = []

	def on_epoch_begin(self, epoch, logs=None):
		if not self.verbose:
			return
		self._target = self.params.get("steps")
		self._progbar = tf.keras.utils.Progbar(
			target=self._target,
			stateful_metrics=self._metric_names,
			unit_name="step",
		)
		self._last_values = []

	def on_train_batch_end(self, batch, logs=None):
		if not self.verbose or self._progbar is None:
			return
		logs = logs or {}
		values: list[tuple[str, float]] = []
		for name in self._metric_names:
			value = logs.get(name)
			if value is not None:
				values.append((name, float(value)))
		if values:
			self._last_values = values
		self._progbar.update(batch + 1, values=values)

	def on_epoch_end(self, epoch, logs=None):
		if not self.verbose or self._progbar is None:
			return
		if self._target is not None:
			self._progbar.update(self._target, values=self._last_values, finalize=True)
		self._progbar = None
		self._last_values = []


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Four-head joint PINN fine-tuning.")
	parser.add_argument("--config", type=Path, required=True, help="Path to experiment config .py file")
	return parser.parse_args()


def main() -> None:
	print("[joint_pinn_fourhead] Starting workflow")
	args = _parse_args()
	cfg = load_experiment_config(args.config)

	joint_config = dict(cfg.JOINT_PINN_CONFIG)
	dataset_config = dict(cfg.DATASET_LOAD_CONFIG)
	loss_config = dict(cfg.LOSS_CONFIG)
	output_base_dir = Path(cfg.OUTPUT_BASE_DIR)

	run_name = str(joint_config.get("run_name", "joint_pinn_fourhead"))
	run_dir = output_base_dir / run_name
	run_dir.mkdir(parents=True, exist_ok=True)
	print(f"[joint_pinn_fourhead] Output dir: {run_dir}")

	model_label = str(joint_config.get("head_model_label", "best_model")).strip().lower()
	im_run_name = str(cfg.IMAGE_HEAD_CONFIG.get("run_name", "image_only"))
	noise_run_name = str(cfg.NOISE_HEAD_CONFIG.get("run_name", "noise_only"))
	psf_run_name = str(cfg.PSF_HEAD_CONFIG.get("run_name", "psf_only"))
	psf_unc_run_name = str(cfg.PSF_UNC_CONFIG.get("run_name", "psf_uncertainty_stage2"))

	image_run_dir = output_base_dir / im_run_name
	noise_run_dir = output_base_dir / noise_run_name
	psf_mean_run_dir = output_base_dir / psf_run_name
	psf_unc_run_dir = output_base_dir / psf_unc_run_name

	image_model_path = _resolve_model_paths(image_run_dir)[model_label]
	noise_model_path = _resolve_model_paths(noise_run_dir)[model_label]
	psf_mean_model_path = _resolve_model_paths(psf_mean_run_dir)[model_label]
	psf_unc_model_path = _resolve_model_paths(psf_unc_run_dir)[model_label]

	print(f"[joint_pinn_fourhead] Image head      : {image_model_path}")
	print(f"[joint_pinn_fourhead] Noise head      : {noise_model_path}")
	print(f"[joint_pinn_fourhead] PSF mean head   : {psf_mean_model_path}")
	print(f"[joint_pinn_fourhead] PSF unc head    : {psf_unc_model_path}")

	print("[joint_pinn_fourhead] Rebuilding and loading pretrained heads")
	image_model, image_head_cfg = _load_independent_head_model(image_model_path)
	noise_model, noise_head_cfg = _load_independent_head_model(noise_model_path)
	psf_mean_model, psf_mean_head_cfg = _load_independent_head_model(psf_mean_model_path)
	psf_unc_model, psf_unc_head_cfg = _load_stage2_head_model(psf_unc_model_path)

	image_model.trainable = bool(joint_config.get("train_image_head", True))
	noise_model.trainable = bool(joint_config.get("train_noise_head", True))
	psf_mean_model.trainable = bool(joint_config.get("train_psf_mean_head", True))
	psf_unc_model.trainable = bool(joint_config.get("train_psf_unc_head", True))
	print(
		"[joint_pinn_fourhead] Trainable heads: "
		f"image={image_model.trainable}, noise={noise_model.trainable}, "
		f"psf_mean={psf_mean_model.trainable}, psf_unc={psf_unc_model.trainable}"
	)

	print("[joint_pinn_fourhead] Loading datasets")
	train_ds, val_ds = make_train_val_datasets(
		dataset_config["data_dir"],
		batch_size=dataset_config["batch_size"],
		val_batch_size=int(dataset_config.get("val_batch_size", dataset_config["batch_size"])),
		shuffle=dataset_config["shuffle"],
		repeat=dataset_config["repeat"],
		seed=dataset_config["seed"],
		channels_last=dataset_config["channels_last"],
		half_n_pix_crop=dataset_config["half_n_pix_crop"],
		fit_im=True,
		fit_psf=True,
		fit_noise=True,
		norm_psf=dataset_config["norm_psf"],
		norm_noise=dataset_config["norm_noise"],
		num_parallel_calls=dataset_config["num_parallel_calls"],
		prefetch=dataset_config["prefetch"],
	)

	pinn_weight = float(joint_config.get("pinn_weight", 1.0))
	im_weight = float(joint_config.get("im_weight", 1.0))
	psf_weight = float(joint_config.get("psf_weight", 1.0))
	noise_weight = float(joint_config.get("noise_weight", 1.0))
	reconstruction_crop = int(joint_config.get("reconstruction_crop", 16))
	psf_mean_source_norm_psf = dict(psf_mean_head_cfg.get("dataset", {})).get("norm_psf")
	psf_unc_input_norm_psf = dict(psf_unc_head_cfg.get("dataset", {})).get("norm_psf")

	model = FourHeadJointPinnModel(
		image_model,
		noise_model,
		psf_mean_model,
		psf_unc_model,
		pinn_weight=pinn_weight,
		im_weight=im_weight,
		psf_weight=psf_weight,
		noise_weight=noise_weight,
		log_sigma=bool(loss_config.get("log_sigma", False)),
		log_min=float(loss_config.get("log_min", -6.0)),
		log_max=float(loss_config.get("log_max", 20.0)),
		sigma2_eps=float(loss_config.get("sigma2_eps", 1e-12)),
		psf_mean_source_norm_psf=psf_mean_source_norm_psf,
		psf_unc_input_norm_psf=psf_unc_input_norm_psf,
		norm_psf=dataset_config.get("norm_psf"),
		norm_noise=dataset_config.get("norm_noise"),
		reconstruction_crop=reconstruction_crop,
		name="joint_pinn_fourhead_model",
	)

	preview_obs, preview_y_true = next(iter(train_ds.take(1)))
	_ = model(preview_obs, training=False)
	startup_psf_audit = _startup_psf_convention_audit(
		model=model,
		psf_mean_model_path=psf_mean_model_path,
		psf_mean_head_cfg=psf_mean_head_cfg,
		psf_unc_head_cfg=psf_unc_head_cfg,
		dataset_half_n_pix_crop=int(dataset_config.get("half_n_pix_crop", 0)),
		obs=preview_obs,
		y_true=preview_y_true,
	)
	print(f"[joint_pinn_fourhead] Startup PSF convention audit: {_format_debug_values(startup_psf_audit)}")

	n_epochs = int(joint_config.get("n_epochs", 100))
	lr_0 = float(joint_config.get("lr_0", 5e-4))
	lr_decay = float(joint_config.get("lr_decay", 10.0))
	n_steps_per_epoch = joint_config.get("n_steps_per_epoch")
	if n_steps_per_epoch is not None:
		n_steps_per_epoch = int(n_steps_per_epoch)
	verbose = bool(joint_config.get("verbose", True))
	jit_compile = bool(joint_config.get("jit_compile", False))
	initial_train_steps = int(joint_config.get("initial_eval_train_steps", 8))
	initial_val_steps = int(joint_config.get("initial_eval_val_steps", 8))

	optimizer = tf.keras.optimizers.Adam(learning_rate=lr_0, clipnorm=1.0)
	print(f"[joint_pinn_fourhead] Keras jit_compile={jit_compile}")
	model.compile(optimizer=optimizer, jit_compile=jit_compile)
	model.summary()
	if not model.trainable_variables:
		raise ValueError("All heads are frozen; at least one train_*_head must be true")

	print("[joint_pinn_fourhead] Evaluating frozen losses before training")
	initial_train_metrics = _evaluate_and_print(model, train_ds, label="train", steps=initial_train_steps)
	initial_val_metrics = _evaluate_and_print(model, val_ds, label="val", steps=initial_val_steps)
	train_truth_audit = _truth_reconstruction_audit(
		train_ds,
		steps=initial_train_steps,
		norm_psf=dataset_config.get("norm_psf"),
		norm_noise=dataset_config.get("norm_noise"),
		reconstruction_crop=reconstruction_crop,
	)
	val_truth_audit = _truth_reconstruction_audit(
		val_ds,
		steps=initial_val_steps,
		norm_psf=dataset_config.get("norm_psf"),
		norm_noise=dataset_config.get("norm_noise"),
		reconstruction_crop=reconstruction_crop,
	)
	if train_truth_audit:
		print(f"[joint_pinn_fourhead] Truth reconstruction audit train: {_format_metrics(train_truth_audit)}")
	if val_truth_audit:
		print(f"[joint_pinn_fourhead] Truth reconstruction audit val: {_format_metrics(val_truth_audit)}")
	train_prediction_audit = _prediction_reconstruction_audit(model, train_ds, steps=initial_train_steps)
	val_prediction_audit = _prediction_reconstruction_audit(model, val_ds, steps=initial_val_steps)
	if train_prediction_audit:
		print(f"[joint_pinn_fourhead] Prediction reconstruction audit train: {_format_metrics(train_prediction_audit)}")
	if val_prediction_audit:
		print(f"[joint_pinn_fourhead] Prediction reconstruction audit val: {_format_metrics(val_prediction_audit)}")

	def _lr_schedule(epoch, _):
		return lr_0 * 10 ** (-(epoch) / lr_decay)

	checkpoint_path = run_dir / "checkpoints" / "joint_pinn_fourhead_best.keras"
	checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

	callbacks: list[tf.keras.callbacks.Callback] = [
		tf.keras.callbacks.LearningRateScheduler(_lr_schedule, verbose=0),
		tf.keras.callbacks.TerminateOnNaN(),
		_TerminateOnNaNWithBatch(),
	]
	callbacks.append(_TrainingProgbar(verbose=verbose))
	callbacks.append(
		tf.keras.callbacks.ModelCheckpoint(
			filepath=str(checkpoint_path),
			monitor="val_loss",
			save_best_only=True,
		)
	)
	callbacks.append(_ValidationLossPrinter(metric_names=[m.name for m in model.metrics if m.name != "loss"], val_dataset=val_ds, verbose=verbose))
	callbacks.append(_SaveBestExamples(val_dataset=val_ds, save_dir=checkpoint_path.parent / "best_examples"))

	batch_history = _BatchHistory(metric_names=[m.name for m in model.metrics if m.name != "loss"])
	callbacks.append(batch_history)

	lr_tracker = _LrTracker()
	callbacks.append(lr_tracker)

	print("[joint_pinn_fourhead] Training model")
	history = model.fit(
		train_ds,
		validation_data=val_ds,
		epochs=n_epochs,
		steps_per_epoch=n_steps_per_epoch,
		verbose=0,
		callbacks=callbacks,
	)

	print("[joint_pinn_fourhead] Saving final model")
	model_path = run_dir / "model_final.keras"
	model.save(model_path)

	print("[joint_pinn_fourhead] Saving metrics")
	metrics_dir = run_dir / "metrics"
	metrics_dir.mkdir(parents=True, exist_ok=True)
	for key, values in history.history.items():
		np.save(metrics_dir / f"history_{key}.npy", np.asarray(values))
	for key, values in batch_history.history.items():
		np.save(metrics_dir / f"history_batch_{key}.npy", np.asarray(values))
	with (metrics_dir / "initial_metrics.json").open("w", encoding="utf-8") as handle:
		json.dump(
			{
				"startup_psf_convention_audit": startup_psf_audit,
				"train": initial_train_metrics,
				"val": initial_val_metrics,
				"truth_reconstruction_audit": {
					"train": train_truth_audit,
					"val": val_truth_audit,
				},
				"prediction_reconstruction_audit": {
					"train": train_prediction_audit,
					"val": val_prediction_audit,
				},
			},
			handle,
			indent=2,
		)

	best_values = history.history.get("val_loss", [])
	best_value = min(best_values) if best_values else None
	best_epoch = int(best_values.index(best_value)) + 1 if best_values else None

	np.save(metrics_dir / "best_metric.npy", np.asarray("val_loss"))
	np.save(metrics_dir / "best_value.npy", np.asarray(best_value))
	np.save(metrics_dir / "best_epoch.npy", np.asarray(best_epoch))
	np.save(metrics_dir / "lr_history.npy", np.asarray(lr_tracker.lr_history))
	np.save(metrics_dir / "checkpoint_path.npy", np.asarray(str(checkpoint_path)))
	np.save(metrics_dir / "model_path.npy", np.asarray(str(model_path)))

	print("[joint_pinn_fourhead] Saving config")
	config_path = run_dir / "training_config.json"
	with config_path.open("w", encoding="utf-8") as handle:
		json.dump(
			{
				"run": run_name,
				"workflow": "joint_pinn_fourhead_training",
				"dataset": dataset_config,
				"loss": {
					"type": "weighted_supervised_nll_plus_pinn_r2",
					"log_sigma": bool(loss_config.get("log_sigma", False)),
					"log_min": float(loss_config.get("log_min", -6.0)),
					"log_max": float(loss_config.get("log_max", 20.0)),
					"sigma2_eps": float(loss_config.get("sigma2_eps", 1e-12)),
					"weights": {
						"pinn": pinn_weight,
						"im": im_weight,
						"psf": psf_weight,
						"noise": noise_weight,
					},
					"reconstruction_crop": reconstruction_crop,
				},
				"normalization": {
					"psf_denorm_factor": float(model._psf_denorm_factor),
					"noise_denorm_factor": float(model._noise_denorm_factor),
					"psf_mean_source_norm_psf": psf_mean_source_norm_psf,
					"psf_unc_input_norm_psf": psf_unc_input_norm_psf,
					"norm_psf": dataset_config.get("norm_psf"),
					"norm_noise": dataset_config.get("norm_noise"),
				},
				"training": {
					"n_epochs": n_epochs,
					"lr_0": lr_0,
					"lr_decay": lr_decay,
					"jit_compile": jit_compile,
					"verbose": verbose,
					"n_steps_per_epoch": n_steps_per_epoch,
					"checkpoint_path": str(checkpoint_path),
					"initial_eval_train_steps": initial_train_steps,
					"initial_eval_val_steps": initial_val_steps,
				},
				"source_models": {
					"image": {
						"model_path": str(image_model_path),
						"run_dir": str(image_run_dir),
						"model_label": model_label,
						"trainable": bool(image_model.trainable),
						"training_config": image_head_cfg,
					},
					"noise": {
						"model_path": str(noise_model_path),
						"run_dir": str(noise_run_dir),
						"model_label": model_label,
						"trainable": bool(noise_model.trainable),
						"training_config": noise_head_cfg,
					},
					"psf_mean": {
						"model_path": str(psf_mean_model_path),
						"run_dir": str(psf_mean_run_dir),
						"model_label": model_label,
						"trainable": bool(psf_mean_model.trainable),
						"training_config": psf_mean_head_cfg,
					},
					"psf_unc": {
						"model_path": str(psf_unc_model_path),
						"run_dir": str(psf_unc_run_dir),
						"model_label": model_label,
						"trainable": bool(psf_unc_model.trainable),
						"training_config": psf_unc_head_cfg,
					},
				},
			},
			handle,
			indent=2,
		)

	print("[joint_pinn_fourhead] Generating plots")
	plots_dir = run_dir / "plots"
	plot_training_outputs(metrics_dir, plots_dir)

	print("[joint_pinn_fourhead] Done")


if __name__ == "__main__":
	main()