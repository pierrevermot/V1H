"""Loss functions, metrics, and prediction splitting utilities."""
from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf

MAKE_LOSS_ALLOWED_KEYS = {
	"loss",
	"log_sigma",
	"log_min",
	"log_max",
	"sigma2_eps",
	"charb_eps",
	"half_n_pix_crop",
	"use_pinn",
	"fit_im",
	"fit_psf",
	"fit_noise",
	"norm_psf",
	"norm_noise",
}


def _var_normalized_mse(truth: tf.Tensor, pred: tf.Tensor, *, eps: float = 1e-12) -> tf.Tensor:
	den = tf.math.reduce_variance(truth) + tf.cast(eps, truth.dtype)
	num = tf.reduce_mean(tf.square(truth - pred))
	return num / den


def _make_prediction_only_loss(metric_name: str):
	def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
		return _var_normalized_mse(y_true, y_pred)

	def components(y_true: tf.Tensor, y_pred: tf.Tensor) -> dict[str, tf.Tensor]:
		return {metric_name: _var_normalized_mse(y_true, y_pred)}

	loss_fn.components = components
	loss_fn.component_names = [metric_name]
	return loss_fn


def _gaussian_nll(truth: tf.Tensor, mean: tf.Tensor, log_sigma2: tf.Tensor) -> tf.Tensor:
	err2 = tf.square(mean - truth)
	return tf.reduce_mean(err2 * tf.exp(-log_sigma2) + log_sigma2)


def _soft_clip(x: tf.Tensor, lo: float, hi: float, softness: float = 0.5) -> tf.Tensor:
	x = lo + softness * tf.nn.softplus((x - lo) / softness)
	x = hi - softness * tf.nn.softplus((hi - x) / softness)
	return x


def _log10_clip_to_ln(x: tf.Tensor, log10_min: float, log10_max: float) -> tf.Tensor:
	ln10 = tf.constant(2.302585092994046, dtype=x.dtype)
	lo = tf.cast(log10_min, x.dtype) * ln10
	hi = tf.cast(log10_max, x.dtype) * ln10
	x = lo + 0.5 * tf.nn.softplus((x - lo) / 0.5)
	x = hi - 0.5 * tf.nn.softplus((hi - x) / 0.5)
	return x


def _pred_to_sigma2(
	pred_unc: tf.Tensor,
	*,
	log_sigma: bool,
	log_min: float,
	log_max: float,
	sigma2_eps: float,
) -> tuple[tf.Tensor, tf.Tensor]:
	if log_sigma:
		log_sigma2 = _log10_clip_to_ln(pred_unc, log_min, log_max)
		sigma2 = tf.exp(log_sigma2) + tf.cast(sigma2_eps, pred_unc.dtype)
		return sigma2, log_sigma2
	sigma2 = tf.nn.softplus(pred_unc) + tf.cast(sigma2_eps, pred_unc.dtype)
	log_sigma2 = _log10_clip_to_ln(tf.math.log(sigma2), log_min, log_max)
	sigma2 = tf.exp(log_sigma2) + tf.cast(sigma2_eps, pred_unc.dtype)
	return sigma2, log_sigma2


def _gaussian_nll_parts(truth: tf.Tensor, pred: tf.Tensor, sigma2: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
	err2 = tf.square(pred - truth)
	residual_term = tf.reduce_mean(err2 / sigma2)
	logsigma2_term = tf.reduce_mean(tf.math.log(sigma2))
	total = residual_term + logsigma2_term
	return total, residual_term, logsigma2_term


def _split_truth(y_true: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
	channels = tf.shape(y_true)[-1]
	n_frames = (channels - 1) // 2
	im = y_true[..., :1]
	psf = y_true[..., 1 : 1 + n_frames]
	res = y_true[..., 1 + n_frames : 1 + 2 * n_frames]
	return im, psf, res, n_frames


def _split_pred(pred: tf.Tensor, n_frames: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
	start = 0
	im = pred[..., start : start + 1]
	start += 1
	psf = pred[..., start : start + n_frames]
	start += n_frames
	res = pred[..., start : start + n_frames]
	return im, psf, res


def _normalized_residual(truth: np.ndarray, pred: np.ndarray, sigma: np.ndarray | None, *, eps: float = 1e-12) -> np.ndarray:
	truth = np.asarray(truth, dtype=np.float32)
	pred = np.asarray(pred, dtype=np.float32)
	if sigma is None:
		return truth - pred
	sigma = np.asarray(sigma, dtype=np.float32)
	return (truth - pred) / np.maximum(sigma, float(eps))


def _filter_make_loss_kwargs(loss_cfg: dict[str, Any]) -> dict[str, Any]:
	return {key: value for key, value in loss_cfg.items() if key in MAKE_LOSS_ALLOWED_KEYS}
