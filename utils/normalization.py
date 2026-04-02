"""Normalization helpers for tensor data."""
from __future__ import annotations

import tensorflow as tf


def _compute_norm_factor(norm_config, n_pix_crop: int | None = None) -> float:
	if norm_config is None:
		return 1.0
	if isinstance(norm_config, str) and norm_config == "npix2":
		if n_pix_crop is None:
			raise ValueError("Cannot compute 'npix2' factor: spatial size unknown")
		return float(n_pix_crop) ** 2
	if isinstance(norm_config, str) and norm_config.lower() == "normalize_max":
		return 1.0
	return float(norm_config)


def _apply_norm_to_tensor(tensor: tf.Tensor, norm_config, *, spatial_axis: int = 1) -> tf.Tensor:
	if norm_config is None:
		return tensor
	if norm_config == "npix2":
		n = tf.cast(tf.shape(tensor)[spatial_axis], tensor.dtype)
		return tensor * (n * n)
	if isinstance(norm_config, str) and norm_config.lower() == "normalize_max":
		x = tf.nn.relu(tensor)
		spatial_axes = (spatial_axis, spatial_axis + 1)
		max_val = tf.reduce_max(x, axis=spatial_axes, keepdims=True)
		eps = tf.cast(1e-12, x.dtype)
		return x / tf.maximum(max_val, eps)
	return tensor * tf.cast(float(norm_config), tensor.dtype)


def _remove_norm_from_tensor(tensor: tf.Tensor, norm_config, *, spatial_axis: int = 1) -> tf.Tensor:
	if norm_config is None:
		return tensor
	if norm_config == "npix2":
		n = tf.cast(tf.shape(tensor)[spatial_axis], tensor.dtype)
		return tensor / (n * n)
	if isinstance(norm_config, str) and norm_config.lower() == "normalize_max":
		raise ValueError("Cannot invert 'normalize_max' normalization for PSF conversion")
	return tensor / tf.cast(float(norm_config), tensor.dtype)


def _convert_normed_tensor(
	tensor: tf.Tensor,
	*,
	source_norm,
	target_norm,
	spatial_axis: int = 1,
) -> tf.Tensor:
	if source_norm == target_norm:
		return tensor
	physical = _remove_norm_from_tensor(tensor, source_norm, spatial_axis=spatial_axis)
	return _apply_norm_to_tensor(physical, target_norm, spatial_axis=spatial_axis)


def _normalize_psf_for_observation(
	psf: tf.Tensor,
	*,
	sigma2_psf: tf.Tensor | None = None,
) -> tuple[tf.Tensor, tf.Tensor | None, tf.Tensor]:
	psf_sum = tf.reduce_sum(psf, axis=(1, 2), keepdims=True)
	eps = tf.cast(1e-12, psf.dtype)
	psf_sum_safe = tf.where(
		psf_sum >= 0,
		tf.maximum(psf_sum, eps),
		tf.minimum(psf_sum, -eps),
	)
	psf_unit = psf / psf_sum_safe
	if sigma2_psf is None:
		return psf_unit, None, psf_sum
	return psf_unit, sigma2_psf / tf.square(psf_sum_safe), psf_sum


def _apply_norm_tf(tensor: tf.Tensor, norm_config, spatial_axis: int = 1) -> tf.Tensor:
	return _apply_norm_to_tensor(tensor, norm_config, spatial_axis=spatial_axis)
