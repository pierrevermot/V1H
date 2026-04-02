
"""TFRecord dataset loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import tensorflow as tf


def list_tfrecord_files(data_dir: str | Path, pattern: str = "*.tfrecord") -> list[str]:
	"""List TFRecord files under a directory."""
	path = Path(data_dir)
	files = sorted(str(p) for p in path.glob(pattern))
	if not files:
		raise FileNotFoundError(f"No TFRecord files found in {path}")
	return files


def _parse_example(example_proto: tf.Tensor) -> dict[str, tf.Tensor]:
	# Accept both "noise" (new) and "residuals" (legacy) TFRecord keys.
	feature_spec = {
		"image": tf.io.FixedLenFeature([], tf.string),
		"obs": tf.io.FixedLenFeature([], tf.string),
		"psf": tf.io.FixedLenFeature([], tf.string),
		"noise": tf.io.FixedLenFeature([], tf.string, default_value=""),
		"residuals": tf.io.FixedLenFeature([], tf.string, default_value=""),
		"ref_psf": tf.io.FixedLenFeature([], tf.string, default_value=""),
		"n_frames": tf.io.FixedLenFeature([], tf.int64),
		"n_pix": tf.io.FixedLenFeature([], tf.int64),
	}
	features = tf.io.parse_single_example(example_proto, feature_spec)

	image = tf.io.parse_tensor(features["image"], out_type=tf.float32)
	obs = tf.io.parse_tensor(features["obs"], out_type=tf.float32)
	psf = tf.io.parse_tensor(features["psf"], out_type=tf.float32)

	# Prefer "noise" key; fall back to legacy "residuals" key.
	has_noise = tf.greater(tf.strings.length(features["noise"]), 0)
	noise = tf.cond(
		has_noise,
		lambda: tf.io.parse_tensor(features["noise"], out_type=tf.float32),
		lambda: tf.io.parse_tensor(features["residuals"], out_type=tf.float32),
	)

	n_frames = tf.cast(features["n_frames"], tf.int32)
	n_pix = tf.cast(features["n_pix"], tf.int32)
	has_ref_psf = tf.greater(tf.strings.length(features["ref_psf"]), 0)
	ref_psf = tf.cond(
		has_ref_psf,
		lambda: tf.reshape(tf.io.parse_tensor(features["ref_psf"], out_type=tf.float32), (n_pix, n_pix)),
		lambda: tf.zeros((n_pix, n_pix), dtype=tf.float32),
	)

	image = tf.reshape(image, (n_pix, n_pix))
	obs = tf.reshape(obs, (n_frames, n_pix, n_pix))
	psf = tf.reshape(psf, (n_frames, n_pix, n_pix))
	noise = tf.reshape(noise, (n_frames, n_pix, n_pix))

	return {
		"image": image,
		"obs": obs,
		"psf": psf,
		"noise": noise,
		"ref_psf": ref_psf,
		"has_ref_psf": has_ref_psf,
		"n_frames": n_frames,
		"n_pix": n_pix,
	}


def _apply_norm_tf(tensor: tf.Tensor, norm_config, spatial_axis: int = 1) -> tf.Tensor:
	"""Multiply *tensor* by a normalization factor.

	Parameters
	----------
	norm_config : str | float | None
		``"npix2"`` → multiply by n_pix², a ``float`` → multiply by that value,
		``None`` → identity (no normalization).
	spatial_axis : int
		Axis index of the first spatial dimension used to infer *n_pix*.
	"""
	if norm_config is None:
		return tensor
	if norm_config == "npix2":
		n = tf.cast(tf.shape(tensor)[spatial_axis], tensor.dtype)
		return tensor * (n * n)
	if isinstance(norm_config, str) and norm_config.lower() == "normalize_max":
		# Per-frame max-normalization (after ReLU) over spatial dimensions.
		# Works for PSF tensors shaped (F,H,W) with spatial_axis=1.
		x = tf.nn.relu(tensor)
		spatial_axes = (spatial_axis, spatial_axis + 1)
		max_val = tf.reduce_max(x, axis=spatial_axes, keepdims=True)
		eps = tf.cast(1e-12, x.dtype)
		return x / tf.maximum(max_val, eps)
	return tensor * tf.cast(float(norm_config), tensor.dtype)


def _prepare_example(
	features: dict[str, tf.Tensor],
	*,
	channels_last: bool,
	half_n_pix_crop: int,
	fit_im: bool,
	fit_psf: bool,
	fit_noise: bool,
	norm_psf: str | float | None = "npix2",
	norm_noise: str | float | None = "npix2",
) -> tuple[tf.Tensor, tf.Tensor]:
	image = features["image"]
	obs = features["obs"]
	psf = features["psf"]
	noise = features["noise"]

	if half_n_pix_crop > 0:
		c = int(half_n_pix_crop)
		image = image[c:-c, c:-c]
		obs = obs[:, c:-c, c:-c]
		psf = psf[:, c:-c, c:-c]
		noise = noise[:, c:-c, c:-c]

	psf = _apply_norm_tf(psf, norm_psf, spatial_axis=1)
	noise = _apply_norm_tf(noise, norm_noise, spatial_axis=1)

	output_parts = []
	if fit_im:
		image_cube = tf.expand_dims(image, axis=0)
		output_parts.append(image_cube)
	if fit_psf:
		output_parts.append(psf)
	if fit_noise:
		output_parts.append(noise)
	if not output_parts:
		raise ValueError("At least one of fit_im/fit_psf/fit_noise must be True")
	output_cube = tf.concat(output_parts, axis=0)

	if channels_last:
		obs = tf.transpose(obs, perm=(1, 2, 0))
		output_cube = tf.transpose(output_cube, perm=(1, 2, 0))

	return obs, output_cube


def make_dataset(
	data_dir: str | Path,
	*,
	batch_size: int = 8,
	shuffle: bool = True,
	repeat: bool = True,
	seed: int | None = None,
	channels_last: bool = True,
	half_n_pix_crop: int = 12,
	fit_im: bool = True,
	fit_psf: bool = True,
	fit_noise: bool = True,
	norm_psf: str | float | None = "npix2",
	norm_noise: str | float | None = "npix2",
	num_parallel_calls: int | None = None,
	prefetch: bool = True,
) -> tf.data.Dataset:
	"""Create a tf.data.Dataset for training.

	Parameters
	----------
	data_dir : str or Path
		Directory containing TFRecord files.
	batch_size : int, optional
		Batch size. Default 8.
	shuffle : bool, optional
		Whether to shuffle. Default True.
	repeat : bool, optional
		Whether to repeat indefinitely. Default True.
	seed : int or None, optional
		Shuffle seed.
	channels_last : bool, optional
		Return tensors as (n_pix, n_pix, channels) when True. Default True.
	half_n_pix_crop : int, optional
		Crop this many pixels from each side. Default 12.
	num_parallel_calls : int or None, optional
		Parallelism for map. Default AUTOTUNE.
	prefetch : bool, optional
		Whether to prefetch. Default True.
	"""
	files = list_tfrecord_files(data_dir)
	if num_parallel_calls is None:
		num_parallel_calls = tf.data.AUTOTUNE

	ds = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
	ds = ds.map(_parse_example, num_parallel_calls=num_parallel_calls)
	ds = ds.map(
		lambda feats: _prepare_example(
			feats,
			channels_last=channels_last,
			half_n_pix_crop=half_n_pix_crop,
			fit_im=fit_im,
			fit_psf=fit_psf,
			fit_noise=fit_noise,
			norm_psf=norm_psf,
			norm_noise=norm_noise,
		),
		num_parallel_calls=num_parallel_calls,
	)
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


def make_train_val_datasets(
	root_dir: str | Path,
	*,
	train_subdir: str = "train",
	val_subdir: str = "val",
	val_batch_size: int | None = None,
	**dataset_kwargs,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
	"""Create train/val datasets from a root directory."""
	root = Path(root_dir)
	train_ds = make_dataset(root / train_subdir, **dataset_kwargs)
	val_kwargs = dict(dataset_kwargs)
	if val_batch_size is not None:
		val_kwargs["batch_size"] = int(val_batch_size)
	val_ds = make_dataset(root / val_subdir, **val_kwargs)
	return train_ds, val_ds
