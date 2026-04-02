"""Global Pooling Kernel Head (GPKH) model builder."""

from __future__ import annotations

import math

import tensorflow as tf

from neural_networks.layers import GroupNormalization


def build_gpkh(
	input_shape: tuple[int, int, int],
	output_shape: tuple[int, int, int],
	*,
	layers_per_block: int = 4,
	base_filters: int = 32,
	normalization: str = "none",
	group_norm_groups: int = 8,
	weight_decay: float | None = 1e-5,
	inner_activation_function="leakyrelu",
	output_activation_function="linear",
	latent_dim: int = 128,
	normalize_output_sum: bool = True,
	normalize_with_first: bool = True,
	normalize_first_only: bool = False,
	normalize_by_mean: bool = False,
) -> tf.keras.Model:
	"""Build a GPKH model.

	Parameters
	----------
	input_shape : tuple[int, int, int]
		Shape as (height, width, channels).
	output_shape : tuple[int, int, int]
		Shape as (height, width, channels) for outputs.
	layers_per_block : int, optional
		Number of convolution layers per encoder block. Default 4.
	base_filters : int, optional
		Number of filters in the first encoder block. Default 32.
	normalization : str, optional
		Normalization to apply after each convolution: "none", "batch",
		"group", or "instance". Default "none".
	group_norm_groups : int, optional
		Number of groups for group normalization. Default 8.
	weight_decay : float or None, optional
		L2 weight decay for kernels. Default 1e-5. Set to 0 or None to disable.
	inner_activation_function : str or callable, optional
		Activation used for inner layers. Default "leakyrelu".
	output_activation_function : str or callable, optional
		Activation for output layer. Default "linear".
	latent_dim : int, optional
		Latent dimension for the shared head. Default 128.
	Returns
	-------
	tf.keras.Model
		GPKH model.
	"""
	if layers_per_block < 1:
		raise ValueError("layers_per_block must be >= 1")
	if base_filters < 1:
		raise ValueError("base_filters must be >= 1")
	if latent_dim < 1:
		raise ValueError("latent_dim must be >= 1")
	if input_shape[:2] != output_shape[:2]:
		raise ValueError("input_shape and output_shape must share spatial dimensions")

	output_height, output_width, output_channels = (
		int(output_shape[0]),
		int(output_shape[1]),
		int(output_shape[2]),
	)
	n_outputs = output_channels

	inputs = tf.keras.Input(shape=input_shape)

	normalization = normalization.lower().strip()
	if normalization not in {"none", "batch", "group", "instance"}:
		raise ValueError("normalization must be one of: none, batch, group, instance")

	if weight_decay is None or weight_decay <= 0:
		kernel_regularizer = None
	else:
		kernel_regularizer = tf.keras.regularizers.l2(weight_decay)

	use_he_init = (
		isinstance(inner_activation_function, str)
		and inner_activation_function.lower() == "leakyrelu"
	)
	kernel_initializer = tf.keras.initializers.HeNormal() if use_he_init else "glorot_uniform"

	def _activation(x):
		if isinstance(inner_activation_function, str) and inner_activation_function.lower() == "leakyrelu":
			return tf.keras.layers.LeakyReLU(negative_slope=0.05)(x)
		return tf.keras.layers.Activation(inner_activation_function)(x)

	def _normalize(x, filters: int):
		if normalization == "batch":
			return tf.keras.layers.BatchNormalization()(x)
		if normalization == "group":
			return GroupNormalization(groups=group_norm_groups)(x)
		if normalization == "instance":
			return GroupNormalization(groups=filters)(x)
		return x

	def _conv_block(x, filters):
		for _ in range(layers_per_block):
			x = tf.keras.layers.Conv2D(
				filters,
				3,
				padding="same",
				kernel_initializer=kernel_initializer,
				kernel_regularizer=kernel_regularizer,
			)(x)
			x = _normalize(x, filters)
			x = _activation(x)
		return x

	def _normalize_output(pos: tf.Tensor) -> tf.Tensor:
		reducer = tf.reduce_mean if normalize_by_mean else tf.reduce_sum
		eps = tf.cast(1e-12, pos.dtype)
		if normalize_first_only:
			first = pos[..., :1] / (reducer(pos[..., :1], axis=(1, 2), keepdims=True) + eps)
			return tf.concat([first, pos[..., 1:]], axis=-1)
		if normalize_with_first:
			return pos / (reducer(pos[..., :1], axis=(1, 2), keepdims=True) + eps)
		return pos / (reducer(pos, axis=(1, 2), keepdims=True) + eps)

	# Determine number of blocks based on spatial size (at least 1, at most 5)
	height, width = int(input_shape[0]), int(input_shape[1])
	min_pix = min(height, width)
	max_blocks = max(1, int(math.floor(math.log2(min_pix))) - 2)
	n_blocks = min(4, max_blocks)
	divisor = 2**n_blocks
	if height % divisor != 0 or width % divisor != 0:
		raise ValueError("height and width must be divisible by 2**n_blocks")

	# Encoder
	filters = int(base_filters)
	x = inputs
	for _ in range(n_blocks):
		x = _conv_block(x, filters)
		x = tf.keras.layers.Conv2D(
			filters * 2,
			3,
			strides=2,
			padding="same",
			kernel_initializer=kernel_initializer,
			kernel_regularizer=kernel_regularizer,
		)(x)
		x = _normalize(x, filters * 2)
		x = _activation(x)
		filters *= 2

	# Bottleneck
	x = _conv_block(x, filters)

	# Use global pooling (XLA-safe) instead of Flatten on feature maps.
	# Flatten can trigger dynamic reshape issues under some TF/XLA builds.
	z_avg = tf.keras.layers.GlobalAveragePooling2D()(x)
	z_max = tf.keras.layers.GlobalMaxPooling2D()(x)
	z0 = tf.keras.layers.Concatenate()([z_avg, z_max])

	# Single latent space
	u = tf.keras.layers.Dense(
		2 * latent_dim,
		kernel_initializer=kernel_initializer,
		kernel_regularizer=kernel_regularizer,
	)(z0)
	u = _activation(u)
	u = tf.keras.layers.Dense(
		latent_dim,
		kernel_initializer=kernel_initializer,
		kernel_regularizer=kernel_regularizer,
	)(u)
	u = _activation(u)

	# Fully connected projection to output cube
	psfs = tf.keras.layers.Dense(
		output_height * output_width * output_channels,
		activation=output_activation_function,
		kernel_initializer=kernel_initializer,
		kernel_regularizer=kernel_regularizer,
	)(u)
	psfs = tf.keras.layers.Reshape((output_height, output_width, output_channels))(psfs)
	if normalize_output_sum:
		psfs = tf.keras.layers.Lambda(
			lambda raw: tf.nn.softplus(raw),
			name="output_softplus",
		)(psfs)
		if normalize_first_only:
			psfs = tf.keras.layers.Lambda(
				_normalize_output,
				name=("output_normalized_first_only_mean1" if normalize_by_mean else "output_normalized_first_only_sum1"),
			)(psfs)
		elif normalize_with_first:
			psfs = tf.keras.layers.Lambda(
				_normalize_output,
				name=("output_normalized_first_mean1" if normalize_by_mean else "output_normalized_first_sum1"),
			)(psfs)
		else:
			psfs = tf.keras.layers.Lambda(
				_normalize_output,
				name=("output_normalized_mean1" if normalize_by_mean else "output_normalized_sum1"),
			)(psfs)

	return tf.keras.Model(inputs=inputs, outputs=psfs, name="gpkh")
