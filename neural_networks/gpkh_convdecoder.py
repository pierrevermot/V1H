"""Global Pooling Kernel Head with convolutional decoder builder."""

from __future__ import annotations

import math

import tensorflow as tf

from neural_networks.layers import GroupNormalization


def build_gpkh_convdecoder(
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
) -> tf.keras.Model:
	"""Build a GPKH variant with a convolutional decoder.

	This builder keeps the original GPKH encoder and global-pooling latent space,
	but replaces the final dense projection to the output cube with:
	- a dense projection to the encoder bottleneck spatial tensor
	- a learned convolutional decoder with bilinear upsampling
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

	def _conv_block(x, filters: int):
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

	height, width = int(input_shape[0]), int(input_shape[1])
	min_pix = min(height, width)
	max_blocks = max(1, int(math.floor(math.log2(min_pix))) - 2)
	n_blocks = min(4, max_blocks)
	divisor = 2**n_blocks
	if height % divisor != 0 or width % divisor != 0:
		raise ValueError("height and width must be divisible by 2**n_blocks")

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

	x = _conv_block(x, filters)
	bottleneck_height = height // divisor
	bottleneck_width = width // divisor
	bottleneck_filters = filters

	z_avg = tf.keras.layers.GlobalAveragePooling2D()(x)
	z_max = tf.keras.layers.GlobalMaxPooling2D()(x)
	z0 = tf.keras.layers.Concatenate()([z_avg, z_max])

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

	x = tf.keras.layers.Dense(
		bottleneck_height * bottleneck_width * bottleneck_filters,
		kernel_initializer=kernel_initializer,
		kernel_regularizer=kernel_regularizer,
		name="decoder_seed_dense",
	)(u)
	x = _activation(x)
	x = tf.keras.layers.Reshape(
		(bottleneck_height, bottleneck_width, bottleneck_filters),
		name="decoder_seed_reshape",
	)(x)
	x = _conv_block(x, bottleneck_filters)

	for _ in range(n_blocks):
		next_filters = max(filters // 2, int(base_filters))
		x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
		x = tf.keras.layers.Conv2D(
			next_filters,
			3,
			padding="same",
			kernel_initializer=kernel_initializer,
			kernel_regularizer=kernel_regularizer,
		)(x)
		x = _normalize(x, next_filters)
		x = _activation(x)
		x = _conv_block(x, next_filters)
		filters = next_filters

	outputs = tf.keras.layers.Conv2D(
		output_channels,
		1,
		activation=output_activation_function,
		padding="same",
		kernel_regularizer=kernel_regularizer,
		name="decoder_output",
	)(x)
	if normalize_output_sum:
		outputs = tf.keras.layers.Lambda(
			lambda raw: tf.nn.softplus(raw),
			name="output_softplus",
		)(outputs)
		if normalize_with_first:
			outputs = tf.keras.layers.Lambda(
				lambda pos: pos / (tf.reduce_sum(pos[..., :1], axis=(1, 2), keepdims=True) + tf.cast(1e-12, pos.dtype)),
				name="output_normalized_first_sum1",
			)(outputs)
		else:
			outputs = tf.keras.layers.Lambda(
				lambda pos: pos / (tf.reduce_sum(pos, axis=(1, 2), keepdims=True) + tf.cast(1e-12, pos.dtype)),
				name="output_normalized_sum1",
			)(outputs)

	return tf.keras.Model(inputs=inputs, outputs=outputs, name="gpkh_convdecoder")
