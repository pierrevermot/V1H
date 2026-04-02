"""TensorFlow U-Net builder."""

from __future__ import annotations

import math

import tensorflow as tf

from neural_networks.layers import GroupNormalization, _upsample_bilinear


def build_unet(
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
):
	"""Build a 2D U-Net model.

	Parameters
	----------
	input_shape : tuple[int, int, int]
		Shape as (height, width, channels). For a cube of n_images of size
		n_pix, use (n_pix, n_pix, n_images).
	output_shape : tuple[int, int, int]
		Shape as (height, width, channels) for outputs.
	layers_per_block : int, optional
		Number of convolution layers per encoder/decoder block. Default 4.
	base_filters : int, optional
		Number of filters in the first encoder block. Default 32.
	normalization : str, optional
		Normalization to apply after each convolution: "none", "batch",
		"group", or "instance". Default "none".
	group_norm_groups : int, optional
		Number of groups for group normalization. Default 8.
	weight_decay : float or None, optional
		L2 weight decay for convolution kernels. Default 1e-5. Set to 0 or None
		to disable.
	inner_activation_function : str or callable, optional
		Activation used for inner layers. Default "leakyrelu".
	output_activation_function : str or callable, optional
		Activation for output layer. Default "linear".

	Returns
	-------
	tf.keras.Model
		U-Net model.
	"""
	if layers_per_block < 1:
		raise ValueError("layers_per_block must be >= 1")
	if base_filters < 1:
		raise ValueError("base_filters must be >= 1")
	if input_shape[:2] != output_shape[:2]:
		raise ValueError("input_shape and output_shape must share spatial dimensions")

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

	# Determine number of blocks based on spatial size (at least 1, at most 5)
	height, width = int(input_shape[0]), int(input_shape[1])
	min_pix = min(height, width)
	max_blocks = max(1, int(math.floor(math.log2(min_pix))) - 2)
	n_blocks = min(4, max_blocks)
	divisor = 2**n_blocks
	if height % divisor != 0 or width % divisor != 0:
		raise ValueError("height and width must be divisible by 2**n_blocks")

	# Encoder
	output_channels = int(output_shape[-1])
	def _clip_filters(value: int) -> int:
		return max(int(value), output_channels)

	skips = []
	filters = int(base_filters)
	x = inputs
	for _ in range(n_blocks):
		x = _conv_block(x, _clip_filters(filters))
		skips.append(x)
		# Downsample with strided conv
		x = tf.keras.layers.Conv2D(
			_clip_filters(filters * 2),
			3,
			strides=2,
			padding="same",
			kernel_initializer=kernel_initializer,
			kernel_regularizer=kernel_regularizer,
		)(x)
		x = _normalize(x, _clip_filters(filters * 2))
		x = _activation(x)
		filters *= 2

	# Bottleneck
	x = _conv_block(x, _clip_filters(filters))

	# Decoder
	for skip in reversed(skips):
		filters //= 2
		x = tf.keras.layers.Lambda(_upsample_bilinear)(x)
		x = tf.keras.layers.Conv2D(
			_clip_filters(filters),
			3,
			padding="same",
			kernel_initializer=kernel_initializer,
			kernel_regularizer=kernel_regularizer,
		)(x)
		x = _normalize(x, _clip_filters(filters))
		x = _activation(x)
		x = tf.keras.layers.Concatenate()([x, skip])
		x = _conv_block(x, _clip_filters(filters))

	# Output
	outputs = tf.keras.layers.Conv2D(
		output_channels,
		1,
		activation=output_activation_function,
		padding="same",
		kernel_regularizer=kernel_regularizer,
	)(x)

	return tf.keras.Model(inputs=inputs, outputs=outputs, name="unet")
