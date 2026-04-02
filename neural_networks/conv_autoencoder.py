"""Purely convolutional autoencoder builders.

This module exposes three public builders:
- ``build_conv_autoencoder``: full encoder-decoder network
- ``build_conv_autoencoder_encoder``: encoder half only
- ``build_conv_autoencoder_decoder``: decoder half only

The latent space is a 3D tensor with shape
``(height / 2**n_layers, width / 2**n_layers, latent_depth)``.
"""

from __future__ import annotations

import tensorflow as tf

from neural_networks.layers import GroupNormalization


VALID_NORMALIZATIONS = {"none", "batch", "group", "instance"}


def _validate_common_params(
	*,
	n_convs_per_layer: int,
	n_filters: int,
	n_layers: int,
	latent_depth: int,
	normalization: str,
) -> str:
	if n_convs_per_layer < 1:
		raise ValueError("n_convs_per_layer must be >= 1")
	if n_filters < 1:
		raise ValueError("n_filters must be >= 1")
	if n_layers < 1:
		raise ValueError("n_layers must be >= 1")
	if latent_depth < 1:
		raise ValueError("latent_depth must be >= 1")
	normalization = normalization.lower().strip()
	if normalization not in VALID_NORMALIZATIONS:
		raise ValueError("normalization must be one of: none, batch, group, instance")
	return normalization


def _make_kernel_regularizer(weight_decay: float | None):
	if weight_decay is None or weight_decay <= 0:
		return None
	return tf.keras.regularizers.l2(weight_decay)


def _make_kernel_initializer(inner_activation_function):
	use_he_init = (
		isinstance(inner_activation_function, str)
		and inner_activation_function.lower() == "leakyrelu"
	)
	return tf.keras.initializers.HeNormal() if use_he_init else "glorot_uniform"


def _make_activation(inner_activation_function):
	def _activation(x):
		if isinstance(inner_activation_function, str) and inner_activation_function.lower() == "leakyrelu":
			return tf.keras.layers.LeakyReLU(negative_slope=0.05)(x)
		return tf.keras.layers.Activation(inner_activation_function)(x)

	return _activation


def _make_normalize(normalization: str, group_norm_groups: int):
	def _normalize(x, filters: int):
		if normalization == "batch":
			return tf.keras.layers.BatchNormalization()(x)
		if normalization == "group":
			return GroupNormalization(groups=group_norm_groups)(x)
		if normalization == "instance":
			return GroupNormalization(groups=filters)(x)
		return x

	return _normalize


def _conv_block(
	x,
	*,
	filters: int,
	n_convs_per_layer: int,
	kernel_initializer,
	kernel_regularizer,
	normalize,
	activation,
):
	for _ in range(n_convs_per_layer):
		x = tf.keras.layers.Conv2D(
			filters,
			3,
			padding="same",
			kernel_initializer=kernel_initializer,
			kernel_regularizer=kernel_regularizer,
		)(x)
		x = normalize(x, filters)
		x = activation(x)
	return x


def _validate_spatial_divisibility(height: int, width: int, n_layers: int) -> None:
	divisor = 2**n_layers
	if height % divisor != 0 or width % divisor != 0:
		raise ValueError(f"height and width must be divisible by 2**n_layers = {divisor}")


def _latent_shape_from_input_shape(
	input_shape: tuple[int, int, int],
	*,
	n_layers: int,
	latent_depth: int,
) -> tuple[int, int, int]:
	height, width = int(input_shape[0]), int(input_shape[1])
	_validate_spatial_divisibility(height, width, n_layers)
	divisor = 2**n_layers
	return (height // divisor, width // divisor, int(latent_depth))


def build_conv_autoencoder_encoder(
	input_shape: tuple[int, int, int],
	*,
	n_convs_per_layer: int = 2,
	n_filters: int = 32,
	n_layers: int = 4,
	latent_depth: int = 128,
	normalization: str = "none",
	group_norm_groups: int = 8,
	weight_decay: float | None = 1e-5,
	inner_activation_function="relu",
) -> tf.keras.Model:
	"""Build the encoder half of the convolutional autoencoder."""
	if len(input_shape) != 3:
		raise ValueError("input_shape must be a 3-tuple (height, width, channels)")
	normalization = _validate_common_params(
		n_convs_per_layer=n_convs_per_layer,
		n_filters=n_filters,
		n_layers=n_layers,
		latent_depth=latent_depth,
		normalization=normalization,
	)
	height, width = int(input_shape[0]), int(input_shape[1])
	_validate_spatial_divisibility(height, width, n_layers)

	kernel_regularizer = _make_kernel_regularizer(weight_decay)
	kernel_initializer = _make_kernel_initializer(inner_activation_function)
	activation = _make_activation(inner_activation_function)
	normalize = _make_normalize(normalization, group_norm_groups)

	inputs = tf.keras.Input(shape=input_shape, name="encoder_input")
	x = inputs
	filters = int(n_filters)
	for _ in range(n_layers):
		x = _conv_block(
			x,
			filters=filters,
			n_convs_per_layer=n_convs_per_layer,
			kernel_initializer=kernel_initializer,
			kernel_regularizer=kernel_regularizer,
			normalize=normalize,
			activation=activation,
		)
		x = tf.keras.layers.Conv2D(
			filters * 2,
			3,
			strides=2,
			padding="same",
			kernel_initializer=kernel_initializer,
			kernel_regularizer=kernel_regularizer,
		)(x)
		x = normalize(x, filters * 2)
		x = activation(x)
		filters *= 2

	x = _conv_block(
		x,
		filters=filters,
		n_convs_per_layer=n_convs_per_layer,
		kernel_initializer=kernel_initializer,
		kernel_regularizer=kernel_regularizer,
		normalize=normalize,
		activation=activation,
	)
	x = tf.keras.layers.Conv2D(
		latent_depth,
		1,
		padding="same",
		kernel_initializer=kernel_initializer,
		kernel_regularizer=kernel_regularizer,
		name="latent_projection",
	)(x)

	return tf.keras.Model(inputs=inputs, outputs=x, name="cae_encoder")


def build_conv_autoencoder_decoder(
	latent_shape: tuple[int, int, int],
	output_shape: tuple[int, int, int],
	*,
	n_convs_per_layer: int = 2,
	n_filters: int = 32,
	n_layers: int = 4,
	latent_depth: int = 128,
	normalization: str = "none",
	group_norm_groups: int = 8,
	weight_decay: float | None = 1e-5,
	inner_activation_function="relu",
	output_activation_function="linear",
	normalize_output_sum: bool = True,
) -> tf.keras.Model:
	"""Build the decoder half of the convolutional autoencoder."""
	if len(latent_shape) != 3:
		raise ValueError("latent_shape must be a 3-tuple (height, width, channels)")
	if len(output_shape) != 3:
		raise ValueError("output_shape must be a 3-tuple (height, width, channels)")
	normalization = _validate_common_params(
		n_convs_per_layer=n_convs_per_layer,
		n_filters=n_filters,
		n_layers=n_layers,
		latent_depth=latent_depth,
		normalization=normalization,
	)
	if int(latent_shape[-1]) != int(latent_depth):
		raise ValueError("latent_shape[-1] must match latent_depth")

	latent_height, latent_width = int(latent_shape[0]), int(latent_shape[1])
	output_height, output_width, output_channels = (
		int(output_shape[0]),
		int(output_shape[1]),
		int(output_shape[2]),
	)
	expected_height = latent_height * (2**n_layers)
	expected_width = latent_width * (2**n_layers)
	if output_height != expected_height or output_width != expected_width:
		raise ValueError(
			"output spatial dimensions must equal latent spatial dimensions multiplied by 2**n_layers"
		)

	kernel_regularizer = _make_kernel_regularizer(weight_decay)
	kernel_initializer = _make_kernel_initializer(inner_activation_function)
	activation = _make_activation(inner_activation_function)
	normalize = _make_normalize(normalization, group_norm_groups)

	inputs = tf.keras.Input(shape=latent_shape, name="decoder_input")
	filters = int(n_filters) * (2**n_layers)
	x = tf.keras.layers.Conv2D(
		filters,
		1,
		padding="same",
		kernel_initializer=kernel_initializer,
		kernel_regularizer=kernel_regularizer,
		name="decoder_input_projection",
	)(inputs)
	x = normalize(x, filters)
	x = activation(x)
	x = _conv_block(
		x,
		filters=filters,
		n_convs_per_layer=n_convs_per_layer,
		kernel_initializer=kernel_initializer,
		kernel_regularizer=kernel_regularizer,
		normalize=normalize,
		activation=activation,
	)

	for _ in range(n_layers):
		next_filters = max(filters // 2, int(n_filters))
		x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
		x = tf.keras.layers.Conv2D(
			next_filters,
			3,
			padding="same",
			kernel_initializer=kernel_initializer,
			kernel_regularizer=kernel_regularizer,
		)(x)
		x = normalize(x, next_filters)
		x = activation(x)
		x = _conv_block(
			x,
			filters=next_filters,
			n_convs_per_layer=n_convs_per_layer,
			kernel_initializer=kernel_initializer,
			kernel_regularizer=kernel_regularizer,
			normalize=normalize,
			activation=activation,
		)
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
			name="decoder_output_softplus",
		)(outputs)
		outputs = tf.keras.layers.Lambda(
			lambda pos: pos / (tf.reduce_sum(pos, axis=(1, 2), keepdims=True) + tf.cast(1e-12, pos.dtype)),
			name="decoder_output_normalized_sum1",
		)(outputs)
	return tf.keras.Model(inputs=inputs, outputs=outputs, name="cae_decoder")


def build_conv_autoencoder(
	input_shape: tuple[int, int, int],
	output_shape: tuple[int, int, int],
	*,
	n_convs_per_layer: int = 2,
	n_filters: int = 32,
	n_layers: int = 4,
	latent_depth: int = 128,
	normalization: str = "none",
	group_norm_groups: int = 8,
	weight_decay: float | None = 1e-5,
	inner_activation_function="relu",
	output_activation_function="linear",
	normalize_output_sum: bool = True,
) -> tf.keras.Model:
	"""Build the full convolutional autoencoder."""
	if input_shape[:2] != output_shape[:2]:
		raise ValueError("input_shape and output_shape must share spatial dimensions")
	encoder = build_conv_autoencoder_encoder(
		input_shape,
		n_convs_per_layer=n_convs_per_layer,
		n_filters=n_filters,
		n_layers=n_layers,
		latent_depth=latent_depth,
		normalization=normalization,
		group_norm_groups=group_norm_groups,
		weight_decay=weight_decay,
		inner_activation_function=inner_activation_function,
	)
	latent_shape = _latent_shape_from_input_shape(
		input_shape,
		n_layers=n_layers,
		latent_depth=latent_depth,
	)
	decoder = build_conv_autoencoder_decoder(
		latent_shape,
		output_shape,
		n_convs_per_layer=n_convs_per_layer,
		n_filters=n_filters,
		n_layers=n_layers,
		latent_depth=latent_depth,
		normalization=normalization,
		group_norm_groups=group_norm_groups,
		weight_decay=weight_decay,
		inner_activation_function=inner_activation_function,
		output_activation_function=output_activation_function,
		normalize_output_sum=normalize_output_sum,
	)
	inputs = tf.keras.Input(shape=input_shape, name="cae_input")
	latent = encoder(inputs)
	outputs = decoder(latent)
	return tf.keras.Model(inputs=inputs, outputs=outputs, name="cae")
