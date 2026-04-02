"""Dense PSF head model builder.

This model is intentionally minimal: it maps the full observation cube to the PSF
output cube using only dense layers. To stay XLA-friendly, it avoids `Flatten`
and uses an explicit static `Reshape((flat_dim,))` like `neural_networks/skh.py`.
"""

from __future__ import annotations

import tensorflow as tf


def build_dense_psf(
	input_shape: tuple[int, int, int],
	output_shape: tuple[int, int, int],
	*,
	n_layers: int = 2,
	n_per_layer: int = 1024,
	normalization: str = "none",
	group_norm_groups: int = 8,
	weight_decay: float | None = 0.0,
	inner_activation_function="relu",
	output_activation_function="linear",
) -> tf.keras.Model:
	"""Build a dense PSF head.

	Parameters are aligned with the other model builders for easier integration.
	`normalization` and `group_norm_groups` are accepted for config compatibility
	but are not used by this pure dense model.
	"""
	if n_layers < 1:
		raise ValueError("n_layers must be >= 1")
	if n_per_layer < 1:
		raise ValueError("n_per_layer must be >= 1")

	input_height, input_width, input_channels = (int(input_shape[0]), int(input_shape[1]), int(input_shape[2]))
	output_height, output_width, output_channels = (int(output_shape[0]), int(output_shape[1]), int(output_shape[2]))
	_ = normalization
	_ = group_norm_groups

	flat_input_dim = int(input_height * input_width * input_channels)
	flat_output_dim = int(output_height * output_width * output_channels)
	if flat_input_dim < 1 or flat_output_dim < 1:
		raise ValueError("input_shape and output_shape must define strictly positive tensor sizes")

	if weight_decay is None or weight_decay <= 0:
		kernel_regularizer = None
	else:
		kernel_regularizer = tf.keras.regularizers.l2(weight_decay)

	use_he_init = isinstance(inner_activation_function, str) and inner_activation_function.lower() in {"relu", "leakyrelu"}
	kernel_initializer = tf.keras.initializers.HeNormal() if use_he_init else "glorot_uniform"

	def _activation(x: tf.Tensor) -> tf.Tensor:
		if isinstance(inner_activation_function, str) and inner_activation_function.lower() == "leakyrelu":
			return tf.keras.layers.LeakyReLU(negative_slope=0.05)(x)
		return tf.keras.layers.Activation(inner_activation_function)(x)

	inputs = tf.keras.Input(shape=input_shape)
	x = tf.keras.layers.Reshape((flat_input_dim,))(inputs)
	for layer_idx in range(n_layers):
		x = tf.keras.layers.Dense(
			n_per_layer,
			kernel_initializer=kernel_initializer,
			kernel_regularizer=kernel_regularizer,
			name=f"dense_hidden_{layer_idx + 1}",
		)(x)
		x = _activation(x)

	x = tf.keras.layers.Dense(
		flat_output_dim,
		activation=output_activation_function,
		kernel_initializer=kernel_initializer,
		kernel_regularizer=kernel_regularizer,
		name="dense_output",
	)(x)
	outputs = tf.keras.layers.Reshape((output_height, output_width, output_channels))(x)
	return tf.keras.Model(inputs=inputs, outputs=outputs, name="dense")
