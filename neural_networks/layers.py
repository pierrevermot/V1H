"""Reusable Keras layers and building blocks.

``GroupNormalization`` and ``_upsample_bilinear`` used to live in
``neural_networks.unet`` but are used across many model builders and
workflow scripts, so they now live here.
"""

from __future__ import annotations

import tensorflow as tf


class GroupNormalization(tf.keras.layers.Layer):
	"""Group Normalization layer (channels-last)."""

	def __init__(self, groups: int = 8, epsilon: float = 1e-5, **kwargs):
		super().__init__(**kwargs)
		self.groups = int(groups)
		self.epsilon = float(epsilon)

	def build(self, input_shape):
		channels = int(input_shape[-1])
		if channels is None or channels < 1:
			raise ValueError("GroupNormalization requires known channel dimension")
		if self.groups < 1:
			raise ValueError("groups must be >= 1")
		if channels % self.groups != 0:
			raise ValueError("channels must be divisible by groups")
		self.gamma = self.add_weight(
			name="gamma",
			shape=(channels,),
			initializer="ones",
			trainable=True,
		)
		self.beta = self.add_weight(
			name="beta",
			shape=(channels,),
			initializer="zeros",
			trainable=True,
		)
		super().build(input_shape)

	def call(self, inputs):
		input_shape = tf.shape(inputs)
		batch, height, width, channels = (
			input_shape[0],
			input_shape[1],
			input_shape[2],
			input_shape[3],
		)
		groups = self.groups
		channels_per_group = channels // groups
		x = tf.reshape(inputs, [batch, height, width, groups, channels_per_group])
		mean, var = tf.nn.moments(x, axes=[1, 2, 4], keepdims=True)
		x = (x - mean) / tf.sqrt(var + self.epsilon)
		x = tf.reshape(x, [batch, height, width, channels])
		return x * self.gamma + self.beta


def _upsample_bilinear(x):
	"""Bilinear 2× upsampling (used as a Lambda layer in the decoder)."""
	spatial = tf.shape(x)[1:3]
	new_size = spatial * 2
	return tf.image.resize(x, new_size, method="bilinear")
