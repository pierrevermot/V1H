from __future__ import annotations
import tensorflow as tf


def _infer_shapes_from_batch(batch) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
	obs, y_true = batch
	input_shape = tuple(obs.shape[1:])
	output_shape = tuple(y_true.shape[1:])
	return input_shape, output_shape


def _apply_output_activation(tensor: tf.Tensor, activation_name: str, *, layer_name: str) -> tf.Tensor:
	activation_name = str(activation_name).strip().lower()
	if activation_name in {"", "linear"}:
		return tensor
	tf.keras.activations.get(activation_name)
	return tf.keras.layers.Activation(activation_name, name=layer_name)(tensor)


def _wrap_model_output_activation(
	model: tf.keras.Model,
	*,
	activation_name: str,
	output_channels: int,
	nll: bool,
) -> tf.keras.Model:
	activation_name = str(activation_name).strip().lower()
	if activation_name in {"", "linear"}:
		return model

	outputs = model.output
	if nll:
		pred_main = tf.keras.layers.Lambda(
			lambda x: x[..., :output_channels],
			name="pred_main_slice",
		)(outputs)
		pred_unc = tf.keras.layers.Lambda(
			lambda x: x[..., output_channels:],
			name="pred_unc_slice",
		)(outputs)
		pred_main = _apply_output_activation(
			pred_main,
			activation_name,
			layer_name=f"pred_main_{activation_name}",
		)
		outputs = tf.keras.layers.Concatenate(axis=-1, name="pred_concat_with_unc")([pred_main, pred_unc])
	else:
		outputs = _apply_output_activation(
			outputs,
			activation_name,
			layer_name=f"pred_{activation_name}",
		)

	return tf.keras.Model(inputs=model.input, outputs=outputs, name=model.name)


def _resolve_model_input_shape(model: tf.keras.Model) -> tuple[int | None, int, int, int]:
	input_shape = getattr(model, "input_shape", None)
	if isinstance(input_shape, list):
		input_shape = input_shape[0] if input_shape else None
	if input_shape is None and getattr(model, "inputs", None):
		input_shape = tuple(model.inputs[0].shape)
	if input_shape is None:
		raise ValueError("Could not determine model input shape")
	shape = tf.TensorShape(input_shape).as_list()
	if len(shape) != 4:
		raise ValueError(f"Expected model input shape of rank 4, got {input_shape}")
	if shape[1] is None or shape[2] is None or shape[3] is None:
		raise ValueError(f"Model input spatial/frame dimensions must be known, got {input_shape}")
	return (None if shape[0] is None else int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3]))


def _resolve_model_input_shape_3d(model: tf.keras.Model) -> tuple[int, int, int]:
	"""Return (H, W, F) dropping the batch dimension."""
	shape = _resolve_model_input_shape(model)
	return (shape[1], shape[2], shape[3])


def _extract_prediction_mean(pred: tf.Tensor, output_channels: tf.Tensor) -> tf.Tensor:
	channels = tf.shape(pred)[-1]
	ratio = tf.math.floordiv(channels, output_channels)
	with tf.control_dependencies(
		[
			tf.debugging.assert_equal(
				channels,
				output_channels * ratio,
				message="Source PSF model output channels must be a multiple of target channels",
			),
			tf.debugging.assert_greater_equal(ratio, 1, message="Source PSF model must output at least one PSF cube"),
			tf.debugging.assert_less_equal(ratio, 2, message="Source PSF model must output either F or 2F channels"),
		]
	):
		return pred[..., :output_channels]


def _split_nll_output(output: tf.Tensor, expected_channels: int | tf.Tensor, *, head_name: str) -> tuple[tf.Tensor, tf.Tensor]:
	actual = output.shape[-1]
	if actual is not None and isinstance(expected_channels, int):
		if int(actual) != 2 * expected_channels:
			raise ValueError(f"{head_name} must output {2 * expected_channels} channels, got {actual}")
		return output[..., :expected_channels], output[..., expected_channels:]
	actual_t = tf.shape(output)[-1]
	expected_t = tf.cast(expected_channels, actual_t.dtype)
	with tf.control_dependencies(
		[tf.debugging.assert_equal(actual_t, 2 * expected_t, message=f"{head_name} output has unexpected channel count")]
	):
		first = output[..., :expected_t]
		second = output[..., expected_t:]
		return first, second


def _extract_mean_output(output: tf.Tensor, expected_channels: int | tf.Tensor, *, head_name: str) -> tf.Tensor:
	actual = output.shape[-1]
	if actual is not None and isinstance(expected_channels, int):
		if int(actual) not in {expected_channels, 2 * expected_channels}:
			raise ValueError(f"{head_name} must output {expected_channels} or {2 * expected_channels} channels, got {actual}")
		return output[..., :expected_channels]
	actual_t = tf.shape(output)[-1]
	expected_t = tf.cast(expected_channels, actual_t.dtype)
	ok = tf.logical_or(tf.equal(actual_t, expected_t), tf.equal(actual_t, 2 * expected_t))
	with tf.control_dependencies(
		[tf.debugging.assert_equal(ok, True, message=f"{head_name} output has unexpected channel count")]
	):
		return output[..., :expected_t]


def _extract_uncertainty_output(output: tf.Tensor, expected_channels: int | tf.Tensor, *, head_name: str) -> tf.Tensor:
	actual = output.shape[-1]
	if actual is not None and isinstance(expected_channels, int):
		if int(actual) == expected_channels:
			return output
		if int(actual) == 2 * expected_channels:
			return output[..., expected_channels:]
		raise ValueError(f"{head_name} must output {expected_channels} or {2 * expected_channels} channels, got {actual}")
	actual_t = tf.shape(output)[-1]
	expected_t = tf.cast(expected_channels, actual_t.dtype)
	ok = tf.logical_or(tf.equal(actual_t, expected_t), tf.equal(actual_t, 2 * expected_t))
	with tf.control_dependencies(
		[tf.debugging.assert_equal(ok, True, message=f"{head_name} output has unexpected channel count")]
	):
		return tf.cond(
			tf.equal(actual_t, expected_t),
			lambda: output,
			lambda: output[..., expected_t:],
		)
