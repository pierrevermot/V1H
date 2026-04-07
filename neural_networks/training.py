
"""Training utilities for U-Net."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np

import tensorflow as tf


class _LossPrinter(tf.keras.callbacks.Callback):
	def __init__(self, metric_names: list[str], verbose: bool = True):
		super().__init__()
		self.metric_names = metric_names
		self.verbose = verbose

	def on_epoch_end(self, epoch, logs=None):
		if not self.verbose or not logs:
			return
		parts = []
		total = logs.get("loss")
		val_total = logs.get("val_loss")
		if total is not None:
			parts.append(f"loss={total:.6f}")
		if val_total is not None:
			parts.append(f"val_loss={val_total:.6f}")
		for name in self.metric_names:
			if name in logs:
				parts.append(f"{name}={logs[name]:.6f}")
			val_name = f"val_{name}"
			if val_name in logs:
				parts.append(f"{val_name}={logs[val_name]:.6f}")
		print(f"Epoch {epoch + 1}: " + ", ".join(parts))


class _LrTracker(tf.keras.callbacks.Callback):
	def __init__(self):
		super().__init__()
		self.lr_history: list[float] = []

	def on_epoch_end(self, epoch, logs=None):
		lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
		self.lr_history.append(lr)


class _SaveBestExamples(tf.keras.callbacks.Callback):
	def __init__(
		self,
		val_dataset: tf.data.Dataset,
		save_dir: Path,
		n_examples: int = 100,
		target_layout: str = "generic",
	):
		super().__init__()
		self.val_dataset = val_dataset
		self.save_dir = save_dir
		self.n_examples = int(n_examples)
		self.target_layout = str(target_layout).strip().lower()
		self.best = None

	def _split_targets(self, y_true: np.ndarray) -> dict[str, np.ndarray]:
		if self.target_layout in {"generic", "target"}:
			return {"target": y_true}
		if self.target_layout in {"im", "image"}:
			return {"image": y_true}
		if self.target_layout == "psf":
			return {"psf": y_true}
		if self.target_layout == "noise":
			return {"noise": y_true}
		if self.target_layout == "joint":
			if y_true.shape[-1] < 3 or (y_true.shape[-1] - 1) % 2 != 0:
				raise ValueError(
					f"Expected joint target layout [image, psf..., noise...], got shape {y_true.shape}"
				)
			n_frames = (y_true.shape[-1] - 1) // 2
			return {
				"image": y_true[..., :1],
				"psf": y_true[..., 1 : 1 + n_frames],
				"noise": y_true[..., 1 + n_frames : 1 + 2 * n_frames],
			}
		if self.target_layout == "stage2_psf_uncertainty":
			if y_true.shape[-1] % 2 != 0:
				raise ValueError(
					f"Expected stage-2 target layout [true_psf, fixed_mean_psf], got shape {y_true.shape}"
				)
			n_frames = y_true.shape[-1] // 2
			return {
				"psf": y_true[..., :n_frames],
				"fixed_mean_psf": y_true[..., n_frames:],
			}
		raise ValueError(f"Unknown best-example target layout: {self.target_layout!r}")

	def on_epoch_end(self, epoch, logs=None):
		if not logs or "val_loss" not in logs:
			return
		val_loss = logs.get("val_loss")
		if val_loss is None:
			return
		if self.best is None or val_loss < self.best:
			self.best = val_loss
			self._save_examples()

	def _save_examples(self):
		obs_list = []
		target_list = []
		named_target_lists: dict[str, list[np.ndarray]] = {}
		pred_list = []
		count = 0

		for obs_batch, y_true_batch in self.val_dataset:
			pred_batch = self.model.predict(obs_batch, verbose=0)
			batch_size = int(obs_batch.shape[0])
			for i in range(batch_size):
				if count >= self.n_examples:
					break
				obs = obs_batch[i].numpy()
				y_true = y_true_batch[i].numpy()
				pred = pred_batch[i]
				named_targets = self._split_targets(y_true)

				obs_list.append(obs)
				target_list.append(y_true)
				for name, value in named_targets.items():
					named_target_lists.setdefault(name, []).append(value)
				pred_list.append(pred)
				count += 1
			if count >= self.n_examples:
				break

		self.save_dir.mkdir(parents=True, exist_ok=True)
		np.save(self.save_dir / "examples_obs.npy", np.asarray(obs_list))
		np.save(self.save_dir / "examples_target.npy", np.asarray(target_list))
		for name, values in named_target_lists.items():
			if name == "target":
				continue
			np.save(self.save_dir / f"examples_{name}.npy", np.asarray(values))
		np.save(self.save_dir / "examples_pred.npy", np.asarray(pred_list))


class _TerminateOnNaNWithBatch(tf.keras.callbacks.Callback):
	def on_train_batch_end(self, batch, logs=None):
		if not logs:
			return
		for key, value in logs.items():
			if value is None:
				continue
			if not np.isfinite(value):
				print(f"NaN/Inf detected in {key} at batch {batch}")
				self.model.stop_training = True
				return


class _BatchHistory(tf.keras.callbacks.Callback):
	def __init__(self, metric_names: list[str]):
		super().__init__()
		self.metric_names = metric_names
		self.history: dict[str, list[float]] = {name: [] for name in ("loss", *metric_names)}

	def on_train_batch_end(self, batch, logs=None):
		if not logs:
			return
		self.history["loss"].append(float(logs.get("loss", float("nan"))))
		for name in self.metric_names:
			self.history[name].append(float(logs.get(name, float("nan"))))


def _make_component_metric(name: str, subloss_fn):
	def metric(y_true, y_pred):
		return tf.reduce_mean(tf.convert_to_tensor(subloss_fn(y_true, y_pred)[name]))

	metric.__name__ = name
	return metric


def _scalarize_metric_value(value) -> tf.Tensor:
	return tf.reduce_mean(tf.convert_to_tensor(value))


def _collect_batch_median_metrics(
	model: tf.keras.Model,
	dataset: tf.data.Dataset,
	*,
	loss_fn,
	metric_fns: dict[str, callable],
) -> dict[str, float]:
	metric_history: dict[str, list[float]] = {"val_loss": []}
	for name in metric_fns:
		metric_history[f"val_{name}"] = []

	for obs_batch, y_true_batch in dataset:
		y_pred_batch = model(obs_batch, training=False)
		loss_value = _scalarize_metric_value(loss_fn(y_true_batch, y_pred_batch))
		if model.losses:
			loss_value = loss_value + tf.add_n(model.losses)
		metric_history["val_loss"].append(float(loss_value.numpy()))

		for name, metric_fn in metric_fns.items():
			value = _scalarize_metric_value(metric_fn(y_true_batch, y_pred_batch))
			metric_history[f"val_{name}"].append(float(value.numpy()))

	return {
		name: float(np.median(values))
		for name, values in metric_history.items()
		if values
	}


class _MedianValidationMetrics(tf.keras.callbacks.Callback):
	def __init__(
		self,
		val_dataset: tf.data.Dataset,
		*,
		loss_fn,
		metric_fns: dict[str, callable],
	):
		super().__init__()
		self.val_dataset = val_dataset
		self.loss_fn = loss_fn
		self.metric_fns = metric_fns
		self.history: dict[str, list[float]] = {"val_loss": []}
		for name in metric_fns:
			self.history[f"val_{name}"] = []

	def on_epoch_end(self, epoch, logs=None):
		if logs is None:
			return
		median_metrics = _collect_batch_median_metrics(
			self.model,
			self.val_dataset,
			loss_fn=self.loss_fn,
			metric_fns=self.metric_fns,
		)
		for name, value in median_metrics.items():
			logs[name] = value
			self.history.setdefault(name, []).append(value)


def train_unet(
	model: tf.keras.Model,
	loss,
	dataset: tf.data.Dataset,
	*,
	val_dataset: tf.data.Dataset | None = None,
	n_epochs: int = 100,
	lr_0: float = 5e-4,
	lr_decay: float = 33.0,
	verbose: bool = True,
	n_steps_per_epoch: int | None = None,
	use_pinn: bool = True,
	checkpoint_path: str | Path = "checkpoints/unet_best.keras",
	subloss_fn=None,
	extra_callbacks: list[tf.keras.callbacks.Callback] | None = None,
	best_examples_target_layout: str = "generic",
) -> dict[str, object]:
	"""Train a U-Net model.

	Parameters
	----------
	model : tf.keras.Model
		Non-compiled model to train (will be compiled inside).
	loss : callable
		Loss function.
	dataset : tf.data.Dataset
		Training dataset.
	val_dataset : tf.data.Dataset, optional
		Validation dataset.
	n_epochs : int, optional
		Number of epochs. Default 100.
	lr_0 : float, optional
		Initial learning rate. Default 5e-4.
	lr_decay : float, optional
		Decay factor for lr schedule: lr = lr_0*10**(-epoch/lr_decay). Default 33.
	verbose : bool, optional
		Whether to print losses each epoch. Default True.
	n_steps_per_epoch : int or None, optional
		Limit number of batches per epoch. Default None.
	use_pinn : bool, optional
		Whether to include the PINN/Charbonnier metric. Default True.
	checkpoint_path : str or Path, optional
		Where to save the best model checkpoint.
	subloss_fn : callable or None, optional
		Function returning sub-loss components dict. If None, uses loss.components when available.
	extra_callbacks : list[tf.keras.callbacks.Callback] or None, optional
		Additional callbacks appended to the default callback list.
	"""
	if subloss_fn is None and hasattr(loss, "components"):
		subloss_fn = loss.components

	metrics = []
	metric_names: list[str] = []
	if subloss_fn is not None:
		metric_list = None
		if hasattr(loss, "component_names"):
			metric_list = list(loss.component_names)
		if metric_list is None:
			metric_list = [
				"nll_im",
				"nll_psf",
				"nll_noise",
				"log_sigma2_im",
				"log_sigma2_psf",
				"log_sigma2_noise",
			]
		if use_pinn and "pinn_charb" not in metric_list:
			metric_list.append("pinn_charb")
		if not use_pinn and "pinn_charb" in metric_list:
			metric_list = [name for name in metric_list if name != "pinn_charb"]
		for name in metric_list:
			metrics.append(_make_component_metric(name, subloss_fn))
			metric_names.append(name)

	optimizer = tf.keras.optimizers.Adam(learning_rate=lr_0, clipnorm=1.0)
	model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
	model.summary()
	metric_fns = {name: metric for name, metric in zip(metric_names, metrics)}

	def _lr_schedule(epoch, _):
		return lr_0 * 10 ** (-(epoch) / lr_decay)

	callbacks: list[tf.keras.callbacks.Callback] = [
		tf.keras.callbacks.LearningRateScheduler(_lr_schedule, verbose=0)
	]
	callbacks.append(tf.keras.callbacks.TerminateOnNaN())
	callbacks.append(_TerminateOnNaNWithBatch())

	median_validation = None
	if val_dataset is not None:
		median_validation = _MedianValidationMetrics(
			val_dataset=val_dataset,
			loss_fn=loss,
			metric_fns=metric_fns,
		)
		callbacks.append(median_validation)

	monitor = "val_loss" if val_dataset is not None else "loss"
	callbacks.append(
		tf.keras.callbacks.ModelCheckpoint(
			filepath=str(checkpoint_path),
			monitor=monitor,
			save_best_only=True,
		)
	)
	callbacks.append(_LossPrinter(metric_names=metric_names, verbose=verbose))
	if val_dataset is not None:
		save_dir = Path(checkpoint_path).parent / "best_examples"
		callbacks.append(
			_SaveBestExamples(
				val_dataset=val_dataset,
				save_dir=save_dir,
				target_layout=best_examples_target_layout,
			)
		)

	batch_history = _BatchHistory(metric_names=metric_names)
	callbacks.append(batch_history)

	lr_tracker = _LrTracker()
	callbacks.append(lr_tracker)
	if extra_callbacks:
		callbacks.extend(extra_callbacks)

	start_time = time.perf_counter()
	history = model.fit(
		dataset,
		validation_data=None,
		epochs=n_epochs,
		steps_per_epoch=n_steps_per_epoch,
		verbose=1 if verbose else 0,
		callbacks=callbacks,
	)
	if median_validation is not None:
		for name, values in median_validation.history.items():
			history.history[name] = list(values)
	end_time = time.perf_counter()

	best_key = "val_loss" if val_dataset is not None else "loss"
	best_values = history.history.get(best_key, [])
	best_value = min(best_values) if best_values else None
	best_epoch = int(best_values.index(best_value)) + 1 if best_values else None

	subloss_history = {
		name: history.history.get(name, [])
		for name in metric_names
	}
	if val_dataset is not None:
		for name in metric_names:
			subloss_history[f"val_{name}"] = history.history.get(f"val_{name}", [])

	return {
		"history": history,
		"model": model,
		"checkpoint_path": str(checkpoint_path),
		"best_metric": best_key,
		"best_value": best_value,
		"best_epoch": best_epoch,
		"lr_history": lr_tracker.lr_history,
		"duration_s": end_time - start_time,
		"subloss_history": subloss_history,
		"batch_history": batch_history.history,
	}
