
"""Plotting utilities for training outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _load_history(metrics_dir: Path, prefix: str) -> dict[str, np.ndarray]:
	data = {}
	for path in metrics_dir.glob(f"{prefix}_*.npy"):
		key = path.stem.replace(f"{prefix}_", "")
		data[key] = np.load(path, allow_pickle=True)
	return data


def plot_training_outputs(metrics_dir: str | Path, plots_dir: str | Path) -> None:
	"""Generate training figures from saved .npy metrics.

	Parameters
	----------
	metrics_dir : str or Path
		Directory containing saved .npy metrics.
	plots_dir : str or Path
		Output directory for plots.
	"""
	metrics_dir = Path(metrics_dir)
	plots_dir = Path(plots_dir)
	plots_dir.mkdir(parents=True, exist_ok=True)

	history = _load_history(metrics_dir, "history")
	subloss = _load_history(metrics_dir, "subloss")
	batch_history = _load_history(metrics_dir, "history_batch")

	if "loss" in history:
		plt.figure(figsize=(6, 4))
		loss_vals = history["loss"]
		plt.plot(loss_vals, label="loss")
		if "val_loss" in history:
			plt.plot(history["val_loss"], label="val_loss")
		plt.xlabel("Epoch")
		plt.ylabel("Loss")
		plt.title("Training Loss")
		if len(loss_vals) > 1000:
			plt.ylim(top=loss_vals[1000])
		plt.legend()
		plt.tight_layout()
		plt.savefig(plots_dir / "loss_curve.png", dpi=150)
		plt.close()

	if "loss" in batch_history:
		plt.figure(figsize=(6, 4))
		batch_loss = batch_history["loss"]
		plt.plot(batch_loss, label="loss")
		plt.xlabel("Batch")
		plt.ylabel("Loss")
		plt.title("Training Loss (batch)")
		if len(batch_loss) > 1000:
			plt.ylim(top=batch_loss[1000])
		plt.legend()
		plt.tight_layout()
		plt.savefig(plots_dir / "loss_curve_batch.png", dpi=150)
		plt.close()

	if subloss:
		plt.figure(figsize=(7, 5))
		for key, values in subloss.items():
			plt.plot(values, label=key)
		plt.xlabel("Epoch")
		plt.ylabel("Value")
		plt.title("Sub-losses")
		values_ref = next(iter(subloss.values()))
		if len(values_ref) > 1000:
			plt.ylim(top=values_ref[1000])
		plt.legend()
		plt.tight_layout()
		plt.savefig(plots_dir / "sublosses.png", dpi=150)
		plt.close()

	if batch_history:
		plt.figure(figsize=(7, 5))
		for key, values in batch_history.items():
			if key == "loss":
				continue
			plt.plot(values, label=key)
		plt.xlabel("Batch")
		plt.ylabel("Value")
		plt.title("Sub-losses (batch)")
		values_ref = next(iter(batch_history.values()))
		if len(values_ref) > 1000:
			plt.ylim(top=values_ref[1000])
		plt.legend()
		plt.tight_layout()
		plt.savefig(plots_dir / "sublosses_batch.png", dpi=150)
		plt.close()

	lr_path = metrics_dir / "lr_history.npy"
	if lr_path.exists():
		lr = np.load(lr_path, allow_pickle=True)
		plt.figure(figsize=(6, 4))
		plt.plot(lr)
		plt.xlabel("Epoch")
		plt.ylabel("Learning Rate")
		plt.title("Learning Rate Schedule")
		plt.tight_layout()
		plt.savefig(plots_dir / "learning_rate.png", dpi=150)
		plt.close()

	best_value_path = metrics_dir / "best_value.npy"
	best_epoch_path = metrics_dir / "best_epoch.npy"
	if best_value_path.exists() and best_epoch_path.exists():
		best_value = np.load(best_value_path, allow_pickle=True)
		best_epoch = np.load(best_epoch_path, allow_pickle=True)
		plt.figure(figsize=(4, 2))
		plt.axis("off")
		plt.text(0.0, 0.6, f"Best epoch: {best_epoch}")
		plt.text(0.0, 0.2, f"Best value: {best_value}")
		plt.tight_layout()
		plt.savefig(plots_dir / "best_summary.png", dpi=150)
		plt.close()
