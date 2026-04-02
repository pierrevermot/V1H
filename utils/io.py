"""I/O helpers: environment parsing, config loading, file utilities."""
from __future__ import annotations

import json
import runpy
import shutil
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# JSON / config loading
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict[str, Any]:
	if not path.exists():
		return {}
	with path.open("r", encoding="utf-8") as handle:
		return json.load(handle)


def _load_head_config(head_dir: Path) -> dict[str, Any]:
	training_cfg = _load_json(head_dir / "training_config.json")
	if training_cfg:
		return training_cfg

	snapshot_path = head_dir / "config_nn.py"
	if not snapshot_path.exists():
		return {}

	namespace = runpy.run_path(str(snapshot_path))
	return {
		"dataset": dict(namespace.get("DATASET_CONFIG", {})),
		"loss": dict(namespace.get("LOSS_CONFIG", {})),
		"model": dict(namespace.get("MODEL_CONFIG", {})),
		"unet": dict(namespace.get("UNET_CONFIG", {})),
		"gpkh": dict(namespace.get("GPKH_CONFIG", {})),
		"gpkh_convdecoder": dict(namespace.get("GPKH_CONVDECODER_CONFIG", {})),
		"skh": dict(namespace.get("SKH_CONFIG", {})),
		"dense": dict(namespace.get("DENSE_PSF_CONFIG", {})),
	}


def _load_run_config(head_dir: Path) -> dict[str, Any]:
	training_cfg = _load_json(head_dir / "training_config.json")
	if training_cfg:
		return training_cfg

	snapshot_path = head_dir / "config_nn.py"
	if not snapshot_path.exists():
		return {}

	namespace = runpy.run_path(str(snapshot_path))
	return {
		"dataset": dict(namespace.get("DATASET_CONFIG", {})),
		"loss": dict(namespace.get("LOSS_CONFIG", {})),
		"unet": dict(namespace.get("UNET_CONFIG", {})),
		"gpkh": dict(namespace.get("GPKH_CONFIG", {})),
		"gpkh_convdecoder": dict(namespace.get("GPKH_CONVDECODER_CONFIG", {})),
		"skh": dict(namespace.get("SKH_CONFIG", {})),
		"dense": dict(namespace.get("DENSE_PSF_CONFIG", {})),
	}


def _resolve_dataset_root(head_cfg: dict[str, Any], *, fallback_data_dir: str | Path | None = None) -> Path:
	data_dir = str(dict(head_cfg.get("dataset", {})).get("data_dir", "")).strip()
	if not data_dir and fallback_data_dir is not None:
		data_dir = str(fallback_data_dir).strip()
	if not data_dir:
		raise ValueError("Dataset data_dir is not configured")
	return Path(data_dir).expanduser().resolve()


def _load_snapshot_config(snapshot_path: Path) -> dict[str, Any]:
	namespace = runpy.run_path(str(snapshot_path))
	return {
		"dataset": dict(namespace.get("DATASET_CONFIG", {})),
		"loss": dict(namespace.get("LOSS_CONFIG", {})),
		"training": dict(namespace.get("TRAINING_CONFIG", {})),
		"joint_training": dict(namespace.get("JOINT_TRAINING_CONFIG", {})),
		"model": dict(namespace.get("MODEL_CONFIG", {})),
		"gpkh": dict(namespace.get("GPKH_CONFIG", {})),
		"unet": dict(namespace.get("UNET_CONFIG", {})),
	}


def _load_joint_run_config(run_dir: Path) -> dict[str, Any]:
	training_cfg = _load_json(run_dir / "training_config.json")
	if training_cfg:
		return training_cfg

	snapshot_path = run_dir / "config_nn.py"
	if snapshot_path.exists():
		return _load_snapshot_config(snapshot_path)

	return {}


# ---------------------------------------------------------------------------
# File utilities
# ---------------------------------------------------------------------------

def _resolve_first_file(directory: Path, pattern: str) -> Path | None:
	matches = sorted(directory.glob(pattern))
	return matches[0] if matches else None


def _write_fits_image(*, image_hw, path: Path, header: dict[str, object] | None = None) -> None:
	import numpy as np
	from astropy.io import fits

	path.parent.mkdir(parents=True, exist_ok=True)
	hdr = fits.Header()
	if header:
		for key, value in header.items():
			try:
				hdr[str(key)[:8].upper()] = value
			except Exception:
				pass
	fits.PrimaryHDU(data=np.asarray(image_hw, dtype=np.float32), header=hdr).writeto(path, overwrite=True)


def _snapshot_file_if_present(src: Path, dst: Path) -> None:
	if src.exists():
		shutil.copy2(src, dst)


def _clear_cache_prefix(cache_path: str | Path) -> None:
	cache_prefix = Path(cache_path)
	cache_prefix.parent.mkdir(parents=True, exist_ok=True)
	for path in cache_prefix.parent.glob(f"{cache_prefix.name}*"):
		if path.is_file() or path.is_symlink():
			path.unlink(missing_ok=True)
		elif path.is_dir():
			shutil.rmtree(path, ignore_errors=True)
