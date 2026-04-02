"""Dynamically load an experiment configuration file."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


_REQUIRED_SECTIONS = (
	"INSTRUMENT_CONFIG",
	"PHASE_CONFIG",
	"SKY_CONFIG",
	"RANDOM_NOISE_CONFIG",
	"DATASET_GEN_CONFIG",
	"DATASET_LOAD_CONFIG",
	"LOSS_CONFIG",
	"IMAGE_HEAD_CONFIG",
	"NOISE_HEAD_CONFIG",
	"PSF_HEAD_CONFIG",
	"PSF_UNC_CONFIG",
	"JOINT_PINN_CONFIG",
	"SLURM_CONFIG",
	"OUTPUT_BASE_DIR",
)

_HEAD_TARGET_TO_CONFIG = {
	"im": "IMAGE_HEAD_CONFIG",
	"psf": "PSF_HEAD_CONFIG",
	"noise": "NOISE_HEAD_CONFIG",
}

_ARCH_KEYS = frozenset({
	"layers_per_block", "base_filters", "normalization", "group_norm_groups",
	"weight_decay", "inner_activation_function", "output_activation_function",
	"latent_dim", "normalize_output_sum", "normalize_with_first",
	"normalize_first_only", "normalize_by_mean",
	"n_layers", "n_per_layer",
	"n_convs_per_layer", "n_filters", "latent_depth",
})

_TRAINING_KEYS = frozenset({
	"n_epochs", "n_steps_per_epoch", "lr_0", "lr_decay",
})


def load_experiment_config(path: str | Path) -> ModuleType:
	"""Import a Python experiment config file and return it as a module.

	Parameters
	----------
	path : str or Path
		Absolute or relative path to the experiment ``.py`` config file.

	Returns
	-------
	ModuleType
		The imported module.  Access any config section as an attribute, e.g.
		``cfg.INSTRUMENT_CONFIG``.

	Raises
	------
	FileNotFoundError
		If the path does not point to an existing file.
	ValueError
		If a required config section is missing from the file.
	"""
	path = Path(path).resolve()
	if not path.is_file():
		raise FileNotFoundError(f"Config file not found: {path}")

	spec = importlib.util.spec_from_file_location("_experiment_config", str(path))
	module = importlib.util.module_from_spec(spec)
	sys.modules["_experiment_config"] = module
	spec.loader.exec_module(module)

	missing = [name for name in _REQUIRED_SECTIONS if not hasattr(module, name)]
	if missing:
		raise ValueError(
			f"Config file {path} is missing required sections: {', '.join(missing)}"
		)
	return module


def get_head_config(cfg: ModuleType, head_target: str) -> dict:
	"""Return the head configuration dict for a given target (im/psf/noise)."""
	attr = _HEAD_TARGET_TO_CONFIG.get(head_target)
	if attr is None:
		raise ValueError(f"Unknown head target {head_target!r}; expected one of {sorted(_HEAD_TARGET_TO_CONFIG)}")
	return dict(getattr(cfg, attr))


def extract_arch_config(head_config: dict) -> dict:
	"""Extract model architecture kwargs from a head config dict."""
	return {k: v for k, v in head_config.items() if k in _ARCH_KEYS}


def extract_training_config(head_config: dict) -> dict:
	"""Extract training hyperparameters from a head config dict."""
	cfg = {k: v for k, v in head_config.items() if k in _TRAINING_KEYS}
	cfg.setdefault("n_epochs", 100)
	cfg.setdefault("lr_0", 5e-4)
	cfg.setdefault("lr_decay", 10.0)
	cfg.setdefault("verbose", True)
	return cfg
