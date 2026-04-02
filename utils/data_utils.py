"""Data preparation and cropping utilities."""
from __future__ import annotations

from typing import Any

import numpy as np


def _prepare_truth_arrays(raw: dict[str, np.ndarray], dataset_cfg: dict[str, Any]) -> dict[str, np.ndarray]:
	image_hh = raw["image_hh"]
	obs_fhh = raw["obs_fhh"]
	psf_fhh = raw["psf_fhh"]
	noise_fhh = raw["noise_fhh"]

	crop = int(dataset_cfg.get("half_n_pix_crop", 0))
	if crop > 0:
		image_hh = image_hh[:, crop:-crop, crop:-crop]
		obs_fhh = obs_fhh[:, :, crop:-crop, crop:-crop]
		psf_fhh = psf_fhh[:, :, crop:-crop, crop:-crop]
		noise_fhh = noise_fhh[:, :, crop:-crop, crop:-crop]

	return {
		"obs_hwf": np.transpose(obs_fhh, (0, 2, 3, 1)).astype(np.float32),
		"image_hw1": image_hh[..., np.newaxis].astype(np.float32),
		"psf_hwf": np.transpose(psf_fhh, (0, 2, 3, 1)).astype(np.float32),
		"noise_hwf": np.transpose(noise_fhh, (0, 2, 3, 1)).astype(np.float32),
	}


def _crop_data_to_model(data_hwf: np.ndarray, model_input_shape: tuple[int, ...], *, keep_all_frames: bool = False) -> np.ndarray:
	expected_h = int(model_input_shape[1])
	expected_w = int(model_input_shape[2])
	expected_f = int(model_input_shape[3])
	h, w, f = data_hwf.shape
	if f < expected_f:
		raise ValueError(f"data.npy has {f} frames but the model expects {expected_f}")
	cube = data_hwf if keep_all_frames else data_hwf[..., :expected_f]
	if h == expected_h and w == expected_w:
		return cube.astype(np.float32)
	if h < expected_h or w < expected_w:
		raise ValueError(f"data.npy spatial shape {(h, w)} is smaller than expected {(expected_h, expected_w)}")
	dy = (h - expected_h) // 2
	dx = (w - expected_w) // 2
	return cube[dy : dy + expected_h, dx : dx + expected_w, :].astype(np.float32)
