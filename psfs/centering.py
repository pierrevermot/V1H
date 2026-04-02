"""Utilities to center PSFs in-array.

The convention used here is to shift the maximum of each PSF to the central
pixel ``(n_pix // 2, n_pix // 2)`` using ``roll`` on the last two axes.
"""

from __future__ import annotations

import numpy as np

from utils.array_backend import get_xp_from_array as _get_xp


def _as_python_int(value) -> int:
	if hasattr(value, "item"):
		return int(value.item())
	return int(value)


def _center_single_psf_peak(psf_2d):
	xp = _get_xp(psf_2d)
	psf_2d = xp.asarray(psf_2d)
	if psf_2d.ndim != 2:
		raise ValueError(f"Expected a 2D PSF array, got shape {psf_2d.shape}")

	n_y, n_x = (int(psf_2d.shape[-2]), int(psf_2d.shape[-1]))
	target_y = n_y // 2
	target_x = n_x // 2
	flat_index = _as_python_int(xp.argmax(psf_2d))
	peak_y, peak_x = divmod(flat_index, n_x)
	shift_y = target_y - peak_y
	shift_x = target_x - peak_x
	centered = xp.roll(psf_2d, shift=shift_y, axis=-2)
	centered = xp.roll(centered, shift=shift_x, axis=-1)
	return centered


def center_psf_peak(psf):
	"""Center a 2D PSF or each frame of a 3D PSF cube by its maximum.

	Parameters
	----------
	psf : array-like
		Either a 2D PSF ``(n_pix, n_pix)`` or a 3D cube
		``(n_frames, n_pix, n_pix)``.

	Returns
	-------
	array-like
		Centered PSF with the same rank/backend as the input.
	"""
	xp = _get_xp(psf)
	psf = xp.asarray(psf)
	if psf.ndim == 2:
		return _center_single_psf_peak(psf)
	if psf.ndim == 3:
		return xp.stack([_center_single_psf_peak(frame) for frame in psf], axis=0)
	raise ValueError(f"Expected a 2D or 3D PSF array, got shape {psf.shape}")
