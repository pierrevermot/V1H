"""Centralised GPU / NumPy dispatch helpers.

These tiny functions used to be duplicated across instruments, phases, psfs,
and skies.  Importing them from here avoids the copies.
"""

from __future__ import annotations

import numpy as np


def is_cupy_array(array) -> bool:
	"""Return True if *array* lives on a CUDA device."""
	return hasattr(array, "__cuda_array_interface__")


def get_xp(use_cupy: bool):
	"""Return ``cupy`` when *use_cupy* is true, else ``numpy``."""
	if use_cupy:
		import cupy as cp

		return cp
	return np


def get_xp_from_array(array):
	"""Return ``cupy`` if *array* lives on GPU, else ``numpy``."""
	if is_cupy_array(array):
		import cupy

		return cupy
	return np


def get_ndimage(use_cupy: bool):
	"""Return ``cupyx.scipy.ndimage`` or ``scipy.ndimage``."""
	if use_cupy:
		from cupyx.scipy import ndimage as cndimage

		return cndimage
	from scipy import ndimage

	return ndimage


def to_numpy(array):
	"""Move *array* to host memory if needed."""
	if is_cupy_array(array):
		return array.get()
	return np.asarray(array)
