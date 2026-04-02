"""LWE phase screen utilities."""

from __future__ import annotations

import numpy as np

from instruments.ao_instrument import compute_lwe_modes, detect_islands


def _get_lwe_modes(ao_instru, modes=None, rotated_index: int = 0):
	xp = getattr(ao_instru, "xp", np)
	if modes is not None:
		modes = xp.asarray(modes)
	else:
		modes = getattr(ao_instru, "lwe_modes_rotated", None)
		if modes is not None and modes.ndim == 5 and modes.shape[2] >= 1:
			modes = modes[:, :, rotated_index]
		else:
			if hasattr(ao_instru, "rotated_pupils") and ao_instru.rotated_pupils is not None:
				pupil_array = ao_instru.rotated_pupils[rotated_index]
			else:
				pupil_array = ao_instru.pupil_array
			islands = detect_islands(pupil_array, use_cupy=getattr(ao_instru, "use_cupy", False))
			modes = compute_lwe_modes(islands)
	return xp.asarray(modes)


def lwe_phase_screen(
	ao_instru,
	weights,
	*,
	rotated_index: int = 0,
	modes=None,
):
	"""
	Generate a phase screen from LWE mode weights.

	Parameters
	----------
	ao_instru : AO_instrument
		AO_instrument object providing the ao_instru mask and backend (xp/cp).
	weights : array-like
		Array of shape (3, n_islands) with [piston, tip, tilt] weights per island.
	modes : array-like or None, optional
		Optional precomputed LWE modes of shape (3, n_islands, n_pix, n_pix).
		If None, uses ``ao_instru.lwe_modes_rotated[..., 0, :, :]`` when available,
		otherwise computes modes from the ao_instru islands.

	Returns
	-------
	xp.ndarray
		Phase screen array with shape (n_pix, n_pix).
	"""
	xp = getattr(ao_instru, "xp", np)
	weights = xp.asarray(weights)

	if weights.ndim != 2 or weights.shape[0] != 3:
		raise ValueError("weights must have shape (3, n_islands)")

	modes = _get_lwe_modes(ao_instru, modes=modes, rotated_index=rotated_index)
	if modes.ndim != 4 or modes.shape[0] != 3:
		raise ValueError("modes must have shape (3, n_islands, n_pix, n_pix)")

	if weights.shape[1] > modes.shape[1]:
		raise ValueError(
			f"Number of islands in weights ({weights.shape[1]}) exceeds modes ({modes.shape[1]})."
		)

	phase = xp.tensordot(weights, modes[:, : weights.shape[1]], axes=([0, 1], [0, 1]))
	return phase


def lwe_phase_screens_vectorized(
	ao_instru,
	weights_stack,
	*,
	rotated_index: int = 0,
	modes=None,
):
	"""
	Generate multiple phase screens from stacked LWE weights.

	Parameters
	----------
	ao_instru : AO_instrument
		AO_instrument object providing the ao_instru mask and backend (xp/cp).
	weights_stack : array-like
		Array of shape (n_screens, 3, n_islands) with LWE weights.
	modes : array-like or None, optional
		Optional precomputed LWE modes of shape (3, n_islands, n_pix, n_pix).
		If None, uses ``ao_instru.lwe_modes_rotated[..., 0, :, :]`` when available,
		otherwise computes modes from the ao_instru islands.

	Returns
	-------
	xp.ndarray
		Phase screen stack with shape (n_screens, n_pix, n_pix).
	"""
	xp = getattr(ao_instru, "xp", np)
	weights_stack = xp.asarray(weights_stack)

	if weights_stack.ndim != 3 or weights_stack.shape[1] != 3:
		raise ValueError("weights_stack must have shape (n_screens, 3, n_islands)")

	modes = _get_lwe_modes(ao_instru, modes=modes, rotated_index=rotated_index)
	if modes.ndim != 4 or modes.shape[0] != 3:
		raise ValueError("modes must have shape (3, n_islands, n_pix, n_pix)")

	n_islands = weights_stack.shape[2]
	if n_islands > modes.shape[1]:
		raise ValueError(
			f"Number of islands in weights ({n_islands}) exceeds modes ({modes.shape[1]})."
		)

	phase_stack = xp.tensordot(weights_stack, modes[:, :n_islands], axes=([1, 2], [0, 1]))
	return phase_stack