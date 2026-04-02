"""Zernike phase screen utilities."""

from __future__ import annotations

import numpy as np

from instruments.ao_instrument import compute_zernike_modes


def zernike_phase_screen(
	ao_instru,
	coeffs,
	*,
	rotated_index: int = 0,
	n_rad: int = 6,
	n_min: int = 1,
	modes=None,
):
	"""
	Generate a phase screen from Zernike coefficients.

	Parameters
	----------
	ao_instru : AO_instrument
		AO_instrument object providing the ao_instru mask and backend (xp/cp).
	coeffs : array-like
		Zernike coefficients (one per mode) ordered as in
		``compute_zernike_modes`` (increasing n, m=-n..n step 2).
	n_rad : int, optional
		Maximum radial order to compute if ``modes`` not provided.
	n_min : int, optional
		Minimum radial order to compute if ``modes`` not provided.
	modes : array-like or None, optional
		Optional precomputed Zernike modes of shape (n_modes, n_pix, n_pix).
		If None, uses ``ao_instru.zernike_modes`` when available, otherwise
		computes modes using ``compute_zernike_modes``.

	Returns
	-------
	xp.ndarray
		Phase screen array with shape (n_pix, n_pix).
	"""
	xp = getattr(ao_instru, "xp", np)
	coeffs = xp.asarray(coeffs)

	if modes is None:
		modes = getattr(ao_instru, "zernike_modes", None)
		if modes is None:
			if hasattr(ao_instru, "rotated_pupils") and ao_instru.rotated_pupils is not None:
				class _PupilView:
					def __init__(self, base, pupil_array):
						self.pupil_array = pupil_array
						self.pixel_scale = base.pixel_scale
						self.xp = getattr(base, "xp", np)

				ao_instru_view = _PupilView(ao_instru, ao_instru.rotated_pupils[rotated_index])
				modes = compute_zernike_modes(ao_instru_view, n_rad=n_rad, n_min=n_min)
			else:
				modes = compute_zernike_modes(ao_instru, n_rad=n_rad, n_min=n_min)

	modes = xp.asarray(modes)
	if modes.ndim != 3:
		raise ValueError("modes must have shape (n_modes, n_pix, n_pix)")

	n_coeffs = int(coeffs.shape[0])
	if n_coeffs > modes.shape[0]:
		raise ValueError(
			f"Number of coeffs ({n_coeffs}) exceeds available modes ({modes.shape[0]})."
		)

	if n_coeffs == 0:
		return xp.zeros(modes.shape[1:], dtype=xp.float64)

	phase = xp.tensordot(coeffs, modes[:n_coeffs], axes=(0, 0))
	return phase


def zernike_phase_screens_vectorized(
	ao_instru,
	coeffs_stack,
	*,
	rotated_index: int = 0,
	n_rad: int = 6,
	n_min: int = 1,
	modes=None,
):
	"""
	Generate multiple phase screens from stacked Zernike coefficients.

	Parameters
	----------
	ao_instru : AO_instrument
		AO_instrument object providing the ao_instru mask and backend (xp/cp).
	coeffs_stack : array-like
		Array of shape (n_screens, n_modes) with Zernike coefficients
		ordered as in ``compute_zernike_modes``.
	n_rad : int, optional
		Maximum radial order to compute if ``modes`` not provided.
	n_min : int, optional
		Minimum radial order to compute if ``modes`` not provided.
	modes : array-like or None, optional
		Optional precomputed Zernike modes of shape (n_modes, n_pix, n_pix).
		If None, uses ``ao_instru.zernike_modes`` when available, otherwise
		computes modes using ``compute_zernike_modes``.

	Returns
	-------
	xp.ndarray
		Phase screen stack with shape (n_screens, n_pix, n_pix).
	"""
	xp = getattr(ao_instru, "xp", np)
	coeffs_stack = xp.asarray(coeffs_stack)

	if coeffs_stack.ndim != 2:
		raise ValueError("coeffs_stack must have shape (n_screens, n_modes)")

	if modes is None:
		modes = getattr(ao_instru, "zernike_modes", None)
		if modes is None:
			if hasattr(ao_instru, "rotated_pupils") and ao_instru.rotated_pupils is not None:
				class _PupilView:
					def __init__(self, base, pupil_array):
						self.pupil_array = pupil_array
						self.pixel_scale = base.pixel_scale
						self.xp = getattr(base, "xp", np)

				ao_instru_view = _PupilView(ao_instru, ao_instru.rotated_pupils[rotated_index])
				modes = compute_zernike_modes(ao_instru_view, n_rad=n_rad, n_min=n_min)
			else:
				modes = compute_zernike_modes(ao_instru, n_rad=n_rad, n_min=n_min)

	modes = xp.asarray(modes)
	if modes.ndim != 3:
		raise ValueError("modes must have shape (n_modes, n_pix, n_pix)")

	n_screens, n_coeffs = int(coeffs_stack.shape[0]), int(coeffs_stack.shape[1])
	if n_coeffs > modes.shape[0]:
		raise ValueError(
			f"Number of coeffs ({n_coeffs}) exceeds available modes ({modes.shape[0]})."
		)

	if n_screens == 0 or n_coeffs == 0:
		return xp.zeros((n_screens, modes.shape[1], modes.shape[2]), dtype=xp.float64)

	phase_stack = xp.tensordot(coeffs_stack, modes[:n_coeffs], axes=(1, 0))
	return phase_stack

