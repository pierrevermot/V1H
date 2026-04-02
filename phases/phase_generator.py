"""Phase screen generator wrapper."""

from __future__ import annotations

import math
import numpy as np

from phases.lwe import lwe_phase_screens_vectorized
from phases.powerlaw import generate_dual_powerlaw_phase
from phases.zernike import zernike_phase_screens_vectorized


def strehl_to_rms(strehl):
	"""Convert Strehl ratio to RMS (radians) using strehl = exp(-rms^2)."""
	if strehl == 1.0:
		return 0.
	elif strehl < 0.0 or strehl > 1.0:
		raise ValueError("strehl must be in the range [0, 1]")
	return float(math.sqrt(-math.log(float(strehl))))


def generate_phase_screens(
	ao_instru,
	rotated_index: int,
	n_screens: int,
	*,
	exponent_lf: float,
	exponent_hf: float,
	cutoff: float,
	rms_lf: float,
	rms_hf: float,
	component_flags,
	zernike_coeffs,
	lwe_weights,
	final_strehl=None,
):
	"""
	Generate composite phase screens from powerlaw, Zernike, and LWE components.

	Parameters
	----------
	ao_instru : AO_instrument
		AO_instrument object providing backend (xp/cp).
	rotated_index : int
		Rotation index for the ao_instru mask.
	n_screens : int
		Number of phase screens to generate.
	exponent_lf, exponent_hf, cutoff, rms_lf, rms_hf : float
		Scalar parameters for the powerlaw components.
	component_flags : tuple
		(powerlaw_on, zernike_on, lwe_on) boolean flags.
	zernike_coeffs : array-like
		2D array of shape (n_screens, n_zernike_modes).
	lwe_weights : array-like
		3D array of shape (n_screens, 3, n_islands).
	final_strehl : float | None
		Desired final Strehl ratio for the composite phase screens.

	Returns
	-------
	xp.ndarray
		Phase screen stack with shape (n_screens, n_pix, n_pix).
	"""
	xp = getattr(ao_instru, "xp", np)

	if n_screens < 1:
		raise ValueError("n_screens must be >= 1")

	exponent_lf = xp.full(n_screens, float(exponent_lf), dtype=xp.float64)
	exponent_hf = xp.full(n_screens, float(exponent_hf), dtype=xp.float64)
	cutoffs = xp.full(n_screens, float(cutoff), dtype=xp.float64)
	rms_lf = xp.full(n_screens, float(rms_lf), dtype=xp.float64)
	rms_hf = xp.full(n_screens, float(rms_hf), dtype=xp.float64)
	try:
		powerlaw_on, zernike_on, lwe_on = component_flags
	except Exception as exc:
		raise ValueError("component_flags must be a 3-tuple of booleans") from exc

	if not (powerlaw_on or zernike_on or lwe_on):
		raise ValueError("At least one component must be enabled")

	n_pix = int(ao_instru.pupil_array.shape[0])
	phase_powerlaw = xp.zeros((n_screens, n_pix, n_pix), dtype=xp.float64)
	phase_zernike = xp.zeros((n_screens, n_pix, n_pix), dtype=xp.float64)
	phase_lwe = xp.zeros((n_screens, n_pix, n_pix), dtype=xp.float64)

	if powerlaw_on:
		phase_powerlaw = generate_dual_powerlaw_phase(
			ao_instru,
			exponent_lf,
			exponent_hf,
			cutoffs,
			rms_lf,
			rms_hf,
			rotated_index=rotated_index,
		)

	if zernike_on:
		zernike_coeffs = xp.asarray(zernike_coeffs)
		if zernike_coeffs.ndim != 2:
			raise ValueError("zernike_coeffs must have shape (n_screens, n_zernike_modes)")
		if zernike_coeffs.shape[0] != n_screens:
			raise ValueError("zernike_coeffs must match number of screens")
		phase_zernike = zernike_phase_screens_vectorized(
			ao_instru,
			zernike_coeffs,
			rotated_index=rotated_index,
		)

	if lwe_on:
		lwe_weights = xp.asarray(lwe_weights)
		if lwe_weights.ndim != 3 or lwe_weights.shape[1] != 3:
			raise ValueError("lwe_weights must have shape (n_screens, 3, n_islands)")
		if lwe_weights.shape[0] != n_screens:
			raise ValueError("lwe_weights must match number of screens")
		phase_lwe = lwe_phase_screens_vectorized(
			ao_instru,
			lwe_weights,
			rotated_index=rotated_index,
		)

	phase_tot = phase_powerlaw + phase_zernike + phase_lwe

	if final_strehl is not None:
		final_strehl = float(final_strehl)
		if not (0.0 < final_strehl <= 1.0):
			raise ValueError("final_strehl must be in the range (0, 1]")
		target_rms = strehl_to_rms(final_strehl)
		mask = ao_instru.pupil_array
		den = xp.sum(mask)
		if float(den) == 0.0:
			raise ValueError("AO_instrument mask sum is zero")
		num = xp.sum(xp.square(phase_tot) * mask[xp.newaxis, :, :], axis=(1, 2))
		current_rms = xp.sqrt(num / den)
		scale_factors = xp.where(current_rms > 0, target_rms / current_rms, 0.0)
		phase_tot *= scale_factors[:, xp.newaxis, xp.newaxis]
	return phase_tot

