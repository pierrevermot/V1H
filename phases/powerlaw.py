from __future__ import annotations

import numpy as np

from instruments import ao_instrument
from utils.array_backend import is_cupy_array, get_xp


def radial_powerlaw_psd(
	ao_instru,
	exponent: float = 11 / 3,
	norm: float = 1.0,
) -> np.ndarray:
	"""
	Generate a radial power-law PSD on the ao_instru FFT frequency grid.

	Parameters
	----------
	ao_instru : AO_instrument
		AO_instrument object providing FFT frequency grids.
	exponent : float, optional
		Power-law exponent (default 11/3).
	norm : float, optional
		Multiplicative normalization factor (default 1).

	Returns
	-------
	np.ndarray
		2D PSD array with the same shape as the ao_instru FFT grid.
	"""
	xp = getattr(ao_instru, "xp", np)
	if hasattr(ao_instru, "fr_m"):
		k = ao_instru.fr_m
	elif hasattr(ao_instru, "fr_pix"):
		k = ao_instru.fr_pix
	elif hasattr(ao_instru, "fx_m") and hasattr(ao_instru, "fy_m"):
		k = xp.sqrt(ao_instru.fx_m**2 + ao_instru.fy_m**2)
	elif hasattr(ao_instru, "fx_pix") and hasattr(ao_instru, "fy_pix"):
		k = xp.sqrt(ao_instru.fx_pix**2 + ao_instru.fy_pix**2)
	else:
		n_pix = ao_instru.n_pix
		fx = xp.fft.fftfreq(n_pix, d=1.0)
		fy = xp.fft.fftfreq(n_pix, d=1.0)
		fxg, fyg = xp.meshgrid(fx, fy)
		k = xp.sqrt(fxg**2 + fyg**2)
	k_safe = xp.where(k == 0, xp.inf, k)
	psd = norm * (k_safe ** (-exponent))
	psd = xp.where(xp.isfinite(psd), psd, 0.0)

	return psd


def radial_powerlaw_psd_vectorized(
	ao_instru,
	exponents,
	norms=None,
) -> np.ndarray:
	"""
	Generate radial power-law PSDs for multiple exponents.

	Parameters
	----------
	ao_instru : AO_instrument
		AO_instrument object providing FFT frequency grids.
	exponents : xp.ndarray
		1D array of power-law exponents.
	norms : xp.ndarray or None, optional
		1D array of multiplicative normalization factors. If None, uses 1.

	Returns
	-------
	xp.ndarray
		3D PSD array with shape (n_exponents, n_pix, n_pix).
	"""
	xp = getattr(ao_instru, "xp", np)
	exponents = xp.asarray(exponents)
	if norms is None:
		norms = xp.ones_like(exponents, dtype=xp.float64)
	else:
		norms = xp.asarray(norms)

	if hasattr(ao_instru, "fr_m"):
		k = ao_instru.fr_m
	elif hasattr(ao_instru, "fr_pix"):
		k = ao_instru.fr_pix
	elif hasattr(ao_instru, "fx_m") and hasattr(ao_instru, "fy_m"):
		k = xp.sqrt(ao_instru.fx_m**2 + ao_instru.fy_m**2)
	elif hasattr(ao_instru, "fx_pix") and hasattr(ao_instru, "fy_pix"):
		k = xp.sqrt(ao_instru.fx_pix**2 + ao_instru.fy_pix**2)
	else:
		n_pix = ao_instru.n_pix
		fx = xp.fft.fftfreq(n_pix, d=1.0)
		fy = xp.fft.fftfreq(n_pix, d=1.0)
		fxg, fyg = xp.meshgrid(fx, fy)
		k = xp.sqrt(fxg**2 + fyg**2)

	k_safe = xp.where(k == 0, xp.inf, k)

	exp_b = exponents[:, None, None]
	norm_b = norms[:, None, None]

	psd = norm_b * (k_safe[None, :, :] ** (-exp_b))
	psd = xp.where(xp.isfinite(psd), psd, 0.0)

	return psd


def apply_cutoff(
	ao_instru,
	psd,
	low_cutoff: float | None = None,
	high_cutoff: float | None = None,
) -> np.ndarray:
	"""
	Apply hard low/high frequency cutoffs to a PSD.

	Parameters
	----------
	ao_instru : AO_instrument
		AO_instrument object providing pixel_scale and frequency grids.
	psd : np.ndarray
		2D PSD array matching the ao_instru FFT grid shape.
	low_cutoff : float or None, optional
		Low-frequency cutoff length in meters. PSD is set to 0 below
		the corresponding cutoff frequency. If None, no low-frequency cutoff.
	high_cutoff : float or None, optional
		High-frequency cutoff length in meters. PSD is set to 0 above
		the corresponding cutoff frequency. If None, no high-frequency cutoff.

	Returns
	-------
	np.ndarray
		PSD array after applying hard cutoffs.
	"""
	xp = getattr(ao_instru, "xp", np)

	if hasattr(ao_instru, "fr_m"):
		k = ao_instru.fr_m
	else:
		n_pix = ao_instru.n_pix
		d = getattr(ao_instru, "pixel_scale", 1.0)
		fx = xp.fft.fftfreq(n_pix, d=d)
		fy = xp.fft.fftfreq(n_pix, d=d)
		fxg, fyg = xp.meshgrid(fx, fy)
		k = xp.sqrt(fxg**2 + fyg**2)

	out = psd.copy()
	if low_cutoff is not None:
		f0 = 1.0 / low_cutoff
		out = xp.where(k < f0, 0.0, out)
	if high_cutoff is not None:
		f1 = 1.0 / high_cutoff
		out = xp.where(k > f1, 0.0, out)

	return out


def apply_cutoff_vectorized(
	ao_instru,
	psd_stack,
	low_cutoffs=None,
	high_cutoffs=None,
) -> np.ndarray:
	"""
	Apply hard low/high frequency cutoffs to a stack of PSDs.

	Parameters
	----------
	ao_instru : AO_instrument
		AO_instrument object providing pixel_scale and frequency grids.
	psd_stack : xp.ndarray
		3D PSD array of shape (n_psd, n_pix, n_pix).
	low_cutoffs : xp.ndarray or None, optional
		1D array of low-frequency cutoff lengths in meters. If None, no low cutoff.
	high_cutoffs : xp.ndarray or None, optional
		1D array of high-frequency cutoff lengths in meters. If None, no high cutoff.

	Returns
	-------
	xp.ndarray
		PSD stack after applying hard cutoffs.
	"""
	xp = getattr(ao_instru, "xp", np)
	psd_stack = xp.asarray(psd_stack)

	if hasattr(ao_instru, "fr_m"):
		k = ao_instru.fr_m
	else:
		n_pix = ao_instru.n_pix
		d = getattr(ao_instru, "pixel_scale", 1.0)
		fx = xp.fft.fftfreq(n_pix, d=d)
		fy = xp.fft.fftfreq(n_pix, d=d)
		fxg, fyg = xp.meshgrid(fx, fy)
		k = xp.sqrt(fxg**2 + fyg**2)

	out = psd_stack.copy()

	if low_cutoffs is not None:
		low_cutoffs = xp.asarray(low_cutoffs)
		f0 = 1.0 / low_cutoffs
		mask_low = k[None, :, :] < f0[:, None, None]
		out = xp.where(mask_low, 0.0, out)

	if high_cutoffs is not None:
		high_cutoffs = xp.asarray(high_cutoffs)
		f1 = 1.0 / high_cutoffs
		mask_high = k[None, :, :] > f1[:, None, None]
		out = xp.where(mask_high, 0.0, out)

	return out


def phase_from_psd(
	ao_instru,
	psd,
	*,
	rotated_index: int = 0,
	rms: float = 1.0,
) -> np.ndarray:
	"""
	Generate a residual phase screen from a PSD.

	Parameters
	----------
	ao_instru : AO_instrument
		AO_instrument object providing ao_instru mask and backend.
	psd : xp.ndarray
		2D PSD array matching the ao_instru FFT grid shape.
	rms : float, optional
		Target RMS of the phase on the ao_instru support. Default is 1.

	Returns
	-------
	xp.ndarray
		2D phase screen.
	"""
	xp = getattr(ao_instru, "xp", np)
	psd = xp.asarray(psd)

	# Random complex spectrum with PSD weighting
	amp = xp.sqrt(psd)
	re = xp.random.standard_normal(psd.shape)
	im = xp.random.standard_normal(psd.shape)
	noise = (re + 1j * im) / xp.sqrt(2.0)
	F = noise * amp
	F = xp.where(xp.isfinite(F), F, 0.0)
	F = xp.asarray(F, dtype=xp.complex64)
	F[0, 0] = 0.0

	phase = xp.real(xp.fft.ifft2(F))

	# Normalize RMS on ao_instru support
	if hasattr(ao_instru, "rotated_pupils") and ao_instru.rotated_pupils is not None:
		mask = ao_instru.rotated_pupils[rotated_index] > 0.5
	else:
		mask = ao_instru.pupil_array > 0.5
	if bool(xp.any(mask)):
		vals = phase[mask]
		mean = xp.mean(vals)
		vals = vals - mean
		var = xp.mean(vals * vals)
		curr_rms = xp.sqrt(xp.maximum(var, 1e-30))
		phase = (phase - mean) * (rms / curr_rms)

	return phase


def phase_from_psd_vectorized(
	ao_instru,
	psd_stack,
	rms_values,
	*,
	rotated_index: int = 0,
) -> np.ndarray:
	"""
	Generate residual phase screens from a stack of PSDs.

	Parameters
	----------
	ao_instru : AO_instrument
		AO_instrument object providing ao_instru mask and backend.
	psd_stack : xp.ndarray
		3D PSD array of shape (n_psd, n_pix, n_pix).
	rms_values : xp.ndarray
		1D array of target RMS values (one per PSD).

	Returns
	-------
	xp.ndarray
		3D phase screen array of shape (n_psd, n_pix, n_pix).
	"""
	xp = getattr(ao_instru, "xp", np)
	psd_stack = xp.asarray(psd_stack)
	rms_values = xp.asarray(rms_values)

	amp = xp.sqrt(psd_stack)
	re = xp.random.standard_normal(psd_stack.shape)
	im = xp.random.standard_normal(psd_stack.shape)
	noise = (re + 1j * im) / xp.sqrt(2.0)
	F = noise * amp
	F = xp.where(xp.isfinite(F), F, 0.0).astype(xp.complex64)
	F[:, 0, 0] = 0.0

	phase = xp.real(xp.fft.ifft2(F, axes=(-2, -1)))

	if hasattr(ao_instru, "rotated_pupils") and ao_instru.rotated_pupils is not None:
		mask = ao_instru.rotated_pupils[rotated_index] > 0.5
	else:
		mask = ao_instru.pupil_array > 0.5
	if bool(xp.any(mask)):
		mask3 = mask[None, :, :]
		pupil_sum = xp.sum(mask)
		mean = xp.sum(phase * mask3, axis=(-2, -1)) / pupil_sum
		phase = phase - mean[:, None, None]
		var = xp.sum((phase ** 2) * mask3, axis=(-2, -1)) / pupil_sum
		curr_rms = xp.sqrt(xp.maximum(var, 1e-30))
		scale = rms_values / curr_rms
		phase = phase * scale[:, None, None]

	return phase


def generate_dual_powerlaw_phase(
	ao_instru,
	exponent_lf,
	exponent_hf,
	cutoff,
	rms_lf,
	rms_hf,
	*,
	rotated_index: int = 0,
):
	"""
	Generate a phase screen (or stack) from LF and HF power-law PSDs.

	Parameters
	----------
	ao_instru : AO_instrument
		AO_instrument object.
	exponent_lf : float or xp.ndarray
		Low-frequency power-law exponent(s).
	exponent_hf : float or xp.ndarray
		High-frequency power-law exponent(s).
	cutoff : float or xp.ndarray
		Cutoff length in meters (same length as exponents for vectorized mode).
	rms_lf : float or xp.ndarray
		Target RMS for LF phase on ao_instru.
	rms_hf : float or xp.ndarray
		Target RMS for HF phase on ao_instru.

	Returns
	-------
	xp.ndarray
		Phase screen or stack of phase screens.
	"""
	xp = getattr(ao_instru, "xp", np)

	# Vectorized path if any input is an array
	if xp.ndim(xp.asarray(exponent_lf)) > 0:
		exp_lf = xp.asarray(exponent_lf)
		exp_hf = xp.asarray(exponent_hf)
		cutoffs = xp.asarray(cutoff)
		rms_lf_arr = xp.asarray(rms_lf)
		rms_hf_arr = xp.asarray(rms_hf)

		psd_lf = radial_powerlaw_psd_vectorized(ao_instru, exp_lf)
		psd_hf = radial_powerlaw_psd_vectorized(ao_instru, exp_hf)

		psd_lf = apply_cutoff_vectorized(ao_instru, psd_lf, low_cutoffs=None, high_cutoffs=cutoffs)
		psd_hf = apply_cutoff_vectorized(ao_instru, psd_hf, low_cutoffs=cutoffs, high_cutoffs=None)

		phase_lf = phase_from_psd_vectorized(
			ao_instru,
			psd_lf,
			rms_lf_arr,
			rotated_index=rotated_index,
		)
		phase_hf = phase_from_psd_vectorized(
			ao_instru,
			psd_hf,
			rms_hf_arr,
			rotated_index=rotated_index,
		)
		return phase_lf + phase_hf

	# Scalar path
	psd_lf = radial_powerlaw_psd(ao_instru, exponent=float(exponent_lf))
	psd_hf = radial_powerlaw_psd(ao_instru, exponent=float(exponent_hf))

	psd_lf = apply_cutoff(ao_instru, psd_lf, low_cutoff=None, high_cutoff=cutoff)
	psd_hf = apply_cutoff(ao_instru, psd_hf, low_cutoff=cutoff, high_cutoff=None)

	phase_lf = phase_from_psd(
		ao_instru,
		psd_lf,
		rms=float(rms_lf),
		rotated_index=rotated_index,
	)
	phase_hf = phase_from_psd(
		ao_instru,
		psd_hf,
		rms=float(rms_hf),
		rotated_index=rotated_index,
	)

	return phase_lf + phase_hf

