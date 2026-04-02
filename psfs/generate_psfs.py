"""PSF generation utilities."""

from __future__ import annotations

import numpy as np


def short_exposure_psf(ao_instru, phase_screen, *, rotated_index: int = 0):
	"""
	Compute a short-exposure PSF from a ao_instru and phase screen.

	Parameters
	----------
	ao_instru : AO_instrument
		AO_instrument object providing ``pupil_array`` and backend (xp/cp).
	phase_screen : array-like
		Phase screen array matching the ao_instru shape.

	Returns
	-------
	xp.ndarray
		Normalized PSF with sum(psf) = 1.
	"""
	xp = getattr(ao_instru, "xp", np)
	phase_screen = xp.asarray(phase_screen)

	if phase_screen.shape != ao_instru.pupil_array.shape:
		raise ValueError(
			"phase_screen must have the same shape as ao_instru.pupil_array"
		)

	if hasattr(ao_instru, "rotated_pupils") and ao_instru.rotated_pupils is not None:
		pupil_array = ao_instru.rotated_pupils[rotated_index]
	else:
		pupil_array = ao_instru.pupil_array

	complex_pupil = pupil_array * xp.exp(1j * phase_screen)
	field = xp.fft.fft2(complex_pupil)
	psf = xp.abs(xp.fft.fftshift(field)) ** 2

	flux = xp.sum(psf)
	if float(flux) == 0.0:
		return psf
	return psf / flux


def short_exposure_psf_vectorized(ao_instru, phase_screens, *, rotated_index: int = 0):
	"""
	Compute short-exposure PSFs for multiple phase screens.

	Parameters
	----------
	ao_instru : AO_instrument
		AO_instrument object providing ``pupil_array`` and backend (xp/cp).
	phase_screens : array-like
		Phase screen stack with shape (n_screens, n_pix, n_pix).

	Returns
	-------
	xp.ndarray
		Normalized PSF stack with shape (n_screens, n_pix, n_pix),
		each with sum(psf) = 1.
	"""
	xp = getattr(ao_instru, "xp", np)
	phase_screens = xp.asarray(phase_screens)

	if phase_screens.ndim != 3:
		raise ValueError("phase_screens must have shape (n_screens, n_pix, n_pix)")
	if phase_screens.shape[1:] != ao_instru.pupil_array.shape:
		raise ValueError(
			"phase_screens must have shape (n_screens, n_pix, n_pix) matching ao_instru"
		)

	if hasattr(ao_instru, "rotated_pupils") and ao_instru.rotated_pupils is not None:
		pupil_array = ao_instru.rotated_pupils[rotated_index]
	else:
		pupil_array = ao_instru.pupil_array

	complex_pupil = pupil_array[None, :, :] * xp.exp(1j * phase_screens)
	field = xp.fft.fft2(complex_pupil, axes=(-2, -1))
	psf = xp.abs(xp.fft.fftshift(field, axes=(-2, -1))) ** 2

	flux = xp.sum(psf, axis=(-2, -1))
	flux_safe = xp.where(flux == 0, 1.0, flux)
	psf = psf / flux_safe[:, None, None]

	return psf


def long_exposure_psf(ao_instru, phase_screens, *, rotated_index: int = 0):
	"""
	Compute a long-exposure PSF by averaging short-exposure PSFs.

	Parameters
	----------
	ao_instru : AO_instrument
		AO_instrument object providing ``pupil_array`` and backend (xp/cp).
	phase_screens : array-like
		Phase screen stack with shape (n_screens, n_pix, n_pix).
	rotated_index : int, optional
		Rotation index for the ao_instru mask. Default is 0.

	Returns
	-------
	xp.ndarray
		Long-exposure PSF with shape (n_pix, n_pix).
	"""
	xp = getattr(ao_instru, "xp", np)
	se_psfs = short_exposure_psf_vectorized(
		ao_instru,
		phase_screens,
		rotated_index=rotated_index,
	)
	return xp.mean(se_psfs, axis=0)


def long_exposure_psfs_vectorized(
	ao_instru,
	phase_screens,
	*,
	rotated_index: int = 0,
	n_se_screens_per_le: int,
	n_le: int,
):
	"""
	Compute multiple long-exposure PSFs by averaging short-exposure PSFs in groups.

	Parameters
	----------
	ao_instru : AO_instrument
		AO_instrument object providing ``pupil_array`` and backend (xp/cp).
	phase_screens : array-like
		Phase screen stack with shape (n_se_screens, n_pix, n_pix).
	rotated_index : int, optional
		Rotation index for the ao_instru mask. Default is 0.
	n_se_screens_per_le : int
		Number of short-exposure screens to average per long exposure.
	n_le : int
		Number of long-exposure PSFs to produce.

	Returns
	-------
	xp.ndarray
		Long-exposure PSF stack with shape (n_le, n_pix, n_pix).
	"""
	xp = getattr(ao_instru, "xp", np)
	phase_screens = xp.asarray(phase_screens)

	if phase_screens.ndim != 3:
		raise ValueError("phase_screens must have shape (n_se_screens, n_pix, n_pix)")
	if phase_screens.shape[1:] != ao_instru.pupil_array.shape:
		raise ValueError(
			"phase_screens must have shape (n_se_screens, n_pix, n_pix) matching ao_instru"
		)
	if n_se_screens_per_le <= 0 or n_le <= 0:
		raise ValueError("n_se_screens_per_le and n_le must be positive")

	n_se_screens = int(phase_screens.shape[0])
	if n_se_screens != n_se_screens_per_le * n_le:
		raise ValueError(
			"n_se_screens must equal n_se_screens_per_le * n_le"
		)

	se_psfs = short_exposure_psf_vectorized(
		ao_instru,
		phase_screens,
		rotated_index=rotated_index,
	)

	se_psfs = se_psfs.reshape(n_le, n_se_screens_per_le, *se_psfs.shape[1:])
	return xp.mean(se_psfs, axis=1)
