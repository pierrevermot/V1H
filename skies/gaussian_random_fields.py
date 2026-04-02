"""Gaussian random field generation and astronomical object functions.

Provides power spectral density (PSD) generators, a GRF sampler, and
three sky-object constructors (nebula, point sources, sharp-edged objects)
that operate on the frequency grids stored in an AO_instrument instance.

All PSD functions work on ``ao_instru.fr_pix_large`` (the 2N × 2N grid).
``grf_from_psd`` draws a realisation on that large grid, then centre-crops
to ``ao_instru.n_pix × ao_instru.n_pix``.
"""

from __future__ import annotations

import numpy as np

from utils.array_backend import is_cupy_array as _is_cupy_array
from utils.array_backend import get_xp_from_array as _get_xp


def _normalize_mean(image, xp):
	"""Normalize image so its mean is 1 (fallback to ones if mean <= 0)."""
	mean_val = float(xp.mean(image))
	if mean_val > 0.0:
		return image / mean_val
	return xp.ones_like(image)


# ---------------------------------------------------------------------------
# PSD functions  (all work on the 2N×2N ``fr_pix_large`` grid)
# ---------------------------------------------------------------------------

def gaussian_psd(ao_instru, std0_pix: float):
	"""Gaussian PSD: P(k) = exp(-k² / (2 σ₀²)).

	Parameters
	----------
	ao_instru : AO_instrument
		Instrument object providing ``fr_pix_large`` and ``xp``.
	std0_pix : float
		Characteristic width in cycles-per-pixel.

	Returns
	-------
	ndarray
		2D PSD array with shape ``(2*n_pix, 2*n_pix)``.
	"""
	xp = getattr(ao_instru, "xp", np)
	k = ao_instru.fr_pix_large
	return xp.exp(-k ** 2 / (2.0 * std0_pix ** 2))


def exponential_psd(ao_instru, k0_pix: float):
	"""Exponential PSD: P(k) = exp(-k / k₀).

	Parameters
	----------
	ao_instru : AO_instrument
		Instrument object providing ``fr_pix_large`` and ``xp``.
	k0_pix : float
		Characteristic frequency in cycles-per-pixel.

	Returns
	-------
	ndarray
		2D PSD array with shape ``(2*n_pix, 2*n_pix)``.
	"""
	xp = getattr(ao_instru, "xp", np)
	k = ao_instru.fr_pix_large
	return xp.exp(-k / k0_pix)


def powerlaw_psd(ao_instru, alpha: float):
	"""Power-law PSD: P(k) ∝ k^{-α}  (DC set to 0).

	Parameters
	----------
	ao_instru : AO_instrument
		Instrument object providing ``fr_pix_large`` and ``xp``.
	alpha : float
		Power-law exponent (positive → red spectrum).

	Returns
	-------
	ndarray
		2D PSD array with shape ``(2*n_pix, 2*n_pix)``.
	"""
	xp = getattr(ao_instru, "xp", np)
	k = ao_instru.fr_pix_large
	psd = xp.zeros_like(k, dtype=xp.float64)
	mask = k > 0
	psd[mask] = k[mask] ** (-alpha)
	return psd


# ---------------------------------------------------------------------------
# GRF sampler
# ---------------------------------------------------------------------------

def grf_from_psd(ao_instru, psd, *, rng=None):
	"""Draw a Gaussian random field realisation from a 2D PSD.

	The field is generated on the 2N × 2N grid (matching ``fr_pix_large``)
	and centre-cropped to ``n_pix × n_pix``.

	Parameters
	----------
	ao_instru : AO_instrument
		Instrument object providing ``n_pix``, ``xp``, and the large
		frequency grid shape.
	psd : ndarray
		2D PSD array of shape ``(2*n_pix, 2*n_pix)``.
	rng : numpy.random.Generator or None, optional
		Random number generator.  If *None*, ``numpy.random`` is used.

	Returns
	-------
	ndarray
		Real-valued 2D array of shape ``(n_pix, n_pix)``.
	"""
	xp = _get_xp(psd)
	n_large = psd.shape[0]
	n_pix = ao_instru.n_pix

	# Draw complex white noise in Fourier space
	if rng is None:
		noise_np = np.random.standard_normal((n_large, n_large)) \
			+ 1j * np.random.standard_normal((n_large, n_large))
	else:
		noise_np = rng.standard_normal((n_large, n_large)) \
			+ 1j * rng.standard_normal((n_large, n_large))

	if xp is not np:
		noise = xp.asarray(noise_np)
	else:
		noise = noise_np

	# Colour the noise
	field_fft = xp.sqrt(psd) * noise
	field_large = xp.real(xp.fft.ifft2(field_fft))

	# Centre-crop to n_pix × n_pix
	start = (n_large - n_pix) // 2
	field = field_large[start:start + n_pix, start:start + n_pix]
	return field


# ---------------------------------------------------------------------------
# Astronomical object generators
# ---------------------------------------------------------------------------

def nebula(ao_instru, exponent: float, percentile: float):
	"""Generate a nebula-like object via a thresholded power-law GRF.

	Parameters
	----------
	ao_instru : AO_instrument
		Instrument providing frequency grids and backend.
	exponent : float
		Power-law exponent for the PSD.
	percentile : float
		Percentile threshold (0–100).  Pixels below the threshold are
		zeroed, creating a sparse, cloudy structure.

	Returns
	-------
	ndarray
		2D image of shape ``(n_pix, n_pix)`` with values ≥ 0.
	"""
	xp = getattr(ao_instru, "xp", np)
	psd = powerlaw_psd(ao_instru, exponent)
	field = grf_from_psd(ao_instru, psd)

	# Threshold at the requested percentile
	if xp is not np:
		thresh = float(xp.percentile(field, percentile))
	else:
		thresh = float(np.percentile(field, percentile))

	image = xp.clip(field - thresh, 0.0, None)
	return _normalize_mean(image, xp)


def point_sources(ao_instru, n: int, exponent: float):
	"""Generate a field of random point sources.

	Each source is a single bright pixel placed at a random position.
	Brightnesses follow a power-law distribution.

	Parameters
	----------
	ao_instru : AO_instrument
		Instrument providing ``n_pix`` and backend.
	n : int
		Number of point sources.
	exponent : float
		Power-law exponent for the brightness distribution.
		Brightnesses are drawn as U^{-1/(exponent-1)} where U ~ Uniform(0,1).

	Returns
	-------
	ndarray
		2D image of shape ``(n_pix, n_pix)``.
	"""
	xp = getattr(ao_instru, "xp", np)
	n_pix = ao_instru.n_pix
	image = xp.zeros((n_pix, n_pix), dtype=xp.float64)

	n = int(n)
	if n < 1:
		return image

	# Random positions (CPU, then transfer if needed)
	ys = np.random.randint(0, n_pix, size=n)
	xs = np.random.randint(0, n_pix, size=n)

	# Power-law brightnesses
	u = np.random.random(n)
	u = np.clip(u, 1e-12, 1.0)
	alpha = max(exponent, 1.01)  # ensure exponent > 1
	brightnesses = u ** (-1.0 / (alpha - 1.0))
	brightnesses /= brightnesses.max()  # normalise to [0, 1]

	if xp is not np:
		brightnesses_xp = xp.asarray(brightnesses)
		for i in range(n):
			image[int(ys[i]), int(xs[i])] += float(brightnesses_xp[i])
	else:
		for i in range(n):
			image[int(ys[i]), int(xs[i])] += brightnesses[i]

	return _normalize_mean(image, xp)


def sharp_edges_object(
	ao_instru,
	exponent_lf: float,
	percentile_lf: float,
	exponent_hf: float,
	vmin_hf: float,
):
	"""Generate an object with sharp edges (e.g. galaxy-like structure).

	A low-frequency GRF defines the overall shape (thresholded at
	*percentile_lf*), while a high-frequency GRF adds internal texture.

	Parameters
	----------
	ao_instru : AO_instrument
		Instrument providing frequency grids and backend.
	exponent_lf : float
		Power-law exponent for the low-frequency (shape) component.
	percentile_lf : float
		Percentile threshold for the shape mask (0–100).
	exponent_hf : float
		Power-law exponent for the high-frequency (texture) component.
	vmin_hf : float
		Minimum clip value for the texture component (in [0, 1] of its
		range), controlling how much internal contrast is retained.

	Returns
	-------
	ndarray
		2D image of shape ``(n_pix, n_pix)`` with values ≥ 0.
	"""
	xp = getattr(ao_instru, "xp", np)

	# Low-frequency shape mask
	psd_lf = powerlaw_psd(ao_instru, exponent_lf)
	field_lf = grf_from_psd(ao_instru, psd_lf)

	if xp is not np:
		thresh_lf = float(xp.percentile(field_lf, percentile_lf))
	else:
		thresh_lf = float(np.percentile(field_lf, percentile_lf))

	mask = (field_lf >= thresh_lf).astype(xp.float64)

	# High-frequency texture
	psd_hf = powerlaw_psd(ao_instru, exponent_hf)
	field_hf = grf_from_psd(ao_instru, psd_hf)

	# Normalise texture to [0, 1]
	hf_min = float(xp.min(field_hf))
	hf_max = float(xp.max(field_hf))
	if hf_max - hf_min > 0:
		field_hf = (field_hf - hf_min) / (hf_max - hf_min)
	else:
		field_hf = xp.ones_like(field_hf)

	# Map the texture to [vmin_hf, 1]
	field_hf = float(vmin_hf) + (1.0 - float(vmin_hf)) * field_hf

	image = mask * field_hf
	return _normalize_mean(image, xp)
