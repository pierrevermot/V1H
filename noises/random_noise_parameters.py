"""Random parameter generators for noise simulation."""

from __future__ import annotations

import numpy as np

from noises.noise import add_gaussian_noise, add_sqrt_noise, add_point_source_noise, pixels_to_zero
from utils.random_compat import rng_random as _rng_random


def draw_noise_rel_std(*, x_min: float = 0.0, x_max: float = 1.0, rng=None):
	"""Draw a 2-element list [x, 1-x] with x uniform in [x_min, x_max]."""
	if x_max < x_min:
		raise ValueError("x_max must be >= x_min")
	if rng is None:
		u = float(np.random.random())
	else:
		u = float(_rng_random(rng))
	x = x_min + (x_max - x_min) * u
	return [x, 1.0 - x]


def draw_peak_snr(*, snr_min: float = 1.0, snr_max: float = 1e4, rng=None):
	"""Draw peak SNR log-uniformly between snr_min and snr_max."""
	if snr_min <= 0 or snr_max <= 0:
		raise ValueError("snr_min and snr_max must be > 0")
	if snr_max < snr_min:
		raise ValueError("snr_max must be >= snr_min")
	log_min = np.log(float(snr_min))
	log_max = np.log(float(snr_max))
	if rng is None:
		u = float(np.random.random())
	else:
		u = float(_rng_random(rng))
	return float(np.exp(log_min + (log_max - log_min) * u))


def draw_point_source_params(
	ao_instru,
	*,
	n_min: int = 0,
	n_max_multiplier: int = 2,
	flux_min: float = 0.0,
	flux_max: float = 1.0,
	rng=None,
):
	"""Draw fluxes for add_point_source_noise.

	Fluxes length is uniform in [n_min, n_max_multiplier*n_frames],
	values uniform in [flux_min, flux_max].
	"""
	if n_min < 0:
		raise ValueError("n_min must be >= 0")
	if n_max_multiplier < 0:
		raise ValueError("n_max_multiplier must be >= 0")
	if flux_max < flux_min:
		raise ValueError("flux_max must be >= flux_min")
	n_frames = int(getattr(ao_instru, "n_frames", 1))
	max_n = max(n_min, n_max_multiplier * n_frames)
	if rng is None:
		n = int(np.random.randint(n_min, max_n + 1))
		fluxes = np.random.random(n)
	else:
		if hasattr(rng, "integers"):
			n = int(rng.integers(n_min, max_n + 1))
		else:
			n = int(rng.randint(n_min, max_n + 1))
		fluxes = np.asarray(_rng_random(rng, size=(n,)))
	fluxes = flux_min + (flux_max - flux_min) * np.asarray(fluxes)
	return (fluxes,)


def draw_pixels_to_zero_params(
	*,
	n_min: int = 0,
	n_max: int = 5,
	prob_same_each_frame: float = 0.5,
	rng=None,
):
	"""Draw parameters for pixels_to_zero.

	n_per_frame is uniform int in [n_min, n_max]; same_each_frame is random bool.
	"""
	if n_min < 0:
		raise ValueError("n_min must be >= 0")
	if n_max < n_min:
		raise ValueError("n_max must be >= n_min")
	if not (0.0 <= prob_same_each_frame <= 1.0):
		raise ValueError("prob_same_each_frame must be between 0 and 1")
	if rng is None:
		n_per_frame = int(np.random.randint(n_min, n_max + 1))
		same_each_frame = bool(np.random.random() < prob_same_each_frame)
	else:
		if hasattr(rng, "integers"):
			n_per_frame = int(rng.integers(n_min, n_max + 1))
		else:
			n_per_frame = int(rng.randint(n_min, n_max + 1))
		same_each_frame = bool(float(_rng_random(rng)) < prob_same_each_frame)
	return (n_per_frame, same_each_frame)


def draw_random_noise_parameters(
	ao_instru,
	*,
	noise_rel_x_min: float = 0.0,
	noise_rel_x_max: float = 1.0,
	peak_snr_min: float = 1.0,
	peak_snr_max: float = 1e4,
	point_sources_n_min: int = 0,
	point_sources_n_max_multiplier: int = 2,
	point_sources_flux_min: float = 0.0,
	point_sources_flux_max: float = 1.0,
	zero_pixels_n_min: int = 0,
	zero_pixels_n_max: int = 5,
	zero_pixels_prob_same_each_frame: float = 0.5,
	rng=None,
):
	"""Draw random parameters to feed noise_simulator.

	Returns
	-------
	tuple
		(noise_functions, noise_params, noise_rel_std, peak_snr,
		 pixel_functions, pixel_params)
	"""
	noise_functions = [add_gaussian_noise, add_sqrt_noise]
	noise_params = [(), ()]
	noise_rel_std = draw_noise_rel_std(
		x_min=noise_rel_x_min,
		x_max=noise_rel_x_max,
		rng=rng,
	)
	peak_snr = draw_peak_snr(
		snr_min=peak_snr_min,
		snr_max=peak_snr_max,
		rng=rng,
	)

	pixel_functions = [add_point_source_noise, pixels_to_zero]
	point_params = draw_point_source_params(
		ao_instru,
		n_min=point_sources_n_min,
		n_max_multiplier=point_sources_n_max_multiplier,
		flux_min=point_sources_flux_min,
		flux_max=point_sources_flux_max,
		rng=rng,
	)
	zero_params = draw_pixels_to_zero_params(
		n_min=zero_pixels_n_min,
		n_max=zero_pixels_n_max,
		prob_same_each_frame=zero_pixels_prob_same_each_frame,
		rng=rng,
	)
	pixel_params = [point_params, zero_params]

	return (
		noise_functions,
		noise_params,
		noise_rel_std,
		peak_snr,
		pixel_functions,
		pixel_params,
	)
