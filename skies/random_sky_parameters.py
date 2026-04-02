"""Random parameter generators for sky objects."""

from __future__ import annotations

import numpy as np

from skies.gaussian_random_fields import nebula, point_sources, sharp_edges_object
from utils.random_compat import rng_random as _rng_random


def draw_n_objects(
	*,
	n_objects_min: int = 1,
	n_objects_max: int = 20,
	rng=None,
):
	"""Draw a log-uniform integer number of objects."""
	if n_objects_min < 1 or n_objects_max < 1:
		raise ValueError("n_objects_min and n_objects_max must be >= 1")
	if n_objects_max < n_objects_min:
		raise ValueError("n_objects_max must be >= n_objects_min")

	log_min = np.log(float(n_objects_min))
	log_max = np.log(float(n_objects_max))
	if rng is None:
		u = np.random.random()
	else:
		u = float(_rng_random(rng))
	val = np.exp(log_min + (log_max - log_min) * u)
	val_int = int(np.clip(np.rint(val), n_objects_min, n_objects_max))
	return val_int


def draw_random_object_function(*, rng=None):
	"""Draw a random sky object generator function."""
	functions = (nebula, point_sources, sharp_edges_object)
	if rng is None:
		idx = int(np.random.randint(0, len(functions)))
	else:
		idx = int(_rng_random(rng) * len(functions))
		idx = max(0, min(len(functions) - 1, idx))
	return functions[idx]


def draw_nebula_params(
	*,
	exponent_min: float = 1.5,
	exponent_max: float = 5.0,
	percentile_min: float = 50.0,
	percentile_max: float = 99.0,
	rng=None,
):
	"""Draw random parameters for nebula generation."""
	if exponent_max < exponent_min:
		raise ValueError("exponent_max must be >= exponent_min")
	if percentile_max < percentile_min:
		raise ValueError("percentile_max must be >= percentile_min")

	if rng is None:
		u1 = np.random.random()
		u2 = np.random.random()
	else:
		u1 = float(_rng_random(rng))
		u2 = float(_rng_random(rng))

	exponent = exponent_min + (exponent_max - exponent_min) * u1
	percentile = percentile_min + (percentile_max - percentile_min) * u2
	return exponent, percentile


def draw_point_sources_params(
	*,
	n_min: int = 1,
	n_max: int = 1000,
	exponent_min: float = 1.5,
	exponent_max: float = 3.5,
	rng=None,
):
	"""Draw random parameters for point source generation."""
	if n_min < 1 or n_max < 1:
		raise ValueError("n_min and n_max must be >= 1")
	if n_max < n_min:
		raise ValueError("n_max must be >= n_min")
	if exponent_max < exponent_min:
		raise ValueError("exponent_max must be >= exponent_min")

	log_min = np.log(float(n_min))
	log_max = np.log(float(n_max))
	if rng is None:
		u = np.random.random()
		u_exp = np.random.random()
	else:
		u = float(_rng_random(rng))
		u_exp = float(_rng_random(rng))

	n_val = np.exp(log_min + (log_max - log_min) * u)
	n_int = int(np.clip(np.rint(n_val), n_min, n_max))
	exponent = exponent_min + (exponent_max - exponent_min) * u_exp
	return n_int, exponent


def draw_sharp_edges_params(
	*,
	exponent_lf_min: float = 1.5,
	exponent_lf_max: float = 5.0,
	percentile_lf_min: float = 50.0,
	percentile_lf_max: float = 99.0,
	exponent_hf_min: float = 1.5,
	exponent_hf_max: float = 5.0,
	vmin_hf_min: float = 0.0,
	vmin_hf_max: float = 1.0,
	rng=None,
):
	"""Draw random parameters for sharp_edges_object generation."""
	if exponent_lf_max < exponent_lf_min:
		raise ValueError("exponent_lf_max must be >= exponent_lf_min")
	if percentile_lf_max < percentile_lf_min:
		raise ValueError("percentile_lf_max must be >= percentile_lf_min")
	if exponent_hf_max < exponent_hf_min:
		raise ValueError("exponent_hf_max must be >= exponent_hf_min")
	if vmin_hf_max < vmin_hf_min:
		raise ValueError("vmin_hf_max must be >= vmin_hf_min")

	if rng is None:
		u1 = np.random.random()
		u2 = np.random.random()
		u3 = np.random.random()
		u4 = np.random.random()
	else:
		u1 = float(_rng_random(rng))
		u2 = float(_rng_random(rng))
		u3 = float(_rng_random(rng))
		u4 = float(_rng_random(rng))

	exponent_lf = exponent_lf_min + (exponent_lf_max - exponent_lf_min) * u1
	percentile_lf = percentile_lf_min + (percentile_lf_max - percentile_lf_min) * u2
	exponent_hf = exponent_hf_min + (exponent_hf_max - exponent_hf_min) * u3
	vmin_hf = vmin_hf_min + (vmin_hf_max - vmin_hf_min) * u4
	return exponent_lf, percentile_lf, exponent_hf, vmin_hf


def draw_uniform_fluxes(
	*,
	n: int,
	rng=None,
):
	"""Draw n uniform fluxes between 0 and 1."""
	if n < 1:
		raise ValueError("n must be >= 1")
	if rng is None:
		return np.random.random(n)
	return np.asarray(_rng_random(rng, size=(n,)))


def draw_random_image_parameters(
	*,
	n_objects_min: int = 1,
	n_objects_max: int = 20,
	# nebula ranges
	nebula_exponent_min: float = 1.5,
	nebula_exponent_max: float = 5.0,
	nebula_percentile_min: float = 50.0,
	nebula_percentile_max: float = 99.0,
	# point_sources ranges
	point_sources_n_min: int = 1,
	point_sources_n_max: int = 1000,
	point_sources_exponent_min: float = 1.5,
	point_sources_exponent_max: float = 3.5,
	# sharp_edges ranges
	sharp_edges_exponent_lf_min: float = 1.5,
	sharp_edges_exponent_lf_max: float = 5.0,
	sharp_edges_percentile_lf_min: float = 50.0,
	sharp_edges_percentile_lf_max: float = 99.0,
	sharp_edges_exponent_hf_min: float = 1.5,
	sharp_edges_exponent_hf_max: float = 5.0,
	sharp_edges_vmin_hf_min: float = 0.0,
	sharp_edges_vmin_hf_max: float = 1.0,
	rng=None,
):
	"""Draw random parameters for image_generator.

	All min/max ranges are forwarded to the individual draw functions.
	"""
	n_objects = draw_n_objects(
		n_objects_min=n_objects_min,
		n_objects_max=n_objects_max,
		rng=rng,
	)

	function_list = []
	params_list = []

	for _ in range(n_objects):
		func = draw_random_object_function(rng=rng)
		function_list.append(func)
		if func is nebula:
			params_list.append(draw_nebula_params(
				exponent_min=nebula_exponent_min,
				exponent_max=nebula_exponent_max,
				percentile_min=nebula_percentile_min,
				percentile_max=nebula_percentile_max,
				rng=rng,
			))
		elif func is point_sources:
			params_list.append(draw_point_sources_params(
				n_min=point_sources_n_min,
				n_max=point_sources_n_max,
				exponent_min=point_sources_exponent_min,
				exponent_max=point_sources_exponent_max,
				rng=rng,
			))
		elif func is sharp_edges_object:
			params_list.append(draw_sharp_edges_params(
				exponent_lf_min=sharp_edges_exponent_lf_min,
				exponent_lf_max=sharp_edges_exponent_lf_max,
				percentile_lf_min=sharp_edges_percentile_lf_min,
				percentile_lf_max=sharp_edges_percentile_lf_max,
				exponent_hf_min=sharp_edges_exponent_hf_min,
				exponent_hf_max=sharp_edges_exponent_hf_max,
				vmin_hf_min=sharp_edges_vmin_hf_min,
				vmin_hf_max=sharp_edges_vmin_hf_max,
				rng=rng,
			))
		else:
			raise ValueError("Unknown object function selected")

	flux_list = draw_uniform_fluxes(n=n_objects, rng=rng)
	return function_list, params_list, flux_list
