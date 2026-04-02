"""Composite image generator from multiple sky-object functions.

Given lists of object-generator functions, their parameters, and per-object
fluxes, this module produces a single normalised sky image by summing the
flux-weighted contributions.
"""

from __future__ import annotations

import numpy as np

from utils.array_backend import is_cupy_array


def image_generator(
	ao_instru,
	function_list,
	params_list,
	flux_list,
	*,
	max_iter: int = 10,
):
	"""Build a composite sky image from a collection of object generators.

	Each function in *function_list* is called as
	``func(ao_instru, *params)`` where *params* comes from the
	corresponding entry in *params_list*.  The resulting image is
	multiplied by the matching flux and accumulated.  The final image
	is normalised by its mean so that the output represents relative
	surface-brightness.

	If a call produces an image whose sum is zero or contains NaN
	values, it is retried up to *max_iter* times before being skipped.

	Parameters
	----------
	ao_instru : AO_instrument
		Instrument object providing ``n_pix``, ``xp``, and frequency grids.
	function_list : list[callable]
		Object generator functions (e.g. ``nebula``, ``point_sources``,
		``sharp_edges_object``).
	params_list : list[tuple | dict | scalar]
		Parameters for each function.  Tuples are unpacked as positional
		args; dicts are unpacked as keyword args; scalars are passed as a
		single positional arg.
	flux_list : array-like
		Per-object flux weights (same length as *function_list*).
	max_iter : int, optional
		Maximum number of retry attempts for an object that produces an
		all-zero or NaN image.  Default is 10.

	Returns
	-------
	ndarray
		2D image of shape ``(n_pix, n_pix)``, normalised by its mean
		(mean ≈ 1).  If no valid objects could be generated the array
		is filled with ones.
	"""
	xp = getattr(ao_instru, "xp", np)
	n_pix = ao_instru.n_pix

	composite = xp.zeros((n_pix, n_pix), dtype=xp.float64)
	flux_arr = np.asarray(flux_list, dtype=np.float64)

	for idx, (func, params, flux) in enumerate(
		zip(function_list, params_list, flux_arr)
	):
		flux = float(flux)
		if flux == 0.0:
			continue

		# Retry loop for degenerate realisations
		for attempt in range(max_iter):
			# Call the generator
			if isinstance(params, dict):
				obj = func(ao_instru, **params)
			elif isinstance(params, (tuple, list)):
				obj = func(ao_instru, *params)
			else:
				obj = func(ao_instru, params)

			# Validate
			obj_sum = float(xp.sum(obj))
			if obj_sum > 0.0 and not np.isnan(obj_sum):
				break
		else:
			# All attempts failed – skip this object
			continue

		composite += flux * obj

	# Normalise by mean (so output mean ≈ 1)
	mean_val = float(xp.mean(composite))
	if mean_val > 0.0:
		composite = composite / mean_val
	else:
		# Fallback: uniform image
		composite = xp.ones((n_pix, n_pix), dtype=xp.float64)

	return composite
