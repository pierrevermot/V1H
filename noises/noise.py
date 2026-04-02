"""Noise generation utilities."""

from __future__ import annotations

import numpy as np


def add_gaussian_noise(ao_instru, data):
	"""Return *data* with additive Gaussian noise using AO_instrument backend.

	Parameters
	----------
	ao_instru : AO_instrument
		Instrument object providing backend via ``xp``.
	data : array-like
		2D image or 3D cube (n_images, ny, nx).

	Returns
	-------
	ndarray
		Noisy data with the same shape as *data*.
	"""
	xp = getattr(ao_instru, "xp", np)
	data_xp = xp.asarray(data)

	if data_xp.ndim == 2:
		noise = xp.random.normal(loc=0.0, scale=1.0, size=data_xp.shape)
		return noise
	if data_xp.ndim == 3:
		noise = xp.random.normal(loc=0.0, scale=1.0, size=data_xp.shape)
		return noise
	raise ValueError("data must be a 2D image or 3D cube")


def add_sqrt_noise(ao_instru, data):
	"""Return *data* with additive Gaussian noise of std = sqrt(data).

	Parameters
	----------
	ao_instru : AO_instrument
		Instrument object providing backend via ``xp``.
	data : array-like
		2D image or 3D cube. Noise is applied elementwise with
		standard deviation equal to ``sqrt(data)``.

	Returns
	-------
	ndarray
		Noisy data with the same shape as *data*.
	"""
	xp = getattr(ao_instru, "xp", np)
	data_xp = xp.asarray(data)
	std = xp.sqrt(xp.maximum(data_xp, 0.0))
	noise = xp.random.normal(loc=0.0, scale=std, size=data_xp.shape)
	return noise


def add_point_source_noise(ao_instru, data, fluxes):
	"""Return a point-source noise map with given fluxes.

	Parameters
	----------
	ao_instru : AO_instrument
		Instrument object providing backend via ``xp``.
	data : array-like
		2D image or 3D cube used only for shape.
	fluxes : array-like
		Sequence of flux values. The noise map will contain one pixel per flux.

	Returns
	-------
	ndarray
		Noise map of zeros with ``len(fluxes)`` pixels set to the flux values.
		If *data* is a cube, this is applied independently per frame.
	"""
	xp = getattr(ao_instru, "xp", np)
	data_xp = xp.asarray(data)
	flux_arr = np.asarray(fluxes, dtype=float)

	if flux_arr.size == 0:
		return data_xp

	if data_xp.ndim == 2:
		ny, nx = data_xp.shape
		noise = xp.zeros((ny, nx), dtype=xp.float64)
		ys = np.random.randint(0, ny, size=flux_arr.size)
		xs = np.random.randint(0, nx, size=flux_arr.size)
		for i in range(flux_arr.size):
			noise[int(ys[i]), int(xs[i])] += float(flux_arr[i])
		return data_xp + noise

	if data_xp.ndim == 3:
		n_frames, ny, nx = data_xp.shape
		noise = xp.zeros((n_frames, ny, nx), dtype=xp.float64)
		for f in range(n_frames):
			ys = np.random.randint(0, ny, size=flux_arr.size)
			xs = np.random.randint(0, nx, size=flux_arr.size)
			for i in range(flux_arr.size):
				noise[f, int(ys[i]), int(xs[i])] += float(flux_arr[i])
		return data_xp + noise

	raise ValueError("data must be a 2D image or 3D cube")


def pixels_to_zero(ao_instru, data, n_per_frame: int, same_each_frame: bool):
	"""Return *data* with random pixels set to zero.

	Parameters
	----------
	ao_instru : AO_instrument
		Instrument object providing backend via ``xp``.
	data : array-like
		2D image or 3D cube.
	n_per_frame : int
		Number of pixels to zero per frame.
	same_each_frame : bool
		If True and *data* is a cube, zero the same pixels in each frame.

	Returns
	-------
	ndarray
		Data with selected pixels set to zero.
	"""
	xp = getattr(ao_instru, "xp", np)
	data_xp = xp.asarray(data)
	result = data_xp.copy()
	if n_per_frame <= 0:
		return result

	if result.ndim == 2:
		ny, nx = result.shape
		ys = np.random.randint(0, ny, size=n_per_frame)
		xs = np.random.randint(0, nx, size=n_per_frame)
		for i in range(n_per_frame):
			result[int(ys[i]), int(xs[i])] = 0.0
		return result

	if result.ndim == 3:
		n_frames, ny, nx = result.shape
		if same_each_frame:
			ys = np.random.randint(0, ny, size=n_per_frame)
			xs = np.random.randint(0, nx, size=n_per_frame)
			for f in range(n_frames):
				for i in range(n_per_frame):
					result[f, int(ys[i]), int(xs[i])] = 0.0
			return result
		for f in range(n_frames):
			ys = np.random.randint(0, ny, size=n_per_frame)
			xs = np.random.randint(0, nx, size=n_per_frame)
			for i in range(n_per_frame):
				result[f, int(ys[i]), int(xs[i])] = 0.0
		return result

	raise ValueError("data must be a 2D image or 3D cube")
