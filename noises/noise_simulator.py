"""Noise simulation utilities."""

from __future__ import annotations

import numpy as np


def noise_simulator(
	ao_instru,
	data,
	noise_functions,
	noise_params,
	noise_rel_std,
	peak_snr: float,
	pixel_functions,
	pixel_params,
):
	"""Apply noise and pixel-level effects to data.

	Steps:
	1) Generate a noise map/cube for each (noise_function, noise_params).
	2) Normalize each noise map so its std equals noise_rel_std.
	3) Sum noises, scale so max(data)/std(noise) = peak_snr, then add to data.
	4) Apply each pixel_function with its parameters to the noisy data.
	5) Normalize final output to mean = 1.

	Parameters
	----------
	ao_instru : AO_instrument
		Instrument object providing backend via ``xp``.
	data : array-like
		Input image or cube.
	noise_functions : list[callable]
		List of noise generators called as func(ao_instru, data, *params).
	noise_params : list
		List of parameter sets for each noise function.
	noise_rel_std : list[float]
		Target standard deviation for each generated noise map/cube.
	peak_snr : float
		Target peak SNR defined as max(data)/std(total_noise).
	pixel_functions : list[callable]
		Pixel-level modifiers called as func(ao_instru, data, *params).
	pixel_params : list
		List of parameter sets for each pixel function.

	Returns
	-------
	ndarray
		Modified data, normalized to mean = 1.
	"""
	xp = getattr(ao_instru, "xp", np)
	data_xp = xp.asarray(data)

	if len(noise_functions) != len(noise_params) or len(noise_functions) != len(noise_rel_std):
		raise ValueError("noise_functions, noise_params, and noise_rel_std must have the same length")
	if len(pixel_functions) != len(pixel_params):
		raise ValueError("pixel_functions and pixel_params must have the same length")
	if peak_snr <= 0:
		raise ValueError("peak_snr must be > 0")

	def _apply(func, params, base):
		if isinstance(params, dict):
			return func(ao_instru, base, **params)
		if isinstance(params, (tuple, list)):
			return func(ao_instru, base, *params)
		return func(ao_instru, base, params)

	# 1) Generate and normalize noise maps
	noise_maps = []
	for func, params, target_std in zip(noise_functions, noise_params, noise_rel_std):
		noise = _apply(func, params, data_xp)
		noise = xp.asarray(noise)
		current_std = float(xp.std(noise))
		if current_std > 0:
			noise = noise * (float(target_std) / current_std)
		noise_maps.append(noise)

	# 2) Sum and scale noise to target peak SNR
	if noise_maps:
		total_noise = noise_maps[0]
		for noise in noise_maps[1:]:
			total_noise = total_noise + noise
	else:
		total_noise = xp.zeros_like(data_xp)

	noise_max = float(xp.max(xp.abs(total_noise)))
	data_peak = float(xp.max(data_xp))
	if noise_max > 0:
		target_std = data_peak / float(peak_snr)
		total_noise = total_noise * (target_std / noise_max)
	result = data_xp + total_noise
	mean_result_int = float(xp.mean(result))

	# 3) Apply pixel functions
	result_means = []
	for func, params in zip(pixel_functions, pixel_params):
		result = _apply(func, params, result)
		result = xp.asarray(result)
		result_means.append(float(xp.mean(result)))

	# 4) Normalize to mean = 1
	mean_val = float(xp.mean(result))
	mean_total_noise = float(xp.mean(total_noise))
	if mean_val > 0:
		result = result / mean_val
	else:
		mean_data = float(xp.mean(data_xp))
		print(f"Warning: mean of result is {mean_val:.3e}, cannot normalize. Returning unnormalized result.")
		print(f"Mean of input data was {mean_data:.3e}.")
		print(f"Mean of total noise was {mean_total_noise:.3e}.")
		print(f"Mean of intermediate result was {mean_result_int:.3e}.")
		print("Means after pixel functions:", result_means)
		print(noise_functions, noise_params, noise_rel_std)
		print(pixel_functions, pixel_params)
		result = xp.ones_like(result)

	return result
