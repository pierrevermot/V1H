"""Random parameter generators for phase screens."""

from __future__ import annotations

import numpy as np

from utils.random_compat import rng_random as _rng_random


def draw_random_phase_parameters(
	ao_instru,
	*,
	rng=None,
	n_frames: int = 32,
	# draw_n_se_screens
	N_SE_SCREENS_MIN: int = 1,
	N_SE_SCREENS_MAX: int = 100,
	# draw_powerlaw_params
	EXPONENT_LF_MIN: float = 2.0,
	EXPONENT_LF_MAX: float = 5.0,
	EXPONENT_HF_MIN: float = 2.0,
	EXPONENT_HF_MAX: float = 5.0,
	EXPONENT_CUTOFF_MIN: float = 0.1,
	EXPONENT_CUTOFF_MAX: float = 2.0,
	EXPONENT_RMS_LF_MIN: float = 0.0,
	EXPONENT_RMS_LF_MAX: float = 1.0,
	EXPONENT_RMS_HF_MIN: float = 0.0,
	EXPONENT_RMS_HF_MAX: float = 1.0,
	# draw_rms_std (zernike)
	ZER_RMS_MIN: float = 0.0,
	ZER_RMS_MAX: float = 1.0,
	ZER_STD_MIN: float = 0.0,
	ZER_STD_MAX: float = 1.0,
	# draw_lwe_coeffs
	LWE_RMS_PISTON_MIN: float = 0.0,
	LWE_RMS_TIPTILT_MIN: float = 0.0,
	LWE_STD_PISTON_MIN: float = 0.0,
	LWE_STD_TIPTILT_MIN: float = 0.0,
	LWE_RMS_PISTON_MAX: float = 1.0,
	LWE_RMS_TIPTILT_MAX: float = 1.0,
	LWE_STD_PISTON_MAX: float = 1.0,
	LWE_STD_TIPTILT_MAX: float = 1.0,
	# draw_component_flags
	PROBA_POWERLAW: float = 0.5,
	PROBA_ZERNIKE: float = 0.5,
	PROBA_LWE: float = 0.5,
	# draw_relative_amplitudes
	RELATIVE_WEIGHT_POWERLAW: float = 1.0,
	RELATIVE_WEIGHT_ZERNIKE: float = 1.0,
	RELATIVE_WEIGHT_LWE: float = 1.0,
	# draw_rms_uniform
	FINAL_STREHL_MIN: float = 0.0,
	FINAL_STREHL_MAX: float = 1.0,
):
	"""Wrapper to draw all random phase parameters and coefficients."""
	rotated_index = draw_rotated_index(
		ao_instru,
		rng=rng,
	)
	
	n_screens_per_frame = draw_n_se_screens(
		N_SE_SCREENS_MIN,
		N_SE_SCREENS_MAX,
		rng=rng,
	)

	n_screens = n_frames * n_screens_per_frame

	powerlaw_on, zernike_on, lwe_on = draw_component_flags(
		proba_powerlaw=PROBA_POWERLAW,
		proba_zernike=PROBA_ZERNIKE,
		proba_lwe=PROBA_LWE,
		rng=rng,
	)

	exponent_lf, exponent_hf, cutoff, rms_lf, rms_hf = draw_powerlaw_params(
		exponent_lf_min=EXPONENT_LF_MIN,
		exponent_lf_max=EXPONENT_LF_MAX,
		exponent_hf_min=EXPONENT_HF_MIN,
		exponent_hf_max=EXPONENT_HF_MAX,
		cutoff_min=EXPONENT_CUTOFF_MIN,
		cutoff_max=EXPONENT_CUTOFF_MAX,
		rms_lf_min=EXPONENT_RMS_LF_MIN,
		rms_lf_max=EXPONENT_RMS_LF_MAX,
		rms_hf_min=EXPONENT_RMS_HF_MIN,
		rms_hf_max=EXPONENT_RMS_HF_MAX,
		rng=rng,
	)

	rms_zer, std_zer = draw_rms_std(
		rms_min=ZER_RMS_MIN,
		rms_max=ZER_RMS_MAX,
		std_min=ZER_STD_MIN,
		std_max=ZER_STD_MAX,
		rng=rng,
	)
	
	zernike_coeffs = draw_zernike_coeffs(
		ao_instru,
		n_screens=n_screens,
		rms=rms_zer,
		std=std_zer,
		rng=rng,
	)

	rms_piston = draw_rms_uniform(
		rms_min=LWE_RMS_PISTON_MIN,
		rms_max=LWE_RMS_PISTON_MAX,
		rng=rng,
	)
	rms_tiptilt = draw_rms_uniform(
		rms_min=LWE_RMS_TIPTILT_MIN,
		rms_max=LWE_RMS_TIPTILT_MAX,
		rng=rng,
	)
	std_piston = draw_rms_uniform(
		rms_min=LWE_STD_PISTON_MIN,
		rms_max=LWE_STD_PISTON_MAX,
		rng=rng,
	)
	std_tiptilt = draw_rms_uniform(
		rms_min=LWE_STD_TIPTILT_MIN,
		rms_max=LWE_STD_TIPTILT_MAX,
		rng=rng,
	)

	lwe_coeffs = draw_lwe_coeffs(
		ao_instru,
		n_screens=n_screens,
		rms_piston=rms_piston,
		rms_tiptilt=rms_tiptilt,
		std_piston=std_piston,
		std_tiptilt=std_tiptilt,
		rng=rng,
	)

	relative_amplitudes = draw_relative_amplitudes(
		relative_weight_powerlaw=RELATIVE_WEIGHT_POWERLAW,
		relative_weight_zernike=RELATIVE_WEIGHT_ZERNIKE,
		relative_weight_lwe=RELATIVE_WEIGHT_LWE,
		rng=rng,
		use_cupy=getattr(ao_instru, "use_cupy", False),
	)

	final_strehl = draw_strehl_uniform(
		strehl_min=FINAL_STREHL_MIN,
		strehl_max=FINAL_STREHL_MAX,
		rng=rng,
	)

	return {
		"rotated_index": rotated_index,
		"n_screens": n_screens,
		"component_flags": (powerlaw_on, zernike_on, lwe_on),
		"powerlaw_params": (exponent_lf, exponent_hf, cutoff, rms_lf, rms_hf),
		"zernike_rms_std": (rms_zer, std_zer),
		"zernike_coeffs": zernike_coeffs,
		"lwe_rms_std": (rms_piston, rms_tiptilt, std_piston, std_tiptilt),
		"lwe_coeffs": lwe_coeffs,
		"relative_amplitudes": relative_amplitudes,
		"final_strehl": final_strehl,
	}



def draw_rotated_index(ao_instru, *, rng=None) -> int:
	"""
	Draw a random rotation index from an AO_instrument's rotated pupils.

	Parameters
	----------
	ao_instru : AO_instrument
		AO_instrument object providing ``rotated_pupils`` or ``angles``.
	rng : np.random.Generator or xp.random.RandomState, optional
		Optional random number generator. If None, uses numpy RNG.

	Returns
	-------
	int
		Random rotation index in [0, n_rotations - 1].
	"""
	n_rotations = None
	if hasattr(ao_instru, "rotated_pupils") and ao_instru.rotated_pupils is not None:
		n_rotations = int(getattr(ao_instru.rotated_pupils, "shape", [0])[0])
	if (n_rotations is None or n_rotations <= 0) and hasattr(ao_instru, "angles"):
		try:
			n_rotations = int(len(ao_instru.angles))
		except Exception:
			n_rotations = None
	if n_rotations is None or n_rotations <= 0:
		raise ValueError("AO_instrument has no rotated pupils or angles to sample from")

	if rng is None:
		return int(np.random.randint(0, n_rotations))
	if hasattr(rng, "integers"):
		return int(rng.integers(0, n_rotations))
	return int(rng.randint(0, n_rotations))


def draw_n_se_screens(
	n_min: int,
	n_max: int,
	*,
	rng=None,
):
	"""
	Draw a random number of short-exposure screens (log-uniform).

	Parameters
	----------
	n_min : int
		Minimum number of screens (inclusive, must be >= 1).
	n_max : int
		Maximum number of screens (inclusive, must be >= n_min).
	rng : np.random.Generator or xp.random.RandomState, optional
		Optional random number generator. If None, uses the backend default RNG.
	use_cupy : bool, optional
		If True, use CuPy backend. Default is False.

	Returns
	-------
	int
		Randomly drawn integer between n_min and n_max (log-uniform).
	"""
	if n_min < 1 or n_max < 1:
		raise ValueError("n_min and n_max must be >= 1")
	if n_max < n_min:
		raise ValueError("n_max must be >= n_min")

	log_min = np.log(float(n_min))
	log_max = np.log(float(n_max))
	if rng is None:
		u = np.random.random()
	else:
		u = float(_rng_random(rng))
	val = np.exp(log_min + (log_max - log_min) * u)
	val_int = int(np.clip(np.rint(val), n_min, n_max))
	return val_int


def draw_powerlaw_params(
	*,
	exponent_lf_min: float,
	exponent_lf_max: float,
	exponent_hf_min: float,
	exponent_hf_max: float,
	cutoff_min: float,
	cutoff_max: float,
	rms_lf_min: float,
	rms_lf_max: float,
	rms_hf_min: float,
	rms_hf_max: float,
	rng=None,
):
	"""
	Draw random parameters for powerlaw phase generation (uniform).

	Returns
	-------
	tuple
		(exponent_lf, exponent_hf, cutoff, rms_lf, rms_hf)
	"""
	def _uniform(low: float, high: float):
		if high < low:
			raise ValueError("max must be >= min")
		if rng is None:
			return low + (high - low) * np.random.random()
		return low + (high - low) * float(_rng_random(rng))

	exponent_lf = _uniform(exponent_lf_min, exponent_lf_max)
	exponent_hf = _uniform(exponent_hf_min, exponent_hf_max)
	cutoff = _uniform(cutoff_min, cutoff_max)
	rms_lf = _uniform(rms_lf_min, rms_lf_max)
	rms_hf = _uniform(rms_hf_min, rms_hf_max)

	return exponent_lf, exponent_hf, cutoff, rms_lf, rms_hf


def draw_rms_std(
	*,
	rms_min: float,
	rms_max: float,
	std_min: float,
	std_max: float,
	rng=None,
):
	"""
	Draw random RMS and STD values (uniform).

	Returns
	-------
	tuple
		(rms, std)
	"""
	def _uniform(low: float, high: float):
		if high < low:
			raise ValueError("max must be >= min")
		if rng is None:
			return low + (high - low) * np.random.random()
		return low + (high - low) * float(_rng_random(rng))

	rms = _uniform(rms_min, rms_max)
	std = _uniform(std_min, std_max)

	return rms, std


def draw_zernike_coeffs(
	ao_instru,
	*,
	n_screens: int,
	rms: float,
	std: float,
	rng=None,
):
	"""
	Draw Zernike coefficient arrays for multiple phase screens.

	Parameters
	----------
	ao_instru : AO_instrument
		AO_instrument object providing ``zernike_modes`` and backend (xp/cp).
	n_screens : int
		Number of phase screens.
	rms : float
		Standard deviation for the base coefficients (coeffs_0).
	std : float
		Standard deviation for per-screen deviations from coeffs_0.
	rng : np.random.Generator or xp.random.RandomState, optional
		Optional random number generator. If None, uses the backend default RNG.

	Returns
	-------
	xp.ndarray
		Array of shape (n_screens, n_modes) of Zernike coefficients.
	"""
	if n_screens < 1:
		raise ValueError("n_screens must be >= 1")

	modes = getattr(ao_instru, "zernike_modes", None)
	if modes is None:
		raise ValueError("AO_instrument has no zernike_modes; compute them first")
	n_modes = int(modes.shape[0])

	use_cupy = getattr(ao_instru, "use_cupy", False)
	if use_cupy:
		import cupy as cp

	if rng is None:
		coeffs_0 = np.random.standard_normal(n_modes) * rms
		if n_screens == 1:
			return cp.asarray(coeffs_0[None, :]) if use_cupy else coeffs_0[None, :]
		noise = np.random.standard_normal((n_screens, n_modes)) * std
	else:
		coeffs_0 = np.asarray(rng.standard_normal(n_modes)) * rms
		if n_screens == 1:
			return cp.asarray(coeffs_0[None, :]) if use_cupy else coeffs_0[None, :]
		noise = np.asarray(rng.standard_normal((n_screens, n_modes))) * std

	result = coeffs_0[None, :] + noise
	return cp.asarray(result) if use_cupy else np.asarray(result)


def draw_lwe_coeffs(
	ao_instru,
	*,
	n_screens: int,
	rms_piston: float,
	rms_tiptilt: float,
	std_piston: float,
	std_tiptilt: float,
	rng=None,
):
	"""
	Draw LWE coefficient arrays for multiple phase screens.

	Parameters
	----------
	ao_instru : AO_instrument
		AO_instrument object providing LWE modes and backend (xp/cp).
	n_screens : int
		Number of phase screens.
	rms_piston : float
		Standard deviation for the base piston coefficients (coeffs_0).
	rms_tiptilt : float
		Standard deviation for the base tip/tilt coefficients (coeffs_0).
	std_piston : float
		Standard deviation for per-screen piston deviations from coeffs_0.
	std_tiptilt : float
		Standard deviation for per-screen tip/tilt deviations from coeffs_0.
	rng : np.random.Generator or xp.random.RandomState, optional
		Optional random number generator. If None, uses the backend default RNG.

	Returns
	-------
	xp.ndarray
		Array of shape (n_screens, 3, n_islands) of LWE coefficients.
	"""
	if n_screens < 1:
		raise ValueError("n_screens must be >= 1")

	modes = getattr(ao_instru, "lwe_modes_rotated", None)
	if modes is not None and getattr(modes, "ndim", 0) == 5:
		n_islands = int(modes.shape[1])
	else:
		# Fallback to rotated_islands if available
		rotated_islands = getattr(ao_instru, "rotated_islands", None)
		if rotated_islands is not None:
			n_islands = int(rotated_islands.shape[0])
		else:
			raise ValueError("AO_instrument has no LWE modes; compute them first")

	use_cupy = getattr(ao_instru, "use_cupy", False)
	if use_cupy:
		import cupy as cp

	if rng is None:
		coeffs_0 = np.random.standard_normal((3, n_islands))
		coeffs_0[0] *= rms_piston
		coeffs_0[1:] *= rms_tiptilt
		if n_screens == 1:
			return cp.asarray(coeffs_0[None, :, :]) if use_cupy else coeffs_0[None, :, :]
		noise = np.random.standard_normal((n_screens, 3, n_islands))
		noise[:, 0, :] *= std_piston
		noise[:, 1:, :] *= std_tiptilt
	else:
		coeffs_0 = np.asarray(rng.standard_normal((3, n_islands)))
		coeffs_0[0] *= rms_piston
		coeffs_0[1:] *= rms_tiptilt
		if n_screens == 1:
			return cp.asarray(coeffs_0[None, :, :]) if use_cupy else coeffs_0[None, :, :]
		noise = np.asarray(rng.standard_normal((n_screens, 3, n_islands)))
		noise[:, 0, :] *= std_piston
		noise[:, 1:, :] *= std_tiptilt

	result = coeffs_0[None, :, :] + noise
	return cp.asarray(result) if use_cupy else np.asarray(result)


def draw_component_flags(
	*,
	proba_powerlaw: float,
	proba_zernike: float,
	proba_lwe: float,
	rng=None,
):
	"""
	Draw boolean flags for powerlaw, zernike, and LWE components.

	Ensures at least one flag is True.

	Returns
	-------
	tuple
		(powerlaw_on, zernike_on, lwe_on)
	"""
	if not (0.0 <= proba_powerlaw <= 1.0):
		raise ValueError("proba_powerlaw must be between 0 and 1")
	if not (0.0 <= proba_zernike <= 1.0):
		raise ValueError("proba_zernike must be between 0 and 1")
	if not (0.0 <= proba_lwe <= 1.0):
		raise ValueError("proba_lwe must be between 0 and 1")

	def _rand() -> float:
		if rng is None:
			return float(np.random.random())
		return float(_rng_random(rng))

	while True:
		powerlaw_on = _rand() < proba_powerlaw
		zernike_on = _rand() < proba_zernike
		lwe_on = _rand() < proba_lwe
		if powerlaw_on or zernike_on or lwe_on:
			return powerlaw_on, zernike_on, lwe_on


def draw_relative_amplitudes(
	*,
	relative_weight_powerlaw: float,
	relative_weight_zernike: float,
	relative_weight_lwe: float,
	rng=None,
	use_cupy: bool = False,
):
	"""
	Draw random relative amplitudes scaled by provided weights.

	Returns
	-------
	np.ndarray
		Array of shape (3,) with amplitudes for powerlaw, zernike, and LWE.
	"""
	weights = np.asarray(
		[relative_weight_powerlaw, relative_weight_zernike, relative_weight_lwe],
		dtype=float,
	)

	if rng is None:
		amps = np.random.random(3)
	else:
		amps = np.asarray(_rng_random(rng, size=(3,)))

	result = amps * weights
	if use_cupy:
		import cupy as cp
		return cp.asarray(result)
	return result


def draw_rms_uniform(
	*,
	rms_min: float,
	rms_max: float,
	rng=None,
):
	"""
	Draw a random RMS value (uniform).

	Returns
	-------
	float
		Random RMS between rms_min and rms_max.
	"""
	if rms_max < rms_min:
		raise ValueError("rms_max must be >= rms_min")
	if rng is None:
		return float(rms_min + (rms_max - rms_min) * np.random.random())
	return float(rms_min + (rms_max - rms_min) * float(_rng_random(rng)))

def draw_strehl_uniform(
	*,
	strehl_min: float,
	strehl_max: float,
	rng=None,
):
	"""
	Draw a random Strehl value (uniform).

	Returns
	-------
	float
		Random Strehl between strehl_min and strehl_max.
	"""
	if strehl_max < strehl_min:
		raise ValueError("strehl_max must be >= strehl_min")
	if rng is None:
		return float(strehl_min + (strehl_max - strehl_min) * np.random.random())
	return float(strehl_min + (strehl_max - strehl_min) * float(_rng_random(rng)))


