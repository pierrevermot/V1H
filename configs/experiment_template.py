"""Experiment configuration template.

Copy this file and modify the parameters at the top to create a new experiment.
The three most commonly changed parameters are at the very top:
  - wavelength
  - angular_pixel_scale
  - n_pix
"""

from __future__ import annotations

import numpy as np


# =============================================================================
# >>> QUICK-EDIT: the three parameters you change most often <<<
# =============================================================================

MAS_TO_RADIANS: float = 1e-3 / 206265

WAVELENGTH: float = 2.18e-6
ANGULAR_PIXEL_SCALE: float = 13 * MAS_TO_RADIANS
N_PIX: int = 160

# =============================================================================
# Output directory (root for all outputs of this experiment)
# =============================================================================

OUTPUT_BASE_DIR: str = "/lustre/fsn1/projects/rech/nab/udl61tt/V1G/experiment_01"


# =============================================================================
# AO instrument configuration (VLT)
# =============================================================================

INSTRUMENT_CONFIG: dict[str, object] = {
	"name": "vlt",
	"n_pix": N_PIX,
	"wavelength": WAVELENGTH,
	"angular_pixel_scale": ANGULAR_PIXEL_SCALE,
	"max_high_res_pixel_scale": 2e-3,
	"angles": np.arange(360),
	"zernike_n_rad": 6,
	"zernike_n_min": 1,
	"n_frames": 1,
	"use_cupy": False,
}


# =============================================================================
# Random phase generation configuration
# =============================================================================

RNG_SEED: int | None = 1234

PHASE_CONFIG: dict[str, object] = {
	"rng": np.random.default_rng(RNG_SEED) if RNG_SEED is not None else None,
	"USE_CUPY": True,
	# draw_n_se_screens
	"N_SE_SCREENS_MIN": 1,
	"N_SE_SCREENS_MAX": 100,
	# draw_powerlaw_params
	"EXPONENT_LF_MIN": 2.0,
	"EXPONENT_LF_MAX": 5.0,
	"EXPONENT_HF_MIN": 2.0,
	"EXPONENT_HF_MAX": 5.0,
	"EXPONENT_CUTOFF_MIN": 0.1,
	"EXPONENT_CUTOFF_MAX": 2.0,
	"EXPONENT_RMS_LF_MIN": 0.0,
	"EXPONENT_RMS_LF_MAX": 1.0,
	"EXPONENT_RMS_HF_MIN": 0.0,
	"EXPONENT_RMS_HF_MAX": 1.0,
	# draw_rms_std (zernike)
	"ZER_RMS_MIN": 0.0,
	"ZER_RMS_MAX": 1.0,
	"ZER_STD_MIN": 0.0,
	"ZER_STD_MAX": 1.0,
	# draw_lwe_coeffs
	"LWE_RMS_PISTON_MIN": 0.0,
	"LWE_RMS_TIPTILT_MIN": 0.0,
	"LWE_STD_PISTON_MIN": 0.0,
	"LWE_STD_TIPTILT_MIN": 0.0,
	"LWE_RMS_PISTON_MAX": 1.0,
	"LWE_RMS_TIPTILT_MAX": 1.0,
	"LWE_STD_PISTON_MAX": 1.0,
	"LWE_STD_TIPTILT_MAX": 1.0,
	# draw_component_flags
	"PROBA_POWERLAW": 0.5,
	"PROBA_ZERNIKE": 0.5,
	"PROBA_LWE": 0.5,
	# draw_relative_amplitudes
	"RELATIVE_WEIGHT_POWERLAW": 1.0,
	"RELATIVE_WEIGHT_ZERNIKE": 1.0,
	"RELATIVE_WEIGHT_LWE": 1.0,
	# draw_rms_uniform
	"FINAL_STREHL_MIN": 0.5,
	"FINAL_STREHL_MAX": 1.0,
}


# =============================================================================
# Random sky generation configuration
# =============================================================================

SKY_CONFIG: dict[str, object] = {
	"rng": np.random.default_rng(RNG_SEED) if RNG_SEED is not None else None,
	# draw_n_objects
	"n_objects_min": 1,
	"n_objects_max": 5,
	# draw_nebula_params
	"nebula_exponent_min": 1.5,
	"nebula_exponent_max": 5.0,
	"nebula_percentile_min": 50.0,
	"nebula_percentile_max": 99.0,
	# draw_point_sources_params
	"point_sources_n_min": 1,
	"point_sources_n_max": 50,
	"point_sources_exponent_min": 1.5,
	"point_sources_exponent_max": 3.5,
	# draw_sharp_edges_params
	"sharp_edges_exponent_lf_min": 1.5,
	"sharp_edges_exponent_lf_max": 5.0,
	"sharp_edges_percentile_lf_min": 50.0,
	"sharp_edges_percentile_lf_max": 99.0,
	"sharp_edges_exponent_hf_min": 1.5,
	"sharp_edges_exponent_hf_max": 5.0,
	"sharp_edges_vmin_hf_min": 0.0,
	"sharp_edges_vmin_hf_max": 1.0,
}


# =============================================================================
# Random noise generation configuration
# =============================================================================

RANDOM_NOISE_CONFIG: dict[str, object] = {
	"rng": np.random.default_rng(RNG_SEED) if RNG_SEED is not None else None,
	# draw_noise_rel_std
	"noise_rel_x_min": 0.0,
	"noise_rel_x_max": 1.0,
	# draw_peak_snr
	"peak_snr_min": 1.0,
	"peak_snr_max": 1e4,
	# draw_point_source_params
	"point_sources_n_min": 0,
	"point_sources_n_max_multiplier": 2,
	"point_sources_flux_min": 0.0,
	"point_sources_flux_max": 1.0,
	# draw_pixels_to_zero_params
	"zero_pixels_n_min": 0,
	"zero_pixels_n_max": 5,
	"zero_pixels_prob_same_each_frame": 0.5,
}


# =============================================================================
# Dataset generation configuration
# =============================================================================

VLT_PRIMARY_DIAMETER_M: float = 8.0
VLT_CENTRAL_OBSCURATION_DIAMETER_M: float = 1.116

DATASET_GEN_CONFIG: dict[str, object] = {
	"output_dir": f"{OUTPUT_BASE_DIR}/dataset",
	"n_batches": 1024,
	"n_ex_per_batch": 1024,
	"n_workers": 40,
	"parallel_mode": "joblib",
	"seed_base": RNG_SEED,
}


# =============================================================================
# Dataset loading configuration
# =============================================================================

DATASET_LOAD_CONFIG: dict[str, object] = {
	"data_dir": f"{OUTPUT_BASE_DIR}/dataset",
	"batch_size": 64,
	"val_batch_size": 1024,
	"shuffle": True,
	"repeat": False,
	"seed": None,
	"channels_last": True,
	"half_n_pix_crop": 16,
	"norm_psf": "npix2",
	"norm_noise": None,
	"num_parallel_calls": None,
	"prefetch": True,
}


# =============================================================================
# Loss configuration (shared defaults, overridden per head where needed)
# =============================================================================

LOSS_CONFIG: dict[str, object] = {
	"log_sigma": False,
	"log_min": -6.0,
	"log_max": 20.0,
	"sigma2_eps": 1e-30,
	"charb_eps": 1e-4,
	"half_n_pix_crop": 16,
}


# =============================================================================
# Image head training configuration (Step 2a)
# =============================================================================

IMAGE_HEAD_CONFIG: dict[str, object] = {
	"model_name": "unet",
	"layers_per_block": 2,
	"base_filters": 32,
	"normalization": "none",
	"group_norm_groups": 8,
	"weight_decay": 1e-5,
	"inner_activation_function": "relu",
	"output_activation_function": "linear",
	"loss_mode": "nll",
	"n_epochs": 100,
	"n_steps_per_epoch": 20000,
	"lr_0": 5e-4,
	"lr_decay": 10.0,
	"run_name": "image_only",
}


# =============================================================================
# Noise head training configuration (Step 2b) — formerly "residual head"
# =============================================================================

NOISE_HEAD_CONFIG: dict[str, object] = {
	"model_name": "unet",
	"layers_per_block": 2,
	"base_filters": 32,
	"normalization": "none",
	"group_norm_groups": 8,
	"weight_decay": 1e-5,
	"inner_activation_function": "relu",
	"output_activation_function": "linear",
	"loss_mode": "nll",
	"n_epochs": 100,
	"n_steps_per_epoch": 20000,
	"lr_0": 5e-4,
	"lr_decay": 10.0,
	"run_name": "noise_only",
}


# =============================================================================
# PSF head training configuration (Step 2c)
# =============================================================================

PSF_HEAD_CONFIG: dict[str, object] = {
	"model_name": "gpkh",
	"layers_per_block": 2,
	"base_filters": 32,
	"latent_dim": 512,
	"normalization": "none",
	"group_norm_groups": 8,
	"weight_decay": 0,
	"inner_activation_function": "relu",
	"output_activation_function": "linear",
	"normalize_output_sum": True,
	"normalize_with_first": True,
	"normalize_first_only": False,
	"normalize_by_mean": False,
	"loss_mode": "r2",
	"n_epochs": 100,
	"n_steps_per_epoch": 20000,
	"lr_0": 5e-4,
	"lr_decay": 10.0,
	"run_name": "psf_only",
}


# =============================================================================
# PSF uncertainty head training configuration (Step 3)
# =============================================================================

PSF_UNC_CONFIG: dict[str, object] = {
	"model_name": "gpkh",
	"layers_per_block": 2,
	"base_filters": 32,
	"latent_dim": 512,
	"normalization": "none",
	"group_norm_groups": 8,
	"weight_decay": 0,
	"inner_activation_function": "relu",
	"output_activation_function": "linear",
	"normalize_output_sum": True,
	"normalize_with_first": True,
	"normalize_first_only": False,
	"normalize_by_mean": False,
	"loss_mode": "nll",
	"n_epochs": 100,
	"n_steps_per_epoch": 20000,
	"lr_0": 5e-4,
	"lr_decay": 10.0,
	"run_name": "psf_uncertainty_stage2",
	"source_psf_model_label": "best_model",
}


# =============================================================================
# Joint PINN four-head training configuration (Step 4)
# =============================================================================

JOINT_PINN_CONFIG: dict[str, object] = {
	"pinn_weight": 1.0,
	"im_weight": 1.0,
	"psf_weight": 1.0,
	"noise_weight": 1.0,
	"n_epochs": 100,
	"n_steps_per_epoch": 20000,
	"lr_0": 5e-4,
	"lr_decay": 10.0,
	"jit_compile": False,
	"reconstruction_crop": 16,
	"train_image_head": True,
	"train_noise_head": True,
	"train_psf_mean_head": True,
	"train_psf_unc_head": True,
	"head_model_label": "best_model",
	"initial_eval_train_steps": 8,
	"initial_eval_val_steps": 8,
	"run_name": "joint_pinn_fourhead",
}


# =============================================================================
# SLURM configuration
# =============================================================================

SLURM_CONFIG: dict[str, object] = {
	"gpu_type": "h100",
	"account": "nab",
	"cpu_account": "nab@cpu",
	# Optional common GPU account override for all GPU types.
	# If omitted, per-type defaults below are used.
	"time_limit": "100:00:00",
	"cpus_per_task": 24,
	"exclude_nodes": "",
	# Dataset generation (CPU job)
	"dataset_n_array_jobs": 10,
	"dataset_cpus_per_task": 40,
	"dataset_time_limit": "10:00:00",
	# Optional GPU account overrides per GPU type
	"v100_account": "nab@v100",
	"h100_account": "nab@h100",
}
