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

WAVELENGTH: float = 4.78e-6
ANGULAR_PIXEL_SCALE: float = 27 * MAS_TO_RADIANS
N_PIX: int = 160

# =============================================================================
# Output directory (root for all outputs of this experiment)
# =============================================================================

OUTPUT_BASE_DIR: str = "/lustre/fsn1/projects/rech/nab/udl61tt/V1H/naco_m_band_02"


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
	"n_batches": 128,
	"n_ex_per_batch": 1024,
	"n_workers": 40,
	"parallel_mode": "joblib",
	"seed_base": RNG_SEED,
}


# =============================================================================
# GalSim testing dataset configuration
# =============================================================================

GALSIM_TEST_CONFIG: dict[str, object] = {
	"output_dir": f"{DATASET_GEN_CONFIG['output_dir']}/galsim_test",
	"seed": RNG_SEED if RNG_SEED is not None else 12345,
	"normalize_scene_mean": True,
	"write_tfrecords": True,
	"tfrecord_name": "batch_0000.tfrecord",
	"n_plot_examples": 20,
	"n_scenes": 10,
	"n_psfs": 11,
	"n_noise_levels": 11,
	"point_source_sigma_pix": 0.1,
	"diam_m": VLT_PRIMARY_DIAMETER_M,
	"central_obscuration_m": VLT_CENTRAL_OBSCURATION_DIAMETER_M,
	"noise_sigma_min": 0.,
	"noise_sigma_max": 50.0,
	"image_config": {
		"primary_sersic_n_range": [0.8, 4.5],
		"primary_half_light_radius_arcsec_range": [0.08, 0.28],
		"primary_flux_range": [2.5e4, 7.5e4],
		"primary_axis_ratio_range": [0.45, 0.95],
		"primary_offset_arcsec_range": [-0.18, 0.18],
		"secondary_disk_probability": 0.9,
		"secondary_half_light_radius_arcsec_range": [0.12, 0.40],
		"secondary_flux_range": [0.7e4, 3.5e4],
		"secondary_axis_ratio_range": [0.35, 0.90],
		"secondary_relative_offset_arcsec_range": [-0.20, 0.20],
		"companion_probability": 0.55,
		"companion_sersic_n_range": [0.8, 2.5],
		"companion_half_light_radius_arcsec_range": [0.03, 0.10],
		"companion_flux_range": [0.2e4, 1.0e4],
		"companion_axis_ratio_range": [0.5, 1.0],
		"companion_offset_arcsec_range": [-0.90, 0.90],
		"n_point_sources_range": [1, 4],
		"point_source_flux_range": [0.25e4, 2.2e4],
		"point_source_offset_arcsec_range": [-1.15, 1.15],
	},
	"psf_config": {
		"aberration_amplitude_range": [0.0, 0.05],
		"aberration_scales": {
			"defocus": 1.00,
			"astig1": 0.70,
			"astig2": 0.70,
			"coma1": 0.45,
			"coma2": 0.45,
			"trefoil1": 0.30,
			"trefoil2": 0.30,
			"spher": 0.25,
		},
		"oversampling": 1.5,
		"pad_factor": 1.5,
	},
	"slurm_cpus_per_task": 4,
	"slurm_time_limit": "02:00:00",
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
	"n_epochs": 3,
	"n_steps_per_epoch": None,
	"lr_0": 1e-4,
	"lr_decay": 1.5,
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
	"n_epochs": 3,
	"n_steps_per_epoch": None,
	"lr_0": 1e-4,
	"lr_decay": 1.50,
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
	"inner_activation_function": "softplus",
	"output_activation_function": "linear",
	"normalize_output_sum": True,
	"normalize_with_first": True,
	"normalize_first_only": False,
	"normalize_by_mean": True,
	"loss_mode": "r2",
	"n_epochs": 5,
	"n_steps_per_epoch": None,
	"lr_0": 1e-4,
	"lr_decay": 2.50,
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
	"inner_activation_function": "softplus",
	"output_activation_function": "linear",
	"normalize_output_sum": False,
	"normalize_with_first": False,
	"normalize_first_only": False,
	"normalize_by_mean": False,
	"loss_mode": "nll",
	"n_epochs": 3,
	"n_steps_per_epoch": None,
	"lr_0": 1e-4,
	"lr_decay": 1.5,
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
	"n_epochs": 3,
	"n_steps_per_epoch": None,
	"lr_0": 1e-5,
	"lr_decay": 1.5,
	"jit_compile": False,
	"reconstruction_crop": 16,
	"train_image_head": True,
	"train_noise_head": False,
	"train_psf_mean_head": False,
	"train_psf_unc_head": False,
	"head_model_label": "best_model",
	"initial_eval_train_steps": 8,
	"initial_eval_val_steps": 8,
	"run_name": "joint_pinn_fourhead",
}


# =============================================================================
# Testing-time evaluation on val and GalSim datasets
# =============================================================================

TEST_ON_GALSIM_CONFIG: dict[str, object] = {
	"algorithm": "joint_pinn",
	"run_name": JOINT_PINN_CONFIG["run_name"],
	"model_label": "best_model",
	"first_batch_only": True,
	"eval_batch_size": DATASET_LOAD_CONFIG["val_batch_size"],
	"plot_examples": 100,
	"plot_dpi": 150,
	"output_dir": f"{OUTPUT_BASE_DIR}/{JOINT_PINN_CONFIG['run_name']}/test_on_galsim",
	"eval_crop_border": 16,
	"slurm_cpus_per_task": 8,
	"slurm_time_limit": "04:00:00",
}


RICHARDSON_LUCY_CONFIG: dict[str, object] = {
	"num_iter": 30,
	"psf_source": "truth",
	"frame_index": 0,
	"clip": False,
	"filter_epsilon": None,
}


WIENER_CONFIG: dict[str, object] = {
	"psf_source": "truth",
	"frame_index": 0,
}


# =============================================================================
# SLURM configuration
# =============================================================================

SLURM_CONFIG: dict[str, object] = {
	"gpu_type": "h100",
	"account": "nab",
	"cpu_account": "nab@cpu",
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
