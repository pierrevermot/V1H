#!/usr/bin/env python3
"""
Generate a GalSim dataset with:
  - 10 different source scenes containing galaxies + point sources
  - 10 different AO-like PSFs
  - 100 clean observations from all scene/PSF pairs
  - 1000 noisy observations using 10 Gaussian noise levels per clean image

Output:
  <outdir>/dataset.npz
  <outdir>/metadata.json
    <outdir>/dataset.tfrecord    (with --write-tfrecords)
    <outdir>/example_stats.json  (with --write-tfrecords)

The saved arrays have shapes:
  scenes      : (10, 128, 128)
  psfs        : (10, 128, 128)
  clean_obs   : (10, 10, 128, 128)
  noisy_obs   : (10, 10, 10, 128, 128)
  noise_sigmas: (10,)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import galsim
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs.load_config import load_experiment_config
from utils.plot_helpers import _save_figure_png_and_pdf

# -----------------------------
# Global simulation parameters
# -----------------------------
N_SCENES = 10
N_PSFS = 10
N_NOISE = 10
N_PIX = 128
PIX_SCALE_ARCSEC = 27e-3  # 27 mas/pixel = 0.027 arcsec/pixel
POINT_SOURCE_SIGMA_PIX = 0.1
POINT_SOURCE_SIGMA_ARCSEC = POINT_SOURCE_SIGMA_PIX * PIX_SCALE_ARCSEC
LAM_NM = 4.78e3           # 4.78 microns = 4780 nm
DIAM_M = 8.0
CENTRAL_OBSCURATION_M = 1.116
OBSCURATION_FRAC = CENTRAL_OBSCURATION_M / DIAM_M

# Broad but usable range of additive Gaussian noise levels in counts/pixel.
NOISE_SIGMAS = np.geomspace(0.5, 24.0, N_NOISE).astype(float)
EXAMPLE_STAT_COMPONENTS = ("image", "obs", "psf", "noise", "ref_psf")
DEFAULT_TFRECORD_NAME = "batch_0000.tfrecord"
DEFAULT_GENERATION_LOG_NAME = "generation_parameter_log.json"
SCENE_CONFIG: dict[str, Any] = {
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
}
PSF_CONFIG: dict[str, Any] = {
    "aberration_amplitude_range": [0.004, 0.030],
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
}


def _merge_nested_dict(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merged[key] = _merge_nested_dict(dict(base[key]), value)
        else:
            merged[key] = value
    return merged


def _range_pair(config: dict[str, Any], key: str) -> tuple[float, float]:
    values = config[key]
    if len(values) != 2:
        raise ValueError(f"Expected {key} to contain exactly two values, got {values}")
    return float(values[0]), float(values[1])


def _get_required_config_section(cfg, name: str) -> dict[str, Any]:
    section = getattr(cfg, name, None)
    if section is None:
        raise ValueError(f"Config file is missing required section {name}")
    return dict(section)


def _configure_from_experiment_config(cfg) -> dict[str, Any]:
    global N_SCENES, N_PSFS, N_NOISE, N_PIX
    global PIX_SCALE_ARCSEC, POINT_SOURCE_SIGMA_PIX, POINT_SOURCE_SIGMA_ARCSEC
    global LAM_NM, DIAM_M, CENTRAL_OBSCURATION_M, OBSCURATION_FRAC, NOISE_SIGMAS
    global SCENE_CONFIG, PSF_CONFIG

    galsim_cfg = _get_required_config_section(cfg, "GALSIM_TEST_CONFIG")

    wavelength_m = float(getattr(cfg, "WAVELENGTH"))
    angular_pixel_scale_rad = float(getattr(cfg, "ANGULAR_PIXEL_SCALE"))

    N_SCENES = int(galsim_cfg.get("n_scenes", 10))
    N_PSFS = int(galsim_cfg.get("n_psfs", 10))
    N_NOISE = int(galsim_cfg.get("n_noise_levels", 10))
    N_PIX = int(getattr(cfg, "N_PIX"))
    PIX_SCALE_ARCSEC = angular_pixel_scale_rad * 206265.0
    POINT_SOURCE_SIGMA_PIX = float(galsim_cfg.get("point_source_sigma_pix", 0.1))
    POINT_SOURCE_SIGMA_ARCSEC = POINT_SOURCE_SIGMA_PIX * PIX_SCALE_ARCSEC
    LAM_NM = wavelength_m * 1e9
    DIAM_M = float(galsim_cfg.get("diam_m", 8.0))
    CENTRAL_OBSCURATION_M = float(galsim_cfg.get("central_obscuration_m", 1.116))
    OBSCURATION_FRAC = CENTRAL_OBSCURATION_M / DIAM_M
    SCENE_CONFIG = _merge_nested_dict(SCENE_CONFIG, dict(galsim_cfg.get("image_config", {})))
    PSF_CONFIG = _merge_nested_dict(PSF_CONFIG, dict(galsim_cfg.get("psf_config", {})))

    if "noise_sigmas" in galsim_cfg:
        noise_sigmas = np.asarray(galsim_cfg["noise_sigmas"], dtype=float)
    else:
        noise_sigma_min = float(galsim_cfg.get("noise_sigma_min", 0.5))
        noise_sigma_max = float(galsim_cfg.get("noise_sigma_max", 24.0))
        noise_sigmas = np.linspace(noise_sigma_min, noise_sigma_max, N_NOISE).astype(float)

    if noise_sigmas.ndim != 1 or len(noise_sigmas) != N_NOISE:
        raise ValueError(
            f"GALSIM_TEST_CONFIG noise specification must produce {N_NOISE} values, got shape {noise_sigmas.shape}"
        )
    NOISE_SIGMAS = noise_sigmas.astype(float)

    default_output_dir = Path(str(cfg.DATASET_GEN_CONFIG.get("output_dir", Path(cfg.OUTPUT_BASE_DIR) / "dataset"))) / "galsim_test"
    output_dir = Path(str(galsim_cfg.get("output_dir", default_output_dir))).expanduser()

    return {
        "output_dir": output_dir,
        "seed": int(galsim_cfg.get("seed", getattr(cfg, "RNG_SEED", 12345) or 12345)),
        "normalize_scene_mean": bool(galsim_cfg.get("normalize_scene_mean", True)),
        "normalize_scene_mean_half_n_pix_crop": int(dict(getattr(cfg, "DATASET_LOAD_CONFIG", {})).get("half_n_pix_crop", 0)),
        "write_tfrecords": bool(galsim_cfg.get("write_tfrecords", True)),
        "tfrecord_name": str(galsim_cfg.get("tfrecord_name", DEFAULT_TFRECORD_NAME)),
        "n_plot_examples": int(galsim_cfg.get("n_plot_examples", 20)),
    }


def _to_builtin(value: Any) -> Any:
    """Convert numpy scalars/arrays recursively into JSON-serializable builtins."""
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _import_tensorflow() -> Any:
    """Import TensorFlow only when TFRecord export is requested."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for TFRecord export. "
            "Install tensorflow or run without --write-tfrecords."
        ) from exc
    return tf


def _import_matplotlib_pyplot() -> Any:
    """Import matplotlib pyplot with a non-interactive backend."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for TFRecord preview plots. "
            "Install matplotlib or set --n-plot-examples 0."
        ) from exc
    return plt


def _import_matplotlib_colors() -> Any:
    """Import matplotlib colors for plot normalization."""
    try:
        from matplotlib import colors
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for TFRecord preview plots. "
            "Install matplotlib or set --n-plot-examples 0."
        ) from exc
    return colors


def _bytes_feature(tf: Any, value: bytes) -> Any:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(tf: Any, value: int) -> Any:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _compute_array_stats(array: np.ndarray) -> dict[str, float]:
    """Return a compact set of scalar statistics for an array."""
    values = np.asarray(array, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "sum": float(np.sum(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def serialize_example(
    tf: Any,
    image: np.ndarray,
    obs: np.ndarray,
    psf: np.ndarray,
    noise: np.ndarray,
    ref_psf: np.ndarray,
) -> bytes:
    """Serialize one example using the TFRecord schema expected by the training pipeline."""
    image = np.asarray(image, dtype=np.float32)
    obs = np.asarray(obs, dtype=np.float32)
    psf = np.asarray(psf, dtype=np.float32)
    noise = np.asarray(noise, dtype=np.float32)
    ref_psf = np.asarray(ref_psf, dtype=np.float32)

    feature = {
        "image": _bytes_feature(tf, tf.io.serialize_tensor(image).numpy()),
        "obs": _bytes_feature(tf, tf.io.serialize_tensor(obs).numpy()),
        "psf": _bytes_feature(tf, tf.io.serialize_tensor(psf).numpy()),
        "noise": _bytes_feature(tf, tf.io.serialize_tensor(noise).numpy()),
        "ref_psf": _bytes_feature(tf, tf.io.serialize_tensor(ref_psf).numpy()),
        "n_frames": _int64_feature(tf, int(obs.shape[0])),
        "n_pix": _int64_feature(tf, int(image.shape[-1])),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def combine_objects(objects: list[galsim.GSObject]) -> galsim.GSObject:
    """Combine a list of GSObjects into a single GSObject sum."""
    if not objects:
        raise ValueError("Need at least one GSObject to combine.")
    result = objects[0]
    for obj in objects[1:]:
        result = result + obj
    return result


def make_scene(scene_id: int, rng: np.random.Generator) -> tuple[galsim.GSObject, dict[str, Any]]:
    """Create one synthetic scene with galaxies + point sources."""
    primary_n_min, primary_n_max = _range_pair(SCENE_CONFIG, "primary_sersic_n_range")
    primary_hlr_min, primary_hlr_max = _range_pair(SCENE_CONFIG, "primary_half_light_radius_arcsec_range")
    primary_flux_min, primary_flux_max = _range_pair(SCENE_CONFIG, "primary_flux_range")
    primary_q_min, primary_q_max = _range_pair(SCENE_CONFIG, "primary_axis_ratio_range")
    primary_offset_min, primary_offset_max = _range_pair(SCENE_CONFIG, "primary_offset_arcsec_range")
    secondary_hlr_min, secondary_hlr_max = _range_pair(SCENE_CONFIG, "secondary_half_light_radius_arcsec_range")
    secondary_flux_min, secondary_flux_max = _range_pair(SCENE_CONFIG, "secondary_flux_range")
    secondary_q_min, secondary_q_max = _range_pair(SCENE_CONFIG, "secondary_axis_ratio_range")
    secondary_offset_min, secondary_offset_max = _range_pair(SCENE_CONFIG, "secondary_relative_offset_arcsec_range")
    companion_n_min, companion_n_max = _range_pair(SCENE_CONFIG, "companion_sersic_n_range")
    companion_hlr_min, companion_hlr_max = _range_pair(SCENE_CONFIG, "companion_half_light_radius_arcsec_range")
    companion_flux_min, companion_flux_max = _range_pair(SCENE_CONFIG, "companion_flux_range")
    companion_q_min, companion_q_max = _range_pair(SCENE_CONFIG, "companion_axis_ratio_range")
    companion_offset_min, companion_offset_max = _range_pair(SCENE_CONFIG, "companion_offset_arcsec_range")
    point_source_count_min, point_source_count_max = _range_pair(SCENE_CONFIG, "n_point_sources_range")
    point_source_flux_min, point_source_flux_max = _range_pair(SCENE_CONFIG, "point_source_flux_range")
    point_source_offset_min, point_source_offset_max = _range_pair(SCENE_CONFIG, "point_source_offset_arcsec_range")

    components: list[galsim.GSObject] = []
    metadata: dict[str, Any] = {
        "scene_id": scene_id,
        "galaxies": [],
        "point_sources": [],
    }

    # Primary galaxy: general Sersic profile.
    main = {
        "type": "Sersic",
        "n": rng.uniform(primary_n_min, primary_n_max),
        "half_light_radius_arcsec": rng.uniform(primary_hlr_min, primary_hlr_max),
        "flux": rng.uniform(primary_flux_min, primary_flux_max),
        "q": rng.uniform(primary_q_min, primary_q_max),
        "beta_deg": rng.uniform(0.0, 180.0),
        "dx_arcsec": rng.uniform(primary_offset_min, primary_offset_max),
        "dy_arcsec": rng.uniform(primary_offset_min, primary_offset_max),
    }
    main_obj = (
        galsim.Sersic(
            n=main["n"],
            half_light_radius=main["half_light_radius_arcsec"],
            flux=main["flux"],
        )
        .shear(q=main["q"], beta=main["beta_deg"] * galsim.degrees)
        .shift(main["dx_arcsec"], main["dy_arcsec"])
    )
    components.append(main_obj)
    metadata["galaxies"].append(main)

    # Secondary disk-like galaxy in most scenes.
    if rng.random() < float(SCENE_CONFIG["secondary_disk_probability"]):
        disk = {
            "type": "Exponential",
            "half_light_radius_arcsec": rng.uniform(secondary_hlr_min, secondary_hlr_max),
            "flux": rng.uniform(secondary_flux_min, secondary_flux_max),
            "q": rng.uniform(secondary_q_min, secondary_q_max),
            "beta_deg": rng.uniform(0.0, 180.0),
            "dx_arcsec": main["dx_arcsec"] + rng.uniform(secondary_offset_min, secondary_offset_max),
            "dy_arcsec": main["dy_arcsec"] + rng.uniform(secondary_offset_min, secondary_offset_max),
        }
        disk_obj = (
            galsim.Exponential(
                half_light_radius=disk["half_light_radius_arcsec"],
                flux=disk["flux"],
            )
            .shear(q=disk["q"], beta=disk["beta_deg"] * galsim.degrees)
            .shift(disk["dx_arcsec"], disk["dy_arcsec"])
        )
        components.append(disk_obj)
        metadata["galaxies"].append(disk)

    # Compact companion or clump in about half the scenes.
    if rng.random() < float(SCENE_CONFIG["companion_probability"]):
        comp = {
            "type": "Sersic",
            "n": rng.uniform(companion_n_min, companion_n_max),
            "half_light_radius_arcsec": rng.uniform(companion_hlr_min, companion_hlr_max),
            "flux": rng.uniform(companion_flux_min, companion_flux_max),
            "q": rng.uniform(companion_q_min, companion_q_max),
            "beta_deg": rng.uniform(0.0, 180.0),
            "dx_arcsec": rng.uniform(companion_offset_min, companion_offset_max),
            "dy_arcsec": rng.uniform(companion_offset_min, companion_offset_max),
        }
        comp_obj = (
            galsim.Sersic(
                n=comp["n"],
                half_light_radius=comp["half_light_radius_arcsec"],
                flux=comp["flux"],
            )
            .shear(q=comp["q"], beta=comp["beta_deg"] * galsim.degrees)
            .shift(comp["dx_arcsec"], comp["dy_arcsec"])
        )
        components.append(comp_obj)
        metadata["galaxies"].append(comp)

    # One to four point sources.
    n_stars = int(rng.integers(int(point_source_count_min), int(point_source_count_max) + 1))
    for star_id in range(n_stars):
        star = {
            "point_source_id": star_id,
            "type": "Gaussian",
            "flux": rng.uniform(point_source_flux_min, point_source_flux_max),
            "sigma_pix": POINT_SOURCE_SIGMA_PIX,
            "sigma_arcsec": POINT_SOURCE_SIGMA_ARCSEC,
            "dx_arcsec": rng.uniform(point_source_offset_min, point_source_offset_max),
            "dy_arcsec": rng.uniform(point_source_offset_min, point_source_offset_max),
        }
        star_obj = galsim.Gaussian(
            flux=star["flux"],
            sigma=POINT_SOURCE_SIGMA_ARCSEC,
        ).shift(star["dx_arcsec"], star["dy_arcsec"])
        components.append(star_obj)
        metadata["point_sources"].append(star)

    scene = combine_objects(components)
    return scene, metadata


def make_psf(psf_id: int, rng: np.random.Generator) -> tuple[galsim.GSObject, dict[str, Any]]:
    """
    Create one AO-like PSF.

    PSF 0 is a perfect Airy pattern.
    PSFs 1..9 are low-aberration OpticalPSFs with small Zernike coefficients.
    Coefficients are in waves, matching the GalSim OpticalPSF convention.
    """
    common = {
        "lam_nm": LAM_NM,
        "diam_m": DIAM_M,
        "central_obscuration_m": CENTRAL_OBSCURATION_M,
        "obscuration_fraction": OBSCURATION_FRAC,
        "scale_unit": "arcsec",
        "flux": 1.0,
    }

    if psf_id == 0:
        psf = galsim.Airy(
            lam=LAM_NM,
            diam=DIAM_M,
            obscuration=OBSCURATION_FRAC,
            scale_unit=galsim.arcsec,
            flux=1.0,
        )
        metadata = {
            "psf_id": psf_id,
            "type": "Airy",
            "aberration_sampling": "linear_template",
            "aberration_amplitude_scale": 0.0,
            "residual_wavefront_rms_waves": 0.0,
            **common,
        }
        return psf, metadata

    # Keep aberrations small to mimic high-Strehl, low-perturbation AO PSFs.
    # The residual-wavefront RMS increases linearly across the aberrated PSFs.
    amp_min, amp_max = _range_pair(PSF_CONFIG, "aberration_amplitude_range")
    amp = np.linspace(amp_min, amp_max, N_PSFS - 1)[psf_id - 1]
    aberration_scales = dict(PSF_CONFIG["aberration_scales"])
    mode_names = list(aberration_scales)
    mode_weights = np.asarray([float(aberration_scales[name]) for name in mode_names], dtype=np.float64)
    mode_signs = np.asarray([1.0 if index % 2 == 0 else -1.0 for index in range(len(mode_names))], dtype=np.float64)
    rms_norm = float(np.sqrt(np.mean(np.square(mode_weights))))
    if not np.isfinite(rms_norm) or rms_norm <= 0.0:
        rms_norm = 1.0
    coeff_vector = amp * mode_signs * mode_weights / rms_norm
    coeffs = {name: float(coeff_vector[index]) for index, name in enumerate(mode_names)}
    residual_wavefront_rms_waves = float(np.sqrt(np.mean(np.square(coeff_vector))))

    psf = galsim.OpticalPSF(
        lam=LAM_NM,
        diam=DIAM_M,
        obscuration=OBSCURATION_FRAC,
        scale_unit=galsim.arcsec,
        flux=1.0,
        oversampling=float(PSF_CONFIG["oversampling"]),
        pad_factor=float(PSF_CONFIG["pad_factor"]),
        **coeffs,
    )

    metadata = {
        "psf_id": psf_id,
        "type": "OpticalPSF",
        "aberration_sampling": "linear_template",
        "aberration_amplitude_scale": amp,
        "residual_wavefront_rms_waves": residual_wavefront_rms_waves,
        "aberration_mode_names": mode_names,
        "aberration_mode_template_weights": mode_weights.tolist(),
        "aberration_mode_template_signs": mode_signs.tolist(),
        **common,
        **coeffs,
    }
    return psf, metadata


def _build_generation_parameter_log(metadata: dict[str, Any]) -> dict[str, Any]:
    scene_metadata = list(metadata.get("scene_metadata", []))
    psf_metadata = list(metadata.get("psf_metadata", []))
    noise_sigmas = np.asarray(metadata.get("noise_model", {}).get("sigmas", NOISE_SIGMAS), dtype=np.float64)

    per_example: list[dict[str, Any]] = []
    for scene_id, _scene_meta in enumerate(scene_metadata):
        for psf_id, psf_meta in enumerate(psf_metadata):
            for noise_level_id, noise_sigma in enumerate(noise_sigmas):
                per_example.append(
                    {
                        "example_index": len(per_example),
                        "scene_id": int(scene_id),
                        "psf_id": int(psf_id),
                        "noise_level_id": int(noise_level_id),
                        "noise_sigma": float(noise_sigma),
                        "psf_type": str(psf_meta.get("type", "unknown")),
                        "psf_aberration_amplitude_scale": float(psf_meta.get("aberration_amplitude_scale", 0.0)),
                        "psf_residual_wavefront_rms_waves": float(psf_meta.get("residual_wavefront_rms_waves", 0.0)),
                    }
                )

    return {
        "path": DEFAULT_GENERATION_LOG_NAME,
        "sampling": {
            "noise_sigma": "linear",
            "psf_residual_wavefront": "linear_template",
        },
        "parameter_names": [
            "noise_sigma",
            "psf_aberration_amplitude_scale",
            "psf_residual_wavefront_rms_waves",
        ],
        "n_examples": len(per_example),
        "noise_levels": [float(value) for value in noise_sigmas.tolist()],
        "psf_levels": [
            {
                "psf_id": int(item.get("psf_id", index)),
                "psf_type": str(item.get("type", "unknown")),
                "psf_aberration_amplitude_scale": float(item.get("aberration_amplitude_scale", 0.0)),
                "psf_residual_wavefront_rms_waves": float(item.get("residual_wavefront_rms_waves", 0.0)),
            }
            for index, item in enumerate(psf_metadata)
        ],
        "per_example": per_example,
    }


def draw_stamp(obj: galsim.GSObject) -> galsim.Image:
    """Draw an object onto a fixed 128x128 image at 27 mas/pixel."""
    return obj.drawImage(nx=N_PIX, ny=N_PIX, scale=PIX_SCALE_ARCSEC)


def _crop_2d_center(array: np.ndarray, half_n_pix_crop: int) -> np.ndarray:
    if half_n_pix_crop <= 0:
        return array
    c = int(half_n_pix_crop)
    if 2 * c >= min(array.shape[-2:]):
        raise ValueError(f"half_n_pix_crop={half_n_pix_crop} is too large for array shape {array.shape}")
    return array[c:-c, c:-c]


def _scale_scene_to_unit_mean(
    scene: galsim.GSObject,
    *,
    half_n_pix_crop: int,
) -> tuple[galsim.GSObject, float, float]:
    """Scale a scene so its rasterized cropped mean equals 1.0 before convolution."""
    scene_image = draw_stamp(scene)
    scene_array = np.asarray(scene_image.array, dtype=np.float64)
    scene_array = _crop_2d_center(scene_array, half_n_pix_crop)
    scene_mean = float(np.mean(scene_array))
    if not np.isfinite(scene_mean) or scene_mean <= 0.0:
        raise ValueError(f"Cannot normalize scene with non-positive drawn mean: {scene_mean}")
    scale_factor = 1.0 / scene_mean
    return scene.withScaledFlux(scale_factor), scene_mean, scale_factor


def build_dataset(
    seed: int,
    *,
    normalize_scene_mean: bool = True,
    normalize_scene_mean_half_n_pix_crop: int = 0,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    rng = np.random.default_rng(seed)

    scenes: list[galsim.GSObject] = []
    psfs: list[galsim.GSObject] = []
    scene_meta: list[dict[str, Any]] = []
    psf_meta: list[dict[str, Any]] = []

    for i in range(N_SCENES):
        scene, meta = make_scene(i, rng)
        if normalize_scene_mean:
            scene, scene_mean_before, scene_scale_factor = _scale_scene_to_unit_mean(
                scene,
                half_n_pix_crop=normalize_scene_mean_half_n_pix_crop,
            )
            meta = {
                **meta,
                "drawn_mean_before_normalization": scene_mean_before,
                "drawn_mean_after_normalization": 1.0,
                "flux_scale_applied": scene_scale_factor,
                "drawn_mean_half_n_pix_crop": int(normalize_scene_mean_half_n_pix_crop),
            }
        scenes.append(scene)
        scene_meta.append(meta)

    for j in range(N_PSFS):
        psf, meta = make_psf(j, rng)
        psfs.append(psf)
        psf_meta.append(meta)

    scenes_arr = np.empty((N_SCENES, N_PIX, N_PIX), dtype=np.float32)
    psfs_arr = np.empty((N_PSFS, N_PIX, N_PIX), dtype=np.float32)
    clean_arr = np.empty((N_SCENES, N_PSFS, N_PIX, N_PIX), dtype=np.float32)
    noisy_arr = np.empty((N_SCENES, N_PSFS, N_NOISE, N_PIX, N_PIX), dtype=np.float32)

    # Save rasterized source scenes and PSFs for reference.
    for i, scene in enumerate(scenes):
        scenes_arr[i] = draw_stamp(scene).array.astype(np.float32)

    for j, psf in enumerate(psfs):
        psfs_arr[j] = draw_stamp(psf).array.astype(np.float32)

    # Generate clean and noisy observations.
    for i, scene in enumerate(scenes):
        for j, psf in enumerate(psfs):
            clean_obj = galsim.Convolve([scene, psf])
            clean_img = draw_stamp(clean_obj)
            clean_arr[i, j] = clean_img.array.astype(np.float32)

            for k, sigma in enumerate(NOISE_SIGMAS):
                noisy_img = galsim.Image(clean_img.array.copy(), scale=PIX_SCALE_ARCSEC)
                noise_seed = seed + 1_000_000 + 10_000 * i + 100 * j + k
                noise_rng = galsim.BaseDeviate(noise_seed)
                noisy_img.addNoise(galsim.GaussianNoise(noise_rng, sigma=float(sigma)))
                noisy_arr[i, j, k] = noisy_img.array.astype(np.float32)

    arrays = {
        "scenes": scenes_arr,
        "psfs": psfs_arr,
        "clean_obs": clean_arr,
        "noisy_obs": noisy_arr,
        "noise_sigmas": NOISE_SIGMAS.astype(np.float32),
    }

    metadata = {
        "seed": seed,
        "grid_shape": {
            "n_scenes": N_SCENES,
            "n_psfs": N_PSFS,
            "n_noise_levels": N_NOISE,
            "n_clean_observations": N_SCENES * N_PSFS,
            "n_noisy_observations": N_SCENES * N_PSFS * N_NOISE,
        },
        "image_geometry": {
            "n_pix": N_PIX,
            "pix_scale_arcsec": PIX_SCALE_ARCSEC,
            "pix_scale_mas": PIX_SCALE_ARCSEC * 1e3,
            "field_of_view_arcsec": N_PIX * PIX_SCALE_ARCSEC,
        },
        "telescope": {
            "diameter_m": DIAM_M,
            "wavelength_micron": LAM_NM / 1e3,
            "wavelength_nm": LAM_NM,
            "central_obscuration_m": CENTRAL_OBSCURATION_M,
            "central_obscuration_fraction": OBSCURATION_FRAC,
        },
        "noise_model": {
            "type": "GaussianNoise",
            "sigmas": NOISE_SIGMAS,
        },
        "normalization": {
            "normalize_scene_mean": bool(normalize_scene_mean),
            "scene_drawn_mean_target": 1.0 if normalize_scene_mean else None,
            "scene_drawn_mean_half_n_pix_crop": int(normalize_scene_mean_half_n_pix_crop),
        },
        "scene_metadata": scene_meta,
        "psf_metadata": psf_meta,
    }

    return arrays, metadata


def build_tfrecord_examples(arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """Pack the grid into TFRecord examples.

    Each example uses one latent source scene, one AO PSF, and one noise level.
    The observation, PSF, and noise tensors therefore contain a single frame.
    """
    examples: list[dict[str, Any]] = []
    psf_cube = np.asarray(arrays["psfs"], dtype=np.float32)

    for scene_id in range(N_SCENES):
        image = np.asarray(arrays["scenes"][scene_id], dtype=np.float32)
        clean_cube = np.asarray(arrays["clean_obs"][scene_id], dtype=np.float32)
        for psf_id in range(N_PSFS):
            psf_frame = np.asarray(psf_cube[psf_id], dtype=np.float32)
            clean_frame = np.asarray(clean_cube[psf_id], dtype=np.float32)
            for noise_level_id, noise_sigma in enumerate(NOISE_SIGMAS):
                obs_frame = np.asarray(arrays["noisy_obs"][scene_id, psf_id, noise_level_id], dtype=np.float32)
                noise_frame = clean_frame - obs_frame
                examples.append(
                    {
                        "scene_id": scene_id,
                        "psf_id": psf_id,
                        "noise_level_id": noise_level_id,
                        "noise_sigma": float(noise_sigma),
                        "image": image,
                        "obs": obs_frame[np.newaxis, ...],
                        "psf": psf_frame[np.newaxis, ...],
                        "noise": noise_frame[np.newaxis, ...],
                        "ref_psf": psf_frame,
                    }
                )

    return examples


def _collapse_frame_cube(array: np.ndarray) -> np.ndarray:
    """Convert a frame cube to a single 2D summary image for plotting."""
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 3:
        if array.shape[0] == 1:
            return array[0]
        return np.mean(array, axis=0, dtype=np.float32)
    return array


def save_example_plots(
    examples: list[dict[str, Any]],
    outdir: Path,
    *,
    n_plot_examples: int,
    seed: int,
) -> dict[str, Any]:
    if n_plot_examples < 0:
        raise ValueError("n_plot_examples must be non-negative.")

    plots_dir = outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for old_plot in plots_dir.glob("example_*.png"):
        old_plot.unlink()

    n_to_plot = min(n_plot_examples, len(examples))
    frame_reduction = "single_frame" if examples and np.asarray(examples[0]["obs"]).shape[0] == 1 else "mean_over_frames"
    if n_to_plot == 0:
        return {
            "directory": "plots",
            "n_examples_plotted": 0,
            "frame_reduction": frame_reduction,
            "selection": "random_without_replacement",
            "selection_seed": seed,
        }

    plt = _import_matplotlib_pyplot()
    colors = _import_matplotlib_colors()
    rng = np.random.default_rng(seed)
    selected_indices = np.sort(rng.choice(len(examples), size=n_to_plot, replace=False))
    for plot_index, example_index in enumerate(selected_indices):
        example = examples[int(example_index)]
        image = np.asarray(example["image"], dtype=np.float32)
        obs = _collapse_frame_cube(example["obs"])
        psf = _collapse_frame_cube(example["psf"])
        noise = _collapse_frame_cube(example["noise"])

        figure, axes = plt.subplots(1, 4, figsize=(16, 4))
        panels = [
            (obs, "Observation", "magma", None),
            (image, "Image", "magma", None),
            (psf, "PSF", "magma", None),
            (noise, "Noise Map", "coolwarm", float(np.max(np.abs(noise))) or None),
        ]

        for axis, (panel_data, title, cmap, symmetric_limit) in zip(axes, panels):
            kwargs: dict[str, Any] = {"origin": "lower", "cmap": cmap}
            if symmetric_limit is not None:
                kwargs["vmin"] = -symmetric_limit
                kwargs["vmax"] = symmetric_limit
            else:
                clipped_panel = np.clip(panel_data, a_min=0.0, a_max=None)
                vmax = float(np.max(clipped_panel))
                if vmax > 0.0:
                    kwargs["norm"] = colors.PowerNorm(gamma=0.5, vmin=0.0, vmax=vmax)
            axis.imshow(panel_data, **kwargs)
            axis.set_title(title)
            axis.set_xticks([])
            axis.set_yticks([])

        figure.suptitle(
            (
                f"Example {int(example_index):03d} | scene={example['scene_id']} | "
                f"psf={example['psf_id']} | noise_level={example['noise_level_id']} | "
                f"sigma={example['noise_sigma']:.3f} | {frame_reduction}"
            ),
            fontsize=11,
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
        _save_figure_png_and_pdf(
            figure,
            plots_dir / f"example_{plot_index:03d}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(figure)

    return {
        "directory": "plots",
        "n_examples_plotted": n_to_plot,
        "frame_reduction": frame_reduction,
        "selection": "random_without_replacement",
        "selection_seed": seed,
        "selected_example_indices": selected_indices.tolist(),
    }


def save_example_statistics(examples: list[dict[str, Any]], outdir: Path) -> dict[str, Any]:
    """Write per-example and dataset-level statistics for the exported examples."""
    stats_path = outdir / "example_stats.json"
    dataset_summary = {
        component: _compute_array_stats(
            np.concatenate(
                [np.ravel(np.asarray(example[component], dtype=np.float32)) for example in examples],
                axis=0,
            )
        )
        for component in EXAMPLE_STAT_COMPONENTS
    }

    per_example = []
    for example_index, example in enumerate(examples):
        per_example.append(
            {
                "example_index": example_index,
                "scene_id": example["scene_id"],
                "psf_id": example["psf_id"],
                "noise_level_id": example["noise_level_id"],
                "noise_sigma": example["noise_sigma"],
                "components": {
                    component: _compute_array_stats(example[component])
                    for component in EXAMPLE_STAT_COMPONENTS
                },
            }
        )

    payload = {
        "n_examples": len(examples),
        "components": list(EXAMPLE_STAT_COMPONENTS),
        "dataset_summary": dataset_summary,
        "per_example": per_example,
    }
    with open(stats_path, "w", encoding="utf-8") as file_obj:
        json.dump(_to_builtin(payload), file_obj, indent=2)

    return {
        "path": stats_path.name,
        "n_examples": len(examples),
        "components": list(EXAMPLE_STAT_COMPONENTS),
        "metrics": ["mean", "median", "std", "sum", "min", "max"],
    }


def _write_tfrecord_file(
    tf: Any,
    examples: list[dict[str, Any]],
    tfrecord_path: Path,
) -> list[dict[str, Any]]:
    tfrecord_path.parent.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    with tf.io.TFRecordWriter(str(tfrecord_path)) as writer:
        for example_index, example in enumerate(examples):
            writer.write(
                serialize_example(
                    tf,
                    image=example["image"],
                    obs=example["obs"],
                    psf=example["psf"],
                    noise=example["noise"],
                    ref_psf=example["ref_psf"],
                )
            )
            manifest.append(
                {
                    "file": tfrecord_path.name,
                    "example_index": example_index,
                    "scene_id": example["scene_id"],
                    "psf_id": example["psf_id"],
                    "noise_level_id": example["noise_level_id"],
                    "noise_sigma": example["noise_sigma"],
                }
            )

    return manifest


def save_tfrecord_dataset(
    arrays: dict[str, np.ndarray],
    outdir: Path,
    *,
    tfrecord_name: str,
    n_plot_examples: int,
    seed: int,
) -> dict[str, Any]:
    tf = _import_tensorflow()
    examples = build_tfrecord_examples(arrays)
    tfrecord_path = outdir / tfrecord_name
    manifest = _write_tfrecord_file(tf, examples, tfrecord_path)
    plot_metadata = save_example_plots(examples, outdir, n_plot_examples=n_plot_examples, seed=seed)
    stats_metadata = save_example_statistics(examples, outdir)

    return {
        "schema": {
            "image": [N_PIX, N_PIX],
            "obs": [1, N_PIX, N_PIX],
            "psf": [1, N_PIX, N_PIX],
            "noise": [1, N_PIX, N_PIX],
            "ref_psf": [N_PIX, N_PIX],
            "dtype": "float32",
        },
        "grouping": {
            "example_unit": "scene_psf_and_noise_level",
            "frame_axis": "singleton_frame",
            "n_examples": len(examples),
            "n_frames_per_example": 1,
            "ref_psf_source": "same_as_example_psf",
            "noise_definition": "noise = clean_obs - obs",
        },
        "tfrecord_file": {
            "path": tfrecord_name,
            "n_examples": len(examples),
        },
        "statistics": stats_metadata,
        "preview_plots": plot_metadata,
        "example_manifest": manifest,
    }


def save_dataset(arrays: dict[str, np.ndarray], metadata: dict[str, Any], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outdir / "dataset.npz", **arrays)
    with open(outdir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(_to_builtin(metadata), f, indent=2)
    generation_log = _build_generation_parameter_log(metadata)
    with open(outdir / DEFAULT_GENERATION_LOG_NAME, "w", encoding="utf-8") as f:
        json.dump(_to_builtin(generation_log), f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a GalSim AO source/PSF/noise grid.")
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment config .py file")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    runtime_cfg = _configure_from_experiment_config(cfg)

    arrays, metadata = build_dataset(
        seed=runtime_cfg["seed"],
        normalize_scene_mean=runtime_cfg["normalize_scene_mean"],
        normalize_scene_mean_half_n_pix_crop=runtime_cfg["normalize_scene_mean_half_n_pix_crop"],
    )
    if runtime_cfg["write_tfrecords"]:
        metadata["tfrecord_dataset"] = save_tfrecord_dataset(
            arrays,
            runtime_cfg["output_dir"],
            tfrecord_name=runtime_cfg["tfrecord_name"],
            n_plot_examples=runtime_cfg["n_plot_examples"],
            seed=runtime_cfg["seed"],
        )
    metadata["runtime_config"] = {
        "n_scenes": N_SCENES,
        "n_psfs": N_PSFS,
        "n_noise_levels": N_NOISE,
        "n_pix": N_PIX,
        "pix_scale_arcsec": PIX_SCALE_ARCSEC,
        "wavelength_nm": LAM_NM,
        "diam_m": DIAM_M,
        "central_obscuration_m": CENTRAL_OBSCURATION_M,
        "point_source_sigma_pix": POINT_SOURCE_SIGMA_PIX,
        "noise_sigmas": NOISE_SIGMAS,
        "normalize_scene_mean": runtime_cfg["normalize_scene_mean"],
        "normalize_scene_mean_half_n_pix_crop": runtime_cfg["normalize_scene_mean_half_n_pix_crop"],
        "image_config": SCENE_CONFIG,
        "psf_config": PSF_CONFIG,
    }
    metadata["generation_parameter_log"] = {
        "path": DEFAULT_GENERATION_LOG_NAME,
        "sampling": {
            "noise_sigma": "linear",
            "psf_residual_wavefront": "linear_template",
        },
    }
    save_dataset(arrays, metadata, outdir=runtime_cfg["output_dir"])

    print(f"Saved dataset to: {runtime_cfg['output_dir'].resolve()}")
    print("dataset.npz contains arrays with shapes:")
    for name, arr in arrays.items():
        print(f"  {name:12s} {arr.shape}")
    print("metadata.json contains the generation parameters and per-scene/per-PSF metadata.")
    print(f"Generation parameter log written to: {(runtime_cfg['output_dir'] / DEFAULT_GENERATION_LOG_NAME).resolve()}")
    if runtime_cfg["write_tfrecords"]:
        n_frames_per_example = metadata["tfrecord_dataset"]["grouping"]["n_frames_per_example"]
        stats_path = metadata["tfrecord_dataset"]["statistics"]["path"]
        print("Single TFRecord file written with loader-compatible schema:")
        print(f"  {runtime_cfg['tfrecord_name']}, frames per example = {n_frames_per_example}")
        print(f"Statistics log written to: {(runtime_cfg['output_dir'] / stats_path).resolve()}")
        print(f"Preview plots written to: {(runtime_cfg['output_dir'] / 'plots').resolve()} ({runtime_cfg['n_plot_examples']} requested)")


if __name__ == "__main__":
    main()
