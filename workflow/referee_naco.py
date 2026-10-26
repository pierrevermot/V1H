"""Compare the same campaign's pretrained and fine-tuned models on NaCo data.

The observation is averaged over its frame axis before one inference per model.
Its input intensity scale is preserved. Optional MISTRAL images must already be
registered and expressed in the same intensity units; no alignment is inferred.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _load_image(path: Path) -> np.ndarray:
    if path.suffix.lower() in (".fits", ".fit", ".fts"):
        from astropy.io import fits

        image = np.asarray(fits.getdata(path), dtype=np.float32)
    elif path.suffix.lower() == ".npy":
        image = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    else:
        raise ValueError(f"Expected a scientific FITS or NPY array: {path}")
    if image.ndim not in (2, 3) or not np.isfinite(image).all():
        raise ValueError(f"Expected a finite 2D image or 3D observation cube: {path}, {image.shape}")
    return image


def _mean_observation(cube: np.ndarray, *, frame_axis: int, input_size: int) -> tuple[np.ndarray, dict]:
    cube = np.asarray(cube, dtype=np.float32)
    n_frames = cube.shape[frame_axis] if cube.ndim == 3 else 1
    image = np.mean(cube, axis=frame_axis, dtype=np.float64).astype(np.float32) if cube.ndim == 3 else cube.copy()
    if image.ndim != 2 or min(image.shape) < input_size:
        raise ValueError(f"Observation {image.shape} is smaller than model input {input_size}x{input_size}")
    y0, x0 = (image.shape[0] - input_size) // 2, (image.shape[1] - input_size) // 2
    cropped = image[y0:y0 + input_size, x0:x0 + input_size].copy()
    return cropped, dict(original_shape=list(cube.shape), frame_axis=frame_axis,
                        n_frames_averaged=int(n_frames), crop_origin_yx=[y0, x0],
                        input_shape=list(cropped.shape), intensity_rescaling="none")


def _radial_profile(image: np.ndarray, *, pixel_scale: float, radius_arcsec: float) -> tuple[np.ndarray, np.ndarray]:
    """Peak-center and normalize signed flux in the same angular aperture."""
    image = np.asarray(image, dtype=float)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("Reference PSF must be a finite 2D scientific image")
    if not np.isfinite(pixel_scale) or pixel_scale <= 0 or not np.isfinite(radius_arcsec) or radius_arcsec <= 0:
        raise ValueError("PSF pixel scale and aperture radius must be finite and positive")
    peak = np.unravel_index(np.argmax(image), image.shape)
    yy, xx = np.indices(image.shape)
    radius_pix = np.hypot(yy - peak[0], xx - peak[1])
    included = radius_pix * pixel_scale <= radius_arcsec
    flux = float(image[included].sum())
    if flux <= 0:
        raise ValueError("PSF flux inside the comparison aperture must be positive")
    bins = np.floor(radius_pix[included]).astype(int)
    sums = np.bincount(bins, weights=image[included] / flux)
    counts = np.bincount(bins)
    # Surface brightness makes comparisons independent of pixel sampling.
    profile = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0) / pixel_scale**2
    return (np.arange(profile.size) + 0.5) * pixel_scale, profile


def _sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _available_psf_radius(image: np.ndarray, pixel_scale: float) -> float:
    peak = np.unravel_index(np.argmax(image), image.shape)
    return min(peak[0] + 0.5, peak[1] + 0.5,
               image.shape[0] - peak[0] - 0.5, image.shape[1] - peak[1] - 0.5) * pixel_scale


def _reference_psf_image(cube: np.ndarray, frame_axis: int) -> tuple[np.ndarray, dict]:
    image = np.mean(cube, axis=frame_axis, dtype=np.float64).astype(np.float32) if cube.ndim == 3 else cube.copy()
    if image.ndim != 2:
        raise ValueError("Averaged reference PSF must be 2D")
    return image, dict(original_shape=list(cube.shape), frame_axis=frame_axis,
                       n_frames_averaged=int(cube.shape[frame_axis]) if cube.ndim == 3 else 1,
                       alignment="No frame registration performed; mean of supplied calibration cube")


def _reference_background(image: np.ndarray, *, pixel_scale: float, radius_arcsec: float) -> float:
    peak = np.unravel_index(np.argmax(image), image.shape)
    yy, xx = np.indices(image.shape)
    radius = np.hypot(yy - peak[0], xx - peak[1]) * pixel_scale
    annulus = image[(radius >= 0.8 * radius_arcsec) & (radius <= radius_arcsec)]
    if not annulus.size:
        raise ValueError("Reference PSF comparison aperture has no outer-annulus pixels")
    return float(np.median(annulus))


def _write_comparison(observed: np.ndarray, components: dict[str, dict[str, np.ndarray]],
                      mistral: np.ndarray | None, out: Path, pixel_scale: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils.plot_helpers import _finish_figure, _imshow, _log_norm_no_clip, _save_figure_png_and_pdf

    panels = [("Observed image", observed),
              ("Pretrained object U-Net", components["pretrained"]["pred_im"]),
              ("Fine-tuned object U-Net", components["joint_pinn"]["pred_im"])]
    if mistral is not None:
        if mistral.ndim != 2 or mistral.shape != observed.shape:
            raise ValueError("MISTRAL must be pre-registered to the displayed observation, with the same 2D shape")
        panels.append(("MISTRAL (supplied reconstruction)", mistral))
    norm = _log_norm_no_clip([array for _, array in panels])
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5.2), squeeze=False)
    positive = observed[observed > 0]
    levels = np.unique(np.percentile(positive, [80, 95, 99])) if positive.size else []
    for ax, (title, image) in zip(axes.flat, panels):
        _imshow(ax, image, title, norm=norm)
        valid_levels = [level for level in levels if float(np.min(image)) < level < float(np.max(image))]
        if valid_levels:
            ax.contour(image, levels=valid_levels, colors="white", linewidths=0.65, alpha=0.7)
        colorbar = fig.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label("Input intensity units")
    _finish_figure(fig, axes, pixel_scale)
    _save_figure_png_and_pdf(fig, out / "naco_comparison", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    offset = (np.arange(observed.shape[0]) - (observed.shape[0] - 1) / 2) * pixel_scale
    center = observed.shape[0] // 2
    for title, image in panels:
        axes[0].plot(offset, image[center, :], label=title)
        axes[1].plot(offset, image[:, center], label=title)
    for ax, name in zip(axes, ("Horizontal", "Vertical")):
        ax.set_title(f"{name} central profile (pixel-coordinate orientation)")
        ax.set_xlabel("Offset (arcsec)")
        ax.set_ylabel("Input intensity units")
        ax.set_yscale("symlog", linthresh=max(float(np.max(np.abs(observed))) * 1e-3, 1e-9))
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=9)
    fig.tight_layout()
    _save_figure_png_and_pdf(fig, out / "naco_profiles", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--frame-axis", type=int, default=-1)
    parser.add_argument("--mistral", type=Path, help="Scientific image already registered and in the same intensity units as the observation")
    parser.add_argument("--reference-psf", type=Path)
    parser.add_argument("--reference-frame-axis", type=int, default=0, help="Frame axis of a reference-star FITS/NPY cube (FITS convention: 0)")
    parser.add_argument("--reference-background", choices=("none", "outer_annulus"), default="outer_annulus",
                        help="Optional constant background estimate from the outer20%% of the common PSF comparison aperture")
    parser.add_argument("--pixel-scale-arcsec", type=float, default=0.027)
    parser.add_argument("--reference-psf-pixel-scale-arcsec", type=float, default=None)
    args = parser.parse_args()
    if not np.isfinite(args.pixel_scale_arcsec) or args.pixel_scale_arcsec <= 0:
        raise ValueError("Pixel scale must be finite and positive")

    import tensorflow as tf
    from astropy.io import fits
    from configs.load_config import load_experiment_config
    from testing.galsim_evaluation import BACKEND_REGISTRY
    from workflow.plot_results import _predict_joint_batch
    from utils.plot_helpers import _plot_inference_example

    cfg = load_experiment_config(args.config)
    if int(cfg.INSTRUMENT_CONFIG.get("n_frames", 1)) != 1:
        raise ValueError("The NaCo comparison uses single-image models")
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    observed, protocol = _mean_observation(
        _load_image(args.data_path), frame_axis=args.frame_axis,
        input_size=int(cfg.N_PIX) - 2 * int(cfg.DATASET_LOAD_CONFIG["half_n_pix_crop"]),
    )
    mistral = _load_image(args.mistral) if args.mistral is not None else None
    if mistral is not None and (mistral.ndim != 2 or mistral.shape != observed.shape):
        raise ValueError("MISTRAL must be pre-registered, same-shape 2D data in the observation's intensity units")
    run_dir = Path(cfg.OUTPUT_BASE_DIR) / cfg.JOINT_PINN_CONFIG["run_name"]
    batch = observed[None, ..., None]
    components, provenance = {}, {}
    for name in ("pretrained", "joint_pinn"):
        backend = BACKEND_REGISTRY[name].from_run(cfg=cfg, run_dir=run_dir, model_label="best_model",
                                                 preview_obs=tf.convert_to_tensor(batch))
        prediction = _predict_joint_batch(backend.model, batch)
        components[name] = {key: np.asarray(value)[0, ..., 0] for key, value in prediction.items()
                            if isinstance(value, np.ndarray)}
        provenance[name] = backend.describe()
        arrays = components[name]
        for key, array in arrays.items():
            header = fits.Header()
            header["PIXSCALE"] = (args.pixel_scale_arcsec, "arcsec/pixel; pixel-coordinate orientation")
            header["MODEL"] = name
            fits.writeto(out / f"{name}_{key}.fits", array, header=header, overwrite=False)
        _plot_inference_example(
            obs_true=observed, pred_im=arrays["pred_im"], pred_psf=arrays["pred_psf_phys"],
            pred_noise=arrays["pred_noise_phys"], pred_obs=arrays["pred_obs"],
            sigma_obs=arrays["sigma_obs"], sigma_im=arrays["sigma_im"],
            sigma_psf=arrays["sigma_psf_phys"], sigma_noise=arrays["sigma_noise_phys"],
            rec_im=arrays["pred_im"], rec_psf=arrays["pred_psf_phys"], rec_noise=arrays["pred_noise_phys"],
            frame="mean input", out_path=out / f"naco_components_{name}", dpi=150,
            pixel_scale_arcsec=args.pixel_scale_arcsec,
        )
    _write_comparison(observed, components, mistral, out, args.pixel_scale_arcsec)
    arrays = {"observed_mean": observed,
              **{f"{name}_{key}": value for name, values in components.items() for key, value in values.items()}}
    if mistral is not None:
        arrays["mistral"] = mistral
    np.savez_compressed(out / "naco_predictions.npz", **arrays)
    metadata = dict(protocol=protocol, models=provenance,
                    observation=dict(path=str(args.data_path.resolve()), sha256=_sha256(args.data_path)),
                    orientation="Unchanged pixel axes; no sky orientation or WCS inferred",
                    pixel_scale_arcsec=args.pixel_scale_arcsec,
                    mistral=dict(available=mistral is not None, path=str(args.mistral) if args.mistral else None,
                                 registration="Must be supplied pre-registered in matching intensity units; no automatic alignment"),
                    uncertainty="Simulation-conditioned marginal predictive scales; independent-pixel/component propagation, not epistemic uncertainty or a real-data calibration proof")
    if args.reference_psf is not None:
        reference, reference_protocol = _reference_psf_image(_load_image(args.reference_psf), args.reference_frame_axis)
        ref_scale = args.reference_psf_pixel_scale_arcsec or args.pixel_scale_arcsec
        if not np.isfinite(ref_scale) or ref_scale <= 0:
            raise ValueError("Reference PSF pixel scale must be finite and positive")
        radius = min(_available_psf_radius(reference, ref_scale), *[
            _available_psf_radius(value["pred_psf_phys"], args.pixel_scale_arcsec)
            for value in components.values()
        ])
        background = _reference_background(reference, pixel_scale=ref_scale, radius_arcsec=radius) if args.reference_background == "outer_annulus" else 0.0
        np.save(out / "reference_psf_mean.npy", reference)
        reference = reference - background
        np.save(out / "reference_psf_for_profile.npy", reference)
        import matplotlib.pyplot as plt
        from utils.plot_helpers import _save_figure_png_and_pdf

        fig, ax = plt.subplots(figsize=(7, 5))
        for name, array, scale in [("Reference star (calibrator)", reference, ref_scale)] + [
            (name, value["pred_psf_phys"], args.pixel_scale_arcsec) for name, value in components.items()
        ]:
            r, profile = _radial_profile(array, pixel_scale=scale, radius_arcsec=radius)
            ax.plot(r, profile, label=name)
        ax.set(xlabel="Radius from PSF peak (arcsec)", ylabel="Aperture-normalized surface brightness (arcsec⁻²)")
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        _save_figure_png_and_pdf(fig, out / "naco_psf_profiles", dpi=150)
        plt.close(fig)
        metadata["reference_psf"] = dict(available=True, path=str(args.reference_psf),
                                          sha256=_sha256(args.reference_psf), protocol=reference_protocol,
                                          normalization="Peak-centered, signed flux normalized inside common angular aperture",
                                          radius_arcsec=radius, pixel_scale_arcsec=ref_scale,
                                          background=dict(method=args.reference_background, value=background,
                                                          annulus_fraction=[0.8, 1.0] if args.reference_background == "outer_annulus" else None),
                                          caveat="Reference-star and science PSFs may differ because of calibration mismatch")
    else:
        metadata["reference_psf"] = dict(available=False)
    (out / "naco_provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
