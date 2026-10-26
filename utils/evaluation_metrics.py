"""Per-image reconstruction and marginal-interval metrics.

Metrics use raw predictions, including negative intensities. PSNR and SSIM use
the same truth-derived range for every method on an example; predictions are
never independently rescaled. Constant truths use max(abs(truth), 1) as range.
"""

from __future__ import annotations

import numpy as np


def _image_batch(array: np.ndarray, crop_border: int = 0) -> np.ndarray:
    array = np.asarray(array, dtype=np.float64)
    if array.ndim == 2:
        array = array[None, ..., None]
    elif array.ndim == 3:
        array = array[..., None]
    if array.ndim != 4:
        raise ValueError("Expected H×W, N×H×W or N×H×W×C arrays")
    crop_border = int(crop_border)
    if crop_border < 0 or 2 * crop_border >= min(array.shape[1:3]):
        raise ValueError("Crop must leave a nonempty image")
    if crop_border:
        array = array[:, crop_border:-crop_border, crop_border:-crop_border, :]
    return array


def object_image_metrics(
    truth: np.ndarray, prediction: np.ndarray, *, crop_border: int = 0,
) -> dict[str, np.ndarray]:
    """Return per-image PSNR, SSIM, range and negative-prediction statistics.

    SSIM uses the largest odd window up to seven pixels and is undefined (NaN)
    when either spatial dimension is smaller than three. Perfect predictions
    have infinite PSNR, following its conventional definition.
    """
    from skimage.metrics import structural_similarity

    truth = _image_batch(truth, crop_border)
    prediction = _image_batch(prediction, crop_border)
    if truth.shape != prediction.shape:
        raise ValueError("Truth and prediction must have identical shapes")
    axes = (1, 2, 3)
    dynamic_range = np.ptp(truth, axis=axes)
    constant = dynamic_range == 0
    dynamic_range = np.where(constant, np.maximum(np.max(np.abs(truth), axis=axes), 1.0), dynamic_range)
    mse = np.mean(np.square(prediction - truth), axis=axes)
    with np.errstate(divide="ignore", invalid="ignore"):
        psnr = 10 * np.log10(np.square(dynamic_range) / mse)
    negative_flux = np.sum(np.maximum(-prediction, 0), axis=axes)
    absolute_flux = np.sum(np.abs(prediction), axis=axes)
    negative_flux_fraction = np.divide(negative_flux, absolute_flux, out=np.zeros_like(negative_flux), where=absolute_flux > 0)
    window = min(7, min(truth.shape[1:3]))
    window -= 1 - window % 2
    ssim = np.full(truth.shape[0], np.nan)
    if window >= 3:
        for i, (target, estimate) in enumerate(zip(truth, prediction)):
            ssim[i] = structural_similarity(
                target, estimate, data_range=float(dynamic_range[i]),
                win_size=window, channel_axis=-1,
            )
    return {
        "psnr_im": psnr,
        "ssim_im": ssim,
        "data_range_im": dynamic_range,
        "constant_truth_im": constant.astype(np.float64),
        "negative_pixel_fraction_im": np.mean(prediction < 0, axis=axes),
        "negative_flux_fraction_im": negative_flux_fraction,
    }


def interval_coverage(
    truth: np.ndarray, mean: np.ndarray, variance: np.ndarray, *, crop_border: int = 0,
) -> dict[str, np.ndarray]:
    """Return per-image marginal 1σ/2σ coverage without clipping predictions.

    Coverage is a simulator-conditioned marginal diagnostic; it does not imply
    independent pixels or account for model-parameter/systematic uncertainty.
    """
    truth, mean, variance = (_image_batch(a, crop_border) for a in (truth, mean, variance))
    if truth.shape != mean.shape or truth.shape != variance.shape:
        raise ValueError("Truth, mean and variance must have identical shapes")
    if not np.all(np.isfinite(variance)) or np.any(variance < 0):
        raise ValueError("Predictive variances must be finite and nonnegative")
    error = np.abs(truth - mean)
    sigma = np.sqrt(variance)
    return {
        "coverage_1sigma": np.mean(error <= sigma, axis=(1, 2, 3)),
        "coverage_2sigma": np.mean(error <= 2 * sigma, axis=(1, 2, 3)),
    }
