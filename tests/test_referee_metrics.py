from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from utils.evaluation_metrics import interval_coverage, object_image_metrics


def test_marginal_coverage_preserves_image_weighting_and_interval_boundaries():
    truth = np.zeros((2, 3, 3, 1))
    prediction = np.zeros_like(truth)
    prediction[0, :, :, 0] = [[0, 1, -1], [2, -2, 3], [0, 0, 0]]
    prediction[1] = 10
    result = interval_coverage(truth, prediction, np.ones_like(truth))
    np.testing.assert_allclose(result["coverage_1sigma"], [6 / 9, 0])
    np.testing.assert_allclose(result["coverage_2sigma"], [8 / 9, 0])


def test_coverage_uses_same_crop_and_rejects_invalid_variances():
    truth = np.zeros((1, 5, 5, 1))
    prediction = np.ones_like(truth) * 10
    prediction[:, 1:-1, 1:-1] = 0
    assert interval_coverage(truth, prediction, np.ones_like(truth), crop_border=1)["coverage_1sigma"][0] == 1
    with pytest.raises(ValueError, match="nonnegative"):
        interval_coverage(truth, prediction, -np.ones_like(truth))
    with pytest.raises(ValueError, match="nonempty"):
        interval_coverage(truth, prediction, np.ones_like(truth), crop_border=3)


def test_psnr_matches_known_error_and_negative_predictions_are_not_clipped():
    pytest.importorskip("skimage")
    truth = np.linspace(0, 1, 81).reshape(1, 9, 9, 1)
    prediction = truth - 0.1
    result = object_image_metrics(truth, prediction)
    np.testing.assert_allclose(result["psnr_im"], [20])
    assert result["negative_pixel_fraction_im"][0] > 0
    assert result["negative_flux_fraction_im"][0] > 0
    assert result["ssim_im"][0] < 1
    assert result["data_range_im"][0] == 1


def test_constant_truth_range_and_perfect_reconstruction_are_explicit():
    pytest.importorskip("skimage")
    truth = np.ones((2, 9, 9, 1)) * 3
    result = object_image_metrics(truth, truth)
    np.testing.assert_array_equal(result["constant_truth_im"], 1)
    np.testing.assert_array_equal(result["data_range_im"], 3)
    assert np.all(np.isposinf(result["psnr_im"]))
    np.testing.assert_allclose(result["ssim_im"], 1)


def test_product_variance_propagation_includes_cross_term():
    tf = pytest.importorskip("tensorflow")
    from utils.uncertainty import propagate_observation_variance

    # A one-pixel convolution is the product of independent scalar variables.
    field = lambda value: tf.constant([[[[value]]]], dtype=tf.float32)
    result = propagate_observation_variance(
        pred_im=field(2), pred_psf_phys=field(3), sigma2_im=field(4),
        sigma2_psf_phys=field(5), sigma2_noise_phys=field(6),
    )
    np.testing.assert_allclose(result.numpy(), 4 * 9 + 4 * 5 + 4 * 5 + 6)


def test_saved_inference_reconstructs_full_field_before_single_crop():
    tf = pytest.importorskip("tensorflow")
    pytest.importorskip("skimage")
    from neural_networks.losses import _convolve_image_with_psfs
    from testing.galsim_evaluation import JointPinnBackend, _evaluate_saved_inference

    image = tf.reshape(tf.range(81, dtype=tf.float32), (1, 9, 9, 1)) / 81
    psf = tf.reshape(tf.range(1, 82, dtype=tf.float32), (1, 9, 9, 1))
    psf = psf / tf.reduce_sum(psf)
    noise = tf.zeros_like(image)
    obs = _convolve_image_with_psfs(image, psf)
    truth = tf.concat([image, psf, noise], axis=-1)
    pred = tf.concat([truth, tf.zeros_like(truth)], axis=-1)
    model = SimpleNamespace(
        _psf_denorm_factor=1, _noise_denorm_factor=1, reconstruction_crop=1,
        log_sigma=False, log_min=-6, log_max=20, sigma2_eps=1e-12,
    )
    backend = JointPinnBackend(model=model, run_dir=Path("."), model_path=Path("unused"), training_cfg={})
    result = _evaluate_saved_inference(
        backend, obs=obs.numpy(), y_true=truth.numpy(), y_pred=pred.numpy(),
        batch_size=1, crop_border=1,
    )
    np.testing.assert_allclose(result["r2_pinn"], 0, atol=1e-10)
    np.testing.assert_allclose(result["ssim_im"], 1)
    np.testing.assert_allclose(result["coverage_1sigma_im"], 1)
