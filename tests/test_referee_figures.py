"""Scientific display conventions must not alter numerical predictions."""

import matplotlib

matplotlib.use("Agg")

from matplotlib.colors import SymLogNorm
import numpy as np
import pytest

from utils import plot_helpers as plots
from workflow.referee_naco import _mean_observation, _radial_profile, _reference_psf_image, _reference_background, _write_comparison


def test_psf_display_uses_recorded_scale_and_preserves_predictions(monkeypatch, tmp_path):
    image = np.arange(64, dtype=float).reshape(8, 8) / 64
    physical_psf = np.zeros((8, 8))
    physical_psf[4, 4] = 0.8
    # The displayed crop is 8x8, but the recorded model scale is 128^2.
    factor = 128**2
    psf = physical_psf * factor
    sigma = np.full((8, 8), 0.01 * factor)
    original = psf.copy()
    figures = []
    monkeypatch.setattr(plots, "_save_figure_png_and_pdf", lambda fig, *args, **kwargs: figures.append(fig))
    plots._plot_truth_vs_prediction(
        obs_true=image, im_true=image, psf_true=psf, noise_true=image * 0,
        obs_pred=image * 0.9, im_pred=image * 0.9, psf_pred=psf, noise_pred=image * 0,
        sigma_obs=np.ones_like(image), sigma_im=np.ones_like(image), sigma_psf=sigma,
        sigma_noise=np.ones_like(image), frame=0, obs_panel_n_pix_zero=1,
        out_path=tmp_path / "comparison.png", dpi=50,
        psf_denorm_factor=factor, pixel_scale_arcsec=0.027,
    )
    axes = figures[0].axes[:24]
    np.testing.assert_allclose(axes[2].images[0].get_array(), physical_psf)
    np.testing.assert_allclose(axes[10].images[0].get_array(), 0.01)
    np.testing.assert_array_equal(psf, original)
    assert axes[0].get_title() == "Observed image"
    assert axes[0].get_xlabel() == "Offset x (arcsec)"
    assert any(text.get_text() == "(a)" for text in axes[0].texts)
    assert isinstance(axes[12].images[0].norm, SymLogNorm)


def test_log_plot_masks_nonpositive_values_without_clipping_the_array():
    values = np.array([[-0.1, 0.0], [1e-8, 1.0]])
    original = values.copy()
    displayed = plots._log_display_data(values)
    np.testing.assert_array_equal(np.ma.getmaskarray(displayed), [[True, True], [False, False]])
    assert displayed[1, 0] == 1e-8
    np.testing.assert_array_equal(values, original)


def test_invalid_recorded_psf_scale_is_rejected():
    with pytest.raises(ValueError, match="finite and positive"):
        plots._physical_psf(np.ones((2, 2)), 0.0)


def test_signed_residual_norm_has_symmetric_limits_and_zero_center():
    norm = plots._symmetric_log_norm([np.array([-100.0, -0.01, 0.0, 0.01, 100.0])])
    assert norm.vmin == -norm.vmax
    assert norm(0.0) == pytest.approx(0.5)
    assert norm(-0.01) < norm(0.0) < norm(0.01)


def test_noise_label_matches_subtractive_model_convention():
    assert plots._panel_title("Pred res").replace("\n", " ") == "Noise correction (subtracted)"
    assert "additive" not in plots._panel_title("Pred res unc").lower()


def test_saved_component_sigmas_do_not_add_a_display_floor(monkeypatch):
    tf = pytest.importorskip("tensorflow")
    from workflow import plot_results

    class Model:
        log_sigma, log_min, log_max, sigma2_eps = False, -30.0, 20.0, 1e-30
        _psf_denorm_factor, _noise_denorm_factor = 16.0, 1.0

        def __call__(self, observation, training=False):
            mean = tf.ones_like(observation)
            return tf.concat([mean, mean, mean * 0, mean * 0, mean * 0, mean * 0], axis=-1)

    def tiny_variance(raw, **kwargs):
        variance = tf.ones_like(raw) * 1e-20
        return variance, tf.math.log(variance)

    monkeypatch.setattr(plot_results, "_pred_to_sigma2", tiny_variance)
    prediction = plot_results._predict_joint_batch(Model(), np.ones((1, 4, 4, 1), dtype=np.float32))
    np.testing.assert_allclose(prediction["sigma_im"]**2, 1e-20, rtol=1e-6, atol=0)
    np.testing.assert_allclose(prediction["sigma_noise_phys"]**2, 1e-20, rtol=1e-6, atol=0)
    np.testing.assert_allclose(prediction["sigma_psf_phys"]**2, 1e-20 / 16**2, rtol=1e-6, atol=0)


def test_naco_mean_input_preserves_intensity_and_records_crop():
    cube = np.arange(12 * 12 * 3, dtype=np.float32).reshape(12, 12, 3)
    original = cube.copy()
    image, metadata = _mean_observation(cube, frame_axis=-1, input_size=8)
    np.testing.assert_allclose(image, cube.mean(axis=-1)[2:10, 2:10])
    np.testing.assert_array_equal(cube, original)
    assert metadata["crop_origin_yx"] == [2, 2]
    assert metadata["n_frames_averaged"] == 3
    assert metadata["intensity_rescaling"] == "none"


def test_reference_psf_radial_profile_is_invariant_to_flux_scale():
    yy, xx = np.indices((15, 15))
    psf = np.exp(-((xx - 7)**2 + (yy - 7)**2) / 4)
    r, profile = _radial_profile(psf, pixel_scale=0.027, radius_arcsec=0.15)
    r2, profile2 = _radial_profile(12 * psf, pixel_scale=0.027, radius_arcsec=0.15)
    np.testing.assert_array_equal(r, r2)
    np.testing.assert_allclose(profile, profile2)


def test_naco_comparison_rejects_unregistered_mistral(tmp_path):
    observed = np.ones((8, 8))
    components = {name: {"pred_im": observed.copy()} for name in ("pretrained", "joint_pinn")}
    with pytest.raises(ValueError, match="pre-registered"):
        _write_comparison(observed, components, np.ones((6, 6)), tmp_path, 0.027)


def test_reference_cube_averaging_and_background_are_explicit():
    yy, xx = np.indices((21, 21))
    psf = np.exp(-((xx - 10)**2 + (yy - 10)**2) / 4)
    cube = np.stack([psf + 3, psf + 5])
    mean, protocol = _reference_psf_image(cube, frame_axis=0)
    np.testing.assert_allclose(mean, psf + 4, rtol=1e-6)
    assert protocol["n_frames_averaged"] == 2
    background = _reference_background(mean, pixel_scale=0.027, radius_arcsec=0.24)
    assert background == pytest.approx(4, abs=1e-4)
