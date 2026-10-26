"""Shared approximate propagation of component predictive variances."""

from __future__ import annotations


def propagate_observation_variance(
    *, pred_im, pred_psf_phys, sigma2_im, sigma2_psf_phys,
    sigma2_noise_phys, eps: float = 1e-12,
):
    """Propagate independent object, PSF and additive-residual variances.

    For y = x*h ± noise, use Var(x)*h² + x²*Var(h) + Var(x)*Var(h)
    (convolutions) + Var(noise). This retains the independent-product term,
    while neglecting spatial/component covariances and uncertainty induced by
    PSF normalization. It is an approximation, not a full posterior covariance.
    """
    import tensorflow as tf
    from neural_networks.losses import _convolve_image_with_psfs

    variance = (
        _convolve_image_with_psfs(sigma2_im, tf.square(pred_psf_phys))
        + _convolve_image_with_psfs(tf.square(pred_im), sigma2_psf_phys)
        + _convolve_image_with_psfs(sigma2_im, sigma2_psf_phys)
        + sigma2_noise_phys
    )
    return tf.maximum(variance, tf.cast(eps, variance.dtype))
