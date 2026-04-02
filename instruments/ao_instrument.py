"""
Telescope ao_instru generation module.

This module provides functions to generate telescope ao_instru masks at specific
pixel scales suitable for FFT-based PSF computation. The key function is
`get_ao_instrument()` which creates a ao_instru array matched to a desired angular pixel
scale for a given wavelength.

Example
-------
>>> ao_instru = get_ao_instrument("vlt", n_pix=256, wavelength=1.6e-6, angular_pixel_scale=5e-3 / 206265)
>>> psf = np.abs(np.fft.fftshift(np.fft.fft2(ao_instru.pupil_array)))**2
"""

from __future__ import annotations

from typing import Callable

import math

import hcipy
import numpy as np
from scipy import ndimage

from utils.array_backend import (
    is_cupy_array as _is_cupy_array,
    get_xp,
    get_ndimage,
    to_numpy,
)


# =============================================================================
# Telescope metadata
# =============================================================================

# Telescope metadata: aperture generator, physical diameter (meters), and info
# Diameters from hcipy source: https://github.com/ehpor/hcipy/blob/main/hcipy/aperture/realistic.py
TELESCOPE_METADATA: dict[str, dict] = {
    "elt": {
        "generator": hcipy.make_elt_aperture,
        "diameter_m": 39.14634,
        "full_name": "Extremely Large Telescope",
    },
    "gmt": {
        "generator": hcipy.make_gmt_aperture,
        "diameter_m": 25.448,
        "full_name": "Giant Magellan Telescope",
    },
    "tmt": {
        "generator": hcipy.make_tmt_aperture,
        "diameter_m": 30.0,
        "full_name": "Thirty Meter Telescope",
    },
    "vlt": {
        "generator": hcipy.make_vlt_aperture,
        "diameter_m": 8.0,
        "full_name": "Very Large Telescope",
    },
    "hst": {
        "generator": hcipy.make_hst_aperture,
        "diameter_m": 2.4,
        "full_name": "Hubble Space Telescope",
    },
    "jwst": {
        "generator": hcipy.make_jwst_aperture,
        "diameter_m": 6.603,
        "full_name": "James Webb Space Telescope",
    },
    "keck": {
        "generator": hcipy.make_keck_aperture,
        "diameter_m": 10.95,
        "full_name": "W. M. Keck Observatory",
    },
    "subaru": {
        "generator": hcipy.make_subaru_aperture,
        "diameter_m": 7.95,
        "full_name": "Subaru Telescope",
    },
    "scexao": {
        "generator": hcipy.make_scexao_aperture,
        "diameter_m": 7.95,
        "full_name": "SCExAO (Subaru Coronagraphic Extreme AO)",
    },
}

# Convenient access to aperture generators by name
APERTURE_GENERATORS: dict[str, Callable] = {
    name: meta["generator"] for name, meta in TELESCOPE_METADATA.items()
}

# Default maximum pixel scale for high-resolution ao_instru generation.
# This value (2 mm) is chosen to adequately resolve fine structures like
# segment gaps and spider vanes in segmented telescope pupils (e.g., ELT, JWST).
DEFAULT_MAX_HIGH_RES_PIXEL_SCALE: float = 2e-3  # meters


# =============================================================================
# Island detection
# =============================================================================


def detect_islands(
    pupil_array: np.ndarray,
    threshold: float = 0.99,
    connectivity: int = 2,
    use_cupy: bool | None = None,
) -> np.ndarray:
    """
    Detect separate "islands" (connected regions) in a ao_instru array.

    Uses connected component labeling to identify distinct open regions
    in the ao_instru mask. This is useful for segmented mirrors or pupils
    with disconnected apertures (e.g., GMT, Keck).

    Parameters
    ----------
    pupil_array : np.ndarray
        2D array representing the ao_instru transmission (0 = blocked, 1 = open).
    threshold : float, optional
        Threshold value to binarize the ao_instru. Pixels with values >= threshold
        are considered open. Default is 0.99.
    connectivity : int, optional
        Connectivity for labeling:
        - 1: 4-connected (only horizontal/vertical neighbors)
        - 2: 8-connected (includes diagonal neighbors)
        Default is 2.

    Returns
    -------
    np.ndarray
        3D array of shape (n_islands, ny, nx) where each slice along the first
        axis is a binary mask for one island. Returns an empty array with shape
        (0, ny, nx) if no islands are detected.

    Examples
    --------
    >>> from ao_instrument import get_ao_instrument, detect_islands
    >>> ao_instru = get_ao_instrument("gmt", n_pix=256, wavelength=1.6e-6,
    ...                   angular_pixel_scale=5e-3 / 206265)
    >>> islands = detect_islands(ao_instru.pupil_array)
    >>> print(f"Detected {islands.shape[0]} islands")
    """
    if use_cupy is None:
        use_cupy = _is_cupy_array(pupil_array)
    xp = get_xp(use_cupy)
    ndi = get_ndimage(use_cupy)

    # Binarize the ao_instru mask
    binary_pupil = (pupil_array >= threshold).astype(xp.int32)

    # Define connectivity structure
    if connectivity == 1:
        # 4-connected: only horizontal/vertical neighbors
        structure = xp.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=xp.int32)
    else:
        # 8-connected: includes diagonal neighbors
        structure = xp.ones((3, 3), dtype=xp.int32)

    # Connected component labeling
    labeled, n_islands = ndi.label(binary_pupil, structure=structure)

    # Create individual masks for each island
    ny, nx = pupil_array.shape
    if n_islands == 0:
        return xp.empty((0, ny, nx), dtype=xp.float32)

    island_masks = xp.zeros((n_islands, ny, nx), dtype=xp.float32)
    for i in range(n_islands):
        island_masks[i] = (labeled == (i + 1)).astype(xp.float32)

    return island_masks


def detect_islands_rotated(
    rotated_pupils: np.ndarray,
    threshold: float = 0.99,
    connectivity: int = 2,
    use_cupy: bool | None = None,
) -> np.ndarray:
    """
    Detect islands for each ao_instru in a rotated pupils array.

    Applies island detection to each rotated ao_instru and returns a 4D array
    containing the island masks for all pupils. Raises an error if the number
    of detected islands is not consistent across all pupils.

    Parameters
    ----------
    rotated_pupils : np.ndarray
        3D array of shape (n_pupils, n_pix, n_pix) containing rotated pupils.
    threshold : float, optional
        Threshold value to binarize the ao_instru. Default is 0.99.
    connectivity : int, optional
        Connectivity for labeling (1 or 2). Default is 2.

    Returns
    -------
    np.ndarray
        4D array of shape (n_islands, n_pupils, n_pix, n_pix) where each slice
        along the first axis contains the masks for one island across all
        rotated pupils.

    Raises
    ------
    ValueError
        If the number of detected islands differs between pupils.

    Examples
    --------
    >>> from ao_instrument import get_ao_instrument, rotate_pupil, detect_islands_rotated
    >>> ao_instru = get_ao_instrument("gmt", n_pix=256, wavelength=1.6e-6,
    ...                   angular_pixel_scale=5e-3 / 206265)
    >>> rotated = rotate_pupil(ao_instru.pupil_array, angles=np.arange(0, 360, 10))
    >>> islands = detect_islands_rotated(rotated)
    >>> print(f"Shape: {islands.shape}")  # (n_islands, n_pupils, n_pix, n_pix)
    """
    if use_cupy is None:
        use_cupy = _is_cupy_array(rotated_pupils)
    xp = get_xp(use_cupy)

    n_pupils, ny, nx = rotated_pupils.shape

    # Detect islands for first ao_instru to get reference count
    first_islands = detect_islands(
        rotated_pupils[0], threshold, connectivity, use_cupy=use_cupy
    )
    n_islands = first_islands.shape[0]

    if n_islands == 0:
        return xp.empty((0, n_pupils, ny, nx), dtype=xp.float32)

    # Allocate output array
    all_islands = xp.zeros((n_islands, n_pupils, ny, nx), dtype=xp.float32)
    all_islands[:, 0, :, :] = first_islands

    # Process remaining pupils
    for i in range(1, n_pupils):
        islands = detect_islands(
            rotated_pupils[i], threshold, connectivity, use_cupy=use_cupy
        )
        if islands.shape[0] != n_islands:
            raise ValueError(
                f"Inconsistent number of islands: ao_instru 0 has {n_islands} islands, "
                f"but ao_instru {i} has {islands.shape[0]} islands."
            )
        all_islands[:, i, :, :] = islands

    return all_islands


# =============================================================================
# LWE modes
# =============================================================================


def compute_lwe_modes(island_masks: np.ndarray) -> np.ndarray:
    """
    Compute Low Wind Effect (LWE) modes for each island.

    Generates piston, tip, and tilt modes for each island mask. These modes
    are commonly used to model the low wind effect in ground-based telescopes
    with segmented or spider-obscured pupils.

    Parameters
    ----------
    island_masks : np.ndarray
        3D array of shape (n_islands, n_pix, n_pix) containing binary masks
        for each island.

    Returns
    -------
    np.ndarray
        4D array of shape (3, n_islands, n_pix, n_pix) containing:
        - lwe_modes[0, i]: piston mode for island i (amplitude = 1)
        - lwe_modes[1, i]: tip mode for island i (std = 1)
        - lwe_modes[2, i]: tilt mode for island i (std = 1)
        All modes are zero outside their respective island mask.

    Examples
    --------
    >>> from ao_instrument import get_ao_instrument, detect_islands, compute_lwe_modes
    >>> ao_instru = get_ao_instrument("vlt", n_pix=128, wavelength=2.2e-6,
    ...                   angular_pixel_scale=13e-3 / 206265)
    >>> islands = detect_islands(ao_instru.pupil_array)
    >>> lwe = compute_lwe_modes(islands)
    >>> print(f"LWE modes shape: {lwe.shape}")  # (3, n_islands, n_pix, n_pix)
    """
    use_cupy = _is_cupy_array(island_masks)
    xp = get_xp(use_cupy)

    n_islands, ny, nx = island_masks.shape

    # Create coordinate grids centered at the array center
    y = xp.arange(ny) - (ny - 1) / 2.0
    x = xp.arange(nx) - (nx - 1) / 2.0
    xx, yy = xp.meshgrid(x, y)

    # Allocate output array
    lwe_modes = xp.zeros((3, n_islands, ny, nx), dtype=xp.float64)

    for i in range(n_islands):
        mask = island_masks[i]
        mask_bool = mask > 0.5

        # Piston: constant value of 1 within the mask
        lwe_modes[0, i] = mask.copy()

        # Tip: x-gradient, masked and normalized to mean=0, std=1
        tip = xx * mask
        if mask_bool.sum() > 0:
            tip_values = tip[mask_bool]
            tip_mean = xp.mean(tip_values)
            tip = tip - tip_mean * mask
            tip_std = xp.std(tip[mask_bool])
            if tip_std > 0:
                tip = tip / tip_std
        lwe_modes[1, i] = tip

        # Tilt: y-gradient, masked and normalized to mean=0, std=1
        tilt = yy * mask
        if mask_bool.sum() > 0:
            tilt_values = tilt[mask_bool]
            tilt_mean = xp.mean(tilt_values)
            tilt = tilt - tilt_mean * mask
            tilt_std = xp.std(tilt[mask_bool])
            if tilt_std > 0:
                tilt = tilt / tilt_std
        lwe_modes[2, i] = tilt

    return lwe_modes


def compute_lwe_modes_rotated(rotated_islands: np.ndarray) -> np.ndarray:
    """
    Compute LWE modes for rotated island masks.

    Wrapper function that applies compute_lwe_modes to each rotation in an
    array of rotated island masks from detect_islands_rotated().

    Parameters
    ----------
    rotated_islands : np.ndarray
        4D array of shape (n_islands, n_rotations, n_pix, n_pix) containing
        island masks for each rotation angle.

    Returns
    -------
    np.ndarray
        5D array of shape (3, n_islands, n_rotations, n_pix, n_pix) containing:
        - lwe_modes[0, i, r]: piston mode for island i at rotation r
        - lwe_modes[1, i, r]: tip mode for island i at rotation r
        - lwe_modes[2, i, r]: tilt mode for island i at rotation r

    Examples
    --------
    >>> from ao_instrument import get_ao_instrument, rotate_pupil, detect_islands_rotated
    >>> from ao_instrument import compute_lwe_modes_rotated
    >>> ao_instru = get_ao_instrument("vlt", n_pix=128, wavelength=2.2e-6,
    ...                   angular_pixel_scale=13e-3 / 206265)
    >>> rotated = rotate_pupil(ao_instru.pupil_array, angles=np.arange(0, 360, 10))
    >>> islands = detect_islands_rotated(rotated)
    >>> lwe = compute_lwe_modes_rotated(islands)
    >>> print(f"LWE shape: {lwe.shape}")  # (3, n_islands, n_rotations, n_pix, n_pix)
    """
    use_cupy = _is_cupy_array(rotated_islands)
    xp = get_xp(use_cupy)

    n_islands, n_rotations, ny, nx = rotated_islands.shape

    # Allocate output array
    lwe_modes = xp.zeros((3, n_islands, n_rotations, ny, nx), dtype=xp.float64)

    # Process each rotation
    for r in range(n_rotations):
        # Extract islands for this rotation: (n_islands, n_pix, n_pix)
        islands_at_rotation = rotated_islands[:, r, :, :]
        # Compute LWE modes: (3, n_islands, n_pix, n_pix)
        modes = compute_lwe_modes(islands_at_rotation)
        # Store in output array
        lwe_modes[:, :, r, :, :] = modes

    return lwe_modes


# =============================================================================
# Zernike modes
# =============================================================================


def compute_zernike_modes(
    ao_instru: "AO_instrument",
    n_rad: int = 6,
    n_min: int = 1,
) -> np.ndarray:
    """
    Compute Zernike polynomials on a ao_instru up to radial order ``n_rad``.

    The Zernike modes are evaluated on the ao_instru support (as defined by the
    ao_instru mask) and zeroed elsewhere. The ao_instru geometry is derived from the
    ``AO_instrument`` object's pixel scale and mask extent so the modes are defined on
    the physical ao_instru, not the full grid.

    Parameters
    ----------
    ao_instru : AO_instrument
        AO_instrument object containing ``pupil_array`` and ``pixel_scale``.
    n_rad : int
        Maximum radial order (inclusive).
    n_min : int, optional
        Minimum radial order (inclusive). Default is 1.

    Returns
    -------
    np.ndarray
        Zernike mode cube of shape (n_modes, n_pix, n_pix), ordered by
        increasing radial order ``n`` and azimuthal order ``m`` from
        ``-n`` to ``n`` in steps of 2.

    Notes
    -----
    Uses standard Zernike normalization on the unit disk:
    - m = 0: sqrt(n + 1)
    - m != 0: sqrt(2 * (n + 1))
    """
    if n_rad < 0:
        raise ValueError(f"n_rad must be >= 0, got {n_rad}")
    if n_min < 0:
        raise ValueError(f"n_min must be >= 0, got {n_min}")
    if n_min > n_rad:
        raise ValueError(f"n_min ({n_min}) must be <= n_rad ({n_rad})")

    xp = getattr(ao_instru, "xp", np)

    pupil_mask = ao_instru.pupil_array > 0.5
    if not bool(xp.any(pupil_mask)):
        n_pix = ao_instru.pupil_array.shape[0]
        return xp.zeros((0, n_pix, n_pix), dtype=xp.float64)

    ny, nx = ao_instru.pupil_array.shape
    y_idx, x_idx = xp.where(pupil_mask)
    cy = xp.mean(y_idx)
    cx = xp.mean(x_idx)

    y = (xp.arange(ny) - cy) * ao_instru.pixel_scale
    x = (xp.arange(nx) - cx) * ao_instru.pixel_scale
    xx, yy = xp.meshgrid(x, y)

    r = xp.sqrt(xx**2 + yy**2)
    r_max = xp.max(r[pupil_mask])
    if r_max == 0:
        n_pix = ao_instru.pupil_array.shape[0]
        return xp.zeros((0, n_pix, n_pix), dtype=xp.float64)

    rho = r / r_max
    theta = xp.arctan2(yy, xx)

    def _zernike_radial(n: int, m: int, rho_vals: np.ndarray) -> np.ndarray:
        m = abs(m)
        if (n - m) % 2 != 0:
            return xp.zeros_like(rho_vals)
        radial = xp.zeros_like(rho_vals, dtype=xp.float64)
        k_max = (n - m) // 2
        for k in range(k_max + 1):
            coeff = (
                ((-1) ** k)
                * math.factorial(n - k)
                / (
                    math.factorial(k)
                    * math.factorial((n + m) // 2 - k)
                    * math.factorial((n - m) // 2 - k)
                )
            )
            radial += coeff * rho_vals ** (n - 2 * k)
        return radial

    nm_list: list[tuple[int, int]] = []
    for n in range(n_min, n_rad + 1):
        for m in range(-n, n + 1, 2):
            nm_list.append((n, m))

    n_modes = len(nm_list)
    modes = xp.zeros((n_modes, ny, nx), dtype=xp.float64)

    for i, (n, m) in enumerate(nm_list):
        radial = _zernike_radial(n, m, rho)
        if m >= 0:
            z = radial * xp.cos(m * theta)
        else:
            z = radial * xp.sin(abs(m) * theta)

        norm = math.sqrt(n + 1) if m == 0 else math.sqrt(2 * (n + 1))
        z *= norm

        z = xp.where(pupil_mask, z, 0.0)
        modes[i] = z

    return modes


# =============================================================================
# Pupil rotation
# =============================================================================


def rotate_pupil(
    pupil_array: np.ndarray,
    angles: np.ndarray | None = None,
    order: int = 1,
    use_cupy: bool | None = None,
) -> list[np.ndarray]:
    """
    Rotate a ao_instru array by multiple angles.

    Uses scipy.ndimage.rotate to create rotated versions of the input ao_instru.
    The rotation is performed around the center of the array with no reshaping,
    so the output arrays have the same shape as the input.

    Parameters
    ----------
    pupil_array : np.ndarray
        2D array representing the ao_instru transmission.
    angles : np.ndarray or None, optional
        Array of rotation angles in degrees. Default is np.arange(360),
        i.e., one rotation per degree from 0 to 359.
    order : int, optional
        Interpolation order for the rotation (0-5). Default is 1 (bilinear).
        - 0: nearest-neighbor
        - 1: bilinear
        - 3: cubic

    Returns
    -------
    list[np.ndarray]
        List of rotated ao_instru arrays, one for each angle.

    Examples
    --------
    >>> from ao_instrument import get_ao_instrument, rotate_pupil
    >>> ao_instru = get_ao_instrument("vlt", n_pix=128, wavelength=2.2e-6,
    ...                   angular_pixel_scale=13e-3 / 206265)
    >>> rotated = rotate_pupil(ao_instru.pupil_array, angles=np.array([0, 90, 180, 270]))
    >>> print(f"Created {len(rotated)} rotated pupils")
    """
    if use_cupy is None:
        use_cupy = _is_cupy_array(pupil_array)
    xp = get_xp(use_cupy)
    ndi = get_ndimage(use_cupy)

    if angles is None:
        angles = xp.arange(360)

    rotated_pupils = []
    for angle in angles:
        rotated = ndi.rotate(
            pupil_array,
            angle,
            reshape=False,
            order=order,
            mode="constant",
            cval=0.0,
        )
        rotated_pupils.append(rotated)

    return xp.array(rotated_pupils)


# =============================================================================
# AO_instrument class
# =============================================================================


class AO_instrument:
    """
    Container for a telescope ao_instru array and its physical pixel scale.

    Attributes
    ----------
    pupil_array : np.ndarray
        2D array representing the ao_instru transmission (0 = blocked, 1 = open).
    pixel_scale : float
        Physical size of each pixel in the ao_instru plane [meters/pixel].
    n_pix : int
        Number of pixels along each axis (assumes square array).
    """

    def __init__(
        self,
        pupil_array: np.ndarray,
        pixel_scale: float,
        angles: np.ndarray | None = None,
        zernike_n_rad: int = 6,
        zernike_n_min: int = 1,
        n_frames: int = 2,
        use_cupy: bool = False,
    ) -> None:
        """
        Initialize an AO_instrument object.

        Parameters
        ----------
        pupil_array : np.ndarray
            2D array representing the ao_instru transmission function.
        pixel_scale : float
            Physical sampling step in the ao_instru plane [meters/pixel].
        angles : np.ndarray or None, optional
            Rotation angles in degrees for precomputing rotated pupils.
            Default is np.array([0]).
        zernike_n_rad : int, optional
            Maximum radial order for Zernike modes (inclusive). Default is 6.
        zernike_n_min : int, optional
            Minimum radial order for Zernike modes (inclusive). Default is 1.
        use_cupy : bool, optional
            If True, use CuPy arrays and cupyx.scipy for computations. Default is False.
        """
        self.use_cupy = use_cupy
        self.xp = get_xp(use_cupy)

        self.pupil_array = self.xp.asarray(pupil_array)
        self.pixel_scale = pixel_scale
        self.n_pix = pupil_array.shape[0]
        self.n_frames = int(n_frames)

        # Spatial coordinate meshes (pixels)
        idx = self.xp.arange(self.n_pix, dtype=self.xp.float64)
        center = (self.n_pix - 1) / 2.0
        x_pix = idx - center
        y_pix = idx - center
        self.xx_pix, self.yy_pix = self.xp.meshgrid(x_pix, y_pix)
        self.rho_pix = self.xp.sqrt(self.xx_pix**2 + self.yy_pix**2)
        self.theta_pix = self.xp.arctan2(self.yy_pix, self.xx_pix)

        # Spatial coordinate meshes (meters)
        self.xx_m = self.xx_pix * self.pixel_scale
        self.yy_m = self.yy_pix * self.pixel_scale
        self.rho_m = self.rho_pix * self.pixel_scale
        self.theta_m = self.theta_pix

        # Frequency coordinate meshes (cycles per pixel)
        fx_pix = self.xp.fft.fftfreq(self.n_pix, d=1.0)
        fy_pix = self.xp.fft.fftfreq(self.n_pix, d=1.0)
        self.fx_pix, self.fy_pix = self.xp.meshgrid(fx_pix, fy_pix)
        self.fr_pix = self.xp.sqrt(self.fx_pix**2 + self.fy_pix**2)
        self.ftheta_pix = self.xp.arctan2(self.fy_pix, self.fx_pix)

        # Larger frequency coordinate meshes (cycles per meter) 
        fx_pix_large = self.xp.fft.fftfreq(self.n_pix * 2, d=1.0)
        fy_pix_large = self.xp.fft.fftfreq(self.n_pix * 2, d=1.0)
        self.fx_pix_large, self.fy_pix_large = self.xp.meshgrid(fx_pix_large, fy_pix_large)
        self.fr_pix_large = self.xp.sqrt(self.fx_pix_large**2 + self.fy_pix_large**2)
        self.ftheta_pix_large = self.xp.arctan2(self.fy_pix_large, self.fx_pix_large)

        # Frequency coordinate meshes (cycles per meter)
        fx_m = self.xp.fft.fftfreq(self.n_pix, d=self.pixel_scale)
        fy_m = self.xp.fft.fftfreq(self.n_pix, d=self.pixel_scale)
        self.fx_m, self.fy_m = self.xp.meshgrid(fx_m, fy_m)
        self.fr_m = self.xp.sqrt(self.fx_m**2 + self.fy_m**2)
        self.ftheta_m = self.xp.arctan2(self.fy_m, self.fx_m)

        if angles is None:
            angles = self.xp.array([0])
        else:
            angles = self.xp.asarray(angles)

        self.angles = angles
        self.rotated_pupils = rotate_pupil(
            self.pupil_array,
            angles=self.angles,
            use_cupy=self.use_cupy,
        )
        self.rotated_islands = detect_islands_rotated(
            self.rotated_pupils,
            use_cupy=self.use_cupy,
        )
        self.lwe_modes_rotated = compute_lwe_modes_rotated(self.rotated_islands)
        self.zernike_modes = compute_zernike_modes(
            self,
            n_rad=zernike_n_rad,
            n_min=zernike_n_min,
        )

    def __repr__(self) -> str:
        return (
            f"AO_instrument(n_pix={self.n_pix}, "
            f"pixel_scale={self.pixel_scale:.4e} m/pix, "
            f"extent={self.n_pix * self.pixel_scale:.3f} m)"
        )


# =============================================================================
# Rebinning utilities
# =============================================================================

# Source - https://stackoverflow.com/a/72848311
# Posted by Pierre Vermot
# Retrieved 2026-02-02, License - CC BY-SA 4.0


def _get_row_compressor(
    old_dimension: int,
    new_dimension: int,
    xp=np,
) -> np.ndarray:
    """
    Create a row compression matrix for rebinning.

    This matrix, when left-multiplied with an array, compresses rows from
    old_dimension to new_dimension while preserving the mean value through
    proper weighting of partial bins.

    Parameters
    ----------
    old_dimension : int
        Original number of rows (must be >= new_dimension).
    new_dimension : int
        Target number of rows after compression.

    Returns
    -------
    np.ndarray
        Compression matrix of shape (new_dimension, old_dimension).

    Raises
    ------
    ValueError
        If dimensions are not positive or new_dimension > old_dimension.
    """
    if old_dimension <= 0 or new_dimension <= 0:
        raise ValueError("Dimensions must be positive integers")
    if new_dimension > old_dimension:
        raise ValueError(
            f"new_dimension ({new_dimension}) must be <= old_dimension ({old_dimension})"
        )

    dim_compressor = xp.zeros((new_dimension, old_dimension))
    bin_size = float(old_dimension) / new_dimension
    next_bin_break = bin_size
    which_row = 0
    which_column = 0

    while which_row < dim_compressor.shape[0] and which_column < dim_compressor.shape[1]:
        if round(next_bin_break - which_column, 10) >= 1:
            dim_compressor[which_row, which_column] = 1
            which_column += 1
        elif abs(next_bin_break - which_column) < 1e-10:
            which_row += 1
            next_bin_break += bin_size
        else:
            partial_credit = next_bin_break - which_column
            dim_compressor[which_row, which_column] = partial_credit
            which_row += 1
            if which_row < dim_compressor.shape[0]:
                dim_compressor[which_row, which_column] = 1 - partial_credit
            which_column += 1
            next_bin_break += bin_size

    dim_compressor /= bin_size
    return dim_compressor


def _rebin(array: np.ndarray, new_shape: tuple[int, int]) -> np.ndarray:
    """
    Rebin a 2D array to a smaller shape while preserving the mean value.

    Uses matrix multiplication with compression matrices to properly weight
    contributions from partial bins.

    Parameters
    ----------
    array : np.ndarray
        Input 2D array to rebin.
    new_shape : tuple[int, int]
        Target shape (ny, nx). Must be <= input shape in both dimensions.

    Returns
    -------
    np.ndarray
        Rebinned array with shape new_shape.

    Raises
    ------
    ValueError
        If new_shape exceeds input array dimensions.
    """
    if new_shape[0] > array.shape[0] or new_shape[1] > array.shape[1]:
        raise ValueError(
            f"new_shape {new_shape} must be <= array shape {array.shape} in both dimensions"
        )

    xp = get_xp(_is_cupy_array(array))
    row_compressor = _get_row_compressor(array.shape[0], new_shape[0], xp=xp)
    col_compressor = _get_row_compressor(array.shape[1], new_shape[1], xp=xp).T

    return xp.asarray(row_compressor @ array @ col_compressor)


# =============================================================================
# Pupil generation
# =============================================================================


def _generate_pupil_array(
    aperture_generator: Callable,
    grid_size: int,
    diameter_m: float,
) -> np.ndarray:
    """
    Generate a telescope ao_instru array using hcipy.

    Parameters
    ----------
    aperture_generator : Callable
        An hcipy aperture generator function (e.g., hcipy.make_vlt_aperture).
    grid_size : int
        Size of the ao_instru grid (grid_size x grid_size pixels).
    diameter_m : float
        Physical diameter of the telescope primary mirror in meters.

    Returns
    -------
    np.ndarray
        2D array containing the ao_instru mask (0 = blocked, 1 = open).
    """
    pupil_grid = hcipy.make_pupil_grid(grid_size, diameter=diameter_m)
    telescope_pupil_generator = aperture_generator(normalized=False)
    return np.array(telescope_pupil_generator(pupil_grid).shaped)


def _generate_obstructed_circular_pupil_array(
    grid_size: int,
    diameter_m: float,
    central_obscuration_diameter_m: float,
) -> np.ndarray:
    """Generate a circular pupil with a central obscuration and no spiders."""
    pupil_grid = hcipy.make_pupil_grid(grid_size, diameter=diameter_m)
    central_obscuration_ratio = float(central_obscuration_diameter_m) / float(diameter_m)
    aperture_generator = hcipy.make_obstructed_circular_aperture(
        diameter_m,
        central_obscuration_ratio,
    )
    return np.array(aperture_generator(pupil_grid).shaped)


def compute_pupil_pixel_scale(
    n_pix: int,
    wavelength: float,
    angular_pixel_scale: float,
) -> float:
    """
    Compute the physical pixel scale in the ao_instru plane for FFT-based PSF generation.

    For an FFT-based optical propagation, the ao_instru and PSF arrays have the same
    number of pixels. The relationship between ao_instru pixel scale (Δx), wavelength
    (λ), array size (N), and angular pixel scale (Δθ) is:

        Δx = λ / (N × Δθ)

    Parameters
    ----------
    n_pix : int
        Number of pixels on a side (same for ao_instru and PSF arrays).
    wavelength : float
        Wavelength in meters.
    angular_pixel_scale : float
        Angular sampling of the PSF in radians per pixel.

    Returns
    -------
    float
        Physical sampling step in the ao_instru plane [meters/pixel].

    Examples
    --------
    >>> # H-band (1.6 μm), 256x256 array, 5 mas/pixel
    >>> scale = compute_pupil_pixel_scale(256, 1.6e-6, 5e-3 / 206265)
    >>> print(f"{scale:.4f} m/pixel")
    """
    return wavelength / (n_pix * angular_pixel_scale)


def _rebin_and_pad(
    pupil_array: np.ndarray,
    pixel_scale: float,
    rebin_factor: int,
    target_n_pix: int,
) -> tuple[np.ndarray, float]:
    """
    Rebin a high-resolution ao_instru and pad to the target array size.

    Parameters
    ----------
    pupil_array : np.ndarray
        High-resolution ao_instru array to rebin.
    pixel_scale : float
        Physical sampling step of the high-resolution ao_instru [meters/pixel].
    rebin_factor : int
        Integer factor by which to rebin (must evenly divide ao_instru size).
    target_n_pix : int
        Final array size after padding.

    Returns
    -------
    tuple[np.ndarray, float]
        Rebinned and padded ao_instru array and its new pixel scale.

    Raises
    ------
    ValueError
        If the rebinned ao_instru is larger than target_n_pix.
    """
    xp = get_xp(_is_cupy_array(pupil_array))

    # Rebin the ao_instru array
    n_pix = pupil_array.shape[0]
    new_size = n_pix // rebin_factor
    rebinned_array = _rebin(pupil_array, (new_size, new_size))
    new_pixel_scale = pixel_scale * rebin_factor

    # Check that rebinned ao_instru fits in target
    if new_size > target_n_pix:
        raise ValueError(
            f"Rebinned ao_instru size ({new_size}) exceeds target size ({target_n_pix}). "
            f"Increase n_pix or decrease the field of view."
        )

    # Calculate padding to reach target_n_pix (handle odd differences)
    pad_total = target_n_pix - new_size
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before  # Handles odd padding correctly

    padded_array = xp.pad(
        rebinned_array,
        pad_width=((pad_before, pad_after), (pad_before, pad_after)),
        mode="constant",
        constant_values=0,
    )

    return padded_array, new_pixel_scale


def get_ao_instrument(
    name: str,
    n_pix: int,
    wavelength: float,
    angular_pixel_scale: float,
    max_high_res_pixel_scale: float = DEFAULT_MAX_HIGH_RES_PIXEL_SCALE,
    angles: np.ndarray | None = None,
    zernike_n_rad: int = 6,
    zernike_n_min: int = 1,
    n_frames: int = 2,
    use_cupy: bool = False,
) -> AO_instrument:
    """
    Generate a telescope ao_instru array suitable for FFT-based PSF computation.

    This function creates a ao_instru mask at the correct physical pixel scale such
    that applying an FFT will produce a PSF with the specified angular pixel scale.

    The workflow is:
    1. Compute the required ao_instru pixel scale from FFT sampling constraints
    2. Generate a high-resolution ao_instru with pixel scale fine enough to resolve
       telescope structures (segment gaps, spiders, etc.)
    3. Rebin the high-resolution ao_instru to the target pixel scale
    4. Pad the array to the requested size (n_pix × n_pix)

    Parameters
    ----------
    name : str
        Telescope name. One of: 'elt', 'gmt', 'tmt', 'vlt', 'hst', 'jwst',
        'keck', 'subaru', 'scexao'.
    n_pix : int
        Size of the output ao_instru array (n_pix × n_pix pixels).
    wavelength : float
        Wavelength in meters.
    angular_pixel_scale : float
        Desired angular pixel scale of the PSF in radians per pixel.
        Tip: to convert from arcseconds, divide by 206265.
    max_high_res_pixel_scale : float, optional
        Maximum pixel scale for the high-resolution ao_instru generation in meters.
        Smaller values better resolve fine structures (segment gaps, spiders)
        but require more memory. Default is 2e-3 m (2 mm), which adequately
        resolves structures in segmented mirrors like ELT and JWST.
    angles : np.ndarray or None, optional
        Rotation angles in degrees for precomputing rotated pupils.
        Default is np.array([0]).
    zernike_n_rad : int, optional
        Maximum radial order for Zernike modes (inclusive). Default is 6.
    zernike_n_min : int, optional
        Minimum radial order for Zernike modes (inclusive). Default is 1.
    use_cupy : bool, optional
        If True, use CuPy arrays and cupyx.scipy for computations. Default is False.

    Returns
    -------
    AO_instrument
        AO_instrument object containing the ao_instru array and its physical pixel scale.

    Raises
    ------
    KeyError
        If the telescope name is not recognized.
    ValueError
        If the ao_instru would be larger than the requested n_pix.

    Examples
    --------
    >>> # VLT ao_instru for H-band imaging at 5 mas/pixel on a 256x256 grid
    >>> ao_instru = get_ao_instrument("vlt", n_pix=256, wavelength=1.6e-6,
    ...                   angular_pixel_scale=5e-3 / 206265)
    >>> psf = np.abs(np.fft.fftshift(np.fft.fft2(ao_instru.pupil_array)))**2
    """
    if name not in TELESCOPE_METADATA:
        available = ", ".join(TELESCOPE_METADATA.keys())
        raise KeyError(f"Unknown telescope '{name}'. Available: {available}")

    metadata = TELESCOPE_METADATA[name]
    aperture_generator = metadata["generator"]
    diameter_m = metadata["diameter_m"]

    # Step 1: Compute required ao_instru pixel scale from FFT constraints
    target_pixel_scale = compute_pupil_pixel_scale(n_pix, wavelength, angular_pixel_scale)

    # Step 2: Determine rebin factor to achieve target pixel scale
    # We need: high_res_pixel_scale * rebin_factor = target_pixel_scale
    # And: high_res_pixel_scale <= max_high_res_pixel_scale
    # So: rebin_factor >= target_pixel_scale / max_high_res_pixel_scale
    rebin_factor = int(np.ceil(target_pixel_scale / max_high_res_pixel_scale))
    rebin_factor = max(1, rebin_factor)  # At least 1

    # Compute the actual high-resolution pixel scale (exactly divides target)
    high_res_pixel_scale = target_pixel_scale / rebin_factor

    # Step 3: Determine grid size to cover the telescope diameter
    # Add a small margin to ensure full coverage
    high_res_grid_size = int(np.ceil(diameter_m / high_res_pixel_scale))

    # Ensure grid size is divisible by rebin_factor for clean rebinning
    remainder = high_res_grid_size % rebin_factor
    if remainder != 0:
        high_res_grid_size += rebin_factor - remainder

    xp = get_xp(use_cupy)

    # Step 4: Generate high-resolution ao_instru
    high_res_pupil_array = _generate_pupil_array(
        aperture_generator, high_res_grid_size, diameter_m
    )
    if use_cupy:
        high_res_pupil_array = xp.asarray(high_res_pupil_array)

    # Step 5: Rebin and pad to target size
    low_res_array, low_res_pixel_scale = _rebin_and_pad(
        high_res_pupil_array,
        high_res_pixel_scale,
        rebin_factor,
        n_pix,
    )

    return AO_instrument(
        low_res_array,
        low_res_pixel_scale,
        angles=angles,
        zernike_n_rad=zernike_n_rad,
        zernike_n_min=zernike_n_min,
        n_frames=n_frames,
        use_cupy=use_cupy,
    )


def get_obstructed_circular_ao_instrument(
    diameter_m: float,
    central_obscuration_diameter_m: float,
    n_pix: int,
    wavelength: float,
    angular_pixel_scale: float,
    max_high_res_pixel_scale: float = DEFAULT_MAX_HIGH_RES_PIXEL_SCALE,
    angles: np.ndarray | None = None,
    zernike_n_rad: int = 6,
    zernike_n_min: int = 1,
    n_frames: int = 2,
    use_cupy: bool = False,
) -> AO_instrument:
    """Generate an obstructed circular AO instrument with no spiders.

    This is useful for building a VLT-like reference pupil that preserves the
    primary diameter and central obscuration while removing spider vanes.
    """
    target_pixel_scale = compute_pupil_pixel_scale(n_pix, wavelength, angular_pixel_scale)

    rebin_factor = int(np.ceil(target_pixel_scale / max_high_res_pixel_scale))
    rebin_factor = max(1, rebin_factor)
    high_res_pixel_scale = target_pixel_scale / rebin_factor

    high_res_grid_size = int(np.ceil(diameter_m / high_res_pixel_scale))
    remainder = high_res_grid_size % rebin_factor
    if remainder != 0:
        high_res_grid_size += rebin_factor - remainder

    xp = get_xp(use_cupy)
    high_res_pupil_array = _generate_obstructed_circular_pupil_array(
        high_res_grid_size,
        diameter_m,
        central_obscuration_diameter_m,
    )
    if use_cupy:
        high_res_pupil_array = xp.asarray(high_res_pupil_array)

    low_res_array, low_res_pixel_scale = _rebin_and_pad(
        high_res_pupil_array,
        high_res_pixel_scale,
        rebin_factor,
        n_pix,
    )

    return AO_instrument(
        low_res_array,
        low_res_pixel_scale,
        angles=angles,
        zernike_n_rad=zernike_n_rad,
        zernike_n_min=zernike_n_min,
        n_frames=n_frames,
        use_cupy=use_cupy,
    )


