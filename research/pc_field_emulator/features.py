"""Build input-only multiscale features for a fixed-gas PC field emulator.

Gaussian widths can be specified in original-grid cells or fixed cMpc and use
periodic boundaries. Fixed physical widths give mixed-resolution training the
same feature definitions. Accuracy on another geometry still requires testing.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter


def _inputs(
    capacity: ArrayLike,
    photons: ArrayLike,
    redshift: float,
    box_len: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rho = np.asarray(capacity, dtype=np.float64)
    source = np.asarray(photons, dtype=np.float64)
    if rho.ndim != 3 or len(set(rho.shape)) != 1 or rho.shape != source.shape:
        raise ValueError(
            "Capacity and photons must share a cubic three-dimensional grid"
        )
    if rho.size == 0 or not np.all(np.isfinite(rho)) or np.any(rho < 0):
        raise ValueError("Capacity must be finite, nonnegative, and nonempty")
    if not np.all(np.isfinite(source)) or np.any(source < 0):
        raise ValueError("Photon counts must be finite and nonnegative")
    if not np.isfinite(np.mean(rho)) or np.mean(rho) <= 0:
        raise ValueError("The box must contain positive finite atom capacity")
    if not np.isfinite(redshift) or redshift < 0:
        raise ValueError("Redshift must be finite and nonnegative")
    if not np.isfinite(box_len) or box_len <= 0:
        raise ValueError("Box length must be finite and positive")
    return rho, source


def build_features(
    capacity: ArrayLike,
    photons: ArrayLike,
    redshift: float,
    box_len: float,
    radii_cells: Sequence[float] = (1, 2, 4, 8),
    *,
    radii_cMpc: Sequence[float] | None = None,
) -> tuple[NDArray[np.float32], list[str]]:
    """Construct local and periodic Gaussian-smoothed input features.

    Parameters
    ----------
    capacity, photons
        Fixed atom capacities and cumulative emitted photons on the same cubic
        grid, in the same count unit. No ionization target is accepted.
    redshift
        Snapshot redshift.
    box_len
        Comoving box length in cMpc.
    radii_cells
        Positive Gaussian standard deviations in original-grid cell widths.
        The unsmoothed fields are always included. Smoothing is truncated at
        four standard deviations and wraps across the periodic box.
    radii_cMpc
        Optional positive Gaussian standard deviations in cMpc. When supplied,
        these replace ``radii_cells``; each kernel uses ``sigma = radius / dx``.
        Physical-scale column names are shared across box lengths and grids.

    Returns
    -------
    features
        A float32 matrix with one C-order flattened cell per row. Density and
        source count are divided by mean atom capacity, then log1p transformed.
        The third feature at each scale is log1p(smoothed source/capacity), set
        to zero where capacity is zero. Source support there remains visible
        through the separate photon feature and zero-capacity indicator.
    names
        Ordered feature names. Final columns are a zero-capacity indicator,
        log1p of the global emitted photons per atom, redshift, and log1p of the
        physical cell width. All statistics are computed from inputs only.
    """
    rho, source = _inputs(capacity, photons, redshift, box_len)
    physical = radii_cMpc is not None
    radii = tuple(float(radius) for radius in (radii_cMpc if physical else radii_cells))
    if any(not np.isfinite(radius) or radius <= 0 for radius in radii):
        raise ValueError("Gaussian radii must be positive and finite")
    if len(set(radii)) != len(radii):
        raise ValueError("Gaussian radii must be unique")
    mean_capacity = float(np.mean(rho))
    dx = box_len / rho.shape[0]
    if not np.isfinite(dx) or dx <= 0:
        raise ValueError("Physical cell width must be positive and finite")
    features = np.empty((rho.size, 3 * (1 + len(radii)) + 4), dtype=np.float32)
    names: list[str] = []
    column = 0
    for radius in (0.0, *radii):
        sigma = radius / dx if physical else radius
        if not np.isfinite(sigma):
            raise ValueError("Gaussian radii overflow the grid cell width")
        density = rho if radius == 0 else gaussian_filter(rho, sigma, mode="wrap")
        emitted = source if radius == 0 else gaussian_filter(source, sigma, mode="wrap")
        unit = "cMpc" if physical and radius != 0 else "cells"
        suffix = f"sigma{radius:g}_{unit}"
        for values, name in (
            (density / mean_capacity, "log1p_density"),
            (emitted / mean_capacity, "log1p_photons"),
            (
                np.divide(
                    emitted, density, out=np.zeros_like(emitted), where=density > 0
                ),
                "log1p_photons_per_atom",
            ),
        ):
            features[:, column] = np.log1p(values).ravel()
            names.append(f"{name}_{suffix}")
            column += 1
    features[:, column] = (rho == 0).ravel()
    features[:, column + 1] = np.log1p(np.mean(source) / mean_capacity)
    features[:, column + 2] = redshift
    features[:, column + 3] = np.log1p(dx)
    names.extend(
        ["zero_capacity", "log1p_emitted_per_atom", "redshift", "log1p_dx_cMpc"]
    )
    if not np.all(np.isfinite(features)):
        raise ValueError("Feature construction overflowed; check input count units")
    return features, names


def physics_baseline(
    capacity: ArrayLike, photons: ArrayLike
) -> tuple[NDArray[np.float64], float]:
    """Construct a local fraction and an input-only residual activation scale.

    Parameters
    ----------
    capacity, photons
        Nonnegative finite atom capacities and cumulative source counts on the
        same cubic grid. Atom capacity must have a positive mean.

    Returns
    -------
    baseline
        ``clip(photons / capacity, 0, 1)``, with zero-capacity cells set to zero.
        This local estimate has not undergone the global photon projection.
    residual_scale
        Zero for a box with no emitted photons, otherwise one. In particular,
        sub-capacity source cells do not disable the learned residual: resolved
        rays can leave a cell before all of its atom capacity has been filled.
    """
    rho, source = _inputs(capacity, photons, 0.0, 1.0)
    local = np.zeros_like(rho)
    local[(rho > 0) & (source >= rho)] = 1.0
    np.divide(source, rho, out=local, where=(rho > 0) & (source < rho))
    return local, float(np.any(source > 0))
