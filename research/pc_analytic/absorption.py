"""Exact no-recombination absorption of a monochromatic plane-parallel fluence.

``photons`` and ``neutral`` are counts in the same units; ``coefficient`` is
the optical depth per unit neutral count (physical sigma / ray-tube area,
including the common count normalization). This is the saturable-absorber
fluence solution, not a new microscopic cross section. Multiple directions
can share it only under the declared equal-column, simultaneous-dose closure.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def absorbed(
    photons: ArrayLike, neutral: ArrayLike, coefficient: ArrayLike
) -> np.ndarray | float:
    """Return absorbed counts, bounded by both photons and available atoms.

    The implementation avoids exponentiating positive optical depths and
    retains small absorptions with expm1/log1p. All arguments broadcast.

    Parameters
    ----------
    photons, neutral : array_like
        Nonnegative photon and initial neutral atom counts in common units.
    coefficient : array_like
        Nonnegative optical depth per unit neutral count, sigma / beam area
        multiplied by the common count normalization.

    Returns
    -------
    ndarray or float
        Absorbed photon counts; equal to the newly ionized atom counts.
    """
    p, h, k = np.broadcast_arrays(
        np.asarray(photons, dtype=float),
        np.asarray(neutral, dtype=float),
        np.asarray(coefficient, dtype=float),
    )
    if (
        not np.isfinite(p).all()
        or not np.isfinite(h).all()
        or not np.isfinite(k).all()
        or np.any(p < 0)
        or np.any(h < 0)
        or np.any(k < 0)
    ):
        raise ValueError(
            "Photon counts, neutral counts and coefficient must be finite and nonnegative"
        )
    result = np.zeros_like(p)
    active = (p > 0) & (h > 0) & (k > 0)
    if np.any(active):
        pp, hh, kk = p[active], h[active], k[active]
        with np.errstate(over="ignore", invalid="ignore"):
            a = kk * np.minimum(pp, hh)
            b = kk * np.maximum(pp, hh)
        small = a < 0.5
        val = np.empty_like(a)
        product = np.expm1(-a[small]) * np.expm1(-b[small])
        val[small] = -np.log1p(-product) / kk[small]
        # This equivalent expression preserves the opaque limit, even when
        # the product coefficient * count overflows but the count is finite.
        with np.errstate(over="ignore"):
            difference = kk[~small] * np.abs(pp[~small] - hh[~small])
        correction = (
            np.log1p(np.exp(-difference) * (-np.expm1(-a[~small]))) / kk[~small]
        )
        val[~small] = np.minimum(pp[~small], hh[~small]) - correction
        result[active] = np.minimum(np.maximum(val, 0), np.minimum(pp, hh))
    return float(result) if result.ndim == 0 else result


def shared_absorption(
    directional_photons: ArrayLike, neutral: ArrayLike, coefficient: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the equal-column closure to simultaneously incident directions.

    Returns surviving photons and remaining neutral counts. All directions
    see one shared atom inventory. Their relative incident rates are assumed
    constant during the exposure; photons do not change direction. This is
    not exact for arbitrary unresolved crossing-ray geometry or illumination
    histories.

    Parameters
    ----------
    directional_photons : array_like
        Incident counts, shape (directions, *neutral.shape).
    neutral : array_like
        Shared neutral atom count in each cell.
    coefficient : array_like
        Optical-depth coefficient, broadcastable to neutral.shape.

    Returns
    -------
    surviving : ndarray
        Photon counts preserving their original angular bins.
    remaining_neutral : ndarray
        Neutral atom counts after all directions' shared absorption.
    """
    photons = np.asarray(directional_photons, dtype=float)
    neutral = np.asarray(neutral, dtype=float)
    if photons.ndim != neutral.ndim + 1 or photons.shape[1:] != neutral.shape:
        raise ValueError(
            "directional_photons must have shape (directions, *neutral.shape)"
        )
    if not np.isfinite(photons).all() or np.any(photons < 0):
        raise ValueError("Directional photon counts must be finite and nonnegative")
    total = photons.sum(axis=0)
    amount = absorbed(total, neutral, coefficient)
    fraction = np.divide(amount, total, out=np.zeros_like(total), where=total > 0)
    return photons * (1 - fraction), neutral - amount
