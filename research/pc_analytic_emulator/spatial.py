"""Bounded local native-input descriptors for an optional spatial field head.

These scalar descriptors expose near-cell clumping and front/source alignment.
They respect periodic translations and the rotations/reflections of a cubic
grid. They do not perform transport, impose a photon budget, or estimate a
physical optical depth. Existing large-scale features remain useful alongside
them; a finite set of local moments cannot specify nonlocal radiation paths.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import convolve, gaussian_filter

DEFAULT_SPATIAL_SCALES = (1.0, 2.0)


def _inputs(
    source: ArrayLike, capacity: ArrayLike, native_es_xHI: ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = tuple(np.asarray(value) for value in (source, capacity, native_es_xHI))
    if any(np.iscomplexobj(array) for array in arrays):
        raise ValueError("Spatial descriptors require real native input arrays")
    photons, atoms, neutral = (np.asarray(array, dtype=np.float64) for array in arrays)
    if (
        photons.ndim != 3
        or len(set(photons.shape)) != 1
        or photons.shape[0] < 2
        or atoms.shape != photons.shape
        or neutral.shape != photons.shape
    ):
        raise ValueError("Native spatial inputs must share a cubic grid of side >= 2")
    if not np.isfinite(photons).all() or np.any(photons < 0):
        raise ValueError("Source photons must be finite and nonnegative")
    if not np.isfinite(atoms).all() or np.any(atoms <= 0):
        raise ValueError("Atom capacity must be finite and strictly positive")
    if not np.isfinite(neutral).all() or np.any((neutral < 0) | (neutral > 1)):
        raise ValueError("Native ES neutral fractions must lie in [0, 1]")
    return photons, atoms, neutral


def _fraction(numerator: ArrayLike, other: ArrayLike) -> np.ndarray:
    """Evaluate a/(a+b) for nonnegative counts without overflowing a+b."""
    first, second = np.broadcast_arrays(numerator, other)
    scale = np.maximum(first, second)
    a = np.divide(first, scale, out=np.zeros_like(scale), where=scale > 0)
    b = np.divide(second, scale, out=np.zeros_like(scale), where=scale > 0)
    return np.divide(a, a + b, out=np.zeros_like(a), where=(a + b) > 0)


def _shell_kernel(squared_radius: int) -> np.ndarray:
    kernel = np.zeros((3, 3, 3), dtype=np.float64)
    for offset in product((-1, 0, 1), repeat=3):
        if sum(component**2 for component in offset) == squared_radius:
            kernel[tuple(component + 1 for component in offset)] = 1
    return kernel / kernel.sum()


def spatial_features(
    source: ArrayLike,
    capacity: ArrayLike,
    native_es_xHI: ArrayLike,
    *,
    scales_cells: Sequence[float] = DEFAULT_SPATIAL_SCALES,
) -> tuple[NDArray[np.float32], list[str]]:
    """Describe local native source, atom and ES structure without teacher labels.

    Parameters
    ----------
    source, capacity, native_es_xHI
        Cubic arrays of cumulative photon counts S, strictly positive atom
        capacities A, and native ES neutral fraction. Counts share a common
        unit. Multiplying both count arrays by one positive factor leaves all
        returned descriptors unchanged up to floating-point roundoff.
    scales_cells
        Distinct positive Gaussian standard deviations in grid cells. The
        defaults add information at dx and 2 dx to existing physical-width
        features. Kernels wrap periodically and truncate at four sigma.

    Returns
    -------
    features, names
        C-order cell rows and deterministic names. The default 64 float32
        columns lie in [-1, 1]. Four compact fields are S/(S+mean(A)),
        A/(A+mean(A)), native xHI, and S/(S+A). For these fields the descriptors
        include face/edge/corner shell means and variances, dimensionless
        central-gradient norms, face contrasts, selected gradient alignments,
        and local Gaussian moments/covariances. Count ratios at each Gaussian
        scale smooth S and A *before* forming their ratio. The ES count
        discrepancy [S-A(1-xHI_ES)]/(S+A) is also supplied locally and smoothed;
        this discrepancy is a feature, not an absorbed or escaped reservoir.

    Notes
    -----
    Grid symmetries are exact mathematical properties of these descriptors;
    floating-point summation order can change values at roundoff level.
    Derivatives use cell coordinates, not physical lengths. A model must also
    receive cell width/box geometry through its existing features. These
    near-grid summaries do not imply resolution-independent predictions.
    """
    photons, atoms, neutral = _inputs(source, capacity, native_es_xHI)
    scales = tuple(float(scale) for scale in scales_cells)
    if (
        not scales
        or any(not np.isfinite(scale) or scale <= 0 for scale in scales)
        or len(scales) != len(set(scales))
    ):
        raise ValueError("Spatial scales must be nonempty, unique, finite and positive")
    maximum_atoms = float(np.max(atoms))
    mean_atoms = maximum_atoms * float(np.mean(atoms / maximum_atoms))
    photon_fraction = _fraction(photons, atoms)
    fields = {
        "source": _fraction(photons, mean_atoms),
        "density": _fraction(atoms, mean_atoms),
        "native_es_xHI": neutral,
        "photon_fraction": photon_fraction,
    }
    es_budget_contrast = photon_fraction - (1 - photon_fraction) * (1 - neutral)
    columns: list[np.ndarray] = []
    names: list[str] = []

    def append(name: str, values: np.ndarray) -> None:
        columns.append(values.ravel())
        names.append(f"spatial_{name}")

    for name, values in fields.items():
        append(name, values)
    append("es_budget_contrast", es_budget_contrast)

    faces = _shell_kernel(1)
    for squared_radius in (1, 2, 3):
        kernel = faces if squared_radius == 1 else _shell_kernel(squared_radius)
        for name, values in fields.items():
            mean = convolve(values, kernel, mode="wrap")
            variance = np.maximum(convolve(values**2, kernel, mode="wrap") - mean**2, 0)
            append(f"{name}_shell_r2_{squared_radius}_mean", mean)
            append(f"{name}_shell_r2_{squared_radius}_variance", variance)

    gradients = {}
    for name, values in fields.items():
        gradient = np.stack(
            [
                (np.roll(values, -1, axis) - np.roll(values, 1, axis)) * 0.5
                for axis in range(3)
            ]
        )
        gradients[name] = gradient
        append(f"{name}_gradient_norm", np.sqrt(np.sum(gradient**2, axis=0) / 3) * 2)
        append(f"{name}_face_contrast", convolve(values, faces, mode="wrap") - values)
    for first, second in (
        ("source", "native_es_xHI"),
        ("source", "density"),
        ("photon_fraction", "native_es_xHI"),
    ):
        ga, gb = gradients[first], gradients[second]
        norm = np.sqrt(np.sum(ga**2, axis=0) * np.sum(gb**2, axis=0))
        alignment = np.divide(
            np.sum(ga * gb, axis=0), norm, out=np.zeros_like(norm), where=norm > 0
        )
        append(f"{first}_{second}_gradient_alignment", np.clip(alignment, -1, 1))

    # Common normalization before filtering preserves the ratio while avoiding
    # overflow from finite but very large count units.
    maximum_count = max(maximum_atoms, float(np.max(photons)))
    normalized_source, normalized_atoms = photons / maximum_count, atoms / maximum_count
    for sigma in scales:
        suffix = f"sigma{sigma:g}_cells"
        means = {}
        for name, values in fields.items():
            mean = gaussian_filter(values, sigma, mode="wrap")
            means[name] = mean
            variance = np.maximum(
                gaussian_filter(values**2, sigma, mode="wrap") - mean**2, 0
            )
            append(f"{name}_{suffix}_mean", mean)
            append(f"{name}_{suffix}_variance", variance)
        for second in ("native_es_xHI", "density"):
            covariance = gaussian_filter(
                fields["source"] * fields[second], sigma, mode="wrap"
            )
            covariance -= means["source"] * means[second]
            append(f"source_{second}_{suffix}_covariance", covariance)
        append(
            f"photon_fraction_{suffix}_count_ratio",
            _fraction(
                gaussian_filter(normalized_source, sigma, mode="wrap"),
                gaussian_filter(normalized_atoms, sigma, mode="wrap"),
            ),
        )
        append(
            f"es_budget_contrast_{suffix}_mean",
            gaussian_filter(es_budget_contrast, sigma, mode="wrap"),
        )
    features = np.column_stack(columns)
    if not np.isfinite(features).all():
        raise FloatingPointError(
            "Spatial descriptor construction produced nonfinite values"
        )
    if np.any(np.abs(features) > 1 + 32 * np.finfo(float).eps):
        raise FloatingPointError("Spatial descriptor exceeded its mathematical bounds")
    return np.clip(features, -1, 1).astype(np.float32), names
