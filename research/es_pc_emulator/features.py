"""Input-only spatial features for native ES to C2-Ray field corrections.

Directional periodic patches retain local geometry; fixed-comoving-scale
averages add context. No teacher state or teacher-derived budget enters these
features. This portable MLP baseline is not a convolutional neural network.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.ndimage import gaussian_filter

FEATURE_SETS = ("es", "es-density", "es-density-source")


def _cube(value, name, *, fraction=False):
    array = np.asarray(value, dtype=np.float64)
    if (array.ndim != 3 or len(set(array.shape)) != 1 or array.shape[0] < 2
            or not np.isfinite(array).all()):
        raise ValueError(f"{name} must be a finite cubic grid")
    if np.any(array < 0) or (fraction and np.any(array > 1)):
        raise ValueError(f"{name} outside its physical bounds")
    return array


def build_features(xhi_es, radius_es_cMpc, *, redshift, box_len_cMpc,
                   feature_set="es", density=None, photons=None,
                   patch_radius=1, radii_cMpc=(6.25, 12.5, 25.0, 50.0)):
    """Return cell-by-feature spatial inputs and their ordered names.

    ``xhi_es`` and the largest ES crossing radius are three-dimensional fields,
    not global means. Density is in mean-nuclei-per-cell units and photons are
    cumulative emitted counts in the same units. Radius and box length are
    comoving Mpc. Different physical grids must retain the cell-width feature.
    """
    xhi = _cube(xhi_es, "ES neutral fraction", fraction=True)
    radius = _cube(radius_es_cMpc, "ES crossing radius")
    if radius.shape != xhi.shape:
        raise ValueError("ES fields must share one grid")
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")
    if (isinstance(patch_radius, bool) or not isinstance(patch_radius, int)
            or not 0 <= patch_radius <= 3):
        raise ValueError("patch_radius must be an integer from zero through three")
    if not np.isfinite([redshift, box_len_cMpc]).all() or redshift < 0 or box_len_cMpc <= 0:
        raise ValueError("Nonnegative redshift and positive comoving box length required")
    radii = tuple(float(r) for r in radii_cMpc)
    if (any(not np.isfinite(r) or r <= 0 for r in radii)
            or len(set(radii)) != len(radii)):
        raise ValueError("Smoothing radii must be distinct positive cMpc values")
    dx = float(box_len_cMpc) / xhi.shape[0]
    channels = {"xhi_es": xhi, "log1p_radius_cMpc": np.log1p(radius)}
    rho = None
    if feature_set != "es":
        rho = _cube(density, "Density")
        if rho.shape != xhi.shape or rho.mean() <= 0:
            raise ValueError("Density must share the ES grid and have positive mean")
        channels["log1p_density"] = np.log1p(rho)
    if feature_set == "es-density-source":
        emitted = _cube(photons, "Emitted photons")
        if emitted.shape != xhi.shape:
            raise ValueError("Photons must share the ES grid")
        channels["log1p_photons"] = np.log1p(emitted)
        channels["log1p_photons_per_atom"] = np.log1p(np.divide(
            emitted, rho, out=np.zeros_like(emitted), where=rho > 0))
    columns, names = [], []
    offsets = tuple(itertools.product(range(-patch_radius, patch_radius + 1), repeat=3))
    for name, field in channels.items():
        for offset in offsets:
            columns.append(np.roll(field, offset, axis=(0, 1, 2)).ravel())
            names.append(f"{name}:offset={offset}")
        for scale in radii:
            columns.append(gaussian_filter(field, scale / dx, mode="wrap").ravel())
            names.append(f"{name}:gaussian_sigma_cMpc={scale:g}")
    columns.extend((np.full(xhi.size, redshift), np.full(xhi.size, np.log1p(dx))))
    names.extend(("redshift", "log1p_dx_cMpc"))
    features = np.asarray(columns, dtype=np.float32).T.copy()
    if not np.isfinite(features).all():
        raise ValueError("Feature construction overflowed")
    return features, names


def from_pair(pair, **configuration):
    """Build features using only the declared inference fields in a pair."""
    metadata = pair["metadata"]
    return build_features(
        pair["xhi_es"], pair["radius_es_cMpc"],
        redshift=metadata["redshift"], box_len_cMpc=metadata["box_len_cMpc"],
        density=pair.get("density"), photons=pair.get("photons"), **configuration)
