"""Native source features for the learned exposure emulator."""

from __future__ import annotations

import ctypes
import os
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

from research.pc_matched.rt import directions


@lru_cache(maxsize=1)
def _kernel() -> ctypes.CDLL:
    source = Path(__file__).with_suffix(".cpp")
    with tempfile.TemporaryDirectory(prefix="pc_exposure_features_") as directory:
        library = Path(directory) / "features.so"
        subprocess.run(
            [
                os.environ.get("CXX", "g++"),
                "-O3",
                "-std=c++17",
                "-fPIC",
                "-shared",
                str(source),
                "-o",
                str(library),
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
        kernel = ctypes.CDLL(str(library))
    kernel.time_features.restype = None
    kernel.time_features.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        *([ctypes.c_void_p] * 6),
        *([ctypes.c_double] * 3),
        *([ctypes.c_void_p] * 4),
    ]
    return kernel


@lru_cache(maxsize=16)
def _offsets(dimension: int, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build donor-centred voxel prefixes inside the periodic source image."""
    vectors, weights = directions(count)
    starts, offsets = [0], []
    for vector in vectors:
        crossing = np.divide(
            0.5, np.abs(vector), where=vector != 0, out=np.full(3, np.inf)
        )
        spacing = 2 * crossing
        shift = np.zeros(3, dtype=np.int32)
        while True:
            step = crossing <= crossing.min() + 1e-10
            shift[step] += np.sign(vector[step]).astype(np.int32)
            crossing[step] += spacing[step]
            if np.any((shift < -(dimension // 2)) | (shift > (dimension - 1) // 2)):
                break
            offsets.append(tuple(shift))
        starts.append(len(offsets))
    return np.array(starts, dtype=np.int32), np.array(offsets, dtype=np.int32), weights


def exposure_features(
    source: np.ndarray,
    atoms: np.ndarray,
    *,
    depth: int,
    gains: tuple[float, float],
    onset: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return normalized native source, near/far dose features and latent onset.

    This is a finite network of shared fixed-stencil layers. Angles specify
    input receptive fields. A layer reads a frozen latent onset cube and pools
    native source rates according to the latest onset on each donor prefix.
    These descriptors carry no evolved photon reservoirs.
    """
    if (
        np.iscomplexobj(source)
        or np.iscomplexobj(atoms)
        or source.ndim != 3
        or len(set(source.shape)) != 1
        or source.shape[0] < 8
        or source.shape != atoms.shape
        or not np.isfinite(source).all()
        or not np.isfinite(atoms).all()
        or np.any(source < 0)
        or np.any(atoms <= 0)
        or type(depth) is not int
        or depth < 1
        or len(gains) != 2
        or not np.isfinite(gains).all()
        or min(gains) <= 0
    ):
        raise ValueError("Invalid native source/atom cubes or exposure layer settings")
    with np.errstate(over="ignore"):
        normalizer = np.max(atoms)
        s = np.ascontiguousarray(source / normalizer, dtype=np.float64)
        a = np.ascontiguousarray(atoms / normalizer, dtype=np.float64)
    if not np.isfinite(s).all() or np.any(a <= 0):
        raise ValueError("Native source and atom dynamic range is too large")
    if onset is None:
        onset = np.divide(a, s, out=np.ones_like(s), where=s > a)
    if onset.shape != s.shape or not np.isfinite(onset).all():
        raise ValueError("Initial exposure estimates must match the input grid")
    current = np.ascontiguousarray(np.clip(onset, 0, 1), dtype=np.float64)
    updated, near, far = [np.empty_like(s) for _ in range(3)]
    escape = np.empty(1)
    start, offsets, weights = _offsets(len(s), 96)
    evaluate = _kernel().time_features
    for _ in range(depth):
        evaluate(
            len(s),
            len(weights),
            *[array.ctypes.data for array in (start, offsets, weights, s, a, current)],
            *gains,
            0.0,
            *[array.ctypes.data for array in (updated, near, far, escape)],
        )
        if np.array_equal(current, updated):
            break
        current, updated = updated, current
    escape_per_atom = float(escape[0] / a.sum())
    return s / a, near / a, far / a, current, escape_per_atom
