"""Use compact ray storage to spend a modest extra budget on angular sampling.

The original cavity bound and equal-weight Fibonacci rule are kept. The density
scale is corrected to remaining neutral atoms by default.
Every selected angular count is multiplied by ``angular_boost`` and rounded up,
up to the explicit actual ``max_angles`` cap. This is angular refinement, not
an improvement of the opaque-voxel spatial model.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from threading import RLock

import numpy as np

from speed_core import module


_path = Path(__file__).resolve().parents[1] / "pc_reconsidered/hybrid.py"
_spec = importlib.util.spec_from_file_location("_pc_budgeted_events", _path)
_backend = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_backend)
_estimate = _backend.estimate_angles
_lock = RLock()


def warmup():
    """Compile the exact compact core before comparing steady-state timings."""
    _backend.library = module("compact").library


def solve(source, capacity, initial_ionized=None, *, angular_boost=1.25,
          effective_density="remaining", **options):
    """Conserve the same photons with increased angular resolution.

    ``max_angles`` remains the maximum actual count. Set angular_boost=1 for a
    bitwise-equivalent version of the original solver with 25% smaller rays
    when also using effective_density='total'.
    Integer widths fall back automatically for unsupported compact layouts.
    """
    if isinstance(angular_boost, bool) or not np.isfinite(angular_boost) or angular_boost < 1:
        raise ValueError("angular_boost must be finite and at least one")
    if effective_density not in ("total", "remaining"):
        raise ValueError("effective_density must be total or remaining")
    cap = options.get("max_angles", 12288)
    if (isinstance(cap, bool) or not isinstance(cap, (int, np.integer))
            or not 1 <= cap <= np.iinfo(np.int32).max):
        raise ValueError("max_angles is outside the supported positive integer range")

    def boosted(source, capacity, remaining, **kwargs):
        density = remaining if effective_density == "remaining" else capacity
        levels, counts, radius, wanted = _estimate(source, density, remaining, **kwargs)
        limit = kwargs.get("max_angles", 12288)
        # Saturate before multiplication to avoid overflowing a large finite
        # boost. Counts remain positive; equal capped levels can share rays.
        values = np.ceil(np.minimum(counts.astype(float), limit/angular_boost)
                         * angular_boost).astype(np.int32)
        values = np.minimum(values, limit)
        unique, inverse = np.unique(values, return_inverse=True)
        return (np.asarray(inverse[levels], dtype=np.int32),
                np.asarray(unique, dtype=np.int32), radius,
                np.minimum(wanted, np.finfo(float).max/angular_boost)*angular_boost)

    with _lock:
        warmup()
        _backend.estimate_angles = boosted
        try:
            result = _backend.solve(source, capacity, initial_ionized, **options)
        finally:
            _backend.estimate_angles = _estimate
    result["audit"].update(angular_boost=float(angular_boost),
                           effective_density=effective_density,
                           ray_storage="compact with automatic wide fallback")
    return result
