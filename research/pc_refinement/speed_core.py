"""Exact arithmetic-path optimizations of the existing angular event kernel.

The index variant avoids integer remainders while a cursor remains in-domain.
The filtered variant additionally uses double estimates to identify separated
face crossings, then computes the winning path parameter using the unchanged
long-double division. Near ties/non-normal intermediates use the original code.
This changes neither quadrature, event ordering nor the opacity model.
"""
from __future__ import annotations

import ctypes as ct
import importlib.util
from pathlib import Path
import subprocess
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "pc_reconsidered/hybrid.py"
_modules = {}


def module(variant="filtered"):
    if variant not in ("original", "index", "filtered", "compact"):
        raise ValueError("variant must be original, index, filtered, or compact")
    if variant not in _modules:
        spec = importlib.util.spec_from_file_location(f"_pc_speed_{variant}", BASE)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if variant != "original":
            output = Path(tempfile.mkdtemp(prefix="pc_speed_")) / "speed.so"
            subprocess.run([
                "g++", "-O3", "-std=c++17", "-shared", "-fPIC",
                f"-DPC_FAST_INDEX={int(variant in ('index', 'filtered'))}",
                f"-DPC_FILTERED_DDA={int(variant == 'filtered')}",
                f"-DPC_COMPACT_RAYS={int(variant == 'compact')}",
                str(HERE / "speed_core.cpp"), "-o", str(output),
            ], check=True)
            lib = ct.CDLL(str(output))
            lib.evolve_adaptive_events.argtypes = [
                ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_void_p,
                ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_void_p,
                ct.c_int, ct.c_void_p, ct.c_double, ct.c_double, ct.c_int,
                *([ct.c_void_p] * 8),
            ]
            lib.evolve_adaptive_events.restype = ct.c_int
            for name in ("hybrid_last_max_heap_nodes", "hybrid_last_heap_updates",
                         "hybrid_last_ray_allocated_bytes", "speed_last_fast_crossings",
                         "speed_last_fallback_crossings"):
                getattr(lib, name).argtypes = []
                getattr(lib, name).restype = ct.c_int64
            mod.library = lambda: lib
        mod.library()  # Warm compilation outside timed calls.
        _modules[variant] = mod
    return _modules[variant]


def solve(source, capacity, initial_ionized=None, *, variant="filtered", **kwargs):
    mod = module(variant)
    result = mod.solve(source, capacity, initial_ionized, **kwargs)
    result["audit"]["exact_optimization"] = variant
    if variant != "original":
        result["audit"]["fast_crossings"] = int(mod.library().speed_last_fast_crossings())
        result["audit"]["fallback_crossings"] = int(mod.library().speed_last_fallback_crossings())
    return result


def raw_run(source, remaining, directions, *, variant="filtered", periodic=False,
            maximum_distance=20., skip=True):
    """Expose custom directions for independent near-corner regression tests."""
    mod = module(variant)
    source = np.ascontiguousarray(source, dtype=np.float64)
    remaining = np.ascontiguousarray(remaining, dtype=np.float64)
    directions = np.ascontiguousarray(directions, dtype=np.float64)
    count = np.ascontiguousarray([len(directions)], dtype=np.int32)
    offset = np.zeros(1, dtype=np.int32)
    level = np.zeros(source.shape, dtype=np.int32)
    clearance = mod.clearance(remaining <= 0) if skip else np.zeros(source.shape, dtype=np.int32)
    arrays = [np.zeros_like(source) for _ in range(4)]
    angular = [np.zeros(len(directions)) for _ in range(3)]
    statistics = np.zeros(6)
    rc = mod.library().evolve_adaptive_events(
        source.shape[0], source.ctypes.data, remaining.ctypes.data, clearance.ctypes.data,
        1, level.ctypes.data, offset.ctypes.data, count.ctypes.data,
        len(directions), directions.ctypes.data, 1., maximum_distance, int(periodic),
        *[a.ctypes.data for a in arrays], *[a.ctypes.data for a in angular], statistics.ctypes.data,
    )
    if rc:
        raise RuntimeError(f"event kernel returned {rc}")
    return dict(arrays=arrays, angular=angular, statistics=statistics)
