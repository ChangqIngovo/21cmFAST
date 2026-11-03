"""Independent, conservative coeval radial allocation.

The inputs are additive photon and atom COUNTS on the same periodic cubic
grid. Gas is frozen; source budgets are cumulative; propagation is instantaneous.
This does not solve directional RT, shadowing, recombination or thermal history.
It has no dependence on SCRIPT and never rescales a field to fit a reference.
"""

from __future__ import annotations

import ctypes
import hashlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time

import numpy as np

_LIBRARY = None
_BUILD_IDENTITY = None
_BUILD_LOCK = threading.Lock()


def _library():
    global _LIBRARY, _BUILD_IDENTITY
    with _BUILD_LOCK:
        if _LIBRARY is not None:
            return _LIBRARY
        source = Path(__file__).with_name("allocation_kernel.cpp")
        compiler = shutil.which(os.environ.get("CXX", "g++"))
        if compiler is None:
            raise RuntimeError("A C++ compiler with OpenMP support is required (CXX or g++)")
        compiler = str(Path(compiler).resolve())
        version = subprocess.run([compiler, "--version"], check=True,
                                 capture_output=True, text=True, timeout=30).stdout
        identity = {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "compiler": compiler, "compiler_version": version,
                    "flags": ["-O3", "-std=c++17", "-shared", "-fPIC", "-fopenmp"]}
        directory = Path(tempfile.mkdtemp(prefix="pc_coeval_kernel_"))
        output = directory / "allocation.so"
        subprocess.run([compiler, *identity["flags"], str(source), "-o", str(output)],
                       check=True, capture_output=True, text=True, timeout=180)
        lib = ctypes.CDLL(str(output))
        lib.allocate_coeval.argtypes = [
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
            ctypes.c_double, ctypes.c_int, ctypes.c_uint64,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ]
        lib.allocate_coeval.restype = ctypes.c_int
        _LIBRARY, _BUILD_IDENTITY = lib, identity
        return lib


def build_identity():
    """Build once outside the source tree and return compiler/kernel identity."""
    _library()
    return dict(_BUILD_IDENTITY)


def solve(source, capacity, *, threads=1, tolerance=1e-8,
          max_iterations=1000, max_interactions=1_000_000_000_000):
    """Allocate cumulative photons without exceeding source or atom budgets.

    All sources propose simultaneously, and exact lattice-distance ties share
    according to available atom capacity. Remaining photons retain their
    original emitter. ``remaining_source`` includes physical surplus and any
    unresolved numerical tails. Limits return a conservative partial result,
    with ``audit['complete'] = False``; the caller must not label it completed.

    ``max_interactions`` bounds source-target visits across all passes. Workers
    reserve small visit chunks, so a limit can stop slightly before that bound.
    An unfinished round is discarded, leaving the last conservative state.
    Compilation is outside ``elapsed_seconds`` and ``kernel_seconds``.
    Nearly saturated, strongly overlapping regions can require many visits;
    the work limit must remain visible in any scientific comparison.
    """
    for name, value in (("threads", threads), ("max_iterations", max_iterations)):
        if type(value) is not int or value < (1 if name == "threads" else 0):
            raise ValueError(f"{name} must be a valid nonnegative integer (threads >= 1)")
        if value > 2**31-1:
            raise ValueError(f"{name} exceeds the kernel integer range")
    if type(max_interactions) is not int or not 0 <= max_interactions < 2**64:
        raise ValueError("max_interactions must be an integer in [0, 2**64)")
    if isinstance(tolerance, bool) or not math.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("tolerance must be finite and strictly between zero and one")
    source = np.ascontiguousarray(source, dtype=np.float64)
    capacity = np.ascontiguousarray(capacity, dtype=np.float64)
    if (source.ndim != 3 or source.shape != capacity.shape
            or len(set(source.shape)) != 1 or source.shape[0] < 1):
        raise ValueError("source and capacity must have the same nonempty cubic 3D shape")
    if source.shape[0] > 2048:
        raise ValueError("Grid dimension exceeds the supported integer geometry range")
    for name, values in (("source", source), ("capacity", capacity)):
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{name} must contain finite nonnegative counts")
        if np.sum(values, dtype=np.longdouble) > np.finfo(float).max:
            raise ValueError(f"{name} total exceeds the representable audit range")
    lib = _library()
    started = time.perf_counter()
    ionized, remaining = np.empty_like(source), np.empty_like(source)
    stats = np.zeros(10, dtype=np.float64)
    counts = np.zeros(6, dtype=np.uint64)
    result = lib.allocate_coeval(source.shape[0], source.ctypes.data,
                                capacity.ctypes.data, threads, tolerance,
                                max_iterations, max_interactions,
                                ionized.ctypes.data, remaining.ctypes.data,
                                stats.ctypes.data, counts.ctypes.data)
    if result not in (0, 1, 2):
        raise FloatingPointError("Coeval allocation failed its per-source or finite-value checks")
    photons = np.sum(source, dtype=np.longdouble)
    ions = np.sum(ionized, dtype=np.longdouble)
    left = np.sum(remaining, dtype=np.longdouble)
    residual = float((photons-ions-left)/photons) if photons > 0 else 0.
    roundoff = 512*np.finfo(float).eps
    if (not np.isfinite(ionized).all() or not np.isfinite(remaining).all()
            or (ionized < 0).any() or (remaining < 0).any()
            or np.any(ionized > capacity) or np.any(remaining > source)
            or abs(residual) > max(1e-11, roundoff*max(1, int(counts[0])))):
        raise FloatingPointError("Independent photon/atom audit failed")
    audit = {
        "method": "coeval-synchronous-radial-shell",
        "complete": result == 0,
        "termination": ("complete", "iteration_limit", "interaction_limit")[result],
        "relative_photon_residual": residual,
        "elapsed_seconds": time.perf_counter()-started,
        "kernel_seconds": float(stats[7]), "geometry_seconds": float(stats[9]),
        "iterations": int(counts[0]), "interactions": int(counts[1]),
        "peak_active_sources": int(counts[2]), "shell_count": int(counts[3]),
        "periodic_offset_count": int(counts[4]),
        "saturation_shortcut": bool(counts[5]),
        "excess_assignment": "proportional to residual emitter budgets after local absorption; surplus positions are bookkeeping only",
        "photon_total": float(photons), "atom_total": float(stats[1]),
        "ionized_total": float(ions), "remaining_photons": float(left),
        "excess_photons": float(stats[4]), "unresolved_photons": float(stats[5]),
        "maximum_roundoff_source_overdraft": float(stats[8]),
        "tolerance": float(tolerance), "threads": threads,
        "max_iterations": max_iterations, "max_interactions": max_interactions,
        "source_anchored": True, "periodic": True,
        "exhausted_inner_shells_skipped": True,
        "scope": "frozen gas; cumulative photons; instantaneous radial allocation; no recombinations",
        "limitations": "No angular photon budgets, foreground opacity or finite light travel time; morphology requires separate validation.",
    }
    return {"ionized": ionized, "remaining_source": remaining, "audit": audit}
