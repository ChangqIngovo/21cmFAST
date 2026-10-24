"""Positive angular weights on unchanged Fibonacci nodes and event equations."""
from __future__ import annotations

import ctypes as ct
from functools import lru_cache
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
from scipy.spatial import SphericalVoronoi

import quadrature

_lib = None


@lru_cache(maxsize=40)
def weights(count, method):
    xyz = quadrature.directions(count, "fibonacci")
    if method == "voronoi-weights":
        answer = SphericalVoronoi(xyz).calculate_areas() / (4*np.pi)
    elif method == "moment-weights":
        x, y, z = xyz.T
        a = np.array([np.ones(count), x, y, z, x*x-y*y,
                      3*z*z-1, x*y, x*z, y*z])
        b = np.zeros(len(a))
        b[0] = 1.
        uniform = np.full(count, 1/count)
        answer = uniform + a.T @ np.linalg.solve(a @ a.T, b-a@uniform)
    elif method == "uniform-weights":
        answer = np.full(count, 1/count)
    else:
        raise ValueError(method)
    if not np.isfinite(answer).all() or np.any(answer <= 0):
        raise ValueError("This direction set cannot use the requested positive weight rule")
    answer = np.ascontiguousarray(answer/answer.sum())
    answer.flags.writeable = False
    return answer


def library():
    global _lib
    if _lib is None:
        source = Path(__file__).with_name("quadrature_weighted.cpp")
        output = Path(tempfile.mkdtemp(prefix="pc_weighted_"))/"weighted.so"
        subprocess.run(["g++", "-O3", "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(output)], check=True)
        lib = ct.CDLL(str(output))
        lib.evolve_adaptive_events.argtypes = [
            ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_void_p,
            ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_void_p,
            ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_double, ct.c_double, ct.c_int,
            *([ct.c_void_p]*8)]
        lib.evolve_adaptive_events.restype = ct.c_int
        lib.hybrid_last_ray_allocated_bytes.argtypes = []
        lib.hybrid_last_ray_allocated_bytes.restype = ct.c_int64
        _lib = lib
    return _lib


def solve(source, capacity, initial_ionized=None, *, method="voronoi-weights",
          min_angles=24, max_angles=12288, rays_per_surface_cell=4., periodic=True,
          maximum_distance=None, influence_radius=None, cavity_bound=True):
    started = time.perf_counter()
    s, c = [np.ascontiguousarray(a, dtype=np.float64) for a in (source, capacity)]
    if s.shape != c.shape or s.ndim != 3 or len(set(s.shape)) != 1 or not s.size:
        raise ValueError("Equal nonempty cubic count grids required")
    old = np.zeros_like(c) if initial_ionized is None else np.asarray(initial_ionized, dtype=float)
    if old.shape != c.shape or not all(np.isfinite(a).all() for a in (s,c,old)):
        raise ValueError("Finite source, capacity and initial ionization required")
    if (s < 0).any() or (old < 0).any() or (old > c).any():
        raise ValueError("Nonnegative sources and bounded initial ionization required")
    rem = np.ascontiguousarray(c-old)
    n = s.shape[0]
    horizon = float(n*np.sqrt(3) if maximum_distance is None else maximum_distance)
    if not np.isfinite(horizon) or horizon <= 0 or horizon > np.iinfo(np.int32).max-n-2:
        raise ValueError("Unsupported distance horizon")
    total = float(s.sum(dtype=np.longdouble))
    if not np.isfinite(total) or n**3 > np.iinfo(np.int32).max:
        raise ValueError("Unsupported aggregate scale")
    if periodic and not np.any(rem > 0):
        result = quadrature.solve(s, c, old, maximum_distance=horizon)
        result["audit"]["quadrature"] = method
        return result
    levels, counts, radius, wanted = quadrature._backend.estimate_angles(
        s, c, rem, min_angles=min_angles, max_angles=max_angles,
        rays_per_surface_cell=rays_per_surface_cell, periodic=periodic,
        maximum_distance=horizon, influence_radius=influence_radius, cavity_bound=cavity_bound)
    used = np.unique(levels[s > 0]) if (s > 0).any() else np.array([0])
    remap = np.zeros(len(counts), dtype=np.int32)
    remap[used] = np.arange(len(used))
    levels = np.ascontiguousarray(remap[levels], dtype=np.int32)
    counts = np.ascontiguousarray(counts[used], dtype=np.int32)
    offsets = np.ascontiguousarray(np.r_[0, np.cumsum(counts[:-1])], dtype=np.int32)
    dirs = np.ascontiguousarray(np.concatenate([quadrature.directions(int(k)) for k in counts]))
    w = np.ascontiguousarray(np.concatenate([weights(int(k), method) for k in counts]))
    active = s > 0
    minimum_weight = np.asarray([weights(int(k), method).min() for k in counts])
    if active.any() and np.any(s[active]*minimum_weight[levels[active]] < np.finfo(float).tiny):
        raise ValueError("Subnormal beam budgets")
    dist = quadrature._backend.clearance(rem <= 0)
    arrays = [np.zeros_like(s) for _ in range(4)]
    angles = [np.zeros(len(dirs)) for _ in range(3)]
    stats = np.zeros(6)
    compile_start = time.perf_counter()
    lib = library()
    compilation = time.perf_counter()-compile_start
    transport = time.perf_counter()
    rc = lib.evolve_adaptive_events(n, s.ctypes.data, rem.ctypes.data, dist.ctypes.data,
        len(counts), levels.ctypes.data, offsets.ctypes.data, counts.ctypes.data,
        len(dirs), dirs.ctypes.data, w.ctypes.data, 1., horizon, int(periodic),
        *[a.ctypes.data for a in arrays], *[a.ctypes.data for a in angles], stats.ctypes.data)
    if rc:
        raise RuntimeError(f"Weighted event status {rc}")
    absorbed, src_abs, escaped, retained = arrays
    expected = np.concatenate([weights(int(k), method)*s[levels == lev].sum(dtype=np.longdouble)
                               for lev, k in enumerate(counts)])
    per_source = s-src_abs-escaped-retained
    residual = total-float(absorbed.sum(dtype=np.longdouble)+escaped.sum(dtype=np.longdouble)+retained.sum(dtype=np.longdouble))
    angle_error = float(np.max(np.abs(expected-sum(angles))))
    source_error = float(np.max(np.abs(per_source[active])/s[active])) if active.any() else 0.
    if (not all(np.isfinite(a).all() for a in [*arrays,*angles]) or source_error > 5e-10
            or abs(residual) > 5e-10*max(total,np.finfo(float).tiny)
            or (absorbed > rem+5e-12*np.maximum(rem,1.)).any()
            or angle_error > 5e-10*max(total,np.finfo(float).tiny)):
        raise RuntimeError("Weighted source/cell/angular ledger failed")
    audit = dict(quadrature=method, photons=total, absorbed=float(absorbed.sum()),
        escaped=float(escaped.sum()), retained=float(retained.sum()), ledger_residual=residual,
        per_source_relative_ledger_max=source_error, per_angle_ledger_max=angle_error,
        capacity_excess=float(np.maximum(absorbed-rem,0).max()),
        rays_materialized=int(stats[1]), ray_allocated_bytes=int(lib.hybrid_last_ray_allocated_bytes()),
        angles_max=int(counts[levels[active]].max()) if active.any() else 0,
        angles_mean=float(counts[levels[active]].mean()) if active.any() else 0.,
        transport_seconds=time.perf_counter()-transport, seconds=time.perf_counter()-started-compilation,
        weight_min=float(w.min()), weight_max=float(w.max()),
        angular_accuracy_certified=False, scope="Same static opaque-cell event equations; positive nonuniform angular weights")
    return dict(absorbed=absorbed, unabsorbed=audit["escaped"]+audit["retained"],
                source_absorbed=src_abs, source_escaped=escaped, source_retained=retained,
                angle_counts=counts[levels], estimated_radius=radius, audit=audit)
