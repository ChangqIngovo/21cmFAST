"""Lazy eight-subcell fronts with original source centres and photon budgets.

Uniform source-only parents evolve as one node. The first incident ray promotes
the parent; fully transparent parents can collapse. This changes data structures,
not the declared eight-subcell opaque-gas approximation. Nonuniform partial
states are retained explicitly between calls.
"""
from __future__ import annotations

import ctypes as ct
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
from threading import RLock
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pc_refinement"))
from front_subcells import octant_directions
import hybrid

_library = None
_lock = RLock()
_INTMAX = np.iinfo(np.int32).max


def library():
    global _library
    with _lock:
        if _library is None:
            source = HERE / "adaptive.cpp"
            digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
            output = Path(tempfile.mkdtemp(prefix=f"pc_front_{digest}_")) / "adaptive.so"
            subprocess.run(["g++", "-O3", "-std=c++17", "-shared", "-fPIC",
                            str(source), "-o", str(output)], check=True)
            lib = ct.CDLL(str(output))
            lib.adaptive_evolve.argtypes = [
                ct.c_int, ct.c_void_p, ct.c_void_p,
                ct.c_int, ct.c_void_p, ct.c_void_p,
                ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_void_p,
                ct.c_int, ct.c_void_p, ct.c_double, ct.c_double, ct.c_int,
                *([ct.c_void_p]*8),
            ]
            lib.adaptive_evolve.restype = ct.c_int
            lib.adaptive_state_count.argtypes = []
            lib.adaptive_state_count.restype = ct.c_int64
            for name, args in (("adaptive_copy_state", [ct.c_void_p, ct.c_void_p]),
                               ("adaptive_copy_remaining", [ct.c_void_p]),
                               ("adaptive_copy_state_absorbed", [ct.c_void_p]),
                               ("adaptive_copy_fine_absorbed", [ct.c_void_p])):
                getattr(lib, name).argtypes = args
                getattr(lib, name).restype = ct.c_int
            _library = lib
    return _library


def _capacity_hash(c):
    return hashlib.sha256(memoryview(c)).hexdigest()


def _fine_from_parent_major(values, n):
    return np.ascontiguousarray(values.reshape(n, n, n, 2, 2, 2)
                                .transpose(0, 3, 1, 4, 2, 5).reshape(2*n, 2*n, 2*n))


def _check_counts(s, c, initial_ionized, initial_state):
    if (s.shape != c.shape or s.ndim != 3 or len(set(s.shape)) != 1 or not s.size
            or not np.isfinite(s).all() or not np.isfinite(c).all()
            or (s < 0).any() or (c < 0).any()):
        raise ValueError("Matching nonempty cubic finite nonnegative photon/atom counts required")
    if s.size > _INTMAX or s.shape[0]*2 > _INTMAX-2:
        raise ValueError("Grid exceeds supported native integer indices")
    if not np.isfinite(float(s.sum(dtype=np.longdouble))):
        raise ValueError("Total photon budget overflows; rescale units")
    fingerprint = _capacity_hash(c)
    if initial_state is not None:
        if initial_ionized is not None:
            raise ValueError("Use initial_ionized or initial_state, not both")
        if initial_state.get("capacity_sha256") != fingerprint:
            raise ValueError("A carried subcell state requires the same static capacity grid")
        old = np.ascontiguousarray(initial_state["ionized"], dtype=float)
        if (old.shape != c.shape or not np.isfinite(old).all()
                or (old < 0).any() or (old > c).any()):
            raise ValueError("Carried ionization must match and fit the atom grid")
        rem = np.ascontiguousarray(c-old)
        ids_raw = np.asarray(initial_state["parent_ids"])
        child_old = np.ascontiguousarray(initial_state["child_ionized"], dtype=float)
        if (ids_raw.ndim != 1 or not np.isfinite(ids_raw).all()
                or (ids_raw != np.floor(ids_raw)).any() or (ids_raw < 0).any()
                or (ids_raw >= s.size).any() or len(np.unique(ids_raw)) != len(ids_raw)):
            raise ValueError("State parent indices must be unique valid integers")
        ids = np.ascontiguousarray(ids_raw, dtype=np.int32)
        if (child_old.shape != (len(ids), 8) or not np.isfinite(child_old).all()
                or (child_old < 0).any()):
            raise ValueError("State child ionization must be finite nonnegative eight-tuples")
        children = np.ascontiguousarray(c.ravel()[ids, None]/8-child_old)
        if len(ids):
            cap8 = c.ravel()[ids, None]/8
            if (child_old > cap8).any():
                raise ValueError("Carried child ionization exceeds original atoms")
            discrepancy = abs(children.sum(axis=1, dtype=np.longdouble)-rem.ravel()[ids])
            if (discrepancy > 2e-12*c.ravel()[ids]).any():
                raise ValueError("Coarse and child state budgets disagree")
            # Near full ionization, C-sum(I_child) suffers cancellation at the
            # parent scale. The resolved child capacities are authoritative.
            rem.ravel()[ids] = children.sum(axis=1, dtype=np.longdouble)
    else:
        old = np.zeros_like(c) if initial_ionized is None else np.asarray(initial_ionized, dtype=float)
        if (old.shape != c.shape or not np.isfinite(old).all()
                or (old < 0).any() or (old > c).any()):
            raise ValueError("initial_ionized must contain bounded ionized atom counts")
        rem = np.ascontiguousarray(c-old)
        ids = np.empty(0, dtype=np.int32)
        children = np.empty((0, 8))
        child_old = np.empty((0, 8))
    if (rem.shape != c.shape or not np.isfinite(rem).all()
            or (rem < 0).any() or (rem > c+2e-12*c).any()):
        raise ValueError("Remaining atom capacities must match the grid and fit total atoms")
    return rem, ids, children, fingerprint, old, child_old


def solve(source, capacity, initial_ionized=None, *, initial_state=None,
          min_angles=24, max_angles=12288, rays_per_surface_cell=4.,
          periodic=True, maximum_distance=None, return_dense=False,
          effective_density="remaining"):
    """Solve a constant-emission, static-gas interval; return counts and state.

    Pass the returned ``state`` to the next call as ``initial_state`` to preserve
    the resolved front. A coarse ``initial_ionized`` instead assumes all eight
    children initially share the same ionized fraction. ``return_dense`` is an
    optional debugging/reference output and defeats sparse output memory.
    """
    lib = library()  # One-time compilation is excluded from timings.
    started = time.perf_counter()
    s = np.ascontiguousarray(source, dtype=float)
    c = np.ascontiguousarray(capacity, dtype=float)
    rem, parent_ids, child_remaining, fingerprint, old, child_old = _check_counts(s, c, initial_ionized, initial_state)
    n = s.shape[0]
    horizon = float(n*np.sqrt(3.) if maximum_distance is None else maximum_distance)
    if not np.isfinite(horizon) or horizon <= 0 or 2*horizon > _INTMAX-2*n-2:
        raise ValueError("Maximum distance exceeds supported positive voxel range")
    for name, count in (("min_angles", min_angles), ("max_angles", max_angles)):
        if (isinstance(count, bool) or not isinstance(count, (int, np.integer))
                or count < 24 or count > _INTMAX or count % 24):
            raise ValueError(f"{name} must be a supported positive multiple of 24")
    if effective_density not in ("remaining", "total"):
        raise ValueError("effective_density must be remaining or total")
    if min_angles > max_angles or not np.isfinite(rays_per_surface_cell) or rays_per_surface_cell <= 0:
        raise ValueError("Ordered angular limits and finite positive surface sampling required")
    if periodic and not np.any(rem > 0):
        # An ionized periodic box has neither an absorber nor an open boundary.
        # Tracing N_source*K rays would only reconstruct this exact reservoir.
        zero = np.zeros_like(s)
        photons = float(s.sum(dtype=np.longdouble))
        state = {"ionized": c.copy(), "parent_ids": np.empty(0, dtype=np.int32),
                 "child_ionized": np.empty((0, 8)), "capacity_sha256": fingerprint}
        audit = {"model": "lazy eight-subcell event transport",
                 "scope": "exact fully transparent periodic limit", "photons": photons,
                 "absorbed": 0., "escaped": 0., "retained": photons, "ledger_residual": 0.,
                 "source_relative_error": 0., "angle_relative_error": 0.,
                 "maximum_local_fraction_violation": 0., "effective_density": effective_density,
                 "source_positions": "original coarse centres", "maximum_distance": horizon,
                 "return_dense": bool(return_dense), "continuum_accuracy_certified": False,
                 "state_bytes": c.nbytes, "preparation_seconds": time.perf_counter()-started}
        for name in ("events", "rays_materialized", "queries", "assignments", "promotions",
                     "peak_live_child_parents", "total_child_slots_highwater", "ray_bytes",
                     "cell_allocated_bytes", "heap_allocated_bytes", "parent_skips",
                     "uniform_coarse_events", "final_sparse_parent_count", "source_launches",
                     "fine_events", "coarsenings"):
            audit[name] = 0
        result = {"absorbed": zero, "source_absorbed": zero.copy(), "source_escaped": zero.copy(),
                  "source_retained": s.copy(), "unabsorbed": photons, "state": state,
                  "angle_counts": np.zeros(s.shape, dtype=np.int32),
                  "estimated_radius": zero.copy(), "audit": audit}
        if return_dense:
            if ((c > 0) & (c/8 == 0)).any():
                raise ValueError("Subcell capacities underflow; rescale units")
            result["subcell_absorbed"] = np.zeros((2*n,)*3)
            result["subcell_ionized"] = c.repeat(2, 0).repeat(2, 1).repeat(2, 2)/8
        audit["seconds"] = time.perf_counter()-started
        return result
    density = rem if effective_density == "remaining" else c
    levels, counts, radius, _ = hybrid.estimate_angles(
        s, density, rem, min_angles=min_angles, max_angles=max_angles,
        rays_per_surface_cell=rays_per_surface_cell, periodic=periodic,
        maximum_distance=horizon)
    used = np.unique(levels[s > 0]) if (s > 0).any() else np.array([0])
    mapping = np.zeros(len(counts), dtype=np.int32)
    mapping[used] = np.arange(len(used))
    levels = np.ascontiguousarray(mapping[levels], dtype=np.int32)
    counts = np.ascontiguousarray(counts[used], dtype=np.int32)
    if counts.sum(dtype=np.int64) > _INTMAX:
        raise ValueError("Combined angular bank exceeds native integer indices")
    active = s > 0
    if active.any() and (s[active]/counts[levels[active]] < np.finfo(float).tiny).any():
        raise ValueError("Per-ray photon budget is subnormal; rescale units")
    offsets = np.ascontiguousarray(np.r_[0, np.cumsum(counts[:-1])], dtype=np.int32)
    directions = np.ascontiguousarray(np.concatenate([octant_directions(int(k)) for k in counts]))
    outputs = [np.zeros_like(s) for _ in range(4)]
    angular = [np.zeros(len(directions)) for _ in range(3)]
    stats = np.zeros(16)
    preparation_seconds = time.perf_counter()-started
    with _lock:
        rc = lib.adaptive_evolve(n, s.ctypes.data, rem.ctypes.data, len(parent_ids),
            parent_ids.ctypes.data, child_remaining.ctypes.data, len(counts), levels.ctypes.data,
            offsets.ctypes.data, counts.ctypes.data, len(directions), directions.ctypes.data,
            1., horizon, int(periodic), *[a.ctypes.data for a in outputs],
            *[a.ctypes.data for a in angular], stats.ctypes.data)
        if rc:
            raise RuntimeError(f"Adaptive subcell core returned status {rc}")
        final_remaining = np.empty_like(c)
        sparse_count = lib.adaptive_state_count()
        if not 0 <= sparse_count <= s.size:
            raise RuntimeError("Invalid sparse front state size")
        final_ids = np.empty(sparse_count, dtype=np.int32)
        final_children = np.empty((sparse_count, 8))
        sparse_absorbed = np.empty((sparse_count, 8))
        if lib.adaptive_copy_remaining(final_remaining.ctypes.data):
            raise RuntimeError("Cannot retrieve coarse state")
        if lib.adaptive_copy_state(final_ids.ctypes.data, final_children.ctypes.data):
            raise RuntimeError("Cannot retrieve resolved front state")
        if lib.adaptive_copy_state_absorbed(sparse_absorbed.ctypes.data):
            raise RuntimeError("Cannot retrieve sparse incremental absorption")
        dense_absorbed = None
        if return_dense:
            dense_absorbed = np.empty((s.size, 8))
            if lib.adaptive_copy_fine_absorbed(dense_absorbed.ctypes.data):
                raise RuntimeError("Cannot retrieve dense reference output")
    a, sa, escaped, retained = outputs
    if not all(np.isfinite(x).all() for x in [*outputs, *angular, final_remaining, final_children]):
        raise RuntimeError("Nonfinite adaptive output")
    photons = s.sum(dtype=np.longdouble)
    residual = photons-sum(x.sum(dtype=np.longdouble) for x in (a, escaped, retained))
    source_residual = s-sa-escaped-retained
    source_error = float(np.max(abs(source_residual[active])/s[active])) if active.any() else 0.
    total_by_level = np.bincount(levels.ravel(), weights=s.ravel(), minlength=len(counts))
    expected_angle = np.repeat(total_by_level/counts, counts)
    angle_error = float(np.max(abs(expected_angle-sum(angular)) /
                               np.maximum(expected_angle, np.finfo(float).tiny)))
    local_violation = float(np.max(np.maximum(np.maximum(a-rem, 0), np.maximum(-a, 0)) /
                                   np.maximum(c, np.finfo(float).tiny)))
    if (abs(residual)/max(photons, np.finfo(float).tiny) > 5e-9
            or source_error > 5e-9 or angle_error > 5e-9 or local_violation > 2e-10):
        raise RuntimeError(f"Adaptive photon ledger failed: {float(residual)=}, {source_error=}, {angle_error=}, {local_violation=}")
    # Retain cumulative ionized counts, not only remaining capacities: a weak
    # increment can be representable in I while C-I rounds back to C.
    ionized = np.minimum(old+a, c)
    ionized[final_remaining == 0] = c[final_remaining == 0]
    final_child_old = np.repeat(old.ravel()[final_ids, None]/8, 8, axis=1)
    if len(parent_ids) and len(final_ids):
        order = np.argsort(parent_ids)
        found = np.searchsorted(parent_ids[order], final_ids)
        matched = found < len(parent_ids)
        matched[matched] &= parent_ids[order[found[matched]]] == final_ids[matched]
        final_child_old[matched] = child_old[order[found[matched]]]
    final_child_ionized = np.minimum(final_child_old+sparse_absorbed, c.ravel()[final_ids, None]/8)
    # Reconstruct parent totals from the children when geometry remains active.
    ionized.ravel()[final_ids] = np.minimum(final_child_ionized.sum(axis=1), c.ravel()[final_ids])
    state = {"ionized": ionized, "parent_ids": final_ids,
             "child_ionized": final_child_ionized, "capacity_sha256": fingerprint}
    result = {"absorbed": a, "source_absorbed": sa, "source_escaped": escaped,
              "source_retained": retained, "unabsorbed": float(escaped.sum(dtype=np.longdouble)+retained.sum(dtype=np.longdouble)),
              "state": state, "angle_counts": counts[levels], "estimated_radius": radius}
    audit = {"model": "lazy eight-subcell event transport", "photons": float(photons),
             "absorbed": float(a.sum(dtype=np.longdouble)), "escaped": float(escaped.sum(dtype=np.longdouble)),
             "retained": float(retained.sum(dtype=np.longdouble)), "ledger_residual": float(residual),
             "source_relative_error": source_error, "angle_relative_error": angle_error,
             "maximum_local_fraction_violation": local_violation,
             "preparation_seconds": preparation_seconds, "seconds": time.perf_counter()-started,
             "return_dense": bool(return_dense), "maximum_distance": horizon,
             "effective_density": effective_density,
             "source_positions": "original coarse centres", "continuum_accuracy_certified": False}
    names = ("events", "rays_materialized", "queries", "assignments", "promotions",
             "peak_live_child_parents", "total_child_slots_highwater", "ray_bytes",
             "cell_allocated_bytes", "heap_allocated_bytes", "parent_skips",
             "uniform_coarse_events", "final_sparse_parent_count", "source_launches",
             "fine_events", "coarsenings")
    audit.update({name: int(value) for name, value in zip(names, stats)})
    audit["state_bytes"] = int(sum(x.nbytes for x in state.values() if isinstance(x, np.ndarray)))
    result["audit"] = audit
    if return_dense:
        result["subcell_absorbed"] = _fine_from_parent_major(dense_absorbed, n)
        fine_old = np.repeat(old.ravel()[:, None]/8, 8, axis=1)
        fine_old[parent_ids] = child_old
        result["subcell_ionized"] = _fine_from_parent_major(
            np.minimum(fine_old+dense_absorbed, c.ravel()[:, None]/8), n)
    return result
