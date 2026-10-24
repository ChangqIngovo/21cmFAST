"""Photon-counting front refinement without moving cell-centred point sources.

This experiment splits each coarse gas cell into eight equal-volume atoms bins,
while retaining the coarse source at the common vertex of its eight bins. Rays
are partitioned into octants; all octants share the original event clock. There
is no post-hoc smoothing or transfer of absorption between coarse gas cells.

The same *total* angular count is used at either spatial resolution. Thus this
tests whether reducing voxel-front error is useful before multiplying angular
work. A sign-balanced, cyclic-permutation quadrature is used on both grids to
give every octant precisely one eighth of the luminosity. It is not asserted to
be optimal; comparison with the original Fibonacci implementation is separate.
"""
from __future__ import annotations

import ctypes as ct
import hashlib
import itertools
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pc_reconsidered"))
import hybrid
from characteristics import clearance

_library = None
SIGNS = np.array(list(itertools.product((-1., 1.), repeat=3)))


def octant_directions(count):
    """Equal-weight quadrature with zero dipole and isotropic second moment."""
    if count < 24 or count % 24:
        raise ValueError("Angular count must be a positive multiple of 24")
    m = count // 24
    j = np.arange(m)
    mu = (j + .5) / m
    phase = (np.pi / 2) * np.mod((j + .5) * ((np.sqrt(5) - 1) / 2), 1.)
    root = np.sqrt(1 - mu * mu)
    base = np.column_stack((mu, root * np.cos(phase), root * np.sin(phase)))
    positive = np.concatenate([np.roll(base, k, axis=1) for k in range(3)])
    return np.ascontiguousarray(np.concatenate([positive * sign for sign in SIGNS]))


def library():
    global _library
    if _library is None:
        path = Path(__file__).with_suffix(".cpp")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        out = Path(tempfile.mkdtemp(prefix=f"pc_subcells_{digest}_")) / "subcells.so"
        subprocess.run(["g++", "-O3", "-std=c++17", "-shared", "-fPIC", str(path), "-o", str(out)], check=True)
        _library = ct.CDLL(str(out))
        _library.evolve_adaptive_events.argtypes = hybrid.library().evolve_adaptive_events.argtypes
        _library.evolve_adaptive_events.restype = ct.c_int
        _library.front_set_parent_skip.argtypes = [ct.c_int]
        _library.front_set_parent_skip.restype = None
        for name in ("hybrid_last_max_heap_nodes", "hybrid_last_heap_updates", "hybrid_last_ray_allocated_bytes"):
            getattr(_library, name).argtypes = []
            getattr(_library, name).restype = ct.c_int64
    return _library


def _split(values):
    return np.ascontiguousarray(values.repeat(2, 0).repeat(2, 1).repeat(2, 2) / 8)


def _combine(values):
    n = values.shape[0] // 2
    return values.reshape(n, 2, n, 2, n, 2).sum(axis=(1, 3, 5))


def solve(source, capacity, initial_ionized=None, *, subcells=2,
          min_angles=24, max_angles=12288, rays_per_surface_cell=4.,
          periodic=True, maximum_distance=None, parent_skip=False,
          initial_subcell_ionized=None):
    """Subcells=1 and 2 share photons, source positions and angular quadrature."""
    if subcells not in (1, 2):
        raise ValueError("This experiment supports one or two cells per coarse axis")
    started = time.perf_counter()
    s = np.asarray(source, dtype=float)
    c = np.asarray(capacity, dtype=float)
    old = np.zeros_like(c) if initial_ionized is None else np.asarray(initial_ionized, dtype=float)
    if (s.shape != c.shape or old.shape != c.shape or s.ndim != 3 or len(set(s.shape)) != 1
            or not s.size or not all(np.isfinite(a).all() for a in (s, c, old))
            or (s < 0).any() or (c < 0).any() or (old < 0).any() or (old > c).any()):
        raise ValueError("Matching finite cubic photon, atom, and initial-ionized counts required")
    if not np.isfinite(np.sum(s, dtype=np.longdouble)) or np.sum(s, dtype=np.longdouble) > np.finfo(float).max:
        raise ValueError("Aggregate photon count overflows; rescale count units")
    n = s.shape[0]
    fine_old = None
    if initial_subcell_ionized is not None:
        if subcells != 2 or initial_ionized is not None:
            raise ValueError("Use either a coarse initial map or an explicit fine map, not both")
        fine_old = np.asarray(initial_subcell_ionized, dtype=float)
        fine_cap = _split(c)
        if (fine_old.shape != fine_cap.shape or not np.isfinite(fine_old).all()
                or (fine_old < 0).any() or (fine_old > fine_cap).any()):
            raise ValueError("Fine initial ionization must fit each subcell's atom capacity")
        old = _combine(fine_old)
    horizon = float(n * np.sqrt(3) if maximum_distance is None else maximum_distance)
    if not np.isfinite(horizon) or horizon <= 0:
        raise ValueError("The photon travel range must be finite and positive")
    levels, counts, radius, _ = hybrid.estimate_angles(s, c, c-old, min_angles=min_angles,
        max_angles=max_angles, rays_per_surface_cell=rays_per_surface_cell,
        periodic=periodic, maximum_distance=horizon)
    if any(k % 24 for k in counts):
        raise ValueError("All dyadic angular levels must be multiples of 24")
    used = np.unique(levels[s > 0]) if np.any(s > 0) else np.array([0])
    remap = np.zeros(len(counts), dtype=np.int32)
    remap[used] = np.arange(len(used))
    levels = remap[levels]
    coarse_counts = np.ascontiguousarray(counts[used], dtype=np.int32)
    dirs = np.ascontiguousarray(np.concatenate([octant_directions(int(k)) for k in coarse_counts]))
    if subcells == 2:
        ss = _split(s)
        rem = _split(c-old) if fine_old is None else np.ascontiguousarray(_split(c)-fine_old)
        fine_levels = np.empty(ss.shape, dtype=np.int32)
        for octant, bits in enumerate(itertools.product((0, 1), repeat=3)):
            fine_levels[bits[0]::2, bits[1]::2, bits[2]::2] = levels * 8 + octant
        levels = fine_levels
        counts = np.ascontiguousarray(np.repeat(coarse_counts // 8, 8), dtype=np.int32)
    else:
        ss, rem = np.ascontiguousarray(s), np.ascontiguousarray(c-old)
        levels = np.ascontiguousarray(levels, dtype=np.int32)
        counts = coarse_counts
    offsets = np.ascontiguousarray(np.r_[0, np.cumsum(counts[:-1])], dtype=np.int32)
    dist = clearance(rem <= 0)
    outputs = [np.zeros_like(ss) for _ in range(4)]
    angles = [np.zeros(len(dirs)) for _ in range(3)]
    statistics = np.zeros(6)
    compilation_start = time.perf_counter()
    lib = library() if subcells == 2 else hybrid.library()
    if subcells == 2:
        lib.front_set_parent_skip(int(parent_skip))
    compilation_seconds = time.perf_counter() - compilation_start
    code = lib.evolve_adaptive_events(n * subcells, ss.ctypes.data, rem.ctypes.data, dist.ctypes.data,
        len(counts), levels.ctypes.data, offsets.ctypes.data, counts.ctypes.data,
        len(dirs), dirs.ctypes.data, 1., horizon * subcells, int(periodic),
        *[a.ctypes.data for a in outputs], *[a.ctypes.data for a in angles], statistics.ctypes.data)
    if code:
        raise RuntimeError(f"Subcell event core returned {code}")
    if not all(np.isfinite(a).all() for a in outputs+angles):
        raise RuntimeError("Nonfinite subcell photon accounting")
    level_photons = np.bincount(levels.ravel(), weights=ss.ravel(), minlength=len(counts))
    expected_angles = np.repeat(level_photons/counts, counts)
    angle_error = float(np.max(np.abs(expected_angles-sum(angles))))
    fine_absorbed = outputs[0] if subcells == 2 else None
    if subcells == 2:
        outputs = [_combine(a) for a in outputs]
    absorbed, source_absorbed, escaped, retained = outputs
    residual = float(np.sum(s, dtype=np.longdouble) - sum(np.sum(a, dtype=np.longdouble)
                      for a in (absorbed, escaped, retained)))
    active = s > 0
    source_error = float(np.max(np.abs(s[active] - source_absorbed[active] - escaped[active] - retained[active]) / s[active])) if active.any() else 0.
    if abs(residual) > 1e-10 * max(float(s.sum()), 1.) or source_error > 1e-9:
        raise RuntimeError("Subcell source ledger failed")
    if (absorbed > c-old + 1e-11*np.maximum(c, 1.)).any():
        raise RuntimeError("Subcell capacity ledger failed")
    audit = {"subcells_per_axis": subcells, "photons": float(s.sum()),
        "parent_skip":bool(parent_skip),
        "absorbed": float(absorbed.sum()), "escaped": float(escaped.sum()), "retained": float(retained.sum()),
        "ledger_residual": residual, "source_relative_error": source_error,
        "maximum_angle_ledger_residual": angle_error,
        "rays_materialized": int(statistics[1]), "events": int(statistics[0]),
        "queries": int(statistics[2]), "ray_bytes": int(lib.hybrid_last_ray_allocated_bytes()),
        "seconds": time.perf_counter()-started-compilation_seconds,
        "quadrature": "octant-balanced, cyclic permutations, equal weights",
        "source_positions": "original coarse cell centres, identical in both spatial resolutions"}
    return {"absorbed": absorbed, "source_absorbed": source_absorbed,
            "source_escaped": escaped, "source_retained": retained,
            "unabsorbed": audit["escaped"]+audit["retained"], "audit": audit,
            "subcell_absorbed": fine_absorbed}
