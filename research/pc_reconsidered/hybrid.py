"""Source-adaptive angular events: an economical sharp-front candidate.

The angular count of each source follows its estimated front surface area.
Initially transparent connected regions enlarge that estimate. All beams share
one event clock, so this does not split strong and weak sources into sequential
transport batches. The model remains infinite-c, optically thick, static-density,
and without recombinations, exactly as the reference event_front model.

The radius predictor is an approximation, not an angular-error certificate.
An initially neutral, collectively ionized cluster can extend weak sources'
fronts well beyond their isolated radii. ``influence_radius`` can supply a
cached ES estimate, and independent high-angle comparisons remain necessary.
"""
from __future__ import annotations

import ctypes as ct
import hashlib
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
from scipy import ndimage

_ALTERNATIVES = Path(__file__).resolve().parents[1] / "pc_alternatives"
if str(_ALTERNATIVES) not in sys.path:
    sys.path.insert(0, str(_ALTERNATIVES))
from characteristics import clearance, directions  # noqa: E402

_library = None


def library():
    """Compile the separate variable-angle core once per Python process."""
    global _library
    if _library is None:
        source = Path(__file__).with_suffix(".cpp")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
        directory = Path(tempfile.mkdtemp(prefix=f"pc_adaptive_{digest}_"))
        output = directory / "adaptive.so"
        subprocess.run(
            ["g++", "-O3", "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(output)],
            check=True,
        )
        lib = ct.CDLL(str(output))
        lib.evolve_adaptive_events.argtypes = [
            ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_void_p,
            ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_void_p,
            ct.c_int, ct.c_void_p, ct.c_double, ct.c_double, ct.c_int,
            *([ct.c_void_p] * 8),
        ]
        lib.evolve_adaptive_events.restype = ct.c_int
        for name in ("hybrid_last_max_heap_nodes", "hybrid_last_heap_updates",
                     "hybrid_last_ray_allocated_bytes"):
            getattr(lib, name).argtypes = []
            getattr(lib, name).restype = ct.c_int64
        _library = lib
    return _library


def _cavity_radius(remaining, active, periodic, horizon):
    """Conservative bounding-box extent of initial transparent components.

Periodic face-connected pieces are joined before their shortest circular
bounding intervals are found. A component covering an entire coordinate axis
uses the requested horizon, since no finite nonwrapping box bounds it.
Components adjoining the source's six faces enter the bound as well: a source
cell with a tiny positive remaining capacity clears almost immediately, and
must not receive drastically fewer angles than the exactly transparent case.
This does not merge neighboring source cells into a newly transparent region.
Only initially transparent cells enter this bound, not regions ionized later.
"""
    labels, count = ndimage.label(remaining <= 0)
    radius = np.zeros_like(remaining)
    if not count:
        return radius
    n = remaining.shape[0]
    if periodic:
        # Join labels across opposite faces. This avoids assigning the full
        # horizon to a small bubble merely because it crosses the box boundary.
        parent = np.arange(count + 1)

        def root(tag):
            while parent[tag] != tag:
                parent[tag] = parent[parent[tag]]
                tag = parent[tag]
            return tag

        for axis in range(3):
            first, last = np.take(labels, 0, axis), np.take(labels, -1, axis)
            mask = (first > 0) & (last > 0)
            for a, b in np.unique(np.stack((first[mask], last[mask]), axis=1), axis=0):
                parent[root(a)] = root(b)
        mapped = np.array([root(tag) for tag in range(count + 1)])
        _, mapped = np.unique(mapped, return_inverse=True)
        labels = mapped[labels]
        count = int(mapped.max())
        coordinates = np.argwhere(labels > 0)
        tags = labels[tuple(coordinates.T)]
        order = np.argsort(tags)
        split = np.searchsorted(tags[order], np.arange(1, count + 2))
        bounds = []
        for tag in range(count):
            points = coordinates[order[split[tag]:split[tag + 1]]]
            box = []
            for axis in range(3):
                occupied = np.unique(points[:, axis])
                gaps = (np.roll(occupied, -1) - occupied - 1) % n
                after = int(np.argmax(gaps))
                start = int(occupied[(after + 1) % len(occupied)])
                box.append(slice(start, start + n - int(gaps[after])))
            bounds.append(tuple(box))
    else:
        bounds = ndimage.find_objects(labels)
    adjacent = labels > 0
    if periodic:
        for axis in range(3):
            adjacent |= np.roll(labels > 0, 1, axis) | np.roll(labels > 0, -1, axis)
    else:
        adjacent = ndimage.binary_dilation(adjacent)
    for location in np.argwhere(active & adjacent):
        tags = {int(labels[tuple(location)])}
        for axis in range(3):
            for side in (-1, 1):
                neighbor = location.copy()
                neighbor[axis] += side
                if periodic:
                    neighbor %= n
                elif not (0 <= neighbor[axis] < n):
                    continue
                tags.add(int(labels[tuple(neighbor)]))
        tags.discard(0)
        if not tags:
            continue
        value = 0.
        for tag in tags:
            box = bounds[tag - 1]
            if periodic and any(interval.stop - interval.start == n for interval in box):
                value = horizon
                break
            sides = []
            for d, interval in enumerate(box):
                coordinate = location[d] + .5
                if periodic:
                    centre = .5 * (interval.start + interval.stop)
                    coordinate += n * np.floor((centre - coordinate) / n + .5)
                sides.append(max(abs(coordinate - interval.start),
                                 abs(interval.stop - coordinate)))
            value = max(value, float(np.linalg.norm(sides)))
        radius[tuple(location)] = min(value, horizon)
    return radius


def estimate_angles(source, capacity, remaining, *, min_angles=24, max_angles=12288,
                    rays_per_surface_cell=4., periodic=True, maximum_distance=None,
                    influence_radius=None, cavity_bound=True):
    """Return a dyadic angular level and radius predictor for each source.

The reference density is the 10th percentile of positive initial capacities,
reducing underestimation in modestly inhomogeneous media. This percentile is a
declared approximation and does not bound underdense tunnels.
"""
    for name, value in (("min_angles", min_angles), ("max_angles", max_angles)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if min_angles > max_angles or not math.isfinite(rays_per_surface_cell) or rays_per_surface_cell <= 0:
        raise ValueError("Ordered angular limits and positive finite surface sampling required")
    n = source.shape[0]
    horizon = n * np.sqrt(3.) if maximum_distance is None else float(maximum_distance)
    positive = capacity[capacity > 0]
    reference_density = float(np.quantile(positive, .1)) if positive.size else 1.
    radius = np.cbrt(3 * source / (4 * np.pi * reference_density))
    if influence_radius is not None:
        supplied = np.asarray(influence_radius, dtype=float)
        if supplied.shape != source.shape or not np.isfinite(supplied).all() or (supplied < 0).any():
            raise ValueError("influence_radius must be a nonnegative finite grid")
        radius = np.maximum(radius, supplied)
    if cavity_bound and (remaining <= 0).any():
        # Adding the isolated new-photon radius allows the new front to advance
        # beyond the initial cavity rather than using its surface as a cutoff.
        radius += _cavity_radius(remaining, source > 0, periodic, horizon)
    radius = np.minimum(radius, horizon)
    wanted = np.maximum(min_angles, np.ceil(4 * np.pi * radius**2 * rays_per_surface_cell))
    counts = [int(min_angles)]
    while counts[-1] < max_angles:
        counts.append(min(2 * counts[-1], int(max_angles)))
    levels = np.searchsorted(counts, wanted, side="left").clip(max=len(counts)-1)
    levels[source == 0] = 0
    return np.asarray(levels, dtype=np.int32), np.array(counts, dtype=np.int32), radius, wanted


def solve(source, capacity, initial_ionized=None, *, min_angles=24, max_angles=12288,
          rays_per_surface_cell=4., periodic=True, maximum_distance=None,
          influence_radius=None, cavity_bound=True):
    """Return newly absorbed photons, unabsorbed photons and independent ledgers.

``source`` gives the photons emitted during a unit-duration step; ``capacity``
gives the total atom count per cell; ``initial_ionized`` gives already ionized
atom counts, not ionized fractions. No existing ionization is counted as new
absorption. ``unabsorbed`` includes both open-boundary escape and distance-limit
retention. These are distinguished in the audit and per-source arrays.
"""
    started = time.perf_counter()
    s = np.ascontiguousarray(source, dtype=np.float64)
    c = np.ascontiguousarray(capacity, dtype=np.float64)
    if s.shape != c.shape or s.ndim != 3 or len(set(s.shape)) != 1 or s.size == 0:
        raise ValueError("Nonempty equally shaped cubic source/capacity arrays required")
    if not np.isfinite(s).all() or not np.isfinite(c).all() or (s < 0).any() or (c < 0).any():
        raise ValueError("Finite nonnegative sources and capacities required")
    old = np.zeros_like(c) if initial_ionized is None else np.asarray(initial_ionized, dtype=float)
    if old.shape != c.shape or not np.isfinite(old).all() or (old < 0).any() or (old > c).any():
        raise ValueError("initial_ionized must be finite atom counts between zero and capacity")
    rem = np.ascontiguousarray(c - old)
    n = s.shape[0]
    horizon = float(n * np.sqrt(3.) if maximum_distance is None else maximum_distance)
    if not np.isfinite(horizon) or horizon <= 0 or horizon > np.iinfo(np.int32).max - n - 2:
        raise ValueError("Maximum distance outside supported positive voxel range")
    if n**3 > np.iinfo(np.int32).max:
        raise ValueError("Grid too large for native indices")
    total = float(np.sum(s, dtype=np.longdouble))
    if not np.isfinite(total):
        raise ValueError("Aggregate photon budget overflows; rescale units")
    # A fully transparent periodic box cannot absorb or let photons leave its
    # domain. Preserve them explicitly without allocating source-angle packets.
    # The corresponding open-boundary problem still needs directional tracing.
    if periodic and not np.any(rem > 0):
        zero = np.zeros_like(s)
        audit = {
            "model": "source-adaptive quadrature of shared sharp-front events",
            "scope": "fully transparent periodic domain; exact retained-photon limit",
            "photons": total, "absorbed": 0., "escaped": 0., "retained": total,
            "ledger_residual": 0., "per_source_relative_ledger_max": 0.,
            "per_angle_ledger_max": 0., "capacity_excess": 0.,
            "angles_min": 0, "angles_max": 0, "angles_mean": 0., "angles_sum": 0,
            "sources_at_angle_limit": 0, "events": 0, "rays_materialized": 0,
            "queries": 0, "ray_allocated_bytes": 0, "transport_seconds": 0.,
            "seconds": time.perf_counter() - started, "periodic": True,
            "maximum_distance": horizon, "angular_accuracy_certified": True,
        }
        return {"absorbed": zero, "unabsorbed": total,
                "source_absorbed": zero.copy(), "source_escaped": zero.copy(),
                "source_retained": s.copy(), "angle_counts": np.zeros(s.shape, dtype=np.int32),
                "estimated_radius": zero.copy(), "audit": audit}
    levels, counts, radius, wanted = estimate_angles(
        s, c, rem, min_angles=min_angles, max_angles=max_angles,
        rays_per_surface_cell=rays_per_surface_cell, periodic=periodic,
        maximum_distance=horizon, influence_radius=influence_radius, cavity_bound=cavity_bound,
    )
    # Retain only used quadratures to avoid constructing large unused levels.
    used = np.unique(levels[s > 0]) if (s > 0).any() else np.array([0])
    remap = np.zeros(len(counts), dtype=np.int32)
    remap[used] = np.arange(len(used))
    levels = np.ascontiguousarray(remap[levels], dtype=np.int32)
    counts = np.ascontiguousarray(counts[used], dtype=np.int32)
    offsets = np.ascontiguousarray(np.r_[0, np.cumsum(counts[:-1])], dtype=np.int32)
    dirs = np.ascontiguousarray(np.concatenate([directions(int(k)) for k in counts]))
    active = s > 0
    if active.any() and np.any(s[active] / counts[levels[active]] < np.finfo(float).tiny):
        raise ValueError("Per-ray budget is subnormal; rescale units")
    dist = clearance(rem <= 0)
    arrays = [np.zeros_like(s) for _ in range(4)]
    angle_arrays = [np.zeros(len(dirs)) for _ in range(3)]
    stats = np.zeros(6)
    compile_started = time.perf_counter()
    lib = library()
    compilation_seconds = time.perf_counter() - compile_started
    # Compilation is deliberately excluded, all predictor/setup work is included.
    transport_started = time.perf_counter()
    rc = lib.evolve_adaptive_events(
        n, s.ctypes.data, rem.ctypes.data, dist.ctypes.data,
        len(counts), levels.ctypes.data, offsets.ctypes.data, counts.ctypes.data,
        len(dirs), dirs.ctypes.data, 1., horizon, int(periodic),
        *[a.ctypes.data for a in arrays], *[a.ctypes.data for a in angle_arrays], stats.ctypes.data,
    )
    if rc:
        raise RuntimeError(f"Adaptive event integrator returned {rc}")
    absorbed, src_abs, escaped, retained = arrays
    expected_angles = np.concatenate([
        np.full(int(k), np.sum(s[levels == lev], dtype=np.longdouble) / k)
        for lev, k in enumerate(counts)
    ])
    angle_error = expected_angles - sum(angle_arrays)
    ledger = total - float(absorbed.sum(dtype=np.longdouble) + escaped.sum(dtype=np.longdouble)
                           + retained.sum(dtype=np.longdouble))
    per_source = s - src_abs - escaped - retained
    if not all(np.isfinite(a).all() for a in [*arrays, *angle_arrays]):
        raise RuntimeError("Nonfinite transport accounting")
    source_relative = float(np.max(np.abs(per_source[active]) / s[active])) if active.any() else 0.
    if (source_relative > 5e-10 or abs(ledger) > 5e-10 * max(total, np.finfo(float).tiny)
            or np.any(absorbed > rem + 5e-12 * np.maximum(rem, 1.))):
        raise RuntimeError("Adaptive photon/capacity ledger failed")
    active_angles = counts[levels[active]]
    audit = {
        "model": "source-adaptive quadrature of shared sharp-front events",
        "scope": "infinite c; static density; no recombinations; approximate radius predictor",
        "photons": total, "absorbed": float(absorbed.sum()),
        "escaped": float(escaped.sum()), "retained": float(retained.sum()),
        "ledger_residual": ledger, "per_source_relative_ledger_max": source_relative,
        "per_angle_ledger_max": float(np.max(np.abs(angle_error))),
        "capacity_excess": float(np.maximum(absorbed - rem, 0).max()),
        "angles_min": int(active_angles.min()) if active.any() else 0,
        "angles_max": int(active_angles.max()) if active.any() else 0,
        "angles_mean": float(active_angles.mean()) if active.any() else 0.,
        "angles_sum": int(active_angles.sum()),
        "sources_at_angle_limit": int(np.count_nonzero(active & (wanted > max_angles))),
        "events": int(stats[0]), "rays_materialized": int(stats[1]), "queries": int(stats[2]),
        "ray_allocated_bytes": int(lib.hybrid_last_ray_allocated_bytes()),
        "transport_seconds": time.perf_counter() - transport_started,
        "seconds": time.perf_counter() - started - compilation_seconds,
        "periodic": bool(periodic), "maximum_distance": horizon,
        "angular_accuracy_certified": False,
    }
    return {"absorbed": absorbed, "unabsorbed": audit["escaped"] + audit["retained"],
            "source_absorbed": src_abs, "source_escaped": escaped,
            "source_retained": retained, "angle_counts": counts[levels],
            "estimated_radius": radius, "audit": audit}
