"""Optional geometric source coarsening strictly inside initially transparent gas.

Nonoverlapping dyadic blocks are accepted only when every member source cell is
initially transparent and the farthest member's displacement from an ORIGINAL
source anchor is at most ``theta`` times a conservative neutral-surface distance.
The latter is min(chessboard_clearance - 1/2) over member cell centers, with opaque
padding at the box faces. It is a lower bound even for a periodic domain.

This is an approximation to source geometry, not a new conservation law. It can
alter angular illumination, shadows, finite-horizon support and source competition.
Original-source ledgers are proportional bookkeeping splits of an anchor's totals,
not ray-resolved attribution. Fixed dyadic boundaries introduce grid dependence.
Theta=0 disables merging exactly; merging is not enabled by default.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from hybrid import clearance


def merge(source, capacity, initial_ionized=None, *, theta=0.):
    """Return aggregated sources and a sparse original-to-anchor ledger mapping."""
    started = time.perf_counter()
    s, c = np.asarray(source, dtype=float), np.asarray(capacity, dtype=float)
    old = np.zeros_like(c) if initial_ionized is None else np.asarray(initial_ionized, dtype=float)
    if s.shape != c.shape or s.ndim != 3 or not s.size or old.shape != c.shape:
        raise ValueError("Equally shaped nonempty 3D source/capacity/initial arrays required")
    if (not all(np.isfinite(a).all() for a in (s, c, old)) or
            any(np.any(a < 0) for a in (s, c, old)) or np.any(old > c)):
        raise ValueError("Finite nonnegative counts with initial_ionized <= capacity required")
    if not np.isfinite(theta) or not 0 <= theta < 1:
        raise ValueError("theta must lie in [0,1)")
    total = np.sum(s, dtype=np.longdouble)
    if not np.isfinite(float(total)):
        raise ValueError("Total source counts overflow; rescale the units")
    original = np.flatnonzero(s)
    source_counts = s.ravel()[original]
    anchor = original.copy()
    coordinates = np.column_stack(np.unravel_index(original, s.shape))
    displacement = np.zeros(len(original))
    surface_bound = np.zeros(len(original))
    tests = accepted_blocks = 0
    if theta > 0 and len(original) > 1:
        transparent = (c-old) == 0
        neutral_distance = np.maximum(clearance(transparent).astype(float)-.5, 0.)
        source_distance = neutral_distance.ravel()[original]
        source_transparent = transparent.ravel()[original]
        width = 1 << int(np.ceil(np.log2(max(s.shape))))

        def visit(ids, lower, side):
            nonlocal tests, accepted_blocks
            if len(ids) < 2 or side < 2:
                return
            tests += 1
            if np.all(source_transparent[ids]):
                weights = source_counts[ids]
                normalized = weights/weights.sum()
                centroid = np.sum(coordinates[ids]*normalized[:, None], axis=0)
                distance2 = np.sum((coordinates[ids]-centroid)**2, axis=1)
                # Select an actual source; equal-centroid ties prefer brighter
                # members, then flat index. This is a declared grid convention.
                best = np.lexsort((original[ids], -weights, distance2))[0]
                chosen = ids[best]
                moved = np.linalg.norm(coordinates[ids]-coordinates[chosen], axis=1)
                bound = float(source_distance[ids].min())
                if moved.max() <= theta*bound:
                    anchor[ids] = original[chosen]
                    displacement[ids] = moved
                    surface_bound[ids] = bound
                    accepted_blocks += 1
                    return
            half = side//2
            octants = ((coordinates[ids] >= lower+half).astype(int)*np.array([4,2,1])).sum(axis=1)
            for tag in np.unique(octants):
                shift = half*np.array([(tag >> 2) & 1, (tag >> 1) & 1, tag & 1])
                visit(ids[octants == tag], lower+shift, half)

        visit(np.arange(len(original)), np.zeros(3, dtype=int), width)
    aggregated = np.zeros_like(s)
    np.add.at(aggregated.ravel(), anchor, source_counts)
    denominator = aggregated.ravel()[anchor]
    weights = np.divide(source_counts, denominator, out=np.zeros_like(source_counts),
                        where=denominator > 0)
    residual = float(total-np.sum(aggregated, dtype=np.longdouble))
    if abs(residual) > 5e-13*max(float(total), np.finfo(float).tiny):
        raise FloatingPointError("Source aggregation count ledger failed")
    moved = displacement > 0
    diagnostics = dict(theta=float(theta), original_sources=len(original),
                       aggregate_sources=int(np.count_nonzero(aggregated)),
                       block_tests=tests, accepted_blocks=accepted_blocks,
                       moved_sources=int(np.count_nonzero(moved)),
                       moved_photon_fraction=float(source_counts[moved].sum()/total) if total else 0.,
                       max_displacement=float(displacement.max(initial=0)),
                       max_displacement_to_clearance=float(np.max(np.divide(
                           displacement, surface_bound, out=np.zeros_like(displacement),
                           where=surface_bound > 0), initial=0)),
                       photon_residual=residual,
                       clearance="opaque-padded chessboard center distance minus half a cell",
                       source_geometry_exact=not bool(np.any(moved)),
                       original_ledger_attribution="proportional anchor totals; not ray-resolved",
                       seconds=time.perf_counter()-started)
    return dict(source=aggregated, original_indices=original, anchor_indices=anchor,
                weights=weights, displacements=displacement,
                surface_distance_lower_bound=surface_bound, diagnostics=diagnostics)


def split_ledger(aggregate_values, grouping):
    """Redistribute a per-anchor count array to original emitters proportionally."""
    values = np.asarray(aggregate_values, dtype=float)
    if values.shape != grouping["source"].shape or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Aggregate ledger must be finite nonnegative counts on the same grid")
    if np.any(values[grouping["source"] == 0] != 0):
        raise ValueError("Aggregate ledger has counts at a non-emitter")
    result = np.zeros_like(values)
    result.ravel()[grouping["original_indices"]] = (
        values.ravel()[grouping["anchor_indices"]]*grouping["weights"])
    return result


def solve(source, capacity, initial_ionized=None, *, theta=0., solver=None, **solver_kwargs):
    """Call an angular solver on merged emitters and expand its source ledgers."""
    if solver is None:
        from hybrid import solve as solver
    grouping = merge(source, capacity, initial_ionized, theta=theta)
    result = dict(solver(grouping["source"], capacity, initial_ionized, **solver_kwargs))
    for name in ("source_absorbed", "source_escaped", "source_retained"):
        if name not in result:
            raise ValueError(f"Transport solver omitted {name}; cannot remap its source ledger")
        result[name] = split_ledger(result[name], grouping)
    expected = np.asarray(source, dtype=float)
    ledger = expected-sum(result[name] for name in
                          ("source_absorbed", "source_escaped", "source_retained"))
    active = expected > 0
    relative = float(np.max(np.abs(ledger[active])/expected[active], initial=0))
    if relative > 5e-10:
        raise FloatingPointError("Expanded original-source ledger failed")
    audit = dict(result.get("audit", {}))
    audit["source_merging"] = grouping["diagnostics"]
    audit["merged_original_source_relative_ledger_max"] = relative
    audit["seconds"] = audit.get("seconds", 0.)+grouping["diagnostics"]["seconds"]
    audit["source_geometry_exact"] = grouping["diagnostics"]["source_geometry_exact"]
    result["audit"] = audit
    result["source_grouping"] = grouping
    return result


def benchmark(*, size=32, radius=10., theta_values=(0., .1, .2, .3), max_angles=6144):
    """Small incremental compact cluster inside a pre-ionized spherical cavity."""
    from hybrid import library
    library()  # Keep native compilation out of the measured comparisons.
    center = np.repeat(size//2, 3)
    xyz = np.moveaxis(np.indices((size,)*3), 0, -1)
    cavity = np.linalg.norm(xyz-center, axis=-1) <= radius
    capacity = np.ones((size,)*3)
    old = cavity.astype(float)
    source = np.zeros_like(capacity)
    for offset in np.ndindex(3,3,3):
        source[tuple(center+np.array(offset)-1)] = 1.
    source *= .03*cavity.sum()/source.sum()
    records, baseline = [], None
    for theta in theta_values:
        result = solve(source, capacity, old, theta=theta, min_angles=24,
                       max_angles=max_angles, rays_per_surface_cell=4., periodic=True,
                       maximum_distance=size/2)
        if baseline is None:
            if theta != 0:
                raise ValueError("First benchmark theta must be zero")
            baseline = result
        record = dict(theta=theta, source_merging=result["audit"]["source_merging"],
                      photons=float(source.sum()), absorbed=float(result["absorbed"].sum()),
                      unabsorbed=result["unabsorbed"],
                      l1_new_absorption_per_emitted_photon=float(np.abs(
                          result["absorbed"]-baseline["absorbed"]).sum()/source.sum()),
                      seconds=result["audit"]["seconds"],
                      speedup=baseline["audit"]["seconds"]/result["audit"]["seconds"],
                      angles_sum=result["audit"]["angles_sum"],
                      events=result["audit"]["events"],
                      per_source_relative_ledger_max=result["audit"]["merged_original_source_relative_ledger_max"])
        records.append(record)
    return dict(scope="27-source compact cluster; 3% cavity-atom photon increment; same angular cap for merged/unmerged; timings are single trials",
                size=size, cavity_radius=radius, max_angles=max_angles,
                code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), records=records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--radius", type=float, default=10.)
    parser.add_argument("--max-angles", type=int, default=6144)
    options = parser.parse_args()
    report = benchmark(size=options.size, radius=options.radius, max_angles=options.max_angles)
    text = json.dumps(report, indent=2)
    if options.output:
        options.output.write_text(text+"\n")
    print(text)
