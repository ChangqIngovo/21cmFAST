"""Conservative elliptic obstacle closure: a diffusion control, not UV transport.

Periodic L=-Laplacian has positive nearest-neighbour conductances. Solve
min_{u>=0} .5*u.Lu+(capacity-source).u. The absorbed field source-Lu
satisfies capacity and positivity constraints at a converged KKT solution.
Inputs are a fresh photon increment and currently available atomic capacity.
No source identity, angular budget, finite light speed or opacity is retained.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def laplacian(u):
    result = 6.0 * u
    for axis in range(3):
        result -= np.roll(u, 1, axis=axis)
        result -= np.roll(u, -1, axis=axis)
    return result


def solve(source, capacity, *, tolerance=1e-9, max_iterations=20000):
    """Accelerated projected gradient, with independently reported feasibility.

    A non-converged result is not a valid allocation. No clipping is used to
    conceal its capacity residual. Tolerance is in photon counts per cell.
    """
    started = time.perf_counter()
    s, c = np.asarray(source, dtype=float), np.asarray(capacity, dtype=float)
    if (s.shape != c.shape or s.ndim != 3 or not np.isfinite(s).all()
            or not np.isfinite(c).all() or (s < 0).any() or (c < 0).any()):
        raise ValueError("Finite nonnegative equally shaped 3D counts required")
    q = float(s.sum())
    if q >= float(c.sum()):
        # The obstacle functional has no minimum for Q>C. Report saturation
        # and an unallocated global reservoir; do not invent source debits.
        return dict(absorbed=c.copy(), potential=None,
                    remaining=q-float(c.sum()), audit=dict(
                        converged=True, saturated=True, iterations=0,
                        photons=q, absorbed=float(c.sum()),
                        remaining=q-float(c.sum()), ledger_residual=0.0,
                        seconds=time.perf_counter()-started))
    u = np.zeros_like(s)
    y = u.copy()
    t = 1.0
    linear = c - s
    converged = False
    violation = np.inf
    for iteration in range(1, max_iterations + 1):
        gradient = laplacian(y) + linear
        next_u = np.maximum(y - gradient/12.0, 0.0)
        next_t = (1 + np.sqrt(1+4*t*t))/2
        next_y = next_u + (t-1)/next_t * (next_u-u)
        # Gradient restart reduces late oscillation near the active boundary.
        if np.sum((y-next_u)*(next_u-u)) > 0:
            next_t = 1.0
            next_y = next_u.copy()
        u, y, t = next_u, next_y, next_t
        if iteration % 20 == 0 or iteration == max_iterations:
            g = laplacian(u) + linear
            projected = u - np.maximum(u - g, 0.0)
            violation = float(np.max(np.abs(projected)))
            if violation <= tolerance:
                converged = True
                break
    absorbed = s - laplacian(u)
    ledger = float(np.sum(s-absorbed, dtype=np.longdouble))
    return dict(absorbed=absorbed, potential=u, remaining=0.0, audit=dict(
        converged=converged, saturated=False, iterations=iteration,
        photons=q, absorbed=float(absorbed.sum()), remaining=0.0,
        ledger_residual=ledger, kkt_residual=violation,
        negative_absorption=float(max(0, -absorbed.min())),
        capacity_excess=float(max(0, (absorbed-c).max())),
        seconds=time.perf_counter()-started))


def probes():
    n = 32
    shape = (n,)*3
    s = np.zeros(shape)
    s[(n//2,)*3] = 4*np.pi*5**3/3
    c = np.ones(shape)
    ball = solve(s, c)
    distance = np.sqrt(sum((a-n//2)**2 for a in np.indices(shape)))
    outside = float(ball['absorbed'][distance > 5.5].sum()/s.sum())
    # A highly absorbing planar wall: diffusion collects photons whose
    # initial straight directions would have headed away from the wall.
    wall_c = c.copy()
    wall_c[n//2+2, :, :] = 1e6
    wall_s = np.zeros(shape)
    wall_s[(n//2,)*3] = 1000
    wall = solve(wall_s, wall_c)
    wall_fraction = float(wall['absorbed'][n//2+2].sum()/1000)
    # The second periodic wall image is far outside the occupied front.
    touches_periodic_edge = float(np.abs(wall['absorbed'][[0,-1]]).sum())
    stronger = solve(s*1.1, c)
    monotone_violation = float(max(0, np.max(ball['absorbed']-stronger['absorbed'])))
    shifted = solve(np.roll(s, 3, axis=0), np.roll(c, 3, axis=0))
    translation_error = float(np.max(np.abs(shifted['absorbed']-
                                              np.roll(ball['absorbed'],3,axis=0))))
    return dict(
        scope='Toy incremental counts; no native 21cmFAST evolution',
        physical_model='elliptic diffusion obstacle, not free-streaming UV',
        isolated=dict(**ball['audit'], fraction_outside_R_plus_half=outside),
        plane=dict(**wall['audit'], wall_fraction=wall_fraction,
                   isotropic_straight_ray_upper_bound=0.5,
                   periodic_edge_absorption=touches_periodic_edge),
        source_increase_monotonicity_violation=monotone_violation,
        translation_error=translation_error,
        preliminary_timings=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('obstacle_results.json'))
    args = parser.parse_args()
    result = probes()
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
