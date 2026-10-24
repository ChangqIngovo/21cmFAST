"""Control static transport truncation while preserving source histories.

Each radius is a fresh solve from the same initial gas state and emitted budget.
Retained source totals are never re-emitted: they lack ray directions and times.
This tests distance convergence only, not angle or spatial convergence.
"""
from __future__ import annotations

import numpy as np

from adaptive import solve


def solve_distance_converged(source, capacity, *, distances,
                             retained_tolerance=1e-4, field_tolerance=1e-3,
                             statistic=None, statistic_rtol=0.01,
                             statistic_atol=0., **options):
    """Return (last solution, convergence report), including failed convergence.

    ``statistic(result)`` may supply a finite scalar such as large-scale HI bias
    squared. A near-zero statistic needs a scientifically chosen absolute
    tolerance. At least two radii are required, and all acceptance conditions
    must hold on the last consecutive pair. Open boundaries are unsupported.
    """
    radii = np.asarray(list(distances), dtype=float)
    if (radii.ndim != 1 or len(radii) < 2 or not np.isfinite(radii).all()
            or (radii <= 0).any() or (np.diff(radii) <= 0).any()):
        raise ValueError('At least two finite positive increasing distances required')
    tolerances = (retained_tolerance, field_tolerance, statistic_rtol, statistic_atol)
    if any(not np.isfinite(v) or v < 0 for v in tolerances):
        raise ValueError('Finite nonnegative tolerances required')
    if options.get('periodic', True) is not True:
        raise ValueError('Distance controller requires periodic boundaries')
    if 'maximum_distance' in options:
        raise ValueError('Supply distances rather than maximum_distance')
    s, c = np.array(source, dtype=float, copy=True), np.array(capacity, dtype=float, copy=True)
    emitted = float(s.sum(dtype=np.longdouble))
    if not np.isfinite(emitted) or emitted <= 0:
        raise ValueError('Positive finite emitted budget required')
    if options.get('angular_reference_distance') is None:
        options['angular_reference_distance'] = float(radii[0])
    previous = None
    history = []
    converged = False
    for radius in radii:
        result = solve(s, c, maximum_distance=float(radius), **options)
        retained = float(result['source_retained'].sum(dtype=np.longdouble))/emitted
        value = None if statistic is None else float(statistic(result))
        if value is not None and not np.isfinite(value):
            raise ValueError('Convergence statistic must be finite')
        row = {'maximum_distance': float(radius), 'retained_fraction': retained,
               'statistic': value, 'seconds': result['audit']['seconds'],
               'field_L1_over_emitted': None, 'statistic_absolute_change': None}
        if previous is not None:
            if not np.array_equal(previous['angle_counts'], result['angle_counts']):
                raise RuntimeError('Angular quadrature changed during distance comparison')
            error = float(np.sum(abs(result['absorbed']-previous['absorbed']), dtype=np.longdouble))/emitted
            delta = None if value is None else abs(value-history[-1]['statistic'])
            row.update(field_L1_over_emitted=error, statistic_absolute_change=delta)
            converged = (retained <= retained_tolerance and error <= field_tolerance
                         and (delta is None or delta <= statistic_atol+statistic_rtol*abs(value)))
        history.append(row)
        if converged:
            break
        previous = result
    return result, {'distance_converged': bool(converged), 'history': history,
                    'retained_tolerance': retained_tolerance, 'field_tolerance': field_tolerance,
                    'statistic_rtol': statistic_rtol, 'statistic_atol': statistic_atol,
                    'angular_reference_distance': options['angular_reference_distance'],
                    'source_budget_rescaled': False, 'retained_photons_reemitted': False,
                    'scope': 'successive-distance stability at fixed angular and spatial discretization'}
