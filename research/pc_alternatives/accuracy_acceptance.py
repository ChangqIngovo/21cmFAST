"""Independent acceptance checks for the repaired research components.

The checks separate implementation agreement from a continuum/model claim.
In particular, a finite collection of converged angular rules cannot certify
arbitrary spatial features, and a finite-c no-recombination transport solver is
not the same closure as quasistatic finite-opacity recombination chemistry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LD = np.longdouble


def event_symmetry_and_clock():
    from event_front import solve

    rng = np.random.default_rng(78193)
    source = np.zeros((5,) * 3)
    source[1, 2, 1], source[3, 1, 4], source[4, 4, 2] = 20, 11, 7
    capacity = rng.uniform(.01, 2, source.shape)
    capacity[rng.random(source.shape) < .3] = 0
    raw = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, 0, -1],
                    [1, 1, 1], [-1, 1, 1], [1, 1 + 1e-10, 0],
                    [-1, -2, 3]], dtype=float)
    directions = raw / np.linalg.norm(raw, axis=1)[:, None]
    errors = []
    keys = ('absorbed', 'source_absorbed', 'source_escaped', 'source_retained')
    for periodic in (False, True):
        reference = solve(source, capacity, ray_directions=directions,
                          periodic=periodic, maximum_distance=8)
        for permutation, signs in [((2, 0, 1), (-1, 1, -1)),
                                   ((1, 0, 2), (1, -1, 1))]:
            def transform(a):
                a = a.transpose(permutation)
                for axis, sign in enumerate(signs):
                    if sign < 0:
                        a = np.flip(a, axis=axis)
                return a.copy()
            actual = solve(transform(source), transform(capacity),
                           ray_directions=directions[:, permutation] * signs,
                           periodic=periodic, maximum_distance=8)
            errors.append(max(float(np.abs(actual[k] - transform(reference[k])).max())
                              for k in keys))
        if periodic:
            shift = (2, -1, 3)
            actual = solve(np.roll(source, shift, (0, 1, 2)),
                           np.roll(capacity, shift, (0, 1, 2)),
                           ray_directions=directions, periodic=True, maximum_distance=8)
            errors.append(max(float(np.abs(actual[k] - np.roll(reference[k], shift, (0, 1, 2))).max())
                              for k in keys))
    reference = solve(source, capacity, ray_directions=directions, maximum_distance=8)
    clock_rows = []
    for rate_scale in (1e-290, 1e-200, 1e-100, 1e-20, 1, 1e20, 1e100, 1e200, 1e290):
        actual = solve(source * rate_scale, capacity, ray_directions=directions,
                       duration=1 / rate_scale, maximum_distance=8)
        clock_rows.append(dict(rate_scale=rate_scale, duration=1 / rate_scale,
                               maximum_field_error=max(float(np.abs(actual[k] - reference[k]).max())
                                                       for k in keys)))
    return dict(symmetry_cases=len(errors), maximum_symmetry_error=max(errors),
                clock_rows=clock_rows, passed=max(errors) < 1e-11 and
                max(v['maximum_field_error'] for v in clock_rows) < 1e-11)


def numeric_domain_rejection():
    from event_front import solve

    cases = [(np.full((1,) * 3, 1e308), np.ones((1,) * 3), 10, 1),
             (np.full((2,) * 3, 1e308), np.ones((2,) * 3), 1e-308, 1),
             (np.full((1,) * 3, 1e-320), np.zeros((1,) * 3), 1, 49152)]
    rows = []
    for source, capacity, duration, angles in cases:
        try:
            solve(source, capacity, duration=duration, angles=angles)
        except (ValueError, FloatingPointError, OverflowError) as error:
            rows.append(dict(rejected=True, reason=str(error)))
        else:
            rows.append(dict(rejected=False))
    return dict(cases=rows, passed=all(v['rejected'] for v in rows))


def packet_scan_reference(capacity, directions, flashes, *, speed, until,
                          periodic=False, maximum_distance=10):
    """Eager all-intersection scan, independent of candidate queue/skip logic.

    Each geometric crossing is scheduled in advance, including transparent
    ones. A packet killed at an earlier crossing contributes zero later. This
    deliberately shares neither mutable cursors nor candidate path caching.
    The ordinary DDA is the previously audited independent geometry reference.
    """
    from validation_event_reference import ordinary_dda

    n = capacity.shape[0]
    events, beam_sources, beam_amounts, emission_times = [], [], [], []
    for xyz, amount, emission in flashes:
        sid = np.ravel_multi_index(xyz, capacity.shape)
        origin = np.asarray(xyz, dtype=LD) + LD('.5')
        for direction in directions:
            beam = len(beam_sources)
            beam_sources.append(sid)
            beam_amounts.append(LD(amount) / len(directions))
            emission_times.append(LD(emission))
            path = ordinary_dda(n, xyz, direction, maximum_distance, periodic=periodic)
            for index, (cell, distance) in enumerate(zip(path.cells, path.entry_distances)):
                events.append((LD(emission) + LD(distance) / LD(speed), beam, cell, index == 0, 'cell'))
            bounds = []
            if not periodic:
                for axis, v in enumerate(direction):
                    if v > 0:
                        bounds.append((LD(n) - origin[axis]) / LD(v))
                    elif v < 0:
                        bounds.append(-origin[axis] / LD(v))
            terminal_distance = min([LD(maximum_distance)] + bounds)
            events.append((LD(emission) + terminal_distance / LD(speed), beam, -1, False, path.terminal))
    events.sort(key=lambda e: e[0])
    remaining = capacity.ravel().astype(LD).copy()
    absorbed = np.zeros_like(remaining)
    source_abs = np.zeros_like(remaining)
    source_escape = np.zeros_like(remaining)
    source_retained = np.zeros_like(remaining)
    photons = np.zeros(len(beam_sources), dtype=LD)
    cursor = 0
    while cursor < len(events) and events[cursor][0] <= LD(until):
        stamp = events[cursor][0]
        grouped = {}
        while cursor < len(events) and events[cursor][0] == stamp:
            _, beam, cell, born, kind = events[cursor]
            cursor += 1
            if born:
                photons[beam] = beam_amounts[beam]
            if kind == 'cell':
                if photons[beam] > 0:
                    grouped.setdefault(cell, []).append(beam)
            else:
                target = source_escape if kind == 'escaped' else source_retained
                target[beam_sources[beam]] += photons[beam]
                photons[beam] = 0
        for cell, ids in grouped.items():
            total = photons[ids].sum()
            loss = min(total, remaining[cell])
            individual = photons[ids] * (loss / total)
            for beam, count in zip(ids, individual):
                source_abs[beam_sources[beam]] += count
            photons[ids] -= individual
            absorbed[cell] += individual.sum()
            remaining[cell] -= individual.sum()
    inflight, emitted = np.zeros_like(remaining), np.zeros_like(remaining)
    for beam, sid in enumerate(beam_sources):
        inflight[sid] += photons[beam]
        if emission_times[beam] <= LD(until):
            emitted[sid] += beam_amounts[beam]
    shape = capacity.shape
    return {k: np.asarray(v, dtype=float).reshape(shape) for k, v in dict(
        absorbed=absorbed, source_absorbed=source_abs, source_escaped=source_escape,
        source_range_retained=source_retained, source_in_flight=inflight,
        source_emitted=emitted).items()}


def packet_reference_cases(count=16):
    from finite_c_packets import FiniteCPackets

    rng = np.random.default_rng(15790)
    directions = np.array([[1., 0, 0], [-1., 0, 0], [0, 1., 0], [0, 0, -1.]])
    rows = []
    for case in range(count):
        capacity = rng.uniform(.01, 1, (4,) * 3)
        capacity[rng.random(capacity.shape) < .5] = 0
        flashes = [(tuple(rng.integers(0, 4, 3)), float(rng.uniform(.01, 4)), stamp)
                   for stamp in (0., .125, .125, .75)]
        periodic = case % 2 == 0
        speed, until, distance = 2., 3.125, 5.75
        solver = FiniteCPackets(capacity, speed=speed, ray_directions=directions,
                               periodic=periodic, maximum_distance=distance)
        for xyz, amount, stamp in flashes:
            solver.schedule_flash(xyz, amount, emission_time=stamp)
        actual = solver.advance(until)
        reference = packet_scan_reference(capacity, directions, flashes, speed=speed,
                                          until=until, periodic=periodic,
                                          maximum_distance=distance)
        errors = {k: float(np.abs(actual[k] - reference[k]).max()) for k in reference}
        rows.append(dict(case=case, periodic=periodic, errors=errors,
                         passed=max(errors.values()) < 1e-12))
    return dict(cases=rows, passed=all(r['passed'] for r in rows))


def source_history_refinement():
    from source_history import RateHistory, SourceLedger

    rng = np.random.default_rng(59143)
    times = np.array([0., .23, .91, 1.75, 2.01, 4.])
    rates = rng.uniform(0, 20, (len(times), 3))
    original = RateHistory.from_knots(times, rates)
    refined_times = np.unique(np.r_[times, rng.uniform(0, 4, 37)])
    refined = RateHistory.from_knots(refined_times, [original.rate_at(t) for t in refined_times])
    exact = np.sum(np.diff(times)[:, None] * (rates[:-1] + rates[1:]) / 2, axis=0)
    ledger = SourceLedger(refined, [100, 100, 100])
    for stop in np.unique(np.r_[rng.uniform(0, 4, 71), 4]):
        ledger.advance(stop)
    return dict(maximum_integral_error=float(np.abs(refined.integrate(0, 4) - exact).max()),
                maximum_subdivision_error=float(np.abs(ledger.emitted - exact).max()),
                passed=bool(np.max(np.abs(ledger.emitted - exact)) < 5e-13))


def ray_segment_reference_cases(count=48):
    """Compare every chord against independent per-voxel slab intersections."""
    from ray_segments import trace_segments

    rng = np.random.default_rng(51947)
    shape = (3, 4, 5)
    errors = []
    for case in range(count):
        width = .2 if case % 2 else 1.
        origin = rng.uniform(.05, .95, 3) * np.array(shape) * width
        velocity = rng.normal(size=3)
        if case == 0:
            origin = np.full(3, .5) * width
            velocity = np.array([1., 1 + 1e-10, 0])
        if case == 1:
            origin = np.array([0., 1.5, 2.5]) * width
            velocity = np.array([1., 0, 0])
        if case == 2:
            origin = np.array([3., 1.5, 2.5]) * width
            velocity = np.array([-1., 0, 0])
        if case == 3:
            origin = np.full(3, .5) * width
            velocity = np.ones(3)
        velocity /= np.linalg.norm(velocity)
        horizon = float(rng.uniform(.1, 6) * width)
        actual = trace_segments(shape, origin, velocity, cell_width=width,
                                maximum_distance=horizon)
        reference = []
        for voxel in np.ndindex(shape):
            lower = np.array(voxel, dtype=LD) * LD(width)
            upper = (np.array(voxel, dtype=LD) + 1) * LD(width)
            lo, hi = LD(0), LD(horizon)
            for axis in range(3):
                v, start = LD(velocity[axis]), LD(origin[axis])
                if v == 0:
                    if not lower[axis] <= start < upper[axis]:
                        hi = LD(-1)
                        break
                else:
                    ends = ((lower[axis] - start) / v, (upper[axis] - start) / v)
                    lo, hi = max(lo, min(ends)), min(hi, max(ends))
            if hi > lo:
                reference.append((lo, int(np.ravel_multi_index(voxel, shape)), hi - lo))
        reference.sort()
        cells = tuple(row[1] for row in reference)
        error = (float(np.max(np.abs(np.array(actual.lengths, dtype=LD) -
                                     np.array([row[2] for row in reference], dtype=LD))))
                 if cells == actual.cells and cells else (0. if cells == actual.cells else float('inf')))
        errors.append(error)
    return dict(cases=count, maximum_chord_error=max(errors), passed=max(errors) < 5e-13,
                reference='Independent slab intersection with each voxel, without DDA traversal')


def coupled_opacity_order_and_slab():
    from coupled_opacity import solve
    from validation_recombination import opacity_reference

    atoms, initial, alpha, tau = [2., .7, 1.3], [.2, .9, .1], [.5, 1.3, .7], [3., .2, 11.]
    paths, rates, duration = [[0, 1, 2], [2, 1, 0], [1]], [.3, 1.1, .2], 1.7
    reference = opacity_reference(atoms, initial, alpha, tau, paths, rates, duration)
    result = solve(atoms, initial, alpha, tau, paths, rates, duration, rtol=1e-8, atol=1e-11)
    reverse = solve(atoms, initial, alpha, tau, paths[::-1], rates[::-1], duration,
                    rtol=1e-8, atol=1e-11)
    slabs = []
    for cells in (1, 2, 8, 32):
        slab = solve(np.full(cells, 3. / cells), np.full(cells, .2), np.zeros(cells),
                     np.full(cells, 5. / cells), [list(range(cells))], [1.3], 1.7, steps=1)
        slabs.append(dict(cells=cells, absorbed=float(slab['absorbed'].sum()),
                          escaped=float(slab['escaped_by_beam'].sum())))
    slab_error = max(abs(v['absorbed'] - slabs[0]['absorbed']) for v in slabs)
    reference_error = float(np.max(np.abs(result['x'] - reference['x'])))
    order_error = float(np.max(np.abs(result['x'] - reverse['x'])))
    return dict(reference_method=reference['method'], reference_x_error=reference_error,
                reversed_beam_order_x_error=order_error, slab_rows=slabs,
                slab_integrated_absorption_maximum_difference=slab_error,
                passed=reference_error < 5e-6 and order_error < 5e-6 and slab_error < 1e-12)


def angular_blind_spot():
    """A converged finite quadrature can miss a finite-solid-angle feature.

    This is an acceptance-scope counterexample, not a bug to conceal by changing
    the tolerances. Angular-controller output must explicitly say empirical.
    """
    from angular_accuracy import solve_controlled

    n = 96
    source = np.zeros((n,) * 3)
    capacity = np.zeros_like(source)
    source[1, n // 2, n // 2] = 1
    capacity[n - 2, n // 2, n // 2] = 10
    result = solve_controlled(source, capacity, initial_angles=3, maximum_angles=48,
                              angular_rtol=1e-12, angular_atol=0, range_rtol=0,
                              range_atol=0, raise_unresolved=False)
    # Exact solid angle of the nearer square face. With the source on the cube
    # symmetry axis, every intersecting ray enters that face.
    distance = LD(n - 3) - LD('.5')
    omega = 4 * np.arctan(LD('.25') / (distance * np.sqrt(distance**2 + LD('.5'))))
    true_absorbed = float(omega / (4 * np.pi))
    return dict(n=n, photon_count=1, exact_geometric_absorbed_fraction=true_absorbed,
                computed_absorbed=float(result['absorbed'].sum()),
                controller_status=result['accuracy']['status'],
                stop_reason=result['accuracy'].get('stop_reason'),
                historical_pre_guard_status='empirically_converged (same maps and tolerances)',
                angular_tolerance_photons=result['accuracy']['angular_tolerance_photons'],
                controller_claims_physical_certificate=result['accuracy']['spatial_or_physical_accuracy_certified'],
                limitation='Zero-hit guard fixes this acceptance error; partially missed features can still agree across finite quadratures',
                passed=result['accuracy']['spatial_or_physical_accuracy_certified'] is False
                and result['accuracy']['status'] == 'unresolved'
                and result['accuracy'].get('stop_reason') == 'zero_absorption_coverage_unresolved')


def run():
    output = dict(scope='Independent discrete-component checks; not a complete coupled Qin20 accuracy certificate',
                  event_symmetry_clock=event_symmetry_and_clock(),
                  numeric_domain=numeric_domain_rejection(),
                  finite_c_reference=packet_reference_cases(),
                  source_history=source_history_refinement(),
                  chord_geometry=ray_segment_reference_cases(),
                  coupled_opacity=coupled_opacity_order_and_slab(),
                  angular_scope_counterexample=angular_blind_spot())
    output['all_declared_checks_passed'] = all(v['passed'] for v in output.values() if isinstance(v, dict) and 'passed' in v)
    output['source_hashes'] = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                             for name in ['accuracy_acceptance.py', 'event_front.py', 'event_front.cpp',
                                          'finite_c_packets.py', 'source_history.py', 'coupled_opacity.py',
                                          'angular_accuracy.py', 'ray_segments.py']}
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=HERE / 'validation_accuracy_acceptance.json')
    args = parser.parse_args()
    result = run()
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
