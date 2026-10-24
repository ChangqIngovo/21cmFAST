#!/usr/bin/env python3
"""Fixed-source front refinement: native morphology, analytic accuracy and cost.

Every native arm uses the same point sources, atom density and measurement grid.
The integrated isolated sphere is an independent continuum accuracy reference.
No field from a cosmological history is treated as a continuum oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
for folder in ('pc_large_scale', 'pc_adaptive_front', 'pc_scale_origin', 'pc_reconsidered'):
    sys.path.insert(0, str(RESEARCH/folder))
sys.path.insert(0, str(HERE))
import adaptive
import point_front
import refined
from compare_models import sphere_reference
from pc_statistics import coarsen_counts, spectrum, total


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()


def write(report, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


def independent_ledger(out, source, capacity):
    emitted = np.sum(source, dtype=np.longdouble)
    delta = emitted-sum(np.sum(out[key], dtype=np.longdouble)
                        for key in ('absorbed', 'source_escaped', 'source_retained'))
    src = sum(out[key] for key in ('source_absorbed', 'source_escaped', 'source_retained'))
    mask = source > 0
    scale = max(float(emitted), np.finfo(float).tiny)
    violation = np.maximum(np.maximum(out['absorbed']-capacity, 0), -out['absorbed'])
    local = np.divide(violation, capacity, out=np.zeros_like(capacity), where=capacity > 0)
    report = dict(relative_count_residual=float(abs(delta)/scale),
                  maximum_source_relative_residual=float(np.max(abs(src[mask]-source[mask])/source[mask])) if mask.any() else 0.,
                  maximum_local_fraction_violation=float(np.max(local)))
    if max(report.values()) > 2e-9:
        raise RuntimeError(f'Independent photon audit failed: {report}')
    return report


def difference(a, b, source, capacity):
    delta = abs(a-b)
    fraction = np.divide(delta, capacity, out=np.zeros_like(capacity), where=capacity > 0)
    return dict(L1_over_emitted=total(delta)/max(total(source), np.finfo(float).tiny),
                maximum_ionized_fraction_difference=float(fraction.max()))


def run_models(source, capacity, *, factors, angles, distance, repeats,
               include_adaptive=True, dense_reference=False, analytic=None,
               template_control=False):
    names = [('refined', f) for f in factors]
    if template_control:
        names += [('refined_plain', f) for f in factors]
    if include_adaptive:
        names.insert(0, ('adaptive', 2))
    def solve(name, factor):
        if name == 'adaptive':
            return adaptive.solve(source, capacity, min_angles=angles, max_angles=angles,
                                  maximum_distance=distance)
        return refined.solve(source, capacity, subcells_per_axis=factor, angles=angles,
                             maximum_distance=distance, use_local_templates=name != 'refined_plain')
    # Compile before first-solve measurements. Cache construction remains timed.
    refined.library()
    if include_adaptive:
        adaptive.library()
    rows = {f'{name}_f{factor}': dict(model=name, factor=factor, seconds_samples=[])
            for name, factor in names}
    for name, factor in names:
        cold = name == 'refined' and hasattr(refined, 'clear_template_cache')
        if cold:
            refined.clear_template_cache()
        started = time.perf_counter()
        first = solve(name, factor)
        rows[f'{name}_f{factor}'].update(first_solve_seconds=time.perf_counter()-started,
                                       first_solve_audit=first['audit'],
                                       first_solve_template_cache_cleared=cold)
    # Populate every cache again after the deliberately cold measurements.
    for name, factor in names:
        solve(name, factor)
    saved = {}
    for repeat in range(repeats):
        for name, factor in (names if repeat % 2 == 0 else names[::-1]):
            key = f'{name}_f{factor}'
            started = time.perf_counter()
            out = solve(name, factor)
            seconds = time.perf_counter()-started
            row = rows[key]
            row['seconds_samples'].append(seconds)
            row.update(audit=out['audit'], independent_ledger=independent_ledger(out, source, capacity),
                       median_seconds=float(np.median(row['seconds_samples'])),
                       absorbed_sha256=array_digest(out['absorbed']),
                       retained_fraction=total(out['source_retained'])/total(source),
                       escaped_fraction=total(out['source_escaped'])/total(source))
            if analytic is not None:
                row['vs_integrated_continuum_sphere'] = difference(out['absorbed'], analytic, source, capacity)
            saved[key] = out
    if 'adaptive_f2' in saved and 'refined_f2' in saved:
        rows['refined_f2']['vs_adaptive'] = difference(saved['refined_f2']['absorbed'], saved['adaptive_f2']['absorbed'], source, capacity)
    if 'adaptive_f2' in rows:
        baseline = rows['adaptive_f2']['median_seconds']
        for row in rows.values():
            row['time_over_existing_adaptive_f2'] = row['median_seconds']/baseline
    if template_control:
        for factor in factors:
            key, plain = f'refined_f{factor}', f'refined_plain_f{factor}'
            check = difference(saved[key]['absorbed'], saved[plain]['absorbed'], source, capacity)
            check['source_ledger_L1_over_emitted'] = {
                field: total(abs(saved[key][field]-saved[plain][field]))/total(source)
                for field in ('source_absorbed', 'source_retained', 'source_escaped')}
            if check['L1_over_emitted'] > 2e-10 or max(check['source_ledger_L1_over_emitted'].values()) > 2e-10:
                raise RuntimeError(f'Local template optimization changed the model: {check}')
            rows[key]['vs_template_disabled'] = check
            rows[key]['time_over_template_disabled'] = rows[key]['median_seconds']/rows[plain]['median_seconds']
    if dense_reference:
        n = len(source)
        coordinates = np.indices(source.shape).reshape(3, -1).T+.5
        active = source.ravel() > 0
        for factor in factors:
            cap = capacity.repeat(factor, 0).repeat(factor, 1).repeat(factor, 2)/factor**3
            ref = point_front.solve(coordinates[active]*factor, source.ravel()[active], cap,
                                    angles=angles, maximum_distance=distance*factor)
            projected = coarsen_counts(ref['absorbed'], n)
            rows[f'refined_f{factor}']['vs_dense_point_reference'] = difference(
                saved[f'refined_f{factor}']['absorbed'], projected, source, capacity)
    return rows, saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native-grid', type=int, default=8)
    parser.add_argument('--measurement-grid', type=int, default=8)
    parser.add_argument('--doses', nargs='+', type=float, default=[.5, .8, .9])
    parser.add_argument('--angles', nargs='+', type=int, default=[384])
    parser.add_argument('--factors', nargs='+', type=int, default=[2, 4, 8])
    parser.add_argument('--horizon-boxes', type=float, default=16.)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--sphere', action='store_true')
    parser.add_argument('--sphere-grid', type=int, default=16)
    parser.add_argument('--sphere-radius', type=float, default=3.75)
    parser.add_argument('--dense-reference', action='store_true')
    parser.add_argument('--no-adaptive', action='store_true')
    parser.add_argument('--template-control', action='store_true')
    args = parser.parse_args()
    if args.repeats < 1 or any(not 0 < q < 1 for q in args.doses):
        raise ValueError('Positive repeats and doses between zero and one required')
    if args.measurement_grid < 2 or args.native_grid % args.measurement_grid:
        raise ValueError('Measurement grid must divide the native source grid')
    deps = [Path(__file__), HERE/'refined.py', HERE/'refined.cpp',
            RESEARCH/'pc_adaptive_front/adaptive.py', RESEARCH/'pc_adaptive_front/adaptive.cpp',
            RESEARCH/'pc_scale_origin/point_front.py', RESEARCH/'pc_scale_origin/point_front.cpp',
            RESEARCH/'pc_large_scale/pc_statistics.py', RESEARCH/'pc_reconsidered/compare_models.py']
    hashes = {str(p.relative_to(RESEARCH)): digest(p) for p in deps}
    report = dict(scope='Static fixed-source gas-front refinement; no cosmological evolution',
                  timing='Compilation excluded; first solve reported separately (cache cleared when supported); one subsequent warm solve; alternating sequential samples; serial transport',
                  code_sha256=hashes, configuration=vars(args).copy(), cases=[])
    report['configuration'] = {k: str(v) if isinstance(v, Path) else v for k, v in report['configuration'].items()}
    jobs = []
    if args.fixture:
        with np.load(args.fixture) as data:
            source = coarsen_counts(data['source'], args.native_grid)
            capacity = coarsen_counts(data['capacity'], args.native_grid)
        report['fixture_sha256'] = digest(args.fixture)
        for dose in args.doses:
            s = source*(dose*total(capacity)/total(source))
            jobs.append((f'native{args.native_grid}_Q{dose}', s, capacity,
                         args.horizon_boxes*args.native_grid, None, {}))
    if args.sphere:
        n, radius = args.sphere_grid, args.sphere_radius
        if radius+1 >= n/2:
            raise ValueError('Sphere must stay inside its periodic box')
        centre = np.full(3, n//2)
        s, c = np.zeros((n,)*3), np.ones((n,)*3)
        s[tuple(centre)] = 4*np.pi*radius**3/3
        exact = sphere_reference(n, radius, centre, quadrature=128)
        check = sphere_reference(n, radius, centre, quadrature=64)
        jobs.append(('sphere', s, c, n/2, exact,
                     dict(radius=radius, quadrature64_to128_L1_over_emitted=total(abs(exact-check))/total(s),
                          relative_sphere_volume_error=total(exact)/total(s)-1)))
    if not jobs:
        raise ValueError('Select --fixture and/or --sphere')
    previous_angles = {}
    report['angular_checks'] = []
    for name, source, capacity, distance, analytic, oracle in jobs:
        for angles in args.angles:
            rows, saved = run_models(source, capacity, factors=args.factors, angles=angles,
                                     distance=distance, repeats=args.repeats,
                                     include_adaptive=not args.no_adaptive,
                                     dense_reference=args.dense_reference, analytic=analytic,
                                     template_control=args.template_control)
            item = dict(name=name, angles=angles, source_sha256=array_digest(source),
                        capacity_sha256=array_digest(capacity), source_grid=len(source),
                        maximum_distance=distance, oracle=oracle, models=rows, contrasts=[])
            for key, out in saved.items():
                if name.startswith('native'):
                    box, band = 135.6, (.04, .08)
                    measured_capacity = coarsen_counts(capacity, args.measurement_grid)
                    measured_absorbed = coarsen_counts(out['absorbed'], args.measurement_grid)
                    edges = np.geomspace(2*np.pi/box*(1-1e-10), np.pi*args.measurement_grid/box*(1+1e-10), 20)
                    stat = spectrum(measured_capacity, measured_absorbed, box, edges, band, 8.)
                    b = stat['k0_band']
                    volume = float(np.mean(np.divide(out['absorbed'], capacity, out=np.zeros_like(capacity), where=capacity > 0)))
                    rows[key]['statistics'] = dict(Q_mass=stat['Q_mass'], Q_volume=volume,
                                                  measurement_Q_volume=stat['Q_volume'],
                                                  measurement_grid=args.measurement_grid, **b,
                                                  signed_cross_bias=b['P_cross']/b['P_matter'],
                                                  density_ionization_covariance=stat['Q_mass']-volume)
            for a, b in zip(args.factors[:-1], args.factors[1:]):
                before, after = f'refined_f{a}', f'refined_f{b}'
                contrast = dict(before=before, after=after,
                                difference=difference(saved[before]['absorbed'], saved[after]['absorbed'], source, capacity),
                                time_ratio=rows[after]['median_seconds']/rows[before]['median_seconds'])
                if name.startswith('native'):
                    contrast['b2_ratio'] = rows[after]['statistics']['b_HI_squared']/rows[before]['statistics']['b_HI_squared']
                item['contrasts'].append(contrast)
            if name in previous_angles:
                old_angles, old_rows, old_saved = previous_angles[name]
                for key in rows:
                    check = dict(case=name, model=key, before_angles=old_angles, after_angles=angles,
                                 field_difference=difference(old_saved[key]['absorbed'], saved[key]['absorbed'], source, capacity))
                    if name.startswith('native'):
                        check['b2_ratio'] = rows[key]['statistics']['b_HI_squared']/old_rows[key]['statistics']['b_HI_squared']
                    report['angular_checks'].append(check)
            previous_angles[name] = (angles, rows, saved)
            report['cases'].append(item)
            write(report, args.output)
            print(json.dumps(dict(name=name, angles=angles, contrasts=item['contrasts'],
                                  models={key: dict(seconds=value['median_seconds'],
                                                    sphere=value.get('vs_integrated_continuum_sphere'),
                                                    b2=value.get('statistics', {}).get('b_HI_squared'))
                                          for key, value in rows.items()})), flush=True)
    final_hashes = {str(p.relative_to(RESEARCH)): digest(p) for p in deps}
    if hashes != final_hashes:
        raise RuntimeError('Code changed during benchmark; result must be rerun')
    write(report, args.output)


if __name__ == '__main__':
    main()
