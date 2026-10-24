#!/usr/bin/env python3
"""Hold source positions and piecewise gas density independently fixed.

An explicit-point reference separates density smoothing, source redistribution,
source-centre relocation, and opaque-front gas resolution. All reported spectra
use one common measurement mesh. This is a frozen small-fixture diagnosis, not a
cosmological volume or continuum convergence claim.
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
sys.path.insert(0, str(RESEARCH/'pc_large_scale'))
from pc_statistics import coarsen_counts, spectrum, total
from point_front import library, solve


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def lift(a, grid):
    """Represent the same piecewise-constant density on a finer gas mesh."""
    factor = grid//len(a)
    if factor < 1 or grid % len(a):
        raise ValueError('Gas grid must refine the density grid by an integer factor')
    return np.ascontiguousarray(a.repeat(factor, 0).repeat(factor, 1).repeat(factor, 2)/factor**3)


def positions(n):
    """Original point locations in units of the box length."""
    return (np.indices((n,)*3).reshape(3, -1).T+.5)/n


def summarize(capacity, absorbed, measurement_grid, box, band):
    c, a = coarsen_counts(capacity, measurement_grid), coarsen_counts(absorbed, measurement_grid)
    edges = np.geomspace(2*np.pi/box*(1-1e-10), np.pi*measurement_grid/box*(1+1e-10), 20)
    stat = spectrum(c, a, box, edges, band, 8.)
    k0 = stat['k0_band']
    if k0 is None or not k0['P_matter'] > 0:
        raise ValueError('Measurement band needs modes and nonzero matter power')
    return a, dict(Q_mass=stat['Q_mass'], Q_volume=stat['Q_volume'], **k0,
                   signed_cross_bias=k0['P_cross']/k0['P_matter'],
                   stochastic_HI_power=k0['P_HI']-k0['P_cross']**2/k0['P_matter'])


def audit_counts(result, photons, cap):
    emitted = np.sum(photons, dtype=np.longdouble)
    residual = emitted-sum(np.sum(result[k], dtype=np.longdouble)
                           for k in ('absorbed', 'source_retained', 'source_escaped'))
    per_source = sum(result[k] for k in ('source_absorbed', 'source_retained', 'source_escaped'))-photons
    active = photons > 0
    report = dict(global_relative_residual=float(abs(residual)/emitted),
                  maximum_source_relative_residual=float(np.max(abs(per_source[active])/photons[active])),
                  maximum_local_fraction_violation=float(np.max(np.maximum(result['absorbed']-cap, 0)/cap)),
                  negative_fraction=float(np.max(np.maximum(-result['absorbed'], 0)/cap)))
    if max(report.values()) > 2e-9:
        raise RuntimeError(f'Independent count audit failed: {report}')
    return report


def run(fixture, output, *, base_grid=16, measurement_grid=8, doses=(.8, .9),
        angles=96, box_hMpc=135.6, band=(.04, .08), horizon_boxes=4.,
        doubled_angles=True, selected=None):
    fixture, output = Path(fixture).resolve(), Path(output).resolve()
    deps = [HERE/'point_front.cpp', HERE/'point_front.py', Path(__file__),
            RESEARCH/'pc_refinement/front_subcells.py', RESEARCH/'pc_large_scale/pc_statistics.py']
    hashes = {str(p.relative_to(RESEARCH)): digest(p) for p in deps}
    fixture_hash = digest(fixture)
    with np.load(fixture) as f:
        sf, cf = np.array(f['source']), np.array(f['capacity'])
    if base_grid % 2 or base_grid % measurement_grid or base_grid <= measurement_grid:
        raise ValueError('Use an even base grid and a proper divisor measurement grid')
    if any(not 0 < q < 1 for q in doses):
        raise ValueError('Sub-unity photon doses required')
    s, c = coarsen_counts(sf, base_grid), coarsen_counts(cf, base_grid)
    ncoarse = base_grid//2
    sc, cc = coarsen_counts(s, ncoarse), coarsen_counts(c, ncoarse)
    ss, cs = lift(sc, base_grid), lift(cc, base_grid)
    # source array specifies positions at that array's cell centres; density is
    # lifted independently. Thus A/B have the identical point sources and A/C
    # have exactly the identical gas field, not just equal total atom counts.
    variants = {
        'A_fine_sources_gas2': (s, c, base_grid*2),
        'B_density_smooth_gas2': (s, cs, base_grid*2),
        'C_aggregate_sources_gas2': (sc, c, base_grid*2),
        'D_both_coarse_inputs_gas2': (sc, cs, base_grid*2),
        'E_coarse_front_gas1': (sc, cs, base_grid),
        'F_source_amplitudes_smooth_gas2': (ss, c, base_grid*2),
        'G_both_smooth_gas2': (ss, cs, base_grid*2),
        'H_fine_sources_gas1': (s, c, base_grid),
        'I_fine_sources_gas4': (s, c, base_grid*4),
    }
    if selected is not None:
        if any(name not in variants for name in selected):
            raise ValueError('Unknown selected variant')
        variants = {name: variants[name] for name in selected}
    report = dict(scope='Frozen source-position/density/front-mesh factorial; all spectra on one mesh',
                  fixture=str(fixture), fixture_sha256=fixture_hash, code_sha256=hashes,
                  config=dict(base_grid=base_grid, measurement_grid=measurement_grid, doses=list(doses),
                              angles=angles, box_hMpc=box_hMpc, band_hMpc_inv=list(band),
                              horizon_boxes=horizon_boxes, doubled_angles=doubled_angles,
                              variants=list(variants)), cases=[], contrasts=[])
    library()
    arrays = {}
    for dose in doses:
        for count in ((angles, 2*angles) if doubled_angles else (angles,)):
            for name, (sources, density, ngas) in variants.items():
                # Refinement and source-position conclusions need an angular
                # control. Other smoothing arms are first-order diagnostics.
                if count != angles and name not in ('A_fine_sources_gas2', 'D_both_coarse_inputs_gas2',
                                                    'E_coarse_front_gas1', 'I_fine_sources_gas4'):
                    continue
                cap = lift(density, ngas)
                photons = sources.ravel()*(dose*total(cap)/total(sources))
                point = positions(len(sources))*ngas
                started = time.perf_counter()
                solution = solve(point, photons, cap, angles=count, maximum_distance=horizon_boxes*ngas)
                elapsed = time.perf_counter()-started
                independent = audit_counts(solution, photons, cap)
                projected, stat = summarize(cap, solution['absorbed'], measurement_grid, box_hMpc, band)
                retained = total(solution['source_retained'])/total(photons)
                key = f'q{dose:.3f}_K{count}_{name}'
                arrays[key] = projected
                row = dict(key=key, variant=name, dose=dose, angles=count,
                           source_grid=len(sources), gas_grid=ngas, density_grid=len(density),
                           seconds=elapsed, retained_fraction=retained,
                           mass_budget_closed_for_comparison=retained < 1e-7,
                           spectra=stat, independent_audit=independent, solver_audit=solution['audit'])
                report['cases'].append(row)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
                print(json.dumps({k: row[k] for k in ('key', 'seconds', 'retained_fraction', 'spectra')}), flush=True)
    lookup = {(r['dose'], r['angles'], r['variant']): r for r in report['cases']}
    pairs = [('A_fine_sources_gas2', 'B_density_smooth_gas2', 'density_at_original_sources'),
             ('A_fine_sources_gas2', 'C_aggregate_sources_gas2', 'source_aggregation_at_original_density'),
             ('B_density_smooth_gas2', 'D_both_coarse_inputs_gas2', 'source_aggregation_at_smoothed_density'),
             ('C_aggregate_sources_gas2', 'D_both_coarse_inputs_gas2', 'density_at_aggregated_sources'),
             ('D_both_coarse_inputs_gas2', 'E_coarse_front_gas1', 'front_resolution_at_fixed_coarse_sources'),
             ('A_fine_sources_gas2', 'F_source_amplitudes_smooth_gas2', 'source_amplitudes_at_original_positions'),
             ('F_source_amplitudes_smooth_gas2', 'C_aggregate_sources_gas2', 'source_relocation_after_amplitude_smoothing'),
             ('B_density_smooth_gas2', 'G_both_smooth_gas2', 'source_amplitudes_at_smoothed_density'),
             ('G_both_smooth_gas2', 'D_both_coarse_inputs_gas2', 'source_relocation_at_smoothed_density'),
             ('H_fine_sources_gas1', 'A_fine_sources_gas2', 'front1_to2_at_fixed_fine_sources'),
             ('A_fine_sources_gas2', 'I_fine_sources_gas4', 'front2_to4_at_fixed_fine_sources'),
             ('A_fine_sources_gas2', 'E_coarse_front_gas1', 'total_coarsening')]
    for dose in doses:
        for count in (angles, 2*angles):
            for before, after, label in pairs:
                if (dose, count, before) not in lookup or (dose, count, after) not in lookup:
                    continue
                a, b = lookup[dose, count, before], lookup[dose, count, after]
                report['contrasts'].append(dict(dose=dose, angles=count, effect=label,
                    before=before, after=after,
                    b2_ratio=b['spectra']['b_HI_squared']/a['spectra']['b_HI_squared'],
                    signed_cross_bias_change=b['spectra']['signed_cross_bias']-a['spectra']['signed_cross_bias'],
                    Q_mass_change=b['spectra']['Q_mass']-a['spectra']['Q_mass'],
                    common_mesh_L1_over_atoms=total(abs(arrays[a['key']]-arrays[b['key']]))/total(c),
                    both_mass_budgets_closed=a['mass_budget_closed_for_comparison'] and b['mass_budget_closed_for_comparison']))
    report['interpretation'] = ('All before/after ratios share one output mesh and total emitted budget. '
        'Density and source aggregation effects may interact; multiplicative chains depend on order. '
        'Gas refinement retains source coordinates and the same piecewise density, so it does not '
        'create new physical density information. A source aggregation comparison changes spatial '
        'positions, not the summed isotropic emission at each coincident point. This diagnoses one '
        'small realization, not the production500 results.')
    if fixture_hash != digest(fixture) or hashes != {str(p.relative_to(RESEARCH)): digest(p) for p in deps}:
        raise RuntimeError('Fixture or reference code changed during experiment')
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base-grid', type=int, default=16)
    p.add_argument('--measurement-grid', type=int, default=8)
    p.add_argument('--doses', nargs='+', type=float, default=[.8, .9])
    p.add_argument('--angles', type=int, default=96)
    p.add_argument('--box-hMpc', type=float, default=135.6)
    p.add_argument('--band', nargs=2, type=float, default=[.04, .08])
    p.add_argument('--horizon-boxes', type=float, default=4.)
    p.add_argument('--no-doubled-angles', action='store_true')
    p.add_argument('--variants', nargs='+')
    a = vars(p.parse_args())
    a['doubled_angles'] = not a.pop('no_doubled_angles')
    a['selected'] = a.pop('variants')
    run(**a)


if __name__ == '__main__':
    main()
