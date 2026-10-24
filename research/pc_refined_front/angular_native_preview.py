#!/usr/bin/env python3
"""Small fixed-source angular convergence diagnostic using a native fixture.

This is a static diagnostic on conservatively coarsened counts. It does not
generate native initial conditions or evolve a cosmological history. The
supplied physical box size defines the spectral coordinates; it is not inferred
from the fixture. Changing the angular count regenerates the non-nested,
octant-balanced quadrature, so successive changes need not be monotone.
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
for folder in ('pc_large_scale', 'pc_reconsidered'):
    sys.path.insert(0, str(RESEARCH/folder))
sys.path.insert(0, str(HERE))
import hybrid
import refined
from fine_state_metrics import fine_state_metrics
from pc_statistics import coarsen_counts, spectrum, total
from run_comparison import independent_ledger


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def run(fixture, output, nbase=8, doses=(.8, .9), multipliers=(1, 2, 4, 8, 16, 32, 64),
        horizon_boxes=16., box_hmpc=135.6, band=(.04, .08), redshift=8.,
        subcells_per_axis=4, power_tolerance=.01, field_tolerance=.01):
    fixture, output = Path(fixture), Path(output)
    doses, multipliers, band = list(doses), list(multipliers), list(band)
    if type(nbase) is not int or nbase < 2:
        raise ValueError('nbase must be an integer at least 2')
    if (not doses or len(set(doses)) != len(doses)
            or any(not np.isfinite(q) or not 0 < q < 1 for q in doses)):
        raise ValueError('Select unique finite photon doses in (0,1)')
    if (len(multipliers) < 2 or multipliers[0] != 1
            or any(type(m) is not int or m < 1 for m in multipliers)
            or multipliers != sorted(set(multipliers))):
        raise ValueError('Select increasing positive integer multipliers starting with 1')
    if subcells_per_axis not in (2, 4, 8):
        raise ValueError('Select 2, 4 or 8 subcells per axis')
    if (not np.isfinite(horizon_boxes) or horizon_boxes <= 0
            or not np.isfinite(box_hmpc) or box_hmpc <= 0
            or not np.isfinite(redshift) or redshift <= -1):
        raise ValueError('Positive finite horizon and box, and redshift > -1 required')
    if (len(band) != 2 or not all(np.isfinite(band))
            or not 0 < band[0] < band[1] < np.pi*nbase/box_hmpc):
        raise ValueError('Spectral band must be positive and below grid Nyquist')
    if any(not np.isfinite(t) or t <= 0 for t in (power_tolerance, field_tolerance)):
        raise ValueError('Positive finite diagnostic tolerances required')
    with np.load(fixture, allow_pickle=False) as data:
        original_source = np.asarray(data['source'], dtype=np.float64)
        original_capacity = np.asarray(data['capacity'], dtype=np.float64)
    if (original_source.shape != original_capacity.shape or original_source.ndim != 3
            or len(set(original_source.shape)) != 1 or original_source.shape[0] % nbase
            or not np.isfinite(original_source).all() or not np.isfinite(original_capacity).all()
            or (original_source < 0).any() or (original_capacity <= 0).any()
            or total(original_source) <= 0):
        raise ValueError('Fixture requires matching finite cubic source/capacity; nbase must divide its side')
    source0 = coarsen_counts(original_source, nbase)
    capacity = coarsen_counts(original_capacity, nbase)
    dependencies = [Path(__file__), HERE/'refined.py', HERE/'refined.cpp',
                    HERE/'fine_state_metrics.py',
                    RESEARCH/'pc_refinement/front_subcells.py',
                    RESEARCH/'pc_reconsidered/hybrid.py',
                    RESEARCH/'pc_alternatives/characteristics.py',
                    RESEARCH/'pc_large_scale/pc_statistics.py',
                    RESEARCH/'pc_large_scale/run_comparison.py']
    hashes = {str(path.relative_to(RESEARCH)): digest(path) for path in dependencies}
    edges = np.geomspace(2*np.pi/box_hmpc*(1-1e-10),
                         np.pi*nbase/box_hmpc*(1+1e-10), 17)
    report = dict(
        schema=1,
        scope='Static small-fixture fixed-source angular diagnostic; not the 500 h^-1 cMpc production comparison or cosmological evolution',
        fixture_name=fixture.name, fixture_sha256=digest(fixture),
        input_native_shape=list(original_source.shape), code_sha256=hashes,
        configuration=dict(nbase=nbase, doses=doses, multipliers=multipliers,
            subcells_per_axis=subcells_per_axis, horizon_boxes=horizon_boxes,
            maximum_distance_original_cells=horizon_boxes*nbase,
            box_hMpc=box_hmpc, band_hMpc_inv=band, redshift=redshift,
            power_tolerance=power_tolerance, field_tolerance=field_tolerance,
            angular_predictor=dict(min_angles=24, max_angles=12288,
                rays_per_surface_cell=4., reference_distance_original_cells=nbase/3)),
        input_construction='Sum native photon and capacity counts into nbase^3 parents; multiply source by Q*sum(capacity)/sum(source); keep centres, capacity and initial neutral state fixed across angular multipliers',
        coarsened_capacity_sha256=array_digest(capacity),
        quadrature='Equal weights, sign-balanced octants, cyclic coordinate permutations; K->2K regenerates non-nested directions because mu=(j+0.5)/(K/24)',
        angular_scaling='Multiply each source original predicted count exactly; do not clip to the original predictor maximum',
        timing='Single sequential solve per case; compilation and prediction excluded; clear local-template cache before each timed solve, so template construction is included; not a repeated performance benchmark',
        cases=[], consecutive_comparisons=[], baseline_comparisons=[],
        single_pair_pass_followed_by_failure=[])
    refined.library()  # Compile outside every timed solve.
    for dose in doses:
        photons = source0*(dose*total(capacity)/total(source0))
        levels, counts, _, _ = hybrid.estimate_angles(
            photons, capacity, capacity, min_angles=24, max_angles=12288,
            rays_per_surface_cell=4., periodic=True, maximum_distance=nbase/3)
        bank = np.ascontiguousarray(counts[levels], dtype=np.int32)
        if int(bank.max()) > np.iinfo(np.int32).max//multipliers[-1]:
            raise ValueError('Scaled angular counts exceed int32')
        baseline = previous = None
        last_comparison = None
        for multiplier in multipliers:
            angular = np.ascontiguousarray(bank.astype(np.int64)*multiplier, dtype=np.int32)
            refined.clear_template_cache()
            started = time.perf_counter()
            result = refined.solve(photons, capacity, subcells_per_axis=subcells_per_axis,
                                   angle_counts=angular, maximum_distance=horizon_boxes*nbase,
                                   periodic=True)
            elapsed = time.perf_counter()-started
            stat = spectrum(capacity, result['absorbed'], box_hmpc, edges, band, redshift)
            if stat['k0_band'] is None or stat['k0_band']['b_HI_squared'] is None:
                raise RuntimeError('No defined matter-bias measurement in requested band')
            ledger = independent_ledger(result, photons, capacity)
            active_angles = angular[photons > 0]
            row = dict(dose=dose, multiplier=multiplier,
                source_sha256=array_digest(photons), angular_bank_sha256=array_digest(angular),
                absorbed_sha256=array_digest(result['absorbed']),
                min_angles=int(active_angles.min()), mean_angles=float(active_angles.mean()),
                max_angles=int(active_angles.max()), seconds=elapsed, ledger=ledger,
                b_HI_squared=stat['k0_band']['b_HI_squared'], k0_band=stat['k0_band'],
                Q_mass=stat['Q_mass'], Q_volume=stat['Q_volume'],
                rays_materialized=result['audit']['rays_materialized'],
                template_build_seconds=result['audit']['template_build_seconds'])
            report['cases'].append(row)

            state = {name: result['state'][name].copy()
                     for name in ('cleared_bits', 'partial_ids', 'partial_ionized')}

            def compare(old):
                old_absorbed, old_row, old_state = old
                difference = dict(dose=dose, multiplier=multiplier,
                    reference_multiplier=old_row['multiplier'],
                    b2_relative_change=row['b_HI_squared']/old_row['b_HI_squared']-1,
                    field_L1_over_emitted=total(abs(result['absorbed']-old_absorbed))/total(photons),
                    Q_mass_change=row['Q_mass']-old_row['Q_mass'],
                    maximum_retained_fraction=max(row['ledger']['retained_fraction'],
                                                  old_row['ledger']['retained_fraction']),
                    resolved_subcells=fine_state_metrics(capacity, subcells_per_axis,
                                                         state, old_state, total(photons)))
                difference['power_threshold_passed'] = abs(difference['b2_relative_change']) < power_tolerance
                difference['field_threshold_passed'] = difference['field_L1_over_emitted'] < field_tolerance
                difference['resolved_subcell_threshold_passed'] = (
                    difference['resolved_subcells']['subcell_L1_over_emitted'] < field_tolerance)
                difference['joint_threshold_passed'] = (difference['power_threshold_passed'] and
                                                         difference['field_threshold_passed'] and
                                                         difference['resolved_subcell_threshold_passed'])
                return difference

            if previous is not None:
                comparison = compare(previous)
                report['consecutive_comparisons'].append(comparison)
                report['baseline_comparisons'].append(compare(baseline))
                if last_comparison is not None:
                    for metric in ('power_threshold_passed', 'field_threshold_passed',
                                   'resolved_subcell_threshold_passed', 'joint_threshold_passed'):
                        if last_comparison[metric] and not comparison[metric]:
                            report['single_pair_pass_followed_by_failure'].append(dict(
                                dose=dose, diagnostic=metric,
                                passed_pair=[last_comparison['reference_multiplier'], last_comparison['multiplier']],
                                failed_pair=[comparison['reference_multiplier'], comparison['multiplier']]))
                last_comparison = comparison
            else:
                baseline = (result['absorbed'].copy(), row, state)
            previous = (result['absorbed'], row, state)
            print(json.dumps(dict(dose=dose, multiplier=multiplier,
                                  b_HI_squared=row['b_HI_squared'], ledger_passed=ledger['passed'])), flush=True)
    report['all_ledgers_passed'] = all(row['ledger']['passed'] for row in report['cases'])
    report['maximum_retained_fraction'] = max(row['ledger']['retained_fraction'] for row in report['cases'])
    report['interpretation'] = ('Non-nested quadrature can produce cancellations: a small baseline-to-high-count change or one passing adjacent pair does not establish convergence. Parent averaging can hide resolved-front differences: joint thresholds include subcell L1 computed from sparse states, without allocating a dense fine grid. Inspect several successive pairs and spatial fields; this preview does not certify the continuum solution or production volume.')
    report['code_unchanged_during_run'] = hashes == {str(path.relative_to(RESEARCH)): digest(path) for path in dependencies}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    if not report['all_ledgers_passed'] or not report['code_unchanged_during_run']:
        raise RuntimeError('Independent photon ledger or code provenance check failed')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--nbase', type=int, default=8)
    parser.add_argument('--doses', type=float, nargs='+', default=[.8, .9])
    parser.add_argument('--multipliers', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument('--horizon-boxes', type=float, default=16.)
    parser.add_argument('--box-hmpc', type=float, default=135.6)
    parser.add_argument('--band', type=float, nargs=2, default=[.04, .08])
    parser.add_argument('--redshift', type=float, default=8.)
    parser.add_argument('--subcells-per-axis', type=int, default=4)
    parser.add_argument('--power-tolerance', type=float, default=.01)
    parser.add_argument('--field-tolerance', type=float, default=.01)
    run(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
