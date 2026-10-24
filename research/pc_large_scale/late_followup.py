#!/usr/bin/env python3
"""Fixed-budget horizon/angle checks on existing immutable native fields.

No native field generation or source recalibration. Each horizon is a fresh
solve from neutral gas, with the angular predictor held at the original horizon.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_comparison import checked_receipt, code_hashes, digest, finish, independent_ledger, write_json
from pc_statistics import spectrum, total
sys.path.insert(0, str(HERE.parent/'pc_adaptive_front'))


def tasks(config):
    rows = []
    for seed in config['seeds']:
        for grid in config['grids']:
            for dose in (.8, .9):
                if dose not in config['photon_doses']:
                    raise ValueError('Parent experiment must contain doses 0.8 and 0.9')
                for distance, angle in ((1, 1), (2, 1), (4, 1), (8, 1), (8, 2)):
                    rows.append(dict(id=len(rows), seed=seed, grid=grid, dose=dose,
                                     distance_factor=distance, angle_factor=angle,
                                     directory=f'seed{seed}_n{grid}_q{dose:.6f}_d{distance}_a{angle}'))
    return rows


def parent_case(parent, spec):
    return parent/'results/cases'/f"seed{spec['seed']}_n{spec['grid']}_q{spec['dose']:.6f}_baseline"


def parent_inputs(parent, config):
    """Verify receipts before recording the exact subset consumed by this study."""
    parent = Path(parent).resolve()
    files = [parent/'run_manifest.json', parent/'results/experiment.json']
    for seed in config['seeds']:
        fixture = parent/'results/fixtures'/f'seed{seed}'
        if not checked_receipt(fixture):
            raise RuntimeError(f'Incomplete parent fixture: {fixture}')
        for grid in config['grids']:
            files += [fixture/f'n{grid}'/name for name in ('source.npy', 'capacity.npy')]
    unique = {(t['seed'], t['grid'], t['dose']): t for t in tasks(config)}
    for spec in unique.values():
        case = parent_case(parent, spec)
        if not checked_receipt(case):
            raise RuntimeError(f'Incomplete parent case: {case}')
        files += [case/'fields.npz', case/'result.json']
    return {str(f): digest(f) for f in files}


def prepare(root, parent):
    root, parent = Path(root).resolve(), Path(parent).resolve()
    if root == parent or parent in root.parents:
        raise ValueError('Follow-up output must be outside the parent run')
    if not (parent/'COMPLETED').is_file():
        raise RuntimeError('Parent run has not completed')
    manifest = json.loads((parent/'run_manifest.json').read_text())
    if manifest.get('experiment') != 'large-scale':
        raise RuntimeError('Parent must be a large-scale comparison run')
    config = json.loads((parent/'results/experiment.json').read_text())['config']
    plan = tasks(config)
    identity = dict(schema=1, scope='Static fixed emitted budget, frozen angular predictor; no mass recalibration',
                    parent_run=str(parent), config=config, tasks=plan,
                    input_sha256=parent_inputs(parent, config), code_sha256=code_hashes())
    root.mkdir(parents=True, exist_ok=True)
    path = root/'experiment.json'
    if path.exists() and json.loads(path.read_text()) != identity:
        raise RuntimeError('Prepared experiment changed; use a new output directory')
    write_json(path, identity)
    return identity


def verify(root):
    identity = json.loads((Path(root)/'experiment.json').read_text())
    if identity['code_sha256'] != code_hashes():
        raise RuntimeError('Follow-up source code changed after preparation')
    for name, expected in identity['input_sha256'].items():
        if digest(name) != expected:
            raise RuntimeError(f'Parent input changed: {name}')
    return identity


def differences(a, b, emitted):
    return {'L1_over_emitted': total(abs(a-b))/emitted,
            'maximum_count_difference': float(np.max(abs(a-b)))}


def run_task(root, task_id):
    import adaptive
    root = Path(root)
    identity = json.loads((root/'experiment.json').read_text())
    if identity['code_sha256'] != code_hashes():
        raise RuntimeError('Follow-up source code changed after preparation')
    c, spec = identity['config'], identity['tasks'][task_id]
    target = root/'cases'/spec['directory']
    parent = Path(identity['parent_run'])
    fixture = parent/'results/fixtures'/f"seed{spec['seed']}"/f"n{spec['grid']}"
    consumed = [fixture/'source.npy', fixture/'capacity.npy',
                parent_case(parent, spec)/'fields.npz', parent_case(parent, spec)/'result.json']
    for path in consumed:
        if digest(path) != identity['input_sha256'][str(path)]:
            raise RuntimeError(f'Parent input changed: {path}')
    if checked_receipt(target):
        return
    target.mkdir(parents=True, exist_ok=True)
    s, cap = (np.load(fixture/name, mmap_mode='r') for name in ('source.npy', 'capacity.npy'))
    emitted = spec['dose']*total(cap)
    photons = s*(emitted/total(s))
    reference = spec['grid']*c['horizon_box_fraction']
    angle = spec['angle_factor']
    options = dict(min_angles=angle*c['min_angles'], max_angles=angle*c['max_angles'],
                   rays_per_surface_cell=angle*c['rays_per_surface_cell'], periodic=True,
                   maximum_distance=spec['distance_factor']*reference,
                   angular_reference_distance=reference)
    adaptive.library()
    started = time.perf_counter()
    result = adaptive.solve(photons, cap, **options)
    elapsed = time.perf_counter()-started
    ledger = independent_ledger(result, photons, cap)
    edges = np.geomspace(2*np.pi/c['box_hMpc']*(1-1e-10),
                         np.pi*c['native_grid']/c['box_hMpc']*(1+1e-10), c['spectral_bins']+1)
    stat = spectrum(cap, result['absorbed'], c['box_hMpc'], edges,
                    c['k0_band_hMpc_inv'], c['redshift'])
    old_dir = parent_case(parent, spec)
    with np.load(old_dir/'fields.npz') as old:
        change = differences(result['absorbed'], old['pc_same_budget_absorbed'], emitted)
    old = json.loads((old_dir/'result.json').read_text())['spectra']['pc_same_budget']
    change.update(Q_mass_change=stat['Q_mass']-old['Q_mass'],
                  b2_relative_change=stat['k0_band']['b_HI_squared']/old['k0_band']['b_HI_squared']-1
                  if old['k0_band']['b_HI_squared'] else None)
    change['baseline_reproduced'] = (change['L1_over_emitted'] <= 1e-10) if spec['distance_factor'] == 1 and angle == 1 else None
    report = dict(task=spec, options=options, seconds=elapsed, ledger=ledger,
                  spectra=stat, versus_original_same_budget=change, audit=result['audit'])
    with (target/'fields.npz.partial').open('wb') as stream:
        np.savez_compressed(stream, absorbed=result['absorbed'], source_absorbed=result['source_absorbed'],
                            source_retained=result['source_retained'], source_escaped=result['source_escaped'],
                            angle_counts=result['angle_counts'])
    (target/'fields.npz.partial').replace(target/'fields.npz')
    write_json(target/'result.json', report)
    for path in consumed:
        if digest(path) != identity['input_sha256'][str(path)]:
            raise RuntimeError(f'Parent input changed during solve: {path}')
    if identity['code_sha256'] != code_hashes():
        raise RuntimeError('Follow-up source code changed during solve')
    finish(target, ['result.json', 'fields.npz'])


def collect(root):
    root = Path(root)
    identity = verify(root)
    reports = []
    for task in identity['tasks']:
        folder = root/'cases'/task['directory']
        if not checked_receipt(folder):
            raise RuntimeError(f'Missing case: {folder}')
        reports.append(json.loads((folder/'result.json').read_text()))
    lookup = {(r['task']['seed'], r['task']['grid'], r['task']['dose'],
               r['task']['distance_factor'], r['task']['angle_factor']): r for r in reports}
    checks = []
    baseline_failures = [r['task'] for r in reports if r['versus_original_same_budget'].get('baseline_reproduced') is False]
    for key, new in lookup.items():
        seed, grid, dose, distance, angle = key
        if distance == 1:
            continue
        prior_key = (seed, grid, dose, distance//2 if angle == 1 else distance, 1)
        old = lookup[prior_key]
        with np.load(root/'cases'/old['task']['directory']/'fields.npz') as a, np.load(root/'cases'/new['task']['directory']/'fields.npz') as b:
            stable_angles = bool(np.array_equal(a['angle_counts'], b['angle_counts']))
            if angle == 1 and not stable_angles:
                raise RuntimeError('Angular allocation changed in a distance-only comparison')
            cap = np.load(Path(identity['parent_run'])/'results/fixtures'/f'seed{seed}'/f'n{grid}'/'capacity.npy', mmap_mode='r')
            change = differences(a['absorbed'], b['absorbed'], dose*total(cap))
        b0 = old['spectra']['k0_band']['b_HI_squared']
        change.update(task=new['task'], comparison='distance' if angle == 1 else 'angle',
                      from_distance_factor=prior_key[3], same_angle_counts=stable_angles,
                      Q_mass_change=new['spectra']['Q_mass']-old['spectra']['Q_mass'],
                      b2_relative_change=new['spectra']['k0_band']['b_HI_squared']/b0-1 if b0 else None,
                      retained_fraction=new['ledger']['retained_fraction'])
        change['retained_over_remaining_neutral'] = (new['ledger']['retained_fraction']*dose/(1-new['spectra']['Q_mass'])
            if new['spectra']['Q_mass'] < 1 else None)
        change['within_requested_b2_tolerance'] = (change['b2_relative_change'] is not None and
            abs(change['b2_relative_change']) <= identity['config']['convergence_relative_tolerance'])
        change['within_field_tolerance'] = change['L1_over_emitted'] <= 1e-3
        change['within_retention_tolerance'] = change['retained_fraction'] <= identity['config']['retained_fraction_tolerance']
        change['all_requested_tolerances_passed'] = (change['within_requested_b2_tolerance'] and
            change['within_field_tolerance'] and change['within_retention_tolerance'])
        checks.append(change)
    out = dict(scope=identity['scope'], cases=len(reports),
               all_ledgers_passed=all(r['ledger']['passed'] for r in reports),
               all_baselines_reproduced=not baseline_failures, baseline_reproduction_failures=baseline_failures,
               tolerances=dict(baseline_L1_over_emitted=1e-10, field_L1_over_emitted=1e-3,
                               b2_relative=identity['config']['convergence_relative_tolerance'],
                               retained_fraction=identity['config']['retained_fraction_tolerance']),
               comparisons=checks, case_results=reports,
               interpretation='All comparisons use fixed emitted photons, not exactly matched ionized mass; inspect Q changes with b2. Distance flags do not certify spatial convergence.')
    write_json(root/'summary.json', out)
    return out


def run(root, workers):
    root = Path(root).resolve()
    identity = verify(root)
    if not 1 <= workers <= 32:
        raise ValueError('Use 1 to 32 independent workers')
    (root/'logs').mkdir(exist_ok=True)
    started = time.perf_counter()
    def launch(task):
        env = dict(os.environ, OMP_NUM_THREADS='1', OMP_PROC_BIND='false',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        with (root/'logs'/f"case_{task['id']:03d}.log").open('a') as stream:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), 'task', '--root', str(root),
                            '--task-id', str(task['id'])], env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(launch, task): task for task in identity['tasks']}
        for future in as_completed(pending):
            task = pending[future]
            try:
                future.result()
                print('Completed:', task['directory'], flush=True)
            except Exception as error:
                failures.append(dict(task=task, error=str(error)))
    write_json(root/'execution.json', dict(seconds_before_collection=time.perf_counter()-started,
                                         transport_workers=workers, failures=failures))
    if failures:
        raise RuntimeError(f'{len(failures)} cases failed; see logs')
    collect(root)
    verify(root)
    (root/'COMPLETED').write_text('All cases and collection complete; inspect scientific checks\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'run', 'task', 'collect'):
        p = sub.add_parser(name); p.add_argument('--root', required=True)
        if name == 'prepare': p.add_argument('--parent', required=True)
        if name == 'run': p.add_argument('--workers', type=int, default=32)
        if name == 'task': p.add_argument('--task-id', type=int, required=True)
    a = vars(parser.parse_args()); command = a.pop('command')
    {'prepare': prepare, 'run': run, 'task': run_task, 'collect': collect}[command](**a)


if __name__ == '__main__':
    main()
