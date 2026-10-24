#!/usr/bin/env python3
"""Refine absorbing fronts at fixed original source positions and angular banks.

Reuse frozen native 21cmFAST fields from a completed pc_large_scale run. Each
case is a separate static solve, not a time step or coupled Qin20 evolution.
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
RESEARCH = HERE.parent
for folder in ("pc_large_scale", "pc_adaptive_front", "pc_reconsidered"):
    sys.path.insert(0, str(RESEARCH/folder))
sys.path.insert(0, str(HERE))
from run_comparison import (checked_receipt, code_hashes as upstream_hashes, digest,
                            finish, independent_ledger, write_json)
from pc_statistics import coarsen_counts, spectrum, total


def code_hashes():
    hashes = upstream_hashes()
    for pattern in ("*.py", "*.cpp", "*.sh", "*.pbs"):
        hashes.update({str(p.relative_to(RESEARCH)): digest(p) for p in HERE.glob(pattern)})
    return dict(sorted(hashes.items()))


def selection(config, grids=None, doses=None, refinements=None, distance_factor=8., contiguous_ray_gb=4.):
    grids = [g for g in (25, 50) if g in config['grids']] if grids is None else list(grids)
    doses = [.8, .9] if doses is None else list(doses)
    refinements = [2, 4] if refinements is None else list(refinements)
    if (not grids or len(set(grids)) != len(grids)
            or any(type(g) is not int or g not in config['grids'] for g in grids)):
        raise ValueError('Select unique parent grids; defaults are 25 and 50')
    if (not doses or len(set(doses)) != len(doses)
            or any(not np.isfinite(q) or not 0 < q < 1 for q in doses)):
        raise ValueError('Select unique photon doses in (0,1)')
    if (not refinements or len(set(refinements)) != len(refinements)
            or 2 not in refinements or any(type(f) is not int or f not in (2, 4, 8) for f in refinements)):
        raise ValueError('Refinements must include 2 and be selected from 2,4,8')
    if not np.isfinite(distance_factor) or distance_factor <= 0:
        raise ValueError('Positive finite distance factor required')
    if (not np.isfinite(contiguous_ray_gb) or contiguous_ray_gb < 0
            or contiguous_ray_gb*1024**3 > np.iinfo(np.int64).max):
        raise ValueError('Contiguous-ray GiB threshold must be finite, nonnegative and representable')
    measure = min(grids)
    if any(g % measure for g in grids):
        raise ValueError('The common measurement grid must divide all selected grids')
    if not config['k0_band_hMpc_inv'][1] < np.pi*measure/config['box_hMpc']:
        raise ValueError('Measurement band must be below the common-grid Nyquist')
    return dict(grids=grids, doses=doses, refinements=refinements,
                distance_factor=float(distance_factor), measurement_grid=measure,
                contiguous_ray_gib=float(contiguous_ray_gb),
                contiguous_ray_bytes=int(contiguous_ray_gb*1024**3),
                contiguous_ray_scope='Reservation/storage-selection threshold, not an RSS or total memory limit; larger potential banks use block storage')


def angle_key(seed, grid, dose):
    return f'seed{seed}_n{grid}_q{dose:.6f}'


def task_list(config, selected):
    rows = []
    models = [('adaptive', 2)]+[('refined', f) for f in selected['refinements']]
    for seed in config['seeds']:
        for grid in selected['grids']:
            for dose in selected['doses']:
                for model, factor in models:
                    key = angle_key(seed, grid, dose)
                    rows.append(dict(id=len(rows), seed=seed, grid=grid, dose=dose,
                                     model=model, subcells_per_axis=factor, angle_key=key,
                                     directory=f'{key}_{model}_f{factor}'))
    if len({r['directory'] for r in rows}) != len(rows):
        raise ValueError('Dose names collide at six decimal places')
    return rows


def fixture_path(parent, seed, grid):
    return Path(parent)/'results/fixtures'/f'seed{seed}'/f'n{grid}'


def source_options(config, grid, selected):
    reference = grid*config['horizon_box_fraction']
    return dict(min_angles=config['min_angles'], max_angles=config['max_angles'],
                rays_per_surface_cell=config['rays_per_surface_cell'], periodic=True,
                maximum_distance=reference*selected['distance_factor'],
                angular_reference_distance=reference)


def input_fields(parent, seed, grid, dose):
    f = fixture_path(parent, seed, grid)
    source, cap = (np.load(f/name, mmap_mode='r') for name in ('source.npy', 'capacity.npy'))
    if (source.shape != (grid,)*3 or cap.shape != source.shape
            or not np.isfinite(source).all() or not np.isfinite(cap).all()
            or (source < 0).any() or (cap <= 0).any() or total(source) <= 0):
        raise ValueError('Invalid parent source/capacity grid')
    return source*(dose*total(cap)/total(source)), cap


def predict_angles(photons, capacity, options):
    import hybrid
    levels, counts, _, _ = hybrid.estimate_angles(
        photons, capacity, capacity, min_angles=options['min_angles'], max_angles=options['max_angles'],
        rays_per_surface_cell=options['rays_per_surface_cell'], periodic=True,
        maximum_distance=options['angular_reference_distance'])
    return np.ascontiguousarray(counts[levels], dtype=np.int32)


def prepare(root, parent, grids=None, doses=None, refinements=None, distance_factor=8., contiguous_ray_gb=4.):
    root, parent = Path(root).resolve(), Path(parent).resolve()
    if root == parent or parent in root.parents:
        raise ValueError('Follow-up output must be outside the parent run')
    if root.exists() and any(root.iterdir()):
        raise ValueError('Preparation requires a fresh empty output directory')
    if not (parent/'COMPLETED').is_file():
        raise RuntimeError('Parent run has not completed')
    manifest = json.loads((parent/'run_manifest.json').read_text())
    if manifest.get('experiment') != 'large-scale':
        raise RuntimeError('Parent must be a large-scale comparison run')
    config = json.loads((parent/'results/experiment.json').read_text())['config']
    chosen = selection(config, grids, doses, refinements, distance_factor, contiguous_ray_gb)
    plan = task_list(config, chosen)
    inputs = [parent/'run_manifest.json', parent/'results/experiment.json']
    for seed in config['seeds']:
        if not checked_receipt(parent/'results/fixtures'/f'seed{seed}'):
            raise RuntimeError(f'Incomplete parent fixture for seed {seed}')
        for grid in chosen['grids']:
            inputs.extend(fixture_path(parent, seed, grid)/name for name in ('source.npy', 'capacity.npy'))
    input_sha = {str(p): digest(p) for p in inputs}
    root.mkdir(parents=True, exist_ok=True)
    (root/'angles').mkdir()
    angular_files, estimates = {}, []
    for seed in config['seeds']:
        for grid in chosen['grids']:
            for dose in chosen['doses']:
                photons, cap = input_fields(parent, seed, grid, dose)
                opts = source_options(config, grid, chosen)
                prediction_start = time.perf_counter()
                bank = predict_angles(photons, cap, opts)
                prediction_seconds = time.perf_counter()-prediction_start
                name = f"angles/{angle_key(seed, grid, dose)}.npy"
                np.save(root/name, bank)
                angular_files[name] = digest(root/name)
                potential_rays = int(bank[photons > 0].sum(dtype=np.int64))
                estimates.append(dict(seed=seed, grid=grid, dose=dose,
                    potential_rays=potential_rays,
                    potential_packed_ray_bytes=potential_rays*48,
                    estimated_contiguous_storage_selected=(chosen['contiguous_ray_bytes'] > 0 and
                                                            potential_rays*48 <= chosen['contiguous_ray_bytes']),
                    ray_byte_estimate_scope='48-byte packed records; capacity estimate is not committed RSS',
                    active_sources=int(np.count_nonzero(photons)),
                    preparation_angle_estimate_seconds=prediction_seconds))
    identity = dict(schema=1, scope='Static fixed-source front refinement of native 21cmFAST fields',
                    parent_run=str(parent), config=config, selection=chosen, tasks=plan,
                    input_sha256=input_sha, prepared_angle_sha256=angular_files,
                    code_sha256=code_hashes(), angular_preflight=estimates,
                    angular_policy='One frozen source-wise bank shared by all spatial refinements',
                    parallelism='Independent static cases; the event kernel is serial')
    write_json(root/'experiment.json', identity)
    verify(root)
    return identity


def verify(root):
    root = Path(root)
    identity = json.loads((root/'experiment.json').read_text())
    if identity['code_sha256'] != code_hashes():
        raise RuntimeError('Follow-up source code changed after preparation')
    for name, expected in identity['input_sha256'].items():
        if digest(name) != expected:
            raise RuntimeError(f'Parent input changed: {name}')
    for name, expected in identity['prepared_angle_sha256'].items():
        if digest(root/name) != expected:
            raise RuntimeError(f'Prepared angular bank changed: {name}')
    return identity


def verify_case_inputs(root, identity, spec):
    if identity['code_sha256'] != code_hashes():
        raise RuntimeError('Follow-up source code changed after preparation')
    for name in ('source.npy', 'capacity.npy'):
        p = fixture_path(identity['parent_run'], spec['seed'], spec['grid'])/name
        if digest(p) != identity['input_sha256'][str(p)]:
            raise RuntimeError(f'Parent input changed: {p}')
    name = f"angles/{spec['angle_key']}.npy"
    if digest(Path(root)/name) != identity['prepared_angle_sha256'][name]:
        raise RuntimeError('Prepared angular bank changed')


def run_task(root, task_id):
    import adaptive
    import refined
    root = Path(root)
    identity = json.loads((root/'experiment.json').read_text())
    if not 0 <= task_id < len(identity['tasks']):
        raise ValueError('Task ID is outside the prepared plan')
    spec, config = identity['tasks'][task_id], identity['config']
    verify_case_inputs(root, identity, spec)
    target = root/'cases'/spec['directory']
    if checked_receipt(target):
        return
    target.mkdir(parents=True, exist_ok=True)
    photons, cap = input_fields(identity['parent_run'], spec['seed'], spec['grid'], spec['dose'])
    angles = np.load(root/'angles'/f"{spec['angle_key']}.npy")
    opts = source_options(config, spec['grid'], identity['selection'])
    module = adaptive if spec['model'] == 'adaptive' else refined
    module.library()
    prediction_seconds = 0.
    if spec['model'] == 'refined':
        # Adaptive estimates this bank inside solve(). Charge the same predictor
        # to refined cases on the same worker, while still consuming the frozen
        # prepared bank for exact geometry-only scientific comparisons.
        prediction_start = time.perf_counter()
        predicted = predict_angles(photons, cap, opts)
        prediction_seconds = time.perf_counter()-prediction_start
        if not np.array_equal(predicted[photons > 0], angles[photons > 0]):
            raise RuntimeError('Remeasured predictor differs from the frozen active-source angular bank')
    started = time.perf_counter()
    if spec['model'] == 'adaptive':
        result = adaptive.solve(photons, cap, **opts)
    else:
        result = refined.solve(photons, cap, subcells_per_axis=spec['subcells_per_axis'],
                               angle_counts=angles, maximum_distance=opts['maximum_distance'], periodic=True,
                               contiguous_ray_bytes=identity['selection']['contiguous_ray_bytes'])
    elapsed = time.perf_counter()-started
    if not np.array_equal(result['angle_counts'][photons > 0], angles[photons > 0]):
        raise RuntimeError('Solver did not use the frozen active-source angular bank')
    audit_elapsed = result['audit'].get('seconds', elapsed)
    preparation_seconds = result['audit'].get('preparation_seconds', 0.)
    timing = dict(raw_solve_seconds=elapsed,
                  separately_measured_angle_estimate_seconds=prediction_seconds if spec['model'] == 'refined' else None,
                  end_to_end_seconds_with_angle_estimate=elapsed+prediction_seconds,
                  preparation_seconds_with_angle_estimate=preparation_seconds+prediction_seconds,
                  transport_and_output_seconds=max(0., audit_elapsed-preparation_seconds),
                  angle_prediction='Inside adaptive.solve' if spec['model'] == 'adaptive' else 'Remeasured on this worker before refined.solve',
                  scope='Compilation excluded. End-to-end includes one angular prediction, solver preparation, transport, template construction and output-state reconstruction. Transport-and-output is not pure kernel time.')
    ledger = independent_ledger(result, photons, cap)
    measurement_grid = identity['selection']['measurement_grid']
    cap_measure = coarsen_counts(cap, measurement_grid)
    absorbed_measure = coarsen_counts(result['absorbed'], measurement_grid)
    edges = np.geomspace(2*np.pi/config['box_hMpc']*(1-1e-10),
                         np.pi*measurement_grid/config['box_hMpc']*(1+1e-10), config['spectral_bins']+1)
    stat = spectrum(cap_measure, absorbed_measure, config['box_hMpc'], edges,
                    config['k0_band_hMpc_inv'], config['redshift'])
    report = dict(task=spec, options=opts, seconds=elapsed, timing=timing, ledger=ledger, spectra=stat,
                  audit=result['audit'], measurement_grid=measurement_grid,
                  contiguous_ray_bytes=(identity['selection']['contiguous_ray_bytes'] if spec['model'] == 'refined' else None),
                  ray_storage_scope=identity['selection']['contiguous_ray_scope'],
                  parent_grid_Q_volume=float(np.mean(result['absorbed']/cap)),
                  spectral_Q_volume_scope='Ratio of conserved counts on the common measurement mesh',
                  angular_bank_sha256=identity['prepared_angle_sha256'][f"angles/{spec['angle_key']}.npy"],
                  fine_cells_per_parent=spec['subcells_per_axis']**3)
    with (target/'fields.npz.partial').open('wb') as stream:
        np.savez_compressed(stream, absorbed=result['absorbed'], measurement_absorbed=absorbed_measure,
                            source_absorbed=result['source_absorbed'], source_retained=result['source_retained'],
                            source_escaped=result['source_escaped'])
    (target/'fields.npz.partial').replace(target/'fields.npz')
    write_json(target/'result.json', report)
    verify_case_inputs(root, identity, spec)
    finish(target, ['result.json', 'fields.npz'])


def relative(new, old):
    return new/old-1 if old is not None and old != 0 and new is not None else None


def compare_reports(root, new, old, emitted):
    with np.load(Path(root)/'cases'/new['task']['directory']/'fields.npz') as a, \
            np.load(Path(root)/'cases'/old['task']['directory']/'fields.npz') as b:
        delta = a['measurement_absorbed']-b['measurement_absorbed']
        change = dict(field_L1_over_emitted=total(abs(delta))/emitted,
                      maximum_measurement_count_difference=float(np.max(abs(delta))))
    change.update(new=new['task'], reference=old['task'],
                  b2_relative_change=relative(new['spectra']['k0_band']['b_HI_squared'],
                                              old['spectra']['k0_band']['b_HI_squared']),
                  Q_mass_change=new['spectra']['Q_mass']-old['spectra']['Q_mass'],
                  runtime_ratio=(new['timing']['end_to_end_seconds_with_angle_estimate']/
                                 old['timing']['end_to_end_seconds_with_angle_estimate']
                                 if old['timing']['end_to_end_seconds_with_angle_estimate'] else None),
                  runtime_ratio_scope='End-to-end with one angular predictor per solve; compilation excluded',
                  raw_solve_runtime_ratio=new['seconds']/old['seconds'] if old['seconds'] else None,
                  transport_and_output_runtime_ratio=(new['timing']['transport_and_output_seconds']/
                                                      old['timing']['transport_and_output_seconds']
                                                      if old['timing']['transport_and_output_seconds'] else None),
                  maximum_retained_fraction=max(new['ledger']['retained_fraction'], old['ledger']['retained_fraction']))
    return change


def paired_summary(comparisons):
    groups = {}
    for row in comparisons:
        new, old = row['new'], row['reference']
        key = (new['grid'], old['grid'], new['dose'], new['model'],
               new['subcells_per_axis'], old['subcells_per_axis'])
        groups.setdefault(key, []).append(row)
    output = []
    for key, rows in sorted(groups.items()):
        values = [r['b2_relative_change'] for r in rows if r['b2_relative_change'] is not None]
        output.append(dict(new_grid=key[0], reference_grid=key[1], dose=key[2], model=key[3],
                           new_subcells_per_axis=key[4], reference_subcells_per_axis=key[5],
                           seeds=[r['new']['seed'] for r in rows],
                           paired_b2_change_mean=float(np.mean(values)) if values else None,
                           paired_b2_change_seed_std=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                           maximum_absolute_Q_mass_change=max(abs(r['Q_mass_change']) for r in rows),
                           maximum_retained_fraction=max(r['maximum_retained_fraction'] for r in rows)))
    return output


def collect(root):
    root = Path(root)
    identity = verify(root)
    rows = []
    for spec in identity['tasks']:
        folder = root/'cases'/spec['directory']
        if not checked_receipt(folder):
            raise RuntimeError(f'Missing completed case: {folder}')
        report = json.loads((folder/'result.json').read_text())
        if report['task'] != spec:
            raise RuntimeError('Completed case does not match the prepared task')
        rows.append(report)
    key = lambda r: (r['task']['seed'], r['task']['grid'], r['task']['dose'],
                     r['task']['model'], r['task']['subcells_per_axis'])
    lookup = {key(r): r for r in rows}
    spatial, implementation, cross_grid = [], [], []
    for row in rows:
        seed, grid, dose, model, factor = key(row)
        cap = np.load(fixture_path(identity['parent_run'], seed, grid)/'capacity.npy', mmap_mode='r')
        emitted = dose*total(cap)
        if model == 'refined':
            if factor == 2:
                prior = lookup[(seed, grid, dose, 'adaptive', 2)]
            else:
                previous = max(f for f in identity['selection']['refinements'] if f < factor)
                prior = lookup[(seed, grid, dose, 'refined', previous)]
            if row['angular_bank_sha256'] != prior['angular_bank_sha256']:
                raise RuntimeError('Angular bank changed in a front-only comparison')
            difference = compare_reports(root, row, prior, emitted)
            difference['same_source_positions_density_and_angles'] = True
            if factor == 2:
                difference['passed'] = difference['field_L1_over_emitted'] <= 1e-9
                implementation.append(difference)
            else:
                spatial.append(difference)
        fine = max(identity['selection']['grids'])
        if grid != fine:
            reference = lookup[(seed, fine, dose, model, factor)]
            comparison = compare_reports(root, reference, row, emitted)
            comparison['scope'] = 'Conservatively coarsened input fields; common output mesh; source locations differ'
            cross_grid.append(comparison)
    out = dict(scope=identity['scope'], cases=len(rows), all_ledgers_passed=all(r['ledger']['passed'] for r in rows),
               all_f2_implementation_checks_passed=all(r['passed'] for r in implementation),
               implementation_checks=implementation, front_refinement=spatial,
               input_grid_comparisons=cross_grid, case_results=rows,
               front_refinement_paired_summary=paired_summary(spatial),
               input_grid_paired_summary=paired_summary(cross_grid),
               measurement_grid=identity['selection']['measurement_grid'],
               interpretation='Fixed emitted photons, not retuned Q. Inspect retained photons and Q alongside power. '
                              'Refinement comparisons share exact source positions and angular banks; angular convergence is separate.')
    write_json(root/'summary.json', out)
    if not out['all_f2_implementation_checks_passed']:
        raise RuntimeError('Refined f2 does not reproduce the adaptive model; inspect summary.json')
    return out


def run(root, workers=32):
    root = Path(root).resolve()
    identity = verify(root)
    if type(workers) is not int or not 1 <= workers <= 32:
        raise ValueError('Use 1 to 32 independent workers')
    (root/'logs').mkdir(exist_ok=True)
    started = time.perf_counter()
    def launch(task):
        env = dict(os.environ, OMP_NUM_THREADS='1', OMP_PROC_BIND='false',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        with (root/'logs'/f"case_{task['id']:03d}.log").open('a') as stream:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), 'task', '--root', str(root),
                            '--task-id', str(task['id'])], env=env, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(launch, task): task for task in identity['tasks']}
        for future in as_completed(pending):
            spec = pending[future]
            try:
                future.result()
                print('Completed:', spec['directory'], flush=True)
            except Exception as error:
                failures.append(dict(task=spec, error=str(error)))
    write_json(root/'execution.json', dict(seconds_before_collection=time.perf_counter()-started,
                                         requested_workers=workers, simultaneous_tasks=min(workers, len(identity['tasks'])),
                                         failures=failures))
    if failures:
        raise RuntimeError(f'{len(failures)} cases failed; see logs')
    collect(root)
    verify(root)
    (root/'COMPLETED').write_text('Static front comparisons completed; inspect convergence measurements\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'run', 'task', 'collect'):
        p = sub.add_parser(name)
        p.add_argument('--root', required=True)
        if name == 'prepare':
            p.add_argument('--parent', required=True)
            p.add_argument('--grids', type=int, nargs='+')
            p.add_argument('--doses', type=float, nargs='+')
            p.add_argument('--refinements', type=int, nargs='+')
            p.add_argument('--distance-factor', type=float, default=8.)
            p.add_argument('--contiguous-ray-gb', type=float, default=4.,
                           help='GiB threshold for contiguous ray reservation; not an RSS cap (0 selects blocks)')
        if name == 'run':
            p.add_argument('--workers', type=int, default=32)
        if name == 'task':
            p.add_argument('--task-id', type=int, required=True)
    args = vars(parser.parse_args())
    command = args.pop('command')
    result = {'prepare': prepare, 'run': run, 'task': run_task, 'collect': collect}[command](**args)
    if command == 'prepare':
        print(json.dumps({'tasks': len(result['tasks']), 'selection': result['selection'],
                          'maximum_potential_rays': max(r['potential_rays'] for r in result['angular_preflight'])}, indent=2))


if __name__ == '__main__':
    main()
