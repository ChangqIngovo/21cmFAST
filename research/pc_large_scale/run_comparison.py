#!/usr/bin/env python3
"""Large-volume, paired-realisation validation of adaptive eight-subcell PC.

Native 21cmFAST fields are generated once per seed on the finest grid. All
coarser inputs are conservative sums of these SAME fields. Independent static
coeval/dose jobs run in separate processes; no photons or shared fronts are
split among workers. The physical model is the tested restricted L-INTEGRAL
adapter. This is not coupled Qin20 evolution or a SCRIPT numerical replication.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
sys.path.insert(0, str(RESEARCH/'pc_reconsidered'))
sys.path.insert(0, str(RESEARCH/'pc_adaptive_front'))
from pc_statistics import coarsen_counts, es_map, es_threshold, input_band_statistics, match_mass, spectrum, total


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def checked_receipt(directory, receipt='COMPLETED.json'):
    path = Path(directory)/receipt
    if not path.exists():
        return False
    row = json.loads(path.read_text())
    for name, expected in row['files_sha256'].items():
        if digest(Path(directory)/name) != expected:
            raise RuntimeError(f'Completed output changed: {Path(directory)/name}')
    return True


def finish(directory, names):
    write_json(Path(directory)/'COMPLETED.json',
               {'files_sha256': {name: digest(Path(directory)/name) for name in names}})


def config_at(root):
    return json.loads((Path(root)/'experiment.json').read_text())['config']


def validate_config(c):
    if c['schema'] != 1:
        raise ValueError('Unknown config schema')
    for key in ('native_grid', 'initial_grid', 'min_angles', 'max_angles', 'spectral_bins'):
        if type(c[key]) is not int or c[key] < 2:
            raise ValueError(f'Invalid integer {key}')
    n = c['native_grid']
    if c['initial_grid'] < n or not c['grids'] or max(c['grids']) != n:
        raise ValueError('Initial/source/coarse grid hierarchy is invalid')
    if any(type(g) is not int or g < 2 or n % g for g in c['grids']):
        raise ValueError('All grids must divide native_grid')
    if len(set(c['grids'])) != len(c['grids']) or len(set(c['seeds'])) != len(c['seeds']):
        raise ValueError('Duplicate grids or seeds')
    if not c['seeds'] or any(type(s) is not int or s < 1 for s in c['seeds']):
        raise ValueError('Positive integer seeds required')
    if not c['photon_doses'] or any(not 0 < q < 1 for q in c['photon_doses']):
        raise ValueError('Photon doses must lie in (0,1)')
    if len(set(c['photon_doses'])) != len(c['photon_doses']):
        raise ValueError('Duplicate photon doses')
    if any(q not in c['photon_doses'] for q in c['validation_doses']):
        raise ValueError('Validation doses must be present in main sweep')
    if (c['min_angles'] < 24 or c['min_angles'] % 24 or c['max_angles'] % 24
            or c['max_angles'] < c['min_angles']):
        raise ValueError('Invalid angular limits')
    for key in ('box_hMpc', 'rays_per_surface_cell', 'horizon_box_fraction',
                'mass_match_tolerance', 'retained_fraction_tolerance', 'convergence_relative_tolerance'):
        if not np.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if c['horizon_box_fraction'] > 1/3:
        raise ValueError('Baseline horizon must respect native R_BUBBLE_MAX <= BOX_LEN/3')
    if not 0 < c['k0_band_hMpc_inv'][0] < c['k0_band_hMpc_inv'][1] < np.pi*min(c['grids'])/c['box_hMpc']:
        raise ValueError('The k0 band must be below the coarsest Nyquist frequency')
    k = 2*np.pi*np.fft.fftfreq(min(c['grids']), d=c['box_hMpc']/min(c['grids']))
    km = np.sqrt(k[:,None,None]**2+k[None,:,None]**2+k[None,None,:]**2)
    if not np.any((km >= c['k0_band_hMpc_inv'][0]) & (km < c['k0_band_hMpc_inv'][1])):
        raise ValueError('The chosen k0 band contains no Fourier modes')


def code_hashes():
    files = list(HERE.glob('*.py'))
    for folder in ('pc_alternatives', 'pc_reconsidered', 'pc_refinement', 'pc_adaptive_front'):
        files += list((RESEARCH/folder).glob('*.py'))+list((RESEARCH/folder).glob('*.cpp'))
    return {str(p.relative_to(RESEARCH)): digest(p) for p in sorted(files)}


def fixture(root, seed, threads):
    from py21cmfast import run_coeval
    from py21cmfast.io.caching import OutputCache
    import py21cmfast
    from native_driver import prepare_native, from_native_coeval, validate_inputs

    root = Path(root)
    c = config_at(root)
    directory = root/'fixtures'/f'seed{seed}'
    directory.mkdir(parents=True, exist_ok=True)
    if checked_receipt(directory):
        return
    h = c['cosmology']['hlittle']
    box = c['box_hMpc']/h
    inputs, _ = prepare_native(dimension=c['native_grid'], box_len=box,
                               threads=threads, random_seed=seed)
    # Keep the validated L-INTEGRAL source prescription; only geometry,
    # cosmology and maximum ES filtering distance change for this experiment.
    inputs = inputs.evolve_input_structs(
        **c['cosmology'], A_s=None, DIM=c['initial_grid'],
        R_BUBBLE_MAX=min(box/3, box*c['horizon_box_fraction']))
    validate_inputs(inputs)
    started = time.perf_counter()
    coeval = run_coeval(inputs=inputs, out_redshifts=c['redshift'], regenerate=True,
                        write=False, cache=OutputCache(directory/'native_cache'))[0]
    native = from_native_coeval(coeval)
    if (native.capacity <= 0).any():
        raise RuntimeError('Zero-density cell: explicit vacuum support is not enabled in this experiment')
    names = []
    meta = {'scope': 'Native 21cmFAST restricted L-INTEGRAL source and 2LPT density',
            'py21cmfast': py21cmfast.__version__, 'seed': seed, 'native': native.metadata,
            'generation_seconds': time.perf_counter()-started,
            'inputs': inputs.asdict(include_cosmo_tables='never', only_cstruct_params=False),
            'source_model_difference_from_paper': 'Qin20 ACG mass-dependent efficiency with L-INTEGRAL; paper uses mass-proportional sources above a hard M_min and N-body density',
            'coarse_units': 'sum of original fine-cell counts; no second density weighting', 'grids': {}}
    for n in c['grids']:
        target = directory/f'n{n}'
        target.mkdir(exist_ok=True)
        s, cap = coarsen_counts(native.source, n), coarsen_counts(native.capacity, n)
        if not np.isclose(total(s), total(native.source), rtol=2e-14, atol=0):
            raise RuntimeError('Coarsening changed photon total')
        if not np.isclose(total(cap), total(native.capacity), rtol=2e-14, atol=0):
            raise RuntimeError('Coarsening changed atom total')
        threshold, audit = es_threshold(s, cap, n*c['horizon_box_fraction'], workers=threads)
        for name, array in [('source', s), ('capacity', cap), ('es_threshold', threshold)]:
            np.save(target/f'{name}.npy', array)
            names.append(f'n{n}/{name}.npy')
        meta['grids'][str(n)] = {'source_total': total(s), 'capacity_total': total(cap),
                                  'cell_hMpc': c['box_hMpc']/n, 'es': audit,
                                  'input_k0_band': input_band_statistics(s, cap, c['box_hMpc'], c['k0_band_hMpc_inv'])}
    native_xhi = np.asarray(coeval.ionized_box.get('neutral_fraction'))
    np.save(directory/'native_es_xhi.npy', native_xhi)
    np.save(directory/'native_es_brightness_mK.npy', np.asarray(coeval.brightness_temp))
    threshold = np.load(directory/f'n{c["native_grid"]}/es_threshold.npy')
    matched_es = es_map(native.source, native.capacity, threshold, 1.)
    native_a = native.capacity*(1-native_xhi.astype(float))
    meta['native_vs_source_matched_ES_at_original_dose'] = {
        'native_Q_mass': total(native_a)/total(native.capacity),
        'matched_Q_mass': total(matched_es)/total(native.capacity),
        'L1_over_emitted': total(abs(native_a-matched_es))/total(native.source),
        'maximum_fraction_difference': float(np.max(abs(native_a-matched_es)/native.capacity)),
        'interpretation': 'Native control recorded separately; exact agreement is not assumed'}
    write_json(directory/'metadata.json', meta)
    finish(directory, names+['metadata.json', 'native_es_xhi.npy', 'native_es_brightness_mK.npy'])


def task_list(c):
    rows = []
    for seed in c['seeds']:
        for n in c['grids']:
            for q in c['photon_doses']:
                rows.append(dict(seed=seed, grid=n, dose=q, variant='baseline'))
    for q in c['validation_doses']:
        for variant in ('angle2', 'distance2'):
            rows.append(dict(seed=c['seeds'][0], grid=c['native_grid'], dose=q, variant=variant))
    for i, row in enumerate(rows):
        row['id'] = i
        row['directory'] = f"seed{row['seed']}_n{row['grid']}_q{row['dose']:.6f}_{row['variant']}"
    if len({r['directory'] for r in rows}) != len(rows):
        raise ValueError('Dose directory names are not unique at six decimal places')
    return rows


def match_pc_mass(solver, source, capacity, target, options, tolerance):
    """Keep the uncalibrated dose map and independently solve a fixed-mass map.

Changing source amplitude is only for the fixed-mass comparison. No fields are
interpolated, and finite-distance retention remains in both photon ledgers.
    """
    C, S = total(capacity), total(source)
    amplitude = target*C/S
    started = time.perf_counter()
    first = solver(source*amplitude, capacity, **options)
    qfirst = total(first['absorbed'])/C
    history = [{'amplitude': amplitude, 'Q_mass': qfirst}]
    best, best_amp, error = first, amplitude, abs(qfirst-target)
    lo, hi = 0., amplitude
    qhi = qfirst
    # Bracket with actual solves; no model of the retained-photon correction.
    while qhi < target-tolerance/4 and len(history) < 12:
        lo = hi
        hi *= min(2., max(1.01, target/max(qhi, 1e-12)*1.01))
        result = solver(source*hi, capacity, **options)
        qhi = total(result['absorbed'])/C
        history.append({'amplitude': hi, 'Q_mass': qhi})
        if abs(qhi-target) < error:
            best, best_amp, error = result, hi, abs(qhi-target)
    if error > tolerance/4 and qhi >= target:
        for _ in range(40):
            mid = (lo+hi)/2
            result = solver(source*mid, capacity, **options)
            q = total(result['absorbed'])/C
            history.append({'amplitude': mid, 'Q_mass': q})
            if abs(q-target) < error:
                best, best_amp, error = result, mid, abs(q-target)
            if error <= tolerance/4 or mid == lo or mid == hi:
                break
            if q < target:
                lo = mid
            else:
                hi = mid
    return first, best, {'input_amplitude': amplitude, 'amplitude': best_amp,
                        'target_Q_mass': target, 'absolute_error': error,
                        'matched': bool(error <= tolerance), 'solves': len(history),
                        'history': history, 'seconds': time.perf_counter()-started,
                        'interpolated_fields': False}


def independent_ledger(result, photons, cap):
    Q = total(photons)
    residual = (np.sum(photons, dtype=np.longdouble)-np.sum(result['absorbed'], dtype=np.longdouble)
                -np.sum(result['source_retained'], dtype=np.longdouble)-np.sum(result['source_escaped'], dtype=np.longdouble))
    per_source = result['source_absorbed']+result['source_retained']+result['source_escaped']-photons
    active = photons > 0
    ledger = {'relative_residual': float(abs(residual)/Q),
              'source_max_relative_residual': float(np.max(abs(per_source[active])/photons[active])),
              'local_fraction_violation': float(np.max(np.maximum(result['absorbed']-cap, 0)/cap)),
              'negative_absorption_fraction': float(np.max(np.maximum(-result['absorbed'], 0)/cap)),
              'retained_fraction': total(result['source_retained'])/Q,
              'escaped_fraction': total(result['source_escaped'])/Q}
    ledger['passed'] = max(ledger[k] for k in ('relative_residual', 'source_max_relative_residual',
                         'local_fraction_violation', 'negative_absorption_fraction')) <= 2e-9
    if not ledger['passed']:
        raise RuntimeError(f'Independent photon audit failed: {ledger}')
    return ledger


def run_task(root, task_id):
    import adaptive

    root = Path(root)
    c = config_at(root)
    spec = task_list(c)[task_id]
    target = root/'cases'/spec['directory']
    target.mkdir(parents=True, exist_ok=True)
    if checked_receipt(target):
        return
    f = root/'fixtures'/f'seed{spec["seed"]}'/f'n{spec["grid"]}'
    source = np.load(f/'source.npy', mmap_mode='r')
    cap = np.load(f/'capacity.npy', mmap_mode='r')
    threshold = np.load(f/'es_threshold.npy', mmap_mode='r')
    amplitude = spec['dose']*total(cap)/total(source)
    variant = spec['variant']
    options = dict(min_angles=c['min_angles'], max_angles=c['max_angles'],
                   rays_per_surface_cell=c['rays_per_surface_cell'], periodic=True,
                   maximum_distance=spec['grid']*c['horizon_box_fraction'])
    if variant == 'angle2':
        options.update(min_angles=2*c['min_angles'], max_angles=2*c['max_angles'],
                       rays_per_surface_cell=2*c['rays_per_surface_cell'])
    if variant == 'distance2':
        options['maximum_distance'] *= 2
    adaptive.library()
    pc_budget, pc, pc_match = match_pc_mass(adaptive.solve, source, cap, spec['dose'],
                                           options, c['mass_match_tolerance'])
    elapsed = pc_match['seconds']
    actual_q = total(pc['absorbed'])/total(cap)
    ledger = independent_ledger(pc, source*pc_match['amplitude'], cap)
    budget_ledger = independent_ledger(pc_budget, source*amplitude, cap)
    es_started = time.perf_counter()
    same_budget = es_map(source, cap, threshold, amplitude)
    matched, match = match_mass(source, cap, threshold, actual_q, c['mass_match_tolerance'])
    es_seconds = time.perf_counter()-es_started
    edges = np.geomspace(2*np.pi/c['box_hMpc']*(1-1e-10),
                         np.pi*c['native_grid']/c['box_hMpc']*(1+1e-10), c['spectral_bins']+1)
    fields = {'pc': pc['absorbed'], 'pc_same_budget': pc_budget['absorbed'],
              'es_same_budget': same_budget, 'es_matched_mass': matched}
    report = {'task': spec, 'pc_source_amplitude': pc_match['amplitude'], 'input_source_amplitude': amplitude, 'pc_options': options,
              'pc_seconds': elapsed, 'es_maps_seconds': es_seconds, 'pc_audit': pc['audit'],
              'independent_ledger': ledger, 'same_budget_ledger': budget_ledger,
              'pc_mass_match': pc_match, 'mass_match': match,
              'requested_vs_actual_Q_mass': actual_q-spec['dose'],
              'retention_acceptable': ledger['retained_fraction'] <= c['retained_fraction_tolerance'],
              'spectra': {}, 'scope': 'independent static z=8 source-dose experiment; no time integration'}
    for name, absorbed in fields.items():
        report['spectra'][name] = spectrum(cap, absorbed, c['box_hMpc'], edges,
                                           c['k0_band_hMpc_inv'], c['redshift'])
        amp = (pc_match['amplitude'] if name == 'pc' else
               match['amplitude'] if name == 'es_matched_mass' else amplitude)
        emitted = total(source)*amp
        report['spectra'][name]['emitted_over_absorbed'] = emitted/total(absorbed)
        report['spectra'][name]['emitted_per_atom'] = emitted/total(cap)
    # Counts remain float64; float32 xHI is only a plotting/storage convenience.
    arrays = {f'{name}_absorbed': a for name, a in fields.items()}
    arrays.update(pc_source_absorbed=pc['source_absorbed'],
                  pc_source_retained=pc['source_retained'], pc_source_escaped=pc['source_escaped'],
                  pc_same_budget_source_absorbed=pc_budget['source_absorbed'],
                  pc_same_budget_source_retained=pc_budget['source_retained'],
                  pc_same_budget_source_escaped=pc_budget['source_escaped'])
    with (target/'fields.npz.partial').open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    (target/'fields.npz.partial').replace(target/'fields.npz')
    write_json(target/'result.json', report)
    finish(target, ['fields.npz', 'result.json'])


def collect(root):
    root = Path(root)
    c = config_at(root)
    reports, rows = [], []
    inputs = []
    for seed in c['seeds']:
        directory = root/'fixtures'/f'seed{seed}'
        if not checked_receipt(directory):
            raise RuntimeError(f'Missing completed fixture: {directory}')
        meta = json.loads((directory/'metadata.json').read_text())
        high = meta['grids'][str(c['native_grid'])]['input_k0_band']
        for n in c['grids']:
            low = meta['grids'][str(n)]['input_k0_band']
            inputs.append({'seed': seed, 'grid': n, 'input_k0_band': low,
                'fine_over_this_P_matter': high['P_matter']/low['P_matter'] if low['P_matter'] else None,
                'fine_over_this_P_source': high['P_source']/low['P_source'] if low['P_source'] else None})
    for task in task_list(c):
        d = root/'cases'/task['directory']
        if not checked_receipt(d):
            raise RuntimeError(f'Missing completed case: {d}')
        report = json.loads((d/'result.json').read_text())
        reports.append(report)
        for model, stat in report['spectra'].items():
            row = {**task, 'model': model, 'Q_mass': stat['Q_mass'], 'Q_volume': stat['Q_volume'],
                   'emitted_over_absorbed': stat['emitted_over_absorbed'],
                   'emitted_per_atom': stat['emitted_per_atom'],
                   'b_HI_squared_k0': stat['k0_band']['b_HI_squared'],
                   'modes_k0': stat['k0_band']['full_fft_modes'],
                   'pc_seconds': report['pc_seconds'], 'pc_solve_count': report['pc_mass_match']['solves'],
                   'mass_match_passed': report['mass_match']['matched'] and report['pc_mass_match']['matched'],
                   'retained_fraction': (report['same_budget_ledger']['retained_fraction'] if model=='pc_same_budget'
                                         else report['independent_ledger']['retained_fraction'] if model=='pc' else None)}
            rows.append(row)
    with (root/'summary.csv').open('w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    lookup = {(r['task']['seed'], r['task']['grid'], r['task']['dose'], r['task']['variant']): r for r in reports}
    checks = []
    for r in reports:
        t = r['task']
        if t['variant'] == 'baseline':
            continue
        base = lookup[t['seed'], t['grid'], t['dose'], 'baseline']
        b, v = (item['spectra']['pc']['k0_band']['b_HI_squared'] for item in (base, r))
        fb = np.load(root/'cases'/base['task']['directory']/'fields.npz')
        fv = np.load(root/'cases'/t['directory']/'fields.npz')
        cap = np.load(root/'fixtures'/f'seed{t["seed"]}'/f'n{t["grid"]}'/'capacity.npy', mmap_mode='r')
        error = total(abs(fb['pc_absorbed']-fv['pc_absorbed']))/(base['spectra']['pc']['emitted_per_atom']*total(cap))
        relative = v/b-1 if b else None
        checks.append({'task': t, 'b2_relative_change': relative,
                       'field_L1_over_emitted': error,
                       'same_budget_b2_relative_change': (r['spectra']['pc_same_budget']['k0_band']['b_HI_squared']/base['spectra']['pc_same_budget']['k0_band']['b_HI_squared']-1
                                                         if base['spectra']['pc_same_budget']['k0_band']['b_HI_squared'] else None),
                       'Q_mass_change': r['spectra']['pc']['Q_mass']-base['spectra']['pc']['Q_mass'],
                       'within_requested_b2_tolerance': bool(relative is not None and abs(relative) <= c['convergence_relative_tolerance']
                                                             and r['pc_mass_match']['matched'] and base['pc_mass_match']['matched'])})
        fb.close(); fv.close()
    # Paired seeds quantify variation in the *ratio*, not independent-box errors.
    resolution = []
    for coarse in c['grids']:
        if coarse == c['native_grid']:
            continue
        for q in c['photon_doses']:
            for model in ('pc', 'es_matched_mass', 'pc_same_budget', 'es_same_budget'):
                ratios = []
                for seed in c['seeds']:
                    fine = lookup[seed, c['native_grid'], q, 'baseline']
                    low = lookup[seed, coarse, q, 'baseline']
                    if model in ('pc', 'es_matched_mass'):
                        if not (fine['pc_mass_match']['matched'] and low['pc_mass_match']['matched']):
                            continue
                        if abs(fine['spectra'][model]['Q_mass']-low['spectra'][model]['Q_mass']) > 2*c['mass_match_tolerance']:
                            continue
                        if model == 'es_matched_mass' and not (fine['mass_match']['matched'] and low['mass_match']['matched']):
                            continue
                    denominator = low['spectra'][model]['k0_band']['b_HI_squared']
                    if denominator:
                        ratios.append(fine['spectra'][model]['k0_band']['b_HI_squared']/denominator)
                mean = float(np.mean(ratios)) if ratios else None
                resolution.append({'fine_grid': c['native_grid'], 'coarse_grid': coarse, 'dose': q,
                                   'model': model, 'paired_seed_ratios': ratios,
                                   'mean_ratio': mean, 'sample_std': float(np.std(ratios, ddof=1)) if len(ratios)>1 else None,
                                   'within_requested_b2_tolerance': bool(mean is not None and len(ratios)==len(c['seeds']) and abs(mean-1) <= c['convergence_relative_tolerance'])})
    summary = {'scope': 'Choudhury & Paranjape 2018 style comparison, with 21cmFAST native sources',
               'cases': len(reports), 'all_ledgers_passed': all(r['independent_ledger']['passed'] and r['same_budget_ledger']['passed'] for r in reports),
               'unmatched_PC_cases': [r['task'] for r in reports if not r['pc_mass_match']['matched']],
               'unmatched_ES_cases': [r['task'] for r in reports if not r['mass_match']['matched']],
               'significant_retention_cases': [r['task'] for r in reports if not r['retention_acceptable']],
               'angle_and_distance_checks': checks, 'resolution_checks': resolution,
               'input_resolution_checks': inputs,
               'tolerance_note': '5 percent is an experiment acceptance target, not a physical theorem; inspect near-zero bias separately',
               'paper_reference': 'https://arxiv.org/abs/1807.00836',
               'paper_comparison': 'same volume, redshift, mass-fraction and bias definitions; different density/source prescription; no SCRIPT run or digitized reference values'}
    write_json(root/'summary.json', summary)
    from plot_large_scale import make_plots
    make_plots(root, reports, rows, resolution, c)
    return summary


def run(root, config, threads, workers):
    root, config = Path(root).resolve(), Path(config).resolve()
    c = json.loads(config.read_text())
    validate_config(c)
    if not 1 <= workers <= threads <= 32:
        raise ValueError('Require 1 <= workers <= allocated threads <= 32')
    root.mkdir(parents=True, exist_ok=True)
    identity = {'config': c, 'code_sha256': code_hashes()}
    path = root/'experiment.json'
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise RuntimeError('Configuration or source changed; use a new run directory')
    else:
        write_json(path, identity)
    write_json(root/'tasks.json', task_list(c))
    logs = root/'logs'
    logs.mkdir(exist_ok=True)
    started = time.perf_counter()
    print(f'Native generation: {threads} threads; independent transport workers: {workers}', flush=True)
    for seed in c['seeds']:
        env = dict(os.environ, OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        with (logs/f'native_seed{seed}.log').open('a') as stream:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), 'fixture', '--root', str(root),
                            '--seed', str(seed), '--threads', str(threads)], env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        print(f'Native seed {seed} ready', flush=True)

    def launch(task):
        env = dict(os.environ, OMP_NUM_THREADS='1', OMP_PROC_BIND='false', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        with (logs/f"case_{task['id']:03d}.log").open('a') as stream:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), 'task', '--root', str(root),
                            '--task-id', str(task['id'])], env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        return task

    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(launch, t): t for t in task_list(c)}
        for future in as_completed(futures):
            t = futures[future]
            try:
                future.result()
                print(f"Completed {t['id']}: {t['directory']}", flush=True)
            except Exception as error:
                failures.append({'task': t, 'error': str(error)})
                print(f"FAILED {t['directory']}; see logs/case_{t['id']:03d}.log", flush=True)
    write_json(root/'execution.json', {'seconds_before_collection': time.perf_counter()-started,
                                       'native_threads': threads, 'transport_workers': workers,
                                       'failures': failures})
    if failures:
        raise RuntimeError(f'{len(failures)} cases failed; completed cases are resumable and hash checked')
    summary = collect(root)
    # Completion is distinct from scientific convergence; failed convergence
    # checks are results, never silently converted into execution failures.
    (root/'COMPLETED').write_text('All planned cases and analysis completed\n')
    print(json.dumps({'cases': summary['cases'], 'all_ledgers_passed': summary['all_ledgers_passed'],
                      'total_seconds': time.perf_counter()-started}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('run')
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--config', type=Path, default=HERE/'default.json')
    p.add_argument('--threads', type=int, default=32)
    p.add_argument('--workers', type=int, default=32)
    p = sub.add_parser('fixture')
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--seed', required=True, type=int)
    p.add_argument('--threads', required=True, type=int)
    p = sub.add_parser('task')
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--task-id', required=True, type=int)
    p = sub.add_parser('collect')
    p.add_argument('--root', required=True, type=Path)
    args = vars(parser.parse_args())
    command = args.pop('command')
    {'run': run, 'fixture': fixture, 'task': run_task, 'collect': collect}[command](**args)
