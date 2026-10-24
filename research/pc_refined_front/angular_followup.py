#!/usr/bin/env python3
"""Angular convergence at fixed sources, gas refinement and photon budgets.

Consume a completed ``refined-front-followup`` run. Every angular bank is an
exact integer multiple of its stored bank; the old predictor maximum is not a
cap. A multiplier-one solve must reproduce its actual parent field. Completion
certifies numerical execution and that reproduction, not angular convergence.
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
from scipy import fft

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import large_followup as base
from fine_state_metrics import fine_state_metrics
from large_followup import (checked_receipt, coarsen_counts, digest, finish,
                           independent_ledger, spectrum, total, write_json)

PHYSICS_FILES = ('pc_refined_front/refined.cpp', 'pc_refined_front/refined.py',
                 'pc_refinement/front_subcells.py', 'pc_large_scale/pc_statistics.py')


def code_hashes():
    return base.code_hashes()


def scaled_angles(bank, multiplier):
    """Scale a stored source-wise bank exactly, without predictor clipping."""
    a = np.asarray(bank)
    if (isinstance(multiplier, (bool, np.bool_))
            or not isinstance(multiplier, (int, np.integer)) or multiplier < 1):
        raise ValueError('Angular multiplier must be a positive integer')
    if (a.dtype.kind not in 'iu' or not a.size or (a < 24).any()
            or (a % 24).any()):
        raise ValueError('Angular bank must contain positive integer multiples of 24')
    if int(a.max()) > np.iinfo(np.int32).max // int(multiplier):
        raise ValueError('Scaled angular bank exceeds int32 range')
    if int(np.unique(a).sum(dtype=np.int64)) > np.iinfo(np.int32).max // int(multiplier):
        raise ValueError('Combined scaled direction bank exceeds int32 range')
    return np.ascontiguousarray(a.astype(np.int64)*int(multiplier), dtype=np.int32)


def selection(parent_identity, grids=None, doses=None, refinements=None,
              multipliers=None, contiguous_ray_gb=4., power_tolerance=.01,
              field_tolerance=.01):
    old = parent_identity['selection']
    grids = list(old['grids'] if grids is None else grids)
    doses = list(old['doses'] if doses is None else doses)
    refinements = list([f for f in (2, 4) if f in old['refinements']]
                       if refinements is None else refinements)
    multipliers = list((1, 2, 4, 8, 16) if multipliers is None else multipliers)
    for name, values, allowed in (('grids', grids, old['grids']),
                                  ('doses', doses, old['doses']),
                                  ('refinements', refinements, old['refinements'])):
        if not values or len(set(values)) != len(values) or any(v not in allowed for v in values):
            raise ValueError(f'Select unique {name} available in the completed parent')
    if any(type(g) is not int for g in grids) or any(type(f) is not int for f in refinements):
        raise ValueError('Grid and refinement factors must be integers')
    if (len(multipliers) < 2 or len(set(multipliers)) != len(multipliers)
            or 1 not in multipliers or any(type(m) is not int or m < 1 for m in multipliers)):
        raise ValueError('Select at least two unique positive integer multipliers including 1')
    if (not np.isfinite(contiguous_ray_gb) or contiguous_ray_gb < 0
            or contiguous_ray_gb*1024**3 > np.iinfo(np.int64).max):
        raise ValueError('Contiguous-ray GiB threshold must be finite and nonnegative')
    if any(not np.isfinite(v) or v <= 0 for v in (power_tolerance, field_tolerance)):
        raise ValueError('Diagnostic tolerances must be positive and finite')
    measure = old['measurement_grid']
    if any(g % measure for g in grids):
        raise ValueError('Parent measurement grid must divide every selected source grid')
    return dict(grids=grids, doses=doses, refinements=refinements,
                multipliers=sorted(multipliers), measurement_grid=measure,
                distance_factor=old['distance_factor'],
                contiguous_ray_gib=float(contiguous_ray_gb),
                contiguous_ray_bytes=int(contiguous_ray_gb*1024**3),
                contiguous_ray_scope='Reservation/storage-selection threshold, not an RSS or total memory limit',
                power_tolerance=float(power_tolerance), field_tolerance=float(field_tolerance),
                Q_mass_tolerance=1e-10, retained_tolerance=1e-4,
                diagnostic_pairs_required=2,
                diagnostic_scope='Two finest adjacent angular pairs at fixed original sources and gas mesh, including resolved-subcell L1; at least three multipliers required; not a continuum certificate')


def _parent_identity(parent):
    parent = Path(parent).resolve()
    if not (parent/'COMPLETED').is_file():
        raise RuntimeError('Refined parent run has not completed')
    manifest = json.loads((parent/'run_manifest.json').read_text())
    if manifest.get('experiment') != 'refined-front-followup':
        raise RuntimeError('Parent must be a completed refined-front-followup run')
    identity = json.loads((parent/'results/experiment.json').read_text())
    for name in PHYSICS_FILES:
        expected = identity['code_sha256'].get(name)
        if expected != digest(base.RESEARCH/name):
            raise RuntimeError(f'Parent physics differs from the current unchanged solver: {name}')
        archived = parent/'source/research'/name
        if archived.exists() and digest(archived) != expected:
            raise RuntimeError(f'Archived parent physics changed: {name}')
    for name, expected in identity['input_sha256'].items():
        if digest(name) != expected:
            raise RuntimeError(f'Parent native input changed: {name}')
    return identity


def task_list(parent_identity, selected):
    rows = []
    for old in parent_identity['tasks']:
        if (old['model'] != 'refined' or old['grid'] not in selected['grids']
                or old['dose'] not in selected['doses']
                or old['subcells_per_axis'] not in selected['refinements']):
            continue
        for multiplier in selected['multipliers']:
            rows.append(dict(id=len(rows), seed=old['seed'], grid=old['grid'], dose=old['dose'],
                             model='refined', subcells_per_axis=old['subcells_per_axis'],
                             multiplier=multiplier, base_angle_key=old['angle_key'],
                             angle_key=f"{old['angle_key']}_a{multiplier}",
                             parent_directory=old['directory'],
                             directory=f"{old['directory']}_a{multiplier}"))
    expected = (len(parent_identity['config']['seeds'])*len(selected['grids'])
                *len(selected['doses'])*len(selected['refinements'])*len(selected['multipliers']))
    if not rows or len(rows) != expected or len({r['directory'] for r in rows}) != len(rows):
        raise RuntimeError('Completed parent does not contain the full requested refined-case matrix')
    return rows


def prepare(root, parent, grids=None, doses=None, refinements=None, multipliers=None,
            contiguous_ray_gb=4., power_tolerance=.01, field_tolerance=.01):
    root, parent = Path(root).resolve(), Path(parent).resolve()
    if root == parent or parent in root.parents:
        raise ValueError('Angular output must be outside the completed parent run')
    if root.exists() and any(root.iterdir()):
        raise ValueError('Preparation requires a fresh empty output directory')
    prior = _parent_identity(parent)
    native_parent = Path(prior['parent_run']).resolve()
    if root == native_parent or native_parent in root.parents:
        raise ValueError('Angular output must be outside the native parent run')
    chosen = selection(prior, grids, doses, refinements, multipliers,
                       contiguous_ray_gb, power_tolerance, field_tolerance)
    plan = task_list(prior, chosen)
    inputs = dict(prior['input_sha256'])
    for p in (parent/'COMPLETED', parent/'run_manifest.json', parent/'results/experiment.json'):
        inputs[str(p)] = digest(p)
    prior_lookup = {r['directory']: r for r in prior['tasks']}
    baseline_reports = {}
    for directory in sorted({r['parent_directory'] for r in plan}):
        folder = parent/'results/cases'/directory
        if not checked_receipt(folder):
            raise RuntimeError(f'Parent refined case lacks a valid receipt: {folder}')
        receipt = json.loads((folder/'COMPLETED.json').read_text())['files_sha256']
        if not {'result.json', 'fields.npz'} <= set(receipt):
            raise RuntimeError('Parent receipt must cover result.json and fields.npz')
        report = json.loads((folder/'result.json').read_text())
        if report['task'] != prior_lookup[directory] or not report['ledger']['passed']:
            raise RuntimeError(f'Parent case identity or ledger is invalid: {directory}')
        if report['measurement_grid'] != chosen['measurement_grid']:
            raise RuntimeError('Parent case uses a different measurement grid')
        baseline_reports[directory] = report
        for name in ('COMPLETED.json', 'result.json', 'fields.npz'):
            inputs[str(folder/name)] = digest(folder/name)
    root.mkdir(parents=True, exist_ok=True)
    (root/'angles').mkdir()
    angle_files, preflight, seen = {}, [], set()
    for spec in plan:
        key = spec['angle_key']
        if key in seen:
            continue
        seen.add(key)
        bank_path = parent/'results/angles'/f"{spec['base_angle_key']}.npy"
        expected = prior['prepared_angle_sha256'][f"angles/{spec['base_angle_key']}.npy"]
        if digest(bank_path) != expected:
            raise RuntimeError('Parent angular bank changed')
        inputs[str(bank_path)] = expected
        original_bank = np.load(bank_path)
        photons, cap = base.input_fields(native_parent, spec['seed'], spec['grid'], spec['dose'])
        if original_bank.shape != photons.shape:
            raise RuntimeError('Parent angular bank shape differs from the source grid')
        bank = scaled_angles(original_bank, spec['multiplier'])
        name = f'angles/{key}.npy'
        np.save(root/name, bank)
        angle_files[name] = digest(root/name)
        active = bank[photons > 0]
        potential = int(active.sum(dtype=np.int64))
        horizon = base.source_options(prior['config'], spec['grid'], chosen)['maximum_distance']
        largest_refinement = max(chosen['refinements'])
        direction_count = int(np.unique(bank).sum(dtype=np.int64))
        compact = (largest_refinement*(horizon+spec['grid']+2) < np.iinfo(np.int16).max
                   and direction_count-1 <= np.iinfo(np.uint16).max)
        ray_bytes = 48 if compact else 64
        preflight.append(dict(seed=spec['seed'], grid=spec['grid'], dose=spec['dose'],
                              multiplier=spec['multiplier'], active_sources=int(active.size),
                              min_angles=int(active.min()), mean_angles=float(active.mean()),
                              max_angles=int(active.max()), potential_rays=potential,
                              combined_direction_bank_size=direction_count,
                              potential_packed_ray_bytes=ray_bytes*potential,
                              estimated_transport_ray_bytes=ray_bytes,
                              estimated_contiguous_storage_selected=(chosen['contiguous_ray_bytes'] > 0
                                  and ray_bytes*potential <= chosen['contiguous_ray_bytes']),
                              estimate_scope='Potential ray capacity for the widest selected spatial refinement; compact 48-byte or wide 64-byte records, not RSS'))
    for report in baseline_reports.values():
        key = report['task']['angle_key']
        if report['angular_bank_sha256'] != prior['prepared_angle_sha256'][f'angles/{key}.npy']:
            raise RuntimeError('Parent case used a different frozen angular bank')
    identity = dict(schema=1, scope='Angular convergence of static fixed-source refined native 21cmFAST fields',
                    parent_run=str(parent), native_parent_run=str(native_parent),
                    config=prior['config'], selection=chosen, tasks=plan,
                    input_sha256=dict(sorted(inputs.items())), prepared_angle_sha256=angle_files,
                    code_sha256=code_hashes(), physics_sha256={n: digest(base.RESEARCH/n) for n in PHYSICS_FILES},
                    angular_preflight=preflight,
                    angular_policy='Exact integer scaling of stored per-source banks; no predictor recomputation and no clipping to its old cap',
                    angular_quadrature='Octant-balanced cyclic-permutation quadrature; banks are non-nested as angular count increases; monotonic changes are not assumed',
                    parallelism='Independent static cases in separate processes; each event kernel is serial',
                    continuum_accuracy_certified=False)
    write_json(root/'experiment.json', identity)
    verify(root)
    return identity


def verify(root):
    root = Path(root)
    identity = json.loads((root/'experiment.json').read_text())
    if identity['code_sha256'] != code_hashes():
        raise RuntimeError('Angular follow-up source code changed after preparation')
    for name, expected in identity['input_sha256'].items():
        if digest(name) != expected:
            raise RuntimeError(f'Parent input changed after preparation: {name}')
    for name, expected in identity['prepared_angle_sha256'].items():
        if digest(root/name) != expected:
            raise RuntimeError('Prepared angular bank changed after preparation')
    return identity


def verify_case_inputs(root, identity, spec):
    if identity['code_sha256'] != code_hashes():
        raise RuntimeError('Angular follow-up source code changed after preparation')
    paths = [base.fixture_path(identity['native_parent_run'], spec['seed'], spec['grid'])/n
             for n in ('source.npy', 'capacity.npy')]
    old = Path(identity['parent_run'])/'results/cases'/spec['parent_directory']
    paths += [old/n for n in ('COMPLETED.json', 'result.json', 'fields.npz')]
    paths += [Path(identity['parent_run'])/'results/angles'/f"{spec['base_angle_key']}.npy"]
    for p in paths:
        if digest(p) != identity['input_sha256'][str(p)]:
            raise RuntimeError(f'Parent case input changed: {p}')
    name = f"angles/{spec['angle_key']}.npy"
    if digest(Path(root)/name) != identity['prepared_angle_sha256'][name]:
        raise RuntimeError('Prepared angular bank changed')


def _edges(config, measure):
    return np.geomspace(2*np.pi/config['box_hMpc']*(1-1e-10),
                        np.pi*measure/config['box_hMpc']*(1+1e-10), config['spectral_bins']+1)


def run_task(root, task_id):
    import refined
    root = Path(root)
    identity = json.loads((root/'experiment.json').read_text())
    if type(task_id) is not int or not 0 <= task_id < len(identity['tasks']):
        raise ValueError('Task ID is outside the prepared plan')
    spec, config = identity['tasks'][task_id], identity['config']
    verify_case_inputs(root, identity, spec)
    folder = root/'cases'/spec['directory']
    if checked_receipt(folder):
        return
    folder.mkdir(parents=True, exist_ok=True)
    photons, cap = base.input_fields(identity['native_parent_run'], spec['seed'], spec['grid'], spec['dose'])
    angles = np.load(root/'angles'/f"{spec['angle_key']}.npy")
    opts = base.source_options(config, spec['grid'], identity['selection'])
    refined.library()  # Compilation is excluded from the reported solve timing.
    started = time.perf_counter()
    result = refined.solve(photons, cap, subcells_per_axis=spec['subcells_per_axis'],
                           angle_counts=angles, maximum_distance=opts['maximum_distance'], periodic=True,
                           contiguous_ray_bytes=identity['selection']['contiguous_ray_bytes'])
    elapsed = time.perf_counter()-started
    if not np.array_equal(result['angle_counts'][photons > 0], angles[photons > 0]):
        raise RuntimeError('Solver changed the frozen scaled angular bank')
    ledger = independent_ledger(result, photons, cap)
    measure = identity['selection']['measurement_grid']
    cm = coarsen_counts(cap, measure)
    am = coarsen_counts(result['absorbed'], measure)
    stat = spectrum(cm, am, config['box_hMpc'], _edges(config, measure),
                    config['k0_band_hMpc_inv'], config['redshift'])
    active = angles[photons > 0]
    report = dict(task=spec, options=opts, seconds=elapsed,
                  timing=dict(raw_solve_seconds=elapsed,
                    scope='Single solve in a separate worker; frozen bank supplied, no predictor cost. Includes preparation, templates, transport and state output; excludes compilation and diagnostics.'),
                  ledger=ledger, spectra=stat, audit=result['audit'], measurement_grid=measure,
                  parent_grid_Q_volume=float(np.mean(result['absorbed']/cap)),
                  fine_cells_per_parent=spec['subcells_per_axis']**3,
                  angular_bank_sha256=identity['prepared_angle_sha256'][f"angles/{spec['angle_key']}.npy"],
                  angular_counts=dict(min=int(active.min()), mean=float(active.mean()), max=int(active.max()),
                                      active_sources=int(active.size)),
                  field_comparison_scope='Original source grid, common measurement grid, and sparse resolved subcells; no dense fine grid stored',
                  contiguous_ray_bytes=identity['selection']['contiguous_ray_bytes'],
                  continuum_accuracy_certified=False)
    with (folder/'fields.npz.partial').open('wb') as stream:
        np.savez_compressed(stream, absorbed=result['absorbed'], measurement_absorbed=am,
                            measurement_capacity=cm, source_absorbed=result['source_absorbed'],
                            source_retained=result['source_retained'], source_escaped=result['source_escaped'],
                            fine_cleared_bits=result['state']['cleared_bits'],
                            fine_partial_ids=result['state']['partial_ids'],
                            fine_partial_ionized=result['state']['partial_ionized'])
    (folder/'fields.npz.partial').replace(folder/'fields.npz')
    write_json(folder/'result.json', report)
    verify_case_inputs(root, identity, spec)
    finish(folder, ['result.json', 'fields.npz'])
    if not ledger['passed']:
        raise RuntimeError('Numerical ledger failed; diagnostics saved in result.json')


def field_metrics(new, old, emitted):
    a, b = np.asarray(new), np.asarray(old)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or emitted <= 0:
        raise ValueError('Finite matching fields and positive emitted budget required')
    delta = a-b
    return dict(field_L1_over_emitted=total(abs(delta))/emitted,
                maximum_count_difference=float(np.max(abs(delta))))


def compare_fields(capnew, anew, capold, aold, box_hMpc, edges, band):
    """Paired fractional-HI powers, correlation and residual on one common mesh."""
    arrays = [np.asarray(v, dtype=float) for v in (capnew, anew, capold, aold)]
    c1, a1, c0, a0 = arrays
    if (c1.ndim != 3 or len(set(c1.shape)) != 1 or any(a.shape != c1.shape for a in arrays)
            or any(not np.isfinite(a).all() for a in arrays)
            or (c1 <= 0).any() or (c0 <= 0).any()):
        raise ValueError('Matching finite cubic fields with positive capacities required')
    h1, h0 = c1-a1, c0-a0
    if h1.mean() <= 0 or h0.mean() <= 0:
        raise ValueError('Fractional HI spectra require a nonzero neutral mean')
    f1, f0 = fft.rfftn(h1/h1.mean()-1), fft.rfftn(h0/h0.mean()-1)
    n = c1.shape[0]
    k = 2*np.pi*fft.fftfreq(n, d=box_hMpc/n)
    kz = 2*np.pi*fft.rfftfreq(n, d=box_hMpc/n)
    km = np.sqrt(k[:,None,None]**2+k[None,:,None]**2+kz[None,None,:]**2)
    w = np.full(kz.size, 2.); w[0] = 1.
    if n % 2 == 0: w[-1] = 1.
    weights = np.broadcast_to(w, km.shape)
    nyquist = np.pi*n/box_hMpc
    keep = (km > 0) & (km <= nyquist*(1+1e-12))
    bins = np.searchsorted(edges, km, side='right')-1
    norm = box_hMpc**3/n**6
    def reduce(mask):
        modes = int(weights[mask].sum())
        if not modes: return None
        ws = weights[mask]*norm/modes
        new = float(np.sum(abs(f1[mask])**2*ws))
        old = float(np.sum(abs(f0[mask])**2*ws))
        cross = float(np.sum((f1[mask]*f0[mask].conj()).real*ws))
        difference = float(np.sum(abs(f1[mask]-f0[mask])**2*ws))
        kval = float(np.sum(km[mask]*weights[mask])/modes)
        return dict(k_hMpc_inv=kval, k_over_nyquist=kval/nyquist, full_fft_modes=modes,
                    P_new=new, P_reference=old, P_cross=cross,
                    r_new_reference=cross/np.sqrt(new*old) if new*old > 0 else None,
                    P_difference=difference, relative_difference_power=difference/old if old > 0 else None,
                    power_ratio=new/old if old > 0 else None)
    shells = []
    for i in range(len(edges)-1):
        row = reduce(keep & (bins == i))
        if row is not None:
            row.update(bin_index=i, k_min=float(edges[i]), k_max=float(edges[i+1]))
            shells.append(row)
    return dict(shells=shells, k0_band=reduce(keep & (km >= band[0]) & (km < band[1])),
                nyquist_hMpc_inv=nyquist,
                definition='Fractional neutral-count fields separately mean-normalized; paired Fourier difference, not difference of power spectra',
                P_units='(h^-1 cMpc)^3')


def _load_fields(folder):
    with np.load(Path(folder)/'fields.npz') as data:
        return {name: data[name] for name in data.files}


def compare_reports(root, identity, new, old, *, old_folder=None):
    spec = new['task']; reference = old['task']
    a = _load_fields(Path(root)/'cases'/spec['directory'])
    b = _load_fields(old_folder or Path(root)/'cases'/reference['directory'])
    photons, cap = base.input_fields(identity['native_parent_run'], spec['seed'], spec['grid'], spec['dose'])
    emitted = total(photons)
    measure = identity['selection']['measurement_grid']
    ca = a.get('measurement_capacity', coarsen_counts(cap, measure))
    if 'measurement_capacity' in b:
        cb = b['measurement_capacity']
    else:
        oldcap = np.load(base.fixture_path(identity['native_parent_run'], reference['seed'], reference['grid'])/'capacity.npy')
        cb = coarsen_counts(oldcap, measure)
    out = field_metrics(a['measurement_absorbed'], b['measurement_absorbed'], emitted)
    out.update(new=spec, reference=reference,
               b2_relative_change=base.relative(new['spectra']['k0_band']['b_HI_squared'], old['spectra']['k0_band']['b_HI_squared']),
               Q_mass_change=new['spectra']['Q_mass']-old['spectra']['Q_mass'],
               maximum_retained_fraction=max(new['ledger']['retained_fraction'], old['ledger']['retained_fraction']),
               raw_solve_runtime_ratio=new['seconds']/old['seconds'] if old['seconds'] else None,
               same_original_sources=(spec['seed'], spec['grid'], spec['dose']) == (reference['seed'], reference['grid'], reference['dose']),
               spectral_comparison=compare_fields(ca, a['measurement_absorbed'], cb, b['measurement_absorbed'],
                    identity['config']['box_hMpc'], _edges(identity['config'], measure), identity['config']['k0_band_hMpc_inv']))
    if out['same_original_sources']:
        out['original_grid'] = field_metrics(a['absorbed'], b['absorbed'], emitted)
        out['original_grid']['volume_mean_absolute_ionized_fraction_difference'] = float(np.mean(abs(a['absorbed']-b['absorbed'])/cap))
        out['source_ledgers'] = {name+'_L1_over_emitted': total(abs(a[name]-b[name]))/emitted
                                for name in ('source_absorbed', 'source_retained', 'source_escaped')}
    else:
        out['original_grid'] = None
        out['source_ledgers'] = None
    out['resolved_subcells'] = None
    fine_names = ('cleared_bits', 'partial_ids', 'partial_ionized')
    if (out['same_original_sources']
            and spec['subcells_per_axis'] == reference['subcells_per_axis']
            and all('fine_'+name in fields for fields in (a, b) for name in fine_names)):
        states = [{name: fields['fine_'+name] for name in fine_names} for fields in (a, b)]
        out['resolved_subcells'] = fine_state_metrics(cap, spec['subcells_per_axis'],
                                                     states[0], states[1], emitted)
    return out


def _summarize(rows):
    groups = {}
    for row in rows:
        new, old = row['new'], row['reference']
        key = (new['grid'], old['grid'], new['dose'], new['subcells_per_axis'],
               old['subcells_per_axis'], new.get('multiplier', 1), old.get('multiplier', 1))
        groups.setdefault(key, []).append(row)
    output = []
    for key, subset in sorted(groups.items()):
        values = [r['b2_relative_change'] for r in subset if r['b2_relative_change'] is not None]
        output.append(dict(new_grid=key[0], reference_grid=key[1], dose=key[2],
            new_subcells_per_axis=key[3], reference_subcells_per_axis=key[4],
            new_multiplier=key[5], reference_multiplier=key[6],
            seeds=[r['new']['seed'] for r in subset],
            paired_b2_change_mean=float(np.mean(values)) if values else None,
            paired_b2_change_seed_std=float(np.std(values, ddof=1)) if len(values)>1 else None,
            mean_field_L1_over_emitted=float(np.mean([r['field_L1_over_emitted'] for r in subset])),
            maximum_field_L1_over_emitted=max(r['field_L1_over_emitted'] for r in subset),
            maximum_absolute_Q_mass_change=max(abs(r['Q_mass_change']) for r in subset),
            maximum_retained_fraction=max(r['maximum_retained_fraction'] for r in subset)))
    return output


def collect(root):
    root = Path(root)
    identity = verify(root); selected = identity['selection']
    reports = []
    for spec in identity['tasks']:
        folder = root/'cases'/spec['directory']
        if not checked_receipt(folder):
            raise RuntimeError(f'Missing completed angular case: {folder}')
        report = json.loads((folder/'result.json').read_text())
        if report['task'] != spec or report['measurement_grid'] != selected['measurement_grid']:
            raise RuntimeError('Case report does not match prepared identity')
        if report['angular_bank_sha256'] != identity['prepared_angle_sha256'][f"angles/{spec['angle_key']}.npy"]:
            raise RuntimeError('Case report angular bank does not match the prepared bank')
        reports.append(report)
    key = lambda r: (r['task']['seed'], r['task']['grid'], r['task']['dose'],
                     r['task']['subcells_per_axis'], r['task']['multiplier'])
    lookup = {key(r): r for r in reports}
    baselines, angular, spatial, crossgrid, checks = [], [], [], [], []
    multipliers = selected['multipliers']
    for row in reports:
        seed, grid, dose, refinement, multiplier = key(row)
        if multiplier == 1:
            folder = Path(identity['parent_run'])/'results/cases'/row['task']['parent_directory']
            old = json.loads((folder/'result.json').read_text())
            comp = compare_reports(root, identity, row, old, old_folder=folder)
            comp['passed'] = bool(comp['original_grid']['field_L1_over_emitted'] <= 1e-9
                and max(comp['source_ledgers'].values()) <= 1e-9
                and abs(comp['Q_mass_change']) <= selected['Q_mass_tolerance'])
            baselines.append(comp)
        else:
            prior_multiplier = max(m for m in multipliers if m < multiplier)
            refs = [prior_multiplier] if prior_multiplier == 1 else [prior_multiplier, 1]
            for m in refs:
                comp = compare_reports(root, identity, row, lookup[(seed, grid, dose, refinement, m)])
                comp['comparison_kind'] = 'adjacent_angular' if m == prior_multiplier else 'angular_to_baseline'
                angular.append(comp)
                if multiplier in multipliers[-2:] and m == prior_multiplier:
                    power = comp['b2_relative_change']
                    check = dict(task=row['task'], reference_multiplier=m,
                        absolute_k0_power_change=abs(power) if power is not None else None,
                        original_grid_field_L1_over_emitted=comp['original_grid']['field_L1_over_emitted'],
                        common_grid_field_L1_over_emitted=comp['field_L1_over_emitted'],
                        resolved_subcell_L1_over_emitted=comp['resolved_subcells']['subcell_L1_over_emitted']
                            if comp['resolved_subcells'] is not None else None,
                        absolute_Q_mass_change=abs(comp['Q_mass_change']),
                        maximum_retained_fraction=comp['maximum_retained_fraction'])
                    check['passed'] = bool(power is not None and abs(power) <= selected['power_tolerance']
                        and check['original_grid_field_L1_over_emitted'] <= selected['field_tolerance']
                        and check['resolved_subcell_L1_over_emitted'] is not None
                        and check['resolved_subcell_L1_over_emitted'] <= selected['field_tolerance']
                        and check['absolute_Q_mass_change'] <= selected['Q_mass_tolerance']
                        and check['maximum_retained_fraction'] <= selected['retained_tolerance'])
                    checks.append(check)
        smaller = [f for f in selected['refinements'] if f < refinement]
        if smaller:
            prior = lookup[(seed, grid, dose, max(smaller), multiplier)]
            if row['angular_bank_sha256'] != prior['angular_bank_sha256']:
                raise RuntimeError('Angular bank changed in a spatial-only comparison')
            spatial.append(compare_reports(root, identity, row, prior))
        if grid != max(selected['grids']):
            fine = lookup[(seed, max(selected['grids']), dose, refinement, multiplier)]
            comp = compare_reports(root, identity, fine, row)
            comp['scope'] = 'Different source and density representations, conservatively projected to the same measurement mesh'
            crossgrid.append(comp)
    summary = dict(scope=identity['scope'], cases=len(reports),
        all_ledgers_passed=all(r['ledger']['passed'] for r in reports),
        all_parent_baselines_passed=bool(baselines) and all(r['passed'] for r in baselines),
        diagnostic_convergence_passed=len(multipliers) >= 3 and bool(checks) and all(r['passed'] for r in checks),
        diagnostic_convergence_has_enough_levels=len(multipliers) >= 3,
        diagnostic_pairs_required=2,
        convergence_thresholds={k: selected[k] for k in ('power_tolerance', 'field_tolerance', 'Q_mass_tolerance', 'retained_tolerance')},
        parent_baselines=baselines, angular_comparisons=angular, front_refinement=spatial,
        input_grid_comparisons=crossgrid, convergence_checks=checks,
        angular_paired_summary=_summarize(angular), front_refinement_paired_summary=_summarize(spatial),
        input_grid_paired_summary=_summarize(crossgrid), case_results=reports,
        measurement_grid=selected['measurement_grid'], continuum_accuracy_certified=False,
        interpretation='Completion means cases passed numerical ledgers and unchanged parent baselines. Both finest adjacent angular pairs must pass diagnostic gates, including resolved-subcell L1, with at least three levels; this is a finite-resolution criterion, not continuum accuracy. Quadrature banks are non-nested and changes need not be monotonic. Small power changes can coexist with spatial redistribution; inspect subcell/original-grid L1 and paired Fourier correlation/residual. Near-Nyquist measurements remain sensitive to input and output discretization.')
    write_json(root/'summary.json', summary)
    if not summary['all_ledgers_passed'] or not summary['all_parent_baselines_passed']:
        raise RuntimeError('Angular validation failed a numerical ledger or parent reproduction; inspect summary.json')
    return summary


def run(root, workers=32):
    root = Path(root).resolve(); identity = verify(root)
    if type(workers) is not int or not 1 <= workers <= 32:
        raise ValueError('Use 1 to 32 independent workers')
    (root/'logs').mkdir(exist_ok=True)
    started = time.perf_counter()
    def launch(spec):
        env = dict(os.environ, OMP_NUM_THREADS='1', OMP_PROC_BIND='false',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        with (root/'logs'/f"case_{spec['id']:03d}.log").open('a') as stream:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), 'task', '--root', str(root),
                            '--task-id', str(spec['id'])], env=env, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(launch, spec): spec for spec in identity['tasks']}
        for future in as_completed(futures):
            spec = futures[future]
            try:
                future.result(); print('Completed:', spec['directory'], flush=True)
            except Exception as error:
                failures.append(dict(task=spec, error=str(error)))
    execution = dict(seconds_before_collection=time.perf_counter()-started, requested_workers=workers,
                     simultaneous_tasks=min(workers, len(identity['tasks'])), failures=failures,
                     timing_scope='Elapsed orchestration time including worker process startup, compilation and diagnostics; excludes preparation')
    write_json(root/'execution.json', execution)
    if failures: raise RuntimeError(f'{len(failures)} cases failed; see logs')
    summary = collect(root)
    verify(root)
    execution.update(total_seconds=time.perf_counter()-started,
                     diagnostic_convergence_passed=summary['diagnostic_convergence_passed'])
    write_json(root/'execution.json', execution)
    (root/'COMPLETED').write_text('Numerical execution and parent reproduction passed. Inspect diagnostic_convergence_passed separately.\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'run', 'task', 'collect'):
        p = sub.add_parser(name); p.add_argument('--root', required=True)
        if name == 'prepare':
            p.add_argument('--parent', required=True)
            for option in ('grids', 'refinements', 'multipliers'):
                p.add_argument('--'+option, type=int, nargs='+')
            p.add_argument('--doses', type=float, nargs='+')
            p.add_argument('--contiguous-ray-gb', type=float, default=4.)
            p.add_argument('--power-tolerance', type=float, default=.01)
            p.add_argument('--field-tolerance', type=float, default=.01)
        if name == 'run': p.add_argument('--workers', type=int, default=32)
        if name == 'task': p.add_argument('--task-id', type=int, required=True)
    args = vars(parser.parse_args()); command = args.pop('command')
    result = {'prepare': prepare, 'run': run, 'task': run_task, 'collect': collect}[command](**args)
    if command == 'prepare':
        print(json.dumps(dict(tasks=len(result['tasks']), selection=result['selection'],
                              maximum_potential_rays=max(r['potential_rays'] for r in result['angular_preflight'])), indent=2))


if __name__ == '__main__':
    main()
