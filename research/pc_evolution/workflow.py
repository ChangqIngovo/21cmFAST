#!/usr/bin/env python3
"""Run native-source, fixed-gas temporal validation in isolated worker processes.

Native endpoint generation uses the allocated OpenMP threads. Independent
carried-state histories then use single-thread processes, with the same frozen
source increments on all temporal meshes. Completion and convergence are
separate outcomes; this driver does not evolve or advect the gas density.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def code_hashes():
    files = []
    for directory in ('pc_evolution', 'pc_refined_front', 'pc_refinement',
                      'pc_reconsidered', 'pc_large_scale', 'pc_alternatives'):
        files += [p for p in (REPO/'research'/directory).glob('*')
                  if p.is_file() and p.suffix in ('.py', '.cpp', '.h', '.sh', '.pbs')]
    return {str(p.relative_to(REPO)): digest(p) for p in sorted(files)}


def worker_environment(threads=1):
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
               PYTHONNOUSERSITE='1')
    return env


def validate_selection(*, threads, dimension, box_len, seeds, angles,
                       base_intervals, z_start, z_end, reference_z):
    if type(threads) is not int or not 1 <= threads <= 32:
        raise ValueError('threads must be an integer from 1 to 32')
    if type(dimension) is not int or dimension < 4:
        raise ValueError('dimension must be an integer at least four')
    if not math.isfinite(box_len) or box_len <= 0:
        raise ValueError('box_len must be finite and positive')
    if type(base_intervals) is not int or base_intervals < 1:
        raise ValueError('base_intervals must be a positive integer')
    if (not seeds or len(seeds) != len(set(seeds))
            or any(type(s) is not int or s < 1 for s in seeds)):
        raise ValueError('seeds must be unique positive integers')
    if (len(angles) < 2 or len(angles) != len(set(angles))
            or any(type(a) is not int or a < 24 or a % 24 for a in angles)):
        raise ValueError('select at least two unique direction counts, multiples of 24')
    if not all(math.isfinite(z) for z in (z_start, z_end, reference_z)):
        raise ValueError('redshifts must be finite')
    if not z_start > z_end >= 0 or reference_z < 0:
        raise ValueError('redshift interval must descend and reference redshift be nonnegative')


def task_list(fixtures, angles):
    """One ES history and twelve default PC histories per native fixture."""
    tasks = []
    for index, fixture in enumerate(fixtures):
        seed = fixture['seed']
        common = dict(fixture_index=index, seed=seed, fixture=fixture['path'],
                      fixture_sha256=fixture['sha256'])
        tasks.append(dict(common, id=len(tasks), model='matched_es',
                          directory=f'seed{seed}_matched_es'))
        for f in (2, 4):
            for count in sorted(angles):
                for stride in (4, 2, 1):
                    tasks.append(dict(common, id=len(tasks), model='refined_pc',
                                      subcells_per_axis=f, f=f, angles=count, stride=stride,
                                      directory=f'seed{seed}_f{f}_k{count}_stride{stride}'))
    return tasks


def run_worker(*, fixture, output, subcells_per_axis=None, angles=None,
               stride=None, control=False):
    from native_history import load_history
    from diagnostics import enrich_history, run_controls
    fixture, output = Path(fixture).resolve(), Path(output).resolve()
    before = digest(fixture)
    hashes = code_hashes()
    history = load_history(fixture)
    if control:
        report = run_controls(history, output)
    else:
        from evolution import run_history
        report = run_history(history, subcells_per_axis=subcells_per_axis,
                             angles=angles, stride=stride,
                             maximum_distance=8*history['capacity'].shape[0],
                             output=output)
        enriched = enrich_history(report, history)
        if enriched is not None:
            report = enriched
    if digest(fixture) != before or code_hashes() != hashes:
        raise RuntimeError('Fixture or executable source changed during the worker')
    report['worker_peak_rss_bytes'] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                                          * (1 if sys.platform == 'darwin' else 1024))
    report['worker_peak_rss_scope'] = 'Measured process resident-memory high-water mark; includes Python and diagnostics, excludes separate compiler subprocesses'
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'summary.json', report)
    checkpoints = {str(p.relative_to(output)): digest(p)
                   for p in sorted(output.rglob('*.npz'))}
    write_json(output/'receipt.json', dict(fixture=str(fixture), fixture_sha256=before,
               code_sha256=hashes, summary_sha256=digest(output/'summary.json'),
               checkpoint_sha256=checkpoints))
    return report


def _run_task(root, task):
    output = root/'cases'/task['directory']
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(HERE/'workflow.py'), '_worker',
               '--fixture', task['fixture'], '--output', str(output)]
    if task['model'] == 'matched_es':
        command += ['--control']
    else:
        command += ['--f', str(task['subcells_per_axis']), '--angles', str(task['angles']),
                    '--stride', str(task['stride'])]
    started = time.perf_counter()
    log = root/'logs'/(task['directory']+'.log')
    with log.open('w') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                env=worker_environment(1), check=False)
    return dict(task=task, returncode=result.returncode,
                elapsed_seconds=time.perf_counter()-started, log=str(log))


def read_checked_report(root, task, expected_code):
    directory = Path(root)/'cases'/task['directory']
    receipt = json.loads((directory/'receipt.json').read_text())
    if (receipt['fixture_sha256'] != task['fixture_sha256']
            or digest(task['fixture']) != task['fixture_sha256']
            or receipt['code_sha256'] != expected_code
            or digest(directory/'summary.json') != receipt['summary_sha256']):
        raise RuntimeError('Worker input, source or summary identity changed: '+task['directory'])
    for name, expected in receipt['checkpoint_sha256'].items():
        if digest(directory/name) != expected:
            raise RuntimeError('Checkpoint changed: '+str(directory/name))
    report = json.loads((directory/'summary.json').read_text())
    report['task'] = task
    return report


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def temporal_gate(row, *, power_tolerance=.01, field_tolerance=.01):
    """Check a nontrivial common epoch without accepting missing statistics."""
    qnew = row.get('new_Q_mass', row.get('Q_mass'))
    qref = row.get('reference_Q_mass', row.get('Q_mass'))
    eligible = (_finite(qnew) and _finite(qref)
                and .1 <= qnew <= .95 and .1 <= qref <= .95)
    result = dict(row, eligible=eligible)
    if not eligible:
        result.update(passed=None, exclusion_reason='Both mass ionized fractions must lie in [0.1, 0.95]')
        return result
    limits = dict(absolute_k0_power_change=power_tolerance,
                  original_grid_field_L1_over_emitted=field_tolerance,
                  resolved_subcell_L1_over_emitted=field_tolerance,
                  absolute_Q_mass_change=1e-10,
                  maximum_retained_fraction=1e-8)
    result['failed_metrics'] = [key for key, limit in limits.items()
                                if not _finite(row.get(key)) or abs(row[key]) > limit]
    if (row.get('source_budgets_match') is not True
            or not _finite(row.get('source_L1_over_emitted'))
            or row['source_L1_over_emitted'] > 1e-12):
        result['failed_metrics'].append('source_budgets_match')
    result['passed'] = not result['failed_metrics']
    return result


def source_quadrature_gate(row, tolerance=.01):
    """Check each relevant epoch; late photons cannot dilute earlier errors."""
    checks = []
    for checkpoint in row.get('checkpoints', []):
        q = checkpoint.get('Q_mass')
        error = checkpoint.get('cumulative_L1_over_emitted')
        eligible = _finite(q) and .1 <= q <= .95
        checks.append(dict(checkpoint, eligible=eligible,
                           passed=bool(_finite(error) and 0 <= error <= tolerance)
                           if eligible else None))
    eligible_checks = [check for check in checks if check['eligible']]
    return dict(row, tolerance=tolerance,
                checkpoint_checks=checks, eligible_epochs=len(eligible_checks),
                passed=bool(row.get('stride') == 2 and eligible_checks
                            and all(check['passed'] for check in eligible_checks)),
                interpretation='Stride-two endpoint trapezoids compared with genuine finest-node trapezoids at every output with Q_mass in [0.1, 0.95]; one adjacent source-quadrature comparison')


def collect(root):
    from native_history import load_history
    from diagnostics import compare_histories
    root = Path(root).resolve()
    identity = json.loads((root/'experiment.json').read_text())
    if code_hashes() != identity['code_sha256']:
        raise RuntimeError('Executable source changed after experiment preparation')
    reports = [read_checked_report(root, task, identity['code_sha256'])
               for task in identity['tasks']]
    pcs = [r for r in reports if r['task']['model'] == 'refined_pc']
    controls = [r for r in reports if r['task']['model'] == 'matched_es']
    lookup = {(r['task']['fixture_index'], r['task']['subcells_per_axis'],
               r['task']['angles'], r['task']['stride']): r for r in pcs}
    temporal, angular, spatial, matched, gates, groups = [], [], [], [], [], []
    control_lookup = {r['task']['fixture_index']:r for r in controls}
    for fixture_index, fixture in enumerate(identity['fixtures']):
        history = load_history(fixture['path'])
        for f in (2, 4):
            for count in sorted(identity['selection']['angles']):
                group_gates = []
                for fine, coarse in ((2, 4), (1, 2)):
                    new = lookup[(fixture_index, f, count, fine)]
                    ref = lookup[(fixture_index, f, count, coarse)]
                    rows = compare_histories(new, ref, history)
                    tag = dict(seed=fixture['seed'], subcells_per_axis=f,
                               angles=count, new_stride=fine, reference_stride=coarse)
                    temporal.extend(dict(r, **tag) for r in rows)
                    current = [temporal_gate(dict(r, **tag)) for r in rows]
                    gates.extend(current)
                    valid = [g for g in current if g['eligible']]
                    group_gates.append(dict(new_stride=fine, reference_stride=coarse,
                                            eligible_epochs=len(valid),
                                            passed=bool(valid) and all(g['passed'] for g in valid)))
                groups.append(dict(seed=fixture['seed'], subcells_per_axis=f, angles=count,
                                   adjacent_pairs=group_gates,
                                   passed=all(g['passed'] for g in group_gates)))
                for stride in (4, 2, 1):
                    if f == 4:
                        new, ref = lookup[(fixture_index, 4, count, stride)], lookup[(fixture_index, 2, count, stride)]
                        spatial.extend(dict(r, seed=fixture['seed'], angles=count, stride=stride,
                                            new_subcells_per_axis=4, reference_subcells_per_axis=2)
                                       for r in compare_histories(new, ref, history))
                    case = lookup[(fixture_index, f, count, stride)]
                    matched.extend(dict(r, seed=fixture['seed'], subcells_per_axis=f,
                                        angles=count, stride=stride)
                                   for r in compare_histories(case, control_lookup[fixture_index], history))
                    previous = [a for a in identity['selection']['angles'] if a < count]
                    if previous:
                        ref_count = max(previous)
                        new, ref = lookup[(fixture_index, f, count, stride)], lookup[(fixture_index, f, ref_count, stride)]
                        angular.extend(dict(r, seed=fixture['seed'], subcells_per_axis=f, stride=stride,
                                            new_angles=count, reference_angles=ref_count)
                                       for r in compare_histories(new, ref, history))
    source_budgets_match = all(r.get('source_budgets_match') is True
                               for r in temporal+angular+spatial+matched)
    numerical = bool(pcs) and all(r.get('all_ledgers_passed') is True for r in pcs)
    source_quadrature = [dict(seed=r['task']['seed'], stride=r['task']['stride'],
                              **r['source_quadrature']) for r in pcs
                         if r['task']['subcells_per_axis'] == 2
                         and r['task']['angles'] == min(identity['selection']['angles'])]
    source_checks = [source_quadrature_gate(r) for r in source_quadrature if r['stride'] == 2]
    source_passed = (len(source_checks) == len(identity['fixtures'])
                     and all(r['passed'] for r in source_checks))
    transport_passed = bool(groups) and all(g['passed'] for g in groups)
    summary = dict(scope=identity['scope'], cases=len(pcs), control_histories=len(controls),
                   all_ledgers_passed=numerical,
                   all_comparison_source_budgets_match=source_budgets_match,
                   diagnostic_temporal_convergence_passed=transport_passed,
                   diagnostic_source_quadrature_passed=source_passed,
                   diagnostic_convergence_passed=transport_passed and source_passed,
                   diagnostic_convergence_scope='Transport time-step and finest-source-quadrature diagnostics, conditional on chosen angular and spatial resolution',
                   failed_diagnostics=([name for name, passed in (
                       ('transport_time_step', transport_passed), ('source_quadrature', source_passed)) if not passed]),
                   temporal_convergence_groups=groups, temporal_checks=gates,
                   source_quadrature_diagnostics=source_quadrature,
                   source_quadrature_checks=source_checks,
                   temporal_comparisons=temporal, angular_comparisons=angular,
                   matched_pc_es_comparisons=matched,
                   spatial_comparisons=spatial, case_results=pcs, control_results=controls,
                   convergence_thresholds=dict(power_tolerance=.01, field_tolerance=.01,
                       Q_mass_tolerance=1e-10, retained_tolerance=1e-8,
                       Q_mass_window=[.1,.95], adjacent_temporal_pairs_required=2),
                   scope_notes=[
                       'Native source rates evolve with redshift; gas capacity remains fixed at the reference redshift.',
                       'Every temporal mesh receives identical per-source integrated photon budgets at common output nodes.',
                       'Source quadrature errors are reported separately from transport time-step differences.',
                       'Native ES is contextual: its instantaneous cumulative proxy differs from the integrated-rate photon history.',
                       'No ion advection, recombination, photoheating feedback or finite light propagation is implemented here.',
                       'A convergence diagnostic is not a continuum accuracy certificate.'])
    write_json(root/'summary.json', summary)
    return summary


def plot_summary(root, summary):
    from diagnostics import plot_results
    grouped = {}
    for row in summary['temporal_comparisons']:
        key = tuple(row[k] for k in ('seed','subcells_per_axis','angles','new_stride','reference_stride'))
        grouped.setdefault(key, []).append(row)
    comparisons = [dict(kind='temporal', new=dict(seed=seed, f=f, angles=angles, stride=new_stride),
                        reference=dict(seed=seed, f=f, angles=angles, stride=reference_stride),
                        checkpoints=rows)
                   for (seed,f,angles,new_stride,reference_stride),rows in grouped.items()]
    plot_results(root, summary['case_results'], summary['control_results'], comparisons)


def run(*, root, threads=32, dimension=32, box_len=200., seeds=(1234,2345,3456,4567),
        z_start=25., z_end=5., base_intervals=80, reference_z=8., angles=(384,768), fixtures=None,
        memory_gb=280.):
    validate_selection(threads=threads, dimension=dimension, box_len=box_len,
                       seeds=list(seeds), angles=list(angles), base_intervals=base_intervals,
                       z_start=z_start, z_end=z_end, reference_z=reference_z)
    if not math.isfinite(memory_gb) or memory_gb <= 0:
        raise ValueError('Worker memory planning budget must be positive and finite')
    root = Path(root).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('Use a fresh results directory; prior histories are not overwritten')
    root.mkdir(parents=True, exist_ok=True)
    (root/'logs').mkdir()
    (root/'fixtures').mkdir()
    started = time.perf_counter()
    hashes = code_hashes()
    execution = dict(requested_workers=threads, native_threads=threads, failures=[],
                     timing_scope='Elapsed orchestration including native endpoint generation, worker startup, compilation, diagnostics and collection; excludes PBS preparation')
    write_json(root/'execution.json', execution)
    fixture_records = []
    try:
        from native_history import load_history
        if fixtures:
            paths = [Path(p).resolve() for p in fixtures]
        else:
            paths = []
            for seed in seeds:
                output = root/'fixtures'/f'seed{seed}.npz'
                command = [sys.executable, str(HERE/'native_history.py'), '--output', str(output),
                           '--dimension', str(dimension), '--box-len', str(box_len),
                           '--threads', str(threads), '--seed', str(seed), '--z-start', str(z_start),
                           '--z-end', str(z_end), '--intervals', str(4*base_intervals),
                           '--reference-z', str(reference_z)]
                with (root/'logs'/f'native_seed{seed}.log').open('w') as stream:
                    result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                            env=worker_environment(threads), check=False)
                if result.returncode:
                    raise RuntimeError(f'Native history generation failed for seed {seed}; inspect logs/native_seed{seed}.log')
                paths.append(output)
        used_seeds = set()
        for index, path in enumerate(paths):
            history = load_history(path)
            if (len(history['redshifts'])-1) % 4:
                raise ValueError('Fixture interval count must be divisible by four')
            meta = history['metadata']
            seed = int(meta.get('seed', meta.get('random_seed', index+1)))
            if seed in used_seeds:
                raise ValueError('Each fixture must have a distinct native random seed')
            used_seeds.add(seed)
            fixture_records.append(dict(seed=seed, path=str(path), sha256=digest(path),
                                        dimension=int(history['capacity'].shape[0]),
                                        intervals=len(history['redshifts'])-1, metadata=meta))
        if not fixture_records:
            raise ValueError('No native fixtures were supplied')
        selection = dict(angles=sorted(angles), refinements=[2,4], strides=[4,2,1],
                         native_requested=dict(dimension=dimension, box_len=box_len, seeds=list(seeds),
                                               z_start=z_start,z_end=z_end,reference_z=reference_z,
                                               intervals=4*base_intervals),
                         external_fixtures=bool(fixtures))
        identity = dict(schema=1, experiment='native-rate-carried-evolution',
                        scope='Time-dependent native 21cmFAST sources and carried fine ionization state in fixed gas; not coupled Qin20 evolution',
                        selection=selection, fixtures=fixture_records, code_sha256=hashes,
                        tasks=task_list(fixture_records, angles), python=sys.executable)
        identity['memory_preflight'] = [dict(seed=record['seed'],
            maximum_angles=max(angles),
            maximum_contiguous_ray_capacity_bytes=record['dimension']**3*max(angles)*48,
            worker_planning_bytes=2*(64*record['dimension']**3*max(angles)
                                    +128*record['dimension']**3*64
                                    +12*(record['intervals']+1)*record['dimension']**3)+512*1024**2,
            scope='Planning estimate: twice full 64-byte ray capacity plus 128 bytes per fine cell plus endpoint arrays, then 512 MiB overhead. Not an RSS measurement or guaranteed bound.')
            for record in fixture_records]
        largest = max(r['worker_planning_bytes'] for r in identity['memory_preflight'])
        memory_workers = int(memory_gb*1024**3//largest)
        if memory_workers < 1:
            raise ValueError('Worker memory planning budget is below one estimated history; use a smaller grid or larger allocation')
        identity['memory_budget_gib'] = memory_gb
        try:
            identity['git_revision'] = subprocess.check_output(
                ['git','-C',str(REPO),'rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            identity['git_revision'] = None
        if code_hashes() != hashes:
            raise RuntimeError('Executable source changed during native preparation')
        write_json(root/'experiment.json', identity)
        execution['native_preparation_seconds'] = time.perf_counter()-started
        execution['simultaneous_tasks'] = min(threads, len(identity['tasks']), memory_workers)
        execution['worker_memory_budget_gib'] = memory_gb
        execution['largest_worker_planning_bytes'] = largest
        execution['memory_limited_worker_count'] = memory_workers
        execution['task_count'] = len(identity['tasks'])
        outcomes = []
        with ThreadPoolExecutor(max_workers=execution['simultaneous_tasks']) as pool:
            futures = [pool.submit(_run_task, root, task) for task in identity['tasks']]
            for future in as_completed(futures):
                row = future.result()
                outcomes.append(row)
                print(json.dumps(dict(completed=len(outcomes), total=len(futures),
                                      task=row['task']['directory'], returncode=row['returncode'])), flush=True)
        execution['task_execution'] = sorted(outcomes,key=lambda r:r['task']['id'])
        execution['failures'] = [row for row in outcomes if row['returncode']]
        execution['seconds_before_collection'] = time.perf_counter()-started
        write_json(root/'execution.json', execution)
        if execution['failures']:
            raise RuntimeError('Evolution worker failures; inspect execution.json and worker logs')
        summary = collect(root)
        plot_summary(root, summary)
        execution['diagnostic_convergence_passed'] = summary['diagnostic_convergence_passed']
        execution['total_seconds'] = time.perf_counter()-started
        write_json(root/'execution.json', execution)
        if not summary['all_ledgers_passed'] or not summary['all_comparison_source_budgets_match']:
            raise RuntimeError('An independent cumulative photon ledger or shared-source comparison failed')
        (root/'COMPLETED').touch()
        return summary
    except BaseException as error:
        execution['total_seconds'] = time.perf_counter()-started
        execution['error'] = f'{type(error).__name__}: {error}'
        write_json(root/'execution.json', execution)
        (root/'FAILED').write_text(execution['error']+'\n')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    command = sub.add_parser('run')
    command.add_argument('--root', type=Path, required=True)
    command.add_argument('--threads', type=int, default=32)
    command.add_argument('--dimension', type=int, default=32)
    command.add_argument('--box-len', type=float, default=200.)
    command.add_argument('--seeds', type=int, nargs='+', default=[1234,2345,3456,4567])
    command.add_argument('--z-start', type=float, default=25.)
    command.add_argument('--z-end', type=float, default=5.)
    command.add_argument('--base-intervals', type=int, default=80)
    command.add_argument('--reference-z', type=float, default=8.)
    command.add_argument('--angles', type=int, nargs='+', default=[384,768])
    command.add_argument('--fixtures', type=Path, nargs='+')
    command.add_argument('--memory-gb', type=float, default=280.,
                         help='Worker memory planning budget in GiB, below the PBS allocation; default 280')
    command = sub.add_parser('collect')
    command.add_argument('--root', type=Path, required=True)
    command = sub.add_parser('_worker')
    command.add_argument('--fixture', type=Path, required=True)
    command.add_argument('--output', type=Path, required=True)
    command.add_argument('--f', dest='subcells_per_axis', type=int)
    command.add_argument('--angles', type=int)
    command.add_argument('--stride', type=int)
    command.add_argument('--control', action='store_true')
    args = vars(parser.parse_args())
    mode = args.pop('command')
    if mode == 'run':
        result = run(**args)
        print(json.dumps({k: result[k] for k in ('cases','all_ledgers_passed','diagnostic_convergence_passed')}),flush=True)
    elif mode == 'collect':
        collect(**args)
    else:
        run_worker(**args)


if __name__ == '__main__':
    main()
