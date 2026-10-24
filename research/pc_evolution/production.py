#!/usr/bin/env python3
"""One large-box PC/ES history with prescribed native sources and fixed gas.

This runs the accepted carried-state model over the complete configured
redshift range. It is not a coupled Qin20 calculation: gas is frozen at the
reference redshift and recombination, heating and feedback are disabled.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from workflow import code_hashes, digest, write_json
from native_history import build_native_history, load_history
from evolution import run_history
from diagnostics import field_spectrum, run_controls

SCOPE = 'Prescribed time-dependent native sources and carried refined PC state in gas frozen at reference redshift; not coupled Qin20 evolution'
DEFAULT_CONFIG = dict(dimension=128, highres_dimension=512, box_len=400., seed=1234,
                      z_start=25., z_end=5., intervals=320, reference_z=8.,
                      subcells_per_axis=4, angles=768, stride=4,
                      maximum_distance=1024.)
INTMAX = np.iinfo(np.int32).max


def validate_config(config):
    if not isinstance(config, dict) or set(config) != set(DEFAULT_CONFIG):
        raise ValueError('Production config must contain exactly: ' + ', '.join(DEFAULT_CONFIG))
    c = dict(config)
    for key in ('dimension', 'highres_dimension', 'seed', 'intervals',
                'subcells_per_axis', 'angles', 'stride'):
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if c['dimension'] < 4 or c['highres_dimension'] < c['dimension']:
        raise ValueError('dimension >= 4 and highres_dimension >= dimension are required')
    if c['highres_dimension'] % c['dimension']:
        raise ValueError('highres_dimension must be a multiple of dimension')
    if c['seed'] > np.iinfo(np.uint32).max:
        raise ValueError('seed must fit a positive uint32')
    if c['subcells_per_axis'] not in (2, 4, 8) or c['stride'] not in (1, 2, 4):
        raise ValueError('subcells_per_axis must be 2, 4 or 8; stride must be 1, 2 or 4')
    if c['intervals'] % 4:
        raise ValueError('intervals must be divisible by four for common output checkpoints')
    if c['angles'] < 24 or c['angles'] % 24:
        raise ValueError('angles must be a positive multiple of 24')
    for key in ('box_len', 'z_start', 'z_end', 'reference_z', 'maximum_distance'):
        if isinstance(c[key], bool) or not isinstance(c[key], (int, float)) or not math.isfinite(c[key]):
            raise ValueError(f'{key} must be finite')
        c[key] = float(c[key])
    if c['box_len'] <= 0 or c['maximum_distance'] <= 0 or not c['z_start'] > c['z_end'] >= 0 or c['reference_z'] < 0:
        raise ValueError('Positive length/horizon and descending nonnegative redshifts required')
    if not c['z_end'] <= c['reference_z'] <= c['z_start']:
        raise ValueError('reference_z must lie within the simulated redshift range')
    if c['dimension']**3 * c['angles'] > INTMAX:
        raise ValueError('dimension^3 * angles exceeds the int32 ray-index limit')
    if c['subcells_per_axis'] * (c['maximum_distance'] + c['dimension'] + 2) > INTMAX:
        raise ValueError('Refined propagation coordinates exceed the int32 limit')
    return c


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _seal(path, payload):
    write_json(path, dict(payload=payload, sha256=hashlib.sha256(_canonical(payload)).hexdigest()))


def _unseal(path):
    record = json.loads(Path(path).read_text())
    payload = record.get('payload')
    if not isinstance(payload, dict) or record.get('sha256') != hashlib.sha256(_canonical(payload)).hexdigest():
        raise RuntimeError(f'Production receipt checksum changed: {path}')
    return payload


def _atomic_text(path, text):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@contextmanager
def _lock(root):
    with (root/'.production.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('This production directory is already running') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _fixture_matches(history, config):
    m = history['metadata']
    mapping = dict(dimension='dimension', highres_dimension='highres_dimension',
                   box_len='box_len_cMpc', seed='random_seed', intervals='intervals',
                   z_start='redshift_start', z_end='redshift_end', reference_z='reference_redshift')
    for key, name in mapping.items():
        if m.get(name) != config[key]:
            raise ValueError(f'Fixture does not match production config: {key}')
    expected = np.linspace(config['z_start'], config['z_end'], config['intervals']+1)
    if not np.array_equal(history['redshifts'], expected) or history['capacity'].shape != (config['dimension'],)*3:
        raise ValueError('Fixture redshift sampling or grid shape differs from production config')
    options = m.get('inputs', {}).get('simulation_options', {})
    if (options.get('DIM') != config['highres_dimension']
            or options.get('HII_DIM') != config['dimension']
            or options.get('BOX_LEN') != config['box_len']):
        raise ValueError('Fixture native geometry does not match production config')


def _receipt(root, phase, experiment_sha256, files):
    payload = dict(phase=phase, experiment_sha256=experiment_sha256,
                   files={str(Path(p).resolve()): digest(p) for p in files})
    _seal(root/(phase+'_receipt.json'), payload)


def _check_receipt(root, phase, experiment_sha256):
    payload = _unseal(root/(phase+'_receipt.json'))
    if payload.get('phase') != phase or payload.get('experiment_sha256') != experiment_sha256:
        raise RuntimeError(f'{phase} receipt refers to a different experiment')
    for name, expected in payload['files'].items():
        if not Path(name).is_file() or digest(name) != expected:
            raise RuntimeError(f'{phase} artifact changed: {name}')
    return payload


def _phase_files(directory, report):
    return [directory/'summary.json', *[Path(r['path']) for r in report['checkpoints']]]


def _preserve_unfinished(root, directory):
    """Keep an interrupted, uncommitted phase rather than deleting its files."""
    if directory.exists():
        destination = root/'incomplete'
        destination.mkdir(exist_ok=True)
        directory.rename(destination/(directory.name+'_'+uuid.uuid4().hex[:12]))


def _load_fields(row, names):
    path = Path(row['path'])
    if digest(path) != row['sha256']:
        raise RuntimeError(f'Checkpoint changed: {path}')
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in names}


def _diagnostic_node(history, pc, es):
    if pc['node_index'] != es['node_index'] or pc['time_seconds'] != es['time_seconds']:
        raise RuntimeError('PC/ES checkpoints refer to different times')
    # Do not load partial_ids/partial_ionized: spectra use the original grid.
    p = _load_fields(pc, ('ionized', 'source_emitted'))
    e = _load_fields(es, ('ionized', 'source_emitted'))
    emitted = float(np.sum(p['source_emitted'], dtype=np.longdouble))
    scale = max(emitted, float(np.sum(e['source_emitted'], dtype=np.longdouble)), np.finfo(float).tiny)
    source_l1 = float(np.sum(np.abs(p['source_emitted']-e['source_emitted']), dtype=np.longdouble)/scale)
    ps = field_spectrum(history, p['ionized'], pc['redshift'])
    espectrum = es.get('spectra')
    if espectrum is None:
        espectrum = field_spectrum(history, e['ionized'], pc['redshift'])
    native_index = pc['node_index']
    return dict(node_index=native_index, redshift=pc['redshift'], time_seconds=pc['time_seconds'],
                Q_mass_PC=pc['Q_mass'], Q_mass_ES=es['Q_mass'],
                xHI_volume_PC=pc['xHI_volume'], xHI_volume_ES=es['xHI_volume'],
                emitted=emitted, source_L1_over_emitted=source_l1,
                source_budgets_match=bool(source_l1 <= 1e-12),
                pc_ledger=pc['ledger'], es_photon_residual_over_emitted=es['photon_residual_over_emitted'],
                native_ES_context_xHI_volume=float(np.mean(history['native_xhi'][native_index], dtype=np.float64)),
                PC_spectra=ps, ES_spectra=espectrum)


def _csv_outputs(root, rows):
    stream = io.StringIO()
    fields = ['node_index', 'redshift', 'Q_mass_PC', 'Q_mass_ES', 'xHI_volume_PC',
              'xHI_volume_ES', 'emitted', 'retained_fraction_PC', 'ledger_relative_residual_PC',
              'source_L1_over_emitted', 'es_photon_residual_over_emitted',
              'native_ES_context_xHI_volume']
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({**{name: row[name] for name in fields if name in row},
                         'retained_fraction_PC': row['pc_ledger']['retained_fraction'],
                         'ledger_relative_residual_PC': row['pc_ledger']['relative_residual']})
    _atomic_text(root/'history.csv', stream.getvalue())
    stream = io.StringIO()
    fields = ['node_index', 'redshift', 'model', 'k_hMpc_inv', 'full_fft_modes',
              'P_HI', 'P_21_mK2_hMpc3', 'Delta2_21_mK2', 'b_HI_squared', 'r_HI_matter']
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        for model in ('PC', 'ES'):
            spectrum = row[model+'_spectra']
            if spectrum is None:
                continue
            for shell in spectrum['shells']:
                writer.writerow({**{name: shell[name] for name in fields if name in shell},
                                 'node_index': row['node_index'], 'redshift': row['redshift'], 'model': model})
    _atomic_text(root/'power_spectra.csv', stream.getvalue())


def _plot(root, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    z = [r['redshift'] for r in rows]
    for model, style in (('PC', '-'), ('ES', '--')):
        axes[0].plot(z, [r['xHI_volume_'+model] for r in rows], style, label=model)
    axes[0].set(xlabel='Redshift', ylabel=r'$\langle x_{\rm HI}\rangle_V$')
    axes[0].invert_xaxis()
    axes[0].legend()
    eligible = [r for r in rows if .1 <= r['Q_mass_PC'] <= .95
                and r['PC_spectra'] is not None and r['ES_spectra'] is not None]
    if eligible:
        chosen = {min(eligible, key=lambda r: abs(r['Q_mass_PC']-q))['node_index'] for q in (.2,.5,.8)}
        for row in eligible:
            if row['node_index'] not in chosen:
                continue
            a = row['PC_spectra']['shells']; b = {r['bin_index']: r for r in row['ES_spectra']['shells']}
            pairs = [(r,b[r['bin_index']]) for r in a if r['bin_index'] in b and b[r['bin_index']]['P_21_mK2_hMpc3'] > 0]
            axes[1].plot([a['k_hMpc_inv'] for a,b in pairs],
                         [a['P_21_mK2_hMpc3']/b['P_21_mK2_hMpc3'] for a,b in pairs],
                         label=f"z={row['redshift']:.2f}")
        axes[1].legend(fontsize=8)
    axes[1].set(xscale='log', xlabel=r'$k\ [h\,{\rm cMpc}^{-1}]$', ylabel=r'$P_{21}^{\rm PC}/P_{21}^{\rm ES}$ (same redshift)')
    axes[1].axhline(1, color='.5', linewidth=.8)
    axes[2].plot(z, [100*r['pc_ledger']['retained_fraction'] for r in rows])
    axes[2].set(xlabel='Redshift', ylabel='Retained / emitted photons [%]')
    axes[2].set_yscale('symlog', linthresh=1e-5)
    axes[2].invert_xaxis()
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle('Prescribed native sources; fixed gas; matched-source ES', fontsize=11)
    for suffix in ('png','pdf'):
        target = root/f'evolution_comparison.{suffix}'
        temporary = target.with_name(target.name+'.tmp')
        fig.savefig(temporary, format=suffix, dpi=170)
        temporary.replace(target)
    plt.close(fig)


def _diagnostics(root, history, pc, es, experiment_sha256):
    directory = root/'diagnostics'
    directory.mkdir(exist_ok=True)
    identity = dict(experiment_sha256=experiment_sha256,
                    pc_summary_sha256=digest(root/'pc/summary.json'),
                    es_summary_sha256=digest(root/'es/summary.json'))
    journal = directory/'progress.json'
    progress = _unseal(journal) if journal.exists() else dict(identity=identity, nodes=[])
    if progress['identity'] != identity:
        raise RuntimeError('Diagnostic inputs changed')
    existing = {row['node_index']: row for row in progress['nodes']}
    expected_indices = [r['node_index'] for r in pc['checkpoints']]
    if [r['node_index'] for r in progress['nodes']] != expected_indices[:len(existing)]:
        raise RuntimeError('Diagnostic checkpoint sequence changed')
    reference = {r['node_index']:r for r in es['checkpoints']}
    if set(reference) != set(expected_indices):
        raise RuntimeError('PC and ES do not share all output checkpoints')
    rows = []
    for pc_row in pc['checkpoints']:
        index = pc_row['node_index']
        path = directory/f'node_{index:05d}.json'
        if index in existing:
            if digest(path) != existing[index]['sha256']:
                raise RuntimeError('Saved diagnostic node changed')
            row = json.loads(path.read_text())
        else:
            row = _diagnostic_node(history, pc_row, reference[index])
            write_json(path, row)
            progress['nodes'].append(dict(node_index=index, sha256=digest(path)))
            _seal(journal, progress)
        rows.append(row)
    return rows


def _run_locked(root, config, threads, resume, fixture):
    started = time.perf_counter()
    experiment_path = root/'experiment.json'
    supplied = None if fixture is None else str(Path(fixture).resolve())
    codes = code_hashes()
    identity = dict(version=1, scope=SCOPE, config=config, code_sha256=codes,
                    supplied_fixture=supplied,
                    supplied_fixture_sha256=digest(supplied) if supplied else None,
                    runtime=dict(python=sys.version, numpy=np.__version__))
    if experiment_path.exists():
        if not resume:
            raise FileExistsError('Production run already exists; use --resume with identical inputs')
        if _unseal(experiment_path) != identity:
            raise RuntimeError('Production config, code, runtime or fixture changed')
    else:
        remaining = [p for p in root.iterdir() if p.name != '.production.lock']
        if remaining:
            raise FileExistsError('Fresh production output must be empty')
        _seal(experiment_path, identity)
    experiment_sha256 = digest(experiment_path)
    if (root/'COMPLETED').exists():
        for phase in ('fixture','pc','es','diagnostics'):
            _check_receipt(root, phase, experiment_sha256)
        return json.loads((root/'summary.json').read_text())
    execution = dict(scope=SCOPE, threads=threads, native_threads=threads,
                     transport_parallelism='one serial latest-refined PC kernel; native source generation uses allocated OpenMP threads',
                     resumed=bool(resume), status='running', phase='fixture', phases_seconds={})
    write_json(root/'execution.json', execution)
    print(json.dumps(dict(stage='production_start', scope=SCOPE, config=config,
                          native_threads=threads, transport_workers=1)), flush=True)
    try:
        phase_start = time.perf_counter()
        print(json.dumps(dict(stage='fixture_start')), flush=True)
        path = Path(supplied) if supplied else root/'native_history.npz'
        if (root/'fixture_receipt.json').exists():
            _check_receipt(root, 'fixture', experiment_sha256)
        elif not path.exists():
            if supplied:
                raise FileNotFoundError(path)
            build_native_history(output=path, dimension=config['dimension'],
                highres_dimension=config['highres_dimension'], box_len=config['box_len'],
                threads=threads, seed=config['seed'], z_start=config['z_start'], z_end=config['z_end'],
                intervals=config['intervals'], reference_z=config['reference_z'])
        history = load_history(path)
        _fixture_matches(history, config)
        if not (root/'fixture_receipt.json').exists():
            _receipt(root, 'fixture', experiment_sha256, [path])
        execution['fixture_sha256'] = digest(path)
        execution['phases_seconds']['fixture'] = time.perf_counter()-phase_start
        execution['phase'] = 'pc'
        write_json(root/'execution.json', execution)
        print(json.dumps(dict(stage='fixture_complete', seconds=execution['phases_seconds']['fixture'],
                              fixture=str(path))), flush=True)
        print(json.dumps(dict(stage='pc_start', progress_journal=str(root/'pc/progress.json'))), flush=True)
        phase_start = time.perf_counter()
        if (root/'pc_receipt.json').exists():
            _check_receipt(root, 'pc', experiment_sha256)
            pc = json.loads((root/'pc/summary.json').read_text())
        else:
            directory = root/'pc'
            continuing = ((directory/'progress.json').exists()
                          or (directory/'checkpoints/node_00000.npz').exists())
            if directory.exists() and not continuing:
                _preserve_unfinished(root, directory)
            pc = run_history(history, output=directory, resume=continuing,
                subcells_per_axis=config['subcells_per_axis'], angles=config['angles'],
                stride=config['stride'], maximum_distance=config['maximum_distance'])
            _receipt(root, 'pc', experiment_sha256,
                     [*_phase_files(directory, pc), directory/'progress.json'])
        execution['phases_seconds']['pc'] = time.perf_counter()-phase_start
        execution['phase'] = 'es'
        write_json(root/'execution.json', execution)
        print(json.dumps(dict(stage='pc_complete', seconds=execution['phases_seconds']['pc'])), flush=True)
        print(json.dumps(dict(stage='matched_es_start')), flush=True)
        phase_start = time.perf_counter()
        if (root/'es_receipt.json').exists():
            _check_receipt(root, 'es', experiment_sha256)
            es = json.loads((root/'es/summary.json').read_text())
        else:
            directory = root/'es'
            _preserve_unfinished(root, directory)
            es = run_controls(history, directory)
            _receipt(root, 'es', experiment_sha256, _phase_files(directory, es))
        execution['phases_seconds']['es'] = time.perf_counter()-phase_start
        execution['phase'] = 'diagnostics'
        write_json(root/'execution.json', execution)
        print(json.dumps(dict(stage='matched_es_complete', seconds=execution['phases_seconds']['es'])), flush=True)
        print(json.dumps(dict(stage='diagnostics_start')), flush=True)
        phase_start = time.perf_counter()
        rows = _diagnostics(root, history, pc, es, experiment_sha256)
        ledger_ok = pc['all_ledgers_passed'] is True
        budgets_ok = all(r['source_budgets_match'] for r in rows)
        if not ledger_ok or not budgets_ok:
            raise RuntimeError('Production photon ledger or matched source budget failed')
        all_ledgers = [r['ledger'] for r in pc['steps']] + [r['ledger'] for r in pc['checkpoints']]
        pre_completion = [r for r in rows if 0 < r['Q_mass_PC'] < 1-1e-8]
        summary = dict(scope=SCOPE, config=config, completed=True,
            all_ledgers_passed=ledger_ok, all_comparison_source_budgets_match=budgets_ok,
            new_convergence_test_performed=False, continuum_accuracy_certified=False,
            accepted_numerical_policy='Retained photons remain unabsorbed in the ledger; no strict 1e-8 retained or 1e-10 inter-run Q gate in this single production history',
            maximum_ledger_relative_residual=max(r['relative_residual'] for r in all_ledgers),
            maximum_source_L1_over_emitted=max(r['source_L1_over_emitted'] for r in rows),
            maximum_precompletion_retained_fraction=max((r['pc_ledger']['retained_fraction'] for r in pre_completion), default=0.),
            final_PC_ledger=pc['final_ledger'],
            final_Q_mass_PC=rows[-1]['Q_mass_PC'], final_Q_mass_ES=rows[-1]['Q_mass_ES'],
            checkpoints=len(rows), native_fixture=str(path), native_fixture_sha256=digest(path),
            raw_PC_summary=str(root/'pc/summary.json'), matched_ES_summary=str(root/'es/summary.json'),
            matched_ES_definition='Native sharp-k centre-cell barrier on the same accumulated rate-integrated photons and fixed capacity; maximum filter radius is half the box',
            native_ES_context='Independent native coeval proxy with evolving density; contextual only, not the matched control',
            scope_notes=['Gas capacity stays fixed at reference_z; native source density/displacements evolve.',
                         'No gas/ion advection, recombination, X-ray heating, spin-temperature evolution or UV feedback.',
                         'Sources start at z_start in neutral gas; no earlier emission is supplied.',
                         'PC and ES compare at the same redshift and emitted source budget, not at matched ionized fraction.',
                         'The run extends the accepted model and does not provide a new angular or spatial convergence certificate.'])
        _csv_outputs(root, rows)
        _plot(root, rows)
        write_json(root/'summary.json', summary)
        if code_hashes() != codes or digest(path) != execution['fixture_sha256']:
            raise RuntimeError('Executable code or fixture changed during production')
        execution['phases_seconds']['diagnostics'] = time.perf_counter()-phase_start
        execution.update(status='complete', phase='complete', this_invocation_seconds=time.perf_counter()-started)
        write_json(root/'execution.json', execution)
        artifacts = [root/name for name in ('summary.json','history.csv','power_spectra.csv',
                     'evolution_comparison.png','evolution_comparison.pdf')]
        artifacts += sorted((root/'diagnostics').glob('*.json'))
        _receipt(root, 'diagnostics', experiment_sha256, artifacts)
        _atomic_text(root/'COMPLETED', 'Numerical execution, photon ledgers and matched per-source budgets passed.\n')
        return summary
    except BaseException as exc:
        execution.update(status='failed', error=f'{type(exc).__name__}: {exc}',
                         this_invocation_seconds=time.perf_counter()-started)
        write_json(root/'execution.json', execution)
        raise


def run(*, root, config, threads=32, resume=False, fixture=None):
    config = validate_config(config)
    if type(threads) is not int or not 1 <= threads <= 32:
        raise ValueError('threads must be an integer from 1 to 32')
    if type(resume) is not bool:
        raise ValueError('resume must be boolean')
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with _lock(root):
        return _run_locked(root, config, threads, resume, fixture)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    command = subparsers.add_parser('run')
    command.add_argument('--root', type=Path, required=True)
    command.add_argument('--config', type=Path, required=True)
    command.add_argument('--threads', type=int, default=32)
    command.add_argument('--resume', action='store_true')
    command.add_argument('--fixture', type=Path)
    args = parser.parse_args()
    result = run(root=args.root, config=json.loads(args.config.read_text()),
                 threads=args.threads, resume=args.resume, fixture=args.fixture)
    print(json.dumps({k:result[k] for k in ('scope','completed','all_ledgers_passed',
                    'all_comparison_source_budgets_match','maximum_precompletion_retained_fraction')}), flush=True)


if __name__ == '__main__':
    main()
