"""Independent controls for fixed-source angular convergence experiments."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import angular_followup as driver
import large_followup as spatial
from test_large_followup import make_parent


def test_scaled_bank_has_exact_counts_without_old_cap():
    bank = np.array([24, 48, 12288], dtype=np.int32).reshape(1, 1, 3)
    for multiplier in (1, 2, 4):
        result = driver.scaled_angles(bank, multiplier)
        assert result.dtype == np.int32
        assert result.flags.c_contiguous
        np.testing.assert_array_equal(result, bank.astype(np.int64)*multiplier)
    assert driver.scaled_angles(bank, 4).max() == 49152
    np.testing.assert_array_equal(bank, np.array([24, 48, 12288]).reshape(1, 1, 3))


@pytest.mark.parametrize('multiplier', [0, -1, True, 1.5, np.nan, np.inf])
def test_invalid_multiplier_is_rejected(multiplier):
    with pytest.raises((ValueError, TypeError)):
        driver.scaled_angles(np.full((2, 2, 2), 24, dtype=np.int32), multiplier)


@pytest.mark.parametrize('bank', [
    np.array([0, 24]), np.array([23, 24]), np.array([25, 48]),
    np.array([24., 48.]), np.array([True, False]),
    np.array([24, np.iinfo(np.int32).max], dtype=np.int64),
])
def test_invalid_bank_is_rejected(bank):
    with pytest.raises((ValueError, TypeError)):
        driver.scaled_angles(bank, 1)


def test_integer_overflow_is_rejected_before_conversion():
    supported = (np.iinfo(np.int32).max//24)*24
    bank = np.array([supported], dtype=np.int64)
    assert int(driver.scaled_angles(bank, 1)[0]) == supported
    with pytest.raises((ValueError, OverflowError)):
        driver.scaled_angles(bank, 2)
    with pytest.raises((ValueError, OverflowError)):
        driver.scaled_angles(np.array([24], dtype=np.int64), 2**63)


def test_combined_direction_offsets_must_fit_int32():
    supported = (np.iinfo(np.int32).max//24)*24
    # Individual counts fit; their concatenated direction-bank offsets do not.
    with pytest.raises((ValueError, OverflowError)):
        driver.scaled_angles(np.array([24, supported], dtype=np.int64), 1)


def test_field_metric_exposes_cancellation_under_coarsening():
    old = np.ones((4,)*3)
    perturbation = np.zeros_like(old)
    perturbation[0, 0, 0] = .25
    perturbation[1, 0, 0] = -.25
    new = old+perturbation
    original = driver.field_metrics(new, old, 64.)
    common = driver.field_metrics(spatial.coarsen_counts(new, 2),
                                  spatial.coarsen_counts(old, 2), 64.)
    assert original['field_L1_over_emitted'] == pytest.approx(.5/64.)
    assert common['field_L1_over_emitted'] == 0.


def test_fourier_comparison_matches_independent_full_fft():
    n, box = 8, 100.
    xyz = np.indices((n,)*3)
    x, y, z = 2*np.pi*xyz/n
    cap = 1+.1*np.cos(x)+.05*np.sin(z)
    old = .45*cap+.07*np.cos(2*y)+.03*np.sin(x+z)
    new = .45*cap+.05*np.cos(2*y+.35)-.02*np.sin(x+z)
    edges = np.array([.01, .08, .14, .3])
    band = [.05, .18]
    output = driver.compare_fields(cap, new, cap, old, box, edges, band)
    fn = np.fft.fftn((cap-new)/np.mean(cap-new)-1)
    fo = np.fft.fftn((cap-old)/np.mean(cap-old)-1)
    k = 2*np.pi*np.fft.fftfreq(n, d=box/n)
    km = np.sqrt(k[:, None, None]**2+k[None, :, None]**2+k[None, None, :]**2)
    mask = (km >= band[0]) & (km < band[1]) & (km <= np.pi*n/box*(1+1e-12))
    norm = box**3/n**6
    pn = np.mean(abs(fn[mask])**2)*norm
    po = np.mean(abs(fo[mask])**2)*norm
    pc = np.mean((fn[mask]*fo[mask].conj()).real)*norm
    pd = np.mean(abs(fn[mask]-fo[mask])**2)*norm
    row = output['k0_band']
    assert row['full_fft_modes'] == np.count_nonzero(mask)
    assert row['P_new'] == pytest.approx(pn, rel=2e-13)
    assert row['P_reference'] == pytest.approx(po, rel=2e-13)
    assert row['P_cross'] == pytest.approx(pc, rel=2e-13)
    assert row['r_new_reference'] == pytest.approx(pc/np.sqrt(pn*po), rel=2e-13)
    assert row['P_difference'] == pytest.approx(pd, rel=2e-13)
    assert row['relative_difference_power'] == pytest.approx(pd/po, rel=2e-13)
    assert row['k_over_nyquist'] == pytest.approx(np.mean(km[mask])/(np.pi*n/box))


def test_equal_power_different_phase_is_detected():
    n = 8
    x = 2*np.pi*np.arange(n)[:, None, None]/n
    cap = np.ones((n,)*3)
    old = .5+.1*np.broadcast_to(np.cos(x), cap.shape)
    new = .5+.1*np.broadcast_to(np.sin(x), cap.shape)
    result = driver.compare_fields(cap, new, cap, old, 100., [.01, .2, .3], [.05, .08])['k0_band']
    assert result['P_new'] == pytest.approx(result['P_reference'])
    assert result['r_new_reference'] == pytest.approx(0., abs=1e-14)
    assert result['relative_difference_power'] == pytest.approx(2.)


def refined_parent(tmp_path, grids=(4, 8), doses=(.8,)):
    """Create native fixture and the completed previous spatial experiment."""
    native, _ = make_parent(tmp_path)
    fixture = native/'results/fixtures/seed1234'
    finecap = np.load(fixture/'n8/capacity.npy')
    xyz = np.indices(finecap.shape)
    # Nonlocal transfer makes the angle test more than a weak-source identity.
    finesource = finecap*np.exp(.8*np.cos(2*np.pi*xyz[0]/8)+.45*np.sin(2*np.pi*xyz[1]/8))
    names = []
    for n in (4, 8):
        np.save(fixture/f'n{n}/source.npy', spatial.coarsen_counts(finesource, n))
        names += [f'n{n}/source.npy', f'n{n}/capacity.npy']
    spatial.finish(fixture, names)
    parent = tmp_path/'refined_parent'
    parent.mkdir()
    (parent/'run_manifest.json').write_text(json.dumps({'experiment': 'refined-front-followup'}))
    identity = spatial.prepare(parent/'results', native, grids=list(grids), doses=list(doses))
    for spec in identity['tasks']:
        spatial.run_task(parent/'results', spec['id'])
    spatial.collect(parent/'results')
    (parent/'COMPLETED').write_text('completed')
    (parent/'results/COMPLETED').write_text('completed')
    return parent


def test_default_matrix_and_explicit_short_sweep():
    config = json.loads((spatial.RESEARCH/'pc_large_scale/default.json').read_text())
    selected = spatial.selection(config)
    parent = dict(config=config, selection=selected, tasks=spatial.task_list(config, selected))
    defaults = driver.selection(parent)
    assert defaults['multipliers'] == [1, 2, 4, 8, 16]
    plan = driver.task_list(parent, defaults)
    assert len(plan) == 160
    assert len({t['directory'] for t in plan}) == 160
    assert {t['model'] for t in plan} == {'refined'}
    assert defaults['measurement_grid'] == 25
    assert defaults['distance_factor'] == 8.
    short = driver.selection(parent, multipliers=[1, 2, 4])
    assert len(driver.task_list(parent, short)) == 96
    for invalid in ([2, 4], [1, 1, 2], [1, True], [1], [1, 2.5]):
        with pytest.raises(ValueError, match='multipliers'):
            driver.selection(parent, multipliers=invalid)
    parent['tasks'].pop()
    with pytest.raises(RuntimeError, match='full requested'):
        driver.task_list(parent, defaults)


def test_prepare_freezes_bank_parent_fields_and_physics(tmp_path):
    parent = refined_parent(tmp_path, grids=(4,))
    identity_path = parent/'results/experiment.json'
    legacy = json.loads(identity_path.read_text())
    # A real previous run predates the validation driver and its tests.
    for name in list(legacy['code_sha256']):
        if 'angular_followup' in name or 'gadi_angular_' in name:
            del legacy['code_sha256'][name]
    identity_path.write_text(json.dumps(legacy))
    root = tmp_path/'angular'
    identity = driver.prepare(root, parent, multipliers=[1, 2, 4])
    assert len(identity['tasks']) == 6
    assert len(identity['angular_preflight']) == 3
    assert identity['selection']['distance_factor'] == 8.
    assert identity['selection']['measurement_grid'] == 4
    for spec in identity['tasks']:
        old = np.load(parent/'results/angles'/f"{spec['base_angle_key']}.npy")
        new = np.load(root/'angles'/f"{spec['angle_key']}.npy")
        np.testing.assert_array_equal(new, old*spec['multiplier'])
    driver.verify(root)
    bank = root/next(iter(identity['prepared_angle_sha256']))
    original = bank.read_bytes()
    altered = np.load(bank)
    altered.flat[0] *= 2
    np.save(bank, altered)
    with pytest.raises(RuntimeError, match='angular bank changed'):
        driver.verify(root)
    bank.write_bytes(original)
    prior_case = parent/'results/cases'/identity['tasks'][0]['parent_directory']/'fields.npz'
    original = prior_case.read_bytes()
    prior_case.write_bytes(original+b'changed')
    with pytest.raises(RuntimeError, match='Parent input changed'):
        driver.verify(root)
    prior_case.write_bytes(original)
    prior = json.loads(identity_path.read_text())
    prior['code_sha256']['pc_refined_front/refined.cpp'] = 'different physics'
    identity_path.write_text(json.dumps(prior))
    with pytest.raises(RuntimeError, match='physics differs'):
        driver.prepare(tmp_path/'different_physics', parent)


def test_preflight_uses_combined_direction_ids_for_wide_storage(tmp_path):
    parent = refined_parent(tmp_path, grids=(4,))
    identity_path = parent/'results/experiment.json'
    prior = json.loads(identity_path.read_text())
    key = prior['tasks'][0]['angle_key']
    path = parent/'results/angles'/f'{key}.npy'
    bank = np.full((4,)*3, 32760, dtype=np.int32)
    bank.flat[0] = 32784
    # This is a preflight-only metadata fixture: no high-angle solve is run.
    np.save(path, bank)
    bank_digest = spatial.digest(path)
    prior['prepared_angle_sha256'][f'angles/{key}.npy'] = bank_digest
    identity_path.write_text(json.dumps(prior))
    for spec in prior['tasks']:
        if spec['model'] != 'refined':
            continue
        folder = parent/'results/cases'/spec['directory']
        report = json.loads((folder/'result.json').read_text())
        report['angular_bank_sha256'] = bank_digest
        spatial.write_json(folder/'result.json', report)
        spatial.finish(folder, ['result.json', 'fields.npz'])
    prepared = driver.prepare(tmp_path/'angular', parent, refinements=[2], multipliers=[1, 2])
    first = next(row for row in prepared['angular_preflight'] if row['multiplier'] == 1)
    assert first['max_angles'] < 65535
    assert first['estimated_transport_ray_bytes'] == 64
    assert first['potential_packed_ray_bytes'] == 64*int(bank.sum(dtype=np.int64))


def test_real_solves_reproduce_parent_and_report_unconverged_fields(tmp_path):
    parent = refined_parent(tmp_path)
    root = tmp_path/'angular'
    identity = driver.prepare(root, parent, multipliers=[1, 2, 4],
                              power_tolerance=1e-12, field_tolerance=1e-12)
    assert len(identity['tasks']) == 12
    for spec in identity['tasks']:
        driver.run_task(root, spec['id'])
    summary = driver.collect(root)
    assert summary['cases'] == 12
    assert summary['all_ledgers_passed']
    assert summary['all_parent_baselines_passed']
    assert len(summary['parent_baselines']) == 4
    assert len(summary['angular_comparisons']) == 12
    assert len(summary['front_refinement']) == 6
    assert len(summary['input_grid_comparisons']) == 6
    assert len(summary['convergence_checks']) == 8
    assert summary['diagnostic_convergence_has_enough_levels']
    assert not summary['diagnostic_convergence_passed']
    assert not summary['continuum_accuracy_certified']
    assert all(r['original_grid']['field_L1_over_emitted'] < 1e-14
               for r in summary['parent_baselines'])
    assert all(max(r['source_ledgers'].values()) < 1e-14
               for r in summary['parent_baselines'])
    assert all(r['resolved_subcells'] is None for r in summary['parent_baselines'])
    assert {c['reference_multiplier'] for c in summary['convergence_checks']} == {1, 2}
    assert all(c['resolved_subcell_L1_over_emitted'] is not None
               for c in summary['convergence_checks'])
    for row in summary['angular_comparisons']:
        assert row['original_grid']['field_L1_over_emitted']+1e-16 >= row['field_L1_over_emitted']
        assert row['resolved_subcells']['subcell_L1_over_emitted']+1e-16 >= row['original_grid']['field_L1_over_emitted']
        assert row['same_original_sources']
    assert all(r['resolved_subcells'] is None for r in summary['front_refinement'])
    assert all(r['resolved_subcells'] is None for r in summary['input_grid_comparisons'])
    # Choose a tolerance that every parent-averaged field passes but at least
    # one resolved state fails; the diagnostic must retain that failure.
    largest_original = max(c['original_grid_field_L1_over_emitted'] for c in summary['convergence_checks'])
    largest_resolved = max(c['resolved_subcell_L1_over_emitted'] for c in summary['convergence_checks'])
    assert largest_resolved > largest_original
    identity['selection']['field_tolerance'] = (largest_original+largest_resolved)/2
    identity['selection']['power_tolerance'] = 10.
    spatial.write_json(root/'experiment.json', identity)
    resolved_gate = driver.collect(root)
    assert all(c['original_grid_field_L1_over_emitted'] < identity['selection']['field_tolerance']
               for c in resolved_gate['convergence_checks'])
    assert not resolved_gate['diagnostic_convergence_passed']
    # A source-by-source mismatch must fail reproduction even if a report's
    # global scalar ledger still says passed and its output receipt is valid.
    spec = next(t for t in identity['tasks'] if t['multiplier'] == 1)
    folder = root/'cases'/spec['directory']
    with np.load(folder/'fields.npz') as data:
        fields = {name: data[name] for name in data.files}
    fields['source_absorbed'].flat[0] += .1
    fields['source_absorbed'].flat[1] -= .1
    np.savez_compressed(folder/'fields.npz', **fields)
    spatial.finish(folder, ['result.json', 'fields.npz'])
    with pytest.raises(RuntimeError, match='parent reproduction'):
        driver.collect(root)
    failed = json.loads((root/'summary.json').read_text())
    assert not failed['all_parent_baselines_passed']
    assert not (root/'COMPLETED').exists()


def test_missing_case_and_failed_ledger_are_not_completed(tmp_path):
    parent = refined_parent(tmp_path, grids=(4,))
    root = tmp_path/'angular'
    identity = driver.prepare(root, parent, refinements=[2], multipliers=[1, 2])
    with pytest.raises(RuntimeError, match='Missing completed angular case'):
        driver.collect(root)
    for spec in identity['tasks']:
        driver.run_task(root, spec['id'])
    summary = driver.collect(root)
    assert summary['all_ledgers_passed']
    assert not summary['diagnostic_convergence_has_enough_levels']
    assert not summary['diagnostic_convergence_passed']
    folder = root/'cases'/identity['tasks'][0]['directory']
    report = json.loads((folder/'result.json').read_text())
    report['ledger']['passed'] = False
    spatial.write_json(folder/'result.json', report)
    spatial.finish(folder, ['result.json', 'fields.npz'])
    with pytest.raises(RuntimeError, match='numerical ledger'):
        driver.collect(root)
    assert not (root/'COMPLETED').exists()


def test_failed_worker_records_failure_without_completion(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, 'verify', lambda root: {'tasks': [{'id': 0, 'directory': 'case0'}]})
    def failed_worker(*args, **kwargs):
        raise RuntimeError('worker failed before making a numerical result')
    monkeypatch.setattr(driver.subprocess, 'run', failed_worker)
    with pytest.raises(RuntimeError, match='1 cases failed'):
        driver.run(tmp_path, workers=1)
    execution = json.loads((tmp_path/'execution.json').read_text())
    assert len(execution['failures']) == 1
    assert 'worker failed' in execution['failures'][0]['error']
    assert not (tmp_path/'COMPLETED').exists()


def test_completion_does_not_claim_diagnostic_convergence(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, 'verify', lambda root: {'tasks': [{'id': 0, 'directory': 'case0'}]})
    monkeypatch.setattr(driver.subprocess, 'run', lambda *args, **kwargs: None)
    monkeypatch.setattr(driver, 'collect', lambda root: {'diagnostic_convergence_passed': False})
    driver.run(tmp_path, workers=1)
    execution = json.loads((tmp_path/'execution.json').read_text())
    assert not execution['diagnostic_convergence_passed']
    assert not execution['failures']
    assert (tmp_path/'COMPLETED').is_file()
    assert 'separately' in (tmp_path/'COMPLETED').read_text()
