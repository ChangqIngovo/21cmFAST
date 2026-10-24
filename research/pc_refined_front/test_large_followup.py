"""Driver contracts: fixed angular banks, immutable parent fields and common mesh."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import large_followup as driver


def make_parent(tmp_path):
    parent = tmp_path/'parent'
    (parent/'results').mkdir(parents=True)
    (parent/'COMPLETED').write_text('complete')
    (parent/'run_manifest.json').write_text(json.dumps({'experiment': 'large-scale'}))
    c = json.loads((driver.RESEARCH/'pc_large_scale/default.json').read_text())
    c.update(seeds=[1234], grids=[8, 4], native_grid=8, min_angles=24, max_angles=24,
             box_hMpc=100., k0_band_hMpc_inv=[.05, .09])
    (parent/'results/experiment.json').write_text(json.dumps({'config': c}))
    x = np.arange(8)[:, None, None]
    cap = np.broadcast_to(1+.1*np.cos(2*np.pi*x/8), (8,)*3).copy()
    fixture = parent/'results/fixtures/seed1234'
    names = []
    for grid in c['grids']:
        folder = fixture/f'n{grid}'
        folder.mkdir(parents=True)
        field = driver.coarsen_counts(cap, grid)
        for name in ('source.npy', 'capacity.npy'):
            np.save(folder/name, field)
            names.append(f'n{grid}/{name}')
    driver.finish(fixture, names)
    return parent, c


def test_default_matrix_and_explicit_refinement():
    c = json.loads((driver.RESEARCH/'pc_large_scale/default.json').read_text())
    selected = driver.selection(c)
    plan = driver.task_list(c, selected)
    assert selected['grids'] == [25, 50]
    assert selected['refinements'] == [2, 4]
    assert selected['distance_factor'] == 8
    assert selected['contiguous_ray_bytes'] == 4*1024**3
    assert len(plan) == 48
    assert len({p['directory'] for p in plan}) == 48
    assert max(p['grid'] for p in plan) == 50
    assert sum(p['model'] == 'adaptive' for p in plan) == 16
    assert len(driver.task_list(c, driver.selection(c, [25, 50, 100], refinements=[2, 4, 8]))) == 96
    with pytest.raises(ValueError, match='include 2'):
        driver.selection(c, refinements=[4])
    with pytest.raises(ValueError, match='unique parent grids'):
        driver.selection(c, [25, 25])
    with pytest.raises(ValueError, match='GiB threshold'):
        driver.selection(c, contiguous_ray_gb=-1)
    assert driver.selection(c, contiguous_ray_gb=0)['contiguous_ray_bytes'] == 0


def test_prepare_freezes_parent_and_angular_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, 'code_hashes', lambda: {'test': 'fixed'})
    parent, _ = make_parent(tmp_path)
    root = tmp_path/'new'
    identity = driver.prepare(root, parent, grids=[4, 8])
    assert len(identity['tasks']) == 12
    assert len(identity['prepared_angle_sha256']) == 4
    assert identity['selection']['measurement_grid'] == 4
    driver.verify(root)
    bank = root/next(iter(identity['prepared_angle_sha256']))
    original = bank.read_bytes()
    a = np.load(bank)
    a.flat[0] *= 2
    np.save(bank, a)
    with pytest.raises(RuntimeError, match='angular bank changed'):
        driver.verify(root)
    bank.write_bytes(original)
    p = parent/'results/fixtures/seed1234/n4/source.npy'
    source = np.load(p)
    source.flat[0] += .1
    np.save(p, source)
    with pytest.raises(RuntimeError, match='Parent input changed'):
        driver.verify(root)


def test_collection_pairs_same_banks_and_common_measurement_mesh(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, 'code_hashes', lambda: {'test': 'fixed'})
    parent, _ = make_parent(tmp_path)
    root = tmp_path/'new'
    identity = driver.prepare(root, parent, grids=[4, 8], doses=[.8])
    calls = []
    def solve(photons, capacity, **options):
        calls.append(options)
        return dict(absorbed=photons.copy(), source_absorbed=photons.copy(),
                    source_retained=np.zeros_like(photons), source_escaped=np.zeros_like(photons),
                    angle_counts=np.full(photons.shape, 24), audit={'test_double': True})
    module = SimpleNamespace(library=lambda: None, solve=solve)
    monkeypatch.setitem(sys.modules, 'adaptive', module)
    monkeypatch.setitem(sys.modules, 'refined', module)
    for task in identity['tasks']:
        driver.run_task(root, task['id'])
    summary = driver.collect(root)
    assert summary['all_ledgers_passed']
    assert summary['all_f2_implementation_checks_passed']
    assert len(summary['implementation_checks']) == 2
    assert len(summary['front_refinement']) == 2
    assert len(summary['input_grid_comparisons']) == 3
    assert all(r['measurement_grid'] == 4 for r in summary['case_results'])
    lookup = {r['task']['directory']: r for r in summary['case_results']}
    for report in summary['case_results']:
        timing = report['timing']
        assert timing['raw_solve_seconds'] == report['seconds']
        if report['task']['model'] == 'refined':
            predictor = timing['separately_measured_angle_estimate_seconds']
            assert predictor > 0
            assert timing['end_to_end_seconds_with_angle_estimate'] == pytest.approx(report['seconds']+predictor)
        else:
            assert timing['separately_measured_angle_estimate_seconds'] is None
            assert timing['end_to_end_seconds_with_angle_estimate'] == report['seconds']
    for comparison in summary['implementation_checks']+summary['front_refinement']:
        new = lookup[comparison['new']['directory']]['timing']
        old = lookup[comparison['reference']['directory']]['timing']
        assert comparison['runtime_ratio'] == pytest.approx(
            new['end_to_end_seconds_with_angle_estimate']/old['end_to_end_seconds_with_angle_estimate'])
        assert 'one angular predictor' in comparison['runtime_ratio_scope']
    for call in calls:
        if 'subcells_per_axis' in call:
            assert call['subcells_per_axis'] in (2, 4)
            assert call['contiguous_ray_bytes'] == 4*1024**3
            assert np.all(call['angle_counts'] == 24)
            assert call['maximum_distance'] in (4*8/3, 8*8/3)
    bad = identity['tasks'][1]
    folder = root/'cases'/bad['directory']
    report = json.loads((folder/'result.json').read_text())
    report['angular_bank_sha256'] = 'changed'
    driver.write_json(folder/'result.json', report)
    driver.finish(folder, ['result.json', 'fields.npz'])
    with pytest.raises(RuntimeError, match='Angular bank changed'):
        driver.collect(root)


def test_rejects_nested_or_nonempty_output_and_bad_task_id(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, 'code_hashes', lambda: {'test': 'fixed'})
    parent, _ = make_parent(tmp_path)
    with pytest.raises(ValueError, match='outside'):
        driver.prepare(parent/'followup', parent, grids=[4])
    root = tmp_path/'new'
    driver.prepare(root, parent, grids=[4])
    with pytest.raises(ValueError, match='fresh empty'):
        driver.prepare(root, parent, grids=[4])
    monkeypatch.setitem(sys.modules, 'refined', SimpleNamespace())
    with pytest.raises(ValueError, match='Task ID'):
        driver.run_task(root, -1)
