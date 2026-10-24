"""Contracts for reuse of fixed-budget parent data and controlled horizons."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from late_followup import tasks, parent_case, prepare, verify, run_task, collect
from run_comparison import finish
from pc_statistics import spectrum, coarsen_counts


def test_followup_contains_all_late_pairs_and_exactly_one_angle_control():
    config = json.loads((Path(__file__).parent/'default.json').read_text())
    plan = tasks(config)
    assert len(plan) == 120
    assert len({v['directory'] for v in plan}) == 120
    assert sum(t['angle_factor'] == 2 for t in plan) == 24
    assert all(t['distance_factor'] == 8 for t in plan if t['angle_factor'] == 2)
    config['photon_doses'] = [.8]
    with pytest.raises(ValueError, match='0.9'):
        tasks(config)


def make_parent(tmp_path, grids=(8,)):
    parent = tmp_path/'parent'
    (parent/'results').mkdir(parents=True)
    (parent/'COMPLETED').write_text('complete')
    (parent/'run_manifest.json').write_text(json.dumps({'experiment':'large-scale'}))
    config = json.loads((Path(__file__).parent/'default.json').read_text())
    config.update(seeds=[1234], grids=list(grids), native_grid=8, min_angles=24, max_angles=24,
                  box_hMpc=100., k0_band_hMpc_inv=[.05,.09])
    (parent/'results/experiment.json').write_text(json.dumps({'config':config}))
    x = np.arange(8)[:,None,None]
    cap = np.broadcast_to(1+.1*np.cos(2*np.pi*x/8), (8,)*3).copy()
    source = cap.copy()
    fixture = parent/'results/fixtures/seed1234'
    for grid in grids:
        (fixture/f'n{grid}').mkdir(parents=True)
        c = coarsen_counts(cap,grid)
        np.save(fixture/f'n{grid}/source.npy',c)
        np.save(fixture/f'n{grid}/capacity.npy',c)
        for dose in (.8,.9):
            folder = parent_case(parent,dict(seed=1234,grid=grid,dose=dose))
            folder.mkdir(parents=True)
            stat = spectrum(c,dose*c,100.,np.linspace(.001,.3,12),[.05,.09],8.)
            # Deliberately different mass-matched map catches accidental field choice.
            np.savez(folder/'fields.npz',pc_same_budget_absorbed=dose*c,pc_absorbed=.5*c)
            (folder/'result.json').write_text(json.dumps({'spectra':{'pc_same_budget':stat}}))
            finish(folder,['fields.npz','result.json'])
    finish(fixture,[f'n{grid}/{name}' for grid in grids for name in ('source.npy','capacity.npy')])
    return parent


def test_parent_data_is_read_only_and_original_budget_field_is_used(tmp_path):
    parent = make_parent(tmp_path)
    output = tmp_path/'new'
    identity = prepare(output,parent)
    assert len(identity['tasks']) == 10
    for task in identity['tasks']:
        run_task(output,task['id'])
    summary = collect(output)
    assert summary['all_ledgers_passed']
    assert summary['all_baselines_reproduced']
    assert all(c['all_requested_tolerances_passed'] for c in summary['comparisons'])
    assert len(summary['comparisons']) == 8
    assert all(c['same_angle_counts'] for c in summary['comparisons'] if c['comparison']=='distance')
    assert max(r['versus_original_same_budget']['L1_over_emitted'] for r in summary['case_results']) < 1e-13
    verify(output)
    p=parent/'results/fixtures/seed1234/n8/source.npy'
    source=np.load(p);source.flat[0]+=.1;np.save(p,source)
    with pytest.raises(RuntimeError,match='Parent input changed'):
        verify(output)


def test_rejects_output_inside_parent(tmp_path):
    parent=make_parent(tmp_path)
    with pytest.raises(ValueError,match='outside'):
        prepare(parent/'followup',parent)
