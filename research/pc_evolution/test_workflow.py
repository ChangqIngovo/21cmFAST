"""Validation gates, isolated execution and provenance checks for evolution."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import workflow


def good_row():
    return dict(node_index=8, new_Q_mass=.8, reference_Q_mass=.8,
                absolute_k0_power_change=.005,
                original_grid_field_L1_over_emitted=.004,
                resolved_subcell_L1_over_emitted=.009,
                absolute_Q_mass_change=0., maximum_retained_fraction=0.,
                source_budgets_match=True, source_L1_over_emitted=0.)


def selection():
    return dict(threads=32,dimension=32,box_len=200.,seeds=[1234,2345],
                angles=[384,768],base_intervals=16,z_start=25.,z_end=6.,reference_z=8.)


def test_gate_retains_fine_cell_failure_despite_small_power_change():
    row = good_row()
    assert workflow.temporal_gate(row)['passed']
    row['resolved_subcell_L1_over_emitted'] = .02
    check = workflow.temporal_gate(row)
    assert check['eligible'] and not check['passed']
    assert check['failed_metrics'] == ['resolved_subcell_L1_over_emitted']


def test_gate_does_not_pass_missing_power_or_resolution_information():
    for key in ('absolute_k0_power_change','resolved_subcell_L1_over_emitted'):
        row = good_row()
        row[key] = None
        result = workflow.temporal_gate(row)
        assert result['eligible'] and not result['passed']
        assert key in result['failed_metrics']


@pytest.mark.parametrize('new,reference', [(0.,0.),(.8,.99),(1.,1.),(.09,.11)])
def test_gate_excludes_epochs_without_disguising_them_as_passes(new,reference):
    row = good_row()
    row.update(new_Q_mass=new,reference_Q_mass=reference)
    check = workflow.temporal_gate(row)
    assert not check['eligible'] and check['passed'] is None


def test_gate_reports_tiny_count_drift_separately():
    row = good_row()
    row['absolute_Q_mass_change'] = 2e-9
    check = workflow.temporal_gate(row)
    assert check['failed_metrics'] == ['absolute_Q_mass_change']


def test_gate_rejects_source_changes_hidden_by_equal_global_photon_count():
    row=good_row()
    row.update(source_budgets_match=False,source_L1_over_emitted=.02)
    assert workflow.temporal_gate(row)['failed_metrics']==['source_budgets_match']


def test_source_quadrature_cannot_hide_behind_matched_transport_budgets():
    checkpoint=dict(node_index=8,redshift=6.,Q_mass=.8,cumulative_L1_over_emitted=.0329)
    row=dict(seed=1234,stride=2,summed_interval_L1_over_emitted=.0329,
             checkpoints=[checkpoint])
    assert not workflow.source_quadrature_gate(row)['passed']
    checkpoint['cumulative_L1_over_emitted']=.002
    assert workflow.source_quadrature_gate(row)['passed']
    row['stride']=1
    assert not workflow.source_quadrature_gate(row)['passed']


def test_source_quadrature_late_rate_ramp_does_not_dilute_early_pulse_error():
    # An early burst is poorly integrated. Large later emission reduces the
    # final relative discrepancy below 1%, but cannot validate that early epoch.
    row=dict(seed=1234,stride=2,summed_interval_L1_over_emitted=.005,
             checkpoints=[
                 dict(node_index=4,redshift=9.,Q_mass=.2,cumulative_photon_dose=.2,
                      cumulative_L1_over_emitted=.03,signed_cumulative_difference_over_emitted=0.),
                 dict(node_index=8,redshift=6.,Q_mass=.9,cumulative_photon_dose=.9,
                      cumulative_L1_over_emitted=.005,signed_cumulative_difference_over_emitted=0.)])
    gate=workflow.source_quadrature_gate(row)
    assert gate['eligible_epochs']==2 and not gate['passed']
    assert [check['passed'] for check in gate['checkpoint_checks']]==[False,True]


def test_source_quadrature_requires_a_resolved_eligible_epoch():
    row=dict(stride=2,summed_interval_L1_over_emitted=0.,checkpoints=[])
    assert not workflow.source_quadrature_gate(row)['passed']
    row['checkpoints']=[dict(Q_mass=.05,cumulative_L1_over_emitted=0.)]
    assert not workflow.source_quadrature_gate(row)['passed']
    row['checkpoints']=[dict(Q_mass=.5,cumulative_L1_over_emitted=-.001)]
    assert not workflow.source_quadrature_gate(row)['passed']


def test_matrix_keeps_angles_fixed_across_all_time_meshes():
    fixtures = [dict(seed=1234,path='/fixture.npz',sha256='abc')]
    rows = workflow.task_list(fixtures,[384,768])
    assert len(rows) == 13
    assert sum(r['model']=='matched_es' for r in rows)==1
    for f in (2,4):
        for angles in (384,768):
            selected = [r for r in rows if r.get('subcells_per_axis')==f and r.get('angles')==angles]
            assert {r['stride'] for r in selected} == {1,2,4}
            assert {r['fixture_sha256'] for r in selected} == {'abc'}
    assert len({r['directory'] for r in rows}) == 13


@pytest.mark.parametrize('key,value', [
    ('threads',0),('threads',33),('angles',[384]),('angles',[384,385]),
    ('angles',[384,384]),('seeds',[1234,1234]),('z_end',26.),('base_intervals',0)])
def test_invalid_matrix_rejected_before_execution(key,value):
    options=selection(); options[key]=value
    with pytest.raises(ValueError):
        workflow.validate_selection(**options)


def test_worker_environment_prevents_native_and_process_oversubscription(monkeypatch):
    monkeypatch.setenv('OMP_NUM_THREADS','32')
    monkeypatch.setenv('OPENBLAS_NUM_THREADS','32')
    assert workflow.worker_environment()['OMP_NUM_THREADS']=='1'
    assert workflow.worker_environment()['OPENBLAS_NUM_THREADS']=='1'
    assert workflow.worker_environment(32)['OMP_NUM_THREADS']=='32'
    assert workflow.worker_environment(32)['OPENBLAS_NUM_THREADS']=='1'


def test_worker_receipt_detects_changed_checkpoint(tmp_path):
    fixture=tmp_path/'fixture.npz'; fixture.write_bytes(b'fixture')
    directory=tmp_path/'cases'/'case'; directory.mkdir(parents=True)
    checkpoint=directory/'node.npz'; checkpoint.write_bytes(b'original field')
    workflow.write_json(directory/'summary.json',dict(all_ledgers_passed=True))
    code={'solver.cpp':'codehash'}
    task=dict(directory='case',fixture=str(fixture),fixture_sha256=workflow.digest(fixture))
    receipt=dict(fixture_sha256=workflow.digest(fixture),code_sha256=code,
                 summary_sha256=workflow.digest(directory/'summary.json'),
                 checkpoint_sha256={'node.npz':workflow.digest(checkpoint)})
    workflow.write_json(directory/'receipt.json',receipt)
    assert workflow.read_checked_report(tmp_path,task,code)['all_ledgers_passed']
    checkpoint.write_bytes(b'altered field')
    with pytest.raises(RuntimeError,match='Checkpoint changed'):
        workflow.read_checked_report(tmp_path,task,code)


def test_existing_result_directory_is_not_overwritten(tmp_path):
    (tmp_path/'summary.json').write_text('old result')
    with pytest.raises(FileExistsError):
        workflow.run(root=tmp_path)
    assert (tmp_path/'summary.json').read_text()=='old result'


def test_json_does_not_emit_nonstandard_nan(tmp_path):
    with pytest.raises(ValueError):
        workflow.write_json(tmp_path/'bad.json',dict(value=float('nan')))
    assert not (tmp_path/'bad.json').exists()
