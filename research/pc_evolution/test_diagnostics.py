import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnostics import checkpoint_fields, compare_histories, enrich_history, run_controls, field_spectrum
from evolution import run_history


def fixture():
    rates = np.zeros((5, 4, 4, 4))
    rates[:, 1, 1, 1] = [0, 3, 5, 3, 0]
    rates[:, 2, 2, 2] = [0, 0, 2, 5, 8]
    return dict(rates=rates, capacity=np.ones((4,)*3),
                times_seconds=np.arange(5.), redshifts=np.arange(12, 7, -1.),
                native_xhi=np.full_like(rates, .5),
                metadata=dict(box_len_cMpc=200., hlittle=.678))


def test_control_uses_accumulated_rate_budget_once(tmp_path):
    h = fixture()
    report = run_controls(h, tmp_path/'es')
    final = checkpoint_fields(report['checkpoints'][-1])
    expected = sum(.5*(h['rates'][i]+h['rates'][i+1]) for i in range(4))
    np.testing.assert_array_equal(final['source_emitted'], expected)
    assert report['checkpoints'][0]['Q_mass'] == 0
    assert report['native_es_context'][-1]['xHI_volume'] == .5
    assert final['ionized'].min() >= 0 and final['ionized'].max() <= 1
    report_path = tmp_path/'es'/'summary.json'
    assert json.loads(report_path.read_text())['model'] == 'matched_es'


def test_temporal_comparison_resolves_fine_state(tmp_path):
    h = fixture()
    a = enrich_history(run_history(h, subcells_per_axis=4, angles=48,
                                  stride=1, output=tmp_path/'a'), h)
    b = enrich_history(run_history(h, subcells_per_axis=4, angles=48,
                                  stride=4, output=tmp_path/'b'), h)
    rows = compare_histories(a, b, h)
    final = rows[-1]
    assert final['source_budgets_match']
    assert final['resolved_subcell_L1_over_emitted'] > 0
    assert final['absolute_Q_mass_change'] < 1e-12
    assert final['absolute_k0_power_change'] is not None
    assert final['resolved_subcell_L1_over_emitted'] >= final['original_grid_field_L1_over_emitted']-1e-12


def test_saturation_has_no_fractional_hi_spectrum():
    h = fixture()
    assert field_spectrum(h, h['capacity'], 8) is None


def test_checkpoint_modification_rejected(tmp_path):
    report = run_controls(fixture(), tmp_path/'es')
    row = report['checkpoints'][-1]
    with open(row['path'], 'ab') as stream:
        stream.write(b'changed')
    with pytest.raises(RuntimeError, match='Checkpoint changed'):
        checkpoint_fields(row)


def test_comparison_rejects_different_times(tmp_path):
    h = fixture()
    a = enrich_history(run_history(h, angles=48, output=tmp_path/'a'), h)
    b = enrich_history(run_history(h, angles=48, output=tmp_path/'b'), h)
    b['checkpoints'][-1]['time_seconds'] += 1
    with pytest.raises(ValueError, match='different physical times'):
        compare_histories(a, b, h)
