from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adaptive import solve
from distance_control import solve_distance_converged


def test_separate_angle_reference_keeps_rays_fixed_as_horizon_changes():
    c = np.ones((12,)*3)
    s = np.zeros_like(c)
    s[6, 6, 6] = 500
    a = solve(s, c, maximum_distance=1., angular_reference_distance=2.)
    b = solve(s, c, maximum_distance=8., angular_reference_distance=2.)
    np.testing.assert_array_equal(a['angle_counts'], b['angle_counts'])
    np.testing.assert_array_equal(a['estimated_radius'], b['estimated_radius'])
    assert b['absorbed'].sum() > a['absorbed'].sum()


def test_distance_control_converges_to_direct_long_solve_without_reemission():
    c = np.ones((12,)*3)
    s = np.zeros_like(c)
    s[6, 6, 6] = 150
    out, audit = solve_distance_converged(s, c, distances=[1., 5., 10.],
                                         min_angles=384, max_angles=384)
    assert audit['distance_converged']
    assert len(audit['history']) == 3
    direct = solve(s, c, maximum_distance=10., angular_reference_distance=1.,
                   min_angles=384, max_angles=384)
    np.testing.assert_array_equal(out['absorbed'], direct['absorbed'])
    assert out['absorbed'].sum() == pytest.approx(150.)


def test_small_retention_does_not_override_failed_field_or_statistic_check():
    c = np.ones((12,)*3)
    s = np.zeros_like(c)
    s[6, 6, 6] = 150
    _, audit = solve_distance_converged(s, c, distances=[1., 5.],
                                      min_angles=384, max_angles=384)
    assert audit['history'][-1]['retained_fraction'] == pytest.approx(0.)
    assert not audit['distance_converged']
    _, audit = solve_distance_converged(s, c, distances=[1., 5.],
                                      field_tolerance=2., statistic=lambda r: r['absorbed'].sum(),
                                      min_angles=384, max_angles=384)
    assert not audit['distance_converged']


def test_insufficient_distance_is_reported_not_hidden_by_source_scaling():
    c = np.ones((12,)*3)
    s = c*.9
    s[:6] *= 2
    s[6:] = 0
    _, audit = solve_distance_converged(s, c, distances=[.1, .2], min_angles=24, max_angles=24)
    assert not audit['distance_converged']
    assert not audit['source_budget_rescaled']


@pytest.mark.parametrize('bad', [0., -1., float('nan'), float('inf')])
def test_invalid_angle_reference_rejected(bad):
    with pytest.raises(ValueError, match='Angular reference'):
        solve(np.ones((2,)*3), np.ones((2,)*3), angular_reference_distance=bad)
