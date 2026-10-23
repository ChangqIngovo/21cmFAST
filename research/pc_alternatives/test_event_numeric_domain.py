import numpy as np
import pytest

from event_front import solve


@pytest.mark.parametrize("n,rate,duration,angles", [
    (1, 1e308, 10., 1),
    (2, 1e308, 1e-308, 1),
    (1, 1e-320, 1., 49152),
    (1, 1e-200, 1e-200, 1),
])
def test_unrepresentable_budget_rejected_before_native_code(n, rate, duration, angles):
    with pytest.raises(ValueError, match="rescale units"):
        solve(np.full((n,)*3, rate), np.zeros((n,)*3), duration=duration, angles=angles)


def test_rescaled_units_preserve_solution_and_budget():
    source = np.zeros((4,)*3); source[0, 1, 1] = 3
    capacity = np.ones_like(source)
    a = solve(source, capacity, angles=48)
    for scale in (1e-120, 1e120):
        b = solve(source*scale, capacity*scale, angles=48)
        np.testing.assert_allclose(b['absorbed']/scale, a['absorbed'], rtol=3e-13, atol=3e-14)
        assert b['audit']['per_source_relative_ledger_max'] < 1e-12


def test_zero_source_allows_extreme_duration_without_nan():
    out = solve(np.zeros((2,)*3), np.ones((2,)*3), duration=1e300, angles=48)
    assert out['audit']['photons'] == out['audit']['ledger_residual'] == 0
