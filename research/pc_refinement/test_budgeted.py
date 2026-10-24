from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from budgeted import solve
from speed_core import solve as original_solve


def test_unit_boost_is_original_model_bitwise():
    rng = np.random.default_rng(718)
    c = rng.uniform(.2, 2, size=(12,)*3)
    s = np.zeros_like(c)
    s[4, 6, 6], s[8, 7, 6] = 140, 75
    original = original_solve(s, c, variant="original")
    actual = solve(s, c, angular_boost=1)
    for name in ("absorbed", "source_absorbed", "source_retained", "angle_counts"):
        np.testing.assert_array_equal(actual[name], original[name])
    assert actual["audit"]["ray_allocated_bytes"] == .75*original["audit"]["ray_allocated_bytes"]


def test_boost_honors_actual_cap_and_keeps_all_photons():
    c = np.ones((12,)*3)
    s = np.zeros_like(c)
    s[6, 6, 6] = 120
    out = solve(s, c, angular_boost=4/3, max_angles=100)
    assert out["angle_counts"][6, 6, 6] == 100
    assert abs(out["absorbed"].sum()+out["unabsorbed"]-120) < 1e-10


@pytest.mark.parametrize("boost", [0, .5, -1, float("inf"), float("nan"), True])
def test_invalid_boost_rejected(boost):
    with pytest.raises(ValueError):
        solve(np.zeros((2,)*3), np.ones((2,)*3), angular_boost=boost)


def test_all_transparent_endpoint_keeps_reservoir():
    c = np.ones((3,)*3)
    s = np.ones_like(c)
    out = solve(s, c, c)
    assert out["unabsorbed"] == 27
    assert out["absorbed"].sum() == 0
