from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adaptive


def test_weak_local_photons_survive_state_continuation():
    c = np.ones((4,)*3)
    s = np.full_like(c, 1e-25)
    state = None
    for step in range(1, 5):
        result = adaptive.solve(s, c, initial_state=state, min_angles=24, max_angles=24)
        state = result["state"]
        np.testing.assert_allclose(state["ionized"], step*s, rtol=2e-14, atol=0)
    assert len(state["parent_ids"]) == 0


def test_cannot_use_carried_state_with_changed_density():
    c = np.ones((4,)*3)
    s = c*.1
    state = adaptive.solve(s, c)["state"]
    with pytest.raises(ValueError, match="same static capacity"):
        adaptive.solve(s, c*2, initial_state=state)


def test_cannot_mix_coarse_and_resolved_initial_state():
    c = np.ones((4,)*3)
    s = c*.1
    state = adaptive.solve(s, c)["state"]
    with pytest.raises(ValueError, match="not both"):
        adaptive.solve(s, c, initial_ionized=c*.1, initial_state=state)


@pytest.mark.parametrize("bad", [float("nan"), -1., float("inf")])
def test_invalid_source_rejected(bad):
    c = np.ones((3,)*3)
    s = np.zeros_like(c)
    s[1, 1, 1] = bad
    with pytest.raises(ValueError):
        adaptive.solve(s, c)


@pytest.mark.parametrize("units", [1e-18, 1e18])
def test_ionization_and_front_state_are_unit_invariant(units):
    c = np.ones((10,)*3)
    s = np.zeros_like(c)
    s[5, 5, 5] = 60
    base = adaptive.solve(s, c, min_angles=384, max_angles=384, return_dense=True)
    rescaled = adaptive.solve(s*units, c*units, min_angles=384, max_angles=384, return_dense=True)
    np.testing.assert_allclose(rescaled["absorbed"]/units, base["absorbed"], atol=1e-12, rtol=1e-10)
    np.testing.assert_allclose(rescaled["subcell_ionized"]/units, base["subcell_ionized"], atol=1e-12, rtol=1e-10)
