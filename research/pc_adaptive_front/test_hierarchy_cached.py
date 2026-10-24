"""Run the same independent geometry regressions against cached block levels."""
from pathlib import Path
import sys

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hierarchy_cached_benchmark as cached
import test_hierarchy as original_tests


@pytest.fixture(autouse=True)
def use_cached(monkeypatch):
    monkeypatch.setattr(original_tests, "solve", cached.solve)
    monkeypatch.setattr(original_tests, "module", cached.module)


@pytest.mark.parametrize("n", [8, 12, 15])
@pytest.mark.parametrize("periodic", [False, True])
def test_random_geometry(n, periodic):
    original_tests.test_random_partial_fields_match_parent_only(n, periodic)


def test_initial_cavity():
    original_tests.test_large_initial_cavity_has_certified_jumps()


def test_dynamic_clearing_and_state():
    original_tests.test_dynamic_clearing_and_carried_state()


def test_near_corner():
    original_tests.test_near_corner_crossing_keeps_thin_opaque_cell()


@pytest.mark.parametrize("n", [10, 12, 24])
def test_non_power_two_periodic_translation(n):
    capacity=np.ones((n,)*3)
    old=np.zeros_like(capacity)
    size=min(8,n//2)
    old[:size,:size,:size]=1
    source=np.zeros_like(capacity)
    source[2,2,2]=.1
    options=dict(min_angles=384,max_angles=384,maximum_distance=n,return_dense=True)
    a=cached.solve(source,capacity,old,**options)
    shift=(n-3,n-5,n-2)
    b=cached.solve(np.roll(source,shift,(0,1,2)),capacity,np.roll(old,shift,(0,1,2)),**options)
    np.testing.assert_allclose(np.roll(a["subcell_absorbed"],tuple(2*x for x in shift),(0,1,2)),
                               b["subcell_absorbed"],rtol=0,atol=1e-14)
