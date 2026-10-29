"""Independent budgets, analytic limits and failure semantics of coeval RT."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from research.pc_matched.rt import MPC_CM
from research.pc_native_rt.solver import solve


def options(**changes):
    result = dict(redshift=8.0, box_len_cMpc=1.0, directions=24,
                  exhaustion_tolerance=1e-7, max_box_crossings=4.0)
    result.update(changes)
    return result


def test_uniform_absorption_matches_closed_form_and_keeps_unfinished_state():
    """Equal initial stocks obey I/C=kt/(1+kt), with no exhaustion shortcut."""
    capacity = np.ones((2, 2, 2))
    sigma = 6.3e-18 * (13.6 / 18) ** 3
    box, crossings, coefficient = 10.0, 0.2, 0.7
    density = coefficient / (sigma * box * MPC_CM * crossings)
    result = solve(capacity, capacity, **options(
        redshift=0, box_len_cMpc=box, hydrogen_density_z0_cm3=density,
        max_box_crossings=crossings))
    np.testing.assert_allclose(result["ionized"], coefficient / (1 + coefficient),
                               rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(result["radiation"].sum(axis=0), 1 / (1 + coefficient),
                               rtol=2e-14, atol=2e-14)
    assert result["audit"]["ledger_passed"]
    assert not result["audit"]["converged"]
    assert not result["audit"]["passed"]
    assert result["audit"]["status"] == "max_box_crossings_reached"


@pytest.mark.parametrize("budget", [0.4, 2.0])
@pytest.mark.parametrize("injection", [0.0, 0.8])
def test_uniform_shortage_or_surplus_preserves_every_photon(budget, injection):
    """Emission is not double counted; supercritical photons survive ionization."""
    capacity = np.ones((3, 3, 3))
    source = budget * capacity
    original = source.copy()
    result = solve(source, capacity, **options(injection_box_crossings=injection))
    audit = result["audit"]
    assert audit["passed"]
    assert audit["emitted"] == pytest.approx(source.sum(), rel=2e-14)
    assert audit["source_injection_complete"]
    np.testing.assert_allclose(result["ionized"], min(budget, 1), atol=2e-7, rtol=0)
    np.testing.assert_allclose(result["radiation"].sum(axis=0), max(budget - 1, 0),
                               atol=2e-7, rtol=0)
    assert audit["ionized"] + audit["radiation_in_flight"] == pytest.approx(source.sum(), rel=2e-13)
    np.testing.assert_array_equal(source, original)
    np.testing.assert_array_equal(capacity, np.ones_like(capacity))


def test_zero_source_stays_neutral_without_propagation():
    capacity = np.ones((2, 2, 2))
    result = solve(np.zeros_like(capacity), capacity, **options(injection_box_crossings=2))
    assert result["audit"]["passed"]
    assert result["metadata"]["steps"] == 0
    np.testing.assert_array_equal(result["ionized"], 0)
    np.testing.assert_array_equal(result["radiation"], 0)


@pytest.mark.parametrize("injection", [0.0, 0.5])
def test_empty_box_has_no_fictitious_absorbers(injection):
    capacity = np.zeros((2, 2, 2))
    source = np.full_like(capacity, 0.8)
    result = solve(source, capacity, **options(injection_box_crossings=injection))
    assert result["audit"]["passed"]
    assert result["audit"]["emitted"] == pytest.approx(source.sum(), rel=2e-14)
    np.testing.assert_array_equal(result["ionized"], 0)
    np.testing.assert_allclose(result["radiation"].sum(axis=0), source, rtol=2e-14)


def test_mixed_vacuum_and_gas_uses_actual_zero_capacity():
    capacity = np.ones((3, 3, 3))
    capacity[0] = 0
    source = np.zeros_like(capacity)
    source[0] = 0.5
    result = solve(source, capacity, **options(max_box_crossings=20))
    assert result["audit"]["passed"]
    np.testing.assert_array_equal(result["ionized"][0], 0)
    assert result["audit"]["ionized"] == pytest.approx(source.sum(), rel=2e-7)
    assert np.all(result["ionized"] <= capacity)


def test_near_equal_budget_is_not_silently_completed():
    """The algebraic equal-stock tail must fail exhaustion rather than be filled."""
    capacity = np.ones((2, 2, 2))
    result = solve(capacity, capacity, **options(max_box_crossings=0.01))
    assert result["audit"]["ledger_passed"]
    assert not result["audit"]["passed"]
    assert result["audit"]["remaining_neutral"] > 0
    assert result["audit"]["radiation_in_flight"] > 0
    np.testing.assert_allclose(result["ionized"] + result["radiation"].sum(axis=0), 1,
                               atol=2e-14, rtol=0)


def test_incomplete_injection_is_not_success_even_if_current_photons_absorb():
    capacity = np.ones((2, 2, 2))
    result = solve(0.2 * capacity, capacity, **options(
        max_box_crossings=0.1, injection_box_crossings=1.0))
    assert result["audit"]["ledger_passed"]
    assert not result["audit"]["source_injection_complete"]
    assert not result["audit"]["passed"]
    assert result["audit"]["emitted"] == pytest.approx(0.1 * 0.2 * capacity.size)


def test_total_step_limit_returns_partial_state_and_preserves_budget():
    capacity = np.ones((3, 3, 3))
    result = solve(capacity, capacity, **options(max_steps=1))
    assert result["metadata"]["steps"] == 1
    assert result["audit"]["status"] == "max_steps_reached"
    assert result["audit"]["ledger_passed"]
    assert not result["audit"]["passed"]


def test_periodic_translation_and_parallel_calls_preserve_solution():
    """No source-position bias or shared mutable radiation state is introduced."""
    rng = np.random.default_rng(421)
    capacity = rng.uniform(0.6, 1.4, (4, 4, 4))
    source = np.zeros_like(capacity)
    source[0, 1, 2] = 0.15 * capacity.sum()
    kwargs = options(max_box_crossings=6)
    serial = solve(source, capacity, **kwargs)
    translated = solve(np.roll(source, 1, axis=0), np.roll(capacity, 1, axis=0), **kwargs)
    np.testing.assert_allclose(translated["ionized"], np.roll(serial["ionized"], 1, axis=0),
                               rtol=3e-14, atol=3e-14)
    np.testing.assert_allclose(translated["radiation"], np.roll(serial["radiation"], 1, axis=1),
                               rtol=3e-14, atol=3e-14)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda threads: solve(source, capacity, threads=threads, **kwargs), [1, 2]))
    for result in results:
        assert result["audit"]["passed"]
        np.testing.assert_array_equal(result["ionized"], serial["ionized"])
        np.testing.assert_array_equal(result["radiation"], serial["radiation"])


@pytest.mark.parametrize("parameter,value", [
    ("max_steps", 0), ("max_steps", True), ("max_box_crossings", -1),
    ("injection_box_crossings", -1), ("directions", 7), ("cfl", 0),
    ("threads", 0), ("exhaustion_tolerance", 0), ("photon_energy_ev", 10),
    ("redshift", -1), ("redshift", np.nan), ("box_len_cMpc", 0),
])
def test_invalid_controls_raise_before_running(parameter, value):
    capacity = np.ones((2, 2, 2))
    with pytest.raises(ValueError):
        solve(capacity, capacity, **options(**{parameter: value}))


@pytest.mark.parametrize("bad", [-1.0, np.nan, np.inf, 1j])
def test_invalid_source_is_never_clipped(bad):
    capacity = np.ones((2, 2, 2))
    with pytest.raises(ValueError):
        solve(np.full(capacity.shape, bad), capacity, **options())
