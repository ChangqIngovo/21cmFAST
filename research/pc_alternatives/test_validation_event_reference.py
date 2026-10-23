"""Independent geometry and temporal comparisons for the event-front candidate."""

import numpy as np
import pytest
from validation_event_reference import (
    axes_and_diagonals,
    compare_case,
    history_cases,
    horizon_error_cases,
    ordinary_dda,
    randomized_cases,
    reference_solve,
    scale_and_tie_cases,
    special_cases,
)


def test_reference_axis_boundaries_and_periodic_path():
    """Check hand-derived integer paths independently of the candidate."""
    direction = np.array([1.0, 0.0, 0.0])
    face = ordinary_dda(2, (0, 0, 0), direction, 0.5)
    assert face.cells == (0,)
    assert face.terminal == "retained"
    exit_path = ordinary_dda(2, (0, 0, 0), direction, 1.5)
    assert exit_path.cells == (0, 4)
    assert exit_path.terminal == "escaped"
    wrap = ordinary_dda(2, (0, 0, 0), direction, 5.1, periodic=True)
    assert wrap.cells == (0, 4, 0, 4, 0, 4)
    assert wrap.terminal == "retained"


def test_reference_corner_crossing_does_not_enter_zero_length_cells():
    """A diagonal ray advances all tied axes at the same geometric crossing."""
    direction = np.ones(3) / np.sqrt(3)
    path = ordinary_dda(3, (1, 1, 1), direction, 2.0)
    assert path.cells == (13, 26)
    negative = ordinary_dda(3, (1, 1, 1), -direction, 2.0)
    assert negative.cells == (13, 0)


def test_reference_matches_analytic_six_axis_clock():
    """The independent reference reproduces the 13/7 source-opening time."""
    source = np.zeros((5,) * 3)
    capacity = np.full_like(source, 100.0)
    source[1, 2, 2] = source[2, 2, 2] = 6
    capacity[1, 2, 2], capacity[2, 2, 2] = 6, 12
    result = reference_solve(source, capacity, axes_and_diagonals()[:6], duration=2)
    expected = np.zeros_like(source)
    expected[1, 2, 2], expected[2, 2, 2] = 6, 12
    expected[3, 2, 2], expected[0, 2, 2] = 2 / 7, 8 / 7
    for y, z in [(1, 2), (3, 2), (2, 1), (2, 3)]:
        expected[1, y, z] = 1
        expected[2, y, z] = 1 / 7
    np.testing.assert_allclose(result["absorbed"], expected, rtol=0, atol=1e-14)


@pytest.mark.parametrize("capacity", [1.0, 1e6, 1e12, 1e20])
def test_weak_source_absorption_is_not_lost_to_capacity_subtraction(capacity):
    """Weak emission must appear in the cell field as well as the source ledger."""
    row = compare_case(
        np.full((1,) * 3, 0.001),
        np.full((1,) * 3, capacity),
        np.array([[1.0, 0.0, 0.0]]),
    )
    assert row["passed"], row


def test_strict_horizons_and_periodic_cycles():
    """Include epsilon-scale offsets about cell and open-box boundaries."""
    failures = [row for row in special_cases() if not row["passed"]]
    assert not failures, failures


def test_random_small_grids_against_independent_paths_and_events():
    """Compare all cells and source categories on 100 deterministic fixtures."""
    failures = [row for row in randomized_cases(100) if not row["passed"]]
    assert not failures, failures


def test_ties_unit_scaling_and_mixed_capacities():
    """Exercise equal event times and scales absent from normal random maps."""
    failures = [row for row in scale_and_tie_cases() if not row["passed"]]
    assert not failures, failures


def test_constant_splitting_and_changing_source_histories():
    """Only unchanged histories compose; reversed emission order may differ."""
    result = history_cases()
    assert result["passed"], result
    assert result["constant_split_max_cell_error"] < 2e-10
    assert result["history_reversal_atom_l1"] == pytest.approx(5 / 3)


def test_retained_budget_bounds_longer_horizon_atom_error():
    """Nested fixed paths can only add absorption, bounded by short retained Q."""
    result = horizon_error_cases()
    assert result["passed"], result
