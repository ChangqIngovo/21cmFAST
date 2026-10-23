"""Regression tests for physical counterexamples and the independent chemistry kernel."""

import numpy as np
from validation_recombination import (
    debt_events,
    local_chemistry_step,
    local_kernel_convergence,
    opacity_reference,
    reopened_screen_probe,
    source_off_probe,
    stromgren_probe,
)


def test_stromgren_converges_first_order_with_closed_ledger():
    """Check convergence to the analytic ionized-volume equation."""
    result = stromgren_probe()
    assert 0.9 < result["rows"][-1]["observed_volume_order"] < 1.1
    assert max(r["atom_ledger_error"] for r in result["rows"]) < 1e-12
    assert result["exact_frozen_coefficient_update_error"] < 1e-14


def test_source_off_needs_nonequilibrium_recombination():
    """Recover parcel recombination even with zero photoionization rate."""
    result = source_off_probe()
    assert result["exact_x"] == 1 / 3
    assert (
        result["rows"][-1]["absolute_error"] < result["rows"][0]["absolute_error"] / 20
    )
    assert max(r["exact_recombination_substep_error"] for r in result["rows"]) < 2e-14
    assert result["Gamma_zero_adapter_absolute_error"] > 0.66


def test_conservative_debt_is_not_finite_opacity_transport():
    """Keep a non-vanishing opacity error despite a closed photon ledger."""
    reference = opacity_reference([1], [1], [1], [1], [[0]], [1], 2)
    for steps in (4, 64, 256):
        result = debt_events([1], [1], [1], [[0]], [1], 2, steps)
        assert result["x"][0] == 1
        assert result["photon_ledger_error"] < 1e-13
        assert result["atom_ledger_max_error"] < 1e-13
        assert result["x"][0] - reference["x"][0] > 0.35
    assert reference["escaped_by_beam"][0] > 1
    assert reference["photon_ledger_error"] < 1e-10


def test_local_chemistry_is_second_order_and_conservative():
    """Verify the proposed chemistry kernel against independently integrated ODEs."""
    result = local_kernel_convergence()
    assert 1.95 < result["rows"][-1]["observed_order"] < 2.05
    assert result["Radau_DOP853_max_difference"] < 1e-10
    assert max(r["photon_ledger_error"] for r in result["rows"]) < 1e-12
    assert max(r["atom_ledger_error"] for r in result["rows"]) < 1e-12


def test_local_chemistry_bounds_zero_opacity_and_stiff_limits():
    """Exercise transparent, dark, saturated and optically thick limiting cases."""
    x = np.array([0.0, 1.0, 0.2, 0.99, 0.5])
    flux = np.array([0.0, 0.0, 100.0, 1.0, 5.0])
    tau = np.array([1.0, 1000.0, 10000.0, 0.01, 0.0])
    end, absorbed, rec, escape = local_chemistry_step(x, flux, 1.0, 1.0, tau, 10.0)
    assert np.all(np.isfinite(end))
    assert np.all((end >= 0) & (end <= 1))
    assert np.all(absorbed >= -1e-13)
    assert np.all(escape >= -1e-13)
    np.testing.assert_allclose(absorbed + escape, flux * 10, atol=1e-12)
    np.testing.assert_allclose(end - x, absorbed - rec, atol=1e-12)
    assert absorbed[-1] == 0


def test_recombined_screen_requires_rebuilt_or_invalidated_rays():
    """Detect downstream leakage from a deliberately stale transparent-cell cache."""
    result = reopened_screen_probe()
    np.testing.assert_allclose(result["stale_cursor_absorption_increment"], [0, 0.2])
    np.testing.assert_allclose(result["rebuilt_absorption_increment"], [0.2, 0])


def test_two_beam_overlap_reference_and_sharp_approximation_are_audited():
    """Verify both ledgers while exposing a multi-beam opacity-closure difference."""
    atoms, initial, alpha = [1, 3, 3], [1, 0, 0], [1, 0.4, 0.4]
    paths, rates = [[0, 1], [0, 2]], [1, 2]
    reference = opacity_reference(atoms, initial, alpha, [1, 4, 4], paths, rates, 1.5)
    result = debt_events(atoms, initial, alpha, paths, rates, 1.5, 128)
    assert reference["photon_ledger_error"] < 1e-10
    assert reference["atom_ledger_max_error"] < 1e-10
    assert result["photon_ledger_error"] < 1e-10
    assert result["atom_ledger_max_error"] < 1e-10
    assert np.max(np.abs(np.asarray(result["x"]) - reference["x"])) > 0.1
