"""Keep a late neutral restart distinct from a full reionization history."""

import numpy as np
import pytest

from research.es_pc_emulator.initial_state import (
    initial_state_audit,
    require_initial_state,
    validate_policy,
)


def fixture(ionized, density):
    """Construct a weighted initial comparison without transport artifacts."""
    return ({"capacity": np.asarray(density), "redshifts": np.array([8.0])},
            {"xhi_es": np.array([1.0 - np.asarray(ionized)]),
             "density_es": np.array([density])})


def test_initial_state_reports_mass_weighted_late_restart():
    """A late neutral restart exposes its large missing initial ion inventory."""
    history, es = fixture([0.0, 0.4], [1.0, 3.0])
    report = initial_state_audit(history, es, initial_xhii=0.0)
    assert report["initial_es_Q_volume"] == pytest.approx(0.2)
    assert report["initial_es_Q_mass_reference_gas"] == pytest.approx(0.3)
    assert report["mass_weighted_initial_MAE"] == pytest.approx(0.3)
    assert report["compatible"] is False
    with pytest.raises(ValueError, match="Start the RT history earlier"):
        require_initial_state(report)
    require_initial_state(report, allow_mismatch=True)
    assert report["compatible"] is False  # An opt-in never changes the measurement.


def test_equal_mean_does_not_hide_different_initial_morphology():
    """Cancelling signed errors cannot bypass the spatial mismatch check."""
    history, es = fixture([0.0, 0.4], [1.0, 1.0])
    report = initial_state_audit(history, es, initial_xhii=0.2)
    assert report["initial_delta_Q_mass"] == pytest.approx(0.0)
    assert report["mass_weighted_initial_MAE"] == pytest.approx(0.2)
    assert report["compatible"] is False


def test_early_neutral_start_passes_without_changing_fields():
    """Screening never replaces the physical initial fractions."""
    history, es = fixture([0.0, 2e-6], [1.0, 2.0])
    original = es["xhi_es"].copy()
    report = initial_state_audit(history, es)
    assert report["compatible"] is True
    require_initial_state(report)
    np.testing.assert_array_equal(original, es["xhi_es"])


@pytest.mark.parametrize("policy", [
    {"max_mass_weighted_error": 0}, {"max_mass_weighted_error": float("nan")},
    {"max_mass_weighted_error": True}, {"allow_mismatch": "yes"}, {"typo": 1},
])
def test_initial_policy_rejects_invalid_or_ambiguous_settings(policy):
    """Require explicit boolean opt-in and a finite, meaningful tolerance."""
    with pytest.raises(ValueError, match=r"initial_state|Initial-state"):
        validate_policy(policy)
