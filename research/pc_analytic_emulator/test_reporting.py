"""Guard scientific summaries against signed cancellation and trivial epochs."""

import pytest

from .train import _test_aggregates


def _row(neutral: float, bias: float, redshift: float) -> dict:
    return {
        "seed": 1008,
        "redshift": redshift,
        "evaluation": {"metrics": {"teacher": {"Q_mass": 1 - neutral}}},
        "field_metrics": {
            name: {
                "MAE": abs(bias),
                "RMSE": 2 * abs(bias),
                "delta_Q_mass": bias,
                "delta_Q_volume": -bias,
            }
            for name in ("raw", "constrained", "native_es")
        },
    }


def test_absolute_errors_do_not_cancel_opposite_budget_biases() -> None:
    """Retain error magnitudes when opposite signed biases average to zero."""
    report = _test_aggregates([_row(0.6, 0.02, 8), _row(0.4, -0.02, 7.5)])
    mean = report["test_metrics_mean_over_snapshots"]["constrained"]
    errors = report["test_absolute_and_worst_errors"]["constrained"]
    assert mean["delta_Q_mass"] == 0
    assert errors["mean_absolute_delta_Q_mass"] == pytest.approx(0.02)
    assert errors["maximum_absolute_delta_Q_volume"] == pytest.approx(0.02)
    assert errors["maximum_snapshot_RMSE"] == pytest.approx(0.04)


def test_phase_summary_keeps_late_errors_separate_from_zero_fields() -> None:
    """Show a late-stage failure independently of a trivial zero-field epoch."""
    rows = [
        _row(0.9, 0.001, 10),
        _row(0.6, 0.01, 8),
        _row(0.1, 0.08, 6.5),
        _row(0, 0, 6),
    ]
    report = _test_aggregates(rows)
    phases = report["test_metrics_by_reionization_phase"]["phases"]
    assert sum(phase["count"] for phase in phases.values()) == len(rows)
    assert all(phase["count"] == 1 for phase in phases.values())
    assert phases["late"]["metrics_mean"]["constrained"]["RMSE"] == 0.16
    assert phases["fully_ionized"]["metrics_mean"]["constrained"]["RMSE"] == 0
    assert phases["late"]["snapshots"][0]["redshift"] == 6.5


def test_empty_phases_have_undefined_metrics() -> None:
    """Do not report perfect accuracy when a phase has no test snapshots."""
    report = _test_aggregates([_row(0.5, 0.01, 7.5)])
    phases = report["test_metrics_by_reionization_phase"]["phases"]
    assert phases["early"]["count"] == 0
    assert phases["early"]["metrics_mean"] is None
    assert phases["late"]["absolute_and_worst_errors"] is None
    assert _test_aggregates([])["test_absolute_and_worst_errors"] is None
