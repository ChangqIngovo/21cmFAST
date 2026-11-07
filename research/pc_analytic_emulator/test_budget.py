"""Check absorption prediction without imposing a closed-box assumption."""

from pathlib import Path

import numpy as np
import pytest

from research.pc_analytic_emulator.budget import (
    TabulatedBudgetModel,
    constant_efficiency_model,
    fit_budget,
    load_budget,
)


def test_table_preserves_escape_and_exact_endpoint(tmp_path: Path) -> None:
    """A bounded table retains finite escape and exact full absorption."""
    model = TabulatedBudgetModel(
        np.array([0.1, 0.5, 1.0]), np.array([1.0, 0.8, 1.0]), 3
    )
    features = np.array([[0.0, 10.0, 3.0], [0.5, 7.0, 1.0], [2.0, 6.0, 0.0]])
    expected = np.array([1.0, 0.8, 1.0])
    np.testing.assert_array_equal(model.predict(features), expected)
    model.save(tmp_path / "budget.npz")
    restored = load_budget(tmp_path / "budget.npz")
    np.testing.assert_array_equal(restored.predict(features), expected)


def test_table_is_fitted_on_training_snapshots_only() -> None:
    """Validation targets select candidates but never alter interpolation knots."""
    xtrain = np.array([[0.1, 10], [0.2, 10], [0.5, 8], [0.6, 8], [1.0, 6], [1.1, 6]])
    eta = np.array([1.0, 1.0, 0.75, 0.85, 1.0, 1.0])
    xval = np.array([[0.15, 10], [0.55, 8], [1.05, 6]])
    options = {"seed": 11, "epochs": 2, "patience": 2, "learning_rate": 1e-3}
    selected, report = fit_budget(
        xtrain, eta, np.ones(6), xval, np.array([1.0, 0.8, 1.0]), np.ones(3), **options
    )
    assert report["selected_kind"] == "tabulated"
    np.testing.assert_allclose(selected.predict(xval), [1.0, 0.8, 1.0], atol=1e-15)
    _, changed = fit_budget(
        xtrain, eta, np.ones(6), xval, np.ones(3), np.ones(3), **options
    )
    assert changed["selected_kind"] == "constant_efficiency"
    assert (
        report["table_knots_log1p_photons_per_atom"]
        == changed["table_knots_log1p_photons_per_atom"]
    )
    assert report["table_efficiencies"] == changed["table_efficiencies"]
    assert not report["test_used"]


def test_candidate_error_uses_actual_absorbed_mass() -> None:
    """The same eta error counts less when fewer photons per atom are supplied."""
    x = np.array([[0.1, 10], [1.0, 6]])
    eta = np.array([0.9, 0.9])
    _, report = fit_budget(
        x,
        eta,
        np.array([0.1, 1.0]),
        x,
        eta,
        np.array([0.1, 1.0]),
        seed=1,
        epochs=1,
        patience=1,
        learning_rate=1e-3,
    )
    constant = next(
        row for row in report["candidates"] if row["kind"] == "constant_efficiency"
    )
    assert constant["validation_mass_mse"] == pytest.approx((0.01**2 + 0.1**2) / 2)


def test_zero_photon_budget_needs_no_network_fit() -> None:
    """Photon-free data choose the exact baseline without zero-weight failure."""
    x = np.array([[0.0, 10], [0.0, 9]])
    model, report = fit_budget(
        x,
        np.ones(2),
        np.zeros(2),
        x,
        np.ones(2),
        np.zeros(2),
        seed=1,
        epochs=1,
        patience=1,
        learning_rate=1e-3,
    )
    assert report["selected_kind"] == "constant_efficiency"
    assert report["network_history"] == []
    np.testing.assert_array_equal(model.predict(x, baseline=np.ones(2)), np.ones(2))


@pytest.mark.parametrize("eta", [np.array([1.01, 0.9]), np.array([np.nan, 0.9])])
def test_nonphysical_budget_targets_fail(eta: np.ndarray) -> None:
    """Invalid scalar labels cannot silently change the fitted physical budget."""
    x = np.array([[0.1, 10], [1.0, 6]])
    with pytest.raises(ValueError, match="Invalid supervised"):
        fit_budget(
            x,
            eta,
            np.ones(2),
            x,
            np.ones(2),
            np.ones(2),
            seed=1,
            epochs=1,
            patience=1,
            learning_rate=1e-3,
        )


def test_reference_efficiency_remains_exact() -> None:
    """The reference can represent a fully absorbed or photon-surplus box."""
    model = constant_efficiency_model(2)
    np.testing.assert_array_equal(
        model.predict(np.array([[0.2, 8], [2.0, 6]]), baseline=np.ones(2)), np.ones(2)
    )
