"""Check absorption prediction without imposing a closed-box assumption."""

from pathlib import Path

import numpy as np
import pytest

from research.pc_analytic_emulator.budget import (
    LocalDeficitBudgetModel,
    TabulatedBudgetModel,
    constant_efficiency_model,
    fit_budget,
    fit_budget_v3,
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


def _native_globals() -> np.ndarray:
    values = np.zeros((6, 14))
    values[:, 0] = [0.1, 0.11, 0.12, 1.0, 1.01, 1.02]
    values[:, 1] = [10, 10, 10, 6, 6, 6]
    values[:, 4:10] = np.arange(6)[:, None] / 20
    return values


def test_local_deficit_preserves_zero_and_finite_escape(tmp_path: Path) -> None:
    """A local zero is exact, while photon surplus alone never forces eta=1."""
    training = _native_globals()
    efficiencies = np.array([1, 1, 1, 0.8, 0.85, 0.9])
    model = LocalDeficitBudgetModel.from_training(training, efficiencies, neighbors=3)
    query = training[[1, 4]].copy()
    query[:, 0] += 0.0001
    predicted = model.predict(query)
    assert predicted[0] == 1
    assert 0.8 < predicted[1] < 0.9
    assert np.expm1(query[1, 0]) > 1
    model.save(tmp_path / "budget.npz")
    restored = load_budget(tmp_path / "budget.npz")
    assert isinstance(restored, LocalDeficitBudgetModel)
    np.testing.assert_array_equal(restored.predict(query), predicted)


def test_local_deficit_uses_native_morphology_at_fixed_photon_load() -> None:
    """Equal photons/atoms can have different absorption from native structure."""
    training = _native_globals()
    training[:, 0] = 1
    training[:, 1] = 6.5
    efficiencies = np.array([1, 1, 1, 0.8, 0.8, 0.8])
    model = LocalDeficitBudgetModel.from_training(training, efficiencies, neighbors=3)
    np.testing.assert_array_equal(model.predict(training[[1, 4]]), [1, 0.8])
    changed_epoch = training[[1, 4]].copy()
    changed_epoch[:, 1] = [100, 0]
    np.testing.assert_array_equal(model.predict(changed_epoch), [1, 0.8])


def test_local_deficit_duplicate_inputs_average_labels() -> None:
    """Indistinguishable native inputs receive their mean empirical deficit."""
    training = np.repeat(_native_globals()[:1], 3, axis=0)
    model = LocalDeficitBudgetModel.from_training(
        training, np.array([0.8, 0.9, 1.0]), neighbors=1
    )
    np.testing.assert_allclose(model.predict(training[:1]), [0.9], atol=1e-15)


def test_local_deficit_normalization_and_table_are_training_only() -> None:
    """Changing validation labels may change ranks, never the local fit."""
    training = _native_globals()
    efficiencies = np.array([1, 1, 1, 0.8, 0.85, 0.9])
    validation = training[[1, 4]].copy()
    validation[:, 0] += 0.005
    options = {"seed": 11, "epochs": 2, "patience": 2, "learning_rate": 1e-3}
    first, report = fit_budget_v3(
        training,
        efficiencies,
        np.ones(6),
        validation,
        np.array([1, 0.8]),
        np.ones(2),
        **options,
    )
    second, _ = fit_budget_v3(
        training,
        efficiencies,
        np.ones(6),
        validation * 1.2,
        np.array([0.4, 0.5]),
        np.ones(2),
        **options,
    )
    assert set(first) == {
        "constant_efficiency",
        "tabulated",
        "mlp",
        "local_k3",
        "local_k5",
    }
    for name in ("local_k3", "local_k5"):
        a, b = first[name], second[name]
        np.testing.assert_array_equal(a.mean, b.mean)
        np.testing.assert_array_equal(a.scale, b.scale)
        np.testing.assert_array_equal(a.training_features, b.training_features)
        np.testing.assert_array_equal(a.deficits, b.deficits)
        np.testing.assert_array_equal(a.predict(validation), b.predict(validation))
    assert not report["test_used"]
    assert report["retained_candidate_count"] == 5


@pytest.mark.parametrize("neighbors", [0, -1, True, 1.5])
def test_local_deficit_rejects_invalid_neighbor_counts(neighbors: int) -> None:
    """Neighborhood size must be a positive integer."""
    with pytest.raises((ValueError, TypeError)):
        LocalDeficitBudgetModel.from_training(
            _native_globals(), np.ones(6), neighbors=neighbors
        )


def test_local_deficit_rejects_invalid_inputs_and_bounded_extrapolation() -> None:
    """Prediction is bounded away from the training set and invalid data fails."""
    training = _native_globals()
    with pytest.raises(ValueError, match="Invalid native-input training"):
        LocalDeficitBudgetModel.from_training(training[:, :2], np.ones(6))
    with pytest.raises(ValueError, match="Invalid native-input training"):
        LocalDeficitBudgetModel.from_training(training, np.full(6, 1.1))
    model = LocalDeficitBudgetModel.from_training(training, np.linspace(0, 1, 6))
    query = training[:2].copy()
    query[:, 0] = [0, 10]
    predicted = model.predict(query)
    assert np.all((predicted >= 0) & (predicted <= 1))
    query[0, 4] = np.nan
    with pytest.raises(ValueError, match="Invalid native budget feature"):
        model.predict(query)


def test_v3_zero_photon_labels_keep_all_non_network_candidates() -> None:
    """Zero absorbed mass never induces an invalid weighted-network fit."""
    training = _native_globals()
    training[:, 0] = 0
    models, report = fit_budget_v3(
        training,
        np.ones(6),
        np.zeros(6),
        training[:2],
        np.ones(2),
        np.zeros(2),
        seed=11,
        epochs=1,
        patience=1,
        learning_rate=1e-3,
    )
    assert "mlp" not in models
    assert report["selected_kind"] == "constant_efficiency"
    assert report["retained_candidate_count"] == 4
    for model in models.values():
        np.testing.assert_array_equal(
            model.predict(training, baseline=np.ones(6)), np.ones(6)
        )
