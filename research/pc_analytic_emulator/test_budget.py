"""Check absorption prediction without imposing a closed-box assumption."""

from pathlib import Path

import numpy as np
import pytest

from research.pc_analytic_emulator.budget import (
    LocalDeficitBudgetModel,
    TabulatedBudgetModel,
    budget_error_metrics,
    constant_efficiency_model,
    fit_budget,
    fit_budget_v3,
    fit_deficit_budget,
    load_budget,
    predict_efficiency,
)
from research.pc_field_emulator.model import (
    _residual_fraction,
    _residual_slope,
    _sigmoid,
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


@pytest.mark.parametrize("geometry_column", [2, 3])
def test_local_geometry_separates_native_domains_and_roundtrips(
    tmp_path: Path, geometry_column: int
) -> None:
    """Equal load/structure can retain different measured escape by geometry."""
    training = np.repeat(_native_globals()[:1], 6, axis=0)
    training[:3, geometry_column] = np.log1p(2.0)
    training[3:, geometry_column] = np.log1p(8.0)
    efficiencies = np.array([0.8, 0.8, 0.8, 0.95, 0.95, 0.95])
    original = LocalDeficitBudgetModel.from_training(training, efficiencies)
    model = LocalDeficitBudgetModel.from_training(
        training, efficiencies, include_geometry=True
    )
    np.testing.assert_allclose(original.predict(training[[0, 3]]), [0.875, 0.875])
    np.testing.assert_allclose(model.predict(training[[0, 3]]), [0.8, 0.95])
    np.testing.assert_array_equal(model.feature_indices, [0, 2, 3, 4, 5, 6, 7, 8, 9])
    constant = np.ptp(training[:, model.feature_indices], axis=0) == 0
    np.testing.assert_array_equal(model.scale[constant], 1.0)
    model.save(tmp_path / "geometry_budget.npz")
    restored = load_budget(tmp_path / "geometry_budget.npz")
    np.testing.assert_array_equal(restored.feature_indices, model.feature_indices)
    np.testing.assert_array_equal(restored.predict(training), model.predict(training))


def test_local_geometry_constant_training_coordinate_has_safe_scale() -> None:
    """An unseen geometry cannot be divided by rounding-level training scatter."""
    training = _native_globals()
    training[:, 2:4] = np.log1p([50 / 16, 50])
    model = LocalDeficitBudgetModel.from_training(
        training, np.linspace(0.8, 0.95, 6), include_geometry=True
    )
    np.testing.assert_array_equal(model.scale[1:3], [1.0, 1.0])
    query = training[:1].copy()
    query[:, 2:4] = np.log1p([256 / 32, 256])
    coordinates = (query[:, model.feature_indices] - model.mean) / model.scale
    assert np.max(np.abs(coordinates[:, 1:3])) < 2.0
    assert 0.8 <= model.predict(query)[0] <= 0.95


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_local_geometry_requires_explicit_boolean(value: object) -> None:
    """Non-boolean values cannot silently enable a new distance convention."""
    with pytest.raises(ValueError, match="include_geometry must be a boolean"):
        LocalDeficitBudgetModel.from_training(
            _native_globals(), np.ones(6), include_geometry=value
        )


def test_v3_geometry_candidates_retain_original_local_models() -> None:
    """Mixed-geometry candidates are additive and share the validation metric."""
    training = _native_globals()
    training[:, 2] = np.log1p([2, 2, 2, 4, 4, 4])
    training[:, 3] = np.log1p([50, 50, 50, 256, 256, 256])
    eta = np.linspace(0.8, 0.95, 6)
    options = {"seed": 11, "epochs": 2, "patience": 2, "learning_rate": 0.001}
    original, _ = fit_budget_v3(
        training, eta, np.ones(6), training[:2], eta[:2], np.ones(2), **options
    )
    expanded, report = fit_budget_v3(
        training,
        eta,
        np.ones(6),
        training[:2],
        eta[:2],
        np.ones(2),
        include_geometry_local=True,
        **options,
    )
    assert set(expanded) - set(original) == {"local_geometry_k3", "local_geometry_k5"}
    for name in ("local_k3", "local_k5"):
        np.testing.assert_array_equal(
            expanded[name].predict(training), original[name].predict(training)
        )
    assert report["local_geometry_feature_indices"] == [0, 2, 3, 4, 5, 6, 7, 8, 9]
    assert report["local_constant_coordinate_scale"] == 1.0
    changed, _ = fit_budget_v3(
        training,
        eta,
        np.ones(6),
        training[:2] * 1.2,
        np.array([0.2, 0.4]),
        np.ones(2),
        include_geometry_local=True,
        **options,
    )
    for name in ("local_geometry_k3", "local_geometry_k5"):
        np.testing.assert_array_equal(expanded[name].mean, changed[name].mean)
        np.testing.assert_array_equal(expanded[name].scale, changed[name].scale)
        np.testing.assert_array_equal(
            expanded[name].predict(training), changed[name].predict(training)
        )


def test_training_geometry_budget_configuration_is_opt_in(tmp_path: Path) -> None:
    """The shared training entry point admits only explicitly enabled models."""
    from research.pc_analytic_emulator.train import validate_config

    default = validate_config({"input_root": str(tmp_path)})
    assert default["include_geometry_budget"] is False
    settings = {
        "input_root": str(tmp_path),
        "training_method": "warm_started_v3",
        "include_geometry_budget": True,
        "budget_candidate_kinds": ["tabulated", "local_geometry_k3"],
    }
    assert validate_config(settings)["include_geometry_budget"] is True
    with pytest.raises(ValueError, match="enabled candidate"):
        validate_config({**settings, "include_geometry_budget": False})
    with pytest.raises(TypeError, match="include_geometry_budget"):
        validate_config({**settings, "include_geometry_budget": 1})
    with pytest.raises(ValueError, match="warm_started_v3"):
        validate_config({**settings, "training_method": "legacy"})


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
        "sigmoid_mlp",
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
    assert report["retained_candidate_count"] == 6


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
    assert "sigmoid_mlp" not in models
    assert report["selected_kind"] == "constant_efficiency"
    assert report["retained_candidate_count"] == 4
    for model in models.values():
        np.testing.assert_array_equal(
            model.predict(training, baseline=np.ones(6)), np.ones(6)
        )


def test_direct_sigmoid_can_leave_the_positive_residual_dead_region() -> None:
    """Finite absorption deficits give no gradient on the old positive branch."""
    target = 0.8
    residual_logit = np.array([0.5])
    sigmoid_logit = residual_logit.copy()
    for _ in range(500):
        amplitude = np.tanh(residual_logit)
        legacy = _residual_fraction(amplitude, np.ones(1), np.ones(1))
        legacy_error = legacy - target
        legacy_slope = _residual_slope(amplitude, np.ones(1), np.ones(1), legacy_error)
        residual_logit -= 0.5 * 2 * legacy_error * legacy_slope
        direct = _sigmoid(sigmoid_logit)
        sigmoid_logit -= 0.5 * 2 * (direct - target) * direct * (1 - direct)
    assert legacy[0] == 1
    assert legacy_slope[0] == 0
    assert residual_logit[0] == 0.5
    assert direct[0] == pytest.approx(target, abs=1e-6)
    assert direct[0] * (1 - direct[0]) > 0


def test_sigmoid_budget_fits_finite_deficit_and_keeps_training_only_prior(
    tmp_path: Path,
) -> None:
    """The new v3 candidate learns bounded deficits and restores its activation."""
    training = _native_globals()
    efficiencies = np.array([0.9, 0.9, 0.9, 0.8, 0.8, 0.8])
    validation = training[[1, 4]].copy()
    options = {
        "seed": 11,
        "epochs": 500,
        "patience": 100,
        "learning_rate": 0.02,
        "hidden": (),
    }
    models, report = fit_budget_v3(
        training,
        efficiencies,
        np.ones(6),
        validation,
        np.array([0.9, 0.8]),
        np.ones(2),
        **options,
    )
    smooth = models["sigmoid_mlp"]
    predicted = predict_efficiency(smooth, validation)
    assert smooth.output_mode == "sigmoid"
    assert smooth.metadata["initial_efficiency_prior"] == pytest.approx(0.85)
    assert report["sigmoid_network_history"]
    assert np.all((predicted > 0) & (predicted < 1))
    np.testing.assert_allclose(predicted, [0.9, 0.8], atol=0.003)
    smooth.save(tmp_path / "budget.npz")
    restored = load_budget(tmp_path / "budget.npz")
    assert restored.output_mode == "sigmoid"
    np.testing.assert_array_equal(predict_efficiency(restored, validation), predicted)

    # Initialization and training updates use no validation labels. With one
    # epoch there is only one checkpoint, so both parameter sets must match.
    options.update(epochs=1, patience=1)
    first, _ = fit_budget_v3(
        training,
        efficiencies,
        np.ones(6),
        validation,
        np.array([0.9, 0.8]),
        np.ones(2),
        **options,
    )
    changed, _ = fit_budget_v3(
        training,
        efficiencies,
        np.ones(6),
        validation,
        np.array([0.2, 0.3]),
        np.ones(2),
        **options,
    )
    first, changed = first["sigmoid_mlp"], changed["sigmoid_mlp"]
    assert (
        first.metadata["initial_efficiency_prior"]
        == changed.metadata["initial_efficiency_prior"]
    )
    for a, b in zip(
        first.weights + first.biases, changed.weights + changed.biases, strict=True
    ):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(first.mean, changed.mean)
    np.testing.assert_array_equal(first.scale, changed.scale)


def test_mixed_budget_metric_uses_mass_and_floored_neutral_fraction() -> None:
    """Late relative-neutral errors supplement actual absorbed-count errors."""
    report = budget_error_metrics(
        np.array([0.8, 0.99]),
        np.array([0.9, 0.999]),
        np.array([0.2, 1.0]),
        relative_weight=0.3,
        neutral_floor=0.01,
    )
    mass_error = np.array([-0.02, -0.009])
    relative = mass_error / np.array([0.82, 0.01])
    assert report["mass_mse"] == pytest.approx(np.mean(mass_error**2))
    assert report["floored_relative_neutral_mse"] == pytest.approx(np.mean(relative**2))
    assert report["mixed_loss"] == pytest.approx(
        np.mean(mass_error**2 + 0.3 * relative**2)
    )
    assert report["neutral_floor_active_count"] == 1


def test_deficit_budget_learns_without_positive_residual_plateau(
    tmp_path: Path,
) -> None:
    """A smooth deficit head learns small positive deficits and serializes them."""
    x = np.column_stack((np.linspace(0.2, 0.8, 12), np.full(12, 7.0)))
    eta = np.full(12, 0.995)
    model, report = fit_deficit_budget(
        x,
        eta,
        np.ones(12),
        x,
        eta,
        np.ones(12),
        seed=11,
        epochs=500,
        patience=100,
        learning_rate=0.02,
        hidden=(),
        relative_weight=0.2,
        neutral_floor=0.01,
    )
    direct = model.predict(x)
    predicted = predict_efficiency(model, x)
    assert np.all(direct * (1 - direct) > 0)
    np.testing.assert_array_equal(predicted, 1 - direct)
    np.testing.assert_allclose(predicted, eta, atol=2e-4)
    assert (
        model.metadata["budget_parameterization"] == "direct_sigmoid_absorption_deficit"
    )
    assert model.metadata["initial_deficit_prior"] == pytest.approx(0.005)
    assert report["validation"]["mixed_loss"] < 2e-6
    assert not report["test_used"]
    model.save(tmp_path / "deficit.npz")
    restored = load_budget(tmp_path / "deficit.npz")
    np.testing.assert_array_equal(predict_efficiency(restored, x), predicted)
    np.testing.assert_array_equal(restored.mean, x.mean(0))


def test_deficit_updates_and_normalizer_exclude_validation_labels() -> None:
    """Only checkpoint selection may depend on validation; first updates cannot."""
    x = _native_globals()
    eta = np.array([1.0, 0.999, 0.998, 0.97, 0.95, 0.9])
    options = {
        "seed": 8,
        "epochs": 1,
        "patience": 1,
        "learning_rate": 0.01,
        "relative_weight": 0.1,
        "neutral_floor": 0.01,
    }
    first, _ = fit_deficit_budget(
        x, eta, np.ones(6), x[:2], np.ones(2), np.ones(2), **options
    )
    changed, _ = fit_deficit_budget(
        x, eta, np.ones(6), x[:2] + 100, np.array([0.3, 0.2]), np.ones(2), **options
    )
    for a, b in zip(
        first.weights + first.biases, changed.weights + changed.biases, strict=True
    ):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(first.mean, x.mean(0))
    np.testing.assert_array_equal(first.mean, changed.mean)
    assert (
        first.metadata["initial_deficit_prior"]
        == changed.metadata["initial_deficit_prior"]
    )


@pytest.mark.parametrize(
    ("weight", "floor"),
    [(-1, 0.01), (np.nan, 0.01), (0.1, 0), (0.1, 1.1), (1.0, 1e-200)],
)
def test_mixed_budget_controls_reject_invalid_values(
    weight: float, floor: float
) -> None:
    """An undefined relative-neutral denominator must never enter optimization."""
    with pytest.raises(ValueError, match=r"relative|floor"):
        budget_error_metrics(
            np.ones(2),
            np.ones(2),
            np.ones(2),
            relative_weight=weight,
            neutral_floor=floor,
        )


def test_deficit_marker_cannot_reinterpret_residual_network() -> None:
    """The tag must not silently change the meaning of legacy bounded residuals."""
    model = constant_efficiency_model(2)
    model.metadata["budget_parameterization"] = "direct_sigmoid_absorption_deficit"
    with pytest.raises(ValueError, match="sigmoid"):
        predict_efficiency(model, np.array([[0.1, 7.0]]))


def test_opt_in_deficit_ranks_every_budget_with_same_mixed_metric() -> None:
    """Candidate-specific training losses cannot determine cross-model ranking."""
    x = _native_globals()
    eta = np.array([1, 1, 0.999, 0.97, 0.95, 0.995])
    models, report = fit_budget_v3(
        x,
        eta,
        np.ones(6),
        x[[1, 4]],
        eta[[1, 4]],
        np.ones(2),
        seed=11,
        epochs=2,
        patience=2,
        learning_rate=0.001,
        include_deficit=True,
        deficit_relative_weight=0.2,
        deficit_neutral_floor=0.03,
    )
    assert "deficit_mlp" in models
    expected = {}
    for name, model in models.items():
        expected[name] = budget_error_metrics(
            predict_efficiency(model, x[[1, 4]]),
            eta[[1, 4]],
            np.ones(2),
            relative_weight=0.2,
            neutral_floor=0.03,
        )["mixed_loss"]
    for record in report["candidates"]:
        assert record["validation_mixed_loss"] == pytest.approx(
            expected[record["kind"]]
        )
    assert report["ranking"] == sorted(expected, key=expected.get)
    assert report["selected_kind"] == min(expected, key=expected.get)


def test_deficit_network_survives_complete_emulator_reload(tmp_path: Path) -> None:
    """Existing model schemas preserve the new budget's tagged target meaning."""
    from research.pc_analytic_emulator.model import Emulator, field_features

    x = _native_globals()
    budget, _ = fit_deficit_budget(
        x,
        np.full(6, 0.99),
        np.ones(6),
        x[:2],
        np.full(2, 0.99),
        np.ones(2),
        seed=11,
        epochs=3,
        patience=3,
        learning_rate=0.001,
    )
    photons = np.full((4, 4, 4), 0.4)
    atoms = np.ones_like(photons)
    neutral = np.full_like(photons, 0.7)
    features, _, _ = field_features(photons, atoms, neutral, 7.0, 50.0)
    model = Emulator(constant_efficiency_model(features.shape[1]), budget)
    before = model.predict(photons, atoms, neutral, 7.0, 50.0)
    model.save(tmp_path / "model")
    restored = Emulator.load(tmp_path / "model")
    after = restored.predict(photons, atoms, neutral, 7.0, 50.0)
    for key in ("xHI", "xHII", "raw_xHI", "absorption_efficiency"):
        np.testing.assert_array_equal(before[key], after[key])
    assert after["absorption_efficiency"] > 0.9
    assert after["absorption_efficiency"] < 1.0
    assert after["photon_ledger"]["relative_photon_residual"] < 1e-10


def test_deficit_opt_in_photon_free_case_retains_exact_reference() -> None:
    """An empty photon budget needs no learned deficit or zero-weight fitting."""
    x = _native_globals()
    models, report = fit_budget_v3(
        x,
        np.ones(6),
        np.zeros(6),
        x[:2],
        np.ones(2),
        np.zeros(2),
        seed=11,
        epochs=1,
        patience=1,
        learning_rate=0.001,
        include_deficit=True,
    )
    assert "deficit_mlp" not in models
    assert report["selected_kind"] == "constant_efficiency"
    assert all(row["validation_mixed_loss"] == 0 for row in report["candidates"])


def test_budget_metric_rejects_complex_values() -> None:
    """Complex labels cannot be silently converted into real photon budgets."""
    with pytest.raises(ValueError, match="real numeric"):
        budget_error_metrics(np.array([0.5 + 0.2j]), np.ones(1), np.ones(1))
