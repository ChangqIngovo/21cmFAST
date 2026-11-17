"""Geometry-conditioned budgets preserve measured finite escape and endpoints."""

from pathlib import Path

import numpy as np
import pytest

from research.pc_analytic_emulator.budget import fit_budget_v3, load_budget
from research.pc_analytic_emulator.geometry_budget import GeometryTabulatedBudgetModel


def _data() -> tuple[np.ndarray, np.ndarray]:
    features = np.zeros((8, 14))
    features[:, 0] = [0.2, 0.3, 1.0, 1.1] * 2
    features[:, 1] = [10, 10, 6, 6] * 2
    features[:4, 2:4] = [1, 3]
    features[4:, 2:4] = [2, 5]
    return features, np.array([1, 1, 0.8, 0.9, 1, 1, 1, 1])


def test_geometry_budget_keeps_finite_escape_and_exact_endpoint(tmp_path: Path) -> None:
    """High photon load cannot force full absorption in a finite-escape domain."""
    x, y = _data()
    model = GeometryTabulatedBudgetModel.from_training(x, y)
    query = x[[3, 7]].copy()
    query[:, 0] = 2
    np.testing.assert_allclose(model.predict(query), [0.85, 1], atol=1e-15)
    model.save(tmp_path / "budget.npz")
    loaded = load_budget(tmp_path / "budget.npz")
    assert isinstance(loaded, GeometryTabulatedBudgetModel)
    np.testing.assert_array_equal(loaded.predict(query), model.predict(query))
    assert not loaded.knots.flags.writeable


def test_geometry_budget_unknown_geometry_uses_original_training_fallback() -> None:
    """Unseen geometry uses the old global table; no geometric extrapolation."""
    x, y = _data()
    model = GeometryTabulatedBudgetModel.from_training(x, y)
    query = x[[3]].copy()
    query[:, 0] = 2
    query[:, 2:4] = [1.5, 4]
    expected = model.predict(query)
    np.testing.assert_allclose(expected, [0.925], atol=1e-15)
    query[:, 2:4] = [9, 10]
    np.testing.assert_array_equal(model.predict(query), expected)
    query[:, 1] = 100
    np.testing.assert_array_equal(model.predict(query), expected)


def test_geometry_budget_candidate_is_opt_in_and_training_only() -> None:
    """Validation label changes do not alter geometry knots or efficiencies."""
    x, y = _data()
    options = {"seed": 1, "epochs": 1, "patience": 1, "learning_rate": 1e-3}
    default, _ = fit_budget_v3(x, y, np.ones(8), x, y, np.ones(8), **options)
    assert "geometry_tabulated" not in default
    fitted, report = fit_budget_v3(
        x, y, np.ones(8), x, y, np.ones(8), include_geometry_tabulated=True, **options
    )
    changed, _ = fit_budget_v3(
        x,
        y,
        np.ones(8),
        x,
        np.ones(8),
        np.ones(8),
        include_geometry_tabulated=True,
        **options,
    )
    a, b = fitted["geometry_tabulated"], changed["geometry_tabulated"]
    np.testing.assert_array_equal(a.knots, b.knots)
    np.testing.assert_array_equal(a.efficiencies, b.efficiencies)
    assert report["geometry_table_fit_scope"].startswith("Training")


@pytest.mark.parametrize("bad", [np.nan, -1.0])
def test_geometry_budget_rejects_invalid_geometry(bad: float) -> None:
    """Malformed coordinates cannot silently select a valid neighboring table."""
    x, y = _data()
    model = GeometryTabulatedBudgetModel.from_training(x, y)
    x[0, 2] = bad
    with pytest.raises(ValueError, match="Invalid"):
        model.predict(x)
    with pytest.raises(ValueError, match="Invalid"):
        GeometryTabulatedBudgetModel.from_training(x, y)


def test_geometry_budget_emulator_roundtrip(tmp_path: Path) -> None:
    """Both recursive heads retain table predictions and the independent ledger."""
    from research.pc_analytic_emulator.model import Emulator, budget_features
    from research.pc_analytic_emulator.test_model import fixture_model, residual_model

    source = np.full((4, 4, 4), 0.4)
    capacity = np.ones_like(source)
    es = np.linspace(0, 1, source.size).reshape(source.shape)
    features, _ = budget_features(source, capacity, es, 7.5, 256)
    base = fixture_model(source, capacity, es)
    base.budget_model = GeometryTabulatedBudgetModel.from_training(
        np.repeat(features, 3, axis=0), np.array([0.8, 0.9, 1.0])
    )
    model = residual_model(base)
    before = model.predict(source, capacity, es, 7.5, 256)
    model.save(tmp_path)
    restored = Emulator.load(tmp_path)
    after = restored.predict(source, capacity, es, 7.5, 256)
    assert isinstance(restored.budget_model, GeometryTabulatedBudgetModel)
    assert isinstance(restored.base_model.budget_model, GeometryTabulatedBudgetModel)
    np.testing.assert_array_equal(before["xHII"], after["xHII"])
    assert before["photon_ledger"] == after["photon_ledger"]
    assert after["absorption_efficiency"] == pytest.approx(0.9)


def test_unknown_geometry_matches_legacy_table_at_every_load() -> None:
    """The explicit fallback reproduces the prior interpolation exactly."""
    from research.pc_analytic_emulator.budget import TabulatedBudgetModel

    x, y = _data()
    x[4:, 0] *= 1.4
    model = GeometryTabulatedBudgetModel.from_training(x, y)
    groups = [x[:, 1] == z for z in np.unique(x[:, 1])]
    coords = np.array([np.mean(x[g, 0]) for g in groups])
    values = np.array([np.mean(y[g]) for g in groups])
    order = np.argsort(coords)
    legacy = TabulatedBudgetModel(coords[order], values[order], x.shape[1])
    queries = np.zeros((11, 14))
    queries[:, 0] = [0, 0.1, 0.2, 0.5, 0.7, 0.9, 1, 1.5, 2, 10, 1e300]
    queries[:, 1] = 6
    queries[:, 2:4] = [9, 20]
    np.testing.assert_array_equal(model.predict(queries), legacy.predict(queries))


def test_geometry_budget_rejects_corrupted_offsets() -> None:
    """Invalid flat-table boundaries cannot silently change the geometry map."""
    x, y = _data()
    model = GeometryTabulatedBudgetModel.from_training(x, y)
    with pytest.raises(ValueError, match="Invalid geometry"):
        GeometryTabulatedBudgetModel(
            model.geometries,
            np.array([0, 0, len(model.knots)]),
            model.knots,
            model.efficiencies,
            model.feature_count,
            model.fallback_knots,
            model.fallback_efficiencies,
        )
