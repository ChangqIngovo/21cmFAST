"""Check the empirical deficit kernel's units, bounds and stored fit identity."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from research.pc_analytic_emulator.budget import (
    budget_error_metrics,
    predict_efficiency,
)
from research.pc_analytic_emulator.deficit_kernel import (
    NATIVE_PREFIX,
    KernelDeficitBudgetModel,
)
from research.pc_analytic_emulator.model import budget_features


def native_features() -> tuple[np.ndarray, list[str]]:
    """Create genuine native-input summaries with known photon-load variation."""
    rng = np.random.default_rng(14)
    atoms = 0.5 + rng.random((4, 4, 4))
    rows = []
    for load in (0.2, 0.5, 1.0, 2.0):
        source = load * atoms * (0.5 + rng.random(atoms.shape))
        neutral = rng.random(atoms.shape)
        features, names = budget_features(source, atoms, neutral, 7, 50)
        rows.append(features)
    return np.concatenate(rows), names


def fit(features: np.ndarray, names: list[str], eta: np.ndarray, **kwargs):
    """Fit one fixed small kernel for numerical-contract tests."""
    return KernelDeficitBudgetModel.from_training(
        features,
        eta,
        feature_names=names,
        feature_indices=[0, 6, 7, 8, 9],
        bandwidth=0.5,
        ridge=0.01,
        **kwargs,
    )


@pytest.mark.parametrize("transform", ["log", "logit"])
def test_serialization_preserves_inference_and_training_state(
    tmp_path: Path, transform: str
) -> None:
    """A restored fit yields bitwise-identical predictions without retraining."""
    features, names = native_features()
    original = features.copy()
    model = fit(features, names, np.array([1, 0.9, 0.8, 0.7]), transform=transform)
    query = features.copy()
    query[:, 0] *= 1.05
    predicted = model.predict(query)
    model.save(tmp_path / "kernel.npz")
    restored = KernelDeficitBudgetModel.load(tmp_path / "kernel.npz")
    np.testing.assert_array_equal(restored.predict(query), predicted)
    np.testing.assert_array_equal(predict_efficiency(restored, query), predicted)
    np.testing.assert_array_equal(features, original)
    assert not restored.training_features.flags.writeable
    assert not restored.mean.flags.writeable
    assert tuple(names[:10]) == NATIVE_PREFIX


def test_photon_surplus_does_not_force_zero_deficit() -> None:
    """Finite empirical deficit survives P/C>1 and changed redshift labels."""
    features, names = native_features()
    model = fit(features, names, np.full(len(features), 0.8))
    query = features.copy()
    query[:, 0] = np.log1p(3)
    query[:, 1] = 0
    np.testing.assert_allclose(model.predict_deficit(query), 0.2, atol=1e-15)
    np.testing.assert_allclose(model.predict(query), 0.8, atol=1e-15)


def test_deficit_is_not_mass_neutral_fraction_below_photon_saturation() -> None:
    """The target obeys Q=s*eta, rather than equating d with neutral mass."""
    metrics = budget_error_metrics(np.array([0.7]), np.array([0.8]), np.array([0.5]))
    assert metrics["mass_MAE"] == pytest.approx(0.05)
    assert metrics["mixed_loss"] == pytest.approx(0.0031944444444444446)
    zero = budget_error_metrics(np.array([0.1]), np.array([1.0]), np.array([0.0]))
    assert zero["mixed_loss"] == 0


def test_common_photon_atom_unit_rescaling_preserves_native_features() -> None:
    """A common change of count units cannot alter the inference input."""
    rng = np.random.default_rng(20)
    atoms = 0.1 + rng.random((4, 4, 4))
    photons = 2 * rng.random(atoms.shape)
    neutral = rng.random(atoms.shape)
    first, names = budget_features(photons, atoms, neutral, 7, 50)
    second, other_names = budget_features(photons * 1e10, atoms * 1e10, neutral, 7, 50)
    assert names == other_names
    np.testing.assert_allclose(first, second, atol=1e-14, rtol=1e-14)


def test_normalizer_and_coefficients_are_unchanged_by_queries() -> None:
    """Validation or distant queries cannot update the training normalizer."""
    features, names = native_features()
    model = fit(features, names, np.array([1.0, 0.9, 0.8, 0.7]))
    mean, scale, coefficients = (
        model.mean.copy(),
        model.scale.copy(),
        model.coefficients.copy(),
    )
    query = features.copy()
    query[:, 0] += 100
    predicted = model.predict_deficit(query)
    assert np.isfinite(predicted).all()
    assert np.all((predicted >= 0) & (predicted <= 1))
    np.testing.assert_array_equal(model.mean, mean)
    np.testing.assert_array_equal(model.scale, scale)
    np.testing.assert_array_equal(model.coefficients, coefficients)
    expected_mean = np.mean(features[:, model.feature_indices], axis=0)
    np.testing.assert_array_equal(model.mean, expected_mean)


def test_target_floor_does_not_floor_predictions() -> None:
    """Zero-label regularization is distinct from an output floor or closure."""
    features, names = native_features()
    model = fit(features, names, np.ones(4), transform="log")
    np.testing.assert_allclose(model.predict_deficit(features), 1e-10, rtol=1e-14)
    small = replace(model, target_offset=np.log(1e-14))
    assert np.all(small.predict_deficit(features) < model.target_floor)
    assert np.all(small.predict_deficit(features) > 0)


@pytest.mark.parametrize("bad", [np.nan, -0.1, 1 + 1j])
def test_invalid_native_inputs_fail_before_inference(bad: complex) -> None:
    """Nonreal or nonphysical budget features cannot be silently coerced."""
    features, names = native_features()
    model = fit(features, names, np.full(4, 0.8))
    changed = features.astype(complex if isinstance(bad, complex) else float)
    changed[0, 0] = bad
    with pytest.raises(ValueError, match=r"finite real|physical range"):
        model.predict(changed)


def test_swapped_native_schema_is_rejected() -> None:
    """A feature-count match cannot mask swapped photon-load/redshift units."""
    features, names = native_features()
    names[0], names[1] = names[1], names[0]
    with pytest.raises(ValueError, match="feature schema"):
        fit(features, names, np.full(4, 0.8))


def test_corrupt_serialized_fit_is_rejected(tmp_path: Path) -> None:
    """Nonfinite coefficients cannot enter a restored inference model."""
    features, names = native_features()
    model = fit(features, names, np.full(4, 0.8))
    path = tmp_path / "kernel.npz"
    model.save(path)
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: np.array(saved[key]) for key in saved.files}
    arrays["coefficients"][0] = np.nan
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="finite real"):
        KernelDeficitBudgetModel.load(path)
