"""Verify diagnostic-only budget interventions without modifying deployment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from research.pc_analytic_emulator.diagnose_errors import decompose_snapshot
from research.pc_analytic_emulator.model import Emulator, field_features
from research.pc_analytic_emulator.test_model import (
    constant_model,
    fixture_model,
    residual_model,
)

METHODS = (
    "deployed",
    "oracle_final_budget",
    "oracle_recursive_budget",
    "teacher_predicted_budget",
)


def _stored_arrays(root: Path) -> dict[str, dict[str, np.ndarray] | dict]:
    records = {}
    for path in sorted(root.rglob("*")):
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                records[str(path.relative_to(root))] = {
                    name: archive[name].copy() for name in archive.files
                }
        elif path.suffix == ".json":
            records[str(path.relative_to(root))] = json.loads(path.read_text())
    return records


def _zero_field_model(
    source: np.ndarray,
    capacity: np.ndarray,
    neutral: np.ndarray,
    projection_mode: str,
    efficiency: float,
) -> Emulator:
    model = fixture_model(source, capacity, neutral, efficiency=efficiency)
    features, _, _ = field_features(source, capacity, neutral, 7.5, 256)
    model.field_model = constant_model(features.shape[1], 0)
    model.projection_mode = projection_mode
    return model


@pytest.mark.parametrize("projection_mode", ["log_odds", "capacity_l2"])
def test_exact_teacher_budget_is_an_identity_without_parameter_mutation(
    tmp_path: Path, projection_mode: str
) -> None:
    """Correct budgets preserve all fields and recursively saved model weights."""
    source = np.full((4, 4, 4), 0.6)
    capacity = np.linspace(0.7, 1.3, source.size).reshape(source.shape)
    neutral = np.linspace(0.15, 0.85, source.size).reshape(source.shape)
    base = _zero_field_model(source, capacity, neutral, projection_mode, 0.8)
    model = residual_model(base)
    teacher = model.predict(source, capacity, neutral, 7.5, 256)["xHI"]
    model.save(tmp_path / "before")
    before = _stored_arrays(tmp_path / "before")

    diagnosed = decompose_snapshot(model, source, capacity, neutral, teacher, 7.5, 256)

    assert set(diagnosed["fields"]) == set(METHODS)
    for field in diagnosed["fields"].values():
        np.testing.assert_allclose(field, teacher, rtol=0, atol=2e-12)
    np.testing.assert_array_equal(
        diagnosed["latent_xHII"]["deployed"],
        diagnosed["latent_xHII"]["oracle_final_budget"],
    )
    model.save(tmp_path / "after")
    after = _stored_arrays(tmp_path / "after")
    assert set(after) == set(before)
    for filename in before:
        assert set(after[filename]) == set(before[filename])
        if filename.endswith(".json"):
            assert after[filename] == before[filename]
        else:
            for name in before[filename]:
                np.testing.assert_array_equal(
                    after[filename][name], before[filename][name]
                )


@pytest.mark.parametrize("projection_mode", ["log_odds", "capacity_l2"])
def test_recursive_oracle_recovers_morphology_lost_by_saturated_frozen_base(
    projection_mode: str,
) -> None:
    """Changing only the final count cannot reconstruct a saturated baseline."""
    source = np.full((4, 4, 4), 2.0)
    capacity = np.ones_like(source)
    teacher = np.linspace(0.1, 0.5, source.size).reshape(source.shape)
    base = _zero_field_model(source, capacity, teacher, projection_mode, 1.0)
    model = residual_model(base)

    result = decompose_snapshot(model, source, capacity, teacher, teacher, 7.5, 256)

    np.testing.assert_array_equal(result["fields"]["deployed"], 0)
    np.testing.assert_array_equal(result["fields"]["teacher_predicted_budget"], 0)
    np.testing.assert_allclose(
        result["fields"]["oracle_final_budget"], np.mean(teacher), rtol=0, atol=2e-12
    )
    np.testing.assert_allclose(
        result["fields"]["oracle_recursive_budget"], teacher, rtol=0, atol=2e-12
    )
    np.testing.assert_array_equal(result["latent_xHII"]["deployed"], 1)
    np.testing.assert_array_equal(result["latent_xHII"]["oracle_final_budget"], 1)
    assert np.std(result["latent_xHII"]["oracle_recursive_budget"]) > 0.1
    assert result["budgets"]["oracle_final_budget"][
        "target_absorbed_photons"
    ] == pytest.approx(result["teacher_absorbed_photons"], abs=1e-12)


@pytest.mark.parametrize("source_per_atom", [0.4, 2.0])
def test_all_counterfactuals_have_independently_checked_photon_inventories(
    source_per_atom: float,
) -> None:
    """Fixed target and unabsorbed reservoirs close for low and surplus supply."""
    source = np.full((4, 4, 4), source_per_atom)
    capacity = np.ones_like(source)
    native = np.linspace(0.1, 0.9, source.size).reshape(source.shape)
    teacher = np.linspace(0.65, 0.95, source.size).reshape(source.shape)
    model = _zero_field_model(source, capacity, native, "capacity_l2", 0.75)
    result = decompose_snapshot(model, source, capacity, native, teacher, 7.5, 256)
    total_source = np.sum(source, dtype=np.longdouble)
    teacher_absorbed = np.sum(
        capacity.astype(np.longdouble) * (1 - teacher.astype(np.longdouble)),
        dtype=np.longdouble,
    )
    assert result["teacher_absorbed_photons"] == pytest.approx(
        float(teacher_absorbed), abs=1e-12
    )
    expected_counts = {
        "deployed": min(float(total_source), capacity.sum()) * 0.75,
        "oracle_final_budget": float(teacher_absorbed),
        "oracle_recursive_budget": float(teacher_absorbed),
        "teacher_predicted_budget": min(float(total_source), capacity.sum()) * 0.75,
    }
    for method, field in result["fields"].items():
        assert np.isfinite(field).all()
        assert np.all((field >= 0) & (field <= 1))
        measured = np.sum(
            capacity.astype(np.longdouble) * (1 - field.astype(np.longdouble))
        )
        ledger = result["budgets"][method]
        assert ledger["target_absorbed_photons"] == pytest.approx(
            expected_counts[method], abs=1e-12
        )
        assert ledger["field_absorbed_photons"] == pytest.approx(
            float(measured), abs=1e-12
        )
        assert ledger["unabsorbed_photons"] >= 0
        assert (
            abs(measured + ledger["unabsorbed_photons"] - total_source) / total_source
            < 1e-12
        )
        assert ledger["relative_photon_residual"] < 1e-12
    np.testing.assert_allclose(
        result["fields"]["oracle_final_budget"],
        result["fields"]["oracle_recursive_budget"],
        rtol=0,
        atol=2e-12,
    )


def test_oracle_rejects_more_teacher_absorption_than_available_photons() -> None:
    """Labels outside the physical supply cannot masquerade as an exact budget."""
    source = np.full((4, 4, 4), 0.1)
    capacity = np.ones_like(source)
    native = np.full_like(source, 0.9)
    model = _zero_field_model(source, capacity, native, "capacity_l2", 0.75)
    with pytest.raises(ValueError, match=r"(?i)(photon|absorp|supply|budget)"):
        decompose_snapshot(
            model, source, capacity, native, np.zeros_like(source), 7.5, 256
        )


@pytest.mark.parametrize("value", [-0.1, 1.1, np.nan])
def test_invalid_teacher_fractions_fail_before_oracle_projection(value: float) -> None:
    """An oracle intervention still requires finite bounded teacher fractions."""
    source = np.full((4, 4, 4), 2.0)
    capacity = np.ones_like(source)
    native = np.full_like(source, 0.5)
    model = _zero_field_model(source, capacity, native, "capacity_l2", 0.75)
    teacher = native.copy()
    teacher.flat[0] = value
    with pytest.raises(ValueError, match=r"(?i)(teacher|neutral|fraction)"):
        decompose_snapshot(model, source, capacity, native, teacher, 7.5, 256)


def test_teacher_grid_must_match_native_inputs() -> None:
    """Broadcastable labels are rejected rather than silently repeated."""
    source = np.full((4, 4, 4), 2.0)
    capacity = np.ones_like(source)
    native = np.full_like(source, 0.5)
    model = _zero_field_model(source, capacity, native, "capacity_l2", 0.75)
    with pytest.raises(ValueError, match=r"(?i)(shape|grid|cubic|teacher)"):
        decompose_snapshot(
            model, source, capacity, native, np.full((4, 4, 1), 0.5), 7.5, 256
        )


@pytest.mark.parametrize("projection_mode", ["capacity_l2", "log_odds"])
def test_budget_only_lower_bound_is_restricted_to_capacity_weighted_mse(
    projection_mode: str,
) -> None:
    """The L2 oracle minimizes weighted error at fixed count, not every metric."""
    source = np.full((4, 4, 4), 2.0)
    capacity = np.linspace(0.7, 1.3, source.size).reshape(source.shape)
    teacher = np.linspace(0.1, 0.9, source.size).reshape(source.shape)
    native = teacher[::-1].copy()
    model = _zero_field_model(source, capacity, native, projection_mode, 0.75)

    result = decompose_snapshot(model, source, capacity, native, teacher, 7.5, 256)

    assert result["teacher_projection_is_weighted_MSE_lower_bound"] is (
        projection_mode == "capacity_l2"
    )
    if projection_mode != "capacity_l2":
        return
    budget_only = result["fields"]["teacher_predicted_budget"]
    bounded_error = np.sum(capacity * (budget_only - teacher) ** 2)
    deployed_error = np.sum(capacity * (result["fields"]["deployed"] - teacher) ** 2)
    assert bounded_error <= deployed_error + 1e-12
    free = np.flatnonzero((budget_only > 0.1) & (budget_only < 0.9))
    assert len(free) >= 2
    first, second = free[:2]
    perturbed = budget_only.copy()
    perturbed.flat[first] += 0.02 / capacity.flat[first]
    perturbed.flat[second] -= 0.02 / capacity.flat[second]
    np.testing.assert_allclose(
        np.sum(capacity * perturbed), np.sum(capacity * budget_only), rtol=0, atol=1e-12
    )
    assert np.sum(capacity * (perturbed - teacher) ** 2) > bounded_error + 1e-6
