"""Check input-only features and independently audited budget constraints."""

import inspect
from pathlib import Path

import numpy as np
import pytest

from research.pc_analytic_emulator.model import (
    Emulator,
    absorption_efficiency,
    budget_features,
    field_features,
)
from research.pc_field_emulator.model import MLPModel


def constant_model(count: int, amplitude: float) -> MLPModel:
    """Create a constant residual response without fitting any labels."""
    return MLPModel(
        [np.zeros((count, 1))],
        [np.array([np.arctanh(amplitude)])],
        np.zeros(count),
        np.ones(count),
        {},
        output_mode="residual",
    )


def fixture_model(
    source: np.ndarray, capacity: np.ndarray, es: np.ndarray, efficiency: float = 0.75
) -> Emulator:
    """Build independent constant field and efficiency heads for checks."""
    field, _, _ = field_features(source, capacity, es, 7.5, 256)
    budget, _ = budget_features(source, capacity, es, 7.5, 256)
    return Emulator(
        constant_model(field.shape[1], 0.1),
        constant_model(budget.shape[1], efficiency - 1),
    )


def test_features_are_input_only_and_include_es() -> None:
    """Expose native ES structure without accepting target fields."""
    source = np.arange(64).reshape(4, 4, 4) / 64
    capacity = np.ones_like(source)
    es = np.full_like(source, 0.7)
    fields, names, baseline = field_features(source, capacity, es, 7.5, 256)
    box, box_names = budget_features(source, capacity, es, 7.5, 256)
    assert fields.shape == (64, 24)
    assert len(names) == 24
    assert box.shape == (1, len(box_names))
    np.testing.assert_allclose(baseline, 0.3, rtol=0, atol=1e-15)
    assert names[-5] == "native_es_xHII"
    for function in (field_features, budget_features, Emulator.predict):
        assert "teacher_xHI" not in inspect.signature(function).parameters
    changed, _, _ = field_features(source, capacity, es * 0, 7.5, 256)
    assert not np.array_equal(fields, changed)


@pytest.mark.parametrize("source_per_atom", [0.0, 0.4, 2.0])
def test_prediction_has_a_bounded_independent_photon_ledger(
    source_per_atom: float,
) -> None:
    """Close predicted reservoirs for empty, partially ionized and surplus boxes."""
    capacity = np.linspace(0.5, 1.5, 64).reshape(4, 4, 4)
    source = capacity * source_per_atom
    es = np.full_like(source, 0.6)
    model = fixture_model(source, capacity, es)
    result = model.predict(source, capacity, es, 7.5, 256)
    emitted = np.sum(source, dtype=np.longdouble)
    target = min(emitted, np.sum(capacity, dtype=np.longdouble)) * 0.75
    absorbed = np.sum(capacity * result["xHII"], dtype=np.longdouble)
    stored_unabsorbed = result["predicted_unabsorbed_photons"]
    assert np.all((result["xHI"] >= 0) & (result["xHI"] <= 1))
    assert abs(absorbed - target) <= 2e-12 * max(float(target), 1)
    assert abs(emitted - absorbed - stored_unabsorbed) <= 2e-12 * max(float(emitted), 1)
    assert result["photon_ledger"]["passed"]
    assert result["photon_ledger"]["relative_photon_residual"] < 1e-10
    if not source_per_atom:
        np.testing.assert_array_equal(result["xHI"], 1)


def test_reservoir_is_predicted_before_field_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deliberately wrong projected field must fail the photon ledger."""
    import research.pc_analytic_emulator.model as implementation

    source = np.full((4, 4, 4), 0.4)
    capacity = np.ones_like(source)
    es = np.full_like(source, 0.6)
    model = fixture_model(source, capacity, es)
    monkeypatch.setattr(
        implementation,
        "project_global_budget",
        lambda *args, **kwargs: (
            np.zeros_like(source),
            {"scope": "deliberate broken field"},
        ),
    )
    with pytest.raises(FloatingPointError, match="photon ledger"):
        model.predict(source, capacity, es, 7.5, 256)


def test_absorption_target_preserves_escape_and_capacity_limits() -> None:
    """Supervise actual field absorption rather than force all photons to absorb."""
    source = np.full((4, 4, 4), 0.8)
    capacity = np.ones_like(source)
    assert absorption_efficiency(
        source, capacity, np.full_like(source, 0.6)
    ) == pytest.approx(0.5)
    assert absorption_efficiency(
        source * 4, capacity, np.zeros_like(source)
    ) == pytest.approx(1)
    assert absorption_efficiency(source * 0, capacity, np.ones_like(source)) == 1
    with pytest.raises(ValueError, match="exceeds"):
        absorption_efficiency(source, capacity, np.zeros_like(source))


def test_model_roundtrip(tmp_path: Path) -> None:
    """Reproduce identical field and ledger after saving both heads."""
    source = np.full((4, 4, 4), 0.4)
    capacity = np.ones_like(source)
    es = np.full_like(source, 0.6)
    model = fixture_model(source, capacity, es)
    model.save(tmp_path)
    restored = Emulator.load(tmp_path)
    before = model.predict(source, capacity, es, 7.5, 256)
    after = restored.predict(source, capacity, es, 7.5, 256)
    np.testing.assert_array_equal(before["xHI"], after["xHI"])
    assert before["photon_ledger"] == after["photon_ledger"]


@pytest.mark.parametrize("bad", [np.nan, -0.1, 1.1])
def test_bad_native_es_rejected(bad: float) -> None:
    """Reject nonfinite or unbounded inference-time ES fields."""
    source = np.ones((4, 4, 4))
    with pytest.raises(ValueError, match="native ES"):
        field_features(source, source, np.full_like(source, bad), 7.5, 256)
