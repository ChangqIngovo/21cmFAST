"""Check input-only features and independently audited budget constraints."""

import inspect
import json
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
@pytest.mark.parametrize("projection_mode", ["log_odds", "capacity_l2"])
def test_prediction_has_a_bounded_independent_photon_ledger(
    source_per_atom: float, projection_mode: str
) -> None:
    """Close predicted reservoirs for empty, partially ionized and surplus boxes."""
    capacity = np.linspace(0.5, 1.5, 64).reshape(4, 4, 4)
    source = capacity * source_per_atom
    es = np.full_like(source, 0.6)
    model = fixture_model(source, capacity, es)
    model.projection_mode = projection_mode
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


@pytest.mark.parametrize("projection_mode", ["log_odds", "capacity_l2"])
def test_model_roundtrip(tmp_path: Path, projection_mode: str) -> None:
    """Reproduce identical field and ledger after saving both heads."""
    source = np.full((4, 4, 4), 0.4)
    capacity = np.ones_like(source)
    es = np.full_like(source, 0.6)
    model = fixture_model(source, capacity, es)
    model.projection_mode = projection_mode
    model.save(tmp_path)
    restored = Emulator.load(tmp_path)
    before = model.predict(source, capacity, es, 7.5, 256)
    after = restored.predict(source, capacity, es, 7.5, 256)
    np.testing.assert_array_equal(before["xHI"], after["xHI"])
    assert before["photon_ledger"] == after["photon_ledger"]
    assert restored.projection_mode == projection_mode


def test_original_schema_loads_with_unchanged_log_odds_inference(
    tmp_path: Path,
) -> None:
    """Restore existing schema-one MLP artifacts without changing their predictions."""
    source = np.full((4, 4, 4), 0.4)
    capacity = np.ones_like(source)
    es = np.full_like(source, 0.6)
    model = fixture_model(source, capacity, es)
    model.save(tmp_path)
    record = json.loads((tmp_path / "model.json").read_text())
    record["schema"] = 1
    for name in ("projection_mode", "budget_kind", "field_kind"):
        record.pop(name)
    (tmp_path / "model.json").write_text(json.dumps(record))
    restored = Emulator.load(tmp_path)
    assert restored.projection_mode == "log_odds"
    np.testing.assert_array_equal(
        restored.predict(source, capacity, es, 7.5, 256)["xHI"],
        model.predict(source, capacity, es, 7.5, 256)["xHI"],
    )


def test_tabulated_budget_roundtrip_retains_partial_absorption(tmp_path: Path) -> None:
    """Load a tagged table budget without substituting an all-absorption target."""
    from research.pc_analytic_emulator.budget import TabulatedBudgetModel

    source = np.full((4, 4, 4), 0.4)
    capacity = np.ones_like(source)
    es = np.full_like(source, 0.6)
    model = fixture_model(source, capacity, es)
    features, _ = budget_features(source, capacity, es, 7.5, 256)
    model.budget_model = TabulatedBudgetModel(
        np.array([0.0, 1.0]), np.array([0.73, 0.73]), features.shape[1]
    )
    model.projection_mode = "capacity_l2"
    model.save(tmp_path)
    restored = Emulator.load(tmp_path)
    result = restored.predict(source, capacity, es, 7.5, 256)
    assert result["absorption_efficiency"] == pytest.approx(0.73)
    assert result["predicted_unabsorbed_photons"] == pytest.approx(0.27 * source.sum())
    assert result["photon_ledger"]["passed"]
    assert (
        json.loads((tmp_path / "model.json").read_text())["budget_kind"] == "tabulated"
    )


def test_latent_projection_roundtrip_retains_neutral_saturation(tmp_path: Path) -> None:
    """Project unbounded scores rather than prematurely clipping their ordering."""
    from research.pc_analytic_emulator.latent import LatentFieldModel
    from research.pc_analytic_emulator.projection import project_capacity_l2

    source = np.full((4, 4, 4), 0.4)
    capacity = np.ones_like(source)
    es = np.linspace(0, 1, source.size).reshape(source.shape)
    model = fixture_model(source, capacity, es)
    features, _, baseline = field_features(source, capacity, es, 7.5, 256)
    model.field_model = LatentFieldModel(
        weights=[np.zeros((features.shape[1], 1))],
        biases=[np.array([-0.6])],
        mean=np.zeros(features.shape[1]),
        scale=np.ones(features.shape[1]),
        metadata={},
    )
    model.projection_mode = "capacity_l2"
    model.save(tmp_path)
    restored = Emulator.load(tmp_path)
    result = restored.predict(source, capacity, es, 7.5, 256)
    expected, _ = project_capacity_l2(
        (baseline - 0.6).reshape(source.shape),
        capacity,
        result["predicted_absorbed_photons"],
    )
    premature, _ = project_capacity_l2(
        result["raw_xHII"], capacity, result["predicted_absorbed_photons"]
    )
    np.testing.assert_array_equal(result["xHII"], expected)
    assert np.any(result["latent_xHII"] < 0)
    assert np.any(result["xHII"] == 0)
    assert np.all(premature > 0)
    assert result["photon_ledger"]["passed"]
    assert result["projection"]["field_kind"] == "latent_linear"
    assert (
        json.loads((tmp_path / "model.json").read_text())["field_kind"]
        == "latent_linear"
    )
    restored.projection_mode = "log_odds"
    with pytest.raises(ValueError, match="require capacity_l2"):
        restored.predict(source, capacity, es, 7.5, 256)


def residual_model(base: Emulator, feature_count: int = 24) -> Emulator:
    """Wrap a deployed emulator in an initially zero linear residual."""
    from research.pc_analytic_emulator.latent import LatentFieldModel

    correction = LatentFieldModel(
        [np.zeros((feature_count, 1))],
        [np.zeros(1)],
        np.zeros(feature_count),
        np.ones(feature_count),
        {},
    )
    return Emulator(
        correction,
        base.budget_model,
        radii_cMpc=base.radii_cMpc,
        projection_mode="capacity_l2",
        base_model=base,
    )


def test_local_deficit_budget_roundtrip_in_recursive_base(tmp_path: Path) -> None:
    """Restore new budget heads at both levels without losing finite escape."""
    from research.pc_analytic_emulator.budget import LocalDeficitBudgetModel

    source = np.full((4, 4, 4), 0.4)
    capacity = np.ones_like(source)
    es = np.linspace(0, 1, source.size).reshape(source.shape)
    features, _ = budget_features(source, capacity, es, 7.5, 256)
    base = fixture_model(source, capacity, es)
    base.budget_model = LocalDeficitBudgetModel.from_training(
        np.repeat(features, 3, axis=0), np.array([0.8, 0.9, 1.0])
    )
    model = residual_model(base)
    before = model.predict(source, capacity, es, 7.5, 256)
    model.save(tmp_path)
    restored = Emulator.load(tmp_path)
    after = restored.predict(source, capacity, es, 7.5, 256)

    assert isinstance(restored.budget_model, LocalDeficitBudgetModel)
    assert isinstance(restored.base_model.budget_model, LocalDeficitBudgetModel)
    for directory in (tmp_path, tmp_path / "base_model"):
        assert (
            json.loads((directory / "model.json").read_text())["budget_kind"]
            == "local_deficit"
        )
    np.testing.assert_array_equal(before["xHII"], after["xHII"])
    assert before["photon_ledger"] == after["photon_ledger"]
    assert after["absorption_efficiency"] == pytest.approx(0.9)
    assert after["predicted_unabsorbed_photons"] == pytest.approx(0.1 * source.sum())


@pytest.mark.parametrize("projection_mode", ["log_odds", "capacity_l2"])
@pytest.mark.parametrize(
    ("source_per_atom", "efficiency"), [(0, 1), (0.7, 0.75), (2, 1)]
)
def test_zero_residual_retains_complete_deployed_base(
    projection_mode: str, source_per_atom: float, efficiency: float
) -> None:
    """Retain the base's budget and spatial field, including endpoint cases."""
    capacity = np.linspace(0.5, 1.5, 64).reshape(4, 4, 4)
    source = source_per_atom * capacity
    es = np.linspace(0, 1, 64).reshape(source.shape)
    base = fixture_model(source, capacity, es, efficiency=efficiency)
    base.projection_mode = projection_mode
    model = residual_model(base)
    expected = base.predict(source, capacity, es, 7.5, 256)
    result = model.predict(source, capacity, es, 7.5, 256)
    np.testing.assert_array_equal(result["raw_xHII"], expected["xHII"])
    np.testing.assert_allclose(result["xHII"], expected["xHII"], rtol=0, atol=2e-12)
    assert result["absorption_efficiency"] == expected["absorption_efficiency"]
    assert (
        result["predicted_absorbed_photons"] == expected["predicted_absorbed_photons"]
    )
    assert result["photon_ledger"]["passed"]


def test_recursive_residual_roundtrip_is_self_contained(tmp_path: Path) -> None:
    """A moved model archive restores the base and corrections without external paths."""
    source = np.full((4, 4, 4), 0.4)
    capacity = np.linspace(0.5, 1.5, 64).reshape(source.shape)
    es = np.linspace(0.1, 0.9, 64).reshape(source.shape)
    first = residual_model(fixture_model(source, capacity, es))
    first.field_model.weights[0][0, 0] = 0.07
    second = residual_model(first)
    second.field_model.weights[0][1, 0] = -0.04
    directory = tmp_path / "original"
    second.save(directory)
    saved = json.loads((directory / "model.json").read_text())
    assert saved["schema"] == 3
    assert saved["base_model"] == "base_model"
    assert (directory / "base_model/base_model/field_model.npz").is_file()
    moved = directory.rename(tmp_path / "moved")
    restored = Emulator.load(moved)
    before = second.predict(source, capacity, es, 7.5, 256)
    after = restored.predict(source, capacity, es, 7.5, 256)
    for name in ("raw_xHI", "xHI", "latent_xHII"):
        np.testing.assert_array_equal(before[name], after[name])
    assert before["photon_ledger"] == after["photon_ledger"]
    assert restored.base_model.base_model is not None


def test_invalid_residual_base_is_rejected(tmp_path: Path) -> None:
    """Reject self-referential bases and incompatible residual parameterizations."""
    source = np.full((4, 4, 4), 0.4)
    es = np.full_like(source, 0.6)
    base = fixture_model(source, np.ones_like(source), es)
    cyclic = residual_model(base)
    cyclic.base_model = cyclic
    with pytest.raises(ValueError, match="cycle"):
        cyclic.save(tmp_path / "cyclic")
    with pytest.raises(ValueError, match="cycle"):
        cyclic.predict(source, np.ones_like(source), es, 7.5, 256)
    base.base_model = fixture_model(source, np.ones_like(source), es)
    with pytest.raises(ValueError, match="latent"):
        base.save(tmp_path / "bounded")


def test_saved_residual_cannot_reference_external_base(tmp_path: Path) -> None:
    """Require the fixed local subdirectory recorded by schema three."""
    source = np.full((4, 4, 4), 0.4)
    model = residual_model(fixture_model(source, np.ones_like(source), 1 - source))
    model.save(tmp_path)
    record = json.loads((tmp_path / "model.json").read_text())
    record["base_model"] = "../elsewhere"
    (tmp_path / "model.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="base"):
        Emulator.load(tmp_path)


@pytest.mark.parametrize("bad", [np.nan, -0.1, 1.1])
def test_bad_native_es_rejected(bad: float) -> None:
    """Reject nonfinite or unbounded inference-time ES fields."""
    source = np.ones((4, 4, 4))
    with pytest.raises(ValueError, match="native ES"):
        field_features(source, source, np.full_like(source, bad), 7.5, 256)
