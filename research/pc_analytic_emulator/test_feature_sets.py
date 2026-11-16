"""Verify optional spatial inputs survive deployment and legacy loading."""

import json
from pathlib import Path

import numpy as np
import pytest

from research.pc_analytic_emulator.model import Emulator, field_features
from research.pc_analytic_emulator.test_model import constant_model, fixture_model


def test_local_features_are_optional_and_keep_legacy_columns() -> None:
    """Augment only explicitly requested models with native spatial columns."""
    photons = np.arange(64).reshape(4, 4, 4) / 64
    atoms = np.ones_like(photons)
    neutral = np.linspace(0, 1, 64).reshape(photons.shape)
    legacy, names, baseline = field_features(photons, atoms, neutral, 7, 50)
    local, local_names, local_baseline = field_features(
        photons, atoms, neutral, 7, 50, feature_set="multiscale_local"
    )
    assert local.shape[1] > legacy.shape[1]
    np.testing.assert_array_equal(local[:, : legacy.shape[1]], legacy)
    assert local_names[: len(names)] == names
    assert len(local_names) == len(set(local_names))
    np.testing.assert_array_equal(local_baseline, baseline)
    with pytest.raises(ValueError, match="feature_set"):
        field_features(photons, atoms, neutral, 7, 50, feature_set="unknown")


@pytest.mark.parametrize("with_base", [False, True])
def test_local_model_roundtrip_and_legacy_base(tmp_path: Path, with_base: bool) -> None:
    """Keep each layer's input convention when nesting an older base model."""
    from research.pc_analytic_emulator.latent import LatentFieldModel

    source = np.full((4, 4, 4), 0.7)
    atoms = np.linspace(0.5, 1.5, 64).reshape(source.shape)
    neutral = np.linspace(0.1, 0.9, 64).reshape(source.shape)
    base = fixture_model(source, atoms, neutral)
    features, _, _ = field_features(
        source, atoms, neutral, 7.5, 50, feature_set="multiscale_local"
    )
    width = features.shape[1]
    weight = np.zeros((width, 1))
    weight[-1] = 0.01
    model = Emulator(
        LatentFieldModel([weight], [np.zeros(1)], np.zeros(width), np.ones(width), {}),
        base.budget_model,
        projection_mode="capacity_l2",
        base_model=base if with_base else None,
        feature_set="multiscale_local",
    )
    before = model.predict(source, atoms, neutral, 7.5, 50)
    model.save(tmp_path)
    record = json.loads((tmp_path / "model.json").read_text())
    assert record["schema"] == 4
    restored = Emulator.load(tmp_path)
    assert restored.feature_set == "multiscale_local"
    if with_base:
        assert restored.base_model.feature_set == "legacy"
    after = restored.predict(source, atoms, neutral, 7.5, 50)
    np.testing.assert_array_equal(before["xHI"], after["xHI"])
    assert after["photon_ledger"]["passed"]
    from research.pc_analytic_emulator.predict import _model_hashes

    hashes = _model_hashes(tmp_path)
    assert len(hashes) == (6 if with_base else 3)
    if with_base:
        assert "base_model/field_model.npz" in hashes


def test_invalid_saved_feature_set_is_rejected(tmp_path: Path) -> None:
    """Reject unknown feature conventions before inference instead of guessing."""
    source = np.ones((4, 4, 4))
    model = fixture_model(source, source, source * 0.5)
    model.save(tmp_path)
    path = tmp_path / "model.json"
    record = json.loads(path.read_text())
    record.update(schema=4, feature_set="unknown")
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="feature_set"):
        Emulator.load(tmp_path)


def test_legacy_default_is_retained_in_schema_two(tmp_path: Path) -> None:
    """Continue writing original schemas when no new features are requested."""
    source = np.ones((4, 4, 4))
    model = fixture_model(source, source, source * 0.5)
    model.field_model = constant_model(24, 0)
    model.save(tmp_path)
    assert json.loads((tmp_path / "model.json").read_text())["schema"] == 2
    assert Emulator.load(tmp_path).feature_set == "legacy"
