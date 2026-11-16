"""Check the standalone frozen-model symmetry audit on synthetic networks."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from research.pc_analytic_emulator.audit_symmetry import audit_model, run
from research.pc_analytic_emulator.latent import LatentFieldModel
from research.pc_analytic_emulator.model import Emulator, field_features
from research.pc_analytic_emulator.test_invariants import _problem


def _spatial_model() -> tuple[Emulator, tuple[np.ndarray, ...]]:
    model, arrays = _problem("residual")
    features, _, _ = field_features(*arrays, 7.5, 96, feature_set="multiscale_local")
    random = np.random.default_rng(474)
    width = features.shape[1]
    model.field_model = LatentFieldModel(
        [random.normal(0, 0.03, (width, 5)), random.normal(0, 0.03, (5, 1))],
        [np.zeros(5), np.zeros(1)],
        np.zeros(width),
        np.ones(width),
        {},
    )
    model.feature_set = "multiscale_local"
    model.budget_model = copy.deepcopy(model.budget_model)
    model.budget_model.output_mode = "sigmoid"
    model.budget_model.metadata["budget_parameterization"] = (
        "direct_sigmoid_absorption_deficit"
    )
    return model, arrays


def test_complete_spatial_deficit_model_passes_all_geometric_and_unit_checks() -> None:
    """Exercise 88-column features, a deficit head and the complete base chain."""
    model, arrays = _spatial_model()
    report, _ = audit_model(model, *arrays, 7.5, 96)
    assert report["checks"] == 55
    assert report["passed"]
    assert report["maximum_photon_residual"] < 1e-10
    assert report["maximum_field_absolute_difference"] < 1e-7
    assert sum(row["kind"] == "cube_symmetry" for row in report["rows"]) == 48


def test_audit_detects_coordinate_dependent_latent_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Passing conservation alone must not hide a broken rotation symmetry."""
    model, arrays = _problem("residual")
    original = model.predict

    def broken(*args: object, **kwargs: object) -> dict:
        prediction = original(*args, **kwargs)
        shape = prediction["latent_xHII"].shape
        prediction["latent_xHII"] += np.arange(shape[0])[:, None, None] * 0.01
        return prediction

    monkeypatch.setattr(model, "predict", broken)
    report, _ = audit_model(model, *arrays, 7.5, 96)
    assert not report["passed"]
    assert report["maximum_field_absolute_difference"] >= 0.01
    assert all(row["independent_photon_ledger"]["passed"] for row in report["rows"])


def test_saved_model_roundtrip_and_inputs_are_audited_without_teacher_labels(
    tmp_path: Path,
) -> None:
    """Persist the fitted model, audit only native inputs, and verify roundtrip."""
    model, arrays = _spatial_model()
    directory = tmp_path / "model"
    model.save(directory)
    source = tmp_path / "native.npz"
    np.savez(
        source,
        source=arrays[0],
        capacity=arrays[1],
        native_es_xhi=arrays[2],
        teacher_xHI=np.full_like(arrays[0], np.nan),
    )
    report = run(directory, source, tmp_path / "audit", redshift=7.5, box_len=96)
    assert report["complete"]
    assert report["passed"]
    assert report["roundtrip"]["passed"]
    assert report["model_weights_unchanged"]
    assert report["input_unchanged"]
    assert report["teacher_labels_used"] is False
    assert all(
        row["maximum_absolute_difference"] == 0
        for row in report["roundtrip"]["fields"].values()
    )
    saved = json.loads((tmp_path / "audit" / "report.json").read_text())
    assert saved == report
