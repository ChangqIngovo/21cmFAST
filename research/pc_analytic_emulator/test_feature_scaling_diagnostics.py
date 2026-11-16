"""Check copied-model interventions and independent feature diagnostics."""

import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from research.pc_analytic_emulator.model import field_features
from research.pc_analytic_emulator.refinement_20261117.reproduction import (
    diagnose_feature_scaling as audit,
)
from research.pc_analytic_emulator.test_invariants import _problem


def _fixture():
    model, arrays = _problem("residual")
    names = field_features(*arrays, 7.5, 96)[1]
    for level in audit.levels(model):
        level.metadata["field_feature_names"] = names
        for radius in (25, 50):
            level.field_model.scale[
                names.index(f"log1p_density_sigma{radius}_cMpc")
            ] = 1e-6
    return model, arrays, names


def test_interventions_use_copies_and_preserve_budgets_and_unselected_weights():
    """Keep original heads and all budget parameters untouched."""
    model, _, names = _fixture()
    original = copy.deepcopy(model)
    changed = audit.intervene(model, "dx_mean_density25_50_floor_recursive", 0.01)
    dx = names.index("log1p_dx_cMpc")
    other = names.index("log1p_photons_sigma0_cells")
    for old, before, new in zip(
        audit.levels(model), audit.levels(original), audit.levels(changed), strict=True
    ):
        np.testing.assert_array_equal(
            old.field_model.weights[0], before.field_model.weights[0]
        )
        np.testing.assert_array_equal(new.field_model.weights[0][dx], 0)
        np.testing.assert_array_equal(
            new.field_model.weights[0][other], old.field_model.weights[0][other]
        )
        np.testing.assert_array_equal(
            new.budget_model.weights[0], old.budget_model.weights[0]
        )
        assert new.field_model.scale[names.index("log1p_density_sigma50_cMpc")] == 0.01


def test_top_only_and_zero_dx_have_the_declared_mean_input_meaning():
    """Distinguish recursive edits and the exact standardized-coordinate clamp."""
    model, arrays, names = _fixture()
    changed = audit.intervene(model, "density25_50_floor_top_only", 0.01)
    np.testing.assert_array_equal(
        changed.base_model.field_model.scale, model.base_model.field_model.scale
    )
    original_features, _, baseline = field_features(*arrays, 7.5, 96)
    features = original_features.astype(float)
    index = names.index("log1p_dx_cMpc")
    features[:, index] = model.field_model.mean[index]
    zeroed = audit.intervene(model, "dx_mean_recursive", 0.01)
    np.testing.assert_allclose(
        zeroed.field_model.latent_predict(original_features, baseline=baseline),
        model.field_model.latent_predict(features, baseline=baseline),
        atol=1e-14,
        rtol=0,
    )


@pytest.mark.parametrize("floor", [0, -1, float("nan"), float("inf")])
def test_invalid_floor_is_rejected(floor):
    """Reject invalid scales even for the unchanged control."""
    model, _, _ = _fixture()
    with pytest.raises(ValueError, match="floor"):
        audit.intervene(model, "original", floor)


def test_saturation_reports_the_actual_standardized_inputs():
    """Recover analytically known z-scores and saturation fractions."""
    head = SimpleNamespace(
        mean=np.array([1.0, 2.0]),
        scale=np.array([1e-6, 1.0]),
        weights=[np.array([[1.0], [0.0]])],
        biases=[np.array([0.0])],
    )
    result = audit.feature_statistics(
        np.array([[1.01, 2.0], [0.99, 2.0]]), ["density", "dx"], head
    )
    assert result["top_coordinates"][0]["max_abs_z"] == pytest.approx(10000)
    assert result["first_hidden_abs_pre_activation_gt10_fraction"] == 1
    assert result["all_coordinates"][1]["max_abs_z"] == 0


def test_sample_interventions_keep_target_fixed_and_do_not_average_zero_power():
    """Keep physical counts fixed and zero reference power explicitly undefined."""
    model, arrays, _ = _fixture()
    source, capacity, native = arrays
    sample = SimpleNamespace(
        seed=1,
        redshift=7.5,
        dimension=len(source),
        box_len=96.0,
        arrays={
            "source": source,
            "capacity": capacity,
            "native_es_xHI": native,
            "teacher_xHI": np.zeros_like(source),
        },
    )
    row = audit.analyze_sample(sample, "regression", model, 0.01)
    targets = [v["predicted_absorbed_photons"] for v in row["interventions"].values()]
    np.testing.assert_array_equal(targets, np.repeat(targets[0], len(targets)))
    assert all(
        v["independent_photon_ledger"]["passed"] for v in row["interventions"].values()
    )
    summary = audit.summarize([row])["regression"]["late"]
    assert summary["count"] == 1
    for variant in summary["variants"].values():
        assert variant["low_k_mean_absolute_power_ratio_error"]["defined_count"] == 0
        assert variant["low_k_mean_absolute_power_ratio_error"]["mean"] is None


def test_runner_keeps_files_unchanged_and_public_report_has_no_absolute_paths(
    tmp_path, monkeypatch
):
    """Audit artifacts without publishing machine-specific directories."""
    model, arrays, _ = _fixture()
    directory = tmp_path / "private_model"
    model.save(directory)
    native = tmp_path / "private_dataset" / "source.npz"
    native.parent.mkdir()
    np.savez(native, source=arrays[0])
    sample = SimpleNamespace(
        seed=1,
        redshift=7.5,
        dimension=len(arrays[0]),
        box_len=96.0,
        arrays={
            "source": arrays[0],
            "capacity": arrays[1],
            "native_es_xHI": arrays[2],
            "teacher_xHI": np.full_like(arrays[0], 0.6),
        },
    )
    dataset = SimpleNamespace(
        root=native.parent,
        samples=[sample],
        file_hashes={str(native): hashlib.sha256(native.read_bytes()).hexdigest()},
        physics={},
    )
    monkeypatch.setattr(audit, "load_dataset", lambda path: dataset)
    monkeypatch.setattr(
        audit, "_physics_compatibility", lambda model, physics: {"available": False}
    )
    result = audit.run({"regression": native.parent}, directory, tmp_path / "output")
    assert result["complete"]
    assert result["model_files_unchanged"]
    assert result["input_files_unchanged"]
    serialized = json.dumps(result)
    assert str(tmp_path) not in serialized
    assert result == json.loads((tmp_path / "output" / "report.json").read_text())
    with pytest.raises(FileExistsError):
        audit.run({"regression": native.parent}, directory, tmp_path / "output")


def test_recursive_cycle_is_rejected():
    """Stop before copying or evaluating a cyclic frozen-base graph."""
    model, _, _ = _fixture()
    model.base_model.base_model = model
    with pytest.raises(ValueError, match="cycle"):
        list(audit.levels(model))
