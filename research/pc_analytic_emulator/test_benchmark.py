"""Verify frozen-model benchmarks and independent saved-field diagnostics."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from research.pc_analytic_emulator import benchmark
from research.pc_analytic_emulator.model import Emulator
from research.pc_analytic_emulator.test_data import make_campaign
from research.pc_analytic_emulator.test_model import fixture_model, residual_model


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    teacher = make_campaign(
        tmp_path / "teacher", seeds=(31,), redshifts=(8.0, 6.0), n=8
    )
    source = np.full((8, 8, 8), 0.6)
    atoms = np.ones_like(source)
    native = np.full_like(source, 0.65)
    model = fixture_model(source, atoms, native, efficiency=2 / 3)
    model.save(tmp_path / "model")
    return teacher, tmp_path / "model"


def test_saved_field_ledger_does_not_construct_its_unabsorbed_reservoir() -> None:
    """A corrupted field must fail despite an otherwise valid stored budget."""
    source = np.full((3, 3, 3), 0.7)
    atoms = np.ones_like(source)
    correct = benchmark.independent_ledger(source, atoms, atoms * 0.6, 0.3 * atoms.size)
    wrong = benchmark.independent_ledger(source, atoms, atoms * 0.2, 0.3 * atoms.size)
    assert correct["passed"]
    assert correct["relative_photon_residual"] < 1e-14
    assert not wrong["passed"]
    assert wrong["relative_photon_residual"] > 0.5


def test_zero_source_ledger_retains_physical_endpoint() -> None:
    """No source and fully neutral gas have a zero, defined photon residual."""
    atoms = np.ones((2, 2, 2))
    result = benchmark.independent_ledger(atoms * 0, atoms, atoms, 0, 0)
    assert result["passed"]
    assert result["relative_photon_residual"] == 0


@pytest.mark.parametrize("bad", [-0.01, 1.01, np.nan])
def test_invalid_stored_fraction_rejected(bad: float) -> None:
    """An invalid saved fraction must not become an apparently passing ledger."""
    atoms = np.ones((2, 2, 2))
    with pytest.raises(ValueError, match="Invalid stored"):
        benchmark.independent_ledger(atoms, atoms, atoms * bad, 0)


def test_benchmark_freezes_hashes_and_retains_undefined_ratios(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise audited input loading, common rays and checksummed artifacts."""
    teacher, model = _fixture(tmp_path)
    output = tmp_path / "benchmark"
    original = Emulator.predict
    calls = []

    def checked_prediction(self: Emulator, *args: object, **kwargs: object) -> dict:
        manifest = json.loads((output / "manifest.json").read_text())
        assert manifest["models"]["selected"]["files_sha256"]
        calls.append(True)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Emulator, "predict", checked_prediction)
    report = benchmark.run(teacher, model, model, output, rays=32)
    assert len(calls) == 4
    assert report["complete"]
    assert report["model_weights_unchanged"]
    assert report["snapshot_count"] == 2
    assert report["maximum_relative_photon_residual"] < 1e-12
    assert report["summaries"]["active"]["count"] == 2
    assert report["summaries"]["z6"]["count"] == 1
    assert report["summaries"]["late"]["count"] == 0
    for row in report["snapshots"]:
        assert (
            row["low_k_21cm"]["maximum_included_mode_k_cMpc_inv"] < np.pi * 8 / 256 / 2
        )
        for method in benchmark.METHODS:
            assert (
                row["low_k_21cm"]["comparisons"][method][
                    "mean_absolute_power_ratio_error"
                ]
                is None
            )
            assert row["mfp"]["ionized"]["comparisons"][method]["available"] is False
        with np.load(
            output / row["files"]["mfp_rays.npz"]["path"], allow_pickle=False
        ) as rays:
            np.testing.assert_array_equal(
                rays["neutral_selected_length_cMpc"],
                rays["neutral_previous_length_cMpc"],
            )
        with np.load(
            output / row["files"]["predictions.npz"]["path"], allow_pickle=False
        ) as saved:
            np.testing.assert_allclose(saved["selected_xHI"], 0.6, atol=2e-12)
    power = report["summaries"]["all"]["methods"]["selected"]["low_k_21cm"][
        "snapshot_mean_absolute_power_ratio_error"
    ]
    assert power == {
        "defined_count": 0,
        "undefined_count": 2,
        "mean": None,
        "maximum": None,
    }
    receipt = json.loads((output / "receipt.json").read_text())
    for name, expected in receipt["files_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    assert (output / "COMPLETED").is_file()


def test_independent_audit_rejects_a_fake_passing_model_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inference's own reported conservation is never taken as benchmark proof."""
    teacher, model = _fixture(tmp_path)
    original = Emulator.predict

    def corrupt(self: Emulator, *args: object, **kwargs: object) -> dict:
        result = original(self, *args, **kwargs)
        result["xHI"] = np.zeros_like(result["xHI"])
        result["photon_ledger"] = {"passed": True, "relative_photon_residual": 0}
        return result

    monkeypatch.setattr(Emulator, "predict", corrupt)
    output = tmp_path / "invalid"
    with pytest.raises(FloatingPointError, match="Independent saved-field"):
        benchmark.run(teacher, model, model, output, rays=8)
    assert not (output / "COMPLETED").exists()


def test_frozen_model_mutation_fails_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file changed after the frozen manifest invalidates the benchmark."""
    teacher, model = _fixture(tmp_path)
    original = Emulator.predict

    def mutate(self: Emulator, *args: object, **kwargs: object) -> dict:
        result = original(self, *args, **kwargs)
        with (model / "field_model.npz").open("ab") as stream:
            stream.write(b"changed-after-model-load")
        return result

    monkeypatch.setattr(Emulator, "predict", mutate)
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="changed during benchmark"):
        benchmark.run(teacher, model, model, output, rays=8)
    assert not (output / "COMPLETED").exists()


def test_schema_four_hashes_include_frozen_base(tmp_path: Path) -> None:
    """New feature schema must not omit the nested model from provenance."""
    _, directory = _fixture(tmp_path)
    model = residual_model(Emulator.load(directory))
    model.save(directory)
    record = json.loads((directory / "model.json").read_text())
    record["schema"] = 4
    (directory / "model.json").write_text(json.dumps(record))
    hashes = benchmark._model_hashes(directory)
    assert len(hashes) == 6
    assert "base_model/field_model.npz" in hashes


def test_output_refuses_to_overwrite_input(tmp_path: Path) -> None:
    """Input archives and previously published benchmark outputs remain intact."""
    teacher, model = _fixture(tmp_path)
    with pytest.raises(ValueError, match="new and outside"):
        benchmark.run(teacher, model, model, teacher / "new", rays=8)
    assert not (teacher / "new").exists()


def _model_physics(teacher: Path, model: Path) -> tuple[Emulator, dict]:
    physics = json.loads(
        (teacher / "seeds" / "seed31" / "native_inputs.json").read_text()
    )
    fitted = Emulator.load(model)
    fitted.metadata["teacher_physics"] = copy.deepcopy(physics)
    return fitted, physics


def test_compatible_geometry_transfer_is_recorded_without_mutating_metadata(
    tmp_path: Path,
) -> None:
    """Grid/box changes are permitted and disclosed; native physics stays fixed."""
    teacher, model = _fixture(tmp_path)
    fitted, physics = _model_physics(teacher, model)
    before = copy.deepcopy(fitted.metadata)
    physics["random_seed"] = 44
    physics["simulation_options"].update(DIM=48, HII_DIM=16, BOX_LEN=128, N_THREADS=8)
    report = benchmark._physics_compatibility(fitted, physics)
    assert report["physical_parameters_matched"]
    assert report["geometry_transfer"]
    assert report["geometry"]["HII_DIM"] == {
        "training": 8,
        "evaluation": 16,
        "changed": True,
    }
    assert fitted.metadata == before


@pytest.mark.parametrize(
    ("group", "key", "value"),
    [
        ("cosmo_params", "OMm", 0.4),
        ("astro_params", "R_BUBBLE_MAX", 30),
        ("astro_options", "USE_TS_FLUCT", True),
        ("matter_options", "SOURCE_MODEL", "changed"),
        ("simulation_options", "UNDECLARED_OPTION", 4),
    ],
)
def test_geometry_permission_never_waives_physical_changes(
    tmp_path: Path,
    group: str,
    key: str,
    value: object,
) -> None:
    """Changes to cosmology, source physics or other options remain failures."""
    teacher, model = _fixture(tmp_path)
    fitted, physics = _model_physics(teacher, model)
    physics[group][key] = value
    with pytest.raises(ValueError, match="physical parameters differ"):
        benchmark._physics_compatibility(fitted, physics)


def test_incompatible_recorded_physics_fails_before_any_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public benchmark must enforce compatibility before model inference."""
    teacher, model = _fixture(tmp_path)
    fitted, _ = _model_physics(teacher, model)
    fitted.metadata["teacher_physics"]["cosmo_params"]["OMm"] = 0.4
    fitted.save(model)

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Incompatible physical inputs reached prediction")

    monkeypatch.setattr(Emulator, "predict", forbidden)
    output = tmp_path / "incompatible"
    with pytest.raises(ValueError, match="physical parameters differ"):
        benchmark.run(teacher, model, model, output, rays=8)
    assert not (output / "COMPLETED").exists()


def test_metadata_free_synthetic_model_reports_compatibility_unavailable(
    tmp_path: Path,
) -> None:
    """Missing metadata must never be represented as a successful match."""
    teacher, model = _fixture(tmp_path)
    physics = json.loads(
        (teacher / "seeds" / "seed31" / "native_inputs.json").read_text()
    )
    report = benchmark._physics_compatibility(Emulator.load(model), physics)
    assert report["available"] is False
    assert report["physical_parameters_matched"] is None
    assert report["geometry_transfer"] is None


def test_frozen_base_cannot_hide_incompatible_training_physics(tmp_path: Path) -> None:
    """A matching outer model must not waive a recorded base-physics mismatch."""
    teacher, model = _fixture(tmp_path)
    base, physics = _model_physics(teacher, model)
    fitted = residual_model(base)
    fitted.metadata["teacher_physics"] = copy.deepcopy(physics)
    base.metadata["teacher_physics"]["cosmo_params"]["OMm"] = 0.4
    with pytest.raises(ValueError, match="physical parameters differ"):
        benchmark._physics_compatibility(fitted, physics)


def test_explicit_training_geometries_replace_the_canonical_transfer_assumption(
    tmp_path: Path,
) -> None:
    """A second trained geometry is supported even when canonical inputs differ."""
    teacher, model = _fixture(tmp_path)
    fitted, physics = _model_physics(teacher, model)
    trained = [
        {"DIM": 24, "HII_DIM": 8, "BOX_LEN": 256.0},
        {"DIM": 48, "HII_DIM": 16, "BOX_LEN": 128.0},
    ]
    fitted.metadata["training_geometries"] = copy.deepcopy(trained)
    before = copy.deepcopy(fitted.metadata)
    physics["simulation_options"].update(trained[1])
    report = benchmark._physics_compatibility(fitted, physics)
    assert report["training_geometries"] == trained
    assert report["evaluated_geometry"] == trained[1]
    assert report["trained_geometry_supported"] is True
    assert report["geometry_transfer"] is False
    assert report["geometry"]["HII_DIM"]["changed"] is True
    assert fitted.metadata == before


def test_explicit_training_geometries_allow_disclosed_benchmark_transfer(
    tmp_path: Path,
) -> None:
    """Benchmarks may test unseen geometry without classifying it as trained."""
    teacher, model = _fixture(tmp_path)
    fitted, physics = _model_physics(teacher, model)
    fitted.metadata["training_geometries"] = [
        {"DIM": 48, "HII_DIM": 16, "BOX_LEN": 128.0}
    ]
    report = benchmark._physics_compatibility(fitted, physics)
    assert report["evaluated_geometry"] == {
        "DIM": 24,
        "HII_DIM": 8,
        "BOX_LEN": 256.0,
    }
    assert report["trained_geometry_supported"] is False
    assert report["geometry_transfer"] is True
    assert all(not item["changed"] for item in report["geometry"].values())


@pytest.mark.parametrize(
    "geometries",
    [
        [],
        None,
        {"DIM": 24, "HII_DIM": 8, "BOX_LEN": 256},
        [{"DIM": 24, "HII_DIM": 8}],
        [{"DIM": True, "HII_DIM": 8, "BOX_LEN": 256}],
        [{"DIM": 24, "HII_DIM": 32, "BOX_LEN": 256}],
        [{"DIM": 24, "HII_DIM": 8, "BOX_LEN": np.nan}],
        [{"DIM": 24, "HII_DIM": 8, "BOX_LEN": 256}] * 2,
    ],
)
def test_malformed_training_geometry_metadata_is_rejected(
    tmp_path: Path, geometries: object
) -> None:
    """An invalid opt-in declaration cannot waive the geometry contract."""
    teacher, model = _fixture(tmp_path)
    fitted, physics = _model_physics(teacher, model)
    fitted.metadata["training_geometries"] = geometries
    with pytest.raises((ValueError, TypeError), match=r"[Gg]eometr"):
        benchmark._physics_compatibility(fitted, physics)


def test_mixed_geometry_permission_keeps_physics_and_frozen_base_checks(
    tmp_path: Path,
) -> None:
    """Support sets are model-local and never weaken nongeometry matching."""
    teacher, model = _fixture(tmp_path)
    base, physics = _model_physics(teacher, model)
    fitted = residual_model(base)
    fitted.metadata["teacher_physics"] = copy.deepcopy(physics)
    fitted.metadata["training_geometries"] = [
        {"DIM": 48, "HII_DIM": 16, "BOX_LEN": 128.0}
    ]
    physics["simulation_options"].update(fitted.metadata["training_geometries"][0])
    report = benchmark._physics_compatibility(fitted, physics)
    assert report["geometry_transfer"] is False
    assert report["frozen_base"]["geometry_transfer"] is True
    assert "training_geometries" not in report["frozen_base"]
    physics["astro_params"]["R_BUBBLE_MAX"] = 30
    with pytest.raises(ValueError, match="physical parameters differ"):
        benchmark._physics_compatibility(fitted, physics)


def test_training_geometry_support_requires_recorded_physics(tmp_path: Path) -> None:
    """A support declaration alone cannot certify physical compatibility."""
    teacher, model = _fixture(tmp_path)
    fitted, physics = _model_physics(teacher, model)
    fitted.metadata.pop("teacher_physics")
    fitted.metadata["training_geometries"] = [
        {"DIM": 24, "HII_DIM": 8, "BOX_LEN": 256.0}
    ]
    with pytest.raises(ValueError, match="require recorded teacher_physics"):
        benchmark._physics_compatibility(fitted, physics)


def test_benchmark_evaluates_each_selected_geometry_before_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mixed dataset retains per-grid reports and uses each native cube size."""
    teacher, model = _fixture(tmp_path)
    small = make_campaign(tmp_path / "small", seeds=(32,), redshifts=(8.0,), n=4)
    original = benchmark.load_dataset(teacher)
    other = benchmark.load_dataset(small)
    mixed = replace(
        original,
        samples=original.samples + other.samples,
        file_hashes={**original.file_hashes, **other.file_hashes},
    )
    monkeypatch.setattr(benchmark, "load_dataset", lambda _: mixed)
    fitted, _ = _model_physics(teacher, model)
    fitted.save(tmp_path / "previous")
    fitted.metadata["training_geometries"] = [
        {"DIM": 12, "HII_DIM": 4, "BOX_LEN": 256.0},
        {"DIM": 24, "HII_DIM": 8, "BOX_LEN": 256.0},
    ]
    fitted.save(model)
    seen_geometries = []
    compatibility = benchmark._physics_compatibility

    def check_geometry(fitted: Emulator, physics: dict) -> dict:
        seen_geometries.append(physics["simulation_options"]["HII_DIM"])
        return compatibility(fitted, physics)

    prediction = Emulator.predict

    def check_prediction(self: Emulator, *args: object, **kwargs: object) -> dict:
        assert seen_geometries == [4, 8, 4, 8]
        return prediction(self, *args, **kwargs)

    monkeypatch.setattr(benchmark, "_physics_compatibility", check_geometry)
    monkeypatch.setattr(Emulator, "predict", check_prediction)
    report = benchmark.run(
        teacher, model, tmp_path / "previous", tmp_path / "mixed_report", rays=8
    )
    selected = report["physical_compatibility"]["selected"]
    previous = report["physical_compatibility"]["previous"]
    assert selected["geometry_transfer"] is False
    assert previous["geometry_transfer"] is True
    assert selected["trained_geometry_supported"] is True
    assert selected["training_geometries"] == fitted.metadata["training_geometries"]
    assert len(selected["by_geometry"]) == 2
    assert all(item["trained_geometry_supported"] for item in selected["by_geometry"])
    assert {row["dimension"] for row in report["snapshots"]} == {4, 8}
    assert report["maximum_relative_photon_residual"] < 1e-12
