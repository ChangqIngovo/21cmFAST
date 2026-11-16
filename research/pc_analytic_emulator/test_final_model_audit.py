"""Check final audit provenance and exact serialization independently."""

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from research.pc_analytic_emulator import predict
from research.pc_analytic_emulator.refinement_20261117.reproduction import (
    audit_final_model as audit,
)
from research.pc_analytic_emulator.test_invariants import _problem


def _sources(tmp_path):
    frozen, released, reference = (
        tmp_path / name for name in ("frozen", "released", "reference")
    )
    expected = {}
    for module_name in audit.INFERENCE_MODULES:
        module = importlib.import_module(module_name)
        relative = module_name.replace(".", "/") + ".py"
        content = Path(module.__file__).read_bytes()
        target = frozen / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        expected[relative] = hashlib.sha256(content).hexdigest()
    train = "research/pc_analytic_emulator/train.py"
    for root, contents in ((frozen, "value=1\n"), (released, "value = 1\n")):
        target = root / train
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents)
    expected[train] = hashlib.sha256((frozen / train).read_bytes()).hexdigest()
    for name in (*audit.KERNEL_FILES, "src/py21cmfast/src/example.c"):
        for root in (frozen, released, reference):
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("unchanged physical source\n")
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps(expected))
    return frozen, released, reference, manifest


def test_frozen_and_loaded_source_hashes_and_kernel_reference_are_checked(tmp_path):
    """Permit formatting-only train edits while retaining exact physical kernels."""
    frozen, released, reference, manifest = _sources(tmp_path)
    report = audit.verify_sources(frozen, manifest, reference, released_source=released)
    assert report["passed"]
    assert not report["train_python_comparison"]["byte_identical"]
    assert report["train_python_comparison"]["AST_identical"]
    (released / "src/py21cmfast/src/example.c").write_text("changed physics\n")
    with pytest.raises(ValueError, match="Physical kernel"):
        audit.verify_sources(frozen, manifest, reference, released_source=released)


def test_frozen_hash_drift_and_training_algorithm_change_are_rejected(tmp_path):
    """Reject source mutations and distinguish arithmetic edits from formatting."""
    frozen, released, reference, manifest = _sources(tmp_path)
    train = "research/pc_analytic_emulator/train.py"
    (released / train).write_text("value = 2\n")
    with pytest.raises(ValueError, match="beyond formatting"):
        audit.verify_sources(frozen, manifest, reference, released_source=released)
    (frozen / train).write_text("value=2\n")
    with pytest.raises(ValueError, match="Frozen source mismatch"):
        audit.verify_sources(frozen, manifest, reference, released_source=released)


def test_loaded_python_formatting_is_reported_and_algorithm_change_rejected(tmp_path):
    """Permit whitespace changes, without calling changed inference bytes identical."""
    frozen, released, reference, manifest = _sources(tmp_path)
    name = "research/pc_analytic_emulator/budget.py"
    archived = frozen / name
    original = archived.read_text()
    archived.write_text(original + "\n\n")
    hashes = json.loads(manifest.read_text())
    hashes[name] = hashlib.sha256(archived.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(hashes))
    report = audit.verify_sources(frozen, manifest, reference, released_source=released)
    assert report["formatted_python_files"] == [
        name,
        "research/pc_analytic_emulator/train.py",
    ]
    comparison = report["python_module_comparisons"][name]
    assert comparison["AST_identical"]
    assert not comparison["byte_identical"]
    assert comparison["frozen_sha256"] != comparison["released_sha256"]
    archived.write_text(original + "\nmaterial_algorithm_change = True\n")
    hashes[name] = hashlib.sha256(archived.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(hashes))
    with pytest.raises(ValueError, match="inference implementation differs beyond"):
        audit.verify_sources(frozen, manifest, reference, released_source=released)


def test_exact_prediction_roundtrip_includes_scalar_photon_partition(tmp_path):
    """A small scalar count change fails even if all bounded fields agree."""
    arrays = {key: np.zeros((2, 2, 2)) for key in predict.FIELD_ARRAYS}
    arrays.update(
        absorption_efficiency=np.array(0.5),
        predicted_absorbed_photons=np.array(1.0),
        predicted_unabsorbed_photons=np.array(1.0),
    )
    first, second = tmp_path / "first.npz", tmp_path / "second.npz"
    np.savez(first, **arrays)
    np.savez(second, **arrays)
    assert audit._exact_predictions(first, second)["passed"]
    arrays["predicted_unabsorbed_photons"] = np.nextafter(
        arrays["predicted_unabsorbed_photons"], 2.0
    )
    np.savez(second, **arrays)
    report = audit._exact_predictions(first, second)
    assert not report["passed"]
    assert not report["array_equal"]["predicted_unabsorbed_photons"]


def test_feature_guard_audit_finds_nonzero_constant_rows_without_changing_weights():
    """Audit declared masks at every frozen base instead of trusting metadata."""
    model, arrays = _problem("residual")
    inputs = dict(zip(("source", "capacity", "native_es_xHI"), arrays, strict=True))
    for level in audit.levels(model):
        level.field_model.metadata.update(
            zero_constant_features=True,
            constant_feature_indices=[0],
            field_feature_scale_floor=0.01,
        )
        level.field_model.weights[0][0] = 0
    before = model.field_model.weights[0].copy()
    assert all(
        row["passed"] for row in audit.feature_guard_audit(model, inputs, 7.5, 96)
    )
    np.testing.assert_array_equal(before, model.field_model.weights[0])
    model.base_model.field_model.weights[0][0, 0] = 0.1
    rows = audit.feature_guard_audit(model, inputs, 7.5, 96)
    assert rows[0]["passed"]
    assert not rows[1]["passed"]


def test_native_geometry_is_read_from_original_metadata(tmp_path):
    """Use original redshift and native geometry, independent of file naming."""
    source = tmp_path / "source.npz"
    np.savez(source, source=np.ones((4, 4, 4)))
    record = {
        "identity": {
            "redshift": 7.5,
            "inputs": {
                "simulation_options": {"DIM": 12, "HII_DIM": 4, "BOX_LEN": 50.0}
            },
        }
    }
    source.with_name("native.json").write_text(json.dumps(record))
    result = audit._native(source)
    assert result["redshift"] == 7.5
    assert result["geometry"] == {"DIM": 12, "HII_DIM": 4, "BOX_LEN": 50.0}
    assert result["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_final_runner_checks_supported_native_geometry_and_exact_inference(
    tmp_path, monkeypatch
):
    """Exercise label-free symmetry and saved prediction audits on a tiny model."""
    from research.pc_analytic_emulator.refinement_20261117.reproduction import (
        audit_selection,
    )

    model, arrays = _problem("residual")
    geometry = {"DIM": 6, "HII_DIM": 6, "BOX_LEN": 96.0}
    physics = {
        "simulation_options": geometry,
        "astro_options": {
            "RECOMB_MODEL": "none",
            "CELL_RECOMB": False,
            "USE_TS_FLUCT": False,
            "USE_MCGS": False,
            "USE_REIONIZATION_PHOTOHEATING_FEEDBACK": False,
            "PHOTON_CONS_TYPE": "no-photoncons",
        },
        "matter_options": {"SOURCE_MODEL": "L-INTEGRAL"},
    }
    model.metadata.update(teacher_physics=physics, training_geometries=[geometry])
    directory = tmp_path / "model"
    model.save(directory)
    source = tmp_path / "native" / "source.npz"
    source.parent.mkdir()
    np.savez(
        source,
        source=arrays[0],
        capacity=arrays[1],
        native_es_xhi=arrays[2],
        teacher_xHI=np.full_like(arrays[0], np.nan),
    )
    source.with_name("native.json").write_text(
        json.dumps(
            {
                "identity": {
                    "redshift": 7.5,
                    "inputs": {
                        **physics,
                        "simulation_options": {**geometry, "N_THREADS": 1},
                    },
                }
            }
        )
    )
    selection = tmp_path / "selection.json"
    selection.write_text("{}")
    monkeypatch.setattr(audit, "verify_sources", lambda *args: {"passed": True})
    monkeypatch.setattr(
        audit_selection, "audit_selection", lambda *args, **kwargs: {"passed": True}
    )
    result = audit.run(
        directory,
        selection,
        {"validation": source},
        {},
        tmp_path / "audit",
        frozen_source=tmp_path,
        source_hashes=tmp_path / "unused.json",
        physics_reference=tmp_path,
    )
    assert result["passed"]
    assert not result["teacher_labels_used_for_symmetry_or_inference"]
    row = result["snapshots"]["validation"]
    assert row["symmetry"]["checks"] == 55
    assert row["symmetry"]["field_atol"] == 1e-7
    assert row["exact_model_roundtrip"]["passed"]
    assert row["saved_prediction_photon_ledger"]["relative_photon_residual"] < 1e-10
