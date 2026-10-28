"""Regression checks for reusing exactly the completed full-history dataset."""
import hashlib
import json

import pytest

from research.es_pc_emulator.history_retrain import prepare, verify


def parent_run(tmp_path):
    """Create immutable stand-ins for three full-history source files."""
    parent = tmp_path / "parent"
    (parent / "native").mkdir(parents=True)
    (parent / "training").mkdir()
    (parent / "COMPLETED").touch()
    records = []
    for split in ("train", "validation", "test"):
        path = parent / "native" / (split + ".npz")
        path.write_bytes(split.encode())
        records.append({"case_id": split + "_000", "split": split, "seed": 111,
                        "path": str(path), "fields_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = parent / "native/dataset_manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "cases": records}))
    old = {"history_steps": 3, "width": 16, "batch_size": 8, "auxiliary_weight": .01,
           "power_floor": 1e-8, "snapshot_scope": "all", "manifest": str(manifest),
           "cases": [{"case_id": r["case_id"], "split": r["split"], "seed": r["seed"],
                      "sha256": r["fields_sha256"]} for r in records]}
    (parent / "training/configuration.json").write_text(json.dumps(old))
    coverage = {"passed": True, "case_count": 3, "cases": records,
                "requested": {"z_start": 20., "z_end": 4., "snapshots": 41,
                              "maximum_final_xHI_volume_and_mass": .01}}
    (parent / "coverage.json").write_text(json.dumps(coverage))
    return parent, records


def test_prepare_retains_cases_and_splits_and_runs_no_simulations(tmp_path):
    """The new workflow changes training choices while keeping every data identity."""
    parent, records = parent_run(tmp_path)
    run = tmp_path / "retrain"
    config = prepare(run, parent, {})
    copied = json.loads((run / "dataset_manifest.json").read_text())
    assert copied["cases"] == records
    assert config["training"]["training_objective"] == "stage-relative"
    assert config["training"]["snapshot_scope"] == "all"
    assert config["training"]["learning_rate"] == pytest.approx(.0003)
    assert config["training"]["external_cases"] == 0
    assert config["native_simulations_run"] is False
    assert verify(run)["parent_run"] == str(parent)


def test_prepare_rejects_changed_training_split_and_file(tmp_path):
    """A completed marker does not override changed source identities or hashes."""
    parent, records = parent_run(tmp_path)
    path = parent / "training/configuration.json"
    old = json.loads(path.read_text())
    old["cases"][0]["seed"] = 999
    path.write_text(json.dumps(old))
    with pytest.raises(ValueError, match="identity"):
        prepare(tmp_path / "retrain", parent, {})
    old["cases"][0]["seed"] = 111
    path.write_text(json.dumps(old))
    from pathlib import Path
    Path(records[0]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        prepare(tmp_path / "retrain2", parent, {})


def test_verification_detects_changes_after_submission(tmp_path):
    """Queued or resumed training cannot silently switch to a changed dataset."""
    parent, _ = parent_run(tmp_path)
    run = tmp_path / "retrain"
    prepare(run, parent, {})
    manifest = run / "dataset_manifest.json"
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="manifest"):
        verify(run)


def test_prepare_rejects_changed_recorded_manifest_or_unusable_prior(tmp_path):
    """Retain the original training manifest provenance and safe float32 bounds."""
    parent, _ = parent_run(tmp_path)
    with pytest.raises(ValueError, match="float32"):
        prepare(tmp_path / "bad_epsilon", parent, {"RETRAIN_PRIOR_EPSILON": "1e-12"})
    path = parent / "training/configuration.json"
    old = json.loads(path.read_text())
    old["manifest_sha256"] = "0" * 64
    path.write_text(json.dumps(old))
    with pytest.raises(ValueError, match="manifest checksum"):
        prepare(tmp_path / "bad_manifest", parent, {})
