"""Protect fitting-ceiling inputs and keep held-out simulations out of fitting."""
import hashlib
import json
from pathlib import Path

import pytest

from research.es_pc_emulator.fit_ceiling_run import prepare, verify


def parent_run(tmp_path: Path, retrained: bool = True) -> tuple[Path, list[dict]]:
    """Create source metadata for two training and three held-out fields."""
    parent = tmp_path / "parent"
    (parent / "native").mkdir(parents=True)
    (parent / "training").mkdir()
    (parent / "COMPLETED").touch()
    rows = []
    for index, split in enumerate(("train", "train", "validation", "test", "external_test")):
        path = parent / "native" / f"{index}.npz"
        path.write_bytes(str(index).encode())
        rows.append({"case_id": f"{split}_{index:03}", "split": split, "seed": 111,
                     "path": str(path), "fields_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = parent / ("dataset_manifest.json" if retrained else "native/dataset_manifest.json")
    manifest.write_text(json.dumps({"schema_version": 1, "cases": rows}))
    config = {"manifest": str(manifest), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
              "snapshot_scope": "all", "batch_size": 8, "history_steps": 3,
              "cases": [{"case_id": r["case_id"], "split": r["split"], "seed": r["seed"],
                         "sha256": r["fields_sha256"]} for r in rows]}
    (parent / "training/configuration.json").write_text(json.dumps(config))
    (parent / "coverage.json").write_text(json.dumps({"passed": True, "case_count": len(rows), "cases": rows}))
    return parent, rows


@pytest.mark.parametrize("retrained", [True, False])
def test_prepare_preserves_nested_training_cases_and_parent_types(tmp_path, retrained):
    """Both full-history and corrected-training parents retain the same two cases."""
    parent, rows = parent_run(tmp_path, retrained)
    run = tmp_path / "fit"
    config = prepare(run, parent, {})
    assert config["selected_training_cases"] == [r["case_id"] for r in rows[:2]]
    assert config["experiment"]["case_counts"] == [1, 2]
    assert config["experiment"]["epochs"] == 400
    assert len(config["experiment"]["arms"]) == 5
    assert config["held_out_cases_used_for_fitting"] == 0
    assert config["native_simulations_run"] is False
    assert verify(run) == config


def test_selected_field_or_original_manifest_change_fails(tmp_path):
    """Queued jobs cannot silently fit changed fields or different source inventory."""
    parent, rows = parent_run(tmp_path)
    run = tmp_path / "fit"
    prepare(run, parent, {})
    Path(rows[0]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Selected training field"):
        verify(run)
    with pytest.raises(ValueError, match="Selected training field"):
        prepare(tmp_path / "again", parent, {})


def test_held_out_field_contents_are_not_read_for_fitting_probe(tmp_path):
    """Only train cases are content-hashed; held-out identity records remain immutable."""
    parent, rows = parent_run(tmp_path)
    run = tmp_path / "fit"
    prepare(run, parent, {})
    Path(rows[-1]["path"]).write_bytes(b"unread held-out content")
    assert verify(run)["held_out_cases_used_for_fitting"] == 0


def test_prepare_rejects_changed_split_and_unsupported_case_count(tmp_path):
    """Do not substitute validation cases or expand the intended fitting ceiling."""
    parent, _ = parent_run(tmp_path)
    with pytest.raises(ValueError, match="case counts"):
        prepare(tmp_path / "bad", parent, {"FIT_CASE_COUNTS": "1,3"})
    config_path = parent / "training/configuration.json"
    config = json.loads(config_path.read_text())
    config["cases"][0]["split"] = "validation"
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="identity"):
        prepare(tmp_path / "bad2", parent, {})


def test_verify_rejects_changed_frozen_configuration(tmp_path):
    """Manifest provenance remains anchored even before any model checkpoint exists."""
    parent, _ = parent_run(tmp_path)
    run = tmp_path / "fit"
    prepare(run, parent, {})
    path = run / "dataset_manifest.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        verify(run)
