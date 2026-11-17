"""Verify sealed cache relocation and leakage checks for residual refinement."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from research.pc_analytic_emulator.finalization_20261117.refine import (
    main,
    verified_cache,
)
from research.pc_analytic_emulator.model import budget_features, field_features
from research.pc_analytic_emulator.test_model import fixture_model


def _cache(root: Path) -> Path:
    root.mkdir()
    source = np.full((4, 4, 4), 0.4)
    atoms = np.ones_like(source)
    neutral = np.linspace(0.1, 0.9, 64).reshape(source.shape)
    features, names, baseline = field_features(source, atoms, neutral, 7.5, 50)
    glob, global_names = budget_features(source, atoms, neutral, 7.5, 50)
    data = {
        "test_labels_in_fit_cache": False,
        "field_feature_names": names,
        "budget_feature_names": global_names,
        "feature_set": "legacy",
    }
    for index, split in enumerate(("training", "validation"), 1):
        name = f"{split}.npz"
        np.savez(
            root / name,
            features=features,
            baseline=baseline,
            teacher_xHI=neutral,
            capacity=atoms,
            source=source,
            global_features=glob,
            native_es_xHI=neutral,
            box_len=np.asarray(50.0),
        )
        data[split + "_snapshots"] = [
            {
                "cache": str(root / name),
                "seed": index,
                "redshift": 7.5,
                "dimension": 4,
                "box_len": 50.0,
            }
        ]
    (root / "features.json").write_text(json.dumps(data))
    _seal(root)
    fixture_model(source, atoms, neutral).save(root / "base")
    return root / "features.json"


def _seal(root: Path) -> None:
    files = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.iterdir()
        if p.is_file() and p.name != "receipt.json"
    }
    receipt = {
        "files": files,
        "record": json.loads((root / "features.json").read_text()),
        "identity": {"configuration": {"train_seeds": [1], "validation_seeds": [2]}},
    }
    (root / "receipt.json").write_text(json.dumps(receipt))


def test_cache_relocation_preserves_original_metadata_and_checks_bytes(
    tmp_path: Path,
) -> None:
    """Copied caches resolve from their current folder, not saved machine paths."""
    features = _cache(tmp_path / "original")
    moved = tmp_path / "moved"
    shutil.copytree(features.parent, moved)
    data, receipt = verified_cache(moved / "features.json")
    assert data["training_snapshots"][0]["cache"] == str(moved / "training.npz")
    assert receipt["record"]["training_snapshots"][0]["cache"] == str(
        features.parent / "training.npz"
    )
    (moved / "training.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        verified_cache(moved / "features.json")


def test_cache_refuses_overlapping_seeds_and_test_labels(tmp_path: Path) -> None:
    """Resealing does not bypass independent split or test-exclusion checks."""
    features = _cache(tmp_path / "cache")
    original = json.loads(features.read_text())
    changed = json.loads(features.read_text())
    changed["validation_snapshots"][0]["seed"] = 1
    features.write_text(json.dumps(changed))
    _seal(features.parent)
    receipt = json.loads((features.parent / "receipt.json").read_text())
    receipt["identity"]["configuration"]["validation_seeds"] = [1]
    (features.parent / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="overlap"):
        verified_cache(features)
    original["test_labels_in_fit_cache"] = True
    features.write_text(json.dumps(original))
    _seal(features.parent)
    with pytest.raises(ValueError, match="test labels"):
        verified_cache(features)


def test_refinement_cli_runs_on_small_sealed_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run the reusable CLI through fitting, checkpoint, and frozen-input checks."""
    features = _cache(tmp_path / "cache")
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "refine",
            "--features",
            str(features),
            "--base",
            str(features.parent / "base"),
            "--output",
            str(output),
            "--epochs",
            "1",
        ],
    )
    main()
    report = json.loads((output / "report.json").read_text())
    assert report["history"][0]["snapshot_updates"] == 1
    assert np.isfinite(report["validation_score"])
    assert (output / "model/model.json").is_file()


@pytest.mark.parametrize("control", ["feature_set", "radii_cMpc"])
def test_refinement_rejects_mismatched_feature_conventions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, control: str
) -> None:
    """A valid cache checksum does not imply compatible deployment features."""
    features = _cache(tmp_path / "cache")
    if control == "feature_set":
        data = json.loads(features.read_text())
        data["feature_set"] = "multiscale_geometry"
        features.write_text(json.dumps(data))
        _seal(features.parent)
    else:
        receipt_path = features.parent / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["identity"]["configuration"]["radii_cMpc"] = [1.0, 2.0]
        receipt_path.write_text(json.dumps(receipt))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "refine",
            "--features",
            str(features),
            "--base",
            str(features.parent / "base"),
            "--output",
            str(tmp_path / "output"),
            "--epochs",
            "1",
        ],
    )
    with pytest.raises(ValueError, match="frozen base"):
        main()
