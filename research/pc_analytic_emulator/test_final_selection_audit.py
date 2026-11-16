"""Independent final-selection audits use sealed synthetic artifacts only."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from research.pc_analytic_emulator.refinement_20261117.reproduction.audit_selection import (
    audit_selection,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hashes(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): _hash(p) for p in root.rglob("*") if p.is_file()}


def _seal(root: Path) -> None:
    for folder in (root / "fit_cache", root / "trials/seed23"):
        files = {k: v for k, v in _hashes(folder).items() if k != "receipt.json"}
        record = json.loads(
            (
                folder / ("trial.json" if "trials" in folder.parts else "features.json")
            ).read_text()
        )
        _write(
            folder / "receipt.json", {"files": files, "identity": {}, "record": record}
        )
    files = {
        k: v
        for k, v in _hashes(root).items()
        if k not in {"completion.json", "COMPLETED"}
    }
    _write(root / "completion.json", {"files": files})
    (root / "COMPLETED").write_text("complete\n")


@pytest.fixture
def artifacts(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    """Seal three-geometry synthetic training and final-selection evidence."""
    run = tmp_path / "train"
    geometries = [(16, 50.0), (32, 50.0), (32, 256.0)]
    snapshots = [
        {"seed": 100 + j, "redshift": z, "dimension": n, "box_len": length}
        for j, (n, length) in enumerate(geometries)
        for z in (8.0, 7.5)
    ]
    features = {"validation_snapshots": snapshots, "test_labels_in_fit_cache": False}
    _write(run / "features.json", features)
    _write(run / "fit_cache/features.json", features)
    field_metadata = {
        "best_epoch": 1,
        "best_validation_loss": 0.2,
        "completed_epochs": 2,
        "validation_snapshots": 6,
        "min_delta": 1e-9,
        "initial_baseline_metrics": {"epoch": 0, "validation_loss": 0.4},
        "selection_objective": "equal-snapshot mean postprojection objective; raw auxiliary excluded",
    }
    for folder in (
        run / "model",
        run / "trials/seed23/model",
        run / "trials/seed23/trained_model",
    ):
        _write(
            folder / "model.json",
            {
                "schema": 4,
                "metadata": {
                    "model_seed": 23,
                    "selected_kind": "warm_started_residual",
                },
            },
        )
        np.savez(folder / "field_model.npz", metadata_json=json.dumps(field_metadata))
        np.savez(
            folder / "budget_model.npz",
            metadata_json=json.dumps({"best_validation_loss": 7.0}),
        )
    history = {
        "field": [
            {
                "epoch": 1,
                "validation_loss": 0.2,
                "validation_raw_auxiliary_loss": 99.0,
                "selected": True,
            },
            {
                "epoch": 2,
                "validation_loss": 0.1999999995,
                "validation_raw_auxiliary_loss": 0.0,
                "selected": False,
            },
        ]
    }
    _write(run / "trials/seed23/history.json", history)
    trial = {
        "model_seed": 23,
        "selected_kind": "warm_started_residual",
        "validation_objective": 0.2,
        "field_best_epoch": 1,
        "field_completed_epochs": 2,
        "test_used": False,
        "model_directory": str(run / "trials/seed23/model"),
        "candidates": [{"kind": "warm_started_residual", "validation_objective": 0.2}],
    }
    _write(run / "trials/seed23/trial.json", trial)
    _write(
        run / "model_selection.json",
        {"selected": trial, "trials": [trial], "test_used_for_selection": False},
    )
    _write(
        run / "summary.json",
        {
            "complete": True,
            "selected_model_seed": 23,
            "selected_kind": "warm_started_residual",
            "selected_validation_objective": 0.2,
            "sample_counts": {"validation": 6},
            "test_used_for_selection": False,
        },
    )
    _seal(run)
    previous = tmp_path / "previous"
    shutil.copytree(run / "model", previous)
    _write(previous / "model.json", {"schema": 4, "metadata": {"role": "previous"}})
    candidates = {}
    for name, model, score in (
        ("guarded_mixed", run / "model", 0.2),
        ("previous_model", previous, 0.3),
    ):
        rows = [
            dict(
                row,
                geometry={
                    "DIM": 96,
                    "HII_DIM": row["dimension"],
                    "BOX_LEN": row["box_len"],
                },
                projected_loss=score,
            )
            for row in snapshots
        ]
        report = {
            "complete": True,
            "fitting_performed": False,
            "test_evaluated": False,
            "model_selection_performed": False,
            "model_weights_unchanged": True,
            "snapshot_count": 6,
            "validation_objective": score,
            "model_sha256": _hashes(model),
            "objective": {"spectral_weight": 0.1},
            "snapshots": rows,
            "by_geometry": [
                {
                    "geometry": {"DIM": 96, "HII_DIM": n, "BOX_LEN": length},
                    "snapshot_count": 2,
                }
                for n, length in geometries
            ],
        }
        report_path = tmp_path / f"{name}.json"
        _write(report_path, report)
        candidates[name] = {
            "validation_objective": score,
            "validation_report": report_path.name,
            "validation_report_sha256": _hash(report_path),
            "model_root": str(model.relative_to(tmp_path)),
            "model_files_sha256": _hashes(model),
        }
    candidates["guarded_mixed"]["training_run"] = "train"
    selected = tmp_path / "selected_model"
    shutil.copytree(run / "model", selected)
    path = tmp_path / "selection.json"
    _write(
        path,
        {
            "schema": 1,
            "selected_experiment": "guarded_mixed",
            "selected_validation_objective": 0.2,
            "selected_model_root": "selected_model",
            "model_files_sha256": _hashes(selected),
            "candidates": candidates,
            "blind_or_transfer_used_for_selection": False,
        },
    )
    return path, selected, {"guarded_mixed": run}


def test_audits_projected_checkpoints_without_confusing_raw_loss(artifacts) -> None:
    """Honor projected min_delta selection despite a smaller raw auxiliary loss."""
    report = audit_selection(*artifacts, expected_validation_count=6)
    assert report["passed"] is True
    assert report["training_runs"]["guarded_mixed"]["validation_geometry_counts"] == [
        2,
        2,
        2,
    ]
    assert report["training_runs"]["guarded_mixed"]["checkpoints"][0]["best_epoch"] == 1
    assert "/workspace/" not in json.dumps(report)


@pytest.mark.parametrize(
    "fault",
    [
        "nonfinite",
        "wrong_winner",
        "test_used",
        "bad_hash",
        "different_model",
        "row_count",
        "geometry_balance",
    ],
)
def test_rejects_invalid_final_evidence(artifacts, fault: str) -> None:
    """Reject changed scores, hashes, weights, selection or validation population."""
    path, model, runs = artifacts
    record = json.loads(path.read_text())
    if fault == "nonfinite":
        record["candidates"]["previous_model"]["validation_objective"] = float("nan")
    elif fault == "wrong_winner":
        record["selected_experiment"] = "previous_model"
    elif fault == "test_used":
        record["blind_or_transfer_used_for_selection"] = True
    elif fault == "bad_hash":
        record["candidates"]["guarded_mixed"]["validation_report_sha256"] = "0" * 64
    elif fault == "different_model":
        (model / "field_model.npz").write_bytes(b"changed")
    else:
        candidate = record["candidates"]["previous_model"]
        report_path = path.parent / candidate["validation_report"]
        report = json.loads(report_path.read_text())
        if fault == "row_count":
            report["snapshots"].pop()
        else:
            report["snapshots"][0]["geometry"] = report["snapshots"][-1]["geometry"]
        _write(report_path, report)
        candidate["validation_report_sha256"] = _hash(report_path)
    _write(path, record)
    with pytest.raises((ValueError, TypeError)):
        audit_selection(path, model, runs, expected_validation_count=6)


@pytest.mark.parametrize(
    "fault", ["completion", "summary", "checkpoint", "history", "cache"]
)
def test_rejects_invalid_training_evidence_even_when_resealed(
    artifacts, fault: str
) -> None:
    """Semantic checks catch forged histories even when their checksums match."""
    path, model, runs = artifacts
    run = runs["guarded_mixed"]
    if fault == "completion":
        (run / "completion.json").unlink()
    elif fault == "summary":
        summary = json.loads((run / "summary.json").read_text())
        summary["selected_validation_objective"] = 0.1
        _write(run / "summary.json", summary)
        _seal(run)
    elif fault in {"checkpoint", "history"}:
        history_path = run / "trials/seed23/history.json"
        history = json.loads(history_path.read_text())
        history["field"][1]["selected"] = True
        if fault == "checkpoint":
            history["field"][1]["validation_loss"] = 0.1
        _write(history_path, history)
        _seal(run)
    else:
        features = json.loads((run / "features.json").read_text())
        features["validation_snapshots"].pop()
        _write(run / "features.json", features)
        _write(run / "fit_cache/features.json", features)
        _seal(run)
    with pytest.raises((ValueError, TypeError, FileNotFoundError)):
        audit_selection(path, model, runs, expected_validation_count=6)


def test_previous_model_wins_exact_final_tie(artifacts) -> None:
    """Retain the unchanged previous model when final objectives tie exactly."""
    path, model, runs = artifacts
    frozen = json.loads(path.read_text())
    previous = frozen["candidates"]["previous_model"]
    report_path = path.parent / previous["validation_report"]
    report = json.loads(report_path.read_text())
    report["validation_objective"] = 0.2
    for row in report["snapshots"]:
        row["projected_loss"] = 0.2
    _write(report_path, report)
    previous["validation_objective"] = 0.2
    previous["validation_report_sha256"] = _hash(report_path)
    _write(path, frozen)
    with pytest.raises(ValueError, match="tie rule"):
        audit_selection(path, model, runs, expected_validation_count=6)
    shutil.rmtree(model)
    shutil.copytree(path.parent / previous["model_root"], model)
    frozen["selected_experiment"] = "previous_model"
    frozen["model_files_sha256"] = previous["model_files_sha256"]
    _write(path, frozen)
    assert audit_selection(path, model, runs, expected_validation_count=6)["passed"]


def test_trial_exact_tie_uses_lowest_model_seed(artifacts) -> None:
    """Enforce the training selector's deterministic seed tie-break."""
    path, model, runs = artifacts
    run = runs["guarded_mixed"]
    selection = json.loads((run / "model_selection.json").read_text())
    alternative = dict(selection["selected"], model_seed=11)
    selection["trials"].append(alternative)
    _write(run / "model_selection.json", selection)
    _seal(run)
    with pytest.raises(ValueError, match="seed tie-break"):
        audit_selection(path, model, runs, expected_validation_count=6)
