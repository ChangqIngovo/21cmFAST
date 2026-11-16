"""Verify portable oracle evidence using only tiny synthetic saved fields."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from research.pc_analytic_emulator import diagnose_errors
from research.pc_analytic_emulator.refinement_20261117.reproduction import (
    oracle_evidence as evidence,
)

PROFILES = ("original", "resolution", "volume", "unseen_box")
METHODS = diagnose_errors.METHODS


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def _snapshot(
    root: Path,
    profile: str,
    seed: int,
    redshift: float,
    teacher: float,
    deployed: float,
) -> tuple[dict, dict]:
    label = f"z{redshift:g}".replace(".", "p")
    benchmark_root = root / f"benchmark_blind_{profile}"
    oracle_root = root / f"oracle_blind_{profile}"
    benchmark_path = benchmark_root / f"seed{seed}" / label / "predictions.npz"
    oracle_path = oracle_root / f"seed{seed}" / label / "fields.npz"
    benchmark_path.parent.mkdir(parents=True)
    oracle_path.parent.mkdir(parents=True)
    source, capacity = np.full((1, 1, 1), 2.0), np.ones((1, 1, 1))
    truth, prediction = np.full_like(source, teacher), np.full_like(source, deployed)
    np.savez_compressed(
        benchmark_path,
        source=source,
        capacity=capacity,
        teacher_xHI=truth,
        selected_xHI=prediction,
    )
    fields = {
        "deployed": prediction,
        "oracle_final_budget": truth,
        "oracle_recursive_budget": truth,
        "teacher_predicted_budget": prediction,
    }
    np.savez_compressed(
        oracle_path,
        source=source,
        capacity=capacity,
        teacher_xHI=truth,
        **{f"{name}_xHI": value for name, value in fields.items()},
        deployed_latent_xHII=1 - prediction,
        oracle_final_budget_latent_xHII=1 - prediction,
        oracle_recursive_budget_latent_xHII=1 - truth,
    )
    teacher_hash = hashlib.sha256(f"synthetic-{seed}-{redshift}".encode()).hexdigest()
    field_metrics = {
        name: {
            "MAE": float(np.mean(np.abs(value - truth))),
            "RMSE": float(np.sqrt(np.mean((value - truth) ** 2))),
            "capacity_weighted_MSE": float(np.mean((value - truth) ** 2)),
            "delta_Q_mass": float(np.mean(truth - value)),
        }
        for name, value in fields.items()
    }
    row = {
        "seed": seed,
        "redshift": redshift,
        "dimension": 1,
        "box_len": 50.0,
        "teacher_xHI_mass": teacher,
        "field_metrics": field_metrics,
        "budgets": {name: {"relative_photon_residual": 0.0} for name in METHODS},
        "teacher_projection_is_weighted_MSE_lower_bound": True,
        "low_k_21cm": {
            "comparisons": {
                name: {
                    "mean_absolute_power_ratio_error": None,
                    "defined_shells": 0,
                    "minimum_cross_correlation": None,
                }
                for name in METHODS
            }
        },
        "files_sha256": {
            "fields.npz": _digest(oracle_path),
            "teacher_fields.npz": teacher_hash,
        },
    }
    reference = {
        "seed": seed,
        "redshift": redshift,
        "files": {
            "predictions.npz": {
                "path": str(benchmark_path.relative_to(benchmark_root)),
                "sha256": _digest(benchmark_path),
            }
        },
        "original_teacher_fields_sha256": teacher_hash,
    }
    return row, reference


def _fixture(tmp_path: Path) -> Path:
    root = tmp_path / "synthetic_campaign"
    model = root / "selected_model"
    _write(model / "model.json", {"schema": 2})
    np.savez_compressed(model / "field_model.npz", synthetic=np.asarray([1.0]))
    np.savez_compressed(model / "budget_model.npz", synthetic=np.asarray([0.9]))
    model_hashes = evidence._model_hashes(model.resolve())
    _write(
        root / "selection.json",
        {
            "model_files_sha256": model_hashes,
            "inference_source_sha256": evidence._inference_source_hashes(),
        },
    )
    _write(root / "evaluation_completion.json", {"complete": True, "synthetic": True})
    for index, profile in enumerate(PROFILES):
        pairs = [
            _snapshot(root, profile, 9001 + index, 7.5, 0.4, 0.3),
            _snapshot(root, profile, 9001 + index, 6.0, 0.01, 0.0),
        ]
        _write(
            root / f"benchmark_blind_{profile}" / "report.json",
            {
                "complete": True,
                "snapshots": [reference for _, reference in pairs],
            },
        )
        (root / f"benchmark_blind_{profile}" / "COMPLETED").write_text(
            "Synthetic complete\n"
        )
        _write(
            root / f"oracle_blind_{profile}" / "summary.json",
            {
                "complete": True,
                "diagnostic_only": True,
                "fitting_performed": False,
                "model_selection_performed": False,
                "model_sha256": model_hashes,
                "diagnostic_source_sha256": _digest(Path(diagnose_errors.__file__)),
                "snapshots": [row for row, _ in pairs],
            },
        )
    return root


def _modify_oracle_report(root: Path, update: Callable[[dict], None]) -> None:
    path = root / "oracle_blind_original" / "summary.json"
    report = json.loads(path.read_text())
    update(report["snapshots"][0])
    _write(path, report)


def test_portable_builder_retains_science_with_path_free_output(tmp_path: Path) -> None:
    """Portable provenance can differ while all scientific evidence stays identical."""
    root = _fixture(tmp_path)
    first = evidence.run(root)
    assert json.loads((root / "oracle_evidence_portable.json").read_text()) == first
    original = copy.deepcopy(first)
    original["evidence_builder_sha256"] = "0" * 64
    original["original_evidence"] = {"ignored_prior_provenance": True}
    original_path = tmp_path / "original_oracle_evidence.json"
    _write(original_path, original)
    destination = tmp_path / "portable" / "evidence.json"

    reproduced = evidence.run(root, destination, original_evidence=original_path)

    assert evidence._scientific_payload(reproduced) == evidence._scientific_payload(
        original
    )
    assert reproduced["original_evidence"]["scientific_payload_identical"] is True
    assert str(tmp_path) not in json.dumps(reproduced)
    assert (
        reproduced["verification"][
            "all_deployed_fields_bitwise_match_frozen_benchmarks"
        ]
        is True
    )
    assert (
        reproduced["verification"][
            "all_applicable_teacher_projection_MSE_lower_bounds_satisfied"
        ]
        is True
    )
    assert reproduced["verification"]["maximum_diagnostic_photon_residual"] == 0
    for profile in reproduced["profiles"].values():
        assert profile["snapshot_count"] == 2
        assert profile["summaries"]["active"]["count"] == 1
        assert profile["summaries"]["late"]["count"] == 1
        assert profile["summaries"]["all"]["methods"]["deployed"][
            "MAE"
        ] == pytest.approx(0.055)


def test_grouped_keeps_undefined_spectra_and_nonadditive_scope(tmp_path: Path) -> None:
    """A uniform teacher has no defined spectrum ratio even for exact fields."""
    root = _fixture(tmp_path)
    rows = json.loads((root / "oracle_blind_original" / "summary.json").read_text())[
        "snapshots"
    ]
    result = evidence.grouped(rows)
    for method in METHODS:
        assert (
            result["methods"][method]["mean_snapshot_low_k_power_ratio_error"] is None
        )
        assert result["methods"][method]["defined_power_snapshots"] == 0
        assert result["methods"][method]["undefined_power_snapshots"] == 2
        assert result["methods"][method]["minimum_cross_correlation"] is None
    assert result["counterfactual_comparisons"]["additive_error_partition"] is False
    oracle = result["counterfactual_comparisons"]["oracle_changes_relative_to_deployed"]
    assert oracle["oracle_final_budget"]["relative_MAE_change"] == -1
    empty = evidence.grouped([])
    assert empty["methods"]["deployed"]["MAE"] is None
    assert (
        empty["counterfactual_comparisons"][
            "teacher_projection_to_deployed_weighted_MSE_ratio"
        ]
        is None
    )


def test_changed_deployed_field_fails_even_after_refreshing_npz_hash(
    tmp_path: Path,
) -> None:
    """Independent saved predictions must match the original benchmark bitwise."""
    root = _fixture(tmp_path)
    fields = root / "oracle_blind_original" / "seed9001" / "z7p5" / "fields.npz"
    with np.load(fields, allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    arrays["deployed_xHI"] += 0.01
    np.savez_compressed(fields, **arrays)
    _modify_oracle_report(
        root, lambda row: row["files_sha256"].update({"fields.npz": _digest(fields)})
    )
    with pytest.raises(ValueError, match=r"(?i)(deployed|bitwise|verification)"):
        evidence.run(root)


def test_frozen_model_weight_mismatch_fails(tmp_path: Path) -> None:
    """The builder cannot attribute controls to a checkpoint changed after selection."""
    root = _fixture(tmp_path)
    np.savez_compressed(
        root / "selected_model" / "field_model.npz", synthetic=np.asarray([2.0])
    )
    with pytest.raises(ValueError, match=r"(?i)(model|hash|weight|verification)"):
        evidence.run(root)


def test_capacity_mse_lower_bound_violation_fails(tmp_path: Path) -> None:
    """A declared optimal teacher projection cannot exceed deployed weighted error."""
    root = _fixture(tmp_path)
    _modify_oracle_report(
        root,
        lambda row: row["field_metrics"]["teacher_predicted_budget"].update(
            capacity_weighted_MSE=1.0
        ),
    )
    with pytest.raises(ValueError, match=r"(?i)(bound|MSE|verification)"):
        evidence.run(root)


def test_nonzero_oracle_mass_bias_fails(tmp_path: Path) -> None:
    """Exact teacher-count interventions must reproduce the teacher ionized mass."""
    root = _fixture(tmp_path)
    _modify_oracle_report(
        root,
        lambda row: row["field_metrics"]["oracle_recursive_budget"].update(
            delta_Q_mass=1e-3
        ),
    )
    with pytest.raises(ValueError, match=r"(?i)(oracle|mass|verification)"):
        evidence.run(root)


def test_changed_original_scientific_payload_fails(tmp_path: Path) -> None:
    """Provenance exemptions must not conceal altered scientific conclusions."""
    root = _fixture(tmp_path)
    original = evidence.run(root)
    original["profiles"]["original"]["snapshot_count"] = 100
    original_path = tmp_path / "changed_original.json"
    _write(original_path, original)
    with pytest.raises(
        ValueError, match=r"(?i)(scientific|original|evidence|verification)"
    ):
        evidence.run(root, tmp_path / "new.json", original_evidence=original_path)


def test_original_evidence_cannot_be_overwritten(tmp_path: Path) -> None:
    """Verification preserves the original evidence file exactly."""
    root = _fixture(tmp_path)
    original = evidence.run(root)
    original_path = root / "oracle_evidence.json"
    _write(original_path, original)
    before = original_path.read_bytes()
    with pytest.raises(
        ValueError, match=r"(?i)(output|overwrite|replace|original|verification)"
    ):
        evidence.run(root, original_path, original_evidence=original_path)
    assert original_path.read_bytes() == before
