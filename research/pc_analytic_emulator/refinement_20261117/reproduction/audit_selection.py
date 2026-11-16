"""Independently audit frozen validation selection and checkpoint evidence.

This reader performs no fitting, prediction, teacher loading or blind evaluation.
It checks saved artifact hashes, equal validation populations, deterministic
selection, and projected checkpoint histories. Raw auxiliary losses are never
substituted for the objective used to select a projected field checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from research.pc_analytic_emulator.predict import _model_hashes


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _score(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Validation scores must be finite nonnegative numbers")
    if not math.isfinite(value) or value < 0:
        raise ValueError("Validation scores must be finite and nonnegative")
    return float(value)


def _equal(left: Any, right: Any, message: str) -> None:
    if not math.isclose(_score(left), _score(right), rel_tol=1e-11, abs_tol=1e-12):
        raise ValueError(message)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def _verify_files(root: Path, manifest: dict) -> set[str]:
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("A completed artifact requires nonempty file checksums")
    for name, expected in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError("Receipt refers to a missing or external file")
        if not isinstance(expected, str) or _digest(path) != expected:
            raise ValueError("Receipt checksum differs from saved artifact")
    return set(files)


def _validation_rows(
    rows: Any, expected_count: int, *, report: bool
) -> tuple[dict, list[tuple]]:
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError("Validation snapshot count differs from declared comparison")
    identities, populations = [], Counter()
    for row in rows:
        if report:
            geometry = row["geometry"]
            dimension, length = geometry["HII_DIM"], geometry["BOX_LEN"]
        else:
            dimension, length = row["dimension"], row["box_len"]
        if (
            type(dimension) is not int
            or dimension < 2
            or not math.isfinite(length)
            or length <= 0
        ):
            raise ValueError("Invalid validation geometry")
        redshift = _score(row["redshift"])
        if type(row["seed"]) is not int:
            raise ValueError("Validation seeds must be integers")
        identity = (row["seed"], redshift, dimension, float(length))
        identities.append(identity)
        populations[(dimension, float(length))] += 1
        if report:
            _score(row["projected_loss"])
    if len(set(identities)) != expected_count:
        raise ValueError("Validation snapshots are duplicated")
    if len(populations) != 3 or set(populations.values()) != {expected_count // 3}:
        raise ValueError("Validation must have equal populations in three geometries")
    return dict(populations), sorted(identities)


def _head_metadata(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        value = archive["metadata_json"]
        if value.shape != () or value.dtype.kind != "U":
            raise ValueError("Checkpoint metadata must be a Unicode JSON scalar")
        metadata = json.loads(str(value))
    if not isinstance(metadata, dict):
        raise TypeError("Checkpoint metadata must be an object")
    return metadata


def _projected_checkpoint(
    folder: Path, trial: dict, expected_count: int
) -> dict | None:
    """Replay the declared strict min_delta rule, including eligible epoch zero."""
    path = folder / "trained_model/field_model.npz"
    if not path.is_file():
        return None
    metadata = _head_metadata(path)
    if "selection_objective" not in metadata:
        return None  # Legacy MLP metadata describes its own raw/weighted MSE.
    initial = metadata.get("initial_baseline_metrics", {})
    if initial.get("epoch") != 0:
        raise ValueError("Projected checkpoint lacks its eligible epoch-zero baseline")
    best, best_epoch = _score(initial["validation_loss"]), 0
    min_delta = _score(metadata["min_delta"])
    history = _json(folder / "history.json")["field"]
    if not isinstance(history, list):
        raise TypeError("Projected checkpoint history must contain epoch rows")
    for index, row in enumerate(history, 1):
        if row["epoch"] != index:
            raise ValueError("Projected checkpoint epochs are missing or duplicated")
        score = _score(row["validation_loss"])
        improved = score < best - min_delta
        if row.get("selected") is not improved:
            raise ValueError(
                "Checkpoint history violates its strict min_delta selection rule"
            )
        if improved:
            best, best_epoch = score, index
    if (
        metadata["best_epoch"] != best_epoch
        or metadata["completed_epochs"] != len(history)
        or metadata["validation_snapshots"] != expected_count
        or trial.get("field_best_epoch") != best_epoch
        or trial.get("field_completed_epochs") != len(history)
    ):
        raise ValueError("Saved checkpoint metadata differs from its projected history")
    _equal(
        metadata["best_validation_loss"],
        best,
        "Saved checkpoint loss differs from history",
    )
    residuals = [
        row
        for row in trial.get("candidates", [])
        if row["kind"] == "warm_started_residual"
    ]
    if residuals:
        _equal(
            residuals[0]["validation_objective"],
            best,
            "Residual candidate differs from saved checkpoint objective",
        )
    if trial["selected_kind"] == "warm_started_residual":
        if _digest(folder / "model/field_model.npz") != _digest(path):
            raise ValueError("Deployed residual field differs from selected checkpoint")
        _equal(
            trial["validation_objective"],
            best,
            "Selected residual objective differs from checkpoint",
        )
    return {
        "model_seed": trial["model_seed"],
        "best_epoch": best_epoch,
        "completed_epochs": len(history),
        "best_projected_validation_objective": best,
        "min_delta": min_delta,
        "field_sha256": _digest(path),
        "raw_auxiliary_excluded": True,
    }


def _audit_run(root: Path, expected_count: int) -> tuple[dict, list[tuple]]:  # noqa: C901
    completion_path = root / "completion.json"
    files = _verify_files(root, _json(completion_path))
    required = {
        "summary.json",
        "model_selection.json",
        "features.json",
        "model/model.json",
        "model/field_model.npz",
        "model/budget_model.npz",
    }
    if not required <= files or not (root / "COMPLETED").is_file():
        raise ValueError("Training completion lacks mandatory final artifacts")
    summary, selection = (
        _json(root / "summary.json"),
        _json(root / "model_selection.json"),
    )
    if (
        summary.get("complete") is not True
        or summary.get("test_used_for_selection") is not False
        or selection.get("test_used_for_selection") is not False
    ):
        raise ValueError(
            "Training must be complete and selection must exclude test labels"
        )
    features = _json(root / "features.json")
    if features.get("test_labels_in_fit_cache") is not False:
        raise ValueError("Fit cache must exclude test labels")
    populations, identities = _validation_rows(
        features["validation_snapshots"], expected_count, report=False
    )
    if summary["sample_counts"]["validation"] != expected_count:
        raise ValueError("Training summary validation count differs from cache")
    cache_records = []
    for receipt_path in sorted(root.rglob("receipt.json")):
        if "incomplete" in receipt_path.relative_to(root).parts:
            continue
        receipt = _json(receipt_path)
        _verify_files(receipt_path.parent, receipt)
        relative = str(receipt_path.relative_to(root))
        if relative not in files:
            raise ValueError("Checkpoint receipt is absent from final completion")
        feature_path = receipt_path.parent / "features.json"
        if feature_path.is_file():
            record = _json(feature_path)
            if receipt["record"] != record:
                raise ValueError("Feature manifest differs from sealed cache record")
            _, cached = _validation_rows(
                record["validation_snapshots"], expected_count, report=False
            )
            if cached != identities:
                raise ValueError(
                    "Sealed cache validation population differs from saved features"
                )
            cache_records.append(relative)
    if not cache_records:
        raise ValueError("Training has no sealed validation feature cache")
    trials = selection.get("trials")
    if not isinstance(trials, list) or not trials:
        raise ValueError("Training selection requires completed trial scores")
    seeds = [row["model_seed"] for row in trials]
    if len(set(seeds)) != len(seeds) or any(type(seed) is not int for seed in seeds):
        raise ValueError("Trial seeds are invalid or duplicated")
    for trial in trials:
        _score(trial["validation_objective"])
    chosen = min(
        trials, key=lambda row: (row["validation_objective"], row["model_seed"])
    )
    if selection["selected"] != chosen:
        raise ValueError(
            "Training selection differs from minimum objective and seed tie-break"
        )
    if (
        summary["selected_model_seed"] != chosen["model_seed"]
        or summary["selected_kind"] != chosen["selected_kind"]
    ):
        raise ValueError("Training summary names a different selected trial")
    _equal(
        summary["selected_validation_objective"],
        chosen["validation_objective"],
        "Training summary objective differs from selected trial",
    )
    checkpoints = []
    for trial in trials:
        folder = root / "trials" / f"seed{trial['model_seed']}"
        receipt, saved = _json(folder / "receipt.json"), _json(folder / "trial.json")
        if (
            receipt["record"] != saved
            or saved != trial
            or trial.get("test_used") is not False
        ):
            raise ValueError("Trial selection differs from its sealed trial record")
        candidates = trial.get("candidates", [])
        for candidate in candidates:
            _score(candidate["validation_objective"])
        matching = [row for row in candidates if row["kind"] == trial["selected_kind"]]
        if len(matching) != 1:
            raise ValueError(
                "Selected trial kind must identify exactly one evaluated candidate"
            )
        _equal(
            matching[0]["validation_objective"],
            trial["validation_objective"],
            "Trial objective differs from its selected candidate",
        )
        if "base_selection" in trial:
            bases = [
                row for row in candidates if row["kind"] != "warm_started_residual"
            ]
            base = min(
                bases,
                key=lambda row: (
                    row["validation_objective"],
                    row["stage"] != "legacy_reference",
                    row["kind"],
                ),
            )
            if trial["base_selection"] != base:
                raise ValueError(
                    "Residual base differs from the fixed-field validation minimum"
                )
            residual = next(
                row for row in candidates if row["kind"] == "warm_started_residual"
            )
            selected = (
                residual
                if residual["validation_objective"]
                < base["validation_objective"] - 1e-12
                else base
            )
            if selected["kind"] != trial["selected_kind"]:
                raise ValueError(
                    "Trial violates the frozen-base residual improvement rule"
                )
        checkpoint = _projected_checkpoint(folder, trial, expected_count)
        if checkpoint is not None:
            checkpoints.append(checkpoint)
    deployed = _model_hashes(root / "model")
    winner = _model_hashes(root / "trials" / f"seed{chosen['model_seed']}" / "model")
    if {k: v for k, v in deployed.items() if k != "model.json"} != {
        k: v for k, v in winner.items() if k != "model.json"
    }:
        raise ValueError("Final training weights differ from the winning trial")
    metadata = _json(root / "model/model.json")["metadata"]
    if (
        metadata.get("model_seed") != chosen["model_seed"]
        or metadata.get("selected_kind") != chosen["selected_kind"]
    ):
        raise ValueError("Final model metadata differs from selected trial")
    return {
        "passed": True,
        "completion_sha256": _digest(completion_path),
        "verified_completion_files": len(files),
        "sealed_feature_caches": cache_records,
        "validation_count": expected_count,
        "validation_geometry_counts": sorted(populations.values()),
        "selected_model_seed": chosen["model_seed"],
        "selected_kind": chosen["selected_kind"],
        "selected_validation_objective": chosen["validation_objective"],
        "trial_count": len(trials),
        "trial_tie_break": "Lowest model seed for an exact objective tie",
        "model_files_sha256": deployed,
        "checkpoints": checkpoints,
    }, identities


def audit_selection(  # noqa: C901
    selection_path: str | Path,
    model_path: str | Path,
    training_runs: Mapping[str, Path],
    expected_validation_count: int = 42,
) -> dict[str, Any]:
    """Verify frozen selection and completed training without executing weights.

    Parameters
    ----------
    selection_path, model_path
        Frozen schema-one selection record and its deployed model directory.
    training_runs
        Candidate-name to completed-run mapping; every declared training run
        must be included. The unchanged previous model needs no training run.
    expected_validation_count
        Validation population, balanced across the three comparison geometries.

    Returns
    -------
    dict
        Portable hashes, selection and checkpoint evidence with ``passed=True``.
        Any mismatch raises; no teacher fields or blind archives are opened.
    """
    if (
        type(expected_validation_count) is not int
        or expected_validation_count <= 0
        or expected_validation_count % 3
    ):
        raise ValueError(
            "Expected validation count must be positive and divisible by three"
        )
    path, model = Path(selection_path).resolve(), Path(model_path).resolve()
    root, selection_hash = path.parent, _digest(path)
    frozen = _json(path)
    if (
        frozen.get("schema") != 1
        or frozen.get("blind_or_transfer_used_for_selection") is not False
    ):
        raise ValueError(
            "Frozen selection must use schema one and exclude blind/transfer test labels"
        )
    if _resolve(root, frozen["selected_model_root"]) != model:
        raise ValueError("Requested model differs from the frozen selected path")
    for stem in ("training_protocol", "validation_config", "training_inputs"):
        if (stem in frozen or f"{stem}_sha256" in frozen) and _digest(
            _resolve(root, frozen[stem])
        ) != frozen[f"{stem}_sha256"]:
            raise ValueError("Frozen training/validation declaration checksum differs")
    candidates = frozen.get("candidates")
    if not isinstance(candidates, dict) or not candidates:
        raise ValueError("Frozen selection requires named candidates")
    reports, populations, runs = {}, None, {}
    common_objective = None
    common_teacher_evidence = None
    for name, candidate in candidates.items():
        score = _score(candidate["validation_objective"])
        candidate_model = _resolve(root, candidate["model_root"])
        hashes = _model_hashes(candidate_model)
        if hashes != candidate["model_files_sha256"]:
            raise ValueError("Candidate model hash differs from frozen selection")
        report_path = _resolve(root, candidate["validation_report"])
        if _digest(report_path) != candidate["validation_report_sha256"]:
            raise ValueError("Candidate validation report hash differs from freeze")
        report = _json(report_path)
        if (
            report.get("complete") is not True
            or report.get("fitting_performed") is not False
            or report.get("test_evaluated") is not False
            or report.get("model_weights_unchanged") is not True
        ):
            raise ValueError(
                "Candidate validation report is incomplete or altered weights"
            )
        if (
            report["model_sha256"] != hashes
            or report["snapshot_count"] != expected_validation_count
        ):
            raise ValueError("Validation report model or population differs")
        counts, identities = _validation_rows(
            report["snapshots"], expected_validation_count, report=True
        )
        teacher_evidence = (
            report.get("teacher_content_digest"),
            sorted(
                (
                    row["seed"],
                    row["redshift"],
                    row["geometry"]["DIM"],
                    row["geometry"]["HII_DIM"],
                    row["geometry"]["BOX_LEN"],
                    row.get("teacher_fields_sha256"),
                )
                for row in report["snapshots"]
            ),
        )
        if (
            common_teacher_evidence is not None
            and teacher_evidence != common_teacher_evidence
        ):
            raise ValueError(
                "Candidate reports refer to different teacher validation evidence"
            )
        common_teacher_evidence = teacher_evidence
        grouped = {
            (row["geometry"]["HII_DIM"], float(row["geometry"]["BOX_LEN"])): row[
                "snapshot_count"
            ]
            for row in report["by_geometry"]
        }
        if len(report["by_geometry"]) != 3 or grouped != counts:
            raise ValueError(
                "Reported geometry counts differ from saved validation rows"
            )
        _equal(
            report["validation_objective"],
            score,
            "Frozen candidate score differs from validation report",
        )
        _equal(
            sum(_score(row["projected_loss"]) for row in report["snapshots"])
            / expected_validation_count,
            score,
            "Validation objective is not the equal-snapshot projected mean",
        )
        if populations is not None and identities != populations:
            raise ValueError(
                "Candidates were evaluated on different validation snapshots"
            )
        populations = identities
        if common_objective is not None and report["objective"] != common_objective:
            raise ValueError("Candidate validation objectives use different settings")
        common_objective = report["objective"]
        if (
            "validation_config_sha256" in frozen
            and report.get("configuration_sha256") != frozen["validation_config_sha256"]
        ):
            raise ValueError("Candidate validation configuration differs from freeze")
        if (
            "inference_source_sha256" in frozen
            and report.get("inference_source_sha256")
            != frozen["inference_source_sha256"]
        ):
            raise ValueError(
                "Candidate validation inference source differs from freeze"
            )
        reports[name] = {
            "validation_objective": score,
            "validation_count": expected_validation_count,
            "validation_geometry_counts": sorted(counts.values()),
            "validation_report_sha256": candidate["validation_report_sha256"],
            "model_files_sha256": hashes,
        }
        if "training_run" in candidate:
            if name not in training_runs or Path(
                training_runs[name]
            ).resolve() != _resolve(root, candidate["training_run"]):
                raise ValueError(
                    "Every declared training run must be explicitly audited"
                )
            runs[name], cached = _audit_run(
                Path(training_runs[name]).resolve(), expected_validation_count
            )
            if cached != identities or runs[name]["model_files_sha256"] != hashes:
                raise ValueError(
                    "Training cache or exported weights differ from evaluated candidate"
                )
            _equal(
                runs[name]["selected_validation_objective"],
                score,
                "Training and frozen candidate projected objectives differ",
            )
            if (
                "completion_sha256" in candidate
                and runs[name]["completion_sha256"] != candidate["completion_sha256"]
            ):
                raise ValueError("Training completion changed after freeze")
    if set(training_runs) != set(runs):
        raise ValueError(
            "Supplied training runs differ from the frozen candidate declarations"
        )
    chosen = min(
        candidates,
        key=lambda name: (
            _score(candidates[name]["validation_objective"]),
            name != "previous_model",
            name,
        ),
    )
    if frozen["selected_experiment"] != chosen:
        raise ValueError(
            "Final choice differs from minimum objective and previous-model tie rule"
        )
    _equal(
        frozen["selected_validation_objective"],
        candidates[chosen]["validation_objective"],
        "Final selected score differs from winning candidate",
    )
    hashes = _model_hashes(model)
    if (
        hashes != frozen["model_files_sha256"]
        or hashes != candidates[chosen]["model_files_sha256"]
    ):
        raise ValueError("Deployed model differs from the frozen winning candidate")
    if _digest(path) != selection_hash:
        raise ValueError("Selection record changed during audit")
    return {
        "schema": 1,
        "passed": True,
        "selection_sha256": selection_hash,
        "selected_experiment": chosen,
        "selected_validation_objective": candidates[chosen]["validation_objective"],
        "model_files_sha256": hashes,
        "candidates": reports,
        "training_runs": runs,
        "final_tie_break": "Previous model wins an exact tie, then candidate name",
        "evaluation_scope": "Saved validation evidence and checkpoints only; no prediction, fitting or blind labels",
    }


def main() -> None:
    """Write an artifact-only selection audit without invoking inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--training-run", action="append", default=[], metavar="NAME=PATH"
    )
    parser.add_argument("--validation-count", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runs = dict(item.split("=", 1) for item in args.training_run)
    result = audit_selection(args.selection, args.model, runs, args.validation_count)
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        sys.stdout.write(rendered)


if __name__ == "__main__":
    main()
