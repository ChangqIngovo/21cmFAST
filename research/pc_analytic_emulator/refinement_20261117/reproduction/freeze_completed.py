"""Select completed candidates on identical validation inputs and freeze weights.

This command performs no training and opens no blind archive. The previous model
is eligible and wins exact ties; a geometry claim comes only from the selected
model's recorded training support.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

from research.pc_analytic_emulator.benchmark import _inference_source_hashes
from research.pc_analytic_emulator.completion_20261116.reproduction.train_and_validate import (
    _dataset_identity,
    _receipt,
    _verify_inputs,
)
from research.pc_analytic_emulator.model import Emulator
from research.pc_analytic_emulator.predict import _model_hashes
from research.pc_analytic_emulator.refinement_20261117.reproduction import (
    evaluate_validation,
)


def digest(path: Path) -> str:
    """Hash an immutable control or result file."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _relative(path: Path, root: Path) -> str:
    return os.path.relpath(path, root)


def _verify_report(report: dict, model: Path, config: Path) -> None:
    if (
        report.get("complete") is not True
        or report.get("model_sha256") != _model_hashes(model)
        or report.get("configuration_sha256") != digest(config)
        or report.get("inference_source_sha256") != _inference_source_hashes()
        or report.get("evaluation_script_sha256")
        != digest(Path(evaluate_validation.__file__))
        or report.get("snapshot_count") != 42
        or report.get("validation_seeds") != [2113, 2114, 2261, 2262, 2271, 2272]
        or report.get("fitting_performed") is not False
        or report.get("test_evaluated") is not False
        or [item["snapshot_count"] for item in report.get("by_geometry", [])]
        != [14, 14, 14]
    ):
        raise ValueError(
            "Validation report does not match the frozen42-snapshot comparison"
        )
    value = report.get("validation_objective")
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("Validation objective must be finite and nonnegative")


def verify(path: str | Path) -> dict:
    """Verify every candidate, validation report and selected-model hash."""
    path = Path(path).absolute()
    root, selected = path.parent, _json(path)
    if (
        digest(Path(__file__)) != selected["freeze_script_sha256"]
        or digest(Path(evaluate_validation.__file__))
        != selected["validation_script_sha256"]
    ):
        raise ValueError("Selection/scoring implementation changed after freeze")
    if _inference_source_hashes() != selected["inference_source_sha256"]:
        raise ValueError("Inference implementation changed after selection freeze")
    config = root / selected["validation_config"]
    protocol = root / selected["training_protocol"]
    if (
        digest(config) != selected["validation_config_sha256"]
        or digest(protocol) != selected["training_protocol_sha256"]
    ):
        raise ValueError("Training/validation protocol changed after freeze")
    if selected.get("blind_or_transfer_used_for_selection") is not False:
        raise ValueError("Selection must exclude blind and transfer test data")
    input_path = root / selected["training_inputs"]
    if digest(input_path) != selected["training_inputs_sha256"]:
        raise ValueError("Selection input identity changed")
    _verify_inputs(_json(input_path))
    candidates = selected["candidates"]
    if set(candidates) != {"guarded_single", "guarded_mixed", "previous_model"}:
        raise ValueError("Unexpected frozen candidate set")
    for candidate in candidates.values():
        model, report_path = (
            root / candidate["model_root"],
            root / candidate["validation_report"],
        )
        if (
            _model_hashes(model) != candidate["model_files_sha256"]
            or digest(report_path) != candidate["validation_report_sha256"]
        ):
            raise ValueError("Candidate weights or validation evidence changed")
        report = _json(report_path)
        _verify_report(report, model, config)
        if report["validation_objective"] != candidate["validation_objective"]:
            raise ValueError("Candidate score differs from its validation report")
        if candidate.get("training_run"):
            run = root / candidate["training_run"]
            _receipt(run, "completion.json", "files")
            if digest(run / "completion.json") != candidate["completion_sha256"]:
                raise ValueError("Training receipt changed after freeze")
    chosen = min(
        candidates,
        key=lambda name: (
            candidates[name]["validation_objective"],
            name != "previous_model",
            name,
        ),
    )
    if (
        selected["selected_experiment"] != chosen
        or selected["selected_validation_objective"]
        != candidates[chosen]["validation_objective"]
    ):
        raise ValueError("Frozen choice differs from the declared validation minimum")
    if (
        _model_hashes(root / selected["selected_model_root"])
        != selected["model_files_sha256"]
        or selected["model_files_sha256"] != candidates[chosen]["model_files_sha256"]
    ):
        raise ValueError("Selected model differs from the winning candidate")
    return selected


def run(
    single_run: str | Path,
    mixed_run: str | Path,
    previous_model: str | Path,
    validation_config: str | Path,
    training_protocol: str | Path,
    output_root: str | Path,
    *,
    baseline_validation: str | Path | None = None,
) -> dict:
    """Freeze the exact common validation minimum from completed artifacts."""
    single_run, mixed_run, previous_model, config, protocol, root = (
        Path(value).absolute()
        for value in (
            single_run,
            mixed_run,
            previous_model,
            validation_config,
            training_protocol,
            output_root,
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "selection.json"
    if destination.exists():
        return verify(destination)
    if (root / "evaluation_started.json").exists():
        raise ValueError("Cannot select a model after blind evaluation started")
    fixed_config = _json(config)
    if (
        fixed_config.get("test_seeds") != []
        or fixed_config.get("external_test_root") is not None
    ):
        raise ValueError(
            "Selection needs an empty test split and no external test data"
        )
    source_hashes = _inference_source_hashes()
    protocol_hash, config_hash = digest(protocol), digest(config)
    inputs = _dataset_identity(Path(fixed_config["input_root"]))
    input_path = root / "selection_inputs.json"
    if input_path.exists():
        if _json(input_path) != inputs:
            raise ValueError("Selection development inputs changed")
    else:
        with input_path.open("x") as stream:
            json.dump(inputs, stream, indent=2, allow_nan=False)
            stream.write("\n")
    input_content_digest = hashlib.sha256(
        json.dumps(sorted(inputs["files_sha256"].values())).encode()
    ).hexdigest()
    candidates = {}
    runs = {"guarded_single": single_run, "guarded_mixed": mixed_run}
    models = {name: folder / "model" for name, folder in runs.items()}
    models["previous_model"] = previous_model
    for name, model in models.items():
        if name in runs:
            _receipt(runs[name], "completion.json", "files")
        report_path = root / f"freeze_validation_{name}.json"
        if report_path.exists():
            report = _json(report_path)
        elif name == "previous_model" and baseline_validation is not None:
            original_report = Path(baseline_validation).absolute()
            report = _json(original_report)
            _verify_report(report, model, config)
            shutil.copyfile(original_report, report_path)
        else:
            report = evaluate_validation.run(model, config, report_path)
        _verify_report(report, model, config)
        if report.get("teacher_content_digest") != input_content_digest:
            raise ValueError("Validation evidence refers to different teacher inputs")
        candidates[name] = {
            "model_root": _relative(model, root),
            "model_files_sha256": _model_hashes(model),
            "validation_objective": report["validation_objective"],
            "validation_report": report_path.name,
            "validation_report_sha256": digest(report_path),
        }
        if name in runs:
            candidates[name].update(
                training_run=_relative(runs[name], root),
                completion_sha256=digest(runs[name] / "completion.json"),
            )
    chosen = min(
        candidates,
        key=lambda name: (
            candidates[name]["validation_objective"],
            name != "previous_model",
            name,
        ),
    )
    chosen_root = root / "selected_model"
    if not chosen_root.exists():
        shutil.copytree(models[chosen], chosen_root)
    if _model_hashes(chosen_root) != candidates[chosen]["model_files_sha256"]:
        raise ValueError("Selected-model copy does not match its source candidate")
    model = Emulator.load(chosen_root)
    selected = {
        "schema": 1,
        "frozen_utc": datetime.now(UTC).isoformat(),
        "selected_experiment": chosen,
        "selected_validation_objective": candidates[chosen]["validation_objective"],
        "selected_model_root": "selected_model",
        "model_files_sha256": _model_hashes(chosen_root),
        "inference_source_sha256": source_hashes,
        "validation_script_sha256": digest(Path(evaluate_validation.__file__)),
        "freeze_script_sha256": digest(Path(__file__)),
        "training_protocol": _relative(protocol, root),
        "training_protocol_sha256": protocol_hash,
        "validation_config": _relative(config, root),
        "validation_config_sha256": config_hash,
        "training_inputs": input_path.name,
        "training_inputs_sha256": digest(input_path),
        "candidates": candidates,
        "selection_rule": "Minimum equal-snapshot projected validation objective across guarded_single,guarded_mixed,previous_model; previous wins exact ties",
        "validation_snapshot_count": 42,
        "validation_geometry_counts": [14, 14, 14],
        "blind_or_transfer_used_for_selection": False,
        "selection_data_scope": "Only the declared development train/validation split; previously inspected simulations may be reused there, while all newly generated blind seeds remain excluded",
        "deployment_support": model.metadata.get("training_geometries", []),
        "deployment_support_scope": "Only recorded training geometries; L100/N32 is always an unseen-geometry check",
    }
    if (
        _inference_source_hashes() != source_hashes
        or digest(protocol) != protocol_hash
        or digest(config) != config_hash
    ):
        raise ValueError("Frozen settings or inference code changed during selection")
    with destination.open("x") as stream:
        json.dump(selected, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return verify(destination)


def main() -> None:
    """Select completed weights without any fitting or blind-data access."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-run", type=Path, required=True)
    parser.add_argument("--mixed-run", type=Path, required=True)
    parser.add_argument("--previous-model", type=Path, required=True)
    parser.add_argument("--validation-config", type=Path, required=True)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--baseline-validation", type=Path)
    args = parser.parse_args()
    report = run(
        args.single_run,
        args.mixed_run,
        args.previous_model,
        args.validation_config,
        args.training_protocol,
        args.output_root,
        baseline_validation=args.baseline_validation,
    )
    sys.stdout.write(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "selected_experiment",
                    "selected_validation_objective",
                    "frozen_utc",
                )
            }
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
