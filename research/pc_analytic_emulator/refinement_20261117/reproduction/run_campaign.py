"""Prepare, generate, train and freeze the preregistered geometry campaign.

Blind evaluation is a separate command requiring an explicit frozen selection
record. Generation and training never evaluate new blind prediction accuracy.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

from research.pc_analytic_emulator import benchmark, train
from research.pc_analytic_emulator.completion_20261116.reproduction.train_and_validate import (
    _dataset_identity,
    _receipt,
    _verify_inputs,
)
from research.pc_analytic_emulator.predict import _model_hashes
from research.pc_analytic_emulator.refinement_20261117.reproduction import (
    evaluate_completed,
    evaluate_validation,
    freeze_completed,
)
from research.pc_analytic_emulator.refinement_20261117.reproduction.campaign_inputs import (
    audit_inputs,
    build_collections,
    generate_inputs,
    prepare_archived_inputs,
    prepare_inputs,
)

MODULE = "research.pc_analytic_emulator.refinement_20261117.reproduction.run_campaign"
HERE = Path(__file__).resolve().parent
BLIND = {
    "blind_original": [2201, 2202],
    "blind_resolution": [2211, 2212],
    "blind_volume": [2221, 2222],
    "blind_unseen_box": [2231, 2232],
}
NAMES = ("guarded_single", "guarded_mixed")


def digest(path: Path) -> str:
    """Hash exact bytes without interpreting scientific results."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict:
    """Read an explicit control record."""
    return json.loads(path.read_text())


def freeze_json(path: Path, value: dict) -> None:
    """Create an immutable JSON record or verify identical resumed inputs."""
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Frozen record changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def implementation_hashes() -> dict:
    """Bind training, inference, scoring and this orchestration to one source."""
    return {
        **train._implementation_hashes(),
        "refinement_orchestration": digest(Path(__file__)),
        "refinement_validation": digest(Path(evaluate_validation.__file__)),
        "benchmark": digest(Path(benchmark.__file__)),
        "independent_benchmark_metrics": digest(Path(benchmark.recompute.__file__)),
        "inference": benchmark._inference_source_hashes(),
    }


def training_runtime() -> dict:
    """Record the actual training interpreter and numeric dependency versions."""
    return {
        "python_executable": str(Path(sys.executable).absolute()),
        "python_binary_sha256": digest(Path(sys.executable)),
        "python_version": list(sys.version_info[:3]),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "matplotlib")
        },
    }


def configurations(root: Path) -> dict[str, dict]:
    """Instantiate the exact two public templates with local collection paths."""
    result = {}
    for name in NAMES:
        folder = root / f"collection_{name}"
        folder.mkdir(parents=True, exist_ok=True)
        config = read_json(HERE / f"{name}_template.json")
        config["input_root"] = str(folder.absolute())
        result[name] = train.validate_config(config)
    if any(
        config["test_seeds"] or config["external_test_root"]
        for config in result.values()
    ):
        raise ValueError("Development training must not open a blind test split")
    return result


def previous_model(previous_root: Path) -> Path:
    """Prefer the selected checkpoint, with documented older archive layouts."""
    for relative in ("selected_model", "model", "train_local_features/model"):
        candidate = previous_root / relative
        if (candidate / "model.json").is_file():
            return candidate
    raise FileNotFoundError(
        "Previous campaign needs selected_model/model.json (or model/model.json)"
    )


def configure(root: Path) -> dict:
    """Freeze settings, code and the previous checkpoint before any fitting."""
    generation = read_json(root / "generation_protocol.json")
    audit_inputs(generation)
    if generation.get("inputs_mode") == "archived":
        build_collections(generation)
    configs = configurations(root)
    previous = previous_model(Path(generation["previous_campaign_root"]))
    record = {
        "schema": 1,
        "generation_protocol_sha256": digest(root / "generation_protocol.json"),
        "source_path": generation["source_path"],
        "training_python": str(Path(sys.executable).absolute()),
        "training_runtime": training_runtime(),
        "implementation_sha256": implementation_hashes(),
        "configurations": configs,
        "previous_model": {
            "root": str(previous),
            "files_sha256": _model_hashes(previous),
        },
        "selection_rule": "Minimum equal-snapshot projected validation objective across both candidates and unchanged previous model; previous wins exact ties",
        "validation_weighting": "42 snapshots,14 at each of three geometries; equal snapshots give equal geometry weight",
        "training_counts": {"guarded_single": 178, "guarded_mixed": 250},
        "validation_seeds": [2113, 2114, 2261, 2262, 2271, 2272],
        "evaluations": {
            name: {
                "root": generation["campaign_configs"][name]["output"],
                "seeds": seeds,
            }
            for name, seeds in BLIND.items()
        },
        "rays_per_phase": 4096,
        "blind_policy": "No scientific evaluation before selection.json exists and evaluate explicitly names that freeze record",
    }
    freeze_json(root / "training_protocol.json", record)
    for name, config in configs.items():
        freeze_json(root / f"{name}.json", config)
    return record


def verify_protocol(root: Path) -> dict:
    """Reject implementation, configuration or previous-checkpoint drift."""
    protocol = read_json(root / "training_protocol.json")
    if (
        digest(root / "generation_protocol.json")
        != protocol["generation_protocol_sha256"]
    ):
        raise ValueError("Generation protocol changed after training preparation")
    audit_inputs(read_json(root / "generation_protocol.json"))
    if implementation_hashes() != protocol["implementation_sha256"]:
        raise ValueError("Frozen training/scoring implementation changed")
    if training_runtime() != protocol["training_runtime"]:
        raise ValueError("Frozen training runtime changed")
    if (
        _model_hashes(protocol["previous_model"]["root"])
        != protocol["previous_model"]["files_sha256"]
    ):
        raise ValueError("Previous checkpoint changed")
    for name, config in protocol["configurations"].items():
        if read_json(root / f"{name}.json") != config:
            raise ValueError("Frozen training configuration changed")
    return protocol


def train_candidates(root: Path) -> dict:
    """Fit only the two predeclared development candidates; support audited resume."""
    protocol = verify_protocol(root)
    if (root / "selection.json").exists():
        return verify_selection(root, protocol)
    if (root / "evaluation_started.json").exists():
        raise ValueError("Evaluation already started; fitting is disabled")
    inputs = {
        name: _dataset_identity(Path(config["input_root"]))
        for name, config in protocol["configurations"].items()
    }
    freeze_json(root / "training_inputs.json", inputs)
    summaries = {}
    for name, config in protocol["configurations"].items():
        _verify_inputs(inputs[name])
        summaries[name] = train.run(config, root / f"train_{name}")
        _receipt(root / f"train_{name}", "completion.json", "files")
        _verify_inputs(inputs[name])
        verify_protocol(root)
        counts = summaries[name]["sample_counts"]
        if counts != {
            "train": protocol["training_counts"][name],
            "validation": 42,
            "test": 0,
        }:
            raise ValueError(
                "Completed training split differs from the frozen protocol"
            )
    return {"complete": True, "candidates": list(summaries), "blind_evaluated": False}


def verify_selection(root: Path, protocol: dict) -> dict:
    """Verify selection through the same standalone freeze-record auditor."""
    del protocol
    return freeze_completed.verify(root / "selection.json")


def freeze_selection(root: Path) -> dict:
    """Use the shared completed-model freezer on identical validation inputs."""
    protocol = verify_protocol(root)
    for identity in read_json(root / "training_inputs.json").values():
        _verify_inputs(identity)
    baseline = root / "baseline_validation.json"
    return freeze_completed.run(
        root / "train_guarded_single",
        root / "train_guarded_mixed",
        protocol["previous_model"]["root"],
        root / "guarded_mixed.json",
        root / "training_protocol.json",
        root,
        baseline_validation=baseline if baseline.is_file() else None,
    )


def evaluate_frozen(root: Path, freeze_record: Path) -> dict:
    """Delegate the explicit post-freeze stage to the standalone evaluator."""
    if (
        freeze_record.absolute() != (root / "selection.json").absolute()
        or not freeze_record.is_file()
    ):
        raise ValueError(
            "Evaluation requires this campaign's existing --freeze-record selection.json"
        )
    protocol = verify_protocol(root)
    verify_selection(root, protocol)
    roots = {Path(spec["root"]).parent for spec in protocol["evaluations"].values()}
    if len(roots) != 1:
        raise ValueError("The four blind profiles must share one archived teacher root")
    return evaluate_completed.run(root, freeze_record, teacher_root=roots.pop())


def execute_frozen(root: Path, stage: str, freeze_record: Path | None = None) -> bool:
    """Re-enter committed source without inheriting a mutable PYTHONPATH."""
    protocol = read_json(root / "generation_protocol.json")
    source = Path(protocol["source_path"]).absolute()
    if HERE.parents[3] == source:
        return False
    environment = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME"):
        environment.pop(key, None)
    environment.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    command = [sys.executable, "-m", MODULE, stage, "--output-root", str(root)]
    if freeze_record is not None:
        command.extend(("--freeze-record", str(freeze_record.absolute())))
    subprocess.run(command, cwd=source, env=environment, check=True)
    return True


def main() -> None:
    """Run one explicit phase; never combine fitting with blind evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "prepare",
            "prepare-training",
            "configure",
            "generate",
            "collections",
            "train",
            "freeze",
            "evaluate",
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--previous-campaign-root", type=Path)
    parser.add_argument("--native-campaign-root", type=Path)
    parser.add_argument("--generated-campaign-root", type=Path)
    parser.add_argument("--native-python", type=Path)
    parser.add_argument("--native-template", type=Path)
    parser.add_argument("--source", type=Path, default=HERE.parents[3])
    parser.add_argument("--commit", default="HEAD")
    parser.add_argument("--freeze-record", type=Path)
    args = parser.parse_args()
    root = args.output_root.expanduser().resolve()
    if args.stage == "prepare-training":
        if any(
            value is None
            for value in (
                args.previous_campaign_root,
                args.native_campaign_root,
                args.generated_campaign_root,
            )
        ):
            parser.error(
                "prepare-training requires previous/native/generated campaign roots"
            )
        prepare_archived_inputs(
            args.source,
            args.previous_campaign_root,
            args.native_campaign_root,
            args.generated_campaign_root,
            root,
            commit=args.commit,
        )
        args.stage = "configure"
    elif args.stage == "prepare":
        if any(
            value is None
            for value in (
                args.previous_campaign_root,
                args.native_campaign_root,
                args.native_python,
                args.native_template,
            )
        ):
            parser.error(
                "prepare requires previous/native campaign roots, native Python and native template"
            )
        prepare_inputs(
            args.source,
            args.previous_campaign_root,
            args.native_campaign_root,
            root,
            args.native_python,
            args.native_template,
            commit=args.commit,
        )
        args.stage = "configure"
    if execute_frozen(root, args.stage, args.freeze_record):
        return
    with (root / ".orchestration.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == "configure":
            result = configure(root)
        elif args.stage == "generate":
            result = generate_inputs(read_json(root / "generation_protocol.json"))
            build_collections(read_json(root / "generation_protocol.json"))
        elif args.stage == "collections":
            result = {
                name: str(path)
                for name, path in build_collections(
                    read_json(root / "generation_protocol.json")
                ).items()
            }
        elif args.stage == "train":
            result = train_candidates(root)
        elif args.stage == "freeze":
            result = freeze_selection(root)
        elif args.stage == "evaluate":
            if args.freeze_record is None:
                parser.error("evaluate requires --freeze-record")
            result = evaluate_frozen(root, args.freeze_record)
        else:
            raise ValueError("Unsupported campaign stage")
    sys.stdout.write(
        json.dumps(
            {
                "stage": args.stage,
                "complete": True,
                "output_root": str(root),
                "result_keys": sorted(result)
                if isinstance(result, dict)
                else [item["campaign"] for item in result],
            }
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
