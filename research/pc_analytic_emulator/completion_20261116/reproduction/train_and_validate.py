"""Train two predeclared emulators, freeze their selection, then benchmark.

Only the development collection is opened before model selection. Blind and
geometry-transfer archives are audited after the selected model, previous
model, inference code, scientific settings and selection rule are frozen.
Completed stages are audited on resume; evaluation can never trigger refitting.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

from research.pc_analytic_emulator import benchmark, train
from research.pc_analytic_emulator.data import load_dataset
from research.pc_analytic_emulator.model import Emulator
from research.pc_analytic_emulator.predict import _model_hashes

EVALUATIONS = {
    "teacher_blind": list(range(2117, 2121)),
    "teacher_transfer_resolution": [2131, 2132],
    "teacher_transfer_volume": [2141, 2142],
}

# Exact scientific settings of the two predeclared completion experiments.
# Only native archive location and execution resources are supplied at runtime.
BASE_CONFIGURATION = {
    "training_method": "warm_started_v3",
    "train_seeds": list(range(2101, 2113)) + list(range(2121, 2129)),
    "validation_seeds": list(range(2113, 2117)),
    "test_seeds": [],
    "allow_empty_test": True,
    "regression_test_seeds": list(range(2101, 2113)),
    "model_seeds": [11, 23, 47, 83],
    "epochs": 300,
    "budget_epochs": 1600,
    "warmstart_epochs": 120,
    "patience": 60,
    "hidden": [32, 32],
    "budget_hidden": [16, 8],
    "samples_per_snapshot": 2048,
    "learning_rate": 0.001,
    "warmstart_learning_rate": 0.0003,
    "include_deficit_budget": True,
    "deficit_relative_weight": 0.1,
    "deficit_neutral_floor": 0.01,
    "spectral_weight": 0.1,
    "spectral_power_floor": 1e-6,
    "field_variance_floor": 0.005,
    "front_weight": 1.0,
    "raw_weight": 0.05,
    "k_fraction": 0.5,
}


def digest(path: Path) -> str:
    """Return a file checksum without reading scientific array contents."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _freeze(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        if _json(path) != value:
            raise ValueError(f"Frozen identity changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def implementation_hashes() -> dict[str, str]:
    """Record training, inference, metric and orchestration implementations."""
    return {
        **train._implementation_hashes(),
        "benchmark.py": digest(Path(benchmark.__file__)),
        "benchmark_metrics.py": digest(Path(benchmark.recompute.__file__)),
        "train_and_validate.py": digest(Path(__file__)),
    }


def configurations(
    campaign_root: Path,
    *,
    workers: int,
    threads: int,
    cpus: int | None = None,
    frame_cache_mib: int = 512,
) -> dict[str, dict[str, Any]]:
    """Build the two frozen scientific configurations with portable input paths.

    Parameters
    ----------
    campaign_root
        Directory containing the generated development collection.
    workers, threads
        Positive process and per-process CPU allocations.
    cpus
        Optional declared CPU allocation, at least workers times threads.
    frame_cache_mib
        Per-worker frame cache limit in MiB; zero disables caching. The
        expanded multiscale collection benefits from 1024 MiB when available.

    Returns
    -------
    dict
        Fully validated settings for the legacy and multiscale-local features.
    """
    if any(type(value) is not int or value < 1 for value in (workers, threads)):
        raise ValueError("Workers and threads must be positive integers")
    cpus = workers * threads if cpus is None else cpus
    if type(cpus) is not int or cpus < workers * threads:
        raise ValueError("Declared cpus must cover all worker threads")
    if type(frame_cache_mib) is not int or frame_cache_mib < 0:
        raise ValueError("frame_cache_mib must be a nonnegative integer per worker")
    result = {}
    for name, feature_set in (
        ("legacy_features", "legacy"),
        ("local_features", "multiscale_local"),
    ):
        config = copy.deepcopy(BASE_CONFIGURATION)
        config.update(
            input_root=str(campaign_root.resolve() / "training_collection"),
            feature_set=feature_set,
            workers=workers,
            threads=threads,
            cpus=cpus,
            frame_cache_MiB=frame_cache_mib,
            external_test_root=None,
        )
        result[name] = train.validate_config(config)
    return result


def _receipt(folder: Path, filename: str, files_key: str) -> dict[str, Any]:
    if not (folder / "COMPLETED").is_file():
        raise ValueError(f"Incomplete stage: {folder}")
    record = _json(folder / filename)
    for name, expected in record[files_key].items():
        path = (folder / name).resolve()
        if (
            not path.is_relative_to(folder.resolve())
            or not path.is_file()
            or digest(path) != expected
        ):
            raise ValueError(f"Completed stage checksum changed: {path}")
    return record


def _dataset_identity(
    root: Path, *, expected_seeds: list[int] | None = None
) -> dict[str, Any]:
    dataset = load_dataset(root)
    if expected_seeds is not None and {
        sample.seed for sample in dataset.samples
    } != set(expected_seeds):
        raise ValueError(f"Evaluation seeds differ from frozen protocol: {root}")
    return {"root": str(root), "files_sha256": dataset.file_hashes}


def _verify_inputs(identity: dict[str, Any]) -> None:
    for filename, expected in identity["files_sha256"].items():
        path = Path(filename)
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Frozen scientific input changed: {path}")


def _score_previous(
    previous: Path, training_root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    details = _json(training_root / "fit_cache/features.json")
    frames = details["validation_snapshots"]
    if {frame["seed"] for frame in frames} != set(config["validation_seeds"]):
        raise ValueError(
            "Reference scoring requires exactly the frozen validation seeds"
        )
    objective = {
        name: config[name]
        for name in (
            "spectral_weight",
            "spectral_power_floor",
            "front_weight",
            "field_variance_floor",
            "k_fraction",
            "raw_weight",
        )
    }
    previous_model = Emulator.load(previous)
    training_model = Emulator.load(training_root / "model")
    physics = training_model.metadata.get("teacher_physics")
    if not isinstance(physics, dict):
        raise TypeError("Reference scoring requires recorded training physics")
    compatibility = benchmark._physics_compatibility(previous_model, physics)
    score, snapshots = train._score_emulator(previous_model, frames, objective)
    if not math.isfinite(score) or score < 0:
        raise ValueError("Previous model must have a finite validation objective")
    return {
        "validation_objective": score,
        "objective": objective,
        "validation_seeds": config["validation_seeds"],
        "physical_compatibility": compatibility,
        "snapshots": snapshots,
        "scope": "Unchanged previous model scored on the identical projected validation objective; no fitting or blind data",
    }


def _selection(output: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    selection_path = output / "selection.json"
    if selection_path.exists():
        selected = _json(selection_path)
        if selected["protocol_sha256"] != digest(output / "protocol.json"):
            raise ValueError("Frozen protocol changed after selection")
        if set(selected["candidates"]) != set(protocol["configurations"]) | {
            "previous_model"
        }:
            raise ValueError("Frozen selection candidate set changed")
        for name, candidate in selected["candidates"].items():
            if name == "previous_model":
                if (
                    digest(output / "validation_reference.json")
                    != candidate["reference_record_sha256"]
                ):
                    raise ValueError("Frozen validation reference changed")
                continue
            root = output / "training" / name
            _receipt(root, "completion.json", "files")
            if digest(root / "completion.json") != candidate["completion_sha256"]:
                raise ValueError("Training completion changed after selection")
        if _model_hashes(output / "model") != selected["model_files_sha256"]:
            raise ValueError("Selected model changed after freeze")
        return selected
    if (output / "evaluation_started.json").exists():
        raise ValueError("Frozen selection missing after evaluation started")
    candidates = {}
    for name, config in protocol["configurations"].items():
        root = output / "training" / name
        _verify_inputs(protocol["training_inputs"])
        result = train.run(config, root)
        _verify_inputs(protocol["training_inputs"])
        _receipt(root, "completion.json", "files")
        score = result.get("selected_validation_objective")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or score < 0
        ):
            raise ValueError("Training must return a finite validation-only objective")
        candidates[name] = {
            "validation_objective": score,
            "completion_sha256": digest(root / "completion.json"),
            "model_files_sha256": _model_hashes(root / "model"),
        }
    previous = Path(protocol["previous_model"]["root"])
    reference = _score_previous(
        previous,
        output / "training/legacy_features",
        protocol["configurations"]["legacy_features"],
    )
    reference.update(
        model_root=str(previous),
        model_files_sha256=protocol["previous_model"]["files_sha256"],
    )
    _freeze(output / "validation_reference.json", reference)
    candidates["previous_model"] = {
        "validation_objective": reference["validation_objective"],
        "reference_record_sha256": digest(output / "validation_reference.json"),
        "model_files_sha256": protocol["previous_model"]["files_sha256"],
    }
    chosen = min(
        candidates,
        key=lambda name: (
            candidates[name]["validation_objective"],
            name != "previous_model",
            name,
        ),
    )
    source = (
        previous
        if chosen == "previous_model"
        else output / "training" / chosen / "model"
    )
    if (output / "model").exists():
        if _model_hashes(output / "model") != candidates[chosen]["model_files_sha256"]:
            raise ValueError("Incomplete or changed selected-model copy")
    else:
        shutil.copytree(source, output / "model")
    if _model_hashes(output / "model") != candidates[chosen]["model_files_sha256"]:
        raise ValueError("Selected-model copy failed checksum verification")
    selected = {
        "schema": 1,
        "selected_experiment": chosen,
        "selected_validation_objective": candidates[chosen]["validation_objective"],
        "selection_rule": "Minimum identical validation objective; unchanged previous model wins exact ties, then experiment-name tie break",
        "protocol_sha256": digest(output / "protocol.json"),
        "candidates": candidates,
        "validation_reference": {
            "validation_objective": reference["validation_objective"],
            "record_sha256": digest(output / "validation_reference.json"),
            "scope": "Unchanged previous model; identical validation objective and inputs",
        },
        "model_files_sha256": _model_hashes(output / "model"),
        "blind_or_transfer_data_used": False,
    }
    _freeze(selection_path, selected)
    return selected


def _verify_benchmark_identity(
    folder: Path,
    specification: dict[str, Any],
    protocol: dict[str, Any],
    output: Path,
    selected: dict[str, Any],
    inputs: dict[str, Any],
) -> None:
    expected = {
        **protocol["benchmark_implementation"],
        "teacher_root": specification["root"],
        "requested_seeds": specification["seeds"],
        "rays_per_phase": protocol["rays_per_phase"],
        "models": {
            "selected": {
                "root": str(output / "model"),
                "files_sha256": selected["model_files_sha256"],
            },
            "previous": protocol["previous_model"],
        },
        "fitting_performed": False,
        "selection_performed": False,
    }
    if (
        not (folder / "manifest.json").is_file()
        or not (folder / "inputs.json").is_file()
    ):
        raise ValueError("Benchmark identity records are missing")
    manifest = _json(folder / "manifest.json")
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Benchmark identity differs from frozen evaluation protocol")
    if _json(folder / "inputs.json").get("files_sha256") != inputs["files_sha256"]:
        raise ValueError("Benchmark identity differs from frozen scientific inputs")


def _run_locked(
    campaign: Path,
    previous: Path,
    output: Path,
    *,
    workers: int,
    threads: int,
    cpus: int | None,
    frame_cache_mib: int,
    rays: int,
) -> dict[str, Any]:
    configs = configurations(
        campaign,
        workers=workers,
        threads=threads,
        cpus=cpus,
        frame_cache_mib=frame_cache_mib,
    )
    previous_hashes = _model_hashes(previous)
    protocol = {
        "schema": 1,
        "configurations": configs,
        "training_inputs": _dataset_identity(campaign / "training_collection"),
        "previous_model": {"root": str(previous), "files_sha256": previous_hashes},
        "implementation_sha256": implementation_hashes(),
        "benchmark_implementation": {
            "benchmark_sha256": digest(Path(benchmark.__file__)),
            "independent_metrics_sha256": digest(Path(benchmark.recompute.__file__)),
            "inference_source_sha256": benchmark._inference_source_hashes(),
        },
        "evaluations": {
            name: {"root": str(campaign / name), "seeds": seeds}
            for name, seeds in EVALUATIONS.items()
        },
        "rays_per_phase": rays,
        "scope": "Two predeclared validation-only fits; model selection and hashes frozen before blind or transfer archives are opened",
    }
    _freeze(output / "protocol.json", protocol)
    for name, config in configs.items():
        _freeze(output / "configs" / f"{name}.json", config)
    selected = _selection(output, protocol)
    _verify_inputs(protocol["training_inputs"])
    if implementation_hashes() != protocol["implementation_sha256"]:
        raise ValueError("Implementation changed during training")
    if _model_hashes(previous) != previous_hashes:
        raise ValueError("Previous model changed during training")
    # This record is immutable. After its creation, no training/selection path
    # is entered again, including after interrupted benchmarks.
    evaluation_identity = {
        "protocol_sha256": digest(output / "protocol.json"),
        "selection_sha256": digest(output / "selection.json"),
        "model_files_sha256": selected["model_files_sha256"],
        "previous_model_files_sha256": previous_hashes,
    }
    _freeze(output / "evaluation_started.json", evaluation_identity)
    evaluation_inputs = {
        name: _dataset_identity(Path(spec["root"]), expected_seeds=spec["seeds"])
        for name, spec in protocol["evaluations"].items()
    }
    _freeze(output / "evaluation_inputs.json", evaluation_inputs)
    reports = {}
    (output / "benchmarks").mkdir(exist_ok=True)
    for name, spec in protocol["evaluations"].items():
        _verify_inputs(evaluation_inputs[name])
        _freeze(output / "evaluation_started.json", evaluation_identity)
        destination = output / "benchmarks" / name
        if destination.exists() and not (destination / "COMPLETED").exists():
            # Preserve failed attempts; restarting uses exactly the frozen
            # model and audited inputs, never a new selected checkpoint.
            destination.rename(
                destination.with_name(f"{name}.partial.{uuid.uuid4().hex}")
            )
        if not destination.exists():
            benchmark.run(
                spec["root"],
                output / "model",
                previous,
                destination,
                seeds=spec["seeds"],
                rays=rays,
            )
        _receipt(destination, "receipt.json", "files_sha256")
        _verify_benchmark_identity(
            destination, spec, protocol, output, selected, evaluation_inputs[name]
        )
        _verify_inputs(evaluation_inputs[name])
        reports[name] = {
            "report": str(destination / "report.json"),
            "receipt_sha256": digest(destination / "receipt.json"),
            "snapshot_count": _json(destination / "report.json")["snapshot_count"],
        }
    if (
        _model_hashes(output / "model") != selected["model_files_sha256"]
        or _model_hashes(previous) != previous_hashes
        or implementation_hashes() != protocol["implementation_sha256"]
    ):
        raise ValueError("Model or implementation changed during evaluation")
    _verify_inputs(protocol["training_inputs"])
    for identity in evaluation_inputs.values():
        _verify_inputs(identity)
    if (
        digest(output / "protocol.json") != evaluation_identity["protocol_sha256"]
        or digest(output / "selection.json") != evaluation_identity["selection_sha256"]
        or digest(output / "validation_reference.json")
        != selected["validation_reference"]["record_sha256"]
    ):
        raise ValueError(
            "Frozen protocol or validation selection changed during evaluation"
        )
    _freeze(output / "evaluation_started.json", evaluation_identity)
    for name, config in configs.items():
        _freeze(output / "configs" / f"{name}.json", config)
    summary = {
        "complete": True,
        "selected_experiment": selected["selected_experiment"],
        "selected_validation_objective": selected["selected_validation_objective"],
        "selection_sha256": digest(output / "selection.json"),
        "protocol_sha256": digest(output / "protocol.json"),
        "benchmarks": reports,
        "blind_and_transfer_used_for_selection": False,
    }
    _freeze(output / "summary.json", summary)
    control_files = [
        "protocol.json",
        "selection.json",
        "evaluation_started.json",
        "evaluation_inputs.json",
        "validation_reference.json",
        "summary.json",
        *[f"configs/{name}.json" for name in configs],
    ]
    _freeze(
        output / "completion.json",
        {"files": {name: digest(output / name) for name in control_files}},
    )
    if not (output / "COMPLETED").exists():
        (output / "COMPLETED").write_text(
            "Paired training and frozen independent benchmarks complete\n"
        )
    _receipt(output, "completion.json", "files")
    return summary


def run(
    campaign_root: str | Path,
    previous_model: str | Path,
    output: str | Path,
    *,
    workers: int = 2,
    threads: int = 2,
    cpus: int | None = None,
    frame_cache_mib: int = 512,
    rays: int = 4096,
) -> dict[str, Any]:
    """Run or audit the portable training and independent-evaluation protocol.

    Parameters
    ----------
    campaign_root
        Generated root with training_collection and the three teacher campaigns.
    previous_model
        Self-contained previous checkpoint, used only as benchmark reference.
    output
        New destination or this same unchanged protocol's resumable directory.
    workers, threads, cpus, rays
        Training resources, optional declared CPU allocation and geometric-MFP
        rays per phase. Omitted cpus defaults to workers times threads.
    frame_cache_mib
        Nonnegative per-worker frame-cache limit; 512 preserves the original
        protocol, 1024 avoids repeated expanded-collection decompression, and
        zero disables caching. It does not change the scientific objective.

    Returns
    -------
    dict
        Frozen selected experiment and audited independent benchmark locations.
    """
    campaign, previous, destination = (
        Path(path).resolve() for path in (campaign_root, previous_model, output)
    )
    configurations(
        campaign,
        workers=workers,
        threads=threads,
        cpus=cpus,
        frame_cache_mib=frame_cache_mib,
    )
    if type(rays) is not int or rays < 2:
        raise ValueError("MFP rays must be an integer >=2")
    inputs = [
        previous,
        campaign / "training_collection",
        *[campaign / name for name in EVALUATIONS],
    ]
    if any(
        destination.is_relative_to(path) or path.is_relative_to(destination)
        for path in inputs
    ):
        raise ValueError("Output must be separate from input archives and models")
    if destination.exists() and not (destination / "protocol.json").is_file():
        raise ValueError("Existing output does not contain this frozen protocol")
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "workflow.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_locked(
            campaign,
            previous,
            destination,
            workers=workers,
            threads=threads,
            cpus=cpus,
            frame_cache_mib=frame_cache_mib,
            rays=rays,
        )


def main() -> None:
    """Run the predeclared completion protocol from portable campaign paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--previous-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--cpus", type=int)
    parser.add_argument(
        "--frame-cache-mib",
        type=int,
        default=512,
        help="Per-worker cache cap: default 512 MiB; recommended 1024 for expanded collection; 0 disables",
    )
    parser.add_argument("--rays", type=int, default=4096)
    args = parser.parse_args()
    result = run(
        args.campaign_root,
        args.previous_model,
        args.output,
        workers=args.workers,
        threads=args.threads,
        cpus=args.cpus,
        frame_cache_mib=args.frame_cache_mib,
        rays=args.rays,
    )
    sys.stdout.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
