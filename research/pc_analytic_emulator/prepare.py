"""Freeze completed analytic teachers and a seed-separated training job."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)
FROZEN_FILES = (
    "commit.txt",
    "teacher_run.txt",
    "python_path.txt",
    "runtime.env",
    "submit.pbs",
    "experiment_config.json",
    "run_manifest.json",
)


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _source_hashes(path: Path) -> dict[str, str]:
    return {
        str(item.relative_to(path)): _digest(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def _integers(text: str, name: str, minimum: int = 1) -> list[int]:
    try:
        values = [int(value) for value in text.replace(",", " ").split()]
    except ValueError as error:
        raise ValueError(f"{name} must contain integers") from error
    if not values or len(values) != len(set(values)) or min(values) < minimum:
        raise ValueError(f"{name} requires unique integers >= {minimum}")
    return values


def latest_teacher(root: Path) -> Path:
    """Locate the newest completed default 56-snapshot teacher campaign.

    Parameters
    ----------
    root
        Directory containing prepared ``analytic_teacher_*`` runs.

    Returns
    -------
    Path
        Selected run directory. Selection uses the recorded UTC timestamp in
        the directory name, not the commit prefix or directory modification
        time. A malformed completed candidate raises instead of silently
        selecting older data. Full file/ledger audits occur during preparation.
    """
    candidates = []
    for path in root.resolve().glob("analytic_teacher_*"):
        matched = re.fullmatch(
            r"analytic_teacher_[0-9a-f]{7,40}_(\d{8}T\d{6})_[A-Za-z0-9]+",
            path.name,
        )
        if (
            matched
            and (path / "COMPLETED").is_file()
            and (path / "results/COMPLETED").is_file()
        ):
            candidates.append((matched[1], path.name, path))
    for _, _, path in sorted(candidates, reverse=True):
        try:
            summary = json.loads((path / "results/summary.json").read_text())
            manifest = json.loads((path / "results/dataset_manifest.json").read_text())
            config = manifest["identity"]["configuration"]
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError(f"Cannot inspect completed teacher: {path}") from error
        if (
            summary.get("complete") is True
            and summary.get("all_photon_ledgers_passed") is True
            and summary.get("completed_snapshots") == 56
            and summary.get("requested_snapshots") == 56
            and manifest.get("schema") == 1
            and manifest.get("complete") is True
            and manifest.get("teacher_method") == "analytic-column-characteristic"
            and sorted(config.get("seeds", [])) == list(range(1001, 1009))
            and sorted(config.get("redshifts", [])) == [6, 6.5, 7, 7.5, 8, 9, 10]
            and config.get("dimension") == 32
            and config.get("box_len") == 256
        ):
            return path.resolve()
        LOGGER.info(
            "Skipping completed campaign outside the default teacher set: %s", path
        )
    raise FileNotFoundError(
        f"No completed matching 56-snapshot teacher campaign under {root}; use --teacher-run for an explicit dataset"
    )


def configuration(input_root: Path, environment: Mapping[str, str]) -> dict[str, Any]:
    """Translate explicit seed splits and bounded training resources.

    Parameters
    ----------
    input_root
        Audited teacher results directory.
    environment
        PC-prefixed controls. PC_THREADS limits BLAS threads per independent
        trial worker, and PC_NCPUS covers their product.

    Returns
    -------
    dict
        Training controls accepted by the module's training entry point.
    """
    workers = int(environment.get("PC_WORKERS", "4"))
    threads = int(environment.get("PC_THREADS", "4"))
    config = {
        "input_root": str(input_root.resolve()),
        "train_seeds": _integers(
            environment.get("PC_TRAIN_SEEDS", "1001,1002,1003,1004,1005,1006"),
            "PC_TRAIN_SEEDS",
        ),
        "validation_seeds": _integers(
            environment.get("PC_VALIDATION_SEEDS", "1007"), "PC_VALIDATION_SEEDS"
        ),
        "test_seeds": _integers(
            environment.get("PC_TEST_SEEDS", "1008"), "PC_TEST_SEEDS"
        ),
        "model_seeds": _integers(
            environment.get("PC_MODEL_SEEDS", "11,23,47,83"), "PC_MODEL_SEEDS", 0
        ),
        "workers": workers,
        "threads": threads,
        "cpus": int(environment.get("PC_NCPUS", str(workers * threads))),
        "epochs": int(environment.get("PC_EMULATOR_EPOCHS", "200")),
        "samples_per_snapshot": int(environment.get("PC_SAMPLES_PER_SNAPSHOT", "8192")),
    }
    if (
        min(
            config[name]
            for name in ("workers", "threads", "cpus", "epochs", "samples_per_snapshot")
        )
        < 1
    ):
        raise ValueError(
            "Worker, thread, CPU, epoch and sample counts must be positive"
        )
    if config["workers"] * config["threads"] > config["cpus"]:
        raise ValueError("PC_NCPUS must cover PC_WORKERS * PC_THREADS")
    method = environment.get("PC_TRAINING_METHOD", "legacy")
    if method not in ("legacy", "projected_multiscale_v2"):
        raise ValueError("Unknown PC_TRAINING_METHOD")
    if method == "projected_multiscale_v2":
        config.update(
            training_method=method,
            v2_epochs=int(environment.get("PC_V2_EPOCHS", "80")),
            spectral_weight=float(environment.get("PC_SPECTRAL_WEIGHT", "0.1")),
            spectral_power_floor=float(
                environment.get("PC_SPECTRAL_POWER_FLOOR", "0.0001")
            ),
            front_weight=float(environment.get("PC_FRONT_WEIGHT", "1")),
            raw_weight=float(environment.get("PC_RAW_WEIGHT", "0.05")),
            field_variance_floor=float(
                environment.get("PC_FIELD_VARIANCE_FLOOR", "0.01")
            ),
            k_fraction=float(environment.get("PC_K_FRACTION", "0.5")),
            regression_test_seeds=[1008],
        )
    if environment.get("PC_EXTERNAL_TEST_ROOT"):
        config["external_test_root"] = str(
            Path(environment["PC_EXTERNAL_TEST_ROOT"]).resolve()
        )
    return config


def _runtime() -> dict[str, Any]:
    import matplotlib
    import numpy
    import scipy

    binary = Path(sys.executable).resolve()
    return {
        "python": str(Path(sys.executable).absolute()),
        "python_binary": str(binary),
        "python_binary_sha256": _digest(binary),
        "versions": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }


def _load_dataset(path: Path) -> Any:
    from research.pc_analytic_emulator.data import load_dataset

    return load_dataset(path)


def _split_dataset(dataset: Any, config: dict[str, Any]) -> dict[str, Any]:
    from research.pc_analytic_emulator.data import split_samples

    return split_samples(
        dataset,
        train_seeds=config["train_seeds"],
        validation_seeds=config["validation_seeds"],
        test_seeds=config["test_seeds"],
    )


def _validated_config(config: dict[str, Any]) -> dict[str, Any]:
    from research.pc_analytic_emulator.train import validate_config

    return validate_config(config)


def validate_external_test(dataset: Any, external: Any) -> None:
    """Check that an external archive supplies new realizations of the same target.

    Parameters
    ----------
    dataset, external
        Independently audited teacher datasets. Neither is modified or combined.
        The external archive is used only after validation selects the final model.

    Raises
    ------
    ValueError
        Physical settings or teacher methods differ, or an initial realization
        has appeared anywhere in the original campaign.
    """
    if dataset.physics != external.physics:
        raise ValueError("External test physical settings differ from training")
    old_seeds = {sample.seed for sample in dataset.samples}
    new_seeds = {sample.seed for sample in external.samples}
    if old_seeds & new_seeds:
        raise ValueError("External test seeds overlap the original teacher campaign")
    old_initial = set(dataset.audit["initial_conditions_sha256"].values())
    new_initial = set(external.audit["initial_conditions_sha256"].values())
    if old_initial & new_initial:
        raise ValueError(
            "External test initial conditions repeat the original campaign"
        )
    for name in ("analytic", "sigma_cm2", "brightness"):
        if dataset.identity["configuration"].get(name) != external.identity[
            "configuration"
        ].get(name):
            raise ValueError(f"External test teacher settings differ: {name}")
    if dataset.identity.get("native_runtime") != external.identity.get(
        "native_runtime"
    ):
        raise ValueError("External test native runtime differs from training")
    for name in ("implementation_sha256", "analytic_build"):
        if dataset.identity.get(name) != external.identity.get(name):
            raise ValueError(f"External test teacher implementation differs: {name}")


def prepare(run: Path, environment: Mapping[str, str]) -> dict[str, Any]:
    """Freeze the selected teacher bytes, input splits and executable environment.

    Parameters
    ----------
    run
        Fresh training directory with committed source and shell metadata.
    environment
        Training/resource overrides, interpreted by :func:`configuration`.

    Returns
    -------
    dict
        Immutable training manifest. Preparation audits existing labels and
        computes no new simulations or fitted model.
    """
    run = run.resolve()
    if (run / "PREPARED").exists() or (run / "run_manifest.json").exists():
        raise FileExistsError(
            "Use a fresh training run; do not rewrite its frozen preparation"
        )
    teacher = Path((run / "teacher_run.txt").read_text().strip()).resolve()
    if run == teacher or run.is_relative_to(teacher):
        raise ValueError("Training output must be separate from the teacher archive")
    dataset = _load_dataset(teacher)
    config = _validated_config(configuration(dataset.root, environment))
    splits = _split_dataset(dataset, config)
    external = None
    if config.get("external_test_root"):
        external = _load_dataset(Path(config["external_test_root"]))
        if run == external.root or run.is_relative_to(external.root):
            raise ValueError(
                "Training output must be outside the external test archive"
            )
        validate_external_test(dataset, external)
        config["external_test_root"] = str(external.root)
    runtime = _runtime()
    _write_json(run / "experiment_config.json", config)
    manifest = {
        "schema": 1,
        "experiment": "analytic-teacher-emulator",
        "configuration": config,
        "teacher_run": str(teacher),
        "input_root": str(dataset.root),
        "input_sha256": dataset.file_hashes,
        "teacher_identity": dataset.identity,
        "teacher_physics": dataset.physics,
        "teacher_audit": dataset.audit,
        "sample_count": len(dataset.samples),
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "split_samples": {
            name: [
                {"seed": sample.seed, "redshift": sample.redshift} for sample in rows
            ]
            for name, rows in splits.items()
        },
        "runtime": runtime,
        "source_sha256": _source_hashes(run / "source"),
        "scope": "Train on completed analytic coeval photon-conserving teachers; physical seeds are disjoint across train, validation and test; no simulation or reference reruns",
        "network_required_on_compute": False,
    }
    if config.get("training_method") == "projected_multiscale_v2":
        manifest["test_interpretation"] = {
            "previously_inspected_regression_seeds": [1008],
            "scope": "Seed 1008 is an inspected regression test, not a fresh held-out realization for this improvement",
        }
    if external is not None:
        manifest["external_test"] = {
            "input_root": str(external.root),
            "input_sha256": external.file_hashes,
            "teacher_identity": external.identity,
            "teacher_physics": external.physics,
            "teacher_audit": external.audit,
            "sample_count": len(external.samples),
            "seeds": sorted({sample.seed for sample in external.samples}),
            "scope": "Separate new realizations; excluded from fitting, normalization and model selection",
        }
    _write_json(run / "run_manifest.json", manifest)
    (run / "prepared.sha256").write_text(
        "".join(f"{_digest(run / name)}  {name}\n" for name in FROZEN_FILES)
    )
    (run / "PREPARED").write_text("Prepared immutable analytic teacher training job.\n")
    return manifest


def verify(run: Path, cpus: int | None = None) -> dict[str, Any]:
    """Verify frozen code, labels, runtime and allocated CPU count.

    Parameters
    ----------
    run
        Prepared training directory.
    cpus
        Actual PBS CPU allocation, when running under the scheduler.

    Returns
    -------
    dict
        Unchanged manifest ready for training or resubmission.
    """
    run = run.resolve()
    expected = [f"{_digest(run / name)}  {name}" for name in FROZEN_FILES]
    if (run / "prepared.sha256").read_text().splitlines() != expected:
        raise ValueError("Prepared metadata changed; prepare a new training run")
    manifest = json.loads((run / "run_manifest.json").read_text())
    if (
        manifest.get("schema") != 1
        or manifest.get("experiment") != "analytic-teacher-emulator"
    ):
        raise ValueError("Incorrect training preparation manifest")
    if _source_hashes(run / "source") != manifest["source_sha256"]:
        raise ValueError("Frozen source changed; prepare a new training run")
    for name, digest in manifest["input_sha256"].items():
        if not Path(name).is_file() or _digest(Path(name)) != digest:
            raise ValueError(f"Teacher input changed: {name}")
    for name, digest in (
        manifest.get("external_test", {}).get("input_sha256", {}).items()
    ):
        if not Path(name).is_file() or _digest(Path(name)) != digest:
            raise ValueError(f"External test input changed: {name}")
    if _runtime() != manifest["runtime"]:
        raise ValueError("Training runtime changed")
    if cpus is not None and cpus != manifest["configuration"]["cpus"]:
        raise ValueError(
            "CPU allocation differs from the frozen training configuration"
        )
    return manifest


def main() -> None:
    """Locate completed teachers, or prepare/verify an immutable training job."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    locate = commands.add_parser("locate")
    locate.add_argument("--root", required=True, type=Path)
    for name in ("prepare", "verify"):
        subparser = commands.add_parser(name)
        subparser.add_argument("--run", required=True, type=Path)
        if name == "verify":
            subparser.add_argument("--cpus", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.command == "locate":
        sys.stdout.write(str(latest_teacher(args.root)) + "\n")
        return
    manifest = (
        prepare(args.run, os.environ)
        if args.command == "prepare"
        else verify(args.run, args.cpus)
    )
    config = manifest["configuration"]
    LOGGER.info(
        "%s: teacher=%s; %s snapshots; split=%s; %s workers x %s BLAS threads",
        args.command,
        manifest["teacher_run"],
        manifest["sample_count"],
        manifest["split_counts"],
        config["workers"],
        config["threads"],
    )


if __name__ == "__main__":
    main()
