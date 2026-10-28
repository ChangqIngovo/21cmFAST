"""Freeze and verify a training-only repeat of completed full-history simulations.

The parent dataset remains external and read-only. Its manifest, individual
field hashes and whole-history split identities are verified before submission
and again inside the allocated job.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np

from research.es_pc_emulator.full_history import write_json
from research.es_pc_emulator.physical_pilot_training import load_manifest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_fields(records: list[dict]) -> None:
    for row in records:
        if _sha(Path(row["path"])) != row["fields_sha256"]:
            raise ValueError("Source field checksum changed: " + row["case_id"])


def prepare(run: Path, parent: Path, environment: dict[str, str]) -> dict:
    """Prepare a training-only configuration preserving the parent's exact cases.

    Parameters
    ----------
    run, parent : pathlib.Path
        New run directory and completed full-history run directory.
    environment : dict
        Optional RETRAIN_* overrides for optimization and random seeds.

    Returns
    -------
    dict
        Recorded configuration with data hashes and explicit training arguments.
    """
    run, parent = run.resolve(), parent.resolve()
    if not (parent / "COMPLETED").is_file():
        raise ValueError("Require a completed full-history parent")
    previous = json.loads((parent / "training/configuration.json").read_text())
    if previous.get("snapshot_scope") != "all":
        raise ValueError("Parent must include all snapshots")
    manifest_path = parent / "native/dataset_manifest.json"
    if Path(previous["manifest"]).resolve() != manifest_path.resolve():
        raise ValueError("Parent training manifest differs from its native dataset")
    manifest = load_manifest(manifest_path)
    if previous.get("manifest_sha256", _sha(manifest_path)) != _sha(manifest_path):
        raise ValueError("Parent manifest checksum differs from its training provenance")
    records = manifest["cases"]
    expected = {row["case_id"]: row for row in previous["cases"]}
    if len(expected) != len(previous["cases"]) or set(expected) != {r["case_id"] for r in records}:
        raise ValueError("Parent case inventory changed")
    for row in records:
        old = expected[row["case_id"]]
        if any(old[key] != row[key] for key in ("split", "seed")) or old["sha256"] != row["fields_sha256"]:
            raise ValueError("Parent identity, split or checksum changed")
    _check_fields(records)
    counts = Counter(row["split"] for row in records)
    if set(counts) - {"train", "validation", "test", "external_test"}:
        raise ValueError("Unexpected split in full-history parent")
    coverage_path = parent / "coverage.json"
    coverage = json.loads(coverage_path.read_text())
    if (not coverage["passed"] or coverage["case_count"] != len(records)
            or {r["case_id"] for r in coverage["cases"]} != set(expected)):
        raise ValueError("Parent temporal coverage is incomplete or mismatched")
    training = {key: previous[key] for key in
                ("history_steps", "width", "batch_size", "auxiliary_weight", "power_floor")}
    for key, default in (("epochs", 120), ("patience", 30), ("diagnostic_interval", 5)):
        value = int(environment.get("RETRAIN_" + key.upper(), default))
        if value < 1:
            raise ValueError(key + " must be positive")
        training[key] = value
    seeds = [int(item) for item in environment.get("RETRAIN_SEEDS", "11,23").split(",")]
    if not seeds or min(seeds) < 0 or len(seeds) != len(set(seeds)):
        raise ValueError("Distinct nonnegative RETRAIN_SEEDS are required")
    learning_rate = float(environment.get("RETRAIN_LEARNING_RATE", .0003))
    relative_floor = float(environment.get("RETRAIN_RELATIVE_POWER_FLOOR", .0001))
    epsilon = float(environment.get("RETRAIN_PRIOR_EPSILON", .0001))
    if not all(math.isfinite(value) and value > 0 for value in (learning_rate, relative_floor, epsilon)):
        raise ValueError("Learning rate, relative-power floor and prior epsilon must be finite and positive")
    if not np.finfo(np.float32).eps <= epsilon < .5:
        raise ValueError("Prior epsilon must be at least float32 machine epsilon and below 0.5")
    training.update(model_seeds=seeds, arms=["thermo_components_p21"], snapshot_scope="all",
                    training_objective="stage-relative", output_transform="logit", prior_epsilon=epsilon,
                    relative_power_floor=relative_floor, learning_rate=learning_rate,
                    component_weight=.01, power_weight=.01,
                    train_cases=counts["train"], validation_cases=counts["validation"],
                    test_cases=counts["test"], external_cases=counts["external_test"])
    run.mkdir(parents=True, exist_ok=True)
    frozen_manifest = run / "dataset_manifest.json"
    write_json(frozen_manifest, manifest)
    config = {"parent_run": str(parent), "parent_manifest": str(manifest_path.resolve()),
              "parent_manifest_sha256": _sha(manifest_path),
              "dataset_manifest_sha256": _sha(frozen_manifest),
              "parent_training_configuration_sha256": _sha(parent / "training/configuration.json"),
              "parent_coverage_sha256": _sha(coverage_path), "coverage_requested": coverage["requested"],
              "training": training, "native_simulations_run": False, "convergence_certified": False}
    write_json(run / "configuration.json", config)
    return config


def verify(run: Path) -> dict:
    """Verify frozen and parent manifests and source fields before fitting.

    Parameters
    ----------
    run : pathlib.Path
        Prepared training-only run directory.

    Returns
    -------
    dict
        Unchanged recorded run configuration.
    """
    config = json.loads((run / "configuration.json").read_text())
    parent = Path(config["parent_run"])
    checks = ((Path(config["parent_manifest"]), "parent_manifest_sha256"),
              (run / "dataset_manifest.json", "dataset_manifest_sha256"),
              (parent / "training/configuration.json", "parent_training_configuration_sha256"),
              (parent / "coverage.json", "parent_coverage_sha256"))
    for path, key in checks:
        if _sha(path) != config[key]:
            raise ValueError("Recorded " + key + " changed")
    _check_fields(load_manifest(run / "dataset_manifest.json")["cases"])
    return config


def main(argv: list[str] | None = None) -> None:
    """Prepare or verify immutable training-only input references.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; defaults to the process arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--run", type=Path, required=True)
    prepare_parser.add_argument("--parent", type=Path, required=True)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.phase == "prepare":
        prepare(args.run, args.parent, dict(os.environ))
    else:
        verify(args.run)


if __name__ == "__main__":
    main()
