"""Prepare immutable training-only inputs for a one/two-simulation fitting test.

The original validation and test fields are never loaded for this experiment.
Their identity metadata is retained solely to verify the parent's split design.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from research.es_pc_emulator.full_history import write_json
from research.es_pc_emulator.physical_pilot_training import load_manifest

ARMS = ("joint_constant16", "joint_cosine16", "joint_cosine32", "field_cosine32", "power_cosine32")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selected_records(manifest: dict, count: int) -> list[dict]:
    records = sorted((row for row in manifest["cases"] if row["split"] == "train"),
                     key=lambda row: row["case_id"])
    if len(records) < count:
        raise ValueError("Insufficient training cases for the requested fitting ceiling")
    return records[:count]


def _check_selected(records: list[dict]) -> None:
    for row in records:
        if _sha(Path(row["path"])) != row["fields_sha256"]:
            raise ValueError("Selected training field checksum changed: " + row["case_id"])


def prepare(run: Path, parent: Path, environment: dict[str, str]) -> dict:
    """Freeze source identities and settings without loading any held-out fields.

    Parameters
    ----------
    run, parent : pathlib.Path
        New fitting-test directory and completed full-history or retraining run.
    environment : dict
        Optional FIT_* overrides for the small diagnostic experiment.

    Returns
    -------
    dict
        Frozen data provenance, selected train identities and experiment options.
    """
    run, parent = run.resolve(), parent.resolve()
    if not (parent / "COMPLETED").is_file():
        raise ValueError("Require a completed full-history or history-retrain parent")
    previous_path = parent / "training/configuration.json"
    previous = json.loads(previous_path.read_text())
    if previous.get("snapshot_scope") != "all":
        raise ValueError("Parent must contain the full snapshot history")
    manifest_path = parent / "dataset_manifest.json"
    if not manifest_path.is_file():
        manifest_path = parent / "native/dataset_manifest.json"
    if Path(previous["manifest"]).resolve() != manifest_path.resolve():
        raise ValueError("Parent training manifest differs from its recorded dataset")
    original_hash = _sha(manifest_path)
    if previous.get("manifest_sha256", original_hash) != original_hash:
        raise ValueError("Parent manifest checksum changed")
    manifest = load_manifest(manifest_path)
    expected = {row["case_id"]: row for row in previous["cases"]}
    if len(expected) != len(previous["cases"]) or set(expected) != {r["case_id"] for r in manifest["cases"]}:
        raise ValueError("Parent case inventory changed")
    for row in manifest["cases"]:
        old = expected[row["case_id"]]
        if (any(old[key] != row[key] for key in ("split", "seed"))
                or old["sha256"] != row["fields_sha256"]):
            raise ValueError("Parent case identity, split or checksum changed")
    coverage_path = parent / "coverage.json"
    coverage = json.loads(coverage_path.read_text())
    if (not coverage["passed"] or coverage["case_count"] != len(expected)
            or {r["case_id"] for r in coverage["cases"]} != set(expected)):
        raise ValueError("Parent coverage gate is incomplete or mismatched")
    options = {"batch_size": previous["batch_size"], "history_steps": previous["history_steps"]}
    for name, default in (("epochs", 400), ("diagnostic_interval", 10)):
        value = int(environment.get("FIT_" + name.upper(), default))
        if value < 1:
            raise ValueError(name + " must be positive")
        options[name] = value
    case_counts = [int(item) for item in environment.get("FIT_CASE_COUNTS", "1,2").split(",")]
    if not case_counts or set(case_counts) - {1, 2} or len(case_counts) != len(set(case_counts)):
        raise ValueError("Distinct case counts from {1,2} are required")
    seeds = [int(item) for item in environment.get("FIT_SEEDS", "11,23").split(",")]
    if not seeds or min(seeds) < 0 or len(seeds) != len(set(seeds)):
        raise ValueError("Distinct nonnegative model seeds are required")
    arms = environment.get("FIT_ARMS", ",".join(ARMS)).split(",")
    if not arms or set(arms) - set(ARMS) or len(arms) != len(set(arms)):
        raise ValueError("Unknown or repeated fitting-ceiling arms")
    learning_rate = float(environment.get("FIT_LEARNING_RATE", .0003))
    minimum_rate = float(environment.get("FIT_MIN_LEARNING_RATE", .000003))
    if not all(math.isfinite(x) for x in (learning_rate, minimum_rate)) or not 0 < minimum_rate <= learning_rate:
        raise ValueError("Require finite 0 < minimum learning rate <= starting learning rate")
    options.update(case_counts=sorted(case_counts), model_seeds=seeds, arms=arms,
                   learning_rate=learning_rate, min_learning_rate=minimum_rate)
    selected = _selected_records(manifest, max(case_counts))
    _check_selected(selected)
    run.mkdir(parents=True, exist_ok=True)
    frozen_manifest = run / "dataset_manifest.json"
    write_json(frozen_manifest, manifest)
    config = {"parent_run": str(parent), "parent_manifest": str(manifest_path.resolve()),
              "parent_manifest_sha256": original_hash, "dataset_manifest_sha256": _sha(frozen_manifest),
              "parent_training_configuration_sha256": _sha(previous_path),
              "parent_coverage_sha256": _sha(coverage_path),
              "selected_training_cases": [row["case_id"] for row in selected],
              "experiment": options, "native_simulations_run": False,
              "held_out_cases_used_for_fitting": 0, "convergence_certified": False,
              "scope": "in-sample fitting ceiling; not held-out accuracy or physical convergence"}
    write_json(run / "configuration.json", config)
    return config


def verify(run: Path) -> dict:
    """Recheck source metadata and selected train-field hashes before each job.

    Parameters
    ----------
    run : pathlib.Path
        Prepared fitting-ceiling run directory.

    Returns
    -------
    dict
        Recorded configuration after all provenance checks pass.
    """
    config = json.loads((run / "configuration.json").read_text())
    parent = Path(config["parent_run"])
    for path, key in ((Path(config["parent_manifest"]), "parent_manifest_sha256"),
                      (run / "dataset_manifest.json", "dataset_manifest_sha256"),
                      (parent / "training/configuration.json", "parent_training_configuration_sha256"),
                      (parent / "coverage.json", "parent_coverage_sha256")):
        if _sha(path) != config[key]:
            raise ValueError("Recorded checksum changed: " + key)
    manifest = load_manifest(run / "dataset_manifest.json")
    selected = _selected_records(manifest, max(config["experiment"]["case_counts"]))
    if [row["case_id"] for row in selected] != config["selected_training_cases"]:
        raise ValueError("Selected training identities changed")
    _check_selected(selected)
    return config


def main(argv: list[str] | None = None) -> None:
    """Prepare or verify an immutable fitting-ceiling experiment.

    Parameters
    ----------
    argv : list of str, optional
        Arguments, or the process command line when omitted.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    create = sub.add_parser("prepare")
    create.add_argument("--run", type=Path, required=True)
    create.add_argument("--parent", type=Path, required=True)
    check = sub.add_parser("verify")
    check.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.phase == "prepare":
        prepare(args.run, args.parent, dict(os.environ))
    else:
        verify(args.run)


if __name__ == "__main__":
    main()
