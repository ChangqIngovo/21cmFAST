"""Refine a frozen emulator using existing disjoint snapshot caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from research.pc_analytic_emulator.fit import fit_field
from research.pc_analytic_emulator.model import Emulator
from research.pc_analytic_emulator.train import _implementation_hashes, _score_emulator


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_cache(features_path: Path) -> tuple[dict, dict]:
    """Verify cached fitting inputs and rebase paths without changing originals.

    Parameters
    ----------
    features_path
        The original features.json alongside its receipt.json and NPZ files.
        The folder may have moved; stored machine paths are not followed.

    Returns
    -------
    features, receipt
        Validated metadata with copied cache rows rebased to the current folder,
        and the unmodified source receipt. Only disjoint training/validation
        seeds are accepted. Every recorded cached payload is checksum checked.
    """
    features_path = Path(features_path).resolve()
    root = features_path.parent
    receipt = json.loads((root / "receipt.json").read_text())
    if not isinstance(receipt.get("identity"), dict):
        raise TypeError("Cache receipt requires its original training identity")
    files = receipt["files"]
    for name, expected in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe cache receipt path")
        path = (root / relative).resolve()
        if (
            not path.is_relative_to(root)
            or not path.is_file()
            or _digest(path) != expected
        ):
            raise ValueError(f"Cache checksum failed: {name}")
    if files.get(features_path.name) != _digest(features_path):
        raise ValueError("Features metadata is not covered by the receipt")
    features = json.loads(features_path.read_text())
    if features != receipt["record"]:
        raise ValueError("Feature metadata differs from cache receipt")
    if features.get("test_labels_in_fit_cache") is not False:
        raise ValueError("Fit cache must explicitly exclude test labels")
    seen_paths, seeds = set(), []
    configuration = receipt.get("identity", {}).get("configuration", {})
    for split in ("training", "validation"):
        rows = features[split + "_snapshots"]
        if not isinstance(rows, list) or not rows:
            raise ValueError("Both fitting splits require snapshots")
        split_seeds = set()
        for row in rows:
            if type(row.get("seed")) is not int or row["seed"] < 0:
                raise ValueError("Snapshot seed must be a nonnegative integer")
            split_seeds.add(row["seed"])
            name = Path(row["cache"]).name
            if name in seen_paths or name not in files or not name.endswith(".npz"):
                raise ValueError("Duplicate or unsealed snapshot cache")
            seen_paths.add(name)
            row["cache"] = str(root / name)
        declared = configuration.get(
            "train_seeds" if split == "training" else "validation_seeds"
        )
        if declared is not None and split_seeds != set(declared):
            raise ValueError(
                "Cached split seeds differ from original training identity"
            )
        seeds.append(split_seeds)
    if seeds[0] & seeds[1]:
        raise ValueError("Training and validation seeds overlap")
    return features, receipt


def main() -> None:
    """Run audited residual fitting from command-line paths and controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase-weight", type=float, default=0)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--spectral-power-floor", type=float, default=1e-4)
    args = parser.parse_args()
    features, receipt = verified_cache(args.features)
    args.output.mkdir(parents=True, exist_ok=False)
    training, validation = (
        features["training_snapshots"],
        features["validation_snapshots"],
    )
    if {r["seed"] for r in training} & {r["seed"] for r in validation}:
        raise ValueError("Training and validation seeds overlap")
    if features.get("test_labels_in_fit_cache"):
        raise ValueError("Fit cache contains test labels")
    base = Emulator.load(args.base)
    if features.get("feature_set", "legacy") != base.feature_set:
        raise ValueError("Cache feature convention differs from the frozen base")
    cached_radii = receipt["identity"].get("configuration", {}).get("radii_cMpc")
    if cached_radii is not None and tuple(cached_radii) != tuple(base.radii_cMpc):
        raise ValueError("Cache smoothing radii differ from the frozen base")
    common_objective = {
        "spectral_weight": 0.1,
        "spectral_power_floor": args.spectral_power_floor,
        "front_weight": 1.0,
        "field_variance_floor": 0.005,
        "k_fraction": 0.5,
        "raw_weight": 0.05,
        "phase_balance_weight": 0.0,
    }
    objective = common_objective | {"phase_balance_weight": args.phase_weight}
    source_hashes = _implementation_hashes() | {
        str(Path(__file__).resolve()): _digest(Path(__file__))
    }
    identity = {
        "implementation_sha256": source_hashes,
        "cache_receipt_sha256": _digest(args.features.parent / "receipt.json"),
        "original_cache_identity": receipt["identity"],
        "base_files": {
            str(p.relative_to(args.base)): _digest(p)
            for p in sorted(args.base.rglob("*"))
            if p.is_file()
        },
        "features_sha256": _digest(args.features),
        "snapshot_cache_hashes": {
            r["cache"]: receipt["files"][Path(r["cache"]).name]
            for r in training + validation
        },
        "optimization_objective": objective,
        "comparison_objective": common_objective,
        "seed": args.seed,
        "epochs": args.epochs,
        "hidden": [32, 32],
        "learning_rate": 0.0003,
        "geometry_balanced_sampling": True,
        "field_feature_scale_floor": 0.01,
        "zero_constant_features": True,
        "validation_selection_only": True,
    }
    (args.output / "protocol.json").write_text(json.dumps(identity, indent=2) + "\n")
    started = time.perf_counter()
    log = (args.output / "epochs.jsonl").open("w")

    def on_epoch(row: dict) -> None:
        log.write(json.dumps(row) + "\n")
        log.flush()
        sys.stdout.write(json.dumps(row) + "\n")
        sys.stdout.flush()

    field, history = fit_field(
        training,
        validation,
        base.budget_model,
        base_model=base,
        seed=args.seed,
        hidden=(32, 32),
        epochs=args.epochs,
        patience=20,
        learning_rate=0.0003,
        objective_config=objective,
        on_epoch=on_epoch,
        frame_cache_max_bytes=4 * 1024**3,
        field_feature_scale_floor=0.01,
        zero_constant_features=True,
        geometry_balanced_sampling=True,
    )
    log.close()
    model = Emulator(
        field,
        base.budget_model,
        radii_cMpc=base.radii_cMpc,
        metadata=dict(base.metadata)
        | {
            "refinement_protocol": identity,
            "model_seed": args.seed,
            "objective": objective,
            "base_kind": "frozen_deployed_refinement",
            "base_files_sha256": identity["base_files"],
            "field_checkpoint_selection": "phase-weighted validation only; final comparison uses common original objective",
        },
        feature_set=base.feature_set,
        projection_mode="capacity_l2",
        base_model=base,
    )
    model.save(args.output / "model")
    baseline_score, baseline_rows = _score_emulator(base, validation, common_objective)
    score, rows = _score_emulator(model, validation, common_objective)
    if (
        _implementation_hashes()
        | {str(Path(__file__).resolve()): _digest(Path(__file__))}
        != source_hashes
    ):
        raise ValueError("Training implementation changed during refinement")
    report = {
        "elapsed_seconds": time.perf_counter() - started,
        "validation_score": score,
        "baseline_validation_score": baseline_score,
        "validation_rows": rows,
        "baseline_validation_rows": baseline_rows,
        "history": history,
        "selected_epoch": field.metadata["best_epoch"],
        "native_physics_unchanged": True,
    }
    for key in ("training_geometries", "native_input_convention"):
        if key in base.metadata:
            assert model.metadata[key] == base.metadata[key]
    for path, expected in identity["snapshot_cache_hashes"].items():
        if _digest(Path(path)) != expected:
            raise ValueError("Snapshot cache changed during refinement")
    for rel, expected in identity["base_files"].items():
        if _digest(args.base / rel) != expected:
            raise ValueError("Frozen base changed")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    sys.stdout.write(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "elapsed_seconds",
                    "validation_score",
                    "baseline_validation_score",
                    "selected_epoch",
                )
            }
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
