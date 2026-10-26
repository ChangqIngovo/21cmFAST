"""Train and evaluate a field emulator using existing committed PC checkpoints.

This experiment predicts coeval coarse ionization maps from cumulative source
counts and density. It learns the existing fixed-gas teacher, including its
approximations. Input photons set the optional global count constraint; held-out
ionization fields are used only for evaluation. No simulation is launched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import multiprocessing
import os
import platform
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import scipy

if __package__:
    from .data import (
        Sample,
        UnresolvedPhotonBudgetError,
        discover_samples,
        load_sample,
        split_samples,
    )
    from .evaluation import evaluate_fields, plot_comparison
    from .features import build_features
    from .model import MLPModel, project_global_budget, train
else:
    from data import (
        Sample,
        UnresolvedPhotonBudgetError,
        discover_samples,
        load_sample,
        split_samples,
    )
    from evaluation import evaluate_fields, plot_comparison
    from features import build_features
    from model import MLPModel, project_global_budget, train

LOGGER = logging.getLogger(__name__)


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Unsupported JSON value: {type(value)}")


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, default=_json_value, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _log(stage: str, **values: Any) -> None:
    LOGGER.info(json.dumps({"stage": stage, **values}, default=_json_value))


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _validate_destination(output: Path, inputs: list[Path]) -> None:
    if output.exists():
        raise FileExistsError(f"Use a fresh training output directory: {output}")
    for source in inputs:
        if output.is_relative_to(source):
            raise ValueError("Training output must be outside every simulation input")


def _select_samples(
    samples: list[Sample], *, q_min: float, q_max: float, maximum: int
) -> list[Sample]:
    if not 0 <= q_min < q_max <= 1 or maximum < 1:
        raise ValueError(
            "Require 0 <= q_min < q_max <= 1 and a positive snapshot limit"
        )
    selected = []
    for seed in sorted({row.seed for row in samples}):
        eligible = sorted(
            [
                row
                for row in samples
                if row.seed == seed and q_min <= row.q_mass <= q_max
            ],
            key=lambda row: row.node_index,
        )
        indices = np.linspace(
            0, len(eligible) - 1, min(len(eligible), maximum), dtype=int
        )
        selected.extend(eligible[index] for index in indices)
    if not selected:
        raise ValueError(
            "No committed snapshots lie inside the selected ionization window"
        )
    return selected


def _prepare_arrays(
    samples: list[Sample], destination: Path, name: str, *, count: int, seed: int
) -> list[str]:
    rng = np.random.default_rng(seed)
    feature_rows, target_rows, feature_names = [], [], None
    for sample in samples:
        fields = load_sample(sample)
        features, names = build_features(
            fields["capacity"], fields["photons"], fields["redshift"], fields["box_len"]
        )
        if feature_names is not None and names != feature_names:
            raise ValueError("Training feature definitions changed between snapshots")
        feature_names = names
        indices = rng.choice(
            len(features), size=min(count, len(features)), replace=False
        )
        feature_rows.append(features[indices])
        target_rows.append(fields["xHII"].reshape(-1)[indices])
        _log("sample_prepared", split=name, seed=sample.seed, node=sample.node_index)
    if not feature_rows:
        raise ValueError(f"Empty {name} sample split")
    np.save(
        destination / f"x_{name}.npy", np.concatenate(feature_rows), allow_pickle=False
    )
    np.save(
        destination / f"y_{name}.npy", np.concatenate(target_rows), allow_pickle=False
    )
    return feature_names


def _coverage(samples: list[Sample]) -> dict:
    return {
        "count": len(samples),
        "seeds": sorted({sample.seed for sample in samples}),
        "redshift_range": (
            [
                min(sample.redshift for sample in samples),
                max(sample.redshift for sample in samples),
            ]
            if samples
            else None
        ),
        "Q_mass_range": (
            [
                min(sample.q_mass for sample in samples),
                max(sample.q_mass for sample in samples),
            ]
            if samples
            else None
        ),
    }


def _qualify_splits(splits: dict, output: Path) -> tuple[dict, dict]:
    """Audit before fitting and exclude only a known projection incompatibility."""
    qualified = {"strategy": splits["strategy"]}
    report = {
        "policy": "Exclude only audited UnresolvedPhotonBudgetError checkpoints",
        "unresolved_fraction_tolerance": 1e-7,
        "eligible": [],
        "excluded": [],
        "coverage": {},
        "seed_roles_reassigned": False,
        "evaluation_scope": "Only snapshots compatible with the emitted-budget projection; see excluded epochs and per-seed coverage",
    }
    for name in ("train", "validation", "test"):
        accepted = []
        for sample in splits[name]:
            record = {
                "split": name,
                "seed": sample.seed,
                "node_index": sample.node_index,
                "redshift": sample.redshift,
                "Q_mass": sample.q_mass,
                "checkpoint": str(sample.checkpoint_path),
                "checkpoint_sha256": sample.checkpoint_sha256,
            }
            # Corrupt files, failed count ledgers and all other errors remain fatal.
            try:
                fields = load_sample(sample)
            except UnresolvedPhotonBudgetError as error:
                record["diagnostics"] = error.diagnostics
                report["excluded"].append(record)
                _log("sample_excluded", **record)
            else:
                record["ledger"] = fields["ledger"]
                report["eligible"].append(record)
                accepted.append(sample)
                del fields
        qualified[name] = accepted
        requested = splits[name]
        per_seed = {}
        for seed in sorted({sample.seed for sample in requested}):
            before = [sample for sample in requested if sample.seed == seed]
            after = [sample for sample in accepted if sample.seed == seed]
            per_seed[str(seed)] = {
                "requested": _coverage(before),
                "accepted": _coverage(after),
                "excluded_count": len(before) - len(after),
            }
        report["coverage"][name] = {
            "requested": _coverage(requested),
            "accepted": _coverage(accepted),
            "excluded_count": len(requested) - len(accepted),
            "by_seed": per_seed,
        }
    report["excluded_count"] = len(report["excluded"])
    report["eligible_count"] = len(report["eligible"])
    report["empty_splits"] = [
        name for name in ("train", "validation", "test") if not qualified[name]
    ]
    _write_json(output / "preflight.json", report)
    if report["empty_splits"]:
        raise ValueError(
            "No budget-compatible snapshots in original split(s): "
            + ", ".join(report["empty_splits"])
            + "; inspect preflight.json. Seed roles were not reassigned."
        )
    _log(
        "preflight_complete",
        eligible_count=report["eligible_count"],
        excluded_count=report["excluded_count"],
        coverage=report["coverage"],
    )
    return qualified, report


def _fit_trial(task: dict) -> dict:
    arrays, output = Path(task["arrays"]), Path(task["output"])
    output.mkdir()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    inputs = [
        np.load(arrays / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in ("x_train", "y_train", "x_validation", "y_validation")
    ]
    start = time.perf_counter()

    def record_epoch(record: dict) -> None:
        _write_json(output / "progress.json", record)
        if record["epoch"] == 1 or record["epoch"] % 10 == 0:
            _log("training_epoch", model_seed=task["seed"], **record)

    model, history = train(
        *inputs,
        seed=task["seed"],
        hidden=tuple(task["hidden"]),
        epochs=task["epochs"],
        batch_size=task["batch_size"],
        patience=task["patience"],
        on_epoch=record_epoch,
    )
    seconds = time.perf_counter() - start
    path = output / "model.npz"
    model.save(path)
    result = {
        "seed": task["seed"],
        "model": str(path),
        "training_seconds": seconds,
        "epochs_run": len(history),
        "training_mse": float(np.mean((model.predict(inputs[0]) - inputs[1]) ** 2)),
        "validation_mse": float(np.mean((model.predict(inputs[2]) - inputs[3]) ** 2)),
    }
    _write_json(output / "history.json", history)
    _write_json(output / "summary.json", result)
    return result


def _choose_trial(trials: list[dict]) -> dict:
    if not trials or any(not np.isfinite(row["validation_mse"]) for row in trials):
        raise ValueError("All training trials must return finite validation losses")
    return min(trials, key=lambda row: (row["validation_mse"], row["seed"]))


def _infer_sample(model: MLPModel, fields: dict) -> tuple[dict, dict]:
    start = time.perf_counter()
    features, _ = build_features(
        fields["capacity"], fields["photons"], fields["redshift"], fields["box_len"]
    )
    after_features = time.perf_counter()
    raw = model.predict(features).reshape(fields["capacity"].shape)
    after_network = time.perf_counter()
    projected, audit = project_global_budget(raw, fields["capacity"], fields["budget"])
    after_projection = time.perf_counter()
    # A simple source-local baseline receives the identical INPUT photon budget.
    local = np.clip(fields["photons"] / fields["capacity"], 0.0, 1.0)
    baseline, baseline_audit = project_global_budget(
        local, fields["capacity"], fields["budget"]
    )
    predicted = {"raw": raw, "projected": projected, "baseline": baseline}
    diagnostic = {
        "features_seconds": after_features - start,
        "network_seconds": after_network - after_features,
        "projection_seconds": after_projection - after_network,
        "emulator_compute_seconds": after_projection - start,
        "timing_scope": "Features + network + global projection; excludes input I/O, diagnostics and output I/O",
        "projection": audit,
        "baseline_projection": baseline_audit,
    }
    return predicted, diagnostic


def _evaluate_test(model: MLPModel, samples: list[Sample], output: Path) -> list[dict]:
    reports = []
    output.mkdir()
    for sample in samples:
        stem = f"seed{sample.seed}_node{sample.node_index:05d}"
        fields = load_sample(sample)
        predicted, timing = _infer_sample(model, fields)
        report = evaluate_fields(
            fields["capacity"],
            fields["xHII"],
            predicted["raw"],
            predicted["projected"],
            fields["redshift"],
            fields["box_len"],
            baseline=predicted["baseline"],
        )
        report.update(
            seed=sample.seed,
            node_index=sample.node_index,
            redshift=sample.redshift,
            teacher_checkpoint=str(sample.checkpoint_path),
            teacher_checkpoint_sha256=sample.checkpoint_sha256,
            input_ledger=fields["ledger"],
            timing=timing,
        )
        _write_json(output / f"{stem}.json", report)
        np.savez_compressed(
            output / f"{stem}.npz",
            capacity=fields["capacity"],
            photons=fields["photons"],
            teacher=fields["xHII"],
            **predicted,
            redshift=np.asarray(sample.redshift),
            box_len=np.asarray(sample.box_len),
        )
        plot_comparison(
            fields["capacity"],
            fields["xHII"],
            predicted["raw"],
            predicted["projected"],
            fields["redshift"],
            fields["box_len"],
            output / f"{stem}.png",
            report=report,
        )
        reports.append(report)
        _log(
            "held_out_evaluated",
            seed=sample.seed,
            node=sample.node_index,
            seconds=timing["emulator_compute_seconds"],
        )
    return reports


def _write_metrics(path: Path, reports: list[dict]) -> None:
    rows = []
    for report in reports:
        for method, metric in report["metrics"].items():
            row = {
                "seed": report["seed"],
                "node": report["node_index"],
                "redshift": report["redshift"],
                "method": method,
            }
            row.update(
                {
                    key: value
                    for key, value in metric.items()
                    if value is None or isinstance(value, (int, float, str, bool))
                }
            )
            rows.append(row)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(
    input_roots: list[Path],
    output: Path,
    *,
    workers: int = 1,
    model_seeds: tuple[int, ...] = (11, 23, 47, 83),
    epochs: int = 100,
    samples_per_snapshot: int = 4096,
    max_snapshots_per_seed: int = 24,
    subcells: int = 4,
    angles: int = 768,
    stride: int = 4,
    q_min: float = 0.01,
    q_max: float = 0.99,
    sample_seed: int = 2026,
    hidden: tuple[int, ...] = (32, 32),
    batch_size: int = 2048,
    patience: int = 12,
) -> dict:
    """Fit on training seeds, select on validation, then evaluate held-out fields.

    Parameters
    ----------
    input_roots : list of Path
        Existing production or evolution-matrix run/results directories.
    output : Path
        Fresh artifact directory outside all input runs.
    workers : int
        Concurrent independent network fits. BLAS threads are set before launch.
    model_seeds : tuple of int
        Network initializations; only validation MSE selects the final model.
    epochs, samples_per_snapshot, max_snapshots_per_seed : int
        Training and uniformly sampled dataset size limits.
    subcells, angles, stride : int
        A single teacher numerical arm shared by all selected data.
    q_min, q_max : float
        Teacher mass-ionized-fraction window for snapshot eligibility.
    sample_seed : int
        Uniform training/validation cell-sampling seed.
    hidden : tuple of int
        Neural hidden-layer widths.
    batch_size, patience : int
        Optimizer batch size and validation stopping patience.

    Returns
    -------
    dict
        Model-selection, held-out accuracy, timing and provenance summary.
    """
    start = time.perf_counter()
    roots = [Path(path).resolve() for path in input_roots]
    output = Path(output).resolve()
    _validate_destination(output, roots)
    if workers < 1 or samples_per_snapshot < 1 or not model_seeds:
        raise ValueError(
            "Positive workers, sample count and at least one model seed required"
        )
    if len(set(model_seeds)) != len(model_seeds):
        raise ValueError("Model initialization seeds must be unique")
    samples = [
        sample
        for root in roots
        for sample in discover_samples(
            root, subcells=subcells, angles=angles, stride=stride
        )
    ]
    keys = [(sample.seed, sample.node_index) for sample in samples]
    if len(keys) != len(set(keys)):
        raise ValueError(
            "Duplicate seed/node histories across inputs; select one history per seed"
        )
    if len({(sample.dimension, sample.box_len) for sample in samples}) != 1:
        raise ValueError(
            "First experiment requires a common physical box and coarse resolution"
        )
    selected = _select_samples(
        samples, q_min=q_min, q_max=q_max, maximum=max_snapshots_per_seed
    )
    initial_splits = split_samples(selected)
    output.mkdir(parents=True)
    splits, preflight = _qualify_splits(initial_splits, output)
    arrays = output / "training_arrays"
    arrays.mkdir()
    manifest = {
        "schema": 1,
        "scope": "Coeval coarse-field emulator of the existing fixed-gas PC teacher",
        "input_roots": roots,
        "selection": {
            "subcells": subcells,
            "angles": angles,
            "stride": stride,
            "q_min": q_min,
            "q_max": q_max,
            "max_snapshots_per_seed": max_snapshots_per_seed,
        },
        "split_strategy": splits["strategy"],
        "sample_eligibility": preflight,
        "splits": {
            name: [asdict(row) for row in splits[name]]
            for name in ("train", "validation", "test")
        },
        "samples_per_snapshot": samples_per_snapshot,
        "sample_seed": sample_seed,
        "model_seeds": model_seeds,
        "workers": min(workers, len(model_seeds)),
        "epochs": epochs,
        "hidden": hidden,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "source_sha256": {
            path.name: _digest(path)
            for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        "global_projection_target": "min(sum(INPUT source_emitted), sum(atom capacity))",
        "conservation_scope": "Global count only; no per-source/directional or reachability guarantee",
        "history_scope": "Independent coeval predictions; not an autoregressive evolution solver",
    }
    _write_json(output / "dataset_manifest.json", manifest)
    feature_names = _prepare_arrays(
        splits["train"], arrays, "train", count=samples_per_snapshot, seed=sample_seed
    )
    validation_names = _prepare_arrays(
        splits["validation"],
        arrays,
        "validation",
        count=samples_per_snapshot,
        seed=sample_seed + 1,
    )
    if validation_names != feature_names:
        raise ValueError("Train/validation feature schemas differ")
    _write_json(output / "features.json", feature_names)
    trials_dir = output / "trials"
    trials_dir.mkdir()
    tasks = [
        {
            "arrays": str(arrays),
            "output": str(trials_dir / f"seed{seed}"),
            "seed": seed,
            "hidden": hidden,
            "epochs": epochs,
            "batch_size": batch_size,
            "patience": patience,
        }
        for seed in model_seeds
    ]
    training_start = time.perf_counter()
    if workers == 1 or len(tasks) == 1:
        trials = [_fit_trial(task) for task in tasks]
    else:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)),
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            trials = list(executor.map(_fit_trial, tasks))
    training_wall_seconds = time.perf_counter() - training_start
    chosen = _choose_trial(trials)
    _write_json(output / "model_selection.json", {"trials": trials, "selected": chosen})
    shutil.copyfile(chosen["model"], output / "model.npz")
    model = MLPModel.load(output / "model.npz")
    # Audited test targets do not enter features, scaling, fitting or model selection.
    reports = _evaluate_test(model, splits["test"], output / "test")
    _write_metrics(output / "test_metrics.csv", reports)
    summary = {
        "scope": manifest["scope"],
        "split_strategy": splits["strategy"],
        "sample_eligibility": preflight,
        "evaluation_scope": preflight["evaluation_scope"],
        "snapshot_counts": {
            name: len(splits[name]) for name in ("train", "validation", "test")
        },
        "selected_trial": chosen,
        "trials": trials,
        "training_wall_seconds": training_wall_seconds,
        "total_seconds": time.perf_counter() - start,
        "feature_names": feature_names,
        "test": reports,
        "global_projection_target": manifest["global_projection_target"],
        "conservation_scope": manifest["conservation_scope"],
        "history_scope": manifest["history_scope"],
        "emulator_compute_seconds_median": float(
            np.median(
                [report["timing"]["emulator_compute_seconds"] for report in reports]
            )
        ),
        "reference_speedup": None,
        "reference_speedup_reason": "No equivalent-task teacher solve was timed during training",
    }
    _write_json(output / "summary.json", summary)
    (output / "COMPLETED").touch()
    _log("completed", output=str(output), total_seconds=summary["total_seconds"])
    return summary


def main() -> None:
    """Run the existing-output training experiment from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[11, 23, 47, 83])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--samples-per-snapshot", type=int, default=4096)
    parser.add_argument("--max-snapshots-per-seed", type=int, default=24)
    parser.add_argument("--subcells", type=int, default=4)
    parser.add_argument("--angles", type=int, default=768)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--q-min", type=float, default=0.01)
    parser.add_argument("--q-max", type=float, default=0.99)
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--hidden", type=int, nargs="+", default=[32, 32])
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--patience", type=int, default=12)
    args = vars(parser.parse_args())
    args["input_roots"] = args.pop("input")
    args["model_seeds"] = tuple(args["model_seeds"])
    args["hidden"] = tuple(args["hidden"])
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_experiment(**args)


if __name__ == "__main__":
    main()
