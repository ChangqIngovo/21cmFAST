"""Train and independently test spatial ES-to-C2-Ray residual emulators.

The label is C2-Ray hydrogen ionization, not an input feature. All models are
selected by validation loss before the held-out test fields are reconstructed.
There is deliberately no one-photon-per-ion projection: the teacher includes
helium, recombinations and secondary ionizations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from research.pc_field_emulator.evaluation import field_metrics, shell_spectra
from research.pc_field_emulator.model import MLPModel
from research.pc_field_emulator.model import train as fit_network

from .features import FEATURE_SETS, from_pair
from .pairs import load_dataset, load_pair


def _json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _weights(xhii, front_weight):
    ionized = xhii >= 0.5
    front = np.zeros_like(ionized)
    for axis in range(3):
        for offset in (-1, 1):
            front |= ionized != np.roll(ionized, offset, axis=axis)
    return np.where(front, front_weight, 1.0).ravel()


def _selected_samples(samples, limit):
    grouped = {}
    for record in samples:
        key = (record["seed"], record["dimension"], record["box_len_cMpc"])
        grouped.setdefault(key, []).append(record)
    result = []
    for key in sorted(grouped):
        records = sorted(grouped[key], key=lambda row: row["redshift"], reverse=True)
        indices = np.unique(np.linspace(0, len(records) - 1, min(limit, len(records))).round().astype(int))
        result.extend(records[int(index)] for index in indices)
    return result


def _arrays(records, configuration, samples_per_snapshot, sampling_seed, front_weight):
    collected = {key: [] for key in ("features", "target", "baseline", "weights")}
    names = None
    inventory = []
    for record in records:
        pair = load_pair(record)
        features, columns = from_pair(pair, **configuration)
        if names is not None and columns != names:
            raise ValueError("Feature definitions differ between snapshots")
        names = columns
        # Stable sample identity makes all ablation arms use identical cells.
        identity = hashlib.sha256(f"{sampling_seed}:{record['sha256']}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(identity[:8], "little"))
        count = min(samples_per_snapshot, len(features))
        indices = np.sort(rng.choice(len(features), count, replace=False))
        target = 1.0 - pair["xhi_c2ray"]
        collected["features"].append(features[indices])
        collected["target"].append(target.ravel()[indices])
        collected["baseline"].append((1.0 - pair["xhi_es"]).ravel()[indices])
        collected["weights"].append(_weights(target, front_weight)[indices])
        inventory.append({"sha256": record["sha256"], "cells": count,
                          "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest()})
    if not records:
        raise ValueError("Empty training or validation split")
    return {key: np.concatenate(value) for key, value in collected.items()}, names, inventory


def predict_pair(model, pair):
    """Predict a neutral-fraction cube using no target-derived quantities."""
    configuration = model.metadata["es_to_c2ray"]["features"]
    features, columns = from_pair(pair, **configuration)
    if columns != model.metadata["es_to_c2ray"]["feature_names"]:
        raise ValueError("Inference feature order differs from the fitted model")
    controls = {"baseline": (1.0 - pair["xhi_es"]).ravel()} if model.output_mode == "residual" else {}
    return 1.0 - model.predict(features, **controls).reshape(pair["xhi_es"].shape)


def _evaluate(model, records, directory):
    directory.mkdir(parents=True, exist_ok=False)
    reports = []
    for index, record in enumerate(records):
        pair = load_pair(record)
        started = time.perf_counter()
        predicted = predict_pair(model, pair)
        elapsed = time.perf_counter() - started
        truth = 1.0 - pair["xhi_c2ray"]
        baseline = 1.0 - pair["xhi_es"]
        rho = pair["density"]
        coefficient = 27.0 * np.sqrt((1.0 + record["redshift"]) / 10.0)
        report = {
            "sample": record, "inference_seconds": elapsed,
            "nn": field_metrics(1.0 - predicted, truth, rho),
            "es": field_metrics(baseline, truth, rho),
            "spectra": shell_spectra({
                "teacher": coefficient * rho * pair["xhi_c2ray"],
                "nn": coefficient * rho * predicted,
                "es": coefficient * rho * pair["xhi_es"],
            }, record["box_len_cMpc"]),
            "brightness_convention": "27 sqrt((1+z)/10) density xHI mK; saturated Ts; no RSD",
            "count_definition": "Hydrogen ion stock relative to teacher; not a complete photon-conservation residual",
        }
        name = f"sample_{index:04d}"
        np.savez_compressed(directory / f"{name}.npz", xhi_predicted=predicted,
                            xhi_es=pair["xhi_es"], xhi_c2ray=pair["xhi_c2ray"],
                            density=rho, metadata=np.asarray(json.dumps(record, sort_keys=True)))
        _json(directory / f"{name}.json", report)
        reports.append(report)
    summary = {
        "snapshots": len(reports),
        "mean_nn_MAE": float(np.mean([r["nn"]["MAE"] for r in reports])),
        "mean_es_MAE": float(np.mean([r["es"]["MAE"] for r in reports])),
        "samples": reports,
    }
    _json(directory / "summary.json", summary)
    return summary


def run(dataset, output, *, feature_sets=FEATURE_SETS, model_seeds=(11, 23, 47, 83),
        epochs=200, samples_per_snapshot=8192, snapshots_per_history=24,
        hidden=(32, 32), patch_radius=1, radii_cMpc=(6.25, 12.5, 25.0, 50.0),
        batch_size=2048, front_weight=4.0, output_mode="residual",
        allow_unconverged_teacher=False):
    """Fit ablations on independent seeds and evaluate each on held-out fields."""
    if (not feature_sets or len(set(feature_sets)) != len(feature_sets)
            or any(item not in FEATURE_SETS for item in feature_sets)):
        raise ValueError("Specify distinct supported feature sets")
    if (not model_seeds or len(set(model_seeds)) != len(model_seeds)
            or any(type(seed) is not int or seed < 0 for seed in model_seeds)):
        raise ValueError("Specify distinct nonnegative integer model seeds")
    for name, value in (("epochs", epochs), ("samples_per_snapshot", samples_per_snapshot),
                        ("snapshots_per_history", snapshots_per_history), ("batch_size", batch_size)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not np.isfinite(front_weight) or front_weight <= 0:
        raise ValueError("front_weight must be finite and positive")
    if output_mode not in ("residual", "direct"):
        raise ValueError("output_mode must be residual or direct")
    dataset = Path(dataset).resolve()
    manifest = load_dataset(dataset)
    if type(allow_unconverged_teacher) is not bool:
        raise ValueError("allow_unconverged_teacher must be boolean")
    convergence_certified = manifest.get("convergence_certified") is True
    if not convergence_certified and not allow_unconverged_teacher:
        raise ValueError("Teacher convergence is not certified. Finish convergence validation, "
                         "or use --allow-unconverged-teacher for an explicitly exploratory fit.")
    output = Path(output).resolve()
    if os.path.lexists(output):
        raise FileExistsError("Use a fresh training output directory")
    # Never publish model selection or completion for an incomplete experiment.
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    splits = {name: _selected_samples(
        [row for row in manifest["samples"] if row["split"] == name], snapshots_per_history)
        for name in ("train", "validation", "test")}
    if any(not rows for rows in splits.values()):
        raise ValueError("Train, validation and test must all contain independent samples")
    with dataset.open("rb") as stream:
        dataset_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    experiment = {
        "schema_version": 1, "experiment": "native-es-to-c2ray-residual-emulation",
        "dataset": str(dataset), "dataset_sha256": dataset_hash,
        "selected_samples": splits, "model_seeds": list(model_seeds),
        "output_mode": output_mode, "front_weight": front_weight,
        "teacher_convergence_certified": convergence_certified,
        "exploratory_teacher_opt_in": allow_unconverged_teacher,
        "model_selection": "Minimum weighted validation MSE; test never selects weights or feature set",
        "scope": "Coeval field correction, not an autoregressive RT solver or guaranteed photon-conserving model",
        "network": {"type": "NumPy spatial-feature MLP", "hidden": list(hidden)},
    }
    _json(output / "experiment.json", experiment)
    selections = []
    for feature_set in feature_sets:
        configuration = {"feature_set": feature_set, "patch_radius": patch_radius,
                         "radii_cMpc": list(radii_cMpc)}
        train_arrays, names, train_inventory = _arrays(
            splits["train"], configuration, samples_per_snapshot, 2026, front_weight)
        validation_arrays, validation_names, validation_inventory = _arrays(
            splits["validation"], configuration, samples_per_snapshot, 2026, front_weight)
        if names != validation_names:
            raise ValueError("Train/validation feature definitions differ")
        arm = output / feature_set
        arm.mkdir()
        _json(arm / "sampling.json", {"train": train_inventory, "validation": validation_inventory})
        trials = []
        for seed in model_seeds:
            trial_dir = arm / f"seed_{seed}"
            trial_dir.mkdir()
            controls = ({"baseline_train": train_arrays["baseline"],
                         "baseline_validation": validation_arrays["baseline"]}
                        if output_mode == "residual" else {})
            model, history = fit_network(
                train_arrays["features"], train_arrays["target"],
                validation_arrays["features"], validation_arrays["target"],
                seed=seed, hidden=tuple(hidden), epochs=epochs, batch_size=batch_size,
                sample_weight_train=train_arrays["weights"],
                sample_weight_validation=validation_arrays["weights"], **controls)
            model.metadata["es_to_c2ray"] = {
                "features": configuration, "feature_names": names,
                "dataset_sha256": dataset_hash, "target": "C2-Ray xHII",
                "baseline": "1 - native ES xHI", "inference_uses_teacher": False,
                "scope": experiment["scope"],
            }
            model.save(trial_dir / "model.npz")
            _json(trial_dir / "history.json", history)
            trials.append({"seed": seed, "model": str(trial_dir / "model.npz"),
                           "validation_loss": model.metadata["best_validation_loss"],
                           "best_epoch": model.metadata["best_epoch"]})
        selected = min(trials, key=lambda row: (row["validation_loss"], row["seed"]))
        _json(arm / "model_selection.json", {"trials": trials, "selected": selected})
        selected_model = MLPModel.load(selected["model"])
        selected_model.save(arm / "model.npz")
        selections.append({"feature_set": feature_set, "model": str(arm / "model.npz"),
                           "validation_loss": selected["validation_loss"]})
        del train_arrays, validation_arrays
    best = min(selections, key=lambda row: (row["validation_loss"], row["feature_set"]))
    _json(output / "model_selection.json", {"arms": selections, "selected": best,
                                           "test_used_for_selection": False})
    MLPModel.load(best["model"]).save(output / "model.npz")
    # Only now reconstruct held-out labels/predictions and report every arm.
    reports = {}
    for selected in selections:
        reports[selected["feature_set"]] = _evaluate(
            MLPModel.load(selected["model"]), splits["test"],
            output / selected["feature_set"] / "test")
    summary = {"schema_version": 1, "selected": best, "arms": reports,
               "dataset_sha256": dataset_hash, "elapsed_seconds": time.perf_counter() - started,
               "scope": experiment["scope"], "strict_photon_conservation": False}
    summary["teacher_convergence_certified"] = convergence_certified
    summary["exploratory_teacher_opt_in"] = allow_unconverged_teacher
    _json(output / "summary.json", summary)
    (output / "COMPLETED").write_text("Independent-seed fitting and held-out evaluation completed.\n")
    return summary


def main():
    """Fit the paired C2-Ray training dataset without regenerating simulations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-sets", nargs="+", choices=FEATURE_SETS, default=list(FEATURE_SETS))
    parser.add_argument("--model-seeds", nargs="+", type=int, default=[11, 23, 47, 83])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--samples-per-snapshot", type=int, default=8192)
    parser.add_argument("--snapshots-per-history", type=int, default=24)
    parser.add_argument("--hidden", nargs="+", type=int, default=[32, 32])
    parser.add_argument("--patch-radius", type=int, default=1)
    parser.add_argument("--radii-cmpc", nargs="*", type=float, default=[6.25, 12.5, 25.0, 50.0])
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--front-weight", type=float, default=4.0)
    parser.add_argument("--output-mode", choices=["residual", "direct"], default="residual")
    parser.add_argument("--allow-unconverged-teacher", action="store_true",
                        help="Explicitly train an exploratory model on labels not certified converged")
    options = vars(parser.parse_args())
    options["radii_cMpc"] = options.pop("radii_cmpc")
    print(json.dumps(run(**options), default=str), flush=True)  # noqa: T201


if __name__ == "__main__":
    main()
