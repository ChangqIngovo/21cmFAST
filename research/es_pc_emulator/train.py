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


def _channel_scale_floor(features, names):
    """Scale filtered inputs by at least their unsmoothed training channel.

    A filter as large as the box is nearly constant within one realization.
    Dividing by its tiny within-box scatter can turn modest between-seed
    mean shifts into enormous standardized values. The unfiltered channel
    supplies a physical reference scale without inspecting validation/test.
    """
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError("Features and names must have matching columns")
    centers = {name.split(":", 1)[0]: i for i, name in enumerate(names)
               if name.endswith(":offset=(0, 0, 0)")}
    floor = np.zeros(len(names), dtype=np.float64)
    for index, name in enumerate(names):
        if ":gaussian_sigma_cMpc=" in name:
            channel = name.split(":", 1)[0]
            if channel not in centers:
                raise ValueError(f"Smoothed channel {channel} has no central unfiltered feature")
            floor[index] = values[:, centers[channel]].std()
    return floor


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


def _accuracy_report(selections, reports, best):
    """Expose failed learning as prominently as successful execution.

    Test scores describe the already selected models; they never select a
    feature set, initialization or epoch. A positive MAE reduction alone is
    not a convergence or photon-conservation certificate.
    """
    arms = {}
    for selection in selections:
        name = selection["feature_set"]
        report = reports[name]
        baseline, prediction = report["mean_es_MAE"], report["mean_nn_MAE"]
        endpoint = min(report["samples"], key=lambda row: row["sample"]["redshift"])
        worst = max(report["samples"], key=lambda row: row["nn"]["MAE"])
        arms[name] = {
            "selected_epoch": selection["best_epoch"],
            "selected_training_mse": selection["training_mse"],
            "selected_validation_mse": selection["validation_mse"],
            "selected_validation_loss": selection["validation_loss"],
            "validation_baseline_loss": selection["validation_baseline_loss"],
            "validation_selected_zero_correction": selection["zero_correction"],
            "test_snapshots": report["snapshots"],
            "mean_es_MAE": baseline,
            "mean_nn_MAE": prediction,
            "MAE_reduction_fraction": (baseline - prediction) / baseline if baseline > 0 else None,
            "test_MAE_improved": prediction < baseline,
            "lowest_redshift_sample": {
                "seed": endpoint["sample"]["seed"], "redshift": endpoint["sample"]["redshift"],
                "ES_MAE": endpoint["es"]["MAE"], "NN_MAE": endpoint["nn"]["MAE"],
            },
            "worst_nn_sample": {
                "seed": worst["sample"]["seed"], "redshift": worst["sample"]["redshift"],
                "ES_MAE": worst["es"]["MAE"], "NN_MAE": worst["nn"]["MAE"],
            },
        }
    return {
        "selected_feature_set": best["feature_set"], "arms": arms,
        "test_used_for_selection": False,
        "MAE_units": "Absolute ionized-fraction difference; multiply by 100 for percentage points",
        "scope": "Held-out field accuracy, not a full photon-budget or convergence certificate",
    }


def run(dataset, output, *, feature_sets=FEATURE_SETS, model_seeds=(11, 23, 47, 83),
        epochs=200, samples_per_snapshot=8192, snapshots_per_history=24,
        hidden=(32, 32), patch_radius=1, radii_cMpc=(6.25, 12.5, 25.0, 50.0),
        batch_size=2048, front_weight=4.0, output_mode="residual",
        allow_unconverged_teacher=False, allow_initial_state_mismatch=False,
        standardization="channel"):
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
    if standardization not in ("channel", "independent"):
        raise ValueError("standardization must be channel or independent")
    dataset = Path(dataset).resolve()
    manifest = load_dataset(dataset)
    if type(allow_unconverged_teacher) is not bool:
        raise ValueError("allow_unconverged_teacher must be boolean")
    if type(allow_initial_state_mismatch) is not bool:
        raise ValueError("allow_initial_state_mismatch must be boolean")
    initial_state_compatible = manifest.get("initial_state_compatible")
    if initial_state_compatible is False and not allow_initial_state_mismatch:
        raise ValueError("ES and RT initial states are incompatible. Generate a history starting "
                         "before appreciable ES ionization, or explicitly reproduce the mismatch "
                         "with --allow-initial-state-mismatch.")
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
        "standardization": standardization,
        "teacher_convergence_certified": convergence_certified,
        "exploratory_teacher_opt_in": allow_unconverged_teacher,
        "initial_state_compatible": initial_state_compatible,
        "initial_state_status": ("compatible" if initial_state_compatible is True else
                                 "mismatch" if initial_state_compatible is False else "legacy_unknown"),
        "exploratory_initial_state_opt_in": allow_initial_state_mismatch,
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
        scale_floor = (_channel_scale_floor(train_arrays["features"], names)
                       if standardization == "channel" else None)
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
                sample_weight_validation=validation_arrays["weights"],
                feature_scale_floor=scale_floor, **controls)
            model.metadata["es_to_c2ray"] = {
                "features": configuration, "feature_names": names,
                "dataset_sha256": dataset_hash, "target": "C2-Ray xHII",
                "baseline": "1 - native ES xHI", "inference_uses_teacher": False,
                "scope": experiment["scope"],
                "standardization": standardization,
            }
            model.save(trial_dir / "model.npz")
            _json(trial_dir / "history.json", history)
            best_epoch = model.metadata["best_epoch"]
            baseline_metrics = model.metadata.get("initial_baseline_metrics")
            selected_metrics = (baseline_metrics if best_epoch == 0 else
                                next(row for row in history if row["epoch"] == best_epoch))
            trials.append({"seed": seed, "model": str(trial_dir / "model.npz"),
                           "validation_loss": model.metadata["best_validation_loss"],
                           "best_epoch": best_epoch,
                           "training_mse": selected_metrics["training_mse"],
                           "validation_mse": selected_metrics["validation_mse"],
                           "validation_baseline_loss": (baseline_metrics["validation_loss"]
                                                        if baseline_metrics else None),
                           "zero_correction": output_mode == "residual" and best_epoch == 0})
        selected = min(trials, key=lambda row: (row["validation_loss"], row["seed"]))
        _json(arm / "model_selection.json", {"trials": trials, "selected": selected})
        selected_model = MLPModel.load(selected["model"])
        selected_model.save(arm / "model.npz")
        selections.append(selected | {"feature_set": feature_set, "model": str(arm / "model.npz")})
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
    summary["initial_state_compatible"] = initial_state_compatible
    summary["initial_state_status"] = experiment["initial_state_status"]
    summary["standardization"] = standardization
    summary["exploratory_initial_state_opt_in"] = allow_initial_state_mismatch
    summary["accuracy_report"] = _accuracy_report(selections, reports, best)
    _json(output / "accuracy_report.json", summary["accuracy_report"])
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
    parser.add_argument("--standardization", choices=["channel", "independent"], default="channel",
                        help="Channel scale floor for smoothed inputs; independent reproduces old scaling")
    parser.add_argument("--allow-unconverged-teacher", action="store_true",
                        help="Explicitly train an exploratory model on labels not certified converged")
    parser.add_argument("--allow-initial-state-mismatch", action="store_true",
                        help="Explicitly reproduce an exploratory history with incompatible ES/RT initial states")
    options = vars(parser.parse_args())
    options["radii_cMpc"] = options.pop("radii_cmpc")
    summary = run(**options)
    print("Held-out accuracy (fraction MAE; 0.01 = one percentage point):", flush=True)  # noqa: T201
    for name, row in summary["accuracy_report"]["arms"].items():
        print(f"{name}: ES={row['mean_es_MAE']:.8g}; NN={row['mean_nn_MAE']:.8g}; "  # noqa: T201
              f"epoch={row['selected_epoch']}; "
              f"zero_correction={row['validation_selected_zero_correction']}; "
              f"test_MAE_improved={row['test_MAE_improved']}", flush=True)
        print(f"  Lowest redshift: {row['lowest_redshift_sample']}; "  # noqa: T201
              f"worst NN snapshot: {row['worst_nn_sample']}", flush=True)
    print(json.dumps(summary, default=str), flush=True)  # noqa: T201


if __name__ == "__main__":
    main()
