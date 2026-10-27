"""Small field-network input ablations for the controlled physical pilot.

The pilot's ES field is a declared budget-ES approximation, not a native
21cmFAST output. Models use no RT state, source-control labels, or test samples
for fitting. The field loss is MSE; power spectra are independently measured
diagnostics, not an assertion that this small MLP achieves a 1% power target.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from research.pc_field_emulator.evaluation import field_metrics, shell_spectra
from research.pc_field_emulator.model import train as fit_network

GROUPS = ("density", "photons", "history", "spectrum", "clumping")
ARMS = {
    "es": (),
    "budget": ("density", "photons"),
    "history": ("density", "photons", "history"),
    "all": GROUPS,
    "no_sources": ("density", "spectrum", "clumping"),
    **{f"no_{group}": tuple(other for other in GROUPS if other != group)
       for group in GROUPS},
}


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_manifest(path):
    """Resolve whole-case splits and reject duplicated files or case identities."""
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    records = manifest.get("cases", [])
    if manifest.get("schema_version") != 1 or not records:
        raise ValueError("A nonempty schema-version-1 physical pilot is required")
    identities, files = set(), set()
    for source in records:
        case_id, split = source["case_id"], source["split"]
        file = Path(source["path"])
        if not file.is_absolute():
            file = path.parent / file
        file = file.resolve()
        if case_id in identities or file in files:
            raise ValueError("Cases or case files must not occur in multiple splits")
        if not isinstance(split, str) or not split:
            raise ValueError("Every case requires a named split")
        if not file.is_file():
            raise ValueError(f"Missing case file: {file}")
        identities.add(case_id)
        files.add(file)
        source["path"] = str(file)
    for split in ("train", "validation", "test"):
        if not any(record["split"] == split for record in records):
            raise ValueError(f"Missing {split} split")
    manifest["source_manifest"] = str(path)
    manifest["source_manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return manifest


def load_case(record):
    """Read a case without pickle and verify its declared split and identity."""
    path = Path(record["path"])
    content = path.read_bytes()
    actual_sha = hashlib.sha256(content).hexdigest()
    if record.get("fields_sha256", record.get("sha256", actual_sha)) != actual_sha:
        raise ValueError("Case file hash does not match manifest")
    with np.load(path, allow_pickle=False) as saved:
        case = {name: saved[name] for name in saved.files if name != "metadata"}
        case["metadata"] = json.loads(str(saved["metadata"].item()))
    for name in ("case_id", "split", "seed"):
        if case["metadata"].get(name) != record[name]:
            raise ValueError(f"Case {name} does not match manifest")
    redshifts = np.asarray(case["redshifts"], dtype=float)
    es = np.asarray(case["xhi_es"], dtype=float)
    if (redshifts.ndim != 1 or es.ndim != 4 or es.shape[0] != len(redshifts)
            or len(set(es.shape[1:])) != 1 or es.shape[1] < 2
            or not np.isfinite(redshifts).all() or np.any(np.diff(redshifts) >= 0)):
        raise ValueError("Expected decreasing redshifts and snapshot-by-cubic ES fields")
    if case["xhi_rt"].shape != es.shape:
        raise ValueError("ES and RT labels must share snapshots and grid")
    for name in ("xhi_rt", "xhi_es"):
        if (not np.isfinite(case[name]).all()
                or np.any((case[name] < 0) | (case[name] > 1))):
            raise ValueError(f"Invalid fraction field: {name}")
    case["sha256"] = actual_sha
    return case


def _field(case, name, snapshot, shape):
    value = np.asarray(case[name], dtype=float)
    value = value[snapshot] if value.ndim == 4 else value
    if value.shape != shape or not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError(f"Invalid nonnegative input field: {name}")
    return value


def build_features(case, snapshot, arm, *, patch_radius=1,
                   radii_cMpc=(6.25, 12.5, 25.0)):
    """Construct deployment-available inputs; RT labels/state are never read.

    Source amplitude, history tilt, source concentration and density contrast
    control labels are deliberately absent. Their effects must be supplied by
    physical input fields, so they cannot leak into omitted-feature ablations.
    """
    if arm not in ARMS:
        raise ValueError(f"Unknown feature arm: {arm}")
    xhi = np.asarray(case["xhi_es"][snapshot], dtype=float)
    shape = xhi.shape
    if (xhi.ndim != 3 or len(set(shape)) != 1 or not np.isfinite(xhi).all()
            or np.any((xhi < 0) | (xhi > 1))):
        raise ValueError("Invalid ES prior")
    if type(patch_radius) is not int or not 0 <= patch_radius <= 3:
        raise ValueError("patch_radius must be an integer from zero to three")
    box = float(case["metadata"]["box_len_cMpc"])
    redshift = float(case["redshifts"][snapshot])
    if not np.isfinite([box, redshift]).all() or box <= 0 or redshift < 0:
        raise ValueError("Invalid box length or redshift")
    radii = tuple(float(radius) for radius in radii_cMpc)
    if any(not np.isfinite(radius) or radius <= 0 for radius in radii):
        raise ValueError("Smoothing radii must be finite and positive")
    groups = ARMS[arm]
    channels = {
        "xhi_es": xhi,
        "log1p_radius": np.log1p(_field(case, "radius_es_cMpc", snapshot, shape)),
    }
    if "density" in groups:
        channels["log1p_density"] = np.log1p(_field(case, "density", snapshot, shape))
    if "photons" in groups:
        channels["log1p_photons"] = np.log1p(_field(case, "photons", snapshot, shape))
    if "history" in groups:
        channels["log1p_emissivity"] = np.log1p(_field(case, "emissivity", snapshot, shape))
        channels["log1p_emission_time_mean"] = np.log1p(
            _field(case, "emission_time_mean", snapshot, shape))
    names, columns = [], []
    offsets = itertools.product(range(-patch_radius, patch_radius + 1), repeat=3)
    offsets = tuple(offsets)
    for name, value in channels.items():
        for offset in offsets:
            columns.append(np.roll(value, offset, axis=(0, 1, 2)).ravel())
            names.append(f"{name}:offset={offset}")
        for radius in radii:
            columns.append(gaussian_filter(value, radius / (box / shape[0]), mode="wrap").ravel())
            names.append(f"{name}:smooth={radius:g}")
        columns.append(np.full(xhi.size, value.mean()))
        names.append(f"{name}:global_mean")
    scalars = {"redshift": redshift, "log1p_dx_cMpc": np.log1p(box / shape[0])}
    parameters = dict(zip(case["metadata"]["parameter_names"], case["metadata"]["theta"], strict=True))
    if "spectrum" in groups:
        scalars["log10_Tbb_K"] = float(parameters["log10_Tbb_K"])
    if "clumping" in groups:
        scalars["log10_clumping"] = float(parameters["log10_clumping"])
    for name, value in scalars.items():
        columns.append(np.full(xhi.size, value))
        names.append(name)
    result = np.asarray(columns, dtype=np.float32).T.copy()
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite features")
    return result, names


def _scale_floor(features, names):
    centers = {name.split(":", 1)[0]: index for index, name in enumerate(names)
               if name.endswith(":offset=(0, 0, 0)")}
    floor = np.zeros(len(names))
    for index, name in enumerate(names):
        if ":smooth=" in name or name.endswith(":global_mean"):
            floor[index] = features[:, centers[name.split(":", 1)[0]]].std()
    return floor


def _arrays(cases, arm, samples_per_snapshot, configuration):
    features, targets, baselines, inventory = [], [], [], []
    names = None
    for case in cases:
        for snapshot in range(len(case["redshifts"])):
            matrix, columns = build_features(case, snapshot, arm, **configuration)
            if names is not None and names != columns:
                raise ValueError("Feature definitions differ between cases")
            names = columns
            key = f"{case['metadata']['case_id']}:{snapshot}:42711"
            rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))
            indices = np.sort(rng.choice(len(matrix), min(samples_per_snapshot, len(matrix)), replace=False))
            features.append(matrix[indices])
            targets.append((1 - case["xhi_rt"][snapshot]).ravel()[indices])
            baselines.append((1 - case["xhi_es"][snapshot]).ravel()[indices])
            inventory.append({"case_id": case["metadata"]["case_id"], "snapshot": snapshot,
                              "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
                              "cells": len(indices)})
    return (np.concatenate(features), np.concatenate(targets), np.concatenate(baselines), names, inventory)


def predict(model, case, snapshot):
    """Predict a snapshot with only the feature definitions saved with the model."""
    metadata = model.metadata["physical_pilot"]
    features, names = build_features(case, snapshot, metadata["arm"], **metadata["configuration"])
    if names != metadata["feature_names"]:
        raise ValueError("Inference feature definitions differ from training")
    base = (1 - case["xhi_es"][snapshot]).ravel()
    return model.predict(features, baseline=base).reshape(case["xhi_es"].shape[1:])


def evaluate(model, cases, output):
    """Retain per-case field errors and spectra, including failed predictions."""
    output.mkdir()
    records = []
    for index, case in enumerate(cases):
        predicted = []
        for snapshot, redshift in enumerate(case["redshifts"]):
            nn = predict(model, case, snapshot)
            predicted.append(nn)
            truth, es = 1 - case["xhi_rt"][snapshot], 1 - case["xhi_es"][snapshot]
            rho = _field(case, "density", snapshot, truth.shape)
            coefficient = 27 * np.sqrt((1 + redshift) / 10)
            spectra = shell_spectra({"teacher": coefficient * rho * (1 - truth),
                                     "nn": coefficient * rho * (1 - nn),
                                     "es": coefficient * rho * (1 - es)},
                                    case["metadata"]["box_len_cMpc"])
            record = {"case_id": case["metadata"]["case_id"], "seed": case["metadata"]["seed"],
                      "split": case["metadata"]["split"], "redshift": float(redshift),
                      "snapshot": snapshot,
                      "primary_snapshot": snapshot >= max(0, len(case["redshifts"]) - 3),
                      "teacher_Q_volume": float(truth.mean()), "nn": field_metrics(nn, truth, rho),
                      "es": field_metrics(es, truth, rho), "spectra": spectra}
            target_k_max = 0.6 * spectra["k_max_cMpc_inv"]
            target_shells = [0.1 <= k <= target_k_max for k in spectra["k_cMpc_inv"]]
            record["target_k_min_cMpc_inv"] = 0.1
            record["target_k_max_cMpc_inv"] = target_k_max
            for name in ("nn", "es"):
                errors = [abs(ratio - 1) for ratio in spectra[f"ratio_{name}"] if ratio is not None]
                record[f"mean_power_error_{name}"] = float(np.mean(errors)) if errors else None
                record[f"max_power_error_{name}"] = float(max(errors)) if errors else None
                target_errors = [abs(ratio - 1) for ratio, valid in zip(
                    spectra[f"ratio_{name}"], target_shells, strict=True) if valid and ratio is not None]
                record[f"target_mean_power_error_{name}"] = float(np.mean(target_errors)) if target_errors else None
                record[f"target_max_power_error_{name}"] = float(max(target_errors)) if target_errors else None
            records.append(record)
        np.savez_compressed(output / f"prediction_{index:04d}.npz", xhii_predicted=np.asarray(predicted),
                            metadata=np.asarray(json.dumps(case["metadata"], sort_keys=True)))
    informative = [row for row in records if 0.05 <= row["teacher_Q_volume"] <= 0.95]
    primary = [row for row in records if row["primary_snapshot"]]
    summary = {"snapshots": len(records), "cases": len(cases), "samples": records,
               "mean_nn_MAE": float(np.mean([row["nn"]["MAE"] for row in records])),
               "mean_es_MAE": float(np.mean([row["es"]["MAE"] for row in records])),
               "informative_snapshots": len(informative),
               "primary_snapshots": len(primary),
               "primary_definition": "last three redshift snapshots of each history, matching the physical controls",
               "target_k_definition": "0.1 <= k <= 0.6*k_Nyquist, in cMpc^-1; all raw shells retained",
               "informative_definition": "0.05 <= teacher volume xHII <= 0.95"}
    for group_name, rows in (("all", records), ("primary", primary), ("informative", informative)):
        for name in ("nn", "es"):
            summary[f"{group_name}_mean_MAE_{name}"] = (
                float(np.mean([row[name]["MAE"] for row in rows])) if rows else None)
            errors = [row[f"mean_power_error_{name}"] for row in rows
                      if row[f"mean_power_error_{name}"] is not None]
            summary[f"{group_name}_mean_power_error_{name}"] = float(np.mean(errors)) if errors else None
            target_errors = [row[f"target_mean_power_error_{name}"] for row in rows
                             if row[f"target_mean_power_error_{name}"] is not None]
            summary[f"{group_name}_target_mean_power_error_{name}"] = (
                float(np.mean(target_errors)) if target_errors else None)
            target_maxima = [row[f"target_max_power_error_{name}"] for row in rows
                             if row[f"target_max_power_error_{name}"] is not None]
            summary[f"{group_name}_target_max_power_error_{name}"] = (
                float(max(target_maxima)) if target_maxima else None)
    _write_json(output / "summary.json", summary)
    return summary


def run(manifest_path, output, *, arms=tuple(ARMS), model_seeds=(11, 23), epochs=120,
        hidden=(32, 32), samples_per_snapshot=128, batch_size=2048,
        patch_radius=1, radii_cMpc=(6.25, 12.5, 25.0), parallel_arms=1):
    """Fit whole-case ablations and select using validation MSE before testing."""
    if (not arms or len(set(arms)) != len(arms) or any(arm not in ARMS for arm in arms)
            or not model_seeds or samples_per_snapshot < 1):
        raise ValueError("Specify supported distinct arms, model seeds and positive sample count")
    if type(parallel_arms) is not int or parallel_arms < 1:
        raise ValueError("parallel_arms must be a positive integer")
    manifest = load_manifest(manifest_path)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cases = [load_case(record) for record in manifest["cases"]]
    train_cases = [case for case in cases if case["metadata"]["split"] == "train"]
    validation_cases = [case for case in cases if case["metadata"]["split"] == "validation"]
    evaluation_splits = sorted({case["metadata"]["split"] for case in cases}
                               - {"train", "validation", "sensitivity", "control", "convergence"})
    configuration = {"patch_radius": patch_radius, "radii_cMpc": radii_cMpc}

    def fit_arm(arm):
        directory = output / arm
        directory.mkdir()
        xt, yt, bt, names, train_inventory = _arrays(train_cases, arm, samples_per_snapshot, configuration)
        xv, yv, bv, validation_names, val_inventory = _arrays(
            validation_cases, arm, samples_per_snapshot, configuration)
        if names != validation_names:
            raise ValueError("Training and validation columns differ")
        _write_json(directory / "sampling.json", {"train": train_inventory, "validation": val_inventory})
        candidates = []
        for seed in model_seeds:
            model, history = fit_network(xt, yt, xv, yv, seed=seed, hidden=hidden,
                epochs=epochs, batch_size=batch_size, patience=25, min_delta=1e-8,
                baseline_train=bt, baseline_validation=bv, feature_scale_floor=_scale_floor(xt, names))
            model.metadata["physical_pilot"] = {"arm": arm, "configuration": configuration,
                                                "feature_names": names}
            model.save(directory / f"model_{seed}.npz")
            _write_json(directory / f"history_{seed}.json", history)
            candidates.append((model.metadata["best_validation_loss"], seed, model))
        loss, seed, model = min(candidates, key=lambda item: (item[0], item[1]))
        model.save(directory / "model.npz")
        selection = {"arm": arm, "seed": seed, "validation_mse": loss,
                     "epoch": model.metadata["best_epoch"], "test_used_for_selection": False}
        _write_json(directory / "selection.json", selection)
        arm_reports = {}
        for split in evaluation_splits:
            split_cases = [case for case in cases if case["metadata"]["split"] == split]
            arm_reports[split] = evaluate(model, split_cases, directory / split)
        print(json.dumps({"completed_arm": arm, "selection": selection,  # noqa: T201
                          "test_MAE": arm_reports["test"]["mean_nn_MAE"]}), flush=True)
        return arm, selection, arm_reports

    if parallel_arms == 1:
        fitted = [fit_arm(arm) for arm in arms]
    else:
        # Read-only input cases are shared; every model has its own RNG,
        # parameters and output directory. map preserves the declared order.
        with ThreadPoolExecutor(max_workers=min(parallel_arms, len(arms))) as executor:
            fitted = list(executor.map(fit_arm, arms))
    selections = [selection for _, selection, _ in fitted]
    reports = {arm: arm_reports for arm, _, arm_reports in fitted}
    selected = min(selections, key=lambda row: (row["validation_mse"], row["arm"]))
    seeds = {split: sorted({case["metadata"]["seed"] for case in cases
                           if case["metadata"]["split"] == split})
             for split in sorted({case["metadata"]["split"] for case in cases})}
    summary = {"selected": selected, "selections": selections, "arms": reports,
               "parallel_arms": parallel_arms,
               "seeds_by_split": seeds,
               "primary_test_seed_disjoint": not bool(set(seeds["train"]) & set(seeds["test"])),
               "source_manifest_sha256": manifest["source_manifest_sha256"],
               "objective": "unweighted voxel ionized-fraction MSE; PS evaluated separately",
               "training_snapshot_scope": "all saved snapshots, including the nearly neutral early phase",
               "test_used_for_selection": False, "strict_photon_conservation": False,
               "feature_groups": {"density": ["density"], "photons": ["photons"],
                                  "history": ["emissivity", "emission_time_mean"],
                                  "spectrum": ["log10_Tbb_K"], "clumping": ["log10_clumping"]},
               "ablation_interpretation": "incremental information beyond retained ES prior; not physical insensitivity",
               "brightness_convention": "saturated Ts, no RSD, shared declared density",
               "scope": "controlled small-box input ablation; not a 1% or universal-ES certificate"}
    _write_json(output / "summary.json", summary)
    (output / "COMPLETED").touch()
    return summary


def main():
    """Run the declared pilot manifest from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=tuple(ARMS), default=list(ARMS))
    parser.add_argument("--model-seeds", nargs="+", type=int, default=[11, 23])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--samples-per-snapshot", type=int, default=128)
    parser.add_argument("--parallel-arms", type=int, default=1,
                        help="Independent arms evaluated concurrently; set BLAS/OpenMP threads to one")
    args = parser.parse_args()
    run(args.manifest, args.output, arms=tuple(args.arms), model_seeds=tuple(args.model_seeds),
        epochs=args.epochs, samples_per_snapshot=args.samples_per_snapshot,
        parallel_arms=args.parallel_arms)


if __name__ == "__main__":
    main()
