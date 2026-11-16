"""Diagnose saved absorption heads without replacing a trained emulator.

Run as a module from the repository root. Optional fitting probes use only the
original training seeds, with held-out seeds excluded even from checkpoint
selection. Teacher fields are labels for diagnosis, never inference features.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from research.pc_analytic_emulator.budget import load_budget, predict_efficiency
from research.pc_analytic_emulator.data import load_dataset
from research.pc_analytic_emulator.model import (
    Emulator,
    absorption_efficiency,
    budget_features,
)
from research.pc_field_emulator.model import MLPModel, train


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _stats(values: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "maximum": float(np.max(values)),
    }


def _errors(
    eta: np.ndarray, prediction: np.ndarray, mass_scale: np.ndarray
) -> dict[str, Any]:
    error = prediction - eta
    mass_error = mass_scale * error
    neutral = 1 - mass_scale * eta
    meaningful = neutral > 1e-8
    return {
        "count": len(eta),
        "efficiency_MAE": float(np.mean(np.abs(error))),
        "efficiency_MSE": float(np.mean(error**2)),
        "mass_Q_MAE": float(np.mean(np.abs(mass_error))),
        "mass_Q_MSE": float(np.mean(mass_error**2)),
        "mass_Q_bias": float(np.mean(mass_error)),
        "maximum_absolute_mass_Q_error": float(np.max(np.abs(mass_error))),
        "relative_neutral_error_count": int(meaningful.sum()),
        "relative_neutral_MAE": float(
            np.mean(np.abs(mass_error[meaningful]) / neutral[meaningful])
        )
        if meaningful.any()
        else None,
    }


def _support(
    features: np.ndarray, names: list[str], rows: list[dict[str, Any]], eta: np.ndarray
) -> dict[str, Any]:
    train_mask = np.array([row["split"] == "train" for row in rows])
    original = features[train_mask]
    # Fixed geometry columns are omitted from distances, not from inference.
    meaningful = np.ptp(original, axis=0) > 1e-12
    mean, scale = original[:, meaningful].mean(0), original[:, meaningful].std(0)
    standardized = (features[:, meaningful] - mean) / scale
    train_indices = np.flatnonzero(train_mask)
    distances = np.sqrt(
        np.mean(
            (standardized[:, None, :] - standardized[train_indices][None, :, :]) ** 2,
            axis=2,
        )
    )
    low, high = original[:, meaningful].min(0), original[:, meaningful].max(0)
    exterior = (features[:, meaningful] < low) | (features[:, meaningful] > high)
    per_split = {}
    for split in ("train", "validation", "regression", "external"):
        indices = np.array([i for i, row in enumerate(rows) if row["split"] == split])
        if split == "train":
            nearest_distances = distances[indices].copy()
            for j, i in enumerate(indices):
                nearest_distances[
                    j,
                    np.array(
                        [rows[k]["seed"] == rows[i]["seed"] for k in train_indices]
                    ),
                ] = np.inf
        else:
            nearest_distances = distances[indices]
        nearest = train_indices[np.argmin(nearest_distances, axis=1)]
        per_split[split] = {
            "nearest_training_different_seed_distance": _stats(
                nearest_distances.min(1)
            ),
            "any_feature_outside_training_range": int(exterior[indices].any(1).sum()),
            "count": len(indices),
            "nearest_pairs": [
                {
                    "query": {key: rows[i][key] for key in ("seed", "redshift")},
                    "training_neighbor": {
                        key: rows[j][key] for key in ("seed", "redshift")
                    },
                    "standardized_RMS_distance": float(nearest_distances[n].min()),
                    "query_eta": float(eta[i]),
                    "neighbor_eta": float(eta[j]),
                    "absolute_eta_difference": float(abs(eta[i] - eta[j])),
                    "out_of_range_features": [
                        name
                        for name, flag in zip(
                            np.array(names)[meaningful], exterior[i], strict=True
                        )
                        if flag
                    ],
                }
                for n, (i, j) in enumerate(zip(indices, nearest, strict=True))
            ],
        }
    pairs = []
    for i, j in itertools.combinations(range(len(rows)), 2):
        if rows[i]["seed"] == rows[j]["seed"]:
            continue
        distance = float(np.sqrt(np.mean((standardized[i] - standardized[j]) ** 2)))
        if abs(eta[i] - eta[j]) >= 0.01:
            pairs.append(
                {
                    "a": {key: rows[i][key] for key in ("seed", "redshift", "split")},
                    "b": {key: rows[j][key] for key in ("seed", "redshift", "split")},
                    "standardized_RMS_distance": distance,
                    "eta_a": float(eta[i]),
                    "eta_b": float(eta[j]),
                    "absolute_eta_difference": float(abs(eta[i] - eta[j])),
                }
            )
    return {
        "distance": "RMS Euclidean difference standardized using training snapshots only",
        "features_used": np.array(names)[meaningful].tolist(),
        "constant_features_omitted_from_distance": np.array(names)[
            ~meaningful
        ].tolist(),
        "training_mean": mean.tolist(),
        "training_std": scale.tolist(),
        "by_split": per_split,
        "closest_different_seed_pairs_with_eta_difference_at_least_0p01": sorted(
            pairs, key=lambda pair: pair["standardized_RMS_distance"]
        )[:12],
        "interpretation": "Finite-sample proximity is evidence of limited coverage or feature resolution; it does not prove that the feature map is non-identifiable.",
    }


def _curve(history: list[dict[str, Any]], model: MLPModel) -> dict[str, Any]:
    return {
        "metadata": model.metadata,
        "first_epoch": history[0],
        "last_epoch": history[-1],
        "best_validation_epoch": min(history, key=lambda row: row["validation_loss"]),
        "last20_validation_loss_change": history[-1]["validation_loss"]
        - history[max(0, len(history) - 21)]["validation_loss"],
        "last20_training_loss_change": history[-1]["training_loss"]
        - history[max(0, len(history) - 21)]["training_loss"],
    }


def _initial_sigmoid(
    features: np.ndarray, targets: np.ndarray, metadata: dict[str, Any]
) -> np.ndarray:
    rng = np.random.default_rng(metadata["seed"])
    widths = (features.shape[1], *metadata["hidden"], 1)
    weights = [
        rng.normal(scale=np.sqrt(2 / (left + right)), size=(left, right))
        for left, right in itertools.pairwise(widths)
    ]
    biases = [np.zeros(width) for width in widths[1:]]
    prior = np.clip(targets.mean(), 1e-4, 1 - 1e-4)
    biases[-1][0] = np.log(prior) - np.log1p(-prior)
    scale = features.std(0)
    scale[scale == 0] = 1
    return MLPModel(weights, biases, features.mean(0), scale, {}).predict(features)


def _loso(
    features: np.ndarray, eta: np.ndarray, mass: np.ndarray, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    records = []
    seeds = np.array([row["seed"] for row in rows])
    for heldout in np.unique(seeds):
        fit, test = seeds != heldout, seeds == heldout
        for output_mode, epochs in itertools.product(
            ("residual", "sigmoid"), (200, 2000)
        ):
            options = {}
            if output_mode == "residual":
                options = {
                    "baseline_train": np.ones(fit.sum()),
                    "baseline_validation": np.ones(fit.sum()),
                }
            model, history = train(
                features[fit],
                eta[fit],
                features[fit],
                eta[fit],
                seed=11,
                hidden=(8,),
                epochs=epochs,
                patience=epochs + 1,
                learning_rate=0.001,
                batch_size=int(fit.sum()),
                min_delta=0,
                sample_weight_train=mass[fit] ** 2,
                sample_weight_validation=mass[fit] ** 2,
                **options,
            )
            prediction = predict_efficiency(model, features[test])
            records.append(
                {
                    "heldout_training_seed": int(heldout),
                    "output_mode": output_mode,
                    "requested_epochs": epochs,
                    "best_training_epoch": model.metadata["best_epoch"],
                    "final_training_loss": history[-1]["training_loss"],
                    "heldout_error": _errors(eta[test], prediction, mass[test]),
                }
            )
    grouped = []
    for mode, epochs in itertools.product(("residual", "sigmoid"), (200, 2000)):
        values = [
            row
            for row in records
            if row["output_mode"] == mode and row["requested_epochs"] == epochs
        ]
        grouped.append(
            {
                "output_mode": mode,
                "epochs": epochs,
                "mean_LOSO_mass_MAE": float(
                    np.mean([v["heldout_error"]["mass_Q_MAE"] for v in values])
                ),
                "mean_LOSO_mass_MSE": float(
                    np.mean([v["heldout_error"]["mass_Q_MSE"] for v in values])
                ),
                "mean_LOSO_relative_neutral_MAE": float(
                    np.mean(
                        [v["heldout_error"]["relative_neutral_MAE"] for v in values]
                    )
                ),
            }
        )
    return {
        "scope": "Diagnostic only. Five original training seeds; fit four, score fifth. Checkpoints selected on fitting data, never held-out fold. Original validation/regression/external labels are never used. No model saved or deployed.",
        "summaries": grouped,
        "folds": records,
    }


def run(training: Path, output: Path, *, probes: bool = False) -> dict[str, Any]:
    """Audit saved budget predictions and write independent diagnostic reports.

    Parameters
    ----------
    training
        Completed emulator training output with frozen dataset identities.
    output
        New destination directory, never an existing training artifact.
    probes
        Run training-only leave-one-seed-out optimizer probes.

    Returns
    -------
    dict
        Compact diagnosis with paths to per-snapshot evidence.
    """
    training = training.resolve()
    output = output.resolve()
    if output.is_relative_to(training):
        raise ValueError("Diagnostic output must be outside training artifacts")
    if output.exists():
        raise FileExistsError(output)
    receipt = _json(training / "completion.json")
    hashes = {name: _digest(training / name) for name in receipt["files"]}
    if hashes != receipt["files"]:
        raise ValueError("Training completion receipt mismatch")
    configuration = _json(training / "identity.json")["configuration"]
    selected = Emulator.load(training / "model")
    models = {"deployed": selected.budget_model}
    models.update(
        {
            path.stem: load_budget(path)
            for path in sorted((training / "budget").glob("*.npz"))
        }
    )
    datasets = [
        load_dataset(configuration[key]) for key in ("input_root", "external_test_root")
    ]
    rows, features, targets, mass_scales = [], [], [], []
    for dataset_index, dataset in enumerate(datasets):
        for sample in dataset.samples:
            if dataset_index:
                split = "external"
            else:
                split = next(
                    (
                        name
                        for name in ("train", "validation")
                        if sample.seed in configuration[f"{name}_seeds"]
                    ),
                    "regression",
                )
            arrays = sample.arrays
            matrix, names = budget_features(
                arrays["source"],
                arrays["capacity"],
                arrays["native_es_xHI"],
                sample.redshift,
                sample.box_len,
                radii_cMpc=selected.radii_cMpc,
            )
            photons = np.sum(arrays["source"], dtype=np.longdouble)
            atoms = np.sum(arrays["capacity"], dtype=np.longdouble)
            eta = absorption_efficiency(
                arrays["source"], arrays["capacity"], arrays["teacher_xHI"]
            )
            mass = float(min(photons, atoms) / atoms)
            rows.append(
                {
                    "seed": sample.seed,
                    "redshift": sample.redshift,
                    "split": split,
                    "teacher_eta": eta,
                    "teacher_deficit": 1 - eta,
                    "photons_per_atom": float(photons / atoms),
                    "teacher_xHI_mass": 1 - mass * eta,
                    "mass_scale": mass,
                }
            )
            features.append(matrix[0])
            targets.append(eta)
            mass_scales.append(mass)
    x, eta, mass = np.array(features), np.array(targets), np.array(mass_scales)
    train_mask = np.array([row["split"] == "train" for row in rows])
    predictions = {name: predict_efficiency(model, x) for name, model in models.items()}
    amplitude = selected.budget_model._forward(
        (x - selected.budget_model.mean) / selected.budget_model.scale
    )[-1][:, 0]
    if selected.budget_model.output_mode != "residual":
        raise ValueError(
            "This diagnostic's plateau audit requires residual deployed budget"
        )
    for index, row in enumerate(rows):
        row.update(
            deployed_residual_amplitude=float(amplitude[index]),
            deployed_zero_gradient_positive_plateau=bool(amplitude[index] > 0),
            predictions={
                name: float(value[index]) for name, value in predictions.items()
            },
        )
    # Recomputed feature and label arrays must match the actual archived fit data.
    with np.load(
        training / "fit_cache/budget_arrays.npz", allow_pickle=False
    ) as cached:
        for split in ("train", "validation"):
            mask = np.array([row["split"] == split for row in rows])
            np.testing.assert_array_equal(x[mask], cached[f"global_{split}"])
            np.testing.assert_array_equal(eta[mask], cached[f"eta_{split}"])
            np.testing.assert_array_equal(mass[mask], cached[f"mass_scale_{split}"])
    by_split, by_redshift = {}, []
    for split in ("train", "validation", "regression", "external"):
        mask = np.array([row["split"] == split for row in rows])
        plateau = mask & (amplitude > 0)
        significant = plateau & ((1 - eta) > 1e-5)
        by_split[split] = {
            "count": int(mask.sum()),
            "teacher_eta": _stats(eta[mask]),
            "positive_plateau_count": int(plateau.sum()),
            "positive_plateau_with_deficit_gt_1e-5_count": int(significant.sum()),
            "positive_plateau_maximum_missing_mass_deficit": float(
                np.max(mass[plateau] * (1 - eta[plateau]))
            )
            if plateau.any()
            else 0,
            "candidate_errors": {
                name: _errors(eta[mask], value[mask], mass[mask])
                for name, value in predictions.items()
            },
        }
        for redshift in sorted({row["redshift"] for row in rows}, reverse=True):
            group = mask & np.array([row["redshift"] == redshift for row in rows])
            by_redshift.append(
                {
                    "split": split,
                    "redshift": redshift,
                    "count": int(group.sum()),
                    "teacher_deficit": _stats(1 - eta[group]),
                    "deployed_deficit": _stats(1 - predictions["deployed"][group]),
                    "sigmoid_deficit": _stats(1 - predictions["sigmoid_mlp"][group]),
                    "candidate_errors": {
                        name: _errors(eta[group], predictions[name][group], mass[group])
                        for name in ("deployed", "sigmoid_mlp")
                    },
                }
            )
    budget_selection = _json(training / "budget/budget_selection.json")["selection"]
    legacy_history = _json(training / "trials/seed47/legacy_reference/history.json")[
        "budget"
    ]
    curves = {
        "deployed": _curve(legacy_history, models["deployed"]),
        "weighted_residual": _curve(budget_selection["network_history"], models["mlp"]),
        "sigmoid": _curve(
            budget_selection["sigmoid_network_history"], models["sigmoid_mlp"]
        ),
        "sigmoid_actual_initial_training_prediction": _stats(
            _initial_sigmoid(
                x[train_mask], eta[train_mask], models["sigmoid_mlp"].metadata
            )
        ),
        "sigmoid_selected_training_prediction": _stats(
            predictions["sigmoid_mlp"][train_mask]
        ),
        "deployed_loss": "Unweighted absolute efficiency MSE; η is near1 for most rows, not relative neutral-fraction error.",
        "shared_candidate_loss": "Weighted efficiency MSE with weights proportional to mass_scale squared, equivalent to absolute mass-ionized-fraction MSE; not relative remaining-neutral error.",
        "scope": "A best checkpoint at the requested final epoch does not establish optimizer convergence.",
    }
    selection_rows = []
    for path in sorted((training / "trials").glob("seed*/trial.json")):
        trial = _json(path)
        selection_rows.append(
            {
                "model_seed": trial["model_seed"],
                "selected_kind": trial["selected_kind"],
                "validation_objective": trial["validation_objective"],
                "base": {
                    key: value
                    for key, value in trial["base_selection"].items()
                    if key != "snapshots"
                },
                "fixed_field_candidates": [
                    {
                        key: value
                        for key, value in candidate.items()
                        if key != "snapshots"
                    }
                    for candidate in trial["candidates"]
                ],
            }
        )
    support = _support(x, names, rows, eta)
    probe_report = (
        _loso(
            x[train_mask],
            eta[train_mask],
            mass[train_mask],
            [row for row in rows if row["split"] == "train"],
        )
        if probes
        else None
    )
    if hashes != {name: _digest(training / name) for name in hashes}:
        raise ValueError("Training artifact changed during read-only diagnosis")
    output.mkdir(parents=True)
    _write(output / "snapshot_budgets.json", rows)
    _write(output / "feature_support.json", support)
    _write(output / "training_curves.json", curves)
    _write(output / "selection.json", selection_rows)
    if probe_report is not None:
        _write(output / "training_only_LOSO_probes.json", probe_report)
    result = {
        "complete": True,
        "training_artifacts_unchanged": True,
        "training_receipt_files_verified": len(hashes),
        "snapshot_count": len(rows),
        "training_root": str(training),
        "script_sha256": _digest(Path(__file__)),
        "plateau_identity": "η=1+min(tanh(logit),0). Positive residual amplitude gives exactly η=1 and dη/dlogit=0. At zero, the training code uses the feasible one-sided derivative.",
        "by_split": by_split,
        "by_redshift": by_redshift,
        "budget_candidate_validation_ranking": budget_selection["ranking"],
        "sigmoid_training": curves["sigmoid"],
        "LOSO_probes": None if probe_report is None else probe_report["summaries"],
        "files": {path.name: _digest(path) for path in sorted(output.glob("*.json"))},
    }
    _write(output / "report.json", result)
    return result


def main() -> None:
    """Parse paths and run an isolated saved-budget diagnostic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probes", action="store_true")
    arguments = parser.parse_args()
    result = run(arguments.training, arguments.output, probes=arguments.probes)
    sys.stdout.write(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "complete",
                    "snapshot_count",
                    "training_artifacts_unchanged",
                    "LOSO_probes",
                )
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
