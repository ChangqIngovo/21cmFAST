"""Train an ES-residual field surrogate on audited coeval analytic teachers.

Independent simulation seeds define the science split. A second small network
predicts the absorbed share of the input photon budget, and a deterministic
global constraint enforces that predicted count. Test labels never participate
in normalization, fitting, checkpoint choice or trial selection.
"""

from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import logging
import multiprocessing
import os
import shutil
import time
import uuid
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from research.pc_analytic_emulator.data import (
    TeacherSample,
    load_dataset,
    split_samples,
)
from research.pc_analytic_emulator.evaluation import (
    evaluate_prediction,
    native_brightness,
)
from research.pc_analytic_emulator.model import (
    Emulator,
    absorption_efficiency,
    budget_features,
    field_features,
)
from research.pc_field_emulator.evaluation import field_metrics, shell_spectra
from research.pc_field_emulator.model import MLPModel
from research.pc_field_emulator.model import train as fit_network

LOGGER = logging.getLogger(__name__)
DEFAULTS = {
    "train_seeds": list(range(1001, 1007)),
    "validation_seeds": [1007],
    "test_seeds": [1008],
    "model_seeds": [11, 23, 47, 83],
    "workers": 4,
    "threads": 4,
    "cpus": 16,
    "epochs": 200,
    "samples_per_snapshot": 8192,
    "hidden": [32, 32],
    "budget_hidden": [8],
    "batch_size": 2048,
    "patience": 20,
    "learning_rate": 1e-3,
    "radii_cMpc": [6.25, 12.5, 25.0, 50.0],
    "sampling_seed": 2026,
    "training_method": "legacy",
    "v2_epochs": 80,
    "spectral_weight": 0.1,
    "spectral_power_floor": 1e-4,
    "front_weight": 1.0,
    "raw_weight": 0.05,
    "field_variance_floor": 0.01,
    "k_fraction": 0.5,
    "regression_test_seeds": [1008],
    "external_test_root": None,
}
SELECTION_SCOPE = "Full-grid constrained validation MSE; test labels are excluded"
V2_SELECTION_SCOPE = "Full-grid projected multiscale validation objective; auxiliary raw loss and test labels are excluded"


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize training settings and reject overlapping simulation splits.

    Parameters
    ----------
    config
        Teacher input path, split seeds, trial seeds and fitting/resources controls.

    Returns
    -------
    dict
        Independent validated settings. No dataset or model is modified.
    """
    if not isinstance(config, dict) or set(config) - (set(DEFAULTS) | {"input_root"}):
        raise ValueError("Unknown analytic-emulator configuration keys")
    result = copy.deepcopy(DEFAULTS)
    result.update(copy.deepcopy(config))
    if not isinstance(result.get("input_root"), (str, Path)):
        raise TypeError("input_root must identify the completed teacher campaign")
    root = Path(result["input_root"]).resolve()
    if not root.is_dir():
        raise ValueError("Teacher input_root does not exist")
    result["input_root"] = str(root)
    for key in (
        "workers",
        "threads",
        "cpus",
        "epochs",
        "samples_per_snapshot",
        "batch_size",
        "patience",
        "v2_epochs",
    ):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if result["workers"] * result["threads"] > result["cpus"]:
        raise ValueError("workers times threads exceeds the declared CPU allocation")
    for key in ("train_seeds", "validation_seeds", "test_seeds", "model_seeds"):
        values = result[key]
        if (
            not isinstance(values, list)
            or not values
            or any(type(value) is not int or value < 0 for value in values)
            or len(values) != len(set(values))
        ):
            raise ValueError(f"{key} must contain unique nonnegative integers")
    groups = [
        set(result[key]) for key in ("train_seeds", "validation_seeds", "test_seeds")
    ]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError(
            "Simulation seeds may not overlap training, validation and test"
        )
    if type(result["sampling_seed"]) is not int or result["sampling_seed"] < 0:
        raise ValueError("sampling_seed must be a nonnegative integer")
    _validate_extended_controls(result)
    for key in ("hidden", "budget_hidden"):
        if not isinstance(result[key], list) or any(
            type(width) is not int or width < 1 for width in result[key]
        ):
            raise ValueError(f"{key} must be a list of positive layer widths")
    learning_rate = result["learning_rate"]
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not np.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("learning_rate must be finite and positive")
    radii = result["radii_cMpc"]
    if (
        not isinstance(radii, list)
        or not radii
        or any(
            isinstance(radius, bool)
            or not isinstance(radius, (int, float))
            or not np.isfinite(radius)
            or radius <= 0
            for radius in radii
        )
        or len(radii) != len(set(radii))
    ):
        raise ValueError("radii_cMpc must contain unique finite positive widths")
    return result


def _validate_extended_controls(result: dict[str, Any]) -> None:
    if result["training_method"] not in {"legacy", "projected_multiscale_v2"}:
        raise ValueError("Unknown training_method")
    regression = result["regression_test_seeds"]
    if (
        not isinstance(regression, list)
        or any(type(seed) is not int or seed < 0 for seed in regression)
        or len(regression) != len(set(regression))
    ):
        raise ValueError("regression_test_seeds requires unique nonnegative seeds")
    if result["training_method"] == "projected_multiscale_v2":
        result["regression_test_seeds"] = sorted(set(regression) | {1008})
    for key in (
        "spectral_weight",
        "spectral_power_floor",
        "front_weight",
        "raw_weight",
        "field_variance_floor",
        "k_fraction",
    ):
        value = result[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{key} must be finite and nonnegative")
    if result["field_variance_floor"] == 0 or not 0 < result["k_fraction"] <= 1:
        raise ValueError(
            "field_variance_floor must be positive and 0 < k_fraction <= 1"
        )
    external = result["external_test_root"]
    if external is not None:
        if not isinstance(external, (str, Path)):
            raise TypeError("external_test_root must be a directory path or null")
        external = Path(external).resolve()
        if not external.is_dir():
            raise ValueError("external_test_root does not exist")
        result["external_test_root"] = str(external)


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _freeze(path: Path, value: dict[str, Any]) -> None:
    if path.is_file():
        if json.loads(path.read_text()) != value:
            raise ValueError(
                "Training identity changed; prepare a fresh output directory"
            )
    else:
        _write_json(path, value)


def _implementation_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    names = [
        "research/pc_analytic_emulator/__init__.py",
        "research/pc_analytic_emulator/data.py",
        "research/pc_analytic_emulator/model.py",
        "research/pc_analytic_emulator/evaluation.py",
        "research/pc_analytic_emulator/train.py",
        "research/pc_analytic_emulator/fit.py",
        "research/pc_analytic_emulator/objectives.py",
        "research/pc_analytic_emulator/projection.py",
        "research/pc_analytic_emulator/budget.py",
        "research/pc_analytic_emulator/latent.py",
        "research/pc_analytic_emulator/prepare.py",
        "research/pc_field_emulator/model.py",
        "research/pc_field_emulator/features.py",
        "research/pc_field_emulator/evaluation.py",
    ]
    return {name: _digest(root / name) for name in names}


def _sampling_indices(
    seed: int, redshift: float, size: int, count: int, sampling_seed: int
) -> np.ndarray:
    if count >= size:
        return np.arange(size)
    text = f"{sampling_seed}:{seed}:{float(redshift).hex()}".encode()
    value = int.from_bytes(hashlib.sha256(text).digest()[:8], "little")
    return np.sort(np.random.default_rng(value).choice(size, count, replace=False))


def _key(sample: TeacherSample) -> str:
    return f"seed{sample.seed}_z{sample.redshift:.12g}".replace(".", "p")


def _sample_record(sample: TeacherSample) -> dict[str, Any]:
    return {
        "seed": sample.seed,
        "redshift": sample.redshift,
        "dimension": sample.dimension,
        "box_len": sample.box_len,
        "fields_path": str(sample.fields_path),
        "native_directory": str(sample.native_directory),
    }


def _input_arguments(
    sample: TeacherSample,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    return (
        sample.arrays["source"],
        sample.arrays["capacity"],
        sample.arrays["native_es_xHI"],
        sample.redshift,
        sample.box_len,
    )


def _archive_partial(folder: Path) -> None:
    if folder.exists():
        destination = folder.parent / "incomplete"
        destination.mkdir(exist_ok=True)
        folder.rename(destination / f"{folder.name}_{uuid.uuid4().hex[:12]}")
    folder.mkdir(parents=True)


def _verified_record(folder: Path, identity: dict[str, Any]) -> dict[str, Any] | None:
    path = folder / "receipt.json"
    if not path.is_file():
        return None
    receipt = json.loads(path.read_text())
    if receipt["identity"] != identity:
        raise ValueError(f"Checkpoint identity changed: {folder}")
    for name, expected in receipt["files"].items():
        file = (folder / name).resolve()
        if (
            not file.is_relative_to(folder.resolve())
            or not file.is_file()
            or _digest(file) != expected
        ):
            raise ValueError(f"Checkpoint checksum failed: {file}")
    return receipt["record"]


def _seal(folder: Path, identity: dict[str, Any], record: dict[str, Any]) -> None:
    files = {
        str(path.relative_to(folder)): _digest(path)
        for path in sorted(folder.rglob("*"))
        if path.is_file()
        and path.name != "receipt.json"
        and not path.name.endswith(".tmp")
    }
    _write_json(
        folder / "receipt.json",
        {"identity": identity, "files": files, "record": record},
    )


def _prepare_cache(
    folder: Path,
    splits: dict[str, tuple[TeacherSample, ...]],
    config: dict[str, Any],
    identity: dict[str, Any],
) -> dict[str, Any]:
    cached = _verified_record(folder, identity)
    if cached is not None:
        return cached
    _archive_partial(folder)
    arrays: dict[str, np.ndarray] = {}
    feature_names, global_names = None, None
    validation_snapshots = []
    for split in ("train", "validation"):
        matrices, targets, baselines, global_inputs, efficiencies = [], [], [], [], []
        for sample in splits[split]:
            args = _input_arguments(sample)
            features, names, baseline = field_features(
                *args, radii_cMpc=config["radii_cMpc"]
            )
            globals_, global_columns = budget_features(
                *args, radii_cMpc=config["radii_cMpc"]
            )
            if feature_names is not None and (
                names != feature_names or global_columns != global_names
            ):
                raise ValueError("Features differ across teacher snapshots")
            feature_names, global_names = names, global_columns
            target = 1.0 - sample.arrays["teacher_xHI"].reshape(-1)
            indices = (
                _sampling_indices(
                    sample.seed,
                    sample.redshift,
                    len(target),
                    config["samples_per_snapshot"],
                    config["sampling_seed"],
                )
                if split == "train"
                else np.arange(len(target))
            )
            matrices.append(features[indices])
            targets.append(target[indices])
            baselines.append(baseline[indices])
            global_inputs.append(globals_)
            efficiencies.append(
                absorption_efficiency(
                    sample.arrays["source"],
                    sample.arrays["capacity"],
                    sample.arrays["teacher_xHI"],
                )
            )
            if split == "validation":
                filename = f"validation_{_key(sample)}.npz"
                np.savez_compressed(
                    folder / filename,
                    **{
                        name: sample.arrays[name]
                        for name in (
                            "source",
                            "capacity",
                            "native_es_xHI",
                            "teacher_xHI",
                        )
                    },
                )
                validation_snapshots.append(
                    _sample_record(sample) | {"cache": filename}
                )
        arrays[f"x_{split}"] = np.concatenate(matrices)
        arrays[f"y_{split}"] = np.concatenate(targets)
        arrays[f"baseline_{split}"] = np.concatenate(baselines)
        arrays[f"global_{split}"] = np.concatenate(global_inputs)
        arrays[f"eta_{split}"] = np.asarray(efficiencies)
    np.savez_compressed(folder / "fit_arrays.npz", **arrays)
    record = {
        "fit_arrays": str(folder / "fit_arrays.npz"),
        "validation_snapshots": validation_snapshots,
        "field_feature_names": feature_names,
        "budget_feature_names": global_names,
        "training_cells": len(arrays["y_train"]),
        "validation_cells": len(arrays["y_validation"]),
        "test_labels_in_fit_cache": False,
    }
    _write_json(folder / "features.json", record)
    _seal(folder, identity, record)
    return record


def _prepare_v2_cache(
    folder: Path,
    splits: dict[str, tuple[TeacherSample, ...]],
    config: dict[str, Any],
    identity: dict[str, Any],
) -> dict[str, Any]:
    cached = _verified_record(folder, identity)
    if cached is not None:
        return cached
    _archive_partial(folder)
    matrices: dict[str, np.ndarray] = {}
    rows: dict[str, list[dict[str, Any]]] = {}
    cells: dict[str, int] = {}
    feature_names, global_names = None, None
    for split in ("train", "validation"):
        rows[split], cells[split] = [], 0
        global_inputs, efficiencies, mass_scales = [], [], []
        for sample in splits[split]:
            arguments = _input_arguments(sample)
            features, names, baseline = field_features(
                *arguments, radii_cMpc=config["radii_cMpc"]
            )
            globals_, global_columns = budget_features(
                *arguments, radii_cMpc=config["radii_cMpc"]
            )
            if feature_names is not None and (
                names != feature_names or global_columns != global_names
            ):
                raise ValueError("Features differ across teacher snapshots")
            feature_names, global_names = names, global_columns
            filename = folder / f"{split}_{_key(sample)}.npz"
            np.savez_compressed(
                filename,
                features=features,
                baseline=baseline,
                global_features=globals_,
                box_len=np.asarray(sample.box_len),
                **{
                    name: sample.arrays[name]
                    for name in ("source", "capacity", "native_es_xHI", "teacher_xHI")
                },
            )
            rows[split].append(_sample_record(sample) | {"cache": str(filename)})
            cells[split] += len(baseline)
            source = np.sum(sample.arrays["source"], dtype=np.longdouble)
            capacity = np.sum(sample.arrays["capacity"], dtype=np.longdouble)
            mass_scales.append(float(min(source, capacity) / capacity))
            efficiencies.append(
                absorption_efficiency(
                    sample.arrays["source"],
                    sample.arrays["capacity"],
                    sample.arrays["teacher_xHI"],
                )
            )
            global_inputs.append(globals_)
        matrices[f"global_{split}"] = np.concatenate(global_inputs)
        matrices[f"eta_{split}"] = np.asarray(efficiencies)
        matrices[f"mass_scale_{split}"] = np.asarray(mass_scales)
    np.savez_compressed(folder / "budget_arrays.npz", **matrices)
    record = {
        "training_method": "projected_multiscale_v2",
        "training_snapshots": rows["train"],
        "validation_snapshots": rows["validation"],
        "field_feature_names": feature_names,
        "budget_feature_names": global_names,
        "training_cells": cells["train"],
        "validation_cells": cells["validation"],
        "sampling": "All cells of every training snapshot; one snapshot per gradient update",
        "test_labels_in_fit_cache": False,
    }
    _write_json(folder / "features.json", record)
    _seal(folder, identity, record)
    return record


def _prepare_v2_budget(
    folder: Path, cache: Path, config: dict[str, Any], identity: dict[str, Any]
) -> dict[str, Any]:
    from research.pc_analytic_emulator.budget import fit_budget

    cached = _verified_record(folder, identity)
    if cached is not None:
        return cached
    _archive_partial(folder)
    with np.load(cache / "budget_arrays.npz", allow_pickle=False) as arrays:
        model, report = fit_budget(
            arrays["global_train"],
            arrays["eta_train"],
            arrays["mass_scale_train"],
            arrays["global_validation"],
            arrays["eta_validation"],
            arrays["mass_scale_validation"],
            seed=config["model_seeds"][0],
            epochs=config["epochs"],
            patience=config["patience"],
            learning_rate=config["learning_rate"],
            hidden=tuple(config["budget_hidden"]),
        )
    model.save(folder / "budget_model.npz")
    record = {
        "model_path": str(folder / "budget_model.npz"),
        "selection": report,
        "initialization_seed": config["model_seeds"][0],
        "selected_before_field_training": True,
        "test_used": False,
    }
    _write_json(folder / "budget_selection.json", record)
    _seal(folder, identity, record)
    return record


def _zero_model(width: int, role: str) -> MLPModel:
    return MLPModel(
        weights=[np.zeros((width, 1))],
        biases=[np.zeros(1)],
        mean=np.zeros(width),
        scale=np.ones(width),
        metadata={
            "role": role,
            "best_epoch": 0,
            "baseline_only": True,
            "standardizer_scope": "fixed identity; no label fitting",
        },
        output_mode="residual",
    )


def _validation_score(
    model: Emulator, cache: Path, rows: list[dict[str, Any]]
) -> tuple[float, list[dict[str, Any]]]:
    result = []
    for row in rows:
        with np.load(cache / row["cache"], allow_pickle=False) as data:
            prediction = model.predict(
                data["source"],
                data["capacity"],
                data["native_es_xHI"],
                row["redshift"],
                row["box_len"],
            )
            error = prediction["xHI"] - data["teacher_xHI"]
            result.append(
                {
                    "seed": row["seed"],
                    "redshift": row["redshift"],
                    "mse": float(np.mean(error**2)),
                }
            )
    return float(np.mean([row["mse"] for row in result])), result


def _fit_legacy_trial(task: dict[str, Any]) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config, identity = task["config"], task["identity"]
    seed = task["model_seed"]
    folder, cache = Path(task["folder"]), Path(task["cache"])
    trial_identity = {"training": identity, "model_seed": seed}
    old = _verified_record(folder, trial_identity)
    if old is not None:
        LOGGER.info("Training trial resumed and verified: initialization=%d", seed)
        return old
    _archive_partial(folder)
    LOGGER.info("Training trial started: initialization=%d", seed)
    started = time.perf_counter()
    with np.load(cache / "fit_arrays.npz", allow_pickle=False) as saved:
        arrays = {name: np.array(saved[name]) for name in saved.files}
    details = json.loads((cache / "features.json").read_text())

    def callback(role: str):
        def log(row: dict[str, Any]) -> None:
            LOGGER.info(
                "Trial %d %s epoch=%d train_mse=%.6g validation_mse=%.6g",
                seed,
                role,
                row["epoch"],
                row["training_mse"],
                row["validation_mse"],
            )

        return log

    field, field_history = fit_network(
        arrays["x_train"],
        arrays["y_train"],
        arrays["x_validation"],
        arrays["y_validation"],
        seed=seed,
        hidden=tuple(config["hidden"]),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        patience=config["patience"],
        learning_rate=config["learning_rate"],
        min_delta=1e-9,
        baseline_train=arrays["baseline_train"],
        baseline_validation=arrays["baseline_validation"],
        on_epoch=callback("field"),
    )
    budget, budget_history = fit_network(
        arrays["global_train"],
        arrays["eta_train"],
        arrays["global_validation"],
        arrays["eta_validation"],
        seed=seed,
        hidden=tuple(config["budget_hidden"]),
        epochs=config["epochs"],
        batch_size=max(len(arrays["eta_train"]), 1),
        patience=config["patience"],
        learning_rate=config["learning_rate"],
        min_delta=1e-12,
        baseline_train=np.ones(len(arrays["eta_train"])),
        baseline_validation=np.ones(len(arrays["eta_validation"])),
        on_epoch=callback("budget"),
    )
    metadata = {
        "field_feature_names": details["field_feature_names"],
        "budget_feature_names": details["budget_feature_names"],
        "model_seed": seed,
        "field_checkpoint_selection": "Raw full-grid validation MSE",
        "budget_checkpoint_selection": "Validation absorption-efficiency MSE",
    }
    trained = Emulator(
        field, budget, radii_cMpc=config["radii_cMpc"], metadata=metadata
    )
    baseline = Emulator(
        _zero_model(arrays["x_train"].shape[1], "ES field baseline"),
        _zero_model(arrays["global_train"].shape[1], "eta=1 baseline"),
        radii_cMpc=config["radii_cMpc"],
        metadata=metadata | {"baseline_only": True},
    )
    candidates = []
    for name, model in (("trained", trained), ("zero_correction", baseline)):
        mse, rows = _validation_score(model, cache, details["validation_snapshots"])
        candidates.append({"kind": name, "validation_mse": mse, "snapshots": rows})
    selected = min(
        candidates,
        key=lambda row: (row["validation_mse"], row["kind"] != "zero_correction"),
    )
    trained.save(folder / "trained_model")
    chosen = trained if selected["kind"] == "trained" else baseline
    chosen.metadata.update(
        selected_kind=selected["kind"], selection_scope=SELECTION_SCOPE
    )
    chosen.save(folder / "model")
    _write_json(
        folder / "history.json", {"field": field_history, "budget": budget_history}
    )
    record = {
        "model_seed": seed,
        "selected_kind": selected["kind"],
        "validation_mse": selected["validation_mse"],
        "candidates": candidates,
        "model_directory": str(folder / "model"),
        "elapsed_seconds": time.perf_counter() - started,
        "field_best_epoch": field.metadata["best_epoch"],
        "budget_best_epoch": budget.metadata["best_epoch"],
        "field_completed_epochs": field.metadata["completed_epochs"],
        "budget_completed_epochs": budget.metadata["completed_epochs"],
        "test_used": False,
    }
    _write_json(folder / "trial.json", record)
    _seal(folder, trial_identity, record)
    LOGGER.info(
        "Training trial completed: initialization=%d selected=%s validation_mse=%.6g",
        seed,
        selected["kind"],
        record["validation_mse"],
    )
    return record


def _fit_trial(task: dict[str, Any]) -> dict[str, Any]:
    if task["config"]["training_method"] == "projected_multiscale_v2":
        return _fit_v2_trial(task)
    return _fit_legacy_trial(task)


def _score_emulator(
    model: Emulator, rows: Sequence[dict[str, Any]], objective: dict[str, Any]
) -> tuple[float, list[dict[str, Any]]]:
    from research.pc_analytic_emulator.objectives import projected_objective

    diagnostics = []
    for row in rows:
        with np.load(row["cache"], allow_pickle=False) as frame:
            prediction = model.predict(
                frame["source"],
                frame["capacity"],
                frame["native_es_xHI"],
                row["redshift"],
                row["box_len"],
            )
            truth = 1 - frame["teacher_xHI"]
            loss, _, components = projected_objective(
                prediction["xHII"],
                truth,
                frame["capacity"],
                row["box_len"],
                **{
                    key: value
                    for key, value in objective.items()
                    if key != "raw_weight"
                },
            )
            diagnostics.append(
                {
                    "seed": row["seed"],
                    "redshift": row["redshift"],
                    "projected_loss": loss,
                    "field_mse": float(np.mean((prediction["xHII"] - truth) ** 2)),
                    "objective": components,
                    "projection": prediction["projection"],
                }
            )
    return float(np.mean([row["projected_loss"] for row in diagnostics])), diagnostics


def _fit_v2_trial(task: dict[str, Any]) -> dict[str, Any]:
    from research.pc_analytic_emulator.budget import (
        constant_efficiency_model,
        load_budget,
    )
    from research.pc_analytic_emulator.fit import fit_field, score_field
    from research.pc_analytic_emulator.latent import LatentFieldModel

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config, identity, seed = task["config"], task["identity"], task["model_seed"]
    folder, cache = Path(task["folder"]), Path(task["cache"])
    trial_identity = {"training": identity, "model_seed": seed}
    old = _verified_record(folder, trial_identity)
    if old is not None:
        LOGGER.info("Projected trial resumed and verified: initialization=%d", seed)
        return old
    restored_legacy = None
    legacy_folder = folder / "legacy_reference"
    if (
        legacy_folder.is_dir()
        and _verified_record(legacy_folder, trial_identity) is not None
    ):
        restored_legacy = folder.parent / f"legacy_restore_{uuid.uuid4().hex}"
        legacy_folder.rename(restored_legacy)
    _archive_partial(folder)
    if restored_legacy is not None:
        restored_legacy.rename(legacy_folder)
    started = time.perf_counter()
    details = json.loads((cache / "features.json").read_text())
    budget = load_budget(task["budget_model_path"])
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

    def log(row: dict[str, Any]) -> None:
        LOGGER.info(
            "Projected trial %d epoch metrics: %s",
            seed,
            json.dumps(row, sort_keys=True),
        )

    field, history = fit_field(
        details["training_snapshots"],
        details["validation_snapshots"],
        budget,
        seed=seed,
        hidden=tuple(config["hidden"]),
        epochs=config["v2_epochs"],
        learning_rate=config["learning_rate"],
        patience=config["patience"],
        objective_config=objective,
        on_epoch=log,
    )
    width = len(details["field_feature_names"])
    zero = LatentFieldModel(
        [np.zeros((width, 1))],
        [np.zeros(1)],
        np.zeros(width),
        np.ones(width),
        {
            "role": "ES zero-correction reference",
            "best_epoch": 0,
            "baseline_only": True,
        },
    )
    unity = constant_efficiency_model(len(details["budget_feature_names"]))
    metadata = {
        "training_method": config["training_method"],
        "field_feature_names": details["field_feature_names"],
        "budget_feature_names": details["budget_feature_names"],
        "model_seed": seed,
        "field_checkpoint_selection": V2_SELECTION_SCOPE,
        "budget_checkpoint_selection": "Validation mass-absorption-fraction MSE before field training",
        "objective": objective,
    }
    candidates, candidate_models = [], {}
    for kind, field_model, budget_model in (
        ("trained_selected_budget", field, budget),
        ("trained_eta_one", field, unity),
        ("es_selected_budget", zero, budget),
        ("es_eta_one", zero, unity),
    ):
        model = Emulator(
            field_model,
            budget_model,
            radii_cMpc=config["radii_cMpc"],
            metadata=metadata,
            projection_mode="capacity_l2",
        )
        score, rows = score_field(
            field_model,
            details["validation_snapshots"],
            budget_model,
            objective_config=objective,
        )
        candidates.append(
            {
                "kind": kind,
                "validation_objective": score,
                "validation_mse": float(np.mean([row["field_mse"] for row in rows])),
                "snapshots": rows,
            }
        )
        candidate_models[kind] = model
    legacy_task = dict(task, folder=str(legacy_folder), cache=task["legacy_cache"])
    legacy_record = _fit_legacy_trial(legacy_task)
    for kind, path in (
        ("legacy_reference", Path(legacy_record["model_directory"])),
        ("legacy_trained", legacy_folder / "trained_model"),
    ):
        model = Emulator.load(path)
        score, rows = _score_emulator(model, details["validation_snapshots"], objective)
        candidates.append(
            {
                "kind": kind,
                "validation_objective": score,
                "validation_mse": float(np.mean([row["field_mse"] for row in rows])),
                "snapshots": rows,
            }
        )
        candidate_models[kind] = model
    selected = min(
        candidates,
        key=lambda row: (
            row["validation_objective"],
            row["kind"].startswith("trained"),
            row["kind"],
        ),
    )
    candidate_models["trained_selected_budget"].save(folder / "trained_model")
    chosen = candidate_models[selected["kind"]]
    chosen.metadata = dict(chosen.metadata) | {
        "selected_kind": selected["kind"],
        "selection_scope": V2_SELECTION_SCOPE,
        "selection_training_method": config["training_method"],
    }
    chosen.save(folder / "model")
    _write_json(
        folder / "history.json",
        {"field": history, "budget": "Selected and frozen before all field trials"},
    )
    record = {
        "model_seed": seed,
        "selected_kind": selected["kind"],
        "validation_objective": selected["validation_objective"],
        "validation_mse": selected["validation_mse"],
        "candidates": candidates,
        "model_directory": str(folder / "model"),
        "elapsed_seconds": time.perf_counter() - started,
        "field_best_epoch": field.metadata["best_epoch"],
        "field_completed_epochs": field.metadata["completed_epochs"],
        "budget_model_path": task["budget_model_path"],
        "legacy_reference": legacy_record,
        "test_used": False,
    }
    _write_json(folder / "trial.json", record)
    _seal(folder, trial_identity, record)
    return record


def _choose_trial(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric = (
        "validation_objective"
        if rows and "validation_objective" in rows[0]
        else "validation_mse"
    )
    if not rows or any(not np.isfinite(row[metric]) for row in rows):
        raise ValueError("Trial selection requires finite validation scores")
    return min(rows, key=lambda row: (row[metric], row["model_seed"]))


@contextmanager
def _thread_environment(threads: int) -> Iterator[None]:
    values = {
        "OPENBLAS_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OMP_NUM_THREADS": str(threads),
        "OMP_THREAD_LIMIT": str(threads),
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_DYNAMIC": "FALSE",
        "OMP_WAIT_POLICY": "PASSIVE",
        "PYTHONDONTWRITEBYTECODE": "1",
        "MPLBACKEND": "Agg",
    }
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, old in previous.items():
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old


def _verify_completion(output: Path) -> dict[str, Any]:
    record = json.loads((output / "completion.json").read_text())
    for name, expected in record["files"].items():
        path = (output / name).resolve()
        if (
            not path.is_relative_to(output.resolve())
            or not path.is_file()
            or _digest(path) != expected
        ):
            raise ValueError(f"Completed training checksum failed: {path}")
    return json.loads((output / "summary.json").read_text())


def _write_figures(
    output: Path,
    samples: Sequence[TeacherSample],
    model: Emulator,
    rows: list[dict[str, Any]],
    evaluation_label: str = "Held-out",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chosen = sorted(samples, key=lambda sample: (sample.seed, -sample.redshift))
    indices = sorted({0, len(chosen) // 2, len(chosen) - 1})
    selected = [chosen[index] for index in indices]
    figure, axes = plt.subplots(
        len(selected),
        4,
        figsize=(15, 3.7 * len(selected)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_axes, sample in zip(axes, selected, strict=True):
        prediction = model.predict(*_input_arguments(sample))
        arrays = sample.arrays
        cut = sample.dimension // 2
        images = (
            arrays["native_es_xHI"],
            arrays["teacher_xHI"],
            prediction["xHI"],
            prediction["xHI"] - arrays["teacher_xHI"],
        )
        for index, (axis, values, title) in enumerate(
            zip(
                row_axes,
                images,
                (
                    "Native ES",
                    "Analytic teacher",
                    "Constrained emulator",
                    "Emulator - teacher",
                ),
                strict=True,
            )
        ):
            limit = (
                1.0
                if index < 3
                else max(float(np.max(np.abs(values[:, :, cut]))), 1e-3)
            )
            image = axis.imshow(
                values[:, :, cut].T,
                origin="lower",
                extent=(0, sample.box_len, 0, sample.box_len),
                cmap="gray" if index < 3 else "RdBu_r",
                vmin=0 if index < 3 else -limit,
                vmax=limit,
            )
            axis.set(
                title=f"{title}\nseed {sample.seed}, z={sample.redshift:g}",
                xlabel="x [cMpc]",
                ylabel="y [cMpc]",
            )
            figure.colorbar(
                image, ax=axis, label="xHI" if index < 3 else "ΔxHI", shrink=0.8
            )
    figure.savefig(output / "test_slices.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(
        1,
        len(selected),
        figsize=(5.2 * len(selected), 4.0),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, sample in zip(axes[0], selected, strict=True):
        prediction = model.predict(*_input_arguments(sample))
        inputs = sample.metadata["input"]["inputs"]
        fields = {
            name: native_brightness(
                xhi, sample.arrays["capacity"], sample.redshift, inputs
            )
            for name, xhi in {
                "teacher": sample.arrays["teacher_xHI"],
                "constrained": prediction["xHI"],
                "raw": prediction["raw_xHI"],
                "native_es": sample.arrays["native_es_xHI"],
            }.items()
        }
        spectra = shell_spectra(fields, sample.box_len)
        for name in ("constrained", "raw", "native_es"):
            axis.plot(
                spectra["k_cMpc_inv"],
                [
                    np.nan if value is None else value
                    for value in spectra[f"ratio_{name}"]
                ],
                label=name,
            )
        if not any(value is not None for value in spectra["ratio_constrained"]):
            axis.text(
                0.5,
                0.5,
                "Power ratio undefined\nTeacher power is numerically zero",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        axis.axhline(1, color="0.5", linewidth=1)
        axis.axvline(
            np.pi * sample.dimension / (2 * sample.box_len),
            color="0.5",
            linestyle=":",
            label="half Nyquist",
        )
        axis.set(
            title=f"{evaluation_label} seed {sample.seed}, z={sample.redshift:g}",
            xlabel="k [cMpc⁻¹]",
            ylabel="21-cm power / teacher",
            xlim=(0, np.pi * sample.dimension / sample.box_len),
        )
        axis.legend(fontsize=8)
    figure.savefig(output / "test_power_ratios.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    redshifts = [row["redshift"] for row in rows]
    for name in ("raw", "constrained", "native_es"):
        axes[0].plot(
            redshifts,
            [row["field_metrics"][name]["MAE"] for row in rows],
            "o-",
            label=name,
        )
        axes[1].plot(
            redshifts,
            [row["field_metrics"][name]["delta_Q_mass"] for row in rows],
            "o-",
            label=name,
        )
    axes[0].set(xlabel="Redshift", ylabel="xHI field MAE")
    axes[1].set(xlabel="Redshift", ylabel="Mass-ionized fraction difference")
    for axis in axes:
        axis.invert_xaxis()
        axis.legend()
    figure.savefig(output / "test_errors.png", dpi=160)
    plt.close(figure)


def _evaluate(
    output: Path,
    samples: Sequence[TeacherSample],
    model: Emulator,
    evaluation_label: str = "Held-out",
) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        started = time.perf_counter()
        prediction = model.predict(*_input_arguments(sample))
        elapsed = time.perf_counter() - started
        report = evaluate_prediction(
            prediction,
            sample.arrays,
            sample.redshift,
            sample.box_len,
            sample.metadata["input"]["inputs"],
        )
        metrics = {
            name: field_metrics(
                1 - values, 1 - sample.arrays["teacher_xHI"], sample.arrays["capacity"]
            )
            for name, values in {
                "raw": prediction["raw_xHI"],
                "constrained": prediction["xHI"],
                "native_es": sample.arrays["native_es_xHI"],
            }.items()
        }
        folder = output / "test" / _key(sample)
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            folder / "predictions.npz",
            raw_xHI=prediction["raw_xHI"],
            emulator_xHI=prediction["xHI"],
            emulator_xHII=prediction["xHII"],
            teacher_xHI=sample.arrays["teacher_xHI"],
            native_es_xHI=sample.arrays["native_es_xHI"],
            predicted_absorbed_photons=np.asarray(
                prediction["predicted_absorbed_photons"]
            ),
            predicted_unabsorbed_photons=np.asarray(
                prediction["predicted_unabsorbed_photons"]
            ),
        )
        row = _sample_record(sample) | {
            "evaluation": report,
            "field_metrics": metrics,
            "photon_ledger": prediction["photon_ledger"],
            "inference_seconds": elapsed,
            "evaluation_role": evaluation_label,
        }
        _write_json(folder / "metrics.json", row)
        rows.append(row)
    with (output / "test_metrics.csv").open("w", newline="") as stream:
        names = [
            "seed",
            "redshift",
            "method",
            "MAE",
            "RMSE",
            "delta_Q_mass",
            "delta_Q_volume",
            "field_correlation",
            "front_MAE",
            "neutral_IoU",
        ]
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        for row in rows:
            for method, metrics in row["field_metrics"].items():
                writer.writerow(
                    {"seed": row["seed"], "redshift": row["redshift"], "method": method}
                    | {name: metrics[name] for name in names[3:]}
                )
    _write_json(output / "test_metrics.json", rows)
    _write_figures(output, samples, model, rows, evaluation_label)
    return rows


def _test_aggregates(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Report equal-snapshot errors without cancelling signed budget biases."""

    def average(selected: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
        if not selected:
            return None
        return {
            name: {
                metric: float(
                    np.mean([row["field_metrics"][name][metric] for row in selected])
                )
                for metric in ("MAE", "RMSE", "delta_Q_mass", "delta_Q_volume")
            }
            for name in ("raw", "constrained", "native_es")
        }

    def errors(selected: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
        if not selected:
            return None
        return {
            name: {
                **{
                    f"{stat}_absolute_{metric}": float(
                        operation(
                            [
                                abs(row["field_metrics"][name][metric])
                                for row in selected
                            ]
                        )
                    )
                    for metric in ("delta_Q_mass", "delta_Q_volume")
                    for stat, operation in (("mean", np.mean), ("maximum", np.max))
                },
                **{
                    f"maximum_snapshot_{metric}": float(
                        max(row["field_metrics"][name][metric] for row in selected)
                    )
                    for metric in ("MAE", "RMSE")
                },
            }
            for name in ("raw", "constrained", "native_es")
        }

    def neutral(row: dict[str, Any]) -> float:
        return 1 - row["evaluation"]["metrics"]["teacher"]["Q_mass"]

    def snapshots(selected: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "seed": row["seed"],
                "redshift": row["redshift"],
                "teacher_xHI_mass": neutral(row),
            }
            for row in selected
        ]

    # Fixed reporting cuts, independent of the test results. The numerical-zero
    # bin prevents fully ionized fields from diluting the late-stage summary.
    phase_definitions = (
        ("early", "teacher xHI_mass > 0.8", lambda x: x > 0.8),
        ("middle", "0.2 < teacher xHI_mass <= 0.8", lambda x: 0.2 < x <= 0.8),
        ("late", "1e-10 < teacher xHI_mass <= 0.2", lambda x: 1e-10 < x <= 0.2),
        ("fully_ionized", "teacher xHI_mass <= 1e-10", lambda x: x <= 1e-10),
    )
    phases = {}
    for name, criterion, contains in phase_definitions:
        selected = [row for row in rows if contains(neutral(row))]
        phases[name] = {
            "criterion": criterion,
            "count": len(selected),
            "snapshots": snapshots(selected),
            "metrics_mean": average(selected),
            "absolute_and_worst_errors": errors(selected),
        }

    active = [
        row
        for row in rows
        if 0.05 < 1 - row["evaluation"]["metrics"]["teacher"]["Q_mass"] < 0.95
    ]
    return {
        "test_metrics_mean_over_snapshots": average(rows),
        "test_metrics_mean_over_active_snapshots": average(active),
        "test_absolute_and_worst_errors": errors(rows),
        "test_metrics_by_reionization_phase": {
            "averaging": "Equal weight per snapshot; empty phases have undefined metrics",
            "phases": phases,
        },
        "test_active_snapshots": {
            "criterion": "0.05 < teacher mass-weighted neutral fraction < 0.95",
            "count": len(active),
            "total_count": len(rows),
            "snapshots": snapshots(active),
            "averaging": "Equal weight per snapshot; undefined when no snapshot is active",
        },
    }


def run(config: dict[str, Any], output: str | Path) -> dict[str, Any]:
    """Fit trials, select on validation and evaluate the held-out simulation seed.

    Parameters
    ----------
    config
        Settings accepted by :func:`validate_config`.
    output
        Fresh result directory or unchanged resumable training directory.

    Returns
    -------
    dict
        Completed split, selection, timing and held-out performance summary.
    """
    config = validate_config(config)
    dataset = load_dataset(config["input_root"])
    improved = config["training_method"] == "projected_multiscale_v2"
    selection_scope = V2_SELECTION_SCOPE if improved else SELECTION_SCOPE
    external = None
    if config["external_test_root"] is not None:
        from research.pc_analytic_emulator.prepare import validate_external_test

        external = load_dataset(config["external_test_root"])
        validate_external_test(dataset, external)
    splits = split_samples(
        dataset,
        train_seeds=config["train_seeds"],
        validation_seeds=config["validation_seeds"],
        test_seeds=config["test_seeds"],
    )
    output = Path(output).resolve()
    if (
        output == dataset.root
        or output.is_relative_to(dataset.root)
        or dataset.root.is_relative_to(output)
    ):
        raise ValueError("Training output must be separate from the teacher archive")
    if external is not None and (
        output == external.root
        or output.is_relative_to(external.root)
        or external.root.is_relative_to(output)
    ):
        raise ValueError("Training output must be separate from the external archive")
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema": 1,
        "configuration": config,
        "teacher_identity": dataset.identity,
        "teacher_file_hashes": dataset.file_hashes,
        "external_test_identity": external.identity if external is not None else None,
        "external_test_file_hashes": external.file_hashes
        if external is not None
        else {},
        "implementation_sha256": _implementation_hashes(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "matplotlib")
        },
    }
    with (output / "training.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _freeze(output / "identity.json", identity)
        if (output / "COMPLETED").is_file():
            LOGGER.info("Completed training verified; no fitting or test re-selection")
            return _verify_completion(output)
        started = time.perf_counter()
        counts = {name: len(samples) for name, samples in splits.items()}
        seed_roles = {
            name: sorted({sample.seed for sample in samples})
            for name, samples in splits.items()
        }
        known_regression = sorted(
            set(config["regression_test_seeds"]) & set(seed_roles["test"])
        )
        test_role = (
            "Regression"
            if known_regression == seed_roles["test"]
            else "Internal evaluation"
            if known_regression
            else "Held-out"
        )
        _write_json(
            output / "dataset_manifest.json",
            {
                "teacher_root": str(dataset.root),
                "teacher_identity": dataset.identity,
                "teacher_file_hashes": dataset.file_hashes,
                "teacher_audit": dataset.audit,
                "physics": dataset.physics,
                "split_seeds": seed_roles,
                "sample_counts": counts,
                "samples": {
                    name: [_sample_record(sample) for sample in samples]
                    for name, samples in splits.items()
                },
                "split_policy": "All redshifts from an initial-condition seed share one split",
                "known_regression_test_seeds": known_regression,
                "internal_test_role": test_role,
                "external_test": None
                if external is None
                else {
                    "root": str(external.root),
                    "identity": external.identity,
                    "file_hashes": external.file_hashes,
                    "seeds": sorted({sample.seed for sample in external.samples}),
                    "sample_count": len(external.samples),
                    "used_for_selection": False,
                },
            },
        )
        _write_json(
            output / "summary.json",
            {"complete": False, "sample_counts": counts, "split_seeds": seed_roles},
        )
        cache = output / "fit_cache"
        details = (
            _prepare_v2_cache(cache, splits, config, identity)
            if improved
            else _prepare_cache(cache, splits, config, identity)
        )
        budget_record = None
        if improved:
            budget_record = _prepare_v2_budget(
                output / "budget", cache, config, identity
            )
            _prepare_cache(output / "legacy_fit_cache", splits, config, identity)
        tasks = [
            {
                "config": config,
                "identity": identity,
                "model_seed": seed,
                "folder": str(output / "trials" / f"seed{seed}"),
                "cache": str(cache),
                "budget_model_path": None
                if budget_record is None
                else budget_record["model_path"],
                "legacy_cache": str(output / "legacy_fit_cache") if improved else None,
            }
            for seed in config["model_seeds"]
        ]
        trials, failures = [], []
        with _thread_environment(config["threads"]):
            if config["workers"] == 1:
                trials.extend(_fit_trial(task) for task in tasks)
            else:
                with ProcessPoolExecutor(
                    max_workers=min(config["workers"], len(tasks)),
                    mp_context=multiprocessing.get_context("spawn"),
                ) as pool:
                    jobs = {
                        pool.submit(_fit_trial, task): task["model_seed"]
                        for task in tasks
                    }
                    for job in as_completed(jobs):
                        try:
                            trials.append(job.result())
                        except Exception as error:
                            LOGGER.exception(
                                "Training initialization %d failed", jobs[job]
                            )
                            failures.append(
                                {"model_seed": jobs[job], "error": str(error)}
                            )
        if failures:
            _write_json(output / "failed_trials.json", failures)
            raise RuntimeError(
                "Training trials failed; verified completed trials remain resumable"
            )
        trials.sort(key=lambda row: row["model_seed"])
        selected = _choose_trial(trials)
        selection = {
            "selection_scope": selection_scope,
            "selected": selected,
            "trials": trials,
            "test_used_for_selection": False,
            "budget_selection": budget_record,
        }
        _write_json(output / "model_selection.json", selection)
        model = Emulator.load(selected["model_directory"])
        model.metadata.update(
            teacher_physics=dataset.physics,
            split_seeds=seed_roles,
            teacher_identity=dataset.identity,
        )
        model.save(output / "model")
        shutil.copyfile(cache / "features.json", output / "features.json")
        rows = _evaluate(output, splits["test"], model, test_role)
        aggregates = _test_aggregates(rows)
        external_summary = None
        if external is not None:
            external_output = output / "external_test"
            external_output.mkdir(exist_ok=True)
            external_rows = _evaluate(
                external_output, external.samples, model, "External test"
            )
            external_summary = {
                "seeds": sorted({sample.seed for sample in external.samples}),
                "sample_count": len(external.samples),
                "used_for_selection": False,
                "maximum_photon_residual": max(
                    row["photon_ledger"]["relative_photon_residual"]
                    for row in external_rows
                ),
                **_test_aggregates(external_rows),
            }
            _write_json(external_output / "summary.json", external_summary)
        # The input and executable identity must still match after the fit.
        if _implementation_hashes() != identity["implementation_sha256"]:
            raise RuntimeError("Training implementation changed during execution")
        for name, expected in dataset.file_hashes.items():
            if _digest(Path(name)) != expected:
                raise RuntimeError(f"Teacher artifact changed during training: {name}")
        if external is not None:
            for name, expected in external.file_hashes.items():
                if _digest(Path(name)) != expected:
                    raise RuntimeError(
                        f"External test artifact changed during evaluation: {name}"
                    )
        summary = {
            "schema": 1,
            "complete": True,
            "training_method": config["training_method"],
            "sample_counts": counts,
            "split_seeds": seed_roles,
            "selected_model_seed": selected["model_seed"],
            "selected_kind": selected["selected_kind"],
            "selection_scope": selection_scope,
            "selected_validation_mse": selected.get("validation_mse"),
            "selected_validation_objective": selected.get("validation_objective"),
            "field_checkpoint_selection": model.metadata.get(
                "field_checkpoint_selection", "Raw full-grid validation MSE"
            ),
            "budget_checkpoint_selection": model.metadata.get(
                "budget_checkpoint_selection", "Validation absorption-efficiency MSE"
            ),
            "budget_selection": budget_record,
            "budget_selection_scope": "Frozen budget for projected candidates; legacy candidates retain their independently trained budget"
            if improved
            else "Legacy field and budget heads",
            "selected_projection_mode": model.projection_mode,
            "test_used_for_selection": False,
            "known_regression_test_seeds": known_regression,
            "internal_test_role": test_role,
            "external_test": external_summary,
            **aggregates,
            "maximum_test_photon_residual": max(
                row["photon_ledger"]["relative_photon_residual"] for row in rows
            ),
            "conservation_scope": "Global predicted absorption plus separately predicted unabsorbed photons; not source-resolved transport or independently measured escape",
            "physical_scope": dataset.manifest.get("physical_scope"),
            "parallelism": {
                "workers": config["workers"],
                "threads_per_worker": config["threads"],
                "declared_cpus": config["cpus"],
                "process_start_method": "spawn",
            },
            "training_cells": details["training_cells"],
            "validation_cells": details["validation_cells"],
            "elapsed_seconds": time.perf_counter() - started,
            "timing_scope": "Training workflow excluding initial teacher archive audit; inference timings exclude I/O and diagnostics, include features and global projection",
        }
        _write_json(output / "summary.json", summary)
        files = {
            str(path.relative_to(output)): _digest(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
            and path.name not in {"training.lock", "completion.json", "COMPLETED"}
            and "incomplete" not in path.relative_to(output).parts
            and not path.name.endswith(".tmp")
        }
        _write_json(output / "completion.json", {"files": files})
        (output / "COMPLETED").write_text(
            "Selected by validation; held-out seed evaluated with a global photon ledger.\n"
        )
        return summary


def main() -> None:
    """Run a frozen JSON-configured analytic teacher training experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(json.loads(arguments.config.read_text()), arguments.output)


if __name__ == "__main__":
    main()
