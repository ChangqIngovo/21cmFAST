"""Compare fixed-field closures before fitting a frozen-base residual model.

All candidates use the same native inputs and validation objective. The field
ablation changes only the predicted global absorption and projection; subsequent
residual fitting starts from the complete deployed field of its selected base.
Teacher fields enter supervised training and scoring, never the inference base.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from research.pc_analytic_emulator.model import Emulator

LOGGER = logging.getLogger(__name__)
SELECTION_SCOPE = (
    "Full-grid projected multiscale validation objective; fixed-field closure "
    "ablation and frozen-base residual; auxiliary raw loss and test labels excluded"
)


def _budget_rows(task: dict[str, Any]) -> list[dict[str, str]]:
    rows = task.get("budget_candidates", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("Warm-start fitting requires named budget candidates")
    names = [row.get("kind") for row in rows if isinstance(row, dict)]
    if (
        len(names) != len(rows)
        or any(
            not isinstance(name, str)
            or not re.fullmatch(r"[a-zA-Z0-9_]+", name)
            or name == "legacy_budget"
            for name in names
        )
        or len(set(names)) != len(names)
    ):
        raise ValueError("Warm-start budget candidate names must be unique and safe")
    for row in rows:
        if (
            not isinstance(row.get("model_path"), str)
            or not Path(row["model_path"]).is_file()
        ):
            raise ValueError("Every warm-start budget candidate requires a saved model")
    return rows


def _score_candidate(
    kind: str,
    model: Emulator,
    frames: Sequence[dict[str, Any]],
    objective: dict[str, Any],
    *,
    stage: str,
    field_sha256: str | None,
    budget_kind: str,
) -> dict[str, Any]:
    from research.pc_analytic_emulator.train import _score_emulator

    score, rows = _score_emulator(model, frames, objective)
    for row, frame in zip(rows, frames, strict=True):
        with np.load(frame["cache"], allow_pickle=False) as data:
            capacity = np.asarray(data["capacity"], dtype=np.longdouble)
            truth = 1 - np.asarray(data["teacher_xHI"], dtype=np.longdouble)
            atoms = np.sum(capacity, dtype=np.longdouble)
            absorbed = np.sum(capacity * truth, dtype=np.longdouble)
            projected = np.longdouble(row["projection"]["corrected_count"])
            row["teacher_Q_mass"] = float(absorbed / atoms)
            row["predicted_Q_mass"] = float(projected / atoms)
            row["delta_Q_mass"] = float((projected - absorbed) / atoms)
    by_seed = []
    for seed in sorted({row["seed"] for row in rows}):
        selected = [row for row in rows if row["seed"] == seed]
        by_seed.append(
            {
                "seed": seed,
                "snapshot_count": len(selected),
                "validation_objective": float(
                    np.mean([row["projected_loss"] for row in selected])
                ),
                "validation_mse": float(
                    np.mean([row["field_mse"] for row in selected])
                ),
                "mean_absolute_delta_Q_mass": float(
                    np.mean([abs(row["delta_Q_mass"]) for row in selected])
                ),
                "maximum_absolute_delta_Q_mass": max(
                    abs(row["delta_Q_mass"]) for row in selected
                ),
            }
        )
    if not np.isfinite(score):
        raise ValueError(
            "Warm-start candidate selection requires finite validation scores"
        )
    return {
        "kind": kind,
        "stage": stage,
        "field_sha256": field_sha256,
        "budget_kind": budget_kind,
        "projection_mode": model.projection_mode,
        "validation_objective": score,
        "validation_mse": float(np.mean([row["field_mse"] for row in rows])),
        "validation_by_seed": by_seed,
        "snapshots": rows,
    }


def _best(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    # Original deployment wins exact ties; residual fitting must improve on it.
    return min(
        rows,
        key=lambda row: (
            row["validation_objective"],
            row["stage"] != "legacy_reference",
            row["kind"],
        ),
    )


def fit_trial(task: dict[str, Any]) -> dict[str, Any]:
    """Fit one legacy initialization, audit closures, and refine its deployed field.

    Parameters
    ----------
    task
        ``config``, immutable ``identity``, ``model_seed``, output ``folder``,
        full-grid ``cache`` and sampled ``legacy_cache``. ``budget_candidates``
        contains unique ``kind`` and ``model_path`` entries fitted independently
        on the training split. The trial sees no test frames or test labels.

    Returns
    -------
    dict
        Receipt-backed trial record with fixed-field ablation scores, selected
        frozen base, per-validation-seed metrics and the final model directory.
        Completed legacy fitting survives an interrupted residual trial.
    """
    from research.pc_analytic_emulator.budget import load_budget
    from research.pc_analytic_emulator.fit import fit_field
    from research.pc_analytic_emulator.train import (
        _archive_partial,
        _digest,
        _fit_legacy_trial,
        _seal,
        _verified_record,
        _write_json,
    )

    budget_rows = _budget_rows(task)
    config, identity, seed = task["config"], task["identity"], task["model_seed"]
    folder, cache = Path(task["folder"]), Path(task["cache"])
    trial_identity = {"training": identity, "model_seed": seed}
    old = _verified_record(folder, trial_identity)
    if old is not None:
        LOGGER.info("Warm-start trial resumed and verified: initialization=%d", seed)
        return old
    legacy_folder = folder / "legacy_reference"
    restored_legacy = None
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
    legacy_record = _fit_legacy_trial(
        dict(task, folder=str(legacy_folder), cache=task["legacy_cache"])
    )
    trained_path = legacy_folder / "trained_model"
    original = Emulator.load(legacy_record["model_directory"])
    trained = Emulator.load(trained_path)
    field_hash = _digest(trained_path / "field_model.npz")
    budgets = [("legacy_budget", trained.budget_model)] + [
        (row["kind"], load_budget(row["model_path"])) for row in budget_rows
    ]
    candidates: list[dict[str, Any]] = []
    models: dict[str, Emulator] = {}
    for budget_kind, budget in budgets:
        for projection in ("log_odds", "capacity_l2"):
            kind = f"fixed_legacy__{budget_kind}__{projection}"
            model = Emulator(
                trained.field_model,
                budget,
                radii_cMpc=trained.radii_cMpc,
                feature_set=trained.feature_set,
                projection_mode=projection,
                metadata=dict(trained.metadata)
                | {
                    "training_method": config["training_method"],
                    "fixed_field_sha256": field_hash,
                    "budget_kind": budget_kind,
                    "budget_checkpoint_selection": "Fixed-field validation closure comparison; predictor fitting remains separate",
                    "selection_scope": SELECTION_SCOPE,
                },
            )
            candidates.append(
                _score_candidate(
                    kind,
                    model,
                    details["validation_snapshots"],
                    objective,
                    stage="fixed_field",
                    field_sha256=field_hash,
                    budget_kind=budget_kind,
                )
            )
            models[kind] = model
    candidates.append(
        _score_candidate(
            "legacy_reference",
            original,
            details["validation_snapshots"],
            objective,
            stage="legacy_reference",
            field_sha256=_digest(
                Path(legacy_record["model_directory"]) / "field_model.npz"
            ),
            budget_kind="legacy_selected_budget",
        )
    )
    models["legacy_reference"] = original
    base_record = _best(candidates)
    base = models[base_record["kind"]]
    base.save(folder / "base_model")
    base_files = {
        str(path.relative_to(folder / "base_model")): _digest(path)
        for path in sorted((folder / "base_model").rglob("*"))
        if path.is_file()
    }

    def log(row: dict[str, Any]) -> None:
        LOGGER.info(
            "Warm-start trial %d epoch metrics: %s",
            seed,
            json.dumps(row, sort_keys=True),
        )

    field, history = fit_field(
        details["training_snapshots"],
        details["validation_snapshots"],
        base.budget_model,
        base_model=base,
        seed=seed,
        hidden=tuple(config["hidden"]),
        epochs=config["warmstart_epochs"],
        learning_rate=config["warmstart_learning_rate"],
        patience=config["patience"],
        objective_config=objective,
        on_epoch=log,
        frame_cache_max_bytes=config.get("frame_cache_MiB", 512) * 1024**2,
        field_feature_scale_floor=config.get("field_feature_scale_floor", 0.0),
        zero_constant_features=config.get("zero_constant_features", False),
    )
    residual = Emulator(
        field,
        base.budget_model,
        radii_cMpc=config["radii_cMpc"],
        feature_set=config.get("feature_set", "legacy"),
        projection_mode="capacity_l2",
        base_model=base,
        metadata={
            "training_method": config["training_method"],
            "model_seed": seed,
            "field_feature_names": details["field_feature_names"],
            "budget_feature_names": details["budget_feature_names"],
            "field_checkpoint_selection": SELECTION_SCOPE,
            "budget_checkpoint_selection": "Frozen selected base budget; no budget fitting during residual training",
            "base_kind": base_record["kind"],
            "base_files_sha256": base_files,
            "objective": objective,
        },
    )
    residual_record = _score_candidate(
        "warm_started_residual",
        residual,
        details["validation_snapshots"],
        objective,
        stage="residual",
        field_sha256=None,
        budget_kind=base_record["budget_kind"],
    )
    candidates.append(residual_record)
    models["warm_started_residual"] = residual
    residual.save(folder / "trained_model")
    # Keep the original base when the apparent gain is only rounding at the
    # second projection, rather than a measurable validation improvement.
    selected = (
        residual_record
        if residual_record["validation_objective"]
        < base_record["validation_objective"] - 1e-12
        else base_record
    )
    chosen = models[selected["kind"]]
    chosen.metadata = dict(chosen.metadata) | {
        "selected_kind": selected["kind"],
        "selection_scope": SELECTION_SCOPE,
        "selection_training_method": config["training_method"],
    }
    chosen.save(folder / "model")
    _write_json(
        folder / "history.json",
        {
            "field": history,
            "budget": "Frozen full deployed base budget; see closure candidates",
        },
    )
    record = {
        "model_seed": seed,
        "selected_kind": selected["kind"],
        "validation_objective": selected["validation_objective"],
        "validation_mse": selected["validation_mse"],
        "validation_by_seed": selected["validation_by_seed"],
        "candidates": candidates,
        "base_selection": base_record,
        "base_files_sha256": base_files,
        "fixed_field_sha256": field_hash,
        "fixed_field_retrained_during_ablation": False,
        "residual_base_frozen": True,
        "model_directory": str(folder / "model"),
        "elapsed_seconds": time.perf_counter() - started,
        "field_best_epoch": field.metadata["best_epoch"],
        "field_completed_epochs": field.metadata["completed_epochs"],
        "budget_candidates": budget_rows,
        "legacy_reference": legacy_record,
        "test_used": False,
    }
    _write_json(folder / "trial.json", record)
    _seal(folder, trial_identity, record)
    return record
