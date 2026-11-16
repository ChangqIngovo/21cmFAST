"""Separate photon-budget and field-shape errors without fitting a model.

Oracle counts use teacher labels for diagnostics only. Counterfactual fields
are never training examples, candidate-selection scores, or deployed models.
The interventions are controlled comparisons, not an additive error partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from research.pc_analytic.validation_20261116.reproduction.recompute import (
    metrics,
    spectra,
    total,
)
from research.pc_analytic_emulator.evaluation import native_brightness
from research.pc_analytic_emulator.model import Emulator, field_features
from research.pc_analytic_emulator.projection import project_capacity_l2
from research.pc_field_emulator.model import project_global_budget

LOGGER = logging.getLogger(__name__)
METHODS = (
    "deployed",
    "oracle_final_budget",
    "oracle_recursive_budget",
    "teacher_predicted_budget",
)


def _project(
    latent: np.ndarray,
    atoms: np.ndarray,
    target: float,
    mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if mode == "capacity_l2":
        return project_capacity_l2(latent, atoms, target, rtol=1e-13, atol=0)
    if mode == "log_odds":
        return project_global_budget(
            np.clip(latent, 0, 1), atoms, target, rtol=1e-13, atol=0
        )
    raise ValueError("Unsupported diagnostic projection mode")


def _teacher_target(
    teacher: np.ndarray,
    source: np.ndarray,
    atoms: np.ndarray,
) -> float:
    if (
        teacher.shape != atoms.shape
        or not np.isfinite(teacher).all()
        or np.any((teacher < 0) | (teacher > 1))
    ):
        raise ValueError(
            "Teacher neutral fractions must share the grid and lie in [0, 1]"
        )
    absorbed = total(atoms.astype(np.longdouble) * (1 - teacher.astype(np.longdouble)))
    limit = min(total(source), total(atoms))
    tolerance = 32 * np.finfo(float).eps * max(limit, np.finfo(float).tiny)
    if absorbed > limit + tolerance:
        raise ValueError(
            "Teacher absorbed count exceeds the photon supply or atom capacity"
        )
    target = float(min(absorbed, limit))
    if np.longdouble(target) > limit:
        target = float(np.nextafter(target, 0))
    return target


def _recursive_oracle(
    model: Emulator,
    source: np.ndarray,
    atoms: np.ndarray,
    native_es: np.ndarray,
    redshift: float,
    box_len: float,
    target: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    features, _, baseline = field_features(
        source,
        atoms,
        native_es,
        redshift,
        box_len,
        radii_cMpc=model.radii_cMpc,
        feature_set=model.feature_set,
    )
    layers: list[dict[str, Any]] = []
    if model.base_model is not None:
        base, _, layers = _recursive_oracle(
            model.base_model,
            source,
            atoms,
            native_es,
            redshift,
            box_len,
            target,
        )
        baseline = base.ravel()
    if model.field_model.output_mode == "latent_linear":
        latent = model.field_model.latent_predict(features, baseline=baseline)
    else:
        latent = model.field_model.predict(features, baseline=baseline)
    latent = latent.reshape(atoms.shape)
    if not np.any(source):
        latent = np.zeros_like(latent)
    ionized, projection = _project(latent, atoms, target, model.projection_mode)
    layers.append(
        {
            "projection_mode": model.projection_mode,
            "field_output_mode": model.field_model.output_mode,
            "target_absorbed_photons": target,
            "projection": projection,
        }
    )
    return ionized, latent, layers


def _budget(
    ionized: np.ndarray, atoms: np.ndarray, source: np.ndarray, target: float
) -> dict:
    source_count = total(source)
    absorbed = total(atoms.astype(np.longdouble) * ionized)
    unabsorbed = float(source_count - np.longdouble(target))
    residual = float(
        abs(source_count - absorbed - np.longdouble(unabsorbed))
        / max(source_count, np.finfo(float).tiny)
    )
    if not np.isfinite(ionized).all() or np.any((ionized < 0) | (ionized > 1)):
        raise ValueError("Counterfactual field is nonfinite or outside [0, 1]")
    if unabsorbed < 0 or residual > 1e-10:
        raise FloatingPointError("Diagnostic field photon ledger failed")
    return {
        "target_absorbed_photons": target,
        "field_absorbed_photons": float(absorbed),
        "unabsorbed_photons": unabsorbed,
        "relative_photon_residual": residual,
    }


def decompose_snapshot(
    model: Emulator,
    source: ArrayLike,
    capacity: ArrayLike,
    native_es_xHI: ArrayLike,
    teacher_xHI: ArrayLike,
    redshift: float,
    box_len: float,
) -> dict[str, Any]:
    """Apply fixed-weight budget interventions to one native snapshot.

    Parameters
    ----------
    model
        Complete deployed emulator, including any frozen base chain.
    source, capacity, native_es_xHI, teacher_xHI
        Common-grid native photons, atom capacities, ES and teacher neutral
        fractions. Teacher labels are used only to construct oracle controls.
    redshift, box_len
        Snapshot redshift and comoving box length in cMpc.

    Returns
    -------
    dict
        Four neutral fields, intermediate latents, target/field photon counts
        and recursive projection records. The original model is not mutated.
        Final-only oracle keeps the original latent; recursive oracle rebuilds
        each frozen base at the teacher count. The budget-only counterfactual
        projects the teacher field itself to the original predicted count.
        Only capacity-L2 makes that last field a capacity-weighted squared-error
        lower bound among all fields at the incorrect count.
    """
    photons, atoms, native, teacher = (
        np.asarray(value, dtype=np.float64)
        for value in (source, capacity, native_es_xHI, teacher_xHI)
    )
    deployed = model.predict(photons, atoms, native, redshift, box_len)
    teacher_target = _teacher_target(teacher, photons, atoms)
    predicted_target = deployed["predicted_absorbed_photons"]
    final, final_projection = _project(
        deployed["latent_xHII"],
        atoms,
        teacher_target,
        model.projection_mode,
    )
    recursive, recursive_latent, layers = _recursive_oracle(
        model,
        photons,
        atoms,
        native,
        redshift,
        box_len,
        teacher_target,
    )
    budget_only, budget_projection = _project(
        1 - teacher,
        atoms,
        predicted_target,
        model.projection_mode,
    )
    ionized = {
        "deployed": deployed["xHII"],
        "oracle_final_budget": final,
        "oracle_recursive_budget": recursive,
        "teacher_predicted_budget": budget_only,
    }
    return {
        "fields": {name: 1 - field for name, field in ionized.items()},
        "latent_xHII": {
            "deployed": deployed["latent_xHII"].copy(),
            "oracle_final_budget": deployed["latent_xHII"].copy(),
            "oracle_recursive_budget": recursive_latent,
        },
        "budgets": {
            name: _budget(
                field,
                atoms,
                photons,
                teacher_target if name.startswith("oracle_") else predicted_target,
            )
            for name, field in ionized.items()
        },
        "teacher_absorbed_photons": teacher_target,
        "projection_mode": model.projection_mode,
        "teacher_projection_is_weighted_MSE_lower_bound": model.projection_mode
        == "capacity_l2",
        "projections": {
            "deployed": deployed["projection"],
            "oracle_final_budget": final_projection,
            "teacher_predicted_budget": budget_projection,
        },
        "oracle_layers": layers,
        "scope": "Teacher-label counterfactuals only; fixed field weights, no fitting or model selection",
    }


def _aggregate(rows: list[dict]) -> dict[str, Any]:
    result: dict[str, Any] = {"count": len(rows), "methods": {}}
    for name in METHODS:
        measured = [row["field_metrics"][name] for row in rows]
        result["methods"][name] = {
            key: float(np.mean([row[key] for row in measured])) if measured else None
            for key in (
                "MAE",
                "RMSE",
                "capacity_weighted_MSE",
                "capacity_weighted_RMSE",
            )
        }
        power = [
            row["low_k_21cm"]["comparisons"][name]["mean_absolute_power_ratio_error"]
            for row in rows
        ]
        power = [value for value in power if value is not None]
        result["methods"][name].update(
            mean_absolute_delta_Q_mass=float(
                np.mean([abs(row["delta_Q_mass"]) for row in measured])
            )
            if measured
            else None,
            mean_snapshot_low_k_power_ratio_error=float(np.mean(power))
            if power
            else None,
            defined_power_snapshots=len(power),
        )
    return result


def _phases(rows: list[dict]) -> dict[str, Any]:
    return {
        "all": _aggregate(rows),
        "active": _aggregate(
            [row for row in rows if 0.05 < row["teacher_xHI_mass"] < 0.95]
        ),
        "late": _aggregate([row for row in rows if row["teacher_xHI_mass"] <= 0.05]),
        "early": _aggregate([row for row in rows if row["teacher_xHI_mass"] >= 0.95]),
    }


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(
    model: Emulator,
    seed_root: Path,
    snapshot: dict,
    output: Path,
) -> dict[str, Any]:
    z, length = snapshot["redshift"], snapshot["box_len"]
    teacher_path = seed_root / "teacher" / f"z{z:.12g}".replace(".", "p") / "fields.npz"
    with np.load(teacher_path, allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key]) for key in archive.files}
    inputs = json.loads((seed_root / "native_inputs.json").read_text())
    source, atoms, native, teacher = (
        arrays[key]
        for key in (
            "source",
            "capacity",
            "native_es_xHI",
            "teacher_xHI",
        )
    )
    decomposition = decompose_snapshot(model, source, atoms, native, teacher, z, length)
    destination = output / seed_root.name / f"z{z:g}".replace(".", "p")
    destination.mkdir(parents=True)
    np.savez_compressed(
        destination / "fields.npz",
        **{
            "source": source,
            "capacity": atoms,
            "teacher_xHI": teacher,
            "native_es_xHI": native,
            **{f"{name}_xHI": value for name, value in decomposition["fields"].items()},
            **{
                f"{name}_latent_xHII": value
                for name, value in decomposition["latent_xHII"].items()
            },
        },
    )
    brightness = {
        name: native_brightness(field, atoms, z, inputs)
        for name, field in {"teacher": teacher, **decomposition["fields"]}.items()
    }
    np.testing.assert_allclose(
        brightness["teacher"], arrays["teacher_brightness"], rtol=3e-6, atol=2e-6
    )
    measured = {}
    for name, field in decomposition["fields"].items():
        measured[name] = metrics(field, teacher, atoms)
        mse = float(
            total(atoms.astype(np.longdouble) * (field - teacher) ** 2) / total(atoms)
        )
        measured[name].update(
            capacity_weighted_MSE=mse, capacity_weighted_RMSE=float(np.sqrt(mse))
        )
    record = {
        "seed": inputs["random_seed"],
        "redshift": z,
        "box_len": length,
        "dimension": atoms.shape[0],
        "teacher_xHI_mass": float(
            total(atoms.astype(np.longdouble) * teacher) / total(atoms)
        ),
        "field_metrics": measured,
        "budgets": decomposition["budgets"],
        "teacher_absorbed_photons": decomposition["teacher_absorbed_photons"],
        "projection_mode": decomposition["projection_mode"],
        "teacher_projection_is_weighted_MSE_lower_bound": decomposition[
            "teacher_projection_is_weighted_MSE_lower_bound"
        ],
        "oracle_layers": decomposition["oracle_layers"],
        "low_k_21cm": spectra(brightness, length, 0.5),
        "files_sha256": {
            "teacher_fields.npz": _digest(teacher_path),
            "fields.npz": _digest(destination / "fields.npz"),
        },
    }
    (destination / "report.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n"
    )
    LOGGER.info("Diagnosed seed %s z=%g", inputs["random_seed"], z)
    return record


def run(
    model_root: Path, teacher_root: Path, output: Path, seeds: list[int]
) -> dict[str, Any]:
    """Evaluate fixed-weight oracle controls on completed native teacher seeds.

    Parameters
    ----------
    model_root
        Directory containing a complete saved Emulator.
    teacher_root
        Teacher campaign with completed ``seeds/seed*/teacher`` archives.
    output
        New directory for fields and reports; existing directories are refused.
    seeds
        Unique seed identifiers to diagnose. No model is fitted or selected.

    Returns
    -------
    dict
        Per-snapshot and aggregated budget, field and low-k 21-cm diagnostics.
    """
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Diagnostics require unique nonempty seed identifiers")
    model = Emulator.load(model_root)
    hashes = {
        str(path.relative_to(model_root)): _digest(path)
        for path in model_root.rglob("*")
        if path.is_file()
    }
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for seed in seeds:
        seed_root = teacher_root / "seeds" / f"seed{seed}"
        summary = json.loads((seed_root / "summary.json").read_text())
        if not (seed_root / "COMPLETED").is_file() or not summary["complete"]:
            raise ValueError(f"Teacher seed {seed} is incomplete")
        rows.extend(
            _snapshot(model, seed_root, snapshot, output)
            for snapshot in summary["snapshots"]
        )
    if hashes != {
        str(path.relative_to(model_root)): _digest(path)
        for path in model_root.rglob("*")
        if path.is_file()
    }:
        raise RuntimeError("Model artifacts changed during diagnostic execution")
    result = {
        "complete": True,
        "diagnostic_only": True,
        "fitting_performed": False,
        "model_selection_performed": False,
        "scope": "Teacher counts are oracle controls, unavailable during deployment. Interventions are not an additive error partition.",
        "methods": {
            "deployed": "Original saved model and all predicted budgets",
            "oracle_final_budget": "Original deployed latent, exact teacher count at final projection only",
            "oracle_recursive_budget": "All frozen-base and final budgets replaced by exact teacher count; weights fixed, latents recomputed",
            "teacher_predicted_budget": "Teacher field projected to original predicted count; budget-only counterfactual, weighted-MSE lower bound only for capacity_l2",
        },
        "model_root": str(model_root.resolve()),
        "teacher_root": str(teacher_root.resolve()),
        "model_sha256": hashes,
        "diagnostic_source_sha256": _digest(Path(__file__)),
        "seeds": seeds,
        "aggregates": _phases(rows),
        "by_seed": {
            str(seed): _phases([row for row in rows if row["seed"] == seed])
            for seed in seeds
        },
        "by_redshift": {
            str(z): _phases([row for row in rows if row["redshift"] == z])
            for z in sorted({row["redshift"] for row in rows}, reverse=True)
        },
        "maximum_photon_residual": {
            name: max(row["budgets"][name]["relative_photon_residual"] for row in rows)
            for name in METHODS
        },
        "snapshots": rows,
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


def main() -> None:
    """Run the diagnostic CLI with explicit model, teacher and output paths."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()
    result = run(args.model, args.teacher, args.output, args.seeds)
    LOGGER.info(json.dumps(result["aggregates"], indent=2))


if __name__ == "__main__":
    main()
