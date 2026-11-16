"""Measure ranking and calibration limits using teacher-labelled monotone oracles.

This is an after-selection diagnostic, not an emulator or a training procedure.
It cannot change a deployed model. Each oracle uses the complete teacher field
of the evaluated snapshot and is therefore not a generalization result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

LOGGER = logging.getLogger(__name__)


def _real(values: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(values)
    if array.dtype.kind not in "fiu":
        raise TypeError("Ranking diagnostics require real numeric arrays")
    return np.asarray(array, dtype=float)


def weighted_monotone_oracle(
    scores: ArrayLike, teacher_ionized: ArrayLike, atoms: ArrayLike
) -> NDArray[np.float64]:
    """Fit the best nondecreasing map of fixed scores to teacher ionization.

    Parameters
    ----------
    scores
        Frozen, possibly unbounded ionization scores. Exact ties are grouped
        before fitting, so tied cells cannot acquire an artificial spatial order.
    teacher_ionized
        Teacher xHII labels in [0, 1], used only for this diagnostic oracle.
    atoms
        Strictly positive cell atom capacities; all arrays have matching shapes.

    Returns
    -------
    ndarray
        Capacity-weighted least-squares monotone fit with the input shape. Its
        weighted sum equals the teacher's absorbed count up to rounding, since
        every fitted block uses the atom-weighted mean of its teacher labels.
        The lower-bound property applies to capacity-weighted field MSE only.
    """
    rank, truth, weight = (_real(value) for value in (scores, teacher_ionized, atoms))
    if not rank.size or rank.shape != truth.shape or rank.shape != weight.shape:
        raise ValueError("Ranking arrays require nonempty matching shapes")
    if not np.isfinite(rank).all() or not np.isfinite(truth).all():
        raise ValueError("Ranking scores and teacher labels must be finite")
    if np.any((truth < 0) | (truth > 1)):
        raise ValueError("Teacher ionization must be bounded by zero and one")
    if not np.isfinite(weight).all() or np.any(weight <= 0):
        raise ValueError("Atom weights must be finite and strictly positive")
    order = np.argsort(rank.ravel(), kind="stable")
    sorted_scores = rank.ravel()[order]
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    # Scale all weights equally before pooling; the fitted map is invariant to
    # count units and avoids overflowing intermediate atom totals.
    scaled = weight.ravel()[order].astype(np.longdouble) / np.max(weight)
    weighted_labels = scaled * truth.ravel()[order].astype(np.longdouble)
    group_weights = np.add.reduceat(scaled, starts)
    group_targets = np.add.reduceat(weighted_labels, starts)
    blocks: list[tuple[int, int, np.longdouble, np.longdouble]] = []
    for index, (mass, target) in enumerate(
        zip(group_weights, group_targets, strict=True)
    ):
        blocks.append((index, index + 1, mass, target))
        while (
            len(blocks) > 1
            and blocks[-2][3] / blocks[-2][2] > blocks[-1][3] / blocks[-1][2]
        ):
            right, left = blocks.pop(), blocks.pop()
            blocks.append((left[0], right[1], left[2] + right[2], left[3] + right[3]))
    group_fit = np.empty(len(starts), dtype=float)
    for first, last, mass, target in blocks:
        group_fit[first:last] = float(target / mass)
    sorted_fit = np.repeat(group_fit, np.diff(np.r_[starts, rank.size]))
    fitted = np.empty(rank.size, dtype=float)
    fitted[order] = sorted_fit
    return fitted.reshape(rank.shape)


def _conditional_row(
    selected: np.ndarray, prediction: np.ndarray, teacher: np.ndarray, atoms: np.ndarray
) -> dict[str, Any]:
    error = prediction[selected] - teacher[selected]
    return {
        "cells": int(selected.sum()),
        "MAE": float(np.mean(np.abs(error))) if error.size else None,
        "RMSE": float(np.sqrt(np.mean(error**2))) if error.size else None,
        "capacity_weighted_MSE": float(np.average(error**2, weights=atoms[selected]))
        if error.size
        else None,
        "absolute_error_over_all_atoms": float(
            np.sum(atoms[selected] * np.abs(error)) / np.sum(atoms)
        ),
    }


def conditional_errors(
    predicted_xhi: ArrayLike, teacher_xhi: ArrayLike, atoms: ArrayLike
) -> dict[str, Any]:
    """Locate field errors by teacher neutral fraction and density quartile.

    Parameters
    ----------
    predicted_xhi, teacher_xhi, atoms
        Matching bounded neutral fractions and positive atom capacities.

    Returns
    -------
    dict
        Conditional field errors and teacher neutral mass contained in cells
        that the deployed prediction sets to exactly zero neutral fraction.
    """
    prediction, teacher, capacity = (
        _real(value) for value in (predicted_xhi, teacher_xhi, atoms)
    )
    if prediction.shape != teacher.shape or teacher.shape != capacity.shape:
        raise ValueError("Conditional fields require matching shapes")
    neutral_mass = np.sum(capacity.astype(np.longdouble) * teacher)
    lost_mass = np.sum((capacity.astype(np.longdouble) * teacher)[prediction == 0])
    bins = []
    lower = -np.inf
    for upper in (1e-6, 0.01, 0.1, 0.5, 0.9, 1.0):
        mask = (teacher > lower) & (teacher <= upper)
        bins.append(
            {
                "lower_exclusive": None if not np.isfinite(lower) else lower,
                "upper_inclusive": upper,
                **_conditional_row(mask, prediction, teacher, capacity),
            }
        )
        lower = upper
    edges = np.unique(np.quantile(capacity, [0, 0.25, 0.5, 0.75, 1]))
    density_bins = []
    if len(edges) == 1:
        edges = np.repeat(edges, 2)
    for index, (low, high) in enumerate(pairwise(edges)):
        mask = (capacity >= low) & (
            (capacity <= high) if index == len(edges) - 2 else (capacity < high)
        )
        density_bins.append(
            {
                "lower_inclusive": float(low),
                "upper": float(high),
                "upper_inclusive": index == len(edges) - 2,
                **_conditional_row(mask, prediction, teacher, capacity),
            }
        )
    return {
        "teacher_neutral_fraction_bins": bins,
        "density_quantile_bins": density_bins,
        "teacher_neutral_mass_in_predicted_zero_xHI_over_atoms": float(
            lost_mass / np.sum(capacity, dtype=np.longdouble)
        ),
        "fraction_of_teacher_neutral_mass_in_predicted_zero_xHI": float(
            lost_mass / neutral_mass
        )
        if neutral_mass > 0
        else None,
    }


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _model_hashes(path: Path) -> dict[str, str]:
    return {
        str(item.relative_to(path)): _digest(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = list(rows[0]["field_metrics"]) if rows else []
    return {
        "snapshots": len(rows),
        "methods": {
            name: {
                **{
                    metric: float(
                        np.mean([row["field_metrics"][name][metric] for row in rows])
                    )
                    for metric in ("MAE", "RMSE", "capacity_weighted_MSE")
                },
                "mean_snapshot_low_k_power_ratio_error": float(
                    np.mean(
                        [
                            row["low_k_21cm"]["comparisons"][name][
                                "mean_absolute_power_ratio_error"
                            ]
                            for row in rows
                            if row["low_k_21cm"]["comparisons"][name][
                                "mean_absolute_power_ratio_error"
                            ]
                            is not None
                        ]
                    )
                )
                if any(
                    row["low_k_21cm"]["comparisons"][name][
                        "mean_absolute_power_ratio_error"
                    ]
                    is not None
                    for row in rows
                )
                else None,
            }
            for name in names
        },
    }


def main() -> None:
    """Run after-selection ranking oracles without modifying the frozen emulator."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "teacher", "training", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    from research.pc_analytic.validation_20261116.reproduction.recompute import (
        metrics,
        spectra,
    )
    from research.pc_analytic_emulator.data import load_dataset
    from research.pc_analytic_emulator.model import Emulator

    if not (args.training / "COMPLETED").is_file():
        raise ValueError("Ranking diagnosis requires completed model selection")
    identity = json.loads((args.training / "identity.json").read_text())
    implementation = identity["implementation_sha256"]
    inference_paths = (
        "research/pc_analytic_emulator/model.py",
        "research/pc_analytic_emulator/budget.py",
        "research/pc_analytic_emulator/latent.py",
        "research/pc_analytic_emulator/projection.py",
        "research/pc_field_emulator/model.py",
        "research/pc_field_emulator/features.py",
    )
    for name in inference_paths:
        if _digest(args.source / name) != implementation[name]:
            raise ValueError(f"Inference implementation changed: {name}")
    model_path = args.training / "model"
    before = _model_hashes(model_path)
    model = Emulator.load(model_path)
    dataset = load_dataset(args.teacher)
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for sample in dataset.samples:
        arrays = sample.arrays
        source, atoms, native, truth = (
            arrays[key]
            for key in ("source", "capacity", "native_es_xHI", "teacher_xHI")
        )
        arguments = (source, atoms, native, sample.redshift, sample.box_len)
        prediction = model.predict(*arguments)
        base = (
            model.base_model.predict(*arguments)
            if model.base_model is not None
            else prediction
        )
        scores = {
            "selected_latent": prediction["latent_xHII"],
            "selected_projected": prediction["xHII"],
            "base_latent": base["latent_xHII"],
            "base_projected": base["xHII"],
            "native_es": 1 - native,
        }
        fields = {
            "selected": prediction["xHI"],
            "base": base["xHI"],
            "native_es": native,
        }
        oracles = {}
        teacher_ions = 1 - truth
        total_atoms = np.sum(atoms, dtype=np.longdouble)
        absorbed = np.sum(atoms.astype(np.longdouble) * teacher_ions)
        for name, score in scores.items():
            ionized = weighted_monotone_oracle(score, teacher_ions, atoms)
            oracle_absorbed = np.sum(atoms.astype(np.longdouble) * ionized)
            relative = float(abs(oracle_absorbed - absorbed) / total_atoms)
            if np.any((ionized < 0) | (ionized > 1)) or relative > 2e-14:
                raise ValueError("Monotone oracle violated bounded teacher budget")
            fields[f"oracle_{name}"] = 1 - ionized
            oracles[name] = {
                "unique_scores": len(np.unique(score)),
                "cells": score.size,
                "fraction_score_at_or_below_zero": float(np.mean(score <= 0)),
                "fraction_score_at_or_above_one": float(np.mean(score >= 1)),
                "teacher_absorption_residual_over_atoms": relative,
            }
        field_metrics = {}
        for name, field in fields.items():
            row = metrics(field, truth, atoms)
            row["capacity_weighted_MSE"] = float(
                np.average((field - truth) ** 2, weights=atoms)
            )
            field_metrics[name] = row
        for oracle, original in (
            ("oracle_selected_latent", "selected"),
            ("oracle_selected_projected", "selected"),
            ("oracle_base_latent", "base"),
            ("oracle_base_projected", "base"),
            ("oracle_native_es", "native_es"),
        ):
            if (
                field_metrics[oracle]["capacity_weighted_MSE"]
                > field_metrics[original]["capacity_weighted_MSE"] + 2e-14
            ):
                raise ValueError("Oracle failed its monotone-remapping MSE bound")
        all_fields = {"teacher": truth, **fields}
        row = {
            "seed": sample.seed,
            "redshift": sample.redshift,
            "teacher_xHI_mass": float(
                np.sum(atoms.astype(np.longdouble) * truth) / total_atoms
            ),
            "teacher_field_sha256": _digest(sample.fields_path),
            "field_metrics": field_metrics,
            "low_k_21cm": spectra(
                {name: atoms * field for name, field in all_fields.items()},
                sample.box_len,
                0.5,
            ),
            "ranking_scores": oracles,
            "deployed_conditional_errors": conditional_errors(
                prediction["xHI"], truth, atoms
            ),
        }
        rows.append(row)
        destination = args.output / f"seed{sample.seed}_z{sample.redshift:g}.npz"
        np.savez_compressed(
            destination,
            **all_fields,
            **{f"score_{name}": value for name, value in scores.items()},
        )
        LOGGER.info(
            "Ranking diagnosis seed%d z%g complete", sample.seed, sample.redshift
        )
    after = _model_hashes(model_path)
    if before != after:
        raise ValueError("Frozen deployed model changed during diagnosis")
    report = {
        "schema": 1,
        "complete": True,
        "scope": "Post-selection teacher-labelled monotone calibration oracle; not deployable or independent generalization evidence. Only capacity-weighted field MSE is minimized, not MAE or power error.",
        "selection_changed": False,
        "model_unchanged": True,
        "teacher": str(args.teacher.resolve()),
        "training": str(args.training.resolve()),
        "model_files_sha256": before,
        "implementation_sha256": {
            name: implementation[name] for name in inference_paths
        },
        "diagnostic_sha256": _digest(Path(__file__)),
        "spectral_helper_sha256": _digest(Path(__file__).with_name("recompute.py")),
        "all": _aggregate(rows),
        "active": _aggregate(
            [row for row in rows if 0.05 < row["teacher_xHI_mass"] < 0.95]
        ),
        "late": _aggregate([row for row in rows if row["teacher_xHI_mass"] <= 0.05]),
        "snapshots": rows,
    }
    (args.output / "ranking_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    (args.output / "COMPLETED").write_text(
        "Teacher-labelled ranking diagnosis completed; model unchanged.\n"
    )


if __name__ == "__main__":
    main()
