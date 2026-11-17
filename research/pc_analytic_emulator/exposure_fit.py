"""Fit the spatial and global heads of the exposure emulator."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear, minimize

from research.pc_analytic_emulator.exposure_emulator import (
    FIELD_TERMS,
    FieldInputs,
    field_inputs,
    predict_fields,
)
from research.pc_analytic_emulator.predict import _digest
from research.pc_analytic_emulator.reliability import accuracy_metrics


@dataclass
class _TrainingFrame:
    """A source-derived feature cube and its supervised objectives."""

    record: dict
    source: np.ndarray
    atoms: np.ndarray
    target: np.ndarray
    inputs: FieldInputs

    def __post_init__(self) -> None:
        self.q = float(np.mean(self.atoms * self.target))
        load = self.inputs.load
        self.excess = max(self.q - max(1 - load, 0), 0.0)
        self.scale = 0.035 + self.q * (1 - self.q)
        frequency = np.fft.fftfreq(len(self.atoms))
        k2 = sum(
            axis**2
            for axis in np.meshgrid(
                frequency, frequency, frequency, sparse=True, indexing="ij"
            )
        )
        self.mask = (k2 > 0) & (k2 < 0.25**2)
        brightness = np.fft.fftn(self.atoms * self.target, norm="ortho")
        self.power = max(
            0.008**2 * np.mean(self.atoms**2),
            np.mean(np.abs(brightness * self.mask) ** 2),
        )


def _load(cache: Path, record: dict) -> _TrainingFrame:
    name = record["cache"]
    if Path(name).name != name:
        raise ValueError("Invalid cache member")
    path = cache / name
    if _digest(path) != record["cache_sha256"]:
        raise ValueError(f"Feature cache checksum mismatch: {path}")
    with np.load(path, allow_pickle=False) as saved:
        source, atoms, target = (saved[name] for name in ("source", "atoms", "target"))
        features = field_inputs(
            source,
            atoms,
            saved["near"],
            saved["far"],
            saved["onset"],
            float(saved["escape"]),
            float(saved["opacity"]),
        )
    return _TrainingFrame(record, source, atoms, target, features)


def fit_field(
    frames: list[_TrainingFrame], spectral: float, count: float, ridge: float
) -> np.ndarray:
    """Fit a clipped field against field, mean and low-k complex residuals."""

    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        loss = 0.0
        derivative = np.zeros_like(weights)
        for frame in frames:
            feature = frame.inputs
            latent = feature.latent + (weights @ feature.basis).reshape(
                frame.atoms.shape
            )
            residual = np.clip(latent, 0, feature.upper) - frame.target
            brightness = frame.atoms * residual
            mean = brightness.mean()
            gradient = (residual + count * mean * frame.atoms) / frame.scale**2
            value = (np.mean(residual**2) + count * mean**2) / frame.scale**2
            if spectral:
                fourier = np.fft.fftn(brightness, norm="ortho") * frame.mask
                value += spectral * np.mean(np.abs(fourier) ** 2) / frame.power
                gradient += (
                    spectral
                    / frame.power
                    * frame.atoms
                    * np.fft.ifftn(fourier, norm="ortho").real
                )
            active = (latent > 0) & (latent < feature.upper)
            derivative += (2 / latent.size) * (
                feature.basis @ (gradient * active).ravel()
            )
            loss += value
        return loss / len(frames) + ridge * (weights @ weights), (
            derivative / len(frames) + 2 * ridge * weights
        )

    fitted = minimize(
        objective,
        np.zeros(FIELD_TERMS),
        method="L-BFGS-B",
        jac=True,
        bounds=[(-0.25, 0.25)] * FIELD_TERMS,
        options={"maxiter": 70, "ftol": 1e-12},
    )
    if not np.isfinite(fitted.fun) or not np.isfinite(fitted.x).all():
        raise FloatingPointError("Field calibration is nonfinite")
    return fitted.x


def fit_budget(frames: list[_TrainingFrame], ridge: float) -> np.ndarray:
    """Learn the excess unabsorbed count over the unavoidable photon surplus."""
    constraints = np.stack([frame.inputs.reservoir for frame in frames])
    target = np.array([frame.excess for frame in frames])
    importance = 1 / np.array([0.006 + frame.q * (1 - frame.q) for frame in frames])
    prior = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    n = len(prior)
    system = np.vstack((constraints * importance[:, None], np.sqrt(ridge) * np.eye(n)))
    rhs = np.concatenate((target * importance, np.sqrt(ridge) * prior))
    return lsq_linear(system, rhs, bounds=(-4, 4)).x


def _evaluate(
    frames: list[_TrainingFrame], field: np.ndarray, budget: np.ndarray
) -> list[dict]:
    rows = []
    for frame in frames:
        prediction = predict_fields(
            frame.inputs, frame.source, frame.atoms, field, budget
        )
        metrics = accuracy_metrics(
            prediction["xHI"], frame.target, frame.atoms, frame.record["box_len"]
        )
        rows.append(
            {
                **{
                    name: frame.record[name]
                    for name in ("seed", "redshift", "box_len", "dimension")
                },
                **metrics,
            }
        )
    return rows


def _ranking(rows: list[dict]) -> tuple[float, float]:
    # Rank by the worst normalized error, then average field error.
    ratios = []
    for row in rows:
        late = row["phase"] == "late"
        ratios.extend(
            [
                row["MAE"] / (0.01 if late else 0.03),
                abs(row["delta_Q_mass"]) / (0.002 if late else 0.005),
            ]
        )
        if late:
            ratios.append(row["brightness_residual_rms_over_neutral_rms"] / 0.01)
        elif row["mean_relative_power_error"] is not None:
            ratios.append(row["mean_relative_power_error"] / 0.1)
    return max(ratios), float(np.mean([row["MAE"] for row in rows]))


def fit(cache: Path, output: Path) -> None:
    """Freeze the validation-selected weights with disjoint seed provenance."""
    data = json.loads((cache / "cache.json").read_text())
    records = data["records"]
    if set(data["seeds"]["train"]) & set(data["seeds"]["validation"]):
        raise ValueError("Training and validation share realization seeds")
    keys = sorted({(row["dim"], row["dimension"], row["box_len"]) for row in records})
    output.mkdir(parents=True, exist_ok=False)
    filters, reservoirs, geometry, outcomes = [], [], [], []
    for dim, dimension, box in keys:
        frames = [
            _load(cache, row)
            for row in records
            if (row["dim"], row["dimension"], row["box_len"]) == (dim, dimension, box)
        ]
        train = [row for row in frames if row.record["split"] == "train"]
        valid = [row for row in frames if row.record["split"] == "validation"]
        if not train or not valid:
            raise ValueError(
                "Every fitted geometry needs training and validation seeds"
            )
        trials = []
        baseline_field = np.zeros(FIELD_TERMS)
        baseline_budget = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        baseline = _evaluate(valid, baseline_field, baseline_budget)
        trial_configs = ((0.05, 5.0, 0.02), (0.10, 1.0, 0.02), (0.03, 2.0, 0.1))
        for spec, count, ridge in trial_configs:
            field_weights = fit_field(train, spec, count, ridge)
            for budget_ridge in (0.0001, 0.01, 0.1):
                budget_weights = fit_budget(train, budget_ridge)
                scores = _evaluate(valid, field_weights, budget_weights)
                trials.append(
                    (
                        _ranking(scores),
                        field_weights,
                        budget_weights,
                        scores,
                        [spec, count, ridge, budget_ridge],
                    )
                )
        ranking, field, budget, validation, controls = min(
            trials, key=lambda entry: entry[0]
        )
        filters.append(field)
        reservoirs.append(budget)
        geometry.append({"DIM": dim, "HII_DIM": dimension, "BOX_LEN": box})
        outcomes.append(
            {
                "geometry": geometry[-1],
                "baseline": baseline,
                "validation": validation,
                "ranking": ranking,
                "fitted_controls": controls,
                "candidates": [{"ranking": t[0], "controls": t[4]} for t in trials],
            }
        )
        logging.getLogger(__name__).info(
            "L%g N%d validation ranking %s", box, dimension, ranking
        )
    opacity = np.array(
        [
            row["opacity"]
            / ((1 + row["redshift"]) ** 2 * row["box_len"] / row["dimension"])
            for row in records
        ]
    )
    if not np.allclose(opacity, opacity[0], rtol=1e-12, atol=0):
        raise ValueError("Teacher opacity normalization differs across samples")
    path = output / "weights.npz"
    np.savez_compressed(path, field=np.stack(filters), budget=np.stack(reservoirs))
    metadata = {
        "schema": 1,
        "method": "learned_coeval_exposure",
        "depth": data["depth"],
        "opacity_prefactor": float(opacity[0]),
        "weights_sha256": _digest(path),
        "training_geometries": geometry,
        "teacher_physics": data["signature"]["physics"],
        "teacher_signature": data["signature"],
        "split_seeds": data["seeds"],
        "fit_cache_sha256": _digest(cache / "cache.json"),
    }
    (output / "model.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    (output / "validation.json").write_text(
        json.dumps(outcomes, indent=2, allow_nan=False)
    )
