"""Fit exposure-head coefficients through the deployed bounded photon budget.

All features must be produced with the same frozen onset head and exposure depth
used at inference. Teacher fields appear only in supervised losses. Low-k shells
receive equal weight, and their exact modes satisfy 0 < |k| < k_Nyquist / 2.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize

from research.pc_analytic_emulator.exposure_emulator import (
    FIELD_TERMS,
    FieldInputs,
    _project,
    predict_fields,
)


def bounded_neutral_vjp(
    neutral: np.ndarray,
    upper: np.ndarray,
    atoms: np.ndarray,
    gradient: np.ndarray,
    *,
    projection_method: str = "local_absorption_bounded_projection",
) -> np.ndarray:
    """Differentiate projected xHI with respect to its latent xHI field.

    On a fixed unsaturated set, xHI = latent - upper * lambda and
    sum(atoms*xHI) is fixed. Consequently the adjoint removes the atom-weighted
    budget direction. Exact clipping boundaries use the zero subgradient.
    """
    if not (neutral.shape == upper.shape == atoms.shape == gradient.shape):
        raise ValueError("Projection adjoint arrays must share a shape")
    output = np.zeros_like(gradient, dtype=np.float64)
    if projection_method in ("scale_local_floor", "saturated"):
        return output
    if projection_method != "local_absorption_bounded_projection":
        raise ValueError("Unknown bounded neutral projection method")
    # 1-(1-upper) can differ from upper by one floating-point ulp. Do not
    # mistake this cancellation at a clipped boundary for an active cell.
    tolerance = 8 * np.finfo(float).eps * np.maximum(1.0, upper)
    active = (atoms > 0) & (neutral > tolerance) & (neutral < upper - tolerance)
    if np.any(active):
        total = np.sum(atoms[active] * upper[active], dtype=np.longdouble)
        correction = (
            np.sum(gradient[active] * upper[active], dtype=np.longdouble) / total
        )
        output[active] = gradient[active] - atoms[active] * float(correction)
    return output


@dataclass
class SpectralTrainingFrame:
    """One supervised snapshot with exact deployed features and a frozen budget.

    Construct from normalized or physical photon/atom arrays sharing one unit.
    ``inputs`` must come from deployed onset/depth, never teacher transport.
    ``record`` is provenance only and does not enter fitting.
    """

    source: np.ndarray
    atoms: np.ndarray
    teacher: np.ndarray
    inputs: FieldInputs
    budget_weights: np.ndarray
    box_len: float
    record: dict[str, Any] = field(default_factory=dict)
    shell_variance_floor: float = 1e-8
    late_threshold: float = 0.05

    def __post_init__(self) -> None:
        """Validate arrays and cache fixed photon targets and shell weights."""
        self.source, self.atoms, self.teacher = (
            np.asarray(value, dtype=np.float64)
            for value in (self.source, self.atoms, self.teacher)
        )
        shape = self.atoms.shape
        if (
            len(shape) != 3
            or len(set(shape)) != 1
            or shape[0] < 4
            or self.source.shape != shape
            or self.teacher.shape != shape
            or not all(
                np.isfinite(x).all() for x in (self.source, self.atoms, self.teacher)
            )
            or np.any(self.source < 0)
            or np.any(self.atoms <= 0)
            or np.any((self.teacher < 0) | (self.teacher > 1))
            or not np.isfinite(self.box_len)
            or self.box_len <= 0
            or not np.isfinite(self.shell_variance_floor)
            or self.shell_variance_floor <= 0
            or not 0 < self.late_threshold < 1
        ):
            raise ValueError("Invalid supervised spectral snapshot")
        if (
            self.inputs.upper.shape != shape
            or self.inputs.latent.shape != shape
            or self.inputs.basis.shape != (FIELD_TERMS, self.atoms.size)
        ):
            raise ValueError("Deployed field feature shapes differ from snapshot")
        prediction = predict_fields(
            self.inputs,
            self.source,
            self.atoms,
            np.zeros(FIELD_TERMS),
            self.budget_weights,
        )
        self.absorbed_count = prediction["predicted_absorbed_photons"]
        # Unit normalization has no effect on any loss or its gradient.
        self.rho = self.atoms / np.mean(self.atoms)
        self.q_mass = float(np.mean(self.rho * self.teacher))
        self.late = self.q_mass <= self.late_threshold
        self.field_scale = max(float(np.var(self.teacher)), 0.01)
        self.neutral_mean_square = float(np.mean(self.rho**2))
        n = shape[0]
        axis = np.fft.fftfreq(n) * n
        norm = np.sqrt(
            axis[:, None, None] ** 2
            + axis[None, :, None] ** 2
            + axis[None, None, :] ** 2
        )
        self.mask = (norm > 0) & (norm < n / 4)
        self.labels = np.floor(norm[self.mask] + 0.5).astype(np.int64)
        self.counts = np.bincount(self.labels)
        self.shells = np.flatnonzero(self.counts)
        self.teacher_fft = np.fft.fftn(self.rho * self.teacher, norm="ortho")
        self.teacher_power = np.bincount(
            self.labels,
            weights=np.abs(self.teacher_fft[self.mask]) ** 2,
            minlength=len(self.counts),
        ) / np.maximum(self.counts, 1)
        # A floor on integrated shell variance has grid-independent units.
        variance_floor = self.shell_variance_floor * self.neutral_mean_square
        self.denominator = np.maximum(
            self.teacher_power,
            variance_floor * self.atoms.size / np.maximum(self.counts, 1),
        )
        self.denominator = np.maximum(self.denominator, np.finfo(float).tiny)
        self.shell_weights = np.zeros(len(self.counts), dtype=np.float64)
        if len(self.shells):
            self.shell_weights[self.shells] = 1 / (
                len(self.shells)
                * self.counts[self.shells]
                * self.denominator[self.shells]
            )


def projected_frame_objective(
    weights: np.ndarray,
    frame: SpectralTrainingFrame,
    *,
    field_weight: float = 1.0,
    spectral_weight: float = 1.0,
    power_weight: float = 0.0,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    """Evaluate exact postprojection field and phase-sensitive spectral loss.

    Late snapshots use absolute brightness residual normalized by the fully
    neutral RMS, rather than ill-conditioned relative power. An optional power
    term is supplementary to the complex residual and defaults to disabled.
    """
    if any(
        not np.isfinite(x) or x < 0
        for x in (field_weight, spectral_weight, power_weight)
    ):
        raise ValueError("Loss weights must be finite and nonnegative")
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (FIELD_TERMS,) or not np.isfinite(weights).all():
        raise ValueError("Invalid field coefficient vector")
    latent = frame.inputs.latent + (weights @ frame.inputs.basis).reshape(
        frame.atoms.shape
    )
    ionized, projection = _project(
        latent, frame.inputs.upper, frame.atoms, frame.absorbed_count
    )
    predicted = 1 - ionized
    residual = predicted - frame.teacher
    field_loss = float(np.mean(residual**2) / frame.field_scale)
    gradient = 2 * field_weight * residual / (residual.size * frame.field_scale)
    brightness_residual = frame.rho * residual
    power_loss = 0.0
    if frame.late:
        spectral_loss = float(
            np.mean(brightness_residual**2) / frame.neutral_mean_square
        )
        gradient += (
            2
            * spectral_weight
            * frame.rho
            * brightness_residual
            / (residual.size * frame.neutral_mean_square)
        )
    else:
        difference = np.fft.fftn(brightness_residual, norm="ortho")
        weighted = np.zeros_like(difference)
        weighted[frame.mask] = (
            difference[frame.mask] * frame.shell_weights[frame.labels]
        )
        spectral_loss = float(np.vdot(difference, weighted).real)
        gradient += (
            2 * spectral_weight * frame.rho * np.fft.ifftn(weighted, norm="ortho").real
        )
        if power_weight and len(frame.shells):
            fourier = difference + frame.teacher_fft
            power = np.bincount(
                frame.labels,
                weights=np.abs(fourier[frame.mask]) ** 2,
                minlength=len(frame.counts),
            ) / np.maximum(frame.counts, 1)
            delta = power - frame.teacher_power
            power_loss = float(
                np.mean((delta[frame.shells] / frame.denominator[frame.shells]) ** 2)
            )
            weighted.fill(0)
            factors = frame.shell_weights * delta / frame.denominator
            weighted[frame.mask] = fourier[frame.mask] * factors[frame.labels]
            gradient += (
                4 * power_weight * frame.rho * np.fft.ifftn(weighted, norm="ortho").real
            )
    latent_gradient = bounded_neutral_vjp(
        predicted,
        frame.inputs.upper,
        frame.atoms,
        gradient,
        projection_method=projection["method"],
    )
    derivative = frame.inputs.basis @ latent_gradient.ravel()
    value = (
        field_weight * field_loss
        + spectral_weight * spectral_loss
        + power_weight * power_loss
    )
    return (
        float(value),
        derivative,
        {
            "field_loss": field_loss,
            "spectral_loss": spectral_loss,
            "power_loss": power_loss,
            "late": frame.late,
            "shell_count": len(frame.shells),
            "predicted_absorbed_photons": frame.absorbed_count,
            "projection_method": projection["method"],
        },
    )


def fit_projected_field(
    frames: Sequence[SpectralTrainingFrame],
    initial_weights: np.ndarray,
    *,
    field_weight: float = 1.0,
    spectral_weight: float = 1.0,
    power_weight: float = 0.0,
    ridge: float = 1e-4,
    maxiter: int = 150,
    coefficient_bound: float = 0.25,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one geometry on training frames only; return coefficients and receipt.

    Candidate selection belongs to a separate seed-disjoint validation driver.
    Budget weights and onset head remain frozen during this spatial fit.
    """
    initial = np.asarray(initial_weights, dtype=np.float64)
    if (
        not frames
        or initial.shape != (FIELD_TERMS,)
        or not np.isfinite(initial).all()
        or not np.isfinite(ridge)
        or ridge < 0
        or not np.isfinite(coefficient_bound)
        or coefficient_bound <= 0
        or maxiter < 1
    ):
        raise ValueError("Invalid projected-head training controls")
    for frame in frames:
        if frame.record.get("split", "train") != "train":
            raise ValueError("Only training frames may enter coefficient fitting")
    controls = {
        "field_weight": field_weight,
        "spectral_weight": spectral_weight,
        "power_weight": power_weight,
    }

    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        value = 0.0
        gradient = np.zeros_like(weights)
        for frame in frames:
            loss, derivative, _ = projected_frame_objective(weights, frame, **controls)
            value += loss
            gradient += derivative
        delta = weights - initial
        return value / len(frames) + ridge * float(delta @ delta), gradient / len(
            frames
        ) + 2 * ridge * delta

    before, _ = objective(initial)
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=[(-coefficient_bound, coefficient_bound)] * FIELD_TERMS,
        options={"maxiter": maxiter, "ftol": 1e-13, "gtol": 1e-9, "maxls": 40},
    )
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise FloatingPointError("Projected-head fit returned nonfinite results")
    if result.fun > before + 1e-12:
        raise RuntimeError("Projected-head fit increased its training objective")
    return result.x, {
        "objective": "Postprojection normalized field MSE plus equal-shell complex brightness residual",
        "budget_frozen": True,
        "snapshots": len(frames),
        "seeds": sorted({int(f.record["seed"]) for f in frames if "seed" in f.record}),
        "controls": controls
        | {"ridge": ridge, "maxiter": maxiter, "coefficient_bound": coefficient_bound},
        "initial_objective": float(before),
        "final_objective": float(result.fun),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "iterations": int(result.nit),
        "gradient_max_abs": float(np.max(np.abs(result.jac))),
        "initial_weights": initial.tolist(),
        "fitted_weights": result.x.tolist(),
    }
