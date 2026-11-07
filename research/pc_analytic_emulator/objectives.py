"""Differentiate full-grid field and phase-sensitive 21-cm Fourier residuals.

The loss is evaluated on the same projected xHII field used at inference.
Fourier shells carry equal weights, so a large number of short-wavelength
modes cannot suppress the contribution of the largest resolved scales.
Weak reference signals use a fixed variance floor relative to fully neutral
gas, preventing almost ionized snapshots from dominating a relative loss.
"""

from __future__ import annotations

from functools import lru_cache
from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _fraction(value: ArrayLike, name: str) -> NDArray[np.float64]:
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise ValueError(f"{name} must contain finite fractions in [0, 1]")
    return result


def _control(value: float, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    if not np.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(
            f"{name} must be finite and {'positive' if positive else 'nonnegative'}"
        )
    return float(value)


@lru_cache(maxsize=16)
def _geometry(
    n: int, fraction: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    component = np.fft.fftfreq(n) * n
    norm = np.sqrt(
        component[:, None, None] ** 2
        + component[None, :, None] ** 2
        + component[None, None, :] ** 2
    )
    selected = (norm > 0) & (norm <= fraction * n / 2 + 1e-12)
    labels = np.floor(norm[selected] + 0.5).astype(np.int64)
    counts = np.bincount(labels)
    for array in (selected, norm, labels, counts):
        array.setflags(write=False)
    return selected, norm, labels, counts


def _fourier_component(
    prediction: NDArray[np.float64],
    teacher: NDArray[np.float64],
    capacity: NDArray[np.float64],
    box_len: float,
    fraction: float,
    spectral_power_floor: float,
) -> tuple[float, NDArray[np.float64], dict[str, Any]]:
    n = prediction.shape[0]
    selected, norm, labels, counts = _geometry(n, fraction)
    occupied = np.flatnonzero(counts > 0)
    # A common density/count-unit factor cancels from relative residuals. This
    # normalization prevents overflow without changing either loss or gradient.
    rho = capacity / np.max(capacity)
    true_brightness = rho * (1 - teacher)
    pred_brightness = rho * (1 - prediction)
    reference = np.fft.fftn(true_brightness - true_brightness.mean(), norm="ortho")
    difference = (
        np.fft.fftn(pred_brightness - pred_brightness.mean(), norm="ortho") - reference
    )
    powers = np.zeros(len(counts), dtype=np.float64)
    error_power = np.zeros_like(powers)
    powers[occupied] = (
        np.bincount(labels, weights=np.abs(reference[selected]) ** 2)[occupied]
        / counts[occupied]
    )
    error_power[occupied] = (
        np.bincount(labels, weights=np.abs(difference[selected]) ** 2)[occupied]
        / counts[occupied]
    )
    floor = max(
        64 * np.finfo(float).eps ** 2 * teacher.size * np.mean(true_brightness**2),
        np.finfo(float).tiny,
    )
    active = occupied[powers[occupied] > floor]
    neutral_mean_square = float(np.mean(rho**2))
    variance_floor = spectral_power_floor * neutral_mean_square
    # For the orthonormal FFT, a shell contributes counts*mean_power/Ncells
    # to real-space brightness variance. Floor this integrated variance,
    # rather than per-mode power, so the threshold has the same physical
    # meaning when the same mode is sampled on a finer grid.
    denominator = np.zeros_like(powers)
    denominator[occupied] = np.maximum(
        powers[occupied], variance_floor * teacher.size / counts[occupied]
    )
    regularized = active[denominator[active] > powers[active]]
    shell_weights = np.zeros_like(powers)
    shell_weights[active] = 1 / (len(active) * counts[active] * denominator[active])
    weighted_difference = np.zeros_like(difference)
    weighted_difference[selected] = difference[selected] * shell_weights[labels]
    gradient = -2 * rho * np.fft.ifftn(weighted_difference, norm="ortho").real
    component = (
        float(np.sum(error_power[active] / denominator[active]) / len(active))
        if len(active)
        else 0.0
    )
    centres = np.bincount(labels, weights=norm[selected])[occupied] / counts[occupied]
    return (
        component,
        gradient,
        {
            "spectral_loss": component if len(active) else None,
            "candidate_shell_count": len(occupied),
            "active_shell_count": len(active),
            "undefined_shell_count": int(len(occupied) - len(active)),
            "regularized_shell_count": len(regularized),
            "spectral_power_floor": spectral_power_floor,
            "neutral_reference_mean_square": neutral_mean_square,
            "shell_variance_floor": variance_floor,
            "shell_k_cMpc_inv": (centres * 2 * np.pi / box_len).tolist(),
            "shell_mode_count": counts[occupied].tolist(),
            "shell_teacher_variance": (
                counts[occupied] * powers[occupied] / teacher.size
            ).tolist(),
            "shell_objective_denominator_variance": (
                counts[occupied] * denominator[occupied] / teacher.size
            ).tolist(),
            "shell_objective_fourier_residual": [
                float(error_power[index] / denominator[index])
                if powers[index] > floor
                else None
                for index in occupied
            ],
            "shell_relative_fourier_residual": [
                float(error_power[index] / powers[index])
                if powers[index] > floor
                else None
                for index in occupied
            ],
            "k_cutoff_cMpc_inv": float(fraction * np.pi * n / box_len),
            "fourier_definition": "Equal-shell mean complex brightness residual variance / max(teacher shell variance, spectral_power_floor * fully-neutral mean-square brightness); mean-subtracted orthonormal full FFT; zero-reference shells undefined",
            "floor_definition": "Fully-neutral reference is rho=capacity/max(capacity); each shell variance is sum(abs(FFT_ortho)**2)/Ncells; default floor 1e-4 corresponds to 1% of neutral RMS brightness per shell, not an accuracy criterion",
        },
    )


def projected_objective(
    predicted_xHII: ArrayLike,
    teacher_xHII: ArrayLike,
    capacity: ArrayLike,
    box_len: float,
    *,
    spectral_weight: float = 0.1,
    spectral_power_floor: float = 1e-4,
    front_weight: float = 1.0,
    field_variance_floor: float = 0.01,
    k_fraction: float = 0.5,
) -> tuple[float, NDArray[np.float64], dict[str, Any]]:
    """Evaluate and differentiate a full-cube projected-field fitting objective.

    Parameters
    ----------
    predicted_xHII, teacher_xHII
        Predicted and reference ionized fractions on the same cubic grid.
    capacity
        Finite nonnegative atom capacities with positive total mass. The
        brightness proxy is capacity times xHI; its common cosmological factor
        cancels from the relative Fourier residual at a fixed redshift.
    box_len
        Periodic comoving box length in cMpc.
    spectral_weight
        Nonnegative coefficient of the relative complex Fourier residual.
    spectral_power_floor
        Nonnegative floor on each shell's integrated reference brightness
        variance, expressed as a fraction of the fully neutral density field's
        mean-square brightness. The default 1e-4 corresponds to 1% of neutral
        RMS brightness per shell. It is a fixed loss-conditioning scale, not
        an accuracy threshold. Zero restores the unfloored relative loss.
        Numerically zero-reference shells remain undefined and excluded.
    front_weight
        Extra field-loss weight for teacher cells adjacent, through any of six
        periodic neighbours, to the opposite side of xHII=0.5. Weights are
        normalized to mean one separately in each snapshot.
    field_variance_floor
        Positive lower bound on the teacher-field variance used to normalize
        the field MSE. This prevents trivial or nearly uniform snapshots from
        producing an unbounded relative objective.
    k_fraction
        Fraction in (0, 1] of the grid Nyquist wavenumber included in the Fourier
        objective. Shells have fundamental-mode width and receive equal weight.

    Returns
    -------
    loss
        Front-weighted normalized field MSE plus weighted Fourier residual.
    gradient
        Derivative of this loss with respect to predicted_xHII, before chaining
        through the global budget projection and network.
    diagnostics
        JSON-safe component losses, front fraction and shell diagnostics.
        Numerically zero-reference shells have null relative errors and are
        excluded; if all are undefined their contribution to fitting is zero.

    Notes
    -----
    Unlike a power-only loss, the complex Fourier residual also tests spatial
    phase. Teacher information is used solely to define the supervised target
    and objective; it is not an inference feature or a projected photon budget.
    """
    prediction = _fraction(predicted_xHII, "Prediction")
    teacher = _fraction(teacher_xHII, "Teacher")
    atoms = np.asarray(capacity, dtype=np.float64)
    if (
        prediction.ndim != 3
        or len(set(prediction.shape)) != 1
        or prediction.shape[0] < 2
        or teacher.shape != prediction.shape
        or atoms.shape != prediction.shape
    ):
        raise ValueError("Prediction, teacher and capacity must share one cubic shape")
    if not np.isfinite(atoms).all() or np.any(atoms < 0) or not np.any(atoms > 0):
        raise ValueError("Capacity must be finite, nonnegative and have positive mass")
    length = _control(box_len, "box_len", positive=True)
    spectral_weight = _control(spectral_weight, "spectral_weight")
    spectral_power_floor = _control(spectral_power_floor, "spectral_power_floor")
    front_weight = _control(front_weight, "front_weight")
    field_variance_floor = _control(
        field_variance_floor, "field_variance_floor", positive=True
    )
    k_fraction = _control(k_fraction, "k_fraction", positive=True)
    if k_fraction > 1:
        raise ValueError("k_fraction cannot exceed the Nyquist fraction of one")
    phase = teacher >= 0.5
    front = np.zeros(teacher.shape, dtype=bool)
    for axis in range(3):
        for direction in (-1, 1):
            front |= phase != np.roll(phase, direction, axis=axis)
    weights = 1 + front_weight * front
    weights /= weights.mean()
    scale = max(float(np.var(teacher)), field_variance_floor)
    error = prediction - teacher
    field_loss = float(np.mean(weights * error**2) / scale)
    gradient = 2 * weights * error / (teacher.size * scale)
    spectral, spectral_gradient, diagnostics = _fourier_component(
        prediction, teacher, atoms, length, k_fraction, spectral_power_floor
    )
    loss = field_loss + spectral_weight * spectral
    gradient += spectral_weight * spectral_gradient
    if not np.isfinite(loss) or not np.isfinite(gradient).all():
        raise FloatingPointError("Projected objective or derivative overflowed")
    diagnostics.update(
        total_loss=loss,
        field_loss=field_loss,
        field_MSE=float(np.mean(error**2)),
        field_variance_scale=scale,
        front_fraction=float(np.mean(front)),
        spectral_weight=spectral_weight,
        spectral_loss_contribution=spectral_weight * spectral,
    )
    return loss, gradient, diagnostics
