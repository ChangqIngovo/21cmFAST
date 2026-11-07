"""Project a latent field onto a bounded global atom count with an exact VJP.

The forward map minimizes ``sum(capacity * (q - raw)**2) / 2`` subject to
``sum(capacity * q) = target`` and ``0 <= q <= 1``. Its solution is a common
additive shift followed by clipping. This is a global constraint, not transport.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _inputs(
    values: ArrayLike, capacity: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    raw = np.asarray(values, dtype=np.float64)
    atoms = np.asarray(capacity, dtype=np.float64)
    if raw.size == 0 or raw.shape != atoms.shape:
        raise ValueError("Projection inputs must have the same nonempty shape")
    if not np.isfinite(raw).all():
        raise ValueError("Projection inputs must be finite")
    if not np.isfinite(atoms).all() or np.any(atoms < 0):
        raise ValueError("Atom capacities must be finite and nonnegative")
    return raw, atoms


def _interior_solution(
    raw: np.ndarray, capacity: np.ndarray, target: np.longdouble
) -> tuple[np.ndarray, float, int]:
    # F(lambda) = sum c*clip(raw+lambda, 0, 1) is piecewise linear. The
    # breakpoints are -raw and 1-raw; accumulate its slope between them.
    latent = raw.astype(np.longdouble)
    atoms = capacity.astype(np.longdouble)
    breakpoints = np.concatenate((-latent, 1 - latent))
    changes = np.concatenate((atoms, -atoms))
    order = np.argsort(breakpoints, kind="stable")
    sorted_breakpoints = breakpoints[order]
    starts = np.flatnonzero(
        np.concatenate(([True], sorted_breakpoints[1:] != sorted_breakpoints[:-1]))
    )
    locations = sorted_breakpoints[starts]
    changes = np.add.reduceat(changes[order], starts)
    slopes = np.maximum(np.cumsum(changes, dtype=np.longdouble), 0)
    counts = np.concatenate(
        (
            np.zeros(1, dtype=np.longdouble),
            np.cumsum(slopes[:-1] * np.diff(locations), dtype=np.longdouble),
        )
    )
    upper = int(np.searchsorted(counts, target, side="left"))
    upper = min(max(upper, 1), len(locations) - 1)
    lower = upper - 1
    if slopes[lower] <= 0:
        raise FloatingPointError("Could not locate an active projection interval")
    # Keep the breakpoint and small offset separate. Forming raw+lambda
    # directly would erase a tiny positive count when lambda is nearly -raw.
    centered = latent + locations[lower]
    offset = (target - counts[lower]) / slopes[lower]
    solution = np.clip(centered + offset, 0, 1)
    iterations = 0
    for _ in range(8):
        active = (solution > 0) & (solution < 1)
        if not np.any(active):
            break
        iterations += 1
        offset = (
            target
            - np.sum(atoms[solution >= 1], dtype=np.longdouble)
            - np.sum(atoms[active] * centered[active], dtype=np.longdouble)
        ) / np.sum(atoms[active], dtype=np.longdouble)
        updated = np.clip(centered + offset, 0, 1)
        if np.array_equal(active, (updated > 0) & (updated < 1)):
            solution = updated
            break
        solution = updated
    return solution.astype(np.float64), float(locations[lower] + offset), iterations


def project_capacity_l2(
    raw: ArrayLike,
    capacity: ArrayLike,
    target: float,
    *,
    rtol: float = 1e-13,
    atol: float = 0.0,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Find the nearest bounded field in the capacity-weighted L2 metric.

    Parameters
    ----------
    raw
        Finite latent ionized fractions. Values outside [0, 1] are allowed.
    capacity
        Nonnegative atom capacities with the same nonempty shape as raw.
        Zero-capacity cells receive output zero by convention.
    target
        Desired absorbed atom count, between zero and total atom capacity.
        At inference this must come from input physics or a budget predictor.
    rtol, atol
        Relative-to-target and absolute tolerances for the count audit.

    Returns
    -------
    corrected
        Bounded fractions with the requested capacity-weighted count.
    diagnostics
        JSON-safe count, additive-shift and active-set diagnostics.

    Notes
    -----
    With the correct teacher count, this projection cannot increase the
    capacity-weighted squared distance from the actual latent input to that
    teacher. This is not a comparison to the separately clipped raw diagnostic.
    It also does not apply to a wrong count or to unweighted field error.
    """
    latent, atoms = _inputs(raw, capacity)
    target = float(target)
    if not np.isfinite(target) or target < 0:
        raise ValueError("Target count must be finite and nonnegative")
    if not np.isfinite(rtol) or not np.isfinite(atol) or rtol < 0 or atol < 0:
        raise ValueError("Projection tolerances must be finite and nonnegative")
    total = np.sum(atoms, dtype=np.longdouble)
    if not np.isfinite(float(total)) or target > float(total):
        raise ValueError("Target count exceeds the supported atom capacity")
    requested = np.longdouble(target)
    positive = atoms > 0
    corrected = np.zeros_like(latent)
    shift, iterations = 0.0, 0
    if np.any(positive):
        if target == 0:
            shift = -float(np.max(latent[positive]))
        elif target == float(total):
            corrected[positive] = 1
            shift = 1 - float(np.min(latent[positive]))
        else:
            corrected[positive], shift, iterations = _interior_solution(
                latent[positive], atoms[positive], requested
            )
    count = np.sum(
        np.multiply(atoms, corrected, dtype=np.longdouble), dtype=np.longdouble
    )
    residual = float(count - requested)
    tolerance = atol + rtol * target
    rounding = 16 * np.finfo(float).eps * max(target, float(count))
    if abs(residual) > tolerance + rounding:
        raise FloatingPointError("Capacity-L2 projection failed its photon count audit")
    raw_count = np.sum(
        np.multiply(atoms, np.clip(latent, 0, 1), dtype=np.longdouble),
        dtype=np.longdouble,
    )
    active = positive & (corrected > 0) & (corrected < 1)
    return corrected, {
        "method": "capacity_l2",
        "scope": "capacity-weighted L2 nearest field at one global count; no local transport",
        "objective_reference": "unbounded latent input; raw diagnostic fractions may be separately clipped",
        "target_count": target,
        "atom_capacity": float(total),
        "raw_count": float(raw_count),
        "corrected_count": float(count),
        "count_residual": residual,
        "relative_count_residual": residual / target if target else 0.0,
        "additive_shift": shift,
        "iterations": iterations,
        "active_cells": int(np.count_nonzero(active)),
        "latent_minimum": float(np.min(latent)),
        "latent_maximum": float(np.max(latent)),
        "gradient_at_all_saturated": "zero subgradient; auxiliary raw/budget supervision required to leave this boundary",
    }


def capacity_l2_vjp(
    corrected: ArrayLike, capacity: ArrayLike, gradient: ArrayLike
) -> tuple[NDArray[np.float64], float]:
    """Propagate an output gradient through a capacity-weighted L2 projection.

    Parameters
    ----------
    corrected
        Bounded output of project_capacity_l2.
    capacity
        Atom capacities used by that projection.
    gradient
        Derivative of a scalar loss with respect to corrected, same shape.

    Returns
    -------
    grad_raw
        Vector-Jacobian product with the unconstrained latent field.
    grad_target
        Derivative with respect to the target atom count.

    Notes
    -----
    The map is differentiable within each fixed active set. Exact clipped
    boundaries use the inactive branch; an all-saturated field returns zero
    for both derivatives. This explicit subgradient convention is not a
    derivative across a change in the active set or outside the feasible count.
    """
    values, atoms = _inputs(corrected, capacity)
    cotangent = np.asarray(gradient, dtype=np.float64)
    if cotangent.shape != values.shape or not np.isfinite(cotangent).all():
        raise ValueError("Projection gradient must be finite and match the field")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("Corrected projection fractions must lie in [0,1]")
    active = (atoms > 0) & (values > 0) & (values < 1)
    result = np.zeros_like(values)
    if not np.any(active):
        return result, 0.0
    total = np.sum(atoms[active], dtype=np.longdouble)
    target_gradient = np.sum(cotangent[active], dtype=np.longdouble) / total
    result[active] = (
        cotangent[active].astype(np.longdouble) - atoms[active] * target_gradient
    ).astype(np.float64)
    if not np.isfinite(result).all() or not np.isfinite(float(target_gradient)):
        raise FloatingPointError("Capacity-L2 projection gradient overflowed")
    return result, float(target_gradient)
