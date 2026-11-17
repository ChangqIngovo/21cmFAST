"""Freeze validation-based emulator routing with an explicit accuracy contract.

The contract is empirical on the supplied validation realizations. Routing uses
only exact native geometry and redshift; a photon ledger is not an accuracy
certificate. Unknown or failed combinations request the unchanged teacher.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from research.pc_analytic_emulator.data import validate_geometries

THRESHOLDS = {
    "late_neutral_mass_max": 0.05,
    "active_MAE_max": 0.03,
    "active_absolute_delta_Q_mass_max": 0.005,
    "active_mean_relative_power_error_max": 0.10,
    "late_MAE_max": 0.01,
    "late_absolute_delta_Q_mass_max": 0.002,
    "brightness_residual_rms_over_neutral_rms_max": 0.01,
    "k_fraction": 0.5,
}


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _key(geometry: dict[str, Any], redshift: float) -> str:
    geometry = validate_geometries([geometry])[0]
    if isinstance(redshift, bool) or not math.isfinite(redshift) or redshift < 0:
        raise ValueError("Redshift must be finite and nonnegative")
    return json.dumps(
        [geometry, float(redshift)], sort_keys=True, separators=(",", ":")
    )


def _acceptance(metric: dict[str, Any], margin: float = 1.0) -> dict[str, bool]:
    late = metric["teacher_xHI_mass"] <= THRESHOLDS["late_neutral_mass_max"]
    prefix = "late" if late else "active"
    checks = {
        "field": metric["MAE"] <= margin * THRESHOLDS[f"{prefix}_MAE_max"],
        "budget": abs(metric["delta_Q_mass"])
        <= margin * THRESHOLDS[f"{prefix}_absolute_delta_Q_mass_max"],
    }
    if late:
        checks["absolute_brightness"] = (
            metric["brightness_residual_rms_over_neutral_rms"]
            <= margin * THRESHOLDS["brightness_residual_rms_over_neutral_rms_max"]
        )
    else:
        error = metric["mean_relative_power_error"]
        checks["relative_power"] = (
            error is None
            or error <= margin * THRESHOLDS["active_mean_relative_power_error_max"]
        )
        if metric["undefined_power_shells"]:
            checks["zero_reference_brightness"] = (
                metric["brightness_residual_rms_over_neutral_rms"]
                <= margin * THRESHOLDS["brightness_residual_rms_over_neutral_rms_max"]
            )
    return checks


def accuracy_metrics(
    predicted_xHI: ArrayLike,
    teacher_xHI: ArrayLike,
    capacity: ArrayLike,
    box_len: float,
) -> dict[str, Any]:
    """Measure field, budget and resolved brightness errors against a teacher.

    Parameters
    ----------
    predicted_xHI, teacher_xHI
        Neutral-fraction cubes on the same grid, bounded in [0, 1].
    capacity
        Positive atom capacities proportional to the native density.
    box_len
        Positive comoving box length in cMpc.

    Returns
    -------
    dict
        Fixed-threshold acceptance, individual measurements and shell counts.
        Power ratios are undefined at numerically zero teacher power. The
        low-k complex brightness residual still includes those modes. Its
        RMS is normalized by fully neutral RMS brightness, not teacher RMS.
    """
    predicted = np.asarray(predicted_xHI, dtype=float)
    teacher = np.asarray(teacher_xHI, dtype=float)
    atoms = np.asarray(capacity, dtype=float)
    if (
        predicted.ndim != 3
        or len(set(predicted.shape)) != 1
        or predicted.shape[0] < 8
        or teacher.shape != predicted.shape
        or atoms.shape != predicted.shape
        or not np.isfinite(atoms).all()
        or np.any(atoms <= 0)
        or not all(
            np.isfinite(a).all() and np.all((a >= 0) & (a <= 1))
            for a in (predicted, teacher)
        )
        or isinstance(box_len, bool)
        or not math.isfinite(box_len)
        or box_len <= 0
    ):
        raise ValueError("Accuracy audit requires valid common cubes with N>=8")
    # Scaling both brightness fields and the neutral reference identically
    # cancels physical units and the saturated-spin cosmological coefficient.
    rho = atoms / np.max(atoms)
    predicted_brightness = rho * predicted
    teacher_brightness = rho * teacher
    n = predicted.shape[0]
    fft_teacher = np.fft.fftn(teacher_brightness, norm="ortho")
    fft_prediction = np.fft.fftn(predicted_brightness, norm="ortho")
    component = np.fft.fftfreq(n) * n
    radius = np.sqrt(
        sum(a**2 for a in np.meshgrid(component, component, component, indexing="ij"))
    )
    selected = (radius > 0) & (radius < THRESHOLDS["k_fraction"] * n / 2)
    labels = np.floor(radius[selected] + 0.5).astype(int)
    counts = np.bincount(labels)
    occupied = np.flatnonzero(counts)
    teacher_power = np.bincount(labels, weights=np.abs(fft_teacher[selected]) ** 2)
    prediction_power = np.bincount(
        labels, weights=np.abs(fft_prediction[selected]) ** 2
    )
    teacher_power[occupied] /= counts[occupied]
    prediction_power[occupied] /= counts[occupied]
    numerical_floor = max(
        64
        * np.finfo(float).eps ** 2
        * teacher.size
        * float(np.mean(teacher_brightness**2)),
        np.finfo(float).tiny,
    )
    defined = occupied[teacher_power[occupied] > numerical_floor]
    power_error = (
        float(np.mean(np.abs(prediction_power[defined] / teacher_power[defined] - 1)))
        if len(defined)
        else None
    )
    residual_rms = float(
        np.sqrt(
            np.sum(np.abs(fft_prediction[selected] - fft_teacher[selected]) ** 2)
            / teacher.size
            / np.mean(rho**2)
        )
    )
    neutral_mass = float(np.sum(rho * teacher) / np.sum(rho))
    delta_q = float(np.sum(rho * (teacher - predicted)) / np.sum(rho))
    mae = float(np.mean(np.abs(predicted - teacher)))
    late = neutral_mass <= THRESHOLDS["late_neutral_mass_max"]
    phase = "late" if late else "active_or_early"
    metric = {
        "schema": 1,
        "thresholds_sha256": _digest(THRESHOLDS),
        "phase": phase,
        "teacher_xHI_mass": neutral_mass,
        "MAE": mae,
        "delta_Q_mass": delta_q,
        "mean_relative_power_error": power_error,
        "brightness_residual_rms_over_neutral_rms": residual_rms,
        "defined_power_shells": len(defined),
        "undefined_power_shells": int(len(occupied) - len(defined)),
        "k_max_cMpc_inv": float(THRESHOLDS["k_fraction"] * np.pi * n / box_len),
    }
    checks = _acceptance(metric)
    return {**metric, "checks": checks, "passed": bool(all(checks.values()))}


def build_policy(
    records: list[dict[str, Any]],
    model_hashes: dict[str, dict[str, str]],
    validation_seeds: list[int],
    *,
    minimum_seeds: int = 2,
    safety_margin: float = 1.0,
) -> dict[str, Any]:
    """Build a model-bound routing policy using validation records only.

    Parameters
    ----------
    records
        Rows with split='validation', candidate, geometry, redshift, seed and
        metrics returned by accuracy_metrics. Every candidate must have the
        same validation seeds at each geometry/redshift combination.
    model_hashes
        Candidate names mapped to complete model artifact path/SHA256 maps.
    validation_seeds
        Explicit seed allowlist fixed before fitting; test seeds are excluded.
    minimum_seeds
        At least two independent realizations per routing combination.
    safety_margin
        Multiplicative margin in (0, 1] applied to validation error limits.
        Reported metric acceptance retains the full declared limits.

    Returns
    -------
    dict
        Hashed policy. A route uses the smallest mean field MAE among candidates
        for which every validation realization passes; other routes use teacher.
    """
    if (
        isinstance(minimum_seeds, bool)
        or not isinstance(minimum_seeds, int)
        or minimum_seeds < 2
    ):
        raise ValueError("At least two independent validation seeds are required")
    if (
        isinstance(safety_margin, bool)
        or not math.isfinite(safety_margin)
        or not 0 < safety_margin <= 1
    ):
        raise ValueError("Validation safety margin must be in (0, 1]")
    if not records:
        raise ValueError("Validation records are required")
    if not model_hashes or any(not value for value in model_hashes.values()):
        raise ValueError("Every candidate needs its complete model hash map")
    allowed = set(validation_seeds)
    if len(allowed) != len(validation_seeds) or any(
        isinstance(s, bool) or not isinstance(s, int) for s in allowed
    ):
        raise ValueError("Validation seeds must be unique integers")
    groups: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    identities = {}
    for row in records:
        if row.get("split") != "validation" or row.get("seed") not in allowed:
            raise ValueError(
                "Only explicitly allowed validation seeds may select routes"
            )
        candidate = row["candidate"]
        if candidate not in model_hashes:
            raise ValueError("Validation candidate has no frozen model hash map")
        metric = row["metrics"]
        if metric.get("thresholds_sha256") != _digest(THRESHOLDS):
            raise ValueError("Validation metrics use different accuracy thresholds")
        if not math.isfinite(metric["MAE"]) or metric["MAE"] < 0:
            raise ValueError("Invalid validation MAE")
        if metric.get("passed") != bool(all(metric["checks"].values())):
            raise ValueError("Validation acceptance differs from its component checks")
        if metric["checks"] != _acceptance(metric):
            raise ValueError("Validation acceptance differs from measured errors")
        key = _key(row["geometry"], row["redshift"])
        seeds = groups.setdefault(key, {}).setdefault(candidate, {})
        if row["seed"] in seeds:
            raise ValueError("Duplicate validation realization")
        seeds[row["seed"]] = metric
        identities[key] = {
            "geometry": validate_geometries([row["geometry"]])[0],
            "redshift": float(row["redshift"]),
        }
    routes = {}
    for key, candidates in sorted(groups.items()):
        if set(candidates) != set(model_hashes):
            raise ValueError(
                "Candidates must share validation geometry/redshift coverage"
            )
        coverage = [set(rows) for rows in candidates.values()]
        if any(seeds != coverage[0] for seeds in coverage):
            raise ValueError("Candidates must be compared on the same validation seeds")
        eligible = []
        for candidate, rows in candidates.items():
            if len(rows) >= minimum_seeds and all(
                all(_acceptance(m, safety_margin).values()) for m in rows.values()
            ):
                eligible.append(
                    (float(np.mean([m["MAE"] for m in rows.values()])), candidate)
                )
        selected_candidate = min(eligible)[1] if eligible else None
        routes[key] = {
            **identities[key],
            "method": "emulator" if selected_candidate else "teacher",
            "candidate": selected_candidate,
            "validation_seeds": sorted(coverage[0]),
            "reason": "all_validation_realizations_pass"
            if selected_candidate
            else "no_candidate_passes_all_validation_realizations",
        }
    result = {
        "schema": 1,
        "thresholds": dict(THRESHOLDS),
        "model_hashes": model_hashes,
        "validation_seeds": sorted(allowed),
        "minimum_seeds": minimum_seeds,
        "safety_margin": float(safety_margin),
        "validation_records_sha256": _digest(records),
        "routes": routes,
        "scope": "Empirical validation routing, not an error guarantee for unseen realizations; exact geometry and redshift only",
    }
    return {**result, "policy_sha256": _digest(result)}


def select_route(
    policy: dict[str, Any],
    geometry: dict[str, Any],
    redshift: float,
    model_hashes: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Select an audited route without reading teacher labels.

    Parameters
    ----------
    policy
        Frozen policy produced by build_policy.
    geometry, redshift
        Native simulation geometry and exact requested coeval redshift.
    model_hashes
        Freshly recomputed hashes of all candidate model artifacts.

    Returns
    -------
    dict
        Emulator candidate or explicit teacher fallback with its reason.
        Altered policy, thresholds or model bytes raise rather than silently
        applying obsolete validation evidence.
    """
    payload = {key: value for key, value in policy.items() if key != "policy_sha256"}
    if policy.get("policy_sha256") != _digest(payload):
        raise ValueError("Frozen routing policy changed")
    if policy.get("thresholds") != THRESHOLDS or policy.get("schema") != 1:
        raise ValueError("Unsupported routing accuracy contract")
    if policy.get("model_hashes") != model_hashes:
        raise ValueError("Model artifacts differ from frozen validation policy")
    route = policy["routes"].get(_key(geometry, redshift))
    if route is None:
        return {
            "method": "teacher",
            "candidate": None,
            "reason": "geometry_or_redshift_not_validated",
        }
    return dict(route)
