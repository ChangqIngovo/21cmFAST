"""Recompute the coeval photon budget from saved arrays, independently of solvers.

One absorbed photon ionizes one hydrogen atom; gas is fixed and recombinations
are disabled. This is not an audit of any model with extra sinks or initial ions.
Saved xHI fields certify the completed allocation target min(Q, A), not an
unwritten photon reservoir. Input-inferred surplus is always labelled as such.
"""
from __future__ import annotations

import math

import numpy as np


def _number(value):
    """JSON-safe number, including invalid output arrays and extreme totals."""
    with np.errstate(over="ignore", invalid="ignore"):
        result = float(value)
    return result if math.isfinite(result) else None


def _relative(numerator, denominator):
    if denominator == 0:
        return 0.0 if numerator == 0 else None
    return _number(numerator / denominator)


def _inputs(source, capacity):
    source = np.asarray(source, dtype=np.float64)
    capacity = np.asarray(capacity, dtype=np.float64)
    if (source.ndim != 3 or source.shape != capacity.shape
            or len(set(source.shape)) != 1 or source.size == 0):
        raise ValueError("Source and capacity must have the same nonempty cubic shape")
    if not np.isfinite(source).all() or (source < 0).any():
        raise ValueError("Source counts must be finite and nonnegative")
    if not np.isfinite(capacity).all() or (capacity <= 0).any():
        raise ValueError("Atom capacities must be finite and strictly positive")
    photons = np.sum(source, dtype=np.longdouble)
    atoms = np.sum(capacity, dtype=np.longdouble)
    if photons > np.finfo(float).max or atoms > np.finfo(float).max:
        raise ValueError("Input totals exceed the JSON float audit range")
    return source, capacity, photons, atoms


def _field_audit(source, capacity, xhi, *, tolerance, remaining_source=None):
    source, capacity, photons, atoms = _inputs(source, capacity)
    xhi = np.asarray(xhi, dtype=np.float64)
    if xhi.shape != source.shape:
        raise ValueError("Neutral field shape differs from the source grid")
    finite = bool(np.isfinite(xhi).all())
    bounds = bool(finite and (xhi >= 0).all() and (xhi <= 1).all())
    # Long-double multiplication as well as accumulation: do not repeat the
    # allocator's float64 ion-count arithmetic or read its claimed residual.
    with np.errstate(invalid="ignore", over="ignore"):
        ions = np.sum(capacity.astype(np.longdouble) *
                      (np.longdouble(1) - xhi.astype(np.longdouble)), dtype=np.longdouble)
        neutral = np.sum(capacity.astype(np.longdouble) * xhi.astype(np.longdouble),
                         dtype=np.longdouble)
    target = min(photons, atoms)
    surplus = max(photons - atoms, np.longdouble(0))
    # Equivalent to Q-I-max(Q-A,0), but stable even if Q >> A.
    defect = target - ions
    # Saving xHI=1-qi rounds away sub-epsilon ionized fractions. This explicit
    # floor is numerical representation uncertainty, not extra physical photons.
    roundoff_floor = np.longdouble(64 * np.finfo(float).eps) * atoms
    allowance = np.longdouble(tolerance) * target + roundoff_floor
    allocation_passed = bool(bounds and np.isfinite(defect) and abs(defect) <= allowance)
    reservoir = {
        "available": False, "passed": None,
        "reason": "No independently saved remaining_source array; surplus inferred from inputs is not a measured reservoir",
    }
    if remaining_source is not None:
        remaining = np.asarray(remaining_source, dtype=np.float64)
        if remaining.shape != source.shape:
            raise ValueError("Remaining-source shape differs from the source grid")
        reservoir_bounds = bool(np.isfinite(remaining).all() and
                                (remaining >= 0).all() and (remaining <= source).all())
        left = np.sum(remaining, dtype=np.longdouble)
        residual = photons - ions - left
        reservoir_passed = bool(bounds and reservoir_bounds and np.isfinite(residual)
                                and abs(residual) <= tolerance * photons + roundoff_floor)
        reservoir = {
            "available": True, "passed": reservoir_passed,
            "remaining_photons_recomputed": _number(left),
            "per_source_bounds_passed": reservoir_bounds,
            "signed_photon_defect": _number(residual),
            "relative_photon_defect": _relative(abs(residual), photons),
            "absolute_defect_per_atom": _relative(abs(residual), atoms),
            "scope": "Independent saved emitter reservoir plus field-recomputed ionized atoms; no inferred surplus added",
        }
    return {
        "field_finite": finite, "field_bounds_passed": bounds,
        "minimum_xHI": _number(np.min(xhi)), "maximum_xHI": _number(np.max(xhi)),
        "ionized_atoms_recomputed": _number(ions), "neutral_atoms_recomputed": _number(neutral),
        "xHI_mass_recomputed": _relative(neutral, atoms),
        "xHI_volume_recomputed": _number(np.mean(xhi, dtype=np.longdouble)),
        "xHI_mass_analytic": _relative(atoms - target, atoms),
        "ionization_target_from_inputs": _number(target),
        "input_inferred_surplus_photons": _number(surplus),
        "surplus_is_independently_observed": False,
        "signed_photon_defect_with_inferred_surplus": _number(defect),
        "relative_photon_defect_with_inferred_surplus": _relative(abs(defect), photons),
        "relative_allocation_defect": _relative(abs(defect), target),
        "absolute_defect_per_atom": _relative(abs(defect), atoms),
        "signed_defect_convention": "Positive means fewer ionized atoms than the completed no-sink budget permits",
        "tolerance": tolerance, "allowed_absolute_allocation_defect": _number(allowance),
        "field_storage_roundoff_allowance": _number(roundoff_floor),
        "zero_source_relative_error": "0 for exact zero defect; null for nonzero defect when Q=0",
        "allocation_budget_passed": allocation_passed,
        "stored_reservoir_ledger": reservoir,
    }


def audit_snapshot(source, capacity, fields, *, remaining_source=None,
                   pc_tolerance=1e-8, script_tolerance=2e-5):
    """Audit arrays; ``fields`` maps pc/script/native_es to neutral fractions.

    PC and SCRIPT must meet the same completed ionization target; each uses its
    documented arithmetic tolerance. ES is diagnostic, since its approximate
    allocation and optional initial residual ionization need not meet that target.
    ``remaining_source`` is optional and, if present, is the independent PC
    emitter reservoir, not ``source - reconstructed_ions`` calculated by caller.
    No input or output field is modified, clipped or rescaled.
    """
    source, capacity, photons, atoms = _inputs(source, capacity)
    for name, value in (("pc_tolerance", pc_tolerance), ("script_tolerance", script_tolerance)):
        if isinstance(value, bool) or not math.isfinite(value) or not 0 < value < 1:
            raise ValueError(f"{name} must be finite and between zero and one")
    required = {"pc", "script", "native_es"}
    if set(fields) != required:
        raise ValueError("Require precisely pc, script, native_es neutral fields")
    methods = {}
    for name in ("pc", "script", "native_es"):
        tolerance = float(pc_tolerance if name == "pc" else script_tolerance)
        methods[name] = _field_audit(source, capacity, fields[name], tolerance=tolerance,
                                    remaining_source=remaining_source if name == "pc" else None)
        methods[name]["required_to_conserve"] = name != "native_es"
    required_passed = all(methods[name]["allocation_budget_passed"] for name in ("pc", "script"))
    stored_passed = methods["pc"]["stored_reservoir_ledger"]["passed"]
    return {
        "schema": 1,
        "scope": "Frozen-gas coeval stellar photon budget; one photon per ionization; no recombinations or other sinks",
        "calculation": "Independent long-double field integration: I=sum(capacity*(1-xHI)); target=min(sum(source),sum(capacity))",
        "input": {"shape": list(source.shape), "source_photons": _number(photons),
                  "atom_capacity": _number(atoms), "photons_per_atom": _relative(photons, atoms)},
        "methods": methods,
        "pc_and_script_allocation_passed": bool(required_passed),
        "independent_reservoir_audit_available": remaining_source is not None,
        "all_available_required_checks_passed": bool(required_passed and stored_passed is not False),
        "native_es_caveat": "ES is a descriptive budget diagnostic; any RECFAST residual ions are outside this stellar-only target",
        "conservation_does_not_certify_morphology": True,
    }
