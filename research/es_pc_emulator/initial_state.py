"""Check whether a fresh RT history starts before appreciable ES ionization.

This is a screening test for omitted pre-start history. Passing does not prove
that the ES proxy and RT model have identical physical histories.
"""

from __future__ import annotations

import numpy as np

DEFAULT_POLICY = {"max_mass_weighted_error": 1e-3, "allow_mismatch": False}


def validate_policy(value=None):
    """Normalize a recorded policy without accepting misspelled settings."""
    if value is None:
        value = {}
    if not isinstance(value, dict) or set(value) - set(DEFAULT_POLICY):
        raise ValueError("Unknown initial_state policy settings")
    result = DEFAULT_POLICY | value
    tolerance = result["max_mass_weighted_error"]
    if (isinstance(tolerance, bool) or not isinstance(tolerance, (int, float))
            or not np.isfinite(tolerance) or not 0 < tolerance < 1):
        raise ValueError("Initial-state maximum mass-weighted error must lie in (0,1)")
    if type(result["allow_mismatch"]) is not bool:
        raise ValueError("Initial-state allow_mismatch must be boolean")
    return result


def initial_state_audit(history, es, *, initial_xhii=1e-6,
                        max_mass_weighted_error=1e-3):
    """Compare the initial ES field to the actual uniform RT initial fraction."""
    policy = validate_policy({"max_mass_weighted_error": max_mass_weighted_error})
    if (isinstance(initial_xhii, bool) or not np.isscalar(initial_xhii)
            or not np.isfinite(initial_xhii) or not 0 <= initial_xhii <= 1):
        raise ValueError("Initial RT hydrogen fraction must lie in [0,1]")
    xhi = np.asarray(es["xhi_es"][0], dtype=np.float64)
    rho = np.asarray(history["capacity"], dtype=np.float64)
    native_rho = np.asarray(es["density_es"][0], dtype=np.float64)
    if (xhi.shape != rho.shape or native_rho.shape != rho.shape
            or not np.isfinite(xhi).all() or not np.isfinite(rho).all()
            or not np.isfinite(native_rho).all() or (rho <= 0).any()
            or (native_rho <= 0).any() or (xhi < 0).any() or (xhi > 1).any()):
        raise ValueError("Invalid initial ES fractions or density weights")
    ionized = 1.0 - xhi
    difference = ionized - initial_xhii
    total = rho.sum(dtype=np.float64)
    error = float(np.sum(rho * np.abs(difference), dtype=np.float64) / total)
    return {
        "initial_redshift": float(history["redshifts"][0]),
        "initial_es_Q_volume": float(ionized.mean()),
        "initial_es_Q_mass_reference_gas": float(np.sum(rho * ionized) / total),
        "initial_es_Q_mass_native_gas": float(np.sum(native_rho * ionized) / native_rho.sum()),
        "teacher_initial_xHII": float(initial_xhii),
        "initial_delta_Q_mass": float(np.sum(rho * difference) / total),
        "mass_weighted_initial_MAE": error,
        "max_mass_weighted_error": policy["max_mass_weighted_error"],
        "compatible": error <= policy["max_mass_weighted_error"],
        "definition": "Initial ES versus uniform RT xHII, weighted by teacher reference nuclei",
        "interpretation": "Missing pre-start history screen; not a matching-physics or convergence certificate",
    }


def require_initial_state(audit, *, allow_mismatch=False):
    """Reject a mismatched fresh RT initial state unless explicitly exploratory."""
    if type(allow_mismatch) is not bool:
        raise ValueError("Initial-state allow_mismatch must be boolean")
    if not audit["compatible"] and not allow_mismatch:
        raise ValueError(
            "Initial ES/RT state mismatch: mass-weighted MAE="
            f"{audit['mass_weighted_initial_MAE']:.6g} exceeds "
            f"{audit['max_mass_weighted_error']:.6g} at z={audit['initial_redshift']:g}. "
            "Start the RT history earlier; initial_state.allow_mismatch=true is "
            "reserved for explicitly exploratory comparisons."
        )
