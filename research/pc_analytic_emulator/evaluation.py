"""Evaluate fields with the teacher's native saturated-spin brightness law."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from research.pc_analytic_emulator.model import absorption_efficiency
from research.pc_field_emulator.evaluation import field_metrics, shell_spectra


def native_brightness(
    xHI: ArrayLike,
    capacity: ArrayLike,
    redshift: float,
    inputs: dict[str, Any],
) -> NDArray[np.float64]:
    """Reconstruct the native no-spin-temperature-evolution brightness field.

    Parameters
    ----------
    xHI, capacity
        Neutral fraction and native 1+delta on the same cubic output grid.
    redshift
        Coeval redshift.
    inputs
        Full native input dictionary, including cosmology and USE_TS_FLUCT.

    Returns
    -------
    ndarray
        Brightness in mK from the explicit BrightnessTemperatureBox.c formula.
        Its cosmological coefficient is computed from inputs, never fitted to
        teacher brightness. This is a formula reconstruction, not a native call.
    """
    if inputs.get("astro_options", {}).get("USE_TS_FLUCT") is not False:
        raise ValueError("Native brightness reconstruction requires USE_TS_FLUCT=False")
    neutral = np.asarray(xHI, dtype=np.float64)
    atoms = np.asarray(capacity, dtype=np.float64)
    if (
        neutral.ndim != 3
        or len(set(neutral.shape)) != 1
        or atoms.shape != neutral.shape
        or not np.isfinite(neutral).all()
        or np.any((neutral < 0) | (neutral > 1))
        or not np.isfinite(atoms).all()
        or np.any(atoms <= 0)
        or not np.isfinite(redshift)
        or redshift < 0
    ):
        raise ValueError("Invalid neutral, density or redshift brightness inputs")
    cosmology = inputs["cosmo_params"]
    baryons, matter, h = (float(cosmology[name]) for name in ("OMb", "OMm", "hlittle"))
    if any(not np.isfinite(value) or value <= 0 for value in (baryons, matter, h)):
        raise ValueError("Native brightness requires positive finite cosmology")
    coefficient = (
        27
        * (baryons * h**2 / 0.023)
        * np.sqrt(0.15 / (matter * h**2) * (1 + redshift) / 10)
    )
    return coefficient * neutral * atoms


def _brightness_audit(
    arrays: dict[str, np.ndarray],
    reconstructed: dict[str, np.ndarray],
) -> dict[str, Any]:
    checks = {}
    for name, stored_name in (
        ("native_es", "native_brightness"),
        ("teacher", "teacher_brightness"),
    ):
        if stored_name not in arrays:
            checks[name] = {"available": False, "passed": None}
            continue
        saved = np.asarray(arrays[stored_name], dtype=float)
        expected = reconstructed[name]
        if saved.shape != expected.shape or not np.isfinite(saved).all():
            raise ValueError("Saved native brightness has an invalid grid or values")
        passed = bool(np.allclose(saved, expected, rtol=3e-6, atol=2e-6))
        checks[name] = {
            "available": True,
            "passed": passed,
            "maximum_absolute_difference_mK": float(np.max(np.abs(saved - expected))),
        }
        if not passed:
            raise ValueError(
                "Saved brightness disagrees with the native saturated-spin formula"
            )
    return {
        "passed": True,
        "relative_tolerance": 3e-6,
        "absolute_tolerance_mK": 2e-6,
        "checks": checks,
    }


def _budget_comparison(
    prediction: dict[str, Any], arrays: dict[str, np.ndarray]
) -> dict[str, Any]:
    source = np.sum(arrays["source"], dtype=np.longdouble)
    absorbed = np.sum(
        np.multiply(arrays["capacity"], 1 - arrays["teacher_xHI"], dtype=np.longdouble),
        dtype=np.longdouble,
    )
    predicted_absorbed = float(prediction["predicted_absorbed_photons"])
    predicted_unabsorbed = float(prediction["predicted_unabsorbed_photons"])
    eta = absorption_efficiency(
        arrays["source"], arrays["capacity"], arrays["teacher_xHI"]
    )
    names = ("remaining_photons", "escaped_by_direction", "unemitted_source")
    components = {
        name: float(np.sum(arrays[name], dtype=np.longdouble))
        for name in names
        if name in arrays
    }
    independent = all(name in components for name in names)
    unabsorbed = (
        sum(np.sum(arrays[name], dtype=np.longdouble) for name in names)
        if independent
        else source - absorbed
    )
    absorbed_error = float(np.longdouble(predicted_absorbed) - absorbed)
    unabsorbed_error = float(np.longdouble(predicted_unabsorbed) - unabsorbed)
    return {
        "predicted_absorption_efficiency": float(prediction["absorption_efficiency"]),
        "teacher_absorption_efficiency": eta,
        "absorption_efficiency_error": float(prediction["absorption_efficiency"]) - eta,
        "predicted_absorbed_photons": predicted_absorbed,
        "teacher_field_absorbed_photons": float(absorbed),
        "absorbed_photon_error": absorbed_error,
        "absorbed_error_over_source": float(absorbed_error / source)
        if source
        else None,
        "predicted_unabsorbed_photons": predicted_unabsorbed,
        "teacher_unabsorbed_photons": float(unabsorbed),
        "unabsorbed_photon_error": unabsorbed_error,
        "unabsorbed_error_over_source": float(unabsorbed_error / source)
        if source
        else None,
        "teacher_reservoir_components": components,
        "teacher_unabsorbed_definition": "independently stored remaining + escaped + unemitted photons"
        if independent
        else "source minus field-derived absorption; independent reservoirs unavailable",
        "independent_teacher_reservoirs_available": independent,
        "scope": "teacher labels used only for evaluation; predicted unabsorbed photons do not separately resolve remaining radiation and escape",
    }


def evaluate_prediction(
    prediction: dict[str, Any],
    arrays: dict[str, np.ndarray],
    redshift: float,
    box_len: float,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    """Compare raw, budget-constrained and ES fields with one teacher snapshot.

    Parameters
    ----------
    prediction
        Complete output of Emulator.predict, including its preprojection budget.
    arrays
        Audited teacher arrays including source, capacity, teacher_xHI and
        native_es_xHI. Saved brightness fields, when present, are checked.
    redshift, box_len
        Snapshot redshift and comoving box length in cMpc.
    inputs
        Original native physical input dictionary for the brightness formula.

    Returns
    -------
    dict
        JSON-safe field errors, same-shell neutral/21-cm auto and cross spectra,
        budget diagnostics and a native-brightness convention check. Undefined
        zero-reference-power ratios remain null.
    """
    capacity = np.asarray(arrays["capacity"], dtype=float)
    fields = {
        "teacher": np.asarray(arrays["teacher_xHI"], dtype=float),
        "raw": np.asarray(prediction["raw_xHI"], dtype=float),
        "constrained": np.asarray(prediction["xHI"], dtype=float),
        "native_es": np.asarray(arrays["native_es_xHI"], dtype=float),
    }
    metrics = {
        name: field_metrics(1 - value, 1 - fields["teacher"], capacity)
        for name, value in fields.items()
    }
    brightness = {
        name: native_brightness(value, capacity, redshift, inputs)
        for name, value in fields.items()
    }
    audit = _brightness_audit(arrays, brightness)
    neutral_spectra = shell_spectra(fields, box_len)
    brightness_spectra = shell_spectra(brightness, box_len)
    for spectrum in (neutral_spectra, brightness_spectra):
        spectrum["below_half_nyquist"] = [
            k < 0.5 * spectrum["k_max_cMpc_inv"] for k in spectrum["k_cMpc_inv"]
        ]
        spectrum["reference"] = (
            "analytic photon-conserving teacher on identical native inputs"
        )
    correction = np.abs(fields["raw"] - fields["constrained"])
    projection = dict(prediction["projection"])
    projection.update(
        raw_to_constrained_MAE=float(np.mean(correction)),
        raw_to_constrained_maximum_absolute_change=float(np.max(correction)),
    )
    return {
        "metrics": metrics,
        "spectra_xHI": neutral_spectra,
        "spectra_21cm": brightness_spectra,
        "photon_ledger": prediction["photon_ledger"],
        "projection": projection,
        "budget_prediction": _budget_comparison(prediction, arrays),
        "brightness_audit": audit,
        "brightness_definition": "Native BrightnessTemperatureBox.c USE_TS_FLUCT=False formula, including OMb/OMm/h; capacity=1+delta; reconstructed independently from physical inputs",
        "comparison": "same redshift, density and sources; no matched-mean renormalization",
    }
