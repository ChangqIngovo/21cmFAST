"""Check native brightness reconstruction and held-out field diagnostics."""

import json
from typing import Any

import numpy as np
import pytest

from research.pc_analytic_emulator.evaluation import (
    evaluate_prediction,
    native_brightness,
)

INPUTS = {
    "cosmo_params": {"OMb": 0.049, "OMm": 0.31, "hlittle": 0.68},
    "astro_options": {"USE_TS_FLUCT": False},
}


def test_brightness_matches_native_saturated_formula() -> None:
    """Retain the native cosmological normalization in mK."""
    xhi = np.full((4, 4, 4), 0.4)
    capacity = np.linspace(0.5, 1.5, 64).reshape(4, 4, 4)
    factor = (
        27 * (0.049 * 0.68**2 / 0.023) * np.sqrt(0.15 / (0.31 * 0.68**2) * 8.5 / 10)
    )
    np.testing.assert_allclose(
        native_brightness(xhi, capacity, 7.5, INPUTS),
        factor * xhi * capacity,
        rtol=1e-15,
        atol=0,
    )


def test_brightness_rejects_spin_temperature_physics() -> None:
    """Do not apply saturated-spin reconstruction to thermal-evolution inputs."""
    fields = np.ones((4, 4, 4))
    inputs = dict(INPUTS, astro_options={"USE_TS_FLUCT": True})
    with pytest.raises(ValueError, match="USE_TS_FLUCT"):
        native_brightness(fields, fields, 7.5, inputs)


def sample_arrays() -> dict[str, np.ndarray]:
    """Make nonuniform fields with consistently normalized brightness."""
    capacity = np.linspace(0.5, 1.5, 64).reshape(4, 4, 4)
    teacher = np.linspace(0.1, 0.9, 64).reshape(4, 4, 4)
    es = np.clip(teacher + 0.03, 0, 1)
    source = capacity * 0.6
    unabsorbed = np.sum(source) - np.sum(capacity * (1 - teacher))
    return {
        "source": source,
        "capacity": capacity,
        "teacher_xHI": teacher,
        "native_es_xHI": es,
        "native_brightness": native_brightness(es, capacity, 7.5, INPUTS).astype(
            np.float32
        ),
        "teacher_brightness": native_brightness(teacher, capacity, 7.5, INPUTS).astype(
            np.float32
        ),
        "remaining_photons": np.full_like(source, unabsorbed / 2 / source.size),
        "escaped_by_direction": np.full(8, unabsorbed / 2 / 8),
        "unemitted_source": np.zeros_like(source),
    }


def sample_prediction(
    arrays: dict[str, np.ndarray], *, neutral_offset: float = 0
) -> dict[str, Any]:
    """Make a deliberately controlled output with an independent global budget."""
    xhi = arrays["teacher_xHI"] + neutral_offset
    absorbed = float(np.sum(arrays["capacity"] * (1 - xhi)))
    source = float(np.sum(arrays["source"]))
    capacity = float(np.sum(arrays["capacity"]))
    return {
        "raw_xHI": arrays["teacher_xHI"].copy(),
        "xHI": xhi,
        "photon_ledger": {"passed": True},
        "absorption_efficiency": absorbed / min(source, capacity) if source else 1,
        "predicted_absorbed_photons": absorbed,
        "predicted_unabsorbed_photons": source - absorbed,
        "projection": {"log_odds_shift": 0.25, "target_count": absorbed},
    }


def test_evaluation_has_physical_brightness_and_same_shells() -> None:
    """Compare all models on identical Fourier shells and original inputs."""
    arrays = sample_arrays()
    prediction = sample_prediction(arrays)
    result = evaluate_prediction(prediction, arrays, 7.5, 256, INPUTS)
    assert result["metrics"]["constrained"]["MAE"] == 0
    assert result["metrics"]["native_es"]["MAE"] == pytest.approx(0.03)
    assert result["brightness_audit"]["passed"]
    assert result["spectra_xHI"]["k_cMpc_inv"] == result["spectra_21cm"]["k_cMpc_inv"]
    assert all(
        value == pytest.approx(1)
        for value in result["spectra_21cm"]["ratio_constrained"]
        if value is not None
    )


def test_changed_brightness_convention_is_rejected() -> None:
    """Refuse a source archive with a different brightness normalization."""
    arrays = sample_arrays()
    arrays["native_brightness"] *= 1.1
    prediction = sample_prediction(arrays)
    with pytest.raises(ValueError, match="brightness"):
        evaluate_prediction(prediction, arrays, 7.5, 256, INPUTS)


def test_projection_and_teacher_budget_errors_are_reported() -> None:
    """Expose morphology corrections and budget error despite a closed ledger."""
    arrays = sample_arrays()
    prediction = sample_prediction(arrays, neutral_offset=0.05)
    result = evaluate_prediction(prediction, arrays, 7.5, 256, INPUTS)
    assert result["projection"]["raw_to_constrained_MAE"] == pytest.approx(0.05)
    assert result["projection"]["raw_to_constrained_maximum_absolute_change"] == (
        pytest.approx(0.05)
    )
    assert result["projection"]["log_odds_shift"] == 0.25
    budget = result["budget_prediction"]
    assert budget["independent_teacher_reservoirs_available"]
    assert budget["absorbed_photon_error"] == pytest.approx(-0.05 * 64)
    assert budget["unabsorbed_photon_error"] == pytest.approx(0.05 * 64)
    assert budget["absorbed_error_over_source"] == pytest.approx(-0.05 / 0.6)
    assert budget["absorption_efficiency_error"] == pytest.approx(-0.05 / 0.6)
    assert set(budget["teacher_reservoir_components"]) == {
        "remaining_photons",
        "escaped_by_direction",
        "unemitted_source",
    }
    json.dumps(result, allow_nan=False)


def test_zero_source_budget_has_explicit_undefined_relative_error() -> None:
    """Return JSON null rather than NaN when no photons define the denominator."""
    arrays = {
        "source": np.zeros((4, 4, 4)),
        "capacity": np.ones((4, 4, 4)),
        "teacher_xHI": np.ones((4, 4, 4)),
        "native_es_xHI": np.ones((4, 4, 4)),
    }
    result = evaluate_prediction(sample_prediction(arrays), arrays, 7.5, 256, INPUTS)
    budget = result["budget_prediction"]
    assert budget["absorbed_error_over_source"] is None
    assert budget["unabsorbed_error_over_source"] is None
    assert budget["teacher_absorption_efficiency"] == 1
    assert not budget["independent_teacher_reservoirs_available"]
    assert budget["teacher_unabsorbed_photons"] == 0
    json.dumps(result, allow_nan=False)
