"""Verify controlled RT comparisons without treating new boxes as paired fields."""

import numpy as np
import pytest

from research.c2ray_teacher.compare import (
    common_endpoints,
    compare_states,
    neutral_spectrum,
    restrict_fields,
)


def run_fixture(n: int = 8, times: tuple = (1.0, 2.0), box: float = 50.0) -> dict:
    """Create a time series whose source-history identity is explicit."""
    shape = (len(times), n, n, n)
    return {
        "manifest": {
            "dimension": n,
            "native_dimension": 8,
            "refinement": n // 8,
            "box_len_cMpc": box,
            "native_array_sha256": {
                "rates": "fixed",
                "capacity": "fixed",
                "redshifts": "fixed",
                "times_seconds": "fixed",
            },
            "cosmology": {"h": 0.7},
            "initial_state": {"xHII": 1e-6},
            "thermal_evolution": True,
            "steps": [{"time_end_seconds": t, "redshift_end": 10 - t} for t in times],
        },
        "fields": {
            "density_total_cm3": np.ones(shape),
            "xHII": np.full(shape, 0.2),
            "temperature_K": np.full(shape, 100.0),
        },
    }


def test_common_endpoints_do_not_interpolate_away_temporal_error() -> None:
    """Only actually saved matching endpoints enter a convergence comparison."""
    left = run_fixture(times=(1.0, 2.0))
    right = run_fixture(times=(0.5, 1.0, 1.5, 2.0))
    assert common_endpoints(left["manifest"], right["manifest"], paired=True) == [
        (0, 1),
        (1, 3),
    ]


def test_restriction_preserves_ionized_atom_count() -> None:
    """A volume-only average would fail on nonuniform child densities."""
    n = 6
    rho = 1 + np.arange(n**3).reshape((n,) * 3) / 20
    ionized = np.linspace(0, 1, n**3).reshape((n,) * 3)
    fields = {"density_total_cm3": rho, "xHII": ionized, "temperature_K": 100 + rho}
    coarse = restrict_fields(fields, 2)
    assert coarse["xHII"].shape == (2, 2, 2)
    assert np.sum(coarse["density_total_cm3"] * coarse["xHII"]) * 27 == pytest.approx(
        np.sum(rho * ionized), rel=1e-13
    )
    assert coarse["xHII"][0, 0, 0] != pytest.approx(ionized[:3, :3, :3].mean())


def test_identical_refined_field_has_zero_paired_error() -> None:
    """Subdivision alone does not manufacture a morphology difference."""
    left = run_fixture()
    right = run_fixture(24)
    result = compare_states(left, right)
    assert result["comparison_kind"] == "paired_transport"
    assert result["snapshots"][0]["paired_errors"]["xHII"]["rms"] == 0
    assert result["convergence_certified"] is False


def test_same_seed_different_source_history_is_not_paired() -> None:
    """IC labels cannot override changed source/gas arrays."""
    left = run_fixture()
    right = run_fixture()
    right["manifest"]["native_array_sha256"]["rates"] = "different"
    result = compare_states(left, right)
    assert result["comparison_kind"] == "statistical_only"
    assert "paired_errors" not in result["snapshots"][0]
    assert "cross_correlation" not in result["snapshots"][0]["neutral_fraction_power"]


def test_different_boxes_are_statistical_even_with_matching_array_hashes() -> None:
    """A repeated random seed does not imply shared physical Fourier phases."""
    result = compare_states(run_fixture(), run_fixture(box=100))
    assert result["comparison_kind"] == "statistical_only"
    assert result["common_power_k_max_cMpc_inv"] == pytest.approx(np.pi * 8 / 100 / 2)


def test_single_mode_spectrum_has_physical_volume_normalization() -> None:
    """A unit sinusoid has variance one half and two complex Fourier modes."""
    n, length = 16, 16.0
    values = np.broadcast_to(
        np.sin(2 * np.pi * np.arange(n) / n)[:, None, None], (n,) * 3
    )
    spectrum = neutral_spectrum(values, length, np.array([0.3, 0.5]))
    assert spectrum["modes"][0] == 6
    assert spectrum["power_cMpc3"][0] * spectrum["modes"][
        0
    ] / length**3 == pytest.approx(0.5)


def test_paired_clock_mismatch_is_rejected() -> None:
    """Same-redshift outputs from inconsistent clocks cannot pass a paired test."""
    left, right = run_fixture(), run_fixture()
    right["manifest"]["steps"][0]["time_end_seconds"] = 20.0
    with pytest.raises(ValueError, match="clock"):
        compare_states(left, right)


def test_exploratory_flags_do_not_certify_empty_spectra() -> None:
    """Constant fields have no nonzero power ratio to certify."""
    left, right = run_fixture(), run_fixture()
    right["fields"]["temperature_K"] *= 1.1
    result = compare_states(left, right, diagnostic_thresholds=True)
    flags = result["snapshots"][0]["exploratory_difference_flags"]
    assert flags["mean_temperature_within_tolerance"] is False
    assert flags["mean_xHII_within_tolerance"] is True
    assert flags["power_within_tolerance"] is None
    assert result["convergence_certified"] is False
