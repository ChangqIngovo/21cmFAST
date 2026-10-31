"""Check the shared-grid diagnostics of the SCRIPT refinement benchmark."""

import numpy as np

from research.pc_causal_rt.benchmarks.script_refinement_benchmark import (
    _field_comparison,
    _pair_metrics,
    _restrict_ionization,
)


def test_restrict_counts_before_ion_fraction():
    """Preserve atoms instead of averaging local neutral fractions."""
    capacity = np.ones((4, 4, 4))
    capacity[:, :, 1::2] = 3.0
    fraction = np.zeros_like(capacity)
    fraction[:, :, 1::2] = 1.0
    atoms, ions, xhi = _restrict_ionization(capacity, fraction, 2)
    np.testing.assert_allclose(atoms, 2.0, rtol=0, atol=1e-15)
    np.testing.assert_allclose(ions, 1.5, rtol=0, atol=1e-15)
    np.testing.assert_allclose(xhi, 0.25, rtol=0, atol=1e-15)
    _, _, empty = _restrict_ionization(np.zeros_like(capacity), fraction, 2)
    np.testing.assert_array_equal(empty, np.ones((2, 2, 2)))


def test_spectrum_denominator_and_unavailable_stage():
    """Use finer-reference power and report missing stages without extrapolation."""
    wave = np.broadcast_to(np.arange(4.0)[:, None, None], (4, 4, 4)).copy()
    fine = {"xhi": np.full_like(wave, 0.9), "brightness": wave}
    coarse = {"xhi": np.full_like(wave, 0.9), "brightness": 2 * wave}
    comparison = _field_comparison(coarse, fine, 50.0)
    np.testing.assert_allclose(
        comparison["power_fractional_error"][0], 3.0, rtol=0, atol=1e-14
    )
    row = {"node": 0, "redshift": 9.0, "xHI_mass": 0.9}
    pair = _pair_metrics([(row, coarse)], [(row, fine)], 50.0)
    assert len(pair["same_time_active"]) == 1
    assert all(not m["available"] for m in pair["matched_mean_xHI"])
