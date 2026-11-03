"""Independent limiting solutions and shared-budget checks for absorption."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from research.pc_analytic.absorption import absorbed, shared_absorption


@pytest.mark.parametrize("opacity", [0.01, 0.1, 1, 10, 100])
@pytest.mark.parametrize("photons", [0.001, 0.01, 0.1, 1, 10, 100])
def test_column_matches_integrated_population_equation(
    opacity: float, photons: float
) -> None:
    """Recover a separately integrated population-depletion equation."""
    # Independently integrate dH/dP = -(1-exp(-k H)), rather than comparing
    # two rearrangements of the implemented closed form.
    numerical = solve_ivp(
        lambda p, h: [np.expm1(-opacity * max(h[0], 0.0))],
        (0, photons),
        [1.0],
        rtol=2e-11,
        atol=2e-13,
    )
    actual = absorbed(photons, 1.0, opacity)
    assert actual == pytest.approx(1 - numerical.y[0, -1], abs=5e-11)
    assert 0 <= actual <= min(photons, 1.0)


def test_extreme_small_absorption_and_opaque_capacity() -> None:
    """Keep weak absorption and the opaque capacity limit numerically stable."""
    assert absorbed(1.0, 1.0, 1e-20) == pytest.approx(1e-20, rel=1e-14)
    assert absorbed(1e-100, 1e-100, 1.0) == pytest.approx(1e-200, rel=1e-14, abs=0)
    assert absorbed(1e100, 1e100, 1e300) == 1e100
    assert absorbed(2.0, 1.0, 1e6) == 1.0
    assert absorbed(0.5, 1.0, 1e6) == 0.5


def test_weak_illumination_is_beer_lambert() -> None:
    """Recover fixed-opacity Beer-Lambert attenuation for weak illumination."""
    photons = 1e-9
    for tau in (0.01, 0.5, 10.0):
        assert absorbed(photons, 1, tau) / photons == pytest.approx(
            -np.expm1(-tau), rel=1e-8
        )


def test_ray_segmentation_and_pulse_subdivision() -> None:
    """Preserve the column solution under spatial and exposure subdivision."""
    for photons in (0.01, 1.0, 10.0):
        whole = absorbed(photons, 2.0, 0.7)
        first = absorbed(photons, 0.8, 0.7)
        second = absorbed(photons - first, 1.2, 0.7)
        assert first + second == pytest.approx(whole, abs=3e-15)
        early = absorbed(photons * 0.3, 2.0, 0.7)
        late = absorbed(photons * 0.7, 2.0 - early, 0.7)
        assert early + late == pytest.approx(whole, abs=3e-15)


def test_multiple_directions_share_atoms_without_direction_transfer() -> None:
    """All directions deplete one inventory without angular photon transfer."""
    photons = np.array([[2.0, 0.0], [1.0, 3.0], [0.0, 0.0]])
    neutral = np.array([1.0, 2.0])
    surviving, left = shared_absorption(photons, neutral, 1000.0)
    assert np.all(surviving >= 0)
    assert np.all(left >= 0)
    np.testing.assert_allclose(
        photons.sum(axis=0) - surviving.sum(axis=0), neutral - left
    )
    np.testing.assert_array_equal(surviving[2], 0)
    assert surviving[0, 0] / surviving[1, 0] == pytest.approx(2.0)


@pytest.mark.parametrize("value", [-1, np.nan, np.inf])
def test_invalid_inputs_rejected(value: float) -> None:
    """Reject negative and nonfinite photon inputs before evaluating exponentials."""
    with pytest.raises(ValueError, match="finite and nonnegative"):
        absorbed(value, 1, 1)
