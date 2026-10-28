"""Analytic and conservation checks for the direct radiation solver."""

from itertools import pairwise

import numpy as np
import pytest

from research.pc_matched.rt import LIGHT_SPEED_CM_S, MPC_CM, directions, evolve_interval


def _uniform_run(photon_total: float, coefficient_dt: float) -> dict:
    capacity = np.ones((4, 4, 4))
    photons = np.full((6, 4, 4, 4), photon_total / 6)
    zero = np.zeros_like(capacity)
    density, sigma = 1e-7, 1e-18
    return evolve_interval(
        capacity, zero, photons, zero, zero,
        dt_seconds=coefficient_dt / (LIGHT_SPEED_CM_S * density * sigma),
        z_start=0.0, z_end=0.0, box_len_cMpc=1e8,
        hydrogen_density_z0_cm3=density, cross_section_cm2=sigma,
    )


@pytest.mark.parametrize("count", [6, 26])
def test_angular_bank_is_positive_and_isotropic(count: int) -> None:
    """The quadrature has isotropic first and second angular moments."""
    vectors, weights = directions(count)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-15)
    np.testing.assert_allclose(weights @ vectors, 0, atol=1e-15)
    np.testing.assert_allclose((vectors.T * weights) @ vectors, np.eye(3) / 3, atol=1e-15)
    assert np.all(weights > 0)
    assert weights.sum() == pytest.approx(1.0)


@pytest.mark.parametrize(("photons", "coefficient", "absorbed"), [
    (1.0, 0.7, 0.7 / 1.7),
    (2.0, np.log(2), 2 / 3),
])
def test_homogeneous_coupled_absorption_exact(
    photons: float, coefficient: float, absorbed: float,
) -> None:
    """Compare against the exact nonlinear photon-neutral reaction solution."""
    result = _uniform_run(photons, coefficient)
    np.testing.assert_allclose(result["ionized"], absorbed, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(result["photons"].sum(axis=0), photons - absorbed, rtol=2e-14)
    assert result["ledger"]["passed"]


def test_nearly_equal_populations_and_tiny_absorption() -> None:
    """Small population differences do not cause catastrophic cancellation."""
    result = _uniform_run(1 + 1e-12, 0.7)
    np.testing.assert_allclose(result["ionized"], 0.7 / 1.7, atol=1e-12, rtol=0)
    tiny = _uniform_run(1.0, 1e-12)
    np.testing.assert_allclose(tiny["ionized"], 1e-12 / (1 + 1e-12), rtol=2e-14)


def test_transparent_periodic_upwind_matches_discrete_solution() -> None:
    """One physical light-crossing fraction matches the conservative FV stencil."""
    n, box = 8, 10.0
    capacity = np.ones((n, n, n))
    zero = np.zeros_like(capacity)
    vectors, _ = directions(6)
    beam = int(np.flatnonzero(np.all(vectors == [1, 0, 0], axis=1))[0])
    photons = np.zeros((6, n, n, n))
    photons[beam] = (1 + 0.5 * np.sin(2 * np.pi * np.arange(n) / n))[:, None, None]
    original = photons.copy()
    result = evolve_interval(
        capacity, zero, photons, zero, zero,
        dt_seconds=0.25 * (box / n) * MPC_CM / LIGHT_SPEED_CM_S,
        z_start=0, z_end=0, box_len_cMpc=box, cross_section_cm2=0,
    )
    expected = 0.75 * original[beam] + 0.25 * np.roll(original[beam], 1, axis=0)
    np.testing.assert_allclose(result["photons"][beam], expected, rtol=2e-15)
    np.testing.assert_array_equal(photons, original)
    np.testing.assert_array_equal(result["ionized"], zero)
    assert result["steps"] == 1
    assert result["ledger"]["passed"]


def test_source_integration_expansion_and_carried_radiation() -> None:
    """Linear emissivity integrates exactly and unabsorbed radiation survives."""
    capacity = np.ones((4, 4, 4))
    zero = np.zeros_like(capacity)
    duration = 1e13
    result = evolve_interval(
        capacity, zero, np.zeros((26, 4, 4, 4)),
        np.full_like(capacity, 2 / duration), np.full_like(capacity, 4 / duration),
        dt_seconds=duration, z_start=9, z_end=8, box_len_cMpc=10,
    )
    assert result["ledger"]["emitted"] == pytest.approx(3 * capacity.size)
    assert result["ledger"]["passed"]
    assert result["photons"].sum() > 0
    assert np.all(result["ionized"] <= capacity)
    np.testing.assert_allclose(result["ionized"], result["ionized"].flat[0], atol=2e-14)


def test_zero_radiation_stays_neutral() -> None:
    """Transport and chemistry do not create photons or ions."""
    capacity = np.ones((4, 4, 4))
    zero = np.zeros_like(capacity)
    result = evolve_interval(
        capacity, zero, np.zeros((6, 4, 4, 4)), zero, zero,
        dt_seconds=1e13, z_start=9, z_end=8, box_len_cMpc=10,
    )
    assert result["ledger"]["passed"]
    np.testing.assert_array_equal(result["ionized"], zero)
    assert result["photons"].sum() == 0


def test_nonuniform_point_source_keeps_bounds_and_budget() -> None:
    """Saturated cells cannot produce negative directional counts by roundoff."""
    rng = np.random.default_rng(42)
    capacity = np.exp(rng.normal(0, 0.2, (8, 8, 8)))
    capacity /= capacity.mean()
    zero = np.zeros_like(capacity)
    rate = zero.copy()
    rate[4, 4, 4] = 0.4 * capacity.sum() / 1e15
    result = evolve_interval(
        capacity, zero, np.zeros((26, 8, 8, 8)), rate, rate,
        dt_seconds=1e15, z_start=8, z_end=7.6, box_len_cMpc=50,
    )
    assert result["ledger"]["passed"]
    assert result["ledger"]["relative_residual"] < 1e-12
    assert result["photons"].min() >= 0
    assert result["ionized"].min() >= 0
    assert np.all(result["ionized"] <= capacity)


def test_invalid_input_rejected() -> None:
    """Negative sources and nonpositive CFL limits never reach the kernel."""
    capacity = np.ones((4, 4, 4))
    zero = np.zeros_like(capacity)
    kwargs = {"dt_seconds": 1e13, "z_start": 8, "z_end": 7, "box_len_cMpc": 10}
    with pytest.raises(ValueError, match="nonnegative"):
        evolve_interval(capacity, zero, np.zeros((6, 4, 4, 4)), -capacity, zero, **kwargs)
    with pytest.raises(ValueError, match="CFL"):
        evolve_interval(capacity, zero, np.zeros((6, 4, 4, 4)), zero, zero, cfl=1.1, **kwargs)


def test_reaction_semigroup_and_photon_limited_absorption() -> None:
    """The exact reaction agrees across subdivisions and preserves limiting stock."""
    capacity = np.full((4, 4, 4), 2.0)
    zero = np.zeros_like(capacity)
    radiation = np.full((6, 4, 4, 4), 1 / 6)
    duration = np.log(2) / (LIGHT_SPEED_CM_S * 1e-7 * 1e-18)
    options = {"z_start": 0, "z_end": 0, "box_len_cMpc": 1e8,
               "hydrogen_density_z0_cm3": 1e-7, "cross_section_cm2": 1e-18}
    whole = evolve_interval(capacity, zero, radiation, zero, zero, dt_seconds=duration, **options)
    first = evolve_interval(capacity, zero, radiation, zero, zero, dt_seconds=duration / 2, **options)
    second = evolve_interval(capacity, first["ionized"], first["photons"], zero, zero,
                             dt_seconds=duration / 2, **options)
    np.testing.assert_allclose(whole["ionized"], 2 / 3, atol=2e-14)
    np.testing.assert_allclose(second["ionized"], whole["ionized"], atol=2e-14)
    np.testing.assert_allclose(second["photons"], whole["photons"], atol=2e-14)


def test_step_budget_and_temporal_refinement() -> None:
    """CFL refinement changes discretization, never the total photon budget."""
    capacity = np.ones((8, 8, 8))
    zero = np.zeros_like(capacity)
    radiation = np.zeros((6, 8, 8, 8))
    vectors, _ = directions(6)
    beam = int(np.flatnonzero(np.all(vectors == [1, 0, 0], axis=1))[0])
    radiation[beam, 0, 0, 0] = 1
    options = {"dt_seconds": MPC_CM / LIGHT_SPEED_CM_S, "z_start": 0,
               "z_end": 0, "box_len_cMpc": 8, "cross_section_cm2": 0}
    coarse = evolve_interval(capacity, zero, radiation, zero, zero, cfl=0.5, **options)
    fine = evolve_interval(capacity, zero, radiation, zero, zero, cfl=0.25, **options)
    assert fine["steps"] > coarse["steps"]
    assert not np.array_equal(fine["photons"], coarse["photons"])
    assert coarse["photons"].sum() == pytest.approx(1, abs=1e-14)
    assert fine["photons"].sum() == pytest.approx(1, abs=1e-14)
    with pytest.raises(RuntimeError, match="substep budget"):
        evolve_interval(capacity, zero, radiation, zero, zero, maximum_steps=1, **options)


def test_continuum_streaming_and_attenuation_converge() -> None:
    """Recover the independently known attenuated traveling Fourier mode.

    Radiation is sufficiently weak that neutral depletion is below 1e-10.
    The continuum cell averages translate by c*t and decay as exp(-K*t).
    This tests the PDE limit, rather than reproducing the discrete stencil.
    """
    vectors, _ = directions(6)
    beam = int(np.flatnonzero(np.all(vectors == [1, 0, 0], axis=1))[0])
    box, flight_fraction, attenuation = 10.0, 0.25, 0.7
    duration = flight_fraction * box * MPC_CM / LIGHT_SPEED_CM_S
    sigma = 1e-18
    density = attenuation / (LIGHT_SPEED_CM_S * sigma * duration)
    errors = []
    for n in (8, 16, 32):
        capacity = np.ones((n, n, n))
        zero = np.zeros_like(capacity)
        photons = np.zeros((6, n, n, n))
        centres = (np.arange(n) + 0.5) / n
        amplitude = 0.3 * np.sinc(1 / n)
        photons[beam] = (1e-10 * (1 + amplitude * np.sin(2 * np.pi * centres)))[:, None, None]
        exact = (1e-10 * np.exp(-attenuation)
                 * (1 + amplitude * np.sin(2 * np.pi * (centres - flight_fraction))))[:, None, None]
        result = evolve_interval(
            capacity, zero, photons, zero, zero, dt_seconds=duration,
            z_start=0, z_end=0, box_len_cMpc=box,
            hydrogen_density_z0_cm3=density, cross_section_cm2=sigma, cfl=0.4,
        )
        actual = result["photons"][beam]
        errors.append(float(np.mean(np.abs(actual - exact)) / np.mean(exact)))
        assert actual.mean() == pytest.approx(float(exact.mean()), rel=1e-9, abs=0)
        assert result["ledger"]["passed"]
    assert 0.05 < errors[0] < 0.07
    assert errors[-1] < 0.018
    for coarse, fine in pairwise(errors):
        assert 0.45 < fine / coarse < 0.60
