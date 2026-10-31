"""Analytic and invariant checks of the higher-order photon transport."""

from itertools import pairwise

import numpy as np
import pytest

from research.pc_causal_rt.transport import (
    ION_BOUND_TOLERANCE,
    LIGHT_SPEED_CM_S,
    MPC_CM,
    directions,
    evolve_interval,
)


def _options(**updates):
    return {
        "dt_seconds": 0.3 * MPC_CM / LIGHT_SPEED_CM_S,
        "z_start": 0.0,
        "z_end": 0.0,
        "box_len_cMpc": 8.0,
        "cross_section_cm2": 0.0,
        **updates,
    }


def _beam_index(vector):
    vectors, _ = directions(6)
    return int(np.flatnonzero(np.all(vectors == vector, axis=1))[0])


def test_smooth_transport_and_absorption_second_order():
    """Compare cell averages with a translated attenuated Fourier mode."""
    beam = _beam_index([1, 0, 0])
    box, flight, attenuation = 10.0, 0.25, 0.7
    duration = flight * box * MPC_CM / LIGHT_SPEED_CM_S
    sigma = 1e-18
    density = attenuation / (LIGHT_SPEED_CM_S * sigma * duration)
    errors = []
    for n in (8, 16, 32):
        capacity = np.ones((n, n, n))
        zero = np.zeros_like(capacity)
        radiation = np.zeros((6, n, n, n))
        centres = (np.arange(n) + 0.5) / n
        amplitude = 0.3 * np.sinc(1 / n)
        radiation[beam] = (1e-10 * (1 + amplitude * np.sin(2 * np.pi * centres)))[
            :, None, None
        ]
        exact = (
            1e-10
            * np.exp(-attenuation)
            * (1 + amplitude * np.sin(2 * np.pi * (centres - flight)))
        )[:, None, None]
        result = evolve_interval(
            capacity,
            zero,
            radiation,
            zero,
            zero,
            dt_seconds=duration,
            z_start=0,
            z_end=0,
            box_len_cMpc=box,
            hydrogen_density_z0_cm3=density,
            cross_section_cm2=sigma,
        )
        errors.append(
            float(np.mean(np.abs(result["photons"][beam] - exact)) / np.mean(exact))
        )
        assert result["photons"][beam].mean() == pytest.approx(
            float(exact.mean()), rel=1e-9, abs=0
        )
        assert result["ledger"]["passed"]
    assert errors[1] < 0.01
    assert errors[2] < 0.003
    for coarse, fine in pairwise(errors):
        assert fine < 0.36 * coarse


def test_oblique_three_dimensional_fourier_mode_converges():
    """A diagonal ordinate exercises all three fluxes and mixed derivatives."""
    velocity = directions(24)[0][5]
    box, flight = 10.0, 0.2
    errors = []
    for n in (8, 16, 32):
        x = (np.arange(n) + 0.5) / n
        phase = 2 * np.pi * (x[:, None, None] + x[None, :, None] + x[None, None, :])
        amplitude = 0.25 * np.sinc(1 / n) ** 3
        capacity = np.ones((n, n, n))
        zero = np.zeros_like(capacity)
        photons = np.zeros((24, n, n, n))
        photons[5] = 1 + amplitude * np.cos(phase)
        exact = 1 + amplitude * np.cos(phase - 2 * np.pi * flight * velocity.sum())
        result = evolve_interval(
            capacity,
            zero,
            photons,
            zero,
            zero,
            **_options(
                dt_seconds=flight * box * MPC_CM / LIGHT_SPEED_CM_S, box_len_cMpc=box
            ),
        )
        errors.append(float(np.mean(np.abs(result["photons"][5] - exact))))
    assert errors[-1] < 0.0015
    assert all(fine < 0.36 * coarse for coarse, fine in pairwise(errors))


@pytest.mark.parametrize("ordinates", [6, 24, 26, 96, 384])
def test_discontinuous_positive_transport_conserves_every_ordinate(ordinates):
    """A sharp pulse cannot borrow negative photons from adjacent voids."""
    n = 5
    capacity = np.zeros((n, n, n))
    radiation = np.zeros((ordinates, n, n, n))
    radiation[:, 2, 2, 2] = np.linspace(0.1, 1, ordinates)
    zero = np.zeros_like(capacity)
    result = evolve_interval(
        capacity,
        zero,
        radiation,
        zero,
        zero,
        **_options(dt_seconds=4 * MPC_CM / LIGHT_SPEED_CM_S, cfl=0.5),
    )
    assert np.all(result["photons"] >= 0)
    np.testing.assert_allclose(
        result["photons"].sum(axis=(1, 2, 3)),
        radiation.sum(axis=(1, 2, 3)),
        rtol=2e-14,
        atol=0,
    )
    np.testing.assert_array_equal(result["ionized"], 0)


def test_sharp_oblique_pulse_obeys_vacuum_maximum_principle():
    """A translated sharp cube cannot develop an intensity above its initial maximum."""
    n, ordinates, beam = 16, 24, 5
    box, flight = 10.0, 0.2
    vacuum = np.zeros((n, n, n))
    radiation = np.zeros((ordinates, n, n, n))
    radiation[beam, n // 4 : n // 2, n // 4 : n // 2, n // 4 : n // 2] = 1.0
    result = evolve_interval(
        vacuum,
        vacuum,
        radiation,
        vacuum,
        vacuum,
        **_options(
            dt_seconds=flight * box * MPC_CM / LIGHT_SPEED_CM_S,
            box_len_cMpc=box,
            cfl=0.4,
        ),
    )
    photons = result["photons"]
    assert np.all(photons >= 0)
    assert photons.max() <= radiation.max() + 2e-13
    np.testing.assert_allclose(
        photons.sum(axis=(1, 2, 3)), radiation.sum(axis=(1, 2, 3)), rtol=2e-14, atol=0
    )
    np.testing.assert_array_equal(result["ionized"], 0)


def test_distant_bright_pulse_does_not_permit_overshoot_at_a_weaker_front():
    """A global intensity bound must not hide local sharp-front overshoots."""
    n, ordinates, beam = 32, 24, 5
    box, flight = 10.0, 0.1
    vacuum = np.zeros((n, n, n))
    radiation = np.zeros((ordinates, n, n, n))
    radiation[beam, 4:8, 4:8, 4:8] = 1.0
    radiation[beam, 20:24, 20:24, 20:24] = 10.0
    result = evolve_interval(
        vacuum,
        vacuum,
        radiation,
        vacuum,
        vacuum,
        **_options(
            dt_seconds=flight * box * MPC_CM / LIGHT_SPEED_CM_S,
            box_len_cMpc=box,
            cfl=0.4,
        ),
    )
    photons = result["photons"]
    # The bright cube starts at 5/8 of the box on every axis. No component
    # can travel farther than 0.1 box, so its exact support stays above 1/2
    # without periodic wrapping. The lower octant can contain only the
    # unit-amplitude pulse; a bound based on the distant maximum 10 is too weak.
    assert photons[beam, : n // 2, : n // 2, : n // 2].max() <= 1.0 + 2e-12
    assert np.all(photons >= 0)
    assert photons.max() <= radiation.max() + 2e-12
    np.testing.assert_allclose(
        photons.sum(axis=(1, 2, 3)), radiation.sum(axis=(1, 2, 3)), rtol=2e-14, atol=0
    )
    np.testing.assert_array_equal(result["ionized"], 0)


def test_no_transverse_spreading_for_cartesian_beam():
    """Confine a Cartesian beam to its original transverse grid line."""
    n = 16
    capacity = np.ones((n, n, n))
    zero = np.zeros_like(capacity)
    radiation = np.zeros((6, n, n, n))
    beam = _beam_index([1, 0, 0])
    radiation[beam, 4, 6, 8] = 1
    result = evolve_interval(
        capacity,
        zero,
        radiation,
        zero,
        zero,
        **_options(dt_seconds=MPC_CM / LIGHT_SPEED_CM_S),
    )
    transverse = result["photons"][beam].sum(axis=0)
    assert transverse[6, 8] == pytest.approx(1, abs=1e-14)
    transverse[6, 8] = 0
    np.testing.assert_array_equal(transverse, 0)


def test_isotropic_source_preserves_cubic_symmetry():
    """Preserve cubic rotations of a centrally emitted isotropic pulse."""
    n, ordinates = 7, 24
    _, weights = directions(ordinates)
    capacity = np.ones((n, n, n))
    zero = np.zeros_like(capacity)
    source = zero.copy()
    source[3, 3, 3] = 10
    radiation = weights[:, None, None, None] * source
    result = evolve_interval(
        capacity,
        zero,
        radiation,
        zero,
        zero,
        **_options(dt_seconds=2 * MPC_CM / LIGHT_SPEED_CM_S),
    )
    summed = result["photons"].sum(axis=0)
    for axes in ((0, 1), (0, 2), (1, 2)):
        np.testing.assert_allclose(
            summed, np.rot90(summed, axes=axes), rtol=5e-14, atol=1e-15
        )


def test_exact_coupled_reaction_and_persistent_ion_state():
    """Recover the analytic coupled reaction and its interval composition."""
    capacity = np.full((4, 4, 4), 3.0)
    initial_ions = np.ones_like(capacity)
    zero = np.zeros_like(capacity)
    radiation = np.full((6, 4, 4, 4), 1 / 6)
    duration = np.log(2) / (LIGHT_SPEED_CM_S * 1e-7 * 1e-18)
    options = _options(
        box_len_cMpc=1e8,
        hydrogen_density_z0_cm3=1e-7,
        cross_section_cm2=1e-18,
        dt_seconds=duration,
    )
    result = evolve_interval(capacity, initial_ions, radiation, zero, zero, **options)
    np.testing.assert_allclose(result["ionized"], 1 + 2 / 3, rtol=0, atol=2e-14)
    np.testing.assert_allclose(result["photons"].sum(axis=0), 1 / 3, rtol=0, atol=2e-14)
    half = evolve_interval(
        capacity,
        initial_ions,
        radiation,
        zero,
        zero,
        **(options | {"dt_seconds": duration / 2}),
    )
    second = evolve_interval(
        capacity,
        half["ionized"],
        half["photons"],
        zero,
        zero,
        **(options | {"dt_seconds": duration / 2}),
    )
    np.testing.assert_allclose(second["ionized"], result["ionized"], atol=2e-14)
    np.testing.assert_allclose(second["photons"], result["photons"], atol=2e-14)


def test_linear_sources_have_exact_trapezoid_and_inputs_are_unchanged():
    """Integrate linear emissivity exactly without modifying caller arrays."""
    rng = np.random.default_rng(71)
    n, nd = 5, 24
    capacity = rng.uniform(0.5, 2, size=(n, n, n))
    ions = 0.1 * capacity
    photons = rng.uniform(0, 0.2, size=(nd, n, n, n))
    first = rng.uniform(0, 1e-14, size=capacity.shape)
    last = rng.uniform(0, 3e-14, size=capacity.shape)
    arrays = capacity, ions, photons, first, last
    before = [a.copy() for a in arrays]
    dt = MPC_CM / LIGHT_SPEED_CM_S
    result = evolve_interval(
        *arrays, **_options(dt_seconds=dt, cross_section_cm2=1e-18)
    )
    expected = np.sum((first.astype(np.longdouble) + last) * (np.longdouble(dt) / 2))
    assert result["ledger"]["emitted"] == pytest.approx(float(expected), rel=2e-15)
    total_before = ions.sum(dtype=np.longdouble) + photons.sum(dtype=np.longdouble)
    total_after = result["ionized"].sum(dtype=np.longdouble) + result["photons"].sum(
        dtype=np.longdouble
    )
    assert float(total_after - total_before) == pytest.approx(
        float(expected), rel=2e-13
    )
    for actual, original in zip(arrays, before, strict=True):
        np.testing.assert_array_equal(actual, original)


def test_threads_are_bitwise_reproducible():
    """Produce identical radiation and ion counts across worker counts."""
    rng = np.random.default_rng(10)
    capacity = rng.uniform(0.2, 2, size=(6, 6, 6))
    zero = np.zeros_like(capacity)
    photons = rng.uniform(0, 0.3, size=(24, 6, 6, 6))
    options = _options(dt_seconds=MPC_CM / LIGHT_SPEED_CM_S, cross_section_cm2=1e-18)
    serial = evolve_interval(capacity, zero, photons, zero, zero, threads=1, **options)
    parallel = evolve_interval(
        capacity, zero, photons, zero, zero, threads=4, **options
    )
    np.testing.assert_array_equal(serial["photons"], parallel["photons"])
    np.testing.assert_array_equal(serial["ionized"], parallel["ionized"])


def test_expanding_opacity_time_quadrature_converges():
    """Uniform weak photons obey exp[-c sigma n0 integral (1+z)^3 dt]."""
    capacity = np.ones((8, 8, 8))
    zero = np.zeros_like(capacity)
    photons = np.full((6, 8, 8, 8), 1e-12 / 6)
    dt = 0.25 * 10 * MPC_CM / LIGHT_SPEED_CM_S
    z_start, z_end, sigma = 3.0, 0.0, 1e-18
    mean_scale_cubed = ((1 + z_start) ** 4 - (1 + z_end) ** 4) / (4 * (z_start - z_end))
    density = 0.7 / (LIGHT_SPEED_CM_S * sigma * dt * mean_scale_cubed)
    exact = 1e-12 * np.exp(-0.7)
    errors = []
    for cfl in (0.4, 0.2, 0.1):
        result = evolve_interval(
            capacity,
            zero,
            photons,
            zero,
            zero,
            dt_seconds=dt,
            z_start=z_start,
            z_end=z_end,
            box_len_cMpc=10,
            hydrogen_density_z0_cm3=density,
            cross_section_cm2=sigma,
            cfl=cfl,
        )
        errors.append(abs(float(result["photons"].sum(axis=0).mean()) / exact - 1))
    assert errors[-1] < 1e-4
    assert errors[1] < 0.3 * errors[0]
    assert errors[2] < 0.3 * errors[1]


def test_exhausted_radiation_tail_uses_cumulative_ledger():
    """Normalize a nearly exhausted radiation tail by the cumulative inventory."""
    capacity = np.ones((4, 4, 4))
    ions = capacity * (1 - 1e-9)
    zero = np.zeros_like(capacity)
    radiation = np.full((6, 4, 4, 4), 1e-25)
    result = evolve_interval(
        capacity, ions, radiation, zero, zero, **_options(cross_section_cm2=1e-14)
    )
    assert result["ledger"]["passed"]
    assert result["ledger"]["normalization"] > 60


def test_fully_ionized_moving_clouds_compose_with_transport_without_clipping():
    """Gas/ion sums may differ by ulps after identical positive transfers."""
    from research.pc_causal_rt.gas import deposit, remap_ions

    rng = np.random.default_rng(2061)
    n = 6
    mass = rng.uniform(0.5, 1.5, size=8 * n**3)
    positions = rng.uniform(0, n, size=(len(mass), 3))
    capacity = deposit(mass, positions, n)
    ions = capacity.copy()
    radiation = rng.uniform(0, 1e-3, size=(24, n, n, n))
    initial_ion_count = ions.sum(dtype=np.longdouble)
    initial_radiation_count = radiation.sum(dtype=np.longdouble)
    for _ in range(4):
        new_positions = positions + rng.normal(0, 0.08, size=positions.shape)
        moved = remap_ions(ions, mass, positions, new_positions, n)
        zero = np.zeros_like(capacity)
        carried = moved["ionized"].copy()
        result = evolve_interval(
            moved["capacity"],
            moved["ionized"],
            radiation,
            zero,
            zero,
            **_options(cross_section_cm2=1e-18),
        )
        # No gas-density reset or clipping is used to satisfy the RT guard.
        assert (
            np.max(np.abs(result["ionized"] - carried))
            < 2e-13 * moved["capacity"].max()
        )
        assert result["ledger"]["passed"]
        ions, radiation, positions = result["ionized"], result["photons"], new_positions
    total = ions.sum(dtype=np.longdouble) + radiation.sum(dtype=np.longdouble)
    assert float(
        total / (initial_ion_count + initial_radiation_count)
    ) == pytest.approx(1.0, abs=2e-14)


@pytest.mark.parametrize("fraction_excess", [np.finfo(float).eps, 1e-12, 1e-11])
def test_roundoff_excess_is_preserved_and_physical_overionization_is_rejected(
    fraction_excess,
):
    """Preserve permitted atom roundoff while rejecting physical overionization."""
    capacity = np.ones((4, 4, 4))
    ions = capacity * (1 + fraction_excess)
    zero = np.zeros_like(capacity)
    radiation = np.ones((6, 4, 4, 4)) * 0.01
    result = evolve_interval(
        capacity, ions, radiation, zero, zero, **_options(cross_section_cm2=1e-18)
    )
    np.testing.assert_array_equal(result["ionized"], ions)
    assert result["ledger"]["bound_tolerance"] == ION_BOUND_TOLERANCE
    with pytest.raises(ValueError, match="ion bounds"):
        evolve_interval(
            capacity,
            capacity * (1 + 1e-10),
            radiation,
            zero,
            zero,
            **_options(cross_section_cm2=1e-18),
        )


def test_long_fully_ionized_gas_history_can_continue_radiation_transport():
    """Small local summation errors accumulate despite an accurate global ledger."""
    from research.pc_causal_rt.gas import TOLERANCE, deposit, remap_ions

    assert TOLERANCE == ION_BOUND_TOLERANCE
    n, particle_mesh = 8, 12
    positions = np.indices((particle_mesh,) * 3).reshape(3, -1).T.astype(float)
    positions = positions * n / particle_mesh + 0.5
    rng = np.random.default_rng(717)
    mass = rng.uniform(0.7, 1.3, len(positions)) * n**3 / len(positions)
    motion = rng.normal(0, 0.0003, positions.shape)
    capacity = deposit(mass, positions, n, cloud_width=2)
    ions = capacity.copy()
    initial_total = ions.sum(dtype=np.longdouble)
    for _ in range(500):
        new_positions = positions + motion
        result = remap_ions(ions, mass, positions, new_positions, n, cloud_width=2)
        positions, ions, capacity = new_positions, result["ionized"], result["capacity"]
    assert abs(float(ions.sum(dtype=np.longdouble) / initial_total - 1)) < 1e-12
    over = ions > capacity
    assert np.max((ions - capacity) / capacity) <= ION_BOUND_TOLERANCE
    radiation = np.full((24, n, n, n), 1e-4)
    zero = np.zeros_like(capacity)
    rt = evolve_interval(
        capacity, ions, radiation, zero, zero, **_options(cross_section_cm2=1e-18)
    )
    np.testing.assert_array_equal(rt["ionized"][over], ions[over])
    assert rt["ledger"]["passed"]
    assert rt["ledger"]["maximum_fraction_excess"] <= rt["ledger"]["bound_tolerance"]


@pytest.mark.parametrize("invalid", [0.0, -0.1, 0.51, float("nan")])
def test_invalid_cfl_rejected(invalid):
    """Reject nonfinite Courant numbers and values outside the positive scheme."""
    capacity = np.ones((4, 4, 4))
    zero = np.zeros_like(capacity)
    with pytest.raises(ValueError, match=r"CFL|finite"):
        evolve_interval(
            capacity, zero, np.zeros((6, 4, 4, 4)), zero, zero, **_options(cfl=invalid)
        )


def test_budget_failure_does_not_mutate_initial_state():
    """Leave caller radiation unchanged when the allowed step count is exceeded."""
    capacity = np.ones((4, 4, 4))
    zero = np.zeros_like(capacity)
    radiation = np.ones((6, 4, 4, 4))
    before = radiation.copy()
    with pytest.raises(RuntimeError, match="substep budget"):
        evolve_interval(
            capacity,
            zero,
            radiation,
            zero,
            zero,
            **_options(dt_seconds=100 * MPC_CM / LIGHT_SPEED_CM_S, maximum_steps=1),
        )
    np.testing.assert_array_equal(radiation, before)
