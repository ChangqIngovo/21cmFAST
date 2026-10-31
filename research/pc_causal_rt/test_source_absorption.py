"""Continuous emission must interact with absorption during the same step.

Homogeneous fields remove spatial streaming errors. An independent implicit
ODE reference then isolates stiff source/reaction coupling, including angular
memory, rather than testing only a photon ledger.
"""

from functools import cache

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from research.pc_causal_rt.transport import (
    LIGHT_SPEED_CM_S,
    MPC_CM,
    directions,
    evolve_interval,
)

N = 4
BOX = 25.0
REDSHIFT = 7.0
DENSITY = 1.9e-7
# Four cell crossing times: CFL0.4 normally gives ten streaming steps.
DURATION = 4 * (BOX / N) * MPC_CM / (LIGHT_SPEED_CM_S * (1 + REDSHIFT))


def _uniform(
    q0: float,
    q1: float,
    tau: float,
    *,
    neutral: float = 1.0,
    initial: tuple[float, ...] | np.ndarray | None = None,
    cfl: float = 0.4,
) -> tuple[np.ndarray, float]:
    initial = np.zeros(6) if initial is None else np.asarray(initial, float)
    capacity = np.ones((N, N, N))
    photons = np.broadcast_to(
        initial[:, None, None, None], (len(initial), N, N, N)
    ).copy()
    sigma = tau / (LIGHT_SPEED_CM_S * DENSITY * (1 + REDSHIFT) ** 3 * DURATION)
    result = evolve_interval(
        capacity,
        capacity * (1 - neutral),
        photons,
        capacity * q0 / DURATION,
        capacity * q1 / DURATION,
        dt_seconds=DURATION,
        z_start=REDSHIFT,
        z_end=REDSHIFT,
        box_len_cMpc=BOX,
        hydrogen_density_z0_cm3=DENSITY,
        cross_section_cm2=sigma,
        cfl=cfl,
    )
    # A homogeneous state must stay homogeneous even for anisotropic light.
    for field in result["photons"]:
        np.testing.assert_allclose(field, field[0, 0, 0], rtol=0, atol=1e-14)
    assert result["ledger"]["passed"]
    return result["photons"][:, 0, 0, 0], float(result["ionized"][0, 0, 0])


@cache
def _reference(
    q0: float,
    q1: float,
    tau: float,
    neutral: float = 1.0,
    initial: tuple[float, ...] = (0.0,) * 6,
) -> tuple[np.ndarray, float]:
    """Integrate directional photons and neutral atoms independently of C++."""
    weights = directions(len(initial))[1]

    def rhs(time: float, state: np.ndarray) -> np.ndarray:
        radiation, atoms = state[:-1], state[-1]
        absorption = tau * atoms * radiation
        return np.r_[weights * (q0 + (q1 - q0) * time) - absorption, -absorption.sum()]

    def jacobian(time: float, state: np.ndarray) -> np.ndarray:
        radiation, atoms = state[:-1], state[-1]
        matrix = np.zeros((len(state), len(state)))
        matrix[:-1, :-1] = -tau * atoms * np.eye(len(initial))
        matrix[:-1, -1] = -tau * radiation
        matrix[-1, :-1] = -tau * atoms
        matrix[-1, -1] = -tau * radiation.sum()
        return matrix

    solution = solve_ivp(
        rhs,
        (0.0, 1.0),
        [*initial, neutral],
        method="Radau",
        jac=jacobian,
        rtol=2e-12,
        atol=2e-15,
    )
    assert solution.success, solution.message
    photons, remaining = solution.y[:-1, -1], solution.y[-1, -1]
    assert (
        abs(
            photons.sum() + 1 - remaining - (sum(initial) + 1 - neutral + (q0 + q1) / 2)
        )
        < 1e-12
    )
    return photons, float(1 - remaining)


@pytest.mark.parametrize("tau", [0.01, 1.0, 100.0, 2548.965093320783, 1e6])
@pytest.mark.parametrize("emitted", [0.4, 1.4])
def test_constant_source_matches_independent_reaction_ode_even_when_stiff(
    tau: float, emitted: float
):
    """Match an independent ODE across weak, stiff, and fully ionized regimes."""
    expected_photons, expected_ions = _reference(emitted, emitted, tau)
    actual_photons, actual_ions = _uniform(emitted, emitted, tau)
    np.testing.assert_allclose(actual_photons, expected_photons, rtol=2e-9, atol=2e-11)
    assert actual_ions == pytest.approx(expected_ions, abs=2e-10)


def test_stiff_source_has_no_half_step_radiation_inventory_floor():
    """At physical18eV opacity, the old split left0.02photons instead of0.000261."""
    tau, emitted = 2548.965093320783, 0.4
    expected_photons, expected_ions = _reference(emitted, emitted, tau)
    for cfl in (0.4, 0.2, 0.1):
        photons, ions = _uniform(emitted, emitted, tau, cfl=cfl)
        assert abs(photons.sum() - expected_photons.sum()) < 2e-10
        assert abs(ions - expected_ions) < 2e-10


def test_isotropic_emission_preserves_the_correct_anisotropic_memory():
    """Retain the surviving anisotropic radiation during isotropic emission."""
    initial = (0.32, 0.04, 0.02, 0.01, 0.005, 0.005)
    photons, ions = _uniform(0.35, 0.35, 3.0, neutral=0.8, initial=initial)
    expected_photons, expected_ions = _reference(0.35, 0.35, 3.0, 0.8, initial)
    np.testing.assert_allclose(photons, expected_photons, rtol=2e-10, atol=2e-12)
    assert ions == pytest.approx(expected_ions, abs=2e-11)
    assert np.ptp(photons) > 0.01  # Isotropizing the total radiation would be wrong.


@pytest.mark.parametrize(("q0", "q1"), [(0.0, 0.8), (0.8, 0.0)])
def test_linearly_varying_emission_converges_in_time_with_exact_budget(
    q0: float, q1: float
):
    """Recover second-order time accuracy while retaining the emitted budget."""
    _, expected_ions = _reference(q0, q1, 5.0)
    errors = []
    for cfl in (0.4, 0.2, 0.1):
        photons, ions = _uniform(q0, q1, 5.0, cfl=cfl)
        assert photons.sum() + ions == pytest.approx((q0 + q1) / 2, abs=2e-13)
        errors.append(abs(ions - expected_ions))
    assert errors[-1] < 2e-5
    assert errors[1] < 0.35 * errors[0]
    assert errors[2] < 0.35 * errors[1]


def test_weak_source_absorption_retains_counts_below_neutral_roundoff():
    """Absorb tiny positive sources without subtracting nearly equal atom counts."""
    emitted, tau = 1e-30, 1000.0
    photons, ions = _uniform(emitted, emitted, tau)
    # Neutral depletion is O(1e-30); the exact linear-absorption limit suffices.
    radiation = emitted * (-np.expm1(-tau)) / tau
    assert photons.sum() == pytest.approx(radiation, rel=2e-10, abs=0)
    assert ions == pytest.approx(emitted - radiation, rel=2e-12, abs=0)


@pytest.mark.parametrize(("tau", "neutral"), [(0.0, 0.7), (1000.0, 0.0)])
def test_absorption_free_limits_keep_incoming_directions_and_all_emission(
    tau: float, neutral: float
):
    """Keep directional photons and emission when no absorption is possible."""
    initial = np.array([0.1, 0.02, 0.03, 0.05, 0.0, 0.07])
    photons, ions = _uniform(0.1, 0.7, tau, neutral=neutral, initial=initial)
    np.testing.assert_allclose(
        photons, initial + directions(6)[1] * 0.4, rtol=2e-14, atol=1e-15
    )
    assert ions == pytest.approx(1 - neutral, abs=2e-14)


def test_opaque_source_free_tail_uses_positive_surviving_inventory():
    """Retain a nonnegative radiation tail after nearly complete absorption."""
    # This front-cell state previously gave survival=-1.0842e-19 after
    # subtracting absorbed/radiation from one, aborting a valid evolution.
    neutral = 0.5999801257685806188391
    initial = (0.1999934282541434016345, 0.0, 0.0, 0.0, 0.0, 0.0)
    photons, ions = _uniform(0.0, 0.0, 31250.0, neutral=neutral, initial=initial)
    assert (photons >= 0).all()
    assert photons.sum() < 1e-250
    assert ions == pytest.approx(1 - neutral + sum(initial), abs=2e-14)
