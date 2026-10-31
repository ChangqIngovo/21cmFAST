"""Continuous emission must interact with absorption during the same step.

Homogeneous fields remove spatial streaming errors. An independent implicit
ODE reference then isolates stiff source/reaction coupling, including angular
memory, rather than testing only a photon ledger.
"""

from functools import lru_cache

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from research.pc_causal_rt.transport import (
    LIGHT_SPEED_CM_S, MPC_CM, directions, evolve_interval,
)


N = 4
BOX = 25.
REDSHIFT = 7.
DENSITY = 1.9e-7
# Four cell crossing times: CFL0.4 normally gives ten streaming steps.
DURATION = 4 * (BOX / N) * MPC_CM / (LIGHT_SPEED_CM_S * (1 + REDSHIFT))


def _uniform(q0, q1, tau, *, neutral=1., initial=None, cfl=.4):
    initial = np.zeros(6) if initial is None else np.asarray(initial, float)
    capacity = np.ones((N, N, N))
    photons = np.broadcast_to(initial[:, None, None, None], (len(initial), N, N, N)).copy()
    sigma = tau / (LIGHT_SPEED_CM_S * DENSITY * (1 + REDSHIFT)**3 * DURATION)
    result = evolve_interval(
        capacity, capacity * (1 - neutral), photons,
        capacity * q0 / DURATION, capacity * q1 / DURATION,
        dt_seconds=DURATION, z_start=REDSHIFT, z_end=REDSHIFT,
        box_len_cMpc=BOX, hydrogen_density_z0_cm3=DENSITY,
        cross_section_cm2=sigma, cfl=cfl,
    )
    # A homogeneous state must stay homogeneous even for anisotropic light.
    for field in result["photons"]:
        np.testing.assert_allclose(field, field[0, 0, 0], rtol=0, atol=1e-14)
    assert result["ledger"]["passed"]
    return result["photons"][:, 0, 0, 0], float(result["ionized"][0, 0, 0])


@lru_cache(maxsize=None)
def _reference(q0, q1, tau, neutral=1., initial=(0.,) * 6):
    """Integrate directional photons and neutral atoms independently of C++."""
    weights = directions(len(initial))[1]

    def rhs(time, state):
        radiation, atoms = state[:-1], state[-1]
        absorption = tau * atoms * radiation
        return np.r_[weights * (q0 + (q1 - q0) * time) - absorption,
                     -absorption.sum()]

    def jacobian(time, state):
        radiation, atoms = state[:-1], state[-1]
        matrix = np.zeros((len(state), len(state)))
        matrix[:-1, :-1] = -tau * atoms * np.eye(len(initial))
        matrix[:-1, -1] = -tau * radiation
        matrix[-1, :-1] = -tau * atoms
        matrix[-1, -1] = -tau * radiation.sum()
        return matrix

    solution = solve_ivp(rhs, (0., 1.), [*initial, neutral], method="Radau",
                         jac=jacobian, rtol=2e-12, atol=2e-15)
    assert solution.success, solution.message
    photons, remaining = solution.y[:-1, -1], solution.y[-1, -1]
    assert abs(photons.sum() + 1 - remaining
               - (sum(initial) + 1 - neutral + (q0 + q1) / 2)) < 1e-12
    return photons, float(1 - remaining)


@pytest.mark.parametrize("tau", [.01, 1., 100., 2548.965093320783, 1e6])
@pytest.mark.parametrize("emitted", [.4, 1.4])
def test_constant_source_matches_independent_reaction_ode_even_when_stiff(tau, emitted):
    expected_photons, expected_ions = _reference(emitted, emitted, tau)
    actual_photons, actual_ions = _uniform(emitted, emitted, tau)
    np.testing.assert_allclose(actual_photons, expected_photons, rtol=2e-9, atol=2e-11)
    assert actual_ions == pytest.approx(expected_ions, abs=2e-10)


def test_stiff_source_has_no_half_step_radiation_inventory_floor():
    """At physical18eV opacity, the old split left0.02photons instead of0.000261."""
    tau, emitted = 2548.965093320783, .4
    expected_photons, expected_ions = _reference(emitted, emitted, tau)
    for cfl in (.4, .2, .1):
        photons, ions = _uniform(emitted, emitted, tau, cfl=cfl)
        assert abs(photons.sum() - expected_photons.sum()) < 2e-10
        assert abs(ions - expected_ions) < 2e-10


def test_isotropic_emission_preserves_the_correct_anisotropic_memory():
    initial = (.32, .04, .02, .01, .005, .005)
    photons, ions = _uniform(.35, .35, 3., neutral=.8, initial=initial)
    expected_photons, expected_ions = _reference(.35, .35, 3., .8, initial)
    np.testing.assert_allclose(photons, expected_photons, rtol=2e-10, atol=2e-12)
    assert ions == pytest.approx(expected_ions, abs=2e-11)
    assert np.ptp(photons) > .01  # Isotropizing the total radiation would be wrong.


@pytest.mark.parametrize("q0,q1", [(0., .8), (.8, 0.)])
def test_linearly_varying_emission_converges_in_time_with_exact_budget(q0, q1):
    expected_photons, expected_ions = _reference(q0, q1, 5.)
    errors = []
    for cfl in (.4, .2, .1):
        photons, ions = _uniform(q0, q1, 5., cfl=cfl)
        assert photons.sum() + ions == pytest.approx((q0 + q1) / 2, abs=2e-13)
        errors.append(abs(ions - expected_ions))
    assert errors[-1] < 2e-5
    assert errors[1] < .35 * errors[0]
    assert errors[2] < .35 * errors[1]


def test_weak_source_absorption_retains_counts_below_neutral_roundoff():
    emitted, tau = 1e-30, 1000.
    photons, ions = _uniform(emitted, emitted, tau)
    # Neutral depletion is O(1e-30); the exact linear-absorption limit suffices.
    radiation = emitted * (-np.expm1(-tau)) / tau
    assert photons.sum() == pytest.approx(radiation, rel=2e-10, abs=0)
    assert ions == pytest.approx(emitted - radiation, rel=2e-12, abs=0)


@pytest.mark.parametrize("tau,neutral", [(0., .7), (1000., 0.)])
def test_absorption_free_limits_keep_incoming_directions_and_all_emission(tau, neutral):
    initial = np.array([.1, .02, .03, .05, 0., .07])
    photons, ions = _uniform(.1, .7, tau, neutral=neutral, initial=initial)
    np.testing.assert_allclose(photons, initial + directions(6)[1] * .4, rtol=2e-14, atol=1e-15)
    assert ions == pytest.approx(1 - neutral, abs=2e-14)
