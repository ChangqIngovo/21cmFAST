"""Photon closure, transport direction, shadows and numerical controls."""

from itertools import pairwise
from typing import Any

import numpy as np
import pytest

from research.pc_analytic.absorption import absorbed
from research.pc_analytic.solver import _library, solve
from research.pc_matched.rt import MPC_CM, directions

OPTIONS = {
    "box_len": 8.0,
    "redshift": 8.0,
    "nH_mean_comoving_cm3": 1.9e-7,
    "sigma_cm2": 3e-18,
    "directions": 24,
    "exposure_steps": 4,
    "threads": 1,
    "exhaustion_tolerance": 1e-8,
}


def run(source: np.ndarray, capacity: np.ndarray, **kwargs: Any) -> dict:
    """Run a small frozen-box fixture with explicitly overridden parameters."""
    return solve(source, capacity, **(OPTIONS | kwargs))


def test_periodic_ledger_and_uniform_ionization() -> None:
    """Close the periodic budget and preserve an exactly homogeneous field."""
    capacity = np.ones((4, 4, 4))
    source = 0.4 * capacity
    result = run(source, capacity)
    assert result["audit"]["complete"]
    assert result["audit"]["ledger_passed"]
    assert abs(result["audit"]["relative_photon_residual"]) < 1e-13
    np.testing.assert_allclose(result["ionized"], 0.4, atol=1e-8)
    np.testing.assert_array_equal(capacity, 1)
    np.testing.assert_array_equal(source, 0.4)


def test_saturation_preserves_surplus_photons() -> None:
    """Retain photons after atom exhaustion without resetting completed ray phases."""
    capacity = np.ones((4, 4, 4))
    source = 2 * capacity
    result = run(source, capacity, exposure_steps=3)
    assert result["audit"]["complete"]
    np.testing.assert_allclose(result["ionized"], capacity, atol=1e-9)
    assert result["remaining_radiation"].sum() == pytest.approx(
        capacity.sum(), abs=1e-8
    )
    assert result["audit"]["escaped_photons"] == 0


def test_opaque_source_cell_absorbs_its_small_budget() -> None:
    """An optically thick source cell absorbs a sub-capacity emission locally."""
    capacity = np.ones((5, 5, 5))
    source = np.zeros_like(capacity)
    source[2, 2, 2] = 0.2
    result = run(source, capacity, exposure_steps=1)
    assert result["audit"]["complete"]
    np.testing.assert_allclose(result["ionized"], source, atol=1e-15, rtol=0)


def test_rotated_vacuum_beam_crosses_tied_faces_once() -> None:
    """A diagonal beam crosses both tied faces without interpolation or splitting."""
    source = np.zeros((7, 7, 7))
    source[3, 3, 3] = 1
    vectors, _ = directions(6)
    beam = int(np.flatnonzero(np.all(vectors == [1, 0, 0], axis=1))[0])
    weights = np.zeros(6)
    weights[beam] = 1
    cosine = np.sqrt(0.5)
    rotation = np.array([[cosine, -cosine, 0], [cosine, cosine, 0], [0, 0, 1]])
    result = run(
        source,
        np.zeros_like(source),
        directions=6,
        emission_weights=weights,
        rotation=rotation,
        boundary="open",
        exposure_steps=1,
        max_sweeps=1,
    )
    expected = np.zeros_like(source)
    expected[4, 4, 3] = 1
    np.testing.assert_allclose(
        result["remaining_radiation"][beam], expected, atol=2e-16, rtol=0
    )
    assert result["escaped"].sum() == 0
    assert not result["audit"]["complete"]


def test_open_vacuum_records_escape_without_absorption() -> None:
    """Count escaping vacuum photons explicitly without making ionizations."""
    source = np.zeros((6, 6, 6))
    source[2, 3, 3] = 1
    result = run(source, np.zeros_like(source), boundary="open")
    assert result["audit"]["complete"]
    assert result["escaped"].sum() == pytest.approx(1.0, abs=3e-9)
    assert result["ionized"].sum() == 0
    assert abs(result["audit"]["relative_photon_residual"]) < 1e-13


def test_cartesian_beam_keeps_shadow_and_direction() -> None:
    """An opaque patch shadows downstream gas without changing photon direction."""
    source = np.zeros((8, 8, 8))
    source[1, :, :] = 1
    capacity = np.zeros_like(source)
    capacity[3, 2:6, 2:6] = 100
    capacity[5, :, :] = 1
    vectors, _ = directions(6)
    beam = int(np.flatnonzero(np.all(vectors == [1, 0, 0], axis=1))[0])
    weights = np.zeros(6)
    weights[beam] = 1
    result = run(
        source,
        capacity,
        directions=6,
        emission_weights=weights,
        transport_fraction=1.0,
        boundary="open",
        exposure_steps=8,
    )
    assert result["audit"]["complete"]
    np.testing.assert_array_equal(result["ionized"][5, 2:6, 2:6], 0)
    assert np.min(result["ionized"][5, :2, :]) > 0.99
    other = [i for i in range(6) if i != beam]
    np.testing.assert_array_equal(result["remaining_radiation"][other], 0)
    np.testing.assert_array_equal(result["escaped"][other], 0)


def test_cubic_symmetry_and_threads() -> None:
    """Preserve grid-axis symmetry and deterministic worker-independent fields."""
    rng = np.random.default_rng(97)
    capacity = np.exp(rng.normal(0, 0.3, (5, 5, 5)))
    source = np.zeros_like(capacity)
    source[1, 2, 3] = 8
    source[3, 3, 2] = 11
    serial = run(source, capacity)
    parallel = run(source, capacity, threads=2)
    np.testing.assert_array_equal(serial["ionized"], parallel["ionized"])
    np.testing.assert_array_equal(
        serial["remaining_radiation"], parallel["remaining_radiation"]
    )
    permuted = run(source.transpose(2, 0, 1), capacity.transpose(2, 0, 1))
    np.testing.assert_allclose(
        permuted["ionized"], serial["ionized"].transpose(2, 0, 1), atol=2e-13, rtol=0
    )


@pytest.mark.parametrize("boundary", ["periodic", "open", "source_periodic_cube"])
def test_parallel_multiblock_inventory_and_fields_are_deterministic(
    boundary: str,
) -> None:
    """Preserve stopping events, inventories and fields across parallel sum blocks."""
    rng = np.random.default_rng(735)
    capacity = np.exp(rng.normal(0, 0.5, (18, 18, 18)))
    source = np.zeros_like(capacity)
    source[::3, ::3, ::3] = rng.lognormal(0, 1, (6, 6, 6))
    source *= 0.45 * capacity.sum() / source.sum()
    initial = 0.1 * capacity
    serial = run(
        source, capacity, boundary=boundary, exposure_steps=2, initial_ionized=initial
    )
    for threads in (2, 4, 8):
        parallel = run(
            source,
            capacity,
            boundary=boundary,
            exposure_steps=2,
            initial_ionized=initial,
            threads=threads,
        )
        for field in ("ionized", "remaining_radiation", "escaped", "unemitted_source"):
            np.testing.assert_array_equal(parallel[field], serial[field])
        for field in (
            "sweeps",
            "completed_exposure_batches",
            "emitted",
            "remaining_photons",
            "escaped_photons",
            "maximum_retained_batch_photons",
        ):
            assert parallel["audit"][field] == serial["audit"][field]
        assert parallel["audit"]["complete"]
        assert abs(parallel["audit"]["relative_photon_residual"]) < 2e-13


def test_parallel_limited_run_preserves_all_unfinished_inventories() -> None:
    """A multi-block sweep-limited run retains the same unfinished photon budget."""
    rng = np.random.default_rng(736)
    capacity = rng.uniform(0.3, 2, (18, 18, 18))
    source = rng.lognormal(0, 3, capacity.shape)
    serial = run(source, capacity, max_sweeps=2, exposure_steps=3)
    parallel = run(source, capacity, max_sweeps=2, exposure_steps=3, threads=8)
    assert not serial["audit"]["complete"]
    assert not parallel["audit"]["complete"]
    for field in ("ionized", "remaining_radiation", "escaped", "unemitted_source"):
        np.testing.assert_array_equal(parallel[field], serial[field])
    recovered = sum(
        np.sum(parallel[field], dtype=np.longdouble)
        for field in ("ionized", "remaining_radiation", "escaped", "unemitted_source")
    )
    assert float(recovered) == pytest.approx(source.sum(dtype=np.longdouble), rel=2e-13)


def test_sweep_limit_keeps_unemitted_and_in_flight_inventory() -> None:
    """A work limit returns every remaining inventory and an incomplete status."""
    source = np.zeros((6, 6, 6))
    source[2, 3, 3] = 100
    capacity = np.ones_like(source)
    result = run(source, capacity, max_sweeps=1)
    assert not result["audit"]["complete"]
    assert result["audit"]["termination"] == "sweep_limit"
    budget = (
        result["ionized"].sum()
        + result["remaining_radiation"].sum()
        + result["unemitted_source"].sum()
    )
    assert budget == pytest.approx(source.sum(), abs=1e-12)
    assert result["unemitted_source"].sum() > 0


def test_exposure_refinement_on_two_source_overlap() -> None:
    """Fine simultaneous doses reduce the controlled two-source field changes."""
    source = np.zeros((8, 8, 8))
    source[2, 3, 4] = 70
    source[5, 4, 4] = 100
    capacity = np.ones_like(source)
    capacity[:, 4:, :] *= 2
    fields = [
        run(source, capacity, exposure_steps=s, boundary="source_periodic_cube")[
            "ionized"
        ]
        / capacity
        for s in (4, 8, 16, 32, 64)
    ]
    differences = [np.abs(b - a).mean() for a, b in pairwise(fields)]
    assert differences[-1] < 0.1 * differences[0]
    assert differences[-1] < 1e-4


@pytest.mark.parametrize("refinement", [1, 2])
@pytest.mark.parametrize("batches", [1, 16])
def test_characteristic_planar_slab_is_exact(refinement: int, batches: int) -> None:
    """Recover the finite-column fluence solution through a resolved planar slab."""
    n = 8 * refinement
    source = np.zeros((n, n, n))
    source[refinement] = refinement
    capacity = np.zeros_like(source)
    capacity[3 * refinement : 4 * refinement] = 1
    vectors, _ = directions(6)
    weights = np.zeros(6)
    weights[np.all(vectors == [1, 0, 0], axis=1)] = 1
    result = run(
        source,
        capacity,
        directions=6,
        emission_weights=weights,
        box_len=8,
        redshift=0,
        nH_mean_comoving_cm3=1,
        sigma_cm2=1 / MPC_CM,
        boundary="open",
        exposure_steps=batches,
    )
    assert result["audit"]["complete"]
    assert result["ionized"].sum() / source.sum() == pytest.approx(
        absorbed(1, 1, 1), abs=3e-14
    )
    assert abs(result["audit"]["relative_photon_residual"]) < 2e-14


def test_source_periodic_cube_uses_source_relative_even_grid_bounds() -> None:
    """The finite tracing cube is translation-invariant with explicit escape."""
    source = np.zeros((8, 8, 8))
    source[0, 2, 3] = 1
    capacity = np.zeros_like(source)
    capacity[4, 2, 3] = 100
    vectors, _ = directions(6)
    positive = np.zeros(6)
    positive[np.all(vectors == [1, 0, 0], axis=1)] = 1
    negative = np.zeros(6)
    negative[np.all(vectors == [-1, 0, 0], axis=1)] = 1
    # Offsets are [-4,3]: +x escapes before offset4; -x includes offset-4.
    forwards = run(
        source,
        capacity,
        directions=6,
        emission_weights=positive,
        boundary="source_periodic_cube",
        exposure_steps=1,
    )
    backwards = run(
        source,
        capacity,
        directions=6,
        emission_weights=negative,
        boundary="source_periodic_cube",
        exposure_steps=1,
    )
    assert forwards["escaped"].sum() == pytest.approx(1.0, abs=1e-14)
    assert forwards["ionized"].sum() == 0
    assert backwards["ionized"].sum() == pytest.approx(1.0, abs=1e-14)
    translated = run(
        np.roll(source, 2, axis=0),
        np.roll(capacity, 2, axis=0),
        directions=6,
        emission_weights=negative,
        boundary="source_periodic_cube",
        exposure_steps=1,
    )
    np.testing.assert_array_equal(
        translated["ionized"], np.roll(backwards["ionized"], 2, axis=0)
    )


@pytest.mark.parametrize(
    ("photons", "coefficients", "expected"),
    [
        ([1.0, 1.0], [0.1, 10.0], [0.04652994863589, 0.9053893710660254]),
        ([2.0, 1.0], [0.1, 10.0], [0.08825889114907796, 0.8782793199928496]),
        ([1.0, 2.0], [1.0, 10.0], [0.15691718583539938, 0.8430823042508834]),
        ([1e20, 1e20], [1.0, 10.0], [0.2668807147330652, 0.7331192852669348]),
    ],
)
def test_unequal_chord_coupling_matches_independent_integrals(
    photons: list[float],
    coefficients: list[float],
    expected: list[float],
) -> None:
    """Different optical paths share atoms without replacing their opacities by a mean."""
    p = np.asarray(photons, dtype=float)
    k = np.asarray(coefficients, dtype=float)
    result = np.zeros_like(p)
    status = _library().analytic_shared_cell(
        len(p), p.ctypes.data, k.ctypes.data, 1.0, result.ctypes.data
    )
    assert status == 0
    np.testing.assert_allclose(result, expected, atol=4e-12, rtol=0)
    assert np.all(result >= 0)
    assert np.all(result <= p)
    assert result.sum() <= 1 + 1e-14


def test_zero_source_and_transparent_periodic_states() -> None:
    """No photons or no opacity cannot create ionized atoms."""
    capacity = np.ones((4, 4, 4))
    zero = np.zeros_like(capacity)
    result = run(zero, capacity, initial_ionized=0.2 * capacity)
    np.testing.assert_array_equal(result["ionized"], 0.2 * capacity)
    assert result["audit"]["relative_photon_residual"] == 0
    result = run(0.2 * capacity, capacity, sigma_cm2=0)
    assert result["audit"]["complete"]
    assert result["audit"]["termination"] == "transparent_medium"
    assert result["ionized"].sum() == 0
    assert result["remaining_radiation"].sum() == pytest.approx(0.2 * capacity.sum())


def test_reject_bad_physical_inputs() -> None:
    """Reject invalid opacity and nonorthogonal angular rotations."""
    capacity = np.ones((4, 4, 4))
    with pytest.raises(ValueError, match="cross section"):
        run(capacity, capacity, sigma_cm2=-1)
    with pytest.raises(ValueError, match="orthogonal"):
        run(capacity, capacity, rotation=np.ones((3, 3)))
