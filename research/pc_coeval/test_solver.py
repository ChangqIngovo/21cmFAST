"""Analytic geometry, independent ledgers and invariances for coeval PC."""

import numpy as np
import pytest

from research.pc_coeval.solver import build_identity, solve


def check_ledger(source, capacity, answer):
    ions, remainder = answer["ionized"], answer["remaining_source"]
    assert np.all(ions >= 0)
    assert np.all(ions <= capacity + 1e-12*np.maximum(capacity, 1))
    assert np.all(remainder >= 0)
    assert np.all(remainder <= source + 1e-12*np.maximum(source, 1))
    assert abs(np.sum(source, dtype=np.longdouble)
               - np.sum(ions, dtype=np.longdouble)
               - np.sum(remainder, dtype=np.longdouble)) <= 1e-11*max(1, source.sum())
    assert abs(answer["audit"]["relative_photon_residual"]) < 1e-11


@pytest.mark.parametrize("n", [1, 4, 5])
def test_zero_partial_vacuum_and_saturation(n):
    cap = np.ones((n,)*3)
    for source, capacity in ((cap*0, cap), (cap*.3, cap),
                             (cap*2, cap), (cap*2, cap*0), (cap*0, cap*0)):
        result = solve(source, capacity)
        check_ledger(source, capacity, result)
        np.testing.assert_array_equal(result["ionized"], np.minimum(source, capacity))
        assert result["audit"]["complete"]
        assert result["audit"]["iterations"] == 0
        assert result["audit"]["unresolved_photons"] < 1e-12


def test_saturation_with_inhomogeneous_sources_and_capacity():
    rng = np.random.default_rng(9)
    cap = rng.lognormal(0, 1, (9,)*3)
    source = np.zeros_like(cap)
    source[0, 0, 0] = .8*cap.sum()
    source[5, 4, 2] = .7*cap.sum()
    result = solve(source, cap, max_interactions=0)
    np.testing.assert_array_equal(result["ionized"], cap)
    assert result["audit"]["saturation_shortcut"]
    assert result["audit"]["complete"]
    assert result["audit"]["remaining_photons"] == pytest.approx(.5*cap.sum())
    check_ledger(source, cap, result)


@pytest.mark.parametrize("threads", [1, 2, 4])
def test_single_source_exact_sphere_and_partial_shell(threads):
    n = 12
    cap = np.ones((n,)*3)
    origin = np.array([6, 6, 6])
    distance2 = ((np.indices(cap.shape)-origin[:, None, None, None])**2).sum(axis=0)
    source = np.zeros_like(cap)
    source[tuple(origin)] = np.count_nonzero(distance2 < 9) + .37*np.count_nonzero(distance2 == 9)
    result = solve(source, cap, threads=threads, tolerance=1e-12)
    expected = (distance2 < 9).astype(float) + .37*(distance2 == 9)
    np.testing.assert_allclose(result["ionized"], expected, rtol=0, atol=1e-12)
    assert result["audit"]["complete"]
    check_ledger(source, cap, result)


def test_partial_shell_is_weighted_by_atom_capacity():
    cap = np.ones((7,)*3)
    cap[4, 3, 3] = 4
    source = np.zeros_like(cap)
    source[3, 3, 3] = 1 + .4*9
    result = solve(source, cap, tolerance=1e-12)
    assert result["ionized"][3, 3, 3] == 1
    assert result["ionized"][4, 3, 3] == pytest.approx(1.6)
    assert result["ionized"][2, 3, 3] == pytest.approx(.4)
    assert result["ionized"].sum() == pytest.approx(source.sum())
    check_ledger(source, cap, result)


def test_periodic_target_across_face_and_empty_capacity():
    cap = np.zeros((8,)*3)
    cap[7, 2, 3] = 2
    cap[3, 2, 3] = 2
    source = np.zeros_like(cap)
    source[0, 2, 3] = 1
    result = solve(source, cap, tolerance=1e-12)
    assert result["ionized"][7, 2, 3] == pytest.approx(1)
    assert result["ionized"][3, 2, 3] == 0
    check_ledger(source, cap, result)


def fixture(seed=42, n=9):
    rng = np.random.default_rng(seed)
    cap = rng.uniform(.1, 2, (n,)*3)
    cap[rng.random(cap.shape) < .1] = 0
    source = np.zeros_like(cap)
    source.ravel()[rng.choice(cap.size, 20, replace=False)] = rng.uniform(3, 15, 20)
    return source, cap


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_overlapping_sources_close_without_overdrawing(seed):
    source, cap = fixture(seed)
    result = solve(source, cap, threads=4, tolerance=1e-12)
    check_ledger(source, cap, result)
    assert result["audit"]["complete"]
    assert result["audit"]["unresolved_photons"] <= 1e-12*source.sum()
    assert result["audit"]["excess_photons"] == 0
    assert result["audit"]["iterations"] > 0


@pytest.mark.parametrize("n", [8, 9])
def test_translation_reflection_axis_permutation_and_thread_scheduling(n):
    source, cap = fixture(70, n=n)
    baseline = solve(source, cap, tolerance=1e-12)
    transforms = [lambda a: np.roll(a, (3, -2, 1), axis=(0, 1, 2)),
                  lambda a: a[::-1], lambda a: a.transpose(2, 0, 1)]
    for transform in transforms:
        answer = solve(transform(source), transform(cap), threads=4, tolerance=1e-12)
        np.testing.assert_allclose(answer["ionized"], transform(baseline["ionized"]),
                                   rtol=0, atol=3e-12)
        np.testing.assert_allclose(answer["remaining_source"],
                                   transform(baseline["remaining_source"]), rtol=0, atol=3e-12)
    for threads in (2, 4):
        answer = solve(source, cap, threads=threads, tolerance=1e-12)
        np.testing.assert_allclose(answer["ionized"], baseline["ionized"], rtol=0, atol=3e-12)


def test_exact_tied_sources_are_treated_symmetrically():
    cap = np.ones((12,)*3)
    source = np.zeros_like(cap)
    source[4, 6, 6] = source[8, 6, 6] = 110
    result = solve(source, cap, threads=4, tolerance=1e-12)
    # Reflection about x=6 exchanges the emitters, leaving inputs unchanged.
    reflected = result["ionized"][(-np.arange(12)) % 12]
    np.testing.assert_allclose(result["ionized"], reflected, rtol=0, atol=2e-12)
    check_ledger(source, cap, result)


def test_partially_occupied_uneven_cells_stay_strictly_bounded_when_filled():
    # Adding later absorptions to the local amount used to overshoot some
    # non-unit capacities by one ULP, producing xHI < 0 downstream.
    source, cap = fixture(0)
    source += .317*cap
    result = solve(source, cap, threads=4, tolerance=1e-12)
    assert result["audit"]["complete"]
    assert np.all(result["ionized"] <= cap)
    assert np.all(result["ionized"] >= 0)
    assert np.all(result["ionized"][cap > 0]/cap[cap > 0] <= 1)
    check_ledger(source, cap, result)


@pytest.mark.parametrize("limit", [0, 1, 50, 300])
def test_work_limit_preserves_last_committed_state(limit):
    source = np.zeros((12,)*3)
    source[6, 6, 6] = 100
    cap = np.ones_like(source)
    result = solve(source, cap, threads=4, max_interactions=limit)
    assert not result["audit"]["complete"]
    assert result["audit"]["termination"] == "interaction_limit"
    assert result["audit"]["interactions"] <= limit
    check_ledger(source, cap, result)


def test_iteration_limit_is_not_claimed_as_complete():
    source, cap = fixture()
    result = solve(source, cap, max_iterations=0)
    assert not result["audit"]["complete"]
    assert result["audit"]["termination"] == "iteration_limit"
    np.testing.assert_array_equal(result["ionized"], np.minimum(source, cap))
    check_ledger(source, cap, result)


def test_inputs_are_not_mutated():
    source, cap = fixture()
    old_source, old_cap = source.copy(), cap.copy()
    solve(source, cap)
    np.testing.assert_array_equal(source, old_source)
    np.testing.assert_array_equal(cap, old_cap)


@pytest.mark.parametrize("factor", [1e-150, 1e150])
def test_additive_count_units_do_not_change_the_field(factor):
    source, cap = fixture(19)
    baseline = solve(source, cap, tolerance=1e-12)
    scaled = solve(source*factor, cap*factor, threads=4, tolerance=1e-12)
    assert scaled["audit"]["complete"]
    assert abs(scaled["audit"]["relative_photon_residual"]) < 1e-12
    np.testing.assert_allclose(scaled["ionized"]/factor, baseline["ionized"],
                               rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("kwargs", [dict(threads=0), dict(threads=True),
                                    dict(tolerance=0), dict(tolerance=float("nan")),
                                    dict(max_iterations=-1), dict(max_interactions=-1)])
def test_invalid_controls(kwargs):
    with pytest.raises(ValueError):
        solve(np.ones((2,)*3), np.ones((2,)*3), **kwargs)


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan")])
def test_invalid_fields(value):
    cap = np.ones((2,)*3)
    bad = cap.copy()
    bad[0, 0, 0] = value
    with pytest.raises(ValueError):
        solve(bad, cap)
    with pytest.raises(ValueError):
        solve(cap, bad)


def test_build_identity_records_actual_kernel_and_compiler():
    identity = build_identity()
    assert len(identity["source_sha256"]) == 64
    assert identity["compiler"]
    assert "-fopenmp" in identity["flags"]
