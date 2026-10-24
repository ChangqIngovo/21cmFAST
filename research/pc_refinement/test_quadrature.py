from itertools import product
import math
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from quadrature import directions, solve, _backend


def double_factorial(n):
    return math.prod(range(n, 0, -2))


@pytest.mark.parametrize("count,degree", [(24, 5), (48, 7), (96, 7), (192, 7)])
def test_exact_angular_moments(count, degree):
    xyz = directions(count, "moment-design")
    for powers in product(range(degree+1), repeat=3):
        if sum(powers) > degree:
            continue
        actual = np.prod(xyz**np.array(powers), axis=1).mean()
        if any(p % 2 for p in powers):
            expected = 0.
        else:
            expected = math.prod(double_factorial(p-1) for p in powers) / double_factorial(sum(powers)+1)
        assert abs(actual-expected) < 3e-15, (count, powers, actual, expected)


def test_antipodal_mean_is_exactly_zero():
    for count in (24, 48, 192, 1536):
        xyz = directions(count, "antipodal")
        np.testing.assert_array_equal(xyz[:count//2], -xyz[count//2:])
        assert np.max(np.abs(xyz.mean(axis=0))) < 1e-16


def test_original_rule_exactly_reproduces_baseline_and_does_not_patch_it():
    sys.path.insert(0, str(HERE.parent / "pc_reconsidered"))
    import hybrid
    original = hybrid.directions
    capacity = np.ones((12,)*3)
    source = np.zeros_like(capacity)
    source[5, 6, 6] = 75.
    source[8, 6, 6] = 45.
    before = hybrid.solve(source, capacity)
    after = solve(source, capacity)
    np.testing.assert_array_equal(after["absorbed"], before["absorbed"])
    assert hybrid.directions is original


@pytest.mark.parametrize("method", ["antipodal", "moment-design", "rotated-fibonacci"])
def test_ledgers_and_constant_rate_time_partition(method):
    capacity = np.ones((12,)*3)
    source = np.zeros_like(capacity)
    source[5, 6, 6] = 75.
    source[8, 6, 6] = 45.
    options = dict(method=method, min_angles=192, max_angles=192)
    whole = solve(source, capacity, **options)
    initial = np.zeros_like(capacity)
    for _ in range(4):
        part = solve(source / 4, capacity, initial, **options)
        initial += part["absorbed"]
        assert abs(part["audit"]["ledger_residual"]) < 1e-12
    np.testing.assert_allclose(initial, whole["absorbed"], atol=2e-14, rtol=0)


def test_invalid_rule_counts_fail():
    with pytest.raises(ValueError):
        directions(25, "antipodal")
    with pytest.raises(ValueError):
        directions(25, "moment-design")
    with pytest.raises(ValueError):
        directions(24, "missing")


@pytest.mark.parametrize("count", [24, 48, 96, 384])
def test_cubic_rule_exact_signed_permutation_symmetry(count):
    from itertools import permutations, product
    xyz = directions(count, "cubic-equal-area")
    original = set(map(tuple, np.round(xyz, 13)))
    for perm in permutations(range(3)):
        for sign in product((-1, 1), repeat=3):
            assert set(map(tuple, np.round(xyz[:, perm]*sign, 13))) == original
    np.testing.assert_allclose(xyz.T@xyz/count, np.eye(3)/3, atol=2e-15, rtol=0)


@pytest.mark.parametrize("method", ["voronoi-weights", "moment-weights"])
def test_positive_weights_and_weighted_ledgers(method):
    from quadrature_weights import weights
    for count in (24, 48, 96, 384):
        w = weights(count, method)
        assert (w > 0).all()
        assert abs(w.sum()-1) < 3e-16
        if method == "moment-weights":
            xyz = directions(count)
            np.testing.assert_allclose(w@xyz, 0., atol=3e-16, rtol=0)
            np.testing.assert_allclose(xyz.T@(w[:,None]*xyz), np.eye(3)/3, atol=3e-16, rtol=0)
    capacity = np.ones((12,)*3)
    source = np.zeros_like(capacity)
    source[5, 6, 6] = 75.
    source[8, 6, 6] = 45.
    whole = solve(source, capacity, method=method, min_angles=192, max_angles=192)
    initial = np.zeros_like(capacity)
    for _ in range(4):
        part = solve(source/4, capacity, initial, method=method, min_angles=192, max_angles=192)
        initial += part["absorbed"]
        assert part["audit"]["per_angle_ledger_max"] < 1e-12
    np.testing.assert_allclose(initial, whole["absorbed"], atol=3e-14, rtol=0)


def test_uniform_weighted_core_matches_original():
    from quadrature_weights import solve as weighted
    capacity = np.ones((12,)*3)
    source = np.zeros_like(capacity)
    source[5,6,6] = 91.
    expected = solve(source, capacity, min_angles=192, max_angles=192)
    actual = weighted(source, capacity, method="uniform-weights", min_angles=192, max_angles=192)
    np.testing.assert_allclose(actual["absorbed"], expected["absorbed"], atol=2e-14, rtol=0)
