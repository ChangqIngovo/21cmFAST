"""Bitwise contracts for arithmetic-path acceleration, including edge rays."""
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from speed_core import raw_run, solve


def same_output(one, two):
    for name in ("absorbed", "source_absorbed", "source_escaped", "source_retained"):
        np.testing.assert_array_equal(one[name], two[name])


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("variant", ["index", "filtered", "compact"])
def test_random_sources_cavities_exact(periodic, variant):
    rng = np.random.default_rng(991)
    capacity = rng.uniform(.2, 2., size=(12,)*3)
    source = np.zeros_like(capacity)
    source[2:10:3, 3:11:3, 2:10:3] = 60.
    initial = capacity * (rng.random(capacity.shape) < .4)
    settings = dict(min_angles=48, max_angles=192, periodic=periodic, maximum_distance=16.)
    expected = solve(source, capacity, initial, variant="original", **settings)
    actual = solve(source, capacity, initial, variant=variant, **settings)
    same_output(actual, expected)
    assert actual["audit"]["events"] == expected["audit"]["events"]
    factor = .75 if variant == "compact" else 1.
    assert actual["audit"]["ray_allocated_bytes"] == factor * expected["audit"]["ray_allocated_bytes"]


@pytest.mark.parametrize("offset", [0., 2**-45, 2**-50, 2**-52])
@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize("variant", ["filtered", "compact"])
def test_near_corner_thin_opaque_voxel_and_exact_tie(offset, skip, variant):
    source = np.zeros((9,)*3)
    source[3, 3, 3] = .25
    capacity = np.zeros_like(source)
    # A positive-length sliver immediately after crossing x but before y must
    # not be skipped; exactly coincident x/y crossings miss that side voxel.
    capacity[4, 3, 3] = 1.
    capacity[5, 5, 3] = 1.
    direction = np.array([[1 + offset, 1., 0.]])
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    args = dict(periodic=False, maximum_distance=6., skip=skip)
    expected = raw_run(source, capacity, direction, variant="original", **args)
    actual = raw_run(source, capacity, direction, variant=variant, **args)
    for a, b in zip(actual["arrays"] + actual["angular"], expected["arrays"] + expected["angular"]):
        np.testing.assert_array_equal(a, b)
    if offset:
        assert actual["arrays"][0][4, 3, 3] == .25
    else:
        assert actual["arrays"][0][4, 3, 3] == 0.


@pytest.mark.parametrize("horizon", [0.5, np.nextafter(.5, 0), np.nextafter(.5, 1)])
@pytest.mark.parametrize("variant", ["filtered", "compact"])
def test_exact_horizon_boundary(horizon, variant):
    source = np.zeros((4,)*3)
    source[3, 1, 1] = 1.
    remaining = np.zeros_like(source)
    direction = np.array([[1., 0., 0.]])
    for periodic in [False, True]:
        one = raw_run(source, remaining, direction, variant="original", periodic=periodic,
                      maximum_distance=horizon, skip=False)
        two = raw_run(source, remaining, direction, variant=variant, periodic=periodic,
                      maximum_distance=horizon, skip=False)
        for a, b in zip(one["arrays"], two["arrays"]):
            np.testing.assert_array_equal(a, b)


def test_compact_falls_back_for_large_horizon_or_quadrature():
    source = np.zeros((4,)*3)
    source[1, 1, 1] = 2.
    capacity = np.ones_like(source)
    for angles, horizon in [(24, 40000.), (65537, 2.)]:
        options = dict(min_angles=angles, max_angles=angles, maximum_distance=horizon)
        one = solve(source, capacity, variant="original", **options)
        two = solve(source, capacity, variant="compact", **options)
        same_output(one, two)
        assert one["audit"]["ray_allocated_bytes"] == two["audit"]["ray_allocated_bytes"]
