"""Physical contracts for the experimental same-origin subcell solver."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_subcells import octant_directions, solve


def test_angular_flux_has_zero_dipole_and_isotropic_second_moment():
    for count in (24, 48, 192):
        d = octant_directions(count)
        np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1, atol=3e-16)
        np.testing.assert_allclose(d.mean(axis=0), 0, atol=1e-16)
        np.testing.assert_allclose(d.T @ d / count, np.eye(3)/3, atol=1e-15)


def test_all_partial_limit_counts_locally_without_transport():
    rng = np.random.default_rng(412)
    capacity = rng.uniform(.5, 1.5, (8,)*3)
    photons = capacity * rng.uniform(.1, .3, capacity.shape)
    out = solve(photons, capacity)
    np.testing.assert_array_equal(out['absorbed'], photons)
    assert out['audit']['rays_materialized'] == 0


def test_original_source_position_and_centroid_are_preserved():
    capacity = np.ones((16,)*3)
    source = np.zeros_like(capacity)
    location = np.array([7, 8, 9])
    source[tuple(location)] = 60
    out = solve(source, capacity, min_angles=1536, max_angles=1536, periodic=False)
    coordinates = np.moveaxis(np.indices(source.shape), 0, -1)
    centroid = np.sum(out['absorbed'][..., None]*coordinates, axis=(0, 1, 2))/source.sum()
    np.testing.assert_allclose(centroid, location, atol=2e-14)
    assert abs(out['audit']['ledger_residual']) < 1e-10


def test_periodic_integer_translation_preserves_subcell_geometry():
    capacity = np.ones((12,)*3)
    source = np.zeros_like(capacity)
    source[1, 3, 5] = 70
    source[4, 4, 6] = 20
    shifts = (6, 4, 3)
    first = solve(source, capacity, min_angles=384, max_angles=384)
    second = solve(np.roll(source, shifts, (0, 1, 2)), capacity, min_angles=384, max_angles=384)
    np.testing.assert_allclose(np.roll(first['absorbed'], shifts, (0, 1, 2)), second['absorbed'], atol=3e-14)


def test_small_horizon_keeps_source_at_original_coarse_centre():
    capacity = np.ones((8,)*3)
    source = np.zeros_like(capacity)
    source[4, 4, 4] = 20
    out = solve(source, capacity, maximum_distance=.3, min_angles=24, max_angles=24)
    assert out['absorbed'][4, 4, 4] == 1
    assert out['absorbed'].sum() == 1
    assert out['unabsorbed'] == pytest.approx(19)


def test_opaque_plane_respects_isotropic_angular_quota():
    capacity = np.zeros((24,)*3)
    capacity[12] = 1e6
    source = np.zeros_like(capacity)
    source[8, 12, 12] = 1
    out = solve(source, capacity, maximum_distance=12, min_angles=12288, max_angles=12288)
    assert out['absorbed'].sum() == pytest.approx(.5*(1-3.5/12), abs=.002)
    assert abs(out['audit']['ledger_residual']) < 1e-12


def test_initial_atoms_are_not_counted_again():
    capacity = np.ones((8,)*3)
    old = np.full_like(capacity, .3)
    source = np.full_like(capacity, .2)
    out = solve(source, capacity, old)
    np.testing.assert_array_equal(out['absorbed'], source)


def test_invalid_capacity_or_quadrature_rejected():
    with pytest.raises(ValueError):
        octant_directions(25)
    with pytest.raises(ValueError):
        solve(np.ones((2,)*3), -np.ones((2,)*3))
    with pytest.raises(ValueError):
        solve(np.full((2,)*3, 1e308), np.ones((2,)*3))
    with pytest.raises(ValueError):
        solve(np.ones((2,)*3), np.ones((2,)*3), maximum_distance=np.inf)


def test_parent_skip_preserves_fixed_angular_transport():
    rng = np.random.default_rng(421)
    capacity = rng.uniform(.1, 1., (12,)*3)
    source = rng.lognormal(-3, 2, capacity.shape)
    for periodic in (True, False):
        regular = solve(source, capacity, min_angles=96, max_angles=96, periodic=periodic, parent_skip=False)
        skipped = solve(source, capacity, min_angles=96, max_angles=96, periodic=periodic, parent_skip=True)
        np.testing.assert_allclose(skipped['absorbed'], regular['absorbed'], atol=2e-13, rtol=0)
        np.testing.assert_allclose(skipped['source_absorbed'], regular['source_absorbed'], atol=2e-13, rtol=0)
        assert skipped['unabsorbed'] == pytest.approx(regular['unabsorbed'], rel=1e-13)
        assert skipped['audit']['queries'] <= regular['audit']['queries']


def test_time_partition_preserves_subcell_state():
    capacity = np.ones((12,)*3)
    source = np.zeros_like(capacity)
    source[4, 6, 6] = 90
    source[7, 6, 6] = 60
    options = dict(min_angles=384, max_angles=384)
    full = solve(source, capacity, **options)
    old = np.zeros((24,)*3)
    coarse = np.zeros_like(capacity)
    for _ in range(4):
        out = solve(source/4, capacity, initial_subcell_ionized=old, **options)
        old += out['subcell_absorbed']
        coarse += out['absorbed']
    np.testing.assert_allclose(coarse, full['absorbed'], atol=3e-14, rtol=0)
    np.testing.assert_allclose(old, full['subcell_absorbed'], atol=3e-14, rtol=0)
