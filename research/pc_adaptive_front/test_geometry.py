"""Independent matched-grid and analytic checks of lazy eight-subcell fronts."""
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pc_refinement"))
sys.path.insert(0, str(HERE))

import adaptive
import front_subcells


def _fine(array):
    return array.repeat(2, 0).repeat(2, 1).repeat(2, 2)/8


def _check_ledgers(result, source, capacity, initial=None):
    old = np.zeros_like(capacity) if initial is None else initial
    q = source.sum(dtype=np.longdouble)
    absorbed = result["absorbed"]
    scale = np.maximum(capacity, np.finfo(float).tiny)
    assert np.max(np.maximum(absorbed-(capacity-old), 0)/scale) < 2e-10
    assert np.max(np.maximum(-absorbed, 0)/scale) < 2e-10
    residual = q-absorbed.sum(dtype=np.longdouble)-result["unabsorbed"]
    assert abs(residual)/max(q, np.finfo(float).tiny) < 2e-10
    actual = result["source_absorbed"] + result["source_escaped"] + result["source_retained"]
    active = source > 0
    if active.any():
        assert np.max(abs(actual[active]-source[active])/source[active]) < 2e-10


def _compare(source, capacity, initial=None, **options):
    common = dict(min_angles=384, max_angles=384, maximum_distance=source.shape[0]/2)
    common.update(options)
    full = front_subcells.solve(source, capacity, initial, parent_skip=True, **common)
    lazy = adaptive.solve(source, capacity, initial, return_dense=True, **common)
    for key in ("absorbed", "source_absorbed", "source_escaped", "source_retained"):
        np.testing.assert_allclose(lazy[key], full[key], rtol=2e-10, atol=2e-12)
    assert lazy["unabsorbed"] == pytest.approx(full["unabsorbed"], rel=2e-10, abs=2e-12)
    np.testing.assert_allclose(lazy["subcell_absorbed"], full["subcell_absorbed"],
                               rtol=2e-10, atol=2e-12)
    _check_ledgers(lazy, source, capacity, initial)
    return lazy, full


def test_uniform_local_partial_needs_no_spatial_expansion():
    rng = np.random.default_rng(981)
    c = rng.uniform(.2, 2., (8,)*3)
    s = c*rng.uniform(.01, .3, c.shape)
    out, _ = _compare(s, c, .4*c)
    np.testing.assert_allclose(out["absorbed"], s, rtol=2e-15, atol=0)
    assert out["audit"]["rays_materialized"] == 0


@pytest.mark.parametrize("periodic", [False, True])
def test_overlapping_sources_share_the_same_fine_capacities(periodic):
    c = np.ones((16,)*3)
    s = np.zeros_like(c)
    s[6, 8, 8], s[10, 8, 8] = 210, 130
    _compare(s, c, periodic=periodic)


@pytest.mark.parametrize("periodic", [False, True])
def test_many_sources_in_nonuniform_partial_medium(periodic):
    rng = np.random.default_rng(19501)
    c = np.exp(rng.normal(0, .6, (10,)*3))
    s = np.exp(rng.normal(-3, 2, c.shape))
    old = c*rng.uniform(0, .8, c.shape)
    # Reproducible completely transparent islands plus uniform-within-parent
    # partial ionization; the latter can stay coarse until an external ray hits.
    transparent = rng.random(c.shape)<.12
    old[transparent] = c[transparent]
    _compare(s, c, old, periodic=periodic, min_angles=96, max_angles=96)


def test_source_point_and_short_horizon_are_unchanged():
    c = np.ones((8,)*3)
    s = np.zeros_like(c)
    s[4, 4, 4] = 20
    out, _ = _compare(s, c, maximum_distance=.3, min_angles=24, max_angles=24)
    assert out["absorbed"].sum() == pytest.approx(1, abs=2e-14)
    assert out["unabsorbed"] == pytest.approx(19, abs=2e-13)


def test_uniform_sphere_keeps_original_source_centroid():
    c = np.ones((16,)*3)
    s = np.zeros_like(c)
    centre = np.array([7, 8, 9])
    s[tuple(centre)] = 60
    out, _ = _compare(s, c, periodic=False, min_angles=1536, max_angles=1536)
    xyz = np.moveaxis(np.indices(c.shape), 0, -1)
    centroid = np.sum(out["absorbed"][..., None]*xyz, axis=(0,1,2))/s.sum()
    np.testing.assert_allclose(centroid, centre, atol=2e-12, rtol=0)


def test_periodic_coarse_translation_preserves_fine_geometry():
    c = np.ones((12,)*3)
    s = np.zeros_like(c)
    s[1, 3, 5], s[4, 4, 6] = 70, 20
    options = dict(min_angles=384, max_angles=384, return_dense=True)
    a = adaptive.solve(s, c, **options)
    shift = (6, 4, 3)
    b = adaptive.solve(np.roll(s, shift, (0,1,2)), c, **options)
    np.testing.assert_allclose(np.roll(a["absorbed"], shift, (0,1,2)), b["absorbed"],
                               atol=2e-12, rtol=0)
    np.testing.assert_allclose(np.roll(a["subcell_absorbed"], tuple(2*x for x in shift), (0,1,2)),
                               b["subcell_absorbed"], atol=2e-12, rtol=0)


def test_front_state_survives_four_constant_rate_time_steps():
    c = np.ones((12,)*3)
    s = np.zeros_like(c)
    s[4, 6, 6], s[7, 6, 6] = 90, 60
    options = dict(min_angles=384, max_angles=384, maximum_distance=6)
    whole, full = _compare(s, c, **options)
    state = None
    summed = np.zeros_like(c)
    fine_summed = np.zeros((24,)*3)
    unabsorbed = 0.
    for _ in range(4):
        out = adaptive.solve(s/4, c, initial_state=state, return_dense=True, **options)
        state = out["state"]
        summed += out["absorbed"]
        fine_summed += out["subcell_absorbed"]
        unabsorbed += out["unabsorbed"]
    np.testing.assert_allclose(summed, whole["absorbed"], atol=3e-12, rtol=0)
    np.testing.assert_allclose(fine_summed, full["subcell_absorbed"], atol=3e-12, rtol=0)
    assert unabsorbed == pytest.approx(whole["unabsorbed"], abs=3e-12)


def test_dense_reference_state_initializes_same_resolved_increment():
    from incremental_benchmark import state_from_dense

    c = np.ones((10,)*3)
    s = np.zeros_like(c)
    s[3, 5, 5], s[6, 5, 5] = 60, 35
    options = dict(min_angles=192, max_angles=192, maximum_distance=5)
    first = front_subcells.solve(s/2, c, parent_skip=True, **options)
    old_fine = first["subcell_absorbed"]
    state = state_from_dense(c, old_fine)
    full = front_subcells.solve(s/2, c, initial_subcell_ionized=old_fine,
                                parent_skip=True, **options)
    lazy = adaptive.solve(s/2, c, initial_state=state, return_dense=True, **options)
    np.testing.assert_allclose(lazy["subcell_absorbed"], full["subcell_absorbed"],
                               atol=2e-12, rtol=0)


def test_uniform_preionization_uses_remaining_neutral_density():
    c = np.ones((20,)*3)
    s = np.zeros_like(c)
    s[10, 10, 10] = .01*4*np.pi*4**3/3
    partial = adaptive.solve(s, c, .99*c, effective_density="remaining")
    equivalent = adaptive.solve(s, .01*c, effective_density="remaining")
    np.testing.assert_array_equal(partial["angle_counts"], equivalent["angle_counts"])
    np.testing.assert_allclose(partial["absorbed"], equivalent["absorbed"],
                               atol=2e-13, rtol=0)


def test_exact_local_exhaustion_at_step_endpoint():
    c = np.ones((6,)*3)
    s = c.copy()
    _compare(s, c, min_angles=24, max_angles=24)


@pytest.mark.parametrize("periodic", [False, True])
def test_transparent_domain_has_no_spurious_absorption(periodic):
    c = np.ones((8,)*3)
    s = np.zeros_like(c)
    s[0, 3, 4], s[7, 4, 5] = 7, 5
    out, _ = _compare(s, c, c, periodic=periodic, min_angles=96, max_angles=96)
    assert out["absorbed"].sum() == 0.
    assert out["unabsorbed"] == pytest.approx(12, abs=2e-12)


def test_saturated_periodic_dense_sources_need_no_rays():
    c = np.ones((6,)*3)
    s = .3*c
    out = adaptive.solve(s, c, c, periodic=True, return_dense=True)
    assert out["audit"]["rays_materialized"] == 0
    assert out["absorbed"].sum() == 0.
    np.testing.assert_array_equal(out["source_retained"], s)
    np.testing.assert_array_equal(out["subcell_ionized"], _fine(c))


def test_long_periodic_transparent_paths_on_non_power_of_two_grid():
    c = np.ones((24,)*3)
    c[18] = 1e5
    old = c.copy()
    old[18] = 0.
    s = np.zeros_like(c)
    s[23, 11, 12] = 1.
    _compare(s, c, old, periodic=True, maximum_distance=35,
             min_angles=384, max_angles=384)


def test_opaque_plane_keeps_isotropic_angular_budget():
    c = np.zeros((24,)*3)
    c[12] = 1e6
    s = np.zeros_like(c)
    s[8,12,12] = 1
    out, _ = _compare(s, c, maximum_distance=12, min_angles=12288, max_angles=12288)
    assert out["absorbed"].sum() == pytest.approx(.5*(1-3.5/12), abs=.002)
