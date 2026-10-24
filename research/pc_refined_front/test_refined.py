"""Independent checks of fixed-source, sparse refined opaque-cell transport.

The dense reference receives explicit point coordinates and a uniformly split
copy of the *same* coarse density. No source is distributed over the refined
volume. All comparisons fix the angular quadrature and physical horizon.
"""
from pathlib import Path
import sys
from copy import deepcopy

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pc_scale_origin"))
import point_front
sys.path.insert(0, str(HERE.parent / "pc_adaptive_front"))
import adaptive
sys.path.insert(0, str(HERE))
import refined


def fine_capacity(capacity, factor):
    return capacity.repeat(factor, 0).repeat(factor, 1).repeat(factor, 2) / factor**3


def project(field, n, factor):
    return field.reshape(n, factor, n, factor, n, factor).sum(axis=(1, 3, 5))


def dense_reference(source, capacity, factor, angles, horizon, periodic):
    active = source.ravel() > 0
    positions = np.indices(source.shape).reshape(3, -1).T + .5
    return point_front.solve(positions[active] * factor, source.ravel()[active],
                             fine_capacity(capacity, factor), angles=angles,
                             maximum_distance=horizon * factor, periodic=periodic)


def independent_ledger(result, source, capacity, factor):
    emitted = source.sum(dtype=np.longdouble)
    absorbed = result['absorbed'].sum(dtype=np.longdouble)
    escaped = result['source_escaped'].sum(dtype=np.longdouble)
    retained = result['source_retained'].sum(dtype=np.longdouble)
    scale = max(float(emitted), np.finfo(float).tiny)
    assert abs(emitted - absorbed - escaped - retained) / scale < 2e-12
    np.testing.assert_allclose(result['source_absorbed'] + result['source_escaped']
                               + result['source_retained'], source,
                               rtol=2e-12, atol=scale * 2e-15)
    assert np.all(result['absorbed'] >= -scale * 2e-15)
    assert np.all(result['subcell_ionized'] >= -scale * 2e-15)
    assert np.all(result['subcell_ionized'] <= fine_capacity(capacity, factor)
                  + scale * 2e-15)
    np.testing.assert_allclose(project(result['subcell_absorbed'], len(source), factor),
                               result['absorbed'], atol=scale * 2e-14, rtol=2e-12)
    np.testing.assert_allclose(project(result['subcell_ionized'], len(source), factor),
                               result['state']['ionized'], atol=scale * 2e-14, rtol=2e-12)


@pytest.mark.parametrize('factor', [2, 4, 8])
@pytest.mark.parametrize('periodic', [True, False])
def test_matches_independent_fixed_position_reference(factor, periodic):
    n = 4
    rng = np.random.default_rng(140)
    capacity = rng.uniform(.2, 1.8, (n,) * 3)
    source = np.zeros_like(capacity)
    source[0, 1, 2] = 12.
    source[2, 2, 1] = 20.
    source[3, 0, 3] = 5.
    angles, horizon = 96, 2 * n
    result = refined.solve(source, capacity, subcells_per_axis=factor, angles=angles,
                           maximum_distance=horizon, periodic=periodic, return_dense=True)
    reference = dense_reference(source, capacity, factor, angles, horizon, periodic)
    np.testing.assert_allclose(result['subcell_absorbed'], reference['absorbed'],
                               atol=3e-12, rtol=2e-10)
    active = source.ravel() > 0
    for name in ('source_absorbed', 'source_escaped', 'source_retained'):
        np.testing.assert_allclose(result[name].ravel()[active], reference[name],
                                   atol=3e-12, rtol=2e-10)
    independent_ledger(result, source, capacity, factor)


def test_factor_two_reproduces_existing_adaptive_geometry():
    rng = np.random.default_rng(9)
    capacity = rng.uniform(.3, 1.4, (5,) * 3)
    source = np.zeros_like(capacity)
    source[1, 2, 3], source[3, 1, 1] = 26., 18.
    result = refined.solve(source, capacity, subcells_per_axis=2, angles=192,
                           maximum_distance=20., return_dense=True)
    reference = adaptive.solve(source, capacity, min_angles=192, max_angles=192,
                               maximum_distance=20., return_dense=True)
    np.testing.assert_allclose(result['subcell_ionized'], reference['subcell_ionized'],
                               atol=4e-12, rtol=2e-10)
    for name in ('absorbed', 'source_absorbed', 'source_escaped', 'source_retained'):
        np.testing.assert_allclose(result[name], reference[name], atol=4e-12, rtol=2e-10)


def test_source_specific_quadrature_matches_existing_predictor():
    capacity = np.ones((6,) * 3)
    source = np.zeros_like(capacity)
    source[0, 0, 0], source[2, 1, 1] = .01, 4.
    source[3, 3, 3], source[4, 4, 4] = 20., 50.
    reference = adaptive.solve(source, capacity, min_angles=24, max_angles=384,
                               maximum_distance=24., return_dense=True)
    assert len(np.unique(reference['angle_counts'][source > 0])) == 4
    result = refined.solve(source, capacity, subcells_per_axis=2,
                           angle_counts=reference['angle_counts'], maximum_distance=24.,
                           return_dense=True)
    np.testing.assert_array_equal(result['angle_counts'], reference['angle_counts'])
    np.testing.assert_allclose(result['subcell_ionized'], reference['subcell_ionized'],
                               atol=4e-12, rtol=2e-10)
    for name in ('source_absorbed', 'source_escaped', 'source_retained'):
        np.testing.assert_allclose(result[name], reference[name], atol=4e-12, rtol=2e-10)


@pytest.mark.parametrize('factor', [2, 4, 8])
def test_source_launches_only_into_central_eight_cells(factor):
    capacity = np.ones((4,) * 3)
    source = np.zeros_like(capacity)
    source[2, 2, 2] = 4. / factor**3
    result = refined.solve(source, capacity, subcells_per_axis=factor, angles=96,
                           return_dense=True)
    expected = np.zeros((4 * factor,) * 3)
    vertex = int(2.5 * factor)
    expected[vertex-1:vertex+1, vertex-1:vertex+1, vertex-1:vertex+1] = source.sum()/8
    np.testing.assert_allclose(result['subcell_absorbed'], expected, atol=1e-17, rtol=2e-14)
    assert np.count_nonzero(result['subcell_absorbed']) == 8
    assert result['audit']['rays_materialized'] == 0


@pytest.mark.parametrize('factor', [2, 4, 8])
def test_two_half_intervals_preserve_resolved_front(factor):
    rng = np.random.default_rng(712)
    capacity = rng.uniform(.4, 1.6, (4,) * 3)
    source = np.zeros_like(capacity)
    source[1, 1, 1], source[2, 3, 2] = 17., 11.
    options = dict(subcells_per_axis=factor, angles=96, maximum_distance=64., return_dense=True)
    whole = refined.solve(source, capacity, **options)
    first = refined.solve(source / 2, capacity, **options)
    second = refined.solve(source / 2, capacity, initial_state=first['state'], **options)
    assert whole['source_retained'].sum() < 1e-12
    assert second['source_retained'].sum() < 1e-12
    np.testing.assert_allclose(second['subcell_ionized'], whole['subcell_ionized'],
                               atol=5e-12, rtol=2e-10)
    np.testing.assert_allclose(first['absorbed'] + second['absorbed'], whole['absorbed'],
                               atol=5e-12, rtol=2e-10)
    np.testing.assert_allclose(first['source_absorbed'] + second['source_absorbed'],
                               whole['source_absorbed'], atol=5e-12, rtol=2e-10)
    independent_ledger(second, source/2, capacity, factor)
    no_emission = refined.solve(source * 0, capacity, initial_state=second['state'], **options)
    np.testing.assert_array_equal(no_emission['subcell_ionized'], second['subcell_ionized'])
    assert no_emission['absorbed'].sum() == 0


def test_periodic_translation_and_reflection():
    rng = np.random.default_rng(33)
    capacity = rng.uniform(.4, 1.6, (5,) * 3)
    source = np.zeros_like(capacity)
    source[1, 2, 3], source[3, 1, 2] = 29., 8.
    options = dict(subcells_per_axis=4, angles=96, maximum_distance=30., return_dense=True)
    original = refined.solve(source, capacity, **options)
    translated = refined.solve(np.roll(source, 1, axis=(0, 1, 2)),
                               np.roll(capacity, 1, axis=(0, 1, 2)), **options)
    np.testing.assert_allclose(translated['subcell_ionized'],
                               np.roll(original['subcell_ionized'], 4, axis=(0, 1, 2)),
                               atol=5e-12, rtol=2e-10)
    reflected = refined.solve(source[::-1, ::-1, ::-1], capacity[::-1, ::-1, ::-1], **options)
    np.testing.assert_allclose(reflected['subcell_ionized'],
                               original['subcell_ionized'][::-1, ::-1, ::-1],
                               atol=5e-12, rtol=2e-10)


@pytest.mark.parametrize('periodic', [True, False])
def test_vacuum_ledger_and_exact_horizon(periodic):
    capacity = np.zeros((2,) * 3)
    source = np.zeros_like(capacity)
    source[1, 1, 1] = 24.
    options = dict(subcells_per_axis=4, angles=24, maximum_distance=1., periodic=periodic,
                   return_dense=True)
    result = refined.solve(source, capacity, **options)
    reference = dense_reference(source, capacity, 4, 24, 1., periodic)
    assert result['absorbed'].sum() == 0
    assert result['source_escaped'].sum() == pytest.approx(reference['source_escaped'].sum())
    assert result['source_retained'].sum() == pytest.approx(reference['source_retained'].sum())
    independent_ledger(result, source, capacity, 4)


def test_zero_sources_and_capacity_state_validation():
    capacity = np.ones((3,) * 3)
    source = np.zeros_like(capacity)
    blank = refined.solve(source, capacity, subcells_per_axis=4, return_dense=True)
    assert np.count_nonzero(blank['subcell_ionized']) == 0
    source[1, 1, 1] = .5
    result = refined.solve(source, capacity, subcells_per_axis=4)
    with pytest.raises(ValueError):
        refined.solve(source, capacity * 2, initial_state=result['state'], subcells_per_axis=4)
    with pytest.raises(ValueError):
        refined.solve(source, capacity, initial_state=result['state'], subcells_per_axis=2)


def test_new_sources_see_carried_fine_geometry():
    capacity = np.ones((4,) * 3)
    first_source = np.zeros_like(capacity)
    first_source[1, 1, 1] = 10.
    second_source = np.zeros_like(capacity)
    second_source[2, 2, 1] = 9.
    options = dict(subcells_per_axis=4, angles=96, maximum_distance=32., return_dense=True)
    first = refined.solve(first_source, capacity, **options)
    second = refined.solve(second_source, capacity, initial_state=first['state'], **options)
    remaining = fine_capacity(capacity, 4) - first['subcell_ionized']
    reference = point_front.solve([[10., 10., 6.]], [9.], remaining,
                                  angles=96, maximum_distance=128.)
    np.testing.assert_allclose(second['subcell_absorbed'], reference['absorbed'],
                               atol=5e-12, rtol=2e-10)
    np.testing.assert_allclose(second['subcell_ionized'],
                               first['subcell_ionized'] + reference['absorbed'],
                               atol=5e-12, rtol=2e-10)


def test_exact_endpoint_clearance_and_transparent_continuation():
    capacity = np.ones((3,) * 3)
    first = refined.solve(capacity, capacity, subcells_per_axis=2, angles=24,
                          return_dense=True)
    np.testing.assert_allclose(first['subcell_ionized'], fine_capacity(capacity, 2),
                               rtol=0., atol=2e-16)
    second = refined.solve(capacity, capacity, subcells_per_axis=2, angles=24,
                           initial_state=first['state'], return_dense=True)
    assert second['absorbed'].sum() == 0.
    np.testing.assert_allclose(second['source_retained'], capacity, atol=2e-15)
    assert second['source_escaped'].sum() == 0.
    assert second['audit']['rays_materialized'] == 0


@pytest.mark.parametrize('corruption', ['duplicate', 'overcapacity', 'cleared_overlap',
                                      'coarse_budget', 'padding_bits'])
def test_corrupted_carried_geometry_is_rejected(corruption):
    capacity = np.ones((3,) * 3)
    source = np.zeros_like(capacity)
    source[1, 1, 1] = .1
    state = deepcopy(refined.solve(source, capacity, subcells_per_axis=2)['state'])
    if corruption == 'duplicate':
        state['partial_ids'][1] = state['partial_ids'][0]
    elif corruption == 'overcapacity':
        state['partial_ionized'][0] = .126
    elif corruption == 'cleared_overlap':
        key = int(state['partial_ids'][0])
        x, y, z = key // 36, (key // 6) % 6, key % 6
        parent = ((x // 2) * 3 + y // 2) * 3 + z // 2
        child = ((x % 2) * 2 + y % 2) * 2 + z % 2
        state['cleared_bits'][parent, 0] |= np.uint64(1) << np.uint64(child)
    elif corruption == 'coarse_budget':
        state['ionized'][1, 1, 1] += .01
    else:
        state['cleared_bits'][0, 0] = np.uint64(1) << np.uint64(8)
    with pytest.raises(ValueError):
        refined.solve(source, capacity, subcells_per_axis=2, initial_state=state)


@pytest.mark.parametrize('bad', [-1., np.nan, np.inf])
def test_invalid_source_rejected(bad):
    capacity = np.ones((3,) * 3)
    source = np.zeros_like(capacity)
    source[1, 1, 1] = bad
    with pytest.raises(ValueError):
        refined.solve(source, capacity)


@pytest.mark.parametrize('factor', [2, 4])
def test_weak_emission_is_not_lost_from_continued_state(factor):
    capacity = np.ones((3,) * 3)
    source = np.zeros_like(capacity)
    source[1, 1, 1] = 1e-25
    state = None
    for step in range(1, 4):
        result = refined.solve(source, capacity, subcells_per_axis=factor,
                               initial_state=state, return_dense=True)
        state = result['state']
        assert result['subcell_ionized'].sum() == pytest.approx(step * 1e-25, rel=2e-13, abs=0.)
        assert result['absorbed'].sum() == pytest.approx(1e-25, rel=2e-13, abs=0.)


def assert_same_resolved_solution(accelerated, reference, emitted):
    scale = max(float(emitted), np.finfo(float).tiny)
    for name in ('absorbed', 'subcell_absorbed', 'subcell_ionized',
                 'source_absorbed', 'source_escaped', 'source_retained'):
        np.testing.assert_allclose(accelerated[name], reference[name],
                                   atol=scale * 3e-13, rtol=2e-10)
    np.testing.assert_allclose(accelerated['state']['ionized'], reference['state']['ionized'],
                               atol=scale * 3e-13, rtol=2e-10)


@pytest.mark.parametrize('factor', [2, 4, 8])
@pytest.mark.parametrize('neighbour_ratio', [.5, 1., 2.])
def test_local_templates_at_foreign_arrival_and_equal_exit_events(factor, neighbour_ratio):
    # The isolated-parent exhaustion clock scales as C/Q. Equal C/Q makes
    # first-exit events simultaneous; unequal neighbours illuminate a parent
    # before or after its own isolated clock has reached that phase.
    capacity = np.ones((4,) * 3)
    source = np.zeros_like(capacity)
    source[1, 1, 1] = 1.4
    source[2, 1, 1] = 1.4 * neighbour_ratio
    options = dict(subcells_per_axis=factor, angles=96, maximum_distance=16.,
                   return_dense=True)
    reference = refined.solve(source, capacity, use_local_templates=False, **options)
    accelerated = refined.solve(source, capacity, use_local_templates=True, **options)
    assert accelerated['audit']['template_parents'] == 2
    assert accelerated['audit']['template_materialisations'] == 2
    assert_same_resolved_solution(accelerated, reference, source.sum())
    independent_ledger(accelerated, source, capacity, factor)


@pytest.mark.parametrize('factor', [4, 8])
@pytest.mark.parametrize('horizon', [.1, .3, .6])
def test_local_templates_respect_horizon_inside_source_parent(factor, horizon):
    capacity = np.ones((3,) * 3)
    source = np.zeros_like(capacity)
    source[1, 1, 1] = 2.
    options = dict(subcells_per_axis=factor, angles=96, maximum_distance=horizon,
                   return_dense=True)
    accelerated = refined.solve(source, capacity, use_local_templates=True, **options)
    fallback = refined.solve(source, capacity, use_local_templates=False, **options)
    dense = dense_reference(source, capacity, factor, 96, horizon, True)
    assert accelerated['audit']['template_parents'] == 0
    assert_same_resolved_solution(accelerated, fallback, source.sum())
    np.testing.assert_allclose(accelerated['subcell_absorbed'], dense['absorbed'],
                               atol=1e-12, rtol=2e-10)
    assert accelerated['source_retained'].sum() > 0
    assert accelerated['source_retained'].sum() == pytest.approx(
        dense['source_retained'].sum(), rel=2e-11, abs=1e-12)


@pytest.mark.parametrize('factor', [2, 4, 8])
def test_local_templates_with_variable_density_and_source_quadrature(factor):
    rng = np.random.default_rng(905 + factor)
    capacity = 10**rng.uniform(-1., 1., (4,) * 3)
    capacity[rng.random(capacity.shape) < .1] = 0.
    source = 10**rng.uniform(-3., .7, capacity.shape)
    source[rng.random(source.shape) < .3] = 0.
    source *= .85 * capacity.sum() / source.sum()
    counts = rng.choice([24, 48, 96], capacity.shape)
    options = dict(subcells_per_axis=factor, angle_counts=counts,
                   maximum_distance=16., return_dense=True)
    reference = refined.solve(source, capacity, use_local_templates=False, **options)
    accelerated = refined.solve(source, capacity, use_local_templates=True, **options)
    assert accelerated['audit']['template_parents'] > 0
    assert_same_resolved_solution(accelerated, reference, source.sum())
    independent_ledger(accelerated, source, capacity, factor)


@pytest.mark.parametrize('factor', [4, 8])
def test_local_templates_preserve_changed_source_continuation(factor):
    rng = np.random.default_rng(606)
    capacity = rng.uniform(.2, 2., (4,) * 3)
    source1 = rng.uniform(0., .2, capacity.shape)
    source1[1, 1, 1] = 14.
    source2 = source1[::-1, ::-1, ::-1].copy()
    options = dict(subcells_per_axis=factor, angles=96, maximum_distance=32.,
                   return_dense=True)
    first_reference = refined.solve(source1, capacity, use_local_templates=False, **options)
    first_accelerated = refined.solve(source1, capacity, use_local_templates=True, **options)
    assert_same_resolved_solution(first_accelerated, first_reference, source1.sum())
    reference = refined.solve(source2, capacity, initial_state=first_reference['state'],
                              use_local_templates=False, **options)
    accelerated = refined.solve(source2, capacity, initial_state=first_accelerated['state'],
                                use_local_templates=True, **options)
    assert accelerated['audit']['template_parents'] == 0
    assert_same_resolved_solution(accelerated, reference, source2.sum())
    independent_ledger(accelerated, source2, capacity, factor)


@pytest.mark.parametrize('factor', [4, 8])
def test_zero_valued_partial_state_is_equivalent_to_untouched_gas(factor):
    capacity = np.ones((3,) * 3)
    source = np.zeros_like(capacity)
    source[1, 1, 1] = .4
    state = deepcopy(refined.solve(source * 0, capacity, subcells_per_axis=factor)['state'])
    fn = 3 * factor
    corner = int(1.5 * factor) - 1
    state['partial_ids'] = np.array([(corner * fn + corner) * fn + corner], dtype=np.int64)
    state['partial_ionized'] = np.array([0.])
    options = dict(subcells_per_axis=factor, angles=96, maximum_distance=12.,
                   return_dense=True, use_local_templates=True)
    blank = refined.solve(source, capacity, **options)
    represented = refined.solve(source, capacity, initial_state=state, **options)
    assert blank['audit']['template_parents'] == 1
    assert represented['audit']['template_parents'] == 0
    assert_same_resolved_solution(represented, blank, source.sum())


@pytest.mark.parametrize('templates', [False, True])
def test_compact_and_wide_rays_produce_identical_open_boundary_solution(templates):
    # Both cutoffs exceed every source-to-open-boundary path. Only the large
    # declared range forces wide unwrapped coordinates; no long path is traced.
    rng = np.random.default_rng(202610)
    capacity = rng.uniform(.3, 1.6, (4,) * 3)
    source = np.zeros_like(capacity)
    source[1, 1, 1], source[2, 2, 2] = 40., 30.
    options = dict(subcells_per_axis=4, angles=96, periodic=False, return_dense=True,
                   use_local_templates=templates)
    compact = refined.solve(source, capacity, maximum_distance=16., **options)
    wide = refined.solve(source, capacity, maximum_distance=20000., **options)
    assert compact['audit']['transport_ray_bytes'] == 48
    assert wide['audit']['transport_ray_bytes'] == 64
    assert compact['audit']['rays_materialized'] > 0
    assert wide['audit']['rays_materialized'] == compact['audit']['rays_materialized']
    assert wide['source_escaped'].sum() > 0
    assert compact['source_retained'].sum() == wide['source_retained'].sum() == 0
    assert_same_resolved_solution(wide, compact, source.sum())
    for name in ('absorbed', 'subcell_absorbed', 'subcell_ionized',
                 'source_absorbed', 'source_escaped', 'source_retained'):
        np.testing.assert_array_equal(wide[name], compact[name])


def test_direction_bank_above_uint16_materializes_wide_rays():
    # Exercise direction indices above65535, not just the dispatch with an
    # unexpanded weak source. Vacuum with open boundaries has an exact ledger.
    capacity = np.zeros((2,) * 3)
    source = np.zeros_like(capacity)
    source[1, 1, 1] = 1.
    result = refined.solve(source, capacity, subcells_per_axis=4, angles=65544,
                           maximum_distance=8., periodic=False, return_dense=True,
                           use_local_templates=False)
    assert result['audit']['transport_ray_bytes'] == 64
    assert result['audit']['rays_materialized'] == 65544
    assert result['source_escaped'].sum() == pytest.approx(1., rel=2e-14)
    assert result['absorbed'].sum() == result['source_retained'].sum() == 0
    independent_ledger(result, source, capacity, 4)
