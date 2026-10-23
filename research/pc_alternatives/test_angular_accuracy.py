"""Acceptance semantics and physical controls for angular/range refinement."""
import numpy as np
import pytest

from angular_accuracy import AccuracyNotReached, range_atom_l1_bound, solve_controlled
from event_front import solve
from validation_geometry import sphere_inputs


def test_low_angles_are_rejected_instead_of_silently_accepted():
    s, c, *_ = sphere_inputs(8)
    with pytest.raises(AccuracyNotReached) as failure:
        solve_controlled(s, c, maximum_angles=48)
    audit = failure.value.result['accuracy']
    assert audit['status'] == 'unresolved'
    assert not audit['spatial_or_physical_accuracy_certified']
    assert audit['history'][0]['phase_maximum'] > .5 * s.sum()


def test_sphere_refinement_returns_a_conservative_unsmoothed_map():
    s, c, reference, *_ = sphere_inputs(8)
    out = solve_controlled(s, c)
    audit = out['accuracy']
    assert audit['status'] == 'empirically_converged'
    assert audit['selected_angles'] > 768
    exact_same_model = solve(s, c, angles=audit['selected_angles'],
                            maximum_distance=audit['selected_range'])
    for name in ['absorbed', 'source_absorbed', 'source_escaped', 'source_retained']:
        np.testing.assert_array_equal(out[name], exact_same_model[name])
    assert abs(out['absorbed'] - reference).sum() / s.sum() < .06
    # The grid error remains larger than the angular convergence threshold.
    # This API must not label that threshold a true physical error bound.
    assert abs(out['absorbed'] - reference).sum() > audit['angular_tolerance_photons']
    assert not audit['spatial_or_physical_accuracy_certified']
    assert abs(out['audit']['ledger_residual']) < 1e-10
    assert out['audit']['per_source_ledger_max'] < 2e-12 * s.sum()
    assert out['audit']['per_angle_ledger_max'] < 1e-10


def test_zero_and_weak_emission_need_no_relative_division():
    s = np.zeros((2,)*3)
    c = np.ones_like(s)
    for emission in [0, 1e-20]:
        s[0, 0, 0] = emission
        out = solve_controlled(s, c, angular_rtol=0, angular_atol=1e-25,
                               initial_angles=3, maximum_angles=48)
        assert out['accuracy']['status'] == 'empirically_converged'
        assert out['absorbed'][0, 0, 0] == emission
        assert out['accuracy']['history'][-1]['phase_maximum'] == 0


def test_range_is_extended_until_atom_error_bound_satisfied():
    s = np.zeros((6,)*3)
    c = np.zeros_like(s)
    s[1, 3, 3] = 1
    c[4, :, :] = 100
    out = solve_controlled(s, c, angular_rtol=.5, initial_angles=12,
                           maximum_angles=3072, maximum_distance=.25,
                           maximum_range=16, range_rtol=0)
    assert out['accuracy']['range_bound_satisfied']
    assert out['accuracy']['selected_range'] > .25
    assert out['accuracy']['selected_range_atom_l1_bound'] == 0
    assert out['audit']['retained'] == 0
    assert len(out['accuracy']['history'][0]['range_attempts'][0]) > 1


def test_range_limit_is_not_silently_overridden_and_failure_is_explicit():
    s = np.zeros((3,)*3)
    c = np.ones_like(s)
    s[1, 1, 1] = 1
    c[1, 1, 1] = 0
    out = solve_controlled(s, c, maximum_distance=.1, maximum_range=.1,
                           initial_angles=3, maximum_angles=48, raise_unresolved=False)
    assert out['accuracy']['status'] == 'unresolved'
    assert not out['accuracy']['range_bound_satisfied']
    assert out['accuracy']['selected_range'] == .1
    assert out['audit']['retained'] == pytest.approx(1)


def test_fully_ionized_periodic_box_has_zero_map_bound_with_retained_photons():
    s = np.ones((2,)*3)
    c = np.zeros_like(s)
    out = solve_controlled(s, c, maximum_distance=1, maximum_range=1,
                           initial_angles=3, maximum_angles=48, periodic=True)
    assert range_atom_l1_bound(out, c) == 0
    assert out['audit']['retained'] == pytest.approx(s.sum())
    assert not out['accuracy']['retained_photons_are_resumable']


def test_multisource_shadow_is_preserved_after_refinement():
    s = np.zeros((8,)*3)
    c = np.zeros_like(s)
    s[1, 2, 3], s[2, 5, 4] = 2, 3
    c[4, :, :] = 1e6
    c[5:, :, :] = 1
    out = solve_controlled(s, c, angular_rtol=.3, maximum_angles=49152)
    assert not np.any(out['absorbed'][5:, :, :])
    assert out['audit']['absorbed'] > 0
    assert out['audit']['escaped'] > 0
    assert out['audit']['ledger_residual'] == pytest.approx(0, abs=1e-12)
    assert out['audit']['per_source_ledger_max'] < 1e-12
    assert out['audit']['per_angle_ledger_max'] < 1e-12


def test_zero_hit_angular_blindspot_fails_closed():
    # A distant opaque voxel subtends a positive rectangular solid angle,
    # although all three K3/12/48 phases miss it exactly.
    s = np.zeros((96,)*3)
    c = np.zeros_like(s)
    s[1, 48, 48], c[94, 48, 48] = 1, 10
    exact = np.arctan(.25/(92.5*np.sqrt(92.5**2+.5)))/np.pi
    assert exact > 9e-6
    with pytest.raises(AccuracyNotReached) as failure:
        solve_controlled(s, c, initial_angles=3, maximum_angles=48,
                         angular_rtol=1e-12)
    out = failure.value.result
    assert not np.any(out['absorbed'])
    assert out['accuracy']['status'] == 'unresolved'
    assert out['accuracy']['stop_reason'] == 'zero_absorption_coverage_unresolved'


def test_ray_storage_guard_before_first_solve_does_not_fabricate_output():
    s = np.ones((2,)*3)
    c = np.ones_like(s)
    out = solve_controlled(s, c, maximum_ray_bytes=1, raise_unresolved=False)
    assert out['accuracy']['stop_reason'] == 'ray_storage_limit'
    assert out['accuracy']['kernel_calls'] == 0
    assert out['accuracy']['selected_angles'] is None
    assert out['absorbed'] is None


def test_ray_storage_guard_keeps_last_unconverged_map():
    s, c, *_ = sphere_inputs(8)
    limit = (4096+3072)*64
    out = solve_controlled(s, c, maximum_ray_bytes=limit, raise_unresolved=False)
    assert out['accuracy']['stop_reason'] == 'ray_storage_limit'
    assert out['accuracy']['status'] == 'unresolved'
    assert out['accuracy']['selected_angles'] == 3072
    assert out['accuracy']['resource_checks'][-1]['estimated_angular_storage_bytes'] > limit
    assert out['absorbed'] is not None


def test_zero_source_still_has_direction_workspace_guard():
    out = solve_controlled(np.zeros((1,)*3), np.ones((1,)*3), maximum_ray_bytes=1,
                           raise_unresolved=False)
    assert out['accuracy']['stop_reason'] == 'ray_storage_limit'
    assert out['accuracy']['kernel_calls'] == 0


@pytest.mark.parametrize('kwargs', [
    {'angular_rtol': -1}, {'angular_rtol': np.nan}, {'range_atol': np.inf},
    {'initial_angles': 0}, {'initial_angles': True}, {'maximum_angles': 12},
    {'refinement_factor': 1}, {'required_successes': 1},
    {'maximum_distance': 0}, {'maximum_range': .1},
    {'maximum_ray_bytes': 0}, {'maximum_ray_bytes': True},
    {'maximum_angles': 2**31},
    {'rotations': np.array([np.eye(3), np.eye(3)])},
    {'rotations': np.array([np.eye(3), -np.eye(3)])},
])
def test_invalid_controls_are_rejected(kwargs):
    with pytest.raises(ValueError):
        solve_controlled(np.zeros((2,)*3), np.ones((2,)*3), **kwargs)
