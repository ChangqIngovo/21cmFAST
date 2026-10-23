"""Analytic and adversarial checks of persistent finite-speed packet transport."""
import numpy as np
import pytest

from finite_c_packets import FiniteCPackets


def beam_solver(*, speed=2., maximum_distance=None, periodic=False):
    cap = np.zeros((8,)*3)
    cap[6, 1, 1] = 100
    return FiniteCPackets(cap, speed=speed, ray_directions=[[1., 0, 0]],
                          maximum_distance=maximum_distance, periodic=periodic)


def assert_ledger(result, atol=1e-12):
    a = result['audit']
    assert abs(a['ledger_error']) < atol
    assert a['per_source_ledger_max'] < atol
    assert abs(a['absorption_ledger_error']) < atol
    assert a['capacity_excess'] < atol


def test_flash_exact_delay_and_stored_position():
    solver = beam_solver()
    solver.schedule_flash((1, 1, 1), 1)
    result = solver.advance(1)
    assert result['audit']['absorbed'] == 0
    assert result['audit']['in_flight'] == 1
    packet = solver.snapshot(include_packets=True)['packets'][0]
    assert packet['position'] == [3.5, 1.5, 1.5]
    assert packet['travelled'] == 2
    assert packet['direction'] == [1., 0., 0.]
    assert solver.advance(2.249)['audit']['absorbed'] == 0
    assert solver.advance(2.25)['audit']['absorbed'] == 1
    assert_ledger(solver.snapshot())


def test_constant_emission_delay_source_off_arrivals():
    solver = beam_solver()
    rates = np.zeros((8,)*3)
    rates[1, 1, 1] = 1
    solver.schedule_constant(rates, 0, 1, emission_steps=32)
    assert solver.advance(2)['audit']['absorbed'] == 0
    at3 = solver.advance(3)
    assert at3['audit']['absorbed'] == .75
    assert at3['audit']['in_flight'] == .25
    final = solver.advance(3.25)
    assert final['audit']['absorbed'] == 1
    assert final['audit']['emitted'] == 1
    assert_ledger(final)


def test_c_dt_is_not_all_photons_travel_time():
    cap = np.zeros((4,)*3)
    cap[2, 1, 1] = 100
    solver = FiniteCPackets(cap, speed=1, ray_directions=[[1., 0, 0]])
    rates = np.zeros_like(cap)
    rates[1, 1, 1] = 1
    solver.schedule_constant(rates, 0, 1, emission_steps=64)
    result = solver.advance(1)
    assert result['audit']['absorbed'] == .5
    assert result['audit']['in_flight'] == .5
    assert_ledger(result)


def test_range_budget_does_not_restart_at_snapshots():
    solver = beam_solver(maximum_distance=3)
    solver.schedule_flash((1, 1, 1), 1)
    for t in (1, 2, 3, 4, 10):
        solver.advance(t)
    result = solver.snapshot(include_packets=True)
    assert result['audit']['range_retained'] == 1
    assert result['audit']['absorbed'] == 0
    assert result['audit']['in_flight'] == 0
    assert result['packets'][0]['position'] == [4.5, 1.5, 1.5]
    assert result['packets'][0]['travelled'] == 3
    assert result['packets'][0]['status'] == 'range_retained'
    assert_ledger(result)


def test_retained_keeps_direction_instead_of_reisotropizing():
    cap = np.zeros((8,)*3)
    cap[6, 1, 1] = cap[1, 2, 1] = 100
    solver = FiniteCPackets(cap, speed=2,
                           ray_directions=[[1., 0, 0], [0, 1., 0]])
    solver.schedule_flash((1, 1, 1), 1)
    solver.advance(1)
    result = solver.advance(3)
    assert result['absorbed'][1, 2, 1] == .5
    assert result['absorbed'][6, 1, 1] == .5
    assert_ledger(result)


def test_observation_partition_is_bitwise_invariant():
    rng = np.random.default_rng(613)
    cap = rng.uniform(0, .1, (5,)*3)
    rates = np.zeros_like(cap)
    rates[1, 1, 1], rates[3, 3, 2] = 1., 2.
    whole = FiniteCPackets(cap, speed=3, angles=24)
    split = FiniteCPackets(cap, speed=3, angles=24)
    for solver in (whole, split):
        solver.schedule_constant(rates, 0, .7, emission_steps=9)
        solver.schedule_constant(rates*.3, .7, 1.4, emission_steps=11)
    a = whole.advance(3)
    for t in (.1, .4, .7, 1., 1.7, 2.1, 3.):
        b = split.advance(t)
    for key in ('absorbed', 'source_emitted', 'source_absorbed',
                'source_escaped', 'source_in_flight'):
        np.testing.assert_array_equal(a[key], b[key])
    assert_ledger(a)


def test_same_time_shared_capacity_is_proportional_and_order_invariant():
    cap = np.zeros((5,)*3)
    cap[2, 2, 2] = 1
    # Each source has two opposite beams; one half reaches the target.
    dirs = [[1., 0, 0], [-1., 0, 0]]
    outputs = []
    for order in ([(1, 2), (3, 6)], [(3, 6), (1, 2)]):
        solver = FiniteCPackets(cap, ray_directions=dirs)
        for x, amount in order:
            solver.schedule_flash((x, 2, 2), amount)
        result = solver.advance(2)
        assert result['source_absorbed'][1, 2, 2] == .25
        assert result['source_absorbed'][3, 2, 2] == .75
        outputs.append(result)
        assert_ledger(result)
    np.testing.assert_array_equal(outputs[0]['absorbed'], outputs[1]['absorbed'])


def test_opaque_screen_cannot_be_skipped_and_opacity_array_is_copied():
    cap = np.zeros((8,)*3)
    cap[3, 1, 1], cap[6, 1, 1] = .3, 10
    solver = FiniteCPackets(cap, ray_directions=[[1., 0, 0]])
    cap[:] = 0  # Must not mutate the transport opacity.
    solver.schedule_flash((1, 1, 1), 1)
    assert solver.advance(1.49)['audit']['absorbed'] == 0
    assert solver.advance(1.5)['absorbed'][3, 1, 1] == .3
    result = solver.advance(4.5)
    assert result['absorbed'][6, 1, 1] == pytest.approx(.7)
    assert_ledger(result)


def test_open_boundary_and_exact_horizon_tie():
    cap = np.zeros((2,)*3)
    for horizon, escaped, retained in ((.5, 1, 0), (.5-1e-12, 0, 1)):
        solver = FiniteCPackets(cap, ray_directions=[[1., 0, 0]],
                               maximum_distance=horizon)
        solver.schedule_flash((1, 0, 0), 1)
        result = solver.advance(1)
        assert result['audit']['escaped'] == escaped
        assert result['audit']['range_retained'] == retained
        assert_ledger(result)


def test_periodic_wrap_keeps_absolute_distance():
    cap = np.zeros((3,)*3)
    cap[0, 1, 1] = 1
    solver = FiniteCPackets(cap, ray_directions=[[1., 0, 0]], periodic=True,
                           maximum_distance=4)
    solver.schedule_flash((2, 1, 1), 2)
    result = solver.advance(4)
    assert result['audit']['absorbed'] == 1
    assert result['audit']['escaped'] == 0
    assert result['audit']['range_retained'] == 1
    assert_ledger(result)


def test_tiny_absorption_is_not_lost_to_large_capacity_subtraction():
    cap = np.ones((2,)*3)*1e20
    solver = FiniteCPackets(cap, angles=1)
    solver.schedule_flash((0, 0, 0), .001)
    result = solver.advance(0)
    assert result['audit']['absorbed'] == .001
    assert_ledger(result, atol=1e-18)


def test_input_checks_reject_backward_time_and_infinite_periodic_path():
    cap = np.ones((2,)*3)
    with pytest.raises(ValueError):
        FiniteCPackets(cap, periodic=True)
    solver = FiniteCPackets(cap)
    solver.advance(1)
    with pytest.raises(ValueError):
        solver.advance(.5)
    with pytest.raises(ValueError):
        solver.schedule_flash((0, 0, 0), 1, emission_time=.5)
    with pytest.raises(ValueError):
        solver.schedule_constant(cap, 1, 2, emission_steps=0)


def test_near_corner_positive_segment_is_absorbing_exact_corner_is_not():
    cap = np.zeros((4,)*3)
    cap[1, 2, 1] = cap[2, 2, 1] = 1
    for epsilon, thin_expected in ((1e-10, 1), (0., 0)):
        direction = np.array([[1., 1.+epsilon, 0.]])
        direction /= np.linalg.norm(direction)
        solver = FiniteCPackets(cap, ray_directions=direction)
        solver.schedule_flash((1, 1, 1), 1)
        result = solver.advance(2)
        assert result['absorbed'][1, 2, 1] == thin_expected
        assert result['absorbed'][2, 2, 1] == 1-thin_expected
        assert_ledger(result)


def test_optional_packet_state_serializes_with_json():
    import json
    solver = beam_solver(maximum_distance=3)
    solver.schedule_flash((1, 1, 1), 1)
    solver.advance(10)
    encoded = json.dumps(solver.snapshot(include_packets=True)['packets'])
    assert 'range_retained' in encoded


def test_packet_budget_fails_before_mutating_source_schedule():
    cap = np.ones((2,)*3)
    solver = FiniteCPackets(cap, angles=24, max_packets=48)
    source = np.zeros_like(cap)
    source[0, 0, 0] = 1
    with pytest.raises(ValueError, match='budget exceeded'):
        solver.schedule_constant(source, 0, 1, emission_steps=3)
    assert solver.scheduled_photons == 0
    solver.schedule_constant(source, 0, 1, emission_steps=2)
    assert solver.advance(1)['audit']['emitted'] == 1


def test_unrepresentable_total_photons_rejected_before_mutation():
    cap = np.zeros((1,)*3)
    solver = FiniteCPackets(cap, angles=1)
    with pytest.raises(ValueError, match='float64 output range'):
        solver.schedule_constant(np.full_like(cap, 1e308), 0, 10, emission_steps=1)
    assert solver.scheduled_photons == 0
    solver.schedule_flash((0, 0, 0), 1e308)
    with pytest.raises(ValueError, match='float64 output range'):
        solver.schedule_flash((0, 0, 0), 1e308)
    assert solver.scheduled_photons == np.longdouble(1e308)
    result = solver.advance(10)
    assert np.isfinite(result['audit']['emitted'])
    assert_ledger(result)
