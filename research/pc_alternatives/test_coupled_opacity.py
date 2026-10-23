"""Independent ODE and analytic acceptance for conservative opacity coupling."""
import numpy as np
import pytest
from coupled_opacity import instantaneous_flux, solve, solve_history
from validation_recombination import opacity_reference


def test_recombining_screen_recovers_simultaneous_finite_opacity():
    args = ([1], [1], [1], [1], [[0]], [1], 2)
    result = solve(*args, rtol=1e-7, atol=1e-10)
    reference = opacity_reference(*args)
    np.testing.assert_allclose(result['x'], reference['x'], atol=8e-8, rtol=0)
    np.testing.assert_allclose(result['escaped_by_beam'], reference['escaped_by_beam'], atol=2e-7, rtol=0)
    assert result['escaped_by_beam'][0] > 1.46
    for key in ('photon_ledger_error', 'per_beam_ledger_max', 'absorption_crosscheck', 'atom_ledger_max_error'):
        assert result['audit'][key] < 2e-13


def test_shared_beams_second_order_and_order_bias_vanishes():
    cells = ([1, 3, 3], [1, 0, 0], [1, .4, .4], [1, 4, 4])
    paths, rates = [[0, 1], [0, 2]], [1, 2]
    reference = np.asarray(opacity_reference(*cells, paths, rates, 1.5)['x'])
    errors, bias = [], []
    for steps in (8, 16, 32, 64):
        a = solve(*cells, paths, rates, 1.5, steps=steps)
        b = solve(*cells, paths[::-1], rates[::-1], 1.5, steps=steps)
        errors.append(np.abs(a['x'] - reference).max())
        bias.append(np.abs(a['x'] - b['x']).max())
    assert 3.8 < errors[-2] / errors[-1] < 4.2
    assert 3.8 < bias[-2] / bias[-1] < 4.2
    assert errors[-1] < 1e-4


def test_exact_photo_flow_entire_chain_against_independent_ode():
    args = ([1, 2, 3], [.1, .8, 0], [0, 0, 0], [.3, 3, 10], [[0, 1, 2]], [2], 3)
    result = solve(*args, steps=1)
    reference = opacity_reference(*args)
    for key in ('x', 'absorbed', 'escaped_by_beam'):
        np.testing.assert_allclose(result[key], reference[key], atol=3e-11, rtol=0)


def test_plane_parallel_photo_column_is_subdivision_invariant():
    whole = solve([1], [0], [0], [10], [[0]], [1], 1.2, steps=1)
    for count in (2, 8, 64):
        result = solve(np.full(count, 1 / count), np.zeros(count), np.zeros(count),
                       np.full(count, 10 / count), [list(range(count))], [1], 1.2, steps=1)
        np.testing.assert_allclose(result['absorbed'].sum(), whole['absorbed'].sum(), atol=1e-14)
        np.testing.assert_allclose(result['escaped_by_beam'], whole['escaped_by_beam'], atol=1e-14)
        # Every prefix has the same exact integrated neutral column.
        coordinate = np.arange(count + 1) / count
        with np.errstate(divide='ignore'):
            y_column = np.logaddexp(0, np.log(np.expm1(10 * coordinate)) - 12) / 10
        neutral_average = np.diff(y_column) * count
        np.testing.assert_allclose(1 - result['x'], neutral_average, atol=2e-14)


def test_source_off_recombination_and_reopened_screen_are_physical():
    result = solve_history([1, 2], [1, 0], [1, 0], [4, 4], [[0, 1]],
                           [(1, [0]), (.2, [1])], rtol=1e-7, atol=1e-10)
    assert result['intervals'][0]['x'][0] == pytest.approx(.5, abs=1e-15)
    assert result['intervals'][0]['absorbed'].sum() == 0
    assert result['intervals'][1]['absorbed'][0] > .15
    assert result['intervals'][1]['absorbed'][1] > 0
    assert result['photon_ledger_error'] < 1e-13
    assert result['atom_ledger_max_error'] < 1e-13


def test_opposite_beams_share_cell_and_gamma_has_correct_ionized_limit():
    atoms, tau, paths, rates = [1, 2], [3, 4], [[0, 1], [1, 0]], [2, 1]
    flux = instantaneous_flux(atoms, [1, 1], tau, paths, rates)
    np.testing.assert_allclose(flux['Gamma'], [9, 6], atol=1e-14)
    np.testing.assert_allclose(flux['absorption_rate'], 0, atol=0)
    np.testing.assert_allclose(flux['outgoing_rate'], rates, atol=0)
    a = solve(atoms, [.8, .2], [.4, .1], tau, paths, rates, 1, rtol=1e-7, atol=1e-10)
    b = opacity_reference(atoms, [.8, .2], [.4, .1], tau, paths, rates, 1)
    np.testing.assert_allclose(a['x'], b['x'], atol=2e-7, rtol=0)


def test_zero_depth_dark_and_stiff_limits_are_bounded():
    out = solve([1, 1, 1], [1, .5, 0], [1e6, 1, 0], [1e6, 0, 1e6],
                [[0, 1, 2]], [0], 10, steps=1)
    np.testing.assert_allclose(out['x'], [1 / (1 + 1e7), 1 / 12, 0], atol=2e-16)
    out = solve([1, 1], [.5, 0], [0, 0], [0, 1e6], [[0, 1]], [1], 2, steps=1)
    np.testing.assert_allclose(out['x'], [.5, 1], atol=1e-15)
    np.testing.assert_allclose(out['escaped_by_beam'], [1], atol=1e-15)
    assert out['audit']['x_min'] >= 0 and out['audit']['x_max'] <= 1


def test_per_entry_optical_depths_and_empty_beam():
    result = solve([1, 1], [0, 0], [0, 0], [100, 100], [[0, 1], []], [1, 2], 1,
                   path_tau=[[0, 1], []], steps=1)
    assert result['absorbed'][0] == 0
    assert result['absorbed'][1] > 0
    assert result['escaped_by_beam'][1] == 2
    assert result['audit']['photon_ledger_error'] < 1e-14


def test_history_rate_changes_persist_state_without_double_spending():
    cells = ([1, 2], [0, 0], [.1, .2], [2, 3])
    history = solve_history(*cells, [[0, 1]], [(.2, [1]), (.5, [0]), (.3, [2])],
                            rtol=1e-7, atol=1e-10)
    assert history['emitted'] == pytest.approx(.8)
    assert history['photon_ledger_error'] < 1e-13
    x = cells[1]
    for duration, rate in ((.2, [1]), (.5, [0]), (.3, [2])):
        ref = opacity_reference(cells[0], x, cells[2], cells[3], [[0, 1]], rate, duration)
        x = ref['x']
    np.testing.assert_allclose(history['x'], x, atol=5e-7, rtol=0)


def test_unsupported_loops_invalid_inputs_and_accuracy_failure_raise():
    with pytest.raises(ValueError, match='Repeated'):
        solve([1], [0], [0], [1], [[0, 0]], [1], 1)
    with pytest.raises(ValueError, match='nonnegative'):
        solve([1], [0], [0], [1], [[0]], [-1], 1)
    with pytest.raises(RuntimeError, match='unresolved'):
        solve([1], [1], [1], [1], [[0]], [1], 2, max_steps=1)


def test_unrepresentable_tiny_state_change_is_flagged_not_claimed_resolved():
    with pytest.raises(FloatingPointError, match='not resolved'):
        solve([1e20], [.5], [0], [1], [[0]], [.001], 1, steps=1)
    diagnostic = solve([1e20], [.5], [0], [1], [[0]], [.001], 1, steps=1,
                       require_resolved_state=False)
    assert not diagnostic['audit']['state_precision_resolved']
    assert diagnostic['absorbed'][0] > 0
    assert diagnostic['beam_absorbed'][0] == diagnostic['absorbed'][0]
    flux = instantaneous_flux([1e-308], [.5], [0], [[0]], [1e308])
    assert flux['Gamma'][0] == 0
    huge = solve([1], [1], [1e308], [1], [[0]], [0], 10, steps=1)
    assert huge['x'][0] == pytest.approx(1e-309, rel=1e-14, abs=0)
