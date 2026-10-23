import copy

import numpy as np
import pytest

from source_history import RateHistory, SourceLedger, native_sfrd_history


def test_irregular_linear_rate_exact_and_subdivision():
    times = [0., .13, 1.2, 3., 7.]
    rates = 2 + 3 * np.array(times)[:, None]
    history = RateHistory.from_knots(times, rates)
    exact = 2 * 7 + 1.5 * 7**2
    assert history.integrate(0, 7)[0] == exact
    ledger = SourceLedger(history, [100])
    total = sum(ledger.advance(t)[0] for t in [.03, .13, .77, 2, 3, 6.9, 7])
    assert total == pytest.approx(exact, rel=2e-16)
    assert ledger.emitted[0] == pytest.approx(exact, rel=2e-16)
    assert ledger.advance(7)[0] == 0
    with pytest.raises(ValueError):
        ledger.advance(6)


def test_declared_source_off_and_on_jumps():
    history = RateHistory([0, 1, 3, 4], [[2], [0], [5]], [[2], [0], [5]])
    assert history.rate_at(1, side="left")[0] == 2
    assert history.rate_at(1)[0] == 0
    assert history.rate_at(3, side="left")[0] == 0
    assert history.rate_at(3)[0] == 5
    np.testing.assert_array_equal(history.integrate(0, 4), [7])
    np.testing.assert_array_equal(history.integrate(1, 3), [0])


def test_checkpoint_continuation_does_not_respend_cumulative_emission():
    history = RateHistory.from_knots([0, 10], [[1, 2], [1, 2]])
    ledger = SourceLedger(history, [100, 200])
    np.testing.assert_array_equal(ledger.advance(3), [3, 6])
    restored = SourceLedger.from_checkpoint(history, [100, 200], ledger.checkpoint())
    np.testing.assert_array_equal(restored.advance(5), [2, 4])
    np.testing.assert_array_equal(restored.emitted, [5, 10])
    altered = copy.deepcopy(ledger.checkpoint())
    altered["emitted_high"][0] += 1
    with pytest.raises(ValueError):
        SourceLedger.from_checkpoint(history, [100, 200], altered)


def test_checkpoint_rejects_changed_source_clock_and_gas():
    history = RateHistory.from_knots([0, 2], [[1], [2]])
    ledger = SourceLedger(history, [100])
    ledger.advance(1)
    with pytest.raises(ValueError):
        ledger.advance(2, atoms=[101])
    assert ledger.time == 1
    changed = RateHistory.from_knots([0, 2], [[1], [2]], clock="different seconds")
    with pytest.raises(ValueError):
        SourceLedger.from_checkpoint(changed, [100], ledger.checkpoint())
    with pytest.raises(ValueError):
        SourceLedger.from_checkpoint(history, [101], ledger.checkpoint())


@pytest.mark.parametrize("scale", [1e-120, 1e-20, 1., 1e20, 1e120])
def test_rate_unit_rescaling(scale):
    history = RateHistory.from_knots([0, 1, 7], np.array([[2], [4], [8]]) * scale)
    assert history.integrate(0, 7)[0] / scale == pytest.approx(39, rel=3e-16)


def test_large_cumulative_then_small_increment_is_not_subtracted_away():
    history = RateHistory([0, 1, 2], [[1e20], [.001]], [[1e20], [.001]])
    ledger = SourceLedger(history, [1e30])
    ledger.advance(1)
    np.testing.assert_array_equal(ledger.advance(2), [.001])
    assert ledger.emitted_high[0] == 1e20
    assert ledger.emitted_low[0] == .001
    restored = SourceLedger.from_checkpoint(history, [1e30], ledger.checkpoint())
    assert restored.emitted_low[0] == .001


def test_small_duration_avoids_intermediate_rate_sum_overflow():
    history = RateHistory.from_knots([0, 1e-100], [[1e308], [1e308]])
    assert history.integrate(0, 1e-100)[0] == pytest.approx(1e208)
    with pytest.raises(OverflowError):
        RateHistory.from_knots([0, 10], [[1e308], [1e308]]).integrate(0, 10)


def test_negative_invalid_or_out_of_range_sources_rejected():
    for times, rates in (([0, 0], [[1], [2]]), ([0, 1], [[1], [-1]]),
                         ([0, np.nan], [[1], [2]]), ([0, 1], [[1], [np.inf]])):
        with pytest.raises(ValueError):
            RateHistory.from_knots(times, rates)
    history = RateHistory.from_knots([0, 1], [[1], [2]])
    for start, stop in ((-1, 1), (0, 2), (1, 0), (0, np.inf)):
        with pytest.raises(ValueError):
            history.integrate(start, stop)
    # A negative difference of cumulative inventories is not accepted as a rate.
    with pytest.raises(ValueError):
        RateHistory.from_knots([0, 1], [[-.003], [.01]])


def test_native_sfrd_units_and_no_second_density_factor():
    astro = dict(ALPHA_ESC=0., F_ESC10_ACG=-1., F_ESC7_MCG=-2., POP2_ION=100., POP3_ION=1000.)
    history = native_sfrd_history([0, 7], [[2], [2]], [[3], [3]], astro=astro, rho_b=10.)
    np.testing.assert_array_equal(history.integrate(0, 7), [35.])


def test_continuous_sampling_and_jump_are_distinct_explicit_models():
    smoothed = RateHistory.from_knots([0, 1, 2], [[1], [0], [0]])
    switched = RateHistory([0, 1, 2], [[1], [0]], [[1], [0]])
    assert smoothed.integrate(0, 2)[0] == .5
    assert switched.integrate(0, 2)[0] == 1


def test_causal_history_extension_preserves_prior_emission():
    history = RateHistory.from_knots([0, 1], [[2], [2]])
    ledger = SourceLedger(history, [100])
    np.testing.assert_array_equal(ledger.advance(1), [2])
    # Feedback can supply the next interval after the current step. Future
    # samples do not have to be known when creating the initial ledger.
    extended = history.append([1, 3], [[0]], [[0]])
    ledger.extend_history(extended)
    np.testing.assert_array_equal(ledger.advance(3), [0])
    np.testing.assert_array_equal(ledger.emitted, [2])
    restarted = SourceLedger.from_checkpoint(extended, [100], ledger.checkpoint())
    np.testing.assert_array_equal(restarted.emitted, [2])
    changed_past = RateHistory([0, 1, 3], [[3], [0]], [[3], [0]])
    with pytest.raises(ValueError):
        ledger.extend_history(changed_past)
    with pytest.raises(ValueError):
        history.append([1.1, 3], [[0]], [[0]])
