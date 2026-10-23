"""Analytic checks that auto-power comparisons alone miss phase errors."""

import numpy as np

from validation_native_observables import statistics


def test_opposite_field_has_same_auto_power_but_four_times_residual_power():
    field = np.random.default_rng(61023).normal(size=(16,)*3)
    result = statistics(-field, field, 100.)
    assert result['shell_weighted_absolute_power_difference'] == 0.
    np.testing.assert_allclose(result['all_nonzero_modes_residual_power_over_reference'], 4., atol=2e-14)
    for shell in result['shells']:
        np.testing.assert_allclose([shell['auto_power_ratio'], shell['cross_coherence'],
                                   shell['residual_power_over_reference']], [1., -1., 4.], atol=2e-14)


def test_mean_change_is_recorded_but_does_not_change_fluctuation_spectra():
    field = np.random.default_rng(61024).normal(size=(16,)*3)
    result = statistics(field+7., field, 100.)
    np.testing.assert_allclose(result['candidate_mean']-result['reference_mean'], 7., atol=1e-14)
    assert result['shell_weighted_absolute_power_difference'] < 1e-14
    assert result['all_nonzero_modes_residual_power_over_reference'] < 1e-25
