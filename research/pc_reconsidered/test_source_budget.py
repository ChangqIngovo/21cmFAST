"""Independent source/atom count and emission-history checks."""

import numpy as np
import pytest

from source_budget import (
    DecreasingCumulativeBudget,
    allocate_global_history_step,
    audit_cumulative_map_change,
    build_mass_normalized_sources,
    integrate_rate_step,
)


def test_no_overlap_two_cell_counterexample_identifies_weighting_mismatch():
    # The original volume correction has sum(eta)=Q=.2; each cell is partial.
    # Actual local source and ionized-atom counts are rho*eta, totalling .3.
    # No bubble overlap is present, so this .1 is not a lost-photon correction.
    rho = np.array([1.5, .5])
    eta = np.array([.2, 0.])
    sources = build_mass_normalized_sources(rho, {"acg": eta}, {"acg": .2})
    d = sources.diagnostics["species"]["acg"]
    assert d["count_implied_by_volume_normalization"] == pytest.approx(.3)
    assert d["volume_normalization_count_excess"] == pytest.approx(.1)
    np.testing.assert_allclose(sources.total, [.2, 0.], rtol=1e-15, atol=0)
    assert np.sum(sources.total / rho) == pytest.approx(2 / 15)
    assert not sources.total.flags.writeable


def test_species_budgets_and_clustering_are_preserved_separately():
    rho = np.array([.2, 1.8, 1.0])
    eta = {"acg": np.array([1., 3., 2.]), "mcg": np.array([4., 1., 2.])}
    source = build_mass_normalized_sources(rho, eta, {"acg": 9., "mcg": 2.})
    assert source.by_species["acg"].sum() == pytest.approx(9.)
    assert source.by_species["mcg"].sum() == pytest.approx(2.)
    assert source.total.sum() == pytest.approx(11.)
    assert source.by_species["acg"][1] / source.by_species["acg"][0] == pytest.approx(27.)
    # An arbitrary prior species-wise volume correction cancels exactly.
    scaled = build_mass_normalized_sources(rho, {k: 7 * v for k, v in eta.items()},
                                           {"acg": 9., "mcg": 2.})
    np.testing.assert_allclose(source.total, scaled.total, rtol=1e-15)


def test_density_uniform_limit_and_zero_source_support():
    source = build_mass_normalized_sources(np.ones(3), {"a": [1., 3., 2.]}, {"a": 6.})
    np.testing.assert_array_equal(source.total, [1., 3., 2.])
    zero = build_mass_normalized_sources([0., 1.], {"a": [1., 0.]}, {"a": 0.})
    np.testing.assert_array_equal(zero.total, [0., 0.])
    with pytest.raises(ValueError, match="no source support"):
        build_mass_normalized_sources([0., 1.], {"a": [1., 0.]}, {"a": 1.})


def test_cumulative_map_motion_is_not_misidentified_as_new_emission():
    delta, diagnostic = audit_cumulative_map_change([4., 0.], [0., 5.])
    np.testing.assert_array_equal(delta, [-4., 5.])
    assert diagnostic["net_change"] == 1.
    assert diagnostic["extra_photons_if_negative_changes_are_clipped"] == 4.
    assert not diagnostic["can_be_nonnegative_emission_without_remapping"]
    # A globally specified increment of one is placed using the positive rate
    # shape; the old four photons are neither re-emitted nor deleted here.
    step = allocate_global_history_step({"a": 4.}, {"a": 5.}, {"a": [0., 2.]})
    np.testing.assert_array_equal(step.total, [0., 1.])


def test_global_cumulative_decline_is_rejected_per_species_without_clipping():
    with pytest.raises(DecreasingCumulativeBudget, match="acg"):
        allocate_global_history_step({"acg": 4., "mcg": 0.},
                                     {"acg": 3., "mcg": 5.},
                                     {"acg": [1.], "mcg": [1.]})
    with pytest.raises(ValueError, match="no source support"):
        allocate_global_history_step({"a": 0.}, {"a": 1.}, {"a": [0.]})


def test_instantaneous_linear_rate_integral_and_subdivision():
    left = np.array([2., 3.])
    right = np.array([10., 7.])
    whole = integrate_rate_step(left, right, 4.)
    mid = (left + right) / 2
    halves = integrate_rate_step(left, mid, 2.) + integrate_rate_step(mid, right, 2.)
    np.testing.assert_array_equal(whole, [24., 20.])
    np.testing.assert_array_equal(whole, halves)
    # There is deliberately no rho argument: native SFRD already has density.
    step = allocate_global_history_step({"a": 7.}, {"a": 18.}, {"a": whole})
    assert step.total.sum() == pytest.approx(11.)
    assert step.diagnostics["species"]["a"]["rate_rescaling_factor"] == .25


def test_source_off_does_not_reemit_old_cumulative_photons():
    step = allocate_global_history_step({"a": 100.}, {"a": 100.}, {"a": [0., 0.]})
    np.testing.assert_array_equal(step.total, [0., 0.])
    np.testing.assert_array_equal(integrate_rate_step([0., 0.], [0., 0.], 10.), [0., 0.])


@pytest.mark.parametrize("bad", [-1., np.nan, np.inf, 1j])
def test_invalid_source_states_are_rejected(bad):
    with pytest.raises(ValueError):
        build_mass_normalized_sources([1.], {"a": [bad]}, {"a": 1.})
    with pytest.raises(ValueError):
        build_mass_normalized_sources([1.], {"a": [1.]}, {"a": bad})
    with pytest.raises(ValueError):
        integrate_rate_step([1.], [1.], bad)


def test_rate_integration_avoids_intermediate_overflow():
    value = integrate_rate_step([1e308], [1e308], 1e-100)
    assert value[0] == pytest.approx(1e208)
    with pytest.raises(OverflowError):
        integrate_rate_step([1e308], [1e308], 100.)
