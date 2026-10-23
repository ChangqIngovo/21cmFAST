"""Cross-component checks independent of each implementation's own tests."""
from accuracy_acceptance import (
    coupled_opacity_order_and_slab, event_symmetry_and_clock,
    numeric_domain_rejection, packet_reference_cases, source_history_refinement,
    ray_segment_reference_cases, angular_blind_spot,
)


def test_event_cubic_symmetry_periodic_translation_and_clock_units():
    result = event_symmetry_and_clock()
    assert result['passed'], result


def test_event_unsupported_numerical_domains_fail_closed():
    result = numeric_domain_rejection()
    assert result['passed'], result


def test_finite_c_queue_matches_eager_crossing_reference():
    result = packet_reference_cases()
    assert result['passed'], result


def test_random_source_history_insertion_and_ledger_subdivision():
    result = source_history_refinement()
    assert result['passed'], result


def test_multibeam_order_reference_and_column_spatial_invariance():
    result = coupled_opacity_order_and_slab()
    assert result['passed'], result


def test_oblique_chords_match_independent_voxel_slab_intersections():
    result = ray_segment_reference_cases()
    assert result['passed'], result


def test_zero_hit_angular_aliasing_is_unresolved_not_accepted():
    result = angular_blind_spot()
    assert result['exact_geometric_absorbed_fraction'] > 1e-6
    assert result['computed_absorbed'] == 0
    assert result['passed'], result
