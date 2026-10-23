"""Opening bound, eligibility, ledger and disabled-mode checks for source merging."""
import unittest

import numpy as np

from merge_sources import merge, solve, split_ledger


class TestMergeSources(unittest.TestCase):
    def example(self):
        capacity = np.ones((16,)*3)
        initial = capacity.copy()
        source = np.zeros_like(capacity)
        source[6,6,6], source[7,6,6] = 2., 1.
        return source, capacity, initial

    def test_disabled_exact_identity(self):
        source, capacity, initial = self.example()
        result = merge(source, capacity, initial, theta=0.)
        np.testing.assert_array_equal(result['source'], source)
        np.testing.assert_array_equal(result['original_indices'], result['anchor_indices'])
        np.testing.assert_array_equal(result['weights'], 1.)
        self.assertTrue(result['diagnostics']['source_geometry_exact'])

    def test_original_anchor_and_bound(self):
        source, capacity, initial = self.example()
        result = merge(source, capacity, initial, theta=.2)
        self.assertEqual(result['diagnostics']['aggregate_sources'], 1)
        self.assertEqual(result['source'][6,6,6], 3.)
        self.assertTrue(np.all(source.ravel()[result['anchor_indices']] > 0))
        self.assertTrue(np.all(result['displacements'] <= .2*result['surface_distance_lower_bound']))
        self.assertLessEqual(result['diagnostics']['max_displacement_to_clearance'], .2)
        self.assertAlmostEqual(result['source'].sum(), source.sum())

    def test_never_merge_partly_neutral_source(self):
        source, capacity, initial = self.example()
        initial[7,6,6] = np.nextafter(1., 0.)
        result = merge(source, capacity, initial, theta=.9)
        np.testing.assert_array_equal(result['source'], source)

    def test_neutral_wall_and_periodic_padding_are_conservative(self):
        source, capacity, initial = self.example()
        initial[8,:,:] = 0.
        result = merge(source, capacity, initial, theta=.9)
        np.testing.assert_array_equal(result['source'], source)
        # A periodic all-transparent box could allow merging across its edge,
        # but opaque padding and nonwrapping blocks deliberately reject it.
        source.fill(0.)
        source[0,7,7], source[1,7,7] = 2., 1.
        result = merge(source, capacity, capacity, theta=.9)
        np.testing.assert_array_equal(result['source'], source)

    def test_bookkeeping_split_preserves_every_original_budget(self):
        source, capacity, initial = self.example()
        grouping = merge(source, capacity, initial, theta=.2)
        outputs = [split_ledger(grouping['source']*fraction, grouping)
                   for fraction in (.6,.25,.15)]
        np.testing.assert_allclose(sum(outputs), source, rtol=0, atol=2e-15)
        for output, fraction in zip(outputs, (.6,.25,.15)):
            self.assertAlmostEqual(output.sum(), source.sum()*fraction)

    def test_wrapper_remaps_all_three_ledgers(self):
        source, capacity, initial = self.example()
        def transport(merged, cap, old, **kwargs):
            return dict(source_absorbed=merged*.6, source_escaped=merged*.25,
                        source_retained=merged*.15, absorbed=np.zeros_like(cap),
                        audit={'seconds':0.})
        result = solve(source, capacity, initial, theta=.2, solver=transport)
        np.testing.assert_allclose(result['source_absorbed'], source*.6)
        self.assertLess(result['audit']['merged_original_source_relative_ledger_max'], 1e-14)
        self.assertFalse(result['audit']['source_geometry_exact'])

    def test_zero_sources_and_invalid_inputs(self):
        source, capacity, initial = self.example()
        zero = merge(source*0., capacity, initial, theta=.2)
        self.assertEqual(zero['diagnostics']['aggregate_sources'], 0)
        for theta in (-.1,1.,float('nan')):
            with self.assertRaises(ValueError):
                merge(source, capacity, initial, theta=theta)
        with self.assertRaises(ValueError):
            merge(source, capacity, initial*2., theta=.2)
        grouping = merge(source, capacity, initial, theta=.2)
        with self.assertRaises(ValueError):
            split_ledger(np.ones_like(source), grouping)


if __name__ == '__main__':
    unittest.main()
