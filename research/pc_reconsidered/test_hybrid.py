"""Independent contracts for variable angular sampling of shared events."""
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from hybrid import solve
from event_front import solve as fixed_solve


class AdaptiveEventsTests(unittest.TestCase):
    def test_common_level_matches_original_core(self):
        rng = np.random.default_rng(413)
        capacity = rng.uniform(.5, 1.5, size=(12,)*3)
        source = np.zeros_like(capacity)
        source[4, 4, 4] = 71.
        source[7, 5, 6] = 33.
        actual = solve(source, capacity, min_angles=192, max_angles=192,
                       cavity_bound=False, periodic=True)
        expected = fixed_solve(source, capacity, angles=192, periodic=True)
        np.testing.assert_array_equal(actual["absorbed"], expected["absorbed"])
        np.testing.assert_array_equal(actual["source_absorbed"], expected["source_absorbed"])
        self.assertLess(abs(actual["audit"]["ledger_residual"]), 1e-10)

    def test_mixed_levels_match_independent_disjoint_sources(self):
        capacity = np.ones((32,)*3)
        source = np.zeros_like(capacity)
        source[7, 7, 7] = 35.
        source[23, 23, 23] = 3.
        actual = solve(source, capacity, min_angles=24, max_angles=3072,
                       periodic=False)
        expected = np.zeros_like(source)
        counts = actual["angle_counts"]
        self.assertNotEqual(counts[7, 7, 7], counts[23, 23, 23])
        for location in [(7, 7, 7), (23, 23, 23)]:
            one = np.zeros_like(source)
            one[location] = source[location]
            expected += fixed_solve(one, capacity, angles=int(counts[location]))["absorbed"]
        np.testing.assert_allclose(actual["absorbed"], expected, atol=2e-15, rtol=0)
        self.assertLess(actual["audit"]["per_angle_ledger_max"], 1e-12)

    def test_all_partial_remains_local_and_exact(self):
        rng = np.random.default_rng(752)
        capacity = rng.uniform(.5, 2., size=(10,)*3)
        source = .1 * capacity
        out = solve(source, capacity)
        np.testing.assert_array_equal(out["absorbed"], source)
        self.assertEqual(out["unabsorbed"], 0.)
        self.assertEqual(out["audit"]["rays_materialized"], 0)

    def test_initial_cavity_increases_sampling(self):
        n = 20
        grid = np.indices((n,)*3)
        cavity = sum((x-10)**2 for x in grid) < 4**2
        capacity = np.ones((n,)*3)
        initial = cavity.astype(float)
        source = np.zeros_like(capacity)
        source[10, 10, 10] = .1
        out = solve(source, capacity, initial, max_angles=6144, periodic=False)
        self.assertGreater(out["angle_counts"][10, 10, 10], 1000)
        self.assertAlmostEqual(out["absorbed"].sum(), .1, places=12)
        self.assertEqual(out["absorbed"][cavity].sum(), 0.)

    def test_partial_origin_inherits_adjacent_cavity(self):
        n = 20
        grid = np.indices((n,)*3)
        cavity = sum((x-10)**2 for x in grid) < 4**2
        capacity = np.ones((n,)*3)
        source = np.zeros_like(capacity)
        source[10, 10, 10] = .1
        counts = []
        for origin_capacity in (0., 1e-8, .05):
            initial = cavity.astype(float)
            initial[10, 10, 10] -= origin_capacity
            out = solve(source, capacity, initial, max_angles=6144, periodic=False)
            counts.append(out["angle_counts"][10, 10, 10])
            self.assertAlmostEqual(out["absorbed"][10, 10, 10], origin_capacity, places=12)
            self.assertAlmostEqual(out["absorbed"].sum(), .1, places=12)
        self.assertGreater(counts[0], 1000)
        self.assertEqual(counts, [counts[0]] * 3)

    def test_periodic_cavity_translation_preserves_predictor_and_map(self):
        n = 20
        grid = np.indices((n,)*3)
        initial = (sum((x-10)**2 for x in grid) < 4**2).astype(float)
        initial[10, 10, 10] -= .05
        capacity = np.ones_like(initial)
        source = np.zeros_like(initial)
        source[10, 10, 10] = .1
        shift = (8, -9, 0)
        original = solve(source, capacity, initial, periodic=True)
        shifted = solve(np.roll(source, shift, (0, 1, 2)), capacity,
                        np.roll(initial, shift, (0, 1, 2)), periodic=True)
        for key in ("angle_counts", "estimated_radius"):
            np.testing.assert_array_equal(shifted[key], np.roll(original[key], shift, (0, 1, 2)))
        np.testing.assert_allclose(shifted["absorbed"],
                                   np.roll(original["absorbed"], shift, (0, 1, 2)),
                                   atol=2e-15, rtol=0)

    def test_winding_transparent_component_uses_horizon(self):
        n = 12
        capacity = np.ones((n,)*3)
        initial = np.zeros_like(capacity)
        initial[:, 6, 6] = 1.
        source = np.zeros_like(initial)
        source[6, 6, 6] = .01
        horizon = n * np.sqrt(3.)
        out = solve(source, capacity, initial, periodic=True, max_angles=1536)
        self.assertEqual(out["estimated_radius"][6, 6, 6], horizon)
        self.assertEqual(out["angle_counts"][6, 6, 6], 1536)

    def test_fixed_angles_are_invariant_under_time_partition(self):
        capacity = np.ones((18,)*3)
        source = np.zeros_like(capacity)
        source[7, 9, 9] = 160.
        source[11, 9, 9] = 120.
        options = {"min_angles": 384, "max_angles": 384, "periodic": True}
        full = solve(source, capacity, **options)["absorbed"]
        for steps in (2, 4):
            old = np.zeros_like(capacity)
            for _ in range(steps):
                old += solve(source / steps, capacity, old, **options)["absorbed"]
            np.testing.assert_allclose(old, full, atol=2e-14, rtol=0)

    def test_source_off_does_not_silently_reemit_retained_photons(self):
        capacity = np.ones((8,)*3)
        source = np.zeros_like(capacity)
        source[4, 4, 4] = 20.
        first = solve(source, capacity, maximum_distance=.1)
        self.assertAlmostEqual(first["absorbed"].sum(), 1.)
        self.assertAlmostEqual(first["audit"]["retained"], 19.)
        second = solve(np.zeros_like(source), capacity, first["absorbed"])
        np.testing.assert_array_equal(second["absorbed"], np.zeros_like(source))
        self.assertEqual(second["unabsorbed"], 0.)
        self.assertAlmostEqual(first["audit"]["photons"],
                               first["absorbed"].sum() + first["unabsorbed"])

    def test_saturation_retains_surplus(self):
        capacity = np.ones((6,)*3)
        source = np.full_like(capacity, 2.)
        out = solve(source, capacity, min_angles=24, max_angles=24, periodic=True)
        np.testing.assert_array_equal(out["absorbed"], capacity)
        self.assertAlmostEqual(out["unabsorbed"], capacity.sum(), places=10)

    def test_fully_transparent_periodic_limit_needs_no_rays(self):
        capacity = np.ones((12,)*3)
        source = np.random.default_rng(144).uniform(.01, 1., size=capacity.shape)
        out = solve(source, capacity, capacity, periodic=True)
        np.testing.assert_array_equal(out["source_retained"], source)
        self.assertEqual(out["absorbed"].sum(), 0.)
        self.assertAlmostEqual(out["unabsorbed"], source.sum(), places=11)
        self.assertEqual(out["audit"]["rays_materialized"], 0)
        self.assertEqual(out["audit"]["angles_sum"], 0)

    def test_nonphysical_initial_state_is_rejected(self):
        shape = (3,)*3
        with self.assertRaises(ValueError):
            solve(np.zeros(shape), np.ones(shape), np.full(shape, 2.))


if __name__ == "__main__":
    unittest.main()
