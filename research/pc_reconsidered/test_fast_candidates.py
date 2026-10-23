"""Independent count, support, and counterexample tests for fast candidates."""

import unittest

import numpy as np

from fast_candidates import solve, source_consistent_es


METHODS = ("es-global", "es-front", "radial-shell", "voronoi")


class TestFastCandidates(unittest.TestCase):
    def check_counts(self, source, capacity, initial, result):
        absorbed = result["absorbed"]
        self.assertTrue(np.all(absorbed >= 0))
        self.assertLessEqual(float(np.max(absorbed-(capacity-initial))), 1e-10)
        self.assertGreaterEqual(result["unabsorbed"], 0)
        self.assertAlmostEqual(float(source.sum()),
                               float(absorbed.sum())+result["unabsorbed"], places=8)
        np.testing.assert_allclose(result["ionized"], initial+absorbed,
                                   rtol=0, atol=1e-12)

    def test_zero_and_saturation(self):
        cap = np.ones((6,)*3)
        initial = cap*.2
        for method in METHODS:
            with self.subTest(method=method):
                zero = solve(np.zeros_like(cap), cap, initial, method=method)
                np.testing.assert_array_equal(zero["absorbed"], 0)
                saturated = solve(cap, cap, cap, method=method)
                self.assertEqual(saturated["unabsorbed"], cap.size)
                np.testing.assert_array_equal(saturated["absorbed"], 0)

    def test_all_partial_requires_no_correction(self):
        rng = np.random.default_rng(523)
        cap = rng.uniform(.1, 3, (10,)*3)
        initial = cap*rng.uniform(0, .3, cap.shape)
        source = (cap-initial)*rng.uniform(0, .6, cap.shape)
        for method in METHODS:
            with self.subTest(method=method):
                result = solve(source, cap, initial, method=method)
                np.testing.assert_allclose(result["absorbed"], source,
                                           rtol=3e-10, atol=1e-12)
                self.check_counts(source, cap, initial, result)

    def test_random_sparse_and_initial_cavity_counts(self):
        rng = np.random.default_rng(2)
        cap = rng.uniform(.1, 2, (8,)*3)
        source = rng.uniform(0, .6, cap.shape)
        source[source < .5] = 0
        source[(3,)*3] = 100
        initial = np.zeros_like(cap)
        initial[2:5, 2:5, 2:5] = cap[2:5, 2:5, 2:5]
        for method in METHODS:
            with self.subTest(method=method):
                result = solve(source, cap, initial, method=method, max_radius=4)
                self.check_counts(source, cap, initial, result)
                np.testing.assert_array_equal(result["absorbed"][2:5, 2:5, 2:5], 0)

    def test_isolated_shell_symmetry_and_exact_radius(self):
        cap = np.ones((16,)*3)
        center = np.array((8,)*3)
        distance2 = np.sum((np.indices(cap.shape)-center[:, None, None, None])**2,
                           axis=0)
        source = np.zeros_like(cap)
        source[tuple(center)] = np.count_nonzero(distance2 <= 9)
        for method in ("radial-shell", "voronoi"):
            result = solve(source, cap, method=method, max_radius=6)
            np.testing.assert_allclose(result["absorbed"], distance2 <= 9,
                                       rtol=0, atol=1e-13)
            self.check_counts(source, cap, np.zeros_like(cap), result)

    def test_translation_covariance(self):
        source = np.zeros((12,)*3)
        source[2, 3, 4], source[6, 7, 8] = 100, 40
        cap = np.ones_like(source)
        cap[3:5, 4:6, 3:5] = 2
        for method in METHODS:
            with self.subTest(method=method):
                a = solve(source, cap, method=method, max_radius=5)
                b = solve(np.roll(source, 3, axis=0), np.roll(cap, 3, axis=0),
                          method=method, max_radius=5)
                np.testing.assert_allclose(b["absorbed"],
                                           np.roll(a["absorbed"], 3, axis=0),
                                           rtol=0, atol=2e-8)

    def test_radius_is_original_source_distance(self):
        cap = np.ones((14,)*3)
        source = np.zeros_like(cap)
        source[7, 7, 7] = 10000
        distance = np.sqrt(np.sum((np.indices(cap.shape)-7)**2, axis=0))
        for method in ("radial-shell", "voronoi"):
            result = solve(source, cap, method=method, max_radius=3)
            self.assertEqual(result["absorbed"][distance > 3].sum(), 0)
            self.assertGreater(result["unabsorbed"], 0)
            self.assertEqual(np.count_nonzero(result["remaining_source"]), 1)
            self.check_counts(source, cap, np.zeros_like(cap), result)

    def test_global_es_can_violate_disconnected_source_budgets(self):
        # Two disconnected source-target groups within maximum radius 2.
        # A has 10 photons but one atom; B has one photon but 10 atoms.
        # Physical maximum absorption is therefore 1+1=2, not 11.
        source = np.zeros((16,)*3)
        cap = np.zeros_like(source)
        source[2, 2, 2], cap[3, 2, 2] = 10, 1
        source[10, 10, 10], cap[11, 10, 10] = 1, 10
        for method in ("radial-shell", "voronoi"):
            radial = solve(source, cap, method=method, max_radius=2)
            self.assertAlmostEqual(radial["absorbed"].sum(), 2)
            self.assertAlmostEqual(radial["remaining_source"][2, 2, 2], 9)
        for method in ("es-global", "es-front"):
            result = solve(source, cap, method=method, max_radius=2)
            self.assertGreater(result["absorbed"].sum(), 2)
            self.assertFalse(result["diagnostics"]["source_attribution"])

    def test_positive_capacity_screen_is_not_ray_opacity(self):
        # In this frozen increment, a very opaque wall collects all photons
        # before radial shells can reach behind it.  Conversely distant targets
        # can still be offered photons without any explicit opacity integral.
        # This checks the declared limitation flag rather than asserting that
        # spherical ordering is a transport solution.
        cap = np.ones((12,)*3)
        cap[8] = 1e5
        source = np.zeros_like(cap)
        source[6, 6, 6] = 1000
        result = solve(source, cap, method="radial-shell", max_radius=5)
        self.assertIn("foreground opacity", result["diagnostics"]["limitation"])
        self.check_counts(source, cap, np.zeros_like(cap), result)

    def test_control_reports_signed_residual(self):
        cap = np.ones((10,)*3)
        source = np.zeros_like(cap)
        source[4, 5, 5], source[6, 5, 5] = 100, 100
        result = source_consistent_es(source, cap, max_radius=4)
        self.assertAlmostEqual(result["diagnostics"]["photon_residual"],
                               source.sum()-result["absorbed"].sum())

    def test_voronoi_all_nearest_source_ties_are_shared(self):
        # Twelve equally near sources around one target exceed the fast query's
        # k=8, exercising complete tie collection.  Each gets the same debit.
        source = np.zeros((12,)*3)
        cap = np.zeros_like(source)
        center = np.array([6, 6, 6])
        offsets = np.array([[a, b, c] for a in (-1, 0, 1)
                            for b in (-1, 0, 1) for c in (-1, 0, 1)
                            if a*a+b*b+c*c == 2])
        positions = offsets+center
        source[tuple(positions.T)] = 1.
        cap[tuple(center)] = 6.
        result = solve(source, cap, method="voronoi", max_radius=2)
        self.assertAlmostEqual(result["absorbed"].sum(), 6.)
        np.testing.assert_allclose(result["remaining_source"][tuple(positions.T)],
                                   .5, rtol=0, atol=1e-14)

    def test_dense_overlap_retains_numerical_source_tails(self):
        # Numerous rounds leave exhausted-source tails far below the original
        # photon scale. They must remain accounted for without debit overflow.
        rng = np.random.default_rng(500)
        cap = rng.lognormal(0, .7, (32,)*3)
        initial = cap*(rng.random(cap.shape) < .03)
        source = np.zeros_like(cap)
        ids = rng.choice(cap.size, 500, replace=False)
        source.ravel()[ids] = rng.uniform(15, 100, len(ids))
        for method in ("radial-shell", "voronoi"):
            result = solve(source, cap, initial, method=method, max_radius=6)
            self.check_counts(source, cap, initial, result)
            self.assertAlmostEqual(result["remaining_source"].sum(),
                                   result["unabsorbed"])

    def test_invalid_values_rejected(self):
        source = np.zeros((4,)*3)
        cap = np.ones_like(source)
        for method in METHODS:
            with self.subTest(method=method):
                with self.assertRaises(ValueError):
                    solve(source, cap, cap*1.1, method=method)
                with self.assertRaises(ValueError):
                    solve(source-1, cap, method=method)
                with self.assertRaises(ValueError):
                    solve(source, cap, method=method, radii=[2, 1])


if __name__ == "__main__":
    unittest.main()
