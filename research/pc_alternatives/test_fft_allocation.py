"""Independent accounting/symmetry checks and explicit physics counterexamples."""

import unittest

import numpy as np

from fft_allocation import allocate, make_probe_results, monotonicity_probes


class MatchingTests(unittest.TestCase):
    def test_zero_source(self):
        p = np.zeros((8, 8, 8))
        result = allocate(p, np.ones_like(p), [1, 2, 4])
        np.testing.assert_array_equal(result["absorbed"], p)
        np.testing.assert_array_equal(result["remaining_source"], p)

    def test_shared_nonuniform_capacity(self):
        rng = np.random.default_rng(16)
        c = np.exp(rng.normal(size=(8, 8, 8)))
        p = rng.exponential(size=c.shape)
        p[rng.random(c.shape) < 0.8] = 0
        result = allocate(p, c, [1, 2, 4, 7], rounds_per_scale=3)
        self.assertGreaterEqual(result["remaining_source"].min(), 0)
        self.assertGreaterEqual(result["absorbed"].min(), 0)
        self.assertLessEqual(np.max(result["absorbed"] - c), 2e-14)
        self.assertLessEqual(np.max(result["source_debit"] - p), 2e-14)
        self.assertLess(
            abs(
                p.sum()
                - result["source_debit"].sum()
                - result["remaining_source"].sum()
            ),
            1e-12,
        )
        self.assertLess(
            abs(result["absorbed"].sum() - result["source_debit"].sum()), 1e-12
        )

    def test_translation_and_axis_rotation(self):
        rng = np.random.default_rng(7)
        c = rng.uniform(0.1, 3, (12,) * 3)
        p = rng.uniform(0, 8, c.shape) * (rng.random(c.shape) < 0.1)
        a = allocate(p, c, [1, 2, 4], rounds_per_scale=2)
        shift = (3, -2, 5)
        b = allocate(
            np.roll(p, shift, (0, 1, 2)),
            np.roll(c, shift, (0, 1, 2)),
            [1, 2, 4],
            rounds_per_scale=2,
        )
        np.testing.assert_allclose(
            b["absorbed"],
            np.roll(a["absorbed"], shift, (0, 1, 2)),
            atol=2e-13,
            rtol=1e-12,
        )
        d = allocate(
            p.transpose(2, 0, 1), c.transpose(2, 0, 1), [1, 2, 4], rounds_per_scale=2
        )
        np.testing.assert_allclose(
            d["absorbed"], a["absorbed"].transpose(2, 0, 1), atol=2e-13, rtol=1e-12
        )

    def test_saturation_and_surplus(self):
        c = np.ones((8,) * 3)
        p = np.zeros_like(c)
        p[0, 0, 0] = 600
        result = allocate(p, c, [1, 2, 4, 7], rounds_per_scale=3)
        np.testing.assert_allclose(result["absorbed"], c, atol=2e-13)
        self.assertAlmostEqual(result["remaining_source"].sum(), 88, places=10)

    def test_weak_source_and_enumeration(self):
        c = np.ones((8,) * 3)
        p = np.zeros_like(c)
        p[3, 3, 3] = 0.125
        result = allocate(p, c, [1, 2, 4])
        np.testing.assert_array_equal(result["absorbed"], p)
        catalogue = [((1, 1, 1), 12), ((3, 1, 1), 27), ((1, 1, 1), 9)]
        p.fill(0)
        q = np.zeros_like(p)
        for index, photons in catalogue:
            p[index] += photons
        for index, photons in reversed(catalogue):
            q[index] += photons
        a = allocate(p, c, [1, 2, 4])
        b = allocate(q, c, [1, 2, 4])
        np.testing.assert_array_equal(a["absorbed"], b["absorbed"])

    def test_monotonicity_failures_remain_visible(self):
        probes = monotonicity_probes()
        source = probes["source_increase"]["rounds_per_scale_refinement"]["2"]
        self.assertGreater(source["maximum_target_decrease"], 2e-5)
        self.assertGreater(
            probes["uniform_capacity_increase"]["maximum_ionized_fraction_increase"],
            4e-4,
        )

    def test_physics_failures_remain_visible(self):
        probes = make_probe_results()
        self.assertAlmostEqual(
            probes["opaque_screen"]["downstream_absorbed"], 0.5, places=12
        )
        self.assertLess(
            probes["opaque_screen"]["foreground_absorbed"],
            probes["opaque_screen"]["foreground_capacity"],
        )
        self.assertAlmostEqual(
            probes["equal_angle_density_bias"]["ratio"], 100, places=8
        )
        self.assertEqual(probes["unreachable"]["absorbed"], 0)
        self.assertLess(probes["sharp_k_real_kernel"]["minimum"], 0)


if __name__ == "__main__":
    unittest.main()
