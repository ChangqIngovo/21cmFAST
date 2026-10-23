"""Semantic tests of constant-rate, infinite-c, first-opaque-voxel evolution."""

import ctypes as ct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import event_front


class EventFrontTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Compile to a unique directory: concurrent research processes must not
        # rewrite a shared .so while another process has that inode mapped.
        cls.build = tempfile.TemporaryDirectory(prefix="pc_event_review_")
        with patch("event_front.tempfile.gettempdir", return_value=cls.build.name):
            cls.lib = event_front.library()
        if hasattr(cls.lib, "event_front_last_max_heap_nodes"):
            cls.lib.event_front_last_max_heap_nodes.restype = ct.c_int64

    @classmethod
    def tearDownClass(cls):
        cls.build.cleanup()

    def check_ledger(self, result, capacity):
        audit = result["audit"]
        tolerance = 2e-11 * max(1, audit["photons"])
        for key in (
            "ledger_residual",
            "per_source_ledger_max",
            "per_angle_ledger_max",
            "independent_absorption_ledger",
        ):
            self.assertLessEqual(abs(audit[key]), tolerance, key)
        self.assertGreaterEqual(result["absorbed"].min(), -tolerance)
        self.assertLessEqual(np.max(result["absorbed"] - capacity), tolerance)
        if hasattr(self.lib, "event_front_last_max_heap_nodes"):
            self.assertLessEqual(
                self.lib.event_front_last_max_heap_nodes(), capacity.size
            )

    def test_zero_sources_and_zero_capacity(self):
        zero = np.zeros((6,) * 3)
        result = event_front.solve(zero, np.ones_like(zero), angles=12)
        np.testing.assert_array_equal(result["absorbed"], zero)
        self.check_ledger(result, np.ones_like(zero))
        source = zero.copy()
        source[2, 2, 2] = 7
        result = event_front.solve(source, zero, angles=12)
        self.assertAlmostEqual(result["audit"]["escaped"], 7, places=12)
        self.assertEqual(result["audit"]["absorbed"], 0)
        self.check_ledger(result, zero)

    def test_public_input_validation(self):
        source = np.zeros((3,) * 3)
        capacity = np.ones_like(source)
        invalid = (
            [{"duration": value} for value in (0, -1, np.nan, np.inf)]
            + [{"maximum_distance": value} for value in (0, -1, np.nan, np.inf)]
            + [
                {"ray_directions": value}
                for value in (
                    np.array([[0.0, 0, 0, 1]]),
                    np.empty((0, 3)),
                    np.array([[0.0, 0, 0]]),
                    np.array([[np.nan, 0, 1]]),
                )
            ]
        )
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                event_front.solve(source, capacity, **options)

    def test_exact_end_time_exhaustion_needs_no_future_ray(self):
        source = np.zeros((3,) * 3)
        capacity = np.zeros_like(source)
        source[1, 1, 1] = 1
        capacity[1, 1, 1] = 1
        result = event_front.solve(source, capacity, angles=48, duration=1)
        np.testing.assert_array_equal(result["absorbed"], capacity)
        self.assertEqual(result["audit"]["rays_materialized"], 0)
        self.check_ledger(result, capacity)

    def test_analytic_continuous_source_competition(self):
        # Along +x: source A(rate2) -> one atom -> source B(rate3,
        # capacity10) -> four atoms -> escape. Shared B exhausts at t=2.4,
        # the last cell at t=3.2. Source A/B absorb 6.4/9.6 by t=4.
        source = np.zeros((6,) * 3)
        capacity = np.zeros_like(source)
        source[1, 2, 2], source[3, 2, 2] = 2, 3
        capacity[1:5, 2, 2] = [1, 1, 10, 4]
        beam = np.array([[1.0, 0, 0]])
        result = event_front.solve(source, capacity, ray_directions=beam, duration=4)
        np.testing.assert_allclose(result["absorbed"], capacity, atol=3e-14)
        self.assertAlmostEqual(result["source_absorbed"][1, 2, 2], 6.4, places=12)
        self.assertAlmostEqual(result["source_absorbed"][3, 2, 2], 9.6, places=12)
        self.assertAlmostEqual(result["source_escaped"][1, 2, 2], 1.6, places=12)
        self.assertAlmostEqual(result["source_escaped"][3, 2, 2], 2.4, places=12)
        self.check_ledger(result, capacity)
        early = event_front.solve(source, capacity, ray_directions=beam, duration=2)
        expected = np.zeros_like(capacity)
        expected[1:4, 2, 2] = [1, 1, 8]
        np.testing.assert_allclose(early["absorbed"], expected, atol=3e-14)

    def test_saturation_escape_and_periodic_reservoir(self):
        source = np.zeros((4,) * 3)
        capacity = np.zeros_like(source)
        source[0, 1, 1] = 10
        capacity[:, 1, 1] = 1
        beam = np.array([[1.0, 0, 0]])
        for periodic in (False, True):
            result = event_front.solve(
                source,
                capacity,
                ray_directions=beam,
                periodic=periodic,
                maximum_distance=16,
            )
            np.testing.assert_allclose(result["absorbed"], capacity, atol=1e-14)
            self.assertAlmostEqual(
                result["audit"]["retained" if periodic else "escaped"], 6, places=12
            )
            self.assertEqual(result["audit"]["escaped" if periodic else "retained"], 0)
            self.check_ledger(result, capacity)

    def test_periodic_crossing_and_explicit_range_reservoir(self):
        source = np.zeros((8,) * 3)
        capacity = np.zeros_like(source)
        source[6, 3, 3] = 4
        capacity[1, 3, 3] = 100
        beam = np.array([[1.0, 0, 0]])
        wrapped = event_front.solve(
            source, capacity, ray_directions=beam, periodic=True, maximum_distance=3
        )
        self.assertAlmostEqual(wrapped["absorbed"][1, 3, 3], 4, places=12)
        limited = event_front.solve(
            source, capacity, ray_directions=beam, periodic=True, maximum_distance=2
        )
        self.assertEqual(limited["audit"]["absorbed"], 0)
        self.assertEqual(limited["audit"]["retained"], 4)
        self.assertEqual(limited["audit"]["escaped"], 0)
        opened = event_front.solve(
            source, capacity, ray_directions=beam, periodic=False, maximum_distance=3
        )
        self.assertEqual(opened["audit"]["escaped"], 4)
        for result in (wrapped, limited, opened):
            self.check_ledger(result, capacity)

    def test_time_partition_multisource(self):
        rng = np.random.default_rng(2094)
        source = rng.exponential(8, (8,) * 3) * (rng.random((8,) * 3) < 0.04)
        capacity = rng.uniform(0.1, 2, source.shape)
        capacity[rng.random(source.shape) < 0.3] = 0
        full = event_front.solve(source, capacity, angles=24, duration=1)
        remaining, total = capacity.copy(), np.zeros_like(capacity)
        for duration in (0.17, 0.29, 0.54):
            part = event_front.solve(source, remaining, angles=24, duration=duration)
            total += part["absorbed"]
            remaining = np.maximum(remaining - part["absorbed"], 0)
            self.check_ledger(part, remaining + part["absorbed"])
        np.testing.assert_allclose(full["absorbed"], total, atol=2e-13, rtol=2e-12)
        self.check_ledger(full, capacity)

    def test_source_increase_does_not_reduce_absorption(self):
        rng = np.random.default_rng(4704)
        for _ in range(12):
            source = rng.exponential(8, (7,) * 3) * (rng.random((7,) * 3) < 0.06)
            capacity = rng.uniform(0.1, 2, source.shape)
            added = source.copy()
            added[tuple(rng.integers(0, 7, 3))] += 10
            before = event_front.solve(source, capacity, angles=24)
            after = event_front.solve(added, capacity, angles=24)
            self.assertLessEqual(
                float(np.max(before["absorbed"] - after["absorbed"])), 3e-13
            )
            self.check_ledger(after, capacity)

    def test_rotated_oblique_beam(self):
        source = np.zeros((8,) * 3)
        capacity = np.zeros_like(source)
        source[1, 1, 2] = 7
        capacity[2, 2, 2], capacity[3, 3, 2] = 3, 10
        beam = np.array([[1.0, 1, 0]]) / np.sqrt(2)
        first = event_front.solve(source, capacity, ray_directions=beam)
        self.assertAlmostEqual(first["absorbed"][2, 2, 2], 3, places=12)
        self.assertAlmostEqual(first["absorbed"][3, 3, 2], 4, places=12)
        rotated = event_front.solve(
            source.transpose(1, 2, 0),
            capacity.transpose(1, 2, 0),
            ray_directions=beam[:, [1, 2, 0]],
        )
        np.testing.assert_allclose(
            rotated["absorbed"], first["absorbed"].transpose(1, 2, 0), atol=3e-14
        )


if __name__ == "__main__":
    unittest.main()
