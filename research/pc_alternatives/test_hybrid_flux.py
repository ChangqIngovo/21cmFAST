"""Independent geometry and local divergence checks for vacuum face flux."""

import unittest

import numpy as np

from hybrid_flux_probe import (
    IsolatedFaceFlux,
    boundary_stats,
    common_visibility_box_certificate,
    monopole_probe,
    rectangular_face_kernel,
)


class FaceFluxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator = IsolatedFaceFlux((12,) * 3, workers=1)

    def test_face_solid_angle_against_area_quadrature(self):
        nodes, weights = np.polynomial.legendre.leggauss(64)
        for normal, first, second in ((0.5, 0, 0), (4.5, 1, -3), (-1.5, 1, 2)):
            u = first + 0.5 * nodes[:, None]
            v = second + 0.5 * nodes[None, :]
            integral = np.sum(
                0.25
                * weights[:, None]
                * weights[None, :]
                * normal
                / (4 * np.pi * (normal**2 + u * u + v * v) ** 1.5)
            )
            field = rectangular_face_kernel((12,) * 3, 0)
            value = field[int(normal - 0.5) % 24, first % 24, second % 24]
            self.assertAlmostEqual(value, integral, places=14)

    def test_single_source_cube_sides_and_divergence(self):
        source = np.zeros((12,) * 3)
        source[5, 5, 5] = 1
        result = self.evaluator(source)
        for axis, face in enumerate(result["faces"]):
            low = [5, 5, 5]
            high = [5, 5, 5]
            high[axis] += 1
            self.assertAlmostEqual(face[tuple(low)], -1 / 6, places=14)
            self.assertAlmostEqual(face[tuple(high)], 1 / 6, places=14)
        np.testing.assert_allclose(result["divergence"], source, atol=3e-15, rtol=0)

    def test_many_sources_convex_positive_and_cubic_rotation(self):
        rng = np.random.default_rng(836)
        source = np.zeros((12,) * 3)
        source[3:9, 3:9, 3:9] = rng.exponential(size=(6,) * 3)
        result = self.evaluator(source)
        np.testing.assert_allclose(result["divergence"], source, atol=5e-14, rtol=0)
        cavity = np.zeros_like(source, dtype=bool)
        cavity[1:11, 1:11, 1:11] = True
        boundary = boundary_stats(result["faces"], cavity)
        self.assertTrue(common_visibility_box_certificate(source, cavity)["certified"])
        self.assertEqual(boundary["negative_outward_flux"], 0)
        self.assertAlmostEqual(boundary["net_outward_flux"], source.sum(), places=11)
        rotated = self.evaluator(source.transpose(1, 2, 0))
        for new_axis, old_axis in enumerate((1, 2, 0)):
            np.testing.assert_allclose(
                rotated["faces"][new_axis],
                result["faces"][old_axis].transpose(1, 2, 0),
                atol=3e-15,
                rtol=3e-14,
            )

    def test_monopole_uniform_bound(self):
        result = monopole_probe()
        for entry in result["results"]:
            self.assertLess(
                entry["max_absolute_error"], entry["uniform_absolute_bound"]
            )

    def test_visibility_rejects_hole_and_outside_source(self):
        source = np.zeros((12,) * 3)
        source[5, 5, 5] = 1
        cavity = np.zeros_like(source, dtype=bool)
        cavity[1:11, 1:11, 1:11] = True
        cavity[8, 5, 5] = False
        self.assertFalse(common_visibility_box_certificate(source, cavity)["certified"])
        cavity[8, 5, 5] = True
        source[0, 5, 5] = 1
        certificate = common_visibility_box_certificate(source, cavity)
        self.assertFalse(certificate["certified"])
        self.assertFalse(certificate["source_containment"])


if __name__ == "__main__":
    unittest.main()
