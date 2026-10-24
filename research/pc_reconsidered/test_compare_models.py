"""Independent checks of benchmark oracles and reporting, not candidate code."""

from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

import numpy as np

from compare_models import (
    Case,
    field_statistics,
    fixture_cases,
    independent_ledger,
    make_cases,
    power_statistics,
    solve_event,
    sphere_reference,
)


class BenchmarkOracleTests(unittest.TestCase):
    def test_continuum_sphere_integral_and_symmetry(self):
        n = 24
        centre = np.full(3, n // 2)
        radius = 4.0
        field = sphere_reference(n, radius, centre, 64)
        q = 4 * np.pi * radius**3 / 3
        self.assertLess(abs(field.sum() - q) / q, 2e-5)
        self.assertGreater(np.count_nonzero((field > 0) & (field < 1)), 0)
        self.assertGreaterEqual(field.min(), 0)
        self.assertLessEqual(field.max(), 1)
        self.assertLess(np.max(abs(field - field.transpose(1, 0, 2))), 1e-13)

    def test_ledger_rejects_hidden_unabsorbed_photons(self):
        source = np.zeros((4,) * 3)
        source[1, 1, 1] = 1
        case = Case("ledger", source, np.ones_like(source), np.zeros_like(source), 2)
        absorbed = source * 0.4
        self.assertTrue(independent_ledger(absorbed, 0.6, case)["passed"])
        self.assertFalse(independent_ledger(absorbed, 0.0, case)["passed"])
        self.assertFalse(independent_ledger(source * 2, -1, case)["passed"])

    def test_partial_capacity_counts_include_existing_ionization(self):
        source = np.zeros((4,) * 3)
        source[1, 1, 1] = 1
        initial = np.zeros_like(source)
        initial[1, 1, 1] = 0.8
        case = Case("capacity", source, np.ones_like(source), initial, 2)
        self.assertFalse(independent_ledger(source * 0.3, 0.7, case)["passed"])
        self.assertTrue(independent_ledger(source * 0.2, 0.8, case)["passed"])

    def test_global_budget_cannot_hide_local_overionization_in_any_units(self):
        for units in (1e-20, 1., 1e20):
            source = np.zeros((4,) * 3)
            source[1, 1, 1] = 100 * units
            capacity = np.ones_like(source) * units
            capacity[2, 2, 2] = 1e-8 * units
            absorbed = np.zeros_like(source)
            absorbed[2, 2, 2] = 2e-8 * units
            case = Case("local", source, capacity, np.zeros_like(source), 2)
            result = independent_ledger(absorbed, float(source.sum() - absorbed.sum()), case)
            self.assertFalse(result["passed"])
            self.assertAlmostEqual(result["maximum_local_fraction_violation"], 1.)

    def test_weak_global_budget_has_no_absolute_unit_tolerance(self):
        source = np.ones((2,) * 3) * 1e-20
        case = Case("weak", source, np.ones_like(source), np.zeros_like(source), 2)
        self.assertFalse(independent_ledger(source * 2, 0., case)["passed"])
        self.assertTrue(independent_ledger(source, 0., case)["passed"])
        with self.assertRaises(AssertionError):
            independent_ledger(source, float("nan"), case)

    def test_auto_power_does_not_hide_opposite_phase(self):
        rng = np.random.default_rng(291)
        reference = rng.normal(size=(24,) * 3)
        candidate = -reference
        report = power_statistics(candidate, reference)
        self.assertLess(report["weighted_absolute_power_difference"], 1e-14)
        self.assertAlmostEqual(report["weighted_residual_power"], 4.0, places=12)
        self.assertTrue(all(abs(row["cross_coherence"] + 1) < 1e-13 for row in report["shells"]))

    def test_exact_all_partial_event_oracle(self):
        case = make_cases()["all-partial"]
        absorbed, remaining, _, _ = solve_event(case, 24)
        self.assertLess(np.max(abs(absorbed - case.source)), 5e-15)
        self.assertEqual(remaining, 0.0)
        result = field_statistics(absorbed, case.analytic, case)
        self.assertLess(result["L1_absorption_over_emitted"], 1e-14)

    def test_periodic_screen_still_has_correct_finite_range_geometry(self):
        case = make_cases()["opaque-plane"]
        absorbed, remaining, _, _ = solve_event(case, 8192)
        expected = case.metadata["analytic_absorbed_fraction"]
        self.assertLess(abs(absorbed.sum() - expected), 0.003)
        self.assertLess(abs(remaining + absorbed.sum() - 1), 1e-12)
        self.assertLessEqual(absorbed.sum(), 0.5)

    def test_fixture_dose_changes_only_photons_and_is_identified(self):
        source = np.full((4,) * 3, 0.7)
        capacity = np.ones_like(source)
        with tempfile.TemporaryDirectory() as tmp:
            filename = Path(tmp) / "fixture.npz"
            np.savez(filename, source=source, capacity=capacity)
            cases = fixture_cases(filename, [0.2, 0.6])
        self.assertEqual(cases["native-frozen"].source.sum(), source.sum())
        for dose in (0.2, 0.6):
            case = cases[f"native-dose-{dose:g}"]
            self.assertAlmostEqual(case.source.sum() / case.capacity.sum(), dose)
            np.testing.assert_array_equal(case.capacity, capacity)
            self.assertIn("not a cosmological redshift", case.metadata["dose_warning"])

    def test_fixture_rejects_invalid_capacity(self):
        source = np.ones((4,) * 3)
        with tempfile.TemporaryDirectory() as tmp:
            filename = Path(tmp) / "fixture.npz"
            np.savez(filename, source=source, capacity=-source)
            with self.assertRaises(ValueError):
                fixture_cases(filename)


if __name__ == "__main__":
    unittest.main()
