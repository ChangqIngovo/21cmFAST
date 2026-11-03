"""Independent reconstructed-field budget checks, including adversarial outputs."""
import json
import unittest

import numpy as np

from research.pc_coeval.photon_audit import audit_snapshot


class PhotonAuditTests(unittest.TestCase):
    def setUp(self):
        self.capacity = np.arange(1, 9, dtype=float).reshape((2, 2, 2))
        self.source = self.capacity * .25
        self.xhi = np.full((2, 2, 2), .75)

    def audit(self, xhi=None, **kwargs):
        xhi = self.xhi if xhi is None else xhi
        return audit_snapshot(self.source, self.capacity,
                              {name: xhi.copy() for name in ("pc", "script", "native_es")}, **kwargs)

    def test_exact_budget_reconstructed_without_reservoir_claim(self):
        result = self.audit()
        self.assertTrue(result["pc_and_script_allocation_passed"])
        self.assertFalse(result["independent_reservoir_audit_available"])
        pc = result["methods"]["pc"]
        self.assertEqual(pc["ionized_atoms_recomputed"], 9)
        self.assertEqual(pc["relative_photon_defect_with_inferred_surplus"], 0)
        self.assertIsNone(pc["stored_reservoir_ledger"]["passed"])

    def test_corrupted_field_detected_without_solver_metadata(self):
        xhi = self.xhi.copy()
        xhi[1, 1, 1] += .01
        pc = self.audit(xhi)["methods"]["pc"]
        self.assertFalse(pc["allocation_budget_passed"])
        self.assertGreater(pc["signed_photon_defect_with_inferred_surplus"], 0)

    def test_correct_volume_average_can_hide_wrong_mass_budget(self):
        xhi = self.xhi.copy()
        xhi[0, 0, 0] -= .2
        xhi[1, 1, 1] += .2
        pc = self.audit(xhi)["methods"]["pc"]
        self.assertAlmostEqual(pc["xHI_volume_recomputed"], .75)
        self.assertFalse(pc["allocation_budget_passed"])
        self.assertAlmostEqual(pc["signed_photon_defect_with_inferred_surplus"], 1.4)

    def test_full_saturation_inferred_surplus_not_stored_ledger(self):
        self.source = self.capacity * 2
        pc = self.audit(np.zeros_like(self.xhi))["methods"]["pc"]
        self.assertTrue(pc["allocation_budget_passed"])
        self.assertEqual(pc["input_inferred_surplus_photons"], 36)
        self.assertFalse(pc["surplus_is_independently_observed"])
        self.assertFalse(pc["stored_reservoir_ledger"]["available"])

    def test_independently_stored_reservoir(self):
        self.source = self.capacity * 2
        result = self.audit(np.zeros_like(self.xhi), remaining_source=self.capacity)
        self.assertTrue(result["independent_reservoir_audit_available"])
        self.assertTrue(result["all_available_required_checks_passed"])
        self.assertTrue(result["methods"]["pc"]["stored_reservoir_ledger"]["passed"])

    def test_wrong_reservoir_sum_detected(self):
        self.source = self.capacity * 2
        result = self.audit(np.zeros_like(self.xhi), remaining_source=self.capacity * .99)
        self.assertTrue(result["pc_and_script_allocation_passed"])
        self.assertFalse(result["all_available_required_checks_passed"])

    def test_correct_reservoir_sum_but_source_overdraft_detected(self):
        self.source = self.capacity * 2
        remaining = self.capacity.copy()
        remaining[0, 0, 0] += 3
        remaining[1, 1, 1] -= 3
        ledger = self.audit(np.zeros_like(self.xhi), remaining_source=remaining)["methods"]["pc"]["stored_reservoir_ledger"]
        self.assertEqual(ledger["relative_photon_defect"], 0)
        self.assertFalse(ledger["per_source_bounds_passed"])
        self.assertFalse(ledger["passed"])

    def test_unfinished_conservative_allocation_still_fails_target(self):
        # A valid partial ledger does not certify a completed allocation.
        result = self.audit(np.full_like(self.xhi, .875), remaining_source=self.source * .5)
        self.assertTrue(result["methods"]["pc"]["stored_reservoir_ledger"]["passed"])
        self.assertFalse(result["pc_and_script_allocation_passed"])

    def test_bounds_detected_even_when_weighted_budget_unchanged(self):
        xhi = self.xhi.copy()
        xhi[0, 0, 0] = 1.75
        xhi[1, 1, 1] -= 1 / 8
        pc = self.audit(xhi)["methods"]["pc"]
        self.assertEqual(pc["relative_photon_defect_with_inferred_surplus"], 0)
        self.assertFalse(pc["allocation_budget_passed"])

    def test_nan_field_fails_and_report_is_json_safe(self):
        xhi = self.xhi.copy()
        xhi[0, 0, 0] = np.nan
        result = self.audit(xhi)
        self.assertFalse(result["pc_and_script_allocation_passed"])
        json.dumps(result, allow_nan=False)

    def test_zero_sources_exact_neutral_passes(self):
        self.source[:] = 0
        pc = self.audit(np.ones_like(self.xhi))["methods"]["pc"]
        self.assertTrue(pc["allocation_budget_passed"])
        self.assertEqual(pc["relative_photon_defect_with_inferred_surplus"], 0)

    def test_zero_sources_with_unexplained_ions_fails(self):
        self.source[:] = 0
        pc = self.audit()["methods"]["pc"]
        self.assertFalse(pc["allocation_budget_passed"])
        self.assertIsNone(pc["relative_photon_defect_with_inferred_surplus"])
        self.assertAlmostEqual(pc["absolute_defect_per_atom"], .25)

    def test_tiny_sources_lost_in_xhi_storage_have_visible_roundoff_floor(self):
        self.source = self.capacity * 1e-18
        pc = self.audit(np.ones_like(self.xhi))["methods"]["pc"]
        self.assertTrue(pc["allocation_budget_passed"])
        self.assertEqual(pc["relative_photon_defect_with_inferred_surplus"], 1)
        self.assertLess(pc["absolute_defect_per_atom"], 2e-18)

    def test_huge_surplus_does_not_hide_missing_ions(self):
        self.source = self.capacity * 1e100
        pc = self.audit()["methods"]["pc"]
        self.assertFalse(pc["allocation_budget_passed"])
        self.assertAlmostEqual(pc["relative_allocation_defect"], .75)

    def test_mass_budget_invariant_under_common_count_units(self):
        baseline = self.audit()["methods"]["pc"]
        self.source *= 1e-100
        self.capacity *= 1e-100
        pc = self.audit()["methods"]["pc"]
        self.assertTrue(pc["allocation_budget_passed"])
        self.assertAlmostEqual(pc["xHI_mass_recomputed"], baseline["xHI_mass_recomputed"])

    def test_es_diagnostic_does_not_fail_pc_and_script(self):
        fields = {"pc": self.xhi, "script": self.xhi, "native_es": self.xhi * .5}
        result = audit_snapshot(self.source, self.capacity, fields)
        self.assertFalse(result["methods"]["native_es"]["allocation_budget_passed"])
        self.assertTrue(result["pc_and_script_allocation_passed"])

    def test_precisions_have_separate_tolerances(self):
        pc = self.audit(self.xhi - 1e-6)["methods"]
        self.assertFalse(pc["pc"]["allocation_budget_passed"])
        self.assertTrue(pc["script"]["allocation_budget_passed"])

    def test_input_validation(self):
        for kwargs in ({"pc_tolerance": 0}, {"script_tolerance": np.nan}, {"pc_tolerance": True}):
            with self.assertRaises(ValueError):
                self.audit(**kwargs)
        self.source[0, 0, 0] = -1
        with self.assertRaises(ValueError):
            self.audit()

    def test_shape_and_missing_model_validation(self):
        with self.assertRaises(ValueError):
            self.audit(np.ones((1, 1, 1)))
        with self.assertRaises(ValueError):
            self.audit(remaining_source=np.ones((1, 1, 1)))
        with self.assertRaises(ValueError):
            audit_snapshot(self.source, self.capacity, {"pc": self.xhi})


if __name__ == "__main__":
    unittest.main()
