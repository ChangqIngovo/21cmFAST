"""Expected-limit tests: do not mislabel passing assertions as finite-c validation."""
import unittest

import numpy as np

from validation_time_state import (
    Flash, advance_flash, counterexamples, cosmic_interval, delayed_absorption,
    relativistic_radius, native_dtdz_seconds, MYR_S, NATIVE_H0_PER_H,
)


class TimeStateAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = counterexamples()

    def test_finite_c_reference_delays_and_source_off(self):
        self.assertEqual(delayed_absorption(2, 1, 4.5, 2, 1), 0)
        self.assertEqual(delayed_absorption(3, 1, 4.5, 2, 1), .75)
        self.assertEqual(delayed_absorption(10, 1, 4.5, 2, 1), 1)
        self.assertEqual(self.rows['source_off_arrivals']['zero_source_restart_absorption_increment'], 0)

    def test_infinite_c_limitation_is_detected_despite_closed_ledger(self):
        early = self.rows['constant_source_finite_c'][0]
        self.assertEqual(early['event_absorbed'], 1)
        self.assertEqual(early['finite_c_exact_absorbed'], 0)
        self.assertAlmostEqual(early['audit']['ledger_residual'], 0)
        cutoff = self.rows['distance_cutoff_is_not_causal_transport']
        self.assertEqual(cutoff['event_absorbed'], 1)
        self.assertEqual(cutoff['finite_c_exact_absorbed'], .5)

    def test_flash_oracle_preserves_state_and_partition(self):
        whole = advance_flash(Flash(1.5, 1), 3, 2, 6)
        split = Flash(1.5, 1)
        for dt in (.2, .8, .1, .9, 1):
            split = advance_flash(split, dt, 2, 6)
        self.assertEqual(whole, split)
        self.assertEqual(whole.absorbed, 1)
        self.assertEqual(whole.travelled, 4.5)
        self.assertEqual(whole.photons + whole.absorbed, 1)

    def test_restart_original_source_has_no_progress(self):
        rows = self.rows['source_restart_stalls']
        for step in rows['restart_steps']:
            self.assertEqual(step['absorbed'], 0)
            self.assertEqual(step['retained'], 1)
            self.assertEqual(step['ledger_residual'], 0)
        self.assertEqual(rows['finite_c_flash_reference'][-1]['absorbed'], 1)
        self.assertEqual(rows['finite_c_same_t0_to_t1_emission_absorbed_at_t3'], .75)
        self.assertEqual(rows['naive_sum_solver_photons'], 3)

    def test_source_only_reservoir_cannot_preserve_direction(self):
        row = self.rows['re_isotropizing_carry_changes_direction']
        self.assertEqual(row['first_retained_x_only'], .5)
        self.assertEqual(row['restart_spurious_extra_y_absorption'], .25)
        self.assertEqual(row['wrong_fraction_of_carried_count'], .5)
        self.assertEqual(row['restart_audit']['ledger_residual'], 0)

    def test_absolute_range_budget_must_survive_restart(self):
        row = self.rows['range_budget_must_persist']
        self.assertEqual(row['correct']['travelled'], 3)
        self.assertEqual(row['correct']['absorbed'], 0)
        self.assertEqual(row['correct']['photons'], 1)
        self.assertEqual(row['wrong_reset_distance_every_step']['absorbed'], 1)

    def test_valid_infinite_c_partition_and_cumulative_counterexample(self):
        self.assertLess(self.rows['infinite_c_no_retained_restart_is_valid']['split_vs_whole_l1'], 1e-13)
        row = self.rows['cumulative_source_is_not_increment']
        self.assertEqual(row['wrong_absorbed_total'], 8)
        self.assertEqual(row['correct_increment_absorbed_total'], 5)

    def test_relativistic_uniform_front_oracle(self):
        for time in (.001, .1, 1, 100):
            radius = relativistic_radius(time)
            self.assertLessEqual(radius, time)
            self.assertLessEqual(radius, time**(1/3))
            self.assertAlmostEqual(radius + radius**3, time, delta=1e-12 * max(1, time))
        self.assertAlmostEqual(relativistic_radius(1), .6823278038280193, places=14)
        r = relativistic_radius(1, speed=1e9)
        self.assertAlmostEqual(r, 1, places=8)

    def test_cosmic_clock_matches_native_flat_lcdm_analytic_age(self):
        cosmo = {'OMm': .30964144154550644, 'hlittle': .6766}
        om = cosmo['OMm']
        h0 = cosmo['hlittle'] * NATIVE_H0_PER_H
        def age(z):
            return 2 * np.arcsinh(np.sqrt((1-om)/om) / (1+z)**1.5) / (3*h0*np.sqrt(1-om)) / MYR_S
        for hi, lo in ((35, 12), (8, 7.5), (10.316657340130499, 10.094762098167156)):
            dt, distance = cosmic_interval(hi, lo, cosmo)
            self.assertAlmostEqual(dt, age(lo)-age(hi), delta=1e-10)
            self.assertGreater(distance, 0)

    def test_two_native_clocks_and_exact_dtdz_expression(self):
        cosmo = {'OMm': .30964144154550644, 'OMl': .6902725584544936,
                 'OMr': 8.6e-5, 'hlittle': .6766}
        z = 7.5
        om, oml = cosmo['OMm'], cosmo['OMl']
        h0 = cosmo['hlittle'] * NATIVE_H0_PER_H
        x = np.sqrt(oml / om) * (1 + z)**(-1.5)
        dx = -1.5 * np.sqrt(oml / om) * (1 + z)**(-2.5)
        raw = 2*np.sqrt(1+om/oml)/(3*h0) * dx * (1 + x/np.sqrt(x*x+1)) / (x+np.sqrt(x*x+1))
        self.assertAlmostEqual(native_dtdz_seconds(z, cosmo) / raw, 1, places=14)
        dt_h, _ = cosmic_interval(8, 7.5, cosmo, clock='hubble')
        dt_d, _ = cosmic_interval(8, 7.5, cosmo, clock='dtdz')
        self.assertGreater(dt_d, dt_h)
        self.assertLess((dt_d-dt_h)/dt_h, .002)


if __name__ == '__main__':
    unittest.main()
