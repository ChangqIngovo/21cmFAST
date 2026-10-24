"""Independent oracles and adversarial gates for the PC refinement experiment."""
from pathlib import Path
import sys
import unittest

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parent/'pc_reconsidered'))
from audit_accuracy import angular_metrics, analytic_cap_audit, directions, missed_cone, strict_ledger, weighting_counterexample
from compare_models import Case, independent_ledger


class AuditAcceptanceTests(unittest.TestCase):
    def test_plane_cap_analytic_limit_at_high_angle_count(self):
        result=analytic_cap_audit(directions(65536))
        self.assertLess(result['maximum_absolute_fraction_error'],4e-4)
        self.assertAlmostEqual(result['expected'],17/48)

    def test_low_moment_isotropy_does_not_guarantee_cap_integration(self):
        # Six axis rays have exact first/second moments but poor finite caps.
        quadrature=np.vstack((np.eye(3),-np.eye(3)))
        self.assertEqual(angular_metrics(quadrature)['dipole_norm'],0.)
        self.assertLess(angular_metrics(quadrature)['quadrupole_frobenius'],1e-15)
        self.assertGreater(analytic_cap_audit(quadrature)['maximum_absolute_fraction_error'],.1)

    def test_small_probe_set_can_miss_non_negligible_solid_angle(self):
        cone=missed_cone(directions(24),candidates=4096)
        self.assertEqual(cone['probe_hits'],0)
        self.assertGreater(cone['fraction'],.03)

    def test_local_capacity_gate_does_not_loosen_with_distant_luminosity(self):
        cap=np.ones((4,)*3)
        cap[0,0,0]=1e-8
        source=np.zeros_like(cap)
        source[-1,-1,-1]=100.
        a=.5*cap
        a[0,0,0]=2e-8
        remainder=source.sum()-a.sum()
        case=Case('local-gate',source,cap,np.zeros_like(cap),1.)
        # The former global-Q allowance admitted this local 100% excess.
        self.assertLess(1e-8,2e-10*source.sum())
        self.assertFalse(independent_ledger(a,remainder,case)['passed'])
        checked=strict_ledger(a,remainder,source,cap)
        self.assertFalse(checked['passed'])
        self.assertEqual(checked['maximum_local_fraction_violation'],1.)

    def test_strict_gate_is_invariant_under_count_units(self):
        cap=np.ones((3,)*3)
        source=.3*cap
        for scale in (1e-12,1.,1e12):
            self.assertTrue(strict_ledger(.3*scale*cap,0.,scale*source,scale*cap)['passed'])

    def test_photon_weighted_l1_is_not_a_volume_error_bound(self):
        out=weighting_counterexample()
        self.assertTrue(out['capacity_bounds_preserved'])
        self.assertLess(abs(out['global_count_difference']),1e-15)
        self.assertLess(out['photon_count_L1_over_Q'],2e-7)
        self.assertGreater(out['volume_mean_absolute_fraction_error'],.01)
        self.assertAlmostEqual(out['maximum_fraction_error'],.7)


if __name__=='__main__':
    unittest.main()
