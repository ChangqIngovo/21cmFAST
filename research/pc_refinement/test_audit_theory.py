"""Independent tests of weak-source angular error bounds."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_theory import audit_triplet, explicit_solve, random_tests, tail_mask


class WeakSourceBoundTests(unittest.TestCase):
    def test_comparison_principle_with_geometry_and_activation_changes(self):
        rows=random_tests(9)
        self.assertTrue(all(row['passed'] for row in rows))

    def test_factor_two_bound_is_sharp(self):
        cap=np.ones((12,)*3)
        old=np.zeros_like(cap)
        old[4:9,4:9,4:9]=1
        source=np.zeros_like(cap)
        source[6,6,6]=.01
        weak=source>0
        row=audit_triplet(source,cap,old,weak,strong_angles=1,weak_angles=1,
                          periodic=True,horizon=6)
        self.assertTrue(row['passed'])
        self.assertAlmostEqual(row['L1'],2*source.sum(),places=14)

    def test_unchanged_quadrature_reproduces_original_map(self):
        cap=np.ones((10,)*3)
        source=np.zeros_like(cap)
        source[4,5,5]=10.
        source[6,5,5]=15.
        mask=source==10.
        a=explicit_solve(source,cap,np.zeros_like(cap),mask,
                         strong_angles=96,weak_angles=96)
        b=explicit_solve(source,cap,np.zeros_like(cap),np.zeros_like(mask),
                         strong_angles=96,weak_angles=96)
        np.testing.assert_allclose(a['absorbed'],b['absorbed'],rtol=0,atol=1e-14)

    def test_tail_never_exceeds_requested_photon_budget(self):
        source=np.random.default_rng(294).lognormal(size=(5,)*3)
        for fraction in (.001,.1,.5):
            mask=tail_mask(source,fraction)
            self.assertLessEqual(source[mask].sum(),source.sum()*fraction)


if __name__=='__main__':
    unittest.main()
