"""Independent rate units, positivity, temporal integration and failure checks."""
import unittest

import numpy as np

from validation_native_rates import positive_increment, time_interval, uv_rates


class NativeRateAdapterChecks(unittest.TestCase):
    def setUp(self):
        self.astro=dict(ALPHA_ESC=0.,F_ESC10_ACG=-1.,F_ESC7_MCG=-2.,POP2_ION=100.,POP3_ION=1000.)

    def test_mass_time_units_and_species_yields(self):
        rates=uv_rates(np.array([2.]),np.array([3.]),astro=self.astro,rho_b=10.)
        # Native SFRD already has units mass/second/volume and contains rho.
        # 2*.1*100/10=2 and 3*.01*1000/10=3, in inverse seconds.
        np.testing.assert_allclose(rates["acg"],[2.],rtol=0,atol=1e-14)
        np.testing.assert_allclose(rates["mcg"],[3.],rtol=0,atol=1e-14)
        np.testing.assert_allclose(positive_increment(rates["acg"]+rates["mcg"],rates["acg"]+rates["mcg"],7.),[35.],rtol=0,atol=1e-13)

    def test_positive_time_integral_under_declining_rate(self):
        result=positive_increment(np.array([6.,0.]),np.array([2.,0.]),3.)
        np.testing.assert_array_equal(result,[12.,0.])
        # This is emission, not a finite difference of endpoint inventories.
        self.assertGreater(result[0],0)

    def test_linear_rate_partition_and_persistent_ledger(self):
        left=np.array([1.,3.]);right=np.array([5.,1.]);mid=(left+right)/2
        whole=positive_increment(left,right,8.)
        parts=positive_increment(left,mid,4.)+positive_increment(mid,right,4.)
        np.testing.assert_array_equal(parts,whole)
        previous=np.array([100.,200.])
        np.testing.assert_array_equal(previous+parts,[124.,216.])

    def test_negative_rate_not_silently_clipped(self):
        with self.assertRaises(ValueError):
            positive_increment(np.array([-1.]),np.array([2.]),1.)
        with self.assertRaises(ValueError):
            uv_rates(np.array([-1.]),np.array([2.]),astro=self.astro,rho_b=10.)

    def test_mass_dependent_escape_rejected(self):
        with self.assertRaises(ValueError):
            uv_rates([1.],[1.],astro={**self.astro,"ALPHA_ESC":.2},rho_b=10.)

    def test_nonfinite_and_invalid_duration_rejected(self):
        for duration in (-1.,np.nan,np.inf):
            with self.assertRaises(ValueError):
                positive_increment([1.],[2.],duration)
        with self.assertRaises(ValueError):
            uv_rates([1.],[1.],astro={**self.astro,"POP2_ION":np.nan},rho_b=10.)

    def test_cosmic_clock_monotone_and_additive(self):
        c=dict(hlittle=.6766,OMm=.30964144154550644,OMr=8.6e-5,OMk=0.,wl=-1.)
        whole=time_interval(8.,7.5,c)
        parts=time_interval(8.,7.75,c)["integrated_seconds"]+time_interval(7.75,7.5,c)["integrated_seconds"]
        self.assertAlmostEqual(parts/whole["integrated_seconds"],1.,places=12)
        self.assertGreater(whole["native_endpoint_approximation_seconds"],whole["integrated_seconds"])
        with self.assertRaises(ValueError):
            time_interval(7.5,8.,c)

    def test_dtdz_clock_matches_analytic_age_and_keeps_sfr_clock_distinct(self):
        c=dict(hlittle=.6766,OMm=.30964144154550644,OMr=8.6e-5,OMk=0.,wl=-1.)
        om=c['OMm'];ol=1-om;h0=c['hlittle']*3.2407e-18
        def age(z):
            return 2/(3*h0*np.sqrt(ol))*np.arcsinh(np.sqrt(ol/om)/(1+z)**1.5)
        actual=time_interval(8.,7.5,c)
        self.assertAlmostEqual(actual['integrated_seconds']/(age(7.5)-age(8.)),1.,places=12)
        self.assertGreater(actual['integrated_seconds'],actual['native_hubble_integrated_seconds'])

    def test_escape_fraction_saturation_and_zero_yield(self):
        rates=uv_rates([2.],[3.],astro={**self.astro,'F_ESC10_ACG':1.,'POP3_ION':0.},rho_b=10.)
        np.testing.assert_allclose(rates['acg'],[20.],rtol=0,atol=1e-14)
        np.testing.assert_array_equal(rates['mcg'],[0.])


if __name__=="__main__":
    unittest.main()
