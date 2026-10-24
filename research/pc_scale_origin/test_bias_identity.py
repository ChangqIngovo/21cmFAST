"""Check spectral interpretation against independently constructed mass fields."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_large import bias_terms


@pytest.mark.parametrize('q', [.2, .8, .9])
def test_ionized_cross_bias_from_HI_matches_direct_ionized_density_fft(q):
    n = 12
    x = np.arange(n)[:, None, None]*2*np.pi/n
    y = np.arange(n)[None, :, None]*2*np.pi/n
    rho = np.broadcast_to(1+.04*np.cos(x), (n, n, n)).copy()
    ions = np.broadcast_to(q*(1+.05*np.cos(x)+.015*np.sin(y)), (n, n, n)).copy()
    assert np.all((ions >= 0) & (ions <= rho))
    hi = rho-ions
    matter_fft = np.fft.fftn(rho/rho.mean()-1).ravel()[1:]
    ions_fft = np.fft.fftn(ions/ions.mean()-1).ravel()[1:]
    hi_fft = np.fft.fftn(hi/hi.mean()-1).ravel()[1:]
    pm = float(np.vdot(matter_fft, matter_fft).real)
    ph = float(np.vdot(hi_fft, hi_fft).real)
    cross = float(np.vdot(matter_fft, hi_fft).real)
    measured = float(np.vdot(matter_fft, ions_fft).real)/pm
    terms = bias_terms(dict(P_matter=pm, P_HI=ph, P_cross=cross), q)
    assert terms['b_ionized_cross'] == pytest.approx(measured, abs=1e-13)
    orthogonal = hi_fft-terms['b_HI_cross']*matter_fft
    assert terms['matter_orthogonal_b2'] == pytest.approx(
        float(np.vdot(orthogonal, orthogonal).real)/pm, abs=1e-12)
