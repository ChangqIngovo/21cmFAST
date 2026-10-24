"""Projection changes measurement geometry, never the solved photon budget."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_coarsening import compare_projection
from pc_statistics import coarsen_counts


def fields(n=24):
    x = np.arange(n)[:, None, None]
    y = np.arange(n)[None, :, None]
    c = (1+.1*np.cos(2*np.pi*x/n))*np.ones((n, n, n))
    # Include high modes that can alias after block averaging, as well as low k.
    neutral = (.4+.03*np.cos(2*np.pi*x/n)+.01*np.sin(2*np.pi*y/n)
               +.03*np.cos(2*np.pi*(n//2-1)*x/n))*np.ones((n, n, n))
    return c, c-neutral


@pytest.mark.parametrize('nc', [6, 8, 12])
def test_exact_projection_is_unity_on_same_mesh_even_if_native_spectrum_changes(nc):
    c, a = fields()
    cc, ac = coarsen_counts(c, nc), coarsen_counts(a, nc)
    out = compare_projection(c, a, cc, ac, 100., [.05, .09])
    assert out['input_coarsening_and_solver_factor'] == pytest.approx(1., abs=2e-14)
    assert out['same_mesh_HI_cross_correlation'] == pytest.approx(1., abs=2e-14)
    assert out['same_mesh_absorbed_L1_over_atoms'] == 0
    assert out['Q_mass_difference'] == pytest.approx(0., abs=2e-16)
    assert out['factorization_residual'] == pytest.approx(0., abs=2e-14)


def test_equal_auto_power_does_not_hide_phase_error():
    c, a = fields()
    cc, ac = coarsen_counts(c, 12), coarsen_counts(a, 12)
    neutral = cc-ac
    reversed_neutral = 2*neutral.mean()-neutral
    out = compare_projection(c, a, cc, cc-reversed_neutral, 100., [.05, .09])
    assert out['input_coarsening_and_solver_factor'] == pytest.approx(1., abs=2e-14)
    assert out['same_mesh_HI_cross_correlation'] == pytest.approx(-1., abs=2e-14)
    assert out['same_mesh_absorbed_L1_over_atoms'] > .01


def test_different_coarse_atom_field_rejected():
    c, a = fields()
    with pytest.raises(ValueError, match='conservative sum'):
        compare_projection(c, a, 1.01*coarsen_counts(c, 12), coarsen_counts(a, 12), 100., [.05, .09])


def test_band_above_coarse_nyquist_rejected():
    c, a = fields()
    with pytest.raises(ValueError, match='Nyquist'):
        compare_projection(c, a, coarsen_counts(c, 6), coarsen_counts(a, 6), 100., [.2, .3])


def test_uniform_lift_preserves_each_parent_source_and_atom_budget():
    from coarsening_factorial import lift_counts
    rng = np.random.default_rng(82)
    parent = rng.uniform(.01, 10., (4, 4, 4))
    for factor in (2, 3):
        lifted = lift_counts(parent, factor)
        np.testing.assert_allclose(coarsen_counts(lifted, 4), parent, rtol=2e-15)
        assert lifted.sum() == pytest.approx(parent.sum(), rel=2e-15)
