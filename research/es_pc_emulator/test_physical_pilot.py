"""Physical invariants of the diagnostic perturbations and ES prior."""
import copy

import numpy as np
import pytest

from research.es_pc_emulator.physical_pilot import (
    BOUNDS,
    FIDUCIAL,
    budget_es,
    integrated_history,
    transform_history,
)
from research.pc_evolution.native_history import (
    ARRAY_KEYS,
    array_hash,
    validate_history,
)


def base_history():
    """Provide positive source histories without a native runtime dependency."""
    rng = np.random.default_rng(14)
    rates = rng.uniform(.1, 1, (4, 4, 4, 4))
    history = {"rates": rates, "capacity": rng.uniform(.6, 1.4, (4, 4, 4)),
               "redshifts": np.array([20., 15., 10., 8.]),
               "times_seconds": np.array([0., 1., 3., 6.]),
               "native_xhi": np.ones(rates.shape, dtype=np.float32),
               "metadata": {"schema_version": 1, "density_evolution": "frozen_at_reference_redshift",
                            "reference_redshift": 8., "rates_definition": "native_n_ion_times_H_over_t_STAR"}}
    history["metadata"]["array_sha256"] = {k: array_hash(history[k]) for k in ARRAY_KEYS}
    return history


def test_exact_piecewise_linear_time_moment():
    """The temporal summary reproduces an analytic linear source."""
    times = np.array([0., .3, 1., 2.])
    rates = times[:, None, None, None] * np.ones((1, 2, 2, 2))
    photons, mean_time = integrated_history(rates, times)
    np.testing.assert_allclose(photons[:, 0, 0, 0], times**2 / 2)
    np.testing.assert_allclose(mean_time[:, 0, 0, 0], times * 2 / 3)


def test_time_tilt_preserves_each_sources_final_photons_but_changes_age():
    """Timing and total photon count are varied independently."""
    base = base_history()
    early, late = FIDUCIAL.copy(), FIDUCIAL.copy()
    early[1], late[1] = -1, 1
    a, b = transform_history(base, early), transform_history(base, late)
    pa, ta = integrated_history(a["rates"], a["times_seconds"])
    pb, tb = integrated_history(b["rates"], b["times_seconds"])
    np.testing.assert_allclose(pa[-1], pb[-1], rtol=1e-14)
    assert np.all(ta[-1] < tb[-1])
    assert not np.allclose(pa[1], pb[1])


def test_source_contrast_preserves_endpoint_totals_when_tilt_is_zero():
    """Concentrating sources retains the instantaneous global emission."""
    base = base_history()
    a, b = FIDUCIAL.copy(), FIDUCIAL.copy()
    a[2], b[2] = .7, 1.3
    ha, hb = transform_history(base, a), transform_history(base, b)
    np.testing.assert_allclose(ha["rates"].sum(axis=(1, 2, 3)), hb["rates"].sum(axis=(1, 2, 3)), rtol=1e-14)
    assert hb["rates"][-1].std() > ha["rates"][-1].std()


def test_density_contrast_preserves_total_mass_and_does_not_change_sources():
    """Gas contrast is varied without changing mass or source luminosity."""
    base = base_history()
    a, b = FIDUCIAL.copy(), FIDUCIAL.copy()
    a[3], b[3] = .7, 1.3
    ha, hb = transform_history(base, a), transform_history(base, b)
    np.testing.assert_allclose(ha["capacity"].sum(), base["capacity"].sum(), rtol=1e-14)
    np.testing.assert_allclose(hb["capacity"].sum(), base["capacity"].sum(), rtol=1e-14)
    np.testing.assert_array_equal(ha["rates"], hb["rates"])
    assert hb["capacity"].std() > ha["capacity"].std()


def test_physical_transform_requires_provenance_and_rejects_outside_domain():
    """Controlled runs cannot masquerade as untouched native source histories."""
    base = base_history()
    transformed = transform_history(base, FIDUCIAL)
    bad = copy.deepcopy(transformed)
    bad["metadata"]["physical_pilot_transform"].pop("parent_array_sha256")
    with pytest.raises(ValueError, match="parent hashes"):
        validate_history(bad)
    theta = FIDUCIAL.copy()
    theta[0] = BOUNDS[0, 1] + 1
    with pytest.raises(ValueError, match="within declared bounds"):
        transform_history(base, theta)


def test_budget_es_uniform_partial_field_and_periodic_covariance():
    """The prior retains local partial ionization and periodic translation symmetry."""
    density = np.ones((8, 8, 8))
    photons = np.full_like(density, .25)
    neutral, radius = budget_es(photons, density, 50., 1.)
    np.testing.assert_allclose(neutral, .75)
    np.testing.assert_array_equal(radius, 0.)
    photons[0, 1, 2] = 80
    neutral, radius = budget_es(photons, density, 50., 1.)
    shift = (2, -1, 3)
    moved, moved_radius = budget_es(np.roll(photons, shift, (0, 1, 2)), density, 50., 1.)
    np.testing.assert_allclose(moved, np.roll(neutral, shift, (0, 1, 2)), atol=1e-14)
    np.testing.assert_array_equal(moved_radius, np.roll(radius, shift, (0, 1, 2)))
