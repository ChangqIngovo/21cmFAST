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


def endpoint_fixture():
    """Supply explicit substeps and nonuniform gas for native endpoint restriction."""
    history = base_history()
    history["times_seconds"] *= 10.
    refinement = 2
    rho = history["capacity"]
    fine_rho = rho
    for axis in range(3):
        fine_rho = np.repeat(fine_rho, refinement, axis=axis)
    rng = np.random.default_rng(32)
    shape = (6, *fine_rho.shape)
    fields = {"density_total_cm3": np.broadcast_to(fine_rho, shape).copy(),
              "xHII": rng.uniform(.1, .9, shape), "temperature_K": rng.uniform(100., 1000., shape),
              "xHeII": rng.uniform(.01, .1, shape), "xHeIII": rng.uniform(.001, .01, shape)}
    manifest = {"steps": [{"time_end_seconds": x} for x in (5., 10., 20., 30., 45., 60.)],
                "helium_number_fraction": .08, "mean_total_nuclei_density_z0_cm3": .02,
                "initial_state": {"xHII": .002, "xHeII": .004, "xHeIII": .0003,
                                  "temperature_kelvin": 123.}}
    return history, {"fields": fields}, manifest, refinement


def test_initial_snapshot_matches_declared_native_state_and_source_clock():
    """Initial gas labels, zero cumulative photons, endpoint emissivity and spectra align."""
    from research.es_pc_emulator.physical_pilot import MYR, collect_history_fields
    from research.pc_field_emulator.evaluation import shell_spectra

    history, state, manifest, refinement = endpoint_fixture()
    arrays, summary = collect_history_fields(history, state, manifest, refinement, include_initial=True)
    initial = manifest["initial_state"]
    np.testing.assert_array_equal(arrays["redshifts"], history["redshifts"])
    np.testing.assert_array_equal(arrays["times_seconds"], history["times_seconds"])
    np.testing.assert_array_equal(arrays["photons"][0], 0.)
    np.testing.assert_array_equal(arrays["emission_time_mean"][0], 0.)
    np.testing.assert_array_equal(arrays["emissivity"][0], history["rates"][0]*MYR)
    np.testing.assert_array_equal(arrays["xhi_es"][0], 1.)
    np.testing.assert_array_equal(arrays["radius_es_cMpc"][0], 0.)
    np.testing.assert_allclose(arrays["xhi_rt"][0], 1-initial["xHII"], atol=1e-15)
    np.testing.assert_allclose(arrays["temperature_rt"][0], initial["temperature_kelvin"], rtol=1e-15)
    np.testing.assert_allclose(arrays["xheii_rt"][0], initial["xHeII"], rtol=1e-15)
    np.testing.assert_allclose(arrays["xheiii_rt"][0], initial["xHeIII"], rtol=1e-15)
    field = 27*np.sqrt((1+history["redshifts"][0])/10)*history["capacity"]*(1-initial["xHII"])
    expected = shell_spectra({"teacher": field}, 50.)
    np.testing.assert_allclose(summary["P21"][0], expected["P_teacher"], rtol=1e-14)
    assert len(summary["P21_fine"]) == len(summary["xHI_mean"]) == len(history["redshifts"])


def test_initial_inclusion_leaves_existing_endpoint_fields_and_spectra_unchanged():
    """Prepending snap0 cannot shift substep matching or alter legacy labels."""
    from research.c2ray_teacher.compare import restrict_fields
    from research.es_pc_emulator.physical_pilot import collect_history_fields

    history, state, manifest, refinement = endpoint_fixture()
    legacy, old_summary = collect_history_fields(history, state, manifest, refinement)
    full, summary = collect_history_fields(history, state, manifest, refinement, include_initial=True)
    for name in legacy:
        np.testing.assert_array_equal(full[name] if name == "density" else full[name][1:], legacy[name])
    for name in ("P21", "P21_fine", "redshifts", "xHI_mean"):
        np.testing.assert_array_equal(summary[name][1:], old_summary[name])
    for endpoint, step in enumerate((1, 3, 5)):
        fine = {key: value[step] for key, value in state["fields"].items()}
        restricted = restrict_fields(fine, 4)
        np.testing.assert_array_equal(legacy["xhi_rt"][endpoint], 1-restricted["xHII"])
    broken = copy.deepcopy(manifest)
    broken["steps"][1]["time_end_seconds"] = 9.
    broken["steps"][0]["time_end_seconds"] = 10.
    with pytest.raises(ValueError, match="exact source endpoint"):
        collect_history_fields(history, state, broken, refinement)


def test_prepare_freezes_timeline_and_rejects_cached_redshift_mismatch(monkeypatch, tmp_path):
    """Cached templates and declared designs must match the requested full timeline."""
    import json

    from research.es_pc_emulator import physical_pilot as pilot
    from research.pc_evolution.native_history import _write_fresh_history, load_history

    calls = []

    def build(**kwargs):
        calls.append(kwargs)
        h = base_history()
        n = kwargs["intervals"]+1
        h["redshifts"] = np.linspace(kwargs["z_start"], kwargs["z_end"], n)
        h["times_seconds"] = np.arange(n, dtype=float)*10
        h["rates"] = np.ones((n, 4, 4, 4))
        h["native_xhi"] = np.ones((n, 4, 4, 4), dtype=np.float32)
        h["metadata"]["array_sha256"] = {key: array_hash(h[key]) for key in ARRAY_KEYS}
        _write_fresh_history(kwargs["output"], h)

    monkeypatch.setattr(pilot, "build_native_history", build)
    monkeypatch.setattr(pilot.subprocess, "check_output", lambda *args, **kwargs: "test\n")
    root = tmp_path/"run"
    pilot.prepare(root, tmp_path/"upstream", z_end=4., intervals=40, include_initial=True)
    assert len(calls) == 3
    assert all(call["z_start"] == 20. and call["z_end"] == 4. and call["intervals"] == 40 for call in calls)
    settings = json.loads((root/"design.json").read_text())["metadata"]["history_settings"]
    assert settings == {"z_start": 20., "z_end": 4., "intervals": 40, "include_initial": True}
    pilot.prepare(root, tmp_path/"upstream", z_end=4., intervals=40, include_initial=True)
    assert len(calls) == 3
    with pytest.raises(ValueError, match="Existing design differs"):
        pilot.prepare(root, tmp_path/"upstream")
    changed = load_history(root/"base_111.npz")
    changed["redshifts"][1] -= .01
    changed["metadata"]["array_sha256"] = {key: array_hash(changed[key]) for key in ARRAY_KEYS}
    (root/"base_111.npz").unlink()
    _write_fresh_history(root/"base_111.npz", changed)
    with pytest.raises(ValueError, match="Cached base history redshifts"):
        pilot.prepare(root, tmp_path/"upstream", z_end=4., intervals=40, include_initial=True)


def test_history_settings_preserve_legacy_defaults_and_reject_invalid_domain():
    """Opting out of full history retains the previous eight evolved snapshots."""
    from research.es_pc_emulator.physical_pilot import (
        history_settings,
        validate_base_timeline,
    )

    settings = history_settings()
    assert settings == {"z_start": 20., "z_end": 7.9, "intervals": 8, "include_initial": False}
    validate_base_timeline({"redshifts": np.linspace(20., 7.9, 9)}, settings)
    for kwargs in ({"z_end": 21.}, {"z_end": -1.}, {"intervals": 0},
                   {"intervals": True}, {"include_initial": 1}, {"z_start": float("nan")}):
        with pytest.raises(ValueError, match="finite descending"):
            history_settings(**kwargs)
