"""Native adapter contracts; physical restrictions must be executable guards."""
from types import SimpleNamespace

import numpy as np
import pytest

from native_driver import (
    _native_box, _spatial_solver, from_native_coeval, update_reionization_redshift,
    validate_inputs,
)


def fake_coeval():
    inputs = SimpleNamespace(
        astro_options=SimpleNamespace(USE_MCGS=False, USE_TS_FLUCT=False,
            USE_REIONIZATION_PHOTOHEATING_FEEDBACK=False, RECOMB_MODEL="none",
            PHOTON_CONS_TYPE="no-photoncons"),
        matter_options=SimpleNamespace(SOURCE_MODEL="L-INTEGRAL", MINIMIZE_MEMORY=True),
        simulation_options=SimpleNamespace(NON_CUBIC_FACTOR=1),
    )
    source = np.arange(8, dtype=float).reshape((2,) * 3) / 10
    delta = source.copy()  # Correlated density exposes erroneous second weighting.
    return SimpleNamespace(inputs=inputs, redshift=8.,
        emissivity_fields=SimpleNamespace(get=lambda name: source),
        perturbed_field=SimpleNamespace(get=lambda name: delta),
        ionized_box=SimpleNamespace(nion_unconditional_acg=source.mean(), nion_unconditional_mcg=0.))


def test_lagrangian_emissivity_is_not_weighted_by_density_twice():
    coeval = fake_coeval()
    result = from_native_coeval(coeval)
    source = coeval.emissivity_fields.get("n_ion")
    np.testing.assert_array_equal(result.source, source)
    np.testing.assert_array_equal(result.capacity, 1 + source)
    assert result.source.sum() != pytest.approx(np.sum(source * (1 + source)))
    assert result.metadata["source_minus_native_reported_total"] == pytest.approx(0.)
    assert not result.source.flags.writeable
    assert not result.capacity.flags.writeable


@pytest.mark.parametrize("group,name,value", [
    ("astro_options", "USE_MCGS", True),
    ("astro_options", "USE_TS_FLUCT", True),
    ("astro_options", "RECOMB_MODEL", "inhomogeneous"),
    ("astro_options", "USE_REIONIZATION_PHOTOHEATING_FEEDBACK", True),
    ("astro_options", "PHOTON_CONS_TYPE", "global-budget"),
    ("matter_options", "SOURCE_MODEL", "E-INTEGRAL"),
    ("matter_options", "MINIMIZE_MEMORY", False),
    ("simulation_options", "NON_CUBIC_FACTOR", 2),
])
def test_unfinalized_physics_is_rejected(group, name, value):
    coeval = fake_coeval()
    setattr(getattr(coeval.inputs, group), name, value)
    with pytest.raises(ValueError):
        validate_inputs(coeval.inputs)


def test_native_source_deficit_is_not_clipped():
    coeval = fake_coeval()
    coeval.emissivity_fields.get("n_ion")[0, 0, 0] = -1e-12
    with pytest.raises(ValueError, match="does not clip"):
        from_native_coeval(coeval)


@pytest.mark.parametrize("capacity_value,absorption_value", [
    (1e-15, 2e-15), (1e-15, -1e-16), (0., 1e-30),
])
def test_native_box_rejects_local_void_overfill_instead_of_clipping(
        capacity_value, absorption_value):
    capacity = np.full((2,) * 3, capacity_value)
    absorbed = np.full_like(capacity, absorption_value)
    with pytest.raises(ValueError, match="native atom capacity"):
        _native_box(fake_coeval(), absorbed, capacity)


def test_reionization_labels_keep_first_crossing_without_claiming_transport():
    labels = update_reionization_redshift([0., .3, 0.], 10.)
    np.testing.assert_array_equal(labels, [10., -1., 10.])
    later = update_reionization_redshift([.2, 0., 0.], 9., labels)
    np.testing.assert_array_equal(later, [10., 9., 10.])
    with pytest.raises(ValueError, match="later"):
        update_reionization_redshift([0.], 10., [9.])


@pytest.mark.parametrize("bad", [-1., 1.1, np.nan])
def test_invalid_fraction_cannot_be_used_for_reionization_history(bad):
    with pytest.raises(ValueError):
        update_reionization_redshift([bad], 8.)


@pytest.mark.parametrize("solver", ["original", "balanced", "adaptive-front", "refined-front"])
def test_true_native_sources_candidate_and_brightness_end_to_end(tmp_path, solver):
    pytest.importorskip("py21cmfast")
    from native_driver import run_native
    result = run_native(output=tmp_path / "native", dimension=8, box_len=24.,
                        redshift=8., threads=2, max_angles=96,
                        rays_per_surface_cell=1., solver=solver)
    assert result["metadata"]["solver_profile"] == solver
    assert result["metadata"]["solver_parameters"]["surface_sampling_override"] == 1.
    assert "not Qin20 exact" in result["metadata"]["scope"]
    sources = result["source"]
    solution = result["candidate"]
    np.testing.assert_allclose(
        solution["absorbed"].sum() + solution["unabsorbed"],
        sources.source.sum(), rtol=1e-12, atol=1e-12,
    )
    assert np.all(solution["absorbed"] <= sources.capacity + 1e-12)
    assert (tmp_path / "native" / "native_fields.npz").is_file()
    assert (tmp_path / "native" / "native_comparison.json").is_file()
    # Native saturation limit must use each arm's actual xHI and the same gas,
    # not the brightness field inherited from the native ES control.
    factors = []
    for name, box in result["boxes"].items():
        xhi = box.get("neutral_fraction").astype(float)
        brightness = result["brightness"][name].get("brightness_temp").astype(float)
        neutral = xhi * sources.capacity
        mask = neutral > 1e-8
        factor = brightness[mask] / neutral[mask]
        np.testing.assert_allclose(factor, factor.mean(), rtol=3e-7)
        factors.append(factor.mean())
        labels = box.get("z_reion")
        assert np.all(labels[xhi <= 1e-6] == 8.)
        assert np.all(labels[xhi > 1e-6] == -1.)
    np.testing.assert_allclose(factors, factors[0], rtol=3e-7)


def test_unknown_solver_is_rejected_before_compilation():
    with pytest.raises(ValueError, match="Unknown spatial solver"):
        _spatial_solver("unregistered-method")


@pytest.mark.parametrize("profile", ["compact", "incremental"])
def test_selected_profile_preserves_counts_and_reports_identity(profile):
    solve, warmup = _spatial_solver(profile)
    warmup()
    source = np.full((4,)*3, .1)
    capacity = np.ones_like(source)
    result = solve(source, capacity)
    assert result["audit"]["solver_profile"] == profile
    np.testing.assert_array_equal(result["absorbed"], source)
    assert result["unabsorbed"] == 0.
