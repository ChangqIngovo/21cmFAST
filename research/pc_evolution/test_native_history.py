"""Source-clock, no-double-counting and fixture provenance contracts."""
import json

import numpy as np
import pytest

from native_history import (
    ARRAY_KEYS, array_hash, build_native_history, load_history, native_hubble,
    native_time_interval, rate_from_nion, validate_history, _write_fresh_history,
)


COSMO = dict(hlittle=.6751, OMm=.3121, OMl=.6879, OMr=9.e-5, OMk=0., wl=-1.)


def fixture():
    z = np.array([12., 10., 8.])
    rates = np.arange(24, dtype=np.float64).reshape(3, 2, 2, 2) * 1e-17
    result = dict(redshifts=z, times_seconds=np.array([0., 1e15, 3e15]),
                  rates=rates, capacity=np.ones((2, 2, 2)),
                  native_xhi=np.ones(rates.shape, dtype=np.float32),
                  metadata=dict(schema_version=1,
                      density_evolution="frozen_at_reference_redshift",
                      rates_definition="native_n_ion_times_H_over_t_STAR",
                      reference_redshift=8.))
    result["metadata"]["array_sha256"] = {key: array_hash(result[key]) for key in ARRAY_KEYS}
    return result


def test_source_rate_matches_independent_stellar_mass_sfr_count_identity():
    # Derive photons through stellar mass -> SFR -> escaped photons, avoiding
    # n_ion differences or a second multiplication by Eulerian density.
    stellar_mass = np.array([0., 3., 20.])
    rho_b = 7.
    escape, photon_yield, t_star, z = .06, 5000., .5, 9.
    nion = stellar_mass * escape * photon_yield / rho_b
    h = COSMO["hlittle"] * 3.2407e-18 * np.sqrt(
        COSMO["OMm"] * 10**3 + COSMO["OMr"] * 10**4 + COSMO["OMl"])
    sfr = stellar_mass / (t_star / h)
    independent_rate = sfr * escape * photon_yield / rho_b
    np.testing.assert_allclose(rate_from_nion(nion, z, t_star=t_star, cosmo=COSMO),
                               independent_rate, rtol=1e-14, atol=0)


def test_native_time_clock_matches_analytic_matter_lambda_age_difference():
    z_old, z_new = 14., 6.
    om, ol = COSMO["OMm"], COSMO["OMl"]
    h0 = COSMO["hlittle"] * 3.2407e-18
    prefactor = 2 * np.sqrt(om + ol) / (3 * h0 * np.sqrt(ol))
    age = lambda z: prefactor * np.arcsinh(np.sqrt(ol / om) / (1 + z)**1.5)
    duration, error = native_time_interval(z_old, z_new, COSMO)
    assert duration == pytest.approx(age(z_new) - age(z_old), rel=2e-13)
    assert error / duration < 1e-10
    # The SFR clock includes radiation; it must not silently replace dtdz.
    without_radiation = {**COSMO, "OMr": 0.}
    assert native_hubble(z_old, COSMO) > native_hubble(z_old, without_radiation)
    assert native_time_interval(z_old, z_new, without_radiation)[0] == duration


@pytest.mark.parametrize("source", [[-1e-30], [np.nan], [np.inf]])
def test_invalid_native_counts_are_rejected_without_clipping(source):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        rate_from_nion(source, 8., t_star=.5, cosmo=COSMO)


def test_fixture_roundtrip_and_existing_destination_is_preserved(tmp_path):
    history = fixture()
    path = tmp_path / "history.npz"
    _write_fresh_history(path, history)
    before = path.read_bytes()
    loaded = load_history(path)
    for key in ARRAY_KEYS:
        np.testing.assert_array_equal(loaded[key], history[key])
    assert loaded["metadata"] == history["metadata"]
    with np.load(path, allow_pickle=False) as saved:
        assert saved["metadata"].shape == ()
        assert json.loads(saved["metadata"].item())["schema_version"] == 1
    with pytest.raises(FileExistsError):
        _write_fresh_history(path, history)
    assert path.read_bytes() == before


def test_payload_mutation_is_detected_by_hash():
    history = fixture()
    history["rates"][1, 0, 0, 0] *= 2
    with pytest.raises(ValueError, match="hash mismatch: rates"):
        validate_history(history)


@pytest.mark.parametrize("change, message", [
    (lambda h: h["times_seconds"].__setitem__(0, 1.), "increasing times"),
    (lambda h: h["times_seconds"].__setitem__(1, 0.), "increasing times"),
    (lambda h: h["redshifts"].__setitem__(1, 14.), "descending redshifts"),
    (lambda h: h["rates"].__setitem__((0, 0, 0, 0), -1e-30), "nonnegative endpoint"),
    (lambda h: h["native_xhi"].__setitem__((0, 0, 0, 0), 1.1), "neutral fractions"),
    (lambda h: h["capacity"].__setitem__((0, 0, 0), 0.), "strictly positive"),
    (lambda h: h.__setitem__("capacity", h["capacity"].astype(np.float32)), "must be float64"),
    (lambda h: h["metadata"].__setitem__("density_evolution", "changing"), "frozen gas"),
], ids=["time_origin", "time_repeat", "redshift_order", "negative_rate", "xhi_bounds", "zero_capacity", "capacity_dtype", "changing_gas"])
def test_structural_or_physical_corruption_is_rejected(change, message):
    history = fixture()
    change(history)
    with pytest.raises(ValueError, match=message):
        validate_history(history, verify_hashes=False)


@pytest.mark.parametrize("highres_dimension", [None, 32], ids=["default_dim", "explicit_dim"])
def test_true_native_history_shares_ics_and_uses_separate_reference_density(tmp_path, highres_dimension):
    pytest.importorskip("py21cmfast")
    path = tmp_path / "native.npz"
    result = build_native_history(output=path, dimension=8, box_len=24., threads=2,
                                  seed=1234, z_start=10., z_end=8., intervals=2,
                                  reference_z=8.5, highres_dimension=highres_dimension)
    loaded = load_history(path)
    np.testing.assert_array_equal(loaded["redshifts"], [10., 9., 8.])
    assert loaded["rates"].shape == (3, 8, 8, 8)
    expected_dim = 24 if highres_dimension is None else highres_dimension
    assert loaded["metadata"]["highres_dimension"] == expected_dim
    assert loaded["metadata"]["inputs"]["simulation_options"]["DIM"] == expected_dim
    assert loaded["metadata"]["inputs"]["simulation_options"]["HII_DIM"] == 8
    assert loaded["metadata"]["reference_redshift_is_source_endpoint"] is False
    assert loaded["metadata"]["same_initial_conditions_for_all_nodes"] is True
    assert loaded["metadata"]["local_cumulative_n_ion_differenced"] is False
    assert loaded["metadata"]["density_weighting_applied_to_n_ion"] is False
    assert loaded["metadata"]["box_hMpc"] == pytest.approx(24. * loaded["metadata"]["hlittle"])
    assert np.all(np.diff(loaded["times_seconds"]) > 0)
    assert np.any(loaded["rates"] > 0)
    for index, row in enumerate(loaded["metadata"]["source_provenance"]):
        assert row["native_n_ion_sum"] / row["native_sfr_timescale_seconds"] == pytest.approx(
            loaded["rates"][index].sum(dtype=np.float64), rel=1e-14)
    # The reference density is genuinely generated at 8.5, not silently
    # selected from the nearest available emission endpoint.
    assert array_hash(result["capacity"]) not in {
        row["native_density_capacity_sha256_float64"]
        for row in result["metadata"]["source_provenance"]}


@pytest.mark.parametrize("highres_dimension", [True, 0, -1, 7, 8.5, "32"])
def test_invalid_highres_dimension_rejected_before_generation(tmp_path, highres_dimension):
    path = tmp_path / "new" / "native.npz"
    with pytest.raises(ValueError, match="highres_dimension must be an integer >= 8"):
        build_native_history(output=path, dimension=8, highres_dimension=highres_dimension)
    assert not path.parent.exists()


def test_native_dimension_override_preserves_other_inputs(monkeypatch):
    pytest.importorskip("py21cmfast")
    from native_history import ROOT
    monkeypatch.syspath_prepend(str(ROOT / "research/pc_reconsidered"))
    from native_driver import prepare_native
    default, default_changes = prepare_native(dimension=8)
    override, override_changes = prepare_native(dimension=8, highres_dimension=32)
    a = default.asdict(include_cosmo_tables="never", only_cstruct_params=False)
    b = override.asdict(include_cosmo_tables="never", only_cstruct_params=False)
    assert a["simulation_options"]["DIM"] == 24
    assert b["simulation_options"]["DIM"] == 32
    b["simulation_options"]["DIM"] = a["simulation_options"]["DIM"]
    assert a == b
    assert default_changes == override_changes
