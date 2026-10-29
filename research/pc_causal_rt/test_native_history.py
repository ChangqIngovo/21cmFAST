"""Native atom trajectories, time sources and archived history identity."""

from types import SimpleNamespace

import numpy as np
import pytest

from research.pc_causal_rt.native_history import (
    ARRAY_NAMES, PACKET_ARRAY_NAMES, _hash_array, _save, load_history,
    packets_from_initial, positions_at_growth, validate_history,
    background_at_time, _native_age_seconds, _redshift_at_time_fraction,
)


class Initial:
    def __init__(self, fields):
        self.fields = fields

    def get(self, name):
        return self.fields[name]


def inputs(n=2, high=4, box_len=10):
    return SimpleNamespace(simulation_options=SimpleNamespace(HII_DIM=n, DIM=high, BOX_LEN=box_len))


def initial(n=2, high=4):
    fields = {"hires_density": np.zeros((high,) * 3, dtype=np.float32)}
    fields.update({f"lowres_v{axis}{suffix}": np.zeros((n,) * 3, dtype=np.float32)
                   for axis in "xyz" for suffix in ("", "_2LPT")})
    return Initial(fields)


def test_packet_atoms_match_native_normalization_and_integer_nodes():
    packets = packets_from_initial(initial(), inputs(), .01)
    assert packets["mass"].sum() == 8
    assert np.all(packets["mass"] == .125)
    np.testing.assert_array_equal(packets["q"][0], [0, 0, 0])
    np.testing.assert_array_equal(packets["q"][-1], [1.5, 1.5, 1.5])
    np.testing.assert_array_equal(positions_at_growth(packets, .2), packets["q"])


def test_native_nearest_displacement_lookup_and_2lpt_sign():
    ic = initial()
    ic.fields["lowres_vx"][0] = 1
    ic.fields["lowres_vx"][1] = 3
    ic.fields["lowres_vx_2LPT"][:] = 2
    packets = packets_from_initial(ic, inputs(), .01)
    first_at_x = [0, 16, 32, 48]
    # Native floor(i * 2/4 + 0.5) wraps [0,1,1,2] -> [0,1,1,0].
    np.testing.assert_allclose(packets["v1"][first_at_x, 0], [.2, .6, .6, .2], rtol=0, atol=2e-16)
    position = positions_at_growth(packets, .2)
    expected_x = packets["q"][:, 0] + .19 * packets["v1"][:, 0] + 3 / 7 * (.2**2 - .01**2) * .4
    np.testing.assert_allclose(position[:, 0], expected_x, rtol=0, atol=2e-16)


def test_packet_initial_mass_keeps_density_and_rejects_negative_atoms():
    ic = initial()
    ic.fields["hires_density"].flat[0] = 2
    packets = packets_from_initial(ic, inputs(), .1)
    assert packets["mass"][0] == pytest.approx(1.2 / 8)
    ic.fields["hires_density"].flat[1] = -11
    with pytest.raises(ValueError, match="negative mass"):
        packets_from_initial(ic, inputs(), .1)


@pytest.mark.parametrize("growth", [0, -1, np.nan, np.inf])
def test_invalid_growth_is_rejected(growth):
    packets = packets_from_initial(initial(), inputs(), .01)
    with pytest.raises(ValueError, match="Growth"):
        positions_at_growth(packets, growth)


def history_fixture():
    result = {"redshifts": np.array([9., 8.]), "times_seconds": np.array([0., 1e15]),
              "growth_factors": np.array([.1, .12]),
              "rates": np.full((2, 2, 2, 2), 1e-15),
              "native_capacity": np.ones((2, 2, 2, 2)),
              "native_xhi": np.full((2, 2, 2, 2), .5),
              "native_brightness": np.full((2, 2, 2, 2), 10.),
              "packets": packets_from_initial(initial(), inputs(), .01), "records": []}
    result["metadata"] = {"schema": 1, "dimension": 2, "growth_initial": .01, "records": [],
                          "array_sha256": {k: _hash_array(result[k]) for k in ARRAY_NAMES}
                          | {f"packet_{k}": _hash_array(result["packets"][k]) for k in PACKET_ARRAY_NAMES}}
    return result


def test_history_save_load_preserves_rates_trajectories_and_hashes(tmp_path):
    expected = history_fixture()
    _save(tmp_path / "history.npz", expected)
    actual = load_history(tmp_path / "history.npz")
    for key in ARRAY_NAMES:
        np.testing.assert_array_equal(expected[key], actual[key])
    for key in PACKET_ARRAY_NAMES:
        np.testing.assert_array_equal(expected["packets"][key], actual["packets"][key])


def test_history_rejects_changed_packet_positions_without_mass_change():
    data = history_fixture()
    data["packets"]["q"][0, 0] += .01
    with pytest.raises(ValueError, match="packet hash"):
        validate_history(data)


def test_history_rejects_inconsistent_cosmic_clock():
    data = history_fixture()
    data["times_seconds"][1] = -1
    with pytest.raises(ValueError, match="ordered physical time"):
        validate_history(data)


@pytest.mark.parametrize("name,value", [("rates", -1), ("native_capacity", 0), ("native_xhi", 1.1)])
def test_history_rejects_invalid_native_fields(name, value):
    data = history_fixture()
    data[name].flat[0] = value
    with pytest.raises(ValueError, match="violates bounds"):
        validate_history(data)


def test_actual_native_packets_rates_and_restart(tmp_path):
    pytest.importorskip("py21cmfast")
    from research.pc_causal_rt.native_history import build_native_history
    from research.pc_evolution.native_history import rate_from_nion
    config = {"dimension": 4, "highres_dimension": 12, "box_len": 50., "threads": 1,
              "z_start": 10., "z_end": 9., "source_intervals": 1, "native_inputs_json": None}
    history = build_native_history(tmp_path, config, 111)
    metadata = history["metadata"]
    background = background_at_time(history, 1, .5)
    from research.pc_evolution.native_history import native_time_interval
    from py21cmfast.wrapper.cfuncs import get_growth_factor
    from research.pc_causal_rt.native_history import _inputs_from_serialized
    import json
    elapsed, _ = native_time_interval(history["redshifts"][0], background["redshift"], metadata["cosmology"])
    assert elapsed == pytest.approx(history["times_seconds"][1] / 2, rel=2e-13)
    native_inputs = _inputs_from_serialized(json.dumps(metadata["inputs"], sort_keys=True))
    assert background["growth_factor"] == get_growth_factor(
        inputs=native_inputs, redshift=float(np.float32(background["redshift"])))
    assert background_at_time(history, 1, .5) == background
    assert max(row["maximum_absolute_density_difference"] for row in metadata["density_audits"]) < 1e-6
    assert np.ptp([row["packet_atom_total"] for row in metadata["density_audits"]]) == 0
    for i, record in enumerate(history["records"]):
        with np.load(tmp_path / record["fields_path"], allow_pickle=False) as saved:
            expected = rate_from_nion(saved["source"], record["redshift"],
                                      t_star=metadata["inputs"]["astro_params"]["t_STAR"],
                                      cosmo=metadata["cosmology"])
            np.testing.assert_array_equal(history["rates"][i], expected)
    reused = build_native_history(tmp_path, config, 111)
    assert reused["metadata"]["array_sha256"] == metadata["array_sha256"]
    # Native archives are also protected, not just the aggregate history NPZ.
    with (tmp_path / history["records"][0]["fields_path"]).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Checkpoint artifact changed"):
        build_native_history(tmp_path, config, 111)


@pytest.mark.parametrize("om,ol", [(.31, .69), (1., 0.)])
@pytest.mark.parametrize("old,new", [(25., 24.), (10., 7.), (100., 0.)])
def test_analytic_native_age_and_inverse_agree_with_independent_quadrature(om, ol, old, new):
    from research.pc_evolution.native_history import native_time_interval
    cosmo = {"hlittle": .6766, "OMm": om, "OMl": ol, "OMk": 0., "wl": -1.}
    duration, _ = native_time_interval(old, new, cosmo)
    age_difference = _native_age_seconds(new, cosmo) - _native_age_seconds(old, cosmo)
    assert age_difference == pytest.approx(duration, rel=2e-13)
    for fraction in (.125, .5, .875):
        z = _redshift_at_time_fraction(old, new, fraction, cosmo)
        elapsed, _ = native_time_interval(old, z, cosmo)
        assert elapsed / duration == pytest.approx(fraction, rel=2e-12)


def test_background_endpoints_match_saved_native_fields_exactly():
    history = history_fixture()
    for fraction, index in [(0., 0), (1., 1)]:
        value = background_at_time(history, 1, fraction)
        assert value == {"redshift": history["redshifts"][index],
                         "growth_factor": history["growth_factors"][index],
                         "time_seconds": history["times_seconds"][index]}


@pytest.mark.parametrize("node,fraction", [(0, .5), (2, .5), (1, -.1), (1, 1.1), (1, np.nan)])
def test_background_rejects_invalid_substep_coordinates(node, fraction):
    with pytest.raises(ValueError, match="right-endpoint"):
        background_at_time(history_fixture(), node, fraction)
