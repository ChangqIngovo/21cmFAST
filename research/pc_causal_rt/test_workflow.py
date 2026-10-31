"""Persistent evolution, native integration and genuine restart checks."""
import json
from pathlib import Path

import numpy as np
import pytest

from research.pc_causal_rt import workflow as w
from research.pc_causal_rt.gas import deposit


def small_config(**kwargs):
    return w.configuration(dict(dimension=4, threads=1, seeds=[111], source_intervals=4, source_mode="instantaneous",
        z_start=10., z_end=7., checkpoint_stride=2,
        settings=[dict(directions=24, refinement=1, cfl=.4, time_substeps=2, source_stride=1)]) | kwargs)


def test_default_has_two_adjacent_refinements_at_shared_anchor():
    config = w.configuration()
    assert len(config["settings"]) == 11
    anchor = dict(directions=384, refinement=4, cfl=.1, time_substeps=4, source_stride=1)
    for axis in anchor:
        ladder = [r for r in config["settings"] if all(r[k] == anchor[k] for k in anchor if k != axis)]
        assert len(ladder) == 3


@pytest.mark.parametrize("override", [
    {"dimension": 0}, {"threads": 33}, {"z_start": 3, "z_end": 5},
    {"source_intervals": 3}, {"tolerance": 0}, {"native_inputs_json": "inputs.json"},
    {"unknown": True}, {"photon_energy_ev": 10},
])
def test_invalid_configs_rejected(override):
    with pytest.raises((ValueError, TypeError)):
        w.configuration(override)


def test_source_knots_use_physical_time_not_node_number():
    rates = np.array([0, 20, 30, 10.])[:, None, None, None]
    times = np.array([0, 1, 4, 10.])
    result = w.sample_rates(rates, times, 3)
    np.testing.assert_array_equal(result.ravel(), [0, 1, 4, 10])


def test_zero_source_continuation_keeps_ionized_atoms_and_photons(monkeypatch):
    from research.pc_causal_rt import native_history
    monkeypatch.setattr(native_history, "background_at_time", lambda h, i, f:
        dict(redshift=float((1-f)*h["redshifts"][i-1]+f*h["redshifts"][i]),
             growth_factor=float((1-f)*h["growth_factors"][i-1]+f*h["growth_factors"][i])))
    n = 4
    positions = np.indices((n,) * 3).reshape(3, -1).T.astype(float)
    packets = dict(mass=np.ones(n**3), q=positions, v1=np.zeros_like(positions),
                   v2=np.zeros_like(positions), growth_initial=.01, coarse_dimension=n)
    capacity = deposit(packets["mass"], positions, n)
    cfg = small_config()
    setting = cfg["settings"][0]
    history = dict(packets=packets, times_seconds=np.array([0, 1e12, 2e12]),
        redshifts=np.array([10., 9.9, 9.8]), growth_factors=np.array([.1, .11, .12]),
        sampled_rates=np.zeros((3, n, n, n)),
        metadata={"inputs": {"cosmo_params": {"OMb": .049, "hlittle": .6766, "Y_He": .24}}})
    photons = np.full((24, n, n, n), .2 / 24)
    ions = .3 * capacity
    source = np.full_like(capacity, .5, dtype=np.longdouble)
    first = w.advance_node(history, 1, setting, cfg, ions, photons, source)
    second = w.advance_node(history, 2, setting, cfg, first["ionized"], first["photons"], first["cumulative_source"])
    assert second["ionized"].sum() >= first["ionized"].sum() >= ions.sum()
    assert second["ionized"].sum() > .49 * n**3
    np.testing.assert_array_equal(second["cumulative_source"], source)
    assert second["audit"]["photon_ledger_relative_residual"] < 1e-12


def test_audit_rejects_density_reset_that_erases_ions():
    capacity = np.ones((4, 4, 4))
    with pytest.raises(RuntimeError, match="ledger"):
        w.audit_state(capacity, .4 * capacity, np.zeros((6, 4, 4, 4)), .5 * capacity, 1, 64)


def test_native_run_restarts_without_resetting_state(tmp_path, monkeypatch):
    pytest.importorskip("py21cmfast")
    cfg = small_config()
    original = w.advance_node
    called = []
    def interrupted(history, node, *args):
        called.append(node)
        if node == 3:
            raise RuntimeError("simulated interruption")
        return original(history, node, *args)
    monkeypatch.setattr(w, "advance_node", interrupted)
    root = tmp_path / "interrupted"
    with pytest.raises(RuntimeError, match="simulated interruption"):
        w.run(root, cfg)
    assert called == [1, 2, 3]
    called.clear()
    def resumed(history, node, *args):
        called.append(node)
        return original(history, node, *args)
    monkeypatch.setattr(w, "advance_node", resumed)
    result = w.run(root, cfg)
    assert called == [3, 4]
    assert result["all_ledgers_passed"]
    assert not result["convergence_certified"]
    assert (root / "COMPLETED").exists()
    checkpoint = result["cases"][0]["checkpoints"][-1]
    assert checkpoint["maximum_native_density_roundoff"] < 3e-6
    assert checkpoint["total_ionized"] > 0
    assert checkpoint["photon_ledger_relative_residual"] < 2e-10
    called.clear()
    assert w.run(root, cfg) == result
    assert not called
    monkeypatch.setattr(w, "advance_node", original)
    continuous = w.run(tmp_path / "continuous", cfg)
    with np.load(root / checkpoint["fields_path"]) as a, np.load(
            tmp_path / "continuous" / continuous["cases"][0]["checkpoints"][-1]["fields_path"]) as b:
        for key in ("capacity", "ionized", "radiation", "cumulative_source", "rt_brightness"):
            np.testing.assert_array_equal(a[key], b[key])
    path = root / checkpoint["fields_path"]
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        w.run(root, cfg)
