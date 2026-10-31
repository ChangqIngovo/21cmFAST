"""Native cumulative photon budgets under source motion and refinement."""
import numpy as np
import pytest

from research.pc_causal_rt import workflow as w


def fixture(monkeypatch, *, amounts=(.2, .5, .8), moving=True, factor=1, substeps=2):
    from research.pc_causal_rt import native_history
    monkeypatch.setattr(native_history, "background_at_time", lambda h, node, f:
        {"redshift": float((1-f)*h["redshifts"][node-1] + f*h["redshifts"][node]),
         "growth_factor": float((1-f)*h["growth_factors"][node-1] + f*h["growth_factors"][node])})
    n = 4
    q = np.indices((n,) * 3).reshape(3, -1).T.astype(float)
    packets = {"mass": np.ones(n**3), "q": q, "v1": np.zeros_like(q),
               "v2": np.zeros_like(q), "growth_initial": .01, "coarse_dimension": n}
    source_packets = {k: value for k, value in packets.items() if k != "mass"}
    source_packets["v1"] = np.zeros_like(q)
    if moving:
        source_packets["v1"][:, 0] = 6.
    cumulative = np.zeros((3, n, n, n))
    cumulative[:, 0, 0, 0] = amounts
    times = np.array([0., 1e12, 2e12])
    growth = np.array([.1, .2, .3])
    zs = np.array([10., 9.9, 9.8])
    budget = {"cumulative": cumulative, "times_seconds": times,
              "growth_factors": growth, "redshifts": zs,
              "native_totals": cumulative.sum(axis=(1, 2, 3)),
              "packets": source_packets, "metadata": {}}
    history = {"packets": packets, "times_seconds": times, "growth_factors": growth,
               "redshifts": zs, "source_budget": budget, "sampled_cumulative": cumulative,
               "metadata": {"inputs": {"cosmo_params": {"OMb": .049, "hlittle": .6766, "Y_He": .24}}}}
    setting = dict(directions=24, refinement=factor, cfl=.4,
                   time_substeps=substeps, source_stride=1)
    config = w.configuration(dict(dimension=n, threads=1, seeds=[111],
        source_intervals=2, z_start=10., z_end=9.8, settings=[setting]))
    return history, setting, config


def evolve(history, setting, config):
    state = w._initial_state(history, setting, config)
    for node in (1, 2):
        state = w.advance_node(history, node, setting, config, state["ionized"],
                               state["photons"], state["cumulative_source"])
    return state


def test_default_source_is_native_cumulative():
    assert w.configuration()["source_mode"] == "native_cumulative"
    assert w.configuration({"source_mode": "instantaneous"})["source_mode"] == "instantaneous"
    with pytest.raises(ValueError, match="source_mode"):
        w.configuration({"source_mode": "rescaled"})


def test_initial_native_budget_is_radiation_counted_once(monkeypatch):
    h, setting, cfg = fixture(monkeypatch, factor=2)
    state = w._initial_state(h, setting, cfg)
    assert np.count_nonzero(state["ionized"]) == 0
    assert state["photons"].sum() / 8 == pytest.approx(.2, rel=1e-14)
    assert state["cumulative_source"].sum() == pytest.approx(.2, rel=1e-14)
    np.testing.assert_allclose(state["cumulative_source"], w._current_source_map(h, 0), rtol=0, atol=1e-15)
    assert state["audit"]["source_budget_relative_residual"] < 1e-14


def test_source_translation_emits_no_additional_photons(monkeypatch):
    h, setting, cfg = fixture(monkeypatch, amounts=(.2, .2, .2))
    initial = w._initial_state(h, setting, cfg)
    current0, current2 = w._current_source_map(h, 0), w._current_source_map(h, 2)
    assert np.any(current2 - current0 < 0)
    assert not np.array_equal(current0, current2)
    final = evolve(h, setting, cfg)
    np.testing.assert_array_equal(final["cumulative_source"], initial["cumulative_source"])
    assert final["audit"]["total_emitted"] == pytest.approx(.2, rel=1e-14)
    assert final["audit"]["photon_ledger_relative_residual"] < 2e-10


@pytest.mark.parametrize("factor", [1, 2])
def test_growth_and_motion_preserve_exact_native_total(monkeypatch, factor):
    h, setting, cfg = fixture(monkeypatch, factor=factor)
    final = evolve(h, setting, cfg)
    assert final["audit"]["total_emitted"] == pytest.approx(.8, rel=1e-14)
    total = (final["photons"].sum() + final["ionized"].sum()) / factor**3
    assert total == pytest.approx(.8, rel=2e-10)
    assert final["audit"]["source_budget_relative_residual"] < 1e-14
    assert final["audit"]["source_sampling_relative_difference"] == 0


def test_matter_substeps_may_change_emission_map_but_not_budget(monkeypatch):
    h, setting, cfg = fixture(monkeypatch, substeps=1)
    first = evolve(h, setting, cfg)
    setting2 = setting | {"time_substeps": 4}
    second = evolve(h, setting2, cfg)
    assert np.max(np.abs(first["cumulative_source"] - second["cumulative_source"])) > 1e-4
    assert first["cumulative_source"].sum() == pytest.approx(second["cumulative_source"].sum(), rel=1e-14)
    assert first["audit"]["source_budget_relative_residual"] < 1e-14
    assert second["audit"]["source_budget_relative_residual"] < 1e-14


def test_missing_initial_budget_is_rejected_independently(monkeypatch):
    h, _, _ = fixture(monkeypatch)
    incorrect = np.zeros((4, 4, 4))
    incorrect[0, 0, 0] = .8 - .2
    with pytest.raises(RuntimeError, match="native cumulative"):
        w._audit_native_budget(h, 2, incorrect)


def test_native_budget_restart_preserves_radiation_and_source_history(tmp_path, monkeypatch):
    h, setting, cfg = fixture(monkeypatch)
    cfg["checkpoint_stride"] = 1
    archive = tmp_path / "native_archive"
    archive.mkdir()
    (archive / "history.npz").write_bytes(b"immutable original native history")
    h["metadata"]["native_archive_root"] = str(archive)
    budget_path = tmp_path / "budget.npz"
    budget_path.write_bytes(b"immutable exported cumulative packet budget")
    h["source_budget"].update(path=str(budget_path), sha256=w._digest(budget_path))
    h["source_budget"]["metadata"]["history_sha256"] = w._digest(archive / "history.npz")
    monkeypatch.setattr(w, "_runtime_identity", lambda: {"native_extension_sha256": "fixture"})

    def checkpoint(root, folder, history, node, numerical, config, state):
        path = folder / "checkpoints" / f"node{node:05d}.npz"
        w._write_npz(path, **{k: state[k] for k in ("capacity", "ionized", "photons", "cumulative_source")})
        return {"node": node, "redshift": float(history["redshifts"][node]),
                "time_seconds": float(history["times_seconds"][node]),
                "fields_path": str(path.relative_to(root)), "fields_sha256": w._digest(path),
                **state["audit"]}

    monkeypatch.setattr(w, "_checkpoint", checkpoint)
    original = w.advance_node

    def interrupted(history, node, *args):
        if node == 2:
            raise RuntimeError("interrupted after first source increment")
        return original(history, node, *args)

    monkeypatch.setattr(w, "advance_node", interrupted)
    output = tmp_path / "interrupted"
    with pytest.raises(RuntimeError, match="interrupted"):
        w.run_case(output, h, 111, setting, cfg, {})
    calls = []

    def resumed(history, node, *args):
        calls.append(node)
        return original(history, node, *args)

    monkeypatch.setattr(w, "advance_node", resumed)
    resumed_case = w.run_case(output, h, 111, setting, cfg, {})
    assert calls == [2]
    monkeypatch.setattr(w, "advance_node", original)
    continuous_root = tmp_path / "continuous"
    continuous_case = w.run_case(continuous_root, h, 111, setting, cfg, {})
    with np.load(output / resumed_case["checkpoints"][-1]["fields_path"]) as a, np.load(
            continuous_root / continuous_case["checkpoints"][-1]["fields_path"]) as b:
        for key in ("capacity", "ionized", "photons", "cumulative_source"):
            np.testing.assert_array_equal(a[key], b[key])
    assert resumed_case["checkpoints"][-1]["total_emitted"] == pytest.approx(.8, rel=1e-14)
