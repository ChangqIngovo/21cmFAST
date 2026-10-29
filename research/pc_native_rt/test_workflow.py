"""Contract tests for native-source preservation and checkpoint integrity."""
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from research.pc_native_rt import workflow as w


@pytest.mark.parametrize("overrides", [
    {"unexpected": 1}, {"dimension": True}, {"threads": 33},
    {"redshifts": [7, 8]}, {"redshifts": [8, 8]}, {"seeds": [1, 1]},
    {"settings": [{"directions": 97, "refinement": 1, "cfl": .4}]},
    {"settings": [{"directions": 96, "refinement": 0, "cfl": .4}]},
    {"settings": [{"directions": 96, "refinement": 1, "cfl": 0}]},
    {"native_inputs_json": "inputs.json"},
    {"injection_box_crossings": 21}, {"exhaustion_tolerance": 1},
])
def test_configuration_rejects_invalid_or_ambiguous(overrides):
    with pytest.raises(ValueError):
        w.configuration(overrides)


def test_configuration_defaults_are_independent():
    first = w.configuration()
    first["settings"][0]["directions"] = 6
    assert w.configuration()["settings"][0]["directions"] == 96


def test_refinement_preserves_source_density_and_original_cell_counts():
    source = np.arange(27, dtype=float).reshape(3, 3, 3) / 7
    fine = w._repeat(source, 4)
    np.testing.assert_allclose(w._restrict(fine, 3), source, rtol=1e-15, atol=0)
    assert fine.sum() / 4**3 == pytest.approx(source.sum())


@pytest.fixture
def fake_case(tmp_path, monkeypatch):
    """Mock native I/O/brightness only; retain real ledger and field audits."""
    from research.pc_reconsidered import native_driver
    source = np.linspace(.01, .9, 8).reshape(2, 2, 2)
    capacity = np.linspace(.9, 2., 8).reshape(2, 2, 2)
    native_xhi = 1 - .8 * source / capacity
    brightness = 27 * capacity * native_xhi
    input_values = {"cosmo_params": {"OMb": .049, "hlittle": .6751, "Y_He": .245},
                    "astro_options": {"preserved_marker": "native"}}
    inputs = SimpleNamespace(asdict=lambda **kw: input_values)
    coeval = SimpleNamespace(inputs=inputs, redshift=8., perturbed_field=object(),
                             ionized_box=SimpleNamespace(get=lambda key: native_xhi),
                             brightness_temp=brightness)
    extracted = SimpleNamespace(source=source.copy(), capacity=capacity.copy())
    monkeypatch.setattr(native_driver, "from_native_coeval", lambda _: extracted)
    def make_box(_, ionized, atoms):
        return SimpleNamespace(inputs=inputs, get=lambda key: np.asarray(1 - ionized / atoms, dtype=np.float32))
    monkeypatch.setattr(native_driver, "_native_box", make_box)
    modules = {}
    for name in ("py21cmfast", "py21cmfast.drivers", "py21cmfast.drivers.coeval", "py21cmfast.drivers.single_field"):
        modules[name] = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, modules[name])
    modules["py21cmfast.drivers.coeval"].Coeval = SimpleNamespace(from_file=lambda _: coeval)
    def brightness_temperature(**kw):
        assert kw["perturbed_field"] is coeval.perturbed_field
        assert kw["write"] is False
        return SimpleNamespace(get=lambda _: 27 * capacity * kw["ionized_box"].get("neutral_fraction"))
    modules["py21cmfast.drivers.single_field"].brightness_temperature = brightness_temperature
    from research.pc_native_rt import solver
    calls = []
    def solve(s, c, **kw):
        calls.append((s.copy(), c.copy(), kw))
        return {"ionized": .9 * s,
                "radiation": np.broadcast_to(.1 * s / kw["directions"], (kw["directions"], *s.shape)).copy(),
                "audit": {"ledger_passed": True, "converged": True}, "metadata": {"box_crossings": 1.}}
    monkeypatch.setattr(solver, "solve", solve)
    folder = tmp_path / "native" / "seed111" / "z8"
    folder.mkdir(parents=True)
    (folder / "native.h5").write_text("native archive placeholder")
    w._write_npz(folder / "source.npz", source=source, capacity=capacity,
                 native_es_xhi=native_xhi, native_brightness=brightness)
    w._write_json(folder / "native.json", {"identity": {"inputs": input_values}})
    w._write_json(folder / "receipt.json", {"files": {name: w._digest(folder / name)
                  for name in ("native.h5", "source.npz", "native.json")}})
    native = {"seed": 111, "redshift": 8., "h5_path": str((folder / "native.h5").relative_to(tmp_path)),
              "fields_path": str((folder / "source.npz").relative_to(tmp_path)),
              "metadata_path": str((folder / "native.json").relative_to(tmp_path)),
              "source_sha256": w._hash_array(source), "capacity_sha256": w._hash_array(capacity)}
    cfg = w.configuration({"dimension": 2, "threads": 1, "seeds": [111], "redshifts": [8.]})
    return SimpleNamespace(root=tmp_path, source=source, capacity=capacity, coeval=coeval,
                           native=native, config=cfg, calls=calls, inputs=input_values, solve=solve)


def test_case_passes_exact_native_budget_and_each_coeval_density(fake_case):
    f = fake_case
    row = w._run_case(f.root, f.config, f.native, {"directions": 24, "refinement": 2, "cfl": .2})
    s, c, params = f.calls[0]
    np.testing.assert_array_equal(s, w._repeat(f.source, 2))
    np.testing.assert_array_equal(c, w._repeat(f.capacity, 2))
    assert params["redshift"] == 8.
    assert params["hydrogen_density_z0_cm3"] != 1.9e-7
    assert row["source_total"] == pytest.approx(f.source.sum())
    assert row["rt_radiation_total"] == pytest.approx(.1 * f.source.sum())
    assert row["ledger_relative_residual"] < 1e-14
    assert row["native_inputs_unchanged"] and row["source_and_capacity_unchanged"]
    with np.load(f.root / row["fields_path"]) as saved:
        np.testing.assert_array_equal(saved["source"], f.source)
        np.testing.assert_array_equal(saved["capacity"], f.capacity)
        np.testing.assert_allclose(saved["ionized"], .9 * f.source)
        np.testing.assert_array_equal(saved["native_brightness"], f.coeval.brightness_temp)


def test_case_reuse_and_tamper_rejection(fake_case):
    f = fake_case
    setting = {"directions": 24, "refinement": 1, "cfl": .4}
    row = w._run_case(f.root, f.config, f.native, setting)
    assert w._run_case(f.root, f.config, f.native, setting) == row
    assert len(f.calls) == 1
    (f.root / row["fields_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        w._run_case(f.root, f.config, f.native, setting)


def test_case_rejects_photon_loss(fake_case, monkeypatch):
    from research.pc_native_rt import solver
    f = fake_case
    def losing(*args, **kwargs):
        result = f.solve(*args, **kwargs)
        result["radiation"] *= 0
        return result
    monkeypatch.setattr(solver, "solve", losing)
    with pytest.raises(RuntimeError, match="photon ledger"):
        w._run_case(f.root, f.config, f.native, f.config["settings"][0])


def test_case_rejects_changed_native_physics(fake_case, monkeypatch):
    from research.pc_native_rt import solver
    f = fake_case
    def mutating(*args, **kwargs):
        result = f.solve(*args, **kwargs)
        f.inputs["astro_options"]["preserved_marker"] = "changed"
        return result
    monkeypatch.setattr(solver, "solve", mutating)
    with pytest.raises(RuntimeError, match="modified native"):
        w._run_case(f.root, f.config, f.native, f.config["settings"][0])


def test_receipt_detects_identity_and_file_changes(tmp_path):
    (tmp_path / "field").write_bytes(b"original")
    w._write_json(tmp_path / "receipt.json", {"identity": {"seed": 1},
                  "files": {"field": w._digest(tmp_path / "field")}})
    assert w._receipt(tmp_path, "receipt.json", {"seed": 1}) is not None
    with pytest.raises(ValueError, match="identity"):
        w._receipt(tmp_path, "receipt.json", {"seed": 2})
    (tmp_path / "field").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact"):
        w._receipt(tmp_path, "receipt.json", {"seed": 1})


def test_native_custom_inputs_roundtrip_and_unsupported_physics(tmp_path):
    pytest.importorskip("py21cmfast", exc_type=ImportError)
    cfg = w.configuration({"dimension": 4, "seeds": [111], "threads": 1, "redshifts": [8.]})
    inputs, _ = w._native_inputs(cfg, 111)
    raw = w._input_dict(inputs)
    path = tmp_path / "native_inputs.json"
    path.write_text(json.dumps(raw))
    custom = cfg | {"native_inputs_json": str(path)}
    restored, provenance = w._native_inputs(custom, 111)
    assert w._input_dict(restored) == raw
    assert provenance["differences_from_supplied_inputs"] == {}
    with pytest.raises(ValueError, match="disagree"):
        w._native_inputs(custom | {"threads": 2}, 111)
    raw["astro_options"]["RECOMB_MODEL"] = "inhomogeneous"
    raw["node_redshifts"] = [40., 35., 20., 10., 8.]
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="RECOMB_MODEL|round-trip"):
        w._native_inputs(custom, 111)


def test_run_completion_reuse_rejects_changed_artifact(tmp_path, monkeypatch):
    from research.pc_native_rt import report
    cfg = w.configuration({"seeds": [111], "redshifts": [8.]})
    monkeypatch.setattr(w, "_source_hashes", lambda: {"solver": "fixed"})
    monkeypatch.setattr(w, "_runtime_identity", lambda: {"native_extension_sha256": "original"})
    monkeypatch.setattr(w, "_native_snapshots", lambda *_: [{"seed": 111, "redshift": 8.}])
    def fake_run_case(root, *args):
        (root / "saved_field").write_bytes(b"original")
        return {"native_inputs_unchanged": True, "source_and_capacity_unchanged": True,
                "solver_audit": {"ledger_passed": True, "converged": True}, "ledger_relative_residual": 0.}
    monkeypatch.setattr(w, "_run_case", fake_run_case)
    def fake_report(root):
        (root / "native_rt_report.json").write_text('{}')
    monkeypatch.setattr(report, "build_report", fake_report)
    first = w.run(tmp_path, cfg)
    assert (tmp_path / "COMPLETED").exists()
    assert w.run(tmp_path, cfg) == first
    monkeypatch.setattr(w, "_runtime_identity", lambda: {"native_extension_sha256": "changed"})
    with pytest.raises(ValueError, match="native runtime changed"):
        w.run(tmp_path, cfg)
    monkeypatch.setattr(w, "_runtime_identity", lambda: {"native_extension_sha256": "original"})
    (tmp_path / "native_rt_report.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="artifact changed"):
        w.run(tmp_path, cfg)
