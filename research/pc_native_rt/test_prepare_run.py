"""Preparation and scientific-runtime identity regression tests."""

import json
from pathlib import Path

import pytest

from research.pc_native_rt import prepare_run as prep


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    run = tmp_path / "run"
    (run / "source").mkdir(parents=True)
    (run / "source/code.py").write_text("source = 'native'\n")
    for name, value in {
        "python_path.txt": "/test/python\n", "commit.txt": "a" * 40,
        "runtime.env": "PC_MODULES=()\n", "submit.pbs": "#!/bin/bash\n",
    }.items():
        (run / name).write_text(value)
    base = {"native_package": "/base/py21cmfast", "extension_sha256": "original",
            "versions": {"py21cmfast": "4.3.1.dev1+g12345"}}
    monkeypatch.setattr(prep, "runtime_fingerprint", lambda *a, **k: dict(base))
    return run, base


def test_default_job_is_small_coeval_matrix():
    config = prep.configuration({})
    assert config["redshifts"] == [10, 9, 8, 7, 6, 5]
    assert config["settings"] == [{"directions": 96, "refinement": 1, "cfl": .4}]
    assert prep.validate_allocation(config, {})["cases"] == 12
    assert "not cosmological time" in prep.SCOPE


def test_environment_axes_and_physical_parameters_are_preserved():
    config = prep.configuration({
        "PC_DIMENSION": "16", "PC_HIGHRES_DIMENSION": "48", "PC_BOX_LEN": "75",
        "PC_THREADS": "4", "PC_SEEDS": "333", "PC_REDSHIFTS": "8,7.5",
        "PC_DIRECTIONS": "24,96", "PC_REFINEMENTS": "1 2", "PC_CFL_VALUES": ".2",
        "PC_PHOTON_ENERGY_EV": "20", "PC_EXHAUSTION_TOLERANCE": "1e-7",
        "PC_MAX_BOX_CROSSINGS": "12", "PC_INJECTION_BOX_CROSSINGS": "1",
    })
    assert config["dimension"] == 16
    assert config["highres_dimension"] == 48
    assert config["box_len"] == 75
    assert config["seeds"] == [333]
    assert config["redshifts"] == [8, 7.5]
    assert len(config["settings"]) == 4
    assert all(row["cfl"] == .2 for row in config["settings"])
    assert config["injection_box_crossings"] == 1
    assert config["exhaustion_tolerance"] == 1e-7


@pytest.mark.parametrize("environment,match", [
    ({"PC_MEMORY_GB": "nan"}, "positive and finite"),
    ({"PC_MEM": "2GB", "PC_MEMORY_GB": "2"}, "leave at least"),
    ({"PC_MEM": "unlimited"}, "allocation"),
])
def test_memory_allocation_is_not_silently_adjusted(environment, match):
    with pytest.raises(ValueError, match=match):
        prep.validate_allocation(prep.configuration({}), environment)


def test_large_angular_matrix_rejected_before_submission():
    config = prep.configuration({"PC_DIMENSION": "128", "PC_DIRECTIONS": "384"})
    with pytest.raises(ValueError, match="exceed PC_MEMORY_GB"):
        prep.validate_allocation(config, {"PC_MEMORY_GB": "1", "PC_MEM": "4GB"})


def test_shallow_untagged_runtime_is_rejected_before_submission(prepared):
    run, base = prepared
    base["versions"]["py21cmfast"] = "0.0.1.dev1+g12345"
    with pytest.raises(ValueError, match="genuine upstream tags"):
        prep.prepare(run, {})
    assert not (run / "PREPARED").exists()


def test_archived_format_version_is_checked_without_importing_sources(tmp_path):
    compat = tmp_path / "src/py21cmfast/io/_compat"
    compat.mkdir(parents=True)
    (compat / "_v4_2_v4_3.py").write_text('RELEASE = FormatRelease("4.2.0", "4.3.0", [])\n')
    assert prep.validate_native_version(tmp_path, "4.3.1.dev5+g1234") == "4.3.0"
    with pytest.raises(ValueError, match="HDF format 4.3.0"):
        prep.validate_native_version(tmp_path, "4.2.1")


def test_frozen_run_rejects_source_change_and_cpu_mismatch(prepared):
    run, _ = prepared
    manifest = prep.prepare(run, {})
    assert manifest["native_rebuild"] is True
    prep.verify(run, cpus=8)
    with pytest.raises(ValueError, match="CPU allocation"):
        prep.verify(run, cpus=16)
    (run / "source/code.py").write_text("source = 'modified'\n")
    with pytest.raises(ValueError, match="Frozen source"):
        prep.verify(run)


def test_native_inputs_are_copied_and_hash_protected(prepared, tmp_path):
    run, _ = prepared
    original = tmp_path / "custom.json"
    original.write_text(json.dumps({"simulation_options": {"HII_DIM": 8}}))
    manifest = prep.prepare(run, {"PC_NATIVE_INPUTS_JSON": str(original), "PC_SEEDS": "111"})
    copied = Path(manifest["configuration"]["native_inputs_json"])
    assert copied == run / "native_inputs.json"
    original.write_text("{}")
    prep.verify(run)
    copied.write_text("{}")
    with pytest.raises(ValueError, match="native_inputs.json"):
        prep.verify(run)


def test_changed_base_runtime_is_rejected(prepared):
    run, base = prepared
    prep.prepare(run, {})
    base["extension_sha256"] = "changed"
    with pytest.raises(ValueError, match="Base runtime"):
        prep.verify(run)


def test_scientific_runtime_must_use_frozen_build(prepared):
    run, _ = prepared
    prep.prepare(run, {})
    with pytest.raises(ValueError, match="isolated native build"):
        prep.register_native(run)


def test_registered_extension_change_fails_restart(prepared, monkeypatch):
    run, _ = prepared
    prep.prepare(run, {})
    scientific = {"native_package": str(run / "native_build/source/src/py21cmfast"),
                  "extension_sha256": "compiled"}
    monkeypatch.setattr(prep, "runtime_fingerprint", lambda *a, **k: dict(scientific))
    prep.register_native(run, cpus=8)
    prep.register_native(run, cpus=8)
    scientific["extension_sha256"] = "changed"
    with pytest.raises(ValueError, match="Scientific runtime changed"):
        prep.verify_native(run)


def test_existing_results_without_native_registration_are_rejected(prepared):
    run, _ = prepared
    prep.prepare(run, {})
    (run / "results").mkdir()
    with pytest.raises(ValueError, match="no registered"):
        prep.register_native(run)


def test_frozen_pbs_uses_new_workflow_and_shared_native_builder():
    folder = Path(prep.__file__).parent
    body = (folder / "gadi_run.pbs").read_text()
    assert "research.pc_matched.native_build" in body
    assert "research.pc_native_rt.workflow --root" in body
    assert "research.pc_matched.workflow" not in body
    assert body.index("register-native") < body.index("research.pc_native_rt.workflow")
    discovery = (folder / "gadi_prepare.sh").read_text()
    assert discovery.index("runtime_select.py") < discovery.index("source \"$run/runtime.env\"")
    assert discovery.index("source \"$run/runtime.env\"") < discovery.index("research.pc_native_rt.prepare_run")
