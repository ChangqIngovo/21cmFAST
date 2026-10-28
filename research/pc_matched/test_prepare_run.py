"""Test frozen runtime selection and preparation without native simulations."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.pc_matched import prepare_run


def _executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o700)
    return path


def test_native_interpreter_takes_priority_over_training(tmp_path: Path) -> None:
    """Walk completed training parents to their original native runtime."""
    parent = tmp_path / "history_retrain_one"
    native = tmp_path / "full_history_one"
    parent.mkdir()
    native.mkdir()
    (parent / "COMPLETED").touch()
    (parent / "runtime.env").write_text("PC_MODULES=()\n")
    training_python = _executable(tmp_path / "train/bin/python")
    native_python = _executable(tmp_path / "native/bin/python")
    (parent / "python_path.txt").write_text(str(training_python))
    (parent / "parent_run.txt").write_text(str(native))
    (native / "native_python.txt").write_text(str(native_python))
    selected = prepare_run.select_runtime(tmp_path, parent=parent)
    assert selected["python"] == str(native_python)
    assert selected["runtime_env"] == str(parent / "runtime.env")


def test_explicit_python_needs_no_runtime_parent(tmp_path: Path) -> None:
    """An explicitly supplied local runtime can run without environment modules."""
    python = _executable(tmp_path / "venv/bin/python")
    result = prepare_run.select_runtime(tmp_path, python=python)
    assert result == {"python": str(python), "parent": "", "runtime_env": ""}


def test_missing_native_runtime_reports_parent_cycle(tmp_path: Path) -> None:
    """Broken ancestry cannot hang discovery or silently choose no interpreter."""
    (tmp_path / "COMPLETED").touch()
    (tmp_path / "runtime.env").write_text("PC_MODULES=()\n")
    (tmp_path / "parent_run.txt").write_text(str(tmp_path))
    with pytest.raises(ValueError, match="Cycle"):
        prepare_run.select_runtime(tmp_path, parent=tmp_path)


def test_default_configuration_and_overrides() -> None:
    """Defaults are the small common-source experiment, not production scales."""
    config = prepare_run.configuration({})
    assert config["profile"] == "pilot"
    assert config["source_strides"] == [1]
    assert config["dimension"] == 8
    assert config["seeds"] == [111, 222]
    assert config["base_intervals"] == 10
    assert config["directions"] == [6, 26]
    assert config["refinements"] == [1, 2]
    assert config["cfl_values"] == [0.4, 0.2, 0.1]
    assert config["photon_energy_ev"] == 18.0
    assert config["threads"] == 16
    assert prepare_run.configuration({"PC_SEEDS": "7, 11"})["seeds"] == [7, 11]
    with pytest.raises(ValueError, match="seeds"):
        prepare_run.configuration({"PC_SEEDS": "7,7"})
    with pytest.raises(ValueError, match="memory"):
        prepare_run.configuration({"PC_MEMORY_GB": "nan"})
    with pytest.raises(ValueError, match="PC_MEMORY_GB"):
        prepare_run.configuration({"PC_MEM": "32GB"})
    with pytest.raises(ValueError, match="direction"):
        prepare_run.configuration({"PC_DIRECTIONS": "6,7"})
    with pytest.raises(ValueError, match=r"(?i)cfl"):
        prepare_run.configuration({"PC_CFL_VALUES": "0.4,0"})
    with pytest.raises(ValueError, match=r"(?i)photon"):
        prepare_run.configuration({"PC_PHOTON_ENERGY_EV": "12"})


def test_convergence_profile_and_explicit_overrides() -> None:
    """Select a complete convergence matrix while preserving explicit resources."""
    config = prepare_run.configuration({"PC_PROFILE": "convergence"})
    assert config["profile"] == "convergence"
    assert config["base_intervals"] == 40
    assert config["directions"] == [24, 96, 384]
    assert config["refinements"] == [1, 2, 4]
    assert config["source_strides"] == [4, 2, 1]
    assert config["threads"] == 8
    assert config["memory_gb"] == 24
    custom = prepare_run.configuration({"PC_PROFILE": "convergence", "PC_SEEDS": "321",
                                        "PC_THREADS": "2", "PC_MEM": "8GB", "PC_MEMORY_GB": "6"})
    assert custom["seeds"] == [321]
    assert custom["threads"] == 2
    assert custom["memory_gb"] == 6
    with pytest.raises(ValueError, match=r"(?i)profile"):
        prepare_run.configuration({"PC_PROFILE": "unknown"})
    with pytest.raises(ValueError, match="stride"):
        prepare_run.configuration({"PC_PROFILE": "convergence", "PC_SOURCE_STRIDES": "3,2,1"})
    with pytest.raises(ValueError, match="array allocation"):
        prepare_run.configuration({"PC_PROFILE": "convergence", "PC_DIMENSION": "128"})


def test_native_sources_must_match_archive(tmp_path: Path) -> None:
    """A changed native source cannot be admitted by a compatible version string."""
    source = tmp_path / "source"
    native = tmp_path / "native"
    source.mkdir()
    native.mkdir()
    (source / "entry.py").write_text("original\n")
    (native / "entry.py").write_text("changed\n")
    with pytest.raises(ValueError, match=r"entry\.py"):
        prepare_run.native_sources(source, native)


def test_base_fingerprint_records_mismatch_but_scientific_check_rejects_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Relax only source equality during dependency selection, retaining the guard."""
    source = tmp_path / "source"
    archived = source / "src/py21cmfast"
    installed = tmp_path / "installed/py21cmfast"
    archived.mkdir(parents=True)
    installed.mkdir(parents=True)
    (archived / "__init__.py").write_text("new source\n")
    (installed / "__init__.py").write_text("old source\n")
    (installed / "c_21cmfast.so").write_bytes(b"mock binary")
    package = SimpleNamespace(__version__="1", __file__=str(installed / "__init__.py"))
    monkeypatch.setattr(prepare_run.importlib, "import_module", lambda _: package)
    monkeypatch.setattr(prepare_run.shutil, "which", lambda _: sys.executable)
    monkeypatch.setattr(prepare_run.subprocess, "check_output", lambda *_, **__: "12.2\n")
    python = str(Path(sys.executable).absolute())
    fingerprint = prepare_run.runtime_fingerprint(source, python, require_native_match=False)
    assert fingerprint["native_source_matches"] is False
    assert fingerprint["native_source_mismatches"] == ["__init__.py"]
    assert "c_21cmfast.so" in fingerprint["native_footprint"]
    with pytest.raises(ValueError, match="Installed native source differs"):
        prepare_run.runtime_fingerprint(source, python)


def test_prepared_source_changes_are_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Preparation pins scripts, scientific source, runtime and configuration."""
    run = tmp_path / "run"
    (run / "source").mkdir(parents=True)
    (run / "source/one.py").write_text("original\n")
    for name, value in {
        "python_path.txt": "/test/python\n",
        "commit.txt": "a" * 40 + "\n",
        "runtime.env": "PC_MODULES=()\n",
        "submit.pbs": "#PBS\n",
    }.items():
        (run / name).write_text(value)
    monkeypatch.setattr(prepare_run, "runtime_fingerprint", lambda *_, **__: {"test": True, "native_source_matches": True})
    prepare_run.prepare(run, {})
    prepare_run.verify(run, cpus=16)
    with pytest.raises(ValueError, match="CPU"):
        prepare_run.verify(run, cpus=8)
    (run / "source/one.py").write_text("changed\n")
    with pytest.raises(ValueError, match="source"):
        prepare_run.verify(run, cpus=16)


def test_prepared_configuration_changes_are_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A mutated configuration fails before launching any calculation."""
    run = tmp_path / "run"
    (run / "source").mkdir(parents=True)
    (run / "source/one.py").write_text("original\n")
    for name in ("python_path.txt", "commit.txt", "runtime.env", "submit.pbs"):
        (run / name).write_text("frozen\n")
    monkeypatch.setattr(prepare_run, "runtime_fingerprint", lambda *_, **__: {"test": True, "native_source_matches": True})
    prepare_run.prepare(run, {})
    (run / "experiment_config.json").write_text("{}\n")
    with pytest.raises(ValueError, match="experiment_config"):
        prepare_run.verify(run, cpus=16)


def _prepared_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, matches: bool) -> tuple[Path, dict]:
    run = tmp_path / "run"
    (run / "source").mkdir(parents=True)
    (run / "source/one.py").write_text("original\n")
    for name in ("python_path.txt", "commit.txt", "runtime.env", "submit.pbs"):
        (run / name).write_text("frozen\n")
    fingerprint = {"native_source_matches": matches, "native_package": "/existing/py21cmfast"}
    monkeypatch.setattr(prepare_run, "runtime_fingerprint", lambda *_, **__: fingerprint.copy())
    prepare_run.prepare(run, {})
    return run, fingerprint


def test_source_mismatch_schedules_build_but_pins_base_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A stale source may supply dependencies but must not supply scientific fields."""
    run, fingerprint = _prepared_run(tmp_path, monkeypatch, matches=False)
    manifest = prepare_run.verify(run)
    assert manifest["native_rebuild"] is True
    fingerprint["native_package"] = "/changed/runtime"
    with pytest.raises(ValueError, match="runtime or compiler changed"):
        prepare_run.verify(run)


def test_scientific_runtime_registration_is_strict_and_resumable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin matching rebuilt imports once and reject subsequent extension changes."""
    run, fingerprint = _prepared_run(tmp_path, monkeypatch, matches=False)
    fingerprint.update(native_source_matches=True, native_package=str(run / "native_build/source/src/py21cmfast"))
    strict = []

    def capture(*_args: object, require_native_match: bool = True) -> dict:
        strict.append(require_native_match)
        return fingerprint.copy()

    monkeypatch.setattr(prepare_run, "runtime_fingerprint", capture)
    prepare_run.register_scientific_runtime(run)
    recorded = (run / "native_runtime/manifest.json").stat().st_mtime_ns
    prepare_run.register_scientific_runtime(run)
    assert (run / "native_runtime/manifest.json").stat().st_mtime_ns == recorded
    prepare_run.verify_scientific_runtime(run)
    assert strict
    assert all(strict)
    fingerprint["extension_sha256"] = "changed"
    with pytest.raises(ValueError, match="Scientific runtime changed"):
        prepare_run.verify_scientific_runtime(run)


def test_rebuild_cannot_register_base_native_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A scientific process must actually import from the isolated build directory."""
    run, fingerprint = _prepared_run(tmp_path, monkeypatch, matches=False)
    fingerprint["native_source_matches"] = True
    with pytest.raises(ValueError, match="isolated native build"):
        prepare_run.register_scientific_runtime(run)


def test_unregistered_existing_results_are_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Existing fields cannot acquire an unrelated runtime identity after the fact."""
    run, _ = _prepared_run(tmp_path, monkeypatch, matches=True)
    (run / "results").mkdir()
    with pytest.raises(ValueError, match="Existing results"):
        prepare_run.register_scientific_runtime(run)
