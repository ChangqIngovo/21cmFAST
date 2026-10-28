"""Test frozen runtime selection and preparation without native simulations."""

from pathlib import Path

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
    with pytest.raises(ValueError, match="directions"):
        prepare_run.configuration({"PC_DIRECTIONS": "6,7"})
    with pytest.raises(ValueError, match="cfl_values"):
        prepare_run.configuration({"PC_CFL_VALUES": "0.4,0"})
    with pytest.raises(ValueError, match="photon_energy_ev"):
        prepare_run.configuration({"PC_PHOTON_ENERGY_EV": "12"})


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
    monkeypatch.setattr(prepare_run, "runtime_fingerprint", lambda *_: {"test": True})
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
    monkeypatch.setattr(prepare_run, "runtime_fingerprint", lambda *_: {"test": True})
    prepare_run.prepare(run, {})
    (run / "experiment_config.json").write_text("{}\n")
    with pytest.raises(ValueError, match="experiment_config"):
        prepare_run.verify(run, cpus=16)
