"""Check analytic comparison preparation without PBS or a native runtime."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.pc_analytic import prepare


def test_default_configuration_is_an_explicit_small_pilot() -> None:
    """Reject large source grids unless coarsening was explicitly requested."""
    config = prepare.configuration({"dimension": 8}, {})
    assert config["redshifts"] == [8.0, 7.5, 7.0]
    assert config["require_c2ray"] is True
    assert config["coarsen_to"] is None
    assert config["analytic"]["boundary"] == "source_periodic_cube"
    assert config["analytic"]["transport_mode"] == "characteristic"
    assert "transport_fraction" not in config["analytic"]
    assert config["c2ray"]["subbox_size"] == 2
    assert config["c2ray"]["loss_fraction"] == 1e-10
    assert config["c2ray"]["timeout"] == 1800
    assert config["script_timeout_seconds"] == 900
    with pytest.raises(ValueError, match="PC_COARSEN_TO"):
        prepare.configuration({"dimension": 128}, {})
    coarse = prepare.configuration({"dimension": 128}, {"PC_COARSEN_TO": "16"})
    assert coarse["coarsen_to"] == 16
    assert coarse["input_dimension"] == 128


@pytest.mark.parametrize("dimension", [32, 64, 128])
def test_large_grid_requires_explicit_selection(dimension: int) -> None:
    """Allow deliberate refinement without silently launching the native grid."""
    with pytest.raises(ValueError, match="PC_COARSEN_TO"):
        prepare.configuration({"dimension": dimension}, {})
    config = prepare.configuration(
        {"dimension": 128}, {"PC_COARSEN_TO": str(dimension)}
    )
    assert config["input_dimension"] == 128
    assert config["output_dimension"] == dimension
    assert config["coarsen_to"] == dimension
    pilot = prepare.configuration({"dimension": 128}, {"PC_COARSEN_TO": "16"})
    for key in ("analytic", "c2ray", "sigma_cm2", "seeds", "redshifts", "brightness"):
        assert config[key] == pilot[key]


def test_empty_grid_override_does_not_enable_a_large_run() -> None:
    """An exported but empty variable is not an explicit grid selection."""
    with pytest.raises(ValueError, match="PC_COARSEN_TO"):
        prepare.configuration({"dimension": 128}, {"PC_COARSEN_TO": ""})


@pytest.mark.parametrize("key", ["PC_C2_TIMEOUT", "PC_SCRIPT_TIMEOUT"])
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf"])
def test_nonpositive_or_nonfinite_backend_timeouts_are_rejected(
    key: str, value: str
) -> None:
    """Reject unusable subprocess limits before creating a compute job."""
    with pytest.raises(ValueError, match=key):
        prepare.configuration({"dimension": 8}, {key: value})


def test_backend_timeout_overrides_are_recorded() -> None:
    """Give larger-grid backends explicit runtime budgets without changing physics."""
    config = prepare.configuration(
        {"dimension": 128},
        {
            "PC_COARSEN_TO": "32",
            "PC_C2_TIMEOUT": "72000",
            "PC_SCRIPT_TIMEOUT": "1200.5",
        },
    )
    assert config["c2ray"]["timeout"] == 72000
    assert config["script_timeout_seconds"] == 1200.5


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("PC_THREADS", "0"),
        ("PC_THREADS", "33"),
        ("PC_REDSHIFTS", "nan"),
        ("PC_REDSHIFTS", "8,8"),
        ("PC_EXPOSURE_STEPS", "0"),
        ("PC_SIGMA_CM2", "0"),
        ("PC_SIGMA_CM2", "nan"),
        ("PC_C2_TOLERANCE", "nan"),
        ("PC_C2_TOLERANCE", "0"),
        ("PC_C2_MAX_ITERATIONS", "0"),
        ("PC_C2_SUBBOX_SIZE", "0"),
        ("PC_C2_LOSS_FRACTION", "nan"),
        ("PC_C2_LOSS_FRACTION", "1"),
        ("PC_BOUNDARY", "reflecting"),
        ("PC_COARSEN_TO", "3"),
        ("PC_COARSEN_TO", "32"),
    ],
)
def test_invalid_configuration_is_rejected(key: str, value: str) -> None:
    """Reject controls before preparing or submitting any jobs."""
    with pytest.raises(ValueError, match=r"PC_|Invalid"):
        prepare.configuration({"dimension": 8}, {key: value})


def test_seed_selection_is_explicit() -> None:
    """Keep all seeds by default and support a unique integer subset."""
    assert prepare.configuration({"dimension": 8}, {})["seeds"] is None
    config = prepare.configuration({"dimension": 8}, {"PC_SEEDS": "222"})
    assert config["seeds"] == [222]
    with pytest.raises(ValueError, match="PC_SEEDS"):
        prepare.configuration({"dimension": 8}, {"PC_SEEDS": "222,222"})


@pytest.mark.parametrize("boundary", ["periodic", "open", "source_periodic_cube"])
def test_boundary_override_is_recorded(boundary: str) -> None:
    """Retain the declared transport boundary without changing SCRIPT's kernel."""
    config = prepare.configuration({"dimension": 8}, {"PC_BOUNDARY": boundary})
    assert config["analytic"]["boundary"] == boundary


def test_full_cube_c2ray_control_is_explicit() -> None:
    """Allow the no-early-stop C²-Ray control without adjusting photon budgets."""
    config = prepare.configuration(
        {"dimension": 8},
        {"PC_C2_SUBBOX_SIZE": "8", "PC_C2_LOSS_FRACTION": "0"},
    )
    assert config["c2ray"]["subbox_size"] == 8
    assert config["c2ray"]["loss_fraction"] == 0


def test_tree_hashes_detect_added_modified_and_removed_files(tmp_path: Path) -> None:
    """Freeze all files while permitting results outside the source directory."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "solver.py").write_text("pass\n")
    before = prepare.tree_hashes(source)
    (tmp_path / "run.log").write_text("outside source\n")
    assert prepare.tree_hashes(source) == before
    (source / "solver.py").write_text("changed\n")
    assert prepare.tree_hashes(source) != before
    (source / "other.txt").write_text("added\n")
    assert set(prepare.tree_hashes(source)) == {"solver.py", "other.txt"}


def test_checkpoint_files_are_not_silently_reprepared(tmp_path: Path) -> None:
    """An existing manifest is immutable even if a previous run failed."""
    (tmp_path / "run_manifest.json").write_text(json.dumps({"already": "prepared"}))
    with pytest.raises(FileExistsError, match="fresh"):
        prepare.prepare(tmp_path, {})


def _frozen_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    dimension: int = 8,
    environment: dict[str, str] | None = None,
) -> tuple[Path, Path]:
    """Prepare a real frozen manifest with isolated runtime and input fixtures."""
    run, original = tmp_path / "run", tmp_path / "input"
    run.mkdir()
    original.mkdir()
    (original / "experiment_config.json").write_text(
        json.dumps({"dimension": dimension})
    )
    data = original / "input.npz"
    data.write_bytes(b"fixture arrays")
    for name in ("commit.txt", "runtime.env", "python_path.txt", "submit.pbs"):
        (run / name).write_text("fixture\n")
    (run / "input_run.txt").write_text(str(original))
    (run / "source").mkdir()
    (run / "source" / "solver.py").write_text("# frozen source\n")
    (run / "c2ray_upstream").mkdir()
    (run / "c2ray_upstream" / "LICENSE").write_text("fixture upstream\n")
    monkeypatch.setattr(
        prepare,
        "_runtime",
        lambda: {"compilers": {"SCRIPT_FC": {"path": "fixture-fortran"}}},
    )
    monkeypatch.setattr(prepare, "_prepare_upstream", lambda *_args: None)
    monkeypatch.setattr(
        prepare,
        "_selected_inputs",
        lambda *_args: ([str(original)], {str(data): prepare._digest(data)}),
    )
    prepare.prepare(run, environment or {})
    return run, data


def test_results_and_logs_do_not_change_frozen_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retry may write outputs, never mutate the immutable source tree."""
    run, _ = _frozen_run(tmp_path, monkeypatch)
    (run / "run.log").write_text("retry log\n")
    (run / "results").mkdir()
    assert prepare.verify(run, cpus=8)["network_required_on_compute"] is False
    with pytest.raises(ValueError, match="CPU allocation"):
        prepare.verify(run, cpus=4)


@pytest.mark.parametrize("area", ["source", "c2ray_upstream"])
def test_frozen_tree_mutation_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, area: str
) -> None:
    """Any added, changed or missing scientific file invalidates the receipt."""
    run, _ = _frozen_run(tmp_path, monkeypatch)
    (run / area / "accidental.log").write_text("change\n")
    with pytest.raises(ValueError, match="Frozen"):
        prepare.verify(run)


def test_native_input_mutation_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail before computation when a selected archived array changes."""
    run, data = _frozen_run(tmp_path, monkeypatch)
    data.write_bytes(b"different arrays")
    with pytest.raises(ValueError, match="Archived native input"):
        prepare.verify(run)


def test_metadata_mutation_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing a prepared solver setting requires a fresh experiment."""
    run, _ = _frozen_run(tmp_path, monkeypatch)
    (run / "experiment_config.json").write_text("{}\n")
    with pytest.raises(ValueError, match="Prepared metadata"):
        prepare.verify(run)


def test_grid_and_timeouts_are_frozen_with_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify recorded overrides and reject changes to a prepared timeout."""
    run, _ = _frozen_run(
        tmp_path,
        monkeypatch,
        dimension=128,
        environment={
            "PC_COARSEN_TO": "32",
            "PC_C2_TIMEOUT": "72000",
            "PC_SCRIPT_TIMEOUT": "1200",
        },
    )
    config_path = run / "experiment_config.json"
    config = json.loads(config_path.read_text())
    assert prepare.verify(run, cpus=8)["configuration"] == config
    assert config["output_dimension"] == 32
    assert config["c2ray"]["timeout"] == 72000
    assert config["script_timeout_seconds"] == 1200
    config["c2ray"]["timeout"] = 1800
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Prepared metadata"):
        prepare.verify(run)


@pytest.mark.parametrize("select_results", [False, True])
def test_grid_reference_is_resolved_and_frozen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, select_results: bool
) -> None:
    """Pin the completed coarse comparison before launching a finer run."""
    reference = tmp_path / "coarse"
    results = reference / "results"
    results.mkdir(parents=True)
    for name in ("COMPLETED", "comparison_report.json", "identity.json"):
        (results / name).write_text("{}\n")
    selected = results if select_results else reference
    run, _ = _frozen_run(
        tmp_path,
        monkeypatch,
        environment={"PC_GRID_REFERENCE_RUN": str(selected)},
    )
    config = prepare.verify(run)["configuration"]
    assert config["grid_reference_run"] == str(results.resolve())
    assert set(config["grid_reference_sha256"]) == {
        "COMPLETED",
        "comparison_report.json",
        "identity.json",
    }
    (results / "comparison_report.json").write_text('{"changed": true}\n')
    with pytest.raises(ValueError, match="Grid reference"):
        prepare.verify(run)


@pytest.mark.parametrize(
    "missing", ["COMPLETED", "comparison_report.json", "identity.json"]
)
def test_incomplete_grid_reference_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    """A fine-grid campaign cannot start from an incomplete coarse result."""
    reference = tmp_path / "coarse" / "results"
    reference.mkdir(parents=True)
    for name in ("COMPLETED", "comparison_report.json", "identity.json"):
        if name != missing:
            (reference / name).write_text("{}\n")
    with pytest.raises(ValueError, match="PC_GRID_REFERENCE_RUN"):
        _frozen_run(
            tmp_path,
            monkeypatch,
            environment={"PC_GRID_REFERENCE_RUN": str(reference)},
        )


def test_shell_syntax() -> None:
    """The submitted scripts are valid Bash without contacting a scheduler."""
    import subprocess

    for name in ("gadi_prepare.sh", "gadi_run.pbs"):
        subprocess.run(["bash", "-n", str(Path(__file__).with_name(name))], check=True)
