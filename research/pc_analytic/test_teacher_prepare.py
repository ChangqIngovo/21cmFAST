"""Check immutable preparation of parallel analytic teacher campaigns."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from research.pc_analytic import teacher_prepare as prepare


def _template() -> dict:
    return {
        "random_seed": 222,
        "node_redshifts": [],
        "cosmo_params": {"OMm": 0.31},
        "simulation_options": {"HII_DIM": 128, "DIM": 384, "BOX_LEN": 256},
        "matter_options": {"SOURCE_MODEL": "L-INTEGRAL"},
        "astro_params": {"R_BUBBLE_MAX": 50},
        "astro_options": {
            "RECOMB_MODEL": "none",
            "CELL_RECOMB": False,
            "USE_TS_FLUCT": False,
            "USE_MCGS": False,
            "USE_REIONIZATION_PHOTOHEATING_FEEDBACK": False,
            "PHOTON_CONS_TYPE": "no-photoncons",
        },
    }


def _parent(root: Path) -> Path:
    root.mkdir()
    for name in ("experiment_config.json", "runtime.env", "python_path.txt"):
        (root / name).write_text("{}\n")
    results = root / "results"
    history = results / "histories/seed222"
    snapshot = history / "native/seed222/z7p5"
    snapshot.mkdir(parents=True)
    (results / "manifest.json").write_text("{}\n")
    identity = {"inputs": _template(), "extension_sha256": "native", "redshift": 7.5}
    (snapshot / "native.json").write_text(json.dumps({"identity": identity}))
    for name in ("native.h5", "source.npz"):
        (snapshot / name).write_text("fixture\n")
    receipt = {
        "identity": identity,
        "record": {"seed": 222, "redshift": 7.5},
        "files": {
            name: prepare._digest(snapshot / name)
            for name in ("native.json", "native.h5", "source.npz")
        },
    }
    (snapshot / "receipt.json").write_text(json.dumps(receipt))
    (history / "history_manifest.json").write_text('{"seed":222}\n')
    (history / "history_receipt.json").write_text(
        json.dumps(
            {
                "identity": {"seed": 222},
                "files": {
                    name: prepare._digest(history / name)
                    for name in (
                        "history_manifest.json",
                        "native/seed222/z7p5/receipt.json",
                    )
                },
            }
        )
    )
    return root


def test_default_parallel_resources(tmp_path: Path) -> None:
    """Make eight independent seeds without multiplying threads beyond allocation."""
    config = prepare.configuration(tmp_path / "native_template.json", {})
    assert config["seeds"] == list(range(1001, 1009))
    assert config["dimension"] == 32
    assert config["highres_dimension"] == 256
    assert config["box_len"] == 256
    assert config["workers"] == 4
    assert config["threads"] == 4
    assert config["cpus"] == 16
    assert "c2ray" not in config
    assert "script" not in config


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ({"PC_BOX_LEN": "300.5"}, 301),
        ({"PC_DIMENSION": "128"}, 384),
        ({"PC_HIGHRES_DIMENSION": "384"}, 384),
    ],
)
def test_density_grid_default(tmp_path: Path, environment: dict, expected: int) -> None:
    """Keep default density cells <=1 cMpc without changing physics controls."""
    config = prepare.configuration(tmp_path / "native_template.json", environment)
    assert config["highres_dimension"] == expected


@pytest.mark.parametrize(
    "environment",
    [
        {"PC_WORKERS": "5", "PC_NCPUS": "16"},
        {"PC_THREADS": "0"},
        {"PC_NCPUS": "8"},
        {"PC_SEEDS": "1001,1001"},
        {"PC_SEEDS": "0"},
        {"PC_REDSHIFTS": "8,nan"},
        {"PC_BOX_LEN": "inf"},
        {"PC_DIMENSION": "1"},
        {"PC_HIGHRES_DIMENSION": "8"},
        {"PC_DIRECTIONS": "12"},
        {"PC_SIGMA_CM2": "-1"},
    ],
)
def test_invalid_configuration_rejected(tmp_path: Path, environment: dict) -> None:
    """Reject oversubscription and malformed physical settings before qsub."""
    with pytest.raises(ValueError, match=r"PC_|analytic"):
        prepare.configuration(tmp_path / "native_template.json", environment)


def test_memory_accounts_for_concurrent_seeds(tmp_path: Path) -> None:
    """Bound concurrent arrays, not only a single solver instance."""
    one = prepare.configuration(tmp_path / "native_template.json", {"PC_WORKERS": "1"})
    four = prepare.configuration(tmp_path / "native_template.json", {})
    a = prepare.resource_estimate(one, {})
    b = prepare.resource_estimate(four, {})
    assert b["estimated_concurrent_gib"] == pytest.approx(
        4 * a["estimated_concurrent_gib"]
    )
    with pytest.raises(ValueError, match="memory"):
        prepare.resource_estimate(four, {"PC_MEMORY_GB": "0.1"})
    with pytest.raises(ValueError, match="memory"):
        prepare.resource_estimate(four, {"PC_MEMORY_GB": "64", "PC_MEM": "64GB"})


def test_template_follows_receipt_chain(tmp_path: Path) -> None:
    """Copy all native physics from one verified archival snapshot."""
    parent = _parent(tmp_path / "parent")
    template, info = prepare.select_template(parent, 222, 7.5)
    assert template == _template()
    assert info["native_extension_sha256"] == "native"
    assert info["input_sha256"]
    source = parent / "results/histories/seed222/native/seed222/z7p5/source.npz"
    source.write_text("changed\n")
    with pytest.raises(ValueError, match="checksum"):
        prepare.select_template(parent, 222, 7.5)


def _frozen_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    parent = _parent(tmp_path / "parent")
    run = tmp_path / "run"
    run.mkdir()
    (run / "source").mkdir()
    (run / "source/solver.py").write_text("pass\n")
    (run / "parent_run.txt").write_text(str(parent))
    for name in ("commit.txt", "runtime.env", "python_path.txt", "submit.pbs"):
        (run / name).write_text("fixture\n")
    monkeypatch.setattr(
        prepare,
        "_runtime",
        lambda _parent: {
            "native_extension_sha256": "native",
            "CXX": {"path": "/bin/g++"},
        },
    )
    monkeypatch.setattr(prepare, "_preflight", lambda _template, config: config)
    prepare.prepare(run, {})
    return run, parent


def test_frozen_run_and_result_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Allow resumable results while keeping inputs and numerical settings frozen."""
    run, _ = _frozen_run(tmp_path, monkeypatch)
    manifest = prepare.verify(run, cpus=16)
    assert manifest["configuration"]["native_template_sha256"] == prepare._digest(
        run / "native_template.json"
    )
    assert manifest["network_required_on_compute"] is False
    (run / "results").mkdir()
    (run / "results/progress.json").write_text("{}\n")
    assert prepare.verify(run, cpus=16) == manifest
    with pytest.raises(ValueError, match="CPU allocation"):
        prepare.verify(run, cpus=8)
    with pytest.raises(FileExistsError, match="fresh"):
        prepare.prepare(run, {})


@pytest.mark.parametrize(
    "target", ["source/solver.py", "native_template.json", "experiment_config.json"]
)
def test_frozen_mutations_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    """Never accept silently edited frozen source or physical input templates."""
    run, _ = _frozen_run(tmp_path, monkeypatch)
    (run / target).write_text("changed\n")
    with pytest.raises(ValueError, match="changed"):
        prepare.verify(run)


def test_parent_input_mutation_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reused runtime parent remains part of the frozen provenance."""
    run, parent = _frozen_run(tmp_path, monkeypatch)
    (parent / "runtime.env").write_text("changed\n")
    with pytest.raises(ValueError, match="Archived parent"):
        prepare.verify(run)


def test_native_runtime_mutation_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refuse a different native binary before resumed simulations start."""
    run, _ = _frozen_run(tmp_path, monkeypatch)
    monkeypatch.setattr(
        prepare,
        "_runtime",
        lambda _parent: {
            "native_extension_sha256": "changed",
            "CXX": {"path": "/bin/g++"},
        },
    )
    with pytest.raises(ValueError, match="runtime or compiler changed"):
        prepare.verify(run)


def test_multi_node_allocation_rejected(tmp_path: Path) -> None:
    """The process pool must not silently leave extra PBS nodes unused."""
    completed = subprocess.run(
        ["bash", str(Path(__file__).with_name("gadi_teacher_run.pbs"))],
        env=os.environ
        | {
            "PC_TEACHER_RUN": str(tmp_path),
            "PBS_JOBID": "fixture",
            "PBS_NCPUS": "32",
            "PBS_NNODES": "2",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "single-node allocation" in completed.stderr
    assert not list(tmp_path.iterdir())


def test_shell_scripts_parse() -> None:
    """Validate the launch scripts without contacting PBS."""
    root = Path(__file__).parent
    subprocess.run(["bash", "-n", str(root / "gadi_teacher_prepare.sh")], check=True)
    subprocess.run(["bash", "-n", str(root / "gadi_teacher_run.pbs")], check=True)
