"""Check immutable preparation of analytic-teacher training jobs."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.pc_analytic_emulator import prepare


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _completed(root: Path, name: str, *, samples: int = 56) -> Path:
    path = root / name
    (path / "results").mkdir(parents=True)
    (path / "COMPLETED").touch()
    (path / "results/COMPLETED").touch()
    _json(
        path / "results/summary.json",
        {
            "complete": True,
            "all_photon_ledgers_passed": True,
            "completed_snapshots": samples,
            "requested_snapshots": samples,
        },
    )
    _json(
        path / "results/dataset_manifest.json",
        {
            "schema": 1,
            "complete": True,
            "teacher_method": "analytic-column-characteristic",
            "identity": {
                "configuration": {
                    "seeds": list(range(1001, 1009)),
                    "redshifts": [10, 9, 8, 7.5, 7, 6.5, 6],
                    "dimension": 32,
                    "box_len": 256,
                }
            },
        },
    )
    return path


def test_latest_uses_timestamp_and_finished_campaign(tmp_path):
    """Select by creation timestamp, not commit prefix or incomplete output."""
    _completed(tmp_path, "analytic_teacher_ffffffff00_20261101T010101_aaaa")
    newest = _completed(tmp_path, "analytic_teacher_1111111111_20261106T010101_bbbb")
    _completed(tmp_path, "analytic_teacher_ffffffff00_20261107T010101_cccc", samples=2)
    incomplete = _completed(
        tmp_path, "analytic_teacher_ffffffff00_20261108T010101_dddd"
    )
    (incomplete / "COMPLETED").unlink()
    assert prepare.latest_teacher(tmp_path) == newest.resolve()


def test_latest_refuses_malformed_completed_candidate(tmp_path):
    """A corrupt completed candidate cannot silently cause use of older data."""
    _completed(tmp_path, "analytic_teacher_ffffffff00_20261101T010101_aaaa")
    newer = _completed(tmp_path, "analytic_teacher_1111111111_20261106T010101_bbbb")
    (newer / "results/summary.json").write_text("broken JSON")
    with pytest.raises(ValueError, match="Cannot inspect completed teacher"):
        prepare.latest_teacher(tmp_path)


def test_latest_requires_a_matching_campaign(tmp_path):
    """A small unrelated smoke campaign is not selected as the 56-label run."""
    _completed(tmp_path, "analytic_teacher_ffffffff00_20261101T010101_aaaa", samples=2)
    with pytest.raises(FileNotFoundError, match="56-snapshot"):
        prepare.latest_teacher(tmp_path)


def test_configuration_defaults(tmp_path):
    """Default training groups keep all redshifts of each seed together."""
    config = prepare.configuration(tmp_path, {})
    assert config["train_seeds"] == list(range(1001, 1007))
    assert config["validation_seeds"] == [1007]
    assert config["test_seeds"] == [1008]
    assert config["model_seeds"] == [11, 23, 47, 83]
    assert config["epochs"] == 200
    assert config["workers"] * config["threads"] == config["cpus"] == 16
    assert config["samples_per_snapshot"] == 8192


@pytest.mark.parametrize(
    "environment",
    [
        {"PC_WORKERS": "0"},
        {"PC_THREADS": "0"},
        {"PC_NCPUS": "15"},
        {"PC_EMULATOR_EPOCHS": "0"},
        {"PC_SAMPLES_PER_SNAPSHOT": "0"},
        {"PC_MODEL_SEEDS": "11,11"},
        {"PC_TRAIN_SEEDS": "1001,1001"},
        {"PC_VALIDATION_SEEDS": "-1"},
    ],
)
def test_configuration_rejects_invalid_controls(tmp_path, environment):
    """Reject malformed allocations and sampling before any fit starts."""
    with pytest.raises(ValueError, match=r"requires|positive|must cover"):
        prepare.configuration(tmp_path, environment)


def test_configuration_explicit_cpu_split(tmp_path):
    """An explicit smaller test run receives matching CPU and seed controls."""
    config = prepare.configuration(
        tmp_path,
        {
            "PC_WORKERS": "2",
            "PC_THREADS": "3",
            "PC_EMULATOR_EPOCHS": "5",
            "PC_TRAIN_SEEDS": "10 20",
            "PC_VALIDATION_SEEDS": "30",
            "PC_TEST_SEEDS": "40",
            "PC_MODEL_SEEDS": "0,1",
        },
    )
    assert config["cpus"] == 6
    assert config["train_seeds"] == [10, 20]
    assert config["epochs"] == 5


def _prepared_fixture(tmp_path, monkeypatch):
    """Install a minimal audited-data stand-in for preparation unit checks."""
    teacher = tmp_path / "teacher"
    teacher.mkdir()
    results = teacher / "results"
    results.mkdir()
    (results / "fields.npz").write_bytes(b"audited teacher bytes")
    dataset = SimpleNamespace(
        root=results.resolve(),
        file_hashes={
            str(results / "fields.npz"): prepare._digest(results / "fields.npz")
        },
        identity={"configuration": {"seeds": list(range(1001, 1009))}},
        physics={"scope": "test"},
        audit={"all_photon_ledgers_passed": True},
        samples=[
            SimpleNamespace(seed=seed, redshift=7.5) for seed in range(1001, 1009)
        ],
    )
    monkeypatch.setattr(prepare, "_load_dataset", lambda path: dataset)
    monkeypatch.setattr(
        prepare,
        "_split_dataset",
        lambda data, config: {
            key: tuple(
                sample
                for sample in data.samples
                if sample.seed in config[f"{key}_seeds"]
            )
            for key in ("train", "validation", "test")
        },
    )
    monkeypatch.setattr(prepare, "_validated_config", lambda config: config)
    runtime = {"python": "/test/python", "versions": {"numpy": "test"}}
    monkeypatch.setattr(prepare, "_runtime", lambda: runtime.copy())
    run = tmp_path / "training"
    (run / "source").mkdir(parents=True)
    (run / "source/code.py").write_text("pass\n")
    for name, value in {
        "commit.txt": "abc123\n",
        "teacher_run.txt": str(teacher),
        "python_path.txt": "/test/python",
        "runtime.env": "PC_MODULES=()\n",
        "submit.pbs": "#!/bin/bash\n",
    }.items():
        (run / name).write_text(value)
    return run, dataset, runtime


def test_prepare_freezes_inputs_and_split(tmp_path, monkeypatch):
    """Preparation records source/data identity without running simulations."""
    run, dataset, _ = _prepared_fixture(tmp_path, monkeypatch)
    manifest = prepare.prepare(run, {})
    assert manifest["split_counts"] == {"train": 6, "validation": 1, "test": 1}
    assert manifest["input_sha256"] == dataset.file_hashes
    assert (run / "PREPARED").is_file()
    assert prepare.verify(run, 16) == manifest
    assert not (run / "results").exists()


@pytest.mark.parametrize(
    "name", ["source/code.py", "experiment_config.json", "runtime.env"]
)
def test_verify_rejects_frozen_mutation(tmp_path, monkeypatch, name):
    """A changed frozen source or configuration fails before compute."""
    run, _, _ = _prepared_fixture(tmp_path, monkeypatch)
    prepare.prepare(run, {})
    with (run / name).open("a") as stream:
        stream.write("\nchanged\n")
    with pytest.raises(ValueError, match="changed"):
        prepare.verify(run, 16)


def test_verify_rejects_teacher_mutation(tmp_path, monkeypatch):
    """Teacher arrays remain immutable between preparation and fitting."""
    run, dataset, _ = _prepared_fixture(tmp_path, monkeypatch)
    prepare.prepare(run, {})
    Path(next(iter(dataset.file_hashes))).write_bytes(b"different teacher")
    with pytest.raises(ValueError, match="Teacher input changed"):
        prepare.verify(run, 16)


def test_verify_rejects_runtime_and_cpu_mutation(tmp_path, monkeypatch):
    """The selected interpreter/dependency versions and CPU allocation are pinned."""
    run, _, runtime = _prepared_fixture(tmp_path, monkeypatch)
    prepare.prepare(run, {})
    with pytest.raises(ValueError, match="CPU allocation"):
        prepare.verify(run, 8)
    runtime["python"] = "/different/python"
    with pytest.raises(ValueError, match="runtime changed"):
        prepare.verify(run, 16)


def test_existing_preparation_is_not_overwritten(tmp_path, monkeypatch):
    """Resubmission verifies an existing preparation instead of rewriting it."""
    run, _, _ = _prepared_fixture(tmp_path, monkeypatch)
    prepare.prepare(run, {})
    with pytest.raises(FileExistsError):
        prepare.prepare(run, {})


def test_pbs_refuses_multiple_nodes(tmp_path):
    """Process workers cannot silently share an allocation across nodes."""
    script = Path(__file__).with_name("gadi_train.pbs")
    result = subprocess.run(
        ["bash", str(script)],
        env={
            "PATH": "/usr/bin:/bin",
            "PC_ANALYTIC_TRAIN_RUN": str(tmp_path),
            "PBS_JOBID": "test",
            "PBS_NCPUS": "16",
            "PBS_NNODES": "2",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "single-node" in result.stderr


def test_latest_preflights_selected_python(tmp_path):
    """Discovery uses PC_PYTHON and reports how to load the known Gadi runtime."""
    incompatible = tmp_path / "old_python"
    incompatible.write_text("#!/bin/sh\nexit 1\n")
    incompatible.chmod(0o755)
    result = subprocess.run(
        ["bash", str(Path(__file__).with_name("gadi_prepare.sh")), "--latest-teacher"],
        env={
            "PATH": "/usr/bin:/bin",
            "PC_PYTHON": str(incompatible),
            "PC_RUN_ROOT": str(tmp_path),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "module load python3/3.12.1" in result.stderr
