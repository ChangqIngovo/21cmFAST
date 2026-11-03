"""Launcher tests do not import a native 21cmFAST extension or call PBS."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from research.pc_coeval import prepare as frozen


ORIGINAL = {"dimension": 128, "highres_dimension": 384, "z_start": 25.,
            "z_end": 5., "source_intervals": 160, "threads": 32}


def write(path, content="{}\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def archive(root):
    write(root / "experiment_config.json", json.dumps(ORIGINAL))
    for name in ("commit.txt", "runtime.env", "python_path.txt", "results/manifest.json",
                 "native_runtime/manifest.json", "native_runtime/manifest.sha256",
                 "results/histories/seed222/history_receipt.json",
                 "results/histories/seed222/history_manifest.json"):
        write(root / name)
    (root / "native_build/source/src").mkdir(parents=True)
    # A history need not exist for launcher identity checks: the runner reads
    # selected native snapshot files, never this potentially huge array archive.
    return root


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    input_run = archive(tmp_path / "input")
    run = tmp_path / "run"
    write(run / "input_run.txt", str(input_run) + "\n")
    for name in ("commit.txt", "python_path.txt", "runtime.env", "submit.pbs"):
        write(run / name)
    write(run / "source/research/pc_coeval/run.py", "# Frozen source\n")
    monkeypatch.setattr(frozen, "runtime_fingerprint", lambda path: {"runtime": "test"})
    manifest = frozen.prepare(run, {})
    return run, input_run, manifest


@pytest.mark.parametrize("value", ("", "nan", "inf", "-1", "8,8.0", "hello"))
def test_redshifts_reject_invalid(value):
    with pytest.raises(ValueError):
        frozen.parse_redshifts(value)


def test_config_selects_exact_snapshots_and_independent_thread_count():
    config = frozen.configuration(ORIGINAL, {"PC_REDSHIFTS": "7,8,6.375", "PC_THREADS": "8"})
    assert config["redshifts"] == [8., 7., 6.375]
    assert config["threads"] == 8
    assert config["max_interactions"] == 1_000_000_000_000
    assert frozen.configuration(ORIGINAL, {})["redshifts"] == list(frozen.DEFAULT_REDSHIFTS)
    with pytest.raises(ValueError, match="not an archived"):
        frozen.configuration(ORIGINAL, {"PC_REDSHIFTS": "7.3"})


@pytest.mark.parametrize("key,value", [
    ("PC_THREADS", "0"), ("PC_THREADS", "33"), ("PC_THREADS", "1.5"),
    ("PC_TOLERANCE", "nan"), ("PC_TOLERANCE", "0"), ("PC_TOLERANCE", "1"),
    ("PC_MAX_ITERATIONS", "0"), ("PC_MAX_ITERATIONS", "1000001"),
    ("PC_MAX_INTERACTIONS", "0"), ("PC_MAX_INTERACTIONS", str(2**63)),
    ("PC_SCRIPT_TIMEOUT", "inf"), ("PC_SCRIPT_TIMEOUT", "0"),
])
def test_config_rejects_invalid_controls(key, value):
    with pytest.raises(ValueError):
        frozen.configuration(ORIGINAL, {key: value})


def test_memory_preflight_is_explicit_estimate():
    config = frozen.configuration(ORIGINAL, {})
    estimate = frozen.allocation(ORIGINAL, config, "32GB")
    assert 6 < estimate["estimated_peak_gib"] < 32
    assert not estimate["estimate_is_upper_bound"]
    with pytest.raises(ValueError, match="exceeds"):
        frozen.allocation(ORIGINAL | {"dimension": 512, "highres_dimension": 1536}, config, "8GB")
    with pytest.raises(ValueError, match="positive integer"):
        frozen.allocation(ORIGINAL, config, "32gb")


def test_native_binary_is_checked_against_input_manifest(tmp_path, monkeypatch):
    from research.pc_causal_rt import budget_prepare
    input_run = archive(tmp_path / "input")
    write(input_run / "results/manifest.json",
          json.dumps({"native_runtime": {"native_extension_sha256": "expected"}}))
    monkeypatch.setattr(budget_prepare, "runtime_fingerprint",
                        lambda path: {"native_binary_sha256": "expected"})
    assert frozen.runtime_fingerprint(input_run)["native_binary_sha256"] == "expected"
    monkeypatch.setattr(budget_prepare, "runtime_fingerprint",
                        lambda path: {"native_binary_sha256": "changed"})
    with pytest.raises(ValueError, match="Native binary differs"):
        frozen.runtime_fingerprint(input_run)


def test_external_logs_do_not_change_frozen_source(prepared):
    run, input_run, manifest = prepared
    input_before = frozen.input_fingerprint(input_run)
    write(run / "pbs_logs/job.output", "PBS output\n")
    write(run / "run_123.log", "live log\n")
    write(run / "results/snapshots/seed222_z8/receipt.json")
    assert frozen.verify(run, cpus=8) == manifest
    assert frozen.input_fingerprint(input_run) == input_before
    assert manifest["native_build"] is False
    assert "instantaneous sharp-front" in manifest["scope"]
    assert not any("history.npz" in name for name in manifest["input_sha256"])


@pytest.mark.parametrize("operation,key", [("add", "added"), ("change", "changed"), ("remove", "missing")])
def test_source_change_diagnostics_never_rewrite_hashes(prepared, operation, key):
    run, _, _ = prepared
    manifest_bytes = (run / "run_manifest.json").read_bytes()
    source = run / "source/research/pc_coeval/run.py"
    if operation == "add":
        write(run / "source/accidental.pbs.log", "This must not be hidden\n")
    elif operation == "change":
        source.write_text("# changed\n")
    else:
        source.unlink()
    with pytest.raises(ValueError, match="Frozen source changed") as error:
        frozen.verify(run)
    diagnostics = json.loads(str(error.value).split(": ", 1)[1])
    assert diagnostics[key]
    assert (run / "run_manifest.json").read_bytes() == manifest_bytes


def test_input_receipt_mutation_is_rejected(prepared):
    run, input_run, _ = prepared
    write(input_run / "results/histories/seed222/history_receipt.json", '{"changed":true}')
    with pytest.raises(ValueError, match="Archived input identity changed"):
        frozen.verify(run)


def test_runtime_mutation_is_rejected(prepared, monkeypatch):
    run, _, _ = prepared
    monkeypatch.setattr(frozen, "runtime_fingerprint", lambda path: {"runtime": "different"})
    with pytest.raises(ValueError, match="native binary"):
        frozen.verify(run)


def test_cpu_change_and_reprepare_rejected(prepared):
    run, _, _ = prepared
    with pytest.raises(ValueError, match="CPU allocation"):
        frozen.verify(run, cpus=16)
    with pytest.raises(FileExistsError, match="fresh run"):
        frozen.prepare(run, {})


@pytest.mark.parametrize("change", ("missing", "duplicate", "traversal"))
def test_checksum_list_cannot_omit_or_add_entries(prepared, change):
    run, _, _ = prepared
    path = run / "prepared.sha256"
    rows = path.read_text().splitlines()
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.append(rows[0])
    else:
        rows.append("0" * 64 + "  ../input/commit.txt")
    path.write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError):
        frozen.verify(run)


def test_nested_output_is_rejected(tmp_path):
    input_run = archive(tmp_path / "input")
    run = input_run / "output"
    write(run / "input_run.txt", str(input_run))
    with pytest.raises(ValueError, match="separate"):
        frozen.prepare(run, {})


def test_shell_freeze_and_resubmit_keep_all_logs_outside_source(tmp_path):
    """Exercise real git archive and shell argument quoting with a fake qsub."""
    repo = tmp_path / "repo"
    target = repo / "research/pc_coeval"
    target.mkdir(parents=True)
    for name in ("prepare.py", "gadi_prepare.sh", "gadi_run.pbs"):
        shutil.copyfile(Path(__file__).with_name(name), target / name)
    write(target / "run.py", "# scientific runner supplied separately\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c",
                    "user.email=test@example.invalid", "commit", "-qm", "fixture"], check=True)
    input_run = archive(tmp_path / "input")
    wrapper = tmp_path / "python_fixture"
    wrapper.write_text(f"#!{sys.executable}\n" +
        "import os, sys\nsys.path.insert(0, os.getcwd())\n"
        "from research.pc_coeval import prepare\n"
        "prepare.runtime_fingerprint = lambda path: {'runtime': 'fixture'}\n"
        "sys.argv = sys.argv[2:]\nprepare.main()\n")
    wrapper.chmod(0o755)
    write(input_run / "python_path.txt", str(wrapper) + "\n")
    write(input_run / "runtime.env", "PC_MODULES=()\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    qsub = fake_bin / "qsub"
    qsub.write_text(f"#!{sys.executable}\n" +
        "import json, os, sys\nfrom pathlib import Path\n"
        "Path(os.environ['QSUB_RECORD']).write_text(json.dumps({'cwd':os.getcwd(),'args':sys.argv[1:]}))\n"
        "for key in ('-o','-e'):\n"
        "    Path(sys.argv[sys.argv.index(key)+1]).write_text('PBS log outside source\\n')\n"
        "print('123.gadi-pbs')\n")
    qsub.chmod(0o755)
    record = tmp_path / "qsub.json"
    env = {key: value for key, value in os.environ.items() if not key.startswith("PC_")}
    env.update(PC_RUN_ROOT=str(tmp_path / "runs"), QSUB_RECORD=str(record),
               PATH=str(fake_bin) + os.pathsep + os.environ["PATH"],
               PYTHONDONTWRITEBYTECODE="1")
    subprocess.run(["bash", str(target / "gadi_prepare.sh"), "--input-run", str(input_run),
                    "--submit"], check=True, env=env, capture_output=True, text=True)
    run, = list((tmp_path / "runs").iterdir())
    before = frozen.source_fingerprint(run / "source")
    # Retry from within frozen source: the generated script must still cd run.
    subprocess.run(["bash", str(run / "resubmit.sh")], cwd=run / "source", env=env,
                   check=True, capture_output=True, text=True)
    submitted = json.loads(record.read_text())
    assert submitted["cwd"] == str(run)
    for flag in ("-o", "-e"):
        path = Path(submitted["args"][submitted["args"].index(flag) + 1])
        assert path.parent == run / "pbs_logs"
        assert path.is_file()
    assert before == frozen.source_fingerprint(run / "source")
    assert before == json.loads((run / "run_manifest.json").read_text())["source_sha256"]
    subprocess.run(["bash", "-n", str(run / "submit.pbs")], check=True)
    subprocess.run(["bash", "-n", str(run / "resubmit.sh")], check=True)
