"""Postprocessing launcher checks need neither 21cmFAST nor PBS."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from research.pc_coeval import validate_prepare as frozen


def write(path, value="{}\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def archive(path):
    results = path / "results"
    native = path.parent / "native/results/manifest.json"
    write(native, '{"configuration":{"box_len":256}}\n')
    write(results / "identity.json", json.dumps({"input_root": str(native.parent),
          "input_manifest_sha256": frozen.digest(native)}))
    for name in ("coeval_report.json", "report_receipt.json", "COMPLETED",
                 "seed222/z7p5/receipt.json"):
        write(results / name)
    return path


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    input_run = archive(tmp_path / "input")
    run = tmp_path / "validation"
    write(run / "input_run.txt", str(input_run) + "\n")
    for name in ("commit.txt", "python_path.txt", "runtime.env", "submit.pbs"):
        write(run / name)
    write(run / "source/research/pc_coeval/validate.py", "# frozen scientific runner\n")
    monkeypatch.setattr(frozen, "runtime_fingerprint", lambda: {"runtime": "test"})
    manifest = frozen.prepare(run, {})
    return run, input_run, manifest


def test_config_defaults_and_controls():
    config = frozen.configuration({})
    assert config == {"threads": 8, "rays": 100000, "seed": 20261103,
                      "thresholds": [.1, .5, .9], "max_distance_boxes": 1.}
    assert frozen.configuration({"PC_MFP_THRESHOLDS": ".25 .75", "PC_THREADS": "4"})["thresholds"] == [.25, .75]


@pytest.mark.parametrize("key,value", [
    ("PC_THREADS", "0"), ("PC_THREADS", "33"), ("PC_THREADS", "1.5"),
    ("PC_MFP_RAYS", "0"), ("PC_MFP_RAYS", "100000001"),
    ("PC_MFP_SEED", "-1"), ("PC_MFP_SEED", str(2**64)),
    ("PC_MFP_THRESHOLDS", ""), ("PC_MFP_THRESHOLDS", "nan"),
    ("PC_MFP_THRESHOLDS", ".5,.5"), ("PC_MFP_THRESHOLDS", "0"),
    ("PC_MFP_THRESHOLDS", "1"), ("PC_MFP_MAX_DISTANCE_BOXES", "nan"),
    ("PC_MFP_MAX_DISTANCE_BOXES", "0"), ("PC_MFP_MAX_DISTANCE_BOXES", "11"),
])
def test_config_rejects_invalid(key, value):
    with pytest.raises(ValueError):
        frozen.configuration({key: value})


def test_input_run_and_results_paths_are_equivalent(tmp_path):
    input_run = archive(tmp_path / "input")
    assert frozen.result_root(input_run) == input_run / "results"
    assert frozen.input_fingerprint(input_run) == frozen.input_fingerprint(input_run / "results")
    (input_run / "results/COMPLETED").unlink()
    with pytest.raises(ValueError, match="not complete"):
        frozen.input_fingerprint(input_run)


def test_logs_and_outputs_do_not_touch_source_or_input(prepared):
    run, input_run, manifest = prepared
    original = frozen.input_fingerprint(input_run)
    write(run / "pbs_logs/job.output", "PBS output\n")
    write(run / "run_123.log", "live log\n")
    write(run / "results/report.json")
    assert frozen.verify(run, cpus=8) == manifest
    assert frozen.input_fingerprint(input_run) == original
    assert manifest["reruns_simulations"] is False


@pytest.mark.parametrize("operation", ["add", "change", "remove"])
def test_frozen_source_changes_rejected(prepared, operation):
    run, _, _ = prepared
    before = (run / "run_manifest.json").read_bytes()
    path = run / "source/research/pc_coeval/validate.py"
    if operation == "add":
        write(run / "source/accidental.log", "wrong place\n")
    elif operation == "change":
        path.write_text("# changed\n")
    else:
        path.unlink()
    with pytest.raises(ValueError, match="Frozen source changed"):
        frozen.verify(run)
    assert (run / "run_manifest.json").read_bytes() == before


def test_archived_snapshot_receipt_change_rejected(prepared):
    run, input_run, _ = prepared
    write(input_run / "results/seed222/z7p5/receipt.json", "changed")
    with pytest.raises(ValueError, match="input identity changed"):
        frozen.verify(run)


def test_native_box_manifest_change_rejected(prepared):
    run, input_run, _ = prepared
    write(input_run.parent / "native/results/manifest.json", '{"configuration":{"box_len":128}}')
    with pytest.raises(ValueError, match="Native input manifest differs"):
        frozen.verify(run)


def test_allocation_and_runtime_changes_rejected(prepared, monkeypatch):
    run, _, _ = prepared
    with pytest.raises(ValueError, match="CPU allocation"):
        frozen.verify(run, cpus=4)
    monkeypatch.setattr(frozen, "runtime_fingerprint", lambda: {"runtime": "different"})
    with pytest.raises(ValueError, match="Compiler or Python"):
        frozen.verify(run)
    with pytest.raises(FileExistsError, match="fresh validation"):
        frozen.prepare(run, {})


@pytest.mark.parametrize("change", ["missing", "duplicate", "traversal"])
def test_checksum_list_complete_and_constrained(prepared, change):
    run, _, _ = prepared
    path = run / "prepared.sha256"
    rows = path.read_text().splitlines()
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.append(rows[0])
    else:
        rows.append("0" * 64 + "  ../input/results/identity.json")
    path.write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError):
        frozen.verify(run)


def test_nested_output_rejected(tmp_path):
    input_run = archive(tmp_path / "input")
    run = input_run / "validation"
    write(run / "input_run.txt", str(input_run))
    with pytest.raises(ValueError, match="separate"):
        frozen.prepare(run, {})


@pytest.mark.parametrize("latest", [False, True])
def test_real_shell_freeze_and_qsub_keep_original_readonly(tmp_path, latest):
    repo = tmp_path / "repo"
    target = repo / "research/pc_coeval"
    target.mkdir(parents=True)
    for name in ("prepare.py", "validate_prepare.py", "gadi_validate_prepare.sh", "gadi_validate_run.pbs"):
        shutil.copyfile(Path(__file__).with_name(name), target / name)
    write(target / "validate.py", "# runner supplied separately\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-qm", "fixture"], check=True)
    input_run = archive(tmp_path / "runs/coeval_pc_current" if latest else tmp_path / "input")
    if latest:
        old = archive(tmp_path / "runs/coeval_pc_old")
        os.utime(old / "results/COMPLETED", (1, 1))
        incomplete = archive(tmp_path / "runs/coeval_pc_incomplete")
        (incomplete / "results/COMPLETED").unlink()
    wrapper = tmp_path / "python_fixture"
    wrapper.write_text(f"#!{sys.executable}\n" +
        "import os, sys\nsys.path.insert(0, os.getcwd())\n"
        "from research.pc_coeval import validate_prepare\n"
        "validate_prepare.runtime_fingerprint = lambda: {'runtime': 'fixture'}\n"
        "sys.argv = sys.argv[2:]\nvalidate_prepare.main()\n")
    wrapper.chmod(0o755)
    write(input_run / "python_path.txt", str(wrapper) + "\n")
    write(input_run / "runtime.env", "PC_MODULES=()\n")
    input_before = frozen.source_fingerprint(input_run)
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
               PATH=str(fake_bin) + os.pathsep + os.environ["PATH"], PYTHONDONTWRITEBYTECODE="1")
    selection = ["--latest"] if latest else ["--input-run", str(input_run / "results")]
    subprocess.run(["bash", str(target / "gadi_validate_prepare.sh"), *selection,
                    "--submit"], check=True, env=env, capture_output=True, text=True)
    run, = list((tmp_path / "runs").glob("coeval_validation_*"))
    assert (run / "input_run.txt").read_text().strip() == str(input_run / "results")
    before = frozen.source_fingerprint(run / "source")
    subprocess.run(["bash", str(run / "resubmit.sh")], cwd=run / "source", env=env,
                   check=True, capture_output=True, text=True)
    submitted = json.loads(record.read_text())
    assert submitted["cwd"] == str(run)
    for flag in ("-o", "-e"):
        path = Path(submitted["args"][submitted["args"].index(flag) + 1])
        assert path.parent == run / "pbs_logs"
    assert before == frozen.source_fingerprint(run / "source")
    assert before == json.loads((run / "run_manifest.json").read_text())["source_sha256"]
    assert frozen.source_fingerprint(input_run) == input_before
    subprocess.run(["bash", "-n", str(run / "submit.pbs")], check=True)
    subprocess.run(["bash", "-n", str(run / "resubmit.sh")], check=True)
