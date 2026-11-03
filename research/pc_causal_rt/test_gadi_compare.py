"""Exercise campaign submission and failure handling without PBS or native runs."""

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
CHILD_SUBMIT = """#!/usr/bin/env bash
set -euo pipefail
printf 'called\\n' > "$PC_CAUSAL_RT_RUN/child_called"
exit "${TEST_CHILD_RUN_EXIT:-7}"
"""


@pytest.fixture
def campaign_tools(tmp_path):
    """Use an isolated committed repository, fake preparer and recording qsub."""
    for command in ("bash", "git", "python3", "sha256sum", "flock", "realpath"):
        if shutil.which(command) is None:
            pytest.skip(f"Campaign shell tests require {command}")
    repo = tmp_path / "repository"
    folder = repo / "research" / "pc_causal_rt"
    folder.mkdir(parents=True)
    for name in ("gadi_compare_prepare.sh", "gadi_compare_run.pbs"):
        shutil.copyfile(HERE / name, folder / name)
    child_template = tmp_path / "child_submit.pbs"
    child_template.write_text(CHILD_SUBMIT)
    (folder / "gadi_prepare.sh").write_text(
        """#!/usr/bin/env bash
set -euo pipefail
python3 - "$@" <<'PY'
import json
import os
from pathlib import Path
import shutil
import sys
Path(os.environ["TEST_CHILD_PREPARE_CALL"]).write_text(json.dumps({
    "arguments": sys.argv[1:], "parent_root": os.environ["PC_PARENT_ROOT"]}))
if os.environ.get("TEST_CHILD_PREPARE_FAIL"):
    raise SystemExit(29)
run = Path(os.environ["PC_CAUSAL_RT_RUN_ROOT"]) / "causal_rt_fixture"
source = run / "source" / "research" / "pc_causal_rt"
source.mkdir(parents=True)
shutil.copyfile("research/pc_causal_rt/gadi_compare_run.pbs",
                source / "gadi_compare_run.pbs")
shutil.copyfile(os.environ["TEST_CHILD_TEMPLATE"], run / "submit.pbs")
cfg = json.loads(Path(os.environ["PC_CONFIG_JSON"]).read_text())
if "PC_THREADS" in os.environ:
    cfg["threads"] = int(os.environ["PC_THREADS"])
(run / "experiment_config.json").write_text(json.dumps(cfg))
(run / "PREPARED").touch()
PY
"""
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    qsub = bin_dir / "qsub"
    qsub.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
with open(os.environ["TEST_QSUB_CALLS"], "a") as stream:
    stream.write(json.dumps(sys.argv[1:]) + "\\n")
print("12345.gadi-pbs")
"""
    )
    qsub.chmod(0o755)
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("PC_", "PBS_", "GIT_", "TEST_CHILD_", "TEST_QSUB_"))
    }
    root = tmp_path / "campaigns"
    parent_root = tmp_path / "existing_runs"
    calls = tmp_path / "qsub_calls.jsonl"
    env.update(
        PATH=str(bin_dir) + os.pathsep + os.environ["PATH"],
        PC_COMPARE_RUN_ROOT=str(root),
        PC_CAUSAL_RT_RUN_ROOT=str(parent_root),
        TEST_CHILD_TEMPLATE=str(child_template),
        TEST_CHILD_PREPARE_CALL=str(tmp_path / "child_prepare_call.json"),
        TEST_QSUB_CALLS=str(calls),
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
    )
    for arguments in (
        ["init", "--quiet"],
        ["add", "."],
        ["-c", "user.name=Campaign test", "-c", "user.email=test@example.invalid",
         "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null",
         "commit", "--quiet", "-m", "Test campaign fixture"],
    ):
        subprocess.run(["git", *arguments], cwd=repo, env=env, check=True,
                       capture_output=True, text=True)
    return {"repo": repo, "root": root, "env": env, "calls": calls,
            "parent_root": parent_root}


def _prepare(tools, *arguments, extra_env=None):
    return subprocess.run(
        ["bash", "research/pc_causal_rt/gadi_compare_prepare.sh", *arguments],
        cwd=tools["repo"], env=tools["env"] | (extra_env or {}),
        capture_output=True, text=True, timeout=20, check=False,
    )


def _campaign(tools):
    campaigns = list(tools["root"].glob("pc_compare_*"))
    assert len(campaigns) == 1
    return campaigns[0]


def _run_wrapper(tools, campaign, **extra):
    env = tools["env"] | {
        "PC_COMPARE_RUN": str(campaign), "PBS_JOBID": "12345.test", "PBS_NCPUS": "32",
    } | extra
    return subprocess.run(["bash", str(campaign / "submit.pbs")], cwd=tools["repo"],
                          env=env, capture_output=True, text=True, timeout=20, check=False)


def test_prepare_single_case_defaults_and_frozen_files(campaign_tools):
    """Freeze one explicit case and preserve the independently prepared child."""
    result = _prepare(campaign_tools, "--parent", "/example/native_run")
    assert result.returncode == 0, result.stdout + result.stderr
    campaign = _campaign(campaign_tools)
    child = Path((campaign / "run_path.txt").read_text().strip())
    cfg = json.loads((child / "experiment_config.json").read_text())
    assert (cfg["box_len"], cfg["dimension"], cfg["highres_dimension"]) == (256, 128, 384)
    assert (cfg["threads"], cfg["seeds"]) == (32, [222])
    assert (cfg["z_start"], cfg["z_end"], cfg["source_intervals"], cfg["checkpoint_stride"]) == (25, 5.75, 154, 1)
    assert cfg["source_mode"] == "native_cumulative"
    assert cfg["settings"] == [{"directions": 96, "refinement": 1, "cfl": 0.4,
                                "time_substeps": 4, "source_stride": 1}]
    assert (child / "submit.pbs").read_text() == CHILD_SUBMIT
    assert (campaign / "submit.pbs").read_bytes() == (HERE / "gadi_compare_run.pbs").read_bytes()
    assert (campaign / "script_timeout.txt").read_text().strip() == "14400"
    for line in (campaign / "prepared.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((campaign / name).read_bytes()).hexdigest() == digest
    call = json.loads(Path(campaign_tools["env"]["TEST_CHILD_PREPARE_CALL"]).read_text())
    assert call == {"arguments": ["--parent", "/example/native_run"],
                    "parent_root": str(campaign_tools["parent_root"])}
    assert not campaign_tools["calls"].exists()


def test_submit_only_campaign_wrapper_with_requested_allocation(campaign_tools):
    """Submit the complete wrapper once with the advertised PBS allocation."""
    result = _prepare(campaign_tools, "--submit")
    assert result.returncode == 0, result.stdout + result.stderr
    campaign = _campaign(campaign_tools)
    calls = [json.loads(line) for line in campaign_tools["calls"].read_text().splitlines()]
    assert len(calls) == 1
    args = calls[0]
    assert args[-1] == str(campaign / "submit.pbs")
    assert args[args.index("-v") + 1] == "PC_COMPARE_RUN=" + str(campaign)
    assert args[args.index("-q") + 1] == "normal"
    assert args[args.index("-l") + 1] == (
        "ncpus=32,mem=128GB,walltime=48:00:00,jobfs=10GB,storage=gdata/qp00+scratch/qp00"
    )
    assert (campaign / "jobid.txt").read_text().strip() == "12345.gadi-pbs"
    assert str(campaign / "submit.pbs") in (campaign / "resubmit.sh").read_text()


def test_failed_child_preparation_never_submits(campaign_tools):
    """An unsuccessful native preparation must not consume a PBS allocation."""
    result = _prepare(campaign_tools, "--submit", extra_env={"TEST_CHILD_PREPARE_FAIL": "1"})
    assert result.returncode == 29
    campaign = _campaign(campaign_tools)
    assert (campaign / "FAILED_PREPARE").read_text().strip() == "29"
    assert not (campaign / "PREPARED").exists()
    assert not campaign_tools["calls"].exists()


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf"])
def test_invalid_timeout_never_prepares_or_submits(campaign_tools, timeout):
    """Reject unusable operation limits before creating a native run."""
    result = _prepare(campaign_tools, "--submit", extra_env={"PC_SCRIPT_TIMEOUT": timeout})
    assert result.returncode != 0
    assert "PC_SCRIPT_TIMEOUT must be finite and positive" in result.stderr
    assert not Path(campaign_tools["env"]["TEST_CHILD_PREPARE_CALL"]).exists()
    assert not campaign_tools["calls"].exists()


def test_wrapper_propagates_child_failure_without_script_or_completion(campaign_tools):
    """Keep the full campaign incomplete when its native/RT stage fails."""
    result = _prepare(campaign_tools)
    assert result.returncode == 0, result.stdout + result.stderr
    campaign = _campaign(campaign_tools)
    child = Path((campaign / "run_path.txt").read_text().strip())
    result = _run_wrapper(campaign_tools, campaign, TEST_CHILD_RUN_EXIT="7")
    assert result.returncode == 7, result.stdout + result.stderr
    assert (child / "child_called").is_file()
    assert (campaign / "FAILED").read_text().strip() == "7"
    assert not (campaign / "COMPLETED").exists()
    assert not (campaign / "script").exists()


def test_wrapper_refuses_concurrent_campaign(campaign_tools):
    """Prevent duplicate jobs from writing the same campaign simultaneously."""
    result = _prepare(campaign_tools)
    assert result.returncode == 0, result.stdout + result.stderr
    campaign = _campaign(campaign_tools)
    child = Path((campaign / "run_path.txt").read_text().strip())
    with (campaign / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = _run_wrapper(campaign_tools, campaign)
    assert result.returncode == 2
    assert "Another process is using this comparison" in result.stderr
    assert not (child / "child_called").exists()
    assert not (campaign / "COMPLETED").exists()


def test_wrapper_detects_changed_frozen_timeout_before_child(campaign_tools):
    """Verify operational metadata before launching the prepared native child."""
    result = _prepare(campaign_tools)
    assert result.returncode == 0, result.stdout + result.stderr
    campaign = _campaign(campaign_tools)
    child = Path((campaign / "run_path.txt").read_text().strip())
    (campaign / "script_timeout.txt").write_text("3600\n")
    result = _run_wrapper(campaign_tools, campaign)
    assert result.returncode != 0
    assert "FAILED" in result.stdout
    assert not (child / "child_called").exists()
    assert not (campaign / "COMPLETED").exists()
