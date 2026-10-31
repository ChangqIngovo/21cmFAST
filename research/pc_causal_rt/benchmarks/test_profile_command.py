"""Check profiler failure propagation and output-file protection."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from research.pc_causal_rt.benchmarks.profile_command import profile_command


def test_workload_failure_records_resources_and_preserves_exit(tmp_path: Path):
    """A failed scientific job must not appear successful because profiling ran."""
    output = tmp_path / "profile.json"
    command = [sys.executable, "-c", "raise SystemExit(7)"]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "research.pc_causal_rt.benchmarks.profile_command",
            "--output",
            str(output),
            "--",
            *command,
        ],
        check=False,
    )
    assert completed.returncode == 7
    report = json.loads(output.read_text())
    assert report["command"] == command
    assert report["workload_returncode"] == 7
    assert report["status"] == "finished"
    assert report["wall_seconds"] > 0
    assert report["descendant_user_cpu_seconds"] >= 0
    assert report["descendant_system_cpu_seconds"] >= 0
    assert report["start_utc"] <= report["end_utc"]
    assert report["memory"]["peak_rss_includes_prior_children"] is False
    assert "aggregate" in report["memory"]["scope"]


def test_existing_output_prevents_workload_launch(tmp_path: Path):
    """Refusing an old profile must happen before starting any workload."""
    output, marker = tmp_path / "profile.json", tmp_path / "started"
    output.write_text("preserve this result")
    command = [
        sys.executable,
        "-c",
        f"from pathlib import Path; Path({str(marker)!r}).touch()",
    ]
    with pytest.raises(FileExistsError):
        profile_command(command, output)
    assert output.read_text() == "preserve this result"
    assert not marker.exists()


def test_signal_status_is_recorded_and_mapped_to_shell_exit(tmp_path: Path):
    """A signal-killed workload retains its signal identity in the report."""
    output = tmp_path / "profile.json"
    result = profile_command(
        [
            sys.executable,
            "-c",
            "import os, signal; os.kill(os.getpid(), signal.SIGTERM)",
        ],
        output,
    )
    report = json.loads(output.read_text())
    assert result == 143
    assert report["workload_returncode"] == -15
    assert report["termination_signal"] == 15
    assert report["status"] == "signalled"
