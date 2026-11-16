"""Check the isolated Gadi launcher without submitting a real scheduler job."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

RELATIVE = Path("research/pc_analytic_emulator/completion_20261116/reproduction")


@pytest.fixture
def prepared_repo(tmp_path: Path) -> SimpleNamespace:
    """Create a committed minimal checkout and a recording qsub substitute."""
    repo = tmp_path / "repo"
    scripts = repo / RELATIVE
    scripts.mkdir(parents=True)
    for name in ("gadi_train_and_validate.sh", "gadi_train_and_validate.pbs"):
        shutil.copyfile(Path(__file__).with_name(name), scripts / name)
    (scripts / "train_and_validate.py").write_text(
        "import json, pathlib, sys\n"
        "args = dict(zip(sys.argv[1::2], sys.argv[2::2]))\n"
        "out = pathlib.Path(args['--output']); out.mkdir()\n"
        "(out / 'arguments.json').write_text(json.dumps(args))\n"
        "(out / 'COMPLETED').write_text('verified mock workflow')\n"
    )
    for arguments in (
        ["init", "--quiet"],
        ["add", "."],
        [
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "launcher fixture",
        ],
    ):
        subprocess.run(["git", "-C", str(repo), *arguments], check=True)
    campaign, previous = tmp_path / "campaign", tmp_path / "previous"
    (campaign / "training_collection").mkdir(parents=True)
    (campaign / "training_collection/collection_manifest.json").write_text("{}")
    previous.mkdir()
    (previous / "model.json").write_text("{}")
    binary = tmp_path / "bin"
    binary.mkdir()
    mock_qsub = binary / "qsub"
    mock_qsub.write_text(
        f"#!{sys.executable}\nimport json, os, pathlib, sys\n"
        "pathlib.Path(os.environ['QSUB_CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n"
        "print('123.mock-pbs')\n"
    )
    mock_qsub.chmod(0o755)
    runtime = tmp_path / "runtime.env"
    runtime.write_text("PC_MODULES=()\n")
    env = dict(
        os.environ,
        PATH=f"{binary}:{os.environ['PATH']}",
        PC_PYTHON=sys.executable,
        PC_RUNTIME_ENV=str(runtime),
        PC_WORKERS="2",
        PC_THREADS="2",
        PC_NCPUS="4",
        PC_FRAME_CACHE_MIB="1024",
        QSUB_CAPTURE=str(tmp_path / "qsub.json"),
    )
    return SimpleNamespace(
        repo=repo,
        script=scripts / "gadi_train_and_validate.sh",
        campaign=campaign,
        previous=previous,
        output=tmp_path / "run",
        env=env,
    )


def _prepare(case: SimpleNamespace, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "bash",
            str(case.script),
            "--campaign-root",
            str(case.campaign),
            "--previous-model",
            str(case.previous),
            "--output",
            str(case.output),
            *extra,
        ],
        env=case.env,
        capture_output=True,
        text=True,
        check=False,
    )


def _pbs(case: SimpleNamespace, *, cpus: str = "4") -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(case.output / "submit.pbs")],
        env=dict(
            case.env,
            PC_COMPLETION_RUN=str(case.output),
            PBS_JOBID="123.mock-pbs",
            PBS_NCPUS=cpus,
            PBS_NNODES="1",
            PBS_JOBFS=str(case.output / "jobfs"),
        ),
        capture_output=True,
        text=True,
        check=False,
    )


def test_prepare_submit_and_frozen_pbs_execution(
    prepared_repo: SimpleNamespace,
) -> None:
    """Preserve paths/resources and execute only the committed isolated source."""
    case = prepared_repo
    result = _prepare(case, "--submit")
    assert result.returncode == 0, result.stderr + result.stdout
    args = json.loads(Path(case.env["QSUB_CAPTURE"]).read_text())
    assert "ncpus=4,mem=16GB,walltime=24:00:00,storage=gdata/qp00+scratch/qp00" in args
    assert f"PC_COMPLETION_RUN={case.output}" in args
    assert args[-1] == str(case.output / "submit.pbs")
    assert (case.output / "jobid.txt").read_text().strip() == "123.mock-pbs"
    assert (case.output / "runtime.env").read_bytes() == Path(
        case.env["PC_RUNTIME_ENV"]
    ).read_bytes()
    result = _pbs(case)
    assert result.returncode == 0, result.stderr + result.stdout
    assert (case.output / "COMPLETED").is_file()
    recorded = json.loads((case.output / "results/arguments.json").read_text())
    assert recorded == {
        "--campaign-root": str(case.campaign),
        "--previous-model": str(case.previous),
        "--output": str(case.output / "results"),
        "--workers": "2",
        "--threads": "2",
        "--cpus": "4",
        "--rays": "4096",
        "--frame-cache-mib": "1024",
    }


@pytest.mark.parametrize(
    "relative",
    ["runtime.env", str(Path("source") / RELATIVE / "train_and_validate.py")],
)
def test_pbs_refuses_changed_frozen_files(
    prepared_repo: SimpleNamespace, relative: str
) -> None:
    """An edited module/runtime must fail its guard before workflow execution."""
    case = prepared_repo
    assert _prepare(case).returncode == 0
    path = case.output / relative
    path.write_text(path.read_text() + "\n# mutation\n")
    result = _pbs(case)
    assert result.returncode != 0
    assert not (case.output / "results").exists()
    assert (case.output / "FAILED").is_file()


def test_pbs_refuses_insufficient_cpus(prepared_repo: SimpleNamespace) -> None:
    """Do not oversubscribe an allocation smaller than workers times threads."""
    case = prepared_repo
    assert _prepare(case).returncode == 0
    assert _pbs(case, cpus="2").returncode != 0
    assert not (case.output / "results").exists()


def test_dirty_checkout_and_existing_output_are_rejected(
    prepared_repo: SimpleNamespace,
) -> None:
    """Never overwrite a prepared run or silently omit uncommitted code."""
    case = prepared_repo
    path = case.repo / RELATIVE / "train_and_validate.py"
    original = path.read_text()
    path.write_text(original + "\n# dirty\n")
    assert _prepare(case).returncode != 0
    assert not case.output.exists()
    path.write_text(original)
    assert _prepare(case).returncode == 0
    assert _prepare(case).returncode != 0


def test_shell_templates_parse() -> None:
    """Both checked-in shell entry points are valid bash programs."""
    for name in ("gadi_train_and_validate.sh", "gadi_train_and_validate.pbs"):
        subprocess.run(["bash", "-n", str(Path(__file__).with_name(name))], check=True)
