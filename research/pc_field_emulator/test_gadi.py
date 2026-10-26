"""Exercise preparation and PBS dispatch without a scheduler or new simulation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from research.pc_field_emulator.test_data import canonical, make_case

HERE = Path(__file__).parent
DEFAULT_SMALL = (
    "/g/data/qp00/sz4640/21cmfast_runs/evolution_pc_30e8497de2_20261024T122119_qC2V"
)
DEFAULT_LARGE = "/g/data/qp00/sz4640/21cmfast_runs/evolution_production_ecd4bbdb88_20261024T144219_8iMG"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def _setup(tmp_path: Path) -> tuple[Path, list[Path], dict[str, str]]:
    repo = tmp_path / "repo"
    code = repo / "research" / "pc_field_emulator"
    code.mkdir(parents=True)
    roots = [tmp_path / "small input", tmp_path / "large input"]
    make_case(roots[0], seed=10)
    make_case(roots[1], seed=20, production=True, complete=False)
    for name in ("data.py", "gadi_prepare.sh", "gadi_train.pbs"):
        text = (HERE / name).read_text()
        if name == "gadi_prepare.sh":
            text = text.replace(DEFAULT_SMALL, str(roots[0])).replace(
                DEFAULT_LARGE, str(roots[1])
            )
        (code / name).write_text(text)
    (code / "train.py").write_text(
        "import json,sys\n"
        "from pathlib import Path\n"
        "args=sys.argv[1:]\n"
        "out=Path(args[args.index('--output')+1]);out.mkdir()\n"
        "inventory=Path(args[args.index('--sample-manifest')+1])\n"
        "samples=json.loads(inventory.read_text())['samples']\n"
        "result={'arguments':args,'sample_count':len(samples)}\n"
        "(out/'summary.json').write_text(json.dumps(result))\n"
        "(out/'COMPLETED').touch()\n"
    )
    for args in (
        ["init", "--quiet"],
        ["add", "research"],
        [
            "-c",
            "user.name=PC test",
            "-c",
            "user.email=pc-test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "Test preparation source",
        ],
    ):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    (roots[0] / "runtime.env").write_text("PC_MODULES=(mock-python)\n")
    (roots[0] / "python_path.txt").write_text(str(Path(sys.executable).absolute()))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(bin_dir / "module", "#!/bin/bash\nexit 0\n")
    _write_executable(
        bin_dir / "qsub",
        '#!/bin/bash\nprintf "%s\\n" "$@" > "$PC_TEST_QSUB"\necho 123.gadi-pbs\n',
    )
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("PC_") or key in ("PYTHONPATH", "PYTHONHOME"):
            env.pop(key)
    env.update(
        PATH=f"{bin_dir}:{env['PATH']}",
        PC_EMULATOR_RUN_ROOT=str(tmp_path / "runs"),
        PC_TEST_QSUB=str(tmp_path / "qsub.args"),
    )
    return code, roots, env


def _prepare(
    code: Path, env: dict[str, str], *args: str
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(code / "gadi_prepare.sh"), *args],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _run_directory(env: dict[str, str]) -> Path:
    runs = list(Path(env["PC_EMULATOR_RUN_ROOT"]).iterdir())
    assert len(runs) == 1
    return runs[0]


def _hash_inputs(roots: list[Path]) -> dict[str, str]:
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for root in roots
        for path in root.rglob("*")
        if path.is_file()
    }


def test_defaults_freeze_both_existing_runs_and_submit_32_cpus(tmp_path: Path) -> None:
    """Archive both inputs and pin the committed production inventory."""
    code, roots, env = _setup(tmp_path)
    before = _hash_inputs(roots)
    completed = _prepare(code, env, "--submit")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    run = _run_directory(env)
    manifest = json.loads((run / "run_manifest.json").read_text())
    inventory = json.loads((run / "sample_manifest.json").read_text())
    assert manifest["input_roots"] == list(map(str, roots))
    assert inventory["input_roots"] == list(map(str, roots))
    assert inventory["teacher_arm"] == {"subcells": 4, "angles": 768, "stride": 4}
    assert len(inventory["samples"]) == 13
    assert max(s["node_index"] for s in inventory["samples"] if s["seed"] == 20) == 5
    args = manifest["arguments"]
    assert args.count("--input") == 2
    assert args[args.index("--sample-manifest") + 1] == str(
        run / "sample_manifest.json"
    )
    assert args[args.index("--model-mode") + 1] == "residual"
    assert args[args.index("--samples-per-snapshot") + 1] == "8192"
    assert args[args.index("--epochs") + 1] == "200"
    assert args[args.index("--front-weight") + 1] == "4"
    assert args[args.index("--neutral-weight") + 1] == "2"
    assert manifest["requested_cpus"] == 32
    assert manifest["trial_workers"] * manifest["blas_threads_per_worker"] == 32
    assert "sample_manifest.json" in (run / "prepared.sha256").read_text()
    assert "ncpus=32,mem=64GB" in Path(env["PC_TEST_QSUB"]).read_text()
    assert before == _hash_inputs(roots)


@pytest.mark.parametrize("source", ["explicit", "environment"])
def test_single_input_override_still_works(tmp_path: Path, source: str) -> None:
    """An explicit or environment input replaces the two default roots."""
    code, roots, env = _setup(tmp_path)
    args = [str(roots[0])] if source == "explicit" else []
    if source == "environment":
        env["PC_EMULATOR_INPUT"] = str(roots[0])
    completed = _prepare(code, env, *args)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    manifest = json.loads((_run_directory(env) / "run_manifest.json").read_text())
    assert manifest["input_roots"] == [str(roots[0])]
    assert not Path(env["PC_TEST_QSUB"]).exists()


def test_multiple_explicit_inputs_and_missing_input_fail_before_submit(
    tmp_path: Path,
) -> None:
    """Every requested source must exist before a batch job is created."""
    code, roots, env = _setup(tmp_path)
    shutil.rmtree(roots[1])
    completed = _prepare(code, env, str(roots[0]), str(roots[1]), "--submit")
    assert completed.returncode != 0
    assert not Path(env["PC_TEST_QSUB"]).exists()
    assert not Path(env["PC_EMULATOR_RUN_ROOT"]).exists()


def test_pbs_uses_frozen_samples_after_producer_progresses(tmp_path: Path) -> None:
    """New journal rows after preparation cannot enter the training dataset."""
    code, roots, env = _setup(tmp_path)
    completed = _prepare(code, env, *map(str, roots))
    assert completed.returncode == 0, completed.stdout + completed.stderr
    run = _run_directory(env)
    journal = roots[1] / "pc" / "progress.json"
    progress = json.loads(journal.read_text())
    # The stub worker only reads the pinned inventory. Journal growth is deliberate.
    progress["payload"]["checkpoints"].append({"node_index": 6})
    progress["sha256"] = hashlib.sha256(canonical(progress["payload"])).hexdigest()
    journal.write_text(json.dumps(progress))
    env.update(RUN_DIR=str(run), PBS_JOBID="123.gadi-pbs", PBS_NCPUS="32")
    result = subprocess.run(
        ["bash", str(run / "submit.pbs")],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, (run / "run_123.gadi-pbs.log").read_text()
    report = json.loads((run / "results" / "summary.json").read_text())
    assert report["sample_count"] == 13
    assert (run / "COMPLETED").exists()
    assert not (run / "FAILED").exists()
    assert not (run / ".running").exists()


def test_empty_requested_input_fails_before_submit(tmp_path: Path) -> None:
    """An existing but uncommitted large run cannot be silently omitted."""
    code, roots, env = _setup(tmp_path)
    journal = roots[1] / "pc" / "progress.json"
    progress = json.loads(journal.read_text())
    progress["payload"]["checkpoints"] = []
    progress["sha256"] = hashlib.sha256(canonical(progress["payload"])).hexdigest()
    journal.write_text(json.dumps(progress))
    before = _hash_inputs(roots)
    completed = _prepare(code, env, "--submit")
    assert completed.returncode != 0
    assert str(roots[1]) in completed.stderr
    assert "No committed PC" in completed.stderr
    assert not Path(env["PC_TEST_QSUB"]).exists()
    assert not (_run_directory(env) / "PREPARED").exists()
    assert before == _hash_inputs(roots)


def test_pbs_rejects_modified_frozen_inventory(tmp_path: Path) -> None:
    """The inventory belongs to the prepared checksum envelope."""
    code, _, env = _setup(tmp_path)
    completed = _prepare(code, env)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    run = _run_directory(env)
    with (run / "sample_manifest.json").open("a") as stream:
        stream.write("\n")
    env.update(RUN_DIR=str(run), PBS_JOBID="124.gadi-pbs", PBS_NCPUS="32")
    result = subprocess.run(
        ["bash", str(run / "submit.pbs")],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert not (run / "results").exists()
    assert (run / "FAILED").exists()
