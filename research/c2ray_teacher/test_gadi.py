"""Exercise frozen Gadi preparation and dispatch without a scheduler."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).parent
UPSTREAM_REVISION = "a363656ff1c3b4b4c7334f398578d8c469154515"


def _git(directory: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(directory), *arguments], text=True, stderr=subprocess.PIPE
    ).strip()


def _commit(directory: Path) -> str:
    _git(directory, "init", "--quiet")
    _git(directory, "add", ".")
    _git(
        directory,
        "-c",
        "user.name=Adapter test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "Fixture",
    )
    return _git(directory, "rev-parse", "HEAD")


def _executable(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(0o755)


class GadiWorkflowTests(unittest.TestCase):
    """Run shell workflows against local immutable repositories and stubs."""

    def setUp(self) -> None:
        """Create a committed adapter and upstream plus a mocked runtime."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        upstream = self.root / "upstream repository"
        upstream.mkdir()
        (upstream / "LICENSE").write_text("test upstream license\n")
        (upstream / "solver.f90").write_text("! Immutable solver fixture\n")
        self.upstream_revision = _commit(upstream)

        repo = self.root / "repo"
        self.code = repo / "research" / "c2ray_teacher"
        self.code.mkdir(parents=True)
        for name in ("gadi_prepare.sh", "gadi_run.pbs"):
            value = (
                (HERE / name)
                .read_text()
                .replace(UPSTREAM_REVISION, self.upstream_revision)
            )
            (self.code / name).write_text(value)
        (self.code / "runner.py").write_text(
            "import argparse,json\nfrom pathlib import Path\n"
            "p=argparse.ArgumentParser()\n"
            "p.add_argument('--config');p.add_argument('--output')\n"
            "p.add_argument('--upstream');p.add_argument('--workers',type=int)\n"
            "a=p.parse_args();out=Path(a.output);out.mkdir()\n"
            "(out/'summary.json').write_text(json.dumps(vars(a)))\n"
            "(out/'COMPLETED').touch()\n"
        )
        self.revision = _commit(repo)
        self.repo = repo

        packages = self.root / "mock_packages"
        for name in ("py21cmfast", "numpy", "scipy", "h5py"):
            directory = packages / name
            directory.mkdir(parents=True)
            (directory / "__init__.py").write_text('__version__ = "test.1"\n')
        (packages / "py21cmfast" / "native.so").write_bytes(b"compiled fixture")
        parent = self.root / "runtime parent"
        parent.mkdir()
        (parent / "runtime.env").write_text("PC_MODULES=(mock-python)\n")
        python = parent / "python"
        _executable(
            python,
            "#!/bin/bash\n"
            f"export PYTHONPATH={shlex.quote(str(packages))}\n"
            f'exec {shlex.quote(sys.executable)} "$@"\n',
        )
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        for name in ("module", "gfortran", "mpifort"):
            _executable(bin_dir / name, "#!/bin/bash\necho 'mock compiler 1'\n")
        _executable(
            bin_dir / "qsub",
            '#!/bin/bash\nprintf "%s\\n" "$@" > "$C2_TEST_QSUB"\necho 123.gadi-pbs\n',
        )
        self.env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("C2_") and key not in ("PYTHONPATH", "PYTHONHOME")
        }
        self.env.update(
            PATH=f"{bin_dir}:{self.env['PATH']}",
            C2_RUNTIME_PARENT=str(parent),
            C2_PYTHON=str(python),
            C2_UPSTREAM_REPO=str(upstream),
            C2_RUN_ROOT=str(self.root / "runs"),
            C2_TEST_QSUB=str(self.root / "qsub.args"),
        )

    def _prepare(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(self.code / "gadi_prepare.sh"), *arguments],
            env=self.env,
            text=True,
            capture_output=True,
            check=False,
        )

    def _run(self) -> Path:
        directories = list((self.root / "runs").iterdir())
        assert len(directories) == 1
        return directories[0]

    def _dispatch(self, run: Path, *, cpus: int = 32) -> subprocess.CompletedProcess:
        env = dict(
            self.env, RUN_DIR=str(run), PBS_JOBID="123.gadi-pbs", PBS_NCPUS=str(cpus)
        )
        return subprocess.run(
            ["bash", str(run / "submit.pbs")],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_prepare_freezes_both_revisions_and_submits_32_cpus(self) -> None:
        """Preparation archives exact revisions and explicitly submits resources."""
        result = self._prepare("--submit")
        assert result.returncode == 0, result.stdout + result.stderr
        run = self._run()
        manifest = json.loads((run / "run_manifest.json").read_text())
        config = json.loads((run / "run_config.json").read_text())
        assert manifest["commit"] == self.revision
        assert manifest["upstream_commit"] == self.upstream_revision
        assert manifest["requested_workers"] == 32
        assert config["dimension"] == 16
        assert config["highres_dimension"] == 48
        assert config["box_len"] == 50.0
        assert (run / "upstream" / "solver.f90").is_file()
        assert not (run / "upstream" / ".git").exists()
        assert "upstream/solver.f90" in (run / "prepared.sha256").read_text()
        assert "ncpus=32,mem=128GB" in (self.root / "qsub.args").read_text()

    def test_pbs_runs_the_native_pipeline(self) -> None:
        """A prepared job invokes the science runner and checks its completion."""
        result = self._prepare()
        assert result.returncode == 0, result.stdout + result.stderr
        run = self._run()
        result = self._dispatch(run)
        log = (run / "run_123.gadi-pbs.log").read_text()
        assert result.returncode == 0, log
        summary = json.loads((run / "results" / "summary.json").read_text())
        assert summary["workers"] == 32
        assert summary["upstream"] == str(run / "upstream")
        assert (run / "COMPLETED").exists()
        assert not (run / "FAILED").exists()
        assert not (run / ".running").exists()

    def test_modified_upstream_is_rejected_before_science(self) -> None:
        """The job refuses modified upstream numerical code."""
        result = self._prepare()
        assert result.returncode == 0, result.stdout + result.stderr
        run = self._run()
        (run / "upstream" / "solver.f90").write_text("! Modified\n")
        result = self._dispatch(run)
        assert result.returncode != 0
        assert (run / "FAILED").exists()
        assert not (run / "results").exists()

    def test_wrong_allocation_is_rejected(self) -> None:
        """MPI cannot silently oversubscribe a different allocation."""
        result = self._prepare()
        assert result.returncode == 0, result.stdout + result.stderr
        run = self._run()
        assert self._dispatch(run, cpus=16).returncode != 0
        assert not (run / "results").exists()

    def test_dirty_source_fails_before_preparing(self) -> None:
        """Tracked edits are never silently omitted by the git archive."""
        (self.code / "runner.py").write_text("raise RuntimeError('uncommitted')\n")
        result = self._prepare("--submit")
        assert result.returncode != 0
        assert not (self.root / "qsub.args").exists()
        assert not (self.root / "runs").exists()

    def test_invalid_config_fails_without_submitting(self) -> None:
        """Invalid size and clock overrides fail before batch submission."""
        self.env["C2_Z_END"] = "25"
        result = self._prepare("--submit")
        assert result.returncode != 0
        assert not (self.root / "qsub.args").exists()


if __name__ == "__main__":
    unittest.main()
