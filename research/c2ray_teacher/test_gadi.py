"""Exercise frozen Gadi preparation and dispatch without a scheduler."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

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


def _history_fixture(path: Path, **updates: float) -> bytes:
    """Write the geometry metadata read by the specialized accuracy preparer."""
    metadata = {
        "dimension": 16,
        "highres_dimension": 48,
        "box_len_cMpc": 50.0,
        "redshift_start": 20.0,
        "redshift_end": 5.0,
        "intervals": 30,
        "reference_redshift": 8.0,
    } | updates
    np.savez(path, metadata=np.asarray(json.dumps(metadata)))
    return path.read_bytes()


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
        for name in ("gadi_prepare.sh", "gadi_run.pbs", "gadi_accuracy_prepare.sh"):
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
        for name in ("py21cmfast", "scipy", "h5py"):
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
            '#!/bin/bash\nprintf "%s\\n" "$@" > "$C2_TEST_QSUB"\n'
            'printf "CALL\\n%s\\n" "$*" >> "${C2_TEST_QSUB}.all"\n'
            'n=$(wc -l < "${C2_TEST_QSUB}.all"); echo "$n.gadi-pbs"\n',
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

    def test_existing_history_is_copied_and_frozen_with_accuracy_controls(self) -> None:
        """A reused history survives later changes to the original input file."""
        history = self.root / "existing history.npz"
        history.write_bytes(b"native history fixture")
        receipt = self.root / "prepared_path.txt"
        self.env.update(
            C2_HISTORY=str(history),
            C2_REFINEMENT="3",
            C2_TIME_SUBSTEPS="4",
            C2_STRIDE="2",
            C2_BENCHMARK="false",
            C2_PREPARED_PATH_FILE=str(receipt),
        )
        result = self._prepare()
        assert result.returncode == 0, result.stdout + result.stderr
        run = self._run()
        config = json.loads((run / "run_config.json").read_text())
        manifest = json.loads((run / "run_manifest.json").read_text())
        frozen = run / "input" / "native_history.npz"
        assert config["history"] == str(frozen)
        assert frozen.read_bytes() == b"native history fixture"
        assert config["refinement"] == 3
        assert config["time_substeps"] == 4
        assert config["stride"] == 2
        assert config["benchmark"] is False
        assert manifest["reused_history"]["source"] == str(history)
        assert "input/native_history.npz" in (run / "prepared.sha256").read_text()
        assert receipt.read_text().strip() == str(run)
        history.write_bytes(b"changed after preparation")
        assert self._dispatch(run).returncode == 0

    def test_modified_frozen_history_is_rejected(self) -> None:
        """Prepared-history corruption stops dispatch before the runner starts."""
        history = self.root / "history.npz"
        history.write_bytes(b"native history fixture")
        self.env["C2_HISTORY"] = str(history)
        result = self._prepare()
        assert result.returncode == 0, result.stdout + result.stderr
        run = self._run()
        (run / "input" / "native_history.npz").write_bytes(b"modified")
        assert self._dispatch(run).returncode != 0
        assert not (run / "results").exists()

    def test_invalid_accuracy_overrides_do_not_submit(self) -> None:
        """Unsupported refinement, timestep and boolean controls fail early."""
        for name, value in (
            ("C2_REFINEMENT", "2"),
            ("C2_TIME_SUBSTEPS", "0"),
            ("C2_STRIDE", "1.5"),
            ("C2_BENCHMARK", "maybe"),
            ("C2_HISTORY", str(self.root / "missing.npz")),
        ):
            with self.subTest(name=name):
                self.env[name] = value
                result = self._prepare("--submit")
                assert result.returncode != 0
                assert not (self.root / "qsub.args").exists()
                del self.env[name]

    def test_existing_prepared_receipt_is_not_overwritten(self) -> None:
        """A caller's old run receipt must not be replaced by a new submission."""
        receipt = self.root / "prepared_path.txt"
        receipt.write_text("existing run\n")
        self.env["C2_PREPARED_PATH_FILE"] = str(receipt)
        result = self._prepare("--submit")
        assert result.returncode != 0
        assert receipt.read_text() == "existing run\n"
        assert not (self.root / "qsub.args").exists()

    def _accuracy(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(self.code / "gadi_accuracy_prepare.sh"), *arguments],
            env=self.env,
            text=True,
            capture_output=True,
            check=False,
        )

    def _suite(self) -> tuple[Path, dict]:
        roots = list((self.root / "runs").glob("c2ray_accuracy_*"))
        assert len(roots) == 1
        suite = roots[0]
        return suite, json.loads((suite / "suite_manifest.json").read_text())

    def test_accuracy_suite_freezes_eight_cases_and_chains_submissions(self) -> None:
        """Submit one 32-CPU case at a time with controlled accuracy comparisons."""
        history = self.root / "original history.npz"
        history_bytes = _history_fixture(history)
        self.env.update(C2_HISTORY=str(history), C2_BENCHMARK="false")
        result = self._accuracy("--submit")
        assert result.returncode == 0, result.stdout + result.stderr
        suite, manifest = self._suite()
        assert manifest["stage"] == "all"
        assert manifest["submission_policy"] == "sequential_afterok"
        assert (
            manifest["validation_history_sha256"]
            == hashlib.sha256(history_bytes).hexdigest()
        )
        cases = manifest["cases"]
        assert [row["name"] for row in cases] == [
            "time1",
            "time2",
            "time4",
            "space48_dt2",
            "bridge_L50_N16_DIM128",
            "box_L75_N24_DIM192",
            "box_L100_N32_DIM256",
            "finer_L75_N32_DIM192",
        ]
        previous = None
        for index, row in enumerate(cases):
            run = Path(row["run_directory"])
            config = json.loads((run / "run_config.json").read_text())
            assert config["benchmark"] is (index == 0)
            assert row["dependency_job_id"] == previous
            assert row["job_id"]
            previous = row["job_id"]
            assert (run / "PREPARED").is_file()
            assert (run / "job_id.txt").read_text().strip() == row["job_id"]
            if index < 4:
                assert Path(config["history"]).read_bytes() == history_bytes
                assert config["time_substeps"] == (1, 2, 4, 2)[index]
                assert config["refinement"] == (1, 1, 1, 3)[index]
            else:
                assert "history" not in config
                assert config["intervals"] == 60
                assert config["time_substeps"] == 2
        assert cases[4]["config"]["highres_dimension"] == 128
        assert cases[5]["config"]["box_len"] == 75
        assert cases[5]["config"]["dimension"] == 24
        assert cases[6]["config"]["highres_dimension"] == 256
        assert cases[7]["config"]["dimension"] == 32
        assert cases[7]["config"]["box_len"] == 75
        calls = (self.root / "qsub.args.all").read_text()
        assert calls.count("depend=afterok:") == 7
        assert calls.count("ncpus=32,mem=128GB,walltime=48:00:00") == 8
        assert (suite / "SUBMITTED").is_file()

    def test_expanded_stage_does_not_require_previous_history(self) -> None:
        """New larger-volume cases can be prepared independently of old fixtures."""
        self.env["C2_HISTORY"] = str(self.root / "missing.npz")
        result = self._accuracy("--stage", "expanded")
        assert result.returncode == 0, result.stdout + result.stderr
        suite, manifest = self._suite()
        assert len(manifest["cases"]) == 4
        assert all(row["job_id"] is None for row in manifest["cases"])
        assert not (suite / "SUBMITTED").exists()
        assert not (self.root / "qsub.args").exists()

    def test_accuracy_validation_requires_history_before_submission(self) -> None:
        """An unavailable old fixture cannot produce a partially submitted suite."""
        result = self._accuracy(
            "--stage",
            "validation",
            "--history",
            str(self.root / "missing.npz"),
            "--submit",
        )
        assert result.returncode != 0
        assert not (self.root / "qsub.args").exists()

    def test_late_preparation_failure_submits_no_cases(self) -> None:
        """A failed fourth preparation leaves earlier prepared cases unsubmitted."""
        history = self.root / "history.npz"
        history.write_bytes(b"native history fixture")
        prepare = self.code / "gadi_prepare.sh"
        prepare.write_text(
            prepare.read_text().replace(
                "set -euo pipefail",
                "set -euo pipefail\n"
                "if [[ ${C2_REFINEMENT:-1} == 3 ]]; then exit 91; fi",
                1,
            )
        )
        _commit(self.repo)
        result = self._accuracy(
            "--stage", "validation", "--history", str(history), "--submit"
        )
        assert result.returncode != 0
        suite, _ = self._suite()
        assert (suite / "receipts" / "time4.txt").is_file()
        assert not (suite / "receipts" / "space48_dt2.txt").exists()
        assert not (suite / "SUBMITTED").exists()
        assert not (self.root / "qsub.args").exists()

    def test_history_changes_between_cases_prevent_all_submission(self) -> None:
        """Identical history paths cannot hide unequal copied source histories."""
        history = self.root / "history.npz"
        history.write_bytes(b"native history fixture")
        prepare = self.code / "gadi_prepare.sh"
        prepare.write_text(
            prepare.read_text()
            + "\nif [[ ${C2_TIME_SUBSTEPS:-1} == 1 ]]; then\n"
            + '    printf "changed history" > "$C2_HISTORY"\n'
            + "fi\n"
        )
        _commit(self.repo)
        result = self._accuracy(
            "--stage", "validation", "--history", str(history), "--submit"
        )
        assert result.returncode != 0
        assert "changed between case preparations" in result.stderr
        assert not (self.root / "qsub.args").exists()

    def test_accuracy_rejects_incompatible_validation_geometry(self) -> None:
        """Specialized case names cannot silently select a much larger old grid."""
        history = self.root / "wrong_geometry.npz"
        _history_fixture(history, dimension=32)
        result = self._accuracy(
            "--stage", "validation", "--history", str(history), "--submit"
        )
        assert result.returncode != 0
        assert "requires dimension=16" in result.stderr
        assert not (self.root / "qsub.args").exists()

    def test_accuracy_rejects_nonbaseline_history_clock(self) -> None:
        """Source-sampling changes must not masquerade as an RT-only time test."""
        history = self.root / "wrong_clock.npz"
        _history_fixture(history, intervals=60)
        result = self._accuracy(
            "--stage", "validation", "--history", str(history), "--submit"
        )
        assert result.returncode != 0
        assert "requires intervals=30" in result.stderr
        assert not (self.root / "qsub.args").exists()


if __name__ == "__main__":
    unittest.main()
