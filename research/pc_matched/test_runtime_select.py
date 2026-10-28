"""Check runtime discovery before Gadi loads a scientific Python environment."""

import ast
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from research.pc_matched import runtime_select

SOURCE = Path(runtime_select.__file__).parent


def test_bootstrap_keeps_python36_syntax() -> None:
    """Reject syntax and future annotations unsupported by the login Python."""
    tree = ast.parse(Path(runtime_select.__file__).read_text(), feature_version=(3, 6))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module != "__future__", "Bootstrap must run before modern Python loads"
        if isinstance(node, ast.arg) and node.annotation is not None:
            assert isinstance(node.annotation, (ast.Name, ast.Constant))


@pytest.mark.parametrize("explicit", [False, True])
def test_bootstrap_cli_needs_no_site_packages(tmp_path: Path, explicit: bool) -> None:
    """Select the native interpreter using only the standard library."""
    parent = tmp_path / "completed"
    parent.mkdir()
    (parent / "COMPLETED").touch()
    (parent / "native_python.txt").write_text(sys.executable + "\n")
    runtime_env = parent / "runtime.env"
    runtime_env.write_text("PC_MODULES=()\n")
    command = [sys.executable, "-S", str(SOURCE / "runtime_select.py"),
               "select", "--root", str(tmp_path)]
    if explicit:
        command += ["--python", sys.executable]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=True)
    expected = [sys.executable, "", ""] if explicit else [sys.executable, str(parent), str(runtime_env)]
    assert result.stdout.splitlines() == expected


def test_shell_bootstraps_before_loading_modern_preparation(tmp_path: Path) -> None:
    """An old-login-Python shim must hand preparation to the selected runtime."""
    repo = tmp_path / "repo"
    scripts = repo / "research/pc_matched"
    scripts.mkdir(parents=True)
    for name in ("gadi_prepare.sh", "runtime_select.py"):
        shutil.copyfile(SOURCE / name, scripts / name)
    for name in ("workflow.py", "gadi_run.pbs"):
        (scripts / name).touch()
    for args in (["init", "-q"], ["add", "."],
                 ["-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                  "commit", "-qm", "Bootstrap fixture"]):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)

    native_python = tmp_path / "native-python"
    native_python.write_text(
        '#!/bin/bash\nif [[ $1 == -m ]]; then\n'
        '  echo "SELECTED_NATIVE_PREPARATION_REACHED"\n  exit 73\nfi\n'
        f'exec {shlex.quote(sys.executable)} "$@"\n'
    )
    native_python.chmod(0o700)
    parent = tmp_path / "parents/completed"
    parent.mkdir(parents=True)
    (parent / "COMPLETED").touch()
    (parent / "native_python.txt").write_text(str(native_python) + "\n")
    (parent / "runtime.env").write_text("PC_MODULES=()\n")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    login_python = binaries / "python3"
    login_python.write_text(
        '#!/bin/bash\ncase "$1" in\n'
        '  */prepare_run.py) echo "SyntaxError: future feature annotations is not defined" >&2; exit 1 ;;\n'
        f'  */runtime_select.py) exec {shlex.quote(sys.executable)} -S "$@" ;;\n'
        '  *) exit 90 ;;\nesac\n'
    )
    login_python.chmod(0o700)
    environment = {key: value for key, value in os.environ.items() if not key.startswith("PC_")}
    environment.update(PATH=str(binaries) + os.pathsep + os.environ["PATH"],
                       PC_PARENT_ROOT=str(parent.parent), PC_MATCHED_RUN_ROOT=str(tmp_path / "runs"))
    result = subprocess.run(["bash", str(scripts / "gadi_prepare.sh")], cwd=repo,
                            env=environment, capture_output=True, text=True, check=False)
    assert result.returncode == 73, result.stdout + result.stderr
    assert "SELECTED_NATIVE_PREPARATION_REACHED" in result.stdout
    assert "SyntaxError" not in result.stderr
