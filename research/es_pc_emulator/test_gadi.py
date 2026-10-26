"""Exercise the actual preparation and PBS scripts with isolated local stubs."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from research.c2ray_teacher.runner import DEFAULT_CONFIG as TEACHER_DEFAULTS

HERE = Path(__file__).parent
UPSTREAM_REVISION = "a363656ff1c3b4b4c7334f398578d8c469154515"
RELEASE_REVISION = "72d8a2d23702f1b69fbc85abd25d4592a7bcd3f5"


def _git(directory, *args):
    return subprocess.check_output(
        ["git", "-C", str(directory), *args], text=True, stderr=subprocess.PIPE
    ).strip()


def _commit(directory):
    _git(directory, "add", ".")
    _git(directory, "-c", "user.name=Workflow test", "-c", "user.email=test@example.invalid",
         "commit", "--quiet", "-m", "Fixture")
    return _git(directory, "rev-parse", "HEAD")


def _script(path, content):
    path.write_text(content)
    path.chmod(0o755)


@pytest.fixture
def gadi_fixture(tmp_path):
    """Create local repos, a fresh-venv stub and a deterministic dataset backend."""
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _git(upstream, "init", "--quiet")
    (upstream / "solver.f90").write_text("! upstream fixture\n")
    upstream_revision = _commit(upstream)
    repo = tmp_path / "repo"
    code = repo / "research/es_pc_emulator"
    code.mkdir(parents=True)
    _git(repo, "init", "--quiet")
    for name in ("generate.py", "default_dataset.json", "gadi_dataset.pbs"):
        (code / name).write_bytes((HERE / name).read_bytes())
    prepare = (HERE / "gadi_prepare.sh").read_text().replace(
        UPSTREAM_REVISION, upstream_revision
    ).replace(RELEASE_REVISION, "RELEASE_ANCHOR")
    (code / "gadi_prepare.sh").write_text(prepare)
    for directory in (repo / "research/c2ray_teacher", repo / "research/pc_evolution"):
        directory.mkdir(parents=True)
    (repo / "research/c2ray_teacher/runner.py").write_text(
        "from pathlib import Path\n"
        f"DEFAULT_CONFIG={TEACHER_DEFAULTS!r}\n"
        "def validate_config(c): return DEFAULT_CONFIG | c\n"
        "def parallel_layout(w): return (1,w)\n"
        "def run(c,out,upstream,w):\n"
        " out.mkdir(); (out/'COMPLETED').touch(); return {'benchmarks':{'passed':True}}\n"
    )
    (repo / "research/pc_evolution/native_history.py").write_text(
        "def require_es_radius_capability(): pass\n"
        "def build_native_history(**c):\n"
        " c['output'].write_text('history'); c['es_output'].write_text('es')\n"
    )
    (code / "pairs.py").write_text(
        "import json\nfrom pathlib import Path\n"
        "def export_pairs(h,e,t,out,split,expected_seed=None):\n"
        " out.mkdir();m={'samples':[{'seed':expected_seed,'split':split}]};"
        "(out/'manifest.json').write_text(json.dumps(m));return m\n"
        "def combine_manifests(paths,out):\n"
        " m={'samples':sum([json.loads(Path(p).read_text())['samples'] for p in paths],[])};"
        "out.write_text(json.dumps(m));return m\n"
        "def load_dataset(p): return json.loads(Path(p).read_text())\n"
        "def load_pair(r): return r\n"
    )
    package = repo / "src/py21cmfast"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('__version__="test.1"\n')
    release = _commit(repo)
    (code / "gadi_prepare.sh").write_text(prepare.replace("RELEASE_ANCHOR", release))
    revision = _commit(repo)
    _git(repo, "tag", "v4.3.1", release)

    packages = tmp_path / "mock_packages"
    packages.mkdir()
    for name in ("scipy", "h5py"):
        path = packages / name
        path.mkdir()
        (path / "__init__.py").write_text('__version__="test.1"\n')
    parent = tmp_path / "old_runtime"
    (parent / "venv/bin").mkdir(parents=True)
    (parent / "runtime.env").write_text("PC_MODULES=(mock-python)\n")
    wrapper = (
        "#!/usr/bin/env bash\nset -eu\n"
        "if [[ ${1:-} == -m && ${2:-} == venv ]]; then\n"
        ' mkdir -p "$3/bin"; cp "$0" "$3/bin/python"; exit 0; fi\n'
        "if [[ ${1:-} == -m && ${2:-} == pip ]]; then\n"
        " printf '%s\\n' \"$*\" >> \"$TEST_PIP_LOG\"\n"
        " if [[ ${3:-} == install && \"$*\" == *' -e '* ]]; then\n"
        '  target=; for arg in "$@"; do if [[ $arg == */source ]]; then target=$arg; fi; done\n'
        '  printf compiled > "$target/src/py21cmfast/native.so"\n'
        " fi\n"
        " if [[ ${3:-} == freeze ]]; then echo 'mock==1'; fi; exit 0; fi\n"
        'run=$(dirname "$(dirname "$(dirname "$0")")")\n'
        f'export PYTHONPATH="$run/source/src":{shlex.quote(str(packages))}\n'
        f'exec {shlex.quote(sys.executable)} "$@"\n'
    )
    _script(parent / "venv/bin/python", wrapper)
    binary = tmp_path / "bin"
    binary.mkdir()
    for name in ("module", "gfortran", "mpifort"):
        _script(binary / name, "#!/bin/bash\necho 'test compiler 1'\n")
    _script(binary / "qsub", '#!/bin/bash\nprintf "%s\\n" "$@" > "$TEST_QSUB"\necho 123.gadi-pbs\n')
    env = {
        k: v for k, v in os.environ.items()
        if not k.startswith(("ES_PC_", "C2_")) and k not in ("PYTHONPATH", "PYTHONHOME")
    }
    env.update(
        PATH=f"{binary}:{env['PATH']}", ES_PC_RUNTIME_PARENT=str(parent),
        ES_PC_UPSTREAM_REPO=str(upstream), ES_PC_RUN_ROOT=str(tmp_path / "runs"),
        TEST_PIP_LOG=str(tmp_path / "pip.log"), TEST_QSUB=str(tmp_path / "qsub.args"),
    )
    return {"repo": repo, "code": code, "env": env, "revision": revision, "parent": parent}


def _prepare(fixture, *options):
    return subprocess.run(
        ["bash", str(fixture["code"] / "gadi_prepare.sh"), *options],
        env=fixture["env"], text=True, capture_output=True, check=False,
    )


def _run_directory(fixture):
    runs = list(Path(fixture["env"]["ES_PC_RUN_ROOT"]).iterdir())
    assert len(runs) == 1
    return runs[0]


def _dispatch(fixture, run, cpus=32):
    return subprocess.run(
        ["bash", str(run / "submit.pbs")],
        env=fixture["env"] | {"RUN_DIR": str(run), "PBS_JOBID": "123.gadi-pbs", "PBS_NCPUS": str(cpus)},
        text=True, capture_output=True, check=False,
    )


def test_preparation_builds_isolated_runtime_and_submits(gadi_fixture):
    """The old interpreter remains untouched and all seeds are frozen before qsub."""
    result = _prepare(gadi_fixture, "--submit")
    assert result.returncode == 0, result.stdout + result.stderr
    run = _run_directory(gadi_fixture)
    manifest = json.loads((run / "run_manifest.json").read_text())
    assert manifest["commit"] == gadi_fixture["revision"]
    assert manifest["runtime"]["native_package"] == str(run / "source/src/py21cmfast")
    assert manifest["config"]["splits"]["test"] == [4567]
    assert (run / "PREPARED").is_file()
    assert (run / "job_id.txt").read_text().strip() == "123.gadi-pbs"
    assert not (gadi_fixture["parent"] / "source").exists()
    args = Path(gadi_fixture["env"]["TEST_QSUB"]).read_text()
    assert "ncpus=32,mem=128GB,walltime=48:00:00,jobfs=4GB" in args


def test_compute_dispatch_never_installs_packages(gadi_fixture):
    """Actual scripts run the coordinator and preserve the prepared native ABI."""
    result = _prepare(gadi_fixture)
    assert result.returncode == 0, result.stdout + result.stderr
    run = _run_directory(gadi_fixture)
    pip_log = Path(gadi_fixture["env"]["TEST_PIP_LOG"])
    installed = pip_log.read_bytes()
    result = _dispatch(gadi_fixture, run)
    log = (run / "run_123.gadi-pbs.log").read_text()
    assert result.returncode == 0, result.stdout + result.stderr + log
    assert (run / "COMPLETED").is_file()
    manifest = json.loads((run / "results/dataset_manifest.json").read_text())
    assert len(manifest["samples"]) == 4
    assert pip_log.read_bytes() == installed


@pytest.mark.parametrize("change", ["allocation", "config", "native"])
def test_dispatch_rejects_allocation_configuration_or_native_mutation(gadi_fixture, change):
    """No changed runtime or job can publish a completed training dataset."""
    prepared = _prepare(gadi_fixture)
    assert prepared.returncode == 0, prepared.stdout + prepared.stderr
    run = _run_directory(gadi_fixture)
    cpus = 8 if change == "allocation" else 32
    if change == "config":
        (run / "dataset_config.json").write_text("{}\n")
    if change == "native":
        (run / "source/src/py21cmfast/native.so").write_bytes(b"changed compiler product")
    result = _dispatch(gadi_fixture, run, cpus=cpus)
    assert result.returncode != 0
    assert (run / "FAILED").is_file()
    assert not (run / "COMPLETED").exists()
    assert not (run / "results/COMPLETED").exists()
