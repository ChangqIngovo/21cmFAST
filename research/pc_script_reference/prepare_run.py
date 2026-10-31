"""Freeze and verify a small original-SCRIPT run, without building native RT."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def source_fingerprint(source):
    """Include every archived source file; bytecode is disabled by the launcher."""
    return {
        str(path.relative_to(source)): digest(path)
        for path in sorted(source.rglob("*"))
        if path.is_file()
    }


def runtime_fingerprint(input_run):
    # Imports deliberately preflight the dependencies before a job is submitted.
    import matplotlib
    import numpy

    native_root = Path(input_run) / "native_build/source/src"
    native_package = native_root / "py21cmfast"
    if not (native_package / "c_21cmfast.so").is_file():
        raise FileNotFoundError("Existing run has no frozen native binary: " + str(native_package))
    sys.path.insert(0, str(native_root))
    import py21cmfast

    if Path(py21cmfast.__file__).resolve().parent != native_package.resolve():
        raise ValueError("Native import did not use the input run's frozen build")

    compiler = shutil.which(os.environ.get("SCRIPT_FC", "gfortran"))
    if compiler is None:
        raise RuntimeError("gfortran is unavailable; load gcc/12.2.0 before preparing")
    compiler = str(Path(compiler).resolve())
    compiler_version = subprocess.run(
        [compiler, "--version"], check=True, capture_output=True, text=True,
        timeout=15,
    ).stdout
    return {
        "python": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "numpy_version": numpy.__version__,
        "matplotlib_version": matplotlib.__version__,
        "py21cmfast_version": py21cmfast.__version__,
        "native_package": str(native_package.resolve()),
        "native_binary_sha256": digest(native_package / "c_21cmfast.so"),
        "gfortran": compiler,
        "gfortran_sha256": digest(compiler),
        "gfortran_version": compiler_version,
    }


def prepare(run):
    run = Path(run).resolve()
    if (run / "PREPARED").exists() or (run / "run_manifest.json").exists():
        raise FileExistsError("Prepare a fresh run directory")
    input_run = Path((run / "input_run.txt").read_text().strip()).resolve()
    if input_run == run or input_run in run.parents:
        raise ValueError("Write reference results outside the archived input run")
    for name in ("experiment_config.json", "runtime.env", "python_path.txt"):
        if not (input_run / name).is_file():
            raise FileNotFoundError(input_run / name)
    if not list((input_run / "results/histories").glob("seed*/history.npz")):
        raise ValueError("No completed native history archive in input run")
    config = json.loads((input_run / "experiment_config.json").read_text())
    workers = config.get("threads")
    if type(workers) is not int or workers < 1:
        raise ValueError("Input experiment must record a positive integer threads count")
    manifest = {
        "schema": 1,
        "experiment": "original-script-native-archive-reference",
        "run": str(run),
        "input_run": str(input_run),
        "commit": (run / "commit.txt").read_text().strip(),
        "source_sha256": source_fingerprint(run / "source"),
        "runtime": runtime_fingerprint(input_run),
        "input_config_sha256": digest(input_run / "experiment_config.json"),
        "native_build": False,
        "ncpus": workers,
        "scope": "Original SCRIPT applied to existing native archives; no new RT history is evolved.",
    }
    (run / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    files = ("commit.txt", "input_run.txt", "python_path.txt", "runtime.env", "submit.pbs", "run_manifest.json")
    (run / "prepared.sha256").write_text(
        "".join(f"{digest(run / name)}  {name}\n" for name in files)
    )
    (run / "PREPARED").write_text("Prepared; no scientific calculation has run.\n")
    return manifest


def verify(run, cpus=None):
    run = Path(run).resolve()
    if not (run / "PREPARED").is_file():
        raise ValueError("Run is not prepared")
    for line in (run / "prepared.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if Path(name).name != name or digest(run / name) != expected:
            raise ValueError("Prepared file changed: " + name)
    manifest = json.loads((run / "run_manifest.json").read_text())
    if manifest.get("schema") != 1 or manifest.get("experiment") != "original-script-native-archive-reference":
        raise ValueError("Unknown prepared run format")
    if manifest["run"] != str(run):
        raise ValueError("Prepared run moved")
    if cpus is not None and cpus != manifest["ncpus"]:
        raise ValueError("CPU allocation differs from archived native thread count")
    if manifest["source_sha256"] != source_fingerprint(run / "source"):
        raise ValueError("Frozen source changed")
    input_run = Path(manifest["input_run"])
    if manifest["runtime"] != runtime_fingerprint(input_run):
        raise ValueError("Compiler, native binary or Python dependency runtime changed")
    if manifest["input_config_sha256"] != digest(input_run / "experiment_config.json"):
        raise ValueError("Input run configuration changed")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cpus", type=int)
    args = parser.parse_args()
    manifest = prepare(args.run) if args.command == "prepare" else verify(args.run, args.cpus)
    print(json.dumps({"command": args.command, "commit": manifest["commit"], "input_run": manifest["input_run"]}))


if __name__ == "__main__":
    main()
