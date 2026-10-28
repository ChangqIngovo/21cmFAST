"""Compile a frozen native package locally without altering its base environment.

Only the existing CFFI builder is executed. Python dependencies and compilers
come from the selected runtime; no package manager or network is used. Activate
the result in subsequent processes with ``PYTHONPATH=RUN/native_build/source/src``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

if __package__:
    from .prepare_run import native_sources
else:
    from prepare_run import native_sources


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _footprint(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): _digest(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in (".pyc", ".pyo")
    }


def _base_identity() -> dict:
    package = Path(importlib.import_module("py21cmfast").__file__).resolve().parent
    compiler = shutil.which(shlex.split(os.environ.get("CC", "gcc"))[0])
    if compiler is None:
        raise ValueError("A C compiler is required for the isolated native build")
    variables = (
        "CC",
        "CXX",
        "CFLAGS",
        "CPPFLAGS",
        "LDFLAGS",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "CPATH",
        "C_INCLUDE_PATH",
        "LOG_LEVEL",
        "DEBUG",
        "PROFILE",
    )
    return {
        "python": str(Path(sys.executable).absolute()),
        "python_sha256": _digest(Path(sys.executable).resolve()),
        "compiler": str(Path(compiler).resolve()),
        "compiler_sha256": _digest(Path(compiler).resolve()),
        "build_dependencies": {
            name: importlib.metadata.version(name) for name in ("cffi", "setuptools")
        },
        "native_package": str(package),
        "native_footprint": _footprint(package),
        "build_environment": {
            name: os.environ[name] for name in variables if name in os.environ
        },
    }


def _compile(source: Path, log: Path) -> Path:
    output = source.parent / "extension_path.txt"
    code = """
import runpy
import sys
from pathlib import Path
source = Path(sys.argv[1])
ffi = runpy.run_path(str(source / "build_cffi.py"))["ffi"]
extension = ffi.compile(tmpdir=str(source), target=str(source / "src/py21cmfast/c_21cmfast.*"), verbose=True)
Path(sys.argv[2]).write_text(str(Path(extension).resolve()) + "\\n")
"""
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    with log.open("w") as stream:
        subprocess.run(
            [sys.executable, "-c", code, str(source), str(output)],
            cwd=source,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    extension = Path(output.read_text().strip())
    if extension.parent != source / "src/py21cmfast" or not extension.is_file():
        raise ValueError("Compiler produced an unexpected native extension path")
    return extension


def _probe(source: Path, extension: Path, log: Path) -> dict:
    output = source.parent / "import_probe.json"
    code = """
import json
import sys
from pathlib import Path
import py21cmfast
from py21cmfast import c_21cmfast
value = {"package": str(Path(py21cmfast.__file__).resolve().parent), "extension": str(Path(c_21cmfast.__file__).resolve()), "version": str(py21cmfast.__version__)}
Path(sys.argv[1]).write_text(json.dumps(value))
"""
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=str(source / "src"),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
    )
    with log.open("a") as stream:
        subprocess.run(
            [sys.executable, "-c", code, str(output)],
            cwd=source,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    result = json.loads(output.read_text())
    if result["package"] != str(source / "src/py21cmfast") or result[
        "extension"
    ] != str(extension):
        raise ValueError("Native import did not use the isolated compiled package")
    return result


def _validate_archive(source: Path) -> dict[str, str]:
    if (
        not (source / "build_cffi.py").is_file()
        or not (source / "src/py21cmfast").is_dir()
    ):
        raise ValueError("Frozen archive lacks the native package or CFFI builder")
    if any(path.is_symlink() for path in source.rglob("*")):
        raise ValueError("Frozen archive must not contain symbolic links")
    footprint = _footprint(source)
    if any(Path(name).suffix in (".so", ".pyd", ".dylib", ".o") for name in footprint):
        raise ValueError("Frozen source archive contains a compiled binary")
    return footprint


def build_native(run: Path) -> dict:
    """Build or strictly verify an isolated extension from a frozen run archive.

    Parameters
    ----------
    run : pathlib.Path
        Prepared directory with a frozen ``source`` repository archive. Invoke
        with its original Python environment, before activating the new package.

    Returns
    -------
    dict
        Completed build manifest, including source, base runtime and binary hashes.

    Notes
    -----
    The caller holds the run's exclusive lock. Incomplete builds are renamed for
    inspection before retry; changed complete builds are rejected, never rebuilt.
    """
    run = run.resolve()
    archived = run / "source"
    archive_hashes = _validate_archive(archived)
    base = _base_identity()
    directory = run / "native_build"
    source = directory / "source"
    manifest_path = directory / "build_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != 1 or manifest.get("run") != str(run):
            raise ValueError("Unexpected isolated native build manifest")
        if manifest["source_sha256"] != archive_hashes:
            raise ValueError("Frozen native build source changed")
        if manifest["base_runtime"] != base:
            raise ValueError("Isolated native build base runtime or compiler changed")
        native_sources(archived / "src/py21cmfast", source / "src/py21cmfast")
        if manifest["built_source_sha256"] != _footprint(source):
            raise ValueError("Completed isolated native build changed")
        extension = source / manifest["extension"]
        if _digest(extension) != manifest["extension_sha256"]:
            raise ValueError("Completed native extension changed")
        _probe(source, extension, directory / "build.log")
        return manifest
    if directory.exists():
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        directory.rename(run / f"native_build.incomplete-{stamp}-{time.time_ns()}")
    directory.mkdir()
    shutil.copytree(
        archived, source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")
    )
    started = time.perf_counter()
    extension = _compile(source, directory / "build.log")
    shared = native_sources(archived / "src/py21cmfast", source / "src/py21cmfast")
    probe = _probe(source, extension, directory / "build.log")
    if _footprint(archived) != archive_hashes or _base_identity() != base:
        raise ValueError("Frozen source or base runtime changed during compilation")
    manifest = {
        "schema": 1,
        "run": str(run),
        "source_sha256": archive_hashes,
        "base_runtime": base,
        "native_sources": shared,
        "built_source_sha256": _footprint(source),
        "extension": str(extension.relative_to(source)),
        "extension_sha256": _digest(extension),
        "import_probe": probe,
        "elapsed_seconds": time.perf_counter() - started,
    }
    temporary = manifest_path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest_path)
    return manifest


def main(argv: list[str] | None = None) -> None:
    """Compile a frozen native run package using the selected Python runtime.

    Parameters
    ----------
    argv : list of str, optional
        Command arguments, defaulting to process arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build_native(args.run)
    sys.stdout.write(
        json.dumps(
            {
                "native_package": manifest["import_probe"]["package"],
                "extension_sha256": manifest["extension_sha256"],
                "elapsed_seconds": manifest["elapsed_seconds"],
            }
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
