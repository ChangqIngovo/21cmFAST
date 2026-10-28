"""Freeze and verify a small native-source matched photon-conservation run.

Preparation reuses an existing native Python environment. It never builds a
new environment or runs native initial conditions on the login node.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

if __package__:
    from .runtime_select import select_runtime
else:
    from runtime_select import select_runtime

DEFAULTS = {
    "dimension": 8,
    "box_len": 50.0,
    "seeds": [111, 222],
    "z_start": 20.0,
    "z_end": 4.0,
    "base_intervals": 10,
    "reference_z": 8.0,
    "directions": [6, 26],
    "refinements": [1, 2],
    "cfl_values": [0.4, 0.2, 0.1],
    "photon_energy_ev": 18.0,
    "threads": 16,
    "memory_gb": 48.0,
}


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def configuration(environment: Mapping[str, str]) -> dict:
    """Read and validate the small-run configuration from PC environment options.

    Parameters
    ----------
    environment : mapping of str to str
        Environment values, normally ``os.environ``.

    Returns
    -------
    dict
        Validated arguments for the matched workflow.
    """
    config = {}
    for key, default in DEFAULTS.items():
        value = environment.get("PC_" + key.upper())
        if value is None:
            config[key] = default.copy() if isinstance(default, list) else default
        elif isinstance(default, list):
            config[key] = [type(default[0])(item) for item in value.replace(",", " ").split()]
        else:
            config[key] = type(default)(value)
    for key in ("dimension", "base_intervals", "threads"):
        if config[key] < (4 if key == "dimension" else 1):
            raise ValueError(f"Invalid {key}")
    if config["threads"] > 32:
        raise ValueError("threads must not exceed 32")
    for key in ("seeds", "directions", "refinements"):
        values = config[key]
        if not values or len(values) != len(set(values)) or min(values) < 1:
            raise ValueError(f"{key} must be distinct positive integers")
    if len(config["directions"]) < 2 or not set(config["directions"]) <= {6, 26}:
        raise ValueError("directions must contain both supported quadratures: 6 and 26")
    cfl_values = config["cfl_values"]
    if (len(cfl_values) < 2 or len(cfl_values) != len(set(cfl_values))
            or any(not math.isfinite(value) or not 0 < value <= 1 for value in cfl_values)):
        raise ValueError("cfl_values must contain at least two distinct finite values in (0, 1]")
    for key in ("box_len", "z_start", "z_end", "reference_z", "memory_gb", "photon_energy_ev"):
        if not math.isfinite(config[key]):
            raise ValueError(f"Invalid {key}")
    if config["box_len"] <= 0 or config["memory_gb"] <= 0:
        raise ValueError("box_len and memory_gb must be positive")
    if config["photon_energy_ev"] < 13.6:
        raise ValueError("photon_energy_ev must reach the hydrogen ionization threshold")
    allocation = environment.get("PC_MEM", "64GB")
    matched = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*(GB|GIB|MB|MIB|TB|TIB)", allocation.upper())
    if matched is None:
        raise ValueError("PC_MEM must specify a positive MB/GB/TB allocation")
    factors = {"GB": 10**9, "GIB": 2**30, "MB": 10**6, "MIB": 2**20,
               "TB": 10**12, "TIB": 2**40}
    allocated_bytes = float(matched[1]) * factors[matched[2]]
    if config["memory_gb"] * 2**30 + 2**30 > allocated_bytes:
        raise ValueError("PC_MEMORY_GB must leave at least 1 GiB below PC_MEM")
    if not config["z_start"] > config["z_end"] >= 0:
        raise ValueError("redshift interval must descend to a nonnegative endpoint")
    if not config["z_end"] <= config["reference_z"] <= config["z_start"]:
        raise ValueError("reference_z must lie within the simulated interval")
    return config


def _footprint(directory: Path) -> dict[str, str]:
    return {str(path.relative_to(directory)): _digest(path)
            for path in sorted(directory.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts
            and path.suffix not in (".pyc", ".pyo")}


def native_sources(archived: Path, installed: Path) -> dict[str, str]:
    """Require every archived native package file to match the installed copy.

    Parameters
    ----------
    archived, installed : pathlib.Path
        Frozen and imported native-package directories.

    Returns
    -------
    dict
        SHA256 hashes of the shared native source files.
    """
    expected = _footprint(archived)
    if not expected:
        raise ValueError("No archived native sources found")
    for relative, sha in expected.items():
        path = installed / relative
        if not path.is_file() or _digest(path) != sha:
            raise ValueError("Installed native source differs from archived source: " + relative)
    return expected


def runtime_fingerprint(source: Path, python: str, *, require_native_match: bool = True) -> dict:
    """Capture imported package, compiler and executable identities.

    Parameters
    ----------
    source : pathlib.Path
        Frozen repository archive.
    python : str
        Selected absolute interpreter path.
    require_native_match : bool, optional
        Require imported native sources to match the archive. False is only
        for recording and checking the dependency environment before a build.

    Returns
    -------
    dict
        Versions and content hashes needed to recheck the compute runtime.
    """
    if str(Path(sys.executable).absolute()) != python:
        raise ValueError("Preparation must run with the selected native Python")
    versions = {"python": platform.python_version()}
    for name in ("numpy", "scipy", "h5py", "matplotlib", "py21cmfast"):
        module = importlib.import_module(name)
        versions[name] = str(module.__version__)
    native = Path(importlib.import_module("py21cmfast").__file__).resolve().parent
    shared = _footprint(source / "src/py21cmfast")
    if not shared:
        raise ValueError("No archived native sources found")
    footprint = _footprint(native)
    mismatches = [name for name, sha in shared.items() if footprint.get(name) != sha]
    if require_native_match and mismatches:
        raise ValueError("Installed native source differs from archived source: " + mismatches[0])
    if not any(name.endswith((".so", ".pyd", ".dylib")) for name in footprint):
        raise ValueError("The selected runtime has no compiled native extension")
    compiler = shutil.which("g++")
    if compiler is None:
        raise ValueError("g++ is required by the frozen native runtime contract")
    compiler_path = Path(compiler).resolve()
    return {
        "versions": versions,
        "python_binary": str(Path(sys.executable).resolve()),
        "python_sha256": _digest(Path(sys.executable).resolve()),
        "compiler": str(compiler_path),
        "compiler_sha256": _digest(compiler_path),
        "compiler_version": subprocess.check_output(
            [compiler, "-dumpfullversion", "-dumpversion"], text=True).strip(),
        "native_package": str(native),
        "native_source_matches": not mismatches,
        "native_source_mismatches": mismatches,
        "native_shared_sources": shared,
        "native_footprint": footprint,
    }


def prepare(run: Path, environment: Mapping[str, str]) -> dict:
    """Write an immutable preparation manifest for an already archived checkout.

    Parameters
    ----------
    run : pathlib.Path
        Fresh prepared-run directory containing source and runtime pointers.
    environment : mapping of str to str
        Experiment configuration overrides.

    Returns
    -------
    dict
        Written manifest, including source and runtime fingerprints.
    """
    run = run.resolve()
    if (run / "PREPARED").exists() or (run / "run_manifest.json").exists():
        raise FileExistsError("Use a fresh run directory")
    config = configuration(environment)
    python = (run / "python_path.txt").read_text().strip()
    fingerprint = runtime_fingerprint(run / "source", python, require_native_match=False)
    _write_json(run / "experiment_config.json", config)
    manifest = {
        "schema": 2,
        "experiment": "matched-photon-conservation",
        "run": str(run),
        "commit": (run / "commit.txt").read_text().strip(),
        "python": python,
        "configuration": config,
        "source_sha256": _footprint(run / "source"),
        "runtime": fingerprint,
        "native_rebuild": not fingerprint["native_source_matches"],
        "scope": "Direct finite-volume discrete-ordinate radiative transfer with native effective H+He counts and a single H-opacity absorber, sharing prescribed native sources and frozen gas with matched ES; monochromatic, no recombination, heating or feedback; native ES is contextual",
        "completion_is_convergence": False,
    }
    _write_json(run / "run_manifest.json", manifest)
    files = ("experiment_config.json", "run_manifest.json", "python_path.txt",
             "commit.txt", "runtime.env", "submit.pbs")
    (run / "prepared.sha256").write_text("".join(f"{_digest(run / name)}  {name}\n" for name in files))
    (run / "PREPARED").write_text("Prepared; native simulations have not started.\n")
    return manifest


def _verify_prepared(run: Path, cpus: int | None = None) -> dict:
    run = run.resolve()
    if not (run / "PREPARED").is_file():
        raise ValueError("Run has not been prepared")
    for line in (run / "prepared.sha256").read_text().splitlines():
        sha, name = line.split("  ", 1)
        if Path(name).name != name or not (run / name).is_file() or _digest(run / name) != sha:
            raise ValueError("Prepared file changed: " + name)
    manifest = json.loads((run / "run_manifest.json").read_text())
    if manifest.get("schema") != 2 or manifest.get("experiment") != "matched-photon-conservation":
        raise ValueError("Unexpected preparation manifest")
    if manifest["run"] != str(run):
        raise ValueError("Prepared run moved")
    if cpus is not None and cpus != manifest["configuration"]["threads"]:
        raise ValueError("CPU allocation differs from prepared configuration")
    if _footprint(run / "source") != manifest["source_sha256"]:
        raise ValueError("Frozen source archive changed")
    return manifest


def verify(run: Path, *, cpus: int | None = None) -> dict:
    """Reject changed prepared inputs or base dependency environment.

    Parameters
    ----------
    run : pathlib.Path
        Prepared run to verify.
    cpus : int, optional
        Allocated CPUs; must equal the frozen worker count.

    Returns
    -------
    dict
        Verified preparation manifest.
    """
    run = run.resolve()
    manifest = _verify_prepared(run, cpus)
    if runtime_fingerprint(run / "source", manifest["python"], require_native_match=False) != manifest["runtime"]:
        raise ValueError("Native runtime or compiler changed")
    return manifest


def _scientific_fingerprint(run: Path, manifest: dict) -> dict:
    fingerprint = runtime_fingerprint(run / "source", manifest["python"])
    if manifest["native_rebuild"]:
        expected = run / "native_build/source/src/py21cmfast"
        if fingerprint["native_package"] != str(expected):
            raise ValueError("Scientific imports must use the isolated native build")
    elif fingerprint != manifest["runtime"]:
        raise ValueError("Scientific runtime changed from the prepared base runtime")
    return fingerprint


def register_scientific_runtime(run: Path, *, cpus: int | None = None) -> dict:
    """Atomically pin the exact native implementation used for scientific fields.

    Parameters
    ----------
    run : pathlib.Path
        Prepared run, with any required isolated native build completed.
    cpus : int, optional
        Allocated CPU count to compare with the frozen configuration.

    Returns
    -------
    dict
        Immutable scientific runtime identity; reused strictly on restart.
    """
    run = run.resolve()
    destination = run / "native_runtime"
    if destination.exists():
        return verify_scientific_runtime(run, cpus=cpus)
    if (run / "results").exists():
        raise ValueError("Existing results have no registered scientific runtime")
    manifest = _verify_prepared(run, cpus)
    fingerprint = _scientific_fingerprint(run, manifest)
    record = {
        "schema": 1,
        "run": str(run),
        "prepared_manifest_sha256": _digest(run / "run_manifest.json"),
        "runtime": fingerprint,
    }
    with tempfile.TemporaryDirectory(prefix=".native_runtime_", dir=run) as temporary:
        staging = Path(temporary)
        _write_json(staging / "manifest.json", record)
        (staging / "manifest.sha256").write_text(_digest(staging / "manifest.json") + "\n")
        staging.replace(destination)
    return record


def verify_scientific_runtime(run: Path, *, cpus: int | None = None) -> dict:
    """Check scientific source and extension hashes against the registered run.

    Parameters
    ----------
    run : pathlib.Path
        Prepared run with a registered scientific runtime.
    cpus : int, optional
        Allocated CPU count to compare with the frozen configuration.

    Returns
    -------
    dict
        Verified scientific runtime identity.
    """
    run = run.resolve()
    manifest = _verify_prepared(run, cpus)
    directory = run / "native_runtime"
    path = directory / "manifest.json"
    if not path.is_file() or not (directory / "manifest.sha256").is_file():
        raise ValueError("Scientific runtime registration is missing or incomplete")
    if _digest(path) != (directory / "manifest.sha256").read_text().strip():
        raise ValueError("Scientific runtime manifest changed")
    record = json.loads(path.read_text())
    if (record.get("schema") != 1 or record.get("run") != str(run)
            or record.get("prepared_manifest_sha256") != _digest(run / "run_manifest.json")):
        raise ValueError("Scientific runtime registration does not match prepared run")
    if _scientific_fingerprint(run, manifest) != record["runtime"]:
        raise ValueError("Scientific runtime changed")
    return record


def main(argv: list[str] | None = None) -> None:
    """Run runtime discovery, preparation, verification or argument emission.

    Parameters
    ----------
    argv : list of str, optional
        Command arguments; defaults to the current process arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="command", required=True)
    selection = modes.add_parser("select")
    selection.add_argument("--root", type=Path, required=True)
    selection.add_argument("--parent", type=Path)
    selection.add_argument("--python", type=Path)
    selection.add_argument("--runtime-env", type=Path)
    for mode in ("prepare", "verify", "register-native", "verify-native", "arguments"):
        command = modes.add_parser(mode)
        command.add_argument("--run", type=Path, required=True)
        if mode in ("verify", "register-native", "verify-native"):
            command.add_argument("--cpus", type=int)
    args = parser.parse_args(argv)
    if args.command == "select":
        result = select_runtime(args.root, parent=args.parent, python=args.python, runtime_env=args.runtime_env)
        sys.stdout.write("\n".join(result[name] for name in ("python", "parent", "runtime_env")) + "\n")
    elif args.command == "prepare":
        result = prepare(args.run, os.environ)
        sys.stdout.write(json.dumps({"prepared": result["run"], "configuration": result["configuration"],
                                   "native_rebuild_on_compute_node": result["native_rebuild"]}) + "\n")
    elif args.command == "verify":
        verify(args.run, cpus=args.cpus)
    elif args.command == "register-native":
        register_scientific_runtime(args.run, cpus=args.cpus)
    elif args.command == "verify-native":
        verify_scientific_runtime(args.run, cpus=args.cpus)
    else:
        config = json.loads((args.run / "experiment_config.json").read_text())
        for key, value in config.items():
            sys.stdout.write("--" + key.replace("_", "-") + "\n")
            sys.stdout.write("\n".join(str(item) for item in (value if isinstance(value, list) else [value])) + "\n")


if __name__ == "__main__":
    main()
