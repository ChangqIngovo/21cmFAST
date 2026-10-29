"""Freeze native-budget coeval RT inputs and verify the compute-node runtime.

Preparation performs no cosmological simulation. The existing dependency
environment is reused, while the native extension is built from the exact
archived source on the compute node before any scientific fields are generated.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import shutil
import sys
import tempfile
from itertools import product
from pathlib import Path

from packaging.version import InvalidVersion, Version

from research.pc_matched.prepare_run import (
    _digest,
    _footprint,
    _write_json,
    runtime_fingerprint,
)

EXPERIMENT = "native-budget-coeval-rt"
SCOPE = (
    "Independent native cumulative-budget coeval direct RT diagnostic; each "
    "snapshot retains its own native source, density and supported input flags. "
    "Relaxation time is not cosmological time; this is not a continuous RT "
    "history or support for the full 21cmFAST physics configuration."
)


def _values(value, cast):
    return [cast(item) for item in value.replace(",", " ").split()]


def configuration(environment):
    """Translate explicit environment settings into validated workflow inputs."""
    from .workflow import configuration as scientific_configuration

    defaults = scientific_configuration()
    overrides = {}
    integer_keys = ("dimension", "highres_dimension", "threads", "max_steps")
    float_keys = (
        "box_len", "photon_energy_ev", "exhaustion_tolerance",
        "max_box_crossings", "injection_box_crossings",
    )
    for key in integer_keys + float_keys:
        variable = "PC_" + key.upper()
        if variable in environment:
            cast = int if key in integer_keys else float
            overrides[key] = cast(environment[variable])
    for key, cast in (("seeds", int), ("redshifts", float)):
        if "PC_" + key.upper() in environment:
            overrides[key] = _values(environment["PC_" + key.upper()], cast)
    if environment.get("PC_NATIVE_INPUTS_JSON"):
        overrides["native_inputs_json"] = str(
            Path(environment["PC_NATIVE_INPUTS_JSON"]).resolve()
        )
    axes = ("PC_DIRECTIONS", "PC_REFINEMENTS", "PC_CFL_VALUES")
    if any(key in environment for key in axes):
        baseline = defaults["settings"][0]
        directions = _values(environment.get(axes[0], str(baseline["directions"])), int)
        refinements = _values(environment.get(axes[1], str(baseline["refinement"])), int)
        cfl_values = _values(environment.get(axes[2], str(baseline["cfl"])), float)
        overrides["settings"] = [
            {"directions": d, "refinement": f, "cfl": c}
            for d, f, c in product(directions, refinements, cfl_values)
        ]
    return scientific_configuration(overrides)


def resource_estimate(config):
    """Bound the dominant sequential solver arrays without claiming a runtime."""
    largest = 0
    for setting in config["settings"]:
        cells = (config["dimension"] * setting["refinement"]) ** 3
        # Three angular fields plus conservative allowance for scalar arrays.
        largest = max(largest, (3 * setting["directions"] + 40) * cells * 8)
    return {
        "maximum_solver_array_gib": largest / 2**30,
        "cases": len(config["seeds"]) * len(config["redshifts"]) * len(config["settings"]),
        "scope": "Sequential solver arrays only; native ICs, native output objects, "
                 "compression, package and compiler memory are additional.",
    }


def validate_allocation(config, environment):
    """Reject impossible requested memory budgets before PBS submission."""
    budget = float(environment.get("PC_MEMORY_GB", "24"))
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("PC_MEMORY_GB must be positive and finite")
    allocation = environment.get("PC_MEM", "32GB")
    matched = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*(GB|GIB|MB|MIB|TB|TIB)", allocation.upper())
    if matched is None:
        raise ValueError("PC_MEM must be a positive MB/GB/TB allocation")
    factors = {"GB": 10**9, "GIB": 2**30, "MB": 10**6, "MIB": 2**20,
               "TB": 10**12, "TIB": 2**40}
    allocated = float(matched[1]) * factors[matched[2]]
    if budget * 2**30 + 2**30 > allocated:
        raise ValueError("PC_MEMORY_GB must leave at least 1 GiB below PC_MEM")
    estimate = resource_estimate(config)
    if estimate["maximum_solver_array_gib"] > budget:
        raise ValueError("Estimated solver arrays exceed PC_MEMORY_GB")
    return estimate | {"solver_budget_gib": budget, "pbs_memory": allocation}


def validate_native_version(source, version):
    """Require truthful runtime metadata compatible with the archived HDF format.

    A shallow, untagged SCM build can report ``0.0.1.dev1`` even though the
    sources write modern HDF files. The native compatibility reader then cannot
    reload its own outputs. Reject this before submitting a job; do not change
    the scientific package or invent a release version.
    """
    required = []
    for path in (Path(source) / "src/py21cmfast/io/_compat").glob("_v*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "FormatRelease" and len(node.args) >= 2
                    and isinstance(node.args[1], ast.Constant)
                    and isinstance(node.args[1].value, str)):
                required.append(Version(node.args[1].value))
    minimum = max(required, default=Version("4.0.0"))
    try:
        current = Version(version)
    except InvalidVersion as error:
        raise ValueError("Native runtime has invalid version metadata: " + str(version)) from error
    if current < minimum or current.major != minimum.major:
        raise ValueError(
            f"Native runtime reports {current}, but archived sources use HDF format {minimum}. "
            "No job was submitted. Select a compatible existing PC_PYTHON, or rebuild the "
            "native package from its complete Git ancestry and genuine upstream tags so "
            "setuptools_scm records its actual version. Do not manually relabel an old binary."
        )
    return str(minimum)


def prepare(run, environment):
    """Write an immutable manifest for an already archived Git revision."""
    run = Path(run).resolve()
    if (run / "PREPARED").exists() or (run / "run_manifest.json").exists():
        raise FileExistsError("Use a fresh preparation directory")
    config = configuration(environment)
    estimate = validate_allocation(config, environment)
    files = ["experiment_config.json", "run_manifest.json", "python_path.txt",
             "commit.txt", "runtime.env", "submit.pbs"]
    if config["native_inputs_json"] is not None:
        original = Path(config["native_inputs_json"])
        value = json.loads(original.read_text())
        if not isinstance(value, dict):
            raise ValueError("PC_NATIVE_INPUTS_JSON must contain an input dictionary")
        destination = run / "native_inputs.json"
        shutil.copyfile(original, destination)
        config["native_inputs_json"] = str(destination)
        files.append(destination.name)
    python = (run / "python_path.txt").read_text().strip()
    runtime = runtime_fingerprint(run / "source", python, require_native_match=False)
    format_version = validate_native_version(run / "source", runtime["versions"]["py21cmfast"])
    _write_json(run / "experiment_config.json", config)
    manifest = {
        "schema": 1, "experiment": EXPERIMENT, "run": str(run),
        "commit": (run / "commit.txt").read_text().strip(), "python": python,
        "configuration": config, "resource_estimate": estimate,
        "source_sha256": _footprint(run / "source"), "runtime": runtime,
        "native_rebuild": True,
        "native_rebuild_reason": "Always compile the frozen native source to establish binary provenance",
        "native_hdf_format_minimum_version": format_version,
        "scope": SCOPE,
    }
    _write_json(run / "run_manifest.json", manifest)
    (run / "prepared.sha256").write_text(
        "".join(f"{_digest(run / name)}  {name}\n" for name in files)
    )
    (run / "PREPARED").write_text("Prepared; no cosmological calculation has run.\n")
    return manifest


def _verify_prepared(run, cpus=None):
    run = Path(run).resolve()
    if not (run / "PREPARED").is_file():
        raise ValueError("Run has not been prepared")
    for line in (run / "prepared.sha256").read_text().splitlines():
        sha, name = line.split("  ", 1)
        if Path(name).name != name or not (run / name).is_file() or _digest(run / name) != sha:
            raise ValueError("Prepared file changed: " + name)
    manifest = json.loads((run / "run_manifest.json").read_text())
    if manifest.get("schema") != 1 or manifest.get("experiment") != EXPERIMENT:
        raise ValueError("Unexpected preparation manifest")
    if manifest["run"] != str(run):
        raise ValueError("Prepared run moved")
    if cpus is not None and cpus != manifest["configuration"]["threads"]:
        raise ValueError("CPU allocation differs from prepared configuration")
    if _footprint(run / "source") != manifest["source_sha256"]:
        raise ValueError("Frozen source archive changed")
    return manifest


def verify(run, cpus=None):
    """Verify the frozen files and original dependency environment."""
    run = Path(run).resolve()
    manifest = _verify_prepared(run, cpus)
    actual = runtime_fingerprint(run / "source", manifest["python"], require_native_match=False)
    if actual != manifest["runtime"]:
        raise ValueError("Base runtime or compiler changed")
    return manifest


def _scientific_fingerprint(run, manifest):
    actual = runtime_fingerprint(run / "source", manifest["python"])
    expected = run / "native_build/source/src/py21cmfast"
    if actual["native_package"] != str(expected):
        raise ValueError("Scientific imports must use the isolated native build")
    return actual


def register_native(run, cpus=None):
    """Pin the actual source-matched extension before creating scientific output."""
    run = Path(run).resolve()
    destination = run / "native_runtime"
    if destination.exists():
        return verify_native(run, cpus)
    if (run / "results").exists():
        raise ValueError("Existing results have no registered scientific runtime")
    manifest = _verify_prepared(run, cpus)
    record = {
        "schema": 1, "run": str(run),
        "prepared_manifest_sha256": _digest(run / "run_manifest.json"),
        "runtime": _scientific_fingerprint(run, manifest),
    }
    with tempfile.TemporaryDirectory(prefix=".native_runtime_", dir=run) as temporary:
        staging = Path(temporary)
        _write_json(staging / "manifest.json", record)
        (staging / "manifest.sha256").write_text(_digest(staging / "manifest.json") + "\n")
        staging.replace(destination)
    return record


def verify_native(run, cpus=None):
    """Reject changed native sources, extensions or registered run identity."""
    run = Path(run).resolve()
    manifest = _verify_prepared(run, cpus)
    path = run / "native_runtime/manifest.json"
    checksum = path.with_name("manifest.sha256")
    if not path.is_file() or not checksum.is_file():
        raise ValueError("Scientific runtime registration is missing")
    if _digest(path) != checksum.read_text().strip():
        raise ValueError("Scientific runtime manifest changed")
    record = json.loads(path.read_text())
    if (record.get("schema") != 1 or record.get("run") != str(run)
            or record.get("prepared_manifest_sha256") != _digest(run / "run_manifest.json")):
        raise ValueError("Scientific runtime registration differs from prepared run")
    if _scientific_fingerprint(run, manifest) != record["runtime"]:
        raise ValueError("Scientific runtime changed")
    return record


def main(argv=None):
    """Prepare or verify a native coeval RT run without running the solver."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "register-native", "verify-native"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cpus", type=int)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        manifest = prepare(args.run, os.environ)
        sys.stdout.write(json.dumps({
            "prepared": manifest["run"], "configuration": manifest["configuration"],
            "resource_estimate": manifest["resource_estimate"],
            "native_rebuild_on_compute_node": True, "scope": SCOPE,
        }) + "\n")
    else:
        function = {"verify": verify, "register-native": register_native,
                    "verify-native": verify_native}[args.command]
        function(args.run, args.cpus)


if __name__ == "__main__":
    main()
