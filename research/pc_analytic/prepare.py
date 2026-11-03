"""Freeze a small same-input analytic-PC, SCRIPT and C²-Ray experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)
UPSTREAM_FILES = (
    "LICENSE",
    "src/c2ray/photorates.f90",
    "src/c2ray/raytracing.f90",
    "src/c2ray/chemistry.f90",
)
FROZEN_FILES = (
    "commit.txt",
    "input_run.txt",
    "python_path.txt",
    "runtime.env",
    "submit.pbs",
    "experiment_config.json",
    "run_manifest.json",
)


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _numbers(value: str, name: str, *, integers: bool = False) -> list[int | float]:
    try:
        result = [
            int(item) if integers else float(item)
            for item in value.replace(",", " ").split()
        ]
    except ValueError as error:
        raise ValueError(f"Invalid {name}") from error
    if (
        not result
        or len(set(result)) != len(result)
        or any(not math.isfinite(item) or item < 0 for item in result)
    ):
        raise ValueError(f"{name} requires unique finite nonnegative values")
    return result


def configuration(
    original: dict[str, Any], environment: Mapping[str, str]
) -> dict[str, Any]:
    """Create explicit pilot controls without silently reducing the source grid.

    Parameters
    ----------
    original
        Archived configuration containing the original dimension.
    environment
        PC-prefixed numerical and snapshot-selection overrides.

    Returns
    -------
    dict
        Configuration accepted by the comparison runner.
    """
    n = int(original["dimension"])
    coarse_value = environment.get("PC_COARSEN_TO", "")
    coarse = int(coarse_value) if coarse_value else None
    output_n = n if coarse is None else coarse
    if n < 2 or output_n < 2 or output_n > n or n % output_n:
        raise ValueError(
            "PC_COARSEN_TO must divide the original dimension and lie between 2 and that dimension"
        )
    if output_n > 16:
        raise ValueError(
            "This C²-Ray pilot is limited to N<=16; explicitly set PC_COARSEN_TO=16 for a divisible larger input"
        )
    threads = int(environment.get("PC_THREADS", "8"))
    steps = int(environment.get("PC_EXPOSURE_STEPS", "16"))
    c2_steps = int(environment.get("PC_C2_EXPOSURE_STEPS", "32"))
    c2_tolerance = float(environment.get("PC_C2_TOLERANCE", "1e-6"))
    c2_max_iterations = int(environment.get("PC_C2_MAX_ITERATIONS", "2000"))
    c2_subbox_size = int(environment.get("PC_C2_SUBBOX_SIZE", "2"))
    c2_loss_fraction = float(environment.get("PC_C2_LOSS_FRACTION", "1e-10"))
    boundary = environment.get("PC_BOUNDARY", "source_periodic_cube")
    directions = int(environment.get("PC_DIRECTIONS", "24"))
    sigma = float(environment.get("PC_SIGMA_CM2", str(6.3e-18 * (13.6 / 18.0) ** 3)))
    if (
        not 1 <= threads <= 32
        or not 1 <= steps <= 4096
        or not 1 <= c2_steps <= 4096
        or directions < 1
    ):
        raise ValueError("Invalid thread, direction or exposure-step count")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("PC_SIGMA_CM2 must be finite and positive")
    if not math.isfinite(c2_tolerance) or not 0 < c2_tolerance < 1:
        raise ValueError("PC_C2_TOLERANCE must lie in (0,1)")
    if not 1 <= c2_max_iterations <= 100000:
        raise ValueError("Invalid PC_C2_MAX_ITERATIONS")
    if c2_subbox_size < 1:
        raise ValueError("Invalid PC_C2_SUBBOX_SIZE")
    if not math.isfinite(c2_loss_fraction) or not 0 <= c2_loss_fraction < 1:
        raise ValueError("PC_C2_LOSS_FRACTION must lie in [0,1)")
    if boundary not in {"periodic", "open", "source_periodic_cube"}:
        raise ValueError("Invalid PC_BOUNDARY")
    seeds = environment.get("PC_SEEDS", "")
    return {
        "input_dimension": n,
        "output_dimension": output_n,
        "coarsen_to": coarse,
        "redshifts": sorted(
            _numbers(environment.get("PC_REDSHIFTS", "8,7.5,7"), "PC_REDSHIFTS"),
            reverse=True,
        ),
        "seeds": _numbers(seeds, "PC_SEEDS", integers=True) if seeds else None,
        "threads": threads,
        "require_c2ray": True,
        "brightness": "native",
        "sigma_cm2": sigma,
        "sigma_definition": "Explicit common monochromatic cross-section; default 6.3e-18*(13.6/18)^3 cm²",
        "analytic": {
            "directions": directions,
            "exposure_steps": steps,
            "boundary": boundary,
            "transport_mode": "characteristic",
            "exhaustion_tolerance": 1e-7,
            "max_sweeps": 10000,
        },
        "c2ray": {
            "exposure_steps": c2_steps,
            "tolerance": c2_tolerance,
            "max_iterations": c2_max_iterations,
            "subbox_size": c2_subbox_size,
            "loss_fraction": c2_loss_fraction,
        },
        "script_timeout_seconds": 900,
        "mfp_rays": 4096,
        "coarsening_scope": (
            "No coarsening: exact archived native source and density arrays"
            if coarse is None
            else "Explicit block averages of source/capacity, with larger physical cell volumes preserving total photon/atom counts; all methods share these coarsened inputs"
        ),
    }


def tree_hashes(root: Path) -> dict[str, str]:
    """Hash every regular file in a frozen tree.

    Parameters
    ----------
    root
        Frozen source or upstream directory.

    Returns
    -------
    dict
        Relative file paths mapped to SHA-256 hashes.
    """
    return {
        str(path.relative_to(root)): _digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _runtime() -> dict[str, Any]:
    compilers = {}
    for key, requested in (
        ("CXX", os.environ.get("CXX", "g++")),
        ("SCRIPT_FC", os.environ.get("SCRIPT_FC", "gfortran")),
    ):
        found = shutil.which(requested)
        if found is None:
            raise RuntimeError(f"Required compiler unavailable: {requested}")
        path = Path(found).resolve()
        version = subprocess.run(
            [str(path), "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
        compilers[key] = {
            "path": str(path),
            "sha256": _digest(path),
            "version": version,
        }
    return {
        "python": str(Path(sys.executable).resolve()),
        "python_sha256": _digest(Path(sys.executable).resolve()),
        "compilers": compilers,
        "c2ray_fflags": os.environ.get("FFLAGS", ""),
    }


def _selected_inputs(
    input_run: Path, config: dict[str, Any]
) -> tuple[list[str], dict[str, str]]:
    from research.pc_coeval.run import discover

    _, _, snapshots = discover(input_run, config["redshifts"])
    if config["seeds"] is not None:
        snapshots = [
            item for item in snapshots if item["record"]["seed"] in config["seeds"]
        ]
        if {item["record"]["seed"] for item in snapshots} != set(config["seeds"]):
            raise ValueError("Requested seed is absent from the input archive")
    directories = [Path(item["receipt_path"]).parent for item in snapshots]
    files = {
        str(path): _digest(path)
        for path in (
            input_run / "experiment_config.json",
            input_run / "results" / "manifest.json",
        )
    }
    for directory in directories:
        for name in ("receipt.json", "native.json", "source.npz", "native.h5"):
            path = directory / name
            files[str(path)] = _digest(path)
    return [str(path) for path in directories], files


def _prepare_upstream(run: Path, environment: Mapping[str, str]) -> None:
    from research.pc_analytic.c2ray_reference import fetch_upstream

    destination = run / "c2ray_upstream"
    supplied = environment.get("PC_C2RAY_SOURCE")
    if supplied:
        source = Path(supplied).resolve()
        for name in UPSTREAM_FILES:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, target)
    # The backend verifies pinned hashes of existing files; network is only
    # permitted during preparation when a missing file must be downloaded.
    fetch_upstream(destination)


def prepare(run: Path, environment: Mapping[str, str]) -> dict[str, Any]:
    """Freeze source, native inputs, compiler identities and C²-Ray files.

    Parameters
    ----------
    run
        Fresh run with archived source and shell metadata already installed.
    environment
        Preparation-time configuration and optional upstream path.

    Returns
    -------
    dict
        Immutable run manifest.
    """
    run = Path(run).resolve()
    if (run / "run_manifest.json").exists() or (run / "PREPARED").exists():
        raise FileExistsError(
            "Use a fresh run; never rewrite frozen experiment identities"
        )
    input_run = Path((run / "input_run.txt").read_text().strip()).resolve()
    if run == input_run or input_run in run.parents:
        raise ValueError("Output must be separate from the archived input")
    original = json.loads((input_run / "experiment_config.json").read_text())
    config = configuration(original, environment)
    directories, input_hashes = _selected_inputs(input_run, config)
    _prepare_upstream(run, environment)
    config["input_run"] = str(input_run)
    config["snapshot_directories"] = directories
    config["c2ray"]["upstream"] = str(run / "c2ray_upstream")
    runtime = _runtime()
    config["c2ray"]["compiler"] = runtime["compilers"]["SCRIPT_FC"]["path"]
    _write_json(run / "experiment_config.json", config)
    manifest = {
        "schema": 1,
        "configuration": config,
        "source_sha256": tree_hashes(run / "source"),
        "upstream_sha256": tree_hashes(run / "c2ray_upstream"),
        "input_sha256": input_hashes,
        "runtime": runtime,
        "scope": "Small frozen-native-input analytic PC / original SCRIPT / actual C²-Ray comparison; no convergence certification",
        "network_required_on_compute": False,
    }
    _write_json(run / "run_manifest.json", manifest)
    (run / "prepared.sha256").write_text(
        "".join(f"{_digest(run / name)}  {name}\n" for name in FROZEN_FILES)
    )
    (run / "PREPARED").write_text("Prepared immutable same-input comparison.\n")
    return manifest


def verify(run: Path, cpus: int | None = None) -> dict[str, Any]:
    """Verify a prepared experiment without changing its recorded identities.

    Parameters
    ----------
    run
        Prepared run directory.
    cpus
        Actual scheduler allocation, when invoked on a compute node.

    Returns
    -------
    dict
        Verified run manifest.
    """
    run = Path(run).resolve()
    checksum_rows = (run / "prepared.sha256").read_text().splitlines()
    expected_rows = [f"{_digest(run / name)}  {name}" for name in FROZEN_FILES]
    if checksum_rows != expected_rows:
        raise ValueError("Prepared metadata checksum changed")
    manifest = json.loads((run / "run_manifest.json").read_text())
    if tree_hashes(run / "source") != manifest["source_sha256"]:
        raise ValueError("Frozen source changed; prepare a new run")
    if tree_hashes(run / "c2ray_upstream") != manifest["upstream_sha256"]:
        raise ValueError("Frozen C²-Ray upstream changed")
    if any(
        _digest(Path(name)) != checksum
        for name, checksum in manifest["input_sha256"].items()
    ):
        raise ValueError("Archived native input changed")
    if _runtime() != manifest["runtime"]:
        raise ValueError("Python or compiler identity changed")
    if cpus is not None and cpus != manifest["configuration"]["threads"]:
        raise ValueError("CPU allocation differs from prepared experiment")
    return manifest


def main() -> None:
    """Prepare or verify an immutable analytic comparison from the CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify"))
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--cpus", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    manifest = (
        prepare(args.run, os.environ)
        if args.command == "prepare"
        else verify(args.run, args.cpus)
    )
    LOGGER.info(
        "%s: N=%s, snapshots=%s, actual C²-Ray required",
        args.command,
        manifest["configuration"]["output_dimension"],
        len(manifest["configuration"]["snapshot_directories"]),
    )


if __name__ == "__main__":
    main()
