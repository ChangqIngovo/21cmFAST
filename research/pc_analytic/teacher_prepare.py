"""Freeze a parallel native-input analytic teacher campaign for Gadi."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from research.pc_analytic.compare import validate_physics
from research.pc_analytic.prepare import _digest, _numbers, _write_json, tree_hashes

LOGGER = logging.getLogger(__name__)
FROZEN_FILES = (
    "commit.txt",
    "parent_run.txt",
    "python_path.txt",
    "runtime.env",
    "submit.pbs",
    "native_template.json",
    "experiment_config.json",
    "run_manifest.json",
)


def configuration(template: Path, environment: Mapping[str, str]) -> dict[str, Any]:
    """Translate explicit campaign controls without changing template physics.

    Parameters
    ----------
    template
        Frozen full native InputParameters JSON path.
    environment
        PC-prefixed overrides. PC_THREADS is per worker; PC_NCPUS is the total
        allocation and must cover PC_WORKERS times PC_THREADS.

    Returns
    -------
    dict
        Explicit native geometry, realization selection and analytic controls.
    """
    dimension = int(environment.get("PC_DIMENSION", "32"))
    box = float(environment.get("PC_BOX_LEN", "256"))
    if not math.isfinite(box) or box <= 0:
        raise ValueError("PC_BOX_LEN must be finite and positive")
    high = int(
        environment.get("PC_HIGHRES_DIMENSION", str(max(3 * dimension, math.ceil(box))))
    )
    workers = int(environment.get("PC_WORKERS", "4"))
    threads = int(environment.get("PC_THREADS", "4"))
    cpus = int(environment.get("PC_NCPUS", str(workers * threads)))
    sigma = float(environment.get("PC_SIGMA_CM2", str(6.3e-18 * (13.6 / 18) ** 3)))
    seeds = _numbers(
        environment.get("PC_SEEDS", "1001,1002,1003,1004,1005,1006,1007,1008"),
        "PC_SEEDS",
        integers=True,
    )
    redshifts = sorted(
        _numbers(environment.get("PC_REDSHIFTS", "10,9,8,7.5,7,6.5,6"), "PC_REDSHIFTS"),
        reverse=True,
    )
    directions = int(environment.get("PC_DIRECTIONS", "96"))
    steps = int(environment.get("PC_EXPOSURE_STEPS", "32"))
    exhaustion = float(environment.get("PC_EXHAUSTION_TOLERANCE", "1e-7"))
    tail = float(environment.get("PC_BATCH_TAIL_TOLERANCE", "1e-4"))
    sweeps = int(environment.get("PC_MAX_SWEEPS", "10000"))
    boundary = environment.get("PC_BOUNDARY", "source_periodic_cube")
    if dimension < 2 or high < dimension:
        raise ValueError("Invalid native PC_DIMENSION or PC_HIGHRES_DIMENSION")
    if workers < 1 or not 1 <= threads <= 32 or cpus < workers * threads:
        raise ValueError(
            "PC_NCPUS must cover positive PC_WORKERS * PC_THREADS (threads <=32)"
        )
    if any(seed < 1 for seed in seeds):
        raise ValueError("PC_SEEDS must be positive")
    if any(not math.isfinite(value) or value <= 0 for value in (box, sigma)):
        raise ValueError("PC_BOX_LEN and PC_SIGMA_CM2 must be finite and positive")
    if directions not in (6, 24, 26, 96, 384) or steps < 1 or sweeps < 1:
        raise ValueError("Invalid analytic direction, exposure or sweep count")
    if any(
        not math.isfinite(value) or not 0 < value < 1 for value in (exhaustion, tail)
    ):
        raise ValueError(
            "Analytic exhaustion and batch-tail tolerances must lie in (0,1)"
        )
    if boundary not in ("periodic", "open", "source_periodic_cube"):
        raise ValueError("Invalid PC_BOUNDARY")
    return {
        "native_template_json": str(template.resolve()),
        "dimension": dimension,
        "highres_dimension": high,
        "box_len": box,
        "seeds": seeds,
        "redshifts": redshifts,
        "workers": workers,
        "threads": threads,
        "cpus": cpus,
        "sigma_cm2": sigma,
        "brightness": "native",
        "analytic": {
            "directions": directions,
            "exposure_steps": steps,
            "boundary": boundary,
            "transport_mode": "characteristic",
            "exhaustion_tolerance": exhaustion,
            "batch_tail_tolerance": tail,
            "max_sweeps": sweeps,
        },
    }


def resource_estimate(
    config: dict[str, Any], environment: Mapping[str, str]
) -> dict[str, Any]:
    """Check the declared allocation against concurrent dominant arrays.

    Parameters
    ----------
    config
        Explicit campaign configuration.
    environment
        PC_MEM scheduler allocation and PC_MEMORY_GB working budget in GiB.

    Returns
    -------
    dict
        Memory estimate and declared allocation. The estimate is not a measured
        upper bound on all native library allocations.
    """
    allocation = environment.get("PC_MEM", "64GB")
    matched = re.fullmatch(r"([1-9][0-9]*)(MB|GB|TB)", allocation.upper())
    if matched is None:
        raise ValueError("Invalid memory allocation PC_MEM")
    allocated = int(matched[1]) * {"MB": 10**6, "GB": 10**9, "TB": 10**12}[matched[2]]
    budget = float(environment.get("PC_MEMORY_GB", "48"))
    if not math.isfinite(budget) or budget <= 0 or (budget + 1) * 2**30 > allocated:
        raise ValueError(
            "Working memory budget must leave at least 1 GiB within PC_MEM"
        )
    n, high = config["dimension"], config["highres_dimension"]
    concurrent = min(config["workers"], len(config["seeds"]))
    solver = (3 * config["analytic"]["directions"] + 80) * n**3 * 8
    native = 32 * high**3 * 8
    estimate = concurrent * (solver + native + 2 * 2**30) / 2**30
    if estimate > budget:
        raise ValueError("Estimated concurrent memory exceeds PC_MEMORY_GB")
    return {
        "concurrent_seeds": concurrent,
        "solver_arrays_per_worker_gib": solver / 2**30,
        "native_array_allowance_per_worker_gib": native / 2**30,
        "estimated_concurrent_gib": estimate,
        "working_budget_gib": budget,
        "pbs_memory": allocation,
        "scope": "Dominant solver arrays plus native high-resolution array allowance and 2 GiB overhead per worker; not a measured peak-memory guarantee",
    }


def _checked_path(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Archived receipt path escapes its root")
    return path


def _verify_files(
    root: Path, files: dict[str, str], names: tuple[str, ...]
) -> dict[str, str]:
    verified = {}
    for name in names:
        path = _checked_path(root, name)
        if not path.is_file() or files.get(name) != _digest(path):
            raise ValueError(f"Archived checksum failed: {path}")
        verified[str(path)] = files[name]
    return verified


def select_template(
    parent: Path, seed: int, redshift: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read a full physical template through its archived receipt chain.

    Parameters
    ----------
    parent
        Original native-source archive with results/histories.
    seed, redshift
        Existing snapshot used only to inherit physical inputs and native runtime.

    Returns
    -------
    tuple
        Unmodified native inputs and checksums/provenance of their source files.
    """
    parent = parent.resolve()
    history = parent / "results/histories" / f"seed{seed}"
    history_path = history / "history_receipt.json"
    history_receipt = json.loads(history_path.read_text())
    hashes = _verify_files(
        history, history_receipt["files"], ("history_manifest.json",)
    )
    if (
        json.loads((history / "history_manifest.json").read_text())
        != history_receipt["identity"]
    ):
        raise ValueError("Archived history manifest differs from receipt")
    matches = []
    for name in sorted(history_receipt["files"]):
        if not name.startswith("native/") or not name.endswith("/receipt.json"):
            continue
        receipt_path = _checked_path(history, name)
        receipt = json.loads(receipt_path.read_text())
        if (
            int(receipt["record"]["seed"]) == seed
            and float(receipt["record"]["redshift"]) == redshift
        ):
            hashes.update(_verify_files(history, history_receipt["files"], (name,)))
            matches.append((receipt_path, receipt))
    if len(matches) != 1:
        raise ValueError(
            "Require exactly one archived template snapshot at the selected seed/redshift"
        )
    receipt_path, receipt = matches[0]
    hashes.update(
        _verify_files(
            receipt_path.parent,
            receipt["files"],
            ("native.json", "native.h5", "source.npz"),
        )
    )
    identity = receipt["identity"]
    if (
        json.loads((receipt_path.parent / "native.json").read_text())["identity"]
        != identity
    ):
        raise ValueError("Archived native metadata differs from receipt")
    template = identity["inputs"]
    expected = {
        "random_seed",
        "node_redshifts",
        "cosmo_params",
        "simulation_options",
        "matter_options",
        "astro_params",
        "astro_options",
    }
    if (
        set(template) != expected
        or template["random_seed"] != seed
        or float(identity["redshift"]) != redshift
    ):
        raise ValueError("Template must retain its complete native input identity")
    validate_physics(template)
    for path in (
        history_path,
        parent / "experiment_config.json",
        parent / "results/manifest.json",
        parent / "runtime.env",
        parent / "python_path.txt",
    ):
        hashes[str(path)] = _digest(path)
    return template, {
        "parent_run": str(parent),
        "template_seed": seed,
        "template_redshift": redshift,
        "receipt_path": str(receipt_path),
        "native_extension_sha256": identity["extension_sha256"],
        "input_sha256": hashes,
    }


def _runtime(parent: Path) -> dict[str, Any]:
    import py21cmfast
    from py21cmfast import c_21cmfast

    source = (parent / "native_build/source/src/py21cmfast").resolve()
    package = Path(py21cmfast.__file__).resolve().parent
    extension = Path(c_21cmfast.__file__).resolve()
    if package != source or not extension.is_relative_to(source):
        raise ValueError("Use the archived parent's native package through PYTHONPATH")
    compiler = shutil.which(os.environ.get("CXX", "g++"))
    if compiler is None:
        raise RuntimeError("Analytic teacher generation requires a C++/OpenMP compiler")
    compiler_path = Path(compiler).resolve()
    version = subprocess.run(
        [str(compiler_path), "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    return {
        "python": str(Path(sys.executable).resolve()),
        "python_sha256": _digest(Path(sys.executable).resolve()),
        "CXX": {
            "path": str(compiler_path),
            "sha256": _digest(compiler_path),
            "version": version,
        },
        "native_package": str(source),
        "native_package_version": str(py21cmfast.__version__),
        "native_extension": str(extension),
        "native_extension_sha256": _digest(extension),
        "native_source_sha256": {
            str(path.relative_to(source)): _digest(path)
            for path in sorted(source.rglob("*"))
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in (".pyc", ".pyo")
        },
    }


def _preflight(template: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    from research.pc_analytic.campaign import native_inputs_for_seed, validate_config

    config = validate_config(config)
    native_inputs_for_seed(template, config, config["seeds"][0])
    return config


def prepare(run: Path, environment: Mapping[str, str]) -> dict[str, Any]:
    """Freeze a fresh campaign without generating simulations on the login host.

    Parameters
    ----------
    run
        Run directory with its source archive and shell metadata installed.
    environment
        Explicit campaign and scheduler settings.

    Returns
    -------
    dict
        Immutable campaign, runtime and physical-template manifest.
    """
    run = run.resolve()
    if (run / "PREPARED").exists() or (run / "run_manifest.json").exists():
        raise FileExistsError("Use a fresh run; do not rewrite a frozen campaign")
    parent = Path((run / "parent_run.txt").read_text().strip()).resolve()
    if run == parent or run.is_relative_to(parent):
        raise ValueError("Campaign output must be separate from its archived parent")
    seed = int(environment.get("PC_TEMPLATE_SEED", "222"))
    redshift = float(environment.get("PC_TEMPLATE_REDSHIFT", "7.5"))
    if seed < 1 or not math.isfinite(redshift) or redshift < 0:
        raise ValueError("Invalid archived template selection")
    template, origin = select_template(parent, seed, redshift)
    template_path = run / "native_template.json"
    _write_json(template_path, template)
    runtime = _runtime(parent)
    if runtime["native_extension_sha256"] != origin["native_extension_sha256"]:
        raise ValueError("Native binary differs from the verified template archive")
    config = configuration(template_path, environment)
    config["native_template_sha256"] = _digest(template_path)
    config["native_extension_sha256"] = runtime["native_extension_sha256"]
    config = _preflight(template, config)
    resources = resource_estimate(config, environment)
    _write_json(run / "experiment_config.json", config)
    manifest = {
        "schema": 1,
        "configuration": config,
        "resources": resources,
        "source_sha256": tree_hashes(run / "source"),
        "template_origin": origin,
        "runtime": runtime,
        "scope": "Independent native coeval simulations with analytic photon-conserving teacher fields; no neural-network training or external reference runs",
        "network_required_on_compute": False,
    }
    _write_json(run / "run_manifest.json", manifest)
    (run / "prepared.sha256").write_text(
        "".join(f"{_digest(run / name)}  {name}\n" for name in FROZEN_FILES)
    )
    (run / "PREPARED").write_text("Prepared immutable analytic teacher campaign.\n")
    return manifest


def verify(run: Path, cpus: int | None = None) -> dict[str, Any]:
    """Verify frozen source, template, original native runtime and CPU allocation.

    Parameters
    ----------
    run
        Previously prepared campaign.
    cpus
        Actual scheduler CPU allocation when available.

    Returns
    -------
    dict
        Unchanged manifest suitable for execution or resume.
    """
    run = run.resolve()
    expected = [f"{_digest(run / name)}  {name}" for name in FROZEN_FILES]
    if (run / "prepared.sha256").read_text().splitlines() != expected:
        raise ValueError("Prepared metadata changed; prepare a new campaign")
    manifest = json.loads((run / "run_manifest.json").read_text())
    if tree_hashes(run / "source") != manifest["source_sha256"]:
        raise ValueError("Frozen source changed; prepare a new campaign")
    origin = manifest["template_origin"]
    for name, expected_digest in origin["input_sha256"].items():
        if not Path(name).is_file() or _digest(Path(name)) != expected_digest:
            raise ValueError(f"Archived parent input changed: {name}")
    if _runtime(Path(origin["parent_run"])) != manifest["runtime"]:
        raise ValueError("Archived native runtime or compiler changed")
    if cpus is not None and cpus != manifest["configuration"]["cpus"]:
        raise ValueError("CPU allocation differs from the frozen campaign")
    return manifest


def main() -> None:
    """Prepare or verify an immutable teacher campaign from the CLI."""
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
    config = manifest["configuration"]
    LOGGER.info(
        "%s: %s seeds x %s redshifts; N=%s, L=%s cMpc, %s workers x %s threads",
        args.command,
        len(config["seeds"]),
        len(config["redshifts"]),
        config["dimension"],
        config["box_len"],
        config["workers"],
        config["threads"],
    )


if __name__ == "__main__":
    main()
