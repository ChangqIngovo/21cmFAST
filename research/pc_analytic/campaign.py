"""Generate independently audited analytic-PC teachers from many native seeds.

Every seed has its own process, native initial conditions, cache and resumable
snapshot receipts. Redshifts share that seed's initial conditions but are
independent cumulative-budget coeval solutions. No SCRIPT, C2-Ray or neural
network is run by this campaign.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import itertools
import json
import logging
import multiprocessing
import os
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from research.pc_analytic.compare import (
    _brightness,
    _digest,
    _write_json,
    independent_budget,
    load_native_snapshot,
    validate_physics,
)
from research.pc_analytic.solver import build_identity, solve
from research.pc_matched.rt import SUPPORTED_DIRECTION_COUNTS
from research.pc_native_rt.workflow import (
    _hash_array,
    _input_dict,
    _native_snapshots,
    _receipt,
    _redshift_name,
    _runtime_identity,
    _write_npz,
)

LOGGER = logging.getLogger(__name__)
LEDGER_TOLERANCE = 1e-10
DEFAULTS = {
    "dimension": 32,
    "highres_dimension": 256,
    "box_len": 256.0,
    "seeds": list(range(1001, 1009)),
    "redshifts": [10.0, 9.0, 8.0, 7.5, 7.0, 6.5, 6.0],
    "workers": 4,
    "threads": 4,
    "cpus": 16,
    "brightness": "native",
    "sigma_cm2": 2.7173135802469135e-18,
    "analytic": {
        "directions": 96,
        "exposure_steps": 32,
        "boundary": "source_periodic_cube",
        "transport_mode": "characteristic",
        "exhaustion_tolerance": 1e-7,
        "batch_tail_tolerance": 1e-4,
        "max_sweeps": 10000,
    },
}
TEMPLATE_KEYS = {
    "random_seed",
    "node_redshifts",
    "cosmo_params",
    "simulation_options",
    "matter_options",
    "astro_params",
    "astro_options",
}


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate a frozen teacher campaign and its declared CPU allocation.

    Parameters
    ----------
    config
        Native input-template path, geometry, seeds, redshifts and parallel
        controls. Workers times threads may not exceed cpus.

    Returns
    -------
    dict
        Independent normalized settings with a verified native-template hash.
    """
    allowed = set(DEFAULTS) | {
        "native_template_json",
        "native_template_sha256",
        "native_extension_sha256",
    }
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError("Unknown teacher campaign configuration keys")
    result = copy.deepcopy(DEFAULTS)
    result.update(copy.deepcopy(config))
    if not isinstance(config.get("analytic", {}), dict):
        raise TypeError("analytic controls must be a dictionary")
    controls = copy.deepcopy(DEFAULTS["analytic"])
    controls.update(config.get("analytic", {}))
    if set(controls) != set(DEFAULTS["analytic"]):
        raise ValueError("Unsupported analytic teacher control")
    result["analytic"] = controls
    for name, minimum in (
        ("dimension", 2),
        ("highres_dimension", 2),
        ("workers", 1),
        ("threads", 1),
        ("cpus", 1),
    ):
        if type(result[name]) is not int or result[name] < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if result["workers"] * result["threads"] > result["cpus"]:
        raise ValueError("workers * threads exceeds the declared CPU allocation")
    if result["threads"] > 32 or result["highres_dimension"] < result["dimension"]:
        raise ValueError("Require threads<=32 and highres_dimension>=dimension")
    for name in ("box_len", "sigma_cm2"):
        if (
            isinstance(result[name], bool)
            or not isinstance(result[name], (int, float))
            or not np.isfinite(result[name])
            or result[name] <= 0
        ):
            raise ValueError(f"{name} must be finite and positive")
    seeds, redshifts = result["seeds"], result["redshifts"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(seed) is not int or seed < 1 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Seeds must be unique positive integers")
    if (
        not isinstance(redshifts, list)
        or not redshifts
        or any(
            isinstance(z, bool)
            or not isinstance(z, (int, float))
            or not np.isfinite(z)
            or z < 0
            for z in redshifts
        )
        or any(a <= b for a, b in itertools.pairwise(redshifts))
    ):
        raise ValueError(
            "Redshifts must be finite, nonnegative and strictly descending"
        )
    if len({_redshift_name(z) for z in redshifts}) != len(redshifts):
        raise ValueError(
            "Redshifts must have distinct checkpoint names at 12-digit precision"
        )
    if result["brightness"] not in {"native", "none", "saturated_proxy"}:
        raise ValueError("Unsupported teacher brightness definition")
    _validate_analytic(controls)
    if not isinstance(result.get("native_template_json"), (str, Path)):
        raise TypeError("native_template_json must be a path")
    template_path = Path(result["native_template_json"]).resolve()
    if not template_path.is_file():
        raise ValueError(
            "native_template_json must identify a full native input template"
        )
    digest = _digest(template_path)
    if result.get("native_template_sha256", digest) != digest:
        raise ValueError("Native template checksum changed")
    template = json.loads(template_path.read_text())
    if set(template) != TEMPLATE_KEYS:
        raise ValueError(
            "Native template must contain all seven InputParameters groups"
        )
    validate_physics(template)
    result["native_template_json"] = str(template_path)
    result["native_template_sha256"] = digest
    return result


def _validate_analytic(controls: dict[str, Any]) -> None:
    for name in ("directions", "exposure_steps", "max_sweeps"):
        if type(controls[name]) is not int or controls[name] < 1:
            raise ValueError(f"analytic.{name} must be a positive integer")
    if controls["directions"] not in SUPPORTED_DIRECTION_COUNTS:
        raise ValueError("Unsupported analytic direction count")
    if controls["exposure_steps"] > 2**31 - 1 or controls["max_sweeps"] > 2**63 - 1:
        raise ValueError("Analytic work counts exceed supported ranges")
    if (
        controls["boundary"] not in {"periodic", "open", "source_periodic_cube"}
        or controls["transport_mode"] != "characteristic"
    ):
        raise ValueError(
            "Teacher requires characteristic transport and a declared boundary"
        )
    for name in ("exhaustion_tolerance", "batch_tail_tolerance"):
        if (
            isinstance(controls[name], bool)
            or not isinstance(controls[name], (int, float))
            or not np.isfinite(controls[name])
            or not 0 < controls[name] < 1
        ):
            raise ValueError(f"Invalid analytic.{name}")
    if controls["batch_tail_tolerance"] < controls["exhaustion_tolerance"]:
        raise ValueError(
            "analytic.batch_tail_tolerance must be >= exhaustion_tolerance"
        )


def native_inputs_for_seed(
    template: dict[str, Any], config: dict[str, Any], seed: int
) -> dict[str, Any]:
    """Clone the native template, changing only seed and declared grid controls.

    Parameters
    ----------
    template
        Full serialized native InputParameters dictionary.
    config
        Validated campaign geometry and per-process thread count.
    seed
        Requested independent initial-condition seed.

    Returns
    -------
    dict
        Full native serialization with unchanged astrophysics and cosmology.
    """
    from py21cmfast.input_serialization import deserialize_inputs
    from py21cmfast.wrapper.inputs import InputParameters
    from research.pc_reconsidered.native_driver import validate_inputs

    structs = deserialize_inputs(
        {
            key: value
            for key, value in template.items()
            if key not in {"random_seed", "node_redshifts"}
        },
        safe=True,
    )
    original = InputParameters(
        **structs,
        random_seed=template["random_seed"],
        node_redshifts=template["node_redshifts"],
    )
    if _input_dict(original) != template:
        raise ValueError("Native template does not round-trip in the selected runtime")
    inputs = original.evolve_input_structs(
        HII_DIM=config["dimension"],
        DIM=config["highres_dimension"],
        HIRES_TO_LOWRES_FACTOR=None,
        BOX_LEN=config["box_len"],
        LOWRES_CELL_SIZE_MPC=None,
        N_THREADS=config["threads"],
    ).clone(random_seed=seed)
    validate_inputs(inputs)
    updated = _input_dict(inputs)
    for name in (
        "cosmo_params",
        "matter_options",
        "astro_params",
        "astro_options",
        "node_redshifts",
    ):
        if updated[name] != template[name]:
            raise ValueError(
                f"Changing teacher geometry altered physical inputs: {name}"
            )
    geometry = {
        "HII_DIM",
        "DIM",
        "BOX_LEN",
        "N_THREADS",
        "HIRES_TO_LOWRES_FACTOR",
        "LOWRES_CELL_SIZE_MPC",
        "_DIM",
        "_BOX_LEN",
        "_HIRES_TO_LOWRES_FACTOR",
        "_LOWRES_CELL_SIZE_MPC",
    }
    before = {
        key: value
        for key, value in template["simulation_options"].items()
        if key not in geometry
    }
    after = {
        key: value
        for key, value in updated["simulation_options"].items()
        if key not in geometry
    }
    if before != after:
        raise ValueError("Changing teacher geometry altered other simulation controls")
    return updated


def _implementation_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    files = [
        "research/pc_analytic/campaign.py",
        "research/pc_analytic/solver.py",
        "research/pc_analytic/transport_kernel.cpp",
        "research/pc_analytic/absorption.py",
        "research/pc_analytic/compare.py",
        "research/pc_native_rt/workflow.py",
        "research/pc_reconsidered/native_driver.py",
        "research/pc_matched/workflow.py",
        "research/pc_matched/rt.py",
        "research/pc_coeval/run.py",
    ]
    return {name: _digest(root / name) for name in files}


def _freeze(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(
                f"Campaign identity changed; use a fresh output directory: {path}"
            )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(path, value)


@contextmanager
def _thread_environment(threads: int) -> Iterator[None]:
    names = {
        "OMP_NUM_THREADS": str(threads),
        "OMP_THREAD_LIMIT": str(threads),
        "OMP_DYNAMIC": "FALSE",
        "OMP_WAIT_POLICY": "PASSIVE",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    previous = {key: os.environ.get(key) for key in names}
    os.environ.update(names)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _teacher_snapshot(
    native: Path,
    folder: Path,
    config: dict[str, Any],
    campaign_identity: dict[str, Any],
) -> dict[str, Any]:
    identity = {
        "campaign": campaign_identity,
        "native_receipt_sha256": _digest(native / "receipt.json"),
    }
    receipt = _receipt(folder, "receipt.json", identity)
    if receipt is not None:
        if set(receipt["files"]) != {"fields.npz", "metrics.json"}:
            raise ValueError("Teacher receipt does not cover all required artifacts")
        if json.loads((folder / "metrics.json").read_text()) != receipt["record"]:
            raise ValueError("Teacher receipt record differs from saved metrics")
        record = receipt["record"]
        residual = record["photon_ledger"]["relative_photon_residual"]
        if (
            not record.get("complete")
            or not record["photon_ledger"]["bounded"]
            or not np.isfinite(residual)
            or residual > LEDGER_TOLERANCE
        ):
            raise ValueError("Stored teacher receipt does not certify a passing label")
        return receipt["record"]
    data = load_native_snapshot(native, brightness=config["brightness"])
    source, capacity, meta = data["source"], data["capacity"], data["metadata"]
    originals = {
        name: _hash_array(data[name])
        for name in ("source", "capacity", "native_es_xhi", "native_brightness")
    }
    result = solve(
        source,
        capacity,
        redshift=meta["redshift"],
        box_len=meta["box_len"],
        nH_mean_comoving_cm3=meta["nH_mean_comoving_cm3"],
        sigma_cm2=config["sigma_cm2"],
        threads=config["threads"],
        **config["analytic"],
    )
    if not result["audit"].get("complete") or not result["audit"].get("ledger_passed"):
        raise RuntimeError("Teacher solver did not finish with a passing photon ledger")
    ions = np.asarray(result["ionized"], dtype=float)
    if (
        not np.isfinite(ions).all()
        or ions.shape != source.shape
        or np.any(ions < -1e-12 * capacity)
        or np.any(ions > (1 + 1e-12) * capacity)
    ):
        raise RuntimeError("Teacher field violates finite atom-capacity bounds")
    xhi = 1 - ions / capacity
    radiation = np.asarray(result["remaining_radiation"], dtype=float)
    escaped = np.asarray(result["escaped"], dtype=float)
    unemitted = np.asarray(result["unemitted_source"], dtype=float)
    if (
        radiation.ndim != 4
        or radiation.shape[1:] != source.shape
        or escaped.shape != (radiation.shape[0],)
        or unemitted.shape != source.shape
    ):
        raise RuntimeError(
            "Teacher photon reservoirs have inconsistent grid/direction shapes"
        )
    for name, value in (
        ("remaining", radiation),
        ("escaped", escaped),
        ("unemitted", unemitted),
    ):
        if not np.isfinite(value).all() or np.any(value < 0):
            raise RuntimeError(f"Invalid teacher photon reservoir: {name}")
    remaining = radiation.sum(axis=0)
    budget = independent_budget(
        source,
        capacity,
        xhi,
        remaining=float(np.sum(remaining, dtype=np.longdouble)),
        escaped=float(np.sum(escaped, dtype=np.longdouble)),
        unemitted=float(np.sum(unemitted, dtype=np.longdouble)),
    )
    if (
        not budget["bounded"]
        or not np.isfinite(budget["relative_photon_residual"])
        or budget["relative_photon_residual"] > LEDGER_TOLERANCE
    ):
        raise RuntimeError("Independent field-derived teacher photon ledger failed")
    brightness = _brightness(data, {"analytic": xhi}, config["brightness"])
    if brightness is not None and (
        brightness["analytic"].shape != source.shape
        or not np.isfinite(brightness["analytic"]).all()
    ):
        raise RuntimeError("Teacher brightness is non-finite or has an unexpected grid")
    if any(_hash_array(data[name]) != digest for name, digest in originals.items()):
        raise RuntimeError(
            "Teacher solver modified native source, gas or baseline fields"
        )
    arrays = {
        "source": source,
        "capacity": capacity,
        "teacher_xHI": xhi,
        "ionized": ions,
        "remaining_photons": remaining,
        "escaped_by_direction": escaped,
        "unemitted_source": unemitted,
        "native_es_xHI": data["native_es_xhi"],
    }
    if brightness is not None:
        arrays.update(
            teacher_brightness=brightness["analytic"],
            native_brightness=data["native_brightness"],
        )
    row = {
        "seed": int(meta["seed"]),
        "redshift": float(meta["redshift"]),
        "dimension": int(meta["dimension"]),
        "box_len": float(meta["box_len"]),
        "input": meta,
        "photon_ledger": budget,
        "solver_audit": result["audit"],
        "solver_metadata": result["metadata"],
        "radiation_storage": "remaining_photons sums angular intensities per cell; escaped_by_direction retains direction totals; coeval labels are not time-evolution restart states",
        "neutral_fraction_mass": budget["xHI_mass"],
        "neutral_fraction_volume": budget["xHI_volume"],
        "phase_cell_counts": {
            "neutral": int(np.count_nonzero(xhi >= 0.5)),
            "ionized": int(np.count_nonzero(xhi < 0.5)),
            "partial": int(np.count_nonzero((xhi > 0) & (xhi < 1))),
        },
        "native_inputs_unchanged": True,
        "complete": True,
        "fields_path": str(folder / "fields.npz"),
        "receipt_path": str(folder / "receipt.json"),
    }
    folder.mkdir(parents=True, exist_ok=True)
    _write_npz(folder / "fields.npz", **arrays)
    _write_json(folder / "metrics.json", row)
    _write_json(
        folder / "receipt.json",
        {
            "identity": identity,
            "files": {
                name: _digest(folder / name) for name in ("fields.npz", "metrics.json")
            },
            "record": row,
        },
    )
    return row


def _run_seed(
    seed: int, config: dict[str, Any], root: str, identity: dict[str, Any]
) -> list[dict[str, Any]]:
    # Spawned workers do not execute main(), so install their log handler here.
    # Existing caller handlers are retained by basicConfig's default behavior.
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    directory = Path(root) / "seeds" / f"seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _freeze(directory / "identity.json", {"campaign": identity, "seed": seed})
        (directory / "COMPLETED").unlink(missing_ok=True)
        LOGGER.info(
            "Teacher seed started: seed=%d snapshots=%d threads=%d",
            seed,
            len(config["redshifts"]),
            config["threads"],
        )
        if _runtime_identity() != identity["native_runtime"]:
            raise ValueError("Native runtime changed in a teacher worker")
        template = json.loads(Path(config["native_template_json"]).read_text())
        if (
            _digest(Path(config["native_template_json"]))
            != config["native_template_sha256"]
        ):
            raise ValueError("Native template changed during campaign")
        native_inputs = native_inputs_for_seed(template, config, seed)
        native_path = directory / "native_inputs.json"
        _freeze(native_path, native_inputs)
        native_config = {
            "dimension": config["dimension"],
            "highres_dimension": config["highres_dimension"],
            "box_len": config["box_len"],
            "threads": config["threads"],
            "seeds": [seed],
            "redshifts": config["redshifts"],
            "native_inputs_json": str(native_path),
        }
        records = _native_snapshots(directory, native_config)
        LOGGER.info("Native snapshots ready: seed=%d count=%d", seed, len(records))
        rows = []
        for record in records:
            native = (directory / record["h5_path"]).parent
            folder = directory / "teacher" / _redshift_name(record["redshift"])
            resumed = (folder / "receipt.json").is_file()
            rows.append(_teacher_snapshot(native, folder, config, identity))
            LOGGER.info(
                "Teacher %s: seed=%d z=%g ledger=%.3g",
                "resumed and verified" if resumed else "saved",
                seed,
                record["redshift"],
                rows[-1]["photon_ledger"]["relative_photon_residual"],
            )
        if {(row["seed"], row["redshift"]) for row in rows} != {
            (seed, z) for z in config["redshifts"]
        }:
            raise RuntimeError(
                "Native/teacher snapshot set differs from requested seed/redshifts"
            )
        _write_json(
            directory / "summary.json",
            {"seed": seed, "snapshots": rows, "complete": True},
        )
        (directory / "COMPLETED").write_text(
            "All seed snapshots independently audited.\n"
        )
        return rows


def run(config: dict[str, Any], output: str | Path) -> dict[str, Any]:
    """Generate independent native simulations and conservative teacher labels.

    Parameters
    ----------
    config
        Campaign settings accepted by :func:`validate_config`.
    output
        Fresh output directory, or an unchanged resumable campaign directory.

    Returns
    -------
    dict
        Dataset manifest containing all independently verified sample receipts.
    """
    config = validate_config(config)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    runtime = _runtime_identity()
    if (
        config.get("native_extension_sha256", runtime["native_extension_sha256"])
        != runtime["native_extension_sha256"]
    ):
        raise ValueError("Use the template archive's original native extension")
    identity = {
        "schema": 1,
        "configuration": config,
        "native_runtime": runtime,
        "implementation_sha256": _implementation_hashes(),
        "analytic_build": build_identity(),
        "scope": "Independent native cumulative-budget coeval analytic-PC teachers; no cosmological state evolution, SCRIPT or C2-Ray",
    }
    with (output / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _freeze(output / "identity.json", identity)
        (output / "COMPLETED").unlink(missing_ok=True)
        rows, errors = [], []

        def save_summary() -> None:
            _write_json(
                output / "summary.json",
                {
                    "complete": False,
                    "completed_snapshots": len(rows),
                    "requested_snapshots": len(config["seeds"])
                    * len(config["redshifts"]),
                    "failed_seeds": errors,
                },
            )

        save_summary()
        with _thread_environment(config["threads"]):
            if config["workers"] == 1:
                for seed in config["seeds"]:
                    try:
                        rows.extend(_run_seed(seed, config, str(output), identity))
                    except Exception as error:
                        LOGGER.exception("Teacher seed %d failed", seed)
                        errors.append({"seed": seed, "error": str(error)})
                    save_summary()
            else:
                with ProcessPoolExecutor(
                    max_workers=config["workers"],
                    mp_context=multiprocessing.get_context("spawn"),
                ) as pool:
                    futures = {
                        pool.submit(
                            _run_seed, seed, config, str(output), identity
                        ): seed
                        for seed in config["seeds"]
                    }
                    for future in as_completed(futures):
                        try:
                            rows.extend(future.result())
                        except Exception as error:
                            LOGGER.exception("Teacher seed %d failed", futures[future])
                            errors.append(
                                {"seed": futures[future], "error": str(error)}
                            )
                        save_summary()
        if errors:
            raise RuntimeError(
                f"{len(errors)} teacher seed(s) failed; passing snapshot receipts remain resumable; inspect summary.json"
            )
        if (
            _implementation_hashes() != identity["implementation_sha256"]
            or _runtime_identity() != runtime
        ):
            raise RuntimeError(
                "Executable or native runtime changed while producing labels"
            )
        expected = {(seed, z) for seed in config["seeds"] for z in config["redshifts"]}
        if {(row["seed"], row["redshift"]) for row in rows} != expected or len(
            rows
        ) != len(expected):
            raise RuntimeError(
                "Completed teacher sample set is incomplete or duplicated"
            )
        rows.sort(key=lambda row: (row["seed"], -row["redshift"]))
        manifest = {
            "schema": 1,
            "identity": identity,
            "complete": True,
            "convergence_certified": False,
            "teacher_method": "analytic-column-characteristic",
            "source_convention": "Cumulative native n_ion; no extra density multiplier or emissivity time integral",
            "physical_scope": "Frozen coeval density, initially neutral, no recombinations/thermal evolution/photoheating feedback/independent helium chemistry",
            "samples": rows,
            "all_photon_ledgers_passed": True,
            "maximum_relative_photon_residual": max(
                row["photon_ledger"]["relative_photon_residual"] for row in rows
            ),
            "parallelism": {
                "workers": config["workers"],
                "threads_per_worker": config["threads"],
                "declared_cpus": config["cpus"],
                "process_start_method": "spawn",
            },
        }
        _write_json(output / "dataset_manifest.json", manifest)
        _write_json(
            output / "summary.json",
            {
                key: value
                for key, value in manifest.items()
                if key not in {"samples", "identity"}
            }
            | {
                "completed_snapshots": len(rows),
                "requested_snapshots": len(expected),
                "failed_seeds": [],
            },
        )
        _write_json(
            output / "completion.json",
            {
                "identity_sha256": _digest(output / "identity.json"),
                "files": {
                    name: _digest(output / name)
                    for name in ("dataset_manifest.json", "summary.json")
                },
            },
        )
        (output / "COMPLETED").write_text(
            "Every teacher sample has a verified native input and photon ledger.\n"
        )
        return manifest


def main() -> None:
    """Run a JSON-configured teacher campaign from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
