"""Reproduce the native ES/RT initial-history comparison.

Run ``python -m research.es_pc_emulator.validate_history PHASE --output DIR
--upstream C2RAY_CHECKOUT --workers 8`` for ``warm``, ``reset``,
``original-late``, ``time2``, then ``report``. The warm phase makes three
independent native histories. Reset retains the exact final source interval
of seed 333 but discards its earlier RT state; time2 retains the full history
and doubles only RT substeps. The optional ``late-dataset`` phase generates
three independent late-start cases with native IC threads fixed to one for
reproducing the original smoke-test training comparison. These tests diagnose
time-step effects; they do not certify cosmological convergence.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from research.c2ray_teacher.runner import run as run_teacher
from research.es_pc_emulator.generate import _reuse_seed, run_dataset, write_json
from research.es_pc_emulator.pairs import combine_manifests, export_pairs, load_pair
from research.pc_evolution.native_history import (
    ARRAY_KEYS,
    _write_fresh_history,
    array_hash,
    build_native_history,
    load_history,
    validate_history,
    write_es_features,
)
from research.pc_field_emulator.evaluation import field_metrics

ROOT = Path(__file__).resolve().parents[2]
PHASES = ("warm", "reset", "original-late", "late-dataset", "time2", "report")
CONFIG = {
    "schema_version": 1,
    "teacher": {
        "dimension": 16,
        "highres_dimension": 48,
        "box_len": 50.0,
        "z_start": 20.0,
        "z_end": 7.9,
        "intervals": 8,
        "reference_z": 8.0,
        "refinement": 1,
        "time_substeps": 1,
        "stride": 1,
        "transport": "conservative_rays",
        "angular_order": 4,
    },
    "splits": {"train": [111], "validation": [222], "test": [333]},
}


def _file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def truncate_last_interval(history: dict, parent_sha256: str) -> dict:
    """Retain the final source interval and restart the clock at zero.

    Parameters
    ----------
    history : dict
        Validated native history with decoded metadata.
    parent_sha256 : str
        SHA256 of the original history file, retained as provenance.

    Returns
    -------
    dict
        Independent arrays and metadata. Source positions, gas capacity,
        endpoint rates and native ES fields retain their original identity.
    """
    validate_history(history)
    if re.fullmatch(r"[0-9a-f]{64}", parent_sha256) is None:
        raise ValueError("A lowercase SHA256 of the parent history is required")
    count = len(history["redshifts"])
    source_records = history["metadata"].get("source_provenance", [])
    if len(source_records) != count:
        raise ValueError("Source provenance must cover every parent endpoint")
    for row, redshift in zip(source_records, history["redshifts"], strict=True):
        if row.get("redshift") != float(redshift):
            raise ValueError("Source provenance endpoint order differs from history")
    result = copy.deepcopy(history)
    origin = float(history["times_seconds"][-2])
    for key in ARRAY_KEYS:
        if key != "capacity":
            result[key] = np.array(history[key][-2:], copy=True)
    result["times_seconds"] -= origin
    metadata = result["metadata"]
    metadata.update(
        redshift_start=float(result["redshifts"][0]),
        redshift_end=float(result["redshifts"][-1]),
        intervals=1,
        source_provenance=copy.deepcopy(source_records[-2:]),
        controlled_reset_parent_history_sha256=parent_sha256,
        controlled_reset_parent_endpoint_indices=[count - 2, count - 1],
        controlled_reset_parent_time_origin_seconds=origin,
        controlled_reset_definition=(
            "Same exact final source-rate segment and gas; omit all prior RT history"
        ),
        time_origin="zero at retained source endpoint; all earlier RT state omitted",
        redshift_sampling="exact final two endpoints of the identified parent history",
        reference_redshift_is_source_endpoint=(
            metadata["reference_redshift"] in result["redshifts"]
        ),
    )
    if "time_interval_quadrature_errors_seconds" in metadata:
        metadata["time_interval_quadrature_errors_seconds"] = metadata[
            "time_interval_quadrature_errors_seconds"
        ][-1:]
    metadata["array_sha256"] = {key: array_hash(result[key]) for key in ARRAY_KEYS}
    validate_history(result)
    return result


def final_report(manifest_path: Path) -> dict:
    """Measure the lowest-redshift paired field in a validated manifest.

    Parameters
    ----------
    manifest_path : Path
        A per-seed pair manifest produced by ``export_pairs``.

    Returns
    -------
    dict
        Field errors, ionization means and the input-photon inventory.
    """
    manifest = json.loads(manifest_path.read_text())
    record = min(manifest["samples"], key=lambda row: row["redshift"])
    pair = load_pair(record)
    density = pair["density"]
    es, teacher = 1 - pair["xhi_es"], 1 - pair["xhi_c2ray"]
    history_path = manifest_path.parent.parent / "native_history.npz"
    native_threads = None
    if history_path.is_file():
        if _file_hash(history_path) != pair["metadata"]["history_sha256"]:
            raise ValueError("Report source history no longer matches paired data")
        native_threads = load_history(history_path)["metadata"]["native_threads"]
    return {
        "sample": record,
        "native_threads": native_threads,
        "initial_conditions_sha256": record["initial_conditions_sha256"],
        "metrics_es_vs_rt": field_metrics(es, teacher, density),
        "es_Q_volume": float(es.mean()),
        "rt_Q_volume": float(teacher.mean()),
        "rt_Q_mass": float(np.sum(density * teacher) / density.sum()),
        "target_mean_input_photons_per_nucleus": float(
            pair["photons"].sum() / density.sum()
        ),
        "pair_sha256": _file_hash(Path(record["path"])),
        "initial_state_audit": pair["metadata"].get("initial_state_audit"),
        "teacher_transport_primary_residual": pair["metadata"].get(
            "teacher_transport_primary_residual"
        ),
    }


def _provenance(phase: str, output: Path, upstream: Path, workers: int) -> dict:
    return {
        "arguments": {
            "phase": phase,
            "output": str(output),
            "upstream": str(upstream),
            "workers": workers,
        },
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": {
            str(path.relative_to(ROOT)): _file_hash(path)
            for path in sorted((ROOT / "research").rglob("*.py"))
        },
        "driver_sha256": _file_hash(Path(__file__)),
        "thread_protocol": {
            "warm_native_threads": workers,
            "original_late_and_late_dataset_native_threads": 1,
            "teacher_workers": workers,
            "controlled_comparison": "reset/time2 reuse exact warm history; different native thread counts may change IC realization",
            "initial_conditions_identity": "each native_history metadata and pair record carry the actual IC SHA256",
        },
        "definition": (
            "Real native history and C2-Ray; no label mock; no convergence certification"
        ),
    }


def _warm_source(output: Path) -> Path:
    directory = output / "warm/seed_333"
    receipt = _reuse_seed(directory, seed=333, split="test")
    return Path(receipt["manifest"]).parent.parent


def _history_case(phase: str, output: Path, upstream: Path, workers: int) -> None:
    source = _warm_source(output)
    experiment = output / phase
    experiment.mkdir()
    history_path = source / "native_history.npz"
    history = load_history(history_path)
    with np.load(source / "native_es.npz", allow_pickle=False) as archive:
        radius = np.array(archive["radius_es_cMpc"])
        density = np.array(archive["density_es"])
    if phase == "reset":
        history = truncate_last_interval(history, _file_hash(history_path))
        radius, density = radius[-2:], density[-2:]
    history_path, es_path = (
        experiment / "native_history.npz",
        experiment / "native_es.npz",
    )
    _write_fresh_history(history_path, history)
    write_es_features(es_path, history, radius, density)
    teacher = CONFIG["teacher"] | {
        "history": str(history_path),
        "seed": 333,
        "benchmark": False,
        "z_start": float(history["redshifts"][0]),
        "intervals": len(history["redshifts"]) - 1,
        "time_substeps": 2 if phase == "time2" else 1,
    }
    run_teacher(teacher, experiment / "teacher", upstream, workers)
    export_pairs(
        history_path,
        es_path,
        experiment / "teacher",
        experiment / "pairs",
        "test",
        expected_seed=333,
        initial_state_config={"allow_mismatch": phase == "reset"},
    )


def _late_case(
    experiment: Path, upstream: Path, workers: int, seed: int, split: str
) -> Path:
    experiment.mkdir()
    teacher = CONFIG["teacher"] | {
        "z_start": 8.0,
        "intervals": 1,
        "seed": seed,
        "benchmark": False,
    }
    history_path, es_path = (
        experiment / "native_history.npz",
        experiment / "native_es.npz",
    )
    keys = (
        "dimension",
        "highres_dimension",
        "box_len",
        "z_start",
        "z_end",
        "intervals",
        "reference_z",
        "seed",
    )
    build_native_history(
        output=history_path,
        es_output=es_path,
        threads=1,
        **{key: teacher[key] for key in keys},
    )
    run_teacher(
        teacher | {"history": str(history_path)},
        experiment / "teacher",
        upstream,
        workers,
    )
    export_pairs(
        history_path,
        es_path,
        experiment / "teacher",
        experiment / "pairs",
        split,
        expected_seed=seed,
        initial_state_config={"allow_mismatch": True},
    )
    metadata = load_history(history_path)["metadata"]
    write_json(
        experiment / "case_identity.json",
        {
            "seed": seed,
            "split": split,
            "native_threads": metadata["native_threads"],
            "teacher_workers": workers,
            "initial_conditions_sha256": metadata["initial_hires_density_sha256"],
        },
    )
    return experiment / "pairs/manifest.json"


def run(
    phase: str, output: Path, upstream: Path, workers: int = 8, *, resume: bool = False
) -> dict:
    """Run one reproducible initial-history comparison phase.

    Parameters
    ----------
    phase : str
        One of warm, reset, original-late, late-dataset, time2 or report.
    output : Path
        Shared experiment directory, with fresh subdirectories for native runs.
    upstream : Path
        Checkout of the pinned C2-Ray revision.
    workers : int
        Number of CPU workers supported by the C2-Ray runner.
    resume : bool
        Resume the warm dataset using its existing provenance checks.

    Returns
    -------
    dict
        Phase completion status or the complete field-comparison report.
    """
    if phase not in PHASES:
        raise ValueError("Unknown history validation phase")
    if resume and phase != "warm":
        raise ValueError("Only the warm dataset supports --resume")
    output, upstream = Path(output).resolve(), Path(upstream).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / f"{phase}_provenance.json",
        _provenance(phase, output, upstream, workers),
    )
    if phase == "warm":
        return run_dataset(
            copy.deepcopy(CONFIG), output / "warm", upstream, workers, resume=resume
        )
    if phase in ("reset", "time2"):
        _history_case(phase, output, upstream, workers)
    elif phase == "original-late":
        _late_case(output / "original-late", upstream, workers, 333, "test")
    elif phase == "late-dataset":
        destination = output / "late-dataset"
        destination.mkdir()
        manifests = [
            _late_case(destination / f"seed{seed}", upstream, workers, seed, split)
            for split, seeds in CONFIG["splits"].items()
            for seed in seeds
        ]
        manifest = output / "late_dataset_manifest.json"
        combine_manifests(manifests, manifest)
        return {
            "phase": phase,
            "manifest": str(manifest),
            "native_threads": 1,
            "teacher_workers": workers,
        }
    else:
        source = _warm_source(output)
        reports = {"warm_seed333": final_report(source / "pairs/manifest.json")}
        for name in ("reset", "original-late", "time2"):
            path = output / name / "pairs/manifest.json"
            if path.exists():
                reports[name] = final_report(path)
        write_json(output / "comparison.json", reports)
        return reports
    return {"phase": phase, "output": str(output / phase)}


def main() -> None:
    """Run the requested validation phase from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=PHASES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    result = run(**vars(parser.parse_args()))
    sys.stdout.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
