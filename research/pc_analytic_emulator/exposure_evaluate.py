"""Evaluate a frozen coeval exposure emulator on independent teacher campaigns."""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from research.pc_analytic_emulator.data import compatibility_signature, load_dataset
from research.pc_analytic_emulator.exposure_emulator import ExposureEmulator, run
from research.pc_analytic_emulator.predict import _digest
from research.pc_analytic_emulator.reliability import THRESHOLDS, accuracy_metrics
from research.pc_coeval.mfp import compare_mfp, make_draws, measure_mfp

LOGGER = logging.getLogger(__name__)
DRAW_SEED = 20261117


def _write(path: Path, content: dict[str, Any]) -> None:
    with path.open("x") as stream:
        json.dump(content, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _cross_correlation(
    predicted: np.ndarray, teacher: np.ndarray, atoms: np.ndarray
) -> dict[str, Any]:
    density = atoms / np.max(atoms)
    dimension = len(atoms)
    coordinate = np.fft.fftfreq(dimension) * dimension
    radius = np.sqrt(
        coordinate[:, None, None] ** 2
        + coordinate[None, :, None] ** 2
        + coordinate[None, None, :] ** 2
    )
    selected = (radius > 0) & (radius < THRESHOLDS["k_fraction"] * dimension / 2)
    labels = np.floor(radius[selected] + 0.5).astype(int)
    pred = np.fft.fftn(density * predicted, norm="ortho")[selected]
    truth = np.fft.fftn(density * teacher, norm="ortho")[selected]
    pred_power = np.bincount(labels, weights=np.abs(pred) ** 2)
    truth_power = np.bincount(labels, weights=np.abs(truth) ** 2)
    cross = np.bincount(labels, weights=(pred * np.conjugate(truth)).real)
    denominator = np.sqrt(pred_power) * np.sqrt(truth_power)
    numerical_floor = (
        64
        * np.finfo(float).eps ** 2
        * dimension**3
        * max(
            float(np.mean((density * predicted) ** 2)),
            float(np.mean((density * teacher) ** 2)),
        )
    )
    occupied = np.flatnonzero(np.bincount(labels))
    defined = occupied[denominator[occupied] > numerical_floor]
    values = np.clip(cross[defined] / denominator[defined], -1, 1)
    return {
        "minimum_defined": float(np.min(values)) if len(values) else None,
        "defined_shells": len(values),
        "undefined_shells": len(occupied) - len(values),
    }


def _geometric_mfp(
    prediction: np.ndarray, reference: np.ndarray, box_len: float, ray_count: int
) -> dict[str, Any]:
    draws = make_draws(ray_count, seed=DRAW_SEED)
    phases = {}
    for phase in ("ionized", "neutral"):
        controls = {
            "phase": phase,
            "threshold": 0.5,
            "draws": draws,
            "threads": 1,
            "max_distance": box_len,
        }
        predicted = measure_mfp(prediction, box_len, **controls)
        teacher = measure_mfp(reference, box_len, **controls)
        phases[phase] = {
            "predicted": predicted["summary"],
            "teacher": teacher["summary"],
            "comparison": compare_mfp(predicted, teacher),
        }
    return phases


def _verify_files(directory: Path, expected: dict[str, str]) -> None:
    if any(_digest(directory / name) != digest for name, digest in expected.items()):
        raise ValueError(f"Frozen input changed: {directory}")


def _score(
    task: tuple[Path, Path, dict[str, Any], dict[str, str], int],
) -> dict[str, Any]:
    model, result_root, record, frozen, mfp_rays = task
    _verify_files(model, frozen)
    teacher_path = Path(record["teacher_fields"])
    if _digest(teacher_path) != record["teacher_sha256"]:
        raise ValueError("Teacher fields changed after initial audit")
    native = Path(record["native_directory"])
    _verify_files(native, record["native_sha256"])
    redshift, box_len = record["redshift"], record["box_len"]
    result = result_root / f"seed{record['seed']}_z{redshift:g}"
    report = run(model, native, result, redshift=redshift, box_len=box_len)
    _verify_files(native, record["native_sha256"])
    _verify_files(model, frozen)
    saved = result / "fields.npz"
    if (
        _digest(saved) != report["field_sha256"]
        or _digest(teacher_path) != record["teacher_sha256"]
    ):
        raise ValueError("Predicted or reference field changed during validation")
    # Labels are used from this point only to score the already saved prediction.
    with (
        np.load(saved, allow_pickle=False) as prediction,
        np.load(teacher_path, allow_pickle=False) as reference,
    ):
        for name in ("source", "capacity", "native_es_xHI"):
            if not np.array_equal(prediction[name], reference[name]):
                raise ValueError(
                    "Saved prediction and reference use different native inputs"
                )
        predicted, teacher, atoms = (
            prediction["xHI"],
            reference["teacher_xHI"],
            reference["capacity"],
        )
        row = {
            **record,
            "metrics": accuracy_metrics(predicted, teacher, atoms, box_len),
            "low_k_brightness_cross_correlation": _cross_correlation(
                predicted, teacher, atoms
            ),
            "prediction_report_sha256": _digest(result / "report.json"),
            "prediction_fields_sha256": report["field_sha256"],
            "inference_seconds": report["inference_seconds"],
            "photon_ledger": report["saved_field_photon_ledger"],
            "mfp": _geometric_mfp(predicted, teacher, box_len, mfp_rays)
            if mfp_rays
            else None,
        }
    _write(result / "evaluation.json", row)
    return row


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    active_errors = [
        row["metrics"]["mean_relative_power_error"]
        for row in rows
        if row["metrics"]["phase"] == "active_or_early"
        and row["metrics"]["mean_relative_power_error"] is not None
    ]
    late_errors = [
        row["metrics"]["brightness_residual_rms_over_neutral_rms"]
        for row in rows
        if row["metrics"]["phase"] == "late"
    ]
    correlations = [
        row["low_k_brightness_cross_correlation"]["minimum_defined"]
        for row in rows
        if row["low_k_brightness_cross_correlation"]["minimum_defined"] is not None
    ]
    mfp = {}
    for phase in ("ionized", "neutral"):
        cases = [row["mfp"][phase] for row in rows if row["mfp"] is not None]
        defined = [
            case["comparison"] for case in cases if case["comparison"]["available"]
        ]
        mfp[phase] = {
            "defined": len(defined),
            "undefined": len(cases) - len(defined),
            "max_cdf_difference": max(
                (value["restricted_cdf_max_difference"] for value in defined),
                default=None,
            ),
            "empty_phase_mismatches": sum(
                (case["predicted"]["selected_voxel_count"] == 0)
                != (case["teacher"]["selected_voxel_count"] == 0)
                for case in cases
            ),
        }
    return {
        "snapshot_count": len(rows),
        "pure_emulator_snapshots": len(rows),
        "teacher_solver_calls": 0,
        "passed_snapshots": sum(row["metrics"]["passed"] for row in rows),
        "all_passed": all(row["metrics"]["passed"] for row in rows),
        "mean_MAE": float(np.mean([row["metrics"]["MAE"] for row in rows])),
        "max_MAE": max(row["metrics"]["MAE"] for row in rows),
        "max_abs_delta_Q_mass": max(
            abs(row["metrics"]["delta_Q_mass"]) for row in rows
        ),
        "max_active_mean_relative_power_error": max(active_errors, default=None),
        "max_late_relative_to_neutral_rms": max(late_errors, default=None),
        "minimum_defined_low_k_cross_correlation": min(correlations, default=None),
        "max_photon_residual": max(
            row["photon_ledger"]["relative_photon_residual"] for row in rows
        ),
        "sum_inference_seconds": sum(row["inference_seconds"] for row in rows),
        "mfp": mfp,
    }


def evaluate(
    model: Path,
    teachers: list[Path],
    output: Path,
    *,
    workers: int = 1,
    mfp_rays: int = 4096,
) -> dict[str, Any]:
    """Freeze the evaluation inputs and score every independent realization."""
    if output.exists():
        raise FileExistsError(output)
    if (
        type(workers) is not int
        or workers < 1
        or type(mfp_rays) is not int
        or mfp_rays < 0
    ):
        raise ValueError("Worker count must be positive and MFP ray count nonnegative")
    frozen = {name: _digest(model / name) for name in ("model.json", "weights.npz")}
    emulator = ExposureEmulator.load(model)
    if onset_hash := emulator.metadata.get("onset_sha256"):
        frozen["onset.npz"] = onset_hash
    signature = emulator.metadata["teacher_signature"]
    splits = emulator.metadata["split_seeds"]
    if (
        not splits.get("train")
        or not splits.get("validation")
        or any(
            not isinstance(split, list) or any(type(seed) is not int for seed in split)
            for split in splits.values()
        )
        or set(splits["train"]) & set(splits["validation"])
    ):
        raise ValueError(
            "Need valid disjoint training and validation seeds before independent evaluation"
        )
    development = {seed for split in splits.values() for seed in split}
    supported = {
        (row["DIM"], row["HII_DIM"], row["BOX_LEN"])
        for row in emulator.metadata["training_geometries"]
    }
    seen = set()
    records = []
    for teacher in teachers:
        audited = load_dataset(teacher)
        if compatibility_signature(audited, allow_mixed_geometry=True) != signature:
            raise ValueError(
                "Teacher campaign physics or solver controls differ from training"
            )
        for sample in audited.samples:
            key = (sample.seed, sample.redshift)
            if key in seen or sample.seed in development:
                raise ValueError(
                    f"Evaluation sample repeats or overlaps a development seed: {key}"
                )
            geometry = sample.metadata["input"]["inputs"]["simulation_options"]
            if (
                geometry["DIM"],
                geometry["HII_DIM"],
                geometry["BOX_LEN"],
            ) not in supported:
                raise ValueError(
                    "Evaluation uses a geometry absent from the frozen model"
                )
            seen.add(key)
            records.append(
                {
                    "seed": sample.seed,
                    "redshift": sample.redshift,
                    "box_len": sample.box_len,
                    "dimension": sample.dimension,
                    "teacher_fields": str(sample.fields_path),
                    "teacher_sha256": audited.file_hashes[str(sample.fields_path)],
                    "native_directory": str(sample.native_directory),
                    "native_sha256": {
                        name: audited.file_hashes[str(sample.native_directory / name)]
                        for name in (
                            "native.json",
                            "source.npz",
                            "receipt.json",
                            "native.h5",
                        )
                    },
                }
            )
    if not records:
        raise ValueError("Require at least one audited independent teacher sample")
    _verify_files(model, frozen)
    output.mkdir(parents=True)
    protocol = {
        "schema": 1,
        "evaluator_sha256": _digest(Path(__file__)),
        "model_sha256": frozen,
        "teacher_roots": [str(root.resolve()) for root in teachers],
        "records": records,
        "development_seeds": sorted(development),
        "acceptance": THRESHOLDS,
        "workers": workers,
        "mfp_rays_per_phase": mfp_rays,
        "mfp_draw_seed": DRAW_SEED,
        "mfp_threshold_xHI": 0.5,
        "mfp_scope": "Geometric first-exit length; one box length cap; empty phases undefined.",
    }
    _write(output / "protocol.json", protocol)
    start = time.perf_counter()
    tasks = [(model, output, record, frozen, mfp_rays) for record in records]
    rows = []
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for row in pool.map(_score, tasks):
            rows.append(row)
            LOGGER.info(
                "seed %d z=%g MAE=%.5g passed=%s",
                row["seed"],
                row["redshift"],
                row["metrics"]["MAE"],
                row["metrics"]["passed"],
            )
    _verify_files(model, frozen)
    groups = sorted({(row["box_len"], row["dimension"]) for row in rows})
    result = {
        "schema": 1,
        "complete": True,
        "method": emulator.metadata["method"],
        "protocol_sha256": _digest(output / "protocol.json"),
        "summary": _summary(rows),
        "groups": [
            {
                "box_len": box_len,
                "dimension": dimension,
                **_summary(
                    [
                        row
                        for row in rows
                        if (row["box_len"], row["dimension"]) == (box_len, dimension)
                    ]
                ),
            }
            for box_len, dimension in groups
        ],
        "elapsed_seconds": time.perf_counter() - start,
        "rows": rows,
    }
    _write(output / "evaluation.json", result)
    return result


def main() -> None:
    """Score and save a frozen pure model on archived independent teachers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--mfp-rays", type=int, default=4096)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = evaluate(
        args.model,
        args.teacher,
        args.output,
        workers=args.workers,
        mfp_rays=args.mfp_rays,
    )
    LOGGER.info("%s", json.dumps(result["summary"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
