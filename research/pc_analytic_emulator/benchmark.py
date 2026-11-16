"""Independently benchmark frozen emulator models on audited native teachers.

This runner does not fit or select a model. Field, spectrum and ray statistics
use the standalone reproduction helpers, and saved neutral fields plus stored
unabsorbed reservoirs are audited without trusting an inference ledger.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from research.pc_analytic.validation_20261116.reproduction import recompute
from research.pc_analytic_emulator.data import TeacherSample, load_dataset
from research.pc_analytic_emulator.model import Emulator

LOGGER = logging.getLogger(__name__)
METHODS = ("selected", "previous", "native_es")
LEDGER_TOLERANCE = 1e-10


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _model_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    visited: set[Path] = set()

    def visit(folder: Path) -> None:
        resolved = folder.resolve()
        if resolved in visited or not resolved.is_relative_to(root):
            raise ValueError("Benchmark models require self-contained acyclic bases")
        visited.add(resolved)
        for filename in ("model.json", "field_model.npz", "budget_model.npz"):
            path = folder / filename
            if not path.is_file() or not path.resolve().is_relative_to(root):
                raise ValueError(f"Missing or external model artifact: {path}")
            hashes[str(path.relative_to(root))] = _digest(path)
        record = json.loads((folder / "model.json").read_text())
        if "base_model" in record:
            if record["base_model"] != "base_model":
                raise ValueError(
                    "A benchmark base must use its local base_model directory"
                )
            visit(folder / "base_model")

    visit(root)
    return hashes


def _inference_source_hashes() -> dict[str, str]:
    research = Path(__file__).resolve().parents[1]
    return {
        str(path.relative_to(research.parent)): _digest(path)
        for package in ("pc_analytic_emulator", "pc_field_emulator")
        for path in sorted((research / package).glob("*.py"))
        if not path.name.startswith("test_")
    }


def _physics_compatibility(model: Emulator, physics: dict) -> dict:
    """Match physical inputs exactly while recording explicit grid/box transfer.

    Only DIM, HII_DIM and BOX_LEN may change scientific geometry. Random seed
    and N_THREADS are ignored as realization/runtime controls. This is not a
    guarantee of accuracy at another geometry. Metadata-free synthetic models
    remain usable with an explicit unavailable compatibility result.
    """
    expected = model.metadata.get("teacher_physics")
    if expected is None:
        report = {
            "available": False,
            "physical_parameters_matched": None,
            "geometry_transfer": None,
            "reason": "Model has no recorded teacher_physics; compatibility is unavailable",
        }
    else:
        if not isinstance(expected, dict) or not isinstance(physics, dict):
            raise TypeError("Recorded and audited teacher physics must be dictionaries")
        training, evaluated = copy.deepcopy(expected), copy.deepcopy(physics)
        geometry = {}
        for record in (training, evaluated):
            record.pop("random_seed", None)
            if not isinstance(record.get("simulation_options"), dict):
                raise TypeError("Teacher physics lacks a simulation_options dictionary")
            record["simulation_options"].pop("N_THREADS", None)
        for key in ("DIM", "HII_DIM", "BOX_LEN"):
            old = training["simulation_options"].pop(key, None)
            new = evaluated["simulation_options"].pop(key, None)
            if old is None or new is None:
                raise ValueError(f"Teacher physics lacks required geometry: {key}")
            geometry[key] = {"training": old, "evaluation": new, "changed": old != new}
        if training != evaluated:
            groups = sorted(
                key
                for key in set(training) | set(evaluated)
                if training.get(key) != evaluated.get(key)
            )
            raise ValueError(
                "Benchmark physical parameters differ from model training outside "
                f"permitted geometry/runtime changes: {', '.join(groups)}"
            )
        report = {
            "available": True,
            "physical_parameters_matched": True,
            "geometry_transfer": any(item["changed"] for item in geometry.values()),
            "geometry": geometry,
            "ignored_runtime_or_realization": [
                "random_seed",
                "simulation_options.N_THREADS",
            ],
            "scope": "Cosmology, astrophysics, flags and other native options matched exactly; recorded grid/box transfer is evaluated, not assumed accurate",
        }
    if model.base_model is not None:
        report["frozen_base"] = _physics_compatibility(model.base_model, physics)
    return report


def independent_ledger(
    source: np.ndarray,
    capacity: np.ndarray,
    xHI: np.ndarray,
    unabsorbed: float,
    target_absorbed: float | None = None,
) -> dict[str, Any]:
    """Audit stored fields and separately stored counts in extended precision.

    The unabsorbed scalar must be supplied, never inferred from this field.
    This tests global bookkeeping, not transport or the accuracy of the
    absorbed/unabsorbed partition.
    """
    photons, atoms, neutral = (np.asarray(a) for a in (source, capacity, xHI))
    if (
        photons.shape != atoms.shape
        or neutral.shape != atoms.shape
        or not atoms.size
        or any(
            np.iscomplexobj(a) or not np.isfinite(a).all()
            for a in (photons, atoms, neutral)
        )
        or np.any(photons < 0)
        or np.any(atoms <= 0)
        or np.any((neutral < 0) | (neutral > 1))
        or not np.isfinite(unabsorbed)
        or unabsorbed < 0
    ):
        raise ValueError(
            "Invalid stored fields or unabsorbed reservoir for photon audit"
        )
    emitted = recompute.total(photons)
    absorbed = recompute.total(
        atoms.astype(np.longdouble) * (1 - neutral.astype(np.longdouble))
    )
    residual = abs(emitted - absorbed - np.longdouble(unabsorbed))
    denominator = max(emitted, np.longdouble(np.finfo(float).tiny))
    relative = float(residual / denominator)
    target_error = None
    if target_absorbed is not None:
        if not np.isfinite(target_absorbed) or target_absorbed < 0:
            raise ValueError("Stored absorbed target must be finite and nonnegative")
        target_error = float(
            abs(absorbed - np.longdouble(target_absorbed)) / denominator
        )
    return {
        "emitted_photons": float(emitted),
        "absorbed_photons_from_saved_field": float(absorbed),
        "stored_unabsorbed_photons": float(unabsorbed),
        "absolute_photon_residual": float(residual),
        "relative_photon_residual": relative,
        "relative_absorbed_target_error": target_error,
        "bounded": True,
        "passed": relative <= LEDGER_TOLERANCE
        and (target_error is None or target_error <= LEDGER_TOLERANCE),
        "scope": "Global field absorption plus separately saved unabsorbed inventory; no local transport or independent escape measurement",
    }


def _brightness(fields: dict, atoms: np.ndarray, z: float, physics: dict) -> dict:
    if physics.get("astro_options", {}).get("USE_TS_FLUCT") is not False:
        raise ValueError("Benchmark brightness requires USE_TS_FLUCT=False")
    cosmology = physics["cosmo_params"]
    h, ob, om = (float(cosmology[k]) for k in ("hlittle", "OMb", "OMm"))
    if not all(np.isfinite(v) and v > 0 for v in (h, ob, om)):
        raise ValueError("Benchmark brightness requires finite positive cosmology")
    coefficient = 27 * ob * h**2 / 0.023 * np.sqrt(0.15 / (om * h**2) * (1 + z) / 10)
    return {name: coefficient * atoms * field for name, field in fields.items()}


def _endpoint(field: np.ndarray) -> dict:
    return {
        "exactly_fully_ionized": bool(np.all(field == 0)),
        "exactly_fully_neutral": bool(np.all(field == 1)),
        "zero_neutral_cell_fraction": float(np.mean(field == 0)),
        "unit_neutral_cell_fraction": float(np.mean(field == 1)),
    }


def _snapshot(
    sample: TeacherSample,
    models: dict[str, Emulator],
    physics: dict,
    output: Path,
    rays: int,
) -> dict:
    arrays = sample.arrays
    photons, atoms, native, truth = (
        arrays[key] for key in ("source", "capacity", "native_es_xHI", "teacher_xHI")
    )
    predictions, inference_seconds = {}, {}
    for name, model in models.items():
        start = time.perf_counter()
        predictions[name] = model.predict(
            photons, atoms, native, sample.redshift, sample.box_len
        )
        inference_seconds[name] = time.perf_counter() - start
    fields = {
        "teacher": truth,
        "selected": predictions["selected"]["xHI"],
        "previous": predictions["previous"]["xHI"],
        "native_es": native,
    }
    folder = (
        output / f"seed{sample.seed}" / f"z{sample.redshift:.12g}".replace(".", "p")
    )
    folder.mkdir(parents=True)
    stored = {"source": photons, "capacity": atoms}
    stored.update({f"{name}_xHI": field for name, field in fields.items()})
    for name, prediction in predictions.items():
        for key in ("predicted_absorbed_photons", "predicted_unabsorbed_photons"):
            stored[f"{name}_{key}"] = np.asarray(prediction[key], dtype=np.float64)
        stored[f"{name}_latent_xHII"] = prediction["latent_xHII"]
    for name in ("remaining_photons", "escaped_by_direction", "unemitted_source"):
        stored[f"teacher_{name}"] = arrays[name]
    np.savez_compressed(folder / "predictions.npz", **stored)
    # Read the materialized artifact back; do not accept the model's ledger.
    with np.load(folder / "predictions.npz", allow_pickle=False) as saved:
        ledgers = {
            name: independent_ledger(
                saved["source"],
                saved["capacity"],
                saved[f"{name}_xHI"],
                float(saved[f"{name}_predicted_unabsorbed_photons"]),
                float(saved[f"{name}_predicted_absorbed_photons"]),
            )
            for name in models
        }
        teacher_reservoir = sum(
            (
                recompute.total(saved[f"teacher_{name}"])
                for name in (
                    "remaining_photons",
                    "escaped_by_direction",
                    "unemitted_source",
                )
            ),
            np.longdouble(0),
        )
        ledgers["teacher"] = independent_ledger(
            saved["source"], saved["capacity"], saved["teacher_xHI"], teacher_reservoir
        )
    if not all(ledger["passed"] for ledger in ledgers.values()):
        raise FloatingPointError("Independent saved-field photon ledger failed")
    brightness = _brightness(fields, atoms, sample.redshift, physics)
    for name, key in (
        ("teacher", "teacher_brightness"),
        ("native_es", "native_brightness"),
    ):
        if key in arrays and not np.allclose(
            brightness[name], arrays[key], rtol=3e-6, atol=2e-6
        ):
            raise ValueError(
                "Archived brightness differs from common saturated formula"
            )
    field_metrics = {
        name: recompute.metrics(fields[name], truth, atoms) for name in METHODS
    }
    for name, values in field_metrics.items():
        values["absolute_delta_Q_mass"] = abs(values["delta_Q_mass"])
        values["capacity_weighted_MSE"] = float(
            recompute.total(atoms.astype(np.longdouble) * (fields[name] - truth) ** 2)
            / recompute.total(atoms)
        )
    ray_seed = int(
        hashlib.sha256(
            f"benchmark/{sample.seed}/{sample.redshift:.12g}".encode()
        ).hexdigest()[:16],
        16,
    )
    draws = np.random.default_rng(ray_seed).random((rays, 6))
    mfp, ray_arrays = {}, {"draws": draws}
    start = time.perf_counter()
    for phase in ("ionized", "neutral"):
        measurements = {
            name: recompute.geometric_mfp(field, sample.box_len, phase, draws)
            for name, field in fields.items()
        }
        mfp[phase] = {
            "measurements": {
                name: measurement[0] for name, measurement in measurements.items()
            },
            "comparisons": {
                name: recompute.compare_mfp(measurements[name], measurements["teacher"])
                for name in METHODS
            },
        }
        for name, (_, distances, censored) in measurements.items():
            ray_arrays[f"{phase}_{name}_length_cMpc"] = distances
            ray_arrays[f"{phase}_{name}_censored"] = censored
    mfp_seconds = time.perf_counter() - start
    np.savez_compressed(folder / "mfp_rays.npz", **ray_arrays)
    absorbed_es = recompute.total(
        atoms.astype(np.longdouble) * (1 - native.astype(np.longdouble))
    )
    row = {
        "seed": sample.seed,
        "redshift": sample.redshift,
        "dimension": sample.dimension,
        "box_len_cMpc": sample.box_len,
        "teacher_xHI_mass": float(
            recompute.total(atoms.astype(np.longdouble) * truth)
            / recompute.total(atoms)
        ),
        "field_metrics": field_metrics,
        "photon_ledgers": ledgers,
        "endpoints": {name: _endpoint(field) for name, field in fields.items()},
        "low_k_21cm": recompute.spectra(brightness, sample.box_len, 0.5),
        "full_band_21cm": recompute.spectra(brightness, sample.box_len, 1.0),
        "mfp": mfp,
        "mfp_draw_seed": ray_seed,
        "inference_seconds_excluding_io": inference_seconds,
        "mfp_seconds_excluding_io": mfp_seconds,
        "native_es_budget": {
            "absorbed_atoms_from_field": float(absorbed_es),
            "target_minus_field_absorption_over_atoms": float(
                (min(recompute.total(photons), recompute.total(atoms)) - absorbed_es)
                / recompute.total(atoms)
            ),
            "scope": "Closed-budget discrepancy only; native ES has no separately stored unabsorbed reservoir",
        },
        "original_teacher_fields_sha256": _digest(sample.fields_path),
        "files": {
            name: {
                "path": str((folder / name).relative_to(output)),
                "sha256": _digest(folder / name),
            }
            for name in ("predictions.npz", "mfp_rays.npz")
        },
    }
    _write(folder / "metrics.json", row)
    return row


def _defined_summary(values: list[float | None]) -> dict:
    defined = [value for value in values if value is not None]
    return {
        "defined_count": len(defined),
        "undefined_count": len(values) - len(defined),
        "mean": float(np.mean(defined)) if defined else None,
        "maximum": float(np.max(defined)) if defined else None,
    }


def aggregate(rows: list[dict]) -> dict:
    """Summarize snapshots equally while retaining undefined statistics."""
    result = {"count": len(rows), "methods": {}}
    for method in METHODS:
        fields = [row["field_metrics"][method] for row in rows]
        metrics = {
            key: _defined_summary([field[key] for field in fields])
            for key in (
                "MAE",
                "RMSE",
                "absolute_delta_Q_mass",
                "maximum_absolute_error",
                "field_correlation",
                "capacity_weighted_MSE",
            )
        }
        metrics["mean_signed_delta_Q_mass"] = (
            float(np.mean([f["delta_Q_mass"] for f in fields])) if fields else None
        )
        for band in ("low_k_21cm", "full_band_21cm"):
            spectra = [row[band]["comparisons"][method] for row in rows]
            metrics[band] = {
                "snapshot_mean_absolute_power_ratio_error": _defined_summary(
                    [p["mean_absolute_power_ratio_error"] for p in spectra]
                ),
                "snapshot_maximum_absolute_power_ratio_error": _defined_summary(
                    [p["maximum_absolute_power_ratio_error"] for p in spectra]
                ),
                "minimum_defined_cross_correlation": min(
                    (
                        p["minimum_cross_correlation"]
                        for p in spectra
                        if p["minimum_cross_correlation"] is not None
                    ),
                    default=None,
                ),
                "defined_shells": sum(p["defined_shells"] for p in spectra),
            }
        metrics["mfp"] = {}
        for phase in ("ionized", "neutral"):
            comparisons = [row["mfp"][phase]["comparisons"][method] for row in rows]
            metrics["mfp"][phase] = {
                "available_snapshots": sum(c["available"] for c in comparisons),
                "restricted_mean_absolute_difference_cMpc": _defined_summary(
                    [
                        abs(c["restricted_mean_difference_cMpc"])
                        if c["available"]
                        else None
                        for c in comparisons
                    ]
                ),
                "restricted_CDF_maximum_difference": _defined_summary(
                    [
                        c["restricted_CDF_maximum_difference"]
                        if c["available"]
                        else None
                        for c in comparisons
                    ]
                ),
            }
        result["methods"][method] = metrics
    return result


def grouped_summary(rows: list[dict]) -> dict:
    """Keep epoch, realization and teacher-phase averages distinct."""
    return {
        "all": aggregate(rows),
        "early": aggregate([r for r in rows if r["teacher_xHI_mass"] >= 0.95]),
        "active": aggregate([r for r in rows if 0.05 < r["teacher_xHI_mass"] < 0.95]),
        "late": aggregate([r for r in rows if r["teacher_xHI_mass"] <= 0.05]),
        "z6": aggregate([r for r in rows if r["redshift"] == 6]),
        "by_seed": {
            str(seed): aggregate([r for r in rows if r["seed"] == seed])
            for seed in sorted({r["seed"] for r in rows})
        },
        "by_redshift": {
            f"{z:g}": aggregate([r for r in rows if r["redshift"] == z])
            for z in sorted({r["redshift"] for r in rows}, reverse=True)
        },
    }


def run(
    teacher: str | Path,
    model: str | Path,
    reference_model: str | Path,
    output: str | Path,
    *,
    seeds: list[int] | None = None,
    rays: int = 4096,
) -> dict:
    """Benchmark two frozen models without fitting, selection or input changes."""
    roots = {
        "selected": Path(model).resolve(),
        "previous": Path(reference_model).resolve(),
    }
    teacher, output = Path(teacher).resolve(), Path(output).resolve()
    if isinstance(rays, bool) or not isinstance(rays, int) or rays < 2:
        raise ValueError("MFP rays must be an integer >= 2")
    if seeds is not None and (not seeds or len(set(seeds)) != len(seeds)):
        raise ValueError("Requested seeds must be nonempty and unique")
    if output.exists() or any(
        output.is_relative_to(root) for root in (teacher, *roots.values())
    ):
        raise ValueError(
            "Benchmark output must be new and outside all inputs and models"
        )
    before = {name: _model_hashes(root) for name, root in roots.items()}
    implementation_before = _inference_source_hashes()
    output.mkdir(parents=True)
    manifest = {
        "schema": 1,
        "state": "models frozen before teacher evaluation",
        "teacher_root": str(teacher),
        "requested_seeds": seeds,
        "rays_per_phase": rays,
        "models": {
            name: {"root": str(roots[name]), "files_sha256": hashes}
            for name, hashes in before.items()
        },
        "benchmark_sha256": _digest(Path(__file__)),
        "independent_metrics_sha256": _digest(Path(recompute.__file__)),
        "inference_source_sha256": implementation_before,
        "fitting_performed": False,
        "selection_performed": False,
    }
    _write(output / "manifest.json", manifest)
    dataset = load_dataset(teacher)
    available = {sample.seed for sample in dataset.samples}
    if seeds is not None and not set(seeds) <= available:
        raise ValueError("Requested seeds are missing from audited teacher data")
    samples = [
        sample for sample in dataset.samples if seeds is None or sample.seed in seeds
    ]
    if not samples:
        raise ValueError("No audited teacher snapshots selected")
    _write(
        output / "inputs.json",
        {
            "identity": dataset.identity,
            "physics": dataset.physics,
            "audit": dataset.audit,
            "files_sha256": dataset.file_hashes,
        },
    )
    models = {name: Emulator.load(root) for name, root in roots.items()}
    physical_compatibility = {
        name: _physics_compatibility(fitted, dataset.physics)
        for name, fitted in models.items()
    }
    if any(_model_hashes(roots[name]) != hashes for name, hashes in before.items()):
        raise ValueError("Model artifacts changed while loading")
    rows = []
    for sample in samples:
        LOGGER.info("Benchmark seed %s z=%g", sample.seed, sample.redshift)
        rows.append(_snapshot(sample, models, dataset.physics, output, rays))
    after = {name: _model_hashes(root) for name, root in roots.items()}
    if after != before:
        raise ValueError("Model artifacts changed during benchmark")
    if _inference_source_hashes() != implementation_before:
        raise ValueError("Inference implementation changed during benchmark")
    report = {
        "schema": 1,
        "complete": True,
        "snapshot_count": len(rows),
        "model_weights_unchanged": True,
        "inference_implementation_unchanged": True,
        "model_files_after_sha256": after,
        "physical_compatibility": physical_compatibility,
        "fitting_performed": False,
        "model_selection_performed": False,
        "averaging": "Equal snapshot weights; power averages defined shells first; undefined endpoint ratios excluded with counts retained",
        "brightness": "Common saturated-spin real-space C(z)*(1+delta)*xHI, without RSD",
        "mfp_scope": "Geometric first exit at xHI=0.5; periodic box, cap=box length; common random variates, means include capped rays; empty phases undefined",
        "full_band_scope": "Spherical 0<|k|<=k_Nyquist, not FFT cube corner modes",
        "timing_scope": "Individual model.predict calls include features/projection and exclude I/O/diagnostics; MFP traversal/comparison measured separately; no native-ES runtime measured",
        "timings": {
            "inference_seconds_excluding_io": {
                name: _defined_summary(
                    [row["inference_seconds_excluding_io"][name] for row in rows]
                )
                for name in models
            },
            "mfp_seconds_excluding_io": _defined_summary(
                [row["mfp_seconds_excluding_io"] for row in rows]
            ),
        },
        "maximum_relative_photon_residual": max(
            ledger["relative_photon_residual"]
            for row in rows
            for ledger in row["photon_ledgers"].values()
        ),
        "summaries": grouped_summary(rows),
        "snapshots": rows,
        "input_record_sha256": _digest(output / "inputs.json"),
        "frozen_manifest_sha256": _digest(output / "manifest.json"),
    }
    _write(output / "report.json", report)
    hashes = {
        str(path.relative_to(output)): _digest(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    _write(output / "receipt.json", {"complete": True, "files_sha256": hashes})
    (output / "COMPLETED").write_text("Independent frozen-model benchmark complete\n")
    return report


def main() -> None:
    """Run independent diagnostics on an audited campaign or collection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--rays", type=int, default=4096)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = run(
        args.teacher,
        args.model,
        args.reference_model,
        args.output,
        seeds=args.seeds,
        rays=args.rays,
    )
    sys.stdout.write(
        json.dumps(
            {
                "complete": report["complete"],
                "snapshots": report["snapshot_count"],
                "report": str(args.output / "report.json"),
            }
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
