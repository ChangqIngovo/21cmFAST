"""Audit sealed external seeds against validation-frozen emulator checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from recompute import compare_mfp, digest, geometric_mfp, metrics, spectra, total

LOGGER = logging.getLogger(__name__)


def verify_inference_source(source: Path, identity: dict) -> None:
    """Require the exact inference implementation recorded before evaluation."""
    for relative in (
        "research/pc_analytic_emulator/model.py",
        "research/pc_analytic_emulator/budget.py",
        "research/pc_analytic_emulator/latent.py",
        "research/pc_analytic_emulator/projection.py",
        "research/pc_field_emulator/model.py",
        "research/pc_field_emulator/features.py",
    ):
        if digest(source / relative) != identity["implementation_sha256"][relative]:
            raise ValueError(
                f"Inference source differs from frozen training: {relative}"
            )


def verify_fields(fields: dict[str, np.ndarray], atoms: np.ndarray) -> None:
    """Validate saved finite fractions before independently measuring them."""
    for values in fields.values():
        if (
            values.shape != atoms.shape
            or not np.isfinite(values).all()
            or np.any((values < 0) | (values > 1))
        ):
            raise ValueError("Invalid saved ionization field")


def summarize(rows: list[dict]) -> dict:
    """Average snapshot errors without treating correlated voxels as samples."""
    result = {"count": len(rows), "metrics": {}}
    for name in ("selected", "legacy", "raw", "native_es"):
        values = [row["field_metrics"][name] for row in rows]
        power = [
            row["low_k_21cm"]["comparisons"][name]["mean_absolute_power_ratio_error"]
            for row in rows
        ]
        power = [value for value in power if value is not None]
        result["metrics"][name] = {
            key: float(np.mean([value[key] for value in values])) if values else None
            for key in ("MAE", "RMSE")
        }
        result["metrics"][name].update(
            mean_absolute_delta_Q_mass=float(
                np.mean([abs(value["delta_Q_mass"]) for value in values])
            )
            if values
            else None,
            maximum_absolute_delta_Q_mass=max(
                [abs(value["delta_Q_mass"]) for value in values], default=None
            ),
            mean_snapshot_low_k_power_ratio_error=float(np.mean(power))
            if power
            else None,
            defined_power_snapshots=len(power),
        )
    return result


def phases(rows: list[dict]) -> dict:
    """Separate active evolution from nearly neutral and late snapshots."""
    return {
        "all": summarize(rows),
        "active": summarize(
            [row for row in rows if 0.05 < row["teacher_xHI_mass"] < 0.95]
        ),
        "late": summarize([row for row in rows if row["teacher_xHI_mass"] <= 0.05]),
        "early": summarize([row for row in rows if row["teacher_xHI_mass"] >= 0.95]),
    }


def main() -> None:
    """Freeze both model choices before loading any external teacher labels."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--rays", type=int, default=32768)
    args = parser.parse_args()
    started = time.monotonic()
    if not (args.training / "COMPLETED").is_file():
        raise ValueError("Training must finish selection before external inference")
    if len(set(args.seeds)) != len(args.seeds) or args.rays < 1:
        raise ValueError("Unique external seeds and positive ray count required")
    identity = json.loads((args.training / "identity.json").read_text())
    config = identity["configuration"]
    prior_seeds = set(
        config["train_seeds"] + config["validation_seeds"] + config["test_seeds"]
    )
    if prior_seeds.intersection(args.seeds):
        raise ValueError(
            "External seeds overlap fitting, selection or prior regression seeds"
        )
    verify_inference_source(args.source, identity)
    selection = json.loads((args.training / "model_selection.json").read_text())
    if selection["test_used_for_selection"]:
        raise ValueError("Model choice used test labels")
    legacy_objective, legacy_seed = min(
        (
            next(
                row for row in trial["candidates"] if row["kind"] == "legacy_reference"
            )["validation_objective"],
            trial["model_seed"],
        )
        for trial in selection["trials"]
    )
    model_paths = {
        "selected": args.training / "model",
        "legacy": args.training
        / "trials"
        / f"seed{legacy_seed}"
        / "legacy_reference"
        / "model",
    }
    sys.path.insert(0, str(args.source))
    from research.pc_analytic_emulator.model import Emulator

    models = {name: Emulator.load(path) for name, path in model_paths.items()}
    args.output.mkdir(parents=True, exist_ok=True)
    artifact_hashes = {
        name: {
            str(path.relative_to(root)): digest(path)
            for path in root.rglob("*")
            if path.is_file()
        }
        for name, root in model_paths.items()
    }
    draws = np.random.default_rng(20261116).random((args.rays, 6))
    rows = []
    expected = 0
    for seed in args.seeds:
        seed_root = args.teacher / "seeds" / f"seed{seed}"
        seed_summary = json.loads((seed_root / "summary.json").read_text())
        if not (seed_root / "COMPLETED").is_file() or not seed_summary["complete"]:
            raise ValueError("External teacher seed is incomplete")
        expected += len(seed_summary["snapshots"])
        native_inputs = json.loads((seed_root / "native_inputs.json").read_text())
        if native_inputs["astro_options"]["USE_TS_FLUCT"] is not False:
            raise ValueError(
                "Independent brightness formula assumes saturated spin temperature"
            )
        cosmology = native_inputs["cosmo_params"]
        for snapshot in seed_summary["snapshots"]:
            redshift, length = snapshot["redshift"], snapshot["box_len"]
            teacher_path = (
                seed_root
                / "teacher"
                / f"z{redshift:.12g}".replace(".", "p")
                / "fields.npz"
            )
            with np.load(teacher_path, allow_pickle=False) as archive:
                physical = {key: np.array(archive[key]) for key in archive.files}
            source, atoms, native, truth = (
                physical[key]
                for key in ("source", "capacity", "native_es_xHI", "teacher_xHI")
            )
            for name in (
                "source",
                "capacity",
                "ionized",
                "remaining_photons",
                "escaped_by_direction",
                "unemitted_source",
            ):
                if not np.isfinite(physical[name]).all() or np.any(physical[name] < 0):
                    raise ValueError(f"Invalid physical inventory: {name}")
            predictions = {
                name: model.predict(source, atoms, native, redshift, length)
                for name, model in models.items()
            }
            destination = args.output / "predictions" / f"seed{seed}_z{redshift:g}"
            destination.mkdir(parents=True, exist_ok=True)
            arrays = {}
            for name, prediction in predictions.items():
                for key in (
                    "xHI",
                    "xHII",
                    "raw_xHI",
                    "predicted_absorbed_photons",
                    "predicted_unabsorbed_photons",
                    "absorption_efficiency",
                ):
                    arrays[f"{name}_{key}"] = np.asarray(prediction[key])
            prediction_path = destination / "predictions.npz"
            np.savez_compressed(prediction_path, **arrays)
            with np.load(prediction_path, allow_pickle=False) as archive:
                saved = {key: np.array(archive[key]) for key in archive.files}
            fields = {
                "teacher": truth,
                "selected": saved["selected_xHI"],
                "legacy": saved["legacy_xHI"],
                "raw": saved["selected_raw_xHI"],
                "native_es": native,
            }
            verify_fields(fields, atoms)
            emitted, capacity = total(source), total(atoms)
            absorbed_teacher = total(
                atoms.astype(np.longdouble) * (1 - truth.astype(np.longdouble))
            )
            remaining_teacher = sum(
                total(physical[key])
                for key in (
                    "remaining_photons",
                    "escaped_by_direction",
                    "unemitted_source",
                )
            )
            budget = {
                "teacher": {
                    "relative_residual": float(
                        abs(emitted - absorbed_teacher - remaining_teacher) / emitted
                    ),
                    "source_photons": float(emitted),
                    "absorbed": float(absorbed_teacher),
                    "stored_unabsorbed": float(remaining_teacher),
                }
            }
            for name in models:
                ionized = saved[f"{name}_xHII"]
                np.testing.assert_allclose(
                    ionized + fields[name], 1, rtol=0, atol=4 * np.finfo(float).eps
                )
                absorbed = total(atoms.astype(np.longdouble) * ionized)
                unabsorbed = np.longdouble(
                    saved[f"{name}_predicted_unabsorbed_photons"]
                )
                if not np.isfinite(unabsorbed) or unabsorbed < 0:
                    raise ValueError("Invalid predicted photon reservoir")
                budget[name] = {
                    "relative_residual": float(
                        abs(emitted - absorbed - unabsorbed) / emitted
                    ),
                    "absorbed_target_relative_residual": float(
                        abs(absorbed - saved[f"{name}_predicted_absorbed_photons"])
                        / emitted
                    ),
                    "field_absorbed": float(absorbed),
                    "predicted_unabsorbed": float(unabsorbed),
                    "predicted_efficiency": float(
                        saved[f"{name}_absorption_efficiency"]
                    ),
                    "teacher_efficiency": float(
                        absorbed_teacher / min(emitted, capacity)
                    ),
                }
            if any(value["relative_residual"] > 1e-10 for value in budget.values()):
                raise ValueError("Independent photon ledger failed")
            coefficient = (
                27
                * cosmology["OMb"]
                * cosmology["hlittle"] ** 2
                / 0.023
                * np.sqrt(
                    0.15
                    / (cosmology["OMm"] * cosmology["hlittle"] ** 2)
                    * (1 + redshift)
                    / 10
                )
            )
            brightness = {
                name: coefficient * atoms * values for name, values in fields.items()
            }
            for name, stored in (
                ("teacher", "teacher_brightness"),
                ("native_es", "native_brightness"),
            ):
                np.testing.assert_allclose(
                    brightness[name], physical[stored], rtol=3e-6, atol=2e-6
                )
            mfp = {}
            for phase in ("ionized", "neutral"):
                samples = {
                    name: geometric_mfp(fields[name], length, phase, draws)
                    for name in ("teacher", "selected", "legacy", "native_es")
                }
                mfp[phase] = {
                    "fields": {name: sample[0] for name, sample in samples.items()},
                    "comparisons": {
                        name: compare_mfp(samples[name], samples["teacher"])
                        for name in ("selected", "legacy", "native_es")
                    },
                }
            rows.append(
                {
                    "seed": seed,
                    "redshift": redshift,
                    "dimension": len(atoms),
                    "box_len_cMpc": length,
                    "teacher_xHI_mass": float(
                        total(atoms.astype(np.longdouble) * truth) / capacity
                    ),
                    "field_metrics": {
                        name: metrics(values, truth, atoms)
                        for name, values in fields.items()
                    },
                    "budget": budget,
                    "low_k_21cm": spectra(brightness, length, 0.5),
                    "all_resolved_21cm": spectra(brightness, length),
                    "MFP": mfp,
                    "input_sha256": {
                        str(teacher_path): digest(teacher_path),
                        str(prediction_path): digest(prediction_path),
                    },
                }
            )
            LOGGER.info(
                "External seed %s z=%g: selected MAE %.6g; legacy %.6g",
                seed,
                redshift,
                rows[-1]["field_metrics"]["selected"]["MAE"],
                rows[-1]["field_metrics"]["legacy"]["MAE"],
            )
    if len(rows) != expected:
        raise ValueError("Incomplete external snapshot coverage")
    rows.sort(key=lambda row: (row["seed"], -row["redshift"]))
    result = {
        "complete": True,
        "scope": "Independent postselection external inference and field/FFT/MFP audit; no fitting, candidate ranking or selection on external labels. Seed2108 is now regression data and excluded here.",
        "seed_split": {
            key: config[key]
            for key in ("train_seeds", "validation_seeds", "test_seeds")
        },
        "external_seeds": args.seeds,
        "selected": selection["selected"],
        "legacy_selection": {
            "model_seed": legacy_seed,
            "validation_objective": legacy_objective,
        },
        "model_sha256": artifact_hashes,
        "training_root": str(args.training.resolve()),
        "teacher_root": str(args.teacher.resolve()),
        "training_identity_sha256": digest(args.training / "identity.json"),
        "selection_sha256": digest(args.training / "model_selection.json"),
        "scripts_sha256": {
            path.name: digest(path)
            for path in (Path(__file__), Path(__file__).with_name("recompute.py"))
        },
        "MFP_settings": {
            "rays_per_field_phase": args.rays,
            "threshold_xHI": 0.5,
            "seed": 20261116,
            "draws_sha256": hashlib.sha256(draws.tobytes()).hexdigest(),
            "cap": "one box length",
            "interpretation": "Capped geometric first-exit distance; same variates, phase-dependent origins. Same redshift; volume fractions may differ.",
        },
        "aggregates": phases(rows),
        "by_seed": {
            str(seed): phases([row for row in rows if row["seed"] == seed])
            for seed in args.seeds
        },
        "by_redshift": {
            str(z): phases([row for row in rows if row["redshift"] == z])
            for z in sorted({row["redshift"] for row in rows}, reverse=True)
        },
        "maximum_photon_residual": {
            name: max(row["budget"][name]["relative_residual"] for row in rows)
            for name in ("teacher", "selected", "legacy")
        },
        "elapsed_seconds": time.monotonic() - started,
        "rows": rows,
    }
    (args.output / "external_independent_summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    LOGGER.info(
        json.dumps(
            {
                key: result[key]
                for key in ("aggregates", "maximum_photon_residual", "elapsed_seconds")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
