"""Check whether late emulator errors can be attributed to teacher numerics.

Each saved z=6 native realization is rerun with its original d96/e32 settings,
then with d384/e32 and d96/e64 separately. No labels or inference models are fit.
The full-complex-FFT diagnostic selects individual modes below half Nyquist.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from recompute import digest, metrics, spectra, total

LOGGER = logging.getLogger(__name__)


def save_json(path: Path, value: dict[str, Any]) -> None:
    """Write a JSON record atomically without nonfinite scalar values."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def ledger(
    source: np.ndarray,
    atoms: np.ndarray,
    neutral: np.ndarray,
    remaining: np.ndarray,
    escaped: np.ndarray,
    unemitted: np.ndarray,
) -> dict[str, Any]:
    """Reconstruct photon absorption and unabsorbed reservoirs from arrays."""
    emitted, capacity = total(source), total(atoms)
    absorbed = total(atoms.astype(np.longdouble) * (1 - neutral.astype(np.longdouble)))
    reservoirs = total(remaining) + total(escaped) + total(unemitted)
    bounded = bool(
        np.isfinite(neutral).all() and np.all((neutral >= 0) & (neutral <= 1))
    )
    residual = float(
        abs(emitted - absorbed - reservoirs) / max(emitted, np.finfo(float).tiny)
    )
    if not bounded or residual > 1e-10:
        raise ValueError("Independent teacher photon ledger failed")
    return {
        "relative_residual": residual,
        "bounded": bounded,
        "source": float(emitted),
        "atoms": float(capacity),
        "absorbed": float(absorbed),
        "remaining": float(total(remaining)),
        "escaped": float(total(escaped)),
        "unemitted": float(total(unemitted)),
        "xHI_mass": float(total(atoms.astype(np.longdouble) * neutral) / capacity),
        "xHI_volume": float(np.mean(neutral)),
    }


def load_prediction(
    root: Path | None,
    audit: dict[str, Any] | None,
    seed: int,
    redshift: float,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Read saved inference arrays only after matching the external audit hash."""
    if root is None or audit is None:
        return {}, {}
    rows = [
        row
        for row in audit["rows"]
        if row["seed"] == seed and row["redshift"] == redshift
    ]
    if len(rows) != 1 or not audit["complete"]:
        raise ValueError("External prediction audit is incomplete or ambiguous")
    path = root / f"seed{seed}_z{redshift:g}" / "predictions.npz"
    hashes = [
        value
        for key, value in rows[0]["input_sha256"].items()
        if key.endswith("/predictions.npz")
    ]
    if hashes != [digest(path)]:
        raise ValueError("Saved external predictions differ from their audit")
    with np.load(path, allow_pickle=False) as archive:
        fields = {
            name: np.array(archive[key])
            for name, key in (
                ("selected", "selected_xHI"),
                ("raw", "selected_raw_xHI"),
                ("legacy", "legacy_xHI"),
            )
        }
    return fields, {str(path): hashes[0]}


def verify_fields(fields: dict[str, np.ndarray], atoms: np.ndarray) -> None:
    """Require finite neutral fractions on the common native grid."""
    for name, values in fields.items():
        if (
            values.shape != atoms.shape
            or not np.isfinite(values).all()
            or np.any((values < 0) | (values > 1))
        ):
            raise ValueError(f"Invalid diagnostic neutral field: {name}")


def main() -> None:
    """Rerun the declared late-time teacher refinements and save their evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path(__file__).resolve().parents[4]
    )
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[2109, 2110, 2111, 2112]
    )
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--predictions-root", type=Path)
    parser.add_argument("--external-audit", type=Path)
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Also run d384/e64; do not infer joint refinement from separate axes",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.threads < 1 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Require positive threads and unique seeds")
    if (args.predictions_root is None) != (args.external_audit is None):
        raise ValueError("Predictions require their completed external audit")
    sys.path.insert(0, str(args.source.resolve()))
    from research.pc_analytic.compare import load_native_snapshot
    from research.pc_analytic.solver import build_identity, solve

    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError("Use an empty output directory for a new sensitivity study")
    external = (
        None
        if args.external_audit is None
        else json.loads(args.external_audit.read_text())
    )
    identity = {
        "teacher_root": str(args.teacher_root.resolve()),
        "seeds": args.seeds,
        "redshift": 6.0,
        "threads": args.threads,
        "combined": args.combined,
        "teacher_solver_build": build_identity(),
        "implementation_sha256": {
            str(path.relative_to(args.source.resolve())): digest(path)
            for path in (
                args.source.resolve() / "research/pc_analytic/solver.py",
                args.source.resolve() / "research/pc_analytic/transport_kernel.cpp",
                args.source.resolve() / "research/pc_analytic/compare.py",
            )
        },
        "diagnostic_sha256": digest(Path(__file__)),
        "metric_helper_sha256": digest(Path(__file__).with_name("recompute.py")),
        "external_audit_sha256": None
        if args.external_audit is None
        else digest(args.external_audit),
        "scope": "Numerical sensitivity on unchanged native source/atom grids; no training or inference parameter changes",
        "power_scope": "Independent full-complex FFT, selecting individual modes 0 < k < 0.5 k_Nyquist; weak but nonzero signal ratios remain reported with absolute power",
    }
    save_json(args.output / "identity.json", identity)
    rows = []
    started = time.perf_counter()
    for seed in args.seeds:
        folder = args.teacher_root / "seeds" / f"seed{seed}"
        native = folder / "native" / f"seed{seed}" / "z6"
        teacher = folder / "teacher" / "z6"
        receipt = json.loads((teacher / "receipt.json").read_text())
        for name in ("fields.npz", "metrics.json"):
            if digest(teacher / name) != receipt["files"][name]:
                raise ValueError(f"Teacher checksum mismatch: {teacher / name}")
        saved_metrics = json.loads((teacher / "metrics.json").read_text())
        if receipt["record"] != saved_metrics or not saved_metrics["complete"]:
            raise ValueError("Saved teacher record is incomplete or inconsistent")
        if (
            digest(native / "receipt.json")
            != receipt["identity"]["native_receipt_sha256"]
        ):
            raise ValueError("Saved teacher belongs to a different native snapshot")
        config = receipt["identity"]["campaign"]["configuration"]
        data = load_native_snapshot(native, brightness="none")
        source, atoms, meta = data["source"], data["capacity"], data["metadata"]
        if (
            meta["redshift"] != 6.0
            or source.shape != (16, 16, 16)
            or meta["box_len"] != 50.0
        ):
            raise ValueError("This diagnostic expects the declared z6 L50 N16 campaign")
        with np.load(teacher / "fields.npz", allow_pickle=False) as archive:
            saved = {key: np.array(archive[key]) for key in archive.files}
        if not np.array_equal(saved["source"], source) or not np.array_equal(
            saved["capacity"], atoms
        ):
            raise ValueError("Saved teacher input arrays differ from native inputs")
        controls = dict(config["analytic"])
        if (
            controls["directions"] != 96
            or controls["exposure_steps"] != 32
            or controls["boundary"] != "source_periodic_cube"
        ):
            raise ValueError("Unexpected baseline teacher numerical settings")
        physical = {
            "box_len": meta["box_len"],
            "redshift": meta["redshift"],
            "nH_mean_comoving_cm3": meta["nH_mean_comoving_cm3"],
            "sigma_cm2": config["sigma_cm2"],
        }
        fields = {"teacher": saved["teacher_xHI"]}
        predictions, prediction_hash = load_prediction(
            args.predictions_root, external, seed, 6.0
        )
        fields.update(predictions)
        cosmology = meta["inputs"]["cosmo_params"]
        coefficient = (
            27
            * cosmology["OMb"]
            * cosmology["hlittle"] ** 2
            / 0.023
            * np.sqrt(0.15 / (cosmology["OMm"] * cosmology["hlittle"] ** 2) * 7 / 10)
        )
        np.testing.assert_allclose(
            coefficient * atoms * fields["teacher"],
            saved["teacher_brightness"],
            rtol=3e-6,
            atol=2e-6,
        )
        row: dict[str, Any] = {
            "seed": seed,
            "redshift": 6.0,
            "physical": physical,
            "baseline_settings": controls,
            "teacher_receipt_sha256": digest(teacher / "receipt.json"),
            "teacher_fields_sha256": digest(teacher / "fields.npz"),
            "native_receipt_sha256": meta["receipt_sha256"],
            "prediction_sha256": prediction_hash,
            "teacher_ledger": ledger(
                source,
                atoms,
                fields["teacher"],
                saved["remaining_photons"],
                saved["escaped_by_direction"],
                saved["unemitted_source"],
            ),
            "cases": {},
        }
        settings = [
            ("baseline_repeat", 96, 32),
            ("angular_refined", 384, 32),
            ("dose_refined", 96, 64),
        ]
        if args.combined:
            settings.append(("joint_refined", 384, 64))
        for name, directions, exposures in settings:
            run_controls = controls | {
                "directions": directions,
                "exposure_steps": exposures,
            }
            result = solve(
                source, atoms, **physical, **run_controls, threads=args.threads
            )
            if not result["audit"]["complete"]:
                raise RuntimeError(f"Incomplete numerical refinement: {seed} {name}")
            neutral = 1 - result["ionized"] / atoms
            budget = ledger(
                source,
                atoms,
                neutral,
                result["remaining_radiation"],
                result["escaped"],
                result["unemitted_source"],
            )
            fields[name] = neutral
            destination = args.output / f"seed{seed}_{name}.npz"
            np.savez_compressed(
                destination,
                xHI=neutral,
                ionized=result["ionized"],
                remaining=np.sum(result["remaining_radiation"], axis=0),
                escaped=result["escaped"],
                unemitted=result["unemitted_source"],
            )
            row["cases"][name] = {
                "settings": run_controls,
                "audit": result["audit"],
                "independent_ledger": budget,
                "field_metrics_vs_saved_teacher": metrics(
                    neutral, fields["teacher"], atoms
                ),
                "saved_teacher_bitwise_equal": bool(
                    np.array_equal(neutral, fields["teacher"])
                ),
                "output_sha256": digest(destination),
            }
            LOGGER.info(
                "Seed %d %s: xHI_mass %.8g MAE %.8g time %.3fs",
                seed,
                name,
                budget["xHI_mass"],
                row["cases"][name]["field_metrics_vs_saved_teacher"]["MAE"],
                result["audit"]["elapsed_seconds"],
            )
        if not row["cases"]["baseline_repeat"]["saved_teacher_bitwise_equal"]:
            raise ValueError("Baseline does not reproduce the original teacher bitwise")
        verify_fields(fields, atoms)
        brightness = {
            name: coefficient * atoms * values for name, values in fields.items()
        }
        brightness["fully_neutral_density"] = coefficient * atoms
        row["field_metrics_vs_saved_teacher"] = {
            name: metrics(values, fields["teacher"], atoms)
            for name, values in fields.items()
            if name != "teacher"
        }
        row["strict_low_k_21cm"] = spectra(brightness, meta["box_len"], 0.5)
        row["all_resolved_21cm"] = spectra(brightness, meta["box_len"])
        row["teacher_reference_power"] = {
            "Delta2_mK2": row["strict_low_k_21cm"]["Delta2_teacher"],
            "P_mK2_cMpc3": row["strict_low_k_21cm"]["P_teacher"],
            "mean_brightness_mK": float(np.mean(brightness["teacher"])),
            "rms_brightness_fluctuation_mK": float(np.std(brightness["teacher"])),
            "mean_square_relative_to_fully_neutral": float(
                np.mean(brightness["teacher"] ** 2)
                / np.mean(brightness["fully_neutral_density"] ** 2)
            ),
        }
        row["prediction_errors_with_refined_reference"] = {}
        for name in ("angular_refined", "dose_refined", "joint_refined"):
            if name not in fields or not predictions:
                continue
            reference_fields = {"teacher": brightness[name]} | {
                key: brightness[key] for key in predictions
            }
            row["prediction_errors_with_refined_reference"][name] = {
                "field": {
                    key: metrics(value, fields[name], atoms)
                    for key, value in predictions.items()
                },
                "strict_low_k_21cm": spectra(reference_fields, meta["box_len"], 0.5),
            }
        rows.append(row)
        save_json(
            args.output / "summary.json",
            {"complete": False, "identity": identity, "rows": rows},
        )
    summary = {
        "complete": True,
        "identity": identity,
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
        "maximum_independent_photon_residual": max(
            case["independent_ledger"]["relative_residual"]
            for row in rows
            for case in row["cases"].values()
        ),
        "all_baselines_reproduced_bitwise": all(
            row["cases"]["baseline_repeat"]["saved_teacher_bitwise_equal"]
            for row in rows
        ),
        "spatial_convergence_certified": False,
    }
    save_json(args.output / "summary.json", summary)
    compact = []
    for row in rows:
        power = row["strict_low_k_21cm"]["comparisons"]
        compact.append(
            {
                "seed": row["seed"],
                "teacher_xHI_mass": row["teacher_ledger"]["xHI_mass"],
                "field_metrics": row["field_metrics_vs_saved_teacher"],
                "strict_low_k_power": {
                    key: {
                        k: value[k]
                        for k in (
                            "defined_shells",
                            "mean_absolute_power_ratio_error",
                            "maximum_absolute_power_ratio_error",
                            "minimum_cross_correlation",
                        )
                    }
                    for key, value in power.items()
                },
                "teacher_reference_power": row["teacher_reference_power"],
                "refined_xHI_mass": {
                    key: value["independent_ledger"]["xHI_mass"]
                    for key, value in row["cases"].items()
                },
                "prediction_errors_with_refined_reference": {
                    key: value["field"]
                    for key, value in row[
                        "prediction_errors_with_refined_reference"
                    ].items()
                },
            }
        )
    save_json(
        args.output / "compact_summary.json",
        {key: value for key, value in summary.items() if key != "rows"}
        | {
            "rows": compact,
            "full_summary_sha256": digest(args.output / "summary.json"),
        },
    )
    save_json(
        args.output / "receipt.json",
        {
            "files": {
                p.name: digest(p) for p in sorted(args.output.iterdir()) if p.is_file()
            }
        },
    )
    LOGGER.info(
        "Completed late teacher sensitivity: %s", args.output / "compact_summary.json"
    )


if __name__ == "__main__":
    main()
