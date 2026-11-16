"""Verify post-evaluation oracle controls and export compact path-free evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from research.pc_analytic_emulator import diagnose_errors
from research.pc_analytic_emulator.benchmark import (
    _inference_source_hashes,
    _model_hashes,
)

PROFILES = ("original", "resolution", "volume", "unseen_box")
METHODS = diagnose_errors.METHODS


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    """Return the SHA256 of a saved artifact without changing it."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> dict:
    """Read one previously generated evidence record."""
    return json.loads(path.read_text())


def mean(values: list[float]) -> float | None:
    """Average defined values, leaving empty groups undefined."""
    return math.fsum(values) / len(values) if values else None


def grouped(rows: list[dict]) -> dict:
    """Summarize fixed-weight interventions with equal snapshot weights.

    Parameters
    ----------
    rows
        Snapshot records from ``diagnose_errors``.

    Returns
    -------
    dict
        Field, budget and defined-power statistics plus nonadditive controls.
    """
    result = {"count": len(rows), "methods": {}}
    for name in METHODS:
        fields = [row["field_metrics"][name] for row in rows]
        measured = {
            key: mean([field[key] for field in fields])
            for key in ("MAE", "RMSE", "capacity_weighted_MSE")
        }
        measured["mean_absolute_delta_Q_mass"] = mean(
            [abs(field["delta_Q_mass"]) for field in fields]
        )
        measured["maximum_absolute_delta_Q_mass"] = max(
            (abs(field["delta_Q_mass"]) for field in fields), default=None
        )
        power = [row["low_k_21cm"]["comparisons"][name] for row in rows]
        defined = [
            p["mean_absolute_power_ratio_error"]
            for p in power
            if p["mean_absolute_power_ratio_error"] is not None
        ]
        measured.update(
            mean_snapshot_low_k_power_ratio_error=mean(defined),
            defined_power_snapshots=len(defined),
            undefined_power_snapshots=len(rows) - len(defined),
            defined_power_shells=sum(p["defined_shells"] for p in power),
            minimum_cross_correlation=min(
                (
                    p["minimum_cross_correlation"]
                    for p in power
                    if p["minimum_cross_correlation"] is not None
                ),
                default=None,
            ),
        )
        result["methods"][name] = measured
    deployed = result["methods"]["deployed"]
    controls = {}
    for name in METHODS[1:3]:
        oracle = result["methods"][name]
        controls[name] = {
            f"relative_{metric}_change": oracle[metric] / deployed[metric] - 1
            if deployed[metric]
            else None
            for metric in ("MAE", "capacity_weighted_MSE")
        }
    original_mse = deployed["capacity_weighted_MSE"]
    result["counterfactual_comparisons"] = {
        "oracle_changes_relative_to_deployed": controls,
        "teacher_projection_to_deployed_weighted_MSE_ratio": result["methods"][
            "teacher_predicted_budget"
        ]["capacity_weighted_MSE"]
        / original_mse
        if original_mse
        else None,
        "additive_error_partition": False,
    }
    return result


def _scientific_payload(report: dict) -> dict:
    return {
        key: value
        for key, value in report.items()
        if key not in {"evidence_builder_sha256", "original_evidence"}
    }


def run(
    root: Path, output: Path | None = None, *, original_evidence: Path | None = None
) -> dict:
    """Verify completed oracle fields and export a portable evidence record.

    Parameters
    ----------
    root
        Campaign root containing frozen selection, four completed blind
        benchmarks and the four ``oracle_blind_*`` diagnostic directories.
    output
        New report destination; defaults to ``oracle_evidence_portable.json``
        under root. Frozen inputs and the original report cannot be overwritten.
    original_evidence
        Optional original report whose scientific payload must agree exactly.
        When omitted, use ``root/oracle_evidence.json`` if it exists. Its source
        identity is retained separately from this portable implementation.

    Returns
    -------
    dict
        Path-free summaries, array-equality checks and model/source hashes.
        This command performs no fitting, model selection or diagnostic inference.
    """
    root = root.resolve()
    output = (
        output.resolve()
        if output is not None
        else root / "oracle_evidence_portable.json"
    )
    if original_evidence is None and (root / "oracle_evidence.json").is_file():
        original_evidence = root / "oracle_evidence.json"
    if original_evidence is not None:
        original_evidence = original_evidence.resolve()
    protected = [root / "selected_model"]
    protected.extend(
        root / f"{kind}_blind_{name}"
        for kind in ("oracle", "benchmark")
        for name in PROFILES
    )
    controls = {
        root / "selection.json",
        root / "evaluation_completion.json",
        original_evidence,
    }
    _require(
        not (
            output in controls
            or any(output.is_relative_to(path.resolve()) for path in protected)
        ),
        "Oracle output must not replace frozen inputs or original evidence",
    )
    selection = read(root / "selection.json")
    selection_hash = digest(root / "selection.json")
    model = (root / "selected_model").resolve()
    _require(
        (root / "evaluation_completion.json").is_file(),
        "Oracle verification failed: (root / 'evaluation_completion.json').is_file()",
    )
    _require(
        _model_hashes(model) == selection["model_files_sha256"],
        "Oracle verification failed: _model_hashes(model) == selection['model_files_sha256']",
    )
    _require(
        _inference_source_hashes() == selection["inference_source_sha256"],
        "Oracle verification failed: _inference_source_hashes() == selection['inference_source_sha256']",
    )
    result = {
        "schema": 1,
        "complete": True,
        "diagnostic_only": True,
        "fitting_performed": False,
        "model_selection_performed": False,
        "scope": "Posthoc controls after all four blind benchmarks completed. Teacher counts and teacher fields are diagnostic labels only. Fixed weights; interventions are not an additive error partition. Power has no corresponding lower-bound claim.",
        "selection_sha256": selection_hash,
        "evaluation_completion_sha256": digest(root / "evaluation_completion.json"),
        "model_files_sha256": selection["model_files_sha256"],
        "diagnostic_source_sha256": digest(Path(diagnose_errors.__file__)),
        "evidence_builder_sha256": digest(Path(__file__)),
        "phase_definitions": {
            "active": "0.05 < teacher_xHI_mass < 0.95",
            "late": "teacher_xHI_mass <= 0.05, including fully ionized endpoints",
        },
        "profiles": {},
    }
    all_bitwise = all_oracle_mass = all_lower_bounds = True
    maximum_residual = 0.0
    for profile in PROFILES:
        benchmark_folder = root / f"benchmark_blind_{profile}"
        oracle_folder = root / f"oracle_blind_{profile}"
        _require(
            (benchmark_folder / "COMPLETED").is_file(),
            "Oracle verification failed: (benchmark_folder / 'COMPLETED').is_file()",
        )
        benchmark = read(benchmark_folder / "report.json")
        report = read(oracle_folder / "summary.json")
        _require(
            report["complete"] and report["diagnostic_only"],
            "Oracle verification failed: report['complete'] and report['diagnostic_only']",
        )
        _require(
            report["fitting_performed"] is False
            and report["model_selection_performed"] is False,
            "Oracle verification failed: report['fitting_performed'] is False and report['model_selection_performed'] is False",
        )
        _require(
            report["model_sha256"] == selection["model_files_sha256"],
            "Oracle verification failed: report['model_sha256'] == selection['model_files_sha256']",
        )
        _require(
            report["diagnostic_source_sha256"] == result["diagnostic_source_sha256"],
            "Oracle verification failed: report['diagnostic_source_sha256'] == result['diagnostic_source_sha256']",
        )
        reference = {(r["seed"], r["redshift"]): r for r in benchmark["snapshots"]}
        rows = report["snapshots"]
        _require(
            {(r["seed"], r["redshift"]) for r in rows} == set(reference),
            "Oracle verification failed: {(r['seed'], r['redshift']) for r in rows} == set(reference)",
        )
        bitwise_count = 0
        for row in rows:
            ref = reference[row["seed"], row["redshift"]]
            zlabel = f"z{row['redshift']:g}".replace(".", "p")
            fields_path = oracle_folder / f"seed{row['seed']}" / zlabel / "fields.npz"
            benchmark_path = benchmark_folder / ref["files"]["predictions.npz"]["path"]
            _require(
                digest(fields_path) == row["files_sha256"]["fields.npz"],
                "Oracle verification failed: digest(fields_path) == row['files_sha256']['fields.npz']",
            )
            _require(
                digest(benchmark_path) == ref["files"]["predictions.npz"]["sha256"],
                "Oracle verification failed: digest(benchmark_path) == ref['files']['predictions.npz']['sha256']",
            )
            _require(
                row["files_sha256"]["teacher_fields.npz"]
                == ref["original_teacher_fields_sha256"],
                "Oracle verification failed: row['files_sha256']['teacher_fields.npz'] == ref['original_teacher_fields_sha256']",
            )
            with (
                np.load(fields_path, allow_pickle=False) as oracle,
                np.load(benchmark_path, allow_pickle=False) as saved,
            ):
                equal = np.array_equal(oracle["deployed_xHI"], saved["selected_xHI"])
                for key in ("source", "capacity", "teacher_xHI"):
                    _require(
                        np.array_equal(oracle[key], saved[key]),
                        "Oracle verification failed: np.array_equal(oracle[key], saved[key])",
                    )
                _require(
                    np.array_equal(
                        oracle["deployed_latent_xHII"],
                        oracle["oracle_final_budget_latent_xHII"],
                    ),
                    "Oracle verification failed: np.array_equal(oracle['deployed_latent_xHII'], oracle['oracle_final_budget_latent_xHII'])",
                )
                bitwise_count += int(equal)
                all_bitwise &= equal
            for name in METHODS:
                budget = row["budgets"][name]
                residual = budget["relative_photon_residual"]
                _require(
                    0 <= residual <= 1e-10,
                    "Oracle verification failed: 0 <= residual <= 1e-10",
                )
                maximum_residual = max(maximum_residual, residual)
            for name in METHODS[1:3]:
                all_oracle_mass &= (
                    abs(row["field_metrics"][name]["delta_Q_mass"]) <= 1e-10
                )
            if row["teacher_projection_is_weighted_MSE_lower_bound"]:
                mse = row["field_metrics"]
                all_lower_bounds &= (
                    mse["teacher_predicted_budget"]["capacity_weighted_MSE"]
                    <= mse["deployed"]["capacity_weighted_MSE"] + 1e-12
                )
        result["profiles"][profile] = {
            "geometry": {
                "HII_DIM": rows[0]["dimension"],
                "BOX_LEN": rows[0]["box_len"],
            },
            "snapshot_count": len(rows),
            "bitwise_deployed_field_matches": bitwise_count,
            "all_teacher_projection_controls_are_weighted_MSE_lower_bounds": all(
                row["teacher_projection_is_weighted_MSE_lower_bound"] for row in rows
            ),
            "summaries": {
                "all": grouped(rows),
                "active": grouped(
                    [r for r in rows if 0.05 < r["teacher_xHI_mass"] < 0.95]
                ),
                "late": grouped([r for r in rows if r["teacher_xHI_mass"] <= 0.05]),
            },
            "late_snapshots": [
                {
                    "seed": row["seed"],
                    "redshift": row["redshift"],
                    "teacher_xHI_mass": row["teacher_xHI_mass"],
                    "methods": {
                        method: {
                            key: row["field_metrics"][method][key]
                            for key in ("MAE", "capacity_weighted_MSE", "delta_Q_mass")
                        }
                        for method in METHODS
                    },
                }
                for row in rows
                if row["teacher_xHI_mass"] <= 0.05
            ],
            "oracle_summary_sha256": digest(oracle_folder / "summary.json"),
            "benchmark_report_sha256": digest(benchmark_folder / "report.json"),
        }
    _require(
        all_bitwise and all_oracle_mass and all_lower_bounds,
        "Oracle verification failed: all_bitwise and all_oracle_mass and all_lower_bounds",
    )
    _require(
        _model_hashes(model) == selection["model_files_sha256"],
        "Oracle verification failed: _model_hashes(model) == selection['model_files_sha256']",
    )
    _require(
        _inference_source_hashes() == selection["inference_source_sha256"],
        "Oracle verification failed: _inference_source_hashes() == selection['inference_source_sha256']",
    )
    _require(
        digest(root / "selection.json") == selection_hash,
        "Oracle verification failed: digest(root / 'selection.json') == selection_hash",
    )
    result["verification"] = {
        "all_deployed_fields_bitwise_match_frozen_benchmarks": bool(all_bitwise),
        "all_oracle_mass_biases_at_most_1e-10": bool(all_oracle_mass),
        "all_applicable_teacher_projection_MSE_lower_bounds_satisfied": bool(
            all_lower_bounds
        ),
        "maximum_diagnostic_photon_residual": maximum_residual,
        "frozen_model_and_inference_source_unchanged": True,
    }
    if original_evidence is not None:
        original = read(original_evidence)
        _require(
            _scientific_payload(result) == _scientific_payload(original),
            "Portable oracle scientific payload differs from original evidence",
        )
        result["original_evidence"] = {
            "filename": original_evidence.name,
            "evidence_sha256": digest(original_evidence),
            "builder_sha256": original["evidence_builder_sha256"],
            "scientific_payload_identical": True,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return result


def main() -> None:
    """Verify existing oracle diagnostics without recomputing predictions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--original-evidence", type=Path)
    arguments = parser.parse_args()
    run(arguments.root, arguments.output, original_evidence=arguments.original_evidence)


if __name__ == "__main__":
    main()
