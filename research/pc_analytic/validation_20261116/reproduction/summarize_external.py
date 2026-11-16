"""Produce a compact endpoint and geometric morphology summary from audit JSON."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from recompute import digest


def main() -> None:
    """Separate evolution snapshots and late endpoints when summarizing errors."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.input.read_text())
    result = {
        "complete": audit["complete"],
        "scope": audit["scope"],
        "external_seeds": audit["external_seeds"],
        "seed_split": audit["seed_split"],
        "aggregates": audit["aggregates"],
        "by_seed": {key: value["all"] for key, value in audit["by_seed"].items()},
        "by_redshift": {
            key: value["all"] for key, value in audit["by_redshift"].items()
        },
        "maximum_photon_residual": audit["maximum_photon_residual"],
        "MFP_settings": audit["MFP_settings"],
        "scripts_sha256": audit["scripts_sha256"],
        "MFP": {},
        "late_endpoints": [],
        "snapshots": [],
    }
    for stage, chosen in (
        (
            "active",
            [row for row in audit["rows"] if 0.05 < row["teacher_xHI_mass"] < 0.95],
        ),
        ("late", [row for row in audit["rows"] if row["teacher_xHI_mass"] <= 0.05]),
    ):
        result["MFP"][stage] = {}
        for phase in ("ionized", "neutral"):
            phase_summary = {}
            for name in ("selected", "legacy", "native_es"):
                rows = [row["MFP"][phase]["comparisons"][name] for row in chosen]
                valid = [row for row in rows if row["available"]]
                ratios = [abs(row["restricted_mean_ratio"] - 1) for row in valid]
                cdf = [row["restricted_CDF_maximum_difference"] for row in valid]
                phase_summary[name] = {
                    "defined_snapshots": len(valid),
                    "undefined_snapshots": len(rows) - len(valid),
                    "mean_absolute_restricted_mean_ratio_error": float(np.mean(ratios))
                    if ratios
                    else None,
                    "maximum_absolute_restricted_mean_ratio_error": max(
                        ratios, default=None
                    ),
                    "mean_restricted_CDF_maximum_difference": float(np.mean(cdf))
                    if cdf
                    else None,
                    "maximum_restricted_CDF_difference": max(cdf, default=None),
                }
            result["MFP"][stage][phase] = phase_summary
    for row in audit["rows"]:
        snapshot = {
            "seed": row["seed"],
            "redshift": row["redshift"],
            "teacher_xHI_mass": row["teacher_xHI_mass"],
            "field_metrics": {
                name: {
                    key: value[key]
                    for key in ("MAE", "RMSE", "delta_Q_mass", "xHI_mass")
                }
                for name, value in row["field_metrics"].items()
                if name != "teacher"
            },
            "budget": {
                name: {
                    key: value[key]
                    for key in (
                        "relative_residual",
                        "predicted_efficiency",
                        "teacher_efficiency",
                    )
                    if key in value
                }
                for name, value in row["budget"].items()
            },
            "low_k_21cm": {
                name: {
                    key: value[key]
                    for key in (
                        "mean_absolute_power_ratio_error",
                        "maximum_absolute_power_ratio_error",
                        "minimum_cross_correlation",
                    )
                }
                for name, value in row["low_k_21cm"]["comparisons"].items()
            },
            "MFP": {},
        }
        for phase, measured in row["MFP"].items():
            snapshot["MFP"][phase] = {
                "fields": {
                    name: {
                        key: value[key]
                        for key in ("restricted_mean_cMpc", "cap_fraction")
                    }
                    for name, value in measured["fields"].items()
                },
                "CDF_maximum_difference": {
                    name: value.get("restricted_CDF_maximum_difference")
                    for name, value in measured["comparisons"].items()
                },
            }
        result["snapshots"].append(snapshot)
        if row["teacher_xHI_mass"] > 0.05:
            continue
        result["late_endpoints"].append(
            {
                "seed": row["seed"],
                "redshift": row["redshift"],
                "teacher_xHI_mass": row["teacher_xHI_mass"],
                "selected_xHI_mass": row["field_metrics"]["selected"]["xHI_mass"],
                "selected_xHI_volume": row["field_metrics"]["selected"]["xHI_volume"],
                "selected_low_k_mean_power_ratio_error": row["low_k_21cm"][
                    "comparisons"
                ]["selected"]["mean_absolute_power_ratio_error"],
                "teacher_ionized_cap_fraction": row["MFP"]["ionized"]["fields"][
                    "teacher"
                ]["cap_fraction"],
                "selected_ionized_cap_fraction": row["MFP"]["ionized"]["fields"][
                    "selected"
                ]["cap_fraction"],
                "teacher_neutral_MFP_cMpc": row["MFP"]["neutral"]["fields"]["teacher"][
                    "restricted_mean_cMpc"
                ],
                "selected_neutral_MFP_cMpc": row["MFP"]["neutral"]["fields"][
                    "selected"
                ]["restricted_mean_cMpc"],
            }
        )
    result["source_sha256"] = digest(args.input)
    result["script_sha256"] = digest(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n"
    )
    logging.getLogger(__name__).info(
        json.dumps({key: result[key] for key in ("MFP", "late_endpoints")}, indent=2)
    )


if __name__ == "__main__":
    main()
