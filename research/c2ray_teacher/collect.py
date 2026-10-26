"""Collect completed accuracy cases while reporting failed and waiting jobs.

The collector reads immutable completed outputs, verifies their native payloads,
and compares only completed requested pairs. It never submits or resumes jobs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from importlib import import_module
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_compare = import_module("research.c2ray_teacher.compare")
_backend = import_module("research.c2ray_teacher.build")
_runner = import_module("research.c2ray_teacher.runner")

PAIRS = (
    ("time1", "time2"),
    ("time2", "time4"),
    ("time2", "space48_dt2"),
    ("bridge_L50_N16_DIM128", "box_L75_N24_DIM192"),
    ("box_L75_N24_DIM192", "box_L100_N32_DIM256"),
    ("box_L75_N24_DIM192", "finer_L75_N32_DIM192"),
)
PHOTON_COLUMNS = (
    "table_emitted_photons",
    "transported_emitted_photons",
    "primary_absorptions",
    "secondary_ionizations",
    "boundary_outflow_photons",
    "native_boundary_outflow_photons",
    "lls_loss_photons",
    "photon_residual_over_nominal_emitted",
    "photon_residual_over_table_emitted",
    "photon_residual_over_transported_emitted",
    "postchemistry_rate_product_difference",
    "transport_fraction_max_abs_change",
    "transport_fraction_max_relative_change",
    "final_transport_iteration",
)
CSV_COLUMNS = (
    "case",
    "step",
    "redshift",
    "time_seconds",
    "dimension",
    "box_len_cMpc",
    "cell_width_cMpc",
    "xHI_volume",
    "xHI_mass",
    "temperature_volume_mean_K",
    "temperature_nuclei_weighted_K",
    "nominal_emitted_photons",
    "independent_counters_available",
    *PHOTON_COLUMNS,
)


def _manifest(path: Path) -> tuple[dict, str]:
    payload = path.read_bytes()
    manifest = json.loads(payload)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("experiment") != "c2ray-accuracy-volume-suite"
        or not isinstance(manifest.get("cases"), list)
        or not manifest["cases"]
    ):
        raise ValueError("Unsupported or empty accuracy suite manifest")
    names = []
    for case in manifest["cases"]:
        name = case.get("name") if isinstance(case, dict) else None
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("Suite case names must be simple unique identifiers")
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError("Suite case names must be unique")
    return manifest, hashlib.sha256(payload).hexdigest()


def _status(case: dict, suite: Path) -> dict:
    result = {
        "name": case["name"],
        "job_id": case.get("job_id"),
        "family": case.get("family"),
    }
    location = case.get("run_directory")
    if location is None:
        return result | {
            "status": "pending",
            "reason": "No prepared run directory recorded",
            "run_directory": None,
        }
    if not isinstance(location, str) or not location:
        raise ValueError("Suite run_directory must be a nonempty path or null")
    directory = Path(location)
    directory = (
        (suite.parent / directory).resolve()
        if not directory.is_absolute()
        else directory.resolve()
    )
    result["run_directory"] = str(directory)
    completed, failed = (
        (directory / "COMPLETED").is_file(),
        (directory / "FAILED").is_file(),
    )
    if completed and failed:
        raise ValueError(f"Contradictory completed and failed markers: {directory}")
    if failed:
        return result | {"status": "failed", "reason": "Run has a FAILED marker"}
    if completed:
        if not (directory / "results/COMPLETED").is_file():
            raise ValueError(f"Completed run lacks completed results: {directory}")
        return result | {"status": "complete"}
    reason = (
        "Running marker present"
        if (directory / ".running").exists()
        else "No completion marker; may be queued, dependency-blocked or not yet submitted"
    )
    return result | {"status": "pending", "reason": reason}


def _load_completed(status: dict) -> tuple[dict, dict, list[dict]]:
    directory = Path(status["run_directory"]) / "results"
    decoded = _compare.load_run(directory)
    summary = json.loads((directory / "summary.json").read_text())
    if summary.get("validation", {}).get("state_checks_passed") is not True:
        raise ValueError(
            f"Completed run does not report passed state checks: {directory}"
        )
    native = _backend.read_output(directory / "science/native_output.bin")
    sidecar = directory / "science/native_diagnostics/transport_diagnostics.jsonl"
    build_path = directory / "build_science/build_manifest.json"
    enabled = (
        json.loads(build_path.read_text()).get("diagnostics_enabled", False)
        if build_path.exists()
        else False
    )
    advertised = summary["validation"].get(
        "primary_photon_absorption_count_available", False
    )
    transport = None
    if sidecar.exists() or enabled or advertised:
        transport = _runner.read_transport_diagnostics(
            sidecar,
            native,
            summary["timing"]["sed_table_photons_over_nominal"],
            active_spectral_ratio=summary["timing"].get(
                "sed_transport_photons_over_nominal"
            ),
        )
    manifest = decoded["manifest"]
    rows = []
    for index, (step, output_step) in enumerate(
        zip(manifest["steps"], native["steps"], strict=True)
    ):
        if not np.isclose(
            step["time_end_seconds"], output_step["time_seconds"], rtol=2e-12, atol=1e-8
        ):
            raise ValueError("Completed output clock differs from input manifest")
        rho, ionized, temperature = (
            np.asarray(decoded["fields"][name][index]) for name in _compare.FIELDS
        )
        if (
            not all(np.isfinite(field).all() for field in (rho, ionized, temperature))
            or min(rho.min(), temperature.min()) <= 0
            or ionized.min() < 0
            or ionized.max() > 1
        ):
            raise ValueError("Completed output contains invalid gas state")
        row = {
            "case": status["name"],
            "step": index + 1,
            "redshift": step["redshift_end"],
            "time_seconds": output_step["time_seconds"],
            "dimension": manifest["dimension"],
            "box_len_cMpc": manifest["box_len_cMpc"],
            "cell_width_cMpc": manifest["box_len_cMpc"] / manifest["dimension"],
            "xHI_volume": float(np.mean(1 - ionized)),
            "xHI_mass": float(np.sum(rho * (1 - ionized)) / rho.sum()),
            "temperature_volume_mean_K": float(temperature.mean()),
            "temperature_nuclei_weighted_K": float(
                np.sum(rho * temperature) / rho.sum()
            ),
            "nominal_emitted_photons": output_step["emitted_photons"],
            "independent_counters_available": transport is not None,
        }
        if transport is not None:
            row.update({key: transport["steps"][index][key] for key in PHOTON_COLUMNS})
        rows.append(row)
    status.update(
        native_output_sha256=summary["native_output_sha256"],
        steps=len(rows),
        dimension=manifest["dimension"],
        box_len_cMpc=manifest["box_len_cMpc"],
        rt_seconds=summary.get("timing", {}).get("seconds"),
        independent_counters_available=transport is not None,
        cumulative_photon_diagnostics=transport["cumulative"] if transport else None,
        native_history_provenance=manifest.get("native_history_provenance"),
    )
    return decoded, status, rows


def collect_suite(
    suite: Path, output: Path, *, diagnostic_thresholds: bool = False
) -> dict:
    """Collect a fixed suite-status snapshot into a fresh analysis directory.

    Parameters
    ----------
    suite : Path
        ``suite_manifest.json`` created by ``gadi_accuracy_prepare.sh``.
    output : Path
        Fresh directory for history.csv, summary.json and completed pair JSONs.
    diagnostic_thresholds : bool
        Request the comparison module's exploratory difference flags; these
        never certify convergence or turn missing cases into passing results.
    """
    suite, output = Path(suite).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest, digest = _manifest(suite)
    statuses, loaded, history = {}, {}, []
    for case in manifest["cases"]:
        status = _status(case, suite)
        if status["status"] == "complete":
            state, status, rows = _load_completed(status)
            loaded[case["name"]] = state
            history.extend(rows)
        statuses[case["name"]] = status
    comparisons, reports = [], {}
    for reference, candidate in PAIRS:
        if reference not in statuses or candidate not in statuses:
            continue
        item = {"reference": reference, "candidate": candidate}
        if reference in loaded and candidate in loaded:
            report = _compare.compare_states(
                loaded[reference],
                loaded[candidate],
                diagnostic_thresholds=diagnostic_thresholds,
            )
            name = f"{reference}_vs_{candidate}.json"
            reports[name] = report
            item.update(
                status="complete", file=name, comparison_kind=report["comparison_kind"]
            )
        else:
            item.update(
                status="unavailable",
                missing_complete_cases=[
                    name for name in (reference, candidate) if name not in loaded
                ],
            )
        comparisons.append(item)
    counts = {
        name: sum(case["status"] == name for case in statuses.values())
        for name in ("complete", "failed", "pending")
    }
    result = {
        "schema_version": 1,
        "suite_manifest": str(suite),
        "suite_manifest_sha256": digest,
        "status_counts": counts,
        "all_cases_complete": counts["complete"] == len(statuses),
        "convergence_certified": False,
        "history_file": "history.csv",
        "cases": list(statuses.values()),
        "comparisons": comparisons,
        "interpretation": "Pending includes running, queued, blocked and unsubmitted cases; job state is inferred only from durable files, without qstat. Corrupt completed outputs fail collection.",
    }
    output.mkdir(parents=True, exist_ok=False)
    with (output / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(history)
    for name, report in reports.items():
        (output / name).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


def main() -> None:
    """Collect finished suite cases without changing simulations or PBS jobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic-thresholds", action="store_true")
    args = parser.parse_args()
    report = collect_suite(
        args.suite, args.output, diagnostic_thresholds=args.diagnostic_thresholds
    )
    sys.stdout.write(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "status_counts": report["status_counts"],
                "all_cases_complete": report["all_cases_complete"],
            }
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
