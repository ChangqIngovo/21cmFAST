"""Collect partial suites without accepting corrupt completed simulations."""

import csv
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from research.c2ray_teacher.build import FIELD_NAMES, META_NAMES
from research.c2ray_teacher.collect import collect_suite
from research.c2ray_teacher.test_runner import transport_fixture


def completed_case(directory: Path, *, times: tuple = (1.0,)) -> Path:
    """Write a real native-format payload with independently known field means."""
    result = directory / "results"
    (result / "science").mkdir(parents=True)
    n = 8
    binary = result / "science/native_output.bin"
    steps = []
    previous = 0.0
    with binary.open("wb") as stream:
        stream.write(b"C2RTOUT1")
        stream.write(
            struct.pack("<4i", n, len(times), len(FIELD_NAMES), len(META_NAMES))
        )
        for time in times:
            dt = time - previous
            step = {
                "time_end_seconds": time,
                "redshift_end": 10 - time,
                "dt_seconds": dt,
                "emitted_photons": dt * 100,
            }
            steps.append(step)
            meta = dict.fromkeys(META_NAMES, 0.0)
            meta.update(
                time_seconds=time,
                dt_seconds=dt,
                emitted_photons=dt * 100,
                proper_box_cm=8.0,
                redshift_mid=10.0,
            )
            np.asarray([meta[name] for name in META_NAMES], dtype="<f8").tofile(stream)
            for name in FIELD_NAMES:
                value = (
                    1.0
                    if name == "density_total_cm3"
                    else 100.0
                    if name == "temperature_K"
                    else 0.2
                    if name == "xHII"
                    else 0.0
                )
                np.full((n,) * 3, value, dtype="<f8").tofile(stream)
            previous = time
    manifest = {
        "dimension": n,
        "native_dimension": n,
        "refinement": 1,
        "step_count": len(times),
        "box_len_cMpc": 50.0,
        "native_array_sha256": dict.fromkeys(
            ("rates", "capacity", "redshifts", "times_seconds"), "same-history"
        ),
        "cosmology": {"h": 0.7},
        "initial_state": {"xHII": 1e-6},
        "thermal_evolution": True,
        "steps": steps,
    }
    summary = {
        "input_manifest": manifest,
        "native_output_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "validation": {
            "state_checks_passed": True,
            "primary_photon_absorption_count_available": False,
        },
        "timing": {"seconds": 1.0, "sed_table_photons_over_nominal": 1.01},
    }
    (result / "summary.json").write_text(json.dumps(summary))
    (result / "COMPLETED").touch()
    (directory / "COMPLETED").touch()
    return directory


def suite_file(path: Path, cases: list[dict]) -> Path:
    """Write the exact suite manifest schema used by the Gadi preparer."""
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment": "c2ray-accuracy-volume-suite",
                "cases": cases,
            }
        )
    )
    return path


def test_collects_partial_suite_with_verified_pair_and_numeric_history(
    tmp_path: Path,
) -> None:
    """Completed pairs remain useful while failed and waiting cases stay explicit."""
    a = completed_case(tmp_path / "time1")
    b = completed_case(tmp_path / "time2", times=(0.5, 1.0))
    failed = tmp_path / "time4"
    failed.mkdir()
    (failed / "FAILED").write_text("2\n")
    manifest = suite_file(
        tmp_path / "suite_manifest.json",
        [
            {"name": "time1", "run_directory": str(a)},
            {"name": "time2", "run_directory": str(b)},
            {"name": "time4", "run_directory": str(failed)},
            {"name": "space48_dt2", "run_directory": None},
        ],
    )
    output = tmp_path / "analysis"
    report = collect_suite(manifest, output, diagnostic_thresholds=True)
    assert report["status_counts"] == {"complete": 2, "failed": 1, "pending": 1}
    assert report["all_cases_complete"] is False
    assert report["comparisons"][0]["status"] == "complete"
    comparison = json.loads((output / "time1_vs_time2.json").read_text())
    assert len(comparison["snapshots"]) == 1
    assert comparison["snapshots"][0]["paired_errors"]["xHII"]["rms"] == 0
    with (output / "history.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    assert float(rows[0]["xHI_volume"]) == pytest.approx(0.8)
    assert rows[0]["primary_absorptions"] == ""


def test_corrupt_completed_payload_is_fatal_before_writing_report(
    tmp_path: Path,
) -> None:
    """A COMPLETED marker does not override a changed native binary."""
    case = completed_case(tmp_path / "case")
    binary = case / "results/science/native_output.bin"
    with binary.open("ab") as stream:
        stream.write(b"corrupt")
    manifest = suite_file(
        tmp_path / "suite_manifest.json",
        [{"name": "time1", "run_directory": str(case)}],
    )
    with pytest.raises(ValueError, match="hash changed"):
        collect_suite(manifest, tmp_path / "analysis")
    assert not (tmp_path / "analysis").exists()


@pytest.mark.parametrize("active_ratio", [1.0, None])
def test_collector_preserves_optional_transported_band_budget(
    tmp_path: Path, active_ratio: float | None
) -> None:
    """A new normalization reaches CSV; old telemetry remains blank rather than inferred."""
    case = completed_case(tmp_path / "case")
    summary_path = case / "results/summary.json"
    summary = json.loads(summary_path.read_text())
    if active_ratio is not None:
        summary["timing"]["sed_transport_photons_over_nominal"] = active_ratio
    summary_path.write_text(json.dumps(summary))
    sidecar = case / "results/science/native_diagnostics/transport_diagnostics.jsonl"
    sidecar.parent.mkdir()
    record = transport_fixture()
    record["native_boundary_outflow_photons"] = 0.0
    sidecar.write_text(json.dumps(record) + "\n")
    suite = suite_file(
        tmp_path / "suite_manifest.json",
        [{"name": "time1", "run_directory": str(case)}],
    )
    output = tmp_path / "analysis"
    report = collect_suite(suite, output)
    ledger = report["cases"][0]["cumulative_photon_diagnostics"]
    with (output / "history.csv").open() as stream:
        row = next(csv.DictReader(stream))
    if active_ratio is None:
        assert row["transported_emitted_photons"] == ""
        assert ledger["transported_emitted_photons"] is None
    else:
        assert float(row["transported_emitted_photons"]) == 100.0
        assert float(row["photon_residual_over_transported_emitted"]) == pytest.approx(
            0.01
        )
        assert ledger["photon_residual_over_transported_emitted"] == pytest.approx(0.01)
    assert float(row["table_emitted_photons"]) == 101.0


def test_duplicate_case_names_and_existing_output_are_rejected(tmp_path: Path) -> None:
    """Collection cannot overwrite analysis or hide an ambiguous case identity."""
    manifest = suite_file(
        tmp_path / "suite_manifest.json", [{"name": "time1", "run_directory": None}] * 2
    )
    with pytest.raises(ValueError, match="unique"):
        collect_suite(manifest, tmp_path / "analysis")
    suite_file(manifest, [{"name": "time1", "run_directory": None}])
    (tmp_path / "analysis").mkdir()
    with pytest.raises(FileExistsError):
        collect_suite(manifest, tmp_path / "analysis")
