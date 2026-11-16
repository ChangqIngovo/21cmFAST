"""Summarize completed frozen-model benchmarks without loading teacher arrays.

The four benchmark receipts, pre-evaluation gate, selected model, reference model
and inference implementation must agree with the validation-only selection.
Statistics are rebuilt from per-snapshot records rather than copied from their
precomputed aggregate sections. Undefined endpoint ratios remain explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from research.pc_analytic_emulator import benchmark
from research.pc_analytic_emulator.benchmark import (
    METHODS,
    _inference_source_hashes,
    _model_hashes,
)
from research.pc_analytic_emulator.completion_20261116.reproduction.train_and_validate import (
    _receipt,
)
from research.pc_analytic_emulator.refinement_20261117.reproduction import (
    evaluate_completed,
    freeze_completed,
)

PROFILES = {
    "original": (16, 50.0),
    "resolution": (32, 50.0),
    "volume": (32, 256.0),
    "unseen_box": (32, 100.0),
}
PHASES = ("ionized", "neutral")


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError("Control and result files must contain JSON objects")
    return value


def _number(value: float | None, *, optional: bool = False) -> float | None:
    if optional and value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(
            "Scientific statistics must be finite numbers or explicit null"
        )
    return float(value)


def _summary(values: list[float | None]) -> dict:
    checked = [_number(value, optional=True) for value in values]
    defined = [value for value in checked if value is not None]
    return {
        "defined_count": len(defined),
        "undefined_count": len(checked) - len(defined),
        "mean": math.fsum(defined) / len(defined) if defined else None,
        "maximum": max(defined, default=None),
    }


def _spectra(rows: list[dict], method: str) -> dict:
    means, maxima, correlations = [], [], []
    shell_count = defined_count = cross_count = 0
    for row in rows:
        record = row["low_k_21cm"]["comparisons"][method]
        ratios = [_number(x, optional=True) for x in record["power_ratio"]]
        cross = [_number(x, optional=True) for x in record["cross_correlation"]]
        if len(ratios) != len(cross):
            raise ValueError("Power and cross-correlation shell counts differ")
        errors = [abs(value - 1) for value in ratios if value is not None]
        valid_cross = [value for value in cross if value is not None]
        if record["defined_shells"] != len(errors):
            raise ValueError("Declared defined power-shell count is inconsistent")
        means.append(math.fsum(errors) / len(errors) if errors else None)
        maxima.append(max(errors, default=None))
        correlations.extend(valid_cross)
        shell_count += len(ratios)
        defined_count += len(errors)
        cross_count += len(valid_cross)
    return {
        "snapshot_mean_absolute_power_ratio_error": _summary(means),
        "snapshot_maximum_absolute_power_ratio_error": _summary(maxima),
        "minimum_defined_cross_correlation": min(correlations, default=None),
        "total_shells": shell_count,
        "defined_shells": defined_count,
        "undefined_shells": shell_count - defined_count,
        "defined_cross_correlation_shells": cross_count,
        "undefined_cross_correlation_shells": shell_count - cross_count,
    }


def _mfp(rows: list[dict], method: str, phase: str) -> dict:
    comparisons = [row["mfp"][phase]["comparisons"][method] for row in rows]
    measurements = [row["mfp"][phase]["measurements"] for row in rows]
    if any(type(record.get("available")) is not bool for record in comparisons):
        raise ValueError("MFP availability must be explicit")
    result = {"available_snapshots": sum(c["available"] for c in comparisons)}
    for output, source, absolute in (
        (
            "restricted_mean_absolute_difference_cMpc",
            "restricted_mean_difference_cMpc",
            True,
        ),
        (
            "restricted_CDF_maximum_difference",
            "restricted_CDF_maximum_difference",
            False,
        ),
        ("absolute_cap_fraction_difference", "cap_fraction_difference", True),
    ):
        values = [
            _number(record[source]) if record["available"] else None
            for record in comparisons
        ]
        result[output] = _summary(
            [
                abs(value) if value is not None and absolute else value
                for value in values
            ]
        )
    for label, name in (("predicted", method), ("teacher", "teacher")):
        result[f"{label}_cap_fraction"] = _summary(
            [measurement[name]["cap_fraction"] for measurement in measurements]
        )
        result[f"{label}_restricted_mean_cMpc"] = _summary(
            [measurement[name]["restricted_mean_cMpc"] for measurement in measurements]
        )
    return result


def summarize_group(rows: list[dict]) -> dict:
    """Aggregate one snapshot group with equal weights and explicit undefined counts.

    Parameters
    ----------
    rows
        Per-snapshot records written by the independent benchmark runner.

    Returns
    -------
    dict
        Field, low-k power, geometric MFP and photon-ledger statistics for the
        selected emulator, previous emulator and native ES reference.
    """
    result = {
        "count": len(rows),
        "fully_ionized_count": sum(row["teacher_xHI_mass"] <= 1e-10 for row in rows),
        "methods": {},
    }
    for method in METHODS:
        fields = [row["field_metrics"][method] for row in rows]
        metrics = {
            key: _summary([field[key] for field in fields]) for key in ("MAE", "RMSE")
        }
        metrics["absolute_delta_Q_mass"] = _summary(
            [abs(_number(field["delta_Q_mass"])) for field in fields]
        )
        metrics["low_k_21cm"] = _spectra(rows, method)
        metrics["mfp"] = {phase: _mfp(rows, method, phase) for phase in PHASES}
        result["methods"][method] = metrics
    result["photon_ledgers"] = {}
    for method in ("selected", "previous", "teacher"):
        ledgers = [row["photon_ledgers"][method] for row in rows]
        if any(ledger.get("passed") is not True for ledger in ledgers):
            raise ValueError("Independent saved-field photon ledger failed")
        residuals = [_number(item["relative_photon_residual"]) for item in ledgers]
        if any(value < 0 for value in residuals):
            raise ValueError("Relative photon residual cannot be negative")
        result["photon_ledgers"][method] = {
            "count": len(ledgers),
            "all_passed": all(item["passed"] for item in ledgers),
            "maximum_relative_photon_residual": max(residuals, default=None),
        }
    return result


def verify_selection(root: Path) -> dict:
    """Verify the immutable validation-only selection before reading test reports.

    Parameters
    ----------
    root
        Campaign directory containing the existing ``selection.json``.

    Returns
    -------
    dict
        Verified selection record, including all candidate and source hashes.
    """
    return freeze_completed.verify(root / "selection.json")


def _evaluation_code_hashes() -> dict[str, str]:
    return {
        "evaluator_sha256": _digest(Path(evaluate_completed.__file__)),
        "benchmark_sha256": _digest(Path(benchmark.__file__)),
        "independent_metrics_sha256": _digest(Path(benchmark.recompute.__file__)),
    }


def _verify_benchmark(
    folder: Path, selection: dict, profile: str, protocol: dict, inputs: dict
) -> tuple[dict, dict]:
    receipt = _receipt(folder, "receipt.json", "files_sha256")
    required = {"report.json", "manifest.json", "inputs.json"}
    if receipt.get("complete") is not True or not required <= set(
        receipt["files_sha256"]
    ):
        raise ValueError(
            "Complete benchmark receipt must bind report, manifest and inputs"
        )
    report, manifest = _json(folder / "report.json"), _json(folder / "manifest.json")
    expected_models = {
        "selected": selection["model_files_sha256"],
        "previous": selection["candidates"]["previous_model"]["model_files_sha256"],
    }
    spec = protocol["evaluations"][f"blind_{profile}"]
    if (
        any(
            report.get(key) is not True
            for key in (
                "complete",
                "model_weights_unchanged",
                "inference_implementation_unchanged",
            )
        )
        or report.get("fitting_performed") is not False
        or report.get("model_selection_performed") is not False
        or report.get("model_files_after_sha256") != expected_models
        or report.get("frozen_manifest_sha256") != _digest(folder / "manifest.json")
        or report.get("input_record_sha256") != _digest(folder / "inputs.json")
        or manifest.get("inference_source_sha256")
        != selection["inference_source_sha256"]
        or manifest.get("fitting_performed") is not False
        or manifest.get("selection_performed") is not False
        or manifest.get("rays_per_phase") != 4096
        or manifest.get("benchmark_sha256") != protocol["benchmark_sha256"]
        or manifest.get("independent_metrics_sha256")
        != protocol["independent_metrics_sha256"]
        or manifest.get("teacher_root") != str(Path(spec["root"]).resolve())
        or manifest.get("requested_seeds") != spec["seeds"]
        or _json(folder / "inputs.json")["files_sha256"]
        != inputs[f"blind_{profile}"]["files_sha256"]
        or {
            name: item["files_sha256"]
            for name, item in manifest.get("models", {}).items()
        }
        != expected_models
    ):
        raise ValueError(
            "Benchmark differs from the pre-blind models, source or settings"
        )
    rows = report["snapshots"]
    dimension, length = PROFILES[profile]
    if (
        not rows
        or spec["dimension"] != dimension
        or spec["box_len"] != length
        or report.get("snapshot_count") != len(rows)
        or len({(row["seed"], row["redshift"]) for row in rows}) != len(rows)
        or any(
            row["dimension"] != dimension or row["box_len_cMpc"] != length
            for row in rows
        )
    ):
        raise ValueError(
            "Benchmark snapshot count, identity or profile geometry is invalid"
        )
    for row in rows:
        neutral = _number(row["teacher_xHI_mass"])
        if not 0 <= neutral <= 1:
            raise ValueError("Teacher neutral fraction must lie between zero and one")
        for entry in row["files"].values():
            if receipt["files_sha256"].get(entry["path"]) != entry["sha256"]:
                raise ValueError(
                    "Snapshot artifacts are not bound by the benchmark receipt"
                )
    requested = manifest.get("requested_seeds")
    if requested is not None and set(requested) != {row["seed"] for row in rows}:
        raise ValueError("Benchmark seeds differ from its frozen request")
    provenance = {
        "report_sha256": _digest(folder / "report.json"),
        "receipt_sha256": _digest(folder / "receipt.json"),
        "manifest_sha256": _digest(folder / "manifest.json"),
        "input_record_sha256": _digest(folder / "inputs.json"),
        "benchmark_sha256": manifest["benchmark_sha256"],
        "independent_metrics_sha256": manifest["independent_metrics_sha256"],
        "rays_per_phase": manifest["rays_per_phase"],
    }
    return report, provenance


def _timings(rows: list[dict]) -> dict:
    return {
        "inference_seconds_excluding_io": {
            method: _summary(
                [row["inference_seconds_excluding_io"][method] for row in rows]
            )
            for method in ("selected", "previous")
        },
        "mfp_seconds_excluding_io": _summary(
            [row["mfp_seconds_excluding_io"] for row in rows]
        ),
    }


def run(root: Path, output: Path | None = None) -> dict:
    """Write a public, path-free summary after all four benchmarks have completed.

    Parameters
    ----------
    root
        Campaign directory containing selection, pre-blind gate and benchmarks.
    output
        Optional JSON destination; defaults to ``root/scientific_summary.json``.

    Returns
    -------
    dict
        Compact verified selection, geometry-specific statistics and provenance.
        No fitting, selection, array loading or physical simulation is performed.
    """
    root = Path(root).resolve()
    output = (
        Path(output).resolve()
        if output is not None
        else root / "scientific_summary.json"
    )
    selection = verify_selection(root)
    selection_hash = _digest(root / "selection.json")
    gate_path = root / "evaluation_started.json"
    gate = _json(gate_path)
    if gate != {
        "training_protocol_sha256": selection["training_protocol_sha256"],
        "selection_sha256": selection_hash,
        "model_files_sha256": selection["model_files_sha256"],
    }:
        raise ValueError(
            "Pre-blind evaluation gate does not match the frozen selection"
        )
    if _inference_source_hashes() != selection["inference_source_sha256"]:
        raise ValueError("Inference source changed after the pre-blind freeze")
    protocol_path, inputs_path = (
        root / "evaluation_protocol.json",
        root / "evaluation_inputs.json",
    )
    protocol, inputs = _json(protocol_path), _json(inputs_path)
    protocol_hash, inputs_hash = _digest(protocol_path), _digest(inputs_path)
    if (
        protocol.get("schema") != 1
        or protocol.get("selection_sha256") != selection_hash
        or protocol.get("training_protocol_sha256")
        != selection["training_protocol_sha256"]
        or protocol.get("inference_source_sha256")
        != selection["inference_source_sha256"]
        or protocol.get("rays_per_phase") != 4096
        or set(protocol.get("evaluations", {})) != {f"blind_{p}" for p in PROFILES}
        or set(inputs) != {f"blind_{p}" for p in PROFILES}
        or any(
            protocol.get(key) != value
            for key, value in _evaluation_code_hashes().items()
        )
        or {
            name: entry["files_sha256"]
            for name, entry in protocol.get("models", {}).items()
        }
        != {
            "selected": selection["model_files_sha256"],
            "previous": selection["candidates"]["previous_model"]["model_files_sha256"],
        }
    ):
        raise ValueError("Frozen evaluation protocol or metric implementation changed")
    selected_root = root / selection["selected_model_root"]
    previous_root = root / selection["candidates"]["previous_model"]["model_root"]
    protected = [selected_root.resolve(), previous_root.resolve()]
    protected.extend(
        (root / f"benchmark_blind_{profile}").resolve() for profile in PROFILES
    )
    controls = {
        root / "selection.json",
        gate_path,
        protocol_path,
        inputs_path,
        root / selection["training_protocol"],
    }
    if output in controls or any(output.is_relative_to(path) for path in protected):
        raise ValueError(
            "Summary output must not replace frozen inputs or benchmark artifacts"
        )
    profiles, all_rows, before = {}, [], {}
    for profile, (dimension, length) in PROFILES.items():
        folder = root / f"benchmark_blind_{profile}"
        report, provenance = _verify_benchmark(
            folder, selection, profile, protocol, inputs
        )
        before[folder / "report.json"] = provenance["report_sha256"]
        before[folder / "receipt.json"] = provenance["receipt_sha256"]
        rows = report["snapshots"]
        all_rows.extend(rows)
        profiles[profile] = {
            "geometry": {"HII_DIM": dimension, "BOX_LEN": length},
            "seeds": sorted({row["seed"] for row in rows}),
            "redshifts": sorted({row["redshift"] for row in rows}, reverse=True),
            "summaries": {
                "all": summarize_group(rows),
                "active": summarize_group(
                    [r for r in rows if 0.05 < r["teacher_xHI_mass"] < 0.95]
                ),
                "late": summarize_group(
                    [r for r in rows if r["teacher_xHI_mass"] <= 0.05]
                ),
            },
            "timings": _timings(rows),
            "provenance": provenance,
        }
    photon_residual = max(
        ledger["relative_photon_residual"]
        for row in all_rows
        for ledger in row["photon_ledgers"].values()
    )
    candidates = {
        name: {"validation_objective": _number(candidate["validation_objective"])}
        for name, candidate in selection["candidates"].items()
    }
    for name, candidate in selection["candidates"].items():
        if candidate.get("training_run"):
            summary = _json(root / candidate["training_run"] / "summary.json")
            candidates[name]["training_elapsed_seconds"] = _number(
                summary["elapsed_seconds"]
            )
    result = {
        "schema": 1,
        "complete": True,
        "fitting_performed": False,
        "selection_performed": False,
        "verification": {
            "selection_and_candidate_hashes_verified": True,
            "pre_blind_gate_verified": True,
            "benchmark_receipts_verified": True,
            "model_and_inference_hashes_unchanged": True,
            "selection_sha256": selection_hash,
            "evaluation_gate_sha256": _digest(gate_path),
            "evaluation_protocol_sha256": protocol_hash,
            "evaluation_inputs_sha256": inputs_hash,
            "summary_builder_sha256": _digest(Path(__file__)),
        },
        "selection": {
            "experiment": selection["selected_experiment"],
            "validation_objective": selection["selected_validation_objective"],
            "frozen_utc": selection["frozen_utc"],
            "validation_snapshot_count": selection["validation_snapshot_count"],
            "deployment_support": selection["deployment_support"],
            "candidates": candidates,
            "model_files_sha256": selection["model_files_sha256"],
        },
        "scope": {
            "averaging": "Equal snapshots; each power error first averages defined Fourier shells",
            "active": "0.05 < teacher mass-weighted xHI < 0.95",
            "late": "teacher mass-weighted xHI <= 0.05, including fully ionized endpoints",
            "fully_ionized": "teacher mass-weighted xHI <= 1e-10",
            "power": "Common saturated-spin real-space 21-cm brightness, no RSD; 0 < k < 0.5 k_Nyquist; undefined ratios counted",
            "mfp": "Geometric first exit at xHI=0.5; cap equals box length; restricted means include capped rays; empty phases undefined; common 4096 draws per phase",
            "conservation": "Saved-field absorption plus separately stored unabsorbed photons; absolute ledger residual divided by max(emitted photons, float64 tiny); no native-ES reservoir ledger",
            "timing": "Per-model inference includes features/projection and excludes I/O/diagnostics; MFP timed separately; native-ES runtime not measured",
        },
        "snapshot_count": len(all_rows),
        "maximum_relative_photon_residual": photon_residual,
        "profiles": profiles,
    }
    if (
        _digest(root / "selection.json") != selection_hash
        or _json(gate_path) != gate
        or _digest(protocol_path) != protocol_hash
        or _digest(inputs_path) != inputs_hash
        or any(
            protocol.get(key) != value
            for key, value in _evaluation_code_hashes().items()
        )
        or _inference_source_hashes() != selection["inference_source_sha256"]
        or _model_hashes(selected_root.resolve()) != selection["model_files_sha256"]
        or _model_hashes(previous_root.resolve())
        != selection["candidates"]["previous_model"]["model_files_sha256"]
        or any(_digest(path) != expected for path, expected in before.items())
    ):
        raise ValueError("Frozen model, source or reports changed while summarizing")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return result


def main() -> None:
    """Summarize only completed, verified frozen-model results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.root, args.output)


if __name__ == "__main__":
    main()
