"""Test frozen result summaries using synthetic records and real file receipts."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from research.pc_analytic_emulator import benchmark
from research.pc_analytic_emulator.refinement_20261117.reproduction import (
    summarize_results as summary,
)

METHODS = ("selected", "previous", "native_es")
GEOMETRIES = {
    "original": (16, 50.0, 2201),
    "resolution": (32, 50.0, 2211),
    "volume": (32, 256.0, 2221),
    "unseen_box": (32, 100.0, 2231),
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def _row(neutral: float, mae: float, *, redshift: float = 7.5) -> dict:
    power = [1.1, 0.9] if neutral > 0 else [None, None]
    correlation = [0.99, 0.97] if neutral > 0 else [None, None]
    spectral = {
        "power_ratio": power,
        "cross_correlation": correlation,
        "defined_shells": 2 if neutral > 0 else 0,
        "mean_absolute_power_ratio_error": 0.1 if neutral > 0 else None,
        "maximum_absolute_power_ratio_error": 0.1 if neutral > 0 else None,
        "minimum_cross_correlation": 0.97 if neutral > 0 else None,
    }
    field = {
        "MAE": mae,
        "RMSE": 2 * mae,
        "delta_Q_mass": -mae / 3,
        "absolute_delta_Q_mass": mae / 3,
        "delta_Q_volume": -mae / 2,
        "maximum_absolute_error": min(1.0, 3 * mae),
        "capacity_weighted_MSE": 4 * mae**2,
        "field_correlation": 0.9 if neutral > 0 else None,
        "xHI_mass": neutral,
        "xHI_volume": neutral,
    }
    mfp = {}
    for phase in ("ionized", "neutral"):
        empty = phase == "neutral" and neutral == 0
        teacher_cap, predicted_cap = (0.2, 0.3) if phase == "ionized" else (0.1, 0.05)
        teacher_mean, predicted_mean = (20, 23) if phase == "ionized" else (10, 8)
        if neutral == 0 and phase == "ionized":
            teacher_cap = predicted_cap = 1.0
            teacher_mean = predicted_mean = 50.0
        comparison = {"available": not empty}
        if not empty:
            comparison.update(
                restricted_mean_difference_cMpc=predicted_mean - teacher_mean,
                restricted_mean_ratio=predicted_mean / teacher_mean,
                restricted_CDF_maximum_difference=0.15,
                cap_fraction_difference=predicted_cap - teacher_cap,
            )
        measurements = {
            name: {
                "status": "empty_phase" if empty else "measured",
                "restricted_mean_cMpc": None
                if empty
                else teacher_mean
                if name == "teacher"
                else predicted_mean,
                "cap_fraction": None
                if empty
                else teacher_cap
                if name == "teacher"
                else predicted_cap,
            }
            for name in ("teacher", *METHODS)
        }
        mfp[phase] = {
            "measurements": measurements,
            "comparisons": {name: copy.deepcopy(comparison) for name in METHODS},
        }
    return {
        "seed": 2201,
        "redshift": redshift,
        "dimension": 16,
        "box_len_cMpc": 50.0,
        "teacher_xHI_mass": neutral,
        "field_metrics": {name: copy.deepcopy(field) for name in METHODS},
        "photon_ledgers": {
            name: {"passed": True, "relative_photon_residual": 1e-15}
            for name in ("teacher", "selected", "previous")
        },
        "endpoints": {
            name: {"exactly_fully_ionized": neutral == 0}
            for name in ("teacher", *METHODS)
        },
        "low_k_21cm": {
            "k_cMpc_inv": [0.1, 0.2],
            "modes": [6, 12],
            "comparisons": {name: copy.deepcopy(spectral) for name in METHODS},
        },
        "full_band_21cm": {
            "comparisons": {name: copy.deepcopy(spectral) for name in METHODS}
        },
        "mfp": mfp,
        "inference_seconds_excluding_io": {"selected": 0.01, "previous": 0.02},
        "mfp_seconds_excluding_io": 0.03,
        "files": {},
    }


def _seal(folder: Path) -> None:
    _write(
        folder / "receipt.json",
        {
            "complete": True,
            "files_sha256": {
                str(path.relative_to(folder)): _digest(path)
                for path in folder.rglob("*")
                if path.is_file() and path.name not in ("receipt.json", "COMPLETED")
            },
        },
    )
    (folder / "COMPLETED").write_text("Synthetic complete benchmark\n")


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "campaign"
    root.mkdir()
    source_hashes = {"inference.py": hashlib.sha256(b"frozen code").hexdigest()}
    models = {}
    for name in ("selected", "previous"):
        model = root / f"{name}_model"
        _write(model / "model.json", {"synthetic_model": name})
        models[name] = {
            "root": str(model),
            "files_sha256": {"model.json": _digest(model / "model.json")},
        }

    def model_hashes(path: Path) -> dict:
        return {"model.json": _digest(Path(path) / "model.json")}

    selection = {
        "model_files_sha256": models["selected"]["files_sha256"],
        "selected_model_root": "selected_model",
        "inference_source_sha256": source_hashes,
        "candidates": {
            "previous_model": {
                "model_root": "previous_model",
                "model_files_sha256": models["previous"]["files_sha256"],
                "validation_objective": 0.5,
            }
        },
        "selected_experiment": "guarded_mixed",
        "selected_validation_objective": 0.25,
        "frozen_utc": "2026-11-17T00:00:00+00:00",
        "validation_snapshot_count": 42,
        "deployment_support": [{"HII_DIM": 16, "DIM": 96, "BOX_LEN": 50}],
    }
    _write(root / "training_protocol.json", {"synthetic_protocol": True})
    selection.update(
        training_protocol="training_protocol.json",
        training_protocol_sha256=_digest(root / "training_protocol.json"),
    )
    _write(root / "selection.json", selection)
    _write(
        root / "evaluation_started.json",
        {
            "training_protocol_sha256": _digest(root / "training_protocol.json"),
            "selection_sha256": _digest(root / "selection.json"),
            "model_files_sha256": models["selected"]["files_sha256"],
        },
    )
    monkeypatch.setattr(summary, "verify_selection", lambda _: selection)
    monkeypatch.setattr(summary, "_inference_source_hashes", lambda: source_hashes)
    monkeypatch.setattr(summary, "_model_hashes", model_hashes)
    code_hashes = summary._evaluation_code_hashes()
    _write(
        root / "evaluation_protocol.json",
        {
            "schema": 1,
            "selection_sha256": _digest(root / "selection.json"),
            "training_protocol_sha256": _digest(root / "training_protocol.json"),
            "inference_source_sha256": source_hashes,
            "rays_per_phase": 4096,
            "models": models,
            "evaluations": {
                f"blind_{name}": {
                    "root": str(root / f"teacher_blind_{name}"),
                    "seeds": [seed],
                    "dimension": dimension,
                    "box_len": length,
                }
                for name, (dimension, length, seed) in GEOMETRIES.items()
            },
            **code_hashes,
        },
    )
    _write(
        root / "evaluation_inputs.json",
        {f"blind_{name}": {"files_sha256": {}} for name in GEOMETRIES},
    )
    for name, (dimension, length, seed) in GEOMETRIES.items():
        folder = root / f"benchmark_blind_{name}"
        manifest = {
            "schema": 1,
            "state": "models frozen before teacher evaluation",
            "teacher_root": str(root / f"teacher_blind_{name}"),
            "requested_seeds": [seed],
            "rays_per_phase": 4096,
            "models": models,
            "inference_source_sha256": source_hashes,
            "benchmark_sha256": _digest(Path(benchmark.__file__)),
            "independent_metrics_sha256": _digest(Path(benchmark.recompute.__file__)),
            "fitting_performed": False,
            "selection_performed": False,
        }
        _write(folder / "manifest.json", manifest)
        _write(folder / "inputs.json", {"identity": "synthetic", "files_sha256": {}})
        rows = [
            _row(0.5, 0.01),
            _row(0.01, 0.8, redshift=6.5),
            _row(0, 0.2, redshift=6),
        ]
        for row in rows:
            row.update(seed=seed, dimension=dimension, box_len_cMpc=length)
        report = {
            "complete": True,
            "model_weights_unchanged": True,
            "inference_implementation_unchanged": True,
            "fitting_performed": False,
            "model_selection_performed": False,
            "model_files_after_sha256": {
                key: value["files_sha256"] for key, value in models.items()
            },
            "snapshot_count": len(rows),
            "snapshots": rows,
            "input_record_sha256": _digest(folder / "inputs.json"),
            "frozen_manifest_sha256": _digest(folder / "manifest.json"),
            "summaries": {
                "all": {"count": 999, "methods": {"selected": {"MAE": {"mean": 999}}}}
            },
        }
        _write(folder / "report.json", report)
        _seal(folder)
    return root


def test_group_retains_undefined_power_and_phase_counts() -> None:
    """Zero reference power is undefined, not a zero-error success."""
    rows = [_row(0.5, 0.01), _row(0.01, 0.8), _row(0, 0.2)]
    for method in METHODS:
        power = rows[1]["low_k_21cm"]["comparisons"][method]
        power["power_ratio"] = [2.0, 0.0]
        power["cross_correlation"] = [0.8, None]
        # Stored per-row aggregates are deliberately stale. Ratios are primary.
        power["mean_absolute_power_ratio_error"] = 999
        power["maximum_absolute_power_ratio_error"] = 999
        power["minimum_cross_correlation"] = -999
    result = summary.summarize_group(rows)
    assert result["count"] == 3
    assert result["fully_ionized_count"] == 1
    power = result["methods"]["selected"]["low_k_21cm"]
    assert power["snapshot_mean_absolute_power_ratio_error"]["mean"] == pytest.approx(
        0.55
    )
    assert power["snapshot_mean_absolute_power_ratio_error"]["defined_count"] == 2
    assert power["snapshot_mean_absolute_power_ratio_error"]["undefined_count"] == 1
    assert power["minimum_defined_cross_correlation"] == pytest.approx(0.8)
    assert power["defined_shells"] == 4
    neutral = result["methods"]["selected"]["mfp"]["neutral"]
    assert neutral["restricted_mean_absolute_difference_cMpc"]["mean"] == 2
    assert neutral["restricted_mean_absolute_difference_cMpc"]["defined_count"] == 2
    assert neutral["restricted_mean_absolute_difference_cMpc"]["undefined_count"] == 1


def test_group_includes_capped_rays_in_restricted_means_and_cap_statistics() -> None:
    """Fully capped lengths remain defined, while an empty phase remains absent."""
    rows = [_row(0.5, 0.01), _row(0, 0.2)]
    result = summary.summarize_group(rows)
    ionized = result["methods"]["selected"]["mfp"]["ionized"]
    assert ionized["restricted_mean_absolute_difference_cMpc"]["mean"] == pytest.approx(
        1.5
    )
    assert ionized["predicted_cap_fraction"]["mean"] == pytest.approx(0.65)
    assert ionized["teacher_cap_fraction"]["mean"] == pytest.approx(0.6)
    assert ionized["absolute_cap_fraction_difference"]["mean"] == pytest.approx(0.05)
    neutral = result["methods"]["selected"]["mfp"]["neutral"]
    assert neutral["predicted_cap_fraction"]["defined_count"] == 1
    assert neutral["predicted_cap_fraction"]["undefined_count"] == 1


def test_group_rejects_inconsistent_declared_power_shell_count() -> None:
    """The number of valid shells must agree with the actual ratio vector."""
    row = _row(0.5, 0.01)
    row["low_k_21cm"]["comparisons"]["selected"]["defined_shells"] = 99
    with pytest.raises(ValueError, match=r"(?i)(shell|power|defined)"):
        summary.summarize_group([row])


def test_run_recomputes_phase_summaries_and_writes_requested_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Active averages never hide a large late error or trust stale summaries."""
    root = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "custom_destination" / "summary.json"
    result = summary.run(root, output)
    assert json.loads(output.read_text()) == result
    assert str(tmp_path) not in output.read_text()
    assert set(result["profiles"]) == set(GEOMETRIES)
    for profile in result["profiles"].values():
        groups = profile["summaries"]
        assert groups["all"]["count"] == 3
        assert groups["active"]["count"] == 1
        assert groups["late"]["count"] == 2
        assert groups["all"]["methods"]["selected"]["MAE"]["mean"] == pytest.approx(
            1.01 / 3
        )
        assert groups["active"]["methods"]["selected"]["MAE"]["mean"] == 0.01
        assert groups["late"]["methods"]["selected"]["MAE"]["mean"] == 0.5
        assert groups["late"]["fully_ionized_count"] == 1


def test_run_requires_untampered_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-completion edit to benchmark records invalidates their checksum."""
    root = _fixture(tmp_path, monkeypatch)
    report = root / "benchmark_blind_original" / "report.json"
    report.write_text(report.read_text() + " ")
    with pytest.raises(ValueError, match=r"(?i)(checksum|receipt|hash|changed)"):
        summary.run(root)


def test_run_rejects_resealed_model_identity_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid file receipt alone cannot bind a benchmark to a different model."""
    root = _fixture(tmp_path, monkeypatch)
    folder = root / "benchmark_blind_original"
    manifest = json.loads((folder / "manifest.json").read_text())
    manifest["models"]["selected"]["files_sha256"]["model.json"] = "f" * 64
    _write(folder / "manifest.json", manifest)
    report = json.loads((folder / "report.json").read_text())
    report["frozen_manifest_sha256"] = _digest(folder / "manifest.json")
    _write(folder / "report.json", report)
    _seal(folder)
    with pytest.raises(ValueError, match=r"(?i)(model|identity|frozen|selection)"):
        summary.run(root)


def test_run_refuses_incomplete_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All prescribed geometries must have complete benchmark evidence."""
    root = _fixture(tmp_path, monkeypatch)
    (root / "benchmark_blind_volume" / "COMPLETED").unlink()
    with pytest.raises(
        (ValueError, FileNotFoundError),
        match=r"(?i)(complete|COMPLETED|profile|benchmark)",
    ):
        summary.run(root)


def test_run_rejects_selection_gate_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The evaluation gate must identify the actual frozen selection bytes."""
    root = _fixture(tmp_path, monkeypatch)
    path = root / "evaluation_started.json"
    gate = json.loads(path.read_text())
    gate["selection_sha256"] = "0" * 64
    _write(path, gate)
    with pytest.raises(ValueError, match=r"(?i)(selection|gate|evaluation|frozen)"):
        summary.run(root)


def test_run_refuses_overwriting_a_frozen_benchmark_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A custom output cannot silently replace the evidence being summarized."""
    root = _fixture(tmp_path, monkeypatch)
    destination = root / "benchmark_blind_original" / "report.json"
    before = _digest(destination)
    with pytest.raises(ValueError, match=r"(?i)(output|replace|frozen|artifact)"):
        summary.run(root, destination)
    assert _digest(destination) == before


def test_run_requires_snapshot_artifact_hashes_to_match_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resealing JSON cannot hide mismatched per-snapshot prediction identities."""
    root = _fixture(tmp_path, monkeypatch)
    folder = root / "benchmark_blind_original"
    artifact = folder / "seed2201" / "synthetic_predictions.bin"
    artifact.parent.mkdir()
    artifact.write_bytes(b"synthetic predictions without arrays")
    report = json.loads((folder / "report.json").read_text())
    report["snapshots"][0]["files"] = {
        "predictions": {"path": str(artifact.relative_to(folder)), "sha256": "0" * 64}
    }
    _write(folder / "report.json", report)
    _seal(folder)
    with pytest.raises(ValueError, match=r"(?i)(artifact|receipt|snapshot|hash)"):
        summary.run(root)


def test_empty_group_has_no_manufactured_zero_error() -> None:
    """A missing phase group returns undefined means with zero defined samples."""
    group = summary.summarize_group([])
    assert group["count"] == 0
    assert group["fully_ionized_count"] == 0
    field = group["methods"]["selected"]["MAE"]
    assert field["mean"] is None
    assert field["defined_count"] == 0
    power = group["methods"]["selected"]["low_k_21cm"]
    assert power["snapshot_mean_absolute_power_ratio_error"]["mean"] is None
    assert power["minimum_defined_cross_correlation"] is None


@pytest.mark.parametrize("key", ["benchmark_sha256", "independent_metrics_sha256"])
def test_run_rejects_resealed_metric_implementation_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    """Benchmark receipts cannot substitute a different scientific metric code."""
    root = _fixture(tmp_path, monkeypatch)
    folder = root / "benchmark_blind_resolution"
    manifest = json.loads((folder / "manifest.json").read_text())
    manifest[key] = "f" * 64
    _write(folder / "manifest.json", manifest)
    report = json.loads((folder / "report.json").read_text())
    report["frozen_manifest_sha256"] = _digest(folder / "manifest.json")
    _write(folder / "report.json", report)
    _seal(folder)
    with pytest.raises(
        ValueError, match=r"(?i)(metric|implementation|code|source|frozen|pre-blind)"
    ):
        summary.run(root)


def test_run_rejects_resealed_input_identity_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The report must refer to the inputs fixed before scientific evaluation."""
    root = _fixture(tmp_path, monkeypatch)
    folder = root / "benchmark_blind_unseen_box"
    _write(folder / "inputs.json", {"files_sha256": {"changed_teacher": "f" * 64}})
    report = json.loads((folder / "report.json").read_text())
    report["input_record_sha256"] = _digest(folder / "inputs.json")
    _write(folder / "report.json", report)
    _seal(folder)
    with pytest.raises(
        ValueError, match=r"(?i)(input|identity|teacher|frozen|pre-blind)"
    ):
        summary.run(root)
