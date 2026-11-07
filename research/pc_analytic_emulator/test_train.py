"""Training orchestration checks on explicitly synthetic teacher archives."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from research.pc_analytic_emulator import train


def test_sampling_is_stable_and_does_not_repeat() -> None:
    """Sampling uses a stable realization identifier without repeated cells."""
    first = train._sampling_indices(1001, 7.5, 1000, 100, 2026)
    assert np.array_equal(first, train._sampling_indices(1001, 7.5, 1000, 100, 2026))
    assert len(np.unique(first)) == 100
    assert not np.array_equal(
        first, train._sampling_indices(1002, 7.5, 1000, 100, 2026)
    )
    assert np.array_equal(
        train._sampling_indices(1001, 7.5, 8, 100, 2026), np.arange(8)
    )


@pytest.mark.parametrize(
    "change",
    [
        {"workers": 5},
        {"epochs": 0},
        {"epochs": True},
        {"samples_per_snapshot": -2},
        {"learning_rate": float("nan")},
        {"radii_cMpc": [0.0]},
        {"model_seeds": [1, 1]},
        {"unknown_setting": 1},
        {"validation_seeds": [1001]},
        {"training_method": "unknown"},
        {"v2_epochs": 0},
        {"spectral_weight": -1},
        {"spectral_power_floor": -1},
        {"raw_weight": float("nan")},
        {"field_variance_floor": 0},
        {"k_fraction": 1.1},
        {"regression_test_seeds": [1008, 1008]},
    ],
)
def test_bad_training_config_is_rejected(tmp_path: Path, change: dict) -> None:
    """Invalid resources, optimization controls and overlapping seeds fail."""
    with pytest.raises((ValueError, TypeError)):
        train.validate_config({"input_root": str(tmp_path), **change})


def test_training_method_is_explicit_and_legacy_remains_default(tmp_path: Path) -> None:
    """An improved objective requires an explicit method discriminator."""
    config = train.validate_config({"input_root": str(tmp_path)})
    assert config["training_method"] == "legacy"
    assert config["v2_epochs"] == 80
    improved = train.validate_config(
        {
            "input_root": str(tmp_path),
            "training_method": "projected_multiscale_v2",
            "regression_test_seeds": [],
        }
    )
    assert improved["spectral_weight"] == 0.1
    assert improved["spectral_power_floor"] == 1e-4
    assert improved["regression_test_seeds"] == [1008]


def test_candidate_selection_uses_validation_only() -> None:
    """Held-out labels cannot affect trial selection or deterministic ties."""
    rows = [
        {"model_seed": 23, "validation_mse": 0.2, "test_mse": 0.0},
        {"model_seed": 11, "validation_mse": 0.1, "test_mse": 1.0},
    ]
    assert train._choose_trial(rows)["model_seed"] == 11
    rows[0]["validation_mse"] = 0.1
    assert train._choose_trial(rows)["model_seed"] == 11
    with pytest.raises(ValueError, match="finite"):
        train._choose_trial([{"model_seed": 11, "validation_mse": float("nan")}])


def test_active_snapshot_aggregate_excludes_trivial_fields() -> None:
    """Saturated snapshots cannot dilute the separately reported active error."""
    rows = [
        {
            "seed": 1008,
            "redshift": redshift,
            "evaluation": {"metrics": {"teacher": {"Q_mass": q_mass}}},
            "field_metrics": {
                name: dict.fromkeys(
                    ("MAE", "RMSE", "delta_Q_mass", "delta_Q_volume"), error
                )
                for name in ("raw", "constrained", "native_es")
            },
        }
        for redshift, q_mass, error in (
            (9.0, 0.0, 0.0),
            (8.0, 0.5, 0.3),
            (6.0, 1.0, 0.0),
        )
    ]
    result = train._test_aggregates(rows)
    assert result["test_active_snapshots"]["count"] == 1
    assert result["test_active_snapshots"]["total_count"] == 3
    assert result["test_metrics_mean_over_snapshots"]["constrained"][
        "MAE"
    ] == pytest.approx(0.1)
    assert (
        result["test_metrics_mean_over_active_snapshots"]["constrained"]["MAE"] == 0.3
    )
    assert (
        train._test_aggregates(rows[:1])["test_metrics_mean_over_active_snapshots"]
        is None
    )


def test_small_training_and_verified_resume(tmp_path: Path) -> None:
    """An audited synthetic dataset can train, evaluate and resume intact."""
    from research.pc_analytic_emulator.test_data import make_campaign

    inputs = make_campaign(
        tmp_path / "teacher", seeds=(1001, 1007, 1008), redshifts=(8.0, 7.5), n=4
    )
    config = {
        "input_root": str(inputs),
        "train_seeds": [1001],
        "validation_seeds": [1007],
        "test_seeds": [1008],
        "model_seeds": [11, 23],
        "workers": 1,
        "threads": 1,
        "cpus": 1,
        "epochs": 2,
        "patience": 2,
        "samples_per_snapshot": 32,
        "hidden": [4],
        "budget_hidden": [3],
        "batch_size": 32,
    }
    output = tmp_path / "fit"
    summary = train.run(config, output)
    assert summary["complete"]
    assert summary["sample_counts"] == {"train": 2, "validation": 2, "test": 2}
    assert summary["split_seeds"] == {
        "train": [1001],
        "validation": [1007],
        "test": [1008],
    }
    assert (output / "COMPLETED").is_file()
    assert len(list((output / "test").glob("*/predictions.npz"))) == 2
    selection = json.loads((output / "model_selection.json").read_text())
    assert (
        selection["selection_scope"]
        == "Full-grid constrained validation MSE; test labels are excluded"
    )
    model_file = output / "model" / "field_model.npz"
    original = (model_file.read_bytes(), model_file.stat().st_mtime_ns)
    repeated = train.run(config, output)
    assert repeated == summary
    assert original == (model_file.read_bytes(), model_file.stat().st_mtime_ns)
    with pytest.raises(ValueError, match="identity"):
        train.run(config | {"epochs": 3}, output)


def test_corrupted_completed_training_is_rejected(tmp_path: Path) -> None:
    """A changed saved output invalidates the completion receipt."""
    folder = tmp_path / "result"
    folder.mkdir()
    (folder / "summary.json").write_text("{}")
    (folder / "completion.json").write_text(
        json.dumps({"files": {"summary.json": "bad"}})
    )
    with pytest.raises(ValueError, match="checksum"):
        train._verify_completion(folder)


def test_projected_training_uses_full_cubes_and_separate_external_test(
    tmp_path: Path,
) -> None:
    """Improved training excludes both inspected and new test labels from fitting."""
    from research.pc_analytic_emulator.model import Emulator
    from research.pc_analytic_emulator.test_data import make_campaign

    inputs = make_campaign(tmp_path / "teacher", seeds=(1001, 1007, 1008), n=4)
    external = make_campaign(tmp_path / "external", seeds=(2001,), n=4)
    config = {
        "input_root": str(inputs),
        "external_test_root": str(external),
        "training_method": "projected_multiscale_v2",
        "train_seeds": [1001],
        "validation_seeds": [1007],
        "test_seeds": [1008],
        "model_seeds": [11],
        "workers": 1,
        "threads": 1,
        "cpus": 1,
        "epochs": 2,
        "v2_epochs": 2,
        "patience": 2,
        "hidden": [4],
        "budget_hidden": [3],
        "samples_per_snapshot": 2,
    }
    output = tmp_path / "fit_v2"
    summary = train.run(config, output)
    assert summary["training_method"] == "projected_multiscale_v2"
    assert summary["training_cells"] == 2 * 4**3
    assert summary["internal_test_role"] == "Regression"
    assert summary["known_regression_test_seeds"] == [1008]
    assert summary["external_test"]["seeds"] == [2001]
    assert summary["selected_validation_objective"] is not None
    assert summary["budget_selection"]["selected_before_field_training"]
    assert summary["maximum_test_photon_residual"] < 1e-12
    assert (
        Emulator.load(output / "model").projection_mode
        == summary["selected_projection_mode"]
    )
    details = json.loads((output / "features.json").read_text())
    assert {row["seed"] for row in details["training_snapshots"]} == {1001}
    assert {row["seed"] for row in details["validation_snapshots"]} == {1007}
    selection = json.loads((output / "model_selection.json").read_text())
    assert {row["kind"] for row in selection["selected"]["candidates"]} == {
        "trained_selected_budget",
        "trained_eta_one",
        "es_selected_budget",
        "es_eta_one",
        "legacy_reference",
        "legacy_trained",
    }
    assert summary["selected_validation_objective"] == min(
        row["validation_objective"] for row in selection["selected"]["candidates"]
    )
    assert len(list((output / "external_test/test").glob("*/predictions.npz"))) == 2
    assert train.run(config, output) == summary
    legacy_model = output / "trials/seed11/legacy_reference/model/field_model.npz"
    legacy_identity = (legacy_model.read_bytes(), legacy_model.stat().st_mtime_ns)
    (output / "COMPLETED").unlink()
    (output / "trials/seed11/receipt.json").unlink()
    resumed = train.run(config, output)
    assert resumed["complete"]
    assert legacy_identity == (
        legacy_model.read_bytes(),
        legacy_model.stat().st_mtime_ns,
    )
