"""Exercise fixed-field ablations and frozen-base residual training together."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from research.pc_analytic_emulator import train
from research.pc_analytic_emulator.budget import (
    TabulatedBudgetModel,
    constant_efficiency_model,
)
from research.pc_analytic_emulator.data import load_dataset, split_samples
from research.pc_analytic_emulator.model import Emulator
from research.pc_analytic_emulator.test_data import make_campaign


def _task(tmp_path: Path) -> dict:
    root = make_campaign(tmp_path / "teacher", seeds=(1001, 1006, 1007, 1008), n=4)
    config = train.validate_config(
        {
            "input_root": str(root),
            "train_seeds": [1001],
            "validation_seeds": [1006, 1007],
            "test_seeds": [1008],
            "model_seeds": [11],
            "epochs": 2,
            "patience": 2,
            "samples_per_snapshot": 16,
            "hidden": [4],
            "budget_hidden": [3],
            "workers": 1,
            "threads": 1,
            "cpus": 1,
        }
    )
    config.update(
        training_method="warm_started_v3",
        warmstart_epochs=2,
        warmstart_learning_rate=3e-4,
    )
    dataset = load_dataset(root)
    splits = split_samples(
        dataset,
        train_seeds=config["train_seeds"],
        validation_seeds=config["validation_seeds"],
        test_seeds=config["test_seeds"],
    )
    identity = {"configuration": config, "fixture": "synthetic-only"}
    cache, legacy_cache = tmp_path / "cache", tmp_path / "legacy_cache"
    details = train._prepare_v2_cache(cache, splits, config, identity)
    train._prepare_cache(legacy_cache, splits, config, identity)
    width = len(details["budget_feature_names"])
    models = {
        "constant_efficiency": constant_efficiency_model(width),
        "fixture_fraction": TabulatedBudgetModel(
            np.array([0.0]), np.array([2 / 3]), width
        ),
    }
    budget_rows = []
    for kind, model in models.items():
        path = tmp_path / f"{kind}.npz"
        model.save(path)
        budget_rows.append({"kind": kind, "model_path": str(path)})
    return {
        "config": config,
        "identity": identity,
        "model_seed": 11,
        "folder": str(tmp_path / "trial"),
        "cache": str(cache),
        "legacy_cache": str(legacy_cache),
        "budget_candidates": budget_rows,
    }


def test_fixed_field_candidates_and_deployed_warmstart_are_separate(
    tmp_path: Path,
) -> None:
    """Cross budgets/projections on one field before training a frozen-base residual."""
    from research.pc_analytic_emulator.warmstart import fit_trial

    task = _task(tmp_path)
    report = fit_trial(task)
    fixed = [row for row in report["candidates"] if row["stage"] == "fixed_field"]
    assert len(fixed) == 6
    assert len({row["field_sha256"] for row in fixed}) == 1
    assert {row["budget_kind"] for row in fixed} == {
        "legacy_budget",
        "constant_efficiency",
        "fixture_fraction",
    }
    assert {row["projection_mode"] for row in fixed} == {
        "log_odds",
        "capacity_l2",
    }
    for row in report["candidates"]:
        assert {item["seed"] for item in row["validation_by_seed"]} == {1006, 1007}
        assert {item["seed"] for item in row["snapshots"]} == {1006, 1007}
        assert all("delta_Q_mass" in item for item in row["snapshots"])
    assert not report["test_used"]
    base = Emulator.load(Path(task["folder"]) / "base_model")
    trained = Emulator.load(Path(task["folder"]) / "trained_model")
    assert trained.base_model is not None
    assert (
        report["validation_objective"]
        <= report["base_selection"]["validation_objective"] + 1e-12
    )
    details = json.loads((Path(task["cache"]) / "features.json").read_text())
    sample = details["validation_snapshots"][0]
    with np.load(sample["cache"], allow_pickle=False) as frame:
        args = (
            frame["source"],
            frame["capacity"],
            frame["native_es_xHI"],
            sample["redshift"],
            sample["box_len"],
        )
        np.testing.assert_array_equal(
            trained.base_model.predict(*args)["xHII"], base.predict(*args)["xHII"]
        )
    receipt = Path(task["folder"]) / "receipt.json"
    before = (receipt.read_bytes(), receipt.stat().st_mtime_ns)
    assert fit_trial(task) == report
    assert before == (receipt.read_bytes(), receipt.stat().st_mtime_ns)


def test_interrupted_residual_trial_preserves_verified_legacy_fit(
    tmp_path: Path,
) -> None:
    """An interrupted outer trial reuses the completed legacy model unchanged."""
    from research.pc_analytic_emulator.warmstart import fit_trial

    task = _task(tmp_path)
    fit_trial(task)
    folder = Path(task["folder"])
    legacy = folder / "legacy_reference/model/field_model.npz"
    original = (legacy.read_bytes(), legacy.stat().st_mtime_ns)
    (folder / "receipt.json").unlink()
    report = fit_trial(task)
    assert not report["test_used"]
    assert original == (legacy.read_bytes(), legacy.stat().st_mtime_ns)


def test_duplicate_budget_names_fail_before_training(tmp_path: Path) -> None:
    """Ambiguous ablation names cannot silently replace candidate predictors."""
    from research.pc_analytic_emulator.warmstart import fit_trial

    task = _task(tmp_path)
    task["budget_candidates"].append(task["budget_candidates"][0])
    with pytest.raises(ValueError, match=r"budget.*names"):
        fit_trial(task)


def test_residual_workflow_has_two_validation_seeds_and_separate_external_test(
    tmp_path: Path,
) -> None:
    """The complete v3 workflow fits no inspected or external test snapshots."""
    task = _task(tmp_path)
    external = make_campaign(tmp_path / "external", seeds=(2001,), n=4)
    config = task["config"] | {"external_test_root": str(external)}
    output = tmp_path / "full_run"
    summary = train.run(config, output)
    assert summary["complete"]
    assert summary["training_method"] == "warm_started_v3"
    assert summary["split_seeds"]["validation"] == [1006, 1007]
    assert summary["sample_counts"] == {"train": 2, "validation": 4, "test": 2}
    assert summary["internal_test_role"] == "Regression"
    assert summary["known_regression_test_seeds"] == [1008]
    assert summary["external_test"]["seeds"] == [2001]
    assert summary["external_test"]["used_for_selection"] is False
    assert summary["maximum_test_photon_residual"] < 1e-12
    selection = json.loads((output / "model_selection.json").read_text())
    trial = selection["selected"]
    assert {row["seed"] for row in trial["validation_by_seed"]} == {1006, 1007}
    assert selection["test_used_for_selection"] is False
    assert trial["fixed_field_retrained_during_ablation"] is False
    assert trial["residual_base_frozen"]
    fit_cache = json.loads((output / "fit_cache/features.json").read_text())
    assert {row["seed"] for row in fit_cache["training_snapshots"]} == {1001}
    assert {row["seed"] for row in fit_cache["validation_snapshots"]} == {1006, 1007}
    assert (output / "comparison_summary.json").is_file()
    assert train.run(config, output) == summary
