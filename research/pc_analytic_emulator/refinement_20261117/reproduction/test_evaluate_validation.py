"""Check fixed-reference validation scores without fitting or test selection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from research.pc_analytic_emulator import train
from research.pc_analytic_emulator.data import load_dataset
from research.pc_analytic_emulator.model import Emulator
from research.pc_analytic_emulator.objectives import projected_objective
from research.pc_analytic_emulator.refinement_20261117.reproduction import (
    evaluate_validation,
)
from research.pc_analytic_emulator.test_data import make_campaign, write_collection
from research.pc_analytic_emulator.test_model import fixture_model


def _fixture(root: Path) -> tuple[Path, Path, Path]:
    first = make_campaign(root / "first", seeds=(1, 2), n=4, box_len=50)
    second = make_campaign(root / "second", seeds=(3,), n=6, box_len=80)
    collection = write_collection(
        root / "collection", [first, second], allow_mixed_geometry=True
    )
    dataset = load_dataset(first)
    sample = dataset.samples[0]
    model = fixture_model(
        sample.arrays["source"],
        sample.arrays["capacity"],
        sample.arrays["native_es_xHI"],
    )
    model.metadata["teacher_physics"] = dataset.physics
    model.save(root / "model")
    config = {
        "input_root": str(collection),
        "training_method": "warm_started_v3",
        "train_seeds": [1],
        "validation_seeds": [2, 3],
        "test_seeds": [],
        "allow_empty_test": True,
        "spectral_weight": 0.2,
        "spectral_power_floor": 1e-6,
        "field_variance_floor": 0.005,
    }
    path = root / "config.json"
    path.write_text(json.dumps(config))
    return root / "model", collection, path


def test_validation_matches_exact_training_objective_without_fitting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the identical objective, actual grid and validation-only seed list."""
    model_path, teacher, config_path = _fixture(tmp_path)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("A fixed validation evaluator must never train")

    monkeypatch.setattr(train, "run", forbidden)
    output = tmp_path / "baseline.json"
    report = evaluate_validation.run(model_path, config_path, output)
    assert report["complete"]
    assert report["model_weights_unchanged"]
    assert not report["fitting_performed"]
    assert not report["model_selection_performed"]
    assert report["validation_seeds"] == [2, 3]
    assert report["snapshot_count"] == 4
    assert len(report["by_geometry"]) == 2
    assert {row["seed"] for row in report["snapshots"]} == {2, 3}
    assert str(tmp_path) not in output.read_text()
    cfg = train.validate_config(json.loads(config_path.read_text()))
    model = Emulator.load(model_path)
    expected = []
    for sample in load_dataset(teacher).samples:
        if sample.seed == 1:
            continue
        arrays = sample.arrays
        predicted = model.predict(
            arrays["source"],
            arrays["capacity"],
            arrays["native_es_xHI"],
            sample.redshift,
            sample.box_len,
        )
        loss, _, _ = projected_objective(
            predicted["xHII"],
            1 - arrays["teacher_xHI"],
            arrays["capacity"],
            sample.box_len,
            **{name: cfg[name] for name in report["objective"]},
        )
        expected.append(loss)
    assert report["validation_objective"] == pytest.approx(
        np.mean(expected), rel=0, abs=1e-15
    )
    with pytest.raises(FileExistsError):
        evaluate_validation.run(model_path, config_path, output)


def test_configuration_override_is_portable_and_tests_are_forbidden(
    tmp_path: Path,
) -> None:
    """An explicit teacher path can replace a stale machine-local configuration."""
    model, teacher, config = _fixture(tmp_path)
    settings = json.loads(config.read_text())
    settings["input_root"] = "/unavailable/on/this/machine"
    config.write_text(json.dumps(settings))
    result = evaluate_validation.run(
        model, config, tmp_path / "portable.json", teacher_root=teacher
    )
    assert result["snapshot_count"] == 4
    settings["test_seeds"] = [4]
    config.write_text(json.dumps(settings))
    with pytest.raises(ValueError, match="development-only"):
        evaluate_validation.run(
            model, config, tmp_path / "invalid.json", teacher_root=teacher
        )


def test_model_mutation_during_reference_scoring_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No final validation report can certify weights changed during evaluation."""
    model, _, config = _fixture(tmp_path)
    original = evaluate_validation._score_candidate

    def changed(*args: object, **kwargs: object) -> dict:
        result = original(*args, **kwargs)
        with (model / "model.json").open("a") as stream:
            stream.write("\n")
        return result

    monkeypatch.setattr(evaluate_validation, "_score_candidate", changed)
    output = tmp_path / "changed.json"
    with pytest.raises(ValueError, match="changed during"):
        evaluate_validation.run(model, config, output)
    assert not output.exists()
