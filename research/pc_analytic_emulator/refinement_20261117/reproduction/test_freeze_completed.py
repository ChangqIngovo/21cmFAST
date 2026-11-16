"""Exercise the validation-only selection gate without scientific test inputs."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.pc_analytic_emulator.refinement_20261117.reproduction import (
    freeze_completed as freeze,
)


def _setup(tmp_path: Path, monkeypatch, scores: dict[str, float]) -> dict:
    root = tmp_path / "output"
    root.mkdir()
    config = root / "guarded_mixed.json"
    config.write_text(
        json.dumps(
            {"test_seeds": [], "external_test_root": None, "input_root": str(root)}
        )
    )
    protocol = root / "training_protocol.json"
    protocol.write_text('{"schema": 1}\n')
    sources = {}
    for name in ("guarded_single", "guarded_mixed", "previous_model"):
        folder = tmp_path / name
        model = folder if name == "previous_model" else folder / "model"
        model.mkdir(parents=True)
        (model / "model.json").write_text(json.dumps({"name": name}))
        (model / "field_model.npz").write_bytes(name.encode())
        if name != "previous_model":
            (folder / "completion.json").write_text('{"files": {}}\n')
        sources[name] = folder

    def hashes(model):
        return {p.name: freeze.digest(p) for p in Path(model).iterdir() if p.is_file()}

    def score(model, configuration, output):
        name = json.loads((Path(model) / "model.json").read_text())["name"]
        report = {
            "complete": True,
            "model_sha256": hashes(model),
            "configuration_sha256": freeze.digest(configuration),
            "inference_source_sha256": {"code": "fixed"},
            "evaluation_script_sha256": freeze.digest(
                Path(freeze.evaluate_validation.__file__)
            ),
            "snapshot_count": 42,
            "validation_seeds": [2113, 2114, 2261, 2262, 2271, 2272],
            "fitting_performed": False,
            "test_evaluated": False,
            "by_geometry": [{"snapshot_count": 14}] * 3,
            "validation_objective": scores[name],
            "teacher_content_digest": hashlib.sha256(b"[]").hexdigest(),
        }
        Path(output).write_text(json.dumps(report))
        return report

    monkeypatch.setattr(freeze, "_model_hashes", hashes)
    monkeypatch.setattr(freeze, "_inference_source_hashes", lambda: {"code": "fixed"})
    monkeypatch.setattr(freeze, "_receipt", lambda *args: None)
    monkeypatch.setattr(
        freeze,
        "_dataset_identity",
        lambda *args: {"root": str(root), "files_sha256": {}},
    )
    monkeypatch.setattr(freeze.evaluate_validation, "run", score)
    monkeypatch.setattr(
        freeze.Emulator,
        "load",
        lambda path: SimpleNamespace(
            metadata={
                "training_geometries": [
                    json.loads((Path(path) / "model.json").read_text())["name"]
                ]
            }
        ),
    )
    return {
        "single_run": sources["guarded_single"],
        "mixed_run": sources["guarded_mixed"],
        "previous_model": sources["previous_model"],
        "validation_config": config,
        "training_protocol": protocol,
        "output_root": root,
    }


def test_freezer_selects_common_validation_minimum_and_verifies_resume(
    tmp_path, monkeypatch
):
    """Selection copies the winning checkpoint and freezes all candidate evidence."""
    kwargs = _setup(
        tmp_path,
        monkeypatch,
        {"guarded_single": 0.5, "guarded_mixed": 0.2, "previous_model": 1.0},
    )
    selected = freeze.run(**kwargs)
    assert selected["selected_experiment"] == "guarded_mixed"
    assert selected["deployment_support"] == ["guarded_mixed"]
    assert selected["blind_or_transfer_used_for_selection"] is False
    assert freeze.verify(kwargs["output_root"] / "selection.json") == selected
    monkeypatch.setattr(
        freeze.evaluate_validation,
        "run",
        lambda *args: pytest.fail("A frozen resume must not rescore/refit"),
    )
    assert freeze.run(**kwargs) == selected


def test_freezer_keeps_previous_model_on_exact_tie(tmp_path, monkeypatch):
    """No unsupported geometry claim is manufactured when the baseline wins."""
    kwargs = _setup(
        tmp_path,
        monkeypatch,
        {"guarded_single": 0.2, "guarded_mixed": 0.2, "previous_model": 0.2},
    )
    selected = freeze.run(**kwargs)
    assert selected["selected_experiment"] == "previous_model"
    assert selected["deployment_support"] == ["previous_model"]


def test_freezer_detects_changed_candidate_after_freeze(tmp_path, monkeypatch):
    """Frozen selections cannot survive changed model bytes."""
    kwargs = _setup(
        tmp_path,
        monkeypatch,
        {"guarded_single": 0.5, "guarded_mixed": 0.2, "previous_model": 1.0},
    )
    freeze.run(**kwargs)
    (kwargs["single_run"] / "model/field_model.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Candidate weights"):
        freeze.verify(kwargs["output_root"] / "selection.json")


def test_freezer_refuses_selection_after_evaluation_started(tmp_path, monkeypatch):
    """A post-evaluation path cannot generate a new selected checkpoint."""
    kwargs = _setup(
        tmp_path,
        monkeypatch,
        {"guarded_single": 0.5, "guarded_mixed": 0.2, "previous_model": 1.0},
    )
    (kwargs["output_root"] / "evaluation_started.json").write_text("{}")
    monkeypatch.setattr(
        freeze.evaluate_validation,
        "run",
        lambda *args: pytest.fail("Must reject before any scoring"),
    )
    with pytest.raises(ValueError, match="after blind evaluation"):
        freeze.run(**kwargs)


def test_freezer_rejects_nonempty_test_split(tmp_path, monkeypatch):
    """The selection stage never accepts a configuration with test targets."""
    kwargs = _setup(
        tmp_path,
        monkeypatch,
        {"guarded_single": 0.5, "guarded_mixed": 0.2, "previous_model": 1.0},
    )
    kwargs["validation_config"].write_text(json.dumps({"test_seeds": [2201]}))
    with pytest.raises(ValueError, match="empty test split"):
        freeze.run(**kwargs)


def test_freezer_rejects_old_scoring_wrapper_report(tmp_path, monkeypatch):
    """Matching model/config bytes do not excuse a different validation wrapper."""
    kwargs = _setup(
        tmp_path,
        monkeypatch,
        {"guarded_single": 0.5, "guarded_mixed": 0.2, "previous_model": 1.0},
    )
    report_path = tmp_path / "baseline.json"
    report = freeze.evaluate_validation.run(
        kwargs["previous_model"], kwargs["validation_config"], report_path
    )
    report["evaluation_script_sha256"] = "old_wrapper"
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Validation report does not match"):
        freeze.run(**kwargs, baseline_validation=report_path)
    assert not (kwargs["output_root"] / "selection.json").exists()
