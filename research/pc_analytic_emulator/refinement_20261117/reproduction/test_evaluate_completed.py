"""Verify post-freeze evaluation ordering and exact resumption identities."""

import json

import pytest

from research.pc_analytic_emulator.refinement_20261117.reproduction import (
    evaluate_completed as evaluation,
)


def _setup(tmp_path, monkeypatch):
    selection = tmp_path / "selection.json"
    selection.write_text("{}")
    selected = {
        "selected_model_root": "selected_model",
        "model_files_sha256": {"field": "selected"},
        "training_protocol_sha256": "frozen",
        "candidates": {
            "previous_model": {
                "model_root": "previous_model",
                "model_files_sha256": {"field": "previous"},
            }
        },
    }
    monkeypatch.setattr(evaluation.freeze_completed, "verify", lambda path: selected)
    monkeypatch.setattr(
        evaluation.benchmark, "_inference_source_hashes", lambda: {"code": "fixed"}
    )
    opened = []
    for name, (_, dimension, box) in evaluation.PROFILES.items():
        folder = tmp_path / f"teacher_{name}"
        folder.mkdir()
        (folder / "identity.json").write_text(
            json.dumps(
                {
                    "configuration": {
                        "dimension": dimension,
                        "box_len": box,
                        "redshifts": [10, 9, 8, 7.5, 7, 6.5, 6],
                    }
                }
            )
        )

    def inputs(path, expected_seeds):
        assert (tmp_path / "evaluation_started.json").is_file()
        assert expected_seeds
        opened.append(path.name)
        file = path / "identity.json"
        return {"root": str(path), "files_sha256": {str(file): evaluation.digest(file)}}

    def benchmark(teacher, model, previous, destination, *, seeds, rays):
        assert len(opened) == 4
        assert len(json.loads((tmp_path / "evaluation_inputs.json").read_text())) == 4
        protocol = json.loads((tmp_path / "evaluation_protocol.json").read_text())
        all_inputs = json.loads((tmp_path / "evaluation_inputs.json").read_text())
        name = destination.name.removeprefix("benchmark_")
        destination.mkdir()
        manifest = {
            "teacher_root": teacher,
            "requested_seeds": seeds,
            "rays_per_phase": rays,
            "models": protocol["models"],
            "benchmark_sha256": protocol["benchmark_sha256"],
            "independent_metrics_sha256": protocol["independent_metrics_sha256"],
            "inference_source_sha256": protocol["inference_source_sha256"],
            "fitting_performed": False,
            "selection_performed": False,
        }
        (destination / "manifest.json").write_text(json.dumps(manifest))
        (destination / "inputs.json").write_text(json.dumps(all_inputs[name]))
        (destination / "report.json").write_text('{"complete": true}')
        (destination / "receipt.json").write_text("{}")
        (destination / "COMPLETED").write_text("done")

    monkeypatch.setattr(evaluation, "_dataset_identity", inputs)
    monkeypatch.setattr(evaluation.benchmark, "run", benchmark)
    monkeypatch.setattr(evaluation, "_receipt", lambda *args: None)
    return selection


def test_evaluation_pins_all_four_inputs_before_first_prediction(tmp_path, monkeypatch):
    """The immutable selection gate and all input hashes precede benchmarking."""
    selection = _setup(tmp_path, monkeypatch)
    result = evaluation.run(tmp_path, selection)
    assert result["complete"]
    assert not result["refitting_performed"]
    gate = json.loads((tmp_path / "evaluation_started.json").read_text())
    assert set(gate) == {
        "training_protocol_sha256",
        "selection_sha256",
        "model_files_sha256",
    }
    monkeypatch.setattr(
        evaluation.benchmark,
        "run",
        lambda *args, **kwargs: pytest.fail(
            "Completed benchmarks must be audited, not rerun"
        ),
    )
    assert evaluation.run(tmp_path, selection) == result


def test_missing_selection_rejects_before_teacher_access(tmp_path, monkeypatch):
    """An absent model freeze cannot be substituted by evaluation settings."""
    monkeypatch.setattr(
        evaluation,
        "_dataset_identity",
        lambda *args, **kwargs: pytest.fail("No teacher access before freeze"),
    )
    with pytest.raises(ValueError, match="existing campaign selection"):
        evaluation.run(tmp_path, tmp_path / "selection.json")


def test_resume_rejects_changed_teacher_bytes(tmp_path, monkeypatch):
    """Changing an input archive after the first benchmark cannot reuse results."""
    selection = _setup(tmp_path, monkeypatch)
    evaluation.run(tmp_path, selection)
    identity = tmp_path / "teacher_blind_original/identity.json"
    identity.write_text(identity.read_text() + "\n")
    with pytest.raises(ValueError, match="Frozen evaluation record changed"):
        evaluation.run(tmp_path, selection)


def test_resume_rejects_wrong_benchmark_model(tmp_path, monkeypatch):
    """A completed receipt alone cannot bind a benchmark to the selected weights."""
    selection = _setup(tmp_path, monkeypatch)
    evaluation.run(tmp_path, selection)
    path = tmp_path / "benchmark_blind_original/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["models"]["selected"]["files_sha256"] = {"field": "changed"}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="different frozen settings or models"):
        evaluation.run(tmp_path, selection)
