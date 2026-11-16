"""Check paired training stays separated from frozen blind evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.pc_analytic_emulator.completion_20261116.reproduction import (
    train_and_validate as workflow,
)


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _model(path: Path, value: str) -> None:
    _write(path / "model.json", {"schema": 2})
    (path / "field_model.npz").write_text(value)
    (path / "budget_model.npz").write_text("budget")


def _setup_workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    campaign, previous, output = (
        tmp_path / "campaign",
        tmp_path / "old",
        tmp_path / "new",
    )
    _model(previous, "old")
    for name in ("training_collection", *workflow.EVALUATIONS):
        folder = campaign / name
        folder.mkdir(parents=True)
        (folder / "data.bin").write_text(name)
    calls = {"train": [], "benchmark": [], "audit": []}

    def dataset(root: str | Path) -> SimpleNamespace:
        root = Path(root)
        if root.name != "training_collection":
            assert (output / "selection.json").is_file()
            assert (output / "evaluation_started.json").is_file()
        calls["audit"].append(root.name)
        seeds = workflow.EVALUATIONS.get(root.name, list(range(2101, 2117)))
        return SimpleNamespace(
            file_hashes={str(root / "data.bin"): workflow.digest(root / "data.bin")},
            samples=[SimpleNamespace(seed=seed) for seed in seeds],
        )

    def train(config: dict, destination: str | Path) -> dict:
        assert config["test_seeds"] == []
        assert config["external_test_root"] is None
        assert config["allow_empty_test"]
        assert not (output / "evaluation_started.json").exists()
        assert not any(name != "training_collection" for name in calls["audit"])
        calls["train"].append(config)
        destination = Path(destination)
        _model(destination / "model", config["feature_set"])
        summary = {
            "complete": True,
            "selected_validation_objective": 0.4
            if config["feature_set"] == "legacy"
            else 0.2,
        }
        _write(destination / "summary.json", summary)
        files = {
            str(path.relative_to(destination)): workflow.digest(path)
            for path in destination.rglob("*")
            if path.is_file()
        }
        _write(destination / "completion.json", {"files": files})
        (destination / "COMPLETED").write_text("done")
        return summary

    def benchmark(
        teacher: str | Path,
        model: str | Path,
        reference: str | Path,
        destination: str | Path,
        *,
        seeds: list[int],
        rays: int,
    ) -> dict:
        assert len(calls["train"]) == 2
        assert (output / "selection.json").is_file()
        assert (output / "evaluation_started.json").is_file()
        assert (
            workflow._model_hashes(Path(model))
            == workflow._json(output / "selection.json")["model_files_sha256"]
        )
        calls["benchmark"].append(Path(teacher).name)
        destination = Path(destination)
        destination.mkdir()
        protocol = workflow._json(output / "protocol.json")
        _write(
            destination / "manifest.json",
            {
                **protocol["benchmark_implementation"],
                "teacher_root": str(teacher),
                "requested_seeds": seeds,
                "rays_per_phase": rays,
                "models": {
                    "selected": {
                        "root": str(model),
                        "files_sha256": workflow._model_hashes(Path(model)),
                    },
                    "previous": {
                        "root": str(reference),
                        "files_sha256": workflow._model_hashes(Path(reference)),
                    },
                },
                "fitting_performed": False,
                "selection_performed": False,
            },
        )
        _write(
            destination / "inputs.json",
            {
                "files_sha256": workflow._json(output / "evaluation_inputs.json")[
                    Path(teacher).name
                ]["files_sha256"]
            },
        )
        report = {
            "complete": True,
            "snapshot_count": len(seeds),
            "model_weights_unchanged": True,
            "rays": rays,
        }
        _write(destination / "report.json", report)
        _write(
            destination / "receipt.json",
            {
                "complete": True,
                "files_sha256": {
                    name: workflow.digest(destination / name)
                    for name in ("report.json", "manifest.json", "inputs.json")
                },
            },
        )
        (destination / "COMPLETED").write_text("done")
        return report

    monkeypatch.setattr(workflow, "load_dataset", dataset)
    monkeypatch.setattr(workflow.train, "run", train)
    monkeypatch.setattr(workflow.benchmark, "run", benchmark)
    monkeypatch.setattr(workflow, "implementation_hashes", lambda: {"mock": "frozen"})
    monkeypatch.setattr(
        workflow,
        "_score_previous",
        lambda *args: {
            "validation_objective": 0.6,
            "snapshots": [],
            "validation_seeds": [2113, 2114, 2115, 2116],
        },
    )
    return SimpleNamespace(
        campaign=campaign, previous=previous, output=output, calls=calls
    )


def test_freeze_before_blind_and_audited_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both validation-only fits precede every blind read; resume never reselects."""
    case = _setup_workflow(tmp_path, monkeypatch)
    result = workflow.run(case.campaign, case.previous, case.output, rays=32)
    assert result["complete"]
    selection = workflow._json(case.output / "selection.json")
    assert selection["selected_experiment"] == "local_features"
    assert selection["selected_validation_objective"] == 0.2
    assert case.calls["benchmark"] == list(workflow.EVALUATIONS)
    assert len(case.calls["train"]) == 2
    assert all(
        (case.output / "configs" / f"{name}.json").exists()
        for name in ("legacy_features", "local_features")
    )
    again = workflow.run(case.campaign, case.previous, case.output, rays=32)
    assert again == result
    assert len(case.calls["train"]) == 2
    assert len(case.calls["benchmark"]) == 3


@pytest.mark.parametrize(
    "target", ["training", "previous", "selected", "selection", "protocol", "blind"]
)
def test_resume_rejects_frozen_identity_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    """No input/model/selection change may silently create a different experiment."""
    case = _setup_workflow(tmp_path, monkeypatch)
    workflow.run(case.campaign, case.previous, case.output, rays=32)
    path = {
        "training": case.campaign / "training_collection/data.bin",
        "previous": case.previous / "field_model.npz",
        "selected": case.output / "model/field_model.npz",
        "selection": case.output / "selection.json",
        "protocol": case.output / "protocol.json",
        "blind": case.campaign / "teacher_blind/data.bin",
    }[target]
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match=r"changed|checksum|Frozen"):
        workflow.run(case.campaign, case.previous, case.output, rays=32)
    assert len(case.calls["train"]) == 2


def test_failed_benchmark_resumes_without_retraining(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Partial diagnostics are preserved, while chosen weights stay frozen."""
    case = _setup_workflow(tmp_path, monkeypatch)
    original = workflow.benchmark.run
    failed = False

    def once(*args: object, **kwargs: object) -> dict:
        nonlocal failed
        if not failed:
            failed = True
            Path(args[3]).mkdir()
            (Path(args[3]) / "partial.log").write_text("interrupted")
            raise RuntimeError("interrupted benchmark")
        return original(*args, **kwargs)

    monkeypatch.setattr(workflow.benchmark, "run", once)
    with pytest.raises(RuntimeError, match="interrupted"):
        workflow.run(case.campaign, case.previous, case.output, rays=32)
    frozen = workflow.digest(case.output / "selection.json")
    assert workflow.run(case.campaign, case.previous, case.output, rays=32)["complete"]
    assert workflow.digest(case.output / "selection.json") == frozen
    assert len(case.calls["train"]) == 2
    assert len(list((case.output / "benchmarks").glob("teacher_blind.partial.*"))) == 1


def test_changed_resources_do_not_resume_same_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resource changes require a new frozen experiment identity."""
    case = _setup_workflow(tmp_path, monkeypatch)
    workflow.run(case.campaign, case.previous, case.output, rays=32)
    with pytest.raises(ValueError, match="Frozen"):
        workflow.run(
            case.campaign, case.previous, case.output, workers=1, threads=1, rays=32
        )


def test_embedded_scientific_configs_match_campaign_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two predeclared feature sets share exactly the requested development split."""
    case = _setup_workflow(tmp_path, monkeypatch)
    configs = workflow.configurations(case.campaign, workers=2, threads=2)
    first, second = configs["legacy_features"], configs["local_features"]
    assert first["train_seeds"] == list(range(2101, 2113)) + list(range(2121, 2129))
    assert first["validation_seeds"] == list(range(2113, 2117))
    assert first["budget_epochs"] == 1600
    assert first["spectral_power_floor"] == 1e-6
    assert first["feature_set"] == "legacy"
    assert second["feature_set"] == "multiscale_local"
    assert {k: v for k, v in first.items() if k != "feature_set"} == {
        k: v for k, v in second.items() if k != "feature_set"
    }


def test_training_input_mutation_aborts_before_any_blind_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both fits must use the same input bytes frozen before the first fit."""
    case = _setup_workflow(tmp_path, monkeypatch)
    original = workflow.train.run

    def mutate(*args: object, **kwargs: object) -> dict:
        result = original(*args, **kwargs)
        (case.campaign / "training_collection/data.bin").write_text("changed")
        return result

    monkeypatch.setattr(workflow.train, "run", mutate)
    with pytest.raises(ValueError, match="changed"):
        workflow.run(case.campaign, case.previous, case.output, rays=32)
    assert not case.calls["benchmark"]
    assert all(name == "training_collection" for name in case.calls["audit"])


def test_benchmark_input_mutation_prevents_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Successful computation cannot certify an archive changed during use."""
    case = _setup_workflow(tmp_path, monkeypatch)
    original = workflow.benchmark.run

    def mutate(*args: object, **kwargs: object) -> dict:
        result = original(*args, **kwargs)
        (case.campaign / "teacher_blind/data.bin").write_text("changed")
        return result

    monkeypatch.setattr(workflow.benchmark, "run", mutate)
    with pytest.raises(ValueError, match="changed"):
        workflow.run(case.campaign, case.previous, case.output, rays=32)
    assert not (case.output / "COMPLETED").exists()


@pytest.mark.parametrize("score", [0.1, 0.2])
def test_previous_model_wins_when_better_or_tied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, score: float
) -> None:
    """The unchanged old model is eligible under the same validation objective."""
    case = _setup_workflow(tmp_path, monkeypatch)
    monkeypatch.setattr(
        workflow,
        "_score_previous",
        lambda *args: {
            "validation_objective": score,
            "snapshots": [],
            "validation_seeds": [2113, 2114, 2115, 2116],
        },
    )
    result = workflow.run(case.campaign, case.previous, case.output, rays=32)
    assert result["selected_experiment"] == "previous_model"
    selected = workflow._json(case.output / "selection.json")
    assert selected["validation_reference"]["validation_objective"] == score
    assert workflow._model_hashes(case.output / "model") == workflow._model_hashes(
        case.previous
    )
    assert workflow.run(case.campaign, case.previous, case.output, rays=32) == result
    assert len(case.calls["train"]) == 2


def test_self_consistent_foreign_benchmark_is_not_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An internally valid receipt must also match this frozen protocol."""
    case = _setup_workflow(tmp_path, monkeypatch)
    workflow.run(case.campaign, case.previous, case.output, rays=32)
    folder = case.output / "benchmarks/teacher_blind"
    record = workflow._json(folder / "manifest.json")
    record["models"]["selected"]["files_sha256"]["field_model.npz"] = "foreign"
    _write(folder / "manifest.json", record)
    receipt = workflow._json(folder / "receipt.json")
    receipt["files_sha256"]["manifest.json"] = workflow.digest(folder / "manifest.json")
    _write(folder / "receipt.json", receipt)
    with pytest.raises(ValueError, match="Benchmark identity"):
        workflow.run(case.campaign, case.previous, case.output, rays=32)


def test_config_mutation_during_training_prevents_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A modified config cannot be newly sealed as the original settings."""
    case = _setup_workflow(tmp_path, monkeypatch)
    original = workflow.train.run

    def mutate(*args: object, **kwargs: object) -> dict:
        result = original(*args, **kwargs)
        path = case.output / "configs/legacy_features.json"
        record = workflow._json(path)
        record["epochs"] += 1
        _write(path, record)
        return result

    monkeypatch.setattr(workflow.train, "run", mutate)
    with pytest.raises(ValueError, match="Frozen"):
        workflow.run(case.campaign, case.previous, case.output, rays=32)
    assert not (case.output / "COMPLETED").exists()


@pytest.mark.parametrize("cap", [0, 512, 1024])
def test_frame_cache_cap_is_frozen_per_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cap: int
) -> None:
    """Execution cache capacity propagates to both fits and immutable protocol."""
    case = _setup_workflow(tmp_path, monkeypatch)
    workflow.run(
        case.campaign, case.previous, case.output, rays=32, frame_cache_mib=cap
    )
    assert all(config["frame_cache_MiB"] == cap for config in case.calls["train"])
    protocol = workflow._json(case.output / "protocol.json")
    assert all(
        config["frame_cache_MiB"] == cap
        for config in protocol["configurations"].values()
    )
    with pytest.raises(ValueError, match="Frozen"):
        workflow.run(
            case.campaign, case.previous, case.output, rays=32, frame_cache_mib=cap + 1
        )


@pytest.mark.parametrize("cap", [-1, 1.5, True])
def test_frame_cache_cap_requires_nonnegative_integer(
    tmp_path: Path, cap: object
) -> None:
    """Invalid cache controls fail before allocating output or opening data."""
    with pytest.raises(ValueError, match="frame_cache"):
        workflow.configurations(tmp_path, workers=1, threads=1, frame_cache_mib=cap)


@pytest.mark.parametrize("during_evaluation", [False, True])
def test_validation_reference_checksum_enforced_on_seal_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, during_evaluation: bool
) -> None:
    """The unchanged baseline's scored evidence cannot be altered after selection."""
    case = _setup_workflow(tmp_path, monkeypatch)
    original = workflow.benchmark.run

    def change_reference(*args: object, **kwargs: object) -> dict:
        result = original(*args, **kwargs)
        path = case.output / "validation_reference.json"
        path.write_text(path.read_text() + " ")
        return result

    if during_evaluation:
        monkeypatch.setattr(workflow.benchmark, "run", change_reference)
    else:
        workflow.run(case.campaign, case.previous, case.output, rays=32)
        path = case.output / "validation_reference.json"
        path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="validation"):
        workflow.run(case.campaign, case.previous, case.output, rays=32)
    if during_evaluation:
        assert not (case.output / "COMPLETED").exists()
