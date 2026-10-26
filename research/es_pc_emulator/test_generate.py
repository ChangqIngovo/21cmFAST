"""Check split isolation, orchestration failures and immutable preparation."""

from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.es_pc_emulator import generate


@pytest.mark.parametrize(
    "updates",
    [
        {"splits": {"train": [1], "validation": [1], "test": [3]}},
        {"splits": {"train": [1], "validation": [], "test": [3]}},
        {"splits": {"train": [True], "validation": [2], "test": [3]}},
        {"teacher": {"history": "/preexisting/history.npz"}},
        {"teacher": {"transport": "native"}},
        {"teacher": {"stride": 2}},
        {"schema_version": 2},
        {"unexpected": 1},
    ],
)
def test_invalid_dataset_definition_fails(updates: dict) -> None:
    """Reject leakage and source schedules that cannot pair native endpoints."""
    with pytest.raises((ValueError, TypeError)):
        generate.validate_config(updates)


def test_default_config_is_explicit_and_does_not_mutate() -> None:
    """The teacher numerical configuration and all split seeds are pinned."""
    original = copy.deepcopy(generate.DEFAULT_CONFIG)
    config = generate.validate_config({})
    assert config["teacher"]["angular_order"] == 32
    assert config["teacher"]["refinement"] == 3
    assert config["teacher"]["thermal"] is True
    assert "seed" not in config["teacher"]
    assert "benchmark" not in config["teacher"]
    assert original == generate.DEFAULT_CONFIG


def _orchestration(monkeypatch, tmp_path, *, smoke_passes=True, pair_fails=False):
    """Replace expensive kernels while exercising the real dataset coordinator."""
    from research.c2ray_teacher import runner
    from research.pc_evolution import native_history

    calls = []

    def history(**options):
        calls.append(("history", options["seed"]))
        options["output"].write_text("history")
        options["es_output"].write_text("native ES")

    def teacher(config, output, upstream, workers):
        calls.append(("teacher", config["seed"], config["benchmark"], workers))
        assert Path(config["history"]).is_file()
        output.mkdir()
        (output / "COMPLETED").touch()
        return {"benchmarks": {"passed": smoke_passes}}

    def pairs(history_path, es_path, teacher_results, output, split, expected_seed=None):
        calls.append(("pairs", expected_seed, split))
        if pair_fails:
            raise ValueError("Rejected teacher source hash")
        assert history_path.is_file()
        assert es_path.is_file()
        assert (teacher_results / "COMPLETED").is_file()
        output.mkdir()
        manifest = {"samples": [{"seed": expected_seed, "split": split}]}
        generate.write_json(output / "manifest.json", manifest)
        return manifest

    def combine(manifests, output_path):
        samples = []
        for path in manifests:
            samples.extend(json.loads(Path(path).read_text())["samples"])
        manifest = {"samples": samples}
        generate.write_json(output_path, manifest)
        return manifest

    fake_pairs = SimpleNamespace(
        export_pairs=pairs,
        combine_manifests=combine,
        load_dataset=lambda path: json.loads(Path(path).read_text()),
        load_pair=lambda record: record,
    )
    monkeypatch.setitem(sys.modules, "research.es_pc_emulator.pairs", fake_pairs)
    monkeypatch.setattr(native_history, "build_native_history", history)
    monkeypatch.setattr(runner, "run", teacher)
    monkeypatch.setattr(generate, "generation_identity", lambda config: {"config": config, "runtime": "test"})
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    return calls, upstream


def test_dataset_runs_one_actual_smoke_gate_and_seed_disjoint_cases(monkeypatch, tmp_path):
    """Only the first teacher performs the mandatory compiled smoke workflow."""
    calls, upstream = _orchestration(monkeypatch, tmp_path)
    output = tmp_path / "dataset"
    result = generate.run_dataset({}, output, upstream, workers=4)
    teachers = [call for call in calls if call[0] == "teacher"]
    assert teachers == [
        ("teacher", 1234, True, 4), ("teacher", 2345, False, 4),
        ("teacher", 3456, False, 4), ("teacher", 4567, False, 4),
    ]
    assert result["samples"] == 4
    assert (output / "COMPLETED").is_file()
    assert {row["split"] for row in result["cases"]} == {"train", "validation", "test"}
    with pytest.raises(FileExistsError):
        generate.run_dataset({}, output, upstream, workers=4)


@pytest.mark.parametrize(("smoke_passes", "pair_fails"), [(False, False), (True, True)])
def test_failed_native_smoke_or_pair_audit_never_marks_complete(
    monkeypatch, tmp_path, smoke_passes, pair_fails
):
    """Incomplete or unaudited teacher output cannot become training data."""
    calls, upstream = _orchestration(
        monkeypatch, tmp_path, smoke_passes=smoke_passes, pair_fails=pair_fails
    )
    output = tmp_path / "failed_dataset"
    with pytest.raises((ValueError, RuntimeError)):
        generate.run_dataset({}, output, upstream, workers=4)
    assert (output / "FAILED.json").is_file()
    assert not (output / "COMPLETED").exists()
    assert not (output / "dataset_manifest.json").exists()
    assert len([call for call in calls if call[0] == "teacher"]) == 1


def test_interrupted_seed_resume_reuses_verified_success_and_preserves_attempt(monkeypatch, tmp_path):
    """A later failure must not require regenerating an earlier successful seed."""
    from research.c2ray_teacher import runner

    calls, upstream = _orchestration(monkeypatch, tmp_path)
    real_teacher = runner.run

    def interrupted(config, output, source, workers):
        if config["seed"] == 2345:
            output.mkdir()
            (output / "partial_output.bin").write_bytes(b"interrupted native output")
            raise RuntimeError("Simulated walltime termination")
        return real_teacher(config, output, source, workers)

    monkeypatch.setattr(runner, "run", interrupted)
    output = tmp_path / "continuation"
    with pytest.raises(RuntimeError, match="walltime"):
        generate.run_dataset({}, output, upstream, workers=4)
    first_receipt = (output / "seed_1234/COMPLETED.json").read_bytes()
    assert not (output / "COMPLETED").exists()
    monkeypatch.setattr(runner, "run", real_teacher)
    summary = generate.run_dataset({}, output, upstream, workers=4, resume=True)
    assert (output / "seed_1234/COMPLETED.json").read_bytes() == first_receipt
    assert summary["cases"][0]["reused"] is True
    assert (output / "seed_2345/attempt_0001/teacher/partial_output.bin").read_bytes() == b"interrupted native output"
    assert (output / "seed_2345/attempt_0002/pairs/manifest.json").is_file()
    assert [call for call in calls if call[0] == "teacher" and call[1] == 1234] == [
        ("teacher", 1234, True, 4)
    ]
    assert (output / "COMPLETED").is_file()


@pytest.mark.parametrize("mutation", ["output", "runtime", "configuration"])
def test_resume_rejects_completed_seed_tampering_or_changed_identity(monkeypatch, tmp_path, mutation):
    """Hash-validated continuation never silently mixes incompatible simulations."""
    from research.c2ray_teacher import runner

    calls, upstream = _orchestration(monkeypatch, tmp_path)
    real_teacher = runner.run

    def interrupted(config, output, source, workers):
        if config["seed"] == 2345:
            raise RuntimeError("Simulated termination")
        return real_teacher(config, output, source, workers)

    monkeypatch.setattr(runner, "run", interrupted)
    output = tmp_path / "tampered"
    with pytest.raises(RuntimeError):
        generate.run_dataset({}, output, upstream, workers=4)
    config = {}
    if mutation == "output":
        (output / "seed_1234/attempt_0001/native_history.npz").write_bytes(b"changed history")
    elif mutation == "runtime":
        monkeypatch.setattr(generate, "generation_identity", lambda config: {"config": config, "runtime": "changed"})
    else:
        config = {"teacher": {"angular_order": 64}}
    monkeypatch.setattr(runner, "run", real_teacher)
    before = len(calls)
    with pytest.raises(ValueError, match=r"changed|configuration"):
        generate.run_dataset(config, output, upstream, workers=4, resume=True)
    assert len(calls) == before
    assert not (output / "COMPLETED").exists()


def test_prepared_verification_checks_config_allocation_and_runtime(monkeypatch, tmp_path):
    """A prepared immutable job cannot silently change models or executables."""
    (tmp_path / "source").mkdir()
    config = generate.validate_config({})
    runtime = {"native": "unchanged"}
    manifest = {
        "schema_version": 1, "experiment": "es-to-c2ray-training-dataset",
        "run_directory": str(tmp_path), "source_checkout": str(tmp_path / "source"),
        "requested_workers": 4, "config": config, "runtime": runtime, "commit": "pinned",
    }
    generate.write_json(tmp_path / "run_manifest.json", manifest)
    generate.write_json(tmp_path / "dataset_config.json", config)
    monkeypatch.setattr(generate, "runtime_fingerprint", lambda: runtime)
    git_calls = []
    monkeypatch.setattr(generate.subprocess, "run", lambda *a, **k: git_calls.append(a))
    assert generate.verify_prepared(tmp_path, 4) == manifest
    assert git_calls
    with pytest.raises(RuntimeError, match="allocation"):
        generate.verify_prepared(tmp_path, 8)
    monkeypatch.setattr(generate, "runtime_fingerprint", lambda: {"native": "changed"})
    with pytest.raises(RuntimeError, match=r"runtime|interpreter"):
        generate.verify_prepared(tmp_path, 4)
    config["teacher"]["angular_order"] = 64
    generate.write_json(tmp_path / "dataset_config.json", config)
    with pytest.raises(RuntimeError, match="configuration"):
        generate.verify_prepared(tmp_path, 4)


def test_coordinator_and_real_binary_pair_export_interoperate(monkeypatch, tmp_path):
    """Exercise real pair audits and conservative restriction through all splits."""
    from research.c2ray_teacher import runner
    from research.es_pc_emulator.pairs import load_dataset, load_pair
    from research.es_pc_emulator.test_pairs import pair_fixture
    from research.pc_evolution import native_history

    fixtures = {}

    def history(**options):
        fixture = pair_fixture(options["output"].parent / "fixture", seed=options["seed"])
        fixtures[options["seed"]] = fixture
        shutil.copyfile(fixture["history"], options["output"])
        shutil.copyfile(fixture["es"], options["es_output"])

    def teacher(config, output, upstream, workers):
        fixture = fixtures[config["seed"]]
        shutil.copytree(fixture["teacher"], output)
        return {"benchmarks": {"passed": True}}

    monkeypatch.setattr(native_history, "build_native_history", history)
    monkeypatch.setattr(runner, "run", teacher)
    monkeypatch.setattr(generate, "generation_identity", lambda config: {"config": config, "runtime": "binary fixture"})
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    output = tmp_path / "dataset"
    result = generate.run_dataset({}, output, upstream, workers=1)
    manifest = load_dataset(result["dataset_manifest"])
    assert len(manifest["samples"]) == 8
    assert {row["split"] for row in manifest["samples"]} == {"train", "validation", "test"}
    assert all(load_pair(row)["xhi_c2ray"].shape == (2, 2, 2) for row in manifest["samples"])
    # A completed rerun still audits all native artifacts and pair checksums.
    repeated = generate.run_dataset({}, output, upstream, workers=1, resume=True)
    assert all(row["reused"] for row in repeated["cases"])
