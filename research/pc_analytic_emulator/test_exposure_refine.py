"""Check exact deployed descriptors, cohort isolation and portable refit output."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from research.pc_analytic_emulator import exposure_refine as refine
from research.pc_analytic_emulator.exposure_emulator import (
    ExposureEmulator,
    predict_fields,
)
from research.pc_analytic_emulator.exposure_fit import _load
from research.pc_analytic_emulator.exposure_onset import OnsetHead
from research.pc_analytic_emulator.predict import _digest


class _SerialPool:
    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def map(self, function, tasks):
        return map(function, tasks)


def _write_model(path: Path) -> ExposureEmulator:
    path.mkdir()
    np.savez(
        path / "weights.npz",
        field=np.full((1, 10), 0.002),
        budget=np.array([[1.0, 0.0, 0.0, 0.0, 0.0]]),
    )
    onset = OnsetHead(
        roots=np.array([0], dtype=np.int32),
        feature=np.array([0], dtype=np.int32),
        threshold=np.zeros(1),
        left=np.zeros(1, dtype=np.int32),
        right=np.zeros(1, dtype=np.int32),
        leaves=np.ones(1, dtype=bool),
        value=np.array([-0.08]),
        intercept=0.0,
        metadata={"training_seeds": [33]},
    )
    onset.save(path / "onset.npz")
    metadata = {
        "schema": 1,
        "method": "learned_coeval_exposure",
        "depth": 2,
        "opacity_prefactor": 1.73,
        "weights_sha256": _digest(path / "weights.npz"),
        "onset_sha256": _digest(path / "onset.npz"),
        "training_geometries": [{"DIM": 96, "HII_DIM": 16, "BOX_LEN": 50.0}],
        "split_seeds": {"train": [11], "validation": [22]},
        "teacher_signature": {"test_identity": "unchanged"},
        "teacher_physics": {"test_physics": "unchanged"},
    }
    (path / "model.json").write_text(json.dumps(metadata))
    return ExposureEmulator.load(path)


def _native(path: Path, seed: int, split: str) -> dict:
    rng = np.random.default_rng(seed)
    atoms = rng.uniform(0.6, 1.4, (16, 16, 16))
    source = rng.lognormal(-1.1, 1.0, atoms.shape)
    es = np.clip(1 - source / atoms, 0, 1)
    teacher = np.clip(es * 0.99, 0, 1)
    np.savez(path, source=source, capacity=atoms, native_es_xHI=es, teacher_xHI=teacher)
    return {
        "seed": seed,
        "split": split,
        "dim": 96,
        "redshift": 7.5,
        "box_len": 50.0,
        "native_archive": path.name,
        "native_sha256": _digest(path),
    }


def _setup(tmp_path: Path, monkeypatch):
    model_path = tmp_path / "model"
    model = _write_model(model_path)
    rows = [
        _native(tmp_path / "train.npz", 11, "train"),
        _native(tmp_path / "validation.npz", 44, "validation"),
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": 1, "records": rows}))
    monkeypatch.setattr(refine, "ProcessPoolExecutor", _SerialPool)
    return model_path, model, manifest, rows


def test_prepared_features_reproduce_deployed_predictions(tmp_path, monkeypatch):
    """Prepared features reproduce deployed predictions."""
    model_path, model, manifest, _ = _setup(tmp_path, monkeypatch)
    cache = tmp_path / "cache"
    refine.prepare(model_path, manifest, cache)
    data = json.loads((cache / "cache.json").read_text())
    assert data["depth"] == model.metadata["depth"]
    assert data["onset_sha256"] == model.metadata["onset_sha256"]
    for row in data["records"]:
        frame = _load(cache, row)
        with np.load(tmp_path / row["native_archive"]) as saved:
            expected = model.predict(
                saved["source"],
                saved["capacity"],
                row["redshift"],
                row["box_len"],
                native_es_xHI=saved["native_es_xHI"],
            )
        prepared = predict_fields(
            frame.inputs,
            frame.source,
            frame.atoms,
            model.field_weights[0],
            model.budget_weights[0],
        )
        np.testing.assert_allclose(prepared["xHI"], expected["xHI"], rtol=0, atol=2e-15)
        np.testing.assert_allclose(
            prepared["absorption_efficiency"],
            expected["absorption_efficiency"],
            rtol=0,
            atol=2e-15,
        )


def test_input_checksum_change_is_rejected(tmp_path, monkeypatch):
    """Input checksum change is rejected."""
    model_path, _, _, rows = _setup(tmp_path, monkeypatch)
    with (tmp_path / rows[0]["native_archive"]).open("ab") as stream:
        stream.write(b"changed")
    output = tmp_path / "cache"
    output.mkdir()
    with pytest.raises(ValueError, match="archive changed"):
        refine._prepare_one((model_path, tmp_path, rows[0], output))
    assert not list(output.iterdir())


@pytest.mark.parametrize(
    "case",
    [
        "overlap",
        "base_train",
        "base_validation",
        "onset_train",
        "test",
        "duplicate",
        "filename_collision",
    ],
)
def test_prepare_rejects_cohort_leakage_and_ambiguous_records(
    tmp_path, monkeypatch, case
):
    """Prepare rejects cohort leakage and ambiguous records."""
    model_path, _, manifest, rows = _setup(tmp_path, monkeypatch)
    if case == "overlap":
        rows[1]["seed"] = rows[0]["seed"]
        rows[1]["redshift"] = 7.0
    elif case == "base_train":
        rows[0]["seed"], rows[1]["seed"] = 55, 11
    elif case == "base_validation":
        rows[0]["seed"] = 22
    elif case == "onset_train":
        rows[1]["seed"] = 33
    elif case == "test":
        rows[1]["split"] = "test"
    elif case == "duplicate":
        rows.append(dict(rows[0]))
    else:
        rows[0]["redshift"] = 7.5000001
        rows.append({**rows[0], "redshift": 7.5000002})
    manifest.write_text(json.dumps({"schema": 1, "records": rows}))
    with pytest.raises(ValueError, match=r"disjoint|frozen-base|Only train|Duplicate"):
        refine.prepare(model_path, manifest, tmp_path / "cache")
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize(
    "change",
    [
        "base_validation_train",
        "cache_bytes",
        "base_hash",
        "depth",
        "onset",
        "parent_grid",
        "seed_index",
    ],
)
def test_fit_revalidates_frozen_cache(tmp_path, monkeypatch, change):
    """Fit revalidates frozen cache."""
    model_path, _, manifest, _ = _setup(tmp_path, monkeypatch)
    cache = tmp_path / "cache"
    refine.prepare(model_path, manifest, cache)
    config_path = cache / "cache.json"
    config = json.loads(config_path.read_text())
    if change == "base_validation_train":
        config["records"][0]["seed"] = 22
        config["seeds"]["train"] = [22]
    elif change == "cache_bytes":
        with (cache / config["records"][0]["cache"]).open("ab") as stream:
            stream.write(b"changed")
    elif change == "base_hash":
        config["base_model_sha256"] = "wrong"
    elif change == "depth":
        config["depth"] += 1
    elif change == "onset":
        config["onset_sha256"] = "wrong"
    elif change == "parent_grid":
        config["records"][0]["dim"] = 192
    else:
        config["seeds"]["validation"] = [55]
    config_path.write_text(json.dumps(config))
    with pytest.raises(
        ValueError, match=r"frozen-base|checksum|deployed base|geometry|seed index"
    ):
        refine.fit(model_path, cache, tmp_path / "output", maxiter=1)
    assert not (tmp_path / "output" / "model.json").exists()


def test_fit_preserves_onset_physics_and_base_files(tmp_path, monkeypatch):
    """Fit preserves onset physics and base files."""
    model_path, model, manifest, _ = _setup(tmp_path, monkeypatch)
    original = {p.name: p.read_bytes() for p in model_path.iterdir()}
    cache = tmp_path / "cache"
    refine.prepare(model_path, manifest, cache)
    calls = []

    def fake_fit(frames, initial, **kwargs):
        calls.extend(frame.record["split"] for frame in frames)
        return initial.copy(), {"test_fit": True}

    monkeypatch.setattr(refine, "fit_projected_field", fake_fit)
    output = tmp_path / "output"
    report = refine.fit(model_path, cache, output, maxiter=1)
    restored = ExposureEmulator.load(output)
    assert calls
    assert set(calls) == {"train"}
    assert restored.metadata["depth"] == model.metadata["depth"]
    assert restored.metadata["field_fit_depth"] == model.metadata["depth"]
    assert restored.metadata["teacher_signature"] == model.metadata["teacher_signature"]
    assert restored.metadata["teacher_physics"] == model.metadata["teacher_physics"]
    assert (output / "onset.npz").read_bytes() == original["onset.npz"]
    assert original == {p.name: p.read_bytes() for p in model_path.iterdir()}
    assert restored.metadata["split_seeds"]["validation"] == [22, 44]
    for row in report["geometries"]:
        selected = next(c for c in row["candidates"] if c["name"] == row["selected"])
        assert tuple(selected["score"]) == min(
            tuple(c["score"]) for c in row["candidates"]
        )
    with pytest.raises(FileExistsError):
        refine.fit(model_path, cache, output, maxiter=1)


def test_score_enforces_one_percent_goal_without_weak_signal_ratios():
    """Score enforces one percent goal without weak signal ratios."""
    active = {
        "teacher_xHI_mass": 0.5,
        "MAE": 0.001,
        "delta_Q_mass": 0.0001,
        "mean_relative_power_error": 0.012,
        "brightness_residual_rms_over_neutral_rms": 0.001,
    }
    assert refine.score([active])[0] == 1.2
    assert refine.score([{**active, "mean_relative_power_error": 0.008}])[0] == 0.8
    late = {**active, "teacher_xHI_mass": 0.001, "mean_relative_power_error": 10.0}
    assert refine.score([late])[0] == 0.1
    with pytest.raises(ValueError, match="nonempty"):
        refine.score([])
    with pytest.raises(ValueError, match="finite"):
        refine.score([{**active, "MAE": float("nan")}])
