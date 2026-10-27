"""Test physical-pilot leakage controls and full-case ablation training."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from research.es_pc_emulator import physical_pilot_training as pilot
from research.pc_field_emulator.model import MLPModel


def make_case(seed=1, split="train", case_id="one", dimension=4):
    """Make explicitly synthetic fields for software, not physics, tests."""
    rng = np.random.default_rng(seed)
    xhi = 0.3 + 0.4 * rng.random((2, dimension, dimension, dimension))
    return {
        "metadata": {"case_id": case_id, "seed": seed, "split": split,
                     "box_len_cMpc": 50.0,
                     "parameter_names": ["log10_source_amplitude", "history_tilt",
                                         "source_concentration", "density_contrast_scale",
                                         "log10_Tbb_K", "log10_clumping"],
                     "theta": [0.6, 0.0, 1.0, 1.0, np.log10(50000), 0.0]},
        "redshifts": np.array([9.0, 8.0]),
        "density": 0.5 + rng.random((dimension, dimension, dimension)),
        "xhi_es": xhi,
        "xhi_rt": xhi - 0.12,
        "radius_es_cMpc": 4 + 3 * xhi,
        "photons": 0.1 + rng.random(xhi.shape),
        "emissivity": 0.01 + 0.02 * rng.random(xhi.shape),
        "emission_time_mean": 10 + 50 * rng.random(xhi.shape),
        "temperature_rt": np.full(xhi.shape, 10000.0),
        "xheii_rt": 1 - xhi,
        "xheiii_rt": np.full(xhi.shape, 1e-6),
    }


def write_dataset(tmp_path):
    """Write parameter-holdout and external-seed test partitions."""
    records = []
    for index, (seed, split) in enumerate(((111, "train"), (111, "validation"),
                                          (111, "test"), (222, "external_test"))):
        case = make_case(seed, split, f"case_{index}")
        metadata = case.pop("metadata")
        path = tmp_path / f"case_{index}.npz"
        np.savez_compressed(path, **case, metadata=np.asarray(json.dumps(metadata)))
        records.append({"case_id": metadata["case_id"], "seed": seed, "split": split,
                        "path": path.name, "theta": metadata["theta"]})
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": 1, "cases": records}))
    return path


def test_features_do_not_read_rt_or_redundant_control_labels():
    """RT state and upstream parameter labels are not hidden inputs."""
    case = make_case()
    expected, names = pilot.build_features(case, 0, "all")
    changed = copy.deepcopy(case)
    for key in ("xhi_rt", "temperature_rt", "xheii_rt", "xheiii_rt"):
        changed.pop(key)
    changed["metadata"]["theta"][:4] = [1234, -999, 400, -73]
    actual, actual_names = pilot.build_features(changed, 0, "all")
    np.testing.assert_array_equal(actual, expected)
    assert actual_names == names
    assert not any("source_amplitude" in name or "density_contrast_scale" in name for name in names)


@pytest.mark.parametrize(("arm", "keys", "theta_index"), [
    ("no_density", ("density",), None),
    ("no_sources", ("photons", "emissivity", "emission_time_mean"), None),
    ("no_history", ("emissivity", "emission_time_mean"), None),
    ("no_spectrum", (), 4),
    ("no_clumping", (), 5),
])
def test_removed_group_cannot_leak_through_feature_construction(arm, keys, theta_index):
    """An omitted input group is not read through another feature path."""
    case = make_case()
    expected, names = pilot.build_features(case, 0, arm)
    changed = copy.deepcopy(case)
    for key in keys:
        changed.pop(key)
    if theta_index is not None:
        changed["metadata"]["theta"][theta_index] = -100.0
    actual, actual_names = pilot.build_features(changed, 0, arm)
    np.testing.assert_array_equal(actual, expected)
    assert actual_names == names


def test_manifest_rejects_same_case_in_two_splits(tmp_path):
    """A complete realization cannot be reused as a different split."""
    path = write_dataset(tmp_path)
    data = json.loads(path.read_text())
    data["cases"][1]["path"] = data["cases"][0]["path"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="multiple splits"):
        pilot.load_manifest(path)


def test_case_identity_and_hash_are_verified(tmp_path):
    """A changed case or mislabeled split fails closed."""
    path = write_dataset(tmp_path)
    manifest = pilot.load_manifest(path)
    record = manifest["cases"][0]
    pilot.load_case(record)
    with pytest.raises(ValueError, match="hash"):
        pilot.load_case(record | {"sha256": "wrong"})
    with pytest.raises(ValueError, match="split"):
        pilot.load_case(record | {"split": "test"})
    actual_hash = hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="hash"):
        pilot.load_case(record | {"fields_sha256": "wrong", "sha256": actual_hash})
    declared = record | {"fields_sha256": actual_hash}
    pilot.load_case(declared)
    with Path(record["path"]).open("ab") as stream:
        stream.write(b"tampered after generation")
    with pytest.raises(ValueError, match="hash"):
        pilot.load_case(declared)


def test_scale_floor_guards_nearly_constant_global_context():
    """Global and smoothed fields retain the scale of their physical channel."""
    matrix = np.array([[0.0, 1e-12, 1e-12], [1.0, 2e-12, 2e-12]])
    names = ["rho:offset=(0, 0, 0)", "rho:smooth=25", "rho:global_mean"]
    np.testing.assert_array_equal(pilot._scale_floor(matrix, names), [0, 0.5, 0.5])


def test_small_fit_reports_external_seed_and_prediction_without_labels(tmp_path):
    """A small real fit evaluates unseen seeds with label-free inference."""
    path = write_dataset(tmp_path)
    output = tmp_path / "trained"
    summary = pilot.run(path, output, arms=("es", "all"), model_seeds=(11,),
                        epochs=80, hidden=(8,), samples_per_snapshot=64,
                        patch_radius=0, radii_cMpc=(), batch_size=64)
    assert summary["test_used_for_selection"] is False
    assert summary["primary_test_seed_disjoint"] is False
    assert summary["seeds_by_split"]["external_test"] == [222]
    assert summary["selected"]["arm"] in ("es", "all")
    assert (output / "COMPLETED").is_file()
    assert summary["arms"]["es"]["external_test"]["mean_nn_MAE"] < 0.5 * 0.12
    sampling = [json.loads((output / arm / "sampling.json").read_text()) for arm in ("es", "all")]
    assert sampling[0] == sampling[1]
    model = MLPModel.load(output / "es/model.npz")
    record = pilot.load_manifest(path)["cases"][-1]
    case = pilot.load_case(record)
    expected = pilot.predict(model, case, 0)
    case.pop("xhi_rt")
    actual = pilot.predict(model, case, 0)
    np.testing.assert_array_equal(actual, expected)
    assert np.all((actual >= 0) & (actual <= 1))
    report = summary["arms"]["es"]["external_test"]
    assert report["all_mean_power_error_nn"] is not None
    assert "spectra" in report["samples"][0]


def test_target_spectral_range_excludes_near_nyquist_shells(tmp_path, monkeypatch):
    """Keep raw spectra but limit target accuracy to the declared pilot scales."""
    case = make_case(dimension=8, split="test")
    monkeypatch.setattr(pilot, "predict", lambda model, value, snapshot: 1 - value["xhi_rt"][snapshot])
    report = pilot.evaluate(None, [case], tmp_path / "evaluate")
    row = report["samples"][0]
    assert len(row["spectra"]["k_cMpc_inv"]) == 4
    assert sum(0.1 <= k <= row["target_k_max_cMpc_inv"] for k in row["spectra"]["k_cMpc_inv"]) == 2
    assert row["target_mean_power_error_nn"] == 0.0
    assert report["all_target_mean_power_error_nn"] == 0.0
    assert report["primary_snapshots"] == 2
    assert report["primary_target_max_power_error_nn"] == 0.0


def test_parallel_arms_preserve_models_selection_and_metrics(tmp_path):
    """Scheduling independent fits cannot change science output or selection."""
    path = write_dataset(tmp_path)
    configuration = {"arms": ("es", "all"), "model_seeds": (11,), "epochs": 4,
                     "hidden": (4,), "samples_per_snapshot": 16, "patch_radius": 0,
                     "radii_cMpc": (), "batch_size": 32}
    serial = pilot.run(path, tmp_path / "serial", parallel_arms=1, **configuration)
    parallel = pilot.run(path, tmp_path / "parallel", parallel_arms=2, **configuration)
    assert serial["selected"] == parallel["selected"]
    assert serial["selections"] == parallel["selections"]
    assert serial["arms"] == parallel["arms"]
    for arm in configuration["arms"]:
        first = MLPModel.load(tmp_path / "serial" / arm / "model.npz")
        second = MLPModel.load(tmp_path / "parallel" / arm / "model.npz")
        for left, right in zip(first.weights, second.weights, strict=True):
            np.testing.assert_array_equal(left, right)
