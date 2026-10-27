"""Exercise fitting, selection and inference independently of RT execution."""

import hashlib
import json

import numpy as np
import pytest

from research.es_pc_emulator import train
from research.pc_field_emulator.model import (
    MLPModel,
    _residual_fraction,
    _residual_slope,
)
from research.pc_field_emulator.model import train as fit_network


@pytest.fixture
def synthetic_dataset(tmp_path, monkeypatch):  # noqa: PLR0206
    """Provide independent toy seeds without standing in for native RT tests."""
    samples, fields = [], {}
    for seed, split in ((1, "train"), (2, "validation"), (3, "test")):
        rng = np.random.default_rng(seed)
        xhi = 0.25 + 0.5 * rng.random((4, 4, 4))
        metadata = {"seed": seed, "split": split, "redshift": 8.0,
                    "box_len_cMpc": 50.0, "dimension": 4}
        record = metadata | {"sha256": hashlib.sha256(str(seed).encode()).hexdigest(),
                             "path": str(tmp_path / f"seed{seed}.npz")}
        samples.append(record)
        fields[record["sha256"]] = {"xhi_es": xhi, "radius_es_cMpc": 5 + xhi,
            "density": np.ones_like(xhi), "photons": np.ones_like(xhi),
            "xhi_c2ray": xhi - 0.1, "metadata": metadata}
    manifest = tmp_path / "dataset.json"
    manifest.write_text(json.dumps({"samples": samples}))
    monkeypatch.setattr(train, "load_dataset", lambda path: {"samples": samples, "convergence_certified": True})
    monkeypatch.setattr(train, "load_pair", lambda record: fields[record["sha256"]])
    return manifest, samples, fields


def test_residual_training_improves_heldout_correction_and_roundtrips(synthetic_dataset, tmp_path):
    """A fitted correction generalizes to a held-out seed and omits labels at inference."""
    manifest, samples, fields = synthetic_dataset
    output = tmp_path / "trained"
    summary = train.run(manifest, output, feature_sets=("es",), model_seeds=(11,),
                        epochs=120, hidden=(8,), batch_size=32,
                        patch_radius=0, radii_cMpc=(), samples_per_snapshot=64)
    assert (output / "COMPLETED").is_file()
    assert summary["arms"]["es"]["mean_nn_MAE"] < 0.7 * summary["arms"]["es"]["mean_es_MAE"]
    selection = json.loads((output / "model_selection.json").read_text())
    assert selection["test_used_for_selection"] is False
    model = MLPModel.load(output / "model.npz")
    pair = fields[samples[-1]["sha256"]].copy()
    expected = train.predict_pair(model, pair)
    pair.pop("xhi_c2ray")
    actual = train.predict_pair(model, pair)
    np.testing.assert_array_equal(actual, expected)
    assert np.all((actual >= 0) & (actual <= 1))
    assert summary["strict_photon_conservation"] is False
    accuracy = json.loads((output / "accuracy_report.json").read_text())
    assert accuracy["test_used_for_selection"] is False
    assert accuracy["arms"]["es"]["test_MAE_improved"] is True
    assert accuracy["arms"]["es"]["selected_epoch"] == model.metadata["best_epoch"]
    assert accuracy["arms"]["es"]["validation_selected_zero_correction"] is False


def test_ablations_sample_identical_cells(synthetic_dataset, tmp_path):
    """Input ablations share exactly the same sampled training/validation cells."""
    manifest, _, _ = synthetic_dataset
    output = tmp_path / "ablations"
    train.run(manifest, output, model_seeds=(11,), epochs=2, hidden=(4,),
              patch_radius=0, radii_cMpc=(), samples_per_snapshot=16)
    inventories = [json.loads((output / name / "sampling.json").read_text())
                   for name in ("es", "es-density", "es-density-source")]
    assert inventories[0] == inventories[1] == inventories[2]


def test_zero_correction_report_does_not_claim_learning(synthetic_dataset, tmp_path):
    """When ES is already the label, a completed fit must report no gain."""
    manifest, _, fields = synthetic_dataset
    for pair in fields.values():
        pair["xhi_c2ray"] = pair["xhi_es"].copy()
    summary = train.run(manifest, tmp_path / "zero", feature_sets=("es",), model_seeds=(11,),
                        epochs=3, hidden=(4,), patch_radius=0, radii_cMpc=())
    report = summary["accuracy_report"]["arms"]["es"]
    assert report["selected_epoch"] == 0
    assert report["validation_selected_zero_correction"] is True
    assert report["test_MAE_improved"] is False
    assert report["mean_es_MAE"] == report["mean_nn_MAE"] == 0.0
    assert report["MAE_reduction_fraction"] is None


def test_residual_corrects_saturated_es_both_directions():
    """Exact ES endpoints must learn both added and removed ionization.

    This directly tests the zero-initialized output and feasible one-sided
    derivative, independently of any RT target or cosmological assumptions.
    """
    rng = np.random.default_rng(418)
    features = rng.uniform(-1, 1, (768, 4))
    baseline = (features[:, 0] > 0).astype(float)
    target = np.where(baseline == 1, 0.65 + 0.1 * features[:, 1],
                      0.20 + 0.1 * features[:, 1])
    model, _ = fit_network(features[:512], target[:512], features[512:], target[512:],
                          baseline_train=baseline[:512], baseline_validation=baseline[512:],
                          hidden=(16,), epochs=200, batch_size=64)
    prediction = model.predict(features[512:], baseline=baseline[512:])
    assert model.metadata["best_epoch"] > 0
    assert np.all((prediction >= 0) & (prediction <= 1))
    for endpoint in (0, 1):
        mask = baseline[512:] == endpoint
        original = np.mean(abs(baseline[512:][mask] - target[512:][mask]))
        corrected = np.mean(abs(prediction[mask] - target[512:][mask]))
        assert corrected < 0.25 * original


def test_weighted_residual_gradient_direction_at_zero_and_nonzero():
    """Directional finite differences check the actual weighted MSE derivative."""
    baseline = np.array([0., 1., 0.3, 0.7])
    target = np.array([0.2, 0.65, 0.5, 0.4])
    weights = np.array([1., 4., 2., 3.])
    for logits in (np.zeros(4), np.array([0.1, -0.2, 0.3, -0.4])):
        amplitude = np.tanh(logits)
        prediction = _residual_fraction(amplitude, baseline, np.ones(4))
        error = prediction - target
        derivative = 2 * weights * error * _residual_slope(amplitude, baseline, np.ones(4), error) / 4
        direction = -np.sign(derivative)
        step = 1e-7
        trial = _residual_fraction(np.tanh(logits + step * direction), baseline, np.ones(4))
        difference = (np.mean(weights * (trial - target)**2)
                      - np.mean(weights * error**2)) / step
        np.testing.assert_allclose(difference, np.dot(derivative, direction), rtol=2e-6)


def test_channel_scaling_avoids_nearly_constant_context_failure(tmp_path):
    """A tiny within-box scatter must not magnify a modest new-box mean shift.

    The target has the same law in both independent samples. Only the global
    context mean changes, so this isolates feature scaling from RT physics.
    """
    rng = np.random.default_rng(4)
    local_train, local_validation = rng.uniform(0, 1, (2, 1024))
    inputs = np.column_stack((local_train, 0.5 + 1e-6 * local_train))
    validation = np.column_stack((local_validation, 0.55 + 1e-6 * local_validation))
    target, truth = 0.1 + 0.1 * local_train, 0.1 + 0.1 * local_validation
    baseline = np.full(1024, 0.6)
    names = ["xhi_es:offset=(0, 0, 0)", "xhi_es:gaussian_sigma_cMpc=50"]
    floor = train._channel_scale_floor(inputs, names)
    common = {"baseline_train": baseline, "baseline_validation": baseline,
              "epochs": 200, "hidden": (16,), "batch_size": 128, "seed": 11}
    old, _ = fit_network(inputs, target, validation, truth, **common)
    fixed, _ = fit_network(inputs, target, validation, truth, feature_scale_floor=floor, **common)
    old_error = np.mean(abs(old.predict(validation, baseline=baseline) - truth))
    new_error = np.mean(abs(fixed.predict(validation, baseline=baseline) - truth))
    assert old_error > 0.2
    assert new_error < 0.02
    assert new_error < 0.1 * old_error
    np.testing.assert_allclose(fixed.scale[1], inputs[:, 0].std())
    fixed.save(tmp_path / "scaled.npz")
    restored = MLPModel.load(tmp_path / "scaled.npz")
    np.testing.assert_array_equal(restored.predict(validation, baseline=baseline),
                                  fixed.predict(validation, baseline=baseline))


@pytest.mark.parametrize("floor", [np.ones(3), np.array([-1., 0.]), np.array([np.nan, 0.])])
def test_reject_invalid_feature_scale_floor(floor):
    """Bad normalization controls must not silently create a fitted model."""
    inputs = np.ones((4, 2))
    with pytest.raises(ValueError, match="Feature scale floor"):
        fit_network(inputs, np.zeros(4), inputs, np.zeros(4), feature_scale_floor=floor)


def test_reject_existing_output_and_failed_fit_does_not_complete(synthetic_dataset, tmp_path, monkeypatch):
    """Failed fitting never publishes successful completion or overwrites prior work."""
    manifest, _, _ = synthetic_dataset
    existing = tmp_path / "exists"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        train.run(manifest, existing)

    def fail(*args, **kwargs):
        raise RuntimeError("Failed fit")

    monkeypatch.setattr(train, "fit_network", fail)
    output = tmp_path / "failure"
    with pytest.raises(RuntimeError, match="Failed fit"):
        train.run(manifest, output, feature_sets=("es",), model_seeds=(1,),
                  epochs=1, patch_radius=0, radii_cMpc=())
    assert not (output / "COMPLETED").exists()


def test_unconverged_teacher_requires_explicit_pilot_opt_in(synthetic_dataset, tmp_path, monkeypatch):
    """Numerically audited labels do not silently become converged training truth."""
    manifest, samples, _ = synthetic_dataset
    monkeypatch.setattr(train, "load_dataset", lambda path: {"samples": samples, "convergence_certified": False})
    with pytest.raises(ValueError, match="convergence is not certified"):
        train.run(manifest, tmp_path / "blocked")
    assert not (tmp_path / "blocked").exists()
    summary = train.run(manifest, tmp_path / "pilot", feature_sets=("es",),
                        model_seeds=(1,), epochs=1, patch_radius=0, radii_cMpc=(),
                        allow_unconverged_teacher=True)
    assert summary["exploratory_teacher_opt_in"] is True
    assert summary["teacher_convergence_certified"] is False


def test_initial_state_mismatch_requires_separate_opt_in(synthetic_dataset, tmp_path, monkeypatch):
    """Late RT resets cannot silently become normal field-correction training."""
    manifest, samples, _ = synthetic_dataset
    monkeypatch.setattr(train, "load_dataset", lambda path: {
        "samples": samples, "convergence_certified": True, "initial_state_compatible": False})
    with pytest.raises(ValueError, match="initial states"):
        train.run(manifest, tmp_path / "rejected", allow_unconverged_teacher=True)
    result = train.run(manifest, tmp_path / "exploratory", feature_sets=("es",), model_seeds=(11,),
                       epochs=1, hidden=(4,), patch_radius=0, radii_cMpc=(),
                       allow_initial_state_mismatch=True)
    assert result["initial_state_status"] == "mismatch"
    assert result["exploratory_initial_state_opt_in"] is True
