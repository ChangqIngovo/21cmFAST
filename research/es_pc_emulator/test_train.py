"""Exercise fitting, selection and inference independently of RT execution."""

import hashlib
import json

import numpy as np
import pytest

from research.es_pc_emulator import train
from research.pc_field_emulator.model import MLPModel


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


def test_ablations_sample_identical_cells(synthetic_dataset, tmp_path):
    """Input ablations share exactly the same sampled training/validation cells."""
    manifest, _, _ = synthetic_dataset
    output = tmp_path / "ablations"
    train.run(manifest, output, model_seeds=(11,), epochs=2, hidden=(4,),
              patch_radius=0, radii_cMpc=(), samples_per_snapshot=16)
    inventories = [json.loads((output / name / "sampling.json").read_text())
                   for name in ("es", "es-density", "es-density-source")]
    assert inventories[0] == inventories[1] == inventories[2]


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
