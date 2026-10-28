"""Tests of deliberate training-set fitting without held-out leakage."""
import copy
import json

import numpy as np
import pytest
import torch

from research.es_pc_emulator import fit_ceiling as ceiling
from research.es_pc_emulator.small_pilot_training import (
    field_arrays,
    predict_checkpoint,
)
from research.es_pc_emulator.test_physical_pilot_training import make_case


def test_nested_subset_selection_never_uses_heldout_cases():
    """Sort only training records, preserving a nested one/two-case comparison."""
    records = [{"case_id": name, "split": split} for name, split in
               (("aaa", "test"), ("train_002", "train"), ("train_001", "train"),
                ("000", "validation"), ("external", "external_test"))]
    one = ceiling.select_training_records(records, 1)
    two = ceiling.select_training_records(records, 2)
    assert one == two[:1]
    assert [row["case_id"] for row in two] == ["train_001", "train_002"]
    with pytest.raises(ValueError, match="training cases"):
        ceiling.select_training_records(records, 3)


def test_learning_rate_controls_are_matched_and_reach_declared_endpoints():
    """Only the scheduled arm changes LR, including the final training epoch."""
    constant = [ceiling.learning_rate(i, 20, 3e-4, 3e-6, "constant") for i in range(20)]
    cosine = [ceiling.learning_rate(i, 20, 3e-4, 3e-6, "cosine") for i in range(20)]
    np.testing.assert_allclose(constant, 3e-4, rtol=0, atol=0)
    assert cosine[0] == pytest.approx(3e-4)
    assert cosine[-1] == pytest.approx(3e-6)
    assert np.all(np.diff(cosine) < 0)


class ScalePrior(torch.nn.Module):
    """A field scaling whose power response is known exactly."""

    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, inputs, base):
        """Scale the prior for an analytic power-spectrum response."""
        return self.scale * base, None


def test_active_score_is_raw_and_rejects_undefined_power():
    """The 1% diagnostic cannot pass by flooring or excluding difficult bins."""
    case = make_case(dimension=8)
    case["photons"] = np.cumsum(case["photons"], axis=0)
    case["xhi_es"] = case["xhi_rt"].copy()
    arrays, _ = field_arrays([case], 3, True, "all")
    metrics = ceiling.active_metrics(ScalePrior(1.1), arrays, 50., 1)
    assert metrics["power_fractional_error"]["mean"] == pytest.approx(.21, rel=1e-5)
    arrays["truth"].fill_(.5)
    arrays["brightness"].fill_(27.)
    with pytest.raises(ValueError, match="undefined"):
        ceiling.active_metrics(ScalePrior(1.), arrays, 50., 1)


def test_accuracy_status_distinguishes_mean_from_all_bins():
    """A 1% mean does not silently become a per-bin guarantee."""
    metric = {"mean": .009, "max": .03, "undefined": 0, "count": 10}
    result = ceiling.accuracy_status(metric, .01)
    assert result["mean_target_passed"]
    assert not result["all_bins_target_passed"]
    metric["undefined"] = 1
    assert not ceiling.accuracy_status(metric, .01)["mean_target_passed"]


def test_tiny_fit_reloads_without_labels_and_matches_independent_evaluation(tmp_path):
    """Fit diagnostics use no held-out data and survive checkpoint serialization."""
    case = make_case(seed=7, split="train", case_id="train_000", dimension=8)
    case["photons"] = np.cumsum(case["photons"], axis=0)
    case["sha256"] = "synthetic"
    options = ceiling.default_options() | {"epochs": 2, "diagnostic_interval": 1, "batch_size": 2}
    result = ceiling.fit_case_set([case], "joint_cosine16", 11, tmp_path, options)
    loaded = torch.load(tmp_path / "model.pt", weights_only=True)
    stripped = copy.deepcopy(case)
    for key in ("xhi_rt", "temperature_rt", "xheii_rt", "xheiii_rt"):
        stripped.pop(key)
    prediction = predict_checkpoint(loaded, stripped)
    assert prediction["xhi"].shape == case["xhi_es"].shape
    assert result["scope"] == "training-set fit only; not generalization"
    assert result["best_epoch"] in (0, 1, 2)
    assert result["best_selection_score"] <= result["initial_active_reionization"]["power_fractional_error"]["mean"]
    measured = result["active_reionization"]["power_fractional_error"]["mean"]
    assert measured == pytest.approx(result["best_selection_score"], rel=2e-5, abs=2e-7)
    records = json.loads((tmp_path / "predictions" / "metrics.json").read_text())
    assert {row["case_id"] for row in records} == {"train_000"}
    assert (tmp_path / "last_model.pt").is_file()
    assert ceiling._completed_result(tmp_path, "joint_cosine16", 11, [case])["best_epoch"] == result["best_epoch"]
    with pytest.raises(ValueError, match="identity"):
        ceiling._completed_result(tmp_path, "joint_cosine16", 23, [case])
    with (tmp_path / "model.pt").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checkpoint"):
        ceiling._completed_result(tmp_path, "joint_cosine16", 11, [case])
