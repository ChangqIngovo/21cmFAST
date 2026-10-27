"""Selection isolation and restart checks for the small spectral-loss sweep."""
import json

import pytest

pytest.importorskip("torch")
from research.es_pc_emulator import loss_sweep as sweep


def config():
    """Return the seven-setting scan with minimal common arguments."""
    return {"component_weights": [.003, .01, .03], "power_weights": [.003, .01, .03],
            "anchor_component_weight": .01, **dict.fromkeys(sweep.COMMON_KEYS, 1)}


def test_grid_is_one_factor_scan_and_rejects_ambiguous_weights():
    """Preserve the intended controls and reject invalid weight choices."""
    grid = sweep.sweep_grid(config())
    assert len(grid) == 7
    assert grid[0]["arm"] == "thermo"
    assert all(row["power_weight"] == 0 for row in grid[1:4])
    assert all(row["component_weight"] == .01 for row in grid[4:])
    with pytest.raises(ValueError, match="anchor"):
        sweep.sweep_grid(config() | {"anchor_component_weight": .02})
    with pytest.raises(ValueError, match="unique finite"):
        sweep.sweep_grid(config() | {"power_weights": [float("nan")]})


def test_selection_uses_validation_not_test_or_training():
    """A better held-out result cannot override the validation criterion."""
    trials = [{"id": "a", "mean_validation_log_power_mse": .1, "test_error": 10., "train_error": 10.},
              {"id": "b", "mean_validation_log_power_mse": .2, "test_error": 0., "train_error": 0.}]
    assert sweep.select_trial(trials)["id"] == "a"
    trials[0]["test_error"] = -100
    assert sweep.select_trial(trials)["id"] == "a"
    with pytest.raises(ValueError, match="finite validation"):
        sweep.select_trial([{"id": "bad", "mean_validation_log_power_mse": float("nan")}])


def test_training_command_excludes_heldout_evaluation(tmp_path):
    """The actual training CLI receives only train and validation evaluation splits."""
    spec = sweep.sweep_grid(config())[4]
    command = sweep.training_command(spec, 23, config(), tmp_path/"manifest", tmp_path/"run", 2)
    start = command.index("--evaluation-splits")
    assert command[start+1:start+3] == ["train", "validation"]
    from research.es_pc_emulator.small_pilot_training import parser
    args = parser().parse_args(command[3:])
    assert args.evaluation_splits == ["train", "validation"]
    assert args.model_seeds == [23]
    assert args.power_weight == .003
    assert args.component_weight == .01


def test_resume_skips_completed_and_preserves_partial_fit(monkeypatch, tmp_path):
    """Resume reuses completed fits and starts a separate attempt after interruption."""
    spec = sweep.sweep_grid(config())[0]
    root = tmp_path/"trials"/spec["id"]/"seed11"
    partial = root/"attempt_001"
    partial.mkdir(parents=True)
    (partial/"partial.txt").write_text("keep")
    calls = []

    def run(command, check):
        calls.append(command)
        output = command[command.index("--output")+1]
        folder = sweep.Path(output)
        folder.mkdir()
        (folder/"COMPLETED").touch()

    monkeypatch.setattr(sweep.subprocess, "run", run)
    result = sweep.run_trial(spec, 11, config(), tmp_path/"manifest", tmp_path, 2)
    assert result.name == "attempt_002"
    assert (partial/"partial.txt").read_text() == "keep"
    again = sweep.run_trial(spec, 11, config(), tmp_path/"manifest", tmp_path, 2)
    assert again == result
    assert len(calls) == 1
    assert json.loads((tmp_path/"progress.json").read_text())["seed"] == 11
