"""Coverage gates distinguish a full trajectory from a truncated or mislabeled one."""
import copy
import json

import numpy as np
import pytest

from research.es_pc_emulator import full_history


def complete_case():
    """Create a declared initial state followed by middle and late phases."""
    shape = (3, 2, 2, 2)
    xhi = np.broadcast_to(np.array([1-1e-6, .5, .005])[:, None, None, None], shape).copy()
    case = {"metadata": {"case_id": "train_000", "split": "train", "seed": 11,
                         "initial_snapshot_included": True,
                         "initial_state": {"xHII": 1e-6, "xHeII": 1e-6, "xHeIII": 1e-12,
                                           "temperature_kelvin": 100.}},
            "redshifts": np.array([20., 12., 4.]), "times_seconds": np.array([0., 1., 2.]),
            "density": np.ones((2, 2, 2)), "xhi_rt": xhi, "xhi_es": xhi.copy(),
            "radius_es_cMpc": np.zeros(shape), "photons": np.zeros(shape),
            "emission_time_mean": np.zeros(shape), "temperature_rt": np.full(shape, 100.),
            "xheii_rt": np.full(shape, 1e-6), "xheiii_rt": np.full(shape, 1e-12)}
    case["xhi_es"][0] = 1.
    return case


def test_complete_trajectory_passes_but_truncated_and_false_initial_states_fail():
    """Coverage requires the actual requested endpoints and exact declared initial labels."""
    case = complete_case()
    assert full_history.check_coverage([case], 20., 4., 3)["passed"]
    changed = copy.deepcopy(case)
    changed["redshifts"][-1] = 7.9
    changed["xhi_rt"][-1] = .3
    result = full_history.check_coverage([changed], 20., 4., 3)
    assert not result["passed"]
    assert len(result["cases"][0]["errors"]) == 2
    changed = copy.deepcopy(case)
    changed["temperature_rt"][0] = 200.
    result = full_history.check_coverage([changed], 20., 4., 3)
    assert not result["passed"]
    assert "temperature_rt" in result["cases"][0]["errors"][0]


def test_mass_weighted_neutral_remnant_is_not_hidden_by_volume_average():
    """A small dense neutral region fails even if volume-averaged xHI is below1%."""
    case = complete_case()
    case["density"][0, 0, 0] = 100.
    case["xhi_rt"][-1] = 0
    case["xhi_rt"][-1, 0, 0, 0] = .05
    result = full_history.check_coverage([case], 20., 4., 3)
    assert result["cases"][0]["final_xHI_volume"] < .01
    assert result["cases"][0]["final_xHI_mass"] > .01
    assert not result["passed"]


def test_every_case_needs_a_middle_phase_and_zero_initial_photons():
    """An apparently complete ensemble cannot conceal a skipped middle or bad photon origin."""
    case = complete_case()
    changed = copy.deepcopy(case)
    changed["metadata"]["case_id"] = "train_001"
    changed["xhi_rt"][1] = .005
    changed["photons"][0] = 1
    result = full_history.check_coverage([case, changed], 20., 4., 3)
    assert result["failed_case_ids"] == ["train_001"]
    assert len(result["cases"][1]["errors"]) == 2


def test_failed_gate_writes_report_before_stopping(monkeypatch, tmp_path):
    """A batch failure leaves an actionable per-case coverage record."""
    case = complete_case()
    case["xhi_rt"][-1] = .2
    monkeypatch.setattr(full_history, "load_manifest", lambda path: {"cases": [{}]})
    monkeypatch.setattr(full_history, "load_case", lambda record: case)
    output = tmp_path/"coverage.json"
    with pytest.raises(SystemExit, match="Incomplete evolution coverage"):
        full_history.main(["coverage", "--manifest", "unused", "--output", str(output), "--snapshots", "3"])
    assert json.loads(output.read_text())["failed_case_ids"] == ["train_000"]


def test_nonfinite_clock_cannot_pass_a_monotonicity_check():
    """A monotone elapsed-time array ending at infinity is still invalid."""
    case = complete_case()
    case["times_seconds"][-1] = np.inf
    result = full_history.check_coverage([case], 20., 4., 3)
    assert not result["passed"]
    assert "elapsed time" in result["cases"][0]["errors"][0]
