"""Audit summary selection and spectral metrics with explicit software fixtures."""

import copy
import json

import numpy as np
import pytest

from research.es_pc_emulator import physical_pilot_results as results


def split_fixture(split, nn_error):
    """Create measured-spectrum-shaped software fixtures, never physics evidence."""
    records = []
    for case_index in range(2):
        for snapshot, redshift in enumerate((15.0, *results.PRIMARY_REDSHIFTS)):
            # The early snapshot and high-k bin deliberately have enormous errors.
            error = (20.0 if snapshot == 0 else nn_error * (case_index + 1))
            power = np.array([2.0, 4.0, 8.0])
            ratios = np.array([1 + error, 1 + 2 * error, 100.0])
            es_ratios = np.array([1.5, 2.0, 100.0])
            records.append({
                "case_id": f"{split}_{case_index}", "split": split,
                "seed": 111 if split == "test" else 222 + case_index,
                "snapshot": snapshot, "redshift": redshift,
                "primary_snapshot": snapshot > 0,
                "nn": {"MAE": .01 * (case_index + 1)}, "es": {"MAE": .1},
                "spectra": {"k_cMpc_inv": [.16, .28, .4], "k_max_cMpc_inv": .5,
                            "P_teacher": power.tolist(),
                            "P_nn": (power * ratios).tolist(), "ratio_nn": ratios.tolist(),
                            "P_es": (power * es_ratios).tolist(), "ratio_es": es_ratios.tolist()},
            })
    return {"cases": 2, "snapshots": len(records), "samples": records}


def training_fixture():
    """Select all-inputs by validation even though ES-only has lower test error."""
    selections = [{"arm": arm, "validation_mse": 0.0 if arm == "all" else float(index + 1),
                   "test_used_for_selection": False}
                  for index, arm in enumerate(results.ARM_LABELS)]
    return {
        "selected": next(row for row in selections if row["arm"] == "all"),
        "selections": selections, "test_used_for_selection": False,
        "arms": {arm: {split: split_fixture(split, .01 * (index + 1))
                       for split in results.SPLITS}
                 for index, arm in enumerate(results.ARM_LABELS)},
        "objective": "unweighted voxel ionized-fraction MSE; PS evaluated separately",
        "brightness_convention": "saturated Ts, no RSD, shared declared density",
        "primary_test_seed_disjoint": False,
        "seeds_by_split": {"train": [111], "validation": [111],
                           "test": [111], "external_test": [222, 223]},
        "source_manifest_sha256": "software-fixture-only",
    }


def test_last_three_snapshots_and_two_target_bins_are_the_only_primary_errors():
    """Early-time/high-k outliers cannot contaminate the declared diagnostic."""
    measured, _ = results.summarize_split(split_fixture("test", .1), "test")
    assert measured["histories"] == 2
    np.testing.assert_allclose(measured["redshifts"], results.PRIMARY_REDSHIFTS)
    np.testing.assert_allclose(measured["k_cMpc_inv"], [.16, .28])
    expected_cells = np.array([.1, .2] * 3 + [.2, .4] * 3)
    distribution = measured["nn"]["power_error_cells"]
    assert distribution["count"] == 12
    assert distribution["mean"] == pytest.approx(expected_cells.mean())
    assert distribution["p95"] == pytest.approx(np.quantile(expected_cells, .95))
    assert distribution["maximum"] == pytest.approx(.4)
    per_history = measured["nn"]["per_history_mean_power_error"]
    assert per_history["mean"] == pytest.approx(.225)
    assert per_history["p25"] == pytest.approx(.1875)
    assert per_history["p75"] == pytest.approx(.2625)
    assert measured["nn"]["field_MAE_snapshots"]["mean"] == pytest.approx(.015)


def test_validation_winner_is_not_replaced_by_posthoc_test_winner():
    """Report both answers without selecting a feature set using test data."""
    summary = results.summarize(training_fixture())
    assert summary["selected_by_validation"]["arm"] == "all"
    assert summary["lowest_test_mean_power_error_arm_posthoc_only"] == {
        "test": "es", "external_test": "es"}
    assert summary["strict_photon_conservation"] is False
    assert summary["selection_used_test"] is False


@pytest.mark.parametrize("mutation", ["selected", "baseline", "seed", "snapshot", "ratio"])
def test_inconsistent_science_records_fail_closed(mutation):
    """Reject switched selection, targets, ICs, snapshots, or derived ratios."""
    fixture = training_fixture()
    row = fixture["arms"]["all"]["test"]["samples"][1]
    if mutation == "selected":
        fixture["selected"] = fixture["selections"][0]
    elif mutation == "baseline":
        row["spectra"]["P_es"][0] *= 2
        row["spectra"]["ratio_es"][0] *= 2
    elif mutation == "seed":
        row["seed"] = 999
    elif mutation == "snapshot":
        row["primary_snapshot"] = False
    else:
        row["spectra"]["ratio_nn"][0] += .1
    message = {"selected": "validation-MSE winner", "baseline": "different held-out targets",
               "seed": "one seed", "snapshot": "Primary snapshot flags",
               "ratio": "power ratio"}[mutation]
    with pytest.raises(ValueError, match=message):
        results.summarize(fixture)


def test_undefined_reference_power_is_counted_not_hidden_as_success():
    """A missing fractional error remains undefined, including aggregate counts."""
    fixture = split_fixture("test", .1)
    row = fixture["samples"][1]
    row["spectra"]["P_teacher"][0] = 0.0
    row["spectra"]["ratio_nn"][0] = None
    row["spectra"]["ratio_es"][0] = None
    measured, _ = results.summarize_split(fixture, "test")
    for name in ("nn", "es"):
        assert measured[name]["power_error_cells"]["undefined_count"] == 1
        assert measured[name]["power_error_cells"]["finite_count"] == 11
        assert measured[name]["per_history"][0]["power_absolute_relative_error"][0][0] is None
    json.dumps(measured, allow_nan=False)


def test_analysis_checks_per_arm_provenance_and_does_not_overwrite(tmp_path):
    """Compact records must refer to the exact independently saved evaluations."""
    fixture = training_fixture()
    root = tmp_path / "training"
    root.mkdir()
    (root / "COMPLETED").touch()
    (root / "summary.json").write_text(json.dumps(fixture))
    for arm, reports in fixture["arms"].items():
        for split, report in reports.items():
            directory = root / arm / split
            directory.mkdir(parents=True)
            (directory / "summary.json").write_text(json.dumps(report))
    summary = results.analyze(root, tmp_path / "analysis")
    assert len(summary["source_report_sha256"]) == 21
    assert (tmp_path / "analysis/input_ablation.png").stat().st_size > 10000
    assert (tmp_path / "analysis/input_ablation.pdf").stat().st_size > 1000
    with pytest.raises(FileExistsError):
        results.analyze(root, tmp_path / "analysis")
    changed = copy.deepcopy(fixture["arms"]["es"]["test"])
    changed["cases"] = 999
    (root / "es/test/summary.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Per-arm"):
        results.analyze(root, tmp_path / "different")
