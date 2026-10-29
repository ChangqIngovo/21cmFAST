"""Saved-field checks cannot confuse conservation with numerical accuracy."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from research.pc_causal_rt.report import AXES, audit_case, build_report


ANCHOR = {"directions": 384, "refinement": 4, "time_substeps": 4,
          "source_stride": 1, "cfl": .1}


def _case(root, setting=None, *, case_id="example", field_scale=1., node_count=2):
    setting = dict(ANCHOR if setting is None else setting)
    n = 8
    rng = np.random.default_rng(4)
    pattern = rng.uniform(-1, 1, (n, n, n))
    capacity = np.ones_like(pattern)
    case = {"case_id": case_id, "seed": 111, "setting": setting,
            "history_sha256": "same-native-history", "all_ledgers_passed": True, "checkpoints": []}
    for node in range(node_count):
        ions = .4 + .02 * pattern * field_scale if node else np.zeros_like(pattern)
        source = .6 + .02 * pattern if node else np.zeros_like(pattern)
        xhi = 1 - ions
        arrays = {"capacity": capacity, "ionized": ions,
                  "radiation": source - ions, "cumulative_source": source,
                  "rt_xhi": xhi, "rt_brightness": 27 * xhi,
                  "native_xhi": np.full_like(pattern, .7), "native_brightness": np.full_like(pattern, 18.9)}
        path = Path("cases") / case_id / f"node{node}.npz"
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(root / path, **arrays)
        case["checkpoints"].append({"node": node, "redshift": 9. - node,
                                    "time_seconds": float(node), "fields_path": str(path)})
    return case


def _summary(root, cases, *, curved_sources=False, include_history=True):
    histories = []
    if cases and include_history:
        n = 8
        base = .6 + .02 * np.random.default_rng(4).uniform(-1, 1, (n, n, n))
        times = np.linspace(0, 1, 5)
        factors = 1 + times**2 if curved_sources else np.ones(5)
        rates = factors[:, None, None, None] * base
        path = root / "history.npz"
        np.savez(path, times_seconds=times, redshifts=9 - times, rates=rates,
                 packet_mass=np.ones(n**3), native_capacity=np.ones((5, n, n, n)))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        histories = [{"seed": 111, "path": "history.npz", "sha256": digest}]
        for case in cases:
            if case["history_sha256"] == "same-native-history":
                case["history_sha256"] = digest
            for point in case["checkpoints"]:
                point["node"] = 4 if point["time_seconds"] else 0
            if curved_sources and len(case["checkpoints"]) > 1:
                stride = case["setting"]["source_stride"]
                selected = np.arange(0, 5, stride)
                factor = np.trapezoid(factors[selected], times[selected])
                checkpoint = root / case["checkpoints"][1]["fields_path"]
                with np.load(checkpoint) as data:
                    arrays = {k: np.array(data[k]) for k in data.files}
                arrays["cumulative_source"] = base * factor
                arrays["radiation"] = arrays["cumulative_source"] - arrays["ionized"]
                np.savez(checkpoint, **arrays)
    (root / "summary.json").write_text(json.dumps({"configuration": {"box_len": 50., "tolerance": .01},
                                                  "histories": histories, "cases": cases}))


def _star(root):
    cases = [_case(root, case_id="anchor")]
    levels = {"directions": [24, 96], "refinement": [1, 2], "time_substeps": [1, 2],
              "source_stride": [4, 2], "cfl": [.4, .2]}
    for axis in AXES:
        for level in levels[axis]:
            cases.append(_case(root, ANCHOR | {axis: level}, case_id=f"{axis}_{level}"))
    return cases


def _mutate(root, point, name, transform):
    path = root / point["fields_path"]
    with np.load(path) as saved:
        arrays = {k: np.array(saved[k]) for k in saved.files}
    arrays[name] = transform(arrays[name])
    np.savez(path, **arrays)


def test_independent_count_and_state_audit(tmp_path):
    case = _case(tmp_path)
    result = audit_case(tmp_path, case, 50.)
    assert result["all_saved_field_ledgers_passed"]
    assert result["checkpoints"][1]["photon_ledger_relative_residual"] < 1e-14
    assert result["checkpoints"][1]["xHI_mass"] == pytest.approx(.6, abs=.002)


@pytest.mark.parametrize("field", ["ionized", "radiation", "cumulative_source"])
def test_corrupted_saved_budget_rejected(tmp_path, field):
    case = _case(tmp_path)
    _mutate(tmp_path, case["checkpoints"][1], field, lambda a: a + .01)
    with pytest.raises(ValueError, match="ledger|inconsistent"):
        audit_case(tmp_path, case, 50.)


@pytest.mark.parametrize("field", ["rt_xhi", "rt_brightness", "native_xhi", "capacity"])
def test_nonfinite_saved_fields_never_pass(tmp_path, field):
    case = _case(tmp_path)
    _mutate(tmp_path, case["checkpoints"][1], field, lambda a: a * np.nan)
    with pytest.raises(ValueError, match="Invalid saved"):
        audit_case(tmp_path, case, 50.)


def test_atom_inventory_change_rejected(tmp_path):
    case = _case(tmp_path)
    _mutate(tmp_path, case["checkpoints"][0], "capacity", lambda a: a * 1.01)
    with pytest.raises(ValueError, match="ledger"):
        audit_case(tmp_path, case, 50.)


def test_single_setting_cannot_certify_convergence(tmp_path):
    _summary(tmp_path, [_case(tmp_path)])
    result = build_report(tmp_path, make_plot=False)
    assert result["all_saved_field_ledgers_passed"]
    assert not result["convergence_certified"]
    assert all(not axis["two_adjacent_steps_available"] for axis in result["seeds"][0]["axes"].values())


def test_two_successive_steps_every_axis_at_common_anchor(tmp_path):
    _summary(tmp_path, _star(tmp_path))
    result = build_report(tmp_path, make_plot=True)
    assert result["convergence_certified"]
    assert all(axis["adjacent_comparisons"] == 2 for axis in result["seeds"][0]["axes"].values())
    assert (tmp_path / "causal_rt_comparison.png").stat().st_size > 1000
    assert (tmp_path / "causal_rt_report.json").is_file()


def test_power_change_fails_even_with_perfect_photon_ledger(tmp_path):
    cases = _star(tmp_path)
    replaced = next(c for c in cases if c["case_id"] == "directions_96")
    cases[cases.index(replaced)] = _case(tmp_path, replaced["setting"], case_id="directions_96", field_scale=1.2)
    _summary(tmp_path, cases)
    result = build_report(tmp_path, make_plot=False)
    assert result["all_saved_field_ledgers_passed"]
    assert not result["convergence_certified"]
    assert not result["seeds"][0]["axes"]["directions"]["passed"]


def test_no_active_epochs_cannot_certify(tmp_path):
    cases = _star(tmp_path)
    for case in cases:
        case["checkpoints"] = case["checkpoints"][:1]
    _summary(tmp_path, cases)
    result = build_report(tmp_path, make_plot=False)
    assert not result["convergence_certified"]
    assert all(p["active_checkpoints"] == 0 for axis in result["seeds"][0]["axes"].values()
               for p in axis["comparisons"])


def test_zero_active_power_cannot_certify(tmp_path):
    cases = _star(tmp_path)
    for case in cases:
        _mutate(tmp_path, case["checkpoints"][1], "rt_brightness", lambda a: np.full_like(a, 10.))
    _summary(tmp_path, cases)
    result = build_report(tmp_path, make_plot=False)
    assert not result["convergence_certified"]
    assert any(p["undefined_active_power_bins"] for axis in result["seeds"][0]["axes"].values()
               for p in axis["comparisons"])


def test_shared_history_required(tmp_path):
    cases = _star(tmp_path)
    cases[0]["history_sha256"] = "different-initial-conditions"
    _summary(tmp_path, cases)
    with pytest.raises(ValueError, match="same archived native history"):
        build_report(tmp_path, make_plot=False)


def test_shared_cosmic_time_required(tmp_path):
    cases = _star(tmp_path)
    cases[0]["checkpoints"][1]["time_seconds"] = 1.1
    _summary(tmp_path, cases)
    with pytest.raises(ValueError, match="actual cosmic timestamps"):
        build_report(tmp_path, make_plot=False)


def test_source_quadrature_change_is_tested(tmp_path):
    cases = _star(tmp_path)
    _summary(tmp_path, cases, curved_sources=True)
    result = build_report(tmp_path, make_plot=False)
    assert result["all_saved_field_ledgers_passed"]
    assert result["all_native_history_audits_passed"]
    assert not result["seeds"][0]["axes"]["source_stride"]["passed"]


def test_other_refinements_cannot_change_source_budget(tmp_path):
    cases = _star(tmp_path)
    case = next(c for c in cases if c["case_id"] == "directions_24")
    point = case["checkpoints"][1]
    _mutate(tmp_path, point, "cumulative_source", lambda a: a + .02)
    _mutate(tmp_path, point, "radiation", lambda a: a + .02)
    _summary(tmp_path, cases)
    with pytest.raises(ValueError, match="independently integrated native history"):
        build_report(tmp_path, make_plot=False)


def test_empty_report_rejected(tmp_path):
    _summary(tmp_path, [])
    with pytest.raises(ValueError, match="empty run"):
        build_report(tmp_path, make_plot=False)


def test_failed_interval_ledger_blocks_certification(tmp_path):
    cases = _star(tmp_path)
    cases[0]["all_ledgers_passed"] = False
    _summary(tmp_path, cases)
    result = build_report(tmp_path, make_plot=False)
    assert result["all_saved_field_ledgers_passed"]
    assert not result["all_interval_ledgers_passed"]
    assert not result["convergence_certified"]


def test_absent_native_history_prevents_certification(tmp_path):
    _summary(tmp_path, _star(tmp_path), include_history=False)
    result = build_report(tmp_path, make_plot=False)
    assert not result["all_native_history_audits_passed"]
    assert not result["convergence_certified"]


def test_corrupt_native_history_hash_rejected(tmp_path):
    _summary(tmp_path, _star(tmp_path))
    (tmp_path / "history.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="native history hash"):
        build_report(tmp_path, make_plot=False)


def test_checkpoint_receipt_hash_rejected(tmp_path):
    case = _case(tmp_path)
    case["checkpoints"][0]["fields_sha256"] = "not-the-file-hash"
    with pytest.raises(ValueError, match="checkpoint file hash"):
        audit_case(tmp_path, case, 50.)


def test_actual_primary_band_is_explicit(tmp_path):
    _summary(tmp_path, [_case(tmp_path)])
    result = build_report(tmp_path, make_plot=False)
    band = result["actual_primary_power_bands"]["example"]
    assert band["bin_count"] == 1
    assert band["modes_per_bin"] == [18]
    assert band["shell_centres_cMpc_inv"] == pytest.approx([.16036478039875368])
    assert band["maximum_bound_cMpc_inv"] == pytest.approx(np.pi * 8 / (2 * 50))


def test_matched_es_fields_and_spectra_are_retained(tmp_path):
    case = _case(tmp_path)
    for point in case["checkpoints"]:
        path = tmp_path / point["fields_path"]
        with np.load(path) as data:
            arrays = {k: np.array(data[k]) for k in data.files}
        arrays["matched_es_xhi"] = np.full_like(arrays["rt_xhi"], .65)
        arrays["matched_es_brightness"] = arrays["rt_brightness"] * .9
        np.savez(path, **arrays)
    _summary(tmp_path, [case])
    result = build_report(tmp_path, make_plot=True)
    point = result["cases"][0]["checkpoints"][1]
    assert point["matched_es_xHI_volume"] == pytest.approx(.65)
    assert "Delta2_matched_es" in point["spectra"]
    assert "not a causal ES history" in result["matched_es_comparison"]


def test_saved_source_reintegration_is_independent(tmp_path):
    case = _case(tmp_path)
    point = case["checkpoints"][1]
    _mutate(tmp_path, point, "cumulative_source", lambda a: a * 1.1)
    _mutate(tmp_path, point, "radiation", lambda a: a + .1 * (.6 + .02 * np.random.default_rng(4).uniform(-1, 1, a.shape)))
    _summary(tmp_path, [case])
    with pytest.raises(ValueError, match="independently integrated native history"):
        build_report(tmp_path, make_plot=False)
