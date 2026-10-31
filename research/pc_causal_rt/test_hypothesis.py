"""Native-ES model differences and numerical changes must remain distinct."""
from copy import deepcopy
import json

import numpy as np
import pytest

from research.pc_causal_rt import hypothesis
from research.pc_causal_rt.report import _primary_band
from research.pc_field_emulator.evaluation import shell_spectra


SETTING = dict(directions=96, refinement=1, cfl=.4, time_substeps=4, source_stride=1)


def _pattern():
    x = np.indices((8, 8, 8))[0]
    return .01 * np.cos(x * 2 * np.pi / 8)


def _frames(offset=0.):
    result = {}
    for node, mean in enumerate((.9, .7, .5, .3, .1)):
        rt = mean + _pattern() + offset
        native = mean + .05 + _pattern() * .5
        capacity = np.ones_like(rt)
        ions = capacity * (1 - rt)
        radiation = capacity * .02
        arrays = dict(capacity=capacity, ionized=ions, radiation=radiation,
                      cumulative_source=capacity * (1 - mean + .02),
                      rt_xhi=rt, native_xhi=native, rt_brightness=rt * 20,
                      native_brightness=native * 20)
        spectra = shell_spectra({"teacher": arrays["rt_brightness"]}, 50.)
        point = {"node": node, "redshift": 10. - node, "time_seconds": float(node),
                 "xHI_mass": float(rt.mean()), "spectra": spectra,
                 "primary_power_band": _primary_band(spectra, 8, 50.)}
        result[node] = point, arrays
    return result


def _case(name="coarse", setting=None, seed=222):
    return dict(case_id=name, seed=seed, setting=SETTING if setting is None else setting,
                source_mode="native_cumulative", history_sha256="same-history",
                source_budget_sha256="same-budget", input_roots=["/test"],
                checkpoints=[], native_history_audit={"passed": True})


def _manifest():
    hashes = {"research/pc_causal_rt/" + name: name for name in (
        "workflow.py", "source_budget.py", "native_history.py", "gas.py", "gas_kernel.cpp",
        "transport.py", "transport_kernel.cpp")}
    return {"configuration": {"box_len": 50., "dimension": 8, "photon_energy_ev": 18.,
                              "seeds": [222], "settings": [SETTING], "tolerance": .01},
            "native_runtime": {"native_extension_sha256": "same-native"}, "source_hashes": hashes}


def _loaded(manifest=None, name="coarse", setting=None, seed=222, offset=0.):
    case = _case(name, setting, seed)
    return (_manifest() if manifest is None else manifest, [case], {name: _frames(offset)}, {"root": name})


def test_history_gap_is_inventory_minus_native_budget_deficit():
    point, arrays = _frames()[2]
    row = hypothesis._same_time(point, arrays, 50.)
    assert row["rt_minus_native_es_mass"] == pytest.approx(-.05)
    assert row["radiation_per_atom"] == pytest.approx(.02)
    assert row["emitted_minus_native_ions_per_atom"] == pytest.approx(.07)
    assert abs(row["history_decomposition_residual"]) < 1e-14


def test_identical_model_fields_do_not_define_normalized_numerical_error():
    field = {"xhi": .5 + _pattern(), "brightness": 20 * _pattern()}
    row = hypothesis._change_metrics(field, field, field, 50.)
    assert row["field_MAE_coarse_vs_fine"] == 0
    assert row["field_numerical_change_over_fine_model_difference"] is None
    assert all(x is None for x in row["power_numerical_change_over_fine_model_difference"])


def test_numerical_change_normalized_by_model_difference_has_explicit_denominator():
    pattern = _pattern()
    coarse = {"xhi": .6 + pattern, "brightness": pattern * 2}
    fine = {"xhi": .5 + pattern, "brightness": pattern}
    native = {"xhi": .55 + pattern, "brightness": pattern * .5}
    row = hypothesis._change_metrics(coarse, fine, native, 50.)
    assert row["field_MAE_coarse_vs_fine"] == pytest.approx(.1)
    assert row["field_MAE_fine_vs_native_es"] == pytest.approx(.05)
    assert row["field_numerical_change_over_fine_model_difference"] == pytest.approx(2.)
    assert row["power_numerical_change_over_fine_model_difference"][0] == pytest.approx(4.)


def test_matched_native_es_uses_field_interpolation_before_spectrum():
    pattern = _pattern()
    frames = {}
    for node, (mean, sign) in enumerate(((.8, 1.), (.4, -1.))):
        rt = mean + sign * pattern
        native = np.full_like(rt, mean)
        frames[node] = ({"node": node, "redshift": 10. - node}, {
            "rt_xhi": rt, "rt_brightness": 20 * rt,
            "native_xhi": native, "native_brightness": 20 * native})
    result = hypothesis._matched(frames, 50.)
    matched = next(row for row in result if row["target_xHI_volume"] == .6)
    assert matched["available"]
    assert matched["field_MAE_rt_vs_native_es"] < 1e-14
    assert matched["rt_interpolation"]["later_weight"] == pytest.approx(.5)
    assert max(matched["brightness_spectra"]["P_rt"]) < 1e-20
    assert not next(row for row in result if row["target_xHI_volume"] == .2)["available"]


@pytest.mark.parametrize("changed", ["physics", "native", "implementation"])
def test_changed_physical_inputs_cannot_be_called_refinement(changed):
    original = _manifest()
    altered = deepcopy(original)
    if changed == "physics":
        altered["configuration"]["photon_energy_ev"] = 20.
    elif changed == "native":
        altered["native_runtime"]["native_extension_sha256"] = "other"
    else:
        altered["source_hashes"]["research/pc_causal_rt/workflow.py"] = "other"
    with pytest.raises(ValueError, match="changed physical"):
        hypothesis._merge_inputs([_loaded(original), _loaded(altered, "fine", SETTING | {"refinement": 2})])


def test_different_history_or_budget_rejected():
    a, b = _loaded(), _loaded(name="fine", setting=SETTING | {"refinement": 2})
    b[1][0]["history_sha256"] = "other-history"
    with pytest.raises(ValueError, match="different native history"):
        hypothesis._merge_inputs([a, b])
    b[1][0]["history_sha256"] = "same-history"
    b[1][0]["source_budget_sha256"] = "other-budget"
    with pytest.raises(ValueError, match="different native history"):
        hypothesis._merge_inputs([a, b])


def test_subset_seeds_and_analysis_only_changes_are_allowed():
    altered = _manifest()
    altered["configuration"]["seeds"] = [111, 222]
    altered["source_hashes"]["research/pc_causal_rt/hypothesis.py"] = "new-diagnostic"
    altered["source_hashes"]["research/pc_causal_rt/report.py"] = "new-report"
    _, cases, _ = hypothesis._merge_inputs([_loaded(), _loaded(altered, "seed111", seed=111)])
    assert len(cases) == 2


def test_duplicate_settings_deduplicated_only_for_identical_saved_fields():
    _, cases, _ = hypothesis._merge_inputs([_loaded(), _loaded(name="repeat")])
    assert len(cases) == 1
    with pytest.raises(ValueError, match="different saved fields"):
        hypothesis._merge_inputs([_loaded(), _loaded(name="repeat", offset=.001)])


def test_refinements_change_one_axis_and_compare_adjacent_levels_only():
    cases = [_case("f1"), _case("f2", SETTING | {"refinement": 2}),
             _case("f4", SETTING | {"refinement": 4}), _case("d384", SETTING | {"directions": 384})]
    pairs = {(a["case_id"], b["case_id"], axis) for a, b, axis in hypothesis._adjacent_pairs(cases)}
    assert pairs == {("f1", "f2", "refinement"), ("f2", "f4", "refinement"), ("f1", "d384", "directions")}


def test_one_passing_pair_never_certifies_the_physical_hypothesis(monkeypatch):
    records = [_loaded(), _loaded(name="fine", setting=SETTING | {"refinement": 2})]
    monkeypatch.setattr(hypothesis, "_load_root", lambda root: records[0 if root.name == "a" else 1])
    result = hypothesis.build_hypothesis(["/a", "/b"])
    assert result["refinement_comparisons"][0]["passed"]
    assert not result["hypothesis_verified"]
    assert not result["convergence_certified"]
    assert result["hypothesis_status"] == "tested_refinements_small_full_convergence_unverified"


def test_completion_checksum_tamper_fails_before_analysis(tmp_path):
    (tmp_path / "COMPLETED").write_text("done\n")
    (tmp_path / "manifest.json").write_text(json.dumps(_manifest()))
    (tmp_path / "summary.json").write_text("{}")
    completion = {"identity": _manifest(), "files": {"manifest.json": "bad", "summary.json": "bad"}}
    (tmp_path / "completion.json").write_text(json.dumps(completion))
    with pytest.raises(ValueError, match="checksum"):
        hypothesis._load_root(tmp_path)


@pytest.mark.parametrize("changed", ["native_runtime", "source_hashes", "configuration", "setting"])
def test_resealed_receipt_cannot_import_a_different_execution_identity(tmp_path, changed):
    manifest = _manifest()
    manifest["configuration"]["source_mode"] = "native_cumulative"
    case = _case()
    identity = {"configuration": manifest["configuration"], "native_runtime": manifest["native_runtime"],
                "source_hashes": manifest["source_hashes"],
                **{key: case[key] for key in ("seed", "setting", "history_sha256", "source_budget_sha256", "source_mode")}}
    identity = deepcopy(identity)
    identity[changed]["changed"] = True
    folder = tmp_path / "cases" / case["case_id"]
    folder.mkdir(parents=True)
    (folder / "case.json").write_text(json.dumps(case))
    receipt = {"identity": identity, "record": case,
               "files": {"case.json": hypothesis._digest(folder / "case.json")}}
    (folder / "receipt.json").write_text(json.dumps(receipt))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "summary.json").write_text(json.dumps({"configuration": manifest["configuration"], "cases": [case]}))
    completion = {"identity": manifest, "files": {
        str(path.relative_to(tmp_path)): hypothesis._digest(path)
        for path in tmp_path.rglob("*") if path.is_file()}}
    (tmp_path / "completion.json").write_text(json.dumps(completion))
    (tmp_path / "COMPLETED").write_text("done\n")
    with pytest.raises(ValueError, match="receipt identity"):
        hypothesis._load_root(tmp_path)


def test_analysis_output_cannot_mutate_completed_input(tmp_path):
    with pytest.raises(ValueError, match="separate"):
        hypothesis.write_hypothesis([tmp_path], tmp_path / "new_report")
