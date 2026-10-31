"""Check interpolation and shared-budget semantics independently of RT."""
import numpy as np
import pytest

from research.pc_causal_rt.script_comparison import _interpolate, _matched, _same_time, _spectra


def _frame(node, z, mean, pattern):
    field = mean + pattern
    return ({"node": node, "redshift": z}, {"rt_xhi": field, "script_xhi": field,
        "rt_brightness": 20 * field, "script_brightness": 20 * field})


def test_matched_fields_are_interpolated_before_nonlinear_power():
    pattern = np.zeros((4, 4, 4))
    pattern[::2] = .1
    pattern[1::2] = -.1
    rt = [_frame(0, 10., .8, pattern), _frame(1, 9., .4, -pattern)]
    script = [_frame(0, 10., .8, np.zeros_like(pattern)), _frame(1, 9., .4, np.zeros_like(pattern))]
    matched = next(row for row in _matched(rt, script, 50.) if row["target_xHI_volume"] == .6)
    assert matched["available"]
    assert matched["field_MAE"] < 1e-14
    assert matched["rt_interpolation"]["later_weight"] == pytest.approx(.5)
    assert max(matched["brightness_spectra"]["P_rt"]) < 1e-20
    # Averaging endpoint powers would leave a nonzero signal, so this tests
    # field interpolation rather than merely following the implementation.
    first = _spectra(rt[0][1]["rt_brightness"], rt[0][1]["rt_brightness"], 50.)
    assert max(first["P_rt"]) > 1.


def test_no_matched_xhi_extrapolation():
    zero = np.zeros((4, 4, 4))
    frames = [_frame(0, 10., .8, zero), _frame(1, 9., .6, zero)]
    assert _interpolate(frames, .2, xhi_name="rt_xhi", brightness_name="rt_brightness") is None
    exact = _interpolate(frames, .8, xhi_name="rt_xhi", brightness_name="rt_brightness")
    assert exact["interpolation"]["nodes"] == [0, 0]
    assert exact["interpolation"]["later_weight"] == 0.


def _same_time_inputs():
    shape = (4, 4, 4)
    capacity = np.ones(shape)
    es = np.full(shape, .7)
    rt = {"capacity": capacity, "cumulative_source": capacity * .4,
          "source_native_current": capacity * .4, "native_xhi": es,
          "native_brightness": es * 20, "rt_xhi": capacity * .65,
          "rt_brightness": capacity * 13, "radiation": capacity * .05}
    script = {"capacity": capacity, "source": capacity * .4, "native_es_xhi": es,
              "native_brightness": es * 20, "script_xhi": capacity * .6,
              "script_brightness": capacity * 12}
    point = {"node": 1, "redshift": 7.}
    return point, rt, point, script


def test_same_budget_can_have_radiation_and_different_ionized_mass():
    args = _same_time_inputs()
    row = _same_time(*args, {"setting": {"source_stride": 1}}, 50.)
    assert row["source_budget_shared_within_native_roundoff"]
    assert row["rt_xHI_mass"] - row["script_xHI_mass"] == pytest.approx(.05)
    assert row["rt_radiation_per_atom"] == pytest.approx(.05)
    assert row["field_MAE_rt_vs_script"] == pytest.approx(.05)
    assert row["brightness_spectra"]["ratio_rt"] == [None, None]


def test_changed_global_budget_rejected_for_finest_source_history():
    args = _same_time_inputs()
    args[1]["cumulative_source"] *= 2
    with pytest.raises(ValueError, match="emitted budget"):
        _same_time(*args, {"setting": {"source_stride": 1}}, 50.)
    sampled = _same_time(*args, {"setting": {"source_stride": 2}}, 50.)
    assert not sampled["source_budget_shared_within_native_roundoff"]


def test_original_native_comparison_field_must_match():
    args = _same_time_inputs()
    args[1]["native_xhi"] = args[1]["native_xhi"] + .01
    with pytest.raises(ValueError, match="original native fields"):
        _same_time(*args, {"setting": {"source_stride": 1}}, 50.)
