"""Test scale selection and missing-power handling in benchmark summaries."""

import numpy as np
import pytest

from research.pc_causal_rt.benchmarks.multiscale_report import (
    _annotate_pairs,
    _attach_resource_profile,
    _band_summary,
    _geometry,
    _matched_agreement,
    _model_agreement,
    _sampling_comparison,
    _shared_active_nodes,
    _subsample_frames,
    _validate_reference,
)


def test_undefined_bins_are_not_replaced_with_zero():
    """A missing high-k spectrum remains missing in the band summary."""
    row = {
        "power_fractional_error": [0.02, None, 0.1],
        "spectra": {"k_cMpc_inv": [0.1, 0.2, 0.4], "modes": [18, 62, 98]},
    }
    bands = _band_summary(row, 0.5)
    assert bands["below_half_nyquist"]["undefined_bins"] == 1
    assert bands["below_half_nyquist"]["maximum_fractional_change"] is None
    assert bands["below_half_nyquist"]["maximum_defined_fractional_change"] == 0.02
    assert bands["above_half_nyquist"]["maximum_fractional_change"] == 0.1


def test_geometry_uses_native_output_grid():
    """Increasing only allocation refinement does not raise output Nyquist."""
    geometry = _geometry(16, 100.0)
    np.testing.assert_allclose(geometry["fundamental_cMpc_inv"], 2 * np.pi / 100)
    np.testing.assert_allclose(geometry["nyquist_cMpc_inv"], np.pi * 16 / 100)
    assert geometry["native_cell_cMpc"] == 6.25


def test_reference_hash_mismatch_is_rejected():
    """A similar-looking SCRIPT report cannot substitute a different RT run."""
    identity = {
        "rt_manifest_sha256": "m",
        "rt_summary_sha256": "s",
        "rt_completion_sha256": "c",
    }
    _validate_reference(identity, {"manifest": "m", "summary": "s", "completion": "c"})
    with pytest.raises(ValueError, match="summary"):
        _validate_reference(
            identity, {"manifest": "m", "summary": "other", "completion": "c"}
        )


def test_model_agreement_uses_script_power_and_cross_correlation():
    """A scaled brightness wave has the expected power ratio and unit r(k)."""
    wave = np.broadcast_to(np.arange(4.0)[:, None, None], (4, 4, 4)).copy()
    rt = {"xhi": np.full_like(wave, 0.4), "brightness": 2 * wave}
    script = {"xhi": np.full_like(wave, 0.6), "brightness": wave}
    result = _model_agreement(rt, script, 50.0)
    np.testing.assert_allclose(result["field_MAE"], 0.2, rtol=0, atol=1e-15)
    np.testing.assert_allclose(result["P_RT_over_SCRIPT"][0], 4, rtol=0, atol=1e-14)
    np.testing.assert_allclose(result["r_RT_SCRIPT"][0], 1, rtol=0, atol=1e-14)
    zero = {"xhi": np.zeros_like(wave), "brightness": np.zeros_like(wave)}
    absent = _model_agreement(zero, zero, 50.0)
    assert all(x is None for x in absent["P_RT_over_SCRIPT"])
    assert all(x is None for x in absent["r_RT_SCRIPT"])
    tiny = {"xhi": script["xhi"], "brightness": wave * 1e-18}
    below_roundoff = _model_agreement(tiny, tiny, 50.0)
    assert all(x is None for x in below_roundoff["P_RT_over_SCRIPT"])
    zero_numerator = _model_agreement(zero, script, 50.0)
    assert zero_numerator["P_RT_over_SCRIPT"][0] == 0.0
    assert zero_numerator["r_RT_SCRIPT"][0] is None
    opposite = {"xhi": script["xhi"], "brightness": -wave}
    anticorrelation = _model_agreement(opposite, script, 50.0)
    np.testing.assert_allclose(
        anticorrelation["r_RT_SCRIPT"][0], -1, rtol=0, atol=1e-14
    )


def test_matched_model_comparison_keeps_interpolation_brackets():
    """Match each history independently and retain its original time bracket."""
    wave = np.broadcast_to(np.arange(4.0)[:, None, None], (4, 4, 4)).copy()
    frames = [
        (
            {"node": node, "redshift": z},
            {"xhi": np.full_like(wave, mean), "brightness": wave * mean},
        )
        for node, z, mean in ((0, 9.0, 0.7), (1, 8.0, 0.3))
    ]
    result = _matched_agreement(frames, frames, 50.0)
    assert not result[0]["available"]
    row = next(r for r in result if r["target_xHI_volume"] == 0.4)
    assert row["RT_interpolation"]["nodes"] == [0, 1]
    assert row["SCRIPT_interpolation"] == row["RT_interpolation"]
    np.testing.assert_allclose(row["field_MAE"], 0, rtol=0, atol=1e-15)


def test_unequal_refinement_ratios_and_common_active_epochs():
    """Do not equate f2→3 with a factor-two step or mix different active epochs."""
    rt = {
        "coarse_case": "f2",
        "fine_case": "f3",
        "coarse_setting": {"refinement": 2},
        "fine_setting": {"refinement": 3},
        "same_time_active": [
            {"node": 4, "redshift": 8.0},
            {"node": 5, "redshift": 7.5},
        ],
        "matched_mean_xHI": [],
    }
    script = {
        "coarse_refinement": 2,
        "fine_refinement": 3,
        "same_time_active": [
            {"node": 5, "redshift": 7.5},
            {"node": 6, "redshift": 7.0},
        ],
        "matched_mean_xHI": [],
    }
    annotated = _annotate_pairs([rt], np.pi * 16 / 100)[0]
    assert annotated["refinement_ratio"] == 1.5
    np.testing.assert_allclose(
        annotated["coarse_cell_width_cMpc"], 100 / 32, rtol=0, atol=1e-14
    )
    np.testing.assert_allclose(
        annotated["fine_cell_width_cMpc"], 100 / 48, rtol=0, atol=1e-14
    )
    common = _shared_active_nodes([rt], [script])
    assert common[0]["nodes"] == [5]
    assert common[0]["redshifts"] == [7.5]


def test_resource_profiles_match_both_script_input_and_output():
    """Keep RT and SCRIPT resources separate and reject a different output run."""
    dataset = {
        "identity": {"rt_root": "/rt_a", "script_root": "/script_a"},
        "timing": {},
    }
    rt = {"data": {"command": ["python", "--root", "/rt_a"]}}
    script = {
        "data": {"command": ["python", "--rt-root", "/rt_a", "--output", "/script_a"]}
    }
    other_output = {
        "data": {"command": ["python", "--rt-root", "/rt_a", "--output", "/script_b"]}
    }
    other_input = {
        "data": {"command": ["python", "--rt-root", "/rt_b", "--output", "/script_a"]}
    }
    for record in (rt, script, other_output, other_input):
        _attach_resource_profile([dataset], record)
    assert dataset["timing"]["workflow_resource_profiles"] == [rt]
    assert dataset["timing"]["script_workflow_resource_profiles"] == [script]


def test_sampling_diagnostic_retains_endpoints_and_interpolates_fields():
    """Coarsen saved output only, retaining endpoints and taking powers afterward."""
    wave = np.broadcast_to(np.arange(4.0)[:, None, None], (4, 4, 4)).copy()
    frames = [
        (
            {"node": node, "redshift": 9.0 - node / 8},
            {"xhi": np.full_like(wave, mean), "brightness": wave * mean**2},
        )
        for node, mean in enumerate((1.0, 0.8, 0.5, 0.2, 0.0))
    ]
    sparse = _subsample_frames(frames[:4])
    assert [row["node"] for row, _ in sparse] == [0, 2, 3]
    result = _sampling_comparison(frames, 50.0, np.pi * 4 / 50)
    row = next(r for r in result["matched_mean_xHI"] if r["target_xHI_volume"] == 0.4)
    assert row["full_interpolation"]["nodes"] == [2, 3]
    assert row["sparse_interpolation"]["nodes"] == [2, 4]
    np.testing.assert_allclose(row["field_MAE"], 0, rtol=0, atol=1e-14)
    np.testing.assert_allclose(
        row["power_fractional_error"][0], (0.2 / 0.18) ** 2 - 1, rtol=0, atol=1e-14
    )
    truncated = _sampling_comparison(frames[:3], 50.0, np.pi * 4 / 50)
    unavailable = next(
        r for r in truncated["matched_mean_xHI"] if r["target_xHI_volume"] == 0.2
    )
    assert not unavailable["available"]
