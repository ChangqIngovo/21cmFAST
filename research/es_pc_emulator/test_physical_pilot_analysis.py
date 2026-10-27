"""Check that physical responses cannot conceal unmatched controls or signs."""

import copy

import numpy as np
import pytest

from research.es_pc_emulator.physical_pilot_analysis import (
    REFINEMENT_CONFIGS,
    REFINEMENTS,
    summarize,
)


def fixture_cases():
    """Create independently calculable multiplicative power differences."""
    base = {"case_id": "fiducial", "seed": 111, "theta": [0.] * 6,
            "redshifts": [11., 9.5, 8.], "k": [.16, .28, .39, .46],
            "P21": [[10., 20., 30., 40.]] * 3, "xHI_mean": [.9, .7, .5],
            "refinement": 3, "angular_order": 2, "time_substeps": 1}
    cases = {"fiducial": base}
    for j in range(6):
        for side, sign in enumerate((-1, 1)):
            row = copy.deepcopy(base)
            row["case_id"] = f"oat_{j}_{side}"
            row["theta"][j] = float(sign)
            row["P21"] = (np.array(base["P21"]) * (1 + sign * .01 * (j + 1))).tolist()
            row["xHI_mean"] = (np.array(base["xHI_mean"]) + sign * .001 * (j + 1)).tolist()
            cases[row["case_id"]] = row
    for name, delta in zip(REFINEMENTS, (.02, -.005, .03), strict=True):
        row = copy.deepcopy(base)
        row["case_id"] = name
        row.update(REFINEMENT_CONFIGS[name])
        row["P21"] = (np.array(base["P21"]) * (1 + delta)).tolist()
        cases[name] = row
    return cases


def test_control_summary_retains_sign_and_uses_linear_power():
    """An exact 1% power change remains 1%, distinct from log-space errors."""
    result = summarize(fixture_cases())
    assert result["complete"]
    assert result["selected_k_indices"] == [0, 1]
    row = result["physical_controls"][0]
    np.testing.assert_allclose(row["sides"]["low"]["power_relative_signed"], -.01)
    np.testing.assert_allclose(row["sides"]["high"]["power_relative_signed"], .01)
    assert row["power_relative_mean_absolute"] == pytest.approx(.01)
    assert row["response_exceeds_fiducial_refinement_envelope_fraction"] == 0.
    assert result["physical_controls"][-1][
        "response_exceeds_fiducial_refinement_envelope_fraction"] == 1.
    np.testing.assert_allclose(result["fiducial_refinement_envelope"], .03)
    np.testing.assert_allclose(row["sides"]["high"]["xHI_difference_signed"], .001)


@pytest.mark.parametrize("change", ["seed", "other_theta", "resolution", "k"])
def test_unmatched_physical_controls_are_rejected(change):
    """Changing another variable cannot be presented as a one-variable response."""
    cases = fixture_cases()
    row = cases["oat_0_0"]
    if change == "seed":
        row["seed"] = 222
    elif change == "other_theta":
        row["theta"][1] = 1.
    elif change == "resolution":
        row["refinement"] = 5
    else:
        row["k"][0] += .01
    with pytest.raises(ValueError, match=r"identical|OAT"):
        summarize(cases)


def test_partial_results_remain_explicitly_incomplete():
    """Missing refinement runs never become a claimed zero numerical floor."""
    cases = fixture_cases()
    for key in REFINEMENTS:
        del cases[key]
    with pytest.raises(ValueError, match="Missing completed"):
        summarize(cases)
    result = summarize(cases, allow_partial=True)
    assert not result["complete"]
    assert result["fiducial_refinement_envelope"] is None
    assert "response_exceeds_fiducial_refinement_envelope_fraction" not in result["physical_controls"][0]


def test_nonpositive_reference_power_is_rejected():
    """Do not manufacture finite fractional errors at a vanishing denominator."""
    cases = fixture_cases()
    cases["fiducial"]["P21"][0][0] = 0.
    with pytest.raises(ValueError, match="strictly positive"):
        summarize(cases)


def test_final_budget_summary_excludes_earlier_snapshot_response():
    """History controls fix final photons, so endpoint changes stay identifiable."""
    cases = fixture_cases()
    cases["oat_1_0"]["P21"][0] = [20., 40., 60., 80.]
    result = summarize(cases)["physical_controls"][1]
    assert result["power_relative_maximum_absolute"] == pytest.approx(1.)
    assert result["power_relative_final_maximum_absolute"] == pytest.approx(.02)
    assert result["power_relative_final_mean_absolute"] == pytest.approx(.02)
    assert result["xHI_final_difference_maximum_absolute"] == pytest.approx(.002)


def test_mislabeled_numerical_refinement_is_rejected():
    """A simultaneous time and angle change is not an isolated angular check."""
    cases = fixture_cases()
    cases["convergence_angle4"]["time_substeps"] = 2
    with pytest.raises(ValueError, match="refinement numerical configuration"):
        summarize(cases)
