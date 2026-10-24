"""Independent history, storage and photon accounting checks."""
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evolution import checkpoint_state, read_checkpoint, run_history, solve
from fine_state_metrics import fine_state_metrics


def fixture(rates=None):
    rng = np.random.default_rng(22)
    capacity = rng.uniform(.4, 1.6, (3,)*3)
    if rates is None:
        rates = np.zeros((9, 3, 3, 3))
        rates[:, 0, 1, 1] = 1.2
        rates[:, 2, 1, 1] = .9
    return dict(redshifts=np.linspace(12, 8, len(rates)),
                times_seconds=np.arange(len(rates), dtype=float), rates=rates,
                capacity=capacity, metadata={"fixture": "synthetic fixed gas"})


def final(report):
    return report["checkpoints"][-1]["fields"]


def test_constant_source_subdivision_preserves_state_and_source_budget():
    data = fixture()
    reports = [run_history(data, stride=s, angles=96) for s in (1, 2, 4)]
    for report in reports:
        assert report["all_ledgers_passed"]
        assert [r["node_index"] for r in report["checkpoints"]] == [0, 4, 8]
        assert report["source_quadrature"]["summed_interval_L1_over_emitted"] == 0
        np.testing.assert_allclose(final(report)["ionized"], final(reports[0])["ionized"],
                                   atol=3e-13, rtol=1e-12)
        np.testing.assert_allclose(final(report)["source_emitted"],
                                   np.trapezoid(data["rates"], data["times_seconds"], axis=0),
                                   atol=2e-15, rtol=2e-15)


def test_changing_sources_preserve_per_source_counts_and_report_quadrature_error():
    rates = np.zeros((9, 3, 3, 3))
    rates[:, 0, 1, 1] = [0, .1, 2., 4., .2, 0, 0, 0, 0]
    rates[:, 2, 1, 1] = [0, 0, 0, 0, 1., 3., 2., .5, 0]
    data = fixture(rates)
    fine = run_history(data, stride=1, angles=96)
    coarse = run_history(data, stride=4, angles=96)
    np.testing.assert_allclose(final(fine)["source_emitted"], final(coarse)["source_emitted"],
                               atol=2e-15, rtol=2e-15)
    assert coarse["source_quadrature"]["summed_interval_L1_over_emitted"] > .1
    # A different order at the same per-source total is a different history.
    reverse = run_history(fixture(rates[::-1].copy()), stride=1, angles=96)
    np.testing.assert_allclose(final(fine)["source_emitted"], final(reverse)["source_emitted"],
                               atol=2e-15, rtol=2e-15)
    assert np.sum(abs(final(fine)["ionized"]-final(reverse)["ionized"])) > .01


def test_serialized_checkpoint_restart_and_capacity_guard(tmp_path):
    data = fixture()
    report = run_history(data, angles=96, output=tmp_path/"run")
    fields = read_checkpoint(report["checkpoints"][1]["path"])
    state = checkpoint_state(fields)
    increment = data["rates"][0]*4
    resumed = solve(increment, data["capacity"], initial_state=state,
                    subcells_per_axis=4, angles=96, maximum_distance=24)
    reference = read_checkpoint(report["checkpoints"][-1]["path"])
    np.testing.assert_allclose(resumed["state"]["ionized"], reference["ionized"],
                               atol=2e-13, rtol=2e-12)
    fine = fine_state_metrics(data["capacity"], 4, resumed["state"],
                              checkpoint_state(reference), reference["source_emitted"].sum())
    assert fine["subcell_L1_over_emitted"] < 2e-13
    cumulative = fields["source_emitted"] + increment
    np.testing.assert_allclose(cumulative, reference["source_emitted"], atol=2e-15, rtol=2e-15)
    with pytest.raises(ValueError, match="same static capacity"):
        solve(increment, data["capacity"]*1.001, initial_state=state,
              subcells_per_axis=4, angles=96)
    assert (tmp_path/"run/summary.json").is_file()
    with pytest.raises(FileExistsError):
        run_history(data, angles=96, output=tmp_path/"run")


def test_retained_counts_are_carried_in_ledger_not_reemitted():
    rates = np.zeros((9, 3, 3, 3))
    rates[:3, 1, 1, 1] = 8
    data = fixture(rates)
    report = run_history(data, angles=24, maximum_distance=.1)
    assert report["final_ledger"]["retained"] > 0
    assert report["final_ledger"]["passed"]
    # Once emission ceases, neither the old photon budget nor ionization grows.
    first, last = report["checkpoints"][1:]
    np.testing.assert_array_equal(first["fields"]["ionized"], last["fields"]["ionized"])
    np.testing.assert_array_equal(first["fields"]["source_emitted"], last["fields"]["source_emitted"])


def test_zero_source_history_is_valid():
    report = run_history(fixture(np.zeros((9, 3, 3, 3))), angles=24)
    assert report["all_ledgers_passed"]
    assert report["final_ledger"]["emitted"] == 0
    assert report["final_ledger"]["relative_residual"] == 0
    assert np.count_nonzero(final(report)["ionized"]) == 0


@pytest.mark.parametrize("bad", [-1., np.nan, np.inf, 1j])
def test_invalid_source_rates_rejected(bad):
    data = fixture()
    if np.iscomplexobj(bad):
        data["rates"] = data["rates"].astype(complex)
    data["rates"][3, 1, 1, 1] = bad
    with pytest.raises(ValueError):
        run_history(data, angles=24)


def test_history_clock_and_stride_validation():
    data = fixture()
    for stride in (0, True, 3, 2.):
        with pytest.raises(ValueError):
            run_history(data, stride=stride)
    data["times_seconds"][0] = -1
    with pytest.raises(ValueError, match="start at zero"):
        run_history(data)


def test_npz_history_and_incomplete_final_stride(tmp_path):
    data = fixture()
    for key in ("redshifts", "times_seconds", "rates"):
        data[key] = data[key][:7]
    data["metadata"] = np.array('{"box_hMpc": 50.0}')
    path = tmp_path/"history.npz"
    np.savez(path, **data)
    with np.load(path, allow_pickle=False) as archive:
        report = run_history(archive, stride=4, angles=24)
    assert [r["node_index"] for r in report["checkpoints"]] == [0, 4, 6]
    assert report["metadata"]["box_hMpc"] == 50.


def test_carried_history_matches_independent_dense_reference():
    sys.path.insert(0, str(HERE.parent/"pc_scale_origin"))
    import point_front
    rates = np.zeros((5, 3, 3, 3))
    rates[:, 0, 1, 1] = [0, 3, 1, 0, 0]
    rates[:, 2, 1, 1] = [0, 0, 0, 2, 3]
    data = fixture(rates)
    result = run_history(data, angles=96)
    capacity = data["capacity"]
    fine_capacity = capacity.repeat(4, 0).repeat(4, 1).repeat(4, 2)/64
    absorbed = np.zeros_like(fine_capacity)
    positions = (np.indices(capacity.shape).reshape(3, -1).T+.5)*4
    for left, right in zip(rates[:-1], rates[1:]):
        increment = (left+right)/2
        active = increment.ravel() > 0
        reference = point_front.solve(positions[active], increment.ravel()[active],
                                       fine_capacity-absorbed, angles=96,
                                       maximum_distance=96, periodic=True)
        absorbed += reference["absorbed"]
    state = checkpoint_state(final(result))
    # Dense reconstruction is used only in this small independent reference.
    from refined import _dense
    reconstructed = _dense(capacity, 4, state["cleared_bits"],
                           state["partial_ids"], state["partial_ionized"])
    np.testing.assert_allclose(reconstructed, absorbed, atol=3e-13, rtol=2e-11)


def test_early_source_quadrature_error_is_not_diluted_by_late_emission():
    data = fixture(np.zeros((9, 3, 3, 3)))
    # Coarse stride-two endpoint trapezoids miss this early pulse entirely.
    # The late ramp is linear, so its coarse quadrature is exact, and its
    # enormous photon count dilutes any final-only relative error.
    data["rates"][:, 1, 1, 1] = [0, .5*data["capacity"].sum(), 0, 0, 0,
                                 1000, 2000, 3000, 4000]
    report = run_history(data, stride=2, angles=24)
    quadrature = report["source_quadrature"]
    assert quadrature["summed_interval_L1_over_emitted"] < .01
    early = quadrature["checkpoints"][1]
    assert early["node_index"] == 4
    assert early["Q_mass"] == pytest.approx(.5, rel=1e-12)
    assert early["cumulative_photon_dose"] == pytest.approx(.5, rel=1e-12)
    assert early["cumulative_L1_over_emitted"] == pytest.approx(1.)
    assert early["signed_cumulative_difference_over_emitted"] == pytest.approx(-1.)
    checkpoint = report["checkpoints"][1]
    assert checkpoint["source_quadrature"]["coarse_endpoint_emitted"] == 0
    assert np.count_nonzero(checkpoint["fields"]["source_coarse_endpoint_emitted"]) == 0
    assert quadrature["checkpoints"][-1]["cumulative_L1_over_emitted"] < .01
