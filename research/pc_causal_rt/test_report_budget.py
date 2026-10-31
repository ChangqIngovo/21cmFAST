"""Audit native cumulative budgets independently of the evolution driver."""
import hashlib

import numpy as np
import pytest

from research.pc_causal_rt import report
from research.pc_causal_rt.gas import deposit


def _fixture(monkeypatch, tmp_path, *, constant=False):
    from research.pc_causal_rt import native_history, source_budget
    n = 4
    growth = np.array([.1, .5, .9])
    times = np.array([0., 2., 4.])
    coordinates = np.indices((n,) * 3, dtype=float).reshape(3, -1).T
    v1 = np.zeros_like(coordinates)
    v1[0, 0] = 1.
    q = np.zeros((3, n, n, n))
    q[:, 0, 0, 0] = [1., 1., 1.] if constant else [1., 3., 4.]
    packets = dict(q=coordinates, v1=v1, v2=np.zeros_like(v1),
                   growth_initial=.01, coarse_dimension=n)
    history = dict(times_seconds=times, redshifts=np.array([10., 9., 8.]),
                   growth_factors=growth, rates=np.zeros_like(q),
                   native_capacity=np.ones_like(q), packet_mass=np.ones(n**3), records=[])
    budget = dict(cumulative=q, times_seconds=times, redshifts=history["redshifts"],
                  growth_factors=growth, native_totals=q.sum(axis=(1, 2, 3)),
                  packets=packets, metadata={"native_archive_root": str(tmp_path)})
    for node, g in enumerate(growth):
        source = deposit(q[node].ravel(), coordinates + (g - .01) * v1, n)
        path = tmp_path / f"source{node}.npz"
        np.savez(path, source=source)
        history["records"].append({"fields_path": path.name,
            "source_sha256": hashlib.sha256(np.ascontiguousarray(source).tobytes()).hexdigest()})
    monkeypatch.setattr(native_history, "background_at_time", lambda h, node, fraction: {
        "growth_factor": (1 - fraction) * h["growth_factors"][node - 1] + fraction * h["growth_factors"][node]})
    monkeypatch.setattr(native_history, "load_history", lambda path: history)
    monkeypatch.setattr(source_budget, "load_source_budget", lambda path: budget)
    return history, budget


def test_initial_budget_counted_once_and_growth_follows_moving_packet(monkeypatch, tmp_path):
    history, budget = _fixture(monkeypatch, tmp_path)
    result = report._native_budget_history(history, budget, {"source_stride": 1, "time_substeps": 1})
    np.testing.assert_allclose(result["cumulative"].sum(axis=(1, 2, 3)), [1., 3., 4.], atol=1e-14)
    # At node1 the initial unit was deposited at x=.09 and the two new
    # photons at x=.29; those old photons are not moved with the source.
    assert result["cumulative"][1, 0, 0, 0] == pytest.approx(.91 + 2 * .71)
    assert result["cumulative"][1, 1, 0, 0] == pytest.approx(.09 + 2 * .29)
    assert not np.array_equal(result["cumulative"][1], result["current_native"][1])


def test_translation_of_constant_cumulative_source_emits_no_new_photons(monkeypatch, tmp_path):
    history, budget = _fixture(monkeypatch, tmp_path, constant=True)
    result = report._native_budget_history(history, budget, {"source_stride": 1, "time_substeps": 4})
    np.testing.assert_array_equal(result["cumulative"][0], result["cumulative"][2])
    assert not np.array_equal(result["current_native"][0], result["current_native"][2])


def test_decreasing_packet_budget_cannot_be_audited_as_positive_emission(monkeypatch, tmp_path):
    history, budget = _fixture(monkeypatch, tmp_path)
    budget["cumulative"][1, 0, 0, 0] = .5
    with pytest.raises(ValueError, match="decreases"):
        report._native_budget_history(history, budget, {"source_stride": 1, "time_substeps": 1})


def _auditable_case(monkeypatch, tmp_path):
    history, budget = _fixture(monkeypatch, tmp_path)
    history_path = tmp_path / "history.npz"
    np.savez(history_path, **{key: value for key, value in history.items() if key != "records"})
    digest = report._digest(history_path)
    budget["metadata"]["history_sha256"] = digest
    budget_path = tmp_path / "budget.npz"
    budget_path.write_bytes(b"verified fixture budget")
    budget_hash = report._digest(budget_path)
    setting = {"source_stride": 1, "time_substeps": 2, "refinement": 1, "directions": 24, "cfl": .4}
    rebuilt = report._native_budget_history(history, budget, setting)
    case = {"case_id": "native", "seed": 111, "setting": setting,
            "source_mode": "native_cumulative", "source_budget_sha256": budget_hash,
            "history_sha256": digest}
    points = {}
    for node in range(len(history["times_seconds"])):
        points[node] = ({"redshift": float(history["redshifts"][node]),
                         "time_seconds": float(history["times_seconds"][node])}, {
            "capacity": np.ones((4, 4, 4)),
            "cumulative_source": np.array(rebuilt["cumulative"][node]),
            "source_native_current": rebuilt["current_native"][node].copy(),
            "source_sampled_current": rebuilt["current_sampled"][node].copy()})
    summary = {"histories": [{"seed": 111, "path": history_path.name, "sha256": digest,
                             "source_budget_path": budget_path.name, "source_budget_sha256": budget_hash}]}
    return summary, case, {"native": points}


def test_saved_budget_and_actual_native_source_both_audited(monkeypatch, tmp_path):
    summary, case, frames = _auditable_case(monkeypatch, tmp_path)
    report._history_sources(tmp_path, summary, [case], frames)
    audit = case["native_history_audit"]
    assert audit["passed"]
    assert audit["initial_source_total"] == 1.
    assert audit["maximum_native_cumulative_budget_relative_residual"] < 1e-14
    assert audit["maximum_native_source_L1_relative_roundoff"] < 1e-14


def test_missing_initial_inventory_rejected_even_if_later_growth_correct(monkeypatch, tmp_path):
    summary, case, frames = _auditable_case(monkeypatch, tmp_path)
    initial = frames["native"][0][1]["cumulative_source"].copy()
    for _, arrays in frames["native"].values():
        arrays["cumulative_source"] -= initial
    with pytest.raises(ValueError, match="independently integrated native history"):
        report._history_sources(tmp_path, summary, [case], frames)


def test_original_native_source_tamper_rejected(monkeypatch, tmp_path):
    summary, case, frames = _auditable_case(monkeypatch, tmp_path)
    np.savez(tmp_path / "source1.npz", source=np.ones((4, 4, 4)))
    with pytest.raises(ValueError, match="source array hash"):
        report._history_sources(tmp_path, summary, [case], frames)


def test_current_source_cannot_be_replaced_by_historical_emission_map(monkeypatch, tmp_path):
    summary, case, frames = _auditable_case(monkeypatch, tmp_path)
    arrays = frames["native"][1][1]
    arrays["source_sampled_current"] = arrays["cumulative_source"].copy()
    with pytest.raises(ValueError, match="current source map"):
        report._history_sources(tmp_path, summary, [case], frames)


def test_temporal_refinement_source_location_difference_is_diagnostic():
    # Both sources have the same total budget; midpoint sampling of a moving
    # emitter can change historical locations while the RT field converges.
    n = 8
    pattern = np.random.default_rng(42).uniform(.3, .7, (n, n, n))
    spectrum = {"k_cMpc_inv": [.16], "P_teacher": [1.]}
    point = {"time_seconds": 1., "redshift": 8., "xHI_mass": .5,
             "spectra": spectrum, "primary_power_band": {}}
    arrays = {"capacity": np.ones_like(pattern), "native_xhi": pattern,
              "rt_xhi": pattern, "cumulative_source": pattern}
    moved = dict(arrays, cumulative_source=np.roll(pattern, 1, axis=0))
    common = {"history_sha256": "same", "source_mode": "native_cumulative", "source_budget_sha256": "same-budget"}
    coarse, fine = dict(common, case_id="a"), dict(common, case_id="b")
    result = report._pair(coarse, fine, {"a": {1: (point, arrays)}, "b": {1: (point, moved)}},
                          "time_substeps", 50., .01)
    assert result["passed"]
    assert result["worst_source_L1_relative_error"] > .01
    moved["cumulative_source"] = moved["cumulative_source"] * 1.1
    with pytest.raises(ValueError, match="supplied source budget"):
        report._pair(coarse, fine, {"a": {1: (point, arrays)}, "b": {1: (point, moved)}},
                     "time_substeps", 50., .01)
