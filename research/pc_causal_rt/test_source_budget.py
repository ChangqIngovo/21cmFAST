"""Independent source-count tests for moving native cumulative sources."""
import numpy as np
import pytest

from research.pc_causal_rt.source_budget import (
    ARRAYS, PACKET_ARRAYS, _array_hash, _save_budget, _validate_budget,
    load_source_budget, sampled_cumulative, source_initial_map, source_positions,
    source_rate,
)


def fixture(cumulative=None):
    n = 4
    if cumulative is None:
        cumulative = np.ones((3, n, n, n)) * np.arange(1., 4.)[:, None, None, None]
    budget = {"cumulative": cumulative, "times_seconds": np.array([0., 2., 6.]),
              "redshifts": np.array([9., 8., 7.]), "growth_factors": np.array([.2, .4, .6]),
              "native_totals": cumulative.sum(axis=(1, 2, 3)),
              "packets": {"q": np.indices((n,) * 3).reshape(3, -1).T.astype(float),
                          "v1": np.zeros((n**3, 3)), "v2": np.zeros((n**3, 3)),
                          "growth_initial": .1, "coarse_dimension": n}}
    budget["metadata"] = {"schema": 1, "dimension": n, "growth_initial": .1, "identity": {"test": True},
        "array_sha256": {name: _array_hash(budget[name]) for name in ARRAYS}
            | {f"packet_{name}": _array_hash(budget["packets"][name]) for name in PACKET_ARRAYS}}
    return budget


@pytest.mark.parametrize("factor", [1, 2, 4])
def test_moving_constant_cumulative_source_emits_nothing(factor):
    q = np.zeros((3, 4, 4, 4))
    q[:, 0, 0, 0] = 3.
    budget = fixture(q)
    budget["packets"]["v1"][:, 0] = 9.
    rate = source_rate(budget, q, 1, .3, factor)
    assert np.count_nonzero(rate) == 0
    assert source_initial_map(budget, factor).sum() / factor**3 == pytest.approx(3.)


def test_growth_emission_occurs_at_current_source_position():
    q = np.zeros((3, 4, 4, 4))
    q[:, 0, 0, 0] = [1., 5., 9.]
    budget = fixture(q)
    # At D=.5 this source has moved exactly one native cell.
    budget["packets"]["v1"][0, 0] = 2.5
    rate = source_rate(budget, q, 1, .5)
    expected = np.zeros((4, 4, 4))
    expected[1, 0, 0] = 2.
    np.testing.assert_array_equal(rate, expected)


def test_negative_packet_increment_is_not_clipped_or_relocated():
    budget = fixture()
    budget["cumulative"][1, 0, 0, 0] = .9
    with pytest.raises(ValueError, match="decreases"):
        _validate_budget(budget, check_hashes=False)
    with pytest.raises(ValueError, match="Negative packet"):
        source_rate(budget, budget["cumulative"], 1, .3)
    with pytest.raises(ValueError, match="decreases"):
        sampled_cumulative(budget, budget["times_seconds"], 1)


def test_source_knots_interpolate_cumulative_counts_in_proper_time():
    budget = fixture()
    sampled = sampled_cumulative(budget, budget["times_seconds"], 2)
    np.testing.assert_array_equal(sampled[0], budget["cumulative"][0])
    np.testing.assert_array_equal(sampled[-1], budget["cumulative"][-1])
    np.testing.assert_allclose(sampled[1], 1 + 2 / 3)
    initial = source_initial_map(budget).sum()
    increments = sum(source_rate(budget, sampled, node, .3).sum()
                     * np.diff(budget["times_seconds"])[node - 1] for node in (1, 2))
    assert initial + increments == pytest.approx(budget["native_totals"][-1])


def test_fine_cloud_restricts_to_same_native_source():
    budget = fixture()
    budget["packets"]["v1"][:, 0] = .7
    budget["packets"]["v2"][:, 1] = .3
    budget["cumulative"][0] *= np.arange(1., 65.).reshape(4, 4, 4)
    coarse = source_initial_map(budget)
    fine = source_initial_map(budget, 2)
    restricted = fine.reshape(4, 2, 4, 2, 4, 2).mean(axis=(1, 3, 5))
    np.testing.assert_allclose(restricted, coarse, rtol=0, atol=2e-13)


def test_native_trajectory_uses_source_packet_2lpt_and_refinement_offset():
    budget = fixture()
    budget["packets"]["v1"][0] = [1., 2., 3.]
    budget["packets"]["v2"][0] = [3., 2., 1.]
    expected = .2 * np.array([1., 2., 3.]) + 3 / 7 * (.3**2 - .1**2) * np.array([3., 2., 1.])
    np.testing.assert_allclose(source_positions(budget, .3, 2)[0], 2 * expected + .5)


def test_saved_budget_checks_file_and_array_integrity(tmp_path):
    path = tmp_path / "source_budget.npz"
    budget = fixture()
    _save_budget(path, budget)
    restored = load_source_budget(path)
    np.testing.assert_array_equal(restored["cumulative"], budget["cumulative"])
    assert restored["path"] == str(path)
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_source_budget(path)


def test_mismatched_native_total_rejected():
    budget = fixture()
    budget["native_totals"][-1] *= 1.01
    with pytest.raises(ValueError, match="native coeval total"):
        _validate_budget(budget, check_hashes=False)
