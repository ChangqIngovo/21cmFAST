"""Sparse-state differences against an independently constructed dense grid."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fine_state_metrics import fine_state_metrics


def encode(capacity, f, fractions):
    n = len(capacity)
    bits = np.zeros((n**3, (f**3+63)//64), dtype=np.uint64)
    ids, values = [], []
    for xyz in np.ndindex((n*f,)*3):
        x, y, z = xyz
        parent = ((x//f)*n+y//f)*n+z//f
        child = ((x % f)*f+y % f)*f+z % f
        fraction = fractions[xyz]
        if fraction == 1:
            bits[parent, child//64] |= np.uint64(1) << np.uint64(child % 64)
        elif fraction != 0:
            ids.append(np.ravel_multi_index(xyz, fractions.shape))
            values.append(fraction*capacity.ravel()[parent]/f**3)
    return dict(cleared_bits=bits, partial_ids=np.array(ids, dtype=np.int64),
                partial_ionized=np.array(values))


@pytest.mark.parametrize('f', [2, 4, 8])
def test_sparse_l1_matches_dense_and_is_symmetric(f):
    rng = np.random.default_rng(109+f)
    capacity = rng.uniform(.1, 2., (2,)*3)
    values = np.array([0., .001, .1, .7, .999, 1.])
    a, b = (rng.choice(values, size=(2*f,)*3) for _ in range(2))
    ca = capacity.repeat(f, 0).repeat(f, 1).repeat(f, 2)/f**3
    expected = np.sum(abs((a-b)*ca), dtype=np.longdouble)
    sa, sb = encode(capacity, f, a), encode(capacity, f, b)
    for new, old in ((sa, sb), (sb, sa)):
        r = fine_state_metrics(capacity, f, new, old, 3.)
        assert r['subcell_L1_over_emitted'] == pytest.approx(float(expected/3), rel=3e-15)
        assert r['volume_mean_absolute_ionized_fraction_difference'] == pytest.approx(np.mean(abs(a-b)))
        assert r['maximum_ionized_fraction_difference'] == pytest.approx(np.max(abs(a-b)))
    assert fine_state_metrics(capacity, f, sa, sa, 3.)['subcell_L1_over_emitted'] == 0


def test_partial_clear_changes_with_equal_parent_counts_are_visible():
    cap = np.ones((2,)*3)
    a, b = np.zeros((8,)*3), np.zeros((8,)*3)
    a[0, 0, 0], a[0, 0, 1] = 1., .4
    b[0, 0, 0], b[0, 0, 1] = .4, 1.
    # Two modified children are in the same parent and the same bit-mask word.
    r = fine_state_metrics(cap, 4, encode(cap, 4, a), encode(cap, 4, b), 1.)
    assert a.sum() == b.sum()
    assert r['subcell_L1_over_emitted'] == pytest.approx(1.2/64)
    assert r['maximum_ionized_fraction_difference'] == pytest.approx(.6)


def test_empty_partial_states_and_invalid_duplicate_storage():
    cap = np.ones((2,)*3)
    neutral = encode(cap, 2, np.zeros((4,)*3))
    clear = encode(cap, 2, np.ones((4,)*3))
    r = fine_state_metrics(cap, 2, clear, neutral, 8.)
    assert r['subcell_L1_over_emitted'] == 1.
    assert r['volume_mean_absolute_ionized_fraction_difference'] == 1.
    invalid = {**clear, 'partial_ids': np.array([0]), 'partial_ionized': np.array([.01])}
    with pytest.raises(ValueError, match='overlap'):
        fine_state_metrics(cap, 2, invalid, neutral, 8.)
