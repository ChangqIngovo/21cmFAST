import numpy as np
import pytest

from ray_segments import build_paths, trace_segments


def test_diagonal_chords_and_origin_half_cell():
    out = trace_segments((4, 4, 4), [.5, .5, .5], np.ones(3) / np.sqrt(3))
    assert out.cells == (0, 21, 42, 63)
    np.testing.assert_allclose(out.lengths, np.array([.5, 1, 1, 1]) * np.sqrt(3), rtol=2e-15)
    assert out.terminal == "escaped"
    assert sum(out.lengths) == pytest.approx(3.5 * np.sqrt(3))


def test_face_start_is_direction_aware_and_reflects():
    positive = trace_segments((4, 1, 1), [2, .5, .5], [1, 0, 0])
    negative = trace_segments((4, 1, 1), [2, .5, .5], [-1, 0, 0])
    assert positive.cells == (2, 3)
    assert negative.cells == (1, 0)
    assert positive.lengths == negative.lengths == (1., 1.)
    inward = trace_segments((4, 1, 1), [4, .5, .5], [-1, 0, 0])
    assert inward.cells == (3, 2, 1, 0)
    assert trace_segments((4, 1, 1), [4, .5, .5], [1, 0, 0]).cells == ()


def test_near_corner_keeps_real_thin_segment():
    direction = np.array([1, 1 + 1e-10, 0.]); direction /= np.linalg.norm(direction)
    out = trace_segments((3, 3, 1), [.5, .5, .5], direction)
    assert out.cells[:3] == (0, 1, 4)
    assert 0 < out.lengths[1] < 1e-9
    assert all(x > 0 for x in out.lengths)


def test_cutoff_and_exit_at_identical_distance():
    for length, terminal in ((.5, "range_limited"), (1.5, "escaped")):
        out = trace_segments((2, 1, 1), [.5, .5, .5], [1, 0, 0], maximum_distance=length)
        assert out.distance == length
        assert sum(out.lengths) == length
        assert out.terminal == terminal


def test_oblique_beer_lambert_path_depth_and_spatial_refinement():
    direction = np.array([1., .25, .1]); direction /= np.linalg.norm(direction)
    for n in (4, 8, 16, 32):
        dx = 1 / n
        coefficient = np.full((n, n, n), 2.3)
        paths, tau, geom = build_paths(coefficient, [[0, .2, .2]], [direction], cell_width=dx)
        length = 1 / direction[0]
        assert sum(tau[0]) == pytest.approx(2.3 * length, rel=2e-15)
        assert np.exp(-sum(tau[0])) == pytest.approx(np.exp(-2.3 * length), rel=2e-15)
        assert len(paths[0]) == len(set(paths[0]))
        assert sum(geom[0].lengths) == pytest.approx(length)


def test_axis_permutation_preserves_optical_depth():
    rng = np.random.default_rng(25)
    coefficient = rng.uniform(.1, 3., (7, 7, 7))
    origin = np.array([1.23, 2.67, 3.1]); direction = np.array([1, .8, -.2])
    direction /= np.linalg.norm(direction)
    _, a, _ = build_paths(coefficient, [origin], [direction])
    for perm in ((1, 2, 0), (2, 0, 1), (2, 1, 0)):
        _, b, _ = build_paths(coefficient.transpose(perm), [origin[list(perm)]], [direction[list(perm)]])
        np.testing.assert_allclose(a, b, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("kwargs", [dict(cell_width=0), dict(maximum_distance=0), dict(maximum_distance=np.inf)])
def test_invalid_geometry_is_rejected(kwargs):
    with pytest.raises(ValueError):
        trace_segments((2, 2, 2), [.5, .5, .5], [1, 0, 0], **kwargs)
