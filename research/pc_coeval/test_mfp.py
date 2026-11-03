"""Independent geometry checks; MC convergence is not spatial convergence."""
import json

import numpy as np
import pytest

from .mfp import make_draws, trace_rays, measure_mfp, compare_mfp


def test_exact_axial_distances_and_periodic_wrap():
    mask = np.zeros((8, 8, 8), bool)
    mask[[0, 1, 6, 7], :, :] = True
    origins = np.array([[0.2, 3., 3.], [0.2, 3., 3.], [7.3, 3., 3.], [7.3, 3., 3.]])
    directions = np.array([[1., 0., 0.], [-1., 0., 0.], [1., 0., 0.], [-1., 0., 0.]])
    distance, censored = trace_rays(mask, 8., origins, directions, max_distance=8.)
    np.testing.assert_allclose(distance, [1.8, 2.2, 2.7, 1.3], rtol=0, atol=1e-14)
    assert not censored.any()


def test_corner_ties_cross_all_faces():
    mask = np.zeros((3, 3, 3), bool)
    mask[0, 0, 0] = mask[1, 1, 1] = True
    d, c = trace_rays(mask, 3., [[.5, .5, .5]], [[1., 1., 1.]])
    assert d[0] == pytest.approx(1.5*np.sqrt(3))
    assert not c[0]


def test_origin_on_boundary_with_negative_direction():
    mask = np.zeros((4, 4, 4), bool)
    mask[1:3] = True
    d, c = trace_rays(mask, 4., [[1., 2., 2.]], [[-1., 0., 0.]])
    assert d[0] == 0 and not c[0]


def test_random_rays_against_independent_face_intersection_reference():
    rng = np.random.default_rng(85)
    n, cap = 7, 14.
    mask = rng.random((n, n, n))>.2
    selected = np.argwhere(mask)
    origins = selected[rng.integers(len(selected), size=400)]+rng.random((400, 3))
    directions = rng.normal(size=(400, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    result, flags = trace_rays(mask, float(n), origins, directions, max_distance=cap)
    expected, expected_flags = [], []
    # Independently intersect each ray with every unwrapped grid plane in its
    # bounding range, then inspect the interval immediately after every face.
    for origin, direction in zip(origins, directions):
        cuts = [cap]
        for start, velocity in zip(origin, direction):
            end = start+velocity*cap
            planes = np.arange(np.floor(min(start, end))-1, np.ceil(max(start, end))+2)
            times = (planes-start)/velocity
            cuts.extend(times[(times>=0)&(times<cap)])
        cuts = np.unique(cuts)
        answer, censored = cap, True
        for j, crossing in enumerate(cuts[:-1]):
            midpoint = (crossing+cuts[j+1])/2
            cell = np.floor(origin+direction*midpoint).astype(int)%n
            if not mask[tuple(cell)]:
                answer, censored = crossing, False
                break
        expected.append(answer); expected_flags.append(censored)
    np.testing.assert_allclose(result, expected, rtol=0, atol=3e-13)
    np.testing.assert_array_equal(flags, expected_flags)


def test_periodic_open_channel_is_censored_after_multiple_wraps():
    mask = np.zeros((4, 4, 4), bool)
    mask[1] = True
    d, c = trace_rays(mask, 4., [[1.5, .1, .2]], [[0., 1., 1.]], max_distance=20.)
    assert d[0] == 20 and c[0]


def test_slab_known_restricted_mean_and_censoring():
    # Width a=2, cap T=8: E[min(R,T)]=3a/4+(a/2)log(T/a), Pr(R>=T)=a/(2T).
    xhi = np.ones((16, 16, 16))
    xhi[6:10] = 0
    result = measure_mfp(xhi, 8., ray_count=120_000, seed=92, threads=2)
    s = result["summary"]
    assert s["restricted_mean_cMpc"] == pytest.approx(1.5+np.log(4), abs=.03)
    assert s["censored_fraction"] == pytest.approx(.125, abs=.004)
    assert sum(s["histogram_probability"])+s["censored_fraction"] == pytest.approx(1.)
    assert s["boundary_cdf_at_edges"][-1] == pytest.approx(1-s["censored_fraction"])
    assert s["uncensored_mean_cMpc"] is None
    assert s["restricted_mean_standard_error_cMpc"]>0


def test_voxelized_sphere_has_analytic_mean_and_cdf():
    # Continuous sphere has E[R]=3R_sphere/4, CDF(l)=3l/(4R)-l^3/(16R^3).
    # Voxel boundary approximation is covered explicitly by a finite tolerance.
    n, box_len, radius = 96, 8., 2.
    xyz = (np.arange(n)+.5)*box_len/n-box_len/2
    sphere = (xyz[:, None, None]**2+xyz[None, :, None]**2+xyz[None, None, :]**2)<radius**2
    result = measure_mfp((~sphere).astype(float), box_len, ray_count=100_000, seed=1704, threads=2)
    distance = result["distance_cMpc"]
    assert not result["censored"].any()
    assert distance.mean() == pytest.approx(3*radius/4, abs=.035)
    for length in (.5, 1., 2., 3.):
        expected = 3*length/(4*radius)-length**3/(16*radius**3)
        assert np.mean(distance<=length) == pytest.approx(expected, abs=.012)


def test_translation_scaling_and_axis_permutation():
    rng = np.random.default_rng(71)
    mask = rng.random((12, 12, 12))>.25
    selected = np.argwhere(mask)
    origins = selected[rng.integers(len(selected), size=1000)]+rng.random((1000, 3))
    directions = rng.normal(size=(1000, 3))
    original, flags = trace_rays(mask, 12., origins, directions)
    shifted, shifted_flags = trace_rays(np.roll(mask, (3, 2, 1), axis=(0, 1, 2)), 12.,
                                         origins+[3, 2, 1], directions)
    np.testing.assert_allclose(original, shifted, rtol=0, atol=1e-13)
    np.testing.assert_array_equal(flags, shifted_flags)
    scaled, sf = trace_rays(mask, 84., origins*7, directions)
    np.testing.assert_allclose(scaled, original*7, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(flags, sf)
    permuted, pf = trace_rays(mask.transpose(2, 0, 1), 12., origins[:, [2, 0, 1]], directions[:, [2, 0, 1]])
    np.testing.assert_allclose(permuted, original, rtol=0, atol=1e-13)
    np.testing.assert_array_equal(flags, pf)


def test_uniform_direction_sampling_is_not_axis_biased():
    draws = make_draws(120_000, 713)
    summaries = []
    for axis in range(3):
        xhi = np.ones((16, 16, 16))
        sl = [slice(None)]*3
        sl[axis] = slice(6, 10)
        xhi[tuple(sl)] = 0.
        summaries.append(measure_mfp(xhi, 8., draws=draws, threads=2)["summary"])
    assert np.ptp([s["restricted_mean_cMpc"] for s in summaries])<.04
    assert np.ptp([s["censored_fraction"] for s in summaries])<.008


def test_thread_determinism_shared_draws_and_identical_comparison():
    xhi = np.random.default_rng(2).random((10, 10, 10))
    draws = make_draws(3000, 5)
    a = measure_mfp(xhi, 30., draws=draws, threads=1)
    b = measure_mfp(xhi, 30., draws=draws, threads=4)
    np.testing.assert_array_equal(a["distance_cMpc"], b["distance_cMpc"])
    np.testing.assert_array_equal(a["censored"], b["censored"])
    assert a["summary"] == b["summary"]
    comparison = compare_mfp(a, b)
    assert comparison["restricted_cdf_max_difference"] == 0
    assert comparison["restricted_wasserstein1_cMpc"] == 0
    assert comparison["restricted_mean_ratio"] == 1
    assert comparison["paired_restricted_mean_difference_standard_error_cMpc"] == 0


def test_full_and_empty_phase_are_not_spurious_zero_mfp():
    xhi = np.zeros((5, 5, 5))
    ionized = measure_mfp(xhi, 50., ray_count=100)
    s = ionized["summary"]
    assert s["status"] == "all_censored"
    assert s["censored_fraction"] == 1
    assert s["restricted_mean_cMpc"] == 50
    assert s["uncensored_mean_cMpc"] is None
    assert all(v is None for v in s["quantiles_cMpc"].values())
    assert sum(s["histogram_probability"]) == 0
    neutral = measure_mfp(xhi, 50., ray_count=100, phase="neutral")
    assert neutral["summary"]["status"] == "empty_phase"
    assert neutral["summary"]["restricted_mean_cMpc"] is None
    assert compare_mfp(ionized, neutral)["available"] is False
    json.dumps(s, allow_nan=False)
    json.dumps(neutral["summary"], allow_nan=False)


def test_censored_quantiles_and_histogram_normalization():
    xhi = np.ones((8, 8, 8))
    xhi[2:6] = 0.
    result = measure_mfp(xhi, 8., max_distance=2., ray_count=20_000, seed=41)
    s = result["summary"]
    assert s["censored_fraction"] == pytest.approx(.75, abs=.015)
    assert s["quantiles_cMpc"]["0.16"] is not None
    assert s["quantiles_cMpc"]["0.5"] is None
    assert s["quantiles_cMpc"]["0.84"] is None
    assert sum(s["histogram_probability"])+s["censored_fraction"] == pytest.approx(1)


def test_sampling_prefix_and_half_vs_full_diagnostic():
    np.testing.assert_array_equal(make_draws(100, 123), make_draws(200, 123)[:100])
    xhi = np.zeros((5, 5, 5))
    xhi[0] = 1.
    small = measure_mfp(xhi, 5., ray_count=1000, seed=123)
    large = measure_mfp(xhi, 5., ray_count=2000, seed=123)
    np.testing.assert_array_equal(small["distance_cMpc"], large["distance_cMpc"][:1000])
    assert large["summary"]["restricted_mean_first_half_cMpc"] == pytest.approx(small["summary"]["restricted_mean_cMpc"])
    assert 0<=large["summary"]["first_half_vs_all_restricted_cdf_max_difference"]<=1


def test_threshold_partition_and_no_input_mutation():
    xhi = np.arange(27.).reshape((3, 3, 3))/26
    original = xhi.copy()
    ion = measure_mfp(xhi, 3., ray_count=100, threshold=.5)
    neu = measure_mfp(xhi, 3., ray_count=100, threshold=.5, phase="neutral")
    assert ion["summary"]["selected_voxel_count"]+neu["summary"]["selected_voxel_count"] == 27
    assert neu["summary"]["selected_voxel_count"] == 14
    np.testing.assert_array_equal(xhi, original)


@pytest.mark.parametrize("kwargs", [{"threshold": -1}, {"threshold": np.nan}, {"phase": "bad"},
    {"box_len": 0}, {"threads": 0}, {"threads": 1.5}, {"max_distance": np.inf},
    {"bins": 1}, {"ray_count": 0}, {"draws": np.ones((3, 6))}])
def test_bad_settings_rejected(kwargs):
    settings = {"box_len": 10., "ray_count": 100, **kwargs}
    with pytest.raises(ValueError):
        measure_mfp(np.zeros((3, 3, 3)), **settings)


@pytest.mark.parametrize("array", [np.zeros((2, 3, 3)), np.zeros((2, 2)),
    np.full((3, 3, 3), np.nan), np.full((3, 3, 3), -.1), np.full((3, 3, 3), 1.1)])
def test_bad_fields_rejected(array):
    with pytest.raises(ValueError):
        measure_mfp(array, 10., ray_count=100)


def test_bad_ray_origins_and_zero_direction_rejected():
    mask = np.zeros((4, 4, 4), bool)
    mask[1] = True
    with pytest.raises(ValueError, match="selected phase"):
        trace_rays(mask, 4., [[0.5, 1., 1.]], [[1., 0., 0.]])
    with pytest.raises(ValueError, match="nonzero norm"):
        trace_rays(mask, 4., [[1.5, 1., 1.]], [[0., 0., 0.]])
