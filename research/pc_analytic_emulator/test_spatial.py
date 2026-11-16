"""Check label-free local descriptors against physical cube symmetries."""

import inspect
from itertools import permutations, product

import numpy as np
import pytest

from research.pc_analytic_emulator.spatial import spatial_features


def _inputs(size: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    random = np.random.default_rng(752)
    capacity = random.lognormal(0, 0.8, (size,) * 3)
    source = random.lognormal(-0.7, 1.1, capacity.shape)
    source.ravel()[::5] = 0
    native = random.uniform(0, 1, capacity.shape)
    return source, capacity, native


def test_native_only_interface_and_deterministic_bounded_columns() -> None:
    """No labels, reservoirs, fitted normalization or global state are inputs."""
    assert tuple(inspect.signature(spatial_features).parameters) == (
        "source",
        "capacity",
        "native_es_xHI",
        "scales_cells",
    )
    arrays = _inputs()
    copies = [array.copy() for array in arrays]
    features, names = spatial_features(*arrays)
    repeat, repeated_names = spatial_features(*arrays)
    assert features.shape == (arrays[0].size, len(names))
    assert features.dtype == np.float32
    assert len(set(names)) == len(names)
    assert all(name.startswith("spatial_") for name in names)
    assert repeated_names == names
    np.testing.assert_array_equal(features, repeat)
    assert np.isfinite(features).all()
    assert np.all(np.abs(features) <= 1)
    for array, original in zip(arrays, copies, strict=True):
        np.testing.assert_array_equal(array, original)


def test_periodic_translation_moves_every_descriptor_with_its_cell() -> None:
    """A boundary crossing must not change local descriptor values."""
    arrays = _inputs()
    original, names = spatial_features(*arrays)
    moved, moved_names = spatial_features(
        *(np.roll(array, (2, -1, 3), axis=(0, 1, 2)) for array in arrays)
    )
    expected = np.roll(
        original.reshape((*arrays[0].shape, -1)), (2, -1, 3), axis=(0, 1, 2)
    )
    assert moved_names == names
    np.testing.assert_allclose(moved, expected.reshape(moved.shape), rtol=0, atol=2e-7)


def test_all_48_cube_rotations_and_reflections_preserve_scalar_features() -> None:
    """Descriptors are scalar under the complete signed-permutation group."""
    arrays = _inputs(5)
    original, names = spatial_features(*arrays)
    original = original.reshape((*arrays[0].shape, -1))
    for permutation in permutations(range(3)):
        for reflections in product((False, True), repeat=3):
            axes = tuple(
                axis for axis, reflected in enumerate(reflections) if reflected
            )

            def transform(
                array: np.ndarray,
                permutation: tuple[int, ...] = permutation,
                axes: tuple[int, ...] = axes,
            ) -> np.ndarray:
                order = (*permutation, *range(3, array.ndim))
                return np.flip(array.transpose(order), axis=axes)

            changed, changed_names = spatial_features(*(transform(a) for a in arrays))
            assert changed_names == names
            np.testing.assert_allclose(
                changed, transform(original).reshape(changed.shape), rtol=0, atol=2e-7
            )


@pytest.mark.parametrize("unit", [1e-240, 1e-18, 1e18, 1e240])
def test_joint_photon_atom_count_units_do_not_change_features(unit: float) -> None:
    """Count units cancel before local moments and gradients are evaluated."""
    source, capacity, native = _inputs()
    original, names = spatial_features(source, capacity, native)
    changed, changed_names = spatial_features(source * unit, capacity * unit, native)
    assert names == changed_names
    np.testing.assert_allclose(changed, original, rtol=0, atol=2e-7)


def test_shell_statistics_distinguish_nearby_source_structure() -> None:
    """Equal local values and shell means can still differ in local clumping."""
    shape = (7,) * 3
    capacity = np.ones(shape)
    native = np.full(shape, 0.5)
    source_a = np.zeros(shape)
    source_b = np.zeros(shape)
    # The compact source variable S/(S+mean(A)) sums to one in both shells.
    source_a[2, 3, 3] = source_a[4, 3, 3] = 1
    source_b[2, 3, 3] = 3
    source_b[4, 3, 3] = 1 / 3
    first, names = spatial_features(source_a, capacity, native)
    second, _ = spatial_features(source_b, capacity, native)
    cell = np.ravel_multi_index((3, 3, 3), shape)
    mean = names.index("spatial_source_shell_r2_1_mean")
    variance = names.index("spatial_source_shell_r2_1_variance")
    assert first[cell, mean] == pytest.approx(second[cell, mean])
    assert first[cell, variance] != pytest.approx(second[cell, variance])


@pytest.mark.parametrize("photon_ratio", [0, 0.4, 2.0])
def test_uniform_inputs_have_zero_gradients_variance_and_alignment(
    photon_ratio: float,
) -> None:
    """A homogeneous box must not acquire an artificial front descriptor."""
    capacity = np.full((5, 5, 5), 3.0)
    source = capacity * photon_ratio
    native = np.full_like(source, 0.7)
    features, names = spatial_features(source, capacity, native)
    assert np.isfinite(features).all()
    np.testing.assert_allclose(features, np.broadcast_to(features[0], features.shape))
    for index, name in enumerate(names):
        # The ES budget contrast is a count discrepancy, not a spatial contrast.
        if "es_budget_contrast" not in name and any(
            word in name for word in ("variance", "gradient", "contrast", "covariance")
        ):
            np.testing.assert_allclose(features[:, index], 0, rtol=0, atol=2e-7)


def test_smoothed_photon_fraction_uses_counts_before_compacting() -> None:
    """A smoothed source/capacity ratio is not the mean of per-cell ratios."""
    from scipy.ndimage import gaussian_filter

    source, capacity, native = _inputs()
    features, names = spatial_features(source, capacity, native, scales_cells=(1.0,))
    smoothed_source = gaussian_filter(source, 1.0, mode="wrap")
    smoothed_capacity = gaussian_filter(capacity, 1.0, mode="wrap")
    expected = smoothed_source / (smoothed_source + smoothed_capacity)
    column = names.index("spatial_photon_fraction_sigma1_cells_count_ratio")
    np.testing.assert_allclose(features[:, column], expected.ravel(), atol=3e-8)


@pytest.mark.parametrize("scale", [(), (0,), (-1,), (np.inf,), (np.nan,), (1, 1)])
def test_invalid_smoothing_scales_fail(scale: tuple[float, ...]) -> None:
    """Column definitions must be nonempty, finite, positive and unique."""
    with pytest.raises(ValueError, match="scale"):
        spatial_features(*_inputs(), scales_cells=scale)


@pytest.mark.parametrize("kind", ["source", "capacity", "native", "shape"])
def test_invalid_native_inputs_fail_before_feature_construction(kind: str) -> None:
    """Input failures must not silently become bounded feature values."""
    source, capacity, native = _inputs()
    if kind == "source":
        source[0, 0, 0] = -1
    elif kind == "capacity":
        capacity[0, 0, 0] = 0
    elif kind == "native":
        native[0, 0, 0] = 1.1
    else:
        source = source[:-1]
    with pytest.raises(ValueError, match=r"Source|Atom|Native"):
        spatial_features(source, capacity, native)
