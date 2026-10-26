"""Check that mixed-resolution inputs retain the same physical feature scales."""

from __future__ import annotations

import numpy as np
import pytest

from research.pc_field_emulator.features import build_features, physics_baseline


def test_fixed_physical_columns_are_identical_across_geometries() -> None:
    """Constant fields have equal features except their declared cell width."""
    first, names = build_features(
        np.ones((8,) * 3),
        np.full((8,) * 3, 0.3),
        8.0,
        100.0,
        radii_cMpc=(6.25, 12.5, 25.0, 50.0),
    )
    second, other_names = build_features(
        np.ones((32,) * 3),
        np.full((32,) * 3, 0.3),
        8.0,
        200.0,
        radii_cMpc=(6.25, 12.5, 25.0, 50.0),
    )
    assert names == other_names
    assert len(names) == 19
    assert "log1p_density_sigma12.5_cMpc" in names
    assert "log1p_density_sigma0_cells" in names
    np.testing.assert_allclose(
        first[:, :-1], np.broadcast_to(first[0, :-1], first[:, :-1].shape), atol=1e-7
    )
    np.testing.assert_allclose(
        second[:, :-1], np.broadcast_to(second[0, :-1], second[:, :-1].shape), atol=1e-7
    )
    np.testing.assert_allclose(first[0, :-1], second[0, :-1], atol=1e-7)
    assert first[0, -1] != second[0, -1]


def test_gaussian_fourier_mode_has_same_physical_attenuation() -> None:
    """A shared wavelength is attenuated by exp(-k²R²/2) on both grids."""
    physical_radius = 12.5
    wavelength = 100.0
    amplitude = 0.2
    expected = amplitude * np.exp(
        -0.5 * (2 * np.pi * physical_radius / wavelength) ** 2
    )
    recovered = []
    for dimension, box_len in ((32, 200.0), (128, 400.0)):
        coordinate = np.arange(dimension) * box_len / dimension
        cosine = np.cos(2 * np.pi * coordinate / wavelength)
        density = np.broadcast_to(
            1 + amplitude * cosine[:, None, None], (dimension,) * 3
        )
        features, names = build_features(
            density,
            np.full_like(density, 0.25),
            8.0,
            box_len,
            radii_cMpc=(physical_radius,),
        )
        column = names.index("log1p_density_sigma12.5_cMpc")
        smoothed = np.expm1(features[:, column].reshape(density.shape))
        value = 2 * np.mean((smoothed[:, 0, 0] - 1) * cosine)
        recovered.append(value)
        # The production kernel is discretely sampled and truncated at four sigma.
        np.testing.assert_allclose(value, expected, rtol=2e-4, atol=1e-7)
    np.testing.assert_allclose(recovered[0], recovered[1], rtol=2e-4, atol=1e-7)


def test_physical_features_ignore_common_count_unit_and_translate() -> None:
    """Input-only features do not depend on atom units or periodic origin."""
    rng = np.random.default_rng(42)
    density = rng.uniform(0.2, 2.0, (12,) * 3)
    photons = rng.uniform(0.0, 1.0, density.shape)
    kwargs = {"radii_cMpc": (3.125, 6.25, 12.5)}
    first, names = build_features(density, photons, 7.0, 75.0, **kwargs)
    rescaled, other_names = build_features(
        23 * density, 23 * photons, 7.0, 75.0, **kwargs
    )
    shifted, _ = build_features(
        np.roll(density, 3, axis=1), np.roll(photons, 3, axis=1), 7.0, 75.0, **kwargs
    )
    assert names == other_names
    np.testing.assert_allclose(first, rescaled, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(
        shifted.reshape(*density.shape, -1),
        np.roll(first.reshape(*density.shape, -1), 3, axis=1),
        rtol=1e-6,
        atol=1e-7,
    )


@pytest.mark.parametrize("radii", [(0.0,), (-1.0,), (np.nan,), (np.inf,), (1.0, 1.0)])
def test_invalid_physical_scales_rejected(radii: tuple[float, ...]) -> None:
    """Malformed scales cannot silently change the mixed-resolution schema."""
    with pytest.raises(ValueError, match="Gaussian radii"):
        build_features(
            np.ones((3,) * 3), np.ones((3,) * 3), 8.0, 30.0, radii_cMpc=radii
        )


def test_legacy_cell_scale_features_unchanged() -> None:
    """The existing caller continues to use its original cell-scale columns."""
    rho = np.ones((4,) * 3)
    features, names = build_features(rho, 0.25 * rho, 8.0, 40.0)
    assert features.shape == (64, 19)
    assert "log1p_density_sigma2_cells" in names
    assert not any("sigma2_cMpc" in name for name in names)


def test_subcapacity_emission_does_not_disable_residual() -> None:
    """A parent can transmit refined rays before its coarse capacity is full."""
    density = np.ones((4,) * 3)
    photons = np.zeros_like(density)
    photons[1, 1, 1] = 0.75
    baseline, scale = physics_baseline(density, photons)
    np.testing.assert_array_equal(baseline, photons)
    assert scale == 1.0
    empty, zero_scale = physics_baseline(density, np.zeros_like(photons))
    np.testing.assert_array_equal(empty, 0.0)
    assert zero_scale == 0.0


def test_local_baseline_preserves_endpoints_without_division_overflow() -> None:
    """Clipped local fractions remain finite for empty and tiny-capacity cells."""
    capacity = np.ones((2,) * 3)
    capacity[0, 0, 0] = 0.0
    capacity[0, 0, 1] = 1e-200
    photons = np.zeros_like(capacity)
    photons[0, 0, 0] = 1.0
    photons[0, 0, 1] = 1e200
    photons[0, 1, 0] = 0.25
    photons[1, 1, 1] = 1.0
    with np.errstate(all="raise"):
        baseline, scale = physics_baseline(capacity, photons)
    assert scale == 1.0
    assert baseline[0, 0, 0] == 0.0
    assert baseline[0, 0, 1] == 1.0
    assert baseline[0, 1, 0] == 0.25
    assert baseline[1, 1, 1] == 1.0
    assert np.isfinite(baseline).all()
