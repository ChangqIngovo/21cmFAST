"""Checks for inference information and directional spatial context."""

import numpy as np
import pytest

from research.es_pc_emulator.features import build_features, from_pair


def test_directional_patch_distinguishes_neighbor_placement():
    """Opposite source-side geometry remains distinguishable at the same cell."""
    field = np.zeros((5, 5, 5))
    field[1, 2, 2] = 1
    radius = np.ones_like(field)
    reflected = field[::-1]
    left, names = build_features(field, radius, redshift=8, box_len_cMpc=50, radii_cMpc=())
    right, _ = build_features(reflected, radius, redshift=8, box_len_cMpc=50, radii_cMpc=())
    center = np.ravel_multi_index((2, 2, 2), field.shape)
    assert not np.array_equal(left[center], right[center])
    assert "xhi_es:offset=(1, 0, 0)" in names


def test_periodic_translation_equivariance():
    """Moving the complete physical input translates the feature field."""
    rng = np.random.default_rng(1)
    fields = [rng.random((4, 4, 4)) for _ in range(4)]
    original, names = build_features(*fields[:2], density=fields[2], photons=fields[3],
                                    redshift=9, box_len_cMpc=50,
                                    feature_set="es-density-source", radii_cMpc=(6.25,))
    shifted = [np.roll(f, 1, axis=0) for f in fields]
    actual, actual_names = build_features(*shifted[:2], density=shifted[2], photons=shifted[3],
                                         redshift=9, box_len_cMpc=50,
                                         feature_set="es-density-source", radii_cMpc=(6.25,))
    expected = np.roll(original.reshape(4, 4, 4, -1), 1, axis=0).reshape(actual.shape)
    np.testing.assert_allclose(actual, expected, atol=2e-7)
    assert names == actual_names


def test_basic_arm_does_not_read_density_photons_or_teacher():
    """Unused physical fields and targets cannot leak into the basic arm."""
    pair = {"xhi_es": np.full((4, 4, 4), 0.6), "radius_es_cMpc": np.ones((4, 4, 4)),
            "metadata": {"redshift": 8, "box_len_cMpc": 50},
            "density": np.full((4, 4, 4), np.nan), "photons": None,
            "xhi_c2ray": np.full((4, 4, 4), np.nan)}
    features, names = from_pair(pair, feature_set="es")
    assert np.isfinite(features).all()
    assert not any("teacher" in name or "c2ray" in name for name in names)
    with pytest.raises(ValueError, match="Density"):
        from_pair(pair, feature_set="es-density")


@pytest.mark.parametrize(("field", "value"), [("xhi", -0.1), ("xhi", 1.1), ("radius", -1), ("radius", np.inf)])
def test_bad_fields_rejected(field, value):
    """Nonphysical fractions or radii fail before fitting."""
    xhi, radius = np.ones((3, 3, 3)), np.ones((3, 3, 3))
    (xhi if field == "xhi" else radius)[0, 0, 0] = value
    with pytest.raises(ValueError, match=r"bounds|finite"):
        build_features(xhi, radius, redshift=8, box_len_cMpc=50)


def test_cell_width_and_physical_radius_are_explicit():
    """Fixed comoving features retain grid-scale information."""
    cube = np.ones((3, 3, 3))
    first, names = build_features(cube, cube * 5, redshift=8, box_len_cMpc=50,
                                 patch_radius=0, radii_cMpc=(6.25,))
    second, _ = build_features(cube, cube * 5, redshift=8, box_len_cMpc=100,
                              patch_radius=0, radii_cMpc=(6.25,))
    assert first[0, -1] != second[0, -1]
    assert "log1p_radius_cMpc:gaussian_sigma_cMpc=6.25" in names
