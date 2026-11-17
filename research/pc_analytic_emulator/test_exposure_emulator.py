"""Regression checks for the pure exposure model and reservoir projection."""

import numpy as np
import pytest

from research.pc_analytic_emulator.exposure_emulator import (
    BUDGET_TERMS,
    FIELD_TERMS,
    ExposureEmulator,
)
from research.pc_analytic_emulator.exposure_features import exposure_features


def _model() -> ExposureEmulator:
    return ExposureEmulator(
        {
            "depth": 5,
            "opacity_prefactor": 1.73,
            "training_geometries": [{"DIM": 48, "HII_DIM": 8, "BOX_LEN": 50.0}],
        },
        np.zeros((1, FIELD_TERMS)),
        np.eye(1, BUDGET_TERMS),
    )


def test_source_only_and_empty_limits():
    """Recover local absorption for uniform weak sources and a dark box."""
    model = _model()
    atoms = np.ones((8,) * 3)
    empty = model.predict(np.zeros_like(atoms), atoms, 8.0, 50.0)
    np.testing.assert_array_equal(empty["xHI"], 1)
    assert empty["predicted_unabsorbed_photons"] == 0
    weak = model.predict(np.full_like(atoms, 0.1), atoms, 8.0, 50.0)
    np.testing.assert_allclose(weak["xHI"], 0.9, atol=1e-13)
    assert weak["predicted_unabsorbed_photons"] == 0
    with pytest.raises(ValueError, match="outside"):
        model.predict(atoms, atoms, 8.0, 51.0)


def test_donor_features_translate_periodically_and_preserve_count_units():
    """Keep the same descriptors under periodic shifts and count-unit changes."""
    rng = np.random.default_rng(91)
    atoms = rng.uniform(0.5, 2, (8,) * 3)
    photons = rng.lognormal(-0.25, 0.8, atoms.shape)
    ref = exposure_features(photons, atoms, depth=5, gains=(1.0, 1.0))
    shift = (2, -1, -3)
    axis = (0, 1, 2)
    transformed = exposure_features(
        1e9 * np.roll(photons, shift, axis),
        1e9 * np.roll(atoms, shift, axis),
        depth=5,
        gains=(1.0, 1.0),
    )
    for field, moved in zip(ref[:4], transformed[:4], strict=True):
        np.testing.assert_allclose(np.roll(field, shift, axis), moved, atol=3e-13)
    assert transformed[-1] == pytest.approx(ref[-1], abs=1e-13)


def test_predicted_budget_matches_independent_field_inventory():
    """Close the predicted budget against the integrated ionized field."""
    rng = np.random.default_rng(92)
    atoms = rng.uniform(0.5, 2, (8,) * 3)
    source = rng.lognormal(0.0, 0.9, atoms.shape)
    result = _model().predict(source, atoms, 6.5, 50.0)
    ionized_count = np.sum(atoms * result["xHII"])
    assert 0 <= ionized_count <= min(atoms.sum(), source.sum())
    assert ionized_count == pytest.approx(
        result["predicted_absorbed_photons"], rel=1e-12
    )
    assert ionized_count + result["predicted_unabsorbed_photons"] == pytest.approx(
        source.sum(), rel=1e-12
    )
