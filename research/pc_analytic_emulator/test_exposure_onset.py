"""Regression tests for learned onset snapshots and packed tree inference."""

from pathlib import Path

import numpy as np
import pytest

from research.pc_analytic_emulator.exposure_onset import (
    FEATURE_COUNT,
    OnsetHead,
    onset_features,
)


def _head() -> OnsetHead:
    return OnsetHead(
        np.array([0, 3]),
        np.array([0, 0, 0, 0]),
        np.array([0.1, 0.0, 0.0, 0.0]),
        np.array([1, 0, 0, 0]),
        np.array([2, 0, 0, 0]),
        np.array([False, True, True, True]),
        np.array([0.0, -0.25, 0.5, 0.1]),
        0.05,
        {"training_seeds": [1, 2]},
    )


def test_packed_trees_round_trip(tmp_path: Path) -> None:
    """Preserve tree decisions, exact boundary choices, and metadata."""
    head = _head()
    features = np.zeros((4, FEATURE_COUNT), dtype=float)
    features[:, 0] = [-1, 0.1, 0.2, 9]
    np.testing.assert_allclose(
        head.predict(features), [-0.1, -0.1, 0.65, 0.65], rtol=0, atol=1e-15
    )
    target = tmp_path / "onset.npz"
    head.save(target)
    restored = OnsetHead.load(target)
    assert restored.metadata == head.metadata
    np.testing.assert_array_equal(restored.predict(features), head.predict(features))


def test_invalid_forward_tree_rejected() -> None:
    """Reject a malformed tree whose child pointers create a cycle."""
    head = _head()
    head.left[0] = 0
    with pytest.raises(ValueError, match="offsets"):
        head.__post_init__()


def test_onset_features_are_unit_invariant() -> None:
    """Use common native count units and preserve periodic translations."""
    generator = np.random.default_rng(98)
    shape = (16, 16, 16)
    photon = generator.uniform(0, 1.7, size=shape)
    atom = generator.uniform(0.1, 2, size=shape)
    neutral = generator.random(shape)
    inputs = (photon, atom, neutral, 6.5, 50.0)
    head = _head()
    reference = head.initialize(*inputs)
    assert np.all(reference >= 0)
    assert np.all(reference <= np.minimum(atom / photon, 1) + 4e-15)
    scaled = head.initialize(photon * 2.5, atom * 2.5, neutral, 6.5, 50.0)
    np.testing.assert_allclose(scaled, reference, rtol=0, atol=5e-15)
    rotated_inputs = [np.roll(arr, 3, axis=2) for arr in inputs[:3]]
    rotated = head.initialize(*rotated_inputs, 6.5, 50.0)
    np.testing.assert_allclose(
        rotated, np.roll(reference, 3, axis=2), rtol=0, atol=6e-15
    )
    features, local, initial = onset_features(*inputs)
    assert features.shape == (atom.size, FEATURE_COUNT)
    assert np.all(initial <= 1.0)
    assert np.all(local <= 1.0)


def test_exposure_training_rejects_overlapping_seed_sets(tmp_path: Path) -> None:
    """Fail before opening any label archive when a seed is also held out."""
    from research.pc_analytic_emulator.exposure_onset_fit import fit_onset_manifest

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"schema": 1, "training_seeds": [17], "excluded_seeds": [17], "examples": []}'
    )
    with pytest.raises(ValueError, match="disjoint seed sets"):
        fit_onset_manifest(manifest)


def test_exposure_label_interpolation_preserves_censored_cells() -> None:
    """Resolve fractional batch crossing and retain a never transparent cell."""
    from research.pc_analytic_emulator.exposure_onset_fit import _dose_labels

    history = np.array([[[[0.6, 0.9]]], [[[0.2, 0.8]]], [[[0.0, 0.7]]]])
    crossing = np.array([[[3, 0]]], dtype=np.uint8)
    source = np.array([[[0.9, 0.4]]])
    atom = np.ones_like(source)
    times = _dose_labels(history, crossing, source, atom)
    np.testing.assert_allclose(
        times, [[[(2 + 0.5 - 0.125) / 3, 1.0]]], atol=2e-15, rtol=0
    )
