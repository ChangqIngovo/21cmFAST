"""Test teacher-labelled monotone ranking diagnostics independently of inference."""

import numpy as np
import pytest

from research.pc_analytic.validation_20261116.reproduction.diagnose_ranking import (
    conditional_errors,
    weighted_monotone_oracle,
)


def test_correct_ranking_recovers_teacher_and_budget() -> None:
    """Strictly correct ranks can recover arbitrary bounded teacher values."""
    teacher = np.array([0.8, 0.0, 1.0, 0.25, 0.6])
    atoms = np.array([3.0, 0.2, 9.0, 0.7, 2.0])
    fitted = weighted_monotone_oracle(teacher**3, teacher, atoms)
    np.testing.assert_allclose(fitted, teacher, rtol=0, atol=2e-16)
    assert np.sum(atoms * fitted) == pytest.approx(np.sum(atoms * teacher), rel=2e-15)


def test_reversed_ranking_cannot_recover_geometry() -> None:
    """A decreasing teacher is pooled to its weighted mean under increasing ranks."""
    teacher = np.array([0.9, 0.7, 0.2, 0.0])
    atoms = np.array([1.0, 2.0, 4.0, 3.0])
    fitted = weighted_monotone_oracle(np.arange(4), teacher, atoms)
    expected = np.average(teacher, weights=atoms)
    np.testing.assert_allclose(fitted, expected, rtol=0, atol=2e-16)
    assert np.average((fitted - teacher) ** 2, weights=atoms) == pytest.approx(0.0989)


def test_exact_ties_preserve_no_artificial_spatial_order() -> None:
    """Equal input scores receive equal predictions regardless of input ordering."""
    teacher = np.array([0.0, 0.1, 0.8, 1.0])
    atoms = np.array([0.5, 2.0, 4.0, 0.5])
    fitted = weighted_monotone_oracle(np.ones(4), teacher, atoms)
    np.testing.assert_allclose(fitted, np.average(teacher, weights=atoms), atol=2e-16)
    permutation = np.array([3, 1, 2, 0])
    shuffled = weighted_monotone_oracle(
        np.ones(4), teacher[permutation], atoms[permutation]
    )
    np.testing.assert_allclose(shuffled, fitted[permutation], rtol=0, atol=2e-16)


def test_oracle_is_bounded_and_no_worse_than_monotone_projection() -> None:
    """PAVA is a capacity-MSE lower bound for any monotone remapping of these scores."""
    rng = np.random.default_rng(84)
    scores = rng.integers(-4, 5, 1000).reshape(10, 10, 10) / 4
    atoms = rng.lognormal(0, 1, scores.shape)
    truth = rng.uniform(0, 1, scores.shape)
    fitted = weighted_monotone_oracle(scores, truth, atoms)
    candidate = np.clip(scores + 0.3, 0, 1)
    assert np.all((fitted >= 0) & (fitted <= 1))
    assert np.average((fitted - truth) ** 2, weights=atoms) <= np.average(
        (candidate - truth) ** 2, weights=atoms
    )
    assert np.sum(atoms * fitted) == pytest.approx(np.sum(atoms * truth), rel=2e-15)
    scaled = weighted_monotone_oracle(scores, truth, 1e100 * atoms)
    np.testing.assert_allclose(scaled, fitted, rtol=0, atol=3e-16)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -1.0, 0.0])
def test_invalid_atom_weights_rejected(invalid: float) -> None:
    """A nonpositive or nonfinite atom inventory cannot define the weighted oracle."""
    with pytest.raises(ValueError, match="positive"):
        weighted_monotone_oracle([0, 1], [0.1, 0.9], [1, invalid])


def test_invalid_shapes_labels_and_complex_values_rejected() -> None:
    """Reject ambiguous or nonphysical diagnostics rather than silently coercing them."""
    with pytest.raises(ValueError, match="matching"):
        weighted_monotone_oracle([0, 1], [0.1], [1, 1])
    with pytest.raises(ValueError, match="bounded"):
        weighted_monotone_oracle([0, 1], [0.1, 1.1], [1, 1])
    with pytest.raises(TypeError, match="real"):
        weighted_monotone_oracle([1j, 1], [0.1, 0.9], [1, 1])


def test_conditional_errors_measure_lost_teacher_neutral_mass() -> None:
    """Count neutral mass missed by exactly ionized predictions without a floor."""
    truth = np.array([0.0, 0.02, 0.5, 1.0])
    prediction = np.array([0.0, 0.0, 0.4, 1.0])
    atoms = np.array([1.0, 2.0, 3.0, 4.0])
    result = conditional_errors(prediction, truth, atoms)
    assert result[
        "teacher_neutral_mass_in_predicted_zero_xHI_over_atoms"
    ] == pytest.approx(0.004)
    assert result[
        "fraction_of_teacher_neutral_mass_in_predicted_zero_xHI"
    ] == pytest.approx(0.04 / 5.54)
    assert sum(row["cells"] for row in result["teacher_neutral_fraction_bins"]) == 4
