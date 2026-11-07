"""Check the capacity-weighted bounded projection and its implicit gradient."""

import numpy as np
import pytest

from research.pc_analytic_emulator.projection import (
    capacity_l2_vjp,
    project_capacity_l2,
)


@pytest.mark.parametrize("target", [0.0, 0.3, 2.1, 4.7, 5.0])
@pytest.mark.parametrize("unbounded", [False, True])
def test_projection_is_bounded_conservative_and_satisfies_kkt(
    target: float, unbounded: bool
) -> None:
    """Use a common shift on free cells and the correct inequality at each bound."""
    raw = np.array([-0.8, 0.3, 0.7, 1.9] if unbounded else [0.1, 0.3, 0.7, 0.9])
    capacity = np.array([0.5, 2.0, 1.0, 1.5])
    result, diagnostic = project_capacity_l2(raw, capacity, target)
    assert np.all((result >= 0) & (result <= 1))
    assert float(np.sum(capacity * result)) == pytest.approx(target, abs=5e-13)
    shift = diagnostic["additive_shift"]
    np.testing.assert_allclose(result, np.clip(raw + shift, 0, 1), atol=1e-14)
    assert diagnostic["method"] == "capacity_l2"


def test_correct_target_projection_cannot_increase_weighted_teacher_error() -> None:
    """Projection contracts distance to any teacher with the requested atom count."""
    random = np.random.default_rng(7)
    for _ in range(30):
        capacity = random.uniform(0.2, 3, 100)
        teacher = random.uniform(0, 1, 100)
        raw = teacher + random.normal(0.15, 0.7, 100)
        target = float(np.sum(capacity * teacher, dtype=np.longdouble))
        corrected, _ = project_capacity_l2(raw, capacity, target)
        before = np.sum(capacity * (raw - teacher) ** 2)
        after = np.sum(capacity * (corrected - teacher) ** 2)
        assert after <= before + 1e-12


def test_contraction_does_not_compare_to_the_clipped_raw_diagnostic() -> None:
    """Keep the latent-input guarantee distinct from empirical reported raw error."""
    latent = np.array([-10.0, 10.0, 0.8])
    teacher = np.array([0.2, 0.5, 0.6])
    corrected, _ = project_capacity_l2(latent, np.ones(3), 1.3)
    clipped_error = np.sum((np.clip(latent, 0, 1) - teacher) ** 2)
    projected_error = np.sum((corrected - teacher) ** 2)
    latent_error = np.sum((latent - teacher) ** 2)
    assert clipped_error == pytest.approx(0.33)
    assert projected_error == pytest.approx(0.38)
    assert clipped_error < projected_error < latent_error


@pytest.mark.parametrize("target", [0.7, 2.8, 4.6])
@pytest.mark.parametrize("unbounded", [False, True])
def test_implicit_vjp_matches_raw_and_target_finite_differences(
    target: float, unbounded: bool
) -> None:
    """Differentiate the constrained map on both sides of mixed saturation."""
    raw = np.array([-0.8, 0.3, 0.7, 1.9] if unbounded else [0.1, 0.3, 0.7, 0.9])
    capacity = np.array([0.5, 2.0, 1.0, 1.5])
    cotangent = np.array([0.7, -0.3, 1.2, 0.4])
    corrected, _ = project_capacity_l2(raw, capacity, target)
    grad_raw, grad_target = capacity_l2_vjp(corrected, capacity, cotangent)
    step = 1e-6

    def loss(values: np.ndarray, count: float) -> float:
        result, _ = project_capacity_l2(values, capacity, count)
        return float(np.dot(result, cotangent))

    numeric = np.empty_like(raw)
    for i in range(raw.size):
        difference = np.zeros_like(raw)
        difference[i] = step
        numeric[i] = (
            loss(raw + difference, target) - loss(raw - difference, target)
        ) / (2 * step)
    numeric_target = (loss(raw, target + step) - loss(raw, target - step)) / (2 * step)
    np.testing.assert_allclose(grad_raw, numeric, rtol=2e-8, atol=2e-9)
    assert grad_target == pytest.approx(numeric_target, rel=2e-8, abs=2e-9)


def test_vjp_uses_zero_subgradient_for_a_fully_saturated_field() -> None:
    """Identify the nonsmooth all-boundary convention without inventing a gradient."""
    capacity = np.array([1.0, 2.0, 3.0])
    for result in (np.zeros(3), np.ones(3), np.array([0.0, 1.0, 0.0])):
        grad_raw, grad_target = capacity_l2_vjp(result, capacity, np.ones(3))
        np.testing.assert_array_equal(grad_raw, 0)
        assert grad_target == 0


def test_tiny_budget_survives_subtractive_cancellation() -> None:
    """A photon-poor box retains its count when all raw fractions are large."""
    raw = np.full(8, 0.7)
    capacity = np.linspace(0.5, 1.5, 8)
    target = 1e-25
    corrected, diagnostic = project_capacity_l2(raw, capacity, target)
    np.testing.assert_allclose(corrected, target / capacity.sum(), rtol=2e-15, atol=0)
    assert abs(diagnostic["relative_count_residual"]) < 1e-13


def test_zero_capacity_cells_have_zero_output_and_gradient() -> None:
    """Ignore cells without atoms in the projection metric and photon constraint."""
    capacity = np.array([0.0, 1.0, 2.0])
    corrected, _ = project_capacity_l2(np.array([0.9, 0.2, 0.4]), capacity, 1.5)
    assert corrected[0] == 0
    gradient, _ = capacity_l2_vjp(corrected, capacity, np.ones(3))
    assert gradient[0] == 0


@pytest.mark.parametrize("target", [-1, 5.1, np.nan])
def test_invalid_count_is_rejected(target: float) -> None:
    """Refuse a target outside the nonnegative atom capacity."""
    with pytest.raises(ValueError, match=r"[Tt]arget"):
        project_capacity_l2(np.full(4, 0.5), np.full(4, 1.25), target)
