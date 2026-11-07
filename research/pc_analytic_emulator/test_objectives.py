"""Check the projected-field loss against independent Fourier and derivative tests."""

import json

import numpy as np
import pytest

from research.pc_analytic_emulator.objectives import projected_objective


def sample_fields(n: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Make bounded fields with fronts and independently nonuniform gas density."""
    rng = np.random.default_rng(729)
    teacher = rng.uniform(0.1, 0.9, (n, n, n))
    predicted = np.clip(teacher + rng.normal(0, 0.06, teacher.shape), 0.02, 0.98)
    capacity = np.exp(rng.normal(0, 0.7, teacher.shape))
    return predicted, teacher, capacity


@pytest.mark.parametrize("spectral_weight", [0.0, 0.3])
def test_full_objective_gradient_by_finite_difference(spectral_weight: float) -> None:
    """Differentiate all field and nonuniform-density Fourier terms together."""
    predicted, teacher, capacity = sample_fields()
    options = {"spectral_weight": spectral_weight, "k_fraction": 0.8}
    loss, gradient, report = projected_objective(
        predicted, teacher, capacity, 256, **options
    )
    assert loss > 0
    assert report["active_shell_count"] >= 2
    rng = np.random.default_rng(88)
    direction = rng.normal(size=predicted.shape)
    direction /= np.linalg.norm(direction)
    step = 1e-5
    plus = projected_objective(
        predicted + step * direction, teacher, capacity, 256, **options
    )[0]
    minus = projected_objective(
        predicted - step * direction, teacher, capacity, 256, **options
    )[0]
    assert (plus - minus) / (2 * step) == pytest.approx(
        np.sum(gradient * direction), rel=2e-7, abs=1e-10
    )
    for index in ((0, 0, 0), (2, 3, 1), (6, 7, 5)):
        perturbation = np.zeros_like(predicted)
        perturbation[index] = step
        plus = projected_objective(
            predicted + perturbation, teacher, capacity, 256, **options
        )[0]
        minus = projected_objective(
            predicted - perturbation, teacher, capacity, 256, **options
        )[0]
        assert (plus - minus) / (2 * step) == pytest.approx(
            gradient[index], rel=2e-7, abs=1e-10
        )


def test_single_fourier_mode_has_known_relative_residual() -> None:
    """A ten-percent mode-amplitude error gives a 0.01 relative complex loss."""
    n = 16
    phase = 2 * np.pi * np.arange(n)[:, None, None] / n
    teacher = np.broadcast_to(0.5 + 0.1 * np.cos(phase), (n, n, n))
    predicted = 0.5 + 1.1 * (teacher - 0.5)
    _, _, report = projected_objective(predicted, teacher, np.ones_like(teacher), 100)
    assert report["active_shell_count"] == 1
    assert report["undefined_shell_count"] > 0
    assert report["spectral_loss"] == pytest.approx(0.01, rel=1e-12, abs=1e-14)


@pytest.mark.parametrize("n", [8, 16])
def test_weak_mode_floor_has_grid_independent_physical_normalization(n: int) -> None:
    """A weak shell is normalized by fixed neutral-reference variance, not noise."""
    phase = 2 * np.pi * np.arange(n)[:, None, None] / n
    amplitude, floor = 1e-5, 1e-4
    teacher = np.broadcast_to(0.5 + amplitude * np.cos(phase), (n, n, n))
    predicted = 0.5 + 2 * (teacher - 0.5)
    _, _, report = projected_objective(
        predicted, teacher, np.ones_like(teacher), 256, spectral_power_floor=floor
    )
    assert report["active_shell_count"] == report["regularized_shell_count"] == 1
    assert report["spectral_loss"] == pytest.approx(
        amplitude**2 / (2 * floor), rel=1e-10, abs=1e-15
    )
    assert report["neutral_reference_mean_square"] == 1
    assert report["shell_variance_floor"] == floor
    _, _, legacy = projected_objective(
        predicted, teacher, np.ones_like(teacher), 256, spectral_power_floor=0
    )
    assert legacy["regularized_shell_count"] == 0
    assert legacy["spectral_loss"] == pytest.approx(1, rel=1e-10, abs=1e-12)


def test_almost_ionized_signal_cannot_dominate_the_objective() -> None:
    """Near-zero teacher brightness receives finite absolute-scale supervision."""
    rng = np.random.default_rng(19)
    capacity = np.exp(rng.normal(0, 0.3, (8, 8, 8)))
    teacher = np.full(capacity.shape, 1 - 1e-10)
    predicted = np.full(capacity.shape, 1 - 1e-6)
    loss, gradient, report = projected_objective(predicted, teacher, capacity, 256)
    assert report["active_shell_count"] > 0
    assert report["regularized_shell_count"] == report["active_shell_count"]
    assert 0 < loss < 1e-8
    assert np.max(np.abs(gradient)) < 1e-4
    _, _, unregularized = projected_objective(
        predicted, teacher, capacity, 256, spectral_power_floor=0
    )
    assert unregularized["spectral_loss"] > 1e7
    # The mathematical zero-signal ratio stays undefined, not zero or one.
    _, _, ionized = projected_objective(predicted, np.ones_like(teacher), capacity, 256)
    assert ionized["spectral_loss"] is None
    assert ionized["active_shell_count"] == ionized["regularized_shell_count"] == 0


def test_regularized_fourier_gradient_and_count_unit_invariance() -> None:
    """Differentiate the floored objective and retain photon-count unit invariance."""
    rng = np.random.default_rng(421)
    capacity = np.exp(rng.normal(0, 0.2, (8, 8, 8)))
    teacher = 1 - rng.uniform(1e-5, 2e-5, capacity.shape)
    predicted = 1 - rng.uniform(2e-5, 4e-5, capacity.shape)
    options = {"spectral_power_floor": 1e-4, "spectral_weight": 0.5}
    loss, gradient, report = projected_objective(
        predicted, teacher, capacity, 256, **options
    )
    assert report["regularized_shell_count"] == report["active_shell_count"] > 0
    direction = rng.normal(size=predicted.shape)
    direction /= np.linalg.norm(direction)
    step = 1e-7
    plus = projected_objective(
        predicted + step * direction, teacher, capacity, 256, **options
    )[0]
    minus = projected_objective(
        predicted - step * direction, teacher, capacity, 256, **options
    )[0]
    assert (plus - minus) / (2 * step) == pytest.approx(
        np.sum(gradient * direction), rel=1e-6, abs=1e-12
    )
    for index in ((0, 0, 0), (3, 2, 1)):
        delta = np.zeros_like(predicted)
        delta[index] = step
        plus = projected_objective(
            predicted + delta, teacher, capacity, 256, **options
        )[0]
        minus = projected_objective(
            predicted - delta, teacher, capacity, 256, **options
        )[0]
        assert (plus - minus) / (2 * step) == pytest.approx(
            gradient[index], rel=1e-6, abs=1e-12
        )
    scaled_loss, scaled_gradient, scaled = projected_objective(
        predicted, teacher, capacity * 1e100, 256, **options
    )
    assert scaled_loss == pytest.approx(loss, rel=1e-13, abs=1e-20)
    np.testing.assert_allclose(scaled_gradient, gradient, rtol=1e-12, atol=1e-18)
    assert scaled["shell_variance_floor"] == pytest.approx(
        report["shell_variance_floor"], rel=1e-14
    )


def test_fourier_objective_detects_phase_errors() -> None:
    """Equal auto-power is insufficient when ionization structures are displaced."""
    n = 16
    phase = 2 * np.pi * np.arange(n)[:, None, None] / n
    teacher = np.broadcast_to(0.5 + 0.1 * np.cos(phase), (n, n, n))
    predicted = np.roll(teacher, 4, axis=0)
    _, _, report = projected_objective(predicted, teacher, np.ones_like(teacher), 100)
    assert report["spectral_loss"] == pytest.approx(2, rel=1e-12, abs=1e-14)


@pytest.mark.parametrize("teacher_value", [0.0, 0.3, 1.0])
def test_uniform_teacher_has_undefined_fluctuation_spectrum(
    teacher_value: float,
) -> None:
    """Empty Fourier signal is excluded explicitly while the field term remains."""
    teacher = np.full((8, 8, 8), teacher_value)
    predicted = teacher * 0.8 + 0.1
    loss, gradient, report = projected_objective(
        predicted, teacher, np.ones_like(teacher), 256
    )
    assert report["spectral_loss"] is None
    assert report["spectral_loss_contribution"] == 0
    assert report["active_shell_count"] == 0
    assert report["undefined_shell_count"] > 0
    assert loss == pytest.approx(report["field_loss"])
    assert np.isfinite(gradient).all()
    json.dumps(report, allow_nan=False)


def test_nonuniform_neutral_gas_has_a_brightness_spectrum() -> None:
    """Neutral-fraction constancy does not erase density-driven 21-cm structure."""
    _, _, capacity = sample_fields()
    teacher = np.zeros_like(capacity)
    predicted = np.full_like(capacity, 0.1)
    _, _, report = projected_objective(predicted, teacher, capacity, 256)
    assert report["active_shell_count"] > 0
    assert report["spectral_loss"] == pytest.approx(0.01, rel=1e-12)


def test_exact_prediction_has_zero_loss_and_gradient() -> None:
    """The teacher field is an exact stationary point of both loss components."""
    _, teacher, capacity = sample_fields()
    loss, gradient, report = projected_objective(teacher, teacher, capacity, 256)
    assert loss == 0
    assert np.array_equal(gradient, np.zeros_like(gradient))
    assert report["spectral_loss"] == 0


def test_loss_and_gradient_are_periodically_equivariant() -> None:
    """Translated and rotated copies preserve the loss and transform its gradient."""
    fields = sample_fields()
    loss, gradient, _ = projected_objective(*fields, 256)
    translated = [np.roll(field, (1, -3, 2), axis=(0, 1, 2)) for field in fields]
    translated_loss, translated_gradient, _ = projected_objective(*translated, 256)
    assert translated_loss == pytest.approx(loss, rel=1e-13)
    np.testing.assert_allclose(
        translated_gradient,
        np.roll(gradient, (1, -3, 2), axis=(0, 1, 2)),
        rtol=1e-12,
        atol=1e-14,
    )
    rotated = [np.rot90(field, axes=(0, 2)) for field in fields]
    rotated_loss, rotated_gradient, _ = projected_objective(*rotated, 256)
    assert rotated_loss == pytest.approx(loss, rel=1e-13)
    np.testing.assert_allclose(
        rotated_gradient, np.rot90(gradient, axes=(0, 2)), rtol=1e-12, atol=1e-14
    )


def test_capacity_units_and_box_length_preserve_relative_loss() -> None:
    """Common photon-count units and physical k units do not alter relative loss."""
    predicted, teacher, capacity = sample_fields()
    loss, gradient, original = projected_objective(predicted, teacher, capacity, 256)
    scaled_loss, scaled_gradient, scaled = projected_objective(
        predicted, teacher, capacity * 1e100, 512
    )
    assert scaled_loss == pytest.approx(loss, rel=1e-13)
    np.testing.assert_allclose(scaled_gradient, gradient, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(
        scaled["shell_k_cMpc_inv"], np.array(original["shell_k_cMpc_inv"]) / 2
    )


def test_no_modes_under_cutoff_is_explicit() -> None:
    """A valid cutoff can leave no resolved nonzero modes on a small grid."""
    predicted, teacher, capacity = sample_fields(4)
    loss, _, report = projected_objective(
        predicted, teacher, capacity, 256, k_fraction=0.1
    )
    assert loss == pytest.approx(report["field_loss"])
    assert report["candidate_shell_count"] == 0
    assert report["active_shell_count"] == 0
    assert report["spectral_loss"] is None


@pytest.mark.parametrize(
    "options",
    [
        {"spectral_weight": -1},
        {"spectral_weight": float("nan")},
        {"spectral_power_floor": -1},
        {"spectral_power_floor": float("nan")},
        {"spectral_power_floor": True},
        {"front_weight": -1},
        {"field_variance_floor": 0},
        {"k_fraction": 0},
        {"k_fraction": 1.1},
        {"box_len": -1},
        {"spectral_weight": True},
    ],
)
def test_invalid_options_are_rejected(options: dict[str, float]) -> None:
    """Reject invalid numerical controls rather than changing their meaning."""
    kwargs = {"box_len": 256, **options}
    with pytest.raises((ValueError, TypeError)):
        projected_objective(*sample_fields(), **kwargs)


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan")])
def test_invalid_prediction_is_rejected(bad: float) -> None:
    """Require bounded finite physical fields before differentiating the objective."""
    predicted, teacher, capacity = sample_fields()
    predicted[0, 0, 0] = bad
    with pytest.raises(ValueError, match="Prediction"):
        projected_objective(predicted, teacher, capacity, 256)


def test_invalid_density_and_shape_are_rejected() -> None:
    """Reject inconsistent geometry or a gas field without positive total mass."""
    predicted, teacher, capacity = sample_fields()
    with pytest.raises(ValueError, match="shape"):
        projected_objective(predicted, teacher[:2], capacity, 256)
    with pytest.raises(ValueError, match="Capacity"):
        projected_objective(predicted, teacher, np.zeros_like(capacity), 256)
    capacity[0, 0, 0] = -1
    with pytest.raises(ValueError, match="Capacity"):
        projected_objective(predicted, teacher, capacity, 256)
