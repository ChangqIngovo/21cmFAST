"""Verify deployed projection gradients and exact Fourier objective semantics."""

from __future__ import annotations

import numpy as np
import pytest

from research.pc_analytic_emulator.exposure_emulator import (
    FIELD_TERMS,
    FieldInputs,
    _project,
)
from research.pc_analytic_emulator.exposure_spectral_fit import (
    SpectralTrainingFrame,
    bounded_neutral_vjp,
    fit_projected_field,
    projected_frame_objective,
)


def _fixture(*, late: bool = False) -> SpectralTrainingFrame:
    random = np.random.default_rng(921)
    shape = (8, 8, 8)
    atoms = random.uniform(0.7, 1.3, shape)
    atoms /= atoms.mean()
    upper = random.uniform(0.7, 0.95, shape)
    latent = random.uniform(-0.2, 1.1, shape)
    if late:
        latent = random.uniform(-0.12, 0.08, shape)
    source = atoms * (0.99 if late else 0.45)
    features = FieldInputs(
        upper=upper,
        latent=latent,
        basis=random.normal(size=(FIELD_TERMS, atoms.size)) * 0.1,
        reservoir=np.zeros(5),
        load=float(source.mean()),
    )
    truth_ionized, _ = _project(
        latent + random.normal(0, 0.01, shape), upper, atoms, float(source.sum())
    )
    teacher = 1 - truth_ionized
    return SpectralTrainingFrame(
        source,
        atoms,
        teacher,
        features,
        np.zeros(5),
        50.0,
        record={"split": "train", "seed": 1},
    )


@pytest.mark.parametrize("late", [False, True])
@pytest.mark.parametrize("power_weight", [0.0, 0.25])
def test_postprojection_coefficient_gradient(late: bool, power_weight: float) -> None:
    """Postprojection coefficient gradient."""
    frame = _fixture(late=late)
    assert frame.late == late
    coefficients = np.linspace(-0.015, 0.015, FIELD_TERMS)
    loss, analytic, _ = projected_frame_objective(
        coefficients, frame, power_weight=power_weight
    )
    assert np.isfinite(loss)
    numeric = np.empty_like(analytic)
    step = 2e-6
    for index in range(FIELD_TERMS):
        perturb = np.zeros(FIELD_TERMS)
        perturb[index] = step
        high = projected_frame_objective(
            coefficients + perturb, frame, power_weight=power_weight
        )[0]
        low = projected_frame_objective(
            coefficients - perturb, frame, power_weight=power_weight
        )[0]
        numeric[index] = (high - low) / (2 * step)
    np.testing.assert_allclose(analytic, numeric, rtol=2e-5, atol=2e-9)


def test_neutral_projection_adjoint_preserves_budget_direction() -> None:
    """Neutral projection adjoint preserves budget direction."""
    frame = _fixture()
    ionized, report = _project(
        frame.inputs.latent, frame.inputs.upper, frame.atoms, frame.absorbed_count
    )
    neutral = 1 - ionized
    # A loss equal to global neutral atom count cannot change with latent shape.
    gradient = bounded_neutral_vjp(
        neutral,
        frame.inputs.upper,
        frame.atoms,
        frame.atoms,
        projection_method=report["method"],
    )
    np.testing.assert_allclose(gradient, 0, rtol=0, atol=1e-14)
    tolerance = 8 * np.finfo(float).eps
    clipped = (neutral < tolerance) | (neutral >= frame.inputs.upper - tolerance)
    random_gradient = bounded_neutral_vjp(
        neutral,
        frame.inputs.upper,
        frame.atoms,
        np.ones_like(neutral),
        projection_method=report["method"],
    )
    assert np.count_nonzero(clipped)
    np.testing.assert_array_equal(random_gradient[clipped], 0)


def test_exact_cut_includes_no_dc_or_half_nyquist_boundary() -> None:
    """Exact cut includes no dc or half nyquist boundary."""
    frame = _fixture()
    # N=8 means |integer mode|<2; the axis mode (2,0,0) is excluded.
    assert not frame.mask[0, 0, 0]
    assert not frame.mask[2, 0, 0]
    assert frame.mask[1, 1, 1]
    assert int(frame.counts.sum()) == 26
    assert frame.shells.tolist() == [1, 2]


def test_shell_residual_weights_are_equal() -> None:
    """Shell residual weights are equal."""
    frame = _fixture()
    # Equal fractional residual power in either shell contributes equally,
    # regardless of the shell's number of modes or teacher power.
    contributions = []
    for shell in frame.shells:
        contribution = (
            frame.counts[shell] * frame.denominator[shell] * frame.shell_weights[shell]
        )
        contributions.append(contribution)
    np.testing.assert_allclose(
        contributions, np.full(len(frame.shells), 1 / len(frame.shells))
    )


def test_fit_decreases_deployed_loss_and_rejects_validation_labels() -> None:
    """Fit decreases deployed loss and rejects validation labels."""
    frame = _fixture()
    initial = np.zeros(FIELD_TERMS)
    weights, receipt = fit_projected_field([frame], initial, maxiter=30)
    assert receipt["final_objective"] <= receipt["initial_objective"]
    assert np.isfinite(weights).all()
    frame.record["split"] = "validation"
    with pytest.raises(ValueError, match="Only training"):
        fit_projected_field([frame], initial)


def test_fully_ionized_loss_is_absolute_and_finite() -> None:
    """Fully ionized loss is absolute and finite."""
    frame = _fixture(late=True)
    frame.teacher[:] = 0
    frame.__post_init__()
    loss, gradient, info = projected_frame_objective(np.zeros(FIELD_TERMS), frame)
    assert info["late"]
    assert np.isfinite(loss)
    assert np.isfinite(gradient).all()


@pytest.mark.parametrize("floor", [0.0, -1e-8, np.nan, np.inf])
def test_shell_variance_floor_must_be_positive_and_finite(floor: float) -> None:
    """Reject floors that leave a zero-power teacher shell unregularized."""
    frame = _fixture()
    frame.teacher[:] = 0.5
    frame.shell_variance_floor = floor
    with pytest.raises(ValueError, match="Invalid supervised spectral snapshot"):
        frame.__post_init__()


def test_zero_power_teacher_has_finite_regularized_spectral_loss() -> None:
    """A uniform partially neutral teacher has zero power in every fitted shell."""
    shape = (8, 8, 8)
    atoms = np.ones(shape)
    source = np.full(shape, 0.5)
    teacher = np.full(shape, 0.5)
    basis = np.zeros((FIELD_TERMS, atoms.size))
    basis[0] = np.broadcast_to(
        np.cos(2 * np.pi * np.arange(8) / 8)[:, None, None], shape
    ).ravel()
    features = FieldInputs(
        upper=np.ones(shape),
        latent=teacher.copy(),
        basis=basis,
        reservoir=np.zeros(5),
        load=0.5,
    )
    frame = SpectralTrainingFrame(
        source, atoms, teacher, features, np.zeros(5), 50.0
    )
    assert not frame.late
    np.testing.assert_array_equal(frame.teacher_power, 0)
    coefficients = np.zeros(FIELD_TERMS)
    coefficients[0] = 0.01
    loss, gradient, _ = projected_frame_objective(
        coefficients, frame, power_weight=1.0
    )
    assert loss > 0
    assert np.isfinite(loss)
    assert np.isfinite(gradient).all()


def test_clipped_boundary_cancellation_has_zero_derivative() -> None:
    """Clipped boundary cancellation has zero derivative."""
    upper = np.array([0.1, 0.3, 0.0])
    neutral = 1 - (1 - upper)
    result = bounded_neutral_vjp(neutral, upper, np.ones(3), np.arange(3.0))
    np.testing.assert_array_equal(result, 0)
