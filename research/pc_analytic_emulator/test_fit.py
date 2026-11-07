"""Differentiate and train whole-field models through the photon constraint."""

from pathlib import Path

import numpy as np
import pytest

from research.pc_analytic_emulator import fit
from research.pc_field_emulator.model import MLPModel


def _budget() -> MLPModel:
    return MLPModel(
        [np.zeros((1, 1))], [np.zeros(1)], np.zeros(1), np.ones(1), {}, "residual"
    )


def _frame(path: Path, *, offset: float = 0.0, saturated: bool = False) -> dict:
    feature = np.linspace(-1, 1, 64).reshape(4, 4, 4)
    teacher = 0.5 + 0.2 * feature
    baseline = (feature > 0).astype(float) if saturated else np.full_like(feature, 0.5)
    np.savez(
        path,
        features=(feature.reshape(-1, 1) + offset),
        baseline=baseline.ravel(),
        teacher_xHI=1 - teacher,
        capacity=np.ones_like(feature),
        source=np.full_like(feature, 0.5),
        global_features=np.ones((1, 1)),
        box_len=np.asarray(16.0),
    )
    return {"cache": str(path), "seed": 1, "redshift": 8.0}


def test_projected_network_gradient_matches_finite_difference(tmp_path: Path) -> None:
    """Differentiate latent network parameters through a fixed photon budget."""
    record = _frame(tmp_path / "frame.npz")
    frame = fit._load_frame(record)
    model = fit._initialize([record], seed=4, hidden=(3,))
    model.weights[-1][:] = np.array([[0.11], [-0.09], [0.07]])
    model.biases[-1][0] = 0.06
    options = {"spectral_weight": 0.1, "raw_weight": 0.05}
    _, gradients = fit._frame_gradient(model, frame, _budget(), options)
    parameters = [*model.weights, *model.biases]
    step = 1e-6
    for parameter, gradient in zip(parameters, gradients, strict=True):
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + step
            plus, _ = fit._frame_gradient(model, frame, _budget(), options)
            parameter[index] = original - step
            minus, _ = fit._frame_gradient(model, frame, _budget(), options)
            parameter[index] = original
            numeric = (plus["optimization_loss"] - minus["optimization_loss"]) / (
                2 * step
            )
            assert gradient[index] == pytest.approx(numeric, rel=3e-5, abs=3e-7)


def test_auxiliary_latent_loss_escapes_saturated_projection(tmp_path: Path) -> None:
    """The raw auxiliary gives a gradient when every projected cell hits a bound."""
    record = _frame(tmp_path / "frame.npz", saturated=True)
    model = fit._initialize([record], seed=4, hidden=())
    frame = fit._load_frame(record)
    _, absent = fit._frame_gradient(model, frame, _budget(), {"raw_weight": 0.0})
    _, auxiliary = fit._frame_gradient(model, frame, _budget(), {"raw_weight": 0.05})
    assert all(np.all(gradient == 0) for gradient in absent)
    assert any(np.any(np.abs(gradient) > 1e-4) for gradient in auxiliary)


def test_composed_gradient_with_mixed_projection_active_set(tmp_path: Path) -> None:
    """Check the complete Fourier and field gradient with free and clipped cells."""
    record = _frame(tmp_path / "frame.npz")
    frame = fit._load_frame(record)
    model = fit._initialize([record], seed=4, hidden=())
    model.weights[0][0, 0] = 0.87 * model.scale[0]
    model.biases[0][0] = 0.04
    options = {"raw_weight": 0.0, "spectral_weight": 0.1}
    diagnostics, gradients = fit._frame_gradient(model, frame, _budget(), options)
    assert 0 < diagnostics["projection"]["active_cells"] < 64
    step = 1e-6
    original = model.weights[0][0, 0]
    model.weights[0][0, 0] = original + step
    plus, _ = fit._frame_gradient(model, frame, _budget(), options)
    model.weights[0][0, 0] = original - step
    minus, _ = fit._frame_gradient(model, frame, _budget(), options)
    numeric = (plus["projected_loss"] - minus["projected_loss"]) / (2 * step)
    assert gradients[0][0, 0] == pytest.approx(numeric, rel=3e-5, abs=3e-7)


def test_training_standardizer_uses_all_training_cells_only(tmp_path: Path) -> None:
    """Validation feature shifts cannot change the training normalizer."""
    train = _frame(tmp_path / "train.npz", offset=3)
    validation = _frame(tmp_path / "validation.npz", offset=30)
    model, _ = fit.fit_field(
        [train],
        [validation],
        _budget(),
        seed=9,
        hidden=(3,),
        epochs=1,
        objective_config={"spectral_weight": 0.0},
    )
    assert model.mean[0] == pytest.approx(3.0, abs=1e-14)
    assert model.scale[0] == pytest.approx(np.std(np.linspace(-1, 1, 64)), abs=1e-14)
    assert model.metadata["training_cells"] == 64


def test_full_grid_training_reduces_selected_objective_reproducibly(
    tmp_path: Path,
) -> None:
    """Snapshot SGD improves the projected validation score from the ES baseline."""
    record = _frame(tmp_path / "frame.npz")
    options = {"spectral_weight": 0.0, "raw_weight": 0.05}
    first, history = fit.fit_field(
        [record],
        [record],
        _budget(),
        seed=3,
        hidden=(4,),
        epochs=20,
        learning_rate=0.02,
        patience=20,
        objective_config=options,
    )
    second, repeated = fit.fit_field(
        [record],
        [record],
        _budget(),
        seed=3,
        hidden=(4,),
        epochs=20,
        learning_rate=0.02,
        patience=20,
        objective_config=options,
    )
    assert (
        first.metadata["best_validation_loss"]
        < first.metadata["initial_baseline_metrics"]["validation_loss"] * 0.2
    )
    assert history == repeated
    for left, right in zip(first.weights, second.weights, strict=True):
        np.testing.assert_array_equal(left, right)
    score, details = fit.score_field(first, [record], _budget(), options)
    assert score == first.metadata["best_validation_loss"]
    assert len(details) == 1


def test_zero_correction_epoch_is_eligible(tmp_path: Path) -> None:
    """Keep the initial ES field when optimization cannot improve its exact score."""
    record = _frame(tmp_path / "frame.npz")
    with np.load(record["cache"]) as data:
        arrays = {name: np.array(data[name]) for name in data.files}
    arrays["teacher_xHI"] = np.full((4, 4, 4), 0.5)
    np.savez(record["cache"], **arrays)
    model, history = fit.fit_field(
        [record],
        [record],
        _budget(),
        seed=4,
        hidden=(3,),
        epochs=5,
        patience=2,
    )
    assert model.metadata["best_epoch"] == 0
    assert model.metadata["best_validation_loss"] == 0.0
    assert len(history) == 2
    np.testing.assert_array_equal(model.weights[-1], 0)


def test_empty_training_frames_are_rejected() -> None:
    """A normalizer cannot be fitted without a complete training snapshot."""
    with pytest.raises(ValueError, match="snapshot"):
        fit.fit_field([], [], _budget(), seed=1)


@pytest.mark.parametrize("limiter", ["source", "capacity"])
def test_target_count_rounding_matches_inference(limiter: str) -> None:
    """Never round the photon constraint above its long-double physical bound."""
    small = np.zeros((2, 2, 2))
    small.ravel()[:3] = [1.0, 2.0**-53, 2.0**-54]
    large = np.ones_like(small)
    frame = {
        "global_features": np.ones((1, 1)),
        "source": small if limiter == "source" else large,
        "capacity": small if limiter == "capacity" else large,
    }
    limit = np.sum(small, dtype=np.longdouble)
    rounded = float(limit)
    assert np.longdouble(rounded) > limit
    target, efficiency = fit._target_count(frame, _budget())
    assert efficiency == 1.0
    assert target == np.nextafter(rounded, 0.0)
    assert np.longdouble(target) <= limit


def test_frame_cache_is_bounded_readonly_and_scoped(tmp_path: Path) -> None:
    """Reuse immutable arrays, evict the oldest frame and clear memory on exit."""
    first = _frame(tmp_path / "first.npz")
    second = _frame(tmp_path / "second.npz", offset=1.0)
    uncached = fit._load_frame(first)
    size = sum(
        value.nbytes for value in uncached.values() if isinstance(value, np.ndarray)
    )
    with fit._frame_cache_scope(max_bytes=size) as cache:
        one = fit._load_frame(first)
        repeated = fit._load_frame(first)
        assert one["features"] is repeated["features"]
        with pytest.raises(ValueError, match="read-only"):
            one["features"][0, 0] = 0.0
        fit._load_frame(second)
        assert cache.nbytes == size
        assert len(cache.entries) == 1
        assert Path(first["cache"]).resolve() not in cache.entries
    assert cache.nbytes == 0
    assert not cache.entries
    assert fit._ACTIVE_FRAME_CACHE.get() is None


def test_frame_cache_reloads_changed_file_and_skips_oversize(tmp_path: Path) -> None:
    """A changed cache file is re-read, and no oversized frame is retained."""
    record = _frame(tmp_path / "frame.npz")
    with fit._frame_cache_scope(max_bytes=512 * 1024) as cache:
        before = fit._load_frame(record)
        _frame(tmp_path / "frame.npz", offset=7.0)
        after = fit._load_frame(record)
        np.testing.assert_allclose(
            after["features"] - before["features"], 7.0, atol=1e-14
        )
        assert after["features"] is not before["features"]
        assert len(cache.entries) == 1
    with fit._frame_cache_scope(max_bytes=1) as cache:
        fit._load_frame(record)
        assert cache.nbytes == 0
        assert not cache.entries
