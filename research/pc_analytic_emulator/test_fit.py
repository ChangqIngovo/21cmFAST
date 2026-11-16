"""Differentiate and train whole-field models through the photon constraint."""

from pathlib import Path

import numpy as np
import pytest

from research.pc_analytic_emulator import fit
from research.pc_analytic_emulator.model import (
    Emulator,
    budget_features,
    field_features,
)
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


@pytest.mark.parametrize("budget_mode", ["residual", "sigmoid"])
def test_projected_network_gradient_matches_finite_difference(
    tmp_path: Path,
    budget_mode: str,
) -> None:
    """Differentiate latent network parameters through a fixed photon budget."""
    record = _frame(tmp_path / "frame.npz")
    frame = fit._load_frame(record)
    model = fit._initialize([record], seed=4, hidden=(3,))
    model.weights[-1][:] = np.array([[0.11], [-0.09], [0.07]])
    model.biases[-1][0] = 0.06
    budget = _budget()
    if budget_mode == "sigmoid":
        budget.output_mode = "sigmoid"
        budget.biases[-1][0] = np.log(4.0)
    options = {"spectral_weight": 0.1, "raw_weight": 0.05}
    _, gradients = fit._frame_gradient(model, frame, budget, options)
    parameters = [*model.weights, *model.biases]
    step = 1e-6
    for parameter, gradient in zip(parameters, gradients, strict=True):
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + step
            plus, _ = fit._frame_gradient(model, frame, budget, options)
            parameter[index] = original - step
            minus, _ = fit._frame_gradient(model, frame, budget, options)
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


def _warm_fixture(path: Path, *, saturated: bool = False) -> tuple[dict, Emulator]:
    feature = np.linspace(-1, 1, 64).reshape(4, 4, 4)
    source, capacity = np.full_like(feature, 0.5), np.ones_like(feature)
    native = (feature <= 0).astype(float) if saturated else 0.5 - 0.15 * feature
    features, _, baseline = field_features(source, capacity, native, 8.0, 16.0)
    globals_, _ = budget_features(source, capacity, native, 8.0, 16.0)
    field = MLPModel(
        [np.zeros((features.shape[1], 1))],
        [np.array([0.0 if saturated else 0.2])],
        np.zeros(features.shape[1]),
        np.ones(features.shape[1]),
        {},
        "residual",
    )
    budget = MLPModel(
        [np.zeros((globals_.shape[1], 1))],
        [np.zeros(1)],
        np.zeros(globals_.shape[1]),
        np.ones(globals_.shape[1]),
        {},
        "residual",
    )
    base = Emulator(field, budget, projection_mode="capacity_l2")
    np.savez(
        path,
        features=features,
        baseline=baseline,
        source=source,
        capacity=capacity,
        native_es_xHI=native,
        global_features=globals_,
        teacher_xHI=0.5 - 0.2 * feature,
        box_len=np.asarray(16.0),
    )
    return {"cache": str(path), "seed": 1, "redshift": 8.0}, base


def test_zero_residual_score_reproduces_full_frozen_base(tmp_path: Path) -> None:
    """Use the deployed base field including its budget, never teacher labels."""
    record, base = _warm_fixture(tmp_path / "frame.npz")
    frame = fit._load_frame(record)
    residual = fit._initialize([record], seed=7, hidden=(3,))
    score, rows = fit.score_field(
        residual, [record], base.budget_model, base_model=base
    )
    prediction = base.predict(
        frame["source"], frame["capacity"], frame["native_es_xHI"], 8.0, 16.0
    )
    warm = fit._warm_frame(frame, base)
    np.testing.assert_array_equal(warm["baseline"], prediction["xHII"].ravel())
    assert not np.array_equal(warm["baseline"], frame["baseline"])
    expected, _, _ = fit.projected_objective(
        prediction["xHII"], 1 - frame["teacher_xHI"], frame["capacity"], 16.0
    )
    assert score == pytest.approx(expected, rel=1e-12, abs=1e-14)
    assert rows[0]["projection"]["relative_count_residual"] < 1e-12
    changed_teacher = dict(frame, teacher_xHI=np.zeros_like(frame["teacher_xHI"]))
    np.testing.assert_array_equal(
        fit._warm_frame(changed_teacher, base)["baseline"], warm["baseline"]
    )


def test_warm_residual_gradient_matches_finite_difference(tmp_path: Path) -> None:
    """Differentiate new residual parameters through the frozen-base projection."""
    record, base = _warm_fixture(tmp_path / "frame.npz")
    frame = fit._load_frame(record)
    residual = fit._initialize([record], seed=7, hidden=(3,))
    residual.weights[-1][:] = [[0.05], [-0.03], [0.04]]
    _, derivatives = fit._frame_gradient(
        residual, frame, base.budget_model, None, base_model=base
    )
    parameter = residual.weights[-1]
    before, step = parameter[1, 0], 1e-6
    parameter[1, 0] = before + step
    plus, _ = fit._frame_gradient(
        residual, frame, base.budget_model, None, base_model=base
    )
    parameter[1, 0] = before - step
    minus, _ = fit._frame_gradient(
        residual, frame, base.budget_model, None, base_model=base
    )
    assert derivatives[len(residual.weights) - 1][1, 0] == pytest.approx(
        (plus["optimization_loss"] - minus["optimization_loss"]) / (2 * step),
        rel=3e-5,
        abs=3e-7,
    )


def test_warm_training_retains_base_weights_and_can_leave_wrong_saturation(
    tmp_path: Path,
) -> None:
    """The new auxiliary gradient can correct a clipped base without fitting it."""
    record, base = _warm_fixture(tmp_path / "frame.npz", saturated=True)
    digest = fit._inference_digest(base)
    residual, history = fit.fit_field(
        [record],
        [record],
        base.budget_model,
        seed=7,
        hidden=(4,),
        epochs=20,
        learning_rate=0.02,
        patience=20,
        base_model=base,
    )
    assert fit._inference_digest(base) == digest
    assert residual.metadata["base_model_inference_sha256"] == digest
    assert residual.metadata["base_model_frozen"]
    assert history
    assert (
        residual.metadata["best_validation_loss"]
        < residual.metadata["initial_baseline_metrics"]["validation_loss"]
    )
    score, _ = fit.score_field(residual, [record], base.budget_model, base_model=base)
    assert score == residual.metadata["best_validation_loss"]


def test_warm_prediction_cache_is_shared_bounded_and_stale_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Charge immutable base predictions to the frame LRU and reject stale models."""
    record, base = _warm_fixture(tmp_path / "frame.npz")
    frame = fit._load_frame(record)
    size = sum(v.nbytes for v in frame.values() if isinstance(v, np.ndarray))
    prediction_size = frame["capacity"].nbytes
    calls = []
    predict = base.predict

    def count(*args: object, **kwargs: object) -> dict:
        calls.append(1)
        return predict(*args, **kwargs)

    monkeypatch.setattr(base, "predict", count)
    with fit._frame_cache_scope(max_bytes=size + prediction_size) as cache:
        first = fit._warm_frame(fit._load_frame(record), base)
        second = fit._warm_frame(fit._load_frame(record), base)
        assert first["baseline"] is second["baseline"]
        assert len(calls) == 1
        assert cache.nbytes == size + prediction_size
        with pytest.raises(ValueError, match="read-only"):
            first["baseline"][0] = 0.0
        base.field_model.biases[0][0] += 0.01
        with pytest.raises(ValueError, match="Frozen base model changed"):
            fit._warm_frame(fit._load_frame(record), base)
        base.field_model.biases[0][0] -= 0.01
        _warm_fixture(Path(record["cache"]), saturated=True)
        after = fit._warm_frame(fit._load_frame(record), base)
        assert len(calls) == 2
        assert not np.array_equal(after["baseline"], first["baseline"])
        assert cache.nbytes <= size + prediction_size
    assert cache.nbytes == 0
    assert not cache.frozen_bases


def test_warm_start_requires_explicit_native_inputs_and_epoch(tmp_path: Path) -> None:
    """Missing original input metadata cannot be replaced with target information."""
    record, base = _warm_fixture(tmp_path / "frame.npz")
    frame = fit._load_frame(record)
    del frame["native_es_xHI"]
    with pytest.raises(ValueError, match="native_es_xHI"):
        fit._warm_frame(frame, base)
    frame = fit._load_frame(record)
    del frame["record"]["redshift"]
    with pytest.raises(ValueError, match="redshift"):
        fit._warm_frame(frame, base)


def test_warm_score_matches_deployed_emulator_after_training(tmp_path: Path) -> None:
    """Use identical frozen baseline, projection and objective at fit and inference."""
    record, base = _warm_fixture(tmp_path / "frame.npz")
    before = fit._inference_digest(base)
    residual, _ = fit.fit_field(
        [record],
        [record],
        base.budget_model,
        seed=5,
        hidden=(3,),
        epochs=4,
        base_model=base,
    )
    deployed = Emulator(
        residual,
        base.budget_model,
        radii_cMpc=base.radii_cMpc,
        projection_mode="capacity_l2",
        base_model=base,
    )
    frame = fit._load_frame(record)
    prediction = deployed.predict(
        frame["source"], frame["capacity"], frame["native_es_xHI"], 8.0, 16.0
    )
    expected, _, _ = fit.projected_objective(
        prediction["xHII"], 1 - frame["teacher_xHI"], frame["capacity"], 16.0
    )
    score, rows = fit.score_field(
        residual, [record], base.budget_model, base_model=base
    )
    assert score == pytest.approx(expected, rel=1e-12, abs=1e-14)
    assert (
        rows[0]["projection"]["target_count"]
        == prediction["predicted_absorbed_photons"]
    )
    assert prediction["photon_ledger"]["passed"]
    assert fit._inference_digest(base) == before


def test_warm_fit_rejects_base_mutation_from_callback(tmp_path: Path) -> None:
    """A changing frozen baseline cannot silently invalidate cached predictions."""
    record, base = _warm_fixture(tmp_path / "frame.npz")

    def mutate(_: dict) -> None:
        base.field_model.biases[0][0] += 0.01

    with pytest.raises(ValueError, match="Frozen base model changed"):
        fit.fit_field(
            [record],
            [record],
            base.budget_model,
            seed=5,
            hidden=(3,),
            epochs=1,
            base_model=base,
            on_epoch=mutate,
        )
    assert fit._ACTIVE_FRAME_CACHE.get() is None


def test_frozen_digest_tracks_budget_semantics_but_not_provenance(
    tmp_path: Path,
) -> None:
    """A deficit-output tag changes inference and must invalidate cached bases."""
    record, base = _warm_fixture(tmp_path / "frame.npz")
    base.budget_model.output_mode = "sigmoid"
    base.budget_model.biases[0][0] = np.log(4.0)
    before = fit._inference_digest(base)
    base.metadata["training_note"] = "Reporting-only provenance"
    base.budget_model.metadata["best_epoch"] = 12
    assert fit._inference_digest(base) == before
    with fit._frame_cache_scope():
        first = fit._warm_frame(fit._load_frame(record), base)["baseline"]
        base.budget_model.metadata["budget_parameterization"] = (
            "direct_sigmoid_absorption_deficit"
        )
        assert fit._inference_digest(base) != before
        changed = fit._predict_base(fit._load_frame(record), base)
        assert np.max(np.abs(changed - first)) > 0.1
        with pytest.raises(ValueError, match="Frozen base model changed"):
            fit._warm_frame(fit._load_frame(record), base)


def test_explicit_cache_limit_preserves_fitted_weights_and_predictions(
    tmp_path: Path,
) -> None:
    """Changing memory allowance cannot change RNG, optimization or deployment."""
    first, base = _warm_fixture(tmp_path / "first.npz")
    second, _ = _warm_fixture(tmp_path / "second.npz", saturated=True)
    results = []
    for cap in (0, 1024**2):
        field, history = fit.fit_field(
            [first, second],
            [first],
            base.budget_model,
            seed=23,
            hidden=(3,),
            epochs=3,
            patience=4,
            base_model=base,
            frame_cache_max_bytes=cap,
        )
        assert field.metadata["frame_cache_max_bytes"] == cap
        frame = fit._load_frame(first)
        emulator = Emulator(
            field, base.budget_model, base_model=base, projection_mode="capacity_l2"
        )
        prediction = emulator.predict(
            frame["source"], frame["capacity"], frame["native_es_xHI"], 8.0, 16.0
        )
        results.append((field, history, prediction))
    for left, right in zip(
        (*results[0][0].weights, *results[0][0].biases),
        (*results[1][0].weights, *results[1][0].biases),
        strict=True,
    ):
        np.testing.assert_array_equal(left, right)
    assert results[0][1] == results[1][1]
    np.testing.assert_array_equal(results[0][2]["xHI"], results[1][2]["xHI"])


@pytest.mark.parametrize("limit", [-1, True, 1.5, "512"])
def test_invalid_cache_limits_are_rejected(limit: object) -> None:
    """Memory caps must be explicit nonnegative byte counts."""
    with pytest.raises(ValueError, match="cache"):
        fit._FrameCache(limit)
