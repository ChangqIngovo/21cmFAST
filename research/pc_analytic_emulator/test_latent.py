"""Check unconstrained residual scores and portable inference parameters."""

from pathlib import Path

import numpy as np
import pytest

from research.pc_analytic_emulator.latent import LatentFieldModel


def linear_model() -> LatentFieldModel:
    """Build a deterministic single-layer residual with nontrivial scaling."""
    return LatentFieldModel(
        [np.array([[2.0], [-1.0]])],
        [np.array([0.25])],
        np.array([1.0, -2.0]),
        np.array([2.0, 0.5]),
        {"purpose": "test", "best_epoch": 3},
    )


def test_latent_scores_remain_unbounded_and_raw_fields_are_clipped() -> None:
    """Preserve negative scores needed to retain neutral cells after projection."""
    model = linear_model()
    features = np.array([[-1.0, -2.0], [1.0, -2.0], [3.0, -2.0]])
    baseline = np.array([0.0, 0.5, 1.0])
    np.testing.assert_allclose(
        model.latent_predict(features, baseline=baseline),
        [-1.75, 0.75, 3.25],
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        model.predict(features, baseline=baseline), [0.0, 0.75, 1.0], rtol=0, atol=0
    )


def test_hidden_tanh_and_final_linear_activation_gradient() -> None:
    """Expose the activation list and unit derivative of the final bias."""
    model = LatentFieldModel(
        [np.array([[0.7, -0.4], [0.2, 0.9]]), np.array([[2.0], [-3.0]])],
        [np.array([0.1, -0.2]), np.array([4.0])],
        np.zeros(2),
        np.ones(2),
        {},
    )
    inputs = np.array([[0.5, -0.25], [1.0, 2.0]])
    activations = model._forward(inputs)
    hidden = np.tanh(inputs @ model.weights[0] + model.biases[0])
    np.testing.assert_allclose(activations[1], hidden, rtol=0, atol=0)
    np.testing.assert_allclose(
        activations[-1], hidden @ model.weights[-1] + model.biases[-1], rtol=0, atol=0
    )
    assert np.any(np.abs(activations[-1]) > 1)
    step = 1e-5
    model.biases[-1][0] += step
    plus = model._forward(inputs)[-1]
    model.biases[-1][0] -= 2 * step
    minus = model._forward(inputs)[-1]
    np.testing.assert_allclose((plus - minus) / (2 * step), 1.0, rtol=1e-10, atol=1e-10)


def test_prediction_chunking_and_serialization(tmp_path: Path) -> None:
    """Preserve fitted scales, metadata and unbounded scores through an archive."""
    model = linear_model()
    features = np.arange(22, dtype=float).reshape(11, 2) / 10
    baseline = np.linspace(0, 1, 11)
    expected = model.latent_predict(features, baseline=baseline)
    for chunk in (1, 3, 100):
        np.testing.assert_allclose(
            model.latent_predict(features, baseline=baseline, chunk_size=chunk),
            expected,
            rtol=1e-14,
            atol=1e-14,
        )
    path = tmp_path / "latent.npz"
    model.save(path)
    restored = LatentFieldModel.load(path)
    assert restored.metadata == model.metadata
    assert restored.output_mode == "latent_linear"
    np.testing.assert_allclose(
        restored.latent_predict(features, baseline=baseline), expected, rtol=0, atol=0
    )
    np.testing.assert_array_equal(restored.mean, model.mean)
    np.testing.assert_array_equal(restored.scale, model.scale)


@pytest.mark.parametrize("chunk_size", [0, -1, True, 1.5])
def test_prediction_rejects_invalid_chunk_sizes(chunk_size: object) -> None:
    """Reject silently coerced or empty prediction chunks."""
    with pytest.raises(ValueError, match="chunk_size"):
        linear_model().latent_predict(
            np.zeros((2, 2)), baseline=np.zeros(2), chunk_size=chunk_size
        )


@pytest.mark.parametrize(
    ("features", "baseline"),
    [
        (np.zeros((2, 2)), None),
        (np.zeros((2, 2)), np.zeros(3)),
        (np.zeros((2, 2)), np.array([0.0, 1.1])),
        (np.zeros((2, 2)), np.array([0.0, np.nan])),
        (np.array([[np.inf, 0.0]]), np.zeros(1)),
        (np.empty((0, 2)), np.empty(0)),
        (np.ones((2, 3)), np.zeros(2)),
    ],
)
def test_prediction_rejects_invalid_features_or_baselines(
    features: np.ndarray, baseline: np.ndarray | None
) -> None:
    """Require finite matched features and a physical input-only baseline."""
    with pytest.raises(ValueError, match=r"baseline|features"):
        linear_model().latent_predict(features, baseline=baseline)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("scale", np.array([1.0, 0.0])),
        ("mean", np.array([np.nan, 0.0])),
        ("weights", [np.ones((3, 1))]),
        ("weights", [np.ones((2, 2))]),
        ("biases", [np.array([np.inf])]),
        ("weights", []),
        ("output_mode", "residual"),
        ("metadata", {"nonfinite": float("nan")}),
    ],
)
def test_constructor_rejects_invalid_networks(name: str, value: object) -> None:
    """Fail before inference when parameters or the output contract are invalid."""
    model = linear_model()
    arguments = dict(vars(model))
    arguments[name] = value
    with pytest.raises((ValueError, TypeError)):
        LatentFieldModel(**arguments)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("schema", np.array(1.0)),
        ("schema", np.array([1])),
        ("schema", np.array(999)),
        ("layer_count", np.array(1.5)),
        ("output_mode", np.array("residual")),
        ("metadata_json", np.array("[]")),
        ("metadata_json", np.array('{"bad": NaN}')),
        ("weight_0", np.array([[1.0], [np.inf]])),
        ("mean", np.array(["0", "1"])),
        ("unexpected_field", np.array(1)),
    ],
)
def test_load_rejects_corrupt_or_ambiguous_archives(
    tmp_path: Path, name: str, value: np.ndarray
) -> None:
    """Validate scalar types, the exact schema and finite numeric parameters."""
    path = tmp_path / "latent.npz"
    linear_model().save(path)
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: np.array(saved[key]) for key in saved.files}
    arrays[name] = value
    np.savez_compressed(path, **arrays)
    with pytest.raises((ValueError, TypeError)):
        LatentFieldModel.load(path)


def test_network_overflow_is_rejected() -> None:
    """Do not turn a nonfinite latent score into a plausible clipped field."""
    model = linear_model()
    model.weights[0][:] = np.finfo(float).max
    with pytest.raises(FloatingPointError):
        model.predict(np.full((1, 2), 10.0), baseline=np.zeros(1))
