"""Check opt-in normalization guards without changing existing checkpoints."""

from pathlib import Path

import numpy as np
import pytest

from research.pc_analytic_emulator import fit
from research.pc_analytic_emulator.test_fit import _budget, _frame
from research.pc_field_emulator.model import MLPModel, train


def test_mlp_exact_constant_weights_stay_zero_and_survive_reload(
    tmp_path: Path,
) -> None:
    """Constant geometry cannot introduce an arbitrary untrained response."""
    x = np.column_stack(
        (np.linspace(-1, 1, 65), np.full(65, 0.1), np.linspace(-1e-7, 1e-7, 65))
    )
    y = 0.5 + 0.15 * x[:, 0]
    model, _ = train(
        x,
        y,
        x,
        y,
        seed=4,
        hidden=(4,),
        epochs=4,
        zero_constant_features=True,
        feature_scale_floor=np.full(3, 0.01),
    )
    assert model.mean[1] == x[0, 1]
    np.testing.assert_array_equal(model.weights[0][1], 0)
    assert model.scale[2] == 0.01
    changed = x.copy()
    changed[:, 1] = 1e8
    np.testing.assert_array_equal(model.predict(changed), model.predict(x))
    model.save(tmp_path / "model.npz")
    restored = MLPModel.load(tmp_path / "model.npz")
    np.testing.assert_array_equal(restored.predict(changed), model.predict(x))
    assert restored.metadata["zero_constant_features"]
    assert restored.metadata["constant_feature_indices"] == [1]


def test_latent_guards_use_training_support_and_freeze_constant_rows(
    tmp_path: Path,
) -> None:
    """A near-zero varying column is floored; exact constants get no weights."""
    train_frame = _frame(tmp_path / "train.npz")
    validation_frame = _frame(tmp_path / "validation.npz")
    for record, constant in ((train_frame, 0.1), (validation_frame, 2.0)):
        with np.load(record["cache"]) as saved:
            arrays = {key: saved[key] for key in saved.files}
        x = arrays["features"]
        arrays["features"] = np.column_stack((x, np.full(len(x), constant), x * 1e-7))
        np.savez(record["cache"], **arrays)
    model, _ = fit.fit_field(
        [train_frame],
        [validation_frame],
        _budget(),
        seed=4,
        hidden=(4,),
        epochs=3,
        zero_constant_features=True,
        field_feature_scale_floor=0.01,
    )
    assert model.mean[1] == 0.1
    assert model.scale[2] == 0.01
    np.testing.assert_array_equal(model.weights[0][1], 0)
    assert model.metadata["constant_feature_indices"] == [1]
    assert model.metadata["field_feature_scale_floor"] == 0.01


def test_old_initialization_remains_default(tmp_path: Path) -> None:
    """Default initialization keeps its original unmodified RNG sequence."""
    frame = _frame(tmp_path / "frame.npz")
    first = fit._initialize([frame], seed=23, hidden=(4,))
    explicit = fit._initialize(
        [frame],
        seed=23,
        hidden=(4,),
        zero_constant_features=False,
        field_feature_scale_floor=0,
    )
    for left, right in zip(first.weights, explicit.weights, strict=True):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(first.scale, explicit.scale)


@pytest.mark.parametrize("floor", [-1, True, float("nan"), float("inf")])
def test_invalid_latent_scale_floors_rejected(tmp_path: Path, floor: object) -> None:
    """Normalization floors must be finite nonnegative real values."""
    frame = _frame(tmp_path / "frame.npz")
    with pytest.raises(ValueError, match="floor"):
        fit._initialize([frame], seed=23, hidden=(4,), field_feature_scale_floor=floor)


@pytest.mark.parametrize("flag", [1, "true", None])
def test_constant_guard_flag_requires_boolean(flag: object) -> None:
    """Explicit opt-ins must not silently accept truthy invalid values."""
    x = np.ones((8, 2))
    y = np.full(8, 0.5)
    with pytest.raises(ValueError, match="zero_constant_features"):
        train(x, y, x, y, epochs=1, zero_constant_features=flag)
