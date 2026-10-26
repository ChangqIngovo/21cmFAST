"""Unit tests for the lightweight field emulator and its global projection."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest

try:
    from .model import (
        MLPModel,
        _residual_fraction,
        _residual_slope,
        project_global_budget,
        train,
    )
except ImportError:
    from model import (
        MLPModel,
        _residual_fraction,
        _residual_slope,
        project_global_budget,
        train,
    )


class ModelTests(unittest.TestCase):
    """Exercise learning, validation separation and portable inference."""

    @classmethod
    def setUpClass(cls) -> None:
        """Train once on a reproducible analytic nonlinear field."""
        rng = np.random.default_rng(31)
        cls.x_train = rng.normal(size=(256, 3))
        cls.x_validation = rng.normal(size=(80, 3))

        def target(x: np.ndarray) -> np.ndarray:
            logit = 1.4 * x[:, 0] - 0.8 * x[:, 1] + 0.3 * x[:, 2] ** 2
            return 1.0 / (1.0 + np.exp(-logit))

        cls.y_train = target(cls.x_train)
        cls.y_validation = target(cls.x_validation)
        cls.model, cls.history = train(
            cls.x_train,
            cls.y_train,
            cls.x_validation,
            cls.y_validation,
            seed=7,
            hidden=(12, 12),
            epochs=100,
            batch_size=64,
            learning_rate=0.005,
            patience=20,
        )

    def test_learns_nonlinear_field(self) -> None:
        """Improve on the training-mean baseline on independent samples."""
        predicted = self.model.predict(self.x_validation, chunk_size=13)
        baseline = np.mean((self.y_validation - self.y_train.mean()) ** 2)
        assert np.mean((predicted - self.y_validation) ** 2) < baseline / 8
        assert np.all((predicted >= 0) & (predicted <= 1))
        assert self.history[-1]["validation_mse"] < self.history[0]["validation_mse"]

    def test_standardizer_uses_training_only(self) -> None:
        """Keep validation inputs out of fitted feature scaling."""
        np.testing.assert_allclose(
            self.model.mean, self.x_train.mean(axis=0), atol=0, rtol=0
        )
        np.testing.assert_allclose(
            self.model.scale, self.x_train.std(axis=0), atol=0, rtol=0
        )

    def test_save_load_and_chunking(self) -> None:
        """Preserve predictions and metadata without pickle serialization."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            self.model.save(path)
            restored = MLPModel.load(path)
            np.testing.assert_allclose(
                restored.predict(self.x_validation, chunk_size=17),
                self.model.predict(self.x_validation, chunk_size=17),
                atol=0,
                rtol=0,
            )
            assert restored.metadata == self.model.metadata
            with np.load(path, allow_pickle=False) as contents:
                assert all(contents[name].dtype.kind != "O" for name in contents)

    def test_invalid_training_inputs(self) -> None:
        """Reject nonphysical targets, missing values and invalid settings."""
        with pytest.raises(ValueError, match="finite fractions"):
            train(self.x_train, np.full(256, 1.1), self.x_validation, self.y_validation)
        with pytest.raises(ValueError, match="nonfinite"):
            self.model.predict(np.full((3, 3), np.nan))
        with pytest.raises(ValueError, match="epochs must be a positive integer"):
            train(
                self.x_train,
                self.y_train,
                self.x_validation,
                self.y_validation,
                epochs=0,
            )


class ResidualModelTests(unittest.TestCase):
    """Check bounded residuals, their derivatives and weighted fitting."""

    def test_exact_baseline_and_zero_gate(self) -> None:
        """Zero residual and inactive transport preserve exact endpoints."""
        model = MLPModel(
            [np.zeros((1, 1))],
            [np.zeros(1)],
            np.zeros(1),
            np.ones(1),
            {},
            output_mode="residual",
        )
        features = np.arange(4.0)[:, None]
        baseline = np.array([0.0, 0.2, 0.8, 1.0])
        np.testing.assert_array_equal(
            model.predict(features, baseline=baseline), baseline
        )
        model.biases[0][0] = 10.0
        np.testing.assert_array_equal(
            model.predict(features, baseline=baseline, residual_scale=np.zeros(4)),
            baseline,
        )
        positive = model.predict(features, baseline=baseline)
        assert np.all((positive >= baseline) & (positive <= 1))
        model.biases[0][0] = -10.0
        negative = model.predict(features, baseline=baseline)
        assert np.all((negative <= baseline) & (negative >= 0))

    def test_residual_derivative_matches_finite_differences(self) -> None:
        """Differentiate the bounded fraction on both smooth branches."""
        logit = np.array([-1.1, -0.2, 0.3, 1.4])
        baseline = np.array([0.0, 0.8, 0.2, 1.0])
        scale = np.array([0.5, 1.0, 0.4, 1.0])
        step = 1e-6
        numerical = (
            _residual_fraction(np.tanh(logit + step), baseline, scale)
            - _residual_fraction(np.tanh(logit - step), baseline, scale)
        ) / (2 * step)
        analytical = _residual_slope(np.tanh(logit), baseline, scale, np.ones(4))
        np.testing.assert_allclose(analytical, numerical, rtol=1e-8, atol=1e-10)

    def test_endpoint_derivative_allows_inward_learning(self) -> None:
        """At zero correction, choose the feasible loss-decreasing branch."""
        baseline = np.array([0.0, 1.0])
        error = np.array([-0.3, 0.2])
        slope = _residual_slope(np.zeros(2), baseline, np.ones(2), error)
        np.testing.assert_array_equal(slope, np.ones(2))
        step = 1e-7
        direction = -np.sign(error)
        numerical = (
            _residual_fraction(np.tanh(step * direction), baseline, np.ones(2))
            - baseline
        ) / (step * direction)
        np.testing.assert_allclose(slope, numerical, rtol=1e-8, atol=1e-10)

    def test_learns_residual_and_roundtrips(self) -> None:
        """Learn inward changes from saturated cells on an independent sample."""
        rng = np.random.default_rng(27)
        x = rng.normal(size=(480, 3))
        baseline = np.where(x[:, 0] > 0.4, 1.0, np.where(x[:, 0] < -0.4, 0.0, 0.5))
        scale = np.ones(len(x))
        scale[::7] = 0
        target = _residual_fraction(
            np.tanh(0.8 * x[:, 1] - 0.3 * x[:, 2]), baseline, scale
        )
        cut = 360
        model, history = train(
            x[:cut],
            target[:cut],
            x[cut:],
            target[cut:],
            baseline_train=baseline[:cut],
            baseline_validation=baseline[cut:],
            residual_scale_train=scale[:cut],
            residual_scale_validation=scale[cut:],
            seed=17,
            hidden=(12, 12),
            epochs=120,
            batch_size=60,
            learning_rate=0.005,
            patience=25,
        )
        prediction = model.predict(
            x[cut:], baseline=baseline[cut:], residual_scale=scale[cut:]
        )
        reference_mse = np.mean((target[cut:] - baseline[cut:]) ** 2)
        assert np.mean((target[cut:] - prediction) ** 2) < reference_mse / 12
        np.testing.assert_array_equal(
            prediction[scale[cut:] == 0], baseline[cut:][scale[cut:] == 0]
        )
        assert model.output_mode == "residual"
        assert history[-1]["validation_loss"] < history[0]["validation_loss"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "residual.npz"
            model.save(path)
            restored = MLPModel.load(path)
            np.testing.assert_allclose(
                restored.predict(
                    x[cut:],
                    baseline=baseline[cut:],
                    residual_scale=scale[cut:],
                    chunk_size=13,
                ),
                prediction,
                rtol=1e-14,
                atol=1e-14,
            )
            assert restored.output_mode == "residual"

    def test_weighted_objective_changes_fit(self) -> None:
        """Weighted fitting reproduces the weighted constant least-squares optimum."""
        x = np.zeros((20, 1))
        y = np.r_[np.zeros(10), np.ones(10)]
        weights = np.r_[np.ones(10), np.full(10, 9.0)]
        model, history = train(
            x,
            y,
            x.copy(),
            y.copy(),
            hidden=(),
            epochs=160,
            learning_rate=0.08,
            batch_size=20,
            patience=40,
            sample_weight_train=weights,
            sample_weight_validation=weights,
        )
        np.testing.assert_allclose(model.predict(x), 0.9, atol=0.005, rtol=0)
        selected = next(
            row for row in history if row["epoch"] == model.metadata["best_epoch"]
        )
        assert selected["validation_loss"] < 0.091
        assert selected["validation_mse"] > 0.4
        assert model.metadata["best_validation_loss"] == selected["validation_loss"]

    def test_zero_residual_is_a_validation_candidate(self) -> None:
        """Retain the exact baseline when fitting never improves validation."""
        x = np.arange(12.0).reshape(6, 2)
        baseline = np.array([0.0, 0.1, 0.4, 0.6, 0.9, 1.0])
        model, history = train(
            x,
            baseline,
            x.copy(),
            baseline.copy(),
            baseline_train=baseline,
            baseline_validation=baseline,
            epochs=10,
            patience=2,
            batch_size=3,
        )
        assert model.metadata["best_epoch"] == 0
        assert model.metadata["best_validation_loss"] == 0.0
        assert model.metadata["initial_baseline_metrics"]["validation_loss"] == 0.0
        assert len(history) == model.metadata["completed_epochs"] == 2
        assert [row["epoch"] for row in history] == [1, 2]
        np.testing.assert_array_equal(model.predict(x, baseline=baseline), baseline)

    def test_rejects_inconsistent_residual_and_weight_inputs(self) -> None:
        """Reject silently ignored controls and invalid loss definitions."""
        x = np.zeros((4, 1))
        y = np.full(4, 0.5)
        for controls, message in [
            ({"baseline_train": y}, "both training and validation"),
            ({"residual_scale_train": y}, "baseline"),
            (
                {
                    "baseline_train": y,
                    "baseline_validation": y,
                    "residual_scale_train": np.full(4, 2.0),
                },
                "finite fractions",
            ),
            ({"sample_weight_train": np.zeros(4)}, "positive total"),
            ({"sample_weight_validation": np.full(4, np.nan)}, "finite"),
        ]:
            with (
                self.subTest(controls=controls),
                pytest.raises(ValueError, match=message),
            ):
                train(x, y, x, y, **controls)
        model, _ = train(x, y, x, y, epochs=1)
        with pytest.raises(ValueError, match="sigmoid"):
            model.predict(x, baseline=y)

    def test_loads_original_schema_one_sigmoid(self) -> None:
        """Keep already trained sigmoid model archives usable."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.npz"
            np.savez(
                path,
                schema=np.array(1),
                mean=np.zeros(1),
                scale=np.ones(1),
                layer_count=np.array(1),
                metadata_json=np.array("{}"),
                weight_0=np.ones((1, 1)),
                bias_0=np.zeros(1),
            )
            restored = MLPModel.load(path)
            assert restored.output_mode == "sigmoid"
            np.testing.assert_allclose(
                restored.predict(np.array([[0.0], [1.0]])),
                [0.5, 1 / (1 + np.exp(-1))],
                atol=1e-15,
                rtol=0,
            )


class ProjectionTests(unittest.TestCase):
    """The projection must close a global count without changing the inputs."""

    def test_bounds_count_and_no_mutation(self) -> None:
        """Enforce capacity and the requested budget without mutating inputs."""
        predicted = np.array([[0.0, 0.04], [0.75, 1.0]])
        capacity = np.array([[0.0, 0.2], [2.0, 7.0]])
        original = predicted.copy()
        corrected, audit = project_global_budget(predicted, capacity, 3.13)
        assert np.all((corrected >= 0) & (corrected <= 1))
        assert corrected[0, 0] == 0.0
        np.testing.assert_allclose(
            np.sum(corrected * capacity), 3.13, atol=1e-10, rtol=0
        )
        assert abs(audit["count_residual"]) < 1e-10
        np.testing.assert_array_equal(predicted, original)

    def test_zero_and_full_targets(self) -> None:
        """Handle exactly neutral and ionized endpoints including empty cells."""
        capacity = np.array([0.0, 0.1, 2.0])
        for target, expected in [(0.0, [0.0, 0.0, 0.0]), (2.1, [0.0, 1.0, 1.0])]:
            result, _ = project_global_budget(
                np.array([0.5, 0.2, 0.9]), capacity, target
            )
            np.testing.assert_array_equal(result, expected)

    def test_all_zero_or_one_predictions_can_be_corrected(self) -> None:
        """Allow finite-budget correction of saturated neural predictions."""
        capacity = np.array([0.2, 2.0, 0.8])
        for value in [0.0, 1.0]:
            result, _ = project_global_budget(np.full(3, value), capacity, 1.1)
            np.testing.assert_allclose(
                np.dot(result, capacity), 1.1, atol=1e-10, rtol=0
            )

    def test_rejects_unphysical_inputs(self) -> None:
        """Reject invalid capacities, predictions and global count targets."""
        for predicted, capacity, target, message in [
            ([0.5], [-1.0], 0.1, "capacities must be finite and nonnegative"),
            ([0.5], [1.0], 1.1, "exceeds the atom capacity"),
            ([0.5], [1.0], -0.1, "Target ionized count must be finite and nonnegative"),
            ([np.nan], [1.0], 0.1, "Predicted fractions must be finite"),
            ([1.1], [1.0], 0.1, "Predicted fractions must be finite"),
        ]:
            with self.subTest(target=target), pytest.raises(ValueError, match=message):
                project_global_budget(np.array(predicted), np.array(capacity), target)

    def test_tiny_budget_and_wide_capacity_range(self) -> None:
        """Resolve a small count despite a large range of cell capacities."""
        capacity = np.array([1e-12, 1.0, 1e10])
        corrected, audit = project_global_budget(
            np.array([1.0, 0.5, 0.0]), capacity, 1e-6
        )
        assert abs(np.dot(corrected, capacity) - 1e-06) < 2e-12
        assert np.isfinite(audit["log_odds_shift"])

    def test_projection_near_capacity_uses_consistent_accumulation(self) -> None:
        """Bound execution when low-order capacity bits differ from a BLAS sum."""
        # The old double-precision dot product was below the long-double total
        # even at fraction one, making the upper-bracket search nonterminating.
        code = """
import numpy as np
from model import project_global_budget
capacity = np.ones(100)
capacity[0] = 1e16
total = float(np.sum(capacity, dtype=np.longdouble))
target = np.nextafter(total, -np.inf)
result, audit = project_global_budget(np.full(100, 0.5), capacity, target)
assert np.all((result >= 0) & (result <= 1))
assert abs(audit['count_residual']) <= 1e-11 * target + 1e-12
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr


if __name__ == "__main__":
    unittest.main()
