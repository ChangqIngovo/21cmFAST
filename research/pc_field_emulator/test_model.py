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
    from .model import MLPModel, project_global_budget, train
except ImportError:
    from model import MLPModel, project_global_budget, train


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
