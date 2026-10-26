"""Check experiment isolation and validation-only model selection."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from research.pc_field_emulator.train import (
    _choose_trial,
    _select_samples,
    _validate_destination,
)


class PipelineTests(unittest.TestCase):
    """Exercise selection rules without accessing cosmological data."""

    def test_selection_keeps_seed_groups_and_ionization_window(self) -> None:
        """Select snapshots within each seed without duplicating endpoints."""
        samples = [
            SimpleNamespace(seed=seed, node_index=node, q_mass=node / 10)
            for seed in (10, 20)
            for node in range(11)
        ]
        selected = _select_samples(samples, q_min=0.1, q_max=0.9, maximum=3)
        assert len(selected) == 6
        for seed in (10, 20):
            assert [row.node_index for row in selected if row.seed == seed] == [1, 5, 9]

    def test_model_selection_uses_validation_loss(self) -> None:
        """Prefer validation performance independently of training loss."""
        candidates = [
            {"seed": 1, "validation_mse": 0.3, "training_mse": 0.001},
            {"seed": 2, "validation_mse": 0.1, "training_mse": 0.02},
        ]
        assert _choose_trial(candidates)["seed"] == 2
        with pytest.raises(ValueError, match="finite validation losses"):
            _choose_trial([{"seed": 1, "validation_mse": np.nan}])

    def test_destination_cannot_modify_input_tree(self) -> None:
        """Keep all training artifacts outside the simulation run."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "simulation"
            source.mkdir()
            with pytest.raises(ValueError, match="outside every simulation input"):
                _validate_destination(source / "training", [source])
            with pytest.raises(FileExistsError):
                _validate_destination(source, [])
            _validate_destination(root / "training", [source])


if __name__ == "__main__":
    unittest.main()
