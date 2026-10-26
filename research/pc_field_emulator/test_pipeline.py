"""Check experiment isolation and validation-only model selection."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from research.pc_field_emulator.data import UnresolvedPhotonBudgetError
from research.pc_field_emulator.test_data import make_case
from research.pc_field_emulator.train import (
    _choose_trial,
    _load_inventory,
    _qualify_splits,
    _select_samples,
    _training_weights,
    _validate_destination,
    run_experiment,
)


class PipelineTests(unittest.TestCase):
    """Exercise selection rules without accessing cosmological data."""

    def test_selection_keeps_seed_groups_and_ionization_window(self) -> None:
        """Select snapshots within each seed without duplicating endpoints."""
        samples = [
            SimpleNamespace(
                seed=seed,
                node_index=node,
                q_mass=node / 10,
                box_len=200.0,
                dimension=32,
                capacity_sha256=str(seed),
            )
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

    def test_mixed_geometry_selection_preserves_each_history(self) -> None:
        """The snapshot cap applies separately to each size and resolution."""
        samples = [
            SimpleNamespace(
                seed=10,
                node_index=node,
                q_mass=node / 10,
                box_len=length,
                dimension=dimension,
                capacity_sha256=f"{length}_{dimension}",
            )
            for length, dimension in ((200.0, 32), (400.0, 128))
            for node in range(11)
        ]
        selected = _select_samples(samples, q_min=0.1, q_max=0.9, maximum=3)
        assert len(selected) == 6
        assert {s.dimension for s in selected} == {32, 128}

    def test_weighted_model_selection_does_not_choose_unweighted_best(self) -> None:
        """Validation selection follows the declared weighted objective."""
        candidates = [
            {"seed": 1, "validation_mse": 0.01, "validation_loss": 0.1},
            {"seed": 2, "validation_mse": 0.02, "validation_loss": 0.05},
        ]
        assert _choose_trial(candidates)["seed"] == 2

    def test_late_neutral_and_front_cells_receive_loss_weight(self) -> None:
        """Weight targets in training only, retaining a uniform loss option."""
        target = np.ones((4, 4, 4))
        target[0] = 0.0
        weighted = _training_weights(target, np.ones_like(target), 4.0, 2.0)
        assert weighted[0, 0, 0] > weighted[2, 0, 0]
        assert weighted[1, 0, 0] > weighted[2, 0, 0]
        np.testing.assert_array_equal(
            _training_weights(target, np.ones_like(target), 1.0, 1.0),
            np.ones_like(target),
        )

    def test_frozen_manifest_must_match_requested_roots_and_arm(self) -> None:
        """A frozen job cannot silently substitute another source run."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "inventory.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": 1,
                        "input_roots": [str(root)],
                        "teacher_arm": {"subcells": 4, "angles": 768, "stride": 4},
                        "samples": [],
                    }
                )
            )
            with pytest.raises(ValueError, match="roots"):
                _load_inventory(path, [root / "different"], 4, 768, 4)
            with pytest.raises(ValueError, match="arm"):
                _load_inventory(path, [root], 2, 768, 4)

    def test_excluded_large_geometry_cannot_silently_disappear(self) -> None:
        """A mixed job must not silently train only the small input geometry."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            small, large = root / "small", root / "large"
            for seed in (10, 20, 30, 40):
                make_case(small, seed=seed)
            make_case(large, seed=10, dimension=4, box_len=80.0, retained=1.0)
            output = root / "emulator"
            with pytest.raises(
                ValueError, match=r"No eligible snapshots remain.*geometries"
            ):
                run_experiment([small, large], output, radii_cmpc=(2.0, 4.0))
            report = json.loads((output / "preflight.json").read_text())
            assert report["excluded_count"] == 6
            assert not (output / "training_arrays").exists()

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

    def test_budget_preflight_keeps_original_seed_roles(self) -> None:
        """Exclude incompatible train/test snapshots without moving seed roles."""
        splits = _preflight_fixture()

        def checked_load(sample: SimpleNamespace) -> dict:
            if (sample.seed, sample.node_index) in ((10, 2), (40, 2)):
                raise _incompatible(sample)
            return {"ledger": {"emitted": 1.0, "retained": 0.0}}

        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "research.pc_field_emulator.train.load_sample", side_effect=checked_load
            ),
        ):
            qualified, report = _qualify_splits(splits, Path(directory))
            assert [(row.seed, row.node_index) for row in qualified["train"]] == [
                (10, 1),
                (20, 1),
                (20, 2),
            ]
            assert [(row.seed, row.node_index) for row in qualified["test"]] == [
                (40, 1)
            ]
            assert qualified["validation"] == splits["validation"]
            assert report["excluded_count"] == 2
            assert report["coverage"]["test"]["accepted"]["count"] == 1
            assert report["coverage"]["train"]["by_seed"]["10"]["excluded_count"] == 1
            assert (Path(directory) / "preflight.json").is_file()

    def test_budget_preflight_empty_partition_is_fatal(self) -> None:
        """Do not replace an excluded test realization with a training one."""
        splits = _preflight_fixture()

        def checked_load(sample: SimpleNamespace) -> dict:
            if sample.seed == 40:
                raise _incompatible(sample)
            return {"ledger": {}}

        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "research.pc_field_emulator.train.load_sample", side_effect=checked_load
            ),
        ):
            with pytest.raises(
                ValueError, match=r"No budget-compatible snapshots.*test"
            ):
                _qualify_splits(splits, Path(directory))
            assert (Path(directory) / "preflight.json").is_file()

    def test_budget_preflight_integrity_errors_are_not_skipped(self) -> None:
        """Keep corrupt checkpoints fatal even when skipping is enabled."""
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "research.pc_field_emulator.train.load_sample",
                side_effect=ValueError("Independent source/grid photon ledger failed"),
            ),
            pytest.raises(
                ValueError, match="Independent source/grid photon ledger failed"
            ),
        ):
            _qualify_splits(_preflight_fixture(), Path(directory))


def _preflight_fixture() -> dict:
    def records(seed: int) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                seed=seed,
                node_index=node,
                q_mass=node / 10,
                redshift=10.0 - node,
                checkpoint_path=Path(f"seed{seed}/node{node}.npz"),
                checkpoint_sha256=f"sha{seed}_{node}",
                box_len=200.0,
                dimension=32,
                capacity_sha256=f"cap{seed}",
            )
            for node in (1, 2)
        ]

    return {
        "train": records(10) + records(20),
        "validation": records(30),
        "test": records(40),
        "strategy": "independent_seed_holdout",
    }


def _incompatible(sample: SimpleNamespace) -> UnresolvedPhotonBudgetError:
    return UnresolvedPhotonBudgetError(
        {
            "reason": "unresolved_photon_budget",
            "seed": sample.seed,
            "node_index": sample.node_index,
            "redshift": sample.redshift,
            "unresolved_fraction": 0.1,
            "unresolved_tolerance": 1e-7,
            "retained": 0.1,
        }
    )


if __name__ == "__main__":
    unittest.main()
