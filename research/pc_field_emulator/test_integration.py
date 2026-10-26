"""Exercise training on checkpoints written by the actual refined-front producer."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np

from research.pc_evolution.evolution import run_history
from research.pc_evolution.native_history import ARRAY_KEYS, array_hash
from research.pc_field_emulator import train as training
from research.pc_field_emulator.data import discover_samples, sample_stem
from research.pc_field_emulator.test_data import make_case
from research.pc_field_emulator.train import run_experiment


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _make_existing_outputs(root: Path) -> None:
    """Generate tiny synthetic sources through the real PC checkpoint writer."""
    (root / "fixtures").mkdir(parents=True)
    for seed in (101, 202, 303, 404):
        rng = np.random.default_rng(seed)
        capacity = np.exp(rng.normal(0.0, 0.15, size=(4, 4, 4)))
        capacity /= capacity.mean()
        rates = np.zeros((9, 4, 4, 4), dtype=np.float64)
        locations = rng.choice(64, size=4, replace=False)
        rates.reshape(9, -1)[:, locations] = 0.7
        fields = {
            "capacity": capacity,
            "rates": rates,
            "redshifts": np.linspace(10.0, 6.0, 9),
            "times_seconds": np.arange(9, dtype=np.float64),
            "native_xhi": np.ones((9, 4, 4, 4), dtype=np.float32),
        }
        metadata = {
            "schema_version": 1,
            "density_evolution": "frozen_at_reference_redshift",
            "rates_definition": "native_n_ion_times_H_over_t_STAR",
            "reference_redshift": 8.0,
            "dimension": 4,
            "box_len_cMpc": 40.0,
            "random_seed": seed,
            "test_scope": "Synthetic prescribed rates; not native cosmological data",
            "array_sha256": {key: array_hash(fields[key]) for key in ARRAY_KEYS},
        }
        fixture_path = root / "fixtures" / f"seed{seed}.npz"
        np.savez_compressed(
            fixture_path, **fields, metadata=np.asarray(json.dumps(metadata))
        )
        case = root / "cases" / f"seed{seed}_f2_k24_stride4"
        run_history(
            {**fields, "metadata": metadata},
            subcells_per_axis=2,
            angles=24,
            stride=4,
            maximum_distance=32.0,
            output=case,
        )
        receipt = {
            "fixture": str(fixture_path),
            "fixture_sha256": _digest(fixture_path),
            "summary_sha256": _digest(case / "summary.json"),
            "checkpoint_sha256": {
                str(path.relative_to(case)): _digest(path)
                for path in (case / "checkpoints").glob("*.npz")
            },
        }
        (case / "receipt.json").write_text(json.dumps(receipt))


class ProducerIntegrationTests(unittest.TestCase):
    """Check the producer-to-training-to-held-out-output path."""

    def test_real_checkpoint_writer_and_parallel_training(self) -> None:
        """Use disjoint seeds, keep input bytes unchanged, and save predictions."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "teacher"
            _make_existing_outputs(inputs)
            before = {
                path: _digest(path) for path in inputs.rglob("*") if path.is_file()
            }
            output = root / "emulator"
            report = run_experiment(
                [inputs],
                output,
                subcells=2,
                angles=24,
                stride=4,
                workers=2,
                model_seeds=(11, 23),
                epochs=8,
                patience=3,
                hidden=(8, 8),
                samples_per_snapshot=64,
                batch_size=32,
            )
            assert (output / "COMPLETED").is_file()
            assert report["snapshot_counts"] == {"train": 4, "validation": 2, "test": 2}
            manifest = json.loads((output / "dataset_manifest.json").read_text())
            assert {row["seed"] for row in manifest["splits"]["test"]} == {404}
            assert 404 not in {row["seed"] for row in manifest["splits"]["train"]}
            for sample in report["test"]:
                assert abs(sample["metrics"]["projected"]["delta_Q_mass"]) < 1e-10
                assert "ratio_baseline" in sample["spectra"]
            assert len(list((output / "test").glob("*.png"))) == 2
            assert before == {path: _digest(path) for path in before}

    def test_mixed_geometry_residual_training_and_reconstruction(self) -> None:
        """Train on existing larger grids without relabelling them as held out."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            small, large = root / "small", root / "large"
            for seed in (10, 20, 30, 40):
                make_case(small, seed, dimension=4, box_len=40.0)
            make_case(large, 10, dimension=8, box_len=80.0, production=True)
            before = {
                path: _digest(path)
                for folder in (small, large)
                for path in folder.rglob("*")
                if path.is_file()
            }
            output = root / "emulator"
            report = run_experiment(
                [small, large],
                output,
                workers=2,
                model_seeds=(11, 23),
                epochs=3,
                hidden=(4,),
                samples_per_snapshot=32,
                max_snapshots_per_seed=3,
                batch_size=32,
                patience=2,
                model_mode="residual",
                radii_cmpc=(2.0, 4.0, 8.0, 16.0),
                front_weight=4.0,
                neutral_weight=2.0,
            )
            assert (output / "COMPLETED").is_file()
            assert report["snapshot_counts"] == {
                "train": 9,
                "validation": 3,
                "test": 3,
            }
            manifest = json.loads((output / "dataset_manifest.json").read_text())
            train = manifest["splits"]["train"]
            assert {row["seed"] for row in train} == {10, 20}
            assert {row["dimension"] for row in train} == {4, 8}
            assert {row["seed"] for row in manifest["splits"]["validation"]} == {30}
            assert {row["seed"] for row in manifest["splits"]["test"]} == {40}
            assert all(row["dimension"] == 4 for row in manifest["splits"]["test"])
            coverage = {row["dimension"]: row for row in report["geometry_coverage"]}
            assert coverage[8]["train"] == {"count": 3, "seeds": [10]}
            assert coverage[8]["test"]["count"] == 0
            assert coverage[8]["independent_test_available"] is False
            assert coverage[4]["independent_test_available"] is True
            assert report["feature_config"] == {"radii_cMpc": (2.0, 4.0, 8.0, 16.0)}
            assert all(name.endswith("_cMpc") for name in report["feature_names"][3:15])
            with np.load(output / "model.npz", allow_pickle=False) as model:
                assert str(model["output_mode"]) == "residual"
                metadata = json.loads(str(model["metadata_json"].item()))
            assert metadata["feature_config"]["radii_cMpc"] == [2.0, 4.0, 8.0, 16.0]
            assert metadata["loss_config"] == {
                "front_weight": 4.0,
                "late_neutral_weight": 2.0,
            }
            assert metadata["training_samples"] == 9 * 32
            highres = [
                row
                for row in report["training_reconstruction"]
                if row["dimension"] == 8
            ]
            assert len(highres) == 3
            assert all(
                row["evaluation_role"] == "training_reconstruction" for row in highres
            )
            names = [
                path.stem for path in (output / "training_reconstruction").glob("*.npz")
            ]
            assert len(names) == len(set(names)) == 6
            lookup = {
                str(sample.checkpoint_path): sample
                for folder in (small, large)
                for sample in discover_samples(folder)
            }
            for role, reports in (
                ("test", report["test"]),
                ("training_reconstruction", report["training_reconstruction"]),
            ):
                for row in reports:
                    sample = lookup[row["teacher_checkpoint"]]
                    path = output / role / f"{sample_stem(sample)}.npz"
                    assert path.with_suffix(".png").is_file()
                    with np.load(path, allow_pickle=False) as archive:
                        rho = archive["capacity"]
                        assert rho.shape == (sample.dimension,) * 3
                        for name in ("raw", "projected", "baseline"):
                            field = archive[name]
                            assert field.shape == rho.shape
                            assert np.all(np.isfinite(field))
                            assert np.all((field >= 0) & (field <= 1))
                        budget = min(
                            np.sum(archive["photons"], dtype=np.longdouble),
                            np.sum(rho, dtype=np.longdouble),
                        )
                        count = np.sum(rho * archive["projected"], dtype=np.longdouble)
                        assert abs(count - budget) / budget < 2e-10
            assert before == {path: _digest(path) for path in before}

    def test_frozen_inventory_is_not_rediscovered_during_training(self) -> None:
        """Prepared snapshots remain fixed when a producer adds later outputs."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "teacher"
            for seed in (10, 20, 30, 40):
                make_case(inputs, seed, dimension=4)
            samples = [
                sample for sample in discover_samples(inputs) if sample.node_index <= 4
            ]
            inventory = root / "sample_inventory.json"
            inventory.write_text(
                json.dumps(
                    {
                        "schema": 1,
                        "input_roots": [str(inputs)],
                        "teacher_arm": {"subcells": 4, "angles": 768, "stride": 4},
                        "samples": [asdict(sample) for sample in samples],
                    },
                    default=str,
                )
            )
            # Valid later endpoints already exist on disk but are not part of
            # the frozen inventory. A new realization is also created afterward.
            make_case(inputs, 50, dimension=4)
            before = {
                path: _digest(path) for path in inputs.rglob("*") if path.is_file()
            }
            with (
                patch.object(
                    training,
                    "discover_samples",
                    side_effect=AssertionError(
                        "Frozen inventories must not rediscover"
                    ),
                ),
                patch.object(training, "plot_comparison"),
            ):
                report = run_experiment(
                    [inputs],
                    root / "emulator",
                    workers=1,
                    model_seeds=(11,),
                    epochs=2,
                    hidden=(4,),
                    samples_per_snapshot=8,
                    max_snapshots_per_seed=3,
                    batch_size=8,
                    model_mode="residual",
                    radii_cmpc=(2.0, 4.0, 8.0, 16.0),
                    sample_manifest=inventory,
                )
            manifest = json.loads((root / "emulator/dataset_manifest.json").read_text())
            retained = [row for rows in manifest["splits"].values() for row in rows]
            assert {row["seed"] for row in retained} == {10, 20, 30, 40}
            assert all(row["node_index"] <= 4 for row in retained)
            assert report["snapshot_counts"] == {"train": 6, "validation": 3, "test": 3}
            assert {row["seed"] for row in report["test"]} == {40}
            assert manifest["frozen_inventory"]["sha256"] == _digest(inventory)
            assert before == {path: _digest(path) for path in before}

    def test_late_retained_snapshots_are_reported_and_training_finishes(self) -> None:
        """Exclude valid retained-budget cases before fitting any split."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "teacher"
            for seed in (10, 20, 30, 40):
                make_case(
                    inputs,
                    seed=seed,
                    retained=0.01 if seed in (20, 40) else 0.0,
                    retained_from_node=6,
                )
            before = {
                path: _digest(path) for path in inputs.rglob("*") if path.is_file()
            }
            output = root / "emulator"
            report = run_experiment(
                [inputs],
                output,
                workers=1,
                model_seeds=(11,),
                epochs=2,
                hidden=(4,),
                samples_per_snapshot=8,
                batch_size=8,
            )
            assert (output / "COMPLETED").is_file()
            assert report["snapshot_counts"] == {
                "train": 11,
                "validation": 6,
                "test": 5,
            }
            eligibility = report["sample_eligibility"]
            assert {
                (row["seed"], row["node_index"]) for row in eligibility["excluded"]
            } == {(20, 6), (40, 6)}
            assert eligibility["coverage"]["test"]["accepted"]["Q_mass_range"] == [
                0.1,
                0.5,
            ]
            assert {row["seed"] for row in report["test"]} == {40}
            assert 6 not in {row["node_index"] for row in report["test"]}
            with np.load(output / "model.npz", allow_pickle=False) as model:
                metadata = json.loads(str(model["metadata_json"].item()))
            assert metadata["training_samples"] == 11 * 8
            assert before == {path: _digest(path) for path in before}


if __name__ == "__main__":
    unittest.main()
