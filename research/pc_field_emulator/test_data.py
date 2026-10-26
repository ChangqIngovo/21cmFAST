"""Checks for committed training inputs and realization-level data splits."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

try:
    from .data import (
        UnresolvedPhotonBudgetError,
        discover_samples,
        history_key,
        load_sample,
        sample_key,
        sample_stem,
        split_samples,
        validate_physics,
    )
except ImportError:
    from data import (
        UnresolvedPhotonBudgetError,
        discover_samples,
        history_key,
        load_sample,
        sample_key,
        sample_stem,
        split_samples,
        validate_physics,
    )


def digest(path: Path) -> str:
    """Return a test file digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: dict) -> bytes:
    """Serialize a progress payload identically to the producer."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def make_case(
    root: Path,
    seed: int = 10,
    *,
    production: bool = False,
    retained: float = 0.0,
    retained_from_node: int = 0,
    complete: bool = True,
    dimension: int = 2,
    box_len: float = 40.0,
    metadata_updates: dict | None = None,
) -> Path:
    """Write a small real-schema history/checkpoint case for reader tests."""
    capacity = np.ones((dimension, dimension, dimension), dtype=np.float64)
    capacity.flat[0] = 2.0
    z = np.linspace(12.0, 6.0, 7)
    times = np.arange(7, dtype=float)
    metadata = {
        "schema_version": 1,
        "density_evolution": "frozen_at_reference_redshift",
        "rates_definition": "native_n_ion_times_H_over_t_STAR",
        "source_model": "L-INTEGRAL",
        "random_seed": seed,
        "dimension": dimension,
        "box_len_cMpc": box_len,
        "physical_restrictions": [
            "No MCG, recombinations, X-ray/spin-temperature evolution or UV photoheating feedback"
        ],
        "array_sha256": {
            name: hashlib.sha256(value.tobytes()).hexdigest()
            for name, value in {
                "capacity": capacity,
                "redshifts": z,
                "times_seconds": times,
            }.items()
        },
    }
    metadata.update(metadata_updates or {})
    fixture = (
        root / "native_history.npz"
        if production
        else root / "fixtures" / f"seed{seed}.npz"
    )
    fixture.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        fixture,
        capacity=capacity,
        redshifts=z,
        times_seconds=times,
        metadata=json.dumps(metadata),
    )
    case = root / "pc" if production else root / "cases" / f"seed{seed}_f4_k768_stride4"
    (case / "checkpoints").mkdir(parents=True)
    settings = {
        "subcells_per_axis": 4,
        "angles": 768,
        "stride": 4,
        "maximum_distance": 16.0,
        "periodic": True,
    }
    identity = {"version": 1, "settings": settings, "metadata": metadata}
    identity_sha = hashlib.sha256(canonical(identity)).hexdigest()
    rows = []
    for node in range(7):
        retained_here = retained if node >= retained_from_node else 0.0
        ionized = capacity * (node / 10.0)
        photons = np.zeros_like(capacity, dtype=np.longdouble)
        photons.flat[-1] = ionized.sum(dtype=np.longdouble) + retained_here
        remaining = np.zeros_like(photons)
        remaining.flat[-1] = retained_here
        path = case / "checkpoints" / f"node_{node:05d}.npz"
        np.savez_compressed(
            path,
            ionized=ionized,
            source_emitted=photons,
            source_absorbed=photons - remaining,
            source_retained=remaining,
            source_escaped=np.zeros_like(photons),
            state_version=np.array(1),
            subcells_per_axis=np.array(4),
            node_index=np.array(node),
            redshift=np.array(z[node]),
            time_seconds=np.array(times[node]),
            capacity_sha256=np.array(metadata["array_sha256"]["capacity"]),
            evolution_identity_sha256=np.array(identity_sha),
        )
        rows.append(
            {
                "node_index": node,
                "redshift": z[node],
                "time_seconds": times[node],
                "Q_mass": node / 10.0,
                "Q_volume": node / 10.0,
                "path": str(path),
                "sha256": digest(path),
                "ledger": {
                    "passed": True,
                    "emitted": float(photons.sum()),
                    "ionized": float(ionized.sum()),
                    "escaped": 0.0,
                    "retained": retained_here,
                },
            }
        )
    report = dict(
        scope="Sequential positive-source evolution in fixed native gas; not coupled Qin20",
        metadata=metadata,
        **{k: settings[k] for k in ("subcells_per_axis", "angles", "stride")},
        checkpoints=rows,
        seconds=1.0,
        all_ledgers_passed=True,
    )
    if complete:
        (case / "summary.json").write_text(json.dumps(report))
        receipt = {
            "fixture": str(fixture),
            "fixture_sha256": digest(fixture),
            "summary_sha256": digest(case / "summary.json"),
            "checkpoint_sha256": {
                str(Path(row["path"]).relative_to(case)): row["sha256"] for row in rows
            },
        }
        (case / "receipt.json").write_text(json.dumps(receipt))
    else:
        payload = {
            "identity": identity,
            "status": "running",
            "checkpoints": rows[:-1],
            "steps": [],
            "elapsed_seconds": 1.0,
        }
        (case / "progress.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "payload": payload,
                    "sha256": hashlib.sha256(canonical(payload)).hexdigest(),
                }
            )
        )
    return case


def replace_ionized(case: Path, node: int, ionized: np.ndarray) -> None:
    """Publish an independently count-consistent target for bound tests."""
    report = json.loads((case / "summary.json").read_text())
    receipt = json.loads((case / "receipt.json").read_text())
    row = report["checkpoints"][node]
    path = Path(row["path"])
    with np.load(receipt["fixture"], allow_pickle=False) as archive:
        capacity = archive["capacity"]
    with np.load(path, allow_pickle=False) as archive:
        fields = {key: archive[key] for key in archive.files}
    fields["ionized"] = ionized
    emitted = np.zeros_like(ionized, dtype=np.longdouble)
    emitted.flat[-1] = np.sum(ionized, dtype=np.longdouble)
    fields["source_emitted"] = emitted
    fields["source_absorbed"] = emitted.copy()
    fields["source_escaped"] = np.zeros_like(emitted)
    fields["source_retained"] = np.zeros_like(emitted)
    np.savez_compressed(path, **fields)
    row["sha256"] = digest(path)
    row["Q_mass"] = float(emitted.sum() / capacity.sum(dtype=np.longdouble))
    row["Q_volume"] = float(np.mean(ionized / capacity))
    row["ledger"].update(
        emitted=float(emitted.sum()),
        ionized=float(emitted.sum()),
        escaped=0.0,
        retained=0.0,
    )
    (case / "summary.json").write_text(json.dumps(report))
    receipt["summary_sha256"] = digest(case / "summary.json")
    receipt["checkpoint_sha256"][str(path.relative_to(case))] = digest(path)
    (case / "receipt.json").write_text(json.dumps(receipt))


class DataTests(unittest.TestCase):
    """Exercise counts, provenance and split leakage protection."""

    def test_counts_are_divided_by_capacity_and_sources_remain_inputs(self) -> None:
        """Distinguish source counts, ionized counts and target fractions."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_case(root)
            samples = discover_samples(root)
            assert len(samples) == 7
            loaded = load_sample(samples[3])
            np.testing.assert_allclose(loaded["xHII"], 0.3, rtol=0, atol=1e-15)
            assert np.count_nonzero(loaded["photons"]) == 1
            assert loaded["capacity"].flat[0] == 2.0
            assert loaded["redshift"] == 9.0
            assert loaded["box_len"] == 40.0

    def test_only_committed_progress_checkpoints_are_discovered(self) -> None:
        """Ignore a checkpoint published after the durable journal."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_case(root, production=True, complete=False)
            samples = discover_samples(root)
            assert [sample.node_index for sample in samples] == list(range(6))
            load_sample(samples[-1])

    def test_wrong_arm_does_not_silently_mix_resolutions(self) -> None:
        """Require the exact requested numerical configuration."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_case(root)
            with pytest.raises(ValueError, match="selected numerical arm"):
                discover_samples(root, subcells=2)

    def test_checkpoint_mutation_is_rejected(self) -> None:
        """Detect checkpoint changes between discovery and loading."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_case(root)
            sample = discover_samples(root)[2]
            with sample.checkpoint_path.open("ab") as stream:
                stream.write(b"changed")
            with pytest.raises(ValueError, match="checksum"):
                load_sample(sample)

    def test_progress_mutation_is_rejected(self) -> None:
        """Reject a modified progress payload with its old checksum."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = make_case(root, production=True, complete=False)
            record = json.loads((case / "progress.json").read_text())
            record["payload"]["checkpoints"] = []
            (case / "progress.json").write_text(json.dumps(record))
            with pytest.raises(ValueError, match="checksum"):
                discover_samples(root)

    def test_independent_ledger_catches_consistently_rehashed_wrong_fields(
        self,
    ) -> None:
        """Check conservation even when every provenance digest matches."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = make_case(root, retained=0.2)
            report = json.loads((case / "summary.json").read_text())
            path = Path(report["checkpoints"][3]["path"])
            with np.load(path, allow_pickle=False) as archive:
                fields = {key: archive[key] for key in archive.files}
            fields["source_absorbed"].flat[-1] *= 0.5
            np.savez_compressed(path, **fields)
            report["checkpoints"][3]["sha256"] = digest(path)
            (case / "summary.json").write_text(json.dumps(report))
            receipt = json.loads((case / "receipt.json").read_text())
            receipt["summary_sha256"] = digest(case / "summary.json")
            receipt["checkpoint_sha256"][str(path.relative_to(case))] = digest(path)
            (case / "receipt.json").write_text(json.dumps(receipt))
            with pytest.raises(ValueError, match="Independent source/grid") as error:
                load_sample(discover_samples(root)[3])
            assert not isinstance(error.value, UnresolvedPhotonBudgetError)

    def test_native_fixture_changed_after_discovery_is_rejected(self) -> None:
        """Reject native-history mutation after its identity was recorded."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_case(root)
            sample = discover_samples(root)[2]
            with sample.fixture_path.open("ab") as stream:
                stream.write(b"changed")
            with pytest.raises(ValueError, match="fixture changed"):
                load_sample(sample)

    def test_explicit_recombination_is_rejected(self) -> None:
        """Reject a teacher whose sinks contradict the training budget."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_case(root)
            fixture = root / "fixtures" / "seed10.npz"
            with np.load(fixture, allow_pickle=False) as archive:
                fields = {key: archive[key] for key in archive.files}
            metadata = json.loads(fields["metadata"].item())
            metadata["inputs"] = {"astro_options": {"RECOMB_MODEL": "inhomo"}}
            fields["metadata"] = np.array(json.dumps(metadata))
            np.savez_compressed(fixture, **fields)
            with pytest.raises(ValueError, match="Recombining"):
                discover_samples(root)

    def test_verified_fraction_roundoff_is_clipped_only_in_training_target(
        self,
    ) -> None:
        """Preserve raw counts while clipping accepted target roundoff."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = make_case(root)
            ionized = np.ones((2, 2, 2))
            ionized.flat[0] = 2.0 + 2e-15
            replace_ionized(case, 3, ionized)
            loaded = load_sample(discover_samples(root)[3])
            np.testing.assert_array_equal(loaded["xHII"], np.ones((2, 2, 2)))
            assert loaded["ledger"]["ionized"] > 9.0
            assert loaded["ledger"]["maximum_target_bound_roundoff"] > 0.0
            assert loaded["ledger"]["maximum_target_bound_roundoff"] < 2e-15

    def test_physical_fraction_violation_is_not_hidden_by_target_clipping(self) -> None:
        """Reject substantial capacity violations before clipping targets."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = make_case(root)
            ionized = np.ones((2, 2, 2))
            ionized.flat[0] = 2.0 + 2e-5
            replace_ionized(case, 3, ionized)
            with pytest.raises(ValueError, match="Independent source/grid"):
                load_sample(discover_samples(root)[3])

    def test_unresolved_retained_photons_are_rejected_before_projection(self) -> None:
        """Distinguish valid radiation ledgers from usable emitted-count targets."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_case(root, retained=0.2)
            samples = discover_samples(root)
            with pytest.raises(
                UnresolvedPhotonBudgetError, match=r"retained|escaped"
            ) as error:
                load_sample(samples[3])
            audit = error.value.diagnostics
            assert audit["seed"] == 10
            assert audit["node_index"] == 3
            assert audit["redshift"] == 9.0
            assert audit["reason"] == "unresolved_radiation_before_saturation"
            assert audit["unresolved_tolerance"] == 1e-7
            np.testing.assert_allclose(
                [
                    audit["emitted"],
                    audit["ionized"],
                    audit["retained"],
                    audit["atom_capacity"],
                    audit["Q_mass"],
                ],
                [2.9, 2.7, 0.2, 9.0, 0.3],
                rtol=0,
                atol=1e-15,
            )
            np.testing.assert_allclose(
                [
                    audit["unresolved_fraction"],
                    audit["projected_budget"],
                    audit["projected_minus_teacher_count"],
                    audit["projected_minus_teacher_Q_mass"],
                ],
                [0.2 / 2.9, 2.9, 0.2, 0.2 / 9.0],
                rtol=1e-15,
                atol=1e-16,
            )
            assert audit["relative_grid_residual"] < 2e-15
            assert audit["maximum_source_relative_residual"] < 2e-15
            json.dumps(audit, allow_nan=False)

    def test_tiny_retained_tail_remains_eligible_under_original_tolerance(self) -> None:
        """Retain the existing radiation-tail allowance without broadening it."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_case(root, retained=1e-8)
            loaded = load_sample(discover_samples(root)[3])
            assert loaded["ledger"]["unresolved_fraction"] < 1e-7
            np.testing.assert_allclose(loaded["budget"], 2.7 + 1e-8, rtol=0, atol=1e-15)

    def test_four_seed_split_is_disjoint_even_at_matching_times(self) -> None:
        """Keep entire realizations within one dataset partition."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for seed in (11, 22, 33, 44):
                make_case(root, seed)
            split = split_samples(discover_samples(root))
            assert split["strategy"] == "independent_seed_holdout"
            groups = [
                {sample.seed for sample in split[key]}
                for key in ("train", "validation", "test")
            ]
            assert groups == [{11, 22}, {33}, {44}]

    def test_single_seed_split_is_ordered_and_explicit(self) -> None:
        """Identify single-realization testing as temporal extrapolation."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_case(root)
            split = split_samples(discover_samples(root))
            assert split["strategy"] == "within_realization_temporal_extrapolation"
            assert max(sample.node_index for sample in split["train"]) < min(
                sample.node_index for sample in split["validation"]
            )
            assert max(sample.node_index for sample in split["validation"]) < min(
                sample.node_index for sample in split["test"]
            )

    def test_mixed_geometry_preserves_seed_roles_and_unique_names(self) -> None:
        """The same seed in a bigger box stays in its original partition."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for seed in (11, 22, 33, 44):
                make_case(root / "small", seed)
            make_case(root / "large", 11, dimension=4, box_len=80.0)
            samples = discover_samples(root / "small") + discover_samples(
                root / "large"
            )
            split = split_samples(samples)
            assert len(split["train"]) == 21
            assert {row.seed for row in split["validation"]} == {33}
            assert {row.seed for row in split["test"]} == {44}
            assert {row.dimension for row in split["train"]} == {2, 4}
            assert len({history_key(row) for row in samples}) == 5
            assert len({sample_key(row) for row in samples}) == len(samples)
            assert len({sample_stem(row) for row in samples}) == len(samples)

    def test_duplicate_geometry_cannot_hide_behind_another_fixture_path(self) -> None:
        """Copied or changed realizations at the same coordinates are duplicates."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_case(root / "first", 11)
            make_case(root / "copy", 11)
            samples = discover_samples(root / "first") + discover_samples(root / "copy")
            with pytest.raises(ValueError, match="Repeated realization"):
                split_samples(samples)

    def test_temporal_split_keeps_matching_redshifts_together_across_grids(
        self,
    ) -> None:
        """Grid size or node numbering cannot move one epoch across partitions."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_case(root / "small", 11)
            make_case(root / "large", 11, dimension=4, box_len=80.0)
            small = discover_samples(root / "small")
            large = [
                replace(row, node_index=row.node_index + 20)
                for row in discover_samples(root / "large")
            ]
            for additional_seed in (False, True):
                samples = small + large
                if additional_seed:
                    samples += [replace(row, seed=22) for row in small]
                split = split_samples(samples)
                roles = {}
                for name in ("train", "validation", "test"):
                    for row in split[name]:
                        if row.seed == 11:
                            roles.setdefault(row.redshift, set()).add(name)
                assert all(len(names) == 1 for names in roles.values())
                assert min(row.redshift for row in split["train"]) > max(
                    row.redshift for row in split["validation"]
                )

    def test_mixed_geometry_physics_signature_ignores_only_geometry_and_runtime(
        self,
    ) -> None:
        """One physics configuration can span box size, resolution and threads."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = {
                "astro_params": {"F_STAR10": -1.3},
                "cosmo_params": {"OMm": 0.3},
                "matter_options": {"SOURCE_MODEL": "L-INTEGRAL"},
            }
            for name, n, length, seed in (
                ("small", 2, 40.0, 11),
                ("large", 4, 80.0, 22),
            ):
                make_case(
                    root / name,
                    seed,
                    dimension=n,
                    box_len=length,
                    metadata_updates={
                        "reference_redshift": 8.0,
                        "redshift_start": 12.0,
                        "inputs": {
                            **base,
                            "random_seed": seed,
                            "node_redshifts": [12, 6] if n == 2 else [12, 9, 6],
                            "simulation_options": {
                                "HII_DIM": n,
                                "DIM": 3 * n,
                                "BOX_LEN": length,
                                "N_THREADS": n,
                            },
                        },
                    },
                )
            samples = discover_samples(root / "small") + discover_samples(
                root / "large"
            )
            report = validate_physics(samples)
            assert len(report["sha256"]) == 64
            assert len(report["histories"]) == 2
            assert report["settings"]["inputs"]["astro_params"] == base["astro_params"]

    def test_incompatible_native_physics_is_rejected(self) -> None:
        """Do not combine different source efficiencies as geometry training."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, efficiency, seed in (("first", -1.3, 11), ("second", -1.0, 22)):
                make_case(
                    root / name,
                    seed,
                    metadata_updates={
                        "inputs": {"astro_params": {"F_STAR10": efficiency}}
                    },
                )
            with pytest.raises(ValueError, match="incompatible physical"):
                validate_physics(
                    discover_samples(root / "first") + discover_samples(root / "second")
                )

    def test_density_reference_and_emission_start_remain_physical_parameters(
        self,
    ) -> None:
        """Mixed grids cannot silently alter gas reference time or source history."""
        for key in ("reference_redshift", "redshift_start"):
            with (
                self.subTest(parameter=key),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                make_case(root / "first", 11, metadata_updates={key: 8.0})
                make_case(root / "second", 22, metadata_updates={key: 9.0})
                with pytest.raises(ValueError, match="incompatible physical"):
                    validate_physics(
                        discover_samples(root / "first")
                        + discover_samples(root / "second")
                    )


if __name__ == "__main__":
    unittest.main()
