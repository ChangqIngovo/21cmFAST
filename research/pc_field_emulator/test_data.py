"""Checks for committed training inputs and realization-level data splits."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest

try:
    from .data import discover_samples, load_sample, split_samples
except ImportError:
    from data import discover_samples, load_sample, split_samples


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
    complete: bool = True,
) -> Path:
    """Write a small real-schema history/checkpoint case for reader tests."""
    capacity = np.ones((2, 2, 2), dtype=np.float64)
    capacity.flat[0] = 2.0
    z = np.linspace(12.0, 6.0, 7)
    times = np.arange(7, dtype=float)
    metadata = {
        "schema_version": 1,
        "density_evolution": "frozen_at_reference_redshift",
        "rates_definition": "native_n_ion_times_H_over_t_STAR",
        "source_model": "L-INTEGRAL",
        "random_seed": seed,
        "dimension": 2,
        "box_len_cMpc": 40.0,
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
        ionized = capacity * (node / 10.0)
        photons = np.zeros_like(capacity, dtype=np.longdouble)
        photons.flat[-1] = ionized.sum(dtype=np.longdouble) + retained
        remaining = np.zeros_like(photons)
        remaining.flat[-1] = retained
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
                    "retained": retained,
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
            case = make_case(root)
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
            with pytest.raises(ValueError, match="Independent source/grid"):
                load_sample(discover_samples(root)[3])

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
        """Reject unresolved radiation before applying absorbed budgets."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_case(root, retained=0.2)
            samples = discover_samples(root)
            with pytest.raises(ValueError, match=r"retained|escaped"):
                load_sample(samples[3])

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


if __name__ == "__main__":
    unittest.main()
