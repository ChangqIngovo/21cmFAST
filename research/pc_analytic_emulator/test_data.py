"""Verify receipt-backed coeval teacher loading and seed-separated datasets."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from research.pc_analytic.compare import independent_budget
from research.pc_analytic_emulator.data import load_dataset, split_samples


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def make_campaign(
    root: Path,
    seeds: tuple[int, ...] = tuple(range(1001, 1009)),
    redshifts: tuple[float, ...] = (8.0, 7.5),
    n: int = 4,
) -> Path:
    """Construct an explicitly synthetic receipt-complete unit-test campaign.

    Parameters
    ----------
    root
        Destination results directory.
    seeds, redshifts, n
        Independent fixture identities, epochs and cube dimension.

    Returns
    -------
    Path
        Results directory. HDF5 placeholders test byte verification only; these
        fixtures never represent a native simulation or scientific benchmark.
    """
    root.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema": 1,
        "configuration": {
            "seeds": list(seeds),
            "redshifts": list(redshifts),
            "dimension": n,
            "highres_dimension": 3 * n,
            "box_len": 256.0,
            "brightness": "native",
            "analytic": {"directions": 6, "boundary": "source_periodic_cube"},
        },
        "native_runtime": {"native_extension_sha256": "unit-test-native-hash"},
        "scope": "Synthetic unit-test fixture; not native physical data",
    }
    rows = []
    for seed in seeds:
        inputs = {
            "random_seed": seed,
            "node_redshifts": [],
            "cosmo_params": {"hlittle": 0.68, "OMm": 0.31, "OMb": 0.049},
            "simulation_options": {
                "HII_DIM": n,
                "DIM": 3 * n,
                "BOX_LEN": 256.0,
                "N_THREADS": 1,
            },
            "matter_options": {"SOURCE_MODEL": "L-INTEGRAL"},
            "astro_params": {"R_BUBBLE_MAX": 50},
            "astro_options": {
                "RECOMB_MODEL": "none",
                "CELL_RECOMB": False,
                "USE_TS_FLUCT": False,
                "USE_MCGS": False,
                "USE_REIONIZATION_PHOTOHEATING_FEEDBACK": False,
                "PHOTON_CONS_TYPE": "no-photoncons",
            },
        }
        _write(root / "seeds" / f"seed{seed}" / "native_inputs.json", inputs)
        for z in redshifts:
            name = f"z{z:.12g}".replace(".", "p")
            native = root / "seeds" / f"seed{seed}" / "native" / f"seed{seed}" / name
            folder = root / "seeds" / f"seed{seed}" / "teacher" / name
            native.mkdir(parents=True)
            folder.mkdir(parents=True)
            capacity = np.ones((n, n, n))
            source = np.full_like(capacity, 0.6)
            xhi = np.full_like(capacity, 0.6)
            cosmology = inputs["cosmo_params"]
            h2 = cosmology["hlittle"] ** 2
            prefactor = (
                27
                * cosmology["OMb"]
                * h2
                / 0.023
                * np.sqrt(0.15 / (cosmology["OMm"] * h2) * (1 + z) / 10)
            )
            native_fields = {
                "source": source,
                "capacity": capacity,
                "native_es_xhi": np.full_like(capacity, 0.65),
                "native_brightness": prefactor * capacity * 0.65,
            }
            arrays = {
                "source": source,
                "capacity": capacity,
                "teacher_xHI": xhi,
                "ionized": capacity * (1 - xhi),
                "native_es_xHI": native_fields["native_es_xhi"],
                "native_brightness": native_fields["native_brightness"],
                "teacher_brightness": prefactor * capacity * xhi,
                "remaining_photons": np.full_like(capacity, 0.1),
                "escaped_by_direction": np.full(6, 0.1 * capacity.size / 6),
                "unemitted_source": np.zeros_like(capacity),
            }
            np.savez(native / "source.npz", **native_fields)
            (native / "native.h5").write_bytes(b"fixture-only-HDF5-placeholder")
            native_identity = {
                "inputs": inputs,
                "redshift": z,
                "extension_sha256": identity["native_runtime"][
                    "native_extension_sha256"
                ],
            }
            initial = hashlib.sha256(f"fixture-seed-{seed}".encode()).hexdigest()
            hashes = {key: _array_digest(value) for key, value in native_fields.items()}
            _write(
                native / "native.json",
                {
                    "identity": native_identity,
                    "source": {"cumulative_budget_not_rate": True},
                    "initial_conditions_sha256": initial,
                    "array_sha256": hashes,
                },
            )
            native_receipt = {
                "identity": native_identity,
                "record": {
                    "seed": seed,
                    "redshift": z,
                    "initial_conditions_sha256": initial,
                },
                "files": {
                    name: _digest(native / name)
                    for name in ("source.npz", "native.json", "native.h5")
                },
            }
            _write(native / "receipt.json", native_receipt)
            budget = independent_budget(
                source,
                capacity,
                xhi,
                remaining=float(arrays["remaining_photons"].sum(dtype=np.longdouble)),
                escaped=float(arrays["escaped_by_direction"].sum(dtype=np.longdouble)),
            )
            row = {
                "seed": seed,
                "redshift": z,
                "dimension": n,
                "box_len": 256.0,
                "input": {
                    "inputs": inputs,
                    "receipt_sha256": _digest(native / "receipt.json"),
                    "files": native_receipt["files"],
                    "native_dimension": n,
                    "coarsening_factor": 1,
                    "archived_array_sha256": hashes,
                    "brightness_mode": "native",
                },
                "photon_ledger": budget,
                "solver_audit": {"complete": True, "ledger_passed": True},
                "native_inputs_unchanged": True,
                "complete": True,
                "fields_path": str((folder / "fields.npz").resolve()),
                "receipt_path": str((folder / "receipt.json").resolve()),
            }
            np.savez(folder / "fields.npz", **arrays)
            _write(folder / "metrics.json", row)
            _write(
                folder / "receipt.json",
                {
                    "identity": {
                        "campaign": identity,
                        "native_receipt_sha256": _digest(native / "receipt.json"),
                    },
                    "record": row,
                    "files": {
                        name: _digest(folder / name)
                        for name in ("fields.npz", "metrics.json")
                    },
                },
            )
            rows.append(row)
    _write(root / "identity.json", identity)
    _write(
        root / "dataset_manifest.json",
        {
            "schema": 1,
            "identity": identity,
            "teacher_method": "analytic-column-characteristic",
            "complete": True,
            "all_photon_ledgers_passed": True,
            "samples": rows,
        },
    )
    _write(
        root / "summary.json",
        {
            "complete": True,
            "failed_seeds": [],
            "completed_snapshots": len(rows),
            "requested_snapshots": len(rows),
        },
    )
    reseal(root)
    (root / "COMPLETED").write_text("Synthetic unit-test campaign completed.\n")
    return root


def reseal(root: Path) -> None:
    """Refresh only campaign-level completion hashes for a unit-test mutation.

    Parameters
    ----------
    root
        Synthetic campaign directory.
    """
    _write(
        root / "completion.json",
        {
            "identity_sha256": _digest(root / "identity.json"),
            "files": {
                name: _digest(root / name)
                for name in ("dataset_manifest.json", "summary.json")
            },
        },
    )


def _mutate_fields(root: Path, **changes: np.ndarray) -> None:
    path = next(root.glob("seeds/*/teacher/*/fields.npz"))
    with np.load(path) as saved:
        arrays = {name: np.array(saved[name]) for name in saved.files}
    arrays.update(changes)
    np.savez(path, **arrays)
    receipt = json.loads((path.parent / "receipt.json").read_text())
    receipt["files"]["fields.npz"] = _digest(path)
    _write(path.parent / "receipt.json", receipt)


def test_complete_campaign_and_split(tmp_path: Path) -> None:
    """Keep whole seeds together and preserve finite escaped/retained photons."""
    root = make_campaign(tmp_path / "results")
    dataset = load_dataset(tmp_path)
    assert dataset.root == root
    assert len(dataset.samples) == 16
    assert dataset.audit["maximum_relative_photon_residual"] < 1e-15
    assert all(row["escaped_fraction"] > 0 for row in dataset.audit["reservoirs"])
    assert all(row["remaining_fraction"] > 0 for row in dataset.audit["reservoirs"])
    split = split_samples(dataset)
    assert [len(split[key]) for key in ("train", "validation", "test")] == [12, 2, 2]
    assert {row.seed for row in split["train"]} == set(range(1001, 1007))
    assert {row.seed for row in split["validation"]} == {1007}
    assert {row.seed for row in split["test"]} == {1008}
    assert not dataset.samples[0].arrays["teacher_xHI"].flags.writeable
    assert str(root / "completion.json") in dataset.file_hashes


def test_relocated_campaign_uses_verified_relative_layout(tmp_path: Path) -> None:
    """Copied immutable metadata may retain original absolute output paths."""
    original = make_campaign(tmp_path / "original", seeds=(1, 2, 3))
    copy_path = tmp_path / "copy"
    shutil.copytree(original, copy_path)
    dataset = load_dataset(copy_path)
    assert all(row.fields_path.is_relative_to(copy_path) for row in dataset.samples)
    assert (
        len(
            split_samples(
                dataset, train_seeds=[1], validation_seeds=[2], test_seeds=[3]
            )["test"]
        )
        == 2
    )


@pytest.mark.parametrize("name", ["COMPLETED", "completion.json", "identity.json"])
def test_incomplete_campaign_rejected(tmp_path: Path, name: str) -> None:
    """Require completion state and immutable campaign identity."""
    root = make_campaign(tmp_path, seeds=(1,))
    (root / name).unlink()
    with pytest.raises((ValueError, FileNotFoundError), match=r"Missing|No such"):
        load_dataset(root)


@pytest.mark.parametrize(
    "suffix",
    [
        "teacher/z8/fields.npz",
        "native/seed1/z8/native.h5",
        "native/seed1/z8/native.json",
    ],
)
def test_mutated_artifact_rejected(tmp_path: Path, suffix: str) -> None:
    """Catch target and original native archive corruption before fitting."""
    root = make_campaign(tmp_path, seeds=(1,))
    path = root / "seeds/seed1" / suffix
    with path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        load_dataset(root)


def test_missing_snapshot_rejected_even_with_resealed_manifest(tmp_path: Path) -> None:
    """A successful flag cannot substitute for the exact requested sample set."""
    root = make_campaign(tmp_path, seeds=(1, 2, 3))
    manifest = json.loads((root / "dataset_manifest.json").read_text())
    manifest["samples"].pop()
    _write(root / "dataset_manifest.json", manifest)
    reseal(root)
    with pytest.raises(ValueError, match="incomplete or duplicated"):
        load_dataset(root)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_bad_teacher_field_rejected(tmp_path: Path, value: float) -> None:
    """Recompute bounds instead of trusting a saved passing receipt."""
    root = make_campaign(tmp_path, seeds=(1,))
    _mutate_fields(root, teacher_xHI=np.full((4, 4, 4), value))
    with pytest.raises(ValueError, match=r"Non-finite|outside"):
        load_dataset(root)


def test_photon_reservoirs_independently_audited(tmp_path: Path) -> None:
    """A resealed reservoir change must fail the field-based photon ledger."""
    root = make_campaign(tmp_path, seeds=(1,))
    _mutate_fields(root, escaped_by_direction=np.zeros(6))
    with pytest.raises(ValueError, match="Independent field-derived"):
        load_dataset(root)


def test_native_source_not_replaced_by_teacher_archive(tmp_path: Path) -> None:
    """Source substitutions cannot pass by altering both source and reservoir."""
    root = make_campaign(tmp_path, seeds=(1,))
    _mutate_fields(
        root, source=np.full((4, 4, 4), 0.7), remaining_photons=np.full((4, 4, 4), 0.2)
    )
    with pytest.raises(ValueError, match="mutated native"):
        load_dataset(root)


@pytest.mark.parametrize(
    "groups", [([1], [1], [2, 3]), ([1], [2], [4]), ([], [2], [1, 3])]
)
def test_seed_leakage_and_missing_coverage_rejected(
    tmp_path: Path, groups: tuple
) -> None:
    """Split identities must be nonempty, disjoint and exhaustive."""
    dataset = load_dataset(make_campaign(tmp_path, seeds=(1, 2, 3)))
    with pytest.raises(ValueError, match=r"split|Seed"):
        split_samples(
            dataset,
            train_seeds=groups[0],
            validation_seeds=groups[1],
            test_seeds=groups[2],
        )


def test_default_split_requires_eight_seeds(tmp_path: Path) -> None:
    """Small integration fixtures need explicitly declared split identities."""
    dataset = load_dataset(make_campaign(tmp_path, seeds=(1, 2, 3)))
    with pytest.raises(ValueError, match="eight"):
        split_samples(dataset)
    with pytest.raises(ValueError, match="all three"):
        split_samples(dataset, train_seeds=[1])


def test_manifest_cannot_redirect_label_path(tmp_path: Path) -> None:
    """Metadata paths must identify the same canonical sample layout."""
    root = make_campaign(tmp_path, seeds=(1,))
    manifest = json.loads((root / "dataset_manifest.json").read_text())
    manifest["samples"][0]["fields_path"] = "/other/seed999/fields.npz"
    _write(root / "dataset_manifest.json", manifest)
    reseal(root)
    with pytest.raises(ValueError, match="Manifest path"):
        load_dataset(root)


def test_seed_physics_remains_exact(tmp_path: Path) -> None:
    """A changed native physical-input template is detected independently."""
    root = make_campaign(tmp_path, seeds=(1,))
    path = root / "seeds/seed1/native_inputs.json"
    inputs = copy.deepcopy(json.loads(path.read_text()))
    inputs["astro_params"]["R_BUBBLE_MAX"] = 20
    _write(path, inputs)
    with pytest.raises(ValueError, match="seed physical inputs"):
        load_dataset(root)
