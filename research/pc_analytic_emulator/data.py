"""Read verified coeval analytic-PC teachers without importing a native runtime.

The original source, atom and photon-reservoir arrays retain their count units.
No rate integration, extra density factor or closed-box absorption assumption is
introduced. Every seed remains wholly inside one data split.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from research.pc_analytic.compare import independent_budget, validate_physics

LEDGER_TOLERANCE = 1e-10
ARRAY_NAMES = {
    "source",
    "capacity",
    "teacher_xHI",
    "ionized",
    "remaining_photons",
    "escaped_by_direction",
    "unemitted_source",
    "native_es_xHI",
}
NATIVE_NAMES = ("source", "capacity", "native_es_xhi", "native_brightness")


@dataclass(frozen=True)
class TeacherSample:
    """One independently audited coeval training sample.

    Parameters
    ----------
    seed, redshift, dimension, box_len
        Native realization, epoch and output geometry in cMpc.
    fields_path, native_directory
        Verified teacher field file and its original native snapshot directory.
    metadata, arrays, budget
        Unmodified manifest row, read-only arrays and recomputed photon ledger.
    """

    seed: int
    redshift: float
    dimension: int
    box_len: float
    fields_path: Path
    native_directory: Path
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]
    budget: dict[str, Any]


@dataclass(frozen=True)
class TeacherDataset:
    """Complete audited teacher campaign with file-level provenance.

    Parameters
    ----------
    root, samples
        Results directory and ordered, independently audited coeval samples.
    manifest, identity, physics
        Saved campaign information and common native physical parameters.
    file_hashes
        Absolute source file paths mapped to verified SHA256 checksums.
    audit
        Dataset-level counts and measured photon-reservoir fractions.
    """

    root: Path
    samples: tuple[TeacherSample, ...]
    manifest: dict[str, Any]
    identity: dict[str, Any]
    file_hashes: dict[str, str]
    physics: dict[str, Any]
    audit: dict[str, Any]


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


class _Files:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.hashes: dict[str, str] = {}

    def digest(self, path: Path, expected: str | None = None) -> str:
        path = path.resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError(f"Missing file or path outside teacher dataset: {path}")
        name = str(path)
        if name not in self.hashes:
            with path.open("rb") as stream:
                self.hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
        digest = self.hashes[name]
        if expected is not None and expected != digest:
            raise ValueError(f"Teacher dataset checksum failed: {path}")
        return digest

    def receipt(self, folder: Path, names: set[str]) -> dict[str, Any]:
        self.digest(folder / "receipt.json")
        receipt = _json(folder / "receipt.json")
        if set(receipt["files"]) != names:
            raise ValueError(f"Receipt does not cover the expected files: {folder}")
        for name in sorted(names):
            self.digest(folder / name, receipt["files"][name])
        return receipt


def _redshift_name(redshift: float) -> str:
    return f"z{redshift:.12g}".replace(".", "p")


def _archived_path(value: str, suffix: Path) -> None:
    path = Path(value)
    if not path.is_absolute() or path.parts[-len(suffix.parts) :] != suffix.parts:
        raise ValueError("Manifest path does not match the expected sample location")


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: np.array(saved[name], copy=True) for name in saved.files}
    for name, value in arrays.items():
        if value.dtype.kind not in "fiub" or not np.isfinite(value).all():
            raise ValueError(f"Non-finite or nonnumeric stored array: {name}")
        value.setflags(write=False)
    return arrays


def _array_digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _native(
    folder: Path,
    row: dict[str, Any],
    config: dict[str, Any],
    identity: dict[str, Any],
    files: _Files,
) -> tuple[dict[str, np.ndarray], dict[str, Any], str]:
    receipt = files.receipt(folder, {"source.npz", "native.json", "native.h5"})
    metadata = _json(folder / "native.json")
    native_identity = receipt["identity"]
    if metadata["identity"] != native_identity:
        raise ValueError("Native metadata differs from its receipt")
    inputs = native_identity["inputs"]
    validate_physics(inputs)
    if metadata.get("source", {}).get("cumulative_budget_not_rate") is not True:
        raise ValueError("Native source must be cumulative counts, not a rate")
    if (
        native_identity["extension_sha256"]
        != identity["native_runtime"]["native_extension_sha256"]
    ):
        raise ValueError(
            "Native extension identity differs between sample and campaign"
        )
    record = receipt["record"]
    if (
        record["seed"] != row["seed"]
        or inputs["random_seed"] != row["seed"]
        or record["redshift"] != row["redshift"]
        or native_identity["redshift"] != row["redshift"]
        or inputs["simulation_options"]["HII_DIM"] != config["dimension"]
        or inputs["simulation_options"]["DIM"] != config["highres_dimension"]
        or inputs["simulation_options"]["BOX_LEN"] != config["box_len"]
    ):
        raise ValueError("Native sample identity or geometry differs from campaign")
    seed_inputs = folder.parents[2] / "native_inputs.json"
    files.digest(seed_inputs)
    if _json(seed_inputs) != inputs:
        raise ValueError("Native sample differs from its seed physical inputs")
    original = row["input"]
    if (
        original["inputs"] != inputs
        or original["receipt_sha256"] != files.digest(folder / "receipt.json")
        or original["files"] != receipt["files"]
        or original["coarsening_factor"] != 1
        or original["native_dimension"] != config["dimension"]
        or original["brightness_mode"] != config["brightness"]
    ):
        raise ValueError("Teacher source metadata differs from verified native inputs")
    arrays = _load_arrays(folder / "source.npz")
    if set(arrays) != set(NATIVE_NAMES):
        raise ValueError("Unexpected native source archive fields")
    for name, value in arrays.items():
        digest = _array_digest(value)
        if (
            metadata["array_sha256"][name] != digest
            or original["archived_array_sha256"][name] != digest
        ):
            raise ValueError(f"Native source array checksum failed: {name}")
    initial = metadata["initial_conditions_sha256"]
    if initial != record["initial_conditions_sha256"]:
        raise ValueError("Native initial-condition identity differs from its receipt")
    return arrays, inputs, initial


def _audit_arrays(
    arrays: dict[str, np.ndarray], config: dict[str, Any]
) -> dict[str, Any]:
    expected = set(ARRAY_NAMES)
    if config["brightness"] != "none":
        expected.update(("teacher_brightness", "native_brightness"))
    if set(arrays) != expected:
        raise ValueError("Teacher archive fields do not match the declared brightness")
    n = config["dimension"]
    for name, value in arrays.items():
        shape = (
            (config["analytic"]["directions"],)
            if name == "escaped_by_direction"
            else (n, n, n)
        )
        if value.shape != shape:
            raise ValueError(f"Wrong teacher grid or direction shape: {name}")
    for name in (
        "source",
        "capacity",
        "ionized",
        "remaining_photons",
        "escaped_by_direction",
        "unemitted_source",
    ):
        if np.any(arrays[name] < 0):
            raise ValueError(f"Negative photon or atom inventory: {name}")
    capacity, source = arrays["capacity"], arrays["source"]
    if np.any(capacity <= 0):
        raise ValueError("Teacher capacity must be strictly positive")
    for name in ("teacher_xHI", "native_es_xHI"):
        if np.any(arrays[name] < -1e-12) or np.any(arrays[name] > 1 + 1e-12):
            raise ValueError(f"Neutral fraction outside [0,1]: {name}")
    if not np.allclose(
        arrays["ionized"],
        capacity * (1 - arrays["teacher_xHI"]),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("Stored ions disagree with the neutral-fraction field")

    def total(name: str) -> float:
        return float(np.sum(arrays[name], dtype=np.longdouble))

    budget = independent_budget(
        source,
        capacity,
        arrays["teacher_xHI"],
        remaining=total("remaining_photons"),
        escaped=total("escaped_by_direction"),
        unemitted=total("unemitted_source"),
    )
    if not budget["bounded"] or budget["relative_photon_residual"] > LEDGER_TOLERANCE:
        raise ValueError("Independent field-derived photon ledger failed")
    if budget["unemitted_photons"] != 0:
        raise ValueError("Completed teacher still contains unemitted source photons")
    return budget


def _sample(
    root: Path,
    row: dict[str, Any],
    identity: dict[str, Any],
    files: _Files,
) -> tuple[TeacherSample, dict[str, Any], str]:
    config = identity["configuration"]
    seed, redshift = int(row["seed"]), float(row["redshift"])
    suffix = Path("seeds") / f"seed{seed}" / "teacher" / _redshift_name(redshift)
    folder = root / suffix
    _archived_path(row["fields_path"], suffix / "fields.npz")
    _archived_path(row["receipt_path"], suffix / "receipt.json")
    receipt = files.receipt(folder, {"fields.npz", "metrics.json"})
    if receipt["record"] != row or _json(folder / "metrics.json") != row:
        raise ValueError("Teacher metrics, receipt and dataset manifest differ")
    native = (
        root
        / "seeds"
        / f"seed{seed}"
        / "native"
        / f"seed{seed}"
        / _redshift_name(redshift)
    )
    if receipt["identity"] != {
        "campaign": identity,
        "native_receipt_sha256": files.digest(native / "receipt.json"),
    }:
        raise ValueError("Teacher receipt identity or native receipt checksum differs")
    if (
        row["dimension"] != config["dimension"]
        or row["box_len"] != config["box_len"]
        or not row.get("complete")
        or not row.get("native_inputs_unchanged")
        or not row["solver_audit"].get("complete")
        or not row["solver_audit"].get("ledger_passed")
        or not row["photon_ledger"].get("bounded")
        or not np.isfinite(row["photon_ledger"]["relative_photon_residual"])
        or row["photon_ledger"]["relative_photon_residual"] > LEDGER_TOLERANCE
    ):
        raise ValueError("Teacher label is incomplete or inconsistent with campaign")
    original, inputs, initial = _native(native, row, config, identity, files)
    arrays = _load_arrays(folder / "fields.npz")
    budget = _audit_arrays(arrays, config)
    for name, target in (
        ("source", "source"),
        ("capacity", "capacity"),
        ("native_es_xHI", "native_es_xhi"),
        ("native_brightness", "native_brightness"),
    ):
        if name in arrays and not np.array_equal(arrays[name], original[target]):
            raise ValueError(f"Teacher mutated native input or baseline: {name}")
    for name in (
        "emitted_photons",
        "atom_capacity",
        "absorbed_photons_from_field",
        "remaining_photons",
        "escaped_photons",
        "unemitted_photons",
        "xHI_mass",
        "xHI_volume",
    ):
        if not np.isclose(
            row["photon_ledger"][name], budget[name], rtol=1e-12, atol=1e-12
        ):
            raise ValueError(f"Stored photon ledger differs from fields: {name}")
    return (
        TeacherSample(
            seed,
            redshift,
            config["dimension"],
            config["box_len"],
            folder / "fields.npz",
            native,
            row,
            arrays,
            budget,
        ),
        inputs,
        initial,
    )


def load_dataset(root: str | Path) -> TeacherDataset:
    """Load and independently verify a complete analytic teacher campaign.

    Parameters
    ----------
    root
        Completed teacher run or its results directory. A relocated copy is
        accepted only through its unchanged receipts and canonical layout.

    Returns
    -------
    TeacherDataset
        Read-only arrays, exact native physics, independently recomputed ledgers
        and checksums of all consumed inputs, including original HDF5 archives.
    """
    root = Path(root).resolve()
    if (root / "results" / "dataset_manifest.json").is_file():
        root = root / "results"
    files = _Files(root)
    files.digest(root / "COMPLETED")
    files.digest(root / "completion.json")
    completion = _json(root / "completion.json")
    if set(completion["files"]) != {"dataset_manifest.json", "summary.json"}:
        raise ValueError("Campaign completion receipt does not cover expected outputs")
    files.digest(root / "identity.json", completion["identity_sha256"])
    for name, digest in completion["files"].items():
        files.digest(root / name, digest)
    identity, manifest, summary = (
        _json(root / name)
        for name in ("identity.json", "dataset_manifest.json", "summary.json")
    )
    if (
        manifest["identity"] != identity
        or manifest.get("schema") != 1
        or manifest.get("teacher_method") != "analytic-column-characteristic"
        or not manifest.get("complete")
        or not manifest.get("all_photon_ledgers_passed")
        or not summary.get("complete")
        or summary.get("failed_seeds") != []
    ):
        raise ValueError("Dataset lacks a complete matching analytic teacher identity")
    config = identity["configuration"]
    expected = {
        (int(seed), float(z)) for seed in config["seeds"] for z in config["redshifts"]
    }
    rows = manifest["samples"]
    actual = [(int(row["seed"]), float(row["redshift"])) for row in rows]
    if (
        len(expected) != len(config["seeds"]) * len(config["redshifts"])
        or len(actual) != len(expected)
        or set(actual) != expected
        or summary["completed_snapshots"] != len(expected)
        or summary["requested_snapshots"] != len(expected)
    ):
        raise ValueError("Teacher sample set is incomplete or duplicated")
    samples, physics, initial_conditions = [], None, {}
    for row in sorted(rows, key=lambda row: (row["seed"], -row["redshift"])):
        sample, inputs, initial = _sample(root, row, identity, files)
        common = copy.deepcopy(inputs)
        common.pop("random_seed")
        common["simulation_options"].pop("N_THREADS", None)
        if physics is not None and common != physics:
            raise ValueError("Teacher samples have inconsistent physical parameters")
        physics = common
        if (
            sample.seed in initial_conditions
            and initial_conditions[sample.seed] != initial
        ):
            raise ValueError("One seed contains different initial conditions")
        initial_conditions[sample.seed] = initial
        samples.append(sample)
    if len(set(initial_conditions.values())) != len(initial_conditions):
        raise ValueError("Different seeds share identical initial conditions")
    audit = {
        "sample_count": len(samples),
        "seeds": sorted(initial_conditions),
        "redshifts": sorted({sample.redshift for sample in samples}, reverse=True),
        "maximum_relative_photon_residual": max(
            sample.budget["relative_photon_residual"] for sample in samples
        ),
        "initial_conditions_sha256": initial_conditions,
        "reservoirs": [
            {
                "seed": sample.seed,
                "redshift": sample.redshift,
                "absorbed_fraction": sample.budget["absorbed_photons_from_field"]
                / max(sample.budget["source_photon_budget"], np.finfo(float).tiny),
                "remaining_fraction": sample.budget["remaining_photons"]
                / max(sample.budget["source_photon_budget"], np.finfo(float).tiny),
                "escaped_fraction": sample.budget["escaped_photons"]
                / max(sample.budget["source_photon_budget"], np.finfo(float).tiny),
            }
            for sample in samples
        ],
    }
    return TeacherDataset(
        root, tuple(samples), manifest, identity, files.hashes, physics or {}, audit
    )


def split_samples(
    dataset: TeacherDataset,
    *,
    train_seeds: Sequence[int] | None = None,
    validation_seeds: Sequence[int] | None = None,
    test_seeds: Sequence[int] | None = None,
) -> dict[str, tuple[TeacherSample, ...]]:
    """Keep every realization and all of its redshifts in exactly one split.

    Parameters
    ----------
    dataset
        Verified campaign returned by :func:`load_dataset`.
    train_seeds, validation_seeds, test_seeds
        Nonempty, disjoint and exhaustive explicit seed lists. If all are None,
        eight sorted seeds are split into the first six, seventh and eighth.

    Returns
    -------
    dict
        Ordered samples under train, validation and test keys.
    """
    available = sorted({sample.seed for sample in dataset.samples})
    selections = (train_seeds, validation_seeds, test_seeds)
    if all(value is None for value in selections):
        if len(available) != 8:
            raise ValueError("Default split requires eight independent seeds")
        selections = (available[:6], available[6:7], available[7:])
    elif any(value is None for value in selections):
        raise ValueError("Provide all three explicit seed splits")
    assigned = []
    for values in selections:
        if not values or any(type(value) is not int for value in values):
            raise ValueError("Each split requires nonempty integer seed lists")
        assigned.extend(values)
    if len(set(assigned)) != len(assigned) or set(assigned) != set(available):
        raise ValueError("Seed splits must be disjoint and cover the whole dataset")
    return {
        name: tuple(sample for sample in dataset.samples if sample.seed in selected)
        for name, selected in zip(
            ("train", "validation", "test"), selections, strict=True
        )
    }
