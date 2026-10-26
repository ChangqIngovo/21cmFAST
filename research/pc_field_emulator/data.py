"""Read committed fixed-gas PC fields without loading the full source history.

Inputs are cumulative ``source_emitted`` counts. Targets are coarse-cell
``ionized / capacity`` fractions. Neither the target nor absorbed-source
budgets are used to construct the model inputs. Sparse fine states are not
expanded: this first emulator predicts the existing coarse output grid.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


class UnresolvedPhotonBudgetError(ValueError):
    """Identify an audited checkpoint outside the emitted-budget approximation.

    A conserved photon ledger can include photons that remain unabsorbed while
    neutral gas is present. Such a field cannot use the total emitted count as
    its ionized-atom target. This exception is separate from integrity failures
    so a training preflight can record and exclude only this incompatible case.

    Parameters
    ----------
    diagnostics : dict
        Independently verified counts, checkpoint coordinates and the error
        introduced by assigning all emitted photons to ionization. These values
        are eligibility diagnostics, not inputs to the learned model.
    """

    def __init__(self, diagnostics: dict[str, Any]) -> None:
        """Store the audited incompatibility without changing its photon counts."""
        self.diagnostics = dict(diagnostics)
        super().__init__(
            "Significant escaped/retained photons while gas remains neutral; "
            "emitted-budget projection is invalid "
            f"(seed={diagnostics['seed']}, node={diagnostics['node_index']}, "
            f"z={diagnostics['redshift']:g}, "
            f"unresolved_fraction={diagnostics['unresolved_fraction']:.9g}, "
            f"tolerance={diagnostics['unresolved_tolerance']:.9g})"
        )


@dataclass(frozen=True)
class Sample:
    """One committed output and the identities required to load it safely.

    Parameters
    ----------
    fixture_path, checkpoint_path : Path
        Existing native-history and PC checkpoint files, read without changes.
    checkpoint_sha256, capacity_sha256 : str
        Producer-recorded payload identities.
    seed, node_index : int
        Initial-condition identity and source-history endpoint.
    redshift, time_seconds, box_len : float
        Output coordinates and physical comoving box length in cMpc.
    dimension, subcells, angles, stride : int
        Coarse grid and selected teacher numerical settings.
    q_mass, q_volume, emitted, escaped, retained : float
        Producer diagnostics, independently checked when loading the fields.
    fixture_size, fixture_mtime_ns : int
        Fixture stat captured after its array/available file checksums pass.
    identity_sha256 : str or None
        Sealed evolution identity for journaled production runs.
    """

    fixture_path: Path
    checkpoint_path: Path
    checkpoint_sha256: str
    capacity_sha256: str
    seed: int
    node_index: int
    redshift: float
    time_seconds: float
    box_len: float
    dimension: int
    subcells: int
    angles: int
    stride: int
    q_mass: float
    q_volume: float
    emitted: float
    escaped: float
    retained: float
    fixture_size: int
    fixture_mtime_ns: int
    identity_sha256: str | None = None


def _canonical(value: dict) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@lru_cache(maxsize=16)
def _cached_digest(path: Path, size: int, mtime_ns: int) -> str:
    del size, mtime_ns
    return _digest(path)


def _checked_digest(path: Path, expected: str) -> None:
    stat = path.stat()
    if _cached_digest(path, stat.st_size, stat.st_mtime_ns) != expected:
        raise ValueError(f"File checksum mismatch: {path}")


def _array_digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _check_scope(metadata: dict) -> None:
    if metadata.get("schema_version") != 1:
        raise ValueError("Unsupported native-history metadata schema")
    if metadata.get("density_evolution") != "frozen_at_reference_redshift":
        raise ValueError("Emulator requires the declared fixed-gas history")
    if metadata.get("rates_definition") != "native_n_ion_times_H_over_t_STAR":
        raise ValueError("Unsupported source-rate definition")
    if metadata.get("source_model", "L-INTEGRAL") != "L-INTEGRAL":
        raise ValueError("This training experiment requires L-INTEGRAL teacher sources")
    inputs = metadata.get("inputs", {})
    options = inputs.get("astro_options", {})
    for key in ("USE_MCGS", "USE_TS_FLUCT", "USE_REIONIZATION_PHOTOHEATING_FEEDBACK"):
        if options.get(key, False):
            raise ValueError(f"Teacher enables incompatible physical option {key}")
    if options.get("RECOMB_MODEL", "none") != "none":
        raise ValueError("Recombining histories require a different photon budget")


@lru_cache(maxsize=8)
def _fixture(path: Path, size: int, mtime_ns: int) -> dict:
    del size, mtime_ns
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        fields = {
            name: np.asarray(archive[name])
            for name in ("capacity", "redshifts", "times_seconds")
        }
    _check_scope(metadata)
    hashes = metadata.get("array_sha256", {})
    for name, value in fields.items():
        if hashes.get(name) != _array_digest(value):
            raise ValueError(f"Native fixture array checksum mismatch: {name}")
        if np.iscomplexobj(value) or not np.isfinite(value).all():
            raise ValueError(f"Non-finite or complex fixture array: {name}")
        value.flags.writeable = False
    capacity, redshifts, times = (
        fields[name] for name in ("capacity", "redshifts", "times_seconds")
    )
    if (
        capacity.ndim != 3
        or len(set(capacity.shape)) != 1
        or capacity.size == 0
        or np.any(capacity <= 0)
    ):
        raise ValueError("Expected positive cubic atom capacity")
    if (
        redshifts.ndim != 1
        or len(redshifts) < 2
        or times.shape != redshifts.shape
        or times[0] != 0
        or np.any(np.diff(times) <= 0)
        or np.any(np.diff(redshifts) >= 0)
        or np.any(redshifts < 0)
    ):
        raise ValueError("Invalid native history coordinates")
    if metadata.get("dimension") != capacity.shape[0]:
        raise ValueError("Native metadata dimension disagrees with capacity")
    box_len = float(metadata.get("box_len_cMpc", 0.0))
    if not np.isfinite(box_len) or box_len <= 0:
        raise ValueError("Explicit positive box_len_cMpc required")
    if type(metadata.get("random_seed")) is not int:
        raise ValueError(
            "Explicit native random_seed required for leakage-free splitting"
        )
    return dict(fields, metadata=metadata)


def _load_fixture(path: Path) -> dict:
    stat = path.stat()
    return _fixture(path, stat.st_size, stat.st_mtime_ns)


def _read_report(case: Path) -> tuple[dict, dict | None, str | None]:
    receipt_path = case / "receipt.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        _checked_digest(case / "summary.json", receipt["summary_sha256"])
        report = json.loads((case / "summary.json").read_text())
        if not report.get("all_ledgers_passed", False):
            raise ValueError(f"Teacher photon audits did not pass: {case}")
        return report, receipt, None
    progress_path = case / "progress.json"
    if not progress_path.is_file():
        raise ValueError(f"No verified receipt or progress journal: {case}")
    record = json.loads(progress_path.read_text())
    payload = record.get("payload")
    if (
        record.get("version") != 1
        or not isinstance(payload, dict)
        or record.get("sha256") != hashlib.sha256(_canonical(payload)).hexdigest()
    ):
        raise ValueError(f"Evolution progress checksum mismatch: {progress_path}")
    if payload.get("status") not in ("running", "complete"):
        raise ValueError("Unsupported evolution progress status")
    identity = payload["identity"]
    report = dict(
        identity["settings"],
        metadata=identity["metadata"],
        checkpoints=payload["checkpoints"],
    )
    return report, None, hashlib.sha256(_canonical(identity)).hexdigest()


def _results_root(root: Path) -> Path:
    root = Path(root).resolve()
    if (
        not (root / "cases").is_dir()
        and not (root / "pc").is_dir()
        and (root / "results").is_dir()
    ):
        return root / "results"
    return root


def discover_samples(
    root: Path, *, subcells: int = 4, angles: int = 768, stride: int = 4
) -> list[Sample]:
    """Discover one numerical arm from committed existing PC outputs.

    Parameters
    ----------
    root : Path
        A matrix/production ``results`` directory, or its enclosing run.
    subcells, angles, stride : int
        Exact teacher settings. Other arms and ES controls are excluded.

    Returns
    -------
    list of Sample
        Sorted by seed and source node. Unjournaled outputs are ignored.
    """
    root = _results_root(root)
    if any(
        type(value) is not int or value <= 0 for value in (subcells, angles, stride)
    ):
        raise ValueError("Positive integer numerical settings required")
    production = (root / "native_history.npz").is_file()
    if production:
        fixtures = [root / "native_history.npz"]
        cases = [root / "pc"]
    else:
        fixtures = sorted((root / "fixtures").glob("*.npz"))
        cases = sorted((root / "cases").glob(f"*_f{subcells}_k{angles}_stride{stride}"))
    fixture_by_seed = {}
    for fixture_path in fixtures:
        data = _load_fixture(fixture_path)
        seed = data["metadata"]["random_seed"]
        if seed in fixture_by_seed:
            raise ValueError(f"Multiple native fixtures have seed {seed}")
        fixture_by_seed[seed] = fixture_path
    samples = []
    used = set()
    for case in cases:
        if not case.is_dir():
            continue
        report, receipt, identity_sha256 = _read_report(case)
        if (
            report.get("subcells_per_axis"),
            report.get("angles"),
            report.get("stride"),
        ) != (subcells, angles, stride):
            continue
        metadata = report["metadata"]
        _check_scope(metadata)
        seed = metadata["random_seed"]
        if seed not in fixture_by_seed:
            raise ValueError(f"Missing native fixture for seed {seed}")
        fixture_path = fixture_by_seed[seed]
        fixture = _load_fixture(fixture_path)
        if fixture["metadata"] != metadata:
            raise ValueError("Checkpoint and fixture metadata differ")
        if receipt is not None:
            _checked_digest(fixture_path, receipt["fixture_sha256"])
        stat = fixture_path.stat()
        previous_node = -1
        for row in report.get("checkpoints", []):
            node = row["node_index"]
            if type(node) is not int or not previous_node < node < len(
                fixture["redshifts"]
            ):
                raise ValueError(
                    "Checkpoint nodes must be unique, ordered native endpoints"
                )
            previous_node = node
            if (seed, node) in used:
                raise ValueError(
                    "Duplicate realization/node would leak between samples"
                )
            used.add((seed, node))
            if (
                row["redshift"] != fixture["redshifts"][node]
                or row["time_seconds"] != fixture["times_seconds"][node]
            ):
                raise ValueError("Checkpoint coordinates differ from native history")
            ledger = row["ledger"]
            if not ledger.get("passed", False):
                raise ValueError("Checkpoint photon audit failed")
            checkpoint = case / "checkpoints" / f"node_{node:05d}.npz"
            if Path(row["path"]).name != checkpoint.name:
                raise ValueError("Checkpoint path disagrees with native node")
            expected = row["sha256"]
            if (
                receipt is not None
                and receipt["checkpoint_sha256"].get(str(checkpoint.relative_to(case)))
                != expected
            ):
                raise ValueError("Receipt and checkpoint row checksums disagree")
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            samples.append(
                Sample(
                    fixture_path=fixture_path,
                    checkpoint_path=checkpoint,
                    checkpoint_sha256=expected,
                    capacity_sha256=metadata["array_sha256"]["capacity"],
                    seed=seed,
                    node_index=node,
                    redshift=float(row["redshift"]),
                    time_seconds=float(row["time_seconds"]),
                    box_len=float(metadata["box_len_cMpc"]),
                    dimension=int(metadata["dimension"]),
                    subcells=subcells,
                    angles=angles,
                    stride=stride,
                    q_mass=float(row["Q_mass"]),
                    q_volume=float(row["Q_volume"]),
                    emitted=float(ledger["emitted"]),
                    escaped=float(ledger["escaped"]),
                    retained=float(ledger["retained"]),
                    fixture_size=stat.st_size,
                    fixture_mtime_ns=stat.st_mtime_ns,
                    identity_sha256=identity_sha256,
                )
            )
    if not samples:
        raise ValueError(
            f"No committed PC outputs match the selected numerical arm f={subcells}, angles={angles}, stride={stride}"
        )
    return sorted(samples, key=lambda sample: (sample.seed, sample.node_index))


def load_sample(
    sample: Sample,
    *,
    ledger_tolerance: float = 2e-10,
    unresolved_tolerance: float = 1e-7,
) -> dict[str, Any]:
    """Load one teacher field and independently check photon accounting.

    Parameters
    ----------
    sample : Sample
        Record from :func:`discover_samples`.
    ledger_tolerance : float
        Relative count and absolute fractional tolerance of producer audits.
    unresolved_tolerance : float
        Maximum emitted-photon fraction escaped/retained while neutral gas
        remains. The default admits the previously measured tiny horizon tail.

    Returns
    -------
    dict
        Capacity, cumulative emitted ``photons``, target ``xHII``, redshift,
        box length, seed, source node and an independently computed count ledger.
        ``budget`` is ``min(sum(photons), sum(capacity))``; it never uses labels.
        After the raw ledger and bounds pass, numerical fraction overshoot is
        clipped only in ``xHII`` and reported as ``maximum_target_bound_roundoff``.

    Raises
    ------
    UnresolvedPhotonBudgetError
        The independently validated ledger has an unresolved radiation fraction
        above the stated tolerance while neutral gas remains. Diagnostics are
        attached to the exception; this condition does not mean photon loss.
    ValueError
        A checkpoint fails identity, field, ledger or summary validation.
    """
    if not (0 < ledger_tolerance < 1 and 0 <= unresolved_tolerance < 1):
        raise ValueError("Invalid photon audit tolerances")
    stat = sample.fixture_path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (
        sample.fixture_size,
        sample.fixture_mtime_ns,
    ):
        raise ValueError("Native fixture changed after discovery")
    fixture = _load_fixture(sample.fixture_path)
    capacity = fixture["capacity"]
    # Always hash checkpoint contents at use time, including after discovery.
    if _digest(sample.checkpoint_path) != sample.checkpoint_sha256:
        raise ValueError(f"Checkpoint checksum mismatch: {sample.checkpoint_path}")
    with np.load(sample.checkpoint_path, allow_pickle=False) as archive:
        values = {}
        for name in (
            "ionized",
            "source_emitted",
            "source_absorbed",
            "source_escaped",
            "source_retained",
        ):
            raw = archive[name]
            if np.iscomplexobj(raw):
                raise ValueError(f"Complex checkpoint count field: {name}")
            values[name] = np.asarray(raw, dtype=np.longdouble)
        if (
            int(archive["state_version"]) != 1
            or int(archive["node_index"]) != sample.node_index
            or float(archive["redshift"]) != sample.redshift
            or float(archive["time_seconds"]) != sample.time_seconds
            or int(archive["subcells_per_axis"]) != sample.subcells
            or str(archive["capacity_sha256"].item()) != sample.capacity_sha256
        ):
            raise ValueError("Checkpoint coordinates/capacity/settings changed")
        if (
            sample.identity_sha256 is not None
            and str(archive["evolution_identity_sha256"].item())
            != sample.identity_sha256
        ):
            raise ValueError(
                "Checkpoint evolution identity differs from committed journal"
            )
    for name, value in values.items():
        if value.shape != capacity.shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid checkpoint field: {name}")
        if np.any(value < 0):
            raise ValueError(f"Negative checkpoint photon/ion count: {name}")
    emitted = values["source_emitted"]
    absorbed = values["source_absorbed"]
    escaped = values["source_escaped"]
    retained = values["source_retained"]
    ionized = values["ionized"]
    total_emitted = emitted.sum(dtype=np.longdouble)
    total_capacity = capacity.sum(dtype=np.longdouble)
    total_ionized = ionized.sum(dtype=np.longdouble)
    total_escaped = escaped.sum(dtype=np.longdouble)
    total_retained = retained.sum(dtype=np.longdouble)
    scale = max(total_emitted, np.longdouble(np.finfo(float).tiny))
    source_residual = emitted - absorbed - escaped - retained
    active = emitted > 0
    source_error = (
        float(np.max(np.abs(source_residual[active]) / emitted[active]))
        if active.any()
        else 0.0
    )
    source_zero_error = bool(np.any(source_residual[~active] != 0))
    grid_error = float(
        abs(total_emitted - total_ionized - total_escaped - total_retained) / scale
    )
    absorbed_error = float(
        abs(absorbed.sum(dtype=np.longdouble) - total_ionized) / scale
    )
    maximum_violation = float(np.max(np.maximum(ionized - capacity, 0) / capacity))
    if (
        source_zero_error
        or max(source_error, grid_error, absorbed_error, maximum_violation)
        > ledger_tolerance
    ):
        raise ValueError("Independent source/grid photon ledger failed")
    xhii = np.asarray(ionized / capacity, dtype=np.float64)
    q_mass = float(total_ionized / total_capacity)
    q_volume = float(np.mean(xhii, dtype=np.float64))
    if (
        abs(q_mass - sample.q_mass) > ledger_tolerance
        or abs(q_volume - sample.q_volume) > ledger_tolerance
        or abs(total_emitted - sample.emitted) > ledger_tolerance * scale
        or abs(total_escaped - sample.escaped) > ledger_tolerance * scale
        or abs(total_retained - sample.retained) > ledger_tolerance * scale
    ):
        raise ValueError(
            "Checkpoint summary disagrees with independently loaded fields"
        )
    radiation_fraction = float((total_escaped + total_retained) / scale)
    if q_mass < 1 - ledger_tolerance and radiation_fraction > unresolved_tolerance:
        projected_budget = min(total_emitted, total_capacity)
        raise UnresolvedPhotonBudgetError(
            {
                "reason": "unresolved_radiation_before_saturation",
                "seed": sample.seed,
                "node_index": sample.node_index,
                "redshift": sample.redshift,
                "Q_mass": q_mass,
                "emitted": float(total_emitted),
                "ionized": float(total_ionized),
                "atom_capacity": float(total_capacity),
                "escaped": float(total_escaped),
                "retained": float(total_retained),
                "unresolved_fraction": radiation_fraction,
                "unresolved_tolerance": unresolved_tolerance,
                "projected_budget": float(projected_budget),
                "projected_minus_teacher_count": float(
                    projected_budget - total_ionized
                ),
                "projected_minus_teacher_Q_mass": float(
                    (projected_budget - total_ionized) / total_capacity
                ),
                "relative_grid_residual": grid_error,
                "maximum_source_relative_residual": source_error,
            }
        )
    target_roundoff = float(
        max(np.max(np.maximum(-xhii, 0)), np.max(np.maximum(xhii - 1, 0)))
    )
    xhii = np.clip(xhii, 0.0, 1.0)
    photons = np.asarray(emitted, dtype=np.float64)
    if not np.isfinite(photons).all():
        raise OverflowError("Cumulative source field cannot be represented in float64")
    return {
        "capacity": capacity,
        "photons": photons,
        "xHII": xhii,
        "redshift": sample.redshift,
        "box_len": sample.box_len,
        "seed": sample.seed,
        "node_index": sample.node_index,
        "budget": float(min(total_emitted, total_capacity)),
        "ledger": {
            "emitted": float(total_emitted),
            "ionized": float(total_ionized),
            "escaped": float(total_escaped),
            "retained": float(total_retained),
            "relative_grid_residual": grid_error,
            "maximum_source_relative_residual": source_error,
            "maximum_target_bound_roundoff": target_roundoff,
            "unresolved_fraction": radiation_fraction,
        },
        "metadata": fixture["metadata"],
    }


def split_samples(samples: list[Sample]) -> dict[str, Any]:
    """Split entire realizations, with explicit temporal fallback for one seed.

    Parameters
    ----------
    samples : list of Sample
        Selected snapshots, optionally already restricted to a useful epoch.

    Returns
    -------
    dict
        ``train``, ``validation`` and ``test`` lists plus a ``strategy`` label.
        Four seeds give two training, one validation and one test realization.
        A single seed uses contiguous 60/20/20 time blocks, explicitly testing
        only within-realization temporal extrapolation.
    """
    if not samples:
        raise ValueError("Cannot split an empty dataset")
    ordered = sorted(samples, key=lambda sample: (sample.seed, sample.node_index))
    keys = [(sample.seed, sample.node_index) for sample in ordered]
    if len(keys) != len(set(keys)):
        raise ValueError("Repeated realization/node in dataset")
    arms = {(sample.subcells, sample.angles, sample.stride) for sample in ordered}
    if len(arms) != 1:
        raise ValueError("Dataset mixes teacher numerical arms")
    seeds = sorted({sample.seed for sample in ordered})
    if len(seeds) >= 3:
        return {
            "train": [sample for sample in ordered if sample.seed in seeds[:-2]],
            "validation": [sample for sample in ordered if sample.seed == seeds[-2]],
            "test": [sample for sample in ordered if sample.seed == seeds[-1]],
            "strategy": "independent_seed_holdout",
        }
    if len(seeds) == 2:
        first = [sample for sample in ordered if sample.seed == seeds[0]]
        if len(first) < 2:
            raise ValueError(
                "Two-seed split requires at least two training-seed snapshots"
            )
        cut = min(len(first) - 1, max(1, int(0.8 * len(first))))
        return {
            "train": first[:cut],
            "validation": first[cut:],
            "test": [sample for sample in ordered if sample.seed == seeds[-1]],
            "strategy": "heldout_test_seed_with_temporal_validation",
        }
    if len(ordered) < 3:
        raise ValueError(
            "Single-seed split requires at least three committed snapshots"
        )
    first_cut = min(len(ordered) - 2, max(1, int(0.6 * len(ordered))))
    second_cut = min(len(ordered) - 1, max(first_cut + 1, int(0.8 * len(ordered))))
    return {
        "train": ordered[:first_cut],
        "validation": ordered[first_cut:second_cut],
        "test": ordered[second_cut:],
        "strategy": "within_realization_temporal_extrapolation",
    }
