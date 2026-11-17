"""Reproduce supervised onset training from verified offline NPZ archives."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from research.pc_analytic_emulator.exposure_onset import OnsetHead, fit_onset


def _dose_labels(
    xhi_history: np.ndarray,
    crossing: np.ndarray,
    source: np.ndarray,
    capacity: np.ndarray,
) -> np.ndarray:
    batches = len(xhi_history)
    if (
        batches < 2
        or xhi_history.shape != (batches, *capacity.shape)
        or crossing.shape != capacity.shape
        or not np.issubdtype(crossing.dtype, np.integer)
        or np.any((crossing < 0) | (crossing > batches))
        or not np.isfinite(xhi_history).all()
        or np.any((xhi_history < 0) | (xhi_history > 1))
        or np.any(np.diff(xhi_history, axis=0) > 1e-7)
    ):
        raise ValueError("Invalid offline ionization exposure history")
    crossing = crossing.astype(np.int64)
    prev = np.take_along_axis(xhi_history, np.maximum(crossing - 2, 0)[None], axis=0)[0]
    earlier = np.take_along_axis(
        xhi_history, np.maximum(crossing - 3, 0)[None], axis=0
    )[0]
    prev = np.where(crossing > 1, prev, 1.0)
    earlier = np.where(crossing > 2, earlier, 1.0)
    local_rate = source / capacity / batches
    previous_rate = np.maximum(np.maximum(earlier - prev, 1e-9), local_rate)
    offset = np.clip(prev / np.maximum(previous_rate, 1e-7), 0.0, 1.0)
    offset = np.where(
        crossing <= 1,
        np.minimum(capacity / np.maximum(source, 1e-30) * batches, 1.0),
        offset,
    )
    offset = np.maximum(offset - 0.125, 0.0)
    return np.where(crossing > 0, (crossing - 1 + offset) / batches, 1.0)


def fit_onset_manifest(
    manifest: Path,
    *,
    sample_cells: int = 4096,
    steps: int = 220,
) -> OnsetHead:
    """Fit an onset head from a checksum-bound offline training manifest.

    Parameters
    ----------
    manifest
        JSON with schema 1, explicit ``training_seeds`` and ``excluded_seeds``
        lists, and an ordered ``examples`` list. Each example
        records ``seed``, ``redshift``, ``box_len``, ``native_archive``,
        ``native_sha256``, ``onset_archive``, ``onset_sha256``, and
        ``onset_label``. Paths are relative to the JSON directory or
        absolute. Native NPZ archives contain ``source``, ``capacity``
        and ``native_es_xHI``. Label type ``teacher_dose_history`` uses
        a separate NPZ with ``source``, ``capacity``, ``xHI_dose`` and
        ``saturation_batch``. Label type ``offline_continuous_donor_pooling``
        uses a separate NPZ with a finite ``onset`` in [0, 1].
    sample_cells, steps
        Training controls forwarded to :func:`fit_onset`.

    Returns
    -------
    OnsetHead
        Trained model bound to the training manifest checksum.
    """
    payload = manifest.read_bytes()
    config = json.loads(payload)
    if config.get("schema") != 1 or not isinstance(config.get("examples"), list):
        raise ValueError("Invalid onset training manifest")
    training = config.get("training_seeds")
    excluded = config.get("excluded_seeds")
    if (
        not isinstance(training, list)
        or not training
        or not isinstance(excluded, list)
        or any(type(seed) is not int for seed in training + excluded)
        or len(set(training)) != len(training)
        or len(set(excluded)) != len(excluded)
        or set(training) & set(excluded)
    ):
        raise ValueError("Onset manifest requires explicit disjoint seed sets")
    training_set = set(training)
    samples = []
    seeds = []
    identities: set[tuple[int, float]] = set()

    def open_npz(row: dict[str, Any], kind: str) -> Any:
        path = manifest.parent / row[f"{kind}_archive"]
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != row[f"{kind}_sha256"]:
            raise ValueError(f"Checksum mismatch for {path}")
        return np.load(path, allow_pickle=False)

    for row in config["examples"]:
        seed = row["seed"]
        redshift = float(row["redshift"])
        key = seed, redshift
        if type(seed) is not int or seed not in training_set or key in identities:
            raise ValueError("Duplicate or invalid onset training identifiers")
        identities.add(key)
        label_type = row["onset_label"]
        with open_npz(row, "native") as native, open_npz(row, "onset") as labels:
            source, atoms, es = (
                native[k] for k in ("source", "capacity", "native_es_xHI")
            )
            if label_type == "teacher_dose_history":
                if not (
                    np.array_equal(source, labels["source"])
                    and np.array_equal(atoms, labels["capacity"])
                ):
                    raise ValueError(
                        "Onset dose history uses different native input arrays"
                    )
                onset = _dose_labels(
                    labels["xHI_dose"], labels["saturation_batch"], source, atoms
                )
            elif label_type == "offline_continuous_donor_pooling":
                onset = labels["onset"]
            else:
                raise ValueError(f"Unsupported exposure label type: {label_type}")
        samples.append((source, atoms, es, redshift, float(row["box_len"]), onset))
        seeds.append(seed)
    if set(seeds) != training_set:
        raise ValueError("Onset examples do not cover all declared training seeds")
    head = fit_onset(samples, seeds=seeds, sample_cells=sample_cells, steps=steps)
    head.metadata["excluded_seeds"] = sorted(excluded)
    head.metadata["training_manifest_sha256"] = hashlib.sha256(payload).hexdigest()
    return head


def main() -> None:
    """Train and save an onset head using archived observations."""
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--manifest", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--sample-cells", type=int, default=4096)
    cli.add_argument("--steps", type=int, default=220)
    args = cli.parse_args()
    head = fit_onset_manifest(
        args.manifest, sample_cells=args.sample_cells, steps=args.steps
    )
    head.save(args.output)


if __name__ == "__main__":
    main()
