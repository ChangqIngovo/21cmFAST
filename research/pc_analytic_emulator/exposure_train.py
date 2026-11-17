"""Fit and independently evaluate a pure coeval exposure emulator."""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from research.pc_analytic_emulator.data import compatibility_signature, load_dataset
from research.pc_analytic_emulator.exposure_features import exposure_features
from research.pc_analytic_emulator.predict import _digest


def _prepare(task: tuple[dict, Path, int]) -> str:
    record, output, depth = task
    path = Path(record["fields"])
    if _digest(path) != record["sha256"]:
        raise ValueError("Teacher field changed after verification")
    with np.load(path, allow_pickle=False) as arrays:
        source, atoms, target = (
            arrays[name] for name in ("source", "capacity", "teacher_xHI")
        )
    _, near, far, onset, escape = exposure_features(
        source, atoms, depth=depth, gains=(1.0, 1.0)
    )
    mean = atoms.mean()
    np.savez_compressed(
        output,
        source=source / mean,
        atoms=atoms / mean,
        target=target,
        near=near,
        far=far,
        onset=onset,
        escape=escape,
        load=np.mean(source / mean),
        opacity=record["opacity"],
    )
    return str(output)


def prepare(args: argparse.Namespace) -> None:
    """Audit training/validation campaigns and materialize native-only features."""
    if args.workers < 1 or args.depth < 1:
        raise ValueError("Worker count and depth must be positive")
    records = []
    signature = None
    seen = set()
    train_seeds = set(args.train_seeds or [])
    for split in ("train", "validation"):
        roots = getattr(args, split)
        for root in roots:
            dataset = load_dataset(root)
            identity = compatibility_signature(dataset, allow_mixed_geometry=True)
            if signature is None:
                signature = identity
            if signature != identity:
                raise ValueError("Teacher physical and numerical targets differ")
            for case in dataset.samples:
                if split == "train" and train_seeds and case.seed not in train_seeds:
                    continue
                key = (case.seed, case.redshift)
                if key in seen:
                    raise ValueError(f"Duplicate training/validation realization {key}")
                seen.add(key)
                records.append(
                    {
                        "split": split,
                        "seed": case.seed,
                        "redshift": case.redshift,
                        "dimension": case.dimension,
                        "box_len": case.box_len,
                        "dim": case.metadata["input"]["dimension_highres"]
                        if "dimension_highres" in case.metadata["input"]
                        else json.loads(
                            (case.native_directory / "native.json").read_text()
                        )["identity"]["inputs"]["simulation_options"]["DIM"],
                        "opacity": case.metadata["solver_metadata"][
                            "optical_depth_per_normalized_atom_per_cell_chord"
                        ],
                        "fields": str(case.fields_path),
                        "sha256": dataset.file_hashes[str(case.fields_path)],
                    }
                )
    seeds = {
        split: sorted({row["seed"] for row in records if row["split"] == split})
        for split in ("train", "validation")
    }
    if not all(seeds.values()) or set(seeds["train"]) & set(seeds["validation"]):
        raise ValueError("Training and validation need disjoint realization seeds")
    if train_seeds and set(seeds["train"]) != train_seeds:
        raise ValueError("Requested train seed missing")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "protocol.json").write_text(
        json.dumps({"records": records, "seeds": seeds, "depth": args.depth}, indent=2)
    )
    jobs = []
    for record in records:
        name = f"seed{record['seed']}_z{record['redshift']:g}.npz"
        record["cache"] = name
        jobs.append((record, args.output / name, args.depth))
    started = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for record, path in zip(records, pool.map(_prepare, jobs), strict=True):
            record["cache_sha256"] = _digest(Path(path))
            logging.getLogger(__name__).info("Prepared %s", Path(path).name)
    report = {
        "records": records,
        "seeds": seeds,
        "depth": args.depth,
        "signature": signature,
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output / "cache.json").write_text(
        json.dumps(report, indent=2, allow_nan=False)
    )


def main() -> None:
    """Prepare supervised data or fit and freeze an exposure emulator."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cache = commands.add_parser("prepare")
    for split in ("train", "validation"):
        cache.add_argument("--" + split, type=Path, nargs="+", required=True)
    cache.add_argument("--train-seeds", type=int, nargs="+")
    cache.add_argument("--output", type=Path, required=True)
    cache.add_argument("--workers", type=int, default=1)
    cache.add_argument("--depth", type=int, default=26)
    fit_command = commands.add_parser("fit")
    fit_command.add_argument("--cache", type=Path, required=True)
    fit_command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        from research.pc_analytic_emulator.exposure_fit import fit

        fit(args.cache, args.output)


if __name__ == "__main__":
    main()
