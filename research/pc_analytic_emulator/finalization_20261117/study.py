"""Freeze validation-only routes, then evaluate a new native teacher holdout.

This driver keeps learned-only and hybrid errors separate. Fallback timings
include an actual solver call; saved teacher labels are used only for scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from research.pc_analytic_emulator.model import Emulator
from research.pc_analytic_emulator.predict import _model_hashes
from research.pc_analytic_emulator.reliability import accuracy_metrics, build_policy


def _native_frame(frame: dict[str, Any], arrays: Mapping[str, Any]) -> dict[str, Any]:
    from research.pc_analytic.compare import load_native_snapshot

    native = load_native_snapshot(frame["native_directory"], brightness="none")
    meta = native["metadata"]
    if any(
        frame[key] != meta[key] for key in ("seed", "redshift", "dimension", "box_len")
    ):
        raise ValueError("Cached validation frame identity differs from native receipt")
    for cached, archived in (
        ("source", "source"),
        ("capacity", "capacity"),
        ("native_es_xHI", "native_es_xhi"),
    ):
        if not np.array_equal(arrays[cached], native[archived]):
            raise ValueError(
                "Cached validation native arrays differ from original input"
            )
    return {
        "inputs": meta["inputs"],
        "geometry": {
            key: meta["inputs"]["simulation_options"][key]
            for key in ("DIM", "HII_DIM", "BOX_LEN")
        },
        "receipt_sha256": meta["receipt_sha256"],
        "files_sha256": meta["files"],
    }


def _development_seeds(models: dict[str, Emulator], validation: list[int]) -> set[int]:
    seen = set(validation)
    for model in models.values():
        split = model.metadata.get("split_seeds")
        if not isinstance(split, dict) or not {"train", "validation"}.issubset(split):
            raise ValueError("Candidate lacks training/validation seed provenance")
        for values in split.values():
            if not isinstance(values, list) or any(
                type(seed) is not int for seed in values
            ):
                raise ValueError("Candidate has invalid seed provenance")
            seen.update(values)
    return seen


def digest(path: Path) -> str:
    """Return the SHA256 of a local file."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path: Path, record: Any) -> None:
    """Write a new finite JSON evidence record without replacing an old run."""
    with path.open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def validate(cache: Path, candidates: Path, settings: Path, output: Path) -> None:
    """Score audited validation caches and freeze a self-contained deployment.

    Parameters
    ----------
    cache
        Existing fit cache with features.json and checksum receipt.
    candidates
        JSON mapping candidate names to self-contained model directories.
    settings
        Teacher campaign configuration with analytic settings and sigma_cm2.
    output
        New directory for validation metrics and frozen deployment.
    """
    from research.pc_analytic_emulator.benchmark import _physics_compatibility
    from research.pc_analytic_emulator.hybrid import freeze_deployment

    if output.exists():
        raise FileExistsError(output)
    paths = {
        name: Path(path) for name, path in json.loads(candidates.read_text()).items()
    }
    if not paths:
        raise ValueError("At least one candidate is required")
    hashes = {name: _model_hashes(path) for name, path in paths.items()}
    models = {name: Emulator.load(path) for name, path in paths.items()}
    receipt = json.loads((cache / "receipt.json").read_text())["files"]
    if digest(cache / "features.json") != receipt["features.json"]:
        raise ValueError("Validation cache metadata changed")
    frames = json.loads((cache / "features.json").read_text())["validation_snapshots"]
    records, cache_hashes, native_inputs = [], {}, {}
    output.mkdir(parents=True)
    for frame in frames:
        path = cache / Path(frame["cache"]).name
        cache_hashes[path.name] = digest(path)
        if cache_hashes[path.name] != receipt[path.name]:
            raise ValueError("Validation cache field changed")
        with np.load(path, allow_pickle=False) as d:
            native = _native_frame(frame, d)
            native_inputs[str(frame["native_directory"])] = native
            geometry = native["geometry"]
            for name, model in models.items():
                _physics_compatibility(model, native["inputs"])
                if frame["seed"] in model.metadata.get("split_seeds", {}).get(
                    "train", []
                ):
                    raise ValueError("Validation overlaps candidate training seeds")
                started = time.perf_counter()
                prediction = model.predict(
                    d["source"],
                    d["capacity"],
                    d["native_es_xHI"],
                    frame["redshift"],
                    frame["box_len"],
                )
                elapsed = time.perf_counter() - started
                records.append(
                    {
                        "split": "validation",
                        "candidate": name,
                        "geometry": geometry,
                        "seed": frame["seed"],
                        "redshift": frame["redshift"],
                        "metrics": accuracy_metrics(
                            prediction["xHI"],
                            d["teacher_xHI"],
                            d["capacity"],
                            frame["box_len"],
                        ),
                        "inference_seconds": elapsed,
                    }
                )
    if hashes != {name: _model_hashes(path) for name, path in paths.items()}:
        raise ValueError("Model changed during validation")
    policy = build_policy(
        records, hashes, sorted({f["seed"] for f in frames}), safety_margin=0.75
    )
    write(
        output / "validation.json",
        {
            "records": records,
            "cache_hashes": cache_hashes,
            "native_inputs": native_inputs,
            "models": hashes,
        },
    )
    write(output / "policy.json", policy)
    cfg = json.loads(settings.read_text())
    freeze_deployment(
        models=paths,
        policy=policy,
        teacher_physics=next(iter(models.values())).metadata["teacher_physics"],
        teacher_settings={"sigma_cm2": cfg["sigma_cm2"], **cfg["analytic"]},
        output=output / "deployment",
    )
    write(
        output / "freeze.json",
        {
            "created_unix": time.time(),
            "policy_sha256": digest(output / "policy.json"),
            "validation_sha256": digest(output / "validation.json"),
            "blind_evaluated": False,
        },
    )


def evaluate(deployment: Path, teachers: list[Path], output: Path) -> None:
    """Evaluate frozen hybrid routes and every candidate on unseen teachers.

    Parameters
    ----------
    deployment
        Previously frozen self-contained hybrid package.
    teachers
        Independently audited new teacher campaign roots.
    output
        New prediction/result directory; existing results cannot be replaced.
    """
    from research.pc_analytic_emulator.data import load_dataset
    from research.pc_analytic_emulator.hybrid import run

    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    policy = json.loads((deployment / "policy.json").read_text())
    models = {
        name: Emulator.load(deployment / "models" / name)
        for name in policy["model_hashes"]
    }
    development_seeds = _development_seeds(models, policy["validation_seeds"])
    rows = []
    for teacher in teachers:
        data = load_dataset(teacher)
        for sample in data.samples:
            if sample.seed in development_seeds:
                raise ValueError("Holdout overlaps candidate development seeds")
            arrays = sample.arrays
            folder = output / f"seed{sample.seed}_z{sample.redshift:g}"
            report = run(
                deployment=deployment,
                native_directory=sample.native_directory,
                output=folder,
                threads=1,
            )
            with np.load(folder / "fields.npz", allow_pickle=False) as saved:
                metrics = accuracy_metrics(
                    saved["xHI"],
                    arrays["teacher_xHI"],
                    arrays["capacity"],
                    sample.box_len,
                )
            learned = {}
            for name, model in models.items():
                started = time.perf_counter()
                pred = model.predict(
                    arrays["source"],
                    arrays["capacity"],
                    arrays["native_es_xHI"],
                    sample.redshift,
                    sample.box_len,
                )
                elapsed = time.perf_counter() - started
                learned[name] = {
                    "metrics": accuracy_metrics(
                        pred["xHI"],
                        arrays["teacher_xHI"],
                        arrays["capacity"],
                        sample.box_len,
                    ),
                    "inference_seconds": elapsed,
                    "ledger": pred["photon_ledger"],
                }
            rows.append(
                {
                    "seed": sample.seed,
                    "redshift": sample.redshift,
                    "dimension": sample.dimension,
                    "box_len": sample.box_len,
                    "hybrid_metrics": metrics,
                    "hybrid_report": report,
                    "learned": learned,
                }
            )
            logging.getLogger(__name__).info(
                json.dumps(
                    {
                        "seed": sample.seed,
                        "z": sample.redshift,
                        "hybrid_pass": metrics["passed"],
                    }
                ),
            )
    write(
        output / "evaluation.json",
        {
            "complete": True,
            "rows": rows,
            "all_hybrid_pass": all(r["hybrid_metrics"]["passed"] for r in rows),
            "policy_sha256": digest(deployment / "policy.json"),
        },
    )


def main() -> None:
    """Run the explicitly separated validation and holdout phases."""
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    v = sub.add_parser("validate")
    for name in ("cache", "candidates", "settings", "output"):
        v.add_argument("--" + name, type=Path, required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--deployment", type=Path, required=True)
    e.add_argument("--teachers", type=Path, nargs="+", required=True)
    e.add_argument("--output", type=Path, required=True)
    logging.basicConfig(level=logging.INFO)
    args = vars(p.parse_args())
    command = args.pop("command")
    (validate if command == "validate" else evaluate)(**args)


if __name__ == "__main__":
    main()
