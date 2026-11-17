"""Refit pure exposure heads on their deployed features and projected spectra.

The JSON manifest lists checksum-bound native NPZ snapshots with explicit
train/validation splits. Test labels are never accepted by this entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from research.pc_analytic_emulator.data import compatibility_signature, load_dataset
from research.pc_analytic_emulator.exposure_emulator import (
    ExposureEmulator,
)
from research.pc_analytic_emulator.exposure_features import exposure_features
from research.pc_analytic_emulator.exposure_fit import _evaluate, _load, fit_budget
from research.pc_analytic_emulator.exposure_spectral_fit import (
    SpectralTrainingFrame,
    fit_projected_field,
)
from research.pc_analytic_emulator.predict import _digest


def _validate_records(rows: list[dict], model: ExposureEmulator) -> dict[str, set[int]]:
    """Recheck cohort separation at preparation and again before fitting."""
    if not rows:
        raise ValueError("Refinement records must be nonempty")
    seen, names, archives = set(), set(), {}
    seeds = {"train": set(), "validation": set()}
    for row in rows:
        split = row["split"]
        if split not in seeds or type(row["seed"]) is not int:
            raise ValueError(
                "Only train/validation records with integer seeds accepted"
            )
        if (
            isinstance(row["redshift"], bool)
            or not np.isfinite(row["redshift"])
            or row["redshift"] < 0
            or type(row["dim"]) is not int
        ):
            raise ValueError("Invalid refinement redshift or parent grid")
        key = row["seed"], row["redshift"]
        name = f"seed{row['seed']}_z{row['redshift']:g}.npz"
        if key in seen or name in names:
            raise ValueError("Duplicate refinement realization/redshift or cache name")
        seen.add(key)
        names.add(name)
        seeds[split].add(row["seed"])
        archive_hash = row["native_sha256"]
        if archive_hash in archives and archives[archive_hash] != split:
            raise ValueError(
                "Duplicate native archive across training/validation splits"
            )
        archives[archive_hash] = split
        if "dimension" in row:
            geometry = {
                "DIM": row["dim"],
                "HII_DIM": row["dimension"],
                "BOX_LEN": row["box_len"],
            }
            if geometry not in model.metadata["training_geometries"]:
                raise ValueError("Refinement geometry differs from frozen base model")
    if not all(seeds.values()) or seeds["train"] & seeds["validation"]:
        raise ValueError("Require disjoint nonempty train/validation seeds")
    base_train = set(model.metadata["split_seeds"]["train"])
    if model.onset_head is not None:
        base_train.update(model.onset_head.metadata.get("training_seeds", []))
    base_validation = set(model.metadata["split_seeds"]["validation"])
    if seeds["validation"] & base_train or seeds["train"] & base_validation:
        raise ValueError("Refinement split leaks frozen-base development labels")
    return seeds


def _binding(row: dict) -> dict:
    return {
        key: row[key]
        for key in ("seed", "redshift", "dim", "dimension", "box_len", "native_sha256")
    }


def _audit_sources(
    parent: Path, rows: list[dict], model: ExposureEmulator
) -> tuple[list[dict], dict]:
    """Bind supervised arrays to independently audited native/teacher receipts."""
    datasets, checked, files = {}, [], {}
    for row in rows:
        if not row.get("teacher_root"):
            raise ValueError(
                "New refinement inputs require an audited teacher_root campaign"
            )
        root = (parent / row["teacher_root"]).resolve()
        if root not in datasets:
            dataset = load_dataset(root)
            signature = compatibility_signature(dataset, allow_mixed_geometry=True)
            if signature != model.metadata["teacher_signature"]:
                raise ValueError(
                    "Refinement teacher physics or numerical signature differs from base model"
                )
            datasets[root] = {(s.seed, s.redshift): s for s in dataset.samples}
            files.update(dataset.file_hashes)
        sample = datasets[root].get((row["seed"], row["redshift"]))
        if sample is None:
            raise ValueError(
                "Refinement seed/redshift is absent from audited teacher campaign"
            )
        options = sample.metadata["input"]["inputs"]["simulation_options"]
        expected = {
            "seed": sample.seed,
            "redshift": sample.redshift,
            "dim": options["DIM"],
            "dimension": sample.dimension,
            "box_len": sample.box_len,
            "native_sha256": files[str(sample.fields_path.resolve())],
        }
        actual = {**row, "dimension": row.get("dimension", sample.dimension)}
        if _binding(actual) != expected:
            raise ValueError(
                "Refinement sample identity differs from audited teacher receipt"
            )
        native_json = json.loads((sample.native_directory / "native.json").read_text())
        checked.append(
            {
                **actual,
                "provenance": {
                    "sample": expected,
                    "teacher_receipt_sha256": files[
                        str((sample.fields_path.parent / "receipt.json").resolve())
                    ],
                    "native_receipt_sha256": files[
                        str((sample.native_directory / "receipt.json").resolve())
                    ],
                    "initial_conditions_sha256": native_json[
                        "initial_conditions_sha256"
                    ],
                },
            }
        )
    _validate_provenance(checked, model.metadata["teacher_signature"], model)
    return checked, files


def _validate_provenance(
    rows: list[dict], signature: dict, model: ExposureEmulator
) -> None:
    """Recheck portable audited bindings without following original source paths."""
    if signature != model.metadata["teacher_signature"]:
        raise ValueError(
            "Refinement teacher physics or numerical signature differs from base model"
        )
    initial_splits = {}
    for row in rows:
        provenance = row.get("provenance", {})
        if provenance.get("sample") != _binding(row):
            raise ValueError(
                "Cached sample identity differs from audited teacher binding"
            )
        for name in (
            "teacher_receipt_sha256",
            "native_receipt_sha256",
            "initial_conditions_sha256",
        ):
            value = provenance.get(name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError("Invalid audited teacher provenance checksum")
        initial = provenance["initial_conditions_sha256"]
        if initial in initial_splits and initial_splits[initial] != row["split"]:
            raise ValueError("Training/validation share initial conditions")
        initial_splits[initial] = row["split"]


def _validate_cache_provenance(
    config: dict,
    model: ExposureEmulator,
    cache_hash: str,
    legacy_cache_sha256: str | None,
) -> None:
    if config.get("schema") == 1:
        if legacy_cache_sha256 is None or legacy_cache_sha256 != cache_hash:
            raise ValueError(
                "Legacy cache requires explicit matching --legacy-cache-sha256; it has no audited teacher bindings"
            )
    elif config.get("schema") == 2:
        if legacy_cache_sha256 is not None:
            raise ValueError(
                "Legacy cache override is not applicable to audited schema 2"
            )
        _validate_provenance(config["records"], config.get("teacher_signature"), model)
    else:
        raise ValueError("Unsupported refinement cache schema")


def _prepare_one(task: tuple[Path, Path, dict, Path]) -> dict:
    model_path, manifest_parent, row, output = task
    model = ExposureEmulator.load(model_path)
    path = manifest_parent / row["native_archive"]
    if _digest(path) != row["native_sha256"]:
        raise ValueError(f"Native/teacher archive changed: {path}")
    with np.load(path, allow_pickle=False) as data:
        source, atoms, es, target = [
            data[k] for k in ("source", "capacity", "native_es_xHI", "teacher_xHI")
        ]
    if _digest(path) != row["native_sha256"]:
        raise ValueError(f"Native/teacher archive changed during read: {path}")
    if (
        not np.isfinite(target).all()
        or np.any((target < 0) | (target > 1))
        or target.shape != atoms.shape
    ):
        raise ValueError("Invalid teacher field")
    geometry = {"DIM": row["dim"], "HII_DIM": len(source), "BOX_LEN": row["box_len"]}
    if geometry not in model.metadata["training_geometries"]:
        raise ValueError("Refinement geometry differs from frozen base model")
    onset = (
        None
        if model.onset_head is None
        else model.onset_head.initialize(
            source, atoms, es, row["redshift"], row["box_len"]
        )
    )
    _, near, far, onset, escape = exposure_features(
        source, atoms, depth=model.metadata["depth"], gains=(1.0, 1.0), onset=onset
    )
    opacity = (
        model.metadata["opacity_prefactor"]
        * (1 + row["redshift"]) ** 2
        * row["box_len"]
        / len(source)
    )
    mean = atoms.mean()
    name = f"seed{row['seed']}_z{row['redshift']:g}.npz"
    dest = output / name
    np.savez_compressed(
        dest,
        source=source / mean,
        atoms=atoms / mean,
        target=target,
        near=near,
        far=far,
        onset=onset,
        escape=escape,
        opacity=opacity,
    )
    return {
        **row,
        "dimension": len(source),
        "opacity": opacity,
        "cache": name,
        "cache_sha256": _digest(dest),
    }


def prepare(model_path: Path, manifest: Path, output: Path, workers: int = 1) -> None:
    """Freeze exact inference descriptors; supervised targets remain separate."""
    model = ExposureEmulator.load(model_path)
    model_hash = _digest(model_path / "model.json")
    manifest_hash = _digest(manifest)
    config = json.loads(manifest.read_text())
    rows = config["records"]
    if config.get("schema") != 2 or not rows or workers < 1:
        raise ValueError("Invalid refinement manifest or worker count")
    seeds = _validate_records(rows, model)
    rows, audited_files = _audit_sources(manifest.parent, rows, model)
    output.mkdir(parents=True, exist_ok=False)
    tasks = [(model_path, manifest.parent, row, output) for row in rows]
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        records = list(pool.map(_prepare_one, tasks))
    if model_hash != _digest(model_path / "model.json") or manifest_hash != _digest(
        manifest
    ):
        raise ValueError(
            "Frozen base model or refinement manifest changed during preparation"
        )
    if any(_digest(Path(path)) != digest for path, digest in audited_files.items()):
        raise ValueError("Audited teacher campaign changed during preparation")
    payload = {
        "schema": 2,
        "records": records,
        "seeds": {k: sorted(v) for k, v in seeds.items()},
        "base_model_sha256": model_hash,
        "manifest_sha256": manifest_hash,
        "depth": model.metadata["depth"],
        "onset_sha256": model.metadata.get("onset_sha256"),
        "teacher_signature": model.metadata["teacher_signature"],
        "audited_source_files": audited_files,
    }
    (output / "cache.json").write_text(json.dumps(payload, indent=2, allow_nan=False))


def score(rows: list[dict]) -> tuple[float, float]:
    """Minimax fixed-grid validation score, including a one-percent low-k goal."""
    if not rows:
        raise ValueError("Validation scores require nonempty snapshots")
    ratios = []
    for row in rows:
        late = row["teacher_xHI_mass"] <= 0.05
        ratios.extend(
            [
                row["MAE"] / (0.01 if late else 0.03),
                abs(row["delta_Q_mass"]) / (0.002 if late else 0.005),
            ]
        )
        if late or row["mean_relative_power_error"] is None:
            ratios.append(row["brightness_residual_rms_over_neutral_rms"] / 0.01)
        else:
            ratios.append(row["mean_relative_power_error"] / 0.01)
    if not np.isfinite(ratios).all():
        raise ValueError("Validation metrics must be finite")
    return max(ratios), float(np.mean([r["MAE"] for r in rows]))


def fit(
    model_path: Path,
    cache: Path,
    output: Path,
    *,
    maxiter: int = 160,
    legacy_cache_sha256: str | None = None,
) -> dict:
    """Select geometry coefficients by validation; retain one pure inference graph."""
    model = ExposureEmulator.load(model_path)
    cache_bytes = (cache / "cache.json").read_bytes()
    config = json.loads(cache_bytes)
    cache_hash = hashlib.sha256(cache_bytes).hexdigest()
    _validate_cache_provenance(config, model, cache_hash, legacy_cache_sha256)
    if (
        config["base_model_sha256"] != _digest(model_path / "model.json")
        or config["depth"] != model.metadata["depth"]
        or config["onset_sha256"] != model.metadata.get("onset_sha256")
    ):
        raise ValueError("Feature cache does not belong to this deployed base model")
    records = config["records"]
    seed_sets = _validate_records(records, model)
    if config["seeds"] != {key: sorted(value) for key, value in seed_sets.items()}:
        raise ValueError("Refinement cache seed index disagrees with records")
    output.mkdir(parents=True, exist_ok=False)
    field, budget = model.field_weights.copy(), model.budget_weights.copy()
    report = {
        "schema": 1,
        "base_model_sha256": _digest(model_path / "model.json"),
        "cache_sha256": cache_hash,
        "cache_schema": config["schema"],
        "input_provenance": "audited_teacher_campaigns"
        if config["schema"] == 2
        else "legacy_cache_explicit_sha256",
        "geometries": [],
        "target_lowk_mean_absolute_relative_error": 0.01,
        "selection": "Worst normalized validation metric, then field MAE; no test labels",
    }
    for i, geometry in enumerate(model.metadata["training_geometries"]):
        rows = [
            r
            for r in records
            if (r["dim"], r["dimension"], r["box_len"])
            == (geometry["DIM"], geometry["HII_DIM"], geometry["BOX_LEN"])
        ]
        train = [_load(cache, r) for r in rows if r["split"] == "train"]
        validation = [_load(cache, r) for r in rows if r["split"] == "validation"]
        if not train or not validation:
            raise ValueError("Every geometry requires training and validation")
        baseline = _evaluate(validation, field[i], budget[i])
        candidates = [
            {
                "name": "frozen_base",
                "score": score(baseline),
                "validation": baseline,
                "field": field[i].tolist(),
                "budget": budget[i].tolist(),
            }
        ]
        for budget_name, bw in [
            ("frozen", budget[i]),
            ("refit", fit_budget(train, 1e-5)),
        ]:
            budget_only = _evaluate(validation, field[i], bw)
            candidates.append(
                {
                    "name": budget_name + "_budget_only",
                    "score": score(budget_only),
                    "validation": budget_only,
                    "field": field[i].tolist(),
                    "budget": bw.tolist(),
                }
            )
            frames = [
                SpectralTrainingFrame(
                    f.source,
                    f.atoms,
                    f.target,
                    f.inputs,
                    bw,
                    f.record["box_len"],
                    record=f.record,
                )
                for f in train
            ]
            for spectral_weight, power_weight in (
                (0.3, 0.0),
                (1.0, 0.0),
                (3.0, 0.3),
                (10.0, 1.0),
            ):
                fw, receipt = fit_projected_field(
                    frames,
                    field[i],
                    spectral_weight=spectral_weight,
                    power_weight=power_weight,
                    ridge=1e-4,
                    maxiter=maxiter,
                )
                metrics = _evaluate(validation, fw, bw)
                candidate = {
                    "name": f"{budget_name}_spectral{spectral_weight}_power{power_weight}",
                    "score": score(metrics),
                    "validation": metrics,
                    "optimization": receipt,
                    "field": fw.tolist(),
                    "budget": bw.tolist(),
                }
                candidates.append(candidate)
                print(  # noqa: T201 - machine-readable batch progress
                    json.dumps(
                        {
                            "geometry": geometry,
                            "candidate": candidate["name"],
                            "score": candidate["score"],
                        }
                    ),
                    flush=True,
                )
        chosen = min(candidates, key=lambda r: tuple(r["score"]))
        field[i], budget[i] = chosen["field"], chosen["budget"]
        report["geometries"].append(
            {"geometry": geometry, "selected": chosen["name"], "candidates": candidates}
        )
        (output / "fit.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    if report["base_model_sha256"] != _digest(model_path / "model.json"):
        raise ValueError("Frozen base model changed during fitting")
    if cache_hash != _digest(cache / "cache.json"):
        raise ValueError("Frozen refinement cache metadata changed during fitting")
    np.savez(output / "weights.npz", field=field, budget=budget)
    if model.onset_head is not None:
        shutil.copyfile(model_path / "onset.npz", output / "onset.npz")
    metadata = {
        **model.metadata,
        "weights_sha256": _digest(output / "weights.npz"),
        "field_fit_depth": model.metadata["depth"],
        "refinement": {
            "fit_sha256": _digest(output / "fit.json"),
            "objective": "Postprojection field and equal-shell complex brightness",
            "base_model_sha256": report["base_model_sha256"],
            "cache_schema": report["cache_schema"],
            "input_provenance": report["input_provenance"],
        },
        "split_seeds": {
            **model.metadata["split_seeds"],
            "train": sorted(
                set(model.metadata["split_seeds"]["train"]) | seed_sets["train"]
            ),
            "validation": sorted(
                set(model.metadata["split_seeds"]["validation"])
                | seed_sets["validation"]
            ),
        },
    }
    (output / "model.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    ExposureEmulator.load(output)
    return report


def main() -> None:
    """Prepare deployed descriptors or fit seed-separated exposure heads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "fit"])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--maxiter", type=int, default=160)
    parser.add_argument(
        "--legacy-cache-sha256",
        help="Explicit SHA256 pin for reproducing an archived schema-1 cache",
    )
    args = parser.parse_args()
    if args.command == "prepare":
        if args.manifest is None:
            parser.error("prepare requires --manifest")
        prepare(args.model, args.manifest, args.output, args.workers)
    else:
        if args.cache is None:
            parser.error("fit requires --cache")
        fit(
            args.model,
            args.cache,
            args.output,
            maxiter=args.maxiter,
            legacy_cache_sha256=args.legacy_cache_sha256,
        )


if __name__ == "__main__":
    main()
