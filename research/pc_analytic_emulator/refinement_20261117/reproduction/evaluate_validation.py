"""Score unchanged emulator weights on a development config's validation split.

The scorer is the same fixed-field evaluator used by warm-start model selection.
It performs no fitting, no model selection and no standalone-inference geometry
restriction: unseen grids are deliberate validation transfer checks. Physical
inputs must still match. Reports omit machine-local paths unless requested.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from research.pc_analytic_emulator.benchmark import (
    _digest,
    _inference_source_hashes,
    _model_hashes,
    _sample_compatibility,
)
from research.pc_analytic_emulator.data import (
    load_dataset,
    sample_geometries,
    split_samples,
)
from research.pc_analytic_emulator.model import Emulator
from research.pc_analytic_emulator.train import validate_config
from research.pc_analytic_emulator.warmstart import _score_candidate

OBJECTIVE_KEYS = (
    "spectral_weight",
    "spectral_power_floor",
    "front_weight",
    "field_variance_floor",
    "k_fraction",
)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "snapshot_count": len(rows),
        "validation_objective": float(np.mean([r["projected_loss"] for r in rows])),
        "field_mse": float(np.mean([r["field_mse"] for r in rows])),
        "mean_snapshot_field_rmse": float(
            np.mean([np.sqrt(r["field_mse"]) for r in rows])
        ),
        "mean_absolute_delta_Q_mass": float(
            np.mean([abs(r["delta_Q_mass"]) for r in rows])
        ),
        "maximum_absolute_delta_Q_mass": max(abs(r["delta_Q_mass"]) for r in rows),
    }


def run(
    model_root: str | Path,
    configuration: str | Path,
    output: str | Path,
    *,
    teacher_root: str | Path | None = None,
    include_paths: bool = False,
) -> dict[str, Any]:
    """Audit and score a fixed reference with the exact training selection loss.

    Parameters
    ----------
    model_root
        Saved self-contained emulator, including any frozen base models.
    configuration
        Development-only training JSON with explicit train/validation seeds,
        an empty test split and no external test archive.
    output
        New report JSON; existing paths are never overwritten.
    teacher_root
        Optional portable replacement for the JSON's input_root directory.
    include_paths
        Include private absolute provenance paths. The default report uses
        checksums and portable input labels, without machine-local paths.

    Returns
    -------
    dict
        Equal-snapshot validation loss, geometry/seed summaries, per-frame
        objective components, physical compatibility and unchanged-file checks.
    """
    model_root, configuration, output = (
        Path(value).resolve() for value in (model_root, configuration, output)
    )
    if output.exists():
        raise FileExistsError(output)
    if output.is_relative_to(model_root):
        raise ValueError("Validation output must be outside the saved model")
    model_hashes = _model_hashes(model_root)
    source_hashes = _inference_source_hashes()
    script_hash = _digest(Path(__file__))
    config_hash = _digest(configuration)
    raw_config = json.loads(configuration.read_text())
    if teacher_root is not None:
        raw_config["input_root"] = str(Path(teacher_root).resolve())
    config = validate_config(raw_config)
    if config["test_seeds"] != [] or config["external_test_root"] is not None:
        raise ValueError("Fixed validation requires a development-only configuration")
    dataset = load_dataset(config["input_root"])
    archive_roots = [dataset.root] + [
        Path(record["root"]) for record in dataset.identity.get("campaigns", [])
    ]
    if any(output.is_relative_to(root) for root in archive_roots):
        raise ValueError("Validation output must be outside every teacher archive")
    splits = split_samples(
        dataset,
        train_seeds=config["train_seeds"],
        validation_seeds=config["validation_seeds"],
        test_seeds=[],
        allow_empty_test=True,
    )
    samples = list(splits["validation"])
    model = Emulator.load(model_root)
    compatibility = _sample_compatibility(model, dataset.physics, samples)
    objective = {key: config[key] for key in OBJECTIVE_KEYS}
    with tempfile.TemporaryDirectory(prefix="pc_validation_") as temporary:
        frames = []
        for index, sample in enumerate(samples):
            path = Path(temporary) / f"frame{index}.npz"
            np.savez(
                path,
                **{
                    key: sample.arrays[key]
                    for key in ("source", "capacity", "native_es_xHI", "teacher_xHI")
                },
            )
            frames.append(
                {
                    "seed": sample.seed,
                    "redshift": sample.redshift,
                    "box_len": sample.box_len,
                    "cache": str(path),
                }
            )
        result = _score_candidate(
            "unchanged_reference",
            model,
            frames,
            objective,
            stage="frozen_reference",
            field_sha256=model_hashes["field_model.npz"],
            budget_kind=model.metadata.get("budget_kind", "saved_model"),
        )
    rows = result["snapshots"]
    for row, sample in zip(rows, samples, strict=True):
        row.update(
            geometry=sample_geometries([sample])[0],
            teacher_fields_sha256=dataset.file_hashes[
                str(sample.fields_path.resolve())
            ],
        )
    by_geometry = []
    for geometry in sample_geometries(samples):
        selected = [row for row in rows if row["geometry"] == geometry]
        by_geometry.append({"geometry": geometry, **_aggregate(selected)})
    if _model_hashes(model_root) != model_hashes:
        raise ValueError("Model changed during fixed validation")
    if (
        _inference_source_hashes() != source_hashes
        or _digest(Path(__file__)) != script_hash
    ):
        raise ValueError("Implementation changed during fixed validation")
    if _digest(configuration) != config_hash:
        raise ValueError("Configuration changed during fixed validation")
    if any(
        _digest(Path(path)) != digest for path, digest in dataset.file_hashes.items()
    ):
        raise ValueError("Teacher inputs changed during fixed validation")
    public_config = copy.deepcopy(config)
    public_config["input_root"] = "TEACHER_ROOT"
    input_digests = sorted(dataset.file_hashes.values())
    report = {
        "schema": 1,
        "complete": True,
        "fitting_performed": False,
        "model_selection_performed": False,
        "test_evaluated": False,
        "model_weights_unchanged": True,
        "teacher_inputs_unchanged": True,
        "inference_implementation_unchanged": True,
        "averaging": "Equal weight per validation snapshot; identical to warm-start fixed-field candidate selection",
        "scope": "Unchanged reference weights; validation transfer geometries are evaluated without a deployment support restriction; no training or normalization intervention",
        "objective": objective,
        "raw_auxiliary_excluded": True,
        "configuration": public_config,
        "configuration_sha256": config_hash,
        "model_sha256": model_hashes,
        "inference_source_sha256": source_hashes,
        "evaluation_script_sha256": script_hash,
        "teacher_file_count": len(input_digests),
        "teacher_content_digest": hashlib.sha256(
            json.dumps(input_digests).encode()
        ).hexdigest(),
        "validation_seeds": sorted({sample.seed for sample in samples}),
        "physical_compatibility": compatibility,
        **_aggregate(rows),
        "by_geometry": by_geometry,
        "by_seed": result["validation_by_seed"],
        "snapshots": rows,
    }
    if include_paths:
        report["paths"] = {
            "model": str(model_root),
            "configuration": str(configuration),
            "teacher": str(dataset.root),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return report


def main() -> None:
    """Run the portable fixed-model validation command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--teacher-root", type=Path)
    parser.add_argument("--include-paths", action="store_true")
    args = parser.parse_args()
    report = run(
        args.model,
        args.config,
        args.output,
        teacher_root=args.teacher_root,
        include_paths=args.include_paths,
    )
    sys.stdout.write(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "complete",
                    "snapshot_count",
                    "validation_objective",
                    "field_mse",
                    "by_geometry",
                )
            },
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
