"""Benchmark an explicitly frozen refinement selection without any fitting."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import uuid
from pathlib import Path

from research.pc_analytic_emulator import benchmark
from research.pc_analytic_emulator.completion_20261116.reproduction.train_and_validate import (
    _dataset_identity,
    _receipt,
    _verify_inputs,
)
from research.pc_analytic_emulator.refinement_20261117.reproduction import (
    freeze_completed,
)

PROFILES = {
    "blind_original": ([2201, 2202], 16, 50.0),
    "blind_resolution": ([2211, 2212], 32, 50.0),
    "blind_volume": ([2221, 2222], 32, 256.0),
    "blind_unseen_box": ([2231, 2232], 32, 100.0),
}


def digest(path: Path) -> str:
    """Hash exact bytes without interpreting their scientific contents."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _freeze(path: Path, value: dict) -> None:
    if path.exists():
        if _json(path) != value:
            raise ValueError(f"Frozen evaluation record changed: {path}")
        return
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run(
    campaign_root: str | Path,
    selection_path: str | Path,
    *,
    teacher_root: str | Path | None = None,
) -> dict:
    """Audit all four blind archives, then benchmark the unchanged selection.

    The selection must already exist in campaign_root. teacher_root optionally
    locates relocated teacher_blind_* directories without copying their arrays.
    All four input hashes are frozen before the first prediction is computed.
    """
    root, selection_path = Path(campaign_root).resolve(), Path(selection_path).resolve()
    if selection_path != root / "selection.json" or not selection_path.is_file():
        raise ValueError("Evaluation requires the existing campaign selection.json")
    selected = freeze_completed.verify(selection_path)
    selected_model = (root / selected["selected_model_root"]).resolve()
    previous = (root / selected["candidates"]["previous_model"]["model_root"]).resolve()
    teachers = root if teacher_root is None else Path(teacher_root).resolve()
    source_hashes = benchmark._inference_source_hashes()
    protocol = {
        "schema": 1,
        "selection_sha256": digest(selection_path),
        "training_protocol_sha256": selected["training_protocol_sha256"],
        "evaluator_sha256": digest(Path(__file__)),
        "benchmark_sha256": digest(Path(benchmark.__file__)),
        "independent_metrics_sha256": digest(Path(benchmark.recompute.__file__)),
        "inference_source_sha256": source_hashes,
        "rays_per_phase": 4096,
        "python_version": list(sys.version_info[:3]),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "matplotlib")
        },
        "models": {
            "selected": {
                "root": str(selected_model),
                "files_sha256": selected["model_files_sha256"],
            },
            "previous": {
                "root": str(previous),
                "files_sha256": selected["candidates"]["previous_model"][
                    "model_files_sha256"
                ],
            },
        },
        "evaluations": {
            name: {
                "root": str(teachers / f"teacher_{name}"),
                "seeds": seeds,
                "dimension": dimension,
                "box_len": box,
            }
            for name, (seeds, dimension, box) in PROFILES.items()
        },
    }
    _freeze(root / "evaluation_protocol.json", protocol)
    gate = {
        "training_protocol_sha256": selected["training_protocol_sha256"],
        "selection_sha256": digest(selection_path),
        "model_files_sha256": selected["model_files_sha256"],
    }
    _freeze(root / "evaluation_started.json", gate)
    # No new-blind scientific array is opened above this immutable gate.
    inputs = {}
    for name, spec in protocol["evaluations"].items():
        path = Path(spec["root"])
        configuration = _json(path / "identity.json")["configuration"]
        if (
            configuration["dimension"] != spec["dimension"]
            or configuration["box_len"] != spec["box_len"]
            or configuration["redshifts"] != [10, 9, 8, 7.5, 7, 6.5, 6]
        ):
            raise ValueError(
                "Blind geometry/redshift profile differs from the declaration"
            )
        inputs[name] = _dataset_identity(path, expected_seeds=spec["seeds"])
    _freeze(root / "evaluation_inputs.json", inputs)
    outputs = {}
    for name, spec in protocol["evaluations"].items():
        _verify_inputs(inputs[name])
        destination = root / f"benchmark_{name}"
        if not (destination / "COMPLETED").exists():
            if destination.exists():
                destination.rename(
                    destination.with_name(
                        f"{destination.name}.partial.{uuid.uuid4().hex}"
                    )
                )
            benchmark.run(
                spec["root"],
                selected_model,
                previous,
                destination,
                seeds=spec["seeds"],
                rays=4096,
            )
        _receipt(destination, "receipt.json", "files_sha256")
        manifest = _json(destination / "manifest.json")
        expected = {
            "teacher_root": spec["root"],
            "requested_seeds": spec["seeds"],
            "rays_per_phase": 4096,
            "models": protocol["models"],
            "benchmark_sha256": protocol["benchmark_sha256"],
            "independent_metrics_sha256": protocol["independent_metrics_sha256"],
            "inference_source_sha256": source_hashes,
            "fitting_performed": False,
            "selection_performed": False,
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError("Benchmark belongs to different frozen settings or models")
        if (
            _json(destination / "inputs.json")["files_sha256"]
            != inputs[name]["files_sha256"]
        ):
            raise ValueError(
                "Benchmark input identity differs from the frozen archives"
            )
        _verify_inputs(inputs[name])
        freeze_completed.verify(selection_path)
        _freeze(root / "evaluation_protocol.json", protocol)
        _freeze(root / "evaluation_started.json", gate)
        outputs[name] = {
            "report_sha256": digest(destination / "report.json"),
            "receipt_sha256": digest(destination / "receipt.json"),
        }
    if (
        benchmark._inference_source_hashes() != source_hashes
        or digest(Path(__file__)) != protocol["evaluator_sha256"]
    ):
        raise ValueError("Inference/evaluation code changed during benchmarking")
    report = {
        "complete": True,
        "selection_sha256": digest(selection_path),
        "evaluations": outputs,
        "refitting_performed": False,
    }
    _freeze(root / "evaluation_completion.json", report)
    return report


def main() -> None:
    """Run only post-freeze benchmarks of the four declared blind profiles."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path)
    args = parser.parse_args()
    sys.stdout.write(
        json.dumps(
            run(args.campaign_root, args.selection, teacher_root=args.teacher_root)
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
