"""Reproduce the frozen native teacher campaigns and their original receipts.

The native Python executable must already provide the supported 21cmFAST build.
This launcher freezes tracked research code in a detached worktree and records
the actual native runtime. Blind and geometry-transfer outputs are generated
but never scientifically evaluated by this program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)
COMMON_Z = [10, 9, 8, 7.5, 7, 6.5, 6]
PROFILES = {
    "development_original": (list(range(2101, 2109)), COMMON_Z, 16, 50.0),
    "development_external": (list(range(2109, 2113)), COMMON_Z, 16, 50.0),
    "train_new": (
        list(range(2121, 2129)),
        [10, 9, 8, 7.5, 7, 6.75, 6.5, 6.25, 6, 5.75],
        16,
        50.0,
    ),
    "validation": (list(range(2113, 2117)), COMMON_Z, 16, 50.0),
    "blind": (list(range(2117, 2121)), COMMON_Z, 16, 50.0),
    "transfer_resolution": ([2131, 2132], [8, 7.5, 6.5, 6], 32, 50.0),
    "transfer_volume": ([2141, 2142], [8, 7.5, 6.5, 6], 32, 256.0),
}
ANALYTIC = {
    "batch_tail_tolerance": 0.0001,
    "boundary": "source_periodic_cube",
    "directions": 96,
    "exhaustion_tolerance": 1e-7,
    "exposure_steps": 32,
    "max_sweeps": 10000,
    "transport_mode": "characteristic",
}


def digest(path: Path) -> str:
    """Return the SHA256 of an input or receipt."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(path: Path, value: dict[str, Any]) -> None:
    """Write a new immutable record, or require an exact match on resume."""
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Frozen identity changed: {path}")
    else:
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def git(source: Path, *arguments: str) -> str:
    """Read or update only the requested repository worktree."""
    return subprocess.check_output(
        ["git", "-C", str(source), *arguments], text=True
    ).strip()


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    """Freeze source, runtime identity and all selected campaign configurations."""
    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frozen_source = output / "source"
    commit = git(source, "rev-parse", args.commit)
    if not frozen_source.exists():
        git(source, "worktree", "add", "--detach", str(frozen_source), commit)
    if git(frozen_source, "rev-parse", "HEAD") != commit:
        raise ValueError("Output worktree belongs to a different source commit")
    template = Path(__file__).with_name("native_template.json")
    copied_template = output / "native_template.json"
    if copied_template.exists() and digest(copied_template) != digest(template):
        raise ValueError("Frozen native input template changed")
    if not copied_template.exists():
        shutil.copyfile(template, copied_template)
    probe = (
        "import json; from research.pc_analytic.campaign import "
        "_runtime_identity,_implementation_hashes; "
        "print(json.dumps({'native_runtime':_runtime_identity(),"
        "'implementation_sha256':_implementation_hashes()}))"
    )
    # Preserve the venv executable symlink: resolving it would select base Python.
    executable = str(Path(args.native_python).expanduser().absolute())
    runtime = json.loads(
        subprocess.check_output(
            [executable, "-c", probe],
            cwd=frozen_source,
            text=True,
            env=args.environment,
        )
    )
    configs = {}
    for name in args.profiles:
        seeds, redshifts, dimension, box_len = PROFILES[name]
        config = {
            "native_template_json": str(copied_template),
            "native_template_sha256": digest(copied_template),
            "native_extension_sha256": runtime["native_runtime"][
                "native_extension_sha256"
            ],
            "dimension": dimension,
            "highres_dimension": 96,
            "box_len": box_len,
            "seeds": seeds,
            "redshifts": redshifts,
            "workers": args.workers,
            "threads": args.threads,
            "cpus": args.workers * args.threads,
            "brightness": "native",
            "sigma_cm2": 2.7173135802469135e-18,
            "analytic": ANALYTIC,
        }
        config_path = output / f"{name}_config.json"
        freeze(config_path, config)
        configs[name] = {
            "path": str(config_path),
            "sha256": digest(config_path),
            "output": str(output / f"teacher_{name}"),
            "requested_snapshots": len(seeds) * len(redshifts),
        }
    protocol = {
        "schema": 1,
        "source_commit": commit,
        "source_path": str(frozen_source),
        "native_python": executable,
        "runtime": runtime,
        "template_sha256": digest(template),
        "campaign_configs": configs,
        "train_seeds": list(range(2101, 2113)) + list(range(2121, 2129)),
        "validation_seeds": list(range(2113, 2117)),
        "blind_seeds": list(range(2117, 2121)),
        "transfer_seeds": [2131, 2132, 2141, 2142],
        "blind_policy": "No scientific evaluation or selection using blind/transfer outputs until the chosen model and all inference settings are frozen",
        "generator_sha256": digest(Path(__file__)),
    }
    freeze(output / "generation_protocol.json", protocol)
    return protocol


def collection(output: Path, protocol: dict[str, Any]) -> None:
    """Build references to four unchanged campaigns without relabeling receipts."""
    selected = (
        "development_original",
        "development_external",
        "train_new",
        "validation",
    )
    if any(name not in protocol["campaign_configs"] for name in selected):
        return
    campaigns = []
    for name in selected:
        path = Path(protocol["campaign_configs"][name]["output"])
        campaigns.append(
            {
                "name": name,
                "path": str(path),
                "completion_sha256": digest(path / "completion.json"),
                "identity_sha256": digest(path / "identity.json"),
                "manifest_sha256": digest(path / "dataset_manifest.json"),
            }
        )
    destination = output / "training_collection"
    destination.mkdir(exist_ok=True)
    freeze(
        destination / "collection_manifest.json",
        {
            "schema": 1,
            "kind": "analytic-teacher-campaign-collection",
            "campaigns": campaigns,
            "split_protocol_sha256": digest(output / "generation_protocol.json"),
            "creation_scope": "Original campaign receipts retained; 164 training and 28 validation snapshots; blind/transfer outputs excluded",
        },
    )


def main() -> None:
    """Run selected campaigns with frozen identities and bounded parallelism."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path(__file__).resolve().parents[4]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-python", default=sys.executable)
    parser.add_argument("--commit", default="HEAD")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--profiles", choices=tuple(PROFILES), nargs="+", default=list(PROFILES)
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if min(args.workers, args.threads) < 1 or len(set(args.profiles)) != len(
        args.profiles
    ):
        raise ValueError("Use positive resources and unique campaign profiles")
    environment = os.environ.copy()
    environment.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        PYTHONNOUSERSITE="1",
        MPLBACKEND="Agg",
        OMP_NUM_THREADS=str(args.threads),
        OMP_THREAD_LIMIT=str(args.threads),
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OMP_WAIT_POLICY="PASSIVE",
        OMP_DYNAMIC="FALSE",
    )
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    args.environment = environment
    protocol = prepare(args)
    output = args.output.resolve()
    for name, spec in protocol["campaign_configs"].items():
        if digest(Path(spec["path"])) != spec["sha256"]:
            raise ValueError("Campaign configuration changed after freeze")
        for relative, expected in protocol["runtime"]["implementation_sha256"].items():
            if digest(Path(protocol["source_path"]) / relative) != expected:
                raise ValueError("Teacher source changed after freeze")
        command = [
            protocol["native_python"],
            "-m",
            "research.pc_analytic.campaign",
            "--config",
            spec["path"],
            "--output",
            spec["output"],
        ]
        LOGGER.info("Generating %s (%d snapshots)", name, spec["requested_snapshots"])
        with (output / f"{name}.log").open("a") as log:
            subprocess.run(
                command,
                cwd=protocol["source_path"],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        summary = json.loads((Path(spec["output"]) / "summary.json").read_text())
        if not summary["complete"] or not summary["all_photon_ledgers_passed"]:
            raise ValueError("Campaign completion or photon audit failed")
        LOGGER.info(
            "%s completed: %d snapshots; maximum photon residual %.3g",
            name,
            summary["completed_snapshots"],
            summary["maximum_relative_photon_residual"],
        )
    collection(output, protocol)


if __name__ == "__main__":
    main()
