"""Freeze read-only photon/MFP postprocessing of completed coeval fields."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from .prepare import digest, fingerprint_difference, source_fingerprint, write_json

EXPERIMENT = "coeval-photon-mfp-validation"
PREPARED_FILES = ("commit.txt", "input_run.txt", "python_path.txt", "runtime.env",
                  "submit.pbs", "experiment_config.json", "run_manifest.json")


def result_root(path):
    """Accept a coeval run or its results directory, never a native history run."""
    path = Path(path).resolve()
    if (path / "coeval_report.json").is_file():
        return path
    if (path / "results/coeval_report.json").is_file():
        return path / "results"
    raise ValueError("Input must be a completed coeval PC comparison run or its results directory")


def configuration(environment):
    thresholds = [float(value) for value in environment.get(
        "PC_MFP_THRESHOLDS", "0.1,0.5,0.9").replace(",", " ").split()]
    config = {"threads": int(environment.get("PC_THREADS", "8")),
              "rays": int(environment.get("PC_MFP_RAYS", "100000")),
              "seed": int(environment.get("PC_MFP_SEED", "20261103")),
              "thresholds": thresholds,
              "max_distance_boxes": float(environment.get("PC_MFP_MAX_DISTANCE_BOXES", "1"))}
    if not 1 <= config["threads"] <= 32:
        raise ValueError("PC_THREADS must be in [1,32]")
    if not 100 <= config["rays"] <= 100_000_000:
        raise ValueError("PC_MFP_RAYS must be in [100,100000000]")
    if not 0 <= config["seed"] < 2**64:
        raise ValueError("PC_MFP_SEED must be an unsigned 64-bit integer")
    if not thresholds or len(set(thresholds)) != len(thresholds) or any(
            not math.isfinite(value) or not 0 < value < 1 for value in thresholds):
        raise ValueError("PC_MFP_THRESHOLDS must be unique values strictly between zero and one")
    if not math.isfinite(config["max_distance_boxes"]) or not 0 < config["max_distance_boxes"] <= 10:
        raise ValueError("PC_MFP_MAX_DISTANCE_BOXES must be in (0,10]")
    return config


def input_fingerprint(path):
    root = result_root(path)
    if not (root / "COMPLETED").is_file():
        raise ValueError("Coeval input is not complete")
    names = ["identity.json", "coeval_report.json", "report_receipt.json", "COMPLETED"]
    receipts = sorted(root.glob("seed*/z*/receipt.json"))
    if not receipts:
        raise ValueError("Coeval input has no snapshot receipts")
    names.extend(str(receipt.relative_to(root)) for receipt in receipts)
    # Full fields hashes are checked while streaming snapshots by validate.py.
    fingerprints = {name: digest(root / name) for name in names}
    identity = json.loads((root / "identity.json").read_text())
    native_manifest = Path(identity["input_root"]) / "manifest.json"
    native_digest = digest(native_manifest)
    if native_digest != identity["input_manifest_sha256"]:
        raise ValueError("Native input manifest differs from coeval identity")
    fingerprints["native_input_manifest:" + str(native_manifest.resolve())] = native_digest
    return fingerprints


def runtime_fingerprint():
    import matplotlib
    import numpy
    compiler = shutil.which(os.environ.get("CXX", "g++"))
    if compiler is None:
        raise RuntimeError("A C++ compiler with OpenMP support is required (CXX or g++)")
    compiler = str(Path(compiler).resolve())
    version = subprocess.run([compiler, "--version"], check=True, capture_output=True,
                             text=True, timeout=30).stdout
    return {"python": str(Path(sys.executable).resolve()), "python_version": platform.python_version(),
            "numpy_version": numpy.__version__, "matplotlib_version": matplotlib.__version__,
            "cxx": compiler, "cxx_sha256": digest(compiler), "cxx_version": version}


def prepare(run, environment):
    run = Path(run).resolve()
    if (run / "PREPARED").exists() or (run / "run_manifest.json").exists():
        raise FileExistsError("Prepare a fresh validation directory")
    input_root = result_root((run / "input_run.txt").read_text().strip())
    input_run = input_root.parent if input_root.name == "results" else input_root
    if run == input_run or run in input_run.parents or input_run in run.parents:
        raise ValueError("Validation output must be separate from the coeval input run")
    config = configuration(environment)
    manifest = {"schema": 1, "experiment": EXPERIMENT, "run": str(run),
                "input_results": str(input_root), "commit": (run / "commit.txt").read_text().strip(),
                "configuration": config, "source_sha256": source_fingerprint(run / "source"),
                "input_sha256": input_fingerprint(input_root), "runtime": runtime_fingerprint(),
                "ncpus": config["threads"], "reruns_simulations": False,
                "scope": "Independent photon-budget audit and geometric first-exit MFP on saved fields"}
    write_json(run / "experiment_config.json", config)
    write_json(run / "run_manifest.json", manifest)
    (run / "prepared.sha256").write_text("".join(
        f"{digest(run / name)}  {name}\n" for name in PREPARED_FILES))
    (run / "PREPARED").write_text("Frozen read-only coeval field validation.\n")
    return manifest


def verify(run, cpus=None):
    run = Path(run).resolve()
    if not (run / "PREPARED").is_file():
        raise ValueError("Validation run is not prepared")
    checked = []
    for line in (run / "prepared.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if name not in PREPARED_FILES or name in checked or digest(run / name) != expected:
            raise ValueError("Prepared file changed or unexpected checksum entry: " + name)
        checked.append(name)
    if set(checked) != set(PREPARED_FILES):
        raise ValueError("Prepared checksums are incomplete")
    manifest = json.loads((run / "run_manifest.json").read_text())
    if manifest.get("schema") != 1 or manifest.get("experiment") != EXPERIMENT:
        raise ValueError("Unknown validation run format")
    if manifest["run"] != str(run):
        raise ValueError("Prepared validation run moved")
    if cpus is not None and cpus != manifest["ncpus"]:
        raise ValueError("CPU allocation differs from frozen validation setting")
    differences = fingerprint_difference(manifest["source_sha256"], source_fingerprint(run / "source"))
    if any(differences.values()):
        raise ValueError("Frozen source changed: " + json.dumps(differences))
    if manifest["input_sha256"] != input_fingerprint(manifest["input_results"]):
        raise ValueError("Coeval input identity changed")
    if manifest["runtime"] != runtime_fingerprint():
        raise ValueError("Compiler or Python dependencies changed")
    if json.loads((run / "experiment_config.json").read_text()) != manifest["configuration"]:
        raise ValueError("Prepared validation configuration changed")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cpus", type=int)
    args = parser.parse_args()
    manifest = prepare(args.run, os.environ) if args.command == "prepare" else verify(args.run, args.cpus)
    print(json.dumps({"command": args.command, "commit": manifest["commit"],
                      "input_results": manifest["input_results"], "configuration": manifest["configuration"]}))


if __name__ == "__main__":
    main()
