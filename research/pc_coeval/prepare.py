"""Freeze a coeval allocation comparison without rebuilding native histories."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re


EXPERIMENT = "native-coeval-pc-script"
DEFAULT_REDSHIFTS = (10., 9., 8., 7.5, 7.25, 7., 6.75, 6.5, 6.375, 6.)
PREPARED_FILES = ("commit.txt", "input_run.txt", "python_path.txt", "runtime.env",
                  "submit.pbs", "experiment_config.json", "run_manifest.json")
SCOPE = (
    "Frozen coeval native 21cmFAST source and atom grids, periodic boundaries, "
    "instantaneous sharp-front photon allocation, no recombinations or feedback. "
    "This explicitly replaces the causal RT propagation/time approximation; "
    "original SCRIPT is an independent allocation reference, not a truth field."
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_fingerprint(source):
    source = Path(source)
    return {str(path.relative_to(source)): digest(path)
            for path in sorted(source.rglob("*")) if path.is_file()}


def fingerprint_difference(expected, actual):
    return {
        "missing": sorted(expected.keys() - actual.keys()),
        "changed": sorted(name for name in expected.keys() & actual.keys()
                          if expected[name] != actual[name]),
        "added": sorted(actual.keys() - expected.keys()),
    }


def parse_redshifts(value):
    try:
        values = [float(item) for item in value.replace(",", " ").split()]
    except (ValueError, AttributeError) as error:
        raise ValueError("PC_REDSHIFTS must contain numeric redshifts") from error
    if not values or any(not math.isfinite(z) or z < 0 for z in values):
        raise ValueError("PC_REDSHIFTS must contain finite nonnegative redshifts")
    if len(set(values)) != len(values):
        raise ValueError("PC_REDSHIFTS must not repeat snapshots")
    return sorted(values, reverse=True)


def configuration(original, environment):
    config = {
        "threads": int(environment.get("PC_THREADS", "8")),
        "redshifts": parse_redshifts(environment.get(
            "PC_REDSHIFTS", ",".join(map(str, DEFAULT_REDSHIFTS)))),
        "tolerance": float(environment.get("PC_TOLERANCE", "1e-8")),
        "max_iterations": int(environment.get("PC_MAX_ITERATIONS", "1000")),
        "max_interactions": int(environment.get("PC_MAX_INTERACTIONS", "1000000000000")),
        "script_timeout": float(environment.get("PC_SCRIPT_TIMEOUT", "900")),
    }
    if not 1 <= config["threads"] <= 32:
        raise ValueError("PC_THREADS must be an integer between 1 and 32")
    if not math.isfinite(config["tolerance"]) or not 0 < config["tolerance"] < 1:
        raise ValueError("PC_TOLERANCE must lie in (0,1)")
    if not 1 <= config["max_iterations"] <= 1_000_000:
        raise ValueError("PC_MAX_ITERATIONS must lie in [1,1000000]")
    if not 1 <= config["max_interactions"] <= 2**63 - 1:
        raise ValueError("PC_MAX_INTERACTIONS must be a positive int64")
    if not math.isfinite(config["script_timeout"]) or config["script_timeout"] <= 0:
        raise ValueError("PC_SCRIPT_TIMEOUT must be finite and positive")
    start, end = float(original["z_start"]), float(original["z_end"])
    intervals = int(original["source_intervals"])
    if intervals < 1 or not start > end >= 0:
        raise ValueError("Archived native redshift grid is invalid")
    step = (start - end) / intervals
    for z in config["redshifts"]:
        index = round((start - z) / step)
        if not 0 <= index <= intervals or abs(start - index * step - z) > 1e-9:
            raise ValueError(f"Requested redshift {z:g} is not an archived native snapshot")
    return config


def input_fingerprint(input_run):
    """Pin small archive identities; the runner verifies selected snapshot data."""
    root = Path(input_run)
    names = ["experiment_config.json", "commit.txt", "runtime.env", "python_path.txt",
             "results/manifest.json", "native_runtime/manifest.json",
             "native_runtime/manifest.sha256"]
    receipts = sorted((root / "results/histories").glob("seed*/history_receipt.json"))
    if not receipts:
        raise ValueError("Input run has no completed native history receipts")
    for receipt in receipts:
        names += [str(receipt.relative_to(root)),
                  str(receipt.with_name("history_manifest.json").relative_to(root))]
    return {name: digest(root / name) for name in names}


def runtime_fingerprint(input_run):
    # This checks the archived native binary plus the C++ and Fortran compilers.
    # Import lazily so configuration/fingerprint tests need no native runtime.
    from research.pc_causal_rt.budget_prepare import runtime_fingerprint as inspect
    runtime = inspect(input_run)
    archived = json.loads((Path(input_run) / "results/manifest.json").read_text())
    if runtime["native_binary_sha256"] != archived["native_runtime"]["native_extension_sha256"]:
        raise ValueError("Native binary differs from the archived input identity")
    return runtime


def allocation(original, config, memory):
    match = re.fullmatch(r"([1-9][0-9]*)(MB|GB|TB)", memory)
    if not match:
        raise ValueError("PC_MEM must be a positive integer MB/GB/TB value")
    allocated = int(match[1]) * {"MB": 10**6, "GB": 10**9, "TB": 10**12}[match[2]]
    n = int(original["dimension"])
    high = int(original.get("highres_dimension") or 3 * n)
    if n < 2 or high < n:
        raise ValueError("Invalid archived native dimensions")
    # Conservative planning allowance: native one-snapshot objects, input/output
    # and FFT grids, distance ordering, plus thread-local claims. It is an
    # estimate, not an assertion that peak resident memory cannot exceed it.
    estimated = (96 + 2 * config["threads"]) * n**3 * 8 + 8 * high**3 * 8 + 2 * 2**30
    if estimated > allocated:
        raise ValueError(
            f"One-snapshot memory estimate {estimated / 2**30:.2f} GiB exceeds PC_MEM={memory}")
    return {"estimated_peak_gib": estimated / 2**30, "pbs_memory": memory,
            "threads": config["threads"], "streaming_snapshots": True,
            "estimate_is_upper_bound": False}


def prepare(run, environment):
    run = Path(run).resolve()
    if (run / "PREPARED").exists() or (run / "run_manifest.json").exists():
        raise FileExistsError("Prepare a fresh run directory")
    input_run = Path((run / "input_run.txt").read_text().strip()).resolve()
    if run == input_run or run in input_run.parents or input_run in run.parents:
        raise ValueError("Comparison output must be separate from archived input")
    original = json.loads((input_run / "experiment_config.json").read_text())
    config = configuration(original, environment)
    estimate = allocation(original, config, environment.get("PC_MEM", "32GB"))
    manifest = {
        "schema": 1, "experiment": EXPERIMENT, "run": str(run),
        "archive_input_run": str(input_run), "commit": (run / "commit.txt").read_text().strip(),
        "configuration": config, "source_sha256": source_fingerprint(run / "source"),
        "input_sha256": input_fingerprint(input_run), "runtime": runtime_fingerprint(input_run),
        "resource_estimate": estimate,
        "ncpus": config["threads"], "native_build": False, "scope": SCOPE,
    }
    write_json(run / "experiment_config.json", config)
    write_json(run / "run_manifest.json", manifest)
    (run / "prepared.sha256").write_text(
        "".join(f"{digest(run / name)}  {name}\n" for name in PREPARED_FILES))
    (run / "PREPARED").write_text("Frozen coeval allocation comparison; no new native run.\n")
    return manifest


def verify(run, cpus=None):
    run = Path(run).resolve()
    if not (run / "PREPARED").is_file():
        raise ValueError("Run is not prepared")
    rows = (run / "prepared.sha256").read_text().splitlines()
    checked = []
    for line in rows:
        expected, name = line.split("  ", 1)
        if name not in PREPARED_FILES or name in checked or digest(run / name) != expected:
            raise ValueError("Prepared file changed or unexpected checksum entry: " + name)
        checked.append(name)
    if set(checked) != set(PREPARED_FILES):
        raise ValueError("Prepared checksums are incomplete")
    manifest = json.loads((run / "run_manifest.json").read_text())
    if manifest.get("schema") != 1 or manifest.get("experiment") != EXPERIMENT:
        raise ValueError("Unknown prepared run format")
    if manifest["run"] != str(run):
        raise ValueError("Prepared run moved")
    if cpus is not None and cpus != manifest["ncpus"]:
        raise ValueError("CPU allocation differs from frozen comparison setting")
    differences = fingerprint_difference(manifest["source_sha256"], source_fingerprint(run / "source"))
    if any(differences.values()):
        raise ValueError("Frozen source changed: " + json.dumps(differences))
    input_run = Path(manifest["archive_input_run"])
    if manifest["input_sha256"] != input_fingerprint(input_run):
        raise ValueError("Archived input identity changed")
    if manifest["runtime"] != runtime_fingerprint(input_run):
        raise ValueError("Compiler, native binary or Python dependencies changed")
    if json.loads((run / "experiment_config.json").read_text()) != manifest["configuration"]:
        raise ValueError("Prepared comparison configuration changed")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify"))
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--cpus", type=int)
    args = parser.parse_args()
    manifest = prepare(args.run, os.environ) if args.command == "prepare" else verify(args.run, args.cpus)
    print(json.dumps({"command": args.command, "commit": manifest["commit"],
                      "configuration": manifest["configuration"], "scope": SCOPE}))


if __name__ == "__main__":
    main()
