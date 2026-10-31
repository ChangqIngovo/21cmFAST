"""Freeze a short native-cumulative RT run against an immutable native archive."""

from __future__ import annotations

import argparse
from itertools import product
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from research.pc_script_reference.prepare_run import (
    digest, runtime_fingerprint as script_runtime_fingerprint, source_fingerprint,
)


EXPERIMENT = "native-cumulative-rt-archive"


def configuration(original, environment):
    """Keep physical inputs fixed; explicitly select a short numerical setting."""
    from .archive_run import archive_configuration

    config = archive_configuration(original)
    if environment.get("PC_CONFIG_JSON"):
        value = json.loads(Path(environment["PC_CONFIG_JSON"]).read_text())
        if not isinstance(value, dict):
            raise ValueError("PC_CONFIG_JSON must contain a configuration dictionary")
        config.update(value)
    for key, cast in (("max_steps", int), ("checkpoint_stride", int), ("tolerance", float)):
        variable = "PC_" + key.upper()
        if variable in environment:
            config[key] = cast(environment[variable])
    axes = (("PC_DIRECTIONS", "directions", int),
            ("PC_REFINEMENTS", "refinement", int),
            ("PC_CFL_VALUES", "cfl", float),
            ("PC_TIME_SUBSTEPS", "time_substeps", int),
            ("PC_SOURCE_STRIDES", "source_stride", int))
    if any(variable in environment for variable, _, _ in axes):
        baseline = config["settings"][0]
        levels = [[cast(item) for item in environment.get(variable, str(baseline[key])).replace(",", " ").split()]
                  for variable, key, cast in axes]
        config["settings"] = [dict(zip((key for _, key, _ in axes), row)) for row in product(*levels)]
    return validate_configuration(original, config)


def validate_configuration(original, config):
    """Only numerical controls may differ from the archived physical experiment."""
    from .archive_run import archive_configuration

    return archive_configuration(original, config)


def input_fingerprint(input_run):
    """Freeze source-history data, receipts and native identity, excluding old RT cases."""
    input_run = Path(input_run)
    required = ("experiment_config.json", "commit.txt", "runtime.env", "python_path.txt",
                "results/manifest.json", "native_runtime/manifest.json", "native_runtime/manifest.sha256")
    files = {name: digest(input_run / name) for name in required}
    history_root = input_run / "results/histories"
    if not list(history_root.glob("seed*/history_receipt.json")):
        raise ValueError("Input run has no completed native source histories")
    for path in sorted(history_root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".pyo"):
            files[str(path.relative_to(input_run))] = digest(path)
    return files


def runtime_fingerprint(input_run):
    result = script_runtime_fingerprint(input_run)
    compiler = shutil.which("g++")
    if compiler is None:
        raise RuntimeError("g++ is unavailable; load gcc/12.2.0 before preparing")
    compiler = str(Path(compiler).resolve())
    result.update(
        gxx=compiler, gxx_sha256=digest(compiler),
        gxx_version=subprocess.run([compiler, "--version"], check=True, capture_output=True,
                                   text=True, timeout=15).stdout,
    )
    return result


def allocation(config, environment):
    from .prepare_run import resource_estimate

    memory = environment.get("PC_MEM", "8GB")
    match = re.fullmatch(r"([1-9][0-9]*)(MB|GB|TB)", memory)
    if match is None:
        raise ValueError("PC_MEM must be a positive integer MB/GB/TB value")
    allocated = int(match[1]) * {"MB": 10**6, "GB": 10**9, "TB": 10**12}[match[2]]
    estimate = resource_estimate(config)
    if estimate["estimated_array_gib"] * 2**30 + 2**30 > allocated:
        raise ValueError("Estimated RT arrays plus 1 GiB runtime allowance exceed PC_MEM")
    return estimate | {"pbs_memory": memory, "ncpus": config["threads"]}


def prepare(run, environment):
    run = Path(run).resolve()
    if (run / "PREPARED").exists() or (run / "run_manifest.json").exists():
        raise FileExistsError("Prepare a fresh run directory")
    input_run = Path((run / "input_run.txt").read_text().strip()).resolve()
    if input_run == run or input_run in run.parents or run in input_run.parents:
        raise ValueError("Write results outside the archived input run")
    original = json.loads((input_run / "experiment_config.json").read_text())
    config = configuration(original, environment)
    manifest = {
        "schema": 1, "experiment": EXPERIMENT, "run": str(run),
        "archive_input_run": str(input_run),
        "commit": (run / "commit.txt").read_text().strip(),
        "configuration": config,
        "source_sha256": source_fingerprint(run / "source"),
        "input_sha256": input_fingerprint(input_run),
        "runtime": runtime_fingerprint(input_run),
        "resource_estimate": allocation(config, environment),
        "ncpus": config["threads"], "native_build": False,
        "scope": "Reuse archived native fields and binary; evolve native cumulative source increments with short RT and compare with original SCRIPT.",
    }
    (run / "experiment_config.json").write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")
    (run / "run_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    files = ("commit.txt", "input_run.txt", "python_path.txt", "runtime.env", "submit.pbs",
             "experiment_config.json", "run_manifest.json")
    (run / "prepared.sha256").write_text("".join(f"{digest(run / name)}  {name}\n" for name in files))
    (run / "PREPARED").write_text("Prepared; no new native simulation or RT evolution has run.\n")
    return manifest


def verify(run, cpus=None):
    run = Path(run).resolve()
    if not (run / "PREPARED").is_file():
        raise ValueError("Run is not prepared")
    for line in (run / "prepared.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if Path(name).name != name or digest(run / name) != expected:
            raise ValueError("Prepared file changed: " + name)
    manifest = json.loads((run / "run_manifest.json").read_text())
    if manifest.get("schema") != 1 or manifest.get("experiment") != EXPERIMENT:
        raise ValueError("Unknown prepared run format")
    if manifest["run"] != str(run):
        raise ValueError("Prepared run moved")
    if cpus is not None and cpus != manifest["ncpus"]:
        raise ValueError("CPU allocation differs from archived native thread count")
    if manifest["source_sha256"] != source_fingerprint(run / "source"):
        raise ValueError("Frozen source changed")
    input_run = Path(manifest["archive_input_run"])
    if manifest["input_sha256"] != input_fingerprint(input_run):
        raise ValueError("Original native archive changed")
    if manifest["runtime"] != runtime_fingerprint(input_run):
        raise ValueError("Compiler, native binary or Python dependency runtime changed")
    original = json.loads((input_run / "experiment_config.json").read_text())
    config = json.loads((run / "experiment_config.json").read_text())
    if config != manifest["configuration"]:
        raise ValueError("Prepared configuration differs from manifest")
    validate_configuration(original, config)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify"))
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--cpus", type=int)
    args = parser.parse_args()
    manifest = prepare(args.run, os.environ) if args.command == "prepare" else verify(args.run, args.cpus)
    print(json.dumps({"command": args.command, "commit": manifest["commit"],
                      "input_run": manifest["archive_input_run"], "settings": manifest["configuration"]["settings"]}))


if __name__ == "__main__":
    main()
