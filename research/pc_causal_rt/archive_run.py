"""Evolve corrected native cumulative source budgets using readonly old archives."""
from __future__ import annotations

import argparse
import fcntl
import json
import logging
from pathlib import Path

from research.pc_causal_rt.workflow import (
    SCOPE, _digest, _receipt, _runtime_identity, _source_hashes, _write_json,
    configuration, run_case,
)


NUMERICAL_OVERRIDES = {"settings", "max_steps", "checkpoint_stride", "tolerance"}
QUICK_SETTINGS = [dict(directions=96, refinement=1, cfl=.4,
                       time_substeps=4, source_stride=1)]


def archive_configuration(original, supplied=None):
    """Keep every archived physical/native input; change only numerical controls."""
    base = configuration(original | {"source_mode": "native_cumulative",
                                     "settings": QUICK_SETTINGS})
    if supplied is None:
        return base
    if not isinstance(supplied, dict):
        raise ValueError("Archive configuration must be a dictionary")
    candidate = configuration(base | supplied)
    for key, value in candidate.items():
        if key not in NUMERICAL_OVERRIDES and value != base[key]:
            raise ValueError(f"Archive physical configuration cannot be changed: {key}")
    return candidate


def input_results(input_run):
    path = Path(input_run).resolve()
    return path / "results" if (path / "results").is_dir() else path


def run(input_run, output, config=None):
    from research.pc_causal_rt.native_history import load_history
    from research.pc_causal_rt.source_budget import build_source_budget
    from research.pc_causal_rt.report import build_report
    from research.pc_causal_rt.script_comparison import build_comparison
    from research.pc_script_reference.run import run as script_run
    from research.pc_script_reference.run import verify_files

    input_run, output = Path(input_run).resolve(), Path(output).resolve()
    if output.is_relative_to(input_run) or input_run.is_relative_to(output):
        raise ValueError("Corrected RT output must be separate from the readonly input run")
    original_root = input_results(input_run)
    original_manifest_path = original_root / "manifest.json"
    original_manifest = json.loads(original_manifest_path.read_text())
    config = archive_configuration(original_manifest["configuration"], config)
    runtime = _runtime_identity()
    if runtime["native_extension_sha256"] != original_manifest["native_runtime"]["native_extension_sha256"]:
        raise ValueError("Use the original input run's native binary and Python environment")
    output.mkdir(parents=True, exist_ok=True)
    identity = {"schema": 2, "configuration": config, "source_hashes": _source_hashes(),
                "native_runtime": runtime, "input_root": str(original_root),
                "input_manifest_sha256": _digest(original_manifest_path),
                "source_mode": "native_cumulative"}
    with (output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != identity:
                raise ValueError("Corrected RT inputs or implementation changed; use a fresh output directory")
            # Always validate external inputs even when all output cases already exist.
            if (output / "COMPLETED").exists():
                _receipt(output, "completion.json", identity)
        else:
            if any(p.name != ".run.lock" for p in output.iterdir()):
                raise ValueError("New corrected RT output directory must be empty")
            _write_json(manifest_path, identity)
        (output / "COMPLETED").unlink(missing_ok=True)
        cases, histories = [], []
        for seed in config["seeds"]:
            original_history_root = original_root / "histories" / f"seed{seed}"
            receipt_path = original_history_root / "history_receipt.json"
            receipt = json.loads(receipt_path.read_text())
            verify_files(original_history_root, receipt, ("history.npz", "history_manifest.json"))
            if json.loads((original_history_root / "history_manifest.json").read_text()) != receipt["identity"]:
                raise ValueError("Original native history manifest differs from its receipt")
            history_path = original_history_root / "history.npz"
            history = load_history(history_path)
            if history["metadata"]["seed"] != seed:
                raise ValueError("Original native history has the wrong seed")
            if (history["metadata"]["dimension"] != config["dimension"]
                    or history["metadata"]["box_len_cMpc"] != config["box_len"]
                    or len(history["redshifts"]) != config["source_intervals"] + 1):
                raise ValueError("Original native history does not match experiment configuration")
            # All original fields remain untouched. The export is a new, audited sidecar.
            history["_archive_root"] = str(original_history_root)
            history["metadata"] = history["metadata"] | {"native_archive_root": str(original_history_root)}
            budget_path = output / "histories" / f"seed{seed}" / "source_budget.npz"
            budget_path.parent.mkdir(parents=True, exist_ok=True)
            history["source_budget"] = build_source_budget(history, budget_path)
            histories.append({"seed": seed, "path": str(history_path),
                "sha256": _digest(history_path), "metadata": history["metadata"],
                "source_budget_path": str(budget_path.relative_to(output)),
                "source_budget_sha256": _digest(budget_path)})
            for setting in config["settings"]:
                cases.append(run_case(output, history, seed, setting, config, identity["source_hashes"]))
        if _source_hashes() != identity["source_hashes"]:
            raise RuntimeError("Executable sources changed during calculation")
        summary = {"schema": 2, "scope": SCOPE, "configuration": config,
                   "physical_model_authority": "21cmFAST; SCRIPT is an external comparison only",
                   "input_root": str(original_root), "histories": histories, "cases": cases,
                   "all_ledgers_passed": all(c["all_ledgers_passed"] for c in cases),
                   "convergence_certified": False}
        _write_json(output / "summary.json", summary)
        report = build_report(output)
        summary["convergence_certified"] = report["convergence_certified"]
        _write_json(output / "summary.json", summary)
        script_root = output / "script_reference"
        script_run(input_run, script_root)
        build_comparison(output, script_root)
        if _source_hashes() != identity["source_hashes"]:
            raise RuntimeError("Executable sources changed during comparison")
        artifacts = [p for p in output.rglob("*") if p.is_file()
                     and p.name not in (".run.lock", "reference.lock", "completion.json", "COMPLETED")
                     and not p.name.endswith(".tmp") and "_build" not in p.parts]
        _write_json(output / "completion.json", {"identity": identity,
            "files": {str(p.relative_to(output)): _digest(p) for p in artifacts}})
        (output / "COMPLETED").write_text("Native cumulative budget RT and independent SCRIPT comparison completed.\n")
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(args.input_run, args.output,
        json.loads(args.config.read_text()) if args.config else None)


if __name__ == "__main__":
    main()
