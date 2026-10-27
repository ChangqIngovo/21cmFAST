"""Small validation-selected loss-weight sweep on existing pilot simulations."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from research.es_pc_emulator.component_report import case_scores
from research.es_pc_emulator.small_pilot_training import (
    distribution,
    evaluate,
    load_manifest,
    select_cases,
    write_json,
)

COMMON_KEYS = ("history_steps", "width", "batch_size", "learning_rate", "auxiliary_weight",
               "power_floor", "epochs", "patience", "diagnostic_interval", "train_cases",
               "validation_cases", "test_cases", "external_cases")


def atomic_json(path, value):
    """Publish complete progress records even if the batch job is interrupted."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def sweep_grid(configuration):
    """Vary one spectral weight at a time, retaining a common thermo baseline."""
    components = configuration["component_weights"]
    powers = configuration["power_weights"]
    anchor = configuration["anchor_component_weight"]
    for values in (components, powers, [anchor]):
        if not values or len(set(values)) != len(values) or any(
                not math.isfinite(x) or x <= 0 for x in values):
            raise ValueError("Spectral weights must be unique finite positive numbers")
    if anchor not in components:
        raise ValueError("anchor_component_weight must appear in component_weights")
    grid = [{"id": "baseline", "arm": "thermo", "component_weight": 0., "power_weight": 0.}]
    grid += [{"id": f"components_c{weight:g}", "arm": "thermo_components",
              "component_weight": weight, "power_weight": 0.} for weight in components]
    grid += [{"id": f"joint_c{anchor:g}_p{weight:g}", "arm": "thermo_components_p21",
              "component_weight": anchor, "power_weight": weight} for weight in powers]
    if len({row["id"] for row in grid}) != len(grid):
        raise ValueError("Weight labels collide; use fewer significant digits")
    return grid


def select_trial(trials):
    """Use mean validation log-P21 MSE across all seeds; never test summaries."""
    scores = [row["mean_validation_log_power_mse"] for row in trials]
    if not scores or not np.isfinite(scores).all():
        raise ValueError("A complete finite validation score is required for every candidate")
    return min(trials, key=lambda row: (row["mean_validation_log_power_mse"], row["id"]))


def summarize_trial(spec, folders, seeds):
    """Keep seed scatter, selected epoch, and independent error summaries."""
    records = []
    for seed, folder in zip(seeds, folders, strict=True):
        name = f"{spec['arm']}_seed{seed}"
        training = json.loads((folder/name/"training.json").read_text())
        summary = json.loads((folder/"summary.json").read_text())[name]
        history = training["history"]
        records.append({"seed": seed, "folder": str(folder), "model_name": name,
                        "best_epoch": training["best_epoch"], "epochs_run": len(history),
                        "best_validation_log_power_mse": training["best_validation_loss"],
                        "last_validation_log_power_mse": history[-1]["validation_loss"],
                        "selected_diagnostics": training["selected_diagnostics"],
                        "diagnostic_history": [row for row in history if "diagnostics" in row],
                        "train": summary["train"], "validation": summary["validation"]})
    return spec | {"seeds": records,
                   "mean_validation_log_power_mse": float(np.mean([
                       r["best_validation_log_power_mse"] for r in records])),
                   "train_error_mean": float(np.mean([
                       r["train"]["power_fractional_error"]["mean"] for r in records])),
                   "validation_error_mean": float(np.mean([
                       r["validation"]["power_fractional_error"]["mean"] for r in records]))}


def training_command(spec, seed, config, manifest, output, threads):
    """Restrict every candidate's evaluation to training and validation cases."""
    command = [sys.executable, "-m", "research.es_pc_emulator.small_pilot_training",
               "--manifest", str(manifest), "--output", str(output), "--threads", str(threads),
               "--arms", spec["arm"], "--model-seeds", str(seed),
               "--evaluation-splits", "train", "validation",
               "--component-weight", str(spec["component_weight"] or config["anchor_component_weight"]),
               "--power-weight", str(spec["power_weight"] or .01)]
    for key in COMMON_KEYS:
        command += ["--"+key.replace("_", "-"), str(config[key])]
    return command


def run_trial(spec, seed, config, manifest, output, threads):
    """Resume completed fits; retain incomplete attempts and restart only those."""
    root = output/"trials"/spec["id"]/f"seed{seed}"
    root.mkdir(parents=True, exist_ok=True)
    attempts = sorted(root.glob("attempt_*"))
    completed = [folder for folder in attempts if (folder/"COMPLETED").is_file()]
    if completed:
        return completed[-1]
    folder = root/f"attempt_{len(attempts)+1:03d}"
    command = training_command(spec, seed, config, manifest, folder, threads)
    atomic_json(output/"progress.json", {"trial": spec, "seed": seed, "folder": str(folder),
                                        "status": "training", "command": command})
    print(json.dumps({"trial": spec["id"], "seed": seed, "output": str(folder)}), flush=True)  # noqa: T201
    subprocess.run(command, check=True)
    if not (folder/"COMPLETED").is_file():
        raise RuntimeError("Training returned without a completed result")
    return folder


def assess_selected(trials, selected, cases, output):
    """Assess only the validation-selected setting and its paired baseline."""
    chosen = [next(row for row in trials if row["id"] == "baseline")]
    if selected["id"] != "baseline":
        chosen.append(selected)
    heldout = [case for case in cases if case["metadata"]["split"] in ("test", "external_test")]
    result = {"selected_id": selected["id"], "models": {}}
    folder = output/"heldout"
    folder.mkdir(exist_ok=True)
    # A partial held-out evaluation can be repeated without overwriting its files.
    attempts = list(folder.glob("attempt_*"))
    attempt = folder/f"attempt_{len(attempts)+1:03d}"
    attempt.mkdir()
    result["models"]["es"] = evaluate(None, heldout, attempt/"es")
    baseline_scores = {}
    comparisons = {}
    for trial in chosen:
        for record in trial["seeds"]:
            seed = record["seed"]
            name = f"{trial['id']}_seed{seed}"
            checkpoint = Path(record["folder"])/record["model_name"]/"model.pt"
            result["models"][name] = evaluate(checkpoint, heldout, attempt/name)
            scores, closure = case_scores(json.loads((attempt/name/"metrics.json").read_text()))
            if trial["id"] == "baseline":
                baseline_scores[seed] = scores
            else:
                baseline = baseline_scores[seed]
                if scores.keys() != baseline.keys():
                    raise ValueError("Paired comparisons require identical cases")
                comparisons[name] = {"maximum_six_term_closure_relative": closure, "splits": {}}
                for split in sorted({key[0] for key in scores}):
                    rows = [{"case_id": key[1], "baseline_error": baseline[key],
                             "model_error": scores[key], "difference": scores[key]-baseline[key]}
                            for key in sorted(scores) if key[0] == split]
                    comparisons[name]["splits"][split] = {
                        "cases": rows, "improved_cases": sum(row["difference"] < 0 for row in rows),
                        "total_cases": len(rows),
                        "paired_error_difference": distribution(row["difference"] for row in rows)}
    result["paired_comparisons"] = comparisons
    result["predictions_directory"] = str(attempt)
    return result


def plot_results(trials, selected, output):
    """Show common-metric learning curves and validation-only weight selection."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    x = np.arange(len(trials))
    for axis, split in zip(axes, ("train", "validation"), strict=True):
        values = np.array([[record[split]["power_fractional_error"]["mean"]*100
                            for record in row["seeds"]] for row in trials])
        mean = values.mean(axis=1)
        axis.bar(x, mean, color=["#d17a22" if row["id"] == selected["id"] else "#287d8e"
                               for row in trials], alpha=.85)
        axis.errorbar(x, mean, yerr=[mean-values.min(1), values.max(1)-mean],
                      fmt="none", color="black", capsize=3)
        axis.axhline(1, color="#af413b", linestyle="--", linewidth=1)
        axis.set_xticks(x, [row["id"] for row in trials], rotation=35, ha="right", fontsize=8)
        axis.set(title=split, ylabel="Mean absolute P21 error [%]")
        axis.grid(axis="y", alpha=.2)
    fig.suptitle("Loss weights on fixed data and network; orange selected by validation log-P21 MSE\n"
                 "Bars average network seeds; whiskers show their range")
    fig.savefig(output/"loss_sweep.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(math.ceil(len(trials)/2), 2, figsize=(13, 3*math.ceil(len(trials)/2)),
                             constrained_layout=True, squeeze=False)
    for axis, trial in zip(axes.flat, trials, strict=False):
        for i, record in enumerate(trial["seeds"]):
            history = json.loads((Path(record["folder"])/record["model_name"]/"training.json").read_text())["history"]
            rows = [row for row in history if "diagnostics" in row]
            for split, style in (("train", "-"), ("validation", "--")):
                values = [row["diagnostics"][split]["power_fractional_error_mean"] for row in rows]
                axis.plot([row["epoch"] for row in rows],
                          [100*value if value is not None else np.nan for value in values],
                          style, color=f"C{i}", marker=".", markersize=3,
                          label=f"seed {record['seed']} {split}")
            axis.axvline(record["best_epoch"], color=f"C{i}", alpha=.25, linewidth=1)
        axis.axhline(1, color="gray", linestyle=":", linewidth=1)
        axis.set(title=trial["id"], xlabel="Epoch", ylabel="Mean absolute P21 error [%]")
        axis.set_yscale("symlog", linthresh=1)
        axis.grid(alpha=.2)
        axis.legend(fontsize=7, frameon=False)
    for axis in list(axes.flat)[len(trials):]:
        axis.set_axis_off()
    fig.suptitle("Identical primary snapshots and k bins on train and validation\n"
                 "Vertical lines: validation-selected epochs; diagnostics every few epochs")
    fig.savefig(output/"learning_curves.png", dpi=160)
    plt.close(fig)


def main(argv=None):
    """Run a resumable sweep, freeze validation selection, then assess held-out data."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "output", "config"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    args.manifest, args.output = args.manifest.resolve(), args.output.resolve()
    config = json.loads(args.config.read_text())
    missing = set(COMMON_KEYS)-config.keys()
    if missing:
        raise ValueError(f"Missing configuration keys: {sorted(missing)}")
    if args.threads < 1 or config["diagnostic_interval"] < 1:
        raise ValueError("Threads and diagnostic_interval must be positive")
    seeds = config["model_seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Use distinct nonnegative integer model seeds")
    grid = sweep_grid(config)
    provenance = {"configuration": config, "manifest": str(args.manifest),
                  "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                  "threads": args.threads, "grid": grid,
                  "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in [Path(__file__), *[Path(__file__).with_name(name) for name in
                                        ("small_pilot_training.py", "small_pilot_model.py", "spectral_components.py")]]}}
    if args.output.exists():
        if not args.resume:
            raise FileExistsError("Use --resume for the identical sweep or a fresh output")
        saved = json.loads((args.output/"sweep_configuration.json").read_text())
        if saved != provenance:
            raise ValueError("Resume configuration, manifest, source or thread count changed")
        if (args.output/"COMPLETED").is_file():
            print(f"Already complete: {args.output}", flush=True)  # noqa: T201
            return
    else:
        args.output.mkdir(parents=True)
        atomic_json(args.output/"sweep_configuration.json", provenance)
    trials = []
    for spec in grid:
        folders = [run_trial(spec, seed, config, args.manifest, args.output, args.threads) for seed in seeds]
        trials.append(summarize_trial(spec, folders, seeds))
        atomic_json(args.output/"sweep_progress.json", {"completed_trials": trials, "total_trials": len(grid)})
    selected = select_trial(trials)
    selection = {"selected_id": selected["id"], "criterion": "mean best validation log-P21 MSE across network seeds",
                 "score": selected["mean_validation_log_power_mse"], "test_used_for_selection": False,
                 "model_seeds": seeds}
    atomic_json(args.output/"selection.json", selection)
    # Selection is persisted before any candidate's held-out predictions are computed.
    import torch
    torch.set_num_threads(args.threads)
    cases = select_cases(load_manifest(args.manifest), {
        "train": config["train_cases"], "validation": config["validation_cases"],
        "test": config["test_cases"], "external_test": config["external_cases"]})
    heldout = assess_selected(trials, selected, cases, args.output)
    atomic_json(args.output/"selected_summary.json", heldout)
    report = {"scope": "fixed-data loss-weight sweep; existing uncertified RT teacher; saturated Ts, no RSD",
              "selection": selection, "trials": trials, "selected_results": heldout,
              "weights": "one-factor scans: component weight at P21 weight zero; P21 weight at fixed component anchor",
              "seed_aggregation": "mean of separate model errors; not an ensemble prediction",
              "component_error_units": "absolute signed-contribution error divided by true total P21"}
    atomic_json(args.output/"sweep_report.json", report)
    plot_results(trials, selected, args.output)
    atomic_json(args.output/"progress.json", {"status": "completed", "selection": selection})
    (args.output/"COMPLETED").write_text("Loss sweep completed; inspect sweep_report.json\n")
    print(json.dumps({"selected": selected["id"], "report": str(args.output/"sweep_report.json")}), flush=True)  # noqa: T201


if __name__ == "__main__":
    main()
