"""Measure neural fitting capacity on one or two existing training histories.

This deliberately fits and selects on the same cases. It diagnoses optimization,
capacity and loss tradeoffs, not generalization, physical convergence or photon
conservation. No held-out field files or new RT simulations are used.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from research.es_pc_emulator.full_history import write_json
from research.es_pc_emulator.history_objective import (
    denominator_floor,
    relative_errors,
    stage_weights,
    weighted_rows,
)
from research.es_pc_emulator.physical_pilot_training import load_case, load_manifest
from research.es_pc_emulator.small_pilot_model import FieldNet, shell_power
from research.es_pc_emulator.small_pilot_training import (
    distribution,
    evaluate,
    field_arrays,
    field_loss,
    normalization,
    primary_band,
    summarize_records,
    training_component_scale,
)

ARMS = {
    "joint_constant16": {"width": 16, "schedule": "constant", "loss": "joint"},
    "joint_cosine16": {"width": 16, "schedule": "cosine", "loss": "joint"},
    "joint_cosine32": {"width": 32, "schedule": "cosine", "loss": "joint"},
    "field_cosine32": {"width": 32, "schedule": "cosine", "loss": "field"},
    "power_cosine32": {"width": 32, "schedule": "cosine", "loss": "power"},
}
SCOPE = "training-set fit only; not generalization"
SELECTION = "raw mean abs(Ppred/Ptrue-1), 0.1 <= teacher mean xHI < 0.9, primary k bins"


def default_options() -> dict:
    """Return fixed controls shared by all fitting experiments.

    Returns
    -------
    dict
        Optimizer, loss and snapshot controls; no production defaults change.
    """
    return {"epochs": 400, "batch_size": 8, "history_steps": 3, "learning_rate": 3e-4,
            "min_learning_rate": 3e-6, "diagnostic_interval": 10, "power_floor": 1e-8,
            "relative_power_floor": 1e-4, "prior_epsilon": 1e-4, "power_weight": .01,
            "component_weight": .01, "auxiliary_weight": .01, "target": .01}


def select_training_records(records: list[dict], count: int) -> list[dict]:
    """Choose a deterministic nested subset without opening held-out fields.

    Parameters
    ----------
    records : list of dict
        Original manifest records with unchanged split labels.
    count : int
        Number of lexicographically first training cases to use.

    Returns
    -------
    list of dict
        Original training records, sorted by case ID.
    """
    training = sorted((row for row in records if row["split"] == "train"), key=lambda row: row["case_id"])
    if count < 1 or count > len(training):
        raise ValueError("Requested subset exceeds available training cases")
    if len({row["case_id"] for row in training}) != len(training):
        raise ValueError("Duplicate training cases")
    return training[:count]


def learning_rate(epoch: int, epochs: int, initial: float, minimum: float, schedule: str) -> float:
    """Compute the matched constant or cosine learning rate for a training epoch.

    Parameters
    ----------
    epoch, epochs : int
        Zero-based current epoch and fixed total budget.
    initial, minimum : float
        Positive initial and final rates.
    schedule : {"constant", "cosine"}
        Optimizer schedule; cosine reaches the minimum on the last epoch.

    Returns
    -------
    float
        Learning rate for the current optimizer steps.
    """
    if not 0 <= epoch < epochs or not 0 < minimum <= initial or schedule not in ("constant", "cosine"):
        raise ValueError("Invalid learning-rate schedule")
    if schedule == "constant" or epochs == 1:
        return initial
    return minimum + .5*(initial-minimum)*(1+math.cos(math.pi*epoch/(epochs-1)))


@torch.no_grad()
def active_metrics(model: torch.nn.Module, arrays: dict[str, torch.Tensor], box: float,
                   batch_size: int) -> dict:
    """Measure raw active-EoR training errors without a denominator floor.

    Parameters
    ----------
    model : torch.nn.Module
        Field model to evaluate on the same histories used for fitting.
    arrays : dict of torch.Tensor
        Full-cube arrays and teacher means; teacher means are evaluation labels.
    box : float
        Box length in comoving Mpc.
    batch_size : int
        Maximum snapshots per forward pass.

    Returns
    -------
    dict
        Raw fractional power and field-error distributions. Undefined active
        power bins abort rather than silently making the 1% test easier.
    """
    selected = torch.nonzero((arrays["teacher_mean"] >= .1) & (arrays["teacher_mean"] < .9)).flatten()
    if not len(selected) or batch_size < 1:
        raise ValueError("Active-EoR snapshots and a positive batch size are required")
    errors, fields = [], []
    for indices in selected.split(batch_size):
        batch = {key: value[indices] for key, value in arrays.items()}
        prediction, _ = model(batch["inputs"], batch["base"])
        factor = batch["brightness"].double()
        k, pp, _ = shell_power(factor*prediction.double(), box)
        _, pt, valid = shell_power(factor*batch["truth"].double(), box)
        band = primary_band(k, prediction.shape[-1], box)
        if not valid[:, band].all():
            raise ValueError("Active-EoR reference contains undefined power bins")
        relative = (pp[:, band]/pt[:, band]-1).abs()
        if not torch.isfinite(relative).all():
            raise FloatingPointError("Nonfinite active-EoR power error")
        errors.extend(relative.flatten().tolist())
        fields.extend((prediction.double()-batch["truth"].double()).abs().mean((1, 2, 3)).tolist())
    return {"snapshots": len(selected), "power_fractional_error": distribution(errors),
            "field_MAE": distribution(fields)}


def accuracy_status(metric: dict, target: float) -> dict:
    """Distinguish a mean-error goal from an all-bin accuracy guarantee.

    Parameters
    ----------
    metric : dict
        Raw fractional-error distribution including undefined count.
    target : float
        Positive fractional target, normally 0.01.

    Returns
    -------
    dict
        Separate mean and all-bin status; either requires every bin defined.
    """
    valid = bool(metric["count"] and not metric["undefined"] and metric["mean"] is not None)
    return {"target_fraction": target, "mean_target_passed": valid and metric["mean"] <= target,
            "all_bins_target_passed": valid and metric["max"] <= target}


def _emit(value: dict) -> None:
    sys.stdout.write(json.dumps(value, allow_nan=False)+"\n")
    sys.stdout.flush()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _completed_result(folder: Path, arm: str, seed: int, cases: list[dict]) -> dict:
    marker = json.loads((folder/"COMPLETED").read_text())
    if marker["result_sha256"] != _sha(folder/"result.json"):
        raise ValueError("Completed fitting result checksum changed")
    result = json.loads((folder/"result.json").read_text())
    if (result["arm"] != arm or result["seed"] != seed or result["case_count"] != len(cases)
            or result["case_ids"] != [case["metadata"]["case_id"] for case in cases]
            or result["scope"] != SCOPE or not np.isfinite(result["best_selection_score"])):
        raise ValueError("Completed fitting result identity changed")
    for name, expected in result["checkpoint_sha256"].items():
        if not (folder/name).is_file() or _sha(folder/name) != expected:
            raise ValueError("Completed fitting checkpoint checksum changed")
    return result


def fit_case_set(cases: list[dict], arm: str, seed: int, output: Path, options: dict) -> dict:
    """Fit one fixed experiment and independently evaluate its selected checkpoint.

    Parameters
    ----------
    cases : list of dict
        One or two training histories; no validation/test cases are permitted.
    arm : str
        One of the five declared optimizer/capacity/loss experiments.
    seed : int
        Initialization and minibatch-order seed.
    output : pathlib.Path
        Fresh experiment directory; the caller manages completed-fit reuse.
    options : dict
        Shared controls from :func:`default_options`.

    Returns
    -------
    dict
        History, independent metrics, target status and provenance.
    """
    if not cases or any(case["metadata"]["split"] != "train" for case in cases):
        raise ValueError("Fitting-ceiling experiments accept training cases only")
    if arm not in ARMS:
        raise ValueError("Unknown fitting arm")
    first = cases[0]
    if any(case["xhi_es"].shape != first["xhi_es"].shape
           or not np.array_equal(case["redshifts"], first["redshifts"])
           or case["metadata"]["box_len_cMpc"] != first["metadata"]["box_len_cMpc"] for case in cases):
        raise ValueError("Fitting histories must share their grids and timeline")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)
    train, names = field_arrays(cases, options["history_steps"], True, "all")
    mean, scale = normalization(train["inputs"])
    train["inputs"] = (train["inputs"]-mean)/scale
    train["sample_weight"], counts = stage_weights(train["teacher_mean"])
    box = float(first["metadata"]["box_len_cMpc"])
    _, powers, _ = shell_power(train["brightness"]*train["truth"], box)
    floor = denominator_floor(powers, train["sample_weight"], options["relative_power_floor"], options["power_floor"])
    component_scale = training_component_scale(train, box, options["power_floor"])
    spec = ARMS[arm]
    model = FieldNet(len(names), width=spec["width"], auxiliary=True,
                     output_transform="logit", prior_epsilon=options["prior_epsilon"])
    optimizer = torch.optim.Adam(model.parameters(), lr=options["learning_rate"])
    generator = torch.Generator().manual_seed(seed)
    history, best_epoch = [], 0
    start = time.monotonic()
    initial = active_metrics(model, train, box, options["batch_size"])
    best_score = initial["power_fractional_error"]["mean"]
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(options["epochs"]):
        tick = time.monotonic()
        rate = learning_rate(epoch, options["epochs"], options["learning_rate"],
                             options["min_learning_rate"], spec["schedule"])
        for group in optimizer.param_groups:
            group["lr"] = rate
        model.train()
        total = 0.
        order = torch.randperm(len(train["truth"]), generator=generator)
        for indices in order.split(options["batch_size"]):
            batch = {key: value[indices] for key, value in train.items()}
            optimizer.zero_grad(set_to_none=True)
            if spec["loss"] == "power":
                predicted, _ = model(batch["inputs"], batch["base"])
                k, pp, _ = shell_power(batch["brightness"]*predicted, box)
                _, pt, _ = shell_power(batch["brightness"]*batch["truth"], box)
                band = primary_band(k, predicted.shape[-1], box)
                loss = options["power_weight"]*weighted_rows(
                    relative_errors(pp[:, band], pt[:, band], floor[band]), batch["sample_weight"])
            else:
                joint = spec["loss"] == "joint"
                loss = field_loss(model, batch, options["power_weight"] if joint else 0.,
                                  options["auxiliary_weight"] if joint else 0., options["power_floor"], box,
                                  primary_spectra=True, component_indices=tuple(range(6)) if joint else (),
                                  component_scale=component_scale, component_weight=options["component_weight"] if joint else 0.,
                                  power_denominator=floor)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite fitting loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach())*len(indices)
        model.eval()
        active = active_metrics(model, train, box, options["batch_size"])
        score = active["power_fractional_error"]["mean"]
        if score < best_score:
            best_score, best_epoch, best_state = score, epoch+1, copy.deepcopy(model.state_dict())
        elapsed = time.monotonic()-start
        row = {"epoch": epoch+1, "learning_rate": rate, "training_loss": total/len(order),
               "selection_score": score, "active_reionization": active,
               "epoch_seconds": time.monotonic()-tick, "elapsed_seconds": elapsed,
               "estimated_remaining_seconds": elapsed/(epoch+1)*(options["epochs"]-epoch-1)}
        history.append(row)
        if epoch == 0 or (epoch+1) % options["diagnostic_interval"] == 0 or epoch+1 == options["epochs"]:
            _emit({"arm": arm, "seed": seed, "cases": len(cases), **row})
    checkpoint = {"kind": "field", "state_dict": model.state_dict(), "input_mean": mean, "input_scale": scale,
                  "feature_names": names, "history_steps": options["history_steps"], "width": spec["width"],
                  "wide": False, "auxiliary": True, "box_len_cMpc": box, "grid_dimension": first["xhi_es"].shape[-1],
                  "output_transform": "logit", "prior_epsilon": options["prior_epsilon"],
                  "power_denominator_floor": floor, "scope": SCOPE, "selection_metric": SELECTION,
                  "convergence_certified": False, "seed": seed, "arm": arm, "stage_counts": counts,
                  "epoch": options["epochs"]}
    torch.save(checkpoint, output/"last_model.pt")
    checkpoint["state_dict"] = best_state
    checkpoint["epoch"] = best_epoch
    torch.save(checkpoint, output/"model.pt")
    summary = evaluate(checkpoint, cases, output/"predictions", "all")
    records = json.loads((output/"predictions/metrics.json").read_text())
    active = summarize_records([row for row in records if .1 <= row["teacher_xHI_mean"] < .9])
    if active["power_fractional_error"]["undefined"]:
        raise ValueError("Independent active-EoR evaluation has undefined power")
    np.testing.assert_allclose(active["power_fractional_error"]["mean"], best_score, rtol=3e-5, atol=2e-7)
    result = {"scope": SCOPE, "arm": arm, "arm_settings": spec, "seed": seed,
              "case_count": len(cases), "case_ids": [case["metadata"]["case_id"] for case in cases],
              "parameter_count": sum(p.numel() for p in model.parameters()), "initial_active_reionization": initial,
              "best_epoch": best_epoch, "best_selection_score": best_score, "selection_metric": SELECTION,
              "active_reionization": active, "last_active_reionization": history[-1]["active_reionization"],
              "accuracy": accuracy_status(active["power_fractional_error"], options["target"]),
              "any_epoch_all_bins_target_passed": min(initial["power_fractional_error"]["max"],
                  *(row["active_reionization"]["power_fractional_error"]["max"] for row in history)) <= options["target"],
              "summary": summary, "history": history, "elapsed_seconds": time.monotonic()-start,
              "checkpoint_sha256": {name: _sha(output/name) for name in ("model.pt", "last_model.pt")},
              "stage_counts": counts, "convergence_certified": False}
    write_json(output/"result.json", result)
    write_json(output/"COMPLETED", {"result_sha256": _sha(output/"result.json"), "scope": SCOPE})
    return result


def _plots(results: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = sorted({row["case_count"] for row in results})
    arms = [name for name in ARMS if any(row["arm"] == name for row in results)]
    fig, axes = plt.subplots(len(counts), len(arms), figsize=(3.5*len(arms), 3.2*len(counts)),
                             squeeze=False, constrained_layout=True)
    for i, count in enumerate(counts):
        for j, arm in enumerate(arms):
            axis = axes[i, j]
            for row in results:
                if row["case_count"] != count or row["arm"] != arm:
                    continue
                axis.plot([h["epoch"] for h in row["history"]],
                          [100*h["selection_score"] for h in row["history"]], label=f"seed {row['seed']}")
                axis.scatter(row["best_epoch"], 100*row["best_selection_score"], marker="*")
            axis.axhline(1, color="black", linestyle=":")
            axis.set(title=f"{count} history: {arm}", xlabel="Epoch", ylabel="Raw active-EoR P21 error [%]", yscale="log")
            axis.legend(fontsize=7)
            axis.grid(alpha=.2)
    fig.suptitle("Deliberate training-set fitting; no held-out accuracy claim")
    fig.savefig(output/"learning_curves.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    labels = [f"{r['case_count']}h {r['arm']} s{r['seed']}" for r in results]
    for axis, metric, ylabel in ((axes[0], "power_fractional_error", "Raw active-EoR P21 error [%]"),
                                 (axes[1], "field_MAE", "Active-EoR neutral-fraction MAE [pp]")):
        axis.bar(np.arange(len(results)), [100*r["active_reionization"][metric]["mean"] for r in results])
        axis.set_xticks(np.arange(len(results)), labels, rotation=80, fontsize=7)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=.2)
    axes[0].axhline(1, color="black", linestyle=":")
    fig.savefig(output/"comparison.png", dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    """Run the controlled fitting matrix using only existing training histories.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; defaults to process arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--model-seeds", nargs="+", type=int, default=[11, 23])
    parser.add_argument("--case-counts", nargs="+", type=int, choices=[1, 2], default=[1, 2])
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    for key, value in default_options().items():
        parser.add_argument("--"+key.replace("_", "-"), type=type(value), default=value)
    args = parser.parse_args(argv)
    options = {key: getattr(args, key) for key in default_options()}
    if not all(np.isfinite(value) and value > 0 for value in options.values()) or args.threads < 1:
        raise ValueError("Positive finite fitting controls required")
    if options["min_learning_rate"] > options["learning_rate"]:
        raise ValueError("Minimum learning rate exceeds initial rate")
    for values in (args.model_seeds, args.case_counts, args.arms):
        if len(values) != len(set(values)):
            raise ValueError("Experiment axes must contain unique values")
    if min(args.model_seeds) < 0:
        raise ValueError("Nonnegative model seeds required")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest = load_manifest(args.manifest)
    records = select_training_records(manifest["cases"], max(args.case_counts))
    cases = [load_case(record) for record in records]
    sources = [Path(__file__), Path(__file__).with_name("small_pilot_training.py"),
               Path(__file__).with_name("small_pilot_model.py"), Path(__file__).with_name("history_objective.py"),
               Path(__file__).with_name("spectral_components.py"), Path(__file__).with_name("physical_pilot_training.py"),
               Path(__file__).parent.parent/"pc_field_emulator/evaluation.py"]
    config = {"scope": SCOPE, "options": options, "model_seeds": args.model_seeds,
              "case_counts": args.case_counts, "arms": args.arms, "arm_definitions": ARMS,
              "threads": args.threads, "manifest": str(args.manifest.resolve()),
              "manifest_sha256": manifest["source_manifest_sha256"],
              "cases": [{"case_id": c["metadata"]["case_id"], "split": c["metadata"]["split"],
                          "seed": c["metadata"]["seed"], "sha256": c["sha256"]} for c in cases],
              "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              "torch": str(torch.__version__), "selection_metric": SELECTION}
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output/"configuration.json"
    if config_path.is_file() and json.loads(config_path.read_text()) != config:
        raise ValueError("Existing fitting experiment has different controls, code or data")
    write_json(config_path, config)
    results = []
    for count in args.case_counts:
        for arm in args.arms:
            for seed in args.model_seeds:
                folder = args.output/f"cases{count}_{arm}_seed{seed}"
                if (folder/"COMPLETED").is_file():
                    result = _completed_result(folder, arm, seed, cases[:count])
                else:
                    if folder.exists():
                        archive = args.output/"incomplete_attempts"
                        archive.mkdir(exist_ok=True)
                        folder.rename(archive/f"{folder.name}_{time.time_ns()}")
                    _emit({"starting": folder.name, "completed_fits": len(results)})
                    result = fit_case_set(cases[:count], arm, seed, folder, options)
                results.append(result)
                report = {"scope": SCOPE, "configuration": config, "completed_fits": len(results),
                          "expected_fits": len(args.case_counts)*len(args.arms)*len(args.model_seeds),
                          "experiments": results, "interpretation": {
                              "success": "Demonstrates fitting capability on selected training histories only.",
                              "failure": "Does not distinguish missing information, capacity or optimization without the controlled arm comparisons.",
                              "power_only": "A small auto-power error does not imply accurate spatial phases or photon conservation."}}
                write_json(args.output/"fit_ceiling_report.json", report)
    _plots(results, args.output)
    (args.output/"COMPLETED").write_text("Fitting matrix completed; inspect raw active-stage errors and field metrics.\n")


if __name__ == "__main__":
    main()
