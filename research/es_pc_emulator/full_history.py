"""Check complete reionization histories and report errors throughout evolution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.es_pc_emulator.physical_pilot_training import load_case, load_manifest


def write_json(path, value):
    """Save a strict complete report without replacing it with partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")
    temporary.replace(path)


def check_coverage(cases, z_start, z_end, snapshots, end_xhi=.01):
    """Require initial conditions, a resolved middle phase and a neutral-poor end.

    Both volume- and mass-weighted neutral fractions must satisfy the endpoint
    criterion. This checks temporal coverage, not RT numerical convergence.
    """
    if (not np.isfinite([z_start, z_end, end_xhi]).all() or not z_start > z_end >= 0
            or type(snapshots) is not int or snapshots < 3 or not 0 < end_xhi < .1):
        raise ValueError("Require decreasing redshifts, >=3 snapshots and 0 < end_xhi < 0.1")
    expected = np.linspace(z_start, z_end, snapshots)
    records = []
    edges = [0., .01, .1, .5, .9, .99, 1.]
    for case in cases:
        errors = []
        meta = case["metadata"]
        z = np.asarray(case["redshifts"])
        xhi = np.asarray(case["xhi_rt"])
        rho = np.broadcast_to(np.asarray(case["density"]), xhi.shape)
        if not np.isfinite(rho).all() or np.any(rho <= 0):
            raise ValueError("Coverage requires finite positive gas density")
        volume = xhi.mean(axis=(1, 2, 3))
        mass = (xhi*rho).sum(axis=(1, 2, 3))/rho.sum(axis=(1, 2, 3))
        if len(z) != snapshots or not np.allclose(z, expected, rtol=0, atol=1e-9):
            errors.append("saved redshifts do not match the complete requested timeline")
        if not meta.get("initial_snapshot_included", False):
            errors.append("initial snapshot was not explicitly saved")
        times = np.asarray(case.get("times_seconds", []))
        if (times.shape != z.shape or not np.isfinite(times).all()
                or times[0] != 0 or not np.all(np.diff(times) > 0)):
            errors.append("elapsed time does not start at zero and increase through the saved timeline")
        initial = meta.get("initial_state", {})
        for field, key in (("xhi_rt", "xHII"), ("temperature_rt", "temperature_kelvin"),
                           ("xheii_rt", "xHeII"), ("xheiii_rt", "xHeIII")):
            target = initial.get(key)
            if target is None:
                errors.append("missing declared initial state: "+key)
            else:
                target = 1-target if field == "xhi_rt" else target
                if not np.allclose(case[field][0], target, rtol=1e-12, atol=1e-15):
                    errors.append("saved initial state disagrees with RT input: "+field)
        if np.any(case["photons"][0] != 0) or np.any(case["emission_time_mean"][0] != 0):
            errors.append("initial integrated photon count/time moment is not zero")
        if np.any(case["xhi_es"][0] != 1) or np.any(case["radius_es_cMpc"][0] != 0):
            errors.append("initial ES prior is not neutral with zero crossing radius")
        if volume[0] < .99 or mass[0] < .99:
            errors.append("history does not start near neutral")
        if volume[-1] > end_xhi or mass[-1] > end_xhi:
            errors.append("final neutral fraction exceeds the requested volume/mass threshold")
        if not np.any((volume >= .1) & (volume <= .9)):
            errors.append("no saved snapshot samples the middle reionization phase")
        records.append({"case_id": meta["case_id"], "split": meta["split"], "seed": meta["seed"],
                        "passed": not errors, "errors": errors, "redshifts": z.tolist(),
                        "xHI_volume": volume.tolist(), "xHI_mass": mass.tolist(),
                        "snapshots_per_neutral_fraction_bin": np.histogram(volume, bins=edges)[0].tolist(),
                        "initial_xHI_volume": float(volume[0]), "final_xHI_volume": float(volume[-1]),
                        "final_xHI_mass": float(mass[-1])})
    if not records:
        raise ValueError("No histories supplied")
    return {"passed": all(row["passed"] for row in records), "case_count": len(records),
            "failed_case_ids": [row["case_id"] for row in records if not row["passed"]],
            "requested": {"z_start": z_start, "z_end": z_end, "snapshots": snapshots,
                          "maximum_final_xHI_volume_and_mass": end_xhi},
            "neutral_fraction_bin_edges": edges, "cases": records,
            "scope": "temporal and reionization-stage coverage; not numerical convergence",
            "convergence_certified": False}


def _compact_metric(metric):
    names = ("cases", "snapshots", "power_fractional_error", "absolute_power_error", "field_MAE", "r21",
             "regularized_power_error", "selection_score", "selection_metric")
    result = {key: metric[key] for key in names if key in metric}
    result["component_error_mean_over_P21"] = {
        key: value["mean"] for key, value in metric.get("components", {}).items()}
    return result


def learning_points(fit: dict, split: str, metric: str) -> list[tuple[int, float]]:
    """Return diagnostic points including the actual restored checkpoint.

    Parameters
    ----------
    fit : dict
        Training history and restored checkpoint diagnostics.
    split, metric : str
        Dataset split and diagnostic metric to plot.

    Returns
    -------
    list of tuple
        Sorted epoch/value pairs; restored values replace a sampled best epoch.
    """
    points = {row["epoch"]: row["diagnostics"][split][metric]
              for row in fit["history"]
              if metric in row.get("diagnostics", {}).get(split, {})}
    selected = fit.get("selected_diagnostics", {}).get(split, {})
    if metric in selected:
        points[fit["best_epoch"]] = selected[metric]
    return sorted(points.items())


def active_phase_summary(rows: list[dict]) -> dict:
    """Aggregate raw active-reionization errors by their actual sample counts.

    Parameters
    ----------
    rows : list of dict
        Stage-resolved summary rows using the standard neutral-fraction bins.

    Returns
    -------
    dict
        Means and finite counts for 0.1 <= teacher mean xHI < 0.9, without
        inventing aggregate quantiles from per-stage quantiles.
    """
    selected = [row for row in rows if row["lower"] >= .1 and row["upper"] <= .9]
    result = {"lower": .1, "upper": .9, "upper_inclusive": False,
              "snapshots": sum(row["snapshots"] for row in selected),
              "weighting": "equal snapshot/mode weights within the active phase"}
    for metric in ("power_fractional_error", "absolute_power_error", "field_MAE", "r21"):
        values = [row[metric] for row in selected if metric in row]
        count = sum(value["count"] - value.get("undefined", 0) for value in values)
        result[metric] = {
            "count": sum(value["count"] for value in values),
            "undefined": sum(value.get("undefined", 0) for value in values),
            "mean": sum(value["mean"] * (value["count"] - value.get("undefined", 0))
                        for value in values if value["mean"] is not None) / count if count else None}
    return result


def report(run):
    """Write one uploadable report and plots of stages and common-metric learning."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = run/"training"
    config = json.loads((root/"configuration.json").read_text())
    if config.get("snapshot_scope") != "all":
        raise ValueError("Full-history report requires snapshot_scope=all")
    coverage = json.loads((run/"coverage.json").read_text())
    if not coverage["passed"]:
        raise ValueError("Do not label a coverage-failing dataset as complete")
    summary = json.loads((root/"summary.json").read_text())
    models = [name for name in summary if name != "es"]
    training = {name: json.loads((root/name/"training.json").read_text()) for name in models}
    compact = {}
    for name, splits in summary.items():
        compact[name] = {}
        for split, values in splits.items():
            compact[name][split] = _compact_metric(values) | {
                "components": values["components"],
                "active_reionization": active_phase_summary(values["by_neutral_fraction"]),
                "by_snapshot": [{"redshift": row["redshift"], **_compact_metric(row)}
                                for row in values["by_snapshot"]],
                "by_neutral_fraction": [{"lower": row["lower"], "upper": row["upper"],
                                         "upper_inclusive": row["upper_inclusive"], **_compact_metric(row)}
                                        for row in values["by_neutral_fraction"]]}
    result = {"configuration": config, "coverage": coverage, "models": compact,
              "training": training,
              "scope": "raw evaluation: all snapshots, equal snapshot weighting, same primary k bins",
              "training_objective": config.get("training_objective", "legacy"),
              "metric_notes": {
                  "raw_relative": "abs(Ppred/Ptrue-1); no regularization; retained even at low signal",
                  "absolute": "abs(Ppred-Ptrue) in mK^2 cMpc^3",
                  "selection": "validation_loss uses the declared training selection metric; it is not the raw full-history mean",
                  "active_reionization": "0.1 <= teacher volume-mean xHI < 0.9; raw errors weighted by sample count"},
              "signal": "saturated spin temperature; no RSD",
              "teacher": "controlled H+He RT with frozen density contrast; convergence uncertified",
              "model": "snapshot ES correction with causal history features; no recurrent rollout"}
    write_json(root/"full_history_report.json", result)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for axis, split in zip(axes.flat, ("train", "validation", "test", "external_test"), strict=True):
        for i, name in enumerate(["es", *models]):
            if split not in summary[name]:
                continue
            rows = summary[name][split]["by_snapshot"]
            values = [row["power_fractional_error"]["mean"] for row in rows]
            axis.plot([row["redshift"] for row in rows],
                      [100*v if v is not None else np.nan for v in values],
                      "--" if name == "es" else "-", label=name,
                      color="gray" if name == "es" else f"C{i-1}", marker=".", markersize=3)
        axis.axhline(1, linestyle=":", color="black", linewidth=1)
        axis.set(title=split, xlabel="Redshift", ylabel="Mean absolute relative P21 error [%]", yscale="symlog")
        axis.set_ylim(bottom=0)
        axis.invert_xaxis()
        axis.grid(alpha=.2)
        axis.legend(fontsize=7, frameon=False)
    fig.suptitle("Every saved snapshot, including the exact initial state\n"
                 "Stage-resolved and absolute errors are retained in full_history_report.json")
    fig.savefig(root/"evolution_errors.png", dpi=160)
    plt.close(fig)
    regularized = config.get("training_objective") == "stage-relative"
    fig, axes = plt.subplots(2, len(models), figsize=(6*len(models), 8),
                             squeeze=False, constrained_layout=True)
    for column, name in enumerate(models):
        fit = training[name]
        for split, style, color in (("train", "-", "C0"), ("validation", "--", "C1")):
            points = learning_points(fit, split, "power_fractional_error_mean")
            axes[0, column].plot([epoch for epoch, _ in points],
                                 [100*value if value is not None else np.nan for _, value in points],
                                 style, color=color, marker=".", markersize=3, label=split)
            selected = fit.get("selected_diagnostics", {}).get(split, {})
            value = selected.get("power_fractional_error_mean")
            if value is not None:
                axes[0, column].scatter([fit["best_epoch"]], [100*value],
                                        marker="*", s=100, color=color, zorder=5,
                                        label=split+" selected checkpoint")
        axes[0, column].axhline(1, color="black", linestyle=":", linewidth=1)
        axes[0, column].set(title=name, ylabel="Raw mean absolute relative P21 error [%]",
                            yscale="symlog")
        # Every epoch's actual selection score is retained, even between diagnostics.
        multiplier = 100 if regularized else 1
        rows = fit["history"]
        axes[1, column].plot([row["epoch"] for row in rows],
                             [multiplier*row["validation_loss"] for row in rows],
                             "--", color="C1", label="validation selection score")
        train_metric = "selection_score" if regularized else "log_power_mse"
        points = learning_points(fit, "train", train_metric)
        axes[1, column].plot([epoch for epoch, _ in points],
                             [multiplier*value for _, value in points],
                             "-", color="C0", marker=".", label="train same metric")
        axes[1, column].scatter([fit["best_epoch"]], [multiplier*fit["best_validation_loss"]],
                                marker="*", s=100, color="C1", zorder=5, label="selected checkpoint")
        ylabel = ("Stage-balanced regularized P21 error [%]" if regularized
                  else "Validation selection: mean squared log-P21 error")
        axes[1, column].set(xlabel="Epoch", ylabel=ylabel, yscale="symlog")
        for axis in axes[:, column]:
            axis.axvline(fit["best_epoch"], color="gray", alpha=.4)
            axis.set_ylim(bottom=0, top=max(axis.get_ylim()[1]*1.15, 1.1))
            axis.grid(alpha=.2)
            axis.legend(frameon=False, fontsize=7)
    fig.suptitle("Raw relative errors and checkpoint-selection objective are shown separately")
    fig.savefig(root/"learning_curves.png", dpi=160)
    plt.close(fig)


def main(argv=None):
    """Expose the pre-training coverage gate and post-training report."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="phase", required=True)
    coverage_parser = commands.add_parser("coverage")
    coverage_parser.add_argument("--manifest", type=Path, required=True)
    coverage_parser.add_argument("--output", type=Path, required=True)
    coverage_parser.add_argument("--z-start", type=float, default=20.)
    coverage_parser.add_argument("--z-end", type=float, default=4.)
    coverage_parser.add_argument("--snapshots", type=int, default=41)
    coverage_parser.add_argument("--end-xhi", type=float, default=.01)
    report_parser = commands.add_parser("report")
    report_parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.phase == "coverage":
        cases = [load_case(row) for row in load_manifest(args.manifest)["cases"]]
        result = check_coverage(cases, args.z_start, args.z_end, args.snapshots, args.end_xhi)
        result["manifest"] = str(args.manifest.resolve())
        write_json(args.output, result)
        print(json.dumps({"coverage_passed": result["passed"], "failed_cases": result["failed_case_ids"]}), flush=True)  # noqa: T201
        if not result["passed"]:
            raise SystemExit("Incomplete evolution coverage: inspect "+str(args.output))
    else:
        report(args.run.resolve())


if __name__ == "__main__":
    main()
