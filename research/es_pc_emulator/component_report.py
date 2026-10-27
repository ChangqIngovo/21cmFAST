"""Report paired spectrum-component comparisons without fitting another model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.es_pc_emulator.small_pilot_training import distribution, write_json
from research.es_pc_emulator.spectral_components import TERM_NAMES

LABELS = {
    "thermo": "Thermo baseline", "thermo_p21": "+ total P21",
    "thermo_pxx": "+ neutral auto", "thermo_cross": "+ density-neutral cross",
    "thermo_nonlinear": "+ nonlinear terms", "thermo_components": "+ all six terms",
    "thermo_components_p21": "+ six terms and P21",
}


def case_scores(rows):
    """Average primary bins within each simulation before paired comparisons."""
    grouped = {}
    closure = 0.
    for row in rows:
        if "component_closure_relative" in row:
            closure = max(closure, *row["component_closure_relative"].values())
        if not row["primary_snapshot"]:
            continue
        key = (row["split"], row["case_id"])
        grouped.setdefault(key, [])
        for index, use in enumerate(row["primary_k"]):
            if not use:
                continue
            error = row["fractional_power_error"][index]
            if error is not None:
                grouped[key].append(error)
            terms = [values[index] for values in row["component_error_over_P21"]]
            if all(value is not None for value in terms):
                ratio_error = row["P_predicted"][index]/row["P_teacher"][index]-1
                np.testing.assert_allclose(sum(terms), ratio_error, rtol=1e-8, atol=1e-9)
    if any(not values for values in grouped.values()):
        raise ValueError("A case has no defined primary power errors")
    return {key: float(np.mean(values)) for key, values in grouped.items()}, closure


def plot_comparison(summary, models, seeds, output):
    """Plot seed ranges and the signed-term error magnitudes on held-out cases."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    ticks = np.arange(len(models))
    for column, split in enumerate(("test", "external_test")):
        ax, heat = axes[:, column]
        if split not in summary[f"thermo_seed{seeds[0]}"]:
            ax.set_axis_off()
            heat.set_axis_off()
            continue
        values = np.array([[summary[f"{model}_seed{seed}"][split]["power_fractional_error"]["mean"]
                            for seed in seeds] for model in models])*100
        if not np.isfinite(values).all():
            raise ValueError("Undefined total-power summary")
        mean = values.mean(axis=1)
        ax.bar(ticks, mean, color="#2b7b8f", alpha=.8)
        ax.errorbar(ticks, mean, yerr=[mean-values.min(axis=1), values.max(axis=1)-mean],
                    fmt="none", capsize=3, color="black", linewidth=1)
        ax.axhline(1, color="#c24b3f", linestyle="--", linewidth=1, label="1% target")
        ax.set_xticks(ticks, [LABELS[m] for m in models], rotation=30, ha="right")
        ax.set(ylabel="Mean absolute P21 error [%]", title=split)
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=.2)
        components = np.array([[[summary[f"{model}_seed{seed}"][split]["components"][term]["mean"]
                                  for term in TERM_NAMES] for seed in seeds] for model in models])*100
        components = components.mean(axis=1)
        if not np.isfinite(components).all():
            raise ValueError("Undefined component summary")
        image = heat.imshow(np.log10(1+components), cmap="magma", aspect="auto")
        logged = np.log10(1+components)
        contrast = max(float(logged.max()-logged.min()), 1e-12)
        for (row, col), value in np.ndenumerate(components):
            heat.text(col,row,f"{value:.1f}",ha="center",va="center",fontsize=8,
                      color="white" if (logged[row,col]-logged.min())/contrast < .65 else "black")
        heat.set_xticks(np.arange(6), [r"$P_{DD}$", r"$P_{XX}$", r"$2P_{DX}$",
                                      r"$P_{NN}$", r"$2P_{DN}$", r"$2P_{XN}$"])
        heat.set_yticks(ticks, [LABELS[m] for m in models])
        heat.set_title("Mean |component error| / true P21 [%]")
        fig.colorbar(image, ax=heat, label="log10(1 + percent error)", fraction=.04)
    fig.suptitle("Same small network and data; only the supervised spectra change\n"
                 "Bars average network seeds; whiskers show their range", fontsize=14)
    fig.savefig(output/"component_comparison.png", dpi=170)
    plt.close(fig)


def main(argv=None):
    """Validate decomposition and save paired whole-case model differences."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.training
    config = json.loads((root/"configuration.json").read_text())
    summary = json.loads((root/"summary.json").read_text())
    models, seeds = config["arms"], config["model_seeds"]
    if "thermo" not in models or any(model not in LABELS for model in models):
        raise ValueError("Use thermo and the supported thermo_* loss comparisons")
    report = {"scope": "paired whole-simulation differences; no posterior or new training",
              "selection": "epochs selected on validation P21 only; test sets describe performance",
              "units": "absolute fractional power error; negative paired changes are improvements",
              "parameter_seeds": seeds, "models": {}}
    for seed in seeds:
        name = f"thermo_seed{seed}"
        baseline, _ = case_scores(json.loads((root/name/"predictions/metrics.json").read_text()))
        for model in models:
            name = f"{model}_seed{seed}"
            scores, closure = case_scores(json.loads((root/name/"predictions/metrics.json").read_text()))
            if scores.keys() != baseline.keys():
                raise ValueError("Paired model comparisons require identical whole-case splits")
            results = {"maximum_six_term_closure_relative": closure, "splits": {}}
            for split in sorted({key[0] for key in scores}):
                records = [{"case_id": key[1], "baseline_error": baseline[key],
                            "model_error": scores[key], "difference": scores[key]-baseline[key]}
                           for key in sorted(scores) if key[0] == split]
                results["splits"][split] = {
                    "cases": records,
                    "paired_error_difference": distribution(row["difference"] for row in records),
                    "improved_cases": sum(row["difference"] < 0 for row in records),
                    "total_cases": len(records)}
            report["models"][name] = results
    write_json(root/"component_report.json", report)
    plot_comparison(summary, models, seeds, root)
    print(json.dumps({"report": str(root/"component_report.json"),  # noqa: T201
                      "figure": str(root/"component_comparison.png")}), flush=True)


if __name__ == "__main__":
    main()
