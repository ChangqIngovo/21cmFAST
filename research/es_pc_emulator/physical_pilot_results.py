"""Audit and plot held-out field-network ablations from measured pilot reports.

The selected model remains the validation-MSE winner. Test-set rankings are
descriptive only. This small-box experiment cannot certify one-percent physical
accuracy or establish that any omitted physical input is unnecessary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ARM_LABELS = {
    "es": "ES-only NN",
    "budget": "ES + density + photons",
    "history": "Budget + source history",
    "all": "All inputs",
    "no_sources": "All without source inputs",
    "no_density": "All without density",
    "no_photons": "All without cumulative photons",
    "no_history": "All without source history",
    "no_spectrum": "All without spectral temperature",
    "no_clumping": "All without clumping",
}
SPLITS = ("test", "external_test")
PRIMARY_REDSHIFTS = (10.925, 9.4125, 7.9)


def _distribution(values):
    array = np.asarray(values, dtype=float).ravel()
    valid = array[np.isfinite(array)]
    result = {"count": len(array), "finite_count": len(valid),
              "undefined_count": len(array) - len(valid)}
    for key, function in (("mean", np.mean), ("median", np.median),
                          ("minimum", np.min), ("maximum", np.max)):
        result[key] = float(function(valid)) if len(valid) else None
    for key, quantile in (("p25", .25), ("p75", .75), ("p95", .95)):
        result[key] = float(np.quantile(valid, quantile)) if len(valid) else None
    return result


def _errors(row, name, selected):
    spectra = row["spectra"]
    ratios = spectra[f"ratio_{name}"]
    reference = np.asarray(spectra["P_teacher"], dtype=float)
    predicted = np.asarray(spectra[f"P_{name}"], dtype=float)
    if (reference.shape != selected.shape or predicted.shape != selected.shape
            or len(ratios) != len(selected) or not np.isfinite(reference).all()
            or not np.isfinite(predicted).all() or np.any(reference < 0)
            or np.any(predicted < 0)):
        raise ValueError("Invalid measured power spectra")
    errors = []
    for index in np.flatnonzero(selected):
        value = ratios[index]
        if value is None:
            errors.append(None)
        elif (not np.isfinite(value) or reference[index] <= 0
              or not np.isclose(value, predicted[index] / reference[index], rtol=1e-12, atol=1e-14)):
            raise ValueError("Reported linear power ratio does not match measured spectra")
        else:
            errors.append(float(abs(value - 1)))
    return errors


def summarize_split(report, split, *, expected_redshifts=PRIMARY_REDSHIFTS):
    """Measure individual history/redshift/k errors; never treat cells as runs."""
    grouped = {}
    for row in report["samples"]:
        if row["split"] != split:
            raise ValueError("Snapshot has the wrong evaluation split")
        grouped.setdefault(row["case_id"], []).append(row)
    if (not grouped or report["cases"] != len(grouped)
            or report.get("snapshots", len(report["samples"])) != len(report["samples"])):
        raise ValueError("Evaluation case count is inconsistent")
    histories = {"nn": [], "es": []}
    shared_k, shared_z, fingerprints = None, None, {}
    for case_id, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["snapshot"])
        snapshots = [row["snapshot"] for row in rows]
        redshifts = np.array([row["redshift"] for row in rows], dtype=float)
        if (len(rows) < 3 or len(set(snapshots)) != len(rows)
                or not np.isfinite(redshifts).all() or not np.all(np.diff(redshifts) < 0)
                or len({row["seed"] for row in rows}) != 1):
            raise ValueError("Each history needs distinct ordered snapshots and one seed")
        selected_rows = rows[-3:]
        if any(row["primary_snapshot"] != (index >= len(rows) - 3)
               for index, row in enumerate(rows)):
            raise ValueError("Primary snapshot flags do not select the last three redshifts")
        if not np.allclose(redshifts[-3:], expected_redshifts, rtol=0, atol=1e-10):
            raise ValueError("Primary redshifts do not match the controlled pilot")
        for name, entries in histories.items():
            errors, maes = [], []
            for row in selected_rows:
                spectra = row["spectra"]
                k = np.asarray(spectra["k_cMpc_inv"], dtype=float)
                nyquist = spectra["k_max_cMpc_inv"]
                if (k.ndim != 1 or not np.isfinite(k).all() or not np.all(np.diff(k) > 0)
                        or not np.isfinite(nyquist) or nyquist <= 0):
                    raise ValueError("Invalid spectral geometry")
                selected = (k >= .1) & (k <= .6 * nyquist)
                if sum(selected) != 2:
                    raise ValueError("The declared pilot target requires exactly two k shells")
                if shared_k is None:
                    shared_k, shared_z = k[selected], redshifts[-3:]
                elif not np.array_equal(shared_k, k[selected]):
                    raise ValueError("Histories use different target k shells")
                errors.append(_errors(row, name, selected))
                mae = row[name]["MAE"]
                if mae is None or not np.isfinite(mae) or not 0 <= mae <= 1:
                    raise ValueError("Invalid field MAE")
                maes.append(float(mae))
            error_summary = _distribution(errors)
            entries.append({"case_id": case_id, "seed": rows[0]["seed"],
                "power_absolute_relative_error": errors,
                "power_error_distribution": error_summary,
                "field_MAE_by_redshift": maes, "mean_field_MAE": float(np.mean(maes))})
        # Target and ES values must be identical across input ablations.
        fingerprints[case_id] = [{"redshift": row["redshift"], "seed": row["seed"],
            "P_teacher": row["spectra"]["P_teacher"], "P_es": row["spectra"]["P_es"],
            "ratio_es": row["spectra"]["ratio_es"], "es_MAE": row["es"]["MAE"]}
            for row in selected_rows]
    result = {"histories": len(grouped), "redshifts": shared_z.tolist(),
              "k_cMpc_inv": shared_k.tolist(), "seeds": sorted({rows[0]["seed"] for rows in grouped.values()})}
    for name, entries in histories.items():
        result[name] = {
            "power_error_cells": _distribution([r["power_absolute_relative_error"] for r in entries]),
            "per_history_mean_power_error": _distribution([r["power_error_distribution"]["mean"] for r in entries]),
            "field_MAE_snapshots": _distribution([r["field_MAE_by_redshift"] for r in entries]),
            "per_history_mean_field_MAE": _distribution([r["mean_field_MAE"] for r in entries]),
            "per_history": entries,
        }
    return result, fingerprints


def summarize(training):
    """Keep validation model selection distinct from descriptive test rankings."""
    if training.get("test_used_for_selection") is not False:
        raise ValueError("Reports must declare that test was not used for selection")
    if set(training["arms"]) != set(ARM_LABELS):
        raise ValueError("Expected all ten declared input ablations")
    selections = training["selections"]
    if (len(selections) != len(ARM_LABELS) or {r["arm"] for r in selections} != set(ARM_LABELS)
            or any(not np.isfinite(row["validation_mse"]) or row["validation_mse"] < 0
                   or row.get("test_used_for_selection") is not False for row in selections)):
        raise ValueError("Invalid validation model selections")
    selected = min(selections, key=lambda row: (row["validation_mse"], row["arm"]))
    if training["selected"] != selected:
        raise ValueError("Selected arm is not the declared validation-MSE winner")
    summaries, baseline, comparison = {}, {}, {}
    for arm in ARM_LABELS:
        summaries[arm] = {}
        for split in SPLITS:
            result, fingerprint = summarize_split(training["arms"][arm][split], split)
            if split in comparison and fingerprint != comparison[split]:
                raise ValueError("Input ablations use different held-out targets or raw ES baselines")
            comparison[split] = fingerprint
            baseline[split] = {key: value for key, value in result.items() if key != "nn"}
            summaries[arm][split] = {key: value for key, value in result.items() if key != "es"}
    if set(comparison["test"]) & set(comparison["external_test"]):
        raise ValueError("Test and external-test histories overlap")
    lowest_test = {}
    for split in SPLITS:
        scores = [(row[split]["nn"]["power_error_cells"]["mean"], arm)
                  for arm, row in summaries.items()
                  if row[split]["nn"]["power_error_cells"]["undefined_count"] == 0]
        lowest_test[split] = (min(scores)[1] if scores else None)
    return {
        "schema_version": 1, "selected_by_validation": selected,
        "lowest_test_mean_power_error_arm_posthoc_only": lowest_test,
        "scope": "small-box physical-input ablations; not a one-percent or universal-ES certificate",
        "input_absence_interpretation": "Ablations measure incremental predictive information beyond retained ES features; weak improvement cannot establish physical irrelevance",
        "objective": training["objective"],
        "brightness_convention": training["brightness_convention"],
        "selection_used_test": False, "strict_photon_conservation": False,
        "metric_definition": "abs(P21_NN/P21_RT-1) at each held-out history, last-three redshifts and target k shell",
        "target_k_definition": "0.1 <= k <= 0.6*k_Nyquist [cMpc^-1]",
        "distribution_weighting": "Each history/redshift/k cell has equal weight; per-history means average its six cells",
        "field_error_definition": "Unweighted voxel mean absolute xHII error (fraction units; xHI MAE is identical)",
        "figure_definition": "Dot is mean of per-history mean power errors; whisker spans their 25th-75th percentiles",
        "undefined_errors": "Undefined reference-power ratios remain null and are counted; no finite subset certifies target accuracy",
        "primary_test_seed_disjoint": training["primary_test_seed_disjoint"],
        "seeds_by_split": training["seeds_by_split"],
        "arms": summaries, "raw_ES_baseline": baseline,
    }


def plot_ablation(summary, output):
    """Display all measured ablations without selecting by test performance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 7.1), sharey=True)
    selected_arm = summary["selected_by_validation"]["arm"]
    for ax, split, title in zip(axes, SPLITS, ("Held-out parameters · same IC", "External IC seeds"), strict=True):
        baseline = summary["raw_ES_baseline"][split]
        es = baseline["es"]["per_history_mean_power_error"]
        if es["mean"] is not None:
            ax.axvspan(100 * es["p25"], 100 * es["p75"], color="0.75", alpha=.25)
            ax.axvline(100 * es["mean"], color="0.35", linestyle="--", linewidth=1.3)
        ax.axvline(1, color="#387245", linewidth=1.3, linestyle=":")
        undefined = baseline["es"]["power_error_cells"]["undefined_count"] > 0
        for y, arm in enumerate(ARM_LABELS):
            metrics = summary["arms"][arm][split]["nn"]
            values = metrics["per_history_mean_power_error"]
            undefined |= metrics["power_error_cells"]["undefined_count"] > 0
            color = "#b35806" if arm == selected_arm else "#2166ac"
            if values["mean"] is None:
                ax.text(.03, y, "Undefined", transform=ax.get_yaxis_transform(), color=color, va="center")
            else:
                ax.plot(100 * np.array([values["p25"], values["p75"]]), [y, y], color=color, linewidth=2)
                ax.plot(100 * values["mean"], y, marker="D" if arm == selected_arm else "o", color=color, ms=6)
        ax.set_xscale("symlog", linthresh=.1, linscale=.4)
        ax.set_xlim(left=0)
        ax.set_yticks(np.arange(len(ARM_LABELS)), labels=list(ARM_LABELS.values()))
        ax.set_xlabel("Mean absolute fractional power error [%]")
        caveat = "; finite values only" if undefined else ""
        ax.set_title(f"{title}\n{baseline['histories']} histories{caveat}", fontsize=12)
        ax.grid(axis="x", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    fig.suptitle("Field-network input ablations on held-out native RT histories", fontsize=15)
    fig.text(.5, .918, "Last 3 redshifts; 0.1 ≤ k ≤ 0.6 kNy; saturated spin temperature, no RSD", ha="center", fontsize=10)
    fig.legend(handles=[Line2D([], [], color="#2166ac", marker="o", label="NN mean; bars = history 25-75%"),
                        Line2D([], [], color="#b35806", marker="D", linestyle="", label="Selected by validation MSE"),
                        Line2D([], [], color=".35", linestyle="--", label="Raw ES mean"),
                        Patch(facecolor=".75", alpha=.25, label="Raw ES history 25-75%"),
                        Line2D([], [], color="#387245", linestyle=":", label="1% reference")],
               loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, .12, 1, .90))
    for extension in ("png", "pdf"):
        fig.savefig(output / f"input_ablation.{extension}", dpi=180)
    plt.close(fig)


def analyze(root, output):
    """Audit completed aggregate/per-arm reports and preserve file provenance."""
    root, output = Path(root).resolve(), Path(output).resolve()
    if not (root / "COMPLETED").is_file():
        raise ValueError("Field training has not completed")
    if output.exists():
        raise FileExistsError("Use a fresh results output directory")
    path = root / "summary.json"
    training = json.loads(path.read_text())
    sources = {"summary.json": hashlib.sha256(path.read_bytes()).hexdigest()}
    for arm, reports in training["arms"].items():
        for split in SPLITS:
            path = root / arm / split / "summary.json"
            if json.loads(path.read_text()) != reports[split]:
                raise ValueError("Per-arm report differs from the training summary")
            sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    summary = summarize(training)
    summary["source_report_sha256"] = sources
    summary["source_manifest_sha256"] = training["source_manifest_sha256"]
    summary["analysis_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    (output / "field_ablation_summary.json").write_text(json.dumps(
        summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    plot_ablation(summary, output)
    return summary


def main():
    """Render actual completed pilot field diagnostics into a fresh directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze(args.root, args.output)
    print(json.dumps({"output": str(args.output),  # noqa: T201
                      "selected_by_validation": summary["selected_by_validation"]}))


if __name__ == "__main__":
    main()
