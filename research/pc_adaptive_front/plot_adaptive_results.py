"""Plot finalized canonical benchmarks without running any simulations.

The sphere panel compares against a continuum solution integrated over parent
voxels. Native-field panels measure discrete-model costs, not cosmological
convergence or continuum accuracy. Run only after benchmarks.py has finished.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

HERE = Path(__file__).resolve().parent
COLORS = ("#4477AA", "#EE9944", "#228866")
SPHERE_MODELS = ("matched_coarse", "full_subcells", "adaptive_subcells")
NATIVE_MODELS = ("balanced_coarse", "full_subcells", "adaptive_subcells")


def unique_case(report, *, name, angles=None):
    """Reject incomplete or ambiguous benchmark inputs instead of plotting them."""
    matches = [case for case in report["cases"]
               if case["case"] == name and case.get("angles") == angles]
    if len(matches) != 1:
        raise ValueError(f"Expected one finalized case {name!r}, K={angles}; found {len(matches)}")
    return matches[0]


def model_row(case, name):
    rows = [row for row in case["models"] if row["model"] == name]
    if len(rows) != 1:
        raise ValueError(f"Expected model {name!r} in case {case['case']!r}")
    return rows[0]


def panel(ax, values, labels, title, ylabel, subtitle, number_format):
    """Draw comparable grouped bars from a [group, model] value array."""
    values = np.asarray(values, dtype=float)
    if values.shape != (2, 3) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"Invalid panel values for {title}: {values}")
    x = np.arange(2, dtype=float)
    width = 0.235
    for index, color in enumerate(COLORS):
        bars = ax.bar(x+(index-1)*width, values[:, index], width=width,
                      color=color, edgecolor="white", linewidth=0.6, zorder=3)
        ax.bar_label(bars, labels=[format(value, number_format) for value in values[:, index]],
                     padding=3, fontsize=8.5, color="#252525")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, float(values.max())*1.23)
    ax.set_ylabel(ylabel, labelpad=7)
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="semibold", pad=27)
    ax.text(0, 1.035, subtitle, transform=ax.transAxes, fontsize=8.5, color="#666666")
    ax.grid(axis="y", color="#E4E4E4", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color("#AAAAAA")
        ax.spines[side].set_linewidth(0.65)
    ax.tick_params(axis="both", color="#AAAAAA", labelsize=9)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.margins(x=0.10)


def plot(report, output_base, dpi=220):
    sphere_cases = [unique_case(report, name="isolated", angles=6144),
                    unique_case(report, name="isolated-r8", angles=6144)]
    native_cases = [unique_case(report, name="native32"),
                    unique_case(report, name="native64")]
    errors = [[100*model_row(case, name)["vs_continuum_sphere"]["L1_over_emitted"]
               for name in SPHERE_MODELS] for case in sphere_cases]
    times = [[model_row(case, name)["median_seconds"] for name in NATIVE_MODELS]
             for case in native_cases]
    memory = [[model_row(case, name)["process_memory"]["peak_process_rss_bytes"]/1e6
               for name in NATIVE_MODELS] for case in native_cases]
    samples = {len(model_row(case, name)["seconds_samples"])
               for case in native_cases for name in NATIVE_MODELS}
    timing_note = (f"Median of {samples.pop()} serial calls; compile excluded"
                   if len(samples) == 1 else "Median serial wall time; compile excluded")

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelsize": 10, "pdf.fonttype": 42,
                         "savefig.facecolor": "white", "figure.facecolor": "white"}):
        fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.45))
        fig.subplots_adjust(left=0.062, right=0.985, top=0.80, bottom=0.32, wspace=0.32)
        panel(axes[0], errors, [r"$R=4\Delta x$", r"$R=8\Delta x$"],
              "(a) Uniform sphere: continuum error", r"$L_1/Q$ (%)",
              "Matched angular rule; K = 6,144 directions", ".2f")
        panel(axes[1], times, [r"$32^3$", r"$64^3$"],
              "(b) Native fields: serial runtime", "Solve wall time (s)",
              timing_note, ".3f")
        panel(axes[2], memory, [r"$32^3$", r"$64^3$"],
              "(c) Native fields: peak memory", "Peak process RSS (MB)",
              "Independent fresh process per model", ".1f")
        for ax in axes[1:]:
            ax.set_xlabel("Parent grid", labelpad=7)
        handles = [Patch(facecolor=color, label=label) for color, label in zip(
            COLORS, ("Coarse: matched rays (a), balanced (b–c)",
                     "Uniform eight subcells", "Adaptive eight subcells"))]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.51, 0.135),
                   ncol=3, frameon=False, fontsize=9, handlelength=1.5, columnspacing=2)
        fig.text(0.062, 0.075,
                 "Sphere errors use the continuum solution integrated over parent voxels. "
                 "Native panels measure frozen discrete-model cost; they do not establish cosmological convergence.",
                 fontsize=8.2, color="#555555", ha="left")
        fig.text(0.062, 0.032,
                 "Wall time includes solver setup and outputs. RSS includes Python, inputs and outputs; "
                 "MB = 10⁶ bytes. Timings are sequential measurements on a shared host.",
                 fontsize=8.2, color="#555555", ha="left")
        output_base = Path(output_base)
        output_base.parent.mkdir(parents=True, exist_ok=True)
        for extension in ("png", "pdf"):
            target = output_base.with_suffix(f".{extension}")
            fig.savefig(target, dpi=dpi, bbox_inches="tight", pad_inches=0.12,
                        metadata={"Creator": "21cmFAST adaptive front benchmark"})
            print(target)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE/"results.json")
    parser.add_argument("--output-base", type=Path, default=HERE/"report_figure")
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    plot(json.loads(args.results.read_text()), args.output_base, args.dpi)


if __name__ == "__main__":
    main()
