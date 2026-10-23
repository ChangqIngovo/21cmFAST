"""Reproduce the static comparison figure from final benchmark JSONs only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
LABELS = {"es-control": "ES control", "es-global": "ES global", "es-front": "ES front",
          "radial-shell": "Radial shell", "voronoi": "Voronoi", "adaptive-event": "Adaptive event"}
COLORS = {"es-control": "#707070", "es-global": "#D55E00", "es-front": "#E69F00",
          "radial-shell": "#CC79A7", "voronoi": "#0072B2", "adaptive-event": "#009E73"}


def plot(native_report, toy_report, output):
    native = next(c for c in native_report["cases"] if c["name"] == "native-frozen")
    plane = next(c for c in toy_report["cases"] if c["name"] == "opaque-plane")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    ax = axes[0]
    offsets = {"es-control": (7, 10), "es-global": (7, 4), "es-front": (-2, -18),
               "radial-shell": (-83, -17), "voronoi": (-46, 8), "adaptive-event": (8, 8)}
    for model in native["models"]:
        name = model["method"]
        x = model["seconds"]
        y = 100 * model["vs_event_reference"]["L1_absorption_over_emitted"]
        ax.scatter(x, y, s=65, color=COLORS[name], zorder=3)
        ax.annotate(LABELS[name], (x, y), xytext=offsets[name], textcoords="offset points",
                    fontsize=9, color=COLORS[name])
    error = 100 * native["reference_angular_change"]["L1_absorption_over_emitted"]
    ax.axhspan(error / 2, error, color="0.85")
    ax.text(.025, error * .8, f"Reference angular change: {error:.3f}%", fontsize=9, color="0.4")
    ax.set(xscale="log", yscale="log", xlabel="Transport / allocation time (s; one worker)",
           ylabel="Absorption L1 / emitted photons (%)", title="Frozen native 32³ field at z = 7.5",
           xlim=(.018, 60), ylim=(.025, 18))
    ax.grid(alpha=.18)
    ax = axes[1]
    rows = [m for m in plane["models"] if m["method"] != "es-control"]
    xs = list(range(len(rows)))
    fractions = [100 * m["ledger"]["absorbed"] / m["ledger"]["emitted"] for m in rows]
    ax.bar(xs, fractions, color=[COLORS[m["method"]] for m in rows], width=.65)
    analytic = 100 * plane["metadata"]["analytic_absorbed_fraction"]
    ax.axhline(analytic, color="#222222", ls="--", lw=1.4,
               label=f"Isotropic geometric result: {analytic:.3f}%")
    ax.set_xticks(xs, [LABELS[m["method"]].replace(" ", "\n") for m in rows], fontsize=9)
    ax.set(ylabel="Emitted photons absorbed by the plane (%)", ylim=(0, 112),
           title="Opaque plane with finite travel distance")
    ax.legend(loc="upper center", fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=.18)
    fig.suptitle("A correct photon count does not guarantee correct photon placement", fontsize=13)
    for extension in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{extension}"), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, default=HERE / "comparison_native_refined_results.json")
    parser.add_argument("--toy", type=Path, default=HERE / "comparison_final_results.json")
    parser.add_argument("--output", type=Path, default=HERE / "comparison_summary")
    args = parser.parse_args()
    plot(json.loads(args.native.read_text()), json.loads(args.toy.read_text()), args.output)


if __name__ == "__main__":
    main()
