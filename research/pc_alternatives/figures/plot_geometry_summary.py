"""Render saved research results; never rerun a transport solver.

Outputs PNG/PDF figures with data-source hashes for reproducibility. The sphere
panel compares geometric errors, not equal-cost or equal-history algorithms.
The optional overlap map uses exactly the arrays/metadata saved by the independent
event geometry probe. Production 21cmFAST is not imported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import PercentFormatter, ScalarFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
BLUE = "#176487"
ORANGE = "#ce673a"
GRAY = "#66717a"


def _read(name):
    path = ROOT/name
    raw = path.read_bytes()
    return json.loads(raw), {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(raw).hexdigest()}


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": .8,
        "lines.linewidth": 1.8, "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _save(figure, name, provenance):
    for suffix in ("png", "pdf"):
        figure.savefig(OUT/f"{name}.{suffix}", dpi=300)
    (OUT/f"{name}_sources.json").write_text(json.dumps(provenance, indent=2)+"\n")
    plt.close(figure)


def geometry_checks():
    """Plot spatial sphere error and independent angular-cone sampling error."""
    event, event_source = _read("event_geometry_results.json")
    angular, angular_source = _read("angular_resolution_results.json")
    active, active_source = _read("angular_high_order_active_results.json")
    first, first_source = _read("angular_results.json")
    figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.45))
    figure.subplots_adjust(left=.075, right=.98, bottom=.20, top=.86, wspace=.31)
    figure.suptitle("Geometry checks for photon-conserving transport", fontsize=14, y=.98)
    ax = axes[0]
    rows = event["sphere_convergence"]["rows"]
    ax.plot([r["radius_cells"] for r in rows],
            [r["sphere_voxel_volume_l1_per_photon"] for r in rows],
            "o-", color=BLUE, label=r"Event / DDA, $K=\lceil16\pi R^2\rceil$", markersize=5)
    for row in rows:
        ax.annotate(f"K={row['angles']:,}",
                    (row["radius_cells"], row["sphere_voxel_volume_l1_per_photon"]),
                    xytext=(0, -15), textcoords="offset points", ha="center", fontsize=8, color=BLUE)
    muscl = list(angular["resolution"])
    muscl += [r for r in active["sphere_error_cost_curve"]
              if r["method"] == "MUSCL_MC" and r["directions"] == 96]
    muscl.sort(key=lambda r: r["radius_cells"])
    ax.plot([r["radius_cells"] for r in muscl],
            [r["sphere_voxel_L1_over_photons"] for r in muscl],
            "s-", color=ORANGE, label="MUSCL, fixed K=96", markersize=4.8)
    fine = angular["angular_refinement_at_R12"]
    ax.scatter([fine["radius_cells"]], [fine["sphere_voxel_L1_over_photons"]],
               s=48, marker="D", facecolor="white", edgecolor=ORANGE, linewidth=1.6, zorder=5,
               label="MUSCL, K=384 at R=12")
    ax.set(xlim=(3, 13.5), ylim=(0, .49), xticks=(4, 6, 8, 10, 12),
           xlabel=r"Resolved sphere radius $R/\Delta x$",
           ylabel=r"Voxel-reference error $\|A-A_{\rm sphere}\|_1/Q$",
           title="(a) Isolated sphere")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", color="#e5e9ec", linewidth=.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, borderpad=.15, handlelength=1.5)

    ax = axes[1]
    cone = first["one_eighth_cone"]
    angles = [r["directions"] for r in cone]
    ax.plot(angles, [r["rms_absolute_error"] for r in cone], "o-", color=BLUE,
            label="RMS over orientations", markersize=5)
    ax.plot(angles, [r["max_absolute_error"] for r in cone], "s--", color=GRAY,
            label="Largest sampled error", markersize=4)
    ax.set(xscale="log", yscale="log", xlim=(10, 480), ylim=(.0025, .18),
           xticks=angles, xlabel="Angular directions K",
           ylabel="Absolute error in photon fraction",
           title="(b) Sampling the 12.5% overlap cone")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    ax.set_yticks([.003, .01, .03, .1])
    ax.minorticks_off()
    ax.grid(axis="both", color="#e5e9ec", linewidth=.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, borderpad=.2)
    ax.text(.04, .05, "1,000 orientations; equal-weight\nantipodal Fibonacci quadrature",
            transform=ax.transAxes, fontsize=8.5, va="bottom", color=GRAY)
    figure.text(.075, .075,
                "Left: angular resolutions differ; geometric convergence, not an equal-cost comparison. "
                "Right: sampled maxima are not uncertainty intervals.", fontsize=8.2, color=GRAY)
    figure.text(.075, .035,
                "Toy sharp-front tests without recombinations. Event/DDA uses constant rates and infinite c; "
                "MUSCL follows a finite-c flash toward the same sphere reference.", fontsize=8.2, color=GRAY)
    _save(figure, "geometry_checks", {"sources": [event_source, angular_source, active_source, first_source],
                                    "scope": "Geometric error diagnostics; not matched-cost or cosmological validation"})


def overlap_maps():
    """Plot two matched initial-mask increment fields using a common color scale."""
    path = ROOT/"event_overlap_fields.npz"
    if not path.exists():
        return False
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        transparent = np.asarray(data["transparent"], dtype=bool)
        event = np.asarray(data["event_absorbed"], dtype=float)
        reference = np.asarray(data["frozen_dda_absorbed"], dtype=float)
        scalar = np.asarray(data["script_like_absorbed"], dtype=float)
        source = np.asarray(data["source"], dtype=float)
    photons = float(source.sum())
    if not photons > 0 or event.shape != scalar.shape or event.shape != reference.shape:
        raise ValueError("Invalid matched overlap arrays")
    for name, field in (("event", event), ("reference", reference), ("scalar", scalar)):
        if not np.isfinite(field).all() or np.min(field) < 0:
            raise ValueError(f"Invalid {name} photon allocation")
        if np.sum(field[transparent]) > 1e-8*photons:
            raise ValueError(f"{name} assigns photons to preionized interior")
    n = event.shape[0]
    projected = [scalar.sum(axis=2)/photons, event.sum(axis=2)/photons]
    maximum = max(float(p.max()) for p in projected)
    norm = LogNorm(vmin=max(maximum*1e-3, 1e-7), vmax=maximum)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#f7f8fa")
    figure, axes = plt.subplots(1, 2, figsize=(10.7, 5.05), sharex=True, sharey=True)
    figure.subplots_adjust(left=.065, right=.865, bottom=.18, top=.85, wspace=.12)
    figure.suptitle("Where a new photon increment is absorbed", fontsize=14, y=.98)
    figure.text(.465, .915, r"Two preionized bubbles: $R=12\Delta x$, $d=1.5R$; isotropic emission at the left centre",
                ha="center", fontsize=10, color=GRAY)
    source_position = np.unravel_index(np.argmax(source), source.shape)
    z_slice = source_position[2]
    occupied = transparent.any(axis=2)
    extent_indices = np.where(occupied)
    xlimits = (max(-.5, int(extent_indices[0].min())-5.5), min(n-.5, int(extent_indices[0].max())+5.5))
    ylimits = (max(-.5, int(extent_indices[1].min())-5.5), min(n-.5, int(extent_indices[1].max())+5.5))
    for ax, field, title in zip(axes, projected,
                                ("(a) SCRIPT-like radial redistribution", "(b) Event / DDA first absorption"), strict=True):
        mapped = np.ma.masked_less_equal(field.T, 0)
        mesh = ax.imshow(mapped, origin="lower", extent=(-.5, n-.5, -.5, n-.5),
                         cmap=cmap, norm=norm, interpolation="nearest", rasterized=True)
        ax.contour(np.arange(n), np.arange(n), transparent[:, :, z_slice].T,
                   levels=[.5], colors=["#55969a"], linewidths=.9, linestyles="--")
        ax.scatter([source_position[0]], [source_position[1]], marker="*", s=78,
                   facecolor="#168084", edgecolor="white", linewidth=.8, zorder=5)
        ax.set(xlim=xlimits, ylim=ylimits, xlabel=r"$x/\Delta x$", title=title, aspect="equal")
        centroid = float((field*np.arange(n)[:, None]).sum()/field.sum()-source_position[0])
        ax.text(.04, .025, f"Absorption centroid shift: {centroid:+.2f} cells", transform=ax.transAxes,
                fontsize=8.4, color=GRAY, va="bottom")
    axes[0].set_ylabel(r"$y/\Delta x$")
    cax = figure.add_axes([.89, .265, .016, .50])
    colorbar = figure.colorbar(mesh, cax=cax)
    colorbar.set_label(r"Projected absorbed fraction $\sum_z\Delta A/Q$")
    discrepancy = float(abs(event-reference).sum()/photons)
    figure.text(.065, .105,
                f"Same initial two-bubble voxel mask and Q={photons:g} photons. "
                f"Event vs independent DDA: L1/Q={discrepancy:.2g}.", fontsize=9, color=GRAY)
    figure.text(.065, .055,
                "Projection along z; dashed contour marks the initial midplane boundary. "
                "The source is a star. No cell exhausts its new capacity in the DDA increment.", fontsize=8.2, color=GRAY)
    _save(figure, "overlap_photon_deposition", {
        "sources": [{"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}],
        "metadata": metadata, "event_reference_L1_over_Q": discrepancy,
        "color_scale": {"kind": "logarithmic", "vmin": norm.vmin, "vmax": norm.vmax},
        "scope": "Matched initial state and total increment; radial allocation vs frozen first absorption; not equal transport models",
    })
    return True


def main():
    """Generate figures using saved arrays only."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlap-only", action="store_true")
    args = parser.parse_args()
    _style()
    if not args.overlap_only:
        geometry_checks()
    available = overlap_maps()
    print(json.dumps({"output_directory": str(OUT), "overlap_arrays_available": available}))  # noqa: T201


if __name__ == "__main__":
    main()
