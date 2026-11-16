"""Plot independently stored benchmark metrics and neutral-field slices."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(root: Path) -> list[Path]:
    """Render field slices and epoch diagnostics from a completed benchmark.

    Parameters
    ----------
    root
        Benchmark directory containing report.json and saved predictions.

    Returns
    -------
    list of Path
        Paths of the two generated PNG figures.
    """
    report = json.loads((root / "report.json").read_text())
    if not report["complete"] or not (root / "COMPLETED").is_file():
        raise ValueError("A completed independent benchmark is required")
    methods = {
        "previous": "Previous emulator",
        "selected": "Selected emulator",
        "native_es": "Native ES",
    }
    colours = {"previous": "#d77b20", "selected": "#1769aa", "native_es": "#4e8e43"}
    by_z = report["summaries"]["by_redshift"]
    redshifts = np.array(sorted((float(z) for z in by_z), reverse=True))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for method, label in methods.items():
        records = [by_z[f"{z:g}"]["methods"][method] for z in redshifts]
        values = [
            [r["MAE"]["mean"] for r in records],
            [r["absolute_delta_Q_mass"]["mean"] for r in records],
            [
                r["low_k_21cm"]["snapshot_mean_absolute_power_ratio_error"]["mean"]
                for r in records
            ],
        ]
        for axis, points in zip(axes, values, strict=True):
            axis.plot(
                redshifts,
                [np.nan if x is None else x for x in points],
                "o-",
                color=colours[method],
                label=label,
            )
    for axis, label in zip(
        axes,
        [
            r"Neutral-field MAE",
            r"Mean $|\Delta Q_{\rm mass}|$",
            r"Mean $|P/P_{\rm teacher}-1|$",
        ],
        strict=True,
    ):
        axis.set(xlabel="Redshift", ylabel=label)
        axis.invert_xaxis()
        axis.grid(alpha=0.18)
    axes[2].set_yscale("symlog", linthresh=0.1)
    axes[0].legend(fontsize=9)
    fig.suptitle(
        "Frozen-model comparison; equal snapshot weights; power uses individual modes below half Nyquist"
    )
    history = root / "diagnostics.png"
    fig.savefig(history, dpi=180)
    plt.close(fig)

    seed = min(row["seed"] for row in report["snapshots"])
    available = [row for row in report["snapshots"] if row["seed"] == seed]
    chosen = []
    for target in (8, 7.5, 6):
        row = min(available, key=lambda r: abs(r["redshift"] - target))
        if row not in chosen:
            chosen.append(row)
    fig, axes = plt.subplots(
        len(chosen),
        4,
        figsize=(12, 3.2 * len(chosen)),
        squeeze=False,
        layout="constrained",
    )
    images = []
    for row in chosen:
        path = root / row["files"]["predictions.npz"]["path"]
        with np.load(path, allow_pickle=False) as archive:
            fields = {
                name: archive[f"{name}_xHI"]
                for name in ("teacher", "previous", "selected")
            }
        images.append((row, fields))
    maximum = max(
        float(np.max(np.abs(fields["selected"] - fields["teacher"])))
        for _, fields in images
    )
    maximum = max(maximum, 1e-6)
    for row_axes, (row, fields) in zip(axes, images, strict=True):
        length = row["box_len_cMpc"]
        index = fields["teacher"].shape[0] // 2
        for axis, name, label in zip(
            row_axes[:3],
            ("teacher", "previous", "selected"),
            ("Analytic teacher", "Previous emulator", "Selected emulator"),
            strict=True,
        ):
            neutral_image = axis.imshow(
                fields[name][:, :, index].T,
                origin="lower",
                extent=(0, length, 0, length),
                vmin=0,
                vmax=1,
                cmap="gray",
            )
            axis.set_title(f"{label}\nz={row['redshift']:g}")
        difference_image = row_axes[3].imshow(
            (fields["selected"] - fields["teacher"])[:, :, index].T,
            origin="lower",
            extent=(0, length, 0, length),
            cmap="RdBu_r",
            vmin=-maximum,
            vmax=maximum,
        )
        row_axes[3].set_title("Selected - teacher")
        for axis in row_axes:
            axis.set(xlabel="x [cMpc]", ylabel="y [cMpc]")
    fig.colorbar(
        neutral_image,
        ax=axes[:, :3].ravel().tolist(),
        shrink=0.75,
        label=r"$x_{\rm HI}$",
    )
    fig.colorbar(
        difference_image,
        ax=axes[:, 3].ravel().tolist(),
        shrink=0.75,
        label=r"$\Delta x_{\rm HI}$",
    )
    fig.suptitle(f"Independent seed {seed}; central cell slice; common colour scales")
    slices = root / "fields.png"
    fig.savefig(slices, dpi=180)
    plt.close(fig)
    return [history, slices]


def main() -> None:
    """Render a completed benchmark without modifying stored scientific data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    for path in plot(args.root):
        sys.stdout.write(f"{path}\n")


if __name__ == "__main__":
    main()
