"""Plot frozen blind comparisons across the four prescribed geometries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROFILES = ("original", "resolution", "volume", "unseen_box")


def plot(root: Path, output: Path) -> Path:
    """Render completed benchmark summaries without recomputing predictions.

    Parameters
    ----------
    root
        Campaign directory containing benchmark_blind_* subdirectories.
    output
        Destination PNG filename.

    Returns
    -------
    Path
        Written comparison figure.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), layout="constrained")
    styles = {"previous": ("Previous", "#ce7b25"), "selected": ("Updated", "#1769aa")}
    for column, profile in enumerate(PROFILES):
        folder = root / f"benchmark_blind_{profile}"
        report = json.loads((folder / "report.json").read_text())
        if not report["complete"] or not (folder / "COMPLETED").is_file():
            raise ValueError("Every frozen benchmark must be complete")
        row = report["snapshots"][0]
        records = report["summaries"]["by_redshift"]
        redshifts = sorted((float(z) for z in records), reverse=True)
        for method, (label, colour) in styles.items():
            values = [records[f"{z:g}"]["methods"][method] for z in redshifts]
            mae = [value["MAE"]["mean"] for value in values]
            power = [
                value["low_k_21cm"]["snapshot_mean_absolute_power_ratio_error"]["mean"]
                for value in values
            ]
            for axis, points in zip(axes[:, column], (mae, power), strict=True):
                axis.plot(
                    redshifts,
                    [np.nan if value is None else value for value in points],
                    "o-",
                    color=colour,
                    label=label,
                )
        length = row["box_len_cMpc"]
        dimension = row["dimension"]
        heading = f"L={length:g} cMpc, N={dimension}"
        if profile == "unseen_box":
            heading += "\nUnseen box size"
        axes[0, column].set_title(heading)
        for axis in axes[:, column]:
            axis.set_xlim(max(redshifts), min(redshifts))
            axis.grid(alpha=0.2)
            axis.set_xlabel("Redshift")
        axes[1, column].set_yscale("symlog", linthresh=0.1)
        axes[1, column].set_ylim(bottom=0)
    axes[0, 0].set_ylabel(r"Neutral-field MAE")
    axes[1, 0].set_ylabel(r"Mean $|P/P_{\rm teacher}-1|$")
    axes[0, 0].legend()
    fig.suptitle(
        "Fresh seeds; weights frozen before evaluation; two realizations per geometry\n"
        "Power uses modes below half the native grid Nyquist; undefined ratios omitted"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def main() -> None:
    """Plot the completed four-profile campaign."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plot(args.root, args.output)


if __name__ == "__main__":
    main()
