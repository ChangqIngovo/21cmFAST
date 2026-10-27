"""Measure physical-control responses separately from native numerical changes.

This is a small-box, fixed-IC diagnostic. Its controls are not astrophysical
parameter priors, and its refinement envelope is measured at the fiducial
point only. A weak response does not establish that a physical input can be
removed from a field emulator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

PARAMETERS = (
    "log10_source_amplitude", "history_tilt", "source_concentration",
    "density_contrast_scale", "log10_Tbb_K", "log10_clumping",
)
LABELS = (
    "Escaped photon amplitude", "Emission-time tilt", "Source concentration",
    "Gas-density contrast", "Blackbody temperature", "Recombination clumping",
)
REFINEMENTS = {
    "convergence_angle4": "Angular: order 2 to 4",
    "convergence_time2": "Time: one to two substeps",
    "convergence_space5": "Space: RT 24 to 40 cells/axis",
}
REFINEMENT_CONFIGS = {
    "convergence_angle4": {"refinement": 3, "angular_order": 4, "time_substeps": 1},
    "convergence_time2": {"refinement": 3, "angular_order": 2, "time_substeps": 2},
    "convergence_space5": {"refinement": 5, "angular_order": 2, "time_substeps": 1},
}


def _finite_array(value, shape, name):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"Invalid {name}: expected finite {shape}")
    return array


def summarize(cases, *, dimension=8, box_len=50., allow_partial=False):
    """Compare matched snapshots/modes and retain both signs of each response."""
    required = {"fiducial", *REFINEMENTS,
                *(f"oat_{j}_{side}" for j in range(6) for side in (0, 1))}
    missing = sorted(required - set(cases))
    if "fiducial" not in cases or (missing and not allow_partial):
        raise ValueError(f"Missing completed control cases: {missing}")
    fid = cases["fiducial"]
    z = np.asarray(fid["redshifts"], dtype=float)
    k = np.asarray(fid["k"], dtype=float)
    if (z.ndim != 1 or k.ndim != 1 or len(z) < 3 or not np.isfinite(z).all()
            or not np.isfinite(k).all() or not np.all(np.diff(z) < 0)
            or not np.all(np.diff(k) > 0)):
        raise ValueError("Control axes require descending redshift and ascending finite k")
    selected_z = np.arange(len(z) - 3, len(z))
    selected_k = np.flatnonzero((k > 0) & (k <= .6 * np.pi * dimension / box_len))
    if len(selected_k) != 2:
        raise ValueError("This pilot requires exactly two shells below 0.6 Nyquist")
    base_power = _finite_array(fid["P21"], (len(z), len(k)), "fiducial power")
    base_xhi = _finite_array(fid["xHI_mean"], (len(z),), "fiducial neutral fraction")
    reference = base_power[np.ix_(selected_z, selected_k)]
    if np.any(reference <= 0):
        raise ValueError("Relative power requires strictly positive fiducial power")
    theta = _finite_array(fid["theta"], (6,), "fiducial controls")
    if any(fid[key] != value for key, value in
           {"refinement": 3, "angular_order": 2, "time_substeps": 1}.items()):
        raise ValueError("Unexpected fiducial numerical configuration")

    def response(row):
        if (row["seed"] != fid["seed"]
                or not np.array_equal(row["redshifts"], z)
                or not np.array_equal(row["k"], k)):
            raise ValueError("Controlled comparisons require identical seed/redshift/k")
        power = _finite_array(row["P21"], base_power.shape, "power")
        xhi = _finite_array(row["xHI_mean"], base_xhi.shape, "neutral fraction")
        relative = power[np.ix_(selected_z, selected_k)] / reference - 1
        neutral = xhi[selected_z] - base_xhi[selected_z]
        return {
            "case_id": row["case_id"], "theta": row["theta"],
            "power_relative_signed": relative.tolist(),
            "power_relative_mean_absolute": float(np.abs(relative).mean()),
            "power_relative_maximum_absolute": float(np.abs(relative).max()),
            "power_relative_final_mean_absolute": float(np.abs(relative[-1]).mean()),
            "power_relative_final_maximum_absolute": float(np.abs(relative[-1]).max()),
            "xHI_difference_signed": neutral.tolist(),
            "xHI_difference_mean_absolute": float(np.abs(neutral).mean()),
            "xHI_difference_maximum_absolute": float(np.abs(neutral).max()),
            "xHI_final": float(xhi[-1]),
            "xHI_final_difference": float(neutral[-1]),
            "P21_selected": power[np.ix_(selected_z, selected_k)].tolist(),
        }

    numerical = {}
    for name in REFINEMENTS:
        if name in cases:
            if not np.array_equal(cases[name]["theta"], theta):
                raise ValueError("Refinement control changed physical parameters")
            if any(cases[name][key] != value
                   for key, value in REFINEMENT_CONFIGS[name].items()):
                raise ValueError("Unexpected refinement numerical configuration")
            numerical[name] = response(cases[name])
            numerical[name]["numerical_configuration"] = REFINEMENT_CONFIGS[name]
    floor = (np.maximum.reduce([np.abs(v["power_relative_signed"])
                                for v in numerical.values()]) if numerical else None)
    rows = []
    for j, parameter in enumerate(PARAMETERS):
        sides = {}
        for side in (0, 1):
            name = f"oat_{j}_{side}"
            if name not in cases:
                continue
            row_theta = _finite_array(cases[name]["theta"], (6,), "OAT controls")
            if not np.array_equal(np.delete(row_theta, j), np.delete(theta, j)):
                raise ValueError("OAT comparison changes more than its named control")
            if any(cases[name][key] != fid[key] for key in
                   ("refinement", "angular_order", "time_substeps")):
                raise ValueError("OAT comparison changes the numerical configuration")
            sides["low" if side == 0 else "high"] = response(cases[name])
        if not sides:
            continue
        values = np.array([row["power_relative_signed"] for row in sides.values()])
        neutral_values = np.array([row["xHI_difference_signed"] for row in sides.values()])
        row = {"parameter": parameter, "label": LABELS[j], "sides": sides,
               "power_relative_mean_absolute": float(np.abs(values).mean()),
               "power_relative_maximum_absolute": float(np.abs(values).max()),
               "power_relative_final_mean_absolute": float(np.abs(values[:, -1]).mean()),
               "power_relative_final_maximum_absolute": float(np.abs(values[:, -1]).max()),
               "xHI_difference_mean_absolute": float(np.abs(neutral_values).mean()),
               "xHI_difference_maximum_absolute": float(np.abs(neutral_values).max()),
               "xHI_final_difference_maximum_absolute": float(np.abs(neutral_values[:, -1]).max())}
        if floor is not None:
            row["response_exceeds_fiducial_refinement_envelope_fraction"] = float(
                (np.abs(values) > floor[None]).mean())
        rows.append(row)
    return {
        "schema_version": 1,
        "scope": "fixed-IC physical-control sensitivity, not an astrophysical posterior",
        "power_definition": "linear P21 of saturated-Ts, no-RSD density-times-neutral field",
        "comparison": "same redshift and common mass-restricted 8^3 analysis grid",
        "active_photon_budget_normalized": True,
        "history_control_budget_scope": "Per-source final cumulative photons fixed; earlier snapshot cumulative photons may differ",
        "refinement_envelope_scope": "fiducial only; finite differences, not a certified error bound",
        "weak_response_interpretation": "Does not establish that an input is unnecessary; requires field ablation and broader coverage",
        "missing_cases": missing, "complete": not missing,
        "selected_redshift_indices": selected_z.tolist(),
        "selected_k_indices": selected_k.tolist(),
        "redshifts": z[selected_z].tolist(), "k_cMpc_inv": k[selected_k].tolist(),
        "k_max_fraction_of_nyquist": .6,
        "fiducial": {"theta": theta.tolist(), "seed": fid["seed"],
                     "P21_selected": reference.tolist(),
                     "xHI_mean": base_xhi[selected_z].tolist()},
        "physical_controls": rows, "numerical_controls": numerical,
        "fiducial_refinement_envelope": None if floor is None else floor.tolist(),
    }


def _control_value(index, theta):
    value = theta[index]
    if index in (0, 4, 5):
        value = 10**value
    return f"{value / 1000:.1f} kK" if index == 4 else f"{value:.3g}"


def plot_summary(summary, output):
    """Render signed physical responses and separate numerical diagnostics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150})
    z = np.asarray(summary["redshifts"])
    k = np.asarray(summary["k_cMpc_inv"])
    floor = summary["fiducial_refinement_envelope"]
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 7.5), sharex=True)
    records = {row["parameter"]: row for row in summary["physical_controls"]}
    for j, ax in enumerate(axes.flat):
        ax.axhline(0, color="0.35", lw=.8)
        if floor is not None:
            band = 100 * np.max(np.asarray(floor), axis=1)
            ax.fill_between(z, -band, band, color="0.65", alpha=.25, zorder=0)
        row = records.get(PARAMETERS[j])
        if row is not None:
            for side, color in (("low", "#2166ac"), ("high", "#b35806")):
                if side not in row["sides"]:
                    continue
                values = 100 * np.asarray(row["sides"][side]["power_relative_signed"])
                for ik, style in enumerate(("-", "--")):
                    ax.plot(z, values[:, ik], color=color, linestyle=style, marker="o", ms=3.5)
            levels = [f'{side}={_control_value(j, row["sides"][side]["theta"])}'
                      for side in ("low", "high") if side in row["sides"]]
            ax.set_title(LABELS[j] + "\n" + "; ".join(levels), fontsize=11)
        else:
            ax.set_title(LABELS[j] + "\nPending")
        ax.set_xticks(z)
        ax.set_xlim(z[0] + .15, z[-1] - .15)
        ax.grid(alpha=.18)
        ax.set_xlabel("Redshift")
        ax.set_ylabel(r"$100(P_{21}/P_{21,\mathrm{fid}}-1)$ [%]")
    legend = [Line2D([], [], color="#2166ac", label="Lower control value"),
              Line2D([], [], color="#b35806", label="Higher control value"),
              *[Line2D([], [], color="0.25", linestyle=style,
                       label=rf"$k={value:.3f}\,\mathrm{{cMpc}}^{{-1}}$")
                for value, style in zip(k, ("-", "--"), strict=True)]]
    if floor is not None:
        legend.append(Patch(facecolor="0.65", alpha=.25,
                            label="Fiducial refinement envelope (both k bins)"))
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Controlled physical responses at fixed IC and redshift", fontsize=15)
    fig.text(.5, .925, "Actual native RT; active transported photons normalized; physical controls, not astrophysical priors",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .10, 1, .91))
    for extension in ("png", "pdf"):
        fig.savefig(output / f"physical_sensitivity.{extension}")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.1), sharex=True)
    for ax, (name, title) in zip(axes, REFINEMENTS.items(), strict=True):
        row = summary["numerical_controls"].get(name)
        if row is not None:
            values = 100 * np.asarray(row["power_relative_signed"])
            for ik, (color, style) in enumerate((("#2166ac", "-"), ("#b35806", "--"))):
                ax.plot(z, values[:, ik], color=color, linestyle=style, marker="o",
                        label=rf"$k={k[ik]:.3f}$")
        else:
            ax.text(.5, .5, "Pending", ha="center", transform=ax.transAxes)
        ax.axhline(0, color="0.3", lw=.7)
        ax.axhspan(-1, 1, color="0.7", alpha=.18)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Redshift")
        ax.set_ylabel(r"$100(P_{21}/P_{21,\mathrm{fid}}-1)$ [%]")
        ax.set_xticks(z)
        ax.set_xlim(z[0] + .15, z[-1] - .15)
        ax.grid(alpha=.18)
        if row is not None:
            ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Numerical refinement differences at the fiducial point only", fontsize=14)
    fig.text(.5, .025, "Common 8³ analysis grid; grey band is ±1%. These changes are not certified error bounds.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .06, 1, .93))
    for extension in ("png", "pdf"):
        fig.savefig(output / f"numerical_refinement.{extension}")
    plt.close(fig)


def analyze(root, output, *, allow_partial=False):
    """Load completed native cases with hash checks and write fresh diagnostics."""
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh analysis output directory")
    design = json.loads((root / "design.json").read_text())
    cases, provenance = {}, {}
    for row in design["cases"]:
        if row["split"] not in ("control", "convergence"):
            continue
        folder = root / "cases" / row["case_id"]
        if not (folder / "COMPLETED").exists():
            continue
        result_path = folder / "result.json"
        result = json.loads(result_path.read_text())
        if any(result.get(key) != value for key, value in row.items()):
            raise ValueError("Completed control differs from the declared design")
        fields = folder / "fields.npz"
        actual_hash = hashlib.sha256(fields.read_bytes()).hexdigest()
        if actual_hash != result["fields_sha256"]:
            raise ValueError("Completed physical fields hash mismatch")
        with np.load(fields, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata.get("input_photons_are_active_transport_band") is not True:
                raise ValueError("SED controls require actual transported-photon normalization")
            desired_photons = float(archive["photons"][-1].sum())
        manifest = json.loads((folder / "input_manifest.json").read_text())
        actual_photons = (manifest["total_emitted_photons"]
                          * result["sed_transport_photons_over_nominal"]
                          / manifest["mean_native_cell_nuclei"])
        if desired_photons <= 0 or not np.isclose(desired_photons, actual_photons, rtol=5e-12):
            raise ValueError("Measured transported photons do not match desired control budget")
        cases[row["case_id"]] = result
        provenance[row["case_id"]] = {
            "fields_sha256": actual_hash,
            "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
            "active_over_desired_total_photons": actual_photons / desired_photons,
        }
    summary = summarize(cases, dimension=design["metadata"]["dimension"],
                        box_len=design["metadata"]["box_len_cMpc"], allow_partial=allow_partial)
    summary["provenance"] = provenance
    summary["design_sha256"] = hashlib.sha256((root / "design.json").read_bytes()).hexdigest()
    summary["analysis_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    (output / "sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    plot_summary(summary, output)
    return summary


def main():
    """CLI for immutable sensitivity plots and machine-readable measurements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    result = analyze(args.root, args.output, allow_partial=args.allow_partial)
    print(json.dumps({"complete": result["complete"], "missing": result["missing_cases"],  # noqa: T201
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
