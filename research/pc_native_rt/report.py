"""Independently audit and plot the native-input snapshot RT comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from research.pc_field_emulator.evaluation import shell_spectra


def _sum(array):
    return np.asarray(array).sum(dtype=np.longdouble)


def _hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _read(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: np.array(data[name]) for name in data.files}


def audit_case(root, row):
    """Recompute counts from saved fields, without using solver ledger totals."""
    arrays = _read(root / row["fields_path"])
    source, capacity = arrays["source"], arrays["capacity"]
    ions, radiation = arrays["ionized"], arrays["radiation"]
    for name in ("source", "capacity", "ionized", "radiation"):
        value = arrays[name]
        if value.shape != source.shape or not np.isfinite(value).all() or (value < 0).any():
            raise ValueError(f"Invalid saved {name}: {row['case_id']}")
    if np.any(ions - capacity > 2e-12 * np.maximum(capacity, 1e-300)):
        raise ValueError("Saved ions exceed native atom capacity")
    if (_hash(source) != row["native"]["source_sha256"]
            or _hash(capacity) != row["native"]["capacity_sha256"]):
        raise ValueError("Saved RT input differs from original native source/density")
    native = _read(root / row["native"]["fields_path"])
    for name in ("source", "capacity", "native_es_xhi", "native_brightness"):
        if not np.array_equal(arrays[name], native[name]):
            raise ValueError(f"Original native {name} changed")
    total_source, total_atoms = _sum(source), _sum(capacity)
    total_ions, total_radiation = _sum(ions), _sum(radiation)
    residual = abs(total_source - total_ions - total_radiation)
    residual /= max(total_source, np.longdouble(np.finfo(float).tiny))
    if residual > 2e-10:
        raise ValueError("Saved coarse-grid photon ledger fails independently")
    rt_xhi = np.divide(capacity - ions, capacity,
                        out=np.ones_like(capacity), where=capacity > 0)
    if np.max(np.abs(rt_xhi - arrays["rt_xhi"])) > 2e-7:
        raise ValueError("Stored RT neutral fraction disagrees with ion counts")
    native_ions = _sum(capacity * (1 - arrays["native_es_xhi"].astype(float)))
    spectra = shell_spectra(
        {"teacher": arrays["rt_brightness"], "native_es": arrays["native_brightness"]},
        float(row["box_len_cMpc"]),
    )
    return {
        "case_id": row["case_id"], "seed": row["seed"],
        "redshift": row["redshift"], "setting": row["setting"],
        "native_inputs_identical": True, "saved_field_ledger_passed": True,
        "saved_field_ledger_relative_residual": float(residual),
        "source_per_atom": float(total_source / total_atoms),
        "rt_ions_per_atom": float(total_ions / total_atoms),
        "rt_radiation_per_atom": float(total_radiation / total_atoms),
        "native_es_ions_per_atom": float(native_ions / total_atoms),
        "native_es_unassigned_per_atom": float((total_source - native_ions) / total_atoms),
        "rt_xHI_volume": float(rt_xhi.mean()),
        "native_es_xHI_volume": float(arrays["native_es_xhi"].mean(dtype=float)),
        "rt_xHI_mass": float((total_atoms - total_ions) / total_atoms),
        "native_es_xHI_mass": float((total_atoms - native_ions) / total_atoms),
        "field_MAE": float(np.mean(np.abs(rt_xhi - arrays["native_es_xhi"]))),
        "spectra": spectra,
    }


def _setting_key(row):
    return json.dumps(row["setting"], sort_keys=True)


def _plot(root, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seeds = sorted({row["seed"] for row in report["cases"]})
    figure, axes = plt.subplots(len(seeds), 3, figsize=(13, 3.5 * len(seeds)), squeeze=False)
    for axis, seed in zip(axes, seeds, strict=True):
        rows = [row for row in report["cases"] if row["seed"] == seed]
        settings = sorted({_setting_key(row) for row in rows})
        for i, key in enumerate(settings):
            selected = sorted([row for row in rows if _setting_key(row) == key],
                              key=lambda row: -row["redshift"])
            setting = selected[0]["setting"]
            label = f"RT D={setting['directions']}, f={setting['refinement']}, CFL={setting['cfl']}"
            z = [row["redshift"] for row in selected]
            color = f"C{i % 10}"
            axis[0].plot(z, [row["rt_xHI_volume"] for row in selected], "o-", color=color, label=label)
            axis[1].plot(z, [row["rt_xHI_mass"] for row in selected], "o-", color=color, label=label)
            if i == 0:
                axis[0].plot(z, [row["native_es_xHI_volume"] for row in selected], "s--", color="black", label="Original native ES")
                axis[1].plot(z, [row["native_es_xHI_mass"] for row in selected], "s--", color="black", label="Original native ES")
                axis[1].plot(z, [max(0., 1 - row["source_per_atom"]) for row in selected], ":", color="0.5", label="Fully absorbed budget")
            stage = min(selected, key=lambda row: abs(row["native_es_xHI_volume"] - .5))
            spectrum = stage["spectra"]
            axis[2].plot(spectrum["k_cMpc_inv"], spectrum["Delta2_teacher"], "o-", color=color, label=label)
            if i == 0:
                axis[2].plot(spectrum["k_cMpc_inv"], spectrum["Delta2_native_es"], "s--", color="black", label="Original native ES")
                axis[2].set_title(f"Seed {seed}, z={stage['redshift']:g}")
        for panel in axis[:2]:
            panel.set_xlabel("Snapshot redshift")
            panel.invert_xaxis()
            panel.set_ylim(-.02, 1.02)
            panel.set_title(f"Seed {seed}")
        axis[0].set_ylabel("Volume-mean neutral fraction")
        axis[1].set_ylabel("Mass-mean neutral fraction")
        axis[2].set(xlabel=r"$k$ [cMpc$^{-1}$]", ylabel=r"$\Delta^2_{21}$ [mK$^2$]")
        for panel in axis:
            panel.grid(alpha=.2)
            panel.legend(fontsize=7)
    figure.suptitle("Independent native-input snapshot RT; not a causal cosmic history", fontsize=12)
    figure.tight_layout()
    figure.savefig(root / "native_rt_comparison.png", dpi=160)
    plt.close(figure)


def build_report(root, *, make_plot=True):
    """Check archived fields and preserve unaveraged physical power spectra."""
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text())
    cases = []
    for item in summary["cases"]:
        row = dict(item)
        row["box_len_cMpc"] = summary["configuration"]["box_len"]
        cases.append(audit_case(root, row))
    report = {
        "schema": 1,
        "scope": "Independent native-input snapshot pulse RT, not a causal cosmic history",
        "convergence_certified": False,
        "all_saved_field_ledgers_passed": all(row["saved_field_ledger_passed"] for row in cases),
        "all_native_inputs_identical": all(row["native_inputs_identical"] for row in cases),
        "all_exhaustion_checks_passed": summary["all_exhaustion_checks_passed"],
        "maximum_saved_field_ledger_residual": max((row["saved_field_ledger_relative_residual"] for row in cases), default=0),
        "signal": "Both brightness fields evaluated by the same original native brightness function",
        "cases": cases,
    }
    destination = root / "native_rt_report.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(destination)
    if make_plot:
        _plot(root, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    build_report(parser.parse_args().root)
