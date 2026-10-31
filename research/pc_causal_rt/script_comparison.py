"""Compare native-budget RT with the original SCRIPT reference.

SCRIPT remains an independent coeval allocation model. The comparisons do not
certify the RT field or identify either algorithm as an exact physical truth.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research.pc_causal_rt.report import _audit_case, _digest, _load, _primary_band, _sum
from research.pc_field_emulator.evaluation import shell_spectra


BUDGET_ROUNDOFF_TOLERANCE = 1e-6
TARGETS = (.8, .6, .4, .2)


def _checked_files(folder, receipt, root, required=()):
    """Check receipt paths within the containing experiment, including ../ paths."""
    root = Path(root).resolve()
    if any(name not in receipt["files"] for name in required):
        raise ValueError("Comparison receipt is missing a required artifact")
    for name, expected in receipt["files"].items():
        path = (folder / name).resolve()
        if not path.is_relative_to(root) or _digest(path) != expected:
            raise ValueError(f"Comparison archive checksum failed: {path}")


def _script_frames(root):
    completion = json.loads((root / "completion.json").read_text())
    _checked_files(root, completion, root, ("script_report.json",))
    report = json.loads((root / "script_report.json").read_text())
    if completion["identity"] != report["identity"] or not report["all_script_ledgers_passed"]:
        raise ValueError("SCRIPT completion identity or ledger status differs")
    seeds = {}
    for seed in report["seeds"]:
        number = seed["seed"]
        if number in seeds:
            raise ValueError("Duplicate SCRIPT seed")
        frames = {}
        for row in seed["snapshots"]:
            node = row["node"]
            if type(node) is not int or node < 0 or node in frames:
                raise ValueError("Invalid or duplicate SCRIPT checkpoint node")
            folder = root / f"seed{number}" / f"node{node:05d}"
            receipt = json.loads((folder / "receipt.json").read_text())
            _checked_files(folder, receipt, root, ("fields.npz", "metrics.json"))
            identity = receipt["identity"]
            if (identity["history_sha256"] != seed["history_sha256"]
                    or identity["reference_identity"] != report["identity"]
                    or identity["native_record"]["seed"] != number
                    or identity["native_record"]["redshift"] != row["redshift"]
                    or json.loads((folder / "metrics.json").read_text()) != row):
                raise ValueError("SCRIPT snapshot provenance differs from report")
            fields = _load(folder / "fields.npz")
            capacity = fields["capacity"]
            if capacity.ndim != 3 or len(set(capacity.shape)) != 1 or len(capacity) < 2:
                raise ValueError("SCRIPT capacity must be cubic")
            for name in ("source", "capacity", "native_es_xhi", "native_brightness",
                         "script_xhi", "script_brightness"):
                if fields[name].shape != capacity.shape or not np.isfinite(fields[name]).all():
                    raise ValueError(f"Invalid SCRIPT reference field: {name}")
            if ((capacity <= 0).any() or (fields["source"] < 0).any()
                    or (fields["script_xhi"] < 0).any() or (fields["script_xhi"] > 1).any()
                    or (fields["native_es_xhi"] < 0).any() or (fields["native_es_xhi"] > 1).any()):
                raise ValueError("SCRIPT reference counts or fractions violate bounds")
            source, atoms = _sum(fields["source"]), _sum(capacity)
            absorbed = _sum(capacity * (1 - fields["script_xhi"]))
            surplus = max(source - atoms, np.longdouble(0))
            residual = abs(source - absorbed - surplus) / max(source, absorbed + surplus,
                                                              np.longdouble(np.finfo(float).tiny))
            if residual > 2e-5 or row["ledger"].get("passed") is not True:
                raise ValueError("SCRIPT saved-field photon ledger failed")
            frames[node] = (row, fields)
        seeds[number] = (seed["history_sha256"], frames)
    return report, seeds


def _spectra(script, rt, box_len, native=None):
    fields = {"teacher": script, "rt": rt}
    if native is not None:
        fields["native_es"] = native
    result = shell_spectra(fields, box_len)
    result = {key.replace("teacher", "script"): value for key, value in result.items()}
    result["ratio_definition"] = "P_comparison / P_SCRIPT; null where SCRIPT reference power is undefined"
    return result


def _interpolate(frames, target, *, xhi_name, brightness_name):
    """First saved descending neutral-fraction bracket; no extrapolation."""
    for index, (row, fields) in enumerate(frames):
        mean = float(fields[xhi_name].mean(dtype=float))
        if abs(mean - target) <= 2e-14:
            return {"xhi": fields[xhi_name].astype(float),
                    "brightness": fields[brightness_name].astype(float),
                    "interpolation": {"nodes": [row["node"], row["node"]],
                        "redshifts": [row["redshift"], row["redshift"]],
                        "later_weight": 0., "redshift_linear_diagnostic": row["redshift"]}}
        if index == 0:
            continue
        previous, old = frames[index - 1]
        first = float(old[xhi_name].mean(dtype=float))
        if first > target > mean:
            weight = (first - target) / (first - mean)
            return {"xhi": (1 - weight) * old[xhi_name] + weight * fields[xhi_name],
                    "brightness": (1 - weight) * old[brightness_name] + weight * fields[brightness_name],
                    "interpolation": {"nodes": [previous["node"], row["node"]],
                        "redshifts": [previous["redshift"], row["redshift"]],
                        "later_weight": weight,
                        "redshift_linear_diagnostic": (1 - weight) * previous["redshift"] + weight * row["redshift"]}}
    return None


def _matched(rt_frames, script_frames, box_len):
    rows = []
    for target in TARGETS:
        rt = _interpolate(rt_frames, target, xhi_name="rt_xhi", brightness_name="rt_brightness")
        script = _interpolate(script_frames, target, xhi_name="script_xhi", brightness_name="script_brightness")
        row = {"target_xHI_volume": target, "available": rt is not None and script is not None}
        if row["available"]:
            row.update(
                field_MAE=float(np.mean(np.abs(rt["xhi"] - script["xhi"]))),
                rt_interpolation=rt["interpolation"], script_interpolation=script["interpolation"],
                rt_xHI_volume=float(rt["xhi"].mean()), script_xHI_volume=float(script["xhi"].mean()),
                brightness_spectra=_spectra(script["brightness"], rt["brightness"], box_len),
                neutral_fraction_spectra=_spectra(script["xhi"], rt["xhi"], box_len),
            )
        else:
            row["unavailable_histories"] = [name for name, item in (("rt", rt), ("script", script)) if item is None]
        rows.append(row)
    return rows


def _same_time(rt_point, rt, script_point, script, case, box_len):
    if (rt_point["node"] != script_point["node"]
            or rt_point["redshift"] != script_point["redshift"]):
        raise ValueError("Same-time comparison has different nodes/redshifts")
    if (rt["capacity"].shape != script["capacity"].shape
            or np.max(np.abs(rt["capacity"] - script["capacity"])) > 3e-6
            or not np.array_equal(rt["native_xhi"], script["native_es_xhi"])
            or not np.array_equal(rt["native_brightness"], script["native_brightness"])):
        raise ValueError("RT and SCRIPT do not share original native fields")
    native_source = _sum(script["source"])
    emitted = _sum(rt["cumulative_source"])
    difference = float(abs(emitted - native_source) / max(native_source, np.longdouble(np.finfo(float).tiny)))
    shared = difference <= BUDGET_ROUNDOFF_TOLERANCE
    if case["setting"]["source_stride"] == 1 and not shared:
        raise ValueError("Corrected RT emitted budget differs from native coeval SCRIPT budget")
    current_difference = None
    if "source_native_current" in rt:
        current_difference = float(_sum(np.abs(rt["source_native_current"] - script["source"]))
                                   / max(native_source, np.longdouble(np.finfo(float).tiny)))
        if current_difference > BUDGET_ROUNDOFF_TOLERANCE:
            raise ValueError("Native current-source reconstruction differs from SCRIPT input")
    atoms_rt, atoms_script = _sum(rt["capacity"]), _sum(script["capacity"])
    spectra = _spectra(script["script_brightness"], rt["rt_brightness"], box_len,
                       script["native_brightness"])
    return {
        "node": rt_point["node"], "redshift": rt_point["redshift"],
        "rt_xHI_volume": float(rt["rt_xhi"].mean()),
        "script_xHI_volume": float(script["script_xhi"].mean()),
        "native_es_xHI_volume": float(script["native_es_xhi"].mean(dtype=float)),
        "rt_xHI_mass": float(_sum(rt["capacity"] * rt["rt_xhi"]) / atoms_rt),
        "script_xHI_mass": float(_sum(script["capacity"] * script["script_xhi"]) / atoms_script),
        "native_es_xHI_mass": float(_sum(script["capacity"] * script["native_es_xhi"]) / atoms_script),
        "rt_radiation_per_atom": float(_sum(rt["radiation"]) / atoms_rt),
        "rt_emitted_per_atom": float(emitted / atoms_rt),
        "native_source_per_atom": float(native_source / atoms_script),
        "source_budget_shared_within_native_roundoff": bool(shared),
        "source_budget_relative_difference": difference,
        "current_native_source_L1_relative_difference": current_difference,
        "field_MAE_rt_vs_script": float(np.mean(np.abs(rt["rt_xhi"] - script["script_xhi"]))),
        "field_MAE_rt_vs_native_es": float(np.mean(np.abs(rt["rt_xhi"] - script["native_es_xhi"]))),
        "brightness_spectra": spectra,
        "primary_power_band": _primary_band(spectra, len(rt["capacity"]), box_len),
    }


def _plot(root, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seeds = sorted({case["seed"] for case in report["cases"]})
    fig, axes = plt.subplots(len(seeds), 3, figsize=(14, 4 * len(seeds)), squeeze=False)
    for panels, seed in zip(axes, seeds, strict=True):
        cases = [case for case in report["cases"] if case["seed"] == seed]
        for index, case in enumerate(cases):
            points = case["same_time"]
            z = [row["redshift"] for row in points]
            setting = case["setting"]
            label = f"RT D{setting['directions']} f{setting['refinement']} CFL{setting['cfl']} t{setting['time_substeps']} s{setting['source_stride']}"
            line, = panels[0].plot(z, [row["rt_xHI_mass"] for row in points], label=label)
            color = line.get_color()
            if index == 0:
                panels[0].plot(z, [row["script_xHI_mass"] for row in points], "k--", label="Original SCRIPT")
                panels[0].plot(z, [row["native_es_xHI_mass"] for row in points], color="gray", linestyle=":", label="Native ES")
            panels[1].plot(z, [row["field_MAE_rt_vs_script"] for row in points], color=color, label=label)
            # Show the first primary shell with its actual centre; all shells remain in JSON.
            ratios, ks = [], []
            for row in points:
                spectrum = row["brightness_spectra"]
                tested = row["primary_power_band"]["shell_centres_cMpc_inv"]
                k = tested[0] if tested else None
                j = spectrum["k_cMpc_inv"].index(k) if k is not None else None
                ratio = spectrum["ratio_rt"][j] if j is not None else None
                ratios.append(np.nan if ratio is None else ratio)
                ks.append(k)
            panels[2].plot(z, ratios, color=color, label=label)
            if ks and ks[0] is not None:
                panels[2].set_title(f"First primary shell k={ks[0]:.4g} cMpc⁻¹")
        for panel, ylabel in zip(panels, ("Mass-weighted neutral fraction", "Neutral-field MAE: RT vs SCRIPT", "P_RT / P_SCRIPT")):
            panel.set(xlabel="Redshift", ylabel=ylabel)
            panel.invert_xaxis()
            panel.legend(fontsize=6)
            panel.grid(alpha=.2)
        panels[0].set_title(f"Seed {seed}: native cumulative sources")
        panels[2].axhline(1, color="gray", linewidth=.7)
    fig.suptitle("21cmFAST physical inputs; SCRIPT is an independent coeval reference")
    fig.tight_layout()
    fig.savefig(root / "budget_script_comparison.png", dpi=170)
    plt.close(fig)


def build_comparison(rt_root, script_root, *, make_plot=True):
    """Audit and compare completed corrected-RT cases and SCRIPT snapshots."""
    rt_root, script_root = Path(rt_root).resolve(), Path(script_root).resolve()
    summary = json.loads((rt_root / "summary.json").read_text())
    script_report, script_seeds = _script_frames(script_root)
    box_len = summary["configuration"]["box_len"]
    report = {
        "schema": 1, "physical_model_authority": "21cmFAST",
        "reference": "original SCRIPT allocation on archived native coeval sources",
        "comparison_is_truth_error": False, "convergence_certified": False,
        "source_budget_relative_roundoff_tolerance": BUDGET_ROUNDOFF_TOLERANCE,
        "source_scope": "RT retains moving emitters, finite light speed and radiation in flight; SCRIPT allocates a coeval cumulative budget",
        "geometry_scope": "SCRIPT uses full periodic redistribution; native ES retains its filter and R_BUBBLE_MAX",
        "matched_xHI_scope": "First saved bracket at the requested volume-neutral fraction; convex interpolation of xHI and native brightness fields before recomputing spectra. Different epochs and evolving density remain; no extrapolation or exact-RT claim.",
        "script_report_sha256": _digest(script_root / "script_report.json"),
        "script_provenance": script_report["identity"]["script_provenance"],
        "cases": [],
    }
    for case in summary["cases"]:
        if case.get("source_mode") != "native_cumulative":
            raise ValueError("This comparison requires native-cumulative RT cases")
        folder = rt_root / "cases" / case["case_id"]
        receipt = json.loads((folder / "receipt.json").read_text())
        _checked_files(folder, receipt, rt_root, ("case.json",))
        if receipt["record"] != case or json.loads((folder / "case.json").read_text()) != case:
            raise ValueError("RT case differs from its receipt")
        _, rt_frames = _audit_case(rt_root, case, box_len)
        if case["seed"] not in script_seeds:
            raise ValueError("SCRIPT reference is missing an RT seed")
        history_sha, script_frames = script_seeds[case["seed"]]
        if history_sha != case["history_sha256"]:
            raise ValueError("RT and SCRIPT use different native source histories")
        common = sorted(set(rt_frames) & set(script_frames))
        if not common:
            raise ValueError("RT and SCRIPT have no shared saved checkpoints")
        comparisons = [_same_time(*rt_frames[node], *script_frames[node], case, box_len) for node in common]
        report["cases"].append({"case_id": case["case_id"], "seed": case["seed"],
            "setting": case["setting"], "history_sha256": history_sha,
            "shared_saved_checkpoints": len(common),
            "rt_nodes_without_script_checkpoint": sorted(set(rt_frames) - set(script_frames)),
            "script_nodes_without_rt_checkpoint": sorted(set(script_frames) - set(rt_frames)),
            "same_time": comparisons,
            "matched_mean_xHI": _matched([rt_frames[node] for node in sorted(rt_frames)],
                                           [script_frames[node] for node in sorted(script_frames)], box_len),
            "all_same_time_budgets_shared": all(row["source_budget_shared_within_native_roundoff"] for row in comparisons)})
    if not report["cases"]:
        raise ValueError("No completed corrected-RT cases to compare")
    report["all_same_time_budgets_shared"] = all(case["all_same_time_budgets_shared"] for case in report["cases"])
    if make_plot:
        _plot(rt_root, report)
    temporary = rt_root / "budget_script_comparison.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(rt_root / "budget_script_comparison.json")
    return report
