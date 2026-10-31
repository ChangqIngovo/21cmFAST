"""Test RT--native ES differences against controlled RT refinements.

The native ES fields are model comparisons, not exact solutions. This analysis
keeps archived physical inputs fixed and reports numerical changes separately
from RT--ES differences; it never infers a true-field error from either one.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .report import (
    AXES, DECREASING, _audit_case, _digest, _history_sources, _pair,
    _primary_band, _setting_key, _sum,
)
from .script_comparison import TARGETS, _checked_files, _interpolate
from research.pc_field_emulator.evaluation import shell_spectra


NUMERICAL_CONFIG = {"settings", "seeds", "max_steps", "checkpoint_stride", "tolerance"}
ANALYSIS_FILES = {"report.py", "script_comparison.py", "hypothesis.py", "archive_run.py",
                  "prepare_run.py", "budget_prepare.py"}


def _scientific_hashes(hashes):
    """Permit report/launcher/test changes, but require identical scientific code."""
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Missing recorded scientific implementation hashes")
    result = {name: value for name, value in hashes.items()
              if Path(name).name not in ANALYSIS_FILES
              and not Path(name).name.startswith("test_")}
    required = {"research/pc_causal_rt/" + name for name in (
        "workflow.py", "source_budget.py", "native_history.py", "gas.py",
        "gas_kernel.cpp", "transport.py", "transport_kernel.cpp")}
    if not required <= result.keys():
        raise ValueError("Incomplete recorded scientific implementation hashes")
    return result


def _input_signature(manifest):
    config = manifest["configuration"]
    return {
        "physical_configuration": {k: v for k, v in config.items() if k not in NUMERICAL_CONFIG},
        "native_runtime": manifest["native_runtime"],
        "scientific_source_hashes": _scientific_hashes(manifest["source_hashes"]),
    }


def _load_root(root):
    root = Path(root).resolve()
    if not (root / "COMPLETED").is_file():
        raise ValueError("A hypothesis input must be a completed RT result root")
    manifest = json.loads((root / "manifest.json").read_text())
    completion = json.loads((root / "completion.json").read_text())
    _checked_files(root, completion, root, ("summary.json", "manifest.json"))
    if completion["identity"] != manifest:
        raise ValueError("RT completion identity differs from its manifest")
    summary = json.loads((root / "summary.json").read_text())
    if summary["configuration"] != manifest["configuration"]:
        raise ValueError("RT summary configuration differs from its manifest")
    cases, frames = [], {}
    for item in summary["cases"]:
        if item.get("source_mode") != "native_cumulative":
            raise ValueError("Hypothesis testing requires native cumulative RT sources")
        if item["case_id"] in frames:
            raise ValueError("Duplicate RT case identity")
        folder = root / "cases" / item["case_id"]
        receipt = json.loads((folder / "receipt.json").read_text())
        _checked_files(folder, receipt, root, ("case.json",))
        if receipt["record"] != item or json.loads((folder / "case.json").read_text()) != item:
            raise ValueError("RT case differs from its receipt")
        expected_identity = {
            "configuration": manifest["configuration"], "native_runtime": manifest["native_runtime"],
            "source_hashes": manifest["source_hashes"],
            **{key: item[key] for key in ("seed", "setting", "history_sha256", "source_budget_sha256", "source_mode")},
        }
        if receipt["identity"] != expected_identity:
            raise ValueError("RT case receipt identity differs from its manifest or record")
        if (item["seed"] not in manifest["configuration"]["seeds"]
                or item["setting"] not in manifest["configuration"]["settings"]
                or manifest["configuration"].get("source_mode") != "native_cumulative"):
            raise ValueError("RT case was not declared in the manifest configuration")
        case, data = _audit_case(root, item, summary["configuration"]["box_len"])
        if not case["all_interval_ledgers_passed"]:
            raise ValueError("RT interval ledgers did not pass")
        case["elapsed_seconds"] = item["elapsed_seconds"]
        case["input_roots"] = [str(root)]
        cases.append(case)
        frames[case["case_id"]] = data
    if not cases:
        raise ValueError("No completed corrected RT cases")
    _history_sources(root, summary, cases, frames)
    if not all(c["native_history_audit"]["passed"] for c in cases):
        raise ValueError("Native history audit did not pass")
    provenance = {"root": str(root), "manifest_sha256": _digest(root / "manifest.json"),
                  "summary_sha256": _digest(root / "summary.json"),
                  "completion_sha256": _digest(root / "completion.json")}
    return manifest, cases, frames, provenance


def _spectra(native, others, box_len):
    result = shell_spectra({"teacher": native, **others}, box_len)
    result = {key.replace("teacher", "native_es"): value for key, value in result.items()}
    result["ratio_definition"] = "P_comparison / P_native_ES; null for undefined native power"
    return result


def _mean_absolute(a, b):
    return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def _relative_change(change, difference):
    # A vanishing reference-model difference cannot normalize numerical changes.
    return float(change / difference) if difference > 1e-14 else None


def _same_time(point, arrays, box_len):
    atoms = _sum(arrays["capacity"])
    emitted = _sum(arrays["cumulative_source"])
    radiation = _sum(arrays["radiation"])
    ions = _sum(arrays["ionized"])
    native_ions = _sum(arrays["capacity"] * (1 - arrays["native_xhi"]))
    rt_mean, native_mean = 1 - ions / atoms, 1 - native_ions / atoms
    gap = rt_mean - native_mean
    inventory, native_deficit = radiation / atoms, (emitted - native_ions) / atoms
    return {
        "node": point["node"], "redshift": point["redshift"],
        "rt_xHI_mass": float(rt_mean), "native_es_xHI_mass": float(native_mean),
        "rt_xHI_volume": float(arrays["rt_xhi"].mean()),
        "native_es_xHI_volume": float(arrays["native_xhi"].mean(dtype=float)),
        "field_MAE_rt_vs_native_es": _mean_absolute(arrays["rt_xhi"], arrays["native_xhi"]),
        "rt_minus_native_es_mass": float(gap),
        "emitted_per_atom": float(emitted / atoms),
        "radiation_per_atom": float(inventory),
        "emitted_minus_native_ions_per_atom": float(native_deficit),
        "history_decomposition_residual": float(gap - (inventory - native_deficit)),
        "brightness_spectra": _spectra(arrays["native_brightness"],
                                        {"rt": arrays["rt_brightness"]}, box_len),
        "primary_power_band": point["primary_power_band"],
    }


def _interpolated(frames, target, native=False):
    return _interpolate([frames[node] for node in sorted(frames)], target,
                        xhi_name="native_xhi" if native else "rt_xhi",
                        brightness_name="native_brightness" if native else "rt_brightness")


def _matched(frames, box_len):
    rows = []
    for target in TARGETS:
        rt, native = _interpolated(frames, target), _interpolated(frames, target, native=True)
        row = {"target_xHI_volume": target, "available": rt is not None and native is not None}
        if row["available"]:
            row.update(field_MAE_rt_vs_native_es=_mean_absolute(rt["xhi"], native["xhi"]),
                       rt_interpolation=rt["interpolation"], native_es_interpolation=native["interpolation"],
                       rt_xHI_volume=float(rt["xhi"].mean()),
                       native_es_xHI_volume=float(native["xhi"].mean()),
                       brightness_spectra=_spectra(native["brightness"], {"rt": rt["brightness"]}, box_len),
                       neutral_fraction_spectra=_spectra(native["xhi"], {"rt": rt["xhi"]}, box_len))
        else:
            row["unavailable_histories"] = [name for name, value in (("rt", rt), ("native_es", native))
                                            if value is None]
        rows.append(row)
    return rows


def _change_metrics(coarse, fine, native, box_len):
    numerical = _mean_absolute(coarse["xhi"], fine["xhi"])
    model_coarse = _mean_absolute(coarse["xhi"], native["xhi"])
    model_fine = _mean_absolute(fine["xhi"], native["xhi"])
    spectrum = _spectra(native["brightness"],
                        {"coarse": coarse["brightness"], "fine": fine["brightness"]}, box_len)
    pc, pf, pe = (np.asarray(spectrum["P_" + key]) for key in ("coarse", "fine", "native_es"))
    floor = max(float(np.max(np.abs(pe))) * 1e-12, np.finfo(float).eps**2 * box_len**3)
    denominator = np.abs(pf - pe)
    power_ratio = [float(abs(a - b) / d) if d > floor else None
                   for a, b, d in zip(pc, pf, denominator, strict=True)]
    return {
        "field_MAE_coarse_vs_fine": numerical,
        "field_MAE_coarse_vs_native_es": model_coarse,
        "field_MAE_fine_vs_native_es": model_fine,
        "field_numerical_change_over_fine_model_difference": _relative_change(numerical, model_fine),
        "power_numerical_change_over_fine_model_difference": power_ratio,
        "power_change_ratio_definition": "abs(P_coarse-P_fine)/abs(P_fine-P_native_ES); null for vanishing denominator",
        "brightness_spectra": spectrum,
    }


def _pair_diagnostics(coarse, fine, frame_map, axis, box_len, tolerance):
    pair = _pair(coarse, fine, frame_map, axis, box_len, tolerance)
    left, right = frame_map[coarse["case_id"]], frame_map[fine["case_id"]]
    if any(not np.array_equal(left[node][1]["native_brightness"], right[node][1]["native_brightness"])
           for node in left):
        raise ValueError("Refinement changed the native brightness comparison field")
    same_time = []
    for point in pair["checkpoints"]:
        node = point["node"]
        a, b = left[node][1], right[node][1]
        row = _change_metrics({"xhi": a["rt_xhi"], "brightness": a["rt_brightness"]},
                              {"xhi": b["rt_xhi"], "brightness": b["rt_brightness"]},
                              {"xhi": b["native_xhi"], "brightness": b["native_brightness"]}, box_len)
        row.update(node=node, redshift=point["redshift"],
                   rt_mass_neutral_change=float(left[node][0]["xHI_mass"] - right[node][0]["xHI_mass"]))
        same_time.append(row)
    matched = []
    for target in TARGETS:
        a, b, es = _interpolated(left, target), _interpolated(right, target), _interpolated(right, target, True)
        row = {"target_xHI_volume": target, "available": all(x is not None for x in (a, b, es))}
        if row["available"]:
            row.update(_change_metrics(a, b, es, box_len), coarse_interpolation=a["interpolation"],
                       fine_interpolation=b["interpolation"], native_es_interpolation=es["interpolation"])
        matched.append(row)
    return {**pair, "same_time_model_comparison": same_time, "matched_mean_xHI": matched,
            "interpretation": "A refinement difference is an observed numerical change, not a bound on true-field error"}


def _adjacent_pairs(cases):
    for seed in sorted({case["seed"] for case in cases}):
        selected = [case for case in cases if case["seed"] == seed]
        for axis in AXES:
            groups = defaultdict(list)
            for case in selected:
                key = tuple((name, case["setting"][name]) for name in AXES if name != axis)
                groups[key].append(case)
            for group in groups.values():
                group.sort(key=lambda case: case["setting"][axis], reverse=axis in DECREASING)
                for coarse, fine in zip(group, group[1:]):
                    yield coarse, fine, axis


def _merge_inputs(loaded):
    signature = _input_signature(loaded[0][0])
    cases, frames, seeds, settings = [], {}, {}, {}
    for manifest, incoming, data, _ in loaded:
        if _input_signature(manifest) != signature:
            raise ValueError("Hypothesis inputs changed physical configuration, native runtime or scientific implementation")
        for case in incoming:
            identity = (case["history_sha256"], case["source_budget_sha256"])
            if case["seed"] in seeds and seeds[case["seed"]] != identity:
                raise ValueError("A seed has different native history or cumulative source archive")
            seeds[case["seed"]] = identity
            key = (case["seed"], _setting_key(case["setting"]))
            if key in settings:
                previous = settings[key]
                old, new = frames[previous["case_id"]], data[case["case_id"]]
                equal = set(old) == set(new) and all(
                    old[node][0]["redshift"] == new[node][0]["redshift"]
                    and old[node][0]["time_seconds"] == new[node][0]["time_seconds"]
                    and old[node][1].keys() == new[node][1].keys()
                    and all(np.array_equal(old[node][1][k], new[node][1][k]) for k in old[node][1])
                    for node in old)
                if not equal:
                    raise ValueError("Repeated numerical setting has different saved fields")
                previous["input_roots"].extend(case["input_roots"])
                continue
            if case["case_id"] in frames:
                raise ValueError("Different cases reuse the same case identifier")
            settings[key] = case
            frames[case["case_id"]] = data[case["case_id"]]
            cases.append(case)
    return signature, cases, frames


def build_hypothesis(input_roots, *, tolerance=.01):
    """Return a read-only, independently audited native-ES/refinement report."""
    roots = [Path(root).resolve() for root in input_roots]
    if not roots or len(set(roots)) != len(roots):
        raise ValueError("Supply one or more distinct completed RT result roots")
    if not np.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("Tolerance must lie in (0,1)")
    loaded = [_load_root(root) for root in roots]
    signature, cases, frames = _merge_inputs(loaded)
    box_len = signature["physical_configuration"]["box_len"]
    comparisons = [_pair_diagnostics(a, b, frames, axis, box_len, tolerance)
                   for a, b, axis in _adjacent_pairs(cases)]
    status = ("insufficient_refinement_pairs" if not comparisons else
              "unresolved_numerical_sensitivity" if any(not pair["passed"] for pair in comparisons) else
              "tested_refinements_small_full_convergence_unverified")
    outputs = []
    for case in cases:
        data = frames[case["case_id"]]
        outputs.append({k: v for k, v in case.items() if k != "checkpoints"} | {
            "same_time": [_same_time(*data[node], box_len) for node in sorted(data)],
            "matched_mean_xHI": _matched(data, box_len)})
    return {
        "schema": 1, "hypothesis": "RT--native ES differences persist when RT numerical resolution increases at fixed archived physical inputs",
        "hypothesis_status": status, "hypothesis_verified": False, "convergence_certified": False,
        "physical_model_authority": "21cmFAST", "reference": "archived native ES fields",
        "comparison_is_truth_error": False, "tolerance": tolerance,
        "scope": "Restricted no-recombination effective-atom RT; observed refinements do not establish a continuum error bound",
        "matched_xHI_scope": "First saved volume-neutral-fraction bracket; convex xHI and brightness field interpolation before spectra. Epoch and density differences remain; no extrapolation.",
        "history_decomposition": "xHI_RT_mass-xHI_ES_mass = radiation/atoms - (emitted-native_ES_ions)/atoms",
        "history_decomposition_scope": "Accounting identity; radiation inventory can contain numerical transport effects and is not by itself a converged finite-c measurement",
        "input_provenance": [entry[3] for entry in loaded], "fixed_input_signature": signature,
        "analysis_sha256": {path.name: _digest(path) for path in (
            Path(__file__), Path(__file__).with_name("report.py"), Path(__file__).with_name("script_comparison.py"))},
        "cases": outputs, "refinement_comparisons": comparisons,
    }


def write_hypothesis(input_roots, output_dir, *, tolerance=.01):
    output = Path(output_dir).resolve()
    roots = [Path(root).resolve() for root in input_roots]
    if any(output == root or output.is_relative_to(root) or root.is_relative_to(output) for root in roots):
        raise ValueError("Hypothesis output must be separate from all completed input roots")
    result = build_hypothesis(roots, tolerance=tolerance)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "hypothesis_report.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)
    receipt = {"input_provenance": result["input_provenance"],
               "analysis_sha256": result["analysis_sha256"], "files": {path.name: _digest(path)}}
    (output / "hypothesis_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-roots", required=True, type=Path, nargs="+")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tolerance", type=float, default=.01)
    args = parser.parse_args()
    result = write_hypothesis(args.input_roots, args.output_dir, tolerance=args.tolerance)
    print(json.dumps({"hypothesis_status": result["hypothesis_status"],
                      "cases": len(result["cases"]), "refinement_pairs": len(result["refinement_comparisons"]),
                      "report": str(args.output_dir / "hypothesis_report.json")}))


if __name__ == "__main__":
    main()
