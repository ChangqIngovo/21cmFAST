"""Compare RT and SCRIPT spatial sensitivity on sealed native-input archives.

Notes
-----
Self-convergence power differences compare adjacent grids within one method.
The native output grid, its finite-volume averaging, and its Nyquist frequency
remain fixed within a dataset. A smaller refinement change is not an absolute
error estimate. Different box sizes are separate realizations, not another
refinement pair. Model agreement is reported separately at common allocation
refinements. Undefined power ratios remain undefined in every summary.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
from pathlib import Path

import numpy as np

from research.pc_causal_rt.benchmarks.script_refinement_benchmark import (
    _field_comparison,
    _pair_metrics,
)
from research.pc_causal_rt.report import _audit_case
from research.pc_causal_rt.script_comparison import _interpolate
from research.pc_field_emulator.evaluation import shell_spectra
from research.pc_native_rt.workflow import _receipt
from research.pc_script_reference.run import digest, write_json

LOGGER = logging.getLogger(__name__)


def _geometry(dimension: int, box_len: float) -> dict:
    """Describe the fixed output-grid scales in comoving physical units."""
    nyquist = np.pi * dimension / box_len
    return {
        "native_dimension": dimension,
        "box_len_cMpc": box_len,
        "native_cell_cMpc": box_len / dimension,
        "fundamental_cMpc_inv": 2 * np.pi / box_len,
        "half_nyquist_cMpc_inv": nyquist / 2,
        "nyquist_cMpc_inv": nyquist,
        "scope": "Spectra use the common native output grid after conservative restriction, without window deconvolution. Shells above half Nyquist are retained as near-grid diagnostics.",
    }


def _band_summary(row: dict, nyquist: float) -> dict:
    """Preserve undefined bins while summarizing two declared spectral bands."""
    k = np.asarray(row["spectra"]["k_cMpc_inv"], dtype=float)
    errors = row["power_fractional_error"]
    if len(errors) != len(k):
        raise ValueError("Power changes and shell centers have different lengths")
    answer = {}
    for name, selected in (
        ("below_half_nyquist", k <= nyquist / 2 * (1 + 1e-12)),
        ("above_half_nyquist", k > nyquist / 2 * (1 + 1e-12)),
    ):
        indices = np.flatnonzero(selected)
        values = [errors[i] for i in indices if errors[i] is not None]
        undefined = len(indices) - len(values)
        worst = max(values) if values else None
        answer[name] = {
            "shell_count": len(indices),
            "undefined_bins": undefined,
            "minimum_shell_cMpc_inv": float(k[indices].min()) if len(indices) else None,
            "maximum_shell_cMpc_inv": float(k[indices].max()) if len(indices) else None,
            "maximum_fractional_change": worst if undefined == 0 else None,
            "maximum_defined_fractional_change": worst,
        }
    return answer


def _annotate_pairs(pairs: list, nyquist: float) -> list:
    """Add scale diagnostics without altering the recorded field/spectrum data."""
    result = copy.deepcopy(pairs)
    for pair in result:
        if "coarse_setting" in pair:
            coarse = pair["coarse_setting"]["refinement"]
            fine = pair["fine_setting"]["refinement"]
        else:
            coarse, fine = pair["coarse_refinement"], pair["fine_refinement"]
        pair["refinement_ratio"] = fine / coarse
        pair["coarse_cell_width_cMpc"] = float(np.pi / (nyquist * coarse))
        pair["fine_cell_width_cMpc"] = float(np.pi / (nyquist * fine))
        for key in ("same_time_active", "matched_mean_xHI"):
            for row in pair[key]:
                if "spectra" in row:
                    row["scale_bands"] = _band_summary(row, nyquist)
                    row["wavelength_cMpc"] = [
                        float(2 * np.pi / k) for k in row["spectra"]["k_cMpc_inv"]
                    ]
    return result


def _shared_active_nodes(rt_pairs: list, script_pairs: list) -> list:
    """Record the common epochs for like-for-like same-time comparisons."""
    result = []
    for rt in rt_pairs:
        levels = tuple(
            rt[key]["refinement"] for key in ("coarse_setting", "fine_setting")
        )
        for script in script_pairs:
            if levels != (script["coarse_refinement"], script["fine_refinement"]):
                continue
            script_nodes = {row["node"] for row in script["same_time_active"]}
            common = [
                row for row in rt["same_time_active"] if row["node"] in script_nodes
            ]
            result.append(
                {
                    "RT_coarse_case": rt["coarse_case"],
                    "RT_fine_case": rt["fine_case"],
                    "refinement_levels": list(levels),
                    "nodes": [row["node"] for row in common],
                    "redshifts": [row["redshift"] for row in common],
                }
            )
    return result


def _validate_reference(identity: dict, hashes: dict) -> None:
    """Require the SCRIPT benchmark to refer to this exact sealed RT archive."""
    for name in ("manifest", "summary", "completion"):
        if identity[f"rt_{name}_sha256"] != hashes[name]:
            raise ValueError(f"SCRIPT benchmark refers to a different RT {name}")


def _read_rt_frames(rt_root: Path, cases: list, box_len: float) -> dict:
    """Audit each RT case once and retain common-grid fields for all diagnostics."""
    answer = {}
    for case in cases:
        _, frames = _audit_case(rt_root, case, box_len)
        answer[case["case_id"]] = [
            (
                row,
                {
                    "xhi": fields["rt_xhi"],
                    "brightness": fields["rt_brightness"],
                    "capacity": fields["capacity"],
                    "ionized": fields["ionized"],
                    "source": fields["cumulative_source"],
                },
            )
            for _, (row, fields) in sorted(frames.items())
        ]
    return answer


def _read_script_frames(root: Path, checkpoints: list, dimension: int) -> dict:
    """Read each hash-verified SCRIPT field once and check conserved restriction."""
    answer = {}
    seen = set()
    for row in checkpoints:
        key = row["refinement"], row["node"]
        if key in seen:
            raise ValueError("Duplicate SCRIPT refinement/node checkpoint")
        seen.add(key)
        path = (root / row["fields_path"]).resolve()
        if not path.is_relative_to(root) or digest(path) != row["fields_sha256"]:
            raise ValueError("SCRIPT checkpoint path or field hash differs")
        with np.load(path, allow_pickle=False) as saved:
            fields = {
                name: np.array(saved[name])
                for name in ("capacity", "source", "ionized", "xhi", "brightness")
            }
        if any(
            v.shape != (dimension,) * 3 or not np.isfinite(v).all()
            for v in fields.values()
        ):
            raise ValueError("Invalid SCRIPT native-grid field")
        atoms, ions, source = (
            fields[name] for name in ("capacity", "ionized", "source")
        )
        if (
            np.any(atoms < 0)
            or np.any(ions < 0)
            or np.any(source < 0)
            or np.any(ions > atoms + 2e-10 * np.maximum(atoms, 1e-300))
            or float(atoms.sum()) <= 0
        ):
            raise ValueError("SCRIPT restricted counts violate field bounds")
        xhi = np.divide(atoms - ions, atoms, out=np.ones_like(atoms), where=atoms > 0)
        if not np.allclose(xhi, fields["xhi"], rtol=0, atol=2e-12):
            raise ValueError("SCRIPT neutral field differs from restricted counts")
        mass_mean = float(
            np.sum(atoms - ions, dtype=np.longdouble)
            / np.sum(atoms, dtype=np.longdouble)
        )
        volume_mean = float(xhi.mean())
        if (
            abs(mass_mean - row["xHI_mass"]) > 2e-12
            or abs(volume_mean - row["xHI_volume"]) > 2e-12
        ):
            raise ValueError("SCRIPT recorded neutral means differ from saved fields")
        audited = row | {
            "xHI_mass": mass_mean,
            "xHI_volume": volume_mean,
            "source_per_atom": float(
                np.sum(source, dtype=np.longdouble) / np.sum(atoms, dtype=np.longdouble)
            ),
        }
        answer.setdefault(row["refinement"], []).append((audited, fields))
    for rows in answer.values():
        rows.sort(key=lambda r: r[0]["node"])
    return answer


def _rt_pairs_from_frames(cases: list, frames: dict, box_len: float) -> list:
    """Reuse audited RT fields for the original adjacent spatial comparisons."""
    groups = {}
    for case in cases:
        key = tuple(
            sorted((k, v) for k, v in case["setting"].items() if k != "refinement")
        )
        groups.setdefault(key, []).append(case)
    result = []
    for group in groups.values():
        group.sort(key=lambda case: case["setting"]["refinement"])
        for coarse, fine in itertools.pairwise(group):
            result.append(
                {
                    "coarse_case": coarse["case_id"],
                    "fine_case": fine["case_id"],
                    "coarse_setting": coarse["setting"],
                    "fine_setting": fine["setting"],
                    **_pair_metrics(
                        frames[coarse["case_id"]], frames[fine["case_id"]], box_len
                    ),
                }
            )
    return result


def _model_agreement(rt: dict, script: dict, box_len: float) -> dict:
    """Compare model fields with SCRIPT as the power-ratio denominator only."""
    spectra = shell_spectra(
        {"teacher": script["brightness"], "RT": rt["brightness"]}, box_len
    )
    return {
        "field_MAE": float(np.mean(np.abs(rt["xhi"] - script["xhi"]))),
        "RT_mean_xHI_volume": float(rt["xhi"].mean()),
        "SCRIPT_mean_xHI_volume": float(script["xhi"].mean()),
        "k_cMpc_inv": spectra["k_cMpc_inv"],
        "modes": spectra["modes"],
        "P_RT": spectra["P_RT"],
        "P_SCRIPT": spectra["P_teacher"],
        "Delta2_RT": spectra["Delta2_RT"],
        "Delta2_SCRIPT": spectra["Delta2_teacher"],
        "P_RT_over_SCRIPT": spectra["ratio_RT"],
        "r_RT_SCRIPT": spectra["r_RT"],
        "undefined_power_bins": sum(v is None for v in spectra["ratio_RT"]),
        "undefined_correlation_bins": sum(v is None for v in spectra["r_RT"]),
    }


def _matched_agreement(rt_frames: list, script_frames: list, box_len: float) -> list:
    """Retain independent time brackets when matching the two neutral histories."""
    result = []
    for target in (0.8, 0.6, 0.4, 0.2):
        rt = _interpolate(
            rt_frames, target, xhi_name="xhi", brightness_name="brightness"
        )
        script = _interpolate(
            script_frames, target, xhi_name="xhi", brightness_name="brightness"
        )
        row = {
            "target_xHI_volume": target,
            "available": rt is not None and script is not None,
        }
        if row["available"]:
            row.update(
                {
                    "RT_interpolation": rt["interpolation"],
                    "SCRIPT_interpolation": script["interpolation"],
                    **_model_agreement(rt, script, box_len),
                }
            )
        result.append(row)
    return result


def _model_cases(
    cases: list, rt_frames: dict, script_frames: dict, box_len: float
) -> list:
    """Compare each common allocation grid while preserving time-history context."""
    result = []
    for case in cases:
        factor = case["setting"]["refinement"]
        if factor not in script_frames:
            continue
        rt = rt_frames[case["case_id"]]
        rt_nodes = {row["node"] for row, _ in rt}
        script = [
            (row, fields)
            for row, fields in script_frames[factor]
            if row["node"] in rt_nodes
        ]
        if {row["node"] for row, _ in script} != rt_nodes:
            raise ValueError("SCRIPT is missing RT comparison checkpoints")
        same_time = []
        for (a, aa), (b, bb) in zip(rt, script, strict=True):
            if a["node"] != b["node"] or a["redshift"] != b["redshift"]:
                raise ValueError(
                    "Model histories do not share the same cosmic timestamps"
                )
            if not np.allclose(aa["capacity"], bb["capacity"], rtol=2e-11, atol=2e-12):
                raise ValueError("Model histories do not share the same atom grid")
            source_scale = max(
                float(np.sum(bb["source"], dtype=np.longdouble)), np.finfo(float).tiny
            )
            budget_error = float(
                abs(
                    np.sum(aa["source"], dtype=np.longdouble)
                    - np.sum(bb["source"], dtype=np.longdouble)
                )
                / source_scale
            )
            if budget_error > 2e-10:
                raise ValueError(
                    "RT and SCRIPT comparison uses different global photon budgets"
                )
            same_time.append(
                {
                    "node": a["node"],
                    "redshift": a["redshift"],
                    "time_seconds": a["time_seconds"],
                    "RT_xHI_mass": a["xHI_mass"],
                    "RT_xHI_volume": a["xHI_volume"],
                    "SCRIPT_xHI_mass": b["xHI_mass"],
                    "SCRIPT_xHI_volume": b["xHI_volume"],
                    "RT_radiation_per_atom": a["radiation_per_atom"],
                    "RT_source_per_atom": a["source_per_atom"],
                    "SCRIPT_source_per_atom": b["source_per_atom"],
                    "source_budget_relative_difference": budget_error,
                    "field_MAE": float(np.mean(np.abs(aa["xhi"] - bb["xhi"]))),
                }
            )
        result.append(
            {
                "case_id": case["case_id"],
                "setting": case["setting"],
                "common_refinement": factor,
                "same_time_history": same_time,
                "matched_mean_xHI": _matched_agreement(rt, script, box_len),
            }
        )
    return result


def _subsample_frames(frames: list) -> list:
    """Keep every other saved frame, always retaining the first and last."""
    if not frames:
        return []
    selected = frames[::2]
    if selected[-1][0]["node"] != frames[-1][0]["node"]:
        selected.append(frames[-1])
    return selected


def _sampling_comparison(frames: list, box_len: float, nyquist: float) -> dict:
    """Measure matched-stage interpolation sensitivity without rerunning a solver."""
    sparse = _subsample_frames(frames)
    rows = []
    for target in (0.8, 0.6, 0.4, 0.2):
        full = _interpolate(
            frames, target, xhi_name="xhi", brightness_name="brightness"
        )
        coarse = _interpolate(
            sparse, target, xhi_name="xhi", brightness_name="brightness"
        )
        row = {
            "target_xHI_volume": target,
            "available": full is not None and coarse is not None,
        }
        if row["available"]:
            row.update(
                {
                    "full_interpolation": full["interpolation"],
                    "sparse_interpolation": coarse["interpolation"],
                    **_field_comparison(coarse, full, box_len),
                }
            )
            row["scale_bands"] = _band_summary(row, nyquist)
        rows.append(row)
    return {
        "full_nodes": [row["node"] for row, _ in frames],
        "sparse_nodes": [row["node"] for row, _ in sparse],
        "full_node_count": len(frames),
        "sparse_node_count": len(sparse),
        "matched_mean_xHI": rows,
    }


def _sampling_diagnostics(
    cases: list, rt_frames: dict, script_frames: dict, box_len: float, nyquist: float
) -> dict:
    """Compare output sampling at the finest common allocation refinement."""
    common = {case["setting"]["refinement"] for case in cases} & set(script_frames)
    result = {
        "scope": "Output interpolation sensitivity, not temporal solver convergence: reuse completed fields, compare every saved node with every other saved node while retaining first and last. Interpolate xHI and brightness fields before spectra. No new RT or SCRIPT solve, no extrapolation. The all-saved-node result is the denominator, not a continuum reference.",
        "available": bool(common),
        "cases": [],
    }
    if not common:
        return result
    factor = max(common)
    result["finest_common_refinement"] = factor
    for case in cases:
        if case["setting"]["refinement"] != factor:
            continue
        rt = rt_frames[case["case_id"]]
        nodes = {row["node"] for row, _ in rt}
        script = [frame for frame in script_frames[factor] if frame[0]["node"] in nodes]
        if {row["node"] for row, _ in script} != nodes:
            raise ValueError("SCRIPT is missing output sampling comparison nodes")
        result["cases"].append(
            {
                "RT_case_id": case["case_id"],
                "setting": case["setting"],
                "RT": _sampling_comparison(rt, box_len, nyquist),
                "SCRIPT": _sampling_comparison(script, box_len, nyquist),
            }
        )
    return result


def _load_dataset(rt_root: Path, script_root: Path) -> dict:
    """Verify both archives, recompute RT diagnostics, and retain SCRIPT audits."""
    rt_root, script_root = rt_root.resolve(), script_root.resolve()
    for root in (rt_root, script_root):
        if not (root / "COMPLETED").is_file():
            raise ValueError(f"A completed sealed archive is required: {root}")
    rt_manifest = json.loads((rt_root / "manifest.json").read_text())
    if _receipt(rt_root, "completion.json", rt_manifest) is None:
        raise ValueError("Missing RT completion receipt")
    rt_hashes = {
        name: digest(rt_root / f"{name}.json")
        for name in ("manifest", "summary", "completion")
    }
    summary = json.loads((rt_root / "summary.json").read_text())
    script = json.loads((script_root / "report.json").read_text())
    identity = script["identity"]
    if _receipt(script_root, "completion.json", identity) is None:
        raise ValueError("Missing SCRIPT completion receipt")
    if json.loads((script_root / "identity.json").read_text()) != identity:
        raise ValueError("SCRIPT report and identity file differ")
    _validate_reference(identity, rt_hashes)
    if not script["audit_summary"]["all_ledgers_passed"]:
        raise ValueError("SCRIPT photon audits did not pass")
    seed = identity["seed"]
    cases = [case for case in summary["cases"] if case["seed"] == seed]
    if not cases or not all(case["all_ledgers_passed"] for case in cases):
        raise ValueError("Missing or unaudited RT cases")
    dimension = summary["configuration"]["dimension"]
    box_len = float(summary["configuration"]["box_len"])
    if dimension != identity["native_dimension"] or box_len != identity["box_len_cMpc"]:
        raise ValueError("RT and SCRIPT output geometries differ")
    histories = [h for h in summary["histories"] if h["seed"] == seed]
    if len(histories) != 1:
        raise ValueError("The selected RT seed must have one native history")
    for key in ("sha256", "source_budget_sha256"):
        if histories[0][key] != identity["history"][key]:
            raise ValueError("RT and SCRIPT do not share the exact native inputs")
    for case in cases:
        if (
            case["history_sha256"] != histories[0]["sha256"]
            or case["source_budget_sha256"] != histories[0]["source_budget_sha256"]
        ):
            raise ValueError("A selected RT case has different physical inputs")
        folder = rt_root / "cases" / case["case_id"]
        receipt = _receipt(folder, "receipt.json")
        if receipt is None or receipt["record"] != case:
            raise ValueError("RT case differs from its completed receipt")
    geometry = _geometry(dimension, box_len)
    nyquist = geometry["nyquist_cMpc_inv"]
    rt_frames = _read_rt_frames(rt_root, cases, box_len)
    script_frames = _read_script_frames(script_root, script["checkpoints"], dimension)
    rt_comparisons = _rt_pairs_from_frames(cases, rt_frames, box_len)
    if rt_comparisons != script["rt_spatial_comparisons"]:
        raise ValueError("Recomputed RT spatial diagnostics differ from SCRIPT context")
    script_pairs = script["comparisons_by_sampling"]
    allocations = []
    for factor in identity["factors"]:
        rows = [r for r in script["checkpoints"] if r["refinement"] == factor]
        if len(rows) != identity["source_node_count"]:
            raise ValueError("SCRIPT refinement is missing native source nodes")
        if not all(r["ledger"]["passed"] for r in rows):
            raise ValueError("An individual SCRIPT photon audit failed")
        times = [r["ledger"]["elapsed_seconds"] for r in rows]
        allocations.append(
            {
                "refinement": factor,
                "allocation_dimension": dimension * factor,
                "snapshot_count": len(rows),
                "sum_subprocess_wall_seconds": sum(times),
                "maximum_subprocess_wall_seconds": max(times),
                "scope": "Sum of individual serial Fortran subprocess wall times, not parallel elapsed time or measured CPU time.",
            }
        )
    return {
        "label": f"L={box_len:g} cMpc, native {dimension}^3, seed {seed}",
        "geometry": geometry,
        "identity": {
            "rt_root": str(rt_root),
            "script_root": str(script_root),
            "rt_hashes": rt_hashes,
            "script_report_sha256": digest(script_root / "report.json"),
            "script_completion_sha256": digest(script_root / "completion.json"),
            "history_sha256": histories[0]["sha256"],
            "source_budget_sha256": histories[0]["source_budget_sha256"],
            "seed": seed,
            "rt_configuration": summary["configuration"],
            "script_execution": identity.get("execution"),
        },
        "audit_summary": {
            "all_archive_file_hashes_verified": True,
            "rt_saved_fields_independently_reaudited": True,
            "script_restricted_fields_independently_reaudited": True,
            "script": script["audit_summary"],
        },
        "coverage": {
            "RT_redshift_end": summary["configuration"]["z_end"],
            "RT_redshift_start": summary["configuration"]["z_start"],
            "RT_saved_nodes_per_case": {
                name: len(values) for name, values in rt_frames.items()
            },
            "SCRIPT_source_node_count": identity["source_node_count"],
            "common_same_time_active_epochs": _shared_active_nodes(
                rt_comparisons, script_pairs["rt_saved_nodes"]
            ),
            "scope": "Each pair selects its active mass-neutral window independently. Compare methods at shared redshifts or matched neutral fractions; unavailable targets are not extrapolated. Different boxes may end at different redshifts.",
        },
        "self_convergence": {
            "RT": _annotate_pairs(rt_comparisons, nyquist),
            "SCRIPT": _annotate_pairs(script_pairs["rt_saved_nodes"], nyquist),
            "SCRIPT_all_native_nodes": _annotate_pairs(
                script_pairs["all_native_nodes"], nyquist
            ),
        },
        "model_agreement": {
            "scope": "RT versus SCRIPT on common refinement factors and exact native source/gas inputs. Field MAE, power ratios and Fourier cross-correlations describe model agreement, not numerical error or an accuracy ranking. Coeval SCRIPT does not carry the RT photon reservoir.",
            "matched_scope": "Each model is interpolated independently at the same volume-neutral fraction, using identical saved-node cadence. Different effective redshifts and evolving density remain; brackets are retained.",
            "power_ratio_definition": "P_RT/P_SCRIPT; r(k)=P_cross(RT,SCRIPT)/sqrt(P_RT*P_SCRIPT). Undefined bins remain null.",
            "cases": _model_cases(cases, rt_frames, script_frames, box_len),
        },
        "output_sampling_sensitivity": _sampling_diagnostics(
            cases, rt_frames, script_frames, box_len, nyquist
        ),
        "timing": {
            "rt_cases": [
                {
                    "case_id": case["case_id"],
                    "setting": case["setting"],
                    "allocation_dimension": dimension * case["setting"]["refinement"],
                    "threads": summary["configuration"]["threads"],
                    "elapsed_seconds": case["elapsed_seconds"],
                    "scope": "Full causal history, including checkpoints and restart writes; excludes shared native history construction. If resumed, the saved timer covers the final invocation only.",
                }
                for case in cases
            ],
            "script_by_refinement": allocations,
            "script_allocation_wall_seconds": script.get("allocation_wall_seconds"),
            "script_allocation_timing_scope": script.get("allocation_timing_scope"),
            "script_workflow_elapsed_seconds": script["total_elapsed_seconds"],
            "shared_native_preparation_seconds": None,
            "scope": "RT follows continuous finite-c transport; SCRIPT allocates photons independently at each coeval node. Their runtimes describe different numerical tasks with matched physical source/gas inputs. Observed wall times can include contention from concurrent jobs; profile and campaign records retain execution context. Requested threads do not imply exclusive CPU allocation.",
        },
    }


def _curves(dataset: dict):
    """Iterate matched method/refinement curves with stable plot styles."""
    for method, color in (("RT", "#2166ac"), ("SCRIPT", "#b35806")):
        styles = itertools.cycle(("-", "--", ":", "-."))
        seen = {}
        if method == "RT":
            for pair in dataset["self_convergence"][method]:
                key = (
                    pair["coarse_setting"]["refinement"],
                    pair["fine_setting"]["refinement"],
                )
                seen[key] = seen.get(key, 0) + 1
        for pair, style in zip(
            dataset["self_convergence"][method], styles, strict=False
        ):
            if method == "RT":
                a = pair["coarse_setting"]["refinement"]
                b = pair["fine_setting"]["refinement"]
            else:
                a, b = pair["coarse_refinement"], pair["fine_refinement"]
            label = f"{method} f{a}→{b}"
            if method == "RT" and seen[a, b] > 1:
                s = pair["coarse_setting"]
                label += f" D{s['directions']}/CFL{s['cfl']:g}/t{s['time_substeps']}"
            yield pair, color, style, label


def _plot_matched(datasets: list, output: Path) -> list[str]:
    """Plot each method's adjacent-grid power changes across physical scales."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets = (0.8, 0.6, 0.4, 0.2)
    fig, axes = plt.subplots(
        len(datasets),
        4,
        figsize=(16, 3.9 * len(datasets)),
        squeeze=False,
        constrained_layout=True,
    )
    for i, dataset in enumerate(datasets):
        maximum = 0.0
        for j, target in enumerate(targets):
            ax = axes[i, j]
            missing = []
            for pair, color, style, label in _curves(dataset):
                row = next(
                    (
                        r
                        for r in pair["matched_mean_xHI"]
                        if r["target_xHI_volume"] == target and r.get("available", True)
                    ),
                    None,
                )
                if row is None:
                    missing.append(label)
                    continue
                k = row["spectra"]["k_cMpc_inv"]
                values = [
                    np.nan if v is None else 100 * v
                    for v in row["power_fractional_error"]
                ]
                maximum = max([maximum, *(v for v in values if np.isfinite(v))])
                ax.plot(k, values, color=color, ls=style, marker="o", ms=3, label=label)
            half = dataset["geometry"]["half_nyquist_cMpc_inv"]
            nyquist = dataset["geometry"]["nyquist_cMpc_inv"]
            ax.axvspan(half, nyquist, color="0.9", alpha=0.6, zorder=-10)
            ax.axvline(half, color="0.45", lw=0.8, ls=":")
            ax.axvline(nyquist, color="0.45", lw=0.8, ls="--")
            ax.set_xscale("log")
            ax.set_yscale("symlog", linthresh=0.1)
            ax.set_ylim(bottom=0)
            ax.set_xlabel(r"$k\ [\mathrm{cMpc}^{-1}]$")
            ax.set_title(rf"$\bar x_{{\rm HI,V}}={target:g}$")
            ax.grid(alpha=0.18)
            if missing:
                ax.text(
                    0.03,
                    0.97,
                    "Target unavailable:\n" + "\n".join(missing),
                    transform=ax.transAxes,
                    va="top",
                    fontsize=7,
                    color="0.4",
                )
            if j == 0:
                ax.set_ylabel(dataset["label"] + "\nPower change [%]")
                ax.legend(fontsize=8, frameon=False)
        for ax in axes[i]:
            ax.set_ylim(0, max(0.2, 1.6 * maximum))
    fig.suptitle(
        "Spatial refinement at matched neutral fraction\n"
        "Common native output grid; shaded region: k > half Nyquist",
        fontsize=13,
    )
    paths = []
    for suffix in ("png", "pdf"):
        name = f"matched_scale_convergence.{suffix}"
        fig.savefig(output / name, dpi=180)
        paths.append(name)
    plt.close(fig)
    return paths


def _plot_same_time(datasets: list, output: Path) -> list[str]:
    """Show active same-redshift field and spectral-band sensitivity."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(datasets),
        3,
        figsize=(14, 3.8 * len(datasets)),
        squeeze=False,
        constrained_layout=True,
    )
    names = ("below_half_nyquist", "above_half_nyquist")
    titles = (
        "Neutral-field MAE",
        "Maximum power change: k ≤ half Nyquist",
        "Maximum power change: k > half Nyquist",
    )
    for i, dataset in enumerate(datasets):
        maxima = [0.0, 0.0, 0.0]
        for pair, color, style, label in _curves(dataset):
            rows = pair["same_time_active"]
            z = [r["redshift"] for r in rows]
            values = [[r["field_MAE"] for r in rows]]
            values.extend(
                [
                    np.nan
                    if r["scale_bands"][name]["maximum_fractional_change"] is None
                    else 100 * r["scale_bands"][name]["maximum_fractional_change"]
                    for r in rows
                ]
                for name in names
            )
            for j, series in enumerate(values):
                maxima[j] = max([maxima[j], *(v for v in series if np.isfinite(v))])
                axes[i, j].plot(z, series, color=color, ls=style, label=label)
        for j, ax in enumerate(axes[i]):
            ax.set_title(titles[j])
            ax.set_xlabel("Redshift")
            ax.invert_xaxis()
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.18)
            ax.set_ylabel(
                dataset["label"] + ("\nMAE" if j == 0 else "\nPower change [%]")
            )
            if j:
                ax.set_yscale("symlog", linthresh=0.1)
                ax.set_ylim(0, max(0.2, 1.6 * maxima[j]))
            if j == 0:
                ax.legend(fontsize=8, frameon=False)
    fig.suptitle(
        "Same-redshift refinement; both mass-neutral fractions in [0.05, 0.9]",
        fontsize=13,
    )
    paths = []
    for suffix in ("png", "pdf"):
        name = f"same_time_convergence.{suffix}"
        fig.savefig(output / name, dpi=180)
        paths.append(name)
    plt.close(fig)
    return paths


def _plot_model_agreement(datasets: list, output: Path) -> list[str]:
    """Display model power and phase agreement at the common neutral fraction 0.4."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(datasets),
        2,
        figsize=(11.5, 3.8 * len(datasets)),
        squeeze=False,
        constrained_layout=True,
    )
    for i, dataset in enumerate(datasets):
        model_cases = dataset["model_agreement"]["cases"]
        counts = {}
        for case in model_cases:
            f = case["common_refinement"]
            counts[f] = counts.get(f, 0) + 1
        correlation_minimum = 1.0
        missing = []
        for case in model_cases:
            row = next(
                (
                    r
                    for r in case["matched_mean_xHI"]
                    if r["target_xHI_volume"] == 0.4 and r["available"]
                ),
                None,
            )
            if row is None:
                missing.append(case["case_id"])
                continue
            factor = case["common_refinement"]
            grid = dataset["geometry"]["native_dimension"] * factor
            label = f"f{factor}, {grid}³"
            if counts[factor] > 1:
                s = case["setting"]
                label += f", D{s['directions']}/CFL{s['cfl']:g}/t{s['time_substeps']}"
            k = row["k_cMpc_inv"]
            for j, key in enumerate(("P_RT_over_SCRIPT", "r_RT_SCRIPT")):
                values = [np.nan if v is None else v for v in row[key]]
                axes[i, j].plot(k, values, marker="o", ms=3, label=label)
                if j == 1:
                    correlation_minimum = min(
                        [correlation_minimum, *(v for v in values if np.isfinite(v))]
                    )
        half, nyquist = (
            dataset["geometry"][key]
            for key in ("half_nyquist_cMpc_inv", "nyquist_cMpc_inv")
        )
        for j, ax in enumerate(axes[i]):
            ax.axhline(1, color="0.4", lw=0.8)
            ax.axvspan(half, nyquist, color="0.9", alpha=0.6, zorder=-10)
            ax.axvline(half, color="0.45", lw=0.8, ls=":")
            ax.axvline(nyquist, color="0.45", lw=0.8, ls="--")
            ax.set_xscale("log")
            ax.set_xlabel(r"$k\ [\mathrm{cMpc}^{-1}]$")
            ax.set_title(dataset["label"])
            ax.grid(alpha=0.18)
            if missing:
                ax.text(
                    0.03,
                    0.03,
                    "Target unavailable for " + str(len(missing)) + " case(s)",
                    transform=ax.transAxes,
                    va="bottom",
                    fontsize=8,
                    color="0.4",
                )
            ax.set_ylabel(
                r"$P_{\rm RT}/P_{\rm SCRIPT}$" if j == 0 else r"$r_{\rm RT,SCRIPT}(k)$"
            )
            if j == 0:
                ax.legend(fontsize=8, frameon=False)
            else:
                ax.set_ylim(max(-1, correlation_minimum - 0.03), 1.01)
    fig.suptitle(
        r"Model agreement at $\bar x_{\rm HI,V}=0.4$; common allocation refinement"
        "\nShaded region: k > half native-output Nyquist",
        fontsize=13,
    )
    paths = []
    for suffix in ("png", "pdf"):
        name = f"model_agreement_xhi04.{suffix}"
        fig.savefig(output / name, dpi=180)
        paths.append(name)
    plt.close(fig)
    return paths


def _attach_resource_profile(datasets: list, record: dict) -> None:
    """Associate a measured command with its exact RT or SCRIPT archive paths."""
    command = record["data"].get("command", [])
    paths = {}
    for option in ("--root", "--rt-root", "--output"):
        if command.count(option) == 1:
            index = command.index(option) + 1
            if index < len(command):
                paths[option] = str(Path(command[index]).resolve())
    for dataset in datasets:
        identity = dataset["identity"]
        if (
            paths.get("--rt-root") == identity["rt_root"]
            and paths.get("--output") == identity["script_root"]
        ):
            dataset["timing"].setdefault(
                "script_workflow_resource_profiles", []
            ).append(record)
        elif paths.get("--root") == identity["rt_root"]:
            dataset["timing"].setdefault("workflow_resource_profiles", []).append(
                record
            )


def build_multiscale_report(
    pairs: list[tuple[Path, Path]],
    output: Path,
    profile_paths: list[Path] | None = None,
) -> dict:
    """Write scale-resolved comparisons from completed RT and SCRIPT archives.

    Parameters
    ----------
    pairs : list of pairs of pathlib.Path
        Each pair contains a completed RT results root and the corresponding
        completed SCRIPT refinement benchmark root.
    output : pathlib.Path
        New directory for JSON, PNG and PDF outputs. Existing paths are rejected.
    profile_paths : list of pathlib.Path, optional
        Separate externally measured resource/timing records. Original data
        and hashes are retained without interpreting unrecorded timings.

    Returns
    -------
    dict
        Audited scale comparisons and explicitly scoped timing diagnostics.
    """
    if not pairs:
        raise ValueError("At least one RT/SCRIPT archive pair is required")
    output = output.resolve()
    if output.exists():
        raise ValueError("Use a new report directory")
    roots = [root.resolve() for pair in pairs for root in pair]
    if any(
        output.is_relative_to(root) or root.is_relative_to(output) for root in roots
    ):
        raise ValueError("Report directory must be separate from all input archives")
    datasets = [_load_dataset(*pair) for pair in pairs]
    resource_records = []
    for path in profile_paths or []:
        resource_record = {
            "path": str(path.resolve()),
            "sha256": digest(path),
            "data": json.loads(path.read_text()),
        }
        resource_records.append(resource_record)
        _attach_resource_profile(datasets, resource_record)
    repo = Path(__file__).resolve().parents[3]
    analysis_sources = (
        "research/pc_causal_rt/benchmarks/multiscale_report.py",
        "research/pc_causal_rt/benchmarks/script_refinement_benchmark.py",
        "research/pc_causal_rt/report.py",
        "research/pc_causal_rt/script_comparison.py",
        "research/pc_field_emulator/evaluation.py",
        "research/pc_native_rt/workflow.py",
        "research/pc_script_reference/run.py",
    )
    report = {
        "schema": 1,
        "scope": "Same-input RT and original SCRIPT spatial-refinement sensitivity across native-grid Fourier shells, with separate model-agreement diagnostics at common refinement factors. Neither is an error against a known field. Different boxes are not resolution pairs.",
        "power_definition": "Self-convergence uses abs(P_coarse/P_fine - 1), with each field's mean-subtracted native brightness. Model-agreement ratios use P_RT/P_SCRIPT. Undefined ratios remain null; no power floors are interpolated or replaced by zero.",
        "power_validity": "Ratios require reference power above max(1e-12 times its largest shell power, eps_float64^2 times volume times max(mean(brightness**2),1)). Cross-correlation additionally requires the other model above its own corresponding floor. A zero numerator with a valid reference gives ratio zero and undefined r(k).",
        "refinement_scope": "RT grid changes hold CFL fixed, which also changes radiation timestep size. Adjacent refinement ratios are recorded explicitly: f1 to f2 has ratio2, f2 to f3 has ratio1.5. Raw decreases across unequal steps do not establish convergence order.",
        "convergence_certified": False,
        "driver_sha256": digest(Path(__file__)),
        "analysis_source_hashes": {
            name: digest(repo / name) for name in analysis_sources
        },
        "datasets": datasets,
        "external_resource_measurements": resource_records,
    }
    output.mkdir(parents=True)
    write_json(output / "multiscale_report.json", report)
    figures = (
        _plot_matched(datasets, output)
        + _plot_same_time(datasets, output)
        + _plot_model_agreement(datasets, output)
    )
    write_json(
        output / "completion.json",
        {
            "files": {
                name: digest(output / name)
                for name in ["multiscale_report.json", *figures]
            },
            "input_pairs": [[str(p.resolve()) for p in pair] for pair in pairs],
            "driver_sha256": report["driver_sha256"],
        },
    )
    return report


def main() -> None:
    """Parse paired archives and write a standalone scale-comparison report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        nargs=2,
        type=Path,
        action="append",
        required=True,
        metavar=("RT_ROOT", "SCRIPT_ROOT"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", "--timing-json", type=Path, action="append")
    args = parser.parse_args()
    result = build_multiscale_report(
        [tuple(pair) for pair in args.pair], args.output, args.profile
    )
    logging.basicConfig(level=logging.INFO)
    LOGGER.info(
        "Wrote %d dataset(s) to %s", len(result["datasets"]), args.output.resolve()
    )


if __name__ == "__main__":
    main()
