"""Measure original SCRIPT allocation-grid sensitivity on archived native inputs.

Notes
-----
This changes neither the 21cmFAST source/atom physics nor production RT. Each
coeval SCRIPT solve uses the same moving physical source and gas clouds. Fine
cell counts are expressed in fine mean-cell units (the common factor f**3);
physical photons and atoms are unchanged. Ions and atoms are conservatively
restricted before evaluating neutral fractions and native brightness.

The result is numerical sensitivity for the supplied realization and scales,
not an absolute truth error, universal tolerance, or convergence certificate.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import time
from pathlib import Path

import numpy as np

from research.pc_causal_rt.gas import deposit
from research.pc_causal_rt.native_history import load_history
from research.pc_causal_rt.report import _audit_case
from research.pc_causal_rt.script_comparison import _interpolate
from research.pc_causal_rt.source_budget import load_source_budget, source_positions
from research.pc_causal_rt.workflow import _positions
from research.pc_field_emulator.evaluation import shell_spectra
from research.pc_native_rt.workflow import _receipt, _restrict, _runtime_identity
from research.pc_script_reference.kernel import provenance, run
from research.pc_script_reference.run import _native_snapshot, digest, write_json

LOGGER = logging.getLogger(__name__)
TARGETS = (0.8, 0.6, 0.4, 0.2)


def _restrict_ionization(
    capacity: np.ndarray, fraction: np.ndarray, dimension: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Restrict conserved counts before dividing to obtain a neutral fraction."""
    atoms = _restrict(capacity, dimension)
    ions = _restrict(capacity * fraction, dimension)
    neutral = np.divide(atoms - ions, atoms, out=np.ones_like(atoms), where=atoms > 0)
    return atoms, ions, neutral


def _field_comparison(coarse: dict, fine: dict, box_len: float) -> dict:
    """Compare native-grid fields using the finer-refinement power denominator."""
    spectra = shell_spectra(
        {"teacher": fine["brightness"], "coarse": coarse["brightness"]}, box_len
    )
    return {
        "field_MAE": float(np.mean(np.abs(coarse["xhi"] - fine["xhi"]))),
        "coarse_mean_xHI": float(coarse["xhi"].mean()),
        "fine_mean_xHI": float(fine["xhi"].mean()),
        "power_fractional_error": [
            None if ratio is None else abs(ratio - 1)
            for ratio in spectra["ratio_coarse"]
        ],
        "spectra": spectra,
    }


def _pair_metrics(coarse: list, fine: list, box_len: float) -> dict:
    """Report active same-time and matched-mean comparisons without thresholds."""
    same_time = []
    fine_nodes = {row["node"]: (row, fields) for row, fields in fine}
    for row, fields in coarse:
        if row["node"] not in fine_nodes:
            continue
        fine_row, fine_fields = fine_nodes[row["node"]]
        if 0.05 <= row["xHI_mass"] <= 0.9 and 0.05 <= fine_row["xHI_mass"] <= 0.9:
            same_time.append(
                {
                    "node": row["node"],
                    "redshift": row["redshift"],
                    **_field_comparison(fields, fine_fields, box_len),
                }
            )
    matched = []
    for target in TARGETS:
        c = _interpolate(coarse, target, xhi_name="xhi", brightness_name="brightness")
        f = _interpolate(fine, target, xhi_name="xhi", brightness_name="brightness")
        item = {
            "target_xHI_volume": target,
            "available": c is not None and f is not None,
        }
        if item["available"]:
            item.update(
                coarse_interpolation=c["interpolation"],
                fine_interpolation=f["interpolation"],
                **_field_comparison(c, f, box_len),
            )
        matched.append(item)
    return {"same_time_active": same_time, "matched_mean_xHI": matched}


def _load_inputs(rt_root: Path, seed: int) -> tuple[dict, dict, dict, list, dict]:
    """Verify a sealed RT archive and its selected external native history."""
    manifest = json.loads((rt_root / "manifest.json").read_text())
    if not (rt_root / "COMPLETED").is_file():
        raise ValueError("A completed RT archive is required")
    if _receipt(rt_root, "completion.json", manifest) is None:
        raise ValueError("Missing RT completion receipt")
    current_helpers = _source_hashes()
    for name in (
        "gas.py",
        "gas_kernel.cpp",
        "source_budget.py",
        "native_history.py",
        "workflow.py",
    ):
        key = f"research/pc_causal_rt/{name}"
        if manifest["source_hashes"].get(key) != current_helpers[key]:
            raise ValueError(
                f"Input deposition helper differs from the RT archive: {key}"
            )
    summary = json.loads((rt_root / "summary.json").read_text())
    histories = [h for h in summary["histories"] if h["seed"] == seed]
    cases = [c for c in summary["cases"] if c["seed"] == seed]
    if len(histories) != 1 or not cases:
        raise ValueError("Select one recorded native history with completed RT cases")
    record = histories[0]
    history_path = Path(record["path"])
    budget_path = Path(record["source_budget_path"])
    if not history_path.is_absolute():
        history_path = rt_root / history_path
    if not budget_path.is_absolute():
        budget_path = rt_root / budget_path
    if (
        digest(history_path) != record["sha256"]
        or digest(budget_path) != record["source_budget_sha256"]
    ):
        raise ValueError("External history or source-budget hash changed")
    history, budget = load_history(history_path), load_source_budget(budget_path)
    if (
        budget["metadata"]["history_sha256"] != record["sha256"]
        or history["metadata"]["seed"] != seed
        or budget["metadata"]["seed"] != seed
        or not np.array_equal(history["redshifts"], budget["redshifts"])
        or not np.array_equal(history["growth_factors"], budget["growth_factors"])
    ):
        raise ValueError("History and cumulative source budget are not linked")
    for case in cases:
        if (
            case["source_mode"] != "native_cumulative"
            or case["setting"]["source_stride"] != 1
            or case["history_sha256"] != record["sha256"]
            or case["source_budget_sha256"] != record["source_budget_sha256"]
        ):
            raise ValueError(
                "RT case does not use the selected native cumulative history"
            )
    return summary, history, budget, cases, record


def _source_hashes() -> dict:
    """Identify the executed driver and its scientific helper modules."""
    repo = Path(__file__).resolve().parents[3]
    files = [
        "research/pc_causal_rt/gas.py",
        "research/pc_causal_rt/gas_kernel.cpp",
        "research/pc_causal_rt/native_history.py",
        "research/pc_causal_rt/source_budget.py",
        "research/pc_causal_rt/workflow.py",
        "research/pc_causal_rt/report.py",
        "research/pc_causal_rt/script_comparison.py",
        "research/pc_native_rt/workflow.py",
        "research/pc_script_reference/kernel.py",
        "research/pc_script_reference/run.py",
        "research/pc_field_emulator/evaluation.py",
        "research/pc_reconsidered/native_driver.py",
        str(Path(__file__).resolve().relative_to(repo)),
    ]
    return {name: digest(repo / name) for name in files}


def _solve_node(
    history: dict,
    budget: dict,
    node: int,
    factor: int,
    native: object,
    native_fields: dict,
    reference_inputs: tuple[np.ndarray, np.ndarray],
    output: Path,
    timeout: float,
) -> tuple[dict, dict]:
    """Run original SCRIPT and audit refined inputs at one native source node."""
    from py21cmfast.drivers.single_field import brightness_temperature
    from research.pc_reconsidered.native_driver import _native_box

    dimension = history["metadata"]["dimension"]
    growth = history["growth_factors"][node]
    capacity = deposit(
        history["packets"]["mass"] * factor**3,
        _positions(history, growth, factor),
        dimension * factor,
        cloud_width=factor,
    )
    source = deposit(
        budget["cumulative"][node].ravel() * factor**3,
        source_positions(budget, growth, factor),
        dimension * factor,
        cloud_width=factor,
    )
    cap_out, src_out = _restrict(capacity, dimension), _restrict(source, dimension)
    ref_cap, ref_src = reference_inputs
    audits = {
        "maximum_restricted_capacity_difference": float(np.max(abs(cap_out - ref_cap))),
        "maximum_restricted_source_difference": float(np.max(abs(src_out - ref_src))),
        "maximum_native_capacity_difference": float(
            np.max(abs(cap_out - native_fields["capacity"]))
        ),
        "maximum_native_source_difference": float(
            np.max(abs(src_out - native_fields["source"]))
        ),
    }
    if (
        audits["maximum_restricted_capacity_difference"] > 2e-12
        or audits["maximum_restricted_source_difference"]
        > 2e-12 * max(float(ref_src.max()), 1.0)
        or audits["maximum_native_capacity_difference"] > 3e-6
        or audits["maximum_native_source_difference"]
        > 3e-6 * max(float(native_fields["source"].max()), 1.0)
    ):
        raise ValueError("Refined clouds do not preserve the original native inputs")
    fraction, ledger = run(
        source, capacity, build_dir=output / "build", timeout_seconds=timeout
    )
    if not ledger["passed"]:
        raise ValueError(f"SCRIPT ledger failed at factor={factor}, node={node}")
    atoms, ions, xhi = _restrict_ionization(capacity, fraction, dimension)
    brightness = np.asarray(
        brightness_temperature(
            ionized_box=_native_box(native, ions, atoms),
            perturbed_field=native.perturbed_field,
            regenerate=True,
            write=False,
        ).get("brightness_temp")
    )
    path = output / f"f{factor}" / f"node{node:05d}.npz"
    path.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        path,
        capacity=atoms,
        source=src_out,
        ionized=ions,
        xhi=xhi,
        brightness=brightness,
    )
    row = {
        "node": node,
        "redshift": float(history["redshifts"][node]),
        "refinement": factor,
        "xHI_volume": float(xhi.mean()),
        "xHI_mass": float(
            np.sum(atoms - ions, dtype=np.longdouble)
            / np.sum(atoms, dtype=np.longdouble)
        ),
        **audits,
        "fields_path": str(path.relative_to(output)),
        "fields_sha256": digest(path),
        "ledger": ledger,
    }
    return row, {"xhi": xhi, "brightness": brightness}


def _rt_pairs(rt_root: Path, cases: list, box_len: float) -> list:
    """Audit RT fields and compare spatial neighbors with other settings fixed."""
    groups = {}
    for case in cases:
        key = tuple(
            sorted((k, v) for k, v in case["setting"].items() if k != "refinement")
        )
        groups.setdefault(key, []).append(case)
    pairs = []
    for group in groups.values():
        group.sort(key=lambda c: c["setting"]["refinement"])
        for coarse, fine in itertools.pairwise(group):
            frames = []
            for case in (coarse, fine):
                _, raw = _audit_case(rt_root, case, box_len)
                frames.append(
                    [
                        (
                            row,
                            {
                                "xhi": fields["rt_xhi"],
                                "brightness": fields["rt_brightness"],
                            },
                        )
                        for _, (row, fields) in sorted(raw.items())
                    ]
                )
            pairs.append(
                {
                    "coarse_case": coarse["case_id"],
                    "fine_case": fine["case_id"],
                    "coarse_setting": coarse["setting"],
                    "fine_setting": fine["setting"],
                    **_pair_metrics(*frames, box_len),
                }
            )
    return pairs


def benchmark(
    rt_root: Path,
    output: Path,
    seed: int,
    factors: tuple[int, ...],
    timeout: float = 120.0,
) -> dict:
    """Run same-input SCRIPT spatial comparisons with native RT diagnostics.

    Parameters
    ----------
    rt_root : pathlib.Path
        Completed native-cumulative RT results directory with original archives.
    output : pathlib.Path
        New, separate directory; existing paths are never overwritten.
    seed : int
        Seed selected from the input summary.
    factors : tuple of int
        Increasing positive integer allocation-grid refinements.
    timeout : float
        Maximum seconds for each original SCRIPT executable invocation.

    Returns
    -------
    dict
        Provenance, input audits, saved-field receipts, SCRIPT and RT spatial
        sensitivity diagnostics. No RT-versus-SCRIPT acceptance threshold.
    """
    started = time.perf_counter()
    rt_root, output = rt_root.resolve(), output.resolve()
    if (
        output.exists()
        or output.is_relative_to(rt_root)
        or rt_root.is_relative_to(output)
    ):
        raise ValueError("Use a new output directory separate from the input archive")
    if (
        len(factors) < 2
        or any(type(f) is not int or f < 1 for f in factors)
        or tuple(sorted(set(factors))) != factors
    ):
        raise ValueError(
            "At least two increasing positive integer factors are required"
        )
    summary, history, budget, cases, history_record = _load_inputs(rt_root, seed)
    dimension = history["metadata"]["dimension"]
    box_len = float(history["metadata"]["box_len_cMpc"])
    if (
        dimension != summary["configuration"]["dimension"]
        or box_len != summary["configuration"]["box_len"]
    ):
        raise ValueError("RT configuration and native geometry differ")
    runtime = _runtime_identity()
    count = len(history["redshifts"])
    rt_nodes = sorted({row["node"] for case in cases for row in case["checkpoints"]})
    if any(
        sorted(row["node"] for row in case["checkpoints"]) != rt_nodes for case in cases
    ):
        raise ValueError(
            "Selected RT cases must share identical saved checkpoint nodes"
        )
    identity = {
        "schema": 1,
        "seed": seed,
        "rt_root": str(rt_root),
        "rt_manifest_sha256": digest(rt_root / "manifest.json"),
        "rt_completion_sha256": digest(rt_root / "completion.json"),
        "rt_summary_sha256": digest(rt_root / "summary.json"),
        "history": {k: v for k, v in history_record.items() if k != "metadata"},
        "native_runtime": runtime,
        "script_provenance": provenance(),
        "source_files": _source_hashes(),
        "factors": factors,
        "native_dimension": dimension,
        "box_len_cMpc": box_len,
        "source_node_count": count,
        "rt_checkpoint_nodes": rt_nodes,
    }
    rt_pairs = _rt_pairs(rt_root, cases, box_len)
    output.mkdir(parents=True)
    write_json(output / "identity.json", identity)
    frames = {factor: [] for factor in factors}
    rows = []
    for node, growth in enumerate(history["growth_factors"]):
        native, fields, _ = _native_snapshot(
            Path(history["metadata"]["native_archive_root"]),
            history["records"][node],
            history,
            node,
            runtime,
        )
        reference = (
            deposit(
                history["packets"]["mass"], _positions(history, growth, 1), dimension
            ),
            deposit(
                budget["cumulative"][node].ravel(),
                source_positions(budget, growth),
                dimension,
            ),
        )
        for factor in factors:
            result = _solve_node(
                history,
                budget,
                node,
                factor,
                native,
                fields,
                reference,
                output,
                timeout,
            )
            frames[factor].append(result)
            rows.append(result[0])
        if node % 8 == 0:
            LOGGER.info("Completed native node %d/%d", node + 1, count)
    comparisons = {}
    for name, nodes in (
        ("rt_saved_nodes", set(rt_nodes)),
        ("all_native_nodes", set(range(count))),
    ):
        selected = {
            f: [v for v in values if v[0]["node"] in nodes]
            for f, values in frames.items()
        }
        comparisons[name] = [
            {
                "coarse_refinement": c,
                "fine_refinement": f,
                **_pair_metrics(selected[c], selected[f], box_len),
            }
            for c, f in itertools.pairwise(factors)
        ]
    audit = {
        "node_count": count,
        "solve_count": len(rows),
        "all_ledgers_passed": all(r["ledger"]["passed"] for r in rows),
        "maximum_ledger_relative_residual": max(
            r["ledger"]["relative_residual"] for r in rows
        ),
        **{k: max(r[k] for r in rows) for k in rows[0] if k.startswith("maximum_")},
    }
    report = {
        "identity": identity,
        "protocol": "Original coeval SCRIPT; fixed physical source/gas clouds. Fine-cell count units multiply both photons and atoms by f**3; physical counts are unchanged. Restrict ions and atoms separately before xHI and native brightness. No emissivity calibration or global physical rescaling.",
        "power_denominator": "Finer-refinement native brightness after conservative restriction to the native output grid",
        "active_window": "Both mass-weighted neutral fractions in [0.05,0.9]",
        "matched_xHI_scope": "First saved descending bracket; convex interpolation of xHI and native brightness fields before spectra; no extrapolation. Different epochs and evolving density remain.",
        "cadence_scope": "RT saved-node comparisons are directly comparable to recorded RT cadence; all-native-node SCRIPT comparisons diagnose interpolation sensitivity. Sparse late-stage interpolation can suppress errors.",
        "interpretation": "Empirical spatial sensitivity for this realization and scales, not absolute truth error or a universal SCRIPT tolerance. Coeval SCRIPT and causal RT have different transport assumptions. No per-bin RT<=SCRIPT acceptance test is imposed.",
        "convergence_certified": False,
        "audit_summary": audit,
        "checkpoints": rows,
        "comparisons_by_sampling": comparisons,
        "rt_spatial_comparisons": rt_pairs,
        "total_elapsed_seconds": time.perf_counter() - started,
    }
    write_json(output / "report.json", report)
    write_json(
        output / "completion.json",
        {
            "identity": identity,
            "files": {
                "identity.json": digest(output / "identity.json"),
                "report.json": digest(output / "report.json"),
                **{r["fields_path"]: r["fields_sha256"] for r in rows},
            },
        },
    )
    (output / "COMPLETED").write_text(
        "Original SCRIPT spatial sensitivity; all input and photon audits passed.\n"
    )
    return report


def main() -> None:
    """Run the benchmark from explicit archive and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=222)
    parser.add_argument("--factors", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    result = benchmark(
        args.rt_root, args.output, args.seed, tuple(args.factors), args.timeout
    )
    LOGGER.info(
        "Saved %d solves to %s", result["audit_summary"]["solve_count"], args.output
    )


if __name__ == "__main__":
    main()
