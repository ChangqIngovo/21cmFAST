"""Rerun same-input photon ledgers, numerical refinements and SCRIPT comparisons.

The input is an audited native snapshot, never a random cube or a training label.
Closed periodic transport is compared separately from source-domain escape.
Power and geometric MFP differences are descriptive; no reference is assumed
exact and a small ledger residual is not a spatial-convergence certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from research.pc_analytic.compare import (
    _block_mean,
    _completed_snapshot,
    _write_json,
    coarsen_snapshot,
    independent_budget,
    load_native_snapshot,
)
from research.pc_analytic.solver import build_identity, solve
from research.pc_coeval.mfp import compare_mfp, make_draws, measure_mfp
from research.pc_field_emulator.evaluation import field_metrics, shell_spectra


def refinement_plan(
    directions: tuple[int, int, int], exposures: tuple[int, int, int]
) -> tuple[dict[str, dict[str, int]], list[dict[str, str]]]:
    """Construct adjacent one-axis comparisons around the middle setting.

    Parameters
    ----------
    directions, exposures
        Three strictly increasing angular counts and source-dose subdivisions.

    Returns
    -------
    cases, comparisons
        Five unique solver settings and four ordered refinement comparisons.
    """
    for values in (directions, exposures):
        if len(values) != 3 or any(type(v) is not int or v < 1 for v in values):
            raise ValueError("Refinement axes require three positive integers")
        if not values[0] < values[1] < values[2]:
            raise ValueError("Refinement settings must strictly increase")
    if any(v not in {6, 24, 26, 96, 384} for v in directions):
        raise ValueError("Unsupported angular quadrature")
    cases, comparisons = {}, []
    for axis, values in (("directions", directions), ("exposure_steps", exposures)):
        names = []
        for value in values:
            d = value if axis == "directions" else directions[1]
            e = value if axis == "exposure_steps" else exposures[1]
            name = f"d{d}_e{e}"
            cases[name] = {"directions": d, "exposure_steps": e}
            names.append(name)
        comparisons.extend(
            {"axis": axis, "coarse": a, "fine": b} for a, b in pairwise(names)
        )
    return cases, comparisons


def pair_summary(
    candidate: NDArray[np.float64],
    reference: NDArray[np.float64],
    capacity: NDArray[np.float64],
    box_len: float,
    *,
    redshift: float,
) -> dict[str, Any]:
    """Compare neutral fields and their common saturated-brightness proxy.

    Parameters
    ----------
    candidate, reference
        Neutral-fraction grids. Ratios use candidate divided by reference.
    capacity, box_len
        Common normalized atom grid and comoving box side in Mpc.
    redshift
        Common redshift in the explicit proxy 27 sqrt((1+z)/10) A xHI mK.

    Returns
    -------
    dict
        Field differences, all shell spectra/correlations and low-k summaries.
        Undefined power ratios remain null rather than becoming zero errors.
    """
    if not np.isfinite(redshift) or redshift <= -1:
        raise ValueError("Redshift must be finite and greater than -1")
    metrics = field_metrics(1 - candidate, 1 - reference, capacity)
    factor = 27 * np.sqrt((1 + redshift) / 10)
    power = shell_spectra(
        {
            "teacher": factor * capacity * reference,
            "candidate": factor * capacity * candidate,
        },
        box_len,
    )
    k = np.asarray(power["k_cMpc_inv"])
    selected = [
        abs(ratio - 1)
        for wave, ratio in zip(k, power["ratio_candidate"], strict=True)
        if ratio is not None and wave <= 0.5 * power["k_max_cMpc_inv"]
    ]
    return {
        "field": metrics,
        "spectra_21cm": power,
        "power_below_half_nyquist": {
            "valid_bins": len(selected),
            "maximum_absolute_ratio_minus_one": max(selected) if selected else None,
            "mean_absolute_ratio_minus_one": float(np.mean(selected))
            if selected
            else None,
            "band_scope": "diagnostic cut only; no accuracy or convergence guarantee",
        },
    }


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _archived_c2ray(
    reference_run: str | Path,
    source: NDArray[np.float64],
    capacity: NDArray[np.float64],
    metadata: dict[str, Any],
    sigma_cm2: float,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    root = Path(reference_run).resolve()
    if not (root / "COMPLETED").is_file():
        raise ValueError("Archived reference is not completed")
    report = json.loads((root / "comparison_report.json").read_text())
    identity = json.loads((root / "identity.json").read_text())
    if (
        not report.get("complete")
        or not report.get("three_methods_completed")
        or report["identity"] != identity
    ):
        raise ValueError("Archived reference report or identity is inconsistent")
    physics = report["common_physics"]
    for key, expected in (
        ("initial_xHII", 0.0),
        ("recombinations", False),
        ("collisional_ionization", False),
        ("thermal_evolution", False),
        ("photoheating_feedback", False),
    ):
        if physics.get(key) != expected:
            raise ValueError(f"Archived reference physics differs: {key}")
    if physics.get("sigma_cm2") != sigma_cm2:
        raise ValueError("Archived reference cross section differs")
    seed, z = metadata["seed"], metadata["redshift"]
    rows = [
        row
        for row in report["snapshots"]
        if row["seed"] == seed and row["redshift"] == z
    ]
    if len(rows) != 1:
        raise ValueError(
            "Archived reference snapshot selection is missing or ambiguous"
        )
    folder = root / f"seed{seed}" / ("z" + f"{z:g}".replace(".", "p"))
    row = _completed_snapshot(
        folder, _digest(root / "identity.json"), require_c2ray=True
    )
    if row is None or row != rows[0]:
        raise ValueError(
            "Archived reference snapshot differs from its completed report"
        )
    receipt = json.loads((folder / "receipt.json").read_text())
    if not {"fields.npz", "metrics.json"}.issubset(receipt["files"]):
        raise ValueError("Archived reference receipt must verify fields and metrics")
    for key in (
        "inputs",
        "box_len",
        "dimension",
        "source_convention",
        "count_conversion",
    ):
        if row["input"].get(key) != metadata.get(key):
            raise ValueError(f"Archived reference native inputs differ: {key}")
    with np.load(folder / "fields.npz", allow_pickle=False) as saved:
        if not np.array_equal(saved["source"], source) or not np.array_equal(
            saved["capacity"], capacity
        ):
            raise ValueError(
                "Archived reference arrays differ from current native input"
            )
        neutral = np.array(saved["c2ray_xHI"], dtype=float)
    if (
        neutral.shape != source.shape
        or not np.isfinite(neutral).all()
        or np.any(neutral < 0)
        or np.any(neutral > 1)
    ):
        raise ValueError("Archived C2-Ray neutral field is invalid")
    return neutral, {
        "run": str(root),
        "rerun_in_current_diagnostics": False,
        "execution_scope": "Previously executed native C2-Ray field, verified from archived receipts; not a new backend run",
        "files_sha256": {
            "identity.json": _digest(root / "identity.json"),
            "comparison_report.json": _digest(root / "comparison_report.json"),
            "fields.npz": _digest(folder / "fields.npz"),
            "receipt.json": _digest(folder / "receipt.json"),
        },
        "original_budget": row["budgets"]["c2ray"],
        "original_audit": row["audits"]["c2ray"],
        "provenance": row["c2ray_provenance"],
    }


def _morphology(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    box_len: float,
    rays: int,
    threads: int,
) -> dict[str, Any]:
    draws = make_draws(rays, seed=20261116)
    result = {}
    for phase in ("ionized", "neutral"):
        a = measure_mfp(first, box_len, phase=phase, draws=draws, threads=threads)
        b = measure_mfp(second, box_len, phase=phase, draws=draws, threads=threads)
        result[phase] = {
            "candidate": a["summary"],
            "reference": b["summary"],
            "difference": compare_mfp(a, b),
        }
    return result


def run(
    snapshot: str | Path,
    output: str | Path,
    *,
    directions: tuple[int, int, int] = (24, 96, 384),
    exposures: tuple[int, int, int] = (16, 32, 64),
    threads: int = 1,
    sigma_cm2: float = 2.7173135802469135e-18,
    mfp_rays: int = 10000,
    script_build: str | Path | None = None,
    coarse_dimension: int | None = None,
    archived_reference_run: str | Path | None = None,
) -> dict[str, Any]:
    """Execute audited native-array diagnostics and original SCRIPT allocation.

    Parameters
    ----------
    snapshot
        Native snapshot directory with source.npz, native.h5 and receipts.
    output
        New empty directory for fields, provenance and diagnostics.json.
    directions, exposures
        Three increasing settings per numerical axis; the middle pair is base.
    threads
        Positive OpenMP worker count, also used by the geometric MFP tracer.
    sigma_cm2
        Shared monochromatic cross section; default corresponds to 18 eV.
    mfp_rays
        Common random draws per phase; finite-distance censored rays are retained.
    script_build
        Optional cache for the unmodified upstream SCRIPT Fortran executable.
    coarse_dimension
        Explicit smaller grid dividing the native grid; rerun on conservative
        source/atom block means to measure combined input and transport sensitivity.
    archived_reference_run
        Optional completed three-method archive with exactly matching native
        arrays and physics; its C2-Ray field is verified and compared, not rerun.

    Returns
    -------
    dict
        Completed field-derived ledgers, refinement and boundary comparisons,
        SCRIPT differences, MFP statistics and exact input/code hashes.
    """
    from research.pc_script_reference.kernel import run as run_script

    cases, comparisons = refinement_plan(directions, exposures)
    if (
        type(threads) is not int
        or threads < 1
        or type(mfp_rays) is not int
        or mfp_rays < 1
    ):
        raise ValueError("threads and mfp_rays must be positive integers")
    if not np.isfinite(sigma_cm2) or sigma_cm2 <= 0:
        raise ValueError("sigma_cm2 must be finite and positive")
    data = load_native_snapshot(snapshot, brightness="saturated_proxy")
    coarse = None
    if coarse_dimension is not None:
        if (
            type(coarse_dimension) is not int
            or coarse_dimension >= data["source"].shape[0]
        ):
            raise ValueError("coarse_dimension must be a smaller nested grid")
        coarse = coarsen_snapshot(data, coarse_dimension)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Diagnostics require an empty output directory")
    source, capacity, meta = data["source"], data["capacity"], data["metadata"]
    physical = {
        "box_len": meta["box_len"],
        "redshift": meta["redshift"],
        "nH_mean_comoving_cm3": meta["nH_mean_comoving_cm3"],
        "sigma_cm2": sigma_cm2,
    }
    settings = {
        "threads": threads,
        "transport_mode": "characteristic",
        "exhaustion_tolerance": 1e-8,
        "max_sweeps": 1000000,
    }
    repo = Path(__file__).resolve().parents[2]
    implementation = [
        Path(__file__),
        Path(__file__).with_name("solver.py"),
        Path(__file__).with_name("transport_kernel.cpp"),
        Path(__file__).with_name("compare.py"),
        repo / "research/pc_script_reference/kernel.py",
        repo / "research/pc_field_emulator/evaluation.py",
        repo / "research/pc_coeval/mfp.py",
        repo / "research/pc_coeval/mfp_kernel.cpp",
    ]
    report: dict[str, Any] = {
        "schema": 1,
        "complete": False,
        "input": meta,
        "physical": physical,
        "solver_build": build_identity(),
        "implementation_sha256": {
            str(p.relative_to(repo)): _digest(p) for p in implementation
        },
        "numerical_settings": settings,
        "cases": {},
        "refinements": [],
        "spatial_convergence_certified": False,
        "brightness": "27 sqrt((1+z)/10) capacity*xHI mK; common saturated proxy, not native brightness re-extraction",
        "scope": "Frozen initially neutral coeval effective atoms with H opacity; no recombinations, spin-temperature evolution, heating feedback or finite light speed",
        "boundary_scope": "source_periodic_cube permits measured source-domain escape; periodic recycles photons and matches SCRIPT intended closed domain",
    }
    _write_json(output / "diagnostics.json", report)
    fields = {}
    baseline = f"d{directions[1]}_e{exposures[1]}"
    cases["closed_periodic"] = dict(cases[baseline])
    for name, setting in cases.items():
        result = solve(
            source,
            capacity,
            **physical,
            **settings,
            **setting,
            boundary="periodic"
            if name == "closed_periodic"
            else "source_periodic_cube",
        )
        xhi = 1 - result["ionized"] / capacity
        # Reconstruct the inventory from returned arrays, not the solver's scalar ledger.
        budget = independent_budget(
            source,
            capacity,
            xhi,
            remaining=float(np.sum(result["remaining_radiation"], dtype=np.longdouble)),
            escaped=float(np.sum(result["escaped"], dtype=np.longdouble)),
            unemitted=float(np.sum(result["unemitted_source"], dtype=np.longdouble)),
        )
        report["cases"][name] = {
            "settings": setting,
            "audit": result["audit"],
            "field_budget": budget,
        }
        np.savez_compressed(
            output / f"{name}.npz",
            source=source,
            capacity=capacity,
            xHI=xhi,
            ionized=result["ionized"],
            remaining=np.sum(result["remaining_radiation"], axis=0),
            escaped=result["escaped"],
            unemitted=result["unemitted_source"],
        )
        fields[name] = xhi
        _write_json(output / "diagnostics.json", report)
        if (
            not result["audit"]["complete"]
            or not budget["bounded"]
            or budget["relative_photon_residual"] > 2e-10
        ):
            raise RuntimeError(f"Incomplete or invalid analytic case: {name}")
    script_qi, script_audit = run_script(
        source, capacity, build_dir=script_build, timeout_seconds=300
    )
    if not script_audit["passed"]:
        raise RuntimeError("Original SCRIPT failed its independent field audit")
    script_xhi = 1 - script_qi
    np.savez_compressed(
        output / "script.npz", source=source, capacity=capacity, xHI=script_xhi
    )
    report["script"] = script_audit
    report["native_es"] = {
        "field_budget": independent_budget(
            source, capacity, data["native_es_xhi"], remaining=None, escaped=None
        ),
        "comparison_to_script": pair_summary(
            data["native_es_xhi"],
            script_xhi,
            capacity,
            meta["box_len"],
            redshift=meta["redshift"],
        ),
    }
    for pair in comparisons:
        report["refinements"].append(
            pair
            | pair_summary(
                fields[pair["coarse"]],
                fields[pair["fine"]],
                capacity,
                meta["box_len"],
                redshift=meta["redshift"],
            )
        )
    report["boundary_difference"] = pair_summary(
        fields[baseline],
        fields["closed_periodic"],
        capacity,
        meta["box_len"],
        redshift=meta["redshift"],
    )
    report["comparisons_to_script"] = {}
    for name in (baseline, "closed_periodic"):
        report["comparisons_to_script"][name] = pair_summary(
            fields[name],
            script_xhi,
            capacity,
            meta["box_len"],
            redshift=meta["redshift"],
        ) | {
            "geometric_mfp": _morphology(
                fields[name], script_xhi, meta["box_len"], mfp_rays, threads
            )
        }
    if coarse is not None:
        coarse_fields = {}
        coarse_audits = {}
        coarse_source, coarse_capacity = coarse["source"], coarse["capacity"]
        for name, boundary in (
            ("source_periodic_cube", "source_periodic_cube"),
            ("closed_periodic", "periodic"),
        ):
            result = solve(
                coarse_source,
                coarse_capacity,
                **physical,
                **settings,
                **cases[baseline],
                boundary=boundary,
            )
            xhi = 1 - result["ionized"] / coarse_capacity
            audit = independent_budget(
                coarse_source,
                coarse_capacity,
                xhi,
                remaining=float(
                    np.sum(result["remaining_radiation"], dtype=np.longdouble)
                ),
                escaped=float(np.sum(result["escaped"], dtype=np.longdouble)),
                unemitted=float(
                    np.sum(result["unemitted_source"], dtype=np.longdouble)
                ),
            )
            if (
                not result["audit"]["complete"]
                or not audit["bounded"]
                or audit["relative_photon_residual"] > 2e-10
            ):
                raise RuntimeError(f"Invalid coarse analytic case: {name}")
            coarse_fields[name] = xhi
            coarse_audits[name] = {"audit": result["audit"], "field_budget": audit}
            np.savez_compressed(
                output / f"coarse_{name}.npz",
                source=coarse_source,
                capacity=coarse_capacity,
                xHI=xhi,
                ionized=result["ionized"],
                remaining=np.sum(result["remaining_radiation"], axis=0),
                escaped=result["escaped"],
                unemitted=result["unemitted_source"],
            )
        coarse_qi, coarse_script_audit = run_script(
            coarse_source, coarse_capacity, build_dir=script_build, timeout_seconds=300
        )
        if not coarse_script_audit["passed"]:
            raise RuntimeError("Coarse SCRIPT failed independent audit")
        coarse_fields["script"] = 1 - coarse_qi
        coarse_audits["script"] = coarse_script_audit
        np.savez_compressed(
            output / "coarse_script.npz",
            source=coarse_source,
            capacity=coarse_capacity,
            xHI=coarse_fields["script"],
        )
        ratio = (source.shape[0] / coarse_dimension) ** 3
        source_total = np.sum(source, dtype=np.longdouble)
        atom_total = np.sum(capacity, dtype=np.longdouble)
        grid = {
            "fine_dimension": source.shape[0],
            "coarse_dimension": coarse_dimension,
            "coarse_count_unit_over_fine_count_unit": ratio,
            "source_physical_total_relative_change": float(
                (ratio * np.sum(coarse_source, dtype=np.longdouble) - source_total)
                / max(source_total, np.finfo(float).tiny)
            ),
            "capacity_physical_total_relative_change": float(
                (ratio * np.sum(coarse_capacity, dtype=np.longdouble) - atom_total)
                / atom_total
            ),
            "scope": "Same original inputs conservatively restricted; combined input/substructure and transport resolution sensitivity, not pure transport convergence",
            "audits": coarse_audits,
            "comparisons": {},
        }
        for name, fine in (
            ("source_periodic_cube", fields[baseline]),
            ("closed_periodic", fields["closed_periodic"]),
            ("script", script_xhi),
        ):
            restricted = (
                _block_mean(capacity * fine, coarse_dimension) / coarse_capacity
            )
            grid["comparisons"][name] = pair_summary(
                coarse_fields[name],
                restricted,
                coarse_capacity,
                meta["box_len"],
                redshift=meta["redshift"],
            )
            np.savez_compressed(
                output / f"restricted_fine_{name}.npz",
                xHI=restricted,
                capacity=coarse_capacity,
            )
        report["grid_sensitivity"] = grid
    if archived_reference_run is not None:
        reference, record = _archived_c2ray(
            archived_reference_run, source, capacity, meta, sigma_cm2
        )
        record["analytic_comparison"] = pair_summary(
            fields[baseline],
            reference,
            capacity,
            meta["box_len"],
            redshift=meta["redshift"],
        )
        record["geometric_mfp"] = _morphology(
            fields[baseline], reference, meta["box_len"], mfp_rays, threads
        )
        report["archived_c2ray"] = record
    report["complete"] = True
    _write_json(output / "diagnostics.json", report)
    _write_json(
        output / "receipt.json",
        {
            "files": {
                p.name: _digest(p) for p in sorted(output.iterdir()) if p.is_file()
            }
        },
    )
    return report


def main() -> None:
    """Run the reproducible same-input diagnostics from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--directions", type=int, nargs=3, default=(24, 96, 384))
    parser.add_argument("--exposures", type=int, nargs=3, default=(16, 32, 64))
    parser.add_argument("--sigma-cm2", type=float, default=2.7173135802469135e-18)
    parser.add_argument("--mfp-rays", type=int, default=10000)
    parser.add_argument("--script-build")
    parser.add_argument("--coarse-dimension", type=int)
    parser.add_argument("--archived-reference-run")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    result = run(
        args.snapshot,
        args.output,
        directions=tuple(args.directions),
        exposures=tuple(args.exposures),
        threads=args.threads,
        sigma_cm2=args.sigma_cm2,
        mfp_rays=args.mfp_rays,
        script_build=args.script_build,
        coarse_dimension=args.coarse_dimension,
        archived_reference_run=args.archived_reference_run,
    )
    logging.getLogger(__name__).info(
        "%s",
        json.dumps(
            {
                "complete": result["complete"],
                "report": str(Path(args.output).resolve() / "diagnostics.json"),
            }
        ),
    )


if __name__ == "__main__":
    main()
