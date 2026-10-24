#!/usr/bin/env python3
"""Matched full-grid versus lazy eight-subcell front accuracy and cost.

Full subdivision is the discrete-model equivalence oracle, while an integrated
continuum sphere separately measures spatial error. Native morphology changes
relative to coarse cells are not labelled as physical accuracy gains.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pc_refinement"))
sys.path.insert(0, str(HERE.parent / "pc_reconsidered"))
sys.path.insert(0, str(HERE))

import adaptive
import front_subcells
import hybrid
import upgraded
from compare_models import Case, make_cases, sphere_reference

MODELS = ("original_fibonacci", "balanced_coarse", "matched_coarse", "full_subcells", "adaptive_subcells")


def warmup():
    hybrid.library()
    front_subcells.library()
    upgraded.warmup("balanced")
    if hasattr(adaptive, "warmup"):
        adaptive.warmup()
    else:
        adaptive.library()


def call_model(name, source, capacity, initial, options, *, dense=False):
    if name == "original_fibonacci":
        return hybrid.solve(source, capacity, initial, **options)
    if name == "matched_coarse":
        return front_subcells.solve(source, capacity, initial, subcells=1, **options)
    if name == "balanced_coarse":
        return upgraded.solve(source, capacity, initial, profile="balanced", **options)
    if name == "full_subcells":
        return front_subcells.solve(source, capacity, initial, subcells=2,
                                    parent_skip=True, **options)
    if name == "adaptive_subcells":
        return adaptive.solve(source, capacity, initial, return_dense=dense, **options)
    raise ValueError(name)


def statistics(candidate, reference, source, capacity):
    delta = np.asarray(candidate)-reference
    q = float(source.sum(dtype=np.longdouble))
    dx = np.divide(delta, capacity, out=np.zeros_like(delta), where=capacity > 0)
    return {"L1_over_emitted": float(np.abs(delta).sum(dtype=np.longdouble)/q) if q else 0.,
            "maximum_count_difference": float(np.abs(delta).max()),
            "volume_mean_absolute_ionized_fraction_difference": float(np.abs(dx).mean()),
            "maximum_ionized_fraction_difference": float(np.abs(dx).max())}


def ledger(result, source, capacity, initial):
    a = result["absorbed"]
    q = source.sum(dtype=np.longdouble)
    tiny = np.finfo(float).tiny
    residual = q-a.sum(dtype=np.longdouble)-result["unabsorbed"]
    violation = np.maximum(np.maximum(a-(capacity-initial), 0.), np.maximum(-a, 0.))
    local = float(np.max(violation/np.maximum(capacity, tiny)))
    active = source > 0
    src_sum = result["source_absorbed"]+result["source_escaped"]+result["source_retained"]
    src_relative = float(np.max(np.abs(src_sum[active]-source[active])/source[active])) if active.any() else 0.
    relative = float(abs(residual)/max(q, tiny))
    passed = relative < 2e-10 and local < 2e-10 and src_relative < 2e-10
    return {"passed": passed, "relative_count_residual": relative,
            "maximum_local_fraction_violation": local,
            "maximum_source_relative_residual": src_relative}


def _rss_worker(connection, name, source, capacity, initial, options):
    """One independent process per model, avoiding a previous model's high water."""
    try:
        warmup()
        def memory_status():
            lines = Path("/proc/self/status").read_text().splitlines()
            return {line.split(":", 1)[0]: int(line.split()[1])*1024
                    for line in lines if line.startswith(("VmRSS:", "VmHWM:"))}

        # A spawned Python process can inherit the fork parent's RSS high-water
        # mark across exec. Reset our own kernel RSS peak after warmup; otherwise
        # a large prior parent allocation can make every model report one floor.
        Path("/proc/self/clear_refs").write_text("5\n")
        before = memory_status()["VmRSS"]
        started = time.perf_counter()
        out = call_model(name, source, capacity, initial, options)
        seconds = time.perf_counter()-started
        after = memory_status()["VmHWM"]
        connection.send({"peak_process_rss_bytes": after,
                         "warm_process_rss_bytes": before,
                         "increase_in_rss_high_water_bytes": after-before,
                         "seconds": seconds, "audit": out["audit"],
                         "scope": "Fresh Linux process; VmHWM reset after warmup; total RSS includes interpreter, inputs, outputs and transport"})
    except BaseException:
        connection.send({"error": traceback.format_exc()})
    finally:
        connection.close()


def process_memory(name, case, options):
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_rss_worker,
                              args=(sender, name, case.source, case.capacity, case.initial, options))
    process.start()
    sender.close()
    report = receiver.recv()
    receiver.close()
    process.join()
    if process.exitcode or "error" in report:
        raise RuntimeError(report.get("error", f"Memory worker failed with {process.exitcode}"))
    return report


def check_equivalence(lazy, full, case):
    report = statistics(lazy["absorbed"], full["absorbed"], case.source, case.capacity)
    # Count and fraction norms are both checked, so small capacities cannot hide
    # a large ionization-fraction error behind a bright distant source budget.
    assert report["L1_over_emitted"] < 2e-10, report
    assert report["maximum_ionized_fraction_difference"] < 2e-9, report
    report["source_ledgers"] = {}
    for key in ("source_absorbed", "source_escaped", "source_retained"):
        delta = lazy[key]-full[key]
        value = float(np.abs(delta).sum()/max(float(case.source.sum()), np.finfo(float).tiny))
        report["source_ledgers"][key+"_L1_over_emitted"] = value
        assert value < 2e-10, (key, value)
    fine_c = case.capacity.repeat(2, 0).repeat(2, 1).repeat(2, 2)/8
    fine_delta = lazy["subcell_absorbed"]-full["subcell_absorbed"]
    fine_fraction = np.divide(np.abs(fine_delta), fine_c, out=np.zeros_like(fine_c), where=fine_c>0)
    report["maximum_subcell_fraction_difference"] = float(fine_fraction.max())
    report["subcell_L1_over_emitted"] = float(np.abs(fine_delta).sum()/max(float(case.source.sum()), np.finfo(float).tiny))
    assert report["maximum_subcell_fraction_difference"] < 2e-9, report
    return report


def compare_case(case, *, repeats, angles=None, measure_rss=False):
    options = dict(periodic=True, maximum_distance=case.maximum_distance)
    if angles is not None:
        options.update(min_angles=angles, max_angles=angles)
    rows = {name: {"model": name, "seconds_samples": []} for name in MODELS}
    saved = {}
    # Compile-only warmup does not warm large native allocations, page tables,
    # or the solver's persistent workspace. Give every model the same untimed
    # solve before collecting alternating steady-state samples.
    for name in MODELS:
        call_model(name, case.source, case.capacity, case.initial, options)
    for repeat in range(repeats):
        for name in (MODELS if repeat % 2 == 0 else MODELS[::-1]):
            started = time.perf_counter()
            out = call_model(name, case.source, case.capacity, case.initial, options)
            elapsed = time.perf_counter()-started
            check = ledger(out, case.source, case.capacity, case.initial)
            assert check["passed"], (case.name, name, check)
            row = rows[name]
            row["seconds_samples"].append(elapsed)
            row.update(audit=out["audit"], independent_ledger=check,
                       median_seconds=float(np.median(row["seconds_samples"])))
            if case.analytic is not None:
                row["vs_continuum_sphere"] = statistics(out["absorbed"], case.analytic,
                                                        case.source, case.capacity)
            saved[name] = out
    # Materializing dense subcell output is a debug/oracle operation and is not
    # included in adaptive timing or its default-process memory measurement.
    lazy_dense = call_model("adaptive_subcells", case.source, case.capacity, case.initial, options, dense=True)
    equivalence = check_equivalence(lazy_dense, saved["full_subcells"], case)
    for name, row in rows.items():
        row["vs_full_subcell_model"] = statistics(saved[name]["absorbed"], saved["full_subcells"]["absorbed"],
                                                  case.source, case.capacity)
        if measure_rss:
            row["process_memory"] = process_memory(name, case, options)
    return {"case": case.name, "shape": list(case.source.shape), "angles": angles,
            "maximum_distance": case.maximum_distance, "metadata": case.metadata,
            "analytic_reference_audit": case.analytic_audit,
            "input_hashes": {key: hashlib.sha256(np.ascontiguousarray(value, dtype=np.float64).tobytes()).hexdigest()
                             for key, value in (("source", case.source), ("capacity", case.capacity), ("initial", case.initial))},
            "equivalence": equivalence, "models": list(rows.values())}


def run(output, native32=None, native64=None, repeats=3, measure_rss=False, only=None):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    warmup()
    files = [Path(__file__), HERE/"adaptive.py", HERE/"adaptive.cpp",
             HERE.parent/"pc_refinement/front_subcells.py", HERE.parent/"pc_refinement/front_subcells.cpp",
             HERE.parent/"pc_refinement/upgraded.py", HERE.parent/"pc_refinement/budgeted.py",
             HERE.parent/"pc_refinement/speed_core.py", HERE.parent/"pc_refinement/speed_core.cpp",
             HERE.parent/"pc_reconsidered/compare_models.py",
             HERE.parent/"pc_reconsidered/hybrid.py", HERE.parent/"pc_reconsidered/hybrid.cpp"]
    hashes = {str(p.relative_to(HERE.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    report = {"scope": "Exact-model lazy subdivision versus full subdivision; separate analytic sphere spatial error",
              "coarse_controls": "Original Fibonacci and latest balanced profiles use their own angular rules; matched_coarse uses the same octant rule as both fine solvers. Fixed-K tests cap all profiles equally; native defaults retain balanced 1.25x angular boost.",
              "timing": "Warm compilation plus one untimed solve per model/case; three or requested sequential alternating measured repeats; shared host; serial event transport",
              "memory": "Optional separate-process RSS includes all Python/native allocations; per-core audit is separate",
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "platform": platform.platform(),
                              "cpu": next((line.split(":", 1)[1].strip()
                                           for line in Path("/proc/cpuinfo").read_text().splitlines()
                                           if line.startswith("model name")), "unknown"),
                              "threads": {key: os.environ.get(key) for key in
                                          ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}},
              "source_hashes": hashes, "cases": []}
    cases = make_cases(24)
    n, radius = 32, 8.
    centre = np.full(3, n//2)
    source = np.zeros((n,)*3)
    photons = 4*np.pi*radius**3/3
    source[tuple(centre)] = photons
    analytic = sphere_reference(n, radius, centre, quadrature=64)
    lower = sphere_reference(n, radius, centre, quadrature=32)
    larger_sphere = Case("isolated-r8", source, np.ones_like(source), np.zeros_like(source), n/2,
                         {"radius_cells": radius, "centre": centre.tolist(),
                          "scope": "Separate uniform sphere spatial-error test; not cosmological grid convergence"},
                         analytic=analytic,
                         analytic_audit={"kind": "Continuous sphere integrated over parent voxels", "xy_quadrature": 64,
                                         "quadrature32_to64_L1_over_Q": float(np.abs(analytic-lower).sum()/photons),
                                         "relative_volume_error": float(analytic.sum()/photons-1)})
    jobs = [(cases["isolated"], 1536), (cases["isolated"], 6144), (larger_sphere, 6144),
            (cases["equal-overlap"], 1536), (cases["transparent-overlap-increment"], 1536),
            (cases["all-partial"], 96)]
    for name, path in (("native32", native32), ("native64", native64)):
        if path:
            with np.load(path) as data:
                s, c = np.asarray(data["source"], dtype=float), np.asarray(data["capacity"], dtype=float)
            jobs.append((Case(name, s, c, np.zeros_like(c), len(s)/2,
                              {"fixture": str(path), "fixture_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                               "scope": "Frozen native source; no continuum morphology oracle"}), None))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for case, angles in jobs:
        if only and case.name not in only:
            continue
        record = compare_case(case, repeats=repeats, angles=angles,
                              measure_rss=measure_rss and case.name.startswith("native"))
        report["cases"].append(record)
        output.write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps({"case": case.name, "angles": angles,
                          "equivalence": record["equivalence"],
                          "seconds": {row["model"]: row["median_seconds"] for row in record["models"]}}), flush=True)
    if not report["cases"]:
        raise ValueError("No selected benchmark cases")
    final_hashes = {str(p.relative_to(HERE.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    if hashes != final_hashes:
        raise RuntimeError("Source code changed during benchmark; rerun the comparison")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE/"results.json")
    parser.add_argument("--native32", type=Path)
    parser.add_argument("--native64", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--measure-rss", action="store_true")
    parser.add_argument("--only", nargs="+")
    run(**vars(parser.parse_args()))
