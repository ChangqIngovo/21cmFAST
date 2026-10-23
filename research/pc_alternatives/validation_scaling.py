"""Resource-bounded event-front scaling and exact independent-domain probe.

Core files are read-only. Independent cases run sequentially in fresh processes;
the final disjoint-domain experiment uses two processes, not solver threads.
"""

# ruff: noqa: T201 -- stdout is the child-process JSON protocol and CLI report.

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import multiprocessing
import os
import resource
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def core_hashes():
    """Identify the immutable implementation used by each measurement."""
    return {
        name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        for name in ("event_front.cpp", "event_front.py", "characteristics.py")
    }


def limits():
    """Limit one validation process, without changing the solver or precision."""
    resource.setrlimit(resource.RLIMIT_AS, (1024 * 2**20, 1024 * 2**20))
    resource.setrlimit(resource.RLIMIT_CPU, (28, 30))


def prepare(n, regime, density):
    """Keep cell size, sparse source density/rate distribution and range fixed."""
    rng = np.random.default_rng(20261027 + n)
    shape = (n,) * 3
    gas = np.exp(rng.normal(-0.125, 0.5, shape))
    centres = np.zeros(shape, dtype=bool)
    centres.flat[
        rng.choice(centres.size, round(0.01 * centres.size), replace=False)
    ] = True
    source = np.zeros(shape)
    source[centres] = rng.lognormal(np.log(10) - 0.245, 0.7, int(centres.sum()))
    transparent = np.zeros(shape, dtype=bool)
    if regime == "late":
        for offset in itertools.product(range(-3, 4), repeat=3):
            if sum(v * v for v in offset) <= 3.5**2:
                transparent |= np.roll(centres, offset, axis=(0, 1, 2))
    capacity = np.where(transparent, 0.0, gas)
    if density == "dense":
        source = rng.lognormal(-0.245, 0.7, shape)
        source *= 0.5 * capacity.sum() / source.sum()
    return (
        source,
        capacity,
        {
            "cell_size": 1.0,
            "box_side_cells": n,
            "initial_transparent_fraction": float(transparent.mean()),
            "nonzero_source_count": int(np.count_nonzero(source)),
            "nonzero_source_fraction": float(np.count_nonzero(source) / source.size),
            "source_total": float(source.sum()),
            "minimum_positive_source_rate": float(source[source > 0].min()),
            "capacity_total": float(capacity.sum()),
            "photons_per_available_capacity": float(source.sum() / capacity.sum()),
            "source_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
            "capacity_sha256": hashlib.sha256(capacity.tobytes()).hexdigest(),
            "source_model": "1% source cells, fixed lognormal mean rate 10"
            if density == "sparse"
            else "Every cell emits; total rate normalized to 0.5 times available capacity",
        },
    )


def run_case(n, angles, regime, density):
    """Measure one bounded serial solve and return its full ledger audit."""
    import event_front

    limits()
    before_hashes = core_hashes()
    source, capacity, setup = prepare(n, regime, density)
    event_front.library()  # Separate compiler/startup cost from solve cost.
    baseline_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    start = time.perf_counter()
    result = event_front.solve(
        source, capacity, angles=angles, duration=1, maximum_distance=16, periodic=True
    )
    wall = time.perf_counter() - start
    audit = result["audit"]
    return {
        "case": f"{density}_n{n}_k{angles}_{regime}",
        "status": "completed",
        "grid": n,
        "angles": angles,
        "regime": regime,
        "density": density,
        "setup": setup,
        "solve_wall_seconds": wall,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "baseline_peak_rss_mib": baseline_rss,
        "core_hashes_before": before_hashes,
        "core_hashes_after": core_hashes(),
        "audit": audit,
        "absorption_sha256": hashlib.sha256(result["absorbed"].tobytes()).hexdigest(),
        "relative_global_ledger_error": abs(audit["ledger_residual"])
        / audit["photons"],
        "materialized_fraction_of_possible_rays": audit["rays_materialized"]
        / (angles * setup["nonzero_source_count"]),
        "retained_fraction": audit["retained"] / audit["photons"],
    }


def summarize_validation(result):
    """Derive bounds from saved audits; regenerating inputs does not rerun solves."""
    passed = []
    for row in result["cases"]:
        if row["status"] != "completed":
            passed.append(False)
            continue
        _, _, setup = prepare(row["grid"], row["regime"], row["density"])
        for key in ("source_sha256", "capacity_sha256"):
            if setup[key] != row["setup"][key]:
                raise RuntimeError(f"Cannot reproduce saved input for {row['case']}")
        row["setup"]["minimum_positive_source_rate"] = setup[
            "minimum_positive_source_rate"
        ]
        audit = row["audit"]
        # Duration is one. Dividing the largest absolute source error by the
        # smallest positive source budget bounds every source's relative error.
        row["relative_per_source_ledger_error_upper_bound"] = (
            audit["per_source_ledger_max"] / setup["minimum_positive_source_rate"]
        )
        row["relative_per_angle_ledger_error"] = audit["per_angle_ledger_max"] / (
            audit["photons"] / row["angles"]
        )
        row["validation_checks"] = {
            "global_budget": row["relative_global_ledger_error"] < 1e-11,
            "every_source_budget_upper_bound": row[
                "relative_per_source_ledger_error_upper_bound"
            ]
            < 1e-11,
            "every_angle_budget": row["relative_per_angle_ledger_error"] < 1e-11,
            "independent_absorbed_budget": abs(audit["independent_absorption_ledger"])
            / audit["photons"]
            < 1e-11,
            "nonnegative_absorption": audit["negative_absorption"] == 0,
            "capacity_bound": audit["capacity_excess"] == 0,
            "heap_at_most_one_node_per_cell": audit["peak_heap_nodes"]
            <= row["grid"] ** 3,
            "ray_block_memory_bound": 64 * audit["rays_materialized"]
            <= audit["ray_allocated_bytes"]
            < 64 * audit["rays_materialized"] + 4096 * 64,
            "core_unchanged": row["core_hashes_before"]
            == row["core_hashes_after"]
            == result["core_hashes_at_start"],
        }
        passed.append(all(row["validation_checks"].values()))
    complete = [row for row in result["cases"] if row["status"] == "completed"]
    result["validation_summary"] = {
        "cases": len(result["cases"]),
        "completed": len(complete),
        "passed_all_checks": sum(passed),
        "all_cases_pass": bool(passed) and all(passed),
        "maximum_relative_global_budget_error": max(
            (row["relative_global_ledger_error"] for row in complete), default=None
        ),
        "maximum_relative_source_budget_error_upper_bound": max(
            (row["relative_per_source_ledger_error_upper_bound"] for row in complete),
            default=None,
        ),
        "maximum_relative_angle_budget_error": max(
            (row["relative_per_angle_ledger_error"] for row in complete), default=None
        ),
        "source_error_bound_definition": "max absolute source residual / minimum positive source budget; an upper bound, not the observed maximum relative residual",
    }


def group_worker(payload):
    """Solve one geometrically independent source group in a child process."""
    import event_front

    limits()
    source, capacity = payload
    result = event_front.solve(
        source, capacity, angles=192, duration=1, maximum_distance=6, periodic=True
    )
    return result["absorbed"], result["audit"]


def disjoint_parallel_probe():
    """Use a certified disjoint reachable-voxel condition, not opaque labels."""
    import event_front

    n = 48
    capacity = np.ones((n,) * 3)
    first = np.zeros_like(capacity)
    second = np.zeros_like(capacity)
    positions = np.array([[11, 24, 24], [36, 24, 24]])
    first[tuple(positions[0])] = 600
    second[tuple(positions[1])] = 900
    separation = np.minimum(
        abs(positions[0] - positions[1]), n - abs(positions[0] - positions[1])
    )
    distance = float(np.linalg.norm(separation))
    # Any entered voxel centre is at most R + sqrt(3)/2 from its source.
    required = 2 * 6 + np.sqrt(3)
    assert distance > required
    event_front.library()
    start = time.perf_counter()
    combined = event_front.solve(
        first + second,
        capacity,
        angles=192,
        duration=1,
        maximum_distance=6,
        periodic=True,
    )
    monolithic_seconds = time.perf_counter() - start
    start = time.perf_counter()
    sequential = [group_worker((part, capacity)) for part in (first, second)]
    sequential_seconds = time.perf_counter() - start
    start = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("fork")
    ) as pool:
        separate = list(pool.map(group_worker, [(first, capacity), (second, capacity)]))
    parallel_seconds = time.perf_counter() - start
    summed = separate[0][0] + separate[1][0]
    return {
        "scope": "Actual two-process solution of certified noninteracting source groups; each solver remains serial",
        "grid": n,
        "angles": 192,
        "processes": 2,
        "maximum_distance": 6,
        "source_separation_periodic": distance,
        "strict_disjoint_support_threshold": required,
        "support_certificate": "distance > 2*range + sqrt(3)*cell_size excludes any shared entered voxel",
        "monolithic_seconds": monolithic_seconds,
        "two_groups_sequential_seconds": sequential_seconds,
        "two_groups_two_processes_seconds_including_pool_and_transfer": parallel_seconds,
        "bitwise_equal_monolithic": bool(np.array_equal(combined["absorbed"], summed)),
        "bitwise_equal_sequential_groups": bool(
            np.array_equal(sequential[0][0] + sequential[1][0], summed)
        ),
        "maximum_cell_difference": float(np.max(np.abs(combined["absorbed"] - summed))),
        "shared_absorbing_cells": int(
            np.count_nonzero((separate[0][0] > 0) & (separate[1][0] > 0))
        ),
        "monolithic_audit": combined["audit"],
        "group_audits": [entry[1] for entry in separate],
        "core_hashes": core_hashes(),
        "limitation": "Tiny proof-of-equivalence fixture; process overhead may dominate. General interacting source groups are not independent.",
    }


def main():
    """Run isolated measurement processes and save resumable validation data."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "validation_scaling.json")
    parser.add_argument("--child", nargs=4, metavar=("N", "K", "REGIME", "DENSITY"))
    parser.add_argument("--parallel-child", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.child:
        n, angles, regime, density = args.child
        print(json.dumps(run_case(int(n), int(angles), regime, density)))
        return
    if args.parallel_child:
        print(json.dumps(disjoint_parallel_probe()))
        return
    cases = [
        (n, k, regime, "sparse")
        for n in (32, 48, 64)
        for k in (48, 192)
        for regime in ("early", "late")
    ]
    cases += [(n, 48, "late", "dense") for n in (32, 48)]
    result = {
        "scope": "Frozen toy validation at fixed cell size; not a native Qin20 evolution or extrapolation to 128 cubed",
        "resource_limits_per_process": {
            "address_space_mib": 1024,
            "cpu_seconds_soft": 28,
            "cpu_seconds_hard": 30,
            "wall_timeout_seconds": 40,
        },
        "solver_threads": 1,
        "case_concurrency": 1,
        "core_hashes_at_start": core_hashes(),
        "cases": [],
    }
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text())
        if previous["core_hashes_at_start"] == result["core_hashes_at_start"]:
            result = previous
    environment = dict(
        os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    done = {row["case"] for row in result["cases"]}
    for n, angles, regime, density in cases:
        name = f"{density}_n{n}_k{angles}_{regime}"
        if name in done:
            continue
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            str(n),
            str(angles),
            regime,
            density,
        ]
        started = time.perf_counter()
        try:
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=40,
                env=environment,
                check=False,
            )
            if process.returncode == 0:
                row = json.loads(process.stdout)
            else:
                row = {
                    "case": name,
                    "status": "failed_or_resource_limited",
                    "returncode": process.returncode,
                    "stderr_tail": process.stderr[-2000:],
                }
        except subprocess.TimeoutExpired:
            row = {"case": name, "status": "wall_timeout", "wall_timeout_seconds": 40}
        row["child_wall_including_startup_seconds"] = time.perf_counter() - started
        result["cases"].append(row)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                {
                    key: row[key]
                    for key in ("case", "status", "solve_wall_seconds", "peak_rss_mib")
                    if key in row
                }
            ),
            flush=True,
        )
    if "independent_parallel_probe" not in result:
        command = [sys.executable, str(Path(__file__).resolve()), "--parallel-child"]
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=40,
            env=environment,
            check=False,
        )
        result["independent_parallel_probe"] = (
            json.loads(process.stdout)
            if process.returncode == 0
            else {
                "status": "failed",
                "stderr_tail": process.stderr[-2000:],
                "returncode": process.returncode,
            }
        )
    result["core_hashes_at_end"] = core_hashes()
    summarize_validation(result)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["independent_parallel_probe"], indent=2))


if __name__ == "__main__":
    main()
