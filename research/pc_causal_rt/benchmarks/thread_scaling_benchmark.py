"""Measure RT thread scaling while verifying identical physical outputs.

Notes
-----
The unchanged public interval solver evolves a deterministic, inhomogeneous,
partially ionized state with anisotropic radiation and continuous sources.
Timing includes input copies and independent photon audits; compilation and
one single-thread warmup at each grid size are excluded. This measures one
interval workload, not a complete native 21cmFAST trajectory or SCRIPT speed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import resource
import sys
import time
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _initial_state(n: int, steps: int) -> tuple[tuple[np.ndarray, ...], dict]:
    """Construct the original pilot's fixed fields and physical interval."""
    from research.pc_matched.rt import LIGHT_SPEED_CM_S, MPC_CM, directions

    vectors, weights = directions(96)
    coords = np.indices((n, n, n), dtype=float) * (2 * np.pi / n)
    x, y, z = coords
    capacity = 1.0 + 0.2 * np.sin(x) * np.cos(y) * np.cos(z)
    ionized = capacity * (0.45 + 0.2 * np.sin(x + y))
    inventory = 0.02 * (1 + 0.3 * np.cos(x - z))
    angular = weights * (1 + 0.2 * vectors[:, 0])
    photons = angular[:, None, None, None] * inventory[None, ...]
    # Half a stable step short of the endpoint avoids an extra roundoff step.
    duration = (
        (steps - 0.5)
        * 0.4
        * (100 / n)
        * MPC_CM
        / (LIGHT_SPEED_CM_S * 8 * np.abs(vectors).sum(axis=1).max())
    )
    rate = 0.04 * capacity * (1 + 0.2 * np.cos(x + z)) / duration
    return (capacity, ionized, photons, rate, rate), {
        "dt_seconds": duration,
        "z_start": 7.0,
        "z_end": 7.0,
        "box_len_cMpc": 100.0,
        "cfl": 0.4,
    }


def _read_limit(name: str) -> str | None:
    path = Path("/sys/fs/cgroup") / name
    return path.read_text().strip() if path.is_file() else None


def _peak_rss_mib() -> float:
    divisor = 1024**2 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor


def benchmark(
    dimensions: tuple[int, ...] = (32, 48),
    thread_counts: tuple[int, ...] = (1, 4, 8),
    repeats: int = 2,
    steps: int = 8,
) -> dict:
    """Compare thread counts for unchanged synthetic RT interval workloads.

    Parameters
    ----------
    dimensions : tuple of int
        Cubic grid widths to test, each at least four cells.
    thread_counts : tuple of int
        Explicit OpenMP counts between one and 32, run sequentially.
    repeats : int
        Number of measured repetitions after a single-thread grid warmup.
    steps : int
        Number of radiation CFL steps in each measured interval.

    Returns
    -------
    dict
        Implementation provenance, physical settings, wall and CPU timings,
        cumulative process memory peaks, and output agreement diagnostics.

    Notes
    -----
    Each grid has the same dimensionless field pattern. Physical box length,
    redshift, opacity, density, quadrature and CFL remain fixed at the original
    pilot values. Grid-dependent duration is chosen to give ``steps`` updates.
    Memory is the process's cumulative maximum, not an isolated per-row peak.
    """
    from research.pc_causal_rt.transport import evolve_interval

    for values, low, high, name in (
        (dimensions, 4, None, "dimensions"),
        (thread_counts, 1, 32, "thread counts"),
    ):
        if (
            not values
            or len(values) != len(set(values))
            or any(
                type(v) is not int or v < low or (high is not None and v > high)
                for v in values
            )
        ):
            raise ValueError(f"Invalid {name}")
    if type(repeats) is not int or repeats < 1 or type(steps) is not int or steps < 1:
        raise ValueError("Positive integer repeats and radiation steps are required")

    repo = Path(__file__).resolve().parents[3]
    source_paths = (
        "research/pc_causal_rt/benchmarks/thread_scaling_benchmark.py",
        "research/pc_causal_rt/transport.py",
        "research/pc_causal_rt/transport_kernel.cpp",
        "research/pc_matched/rt.py",
    )
    source_hashes = {p: _digest(repo / p) for p in source_paths}
    rows = []
    for n in dimensions:
        state, kwargs = _initial_state(n, steps)
        warm = evolve_interval(*state, threads=1, **kwargs)
        if warm["steps"] != steps or not warm["ledger"]["passed"]:
            raise ValueError("Warmup failed the expected step count or photon audit")
        reference = (warm["ionized"].copy(), warm["photons"].copy())
        del warm
        for threads in thread_counts:
            wall_times, cpu_times = [], []
            difference = maximum_ledger = 0.0
            for _ in range(repeats):
                gc.collect()
                before = resource.getrusage(resource.RUSAGE_SELF)
                start = time.perf_counter()
                result = evolve_interval(*state, threads=threads, **kwargs)
                elapsed = time.perf_counter() - start
                after = resource.getrusage(resource.RUSAGE_SELF)
                wall_times.append(elapsed)
                cpu_times.append(
                    after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime
                )
                if result["steps"] != steps or not result["ledger"]["passed"]:
                    raise ValueError(
                        "Measured run failed the step count or photon audit"
                    )
                for field, expected in zip(
                    (result["ionized"], result["photons"]), reference, strict=True
                ):
                    difference = max(difference, float(np.max(abs(field - expected))))
                maximum_ledger = max(
                    maximum_ledger, result["ledger"]["relative_residual"]
                )
                del result
            row = {
                "dimension": n,
                "directions": 96,
                "threads": threads,
                "radiation_steps": steps,
                "wall_seconds": wall_times,
                "mean_wall_seconds": float(np.mean(wall_times)),
                "cpu_seconds": cpu_times,
                "maximum_field_difference_vs_one_thread": difference,
                "maximum_ledger_relative_residual": maximum_ledger,
                "process_cumulative_peak_rss_MiB": _peak_rss_mib(),
            }
            rows.append(row)
            LOGGER.info("%s", json.dumps(row, allow_nan=False))
        del reference, state, kwargs
        gc.collect()
    if source_hashes != {p: _digest(repo / p) for p in source_paths}:
        raise RuntimeError("Benchmark source changed during measurement")
    return {
        "schema": 1,
        "completed": True,
        "scope": (
            "Public evolve_interval wall time includes copies and independent ledger; "
            "compile and grid warmup excluded. Fixed inhomogeneous partially ionized "
            "state, physical 18 eV opacity, anisotropic radiation and continuous "
            "sources. RSS is cumulative process peak, not isolated per-case."
        ),
        "source_hashes": source_hashes,
        "cpu_max": _read_limit("cpu.max"),
        "memory_max": _read_limit("memory.max"),
        "configuration": {
            "dimensions": dimensions,
            "thread_counts": thread_counts,
            "repeats": repeats,
            "radiation_steps": steps,
            "directions": 96,
            "box_len_cMpc": 100.0,
            "redshift": 7.0,
            "cfl": 0.4,
            "hydrogen_density_z0_cm3": 1.9e-7,
            "photon_energy_ev": 18.0,
        },
        "all_fields_bitwise_equal": all(
            r["maximum_field_difference_vs_one_thread"] == 0.0 for r in rows
        ),
        "rows": rows,
    }


def main() -> None:
    """Write a new thread-scaling report from explicit command-line settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", type=int, nargs="+", default=[32, 48])
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path; recorded timings are not overwritten")
    logging.basicConfig(level=logging.INFO)
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    result = benchmark(
        tuple(args.dimensions), tuple(args.threads), args.repeats, args.steps
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
