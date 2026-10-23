"""Independent ordinary-DDA paths and full-scan event reference.

No candidate clearance, traversal, heap, lazy source creation, or residual-to-
absorption subtraction is reused. Strict horizon convention: a target needs a
positive ray interval before the horizon; an open boundary at the horizon is
an escape. This verifies the declared discrete model, not continuum RT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LD = np.longdouble
INITIAL_FAILURE_HASHES = {
    "event_front.cpp": "aca74136b8383b7202903a4570cc78473feb4ae4f487a832b758cb2db9426c19",
    "event_front.py": "c2c3817b9795e84ba645f181acfc75957d6fbd3ae493ccf150c0314abfab0902",
}


@dataclass(frozen=True)
class RayPath:
    """Precomputed ordered voxel intersections and the endpoint category."""

    cells: tuple[int, ...]
    entry_distances: tuple[float, ...]
    terminal: str


def hashes() -> dict:
    """Identify all candidate code used by this validation process."""
    names = ["event_front.cpp", "event_front.py", "characteristics.py"]
    return {
        name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in names
    }


def ordinary_dda(
    n: int,
    source_cell: tuple,
    direction: np.ndarray,
    maximum_distance: float,
    *,
    periodic: bool = False,
) -> RayPath:
    """Enumerate voxel intervals using integer crossing indices, without nudges."""
    index = np.asarray(source_cell, dtype=np.int64).copy()
    velocity = np.asarray(direction, dtype=LD)
    origin = index.astype(LD) + LD("0.5")
    step = np.sign(velocity).astype(np.int64)
    next_plane = index.copy()
    next_plane[step > 0] += 1
    crossings = np.full(3, LD(np.inf))
    moving = step != 0
    crossings[moving] = (next_plane[moving] - origin[moving]) / velocity[moving]
    limit = LD(maximum_distance)
    cells, entries = [], []
    entry = LD(0)
    # Recompute plane distances from the origin at every crossing, avoiding
    # accumulated tDelta drift. Long double reduces reference rounding.
    bound = int(np.ceil(maximum_distance * np.abs(velocity).sum())) + 12
    for _ in range(bound):
        if not periodic and (np.any(index < 0) or np.any(index >= n)):
            return RayPath(tuple(cells), tuple(entries), "escaped")
        if entry >= limit:
            return RayPath(tuple(cells), tuple(entries), "retained")
        physical = index % n
        cells.append(int((physical[0] * n + physical[1]) * n + physical[2]))
        entries.append(float(entry))
        crossing = crossings.min()
        if crossing > limit:
            return RayPath(tuple(cells), tuple(entries), "retained")
        tied = crossings == crossing
        index[tied] += step[tied]
        next_plane[tied] += step[tied]
        crossings[tied] = (next_plane[tied] - origin[tied]) / velocity[tied]
        entry = crossing
    raise RuntimeError("Independent DDA exceeded its geometric crossing bound")


def reference_solve(
    source: np.ndarray,
    capacity: np.ndarray,
    directions: np.ndarray,
    *,
    duration: float = 1.0,
    periodic: bool = False,
    maximum_distance: float | None = None,
) -> dict:
    """Solve precomputed fixed paths with all-ray, all-cell event scans."""
    source = np.asarray(source, dtype=LD)
    original = np.asarray(capacity, dtype=LD)
    n = source.shape[0]
    horizon = n * np.sqrt(3) if maximum_distance is None else maximum_distance
    source_cells = np.flatnonzero(source.ravel() > 0)
    ray_sources, rates, paths = [], [], []
    for cell in source_cells:
        xyz = tuple(np.unravel_index(cell, source.shape))
        for direction in directions:
            paths.append(ordinary_dda(n, xyz, direction, horizon, periodic=periodic))
            ray_sources.append(cell)
            rates.append(source.ravel()[cell] / len(directions))
    ids = np.asarray(ray_sources, dtype=int)
    rates = np.asarray(rates, dtype=LD)
    cursor = np.zeros(len(paths), dtype=int)
    residual = original.ravel().copy()
    absorbed = np.zeros(residual.size, dtype=LD)
    source_absorbed = np.zeros_like(absorbed)
    source_escaped = np.zeros_like(absorbed)
    source_retained = np.zeros_like(absorbed)
    current, stop = LD(0), LD(duration)
    event_count = 0
    while current < stop:
        target = np.full(len(paths), -1, dtype=int)
        for beam, path in enumerate(paths):
            while (
                cursor[beam] < len(path.cells)
                and residual[path.cells[cursor[beam]]] <= 0
            ):
                cursor[beam] += 1
            if cursor[beam] < len(path.cells):
                target[beam] = path.cells[cursor[beam]]
        active = target >= 0
        incoming = np.zeros_like(residual)
        np.add.at(incoming, target[active], rates[active])
        until = np.full(residual.size, LD(np.inf))
        receives = incoming > 0
        until[receives] = residual[receives] / incoming[receives]
        next_event = until.min()
        elapsed = min(stop - current, next_event)
        if not elapsed > 0:
            raise RuntimeError("Reference event clock failed to advance")
        delta = rates * elapsed
        np.add.at(absorbed, target[active], delta[active])
        np.add.at(source_absorbed, ids[active], delta[active])
        residual[receives] -= incoming[receives] * elapsed
        for beam in np.flatnonzero(~active):
            outgoing = (
                source_escaped if paths[beam].terminal == "escaped" else source_retained
            )
            outgoing[ids[beam]] += delta[beam]
        current += elapsed
        if next_event <= elapsed:
            exhausted = until == next_event
            residual[exhausted] = 0
            event_count += int(exhausted.sum())
        # Long-double rounding may leave simultaneous exhaustion one ulp below
        # zero. No positive capacity is skipped or filled by a tolerance.
        residual[residual < 0] = 0
        if event_count > residual.size:
            raise RuntimeError("A cell filled twice in the independent model")
    shape = source.shape
    return {
        "absorbed": np.asarray(absorbed.reshape(shape), dtype=float),
        "source_absorbed": np.asarray(source_absorbed.reshape(shape), dtype=float),
        "source_escaped": np.asarray(source_escaped.reshape(shape), dtype=float),
        "source_retained": np.asarray(source_retained.reshape(shape), dtype=float),
        "events": event_count,
        "rays": len(paths),
    }


def compare_case(source, capacity, directions, **kwargs) -> dict:
    """Compare every returned cell and every source category to the reference."""
    import event_front

    reference = reference_solve(source, capacity, directions, **kwargs)
    actual = event_front.solve(source, capacity, ray_directions=directions, **kwargs)
    emitted = float(np.sum(np.asarray(source, dtype=LD)) * kwargs.get("duration", 1))
    scale = max(emitted, np.finfo(float).tiny)
    errors = {
        name: float(np.max(np.abs(actual[name] - reference[name])))
        for name in ("absorbed", "source_absorbed", "source_escaped", "source_retained")
    }
    individual_emission = np.asarray(source) * kwargs.get("duration", 1)
    emitting = individual_emission > 0
    relative_source_error = 0.0
    if np.any(emitting):
        for name in ("source_absorbed", "source_escaped", "source_retained"):
            relative_source_error = max(
                relative_source_error,
                float(
                    np.max(
                        np.abs(actual[name][emitting] - reference[name][emitting])
                        / individual_emission[emitting]
                    )
                ),
            )
        source_balance = (
            individual_emission
            - actual["source_absorbed"]
            - actual["source_escaped"]
            - actual["source_retained"]
        )
        relative_source_error = max(
            relative_source_error,
            float(
                np.max(np.abs(source_balance[emitting]) / individual_emission[emitting])
            ),
        )
    tolerance = 2e-11 * scale
    passed = all(error <= tolerance for error in errors.values())
    passed = passed and relative_source_error <= 2e-11
    passed = passed and abs(actual["audit"]["ledger_residual"]) <= tolerance
    passed = passed and abs(actual["audit"]["per_source_ledger_max"]) <= tolerance
    return {
        "passed": bool(passed),
        "emitted": emitted,
        "max_errors": errors,
        "max_cell_error_per_emitted_photon": errors["absorbed"] / scale,
        "max_source_error_per_own_emitted_budget": relative_source_error,
        "global_ledger_residual": actual["audit"]["ledger_residual"],
        "per_source_ledger_max": actual["audit"]["per_source_ledger_max"],
        "reference_events": reference["events"],
        "candidate_events": actual["audit"]["events"],
        "candidate_absorbed_sum": float(actual["absorbed"].sum()),
        "reference_absorbed_sum": float(reference["absorbed"].sum()),
        "candidate_escaped": float(actual["source_escaped"].sum()),
        "reference_escaped": float(reference["source_escaped"].sum()),
        "candidate_retained": float(actual["source_retained"].sum()),
        "reference_retained": float(reference["source_retained"].sum()),
    }


def axes_and_diagonals() -> np.ndarray:
    """Return fixed directions with exact axis and corner crossing ties."""
    raw = np.array(
        [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
            [1, 1, 0],
            [-1, 1, 0],
            [1, 1, 1],
            [-1, -1, -1],
        ],
        dtype=float,
    )
    return raw / np.linalg.norm(raw, axis=1)[:, None]


def special_cases() -> list:
    """Exercise strict horizons, empty boxes, tiny absorption and dynamic range."""
    rows = []
    plus_x = np.array([[1.0, 0.0, 0.0]])
    for cap in [1.0, 1e6, 1e12, 1e20]:
        s, c = np.full((1, 1, 1), 1e-3), np.full((1, 1, 1), cap)
        rows.append(
            {"case": "weak_local_source", "capacity": cap, **compare_case(s, c, plus_x)}
        )
    for periodic in [False, True]:
        for offset in [-2e-10, -5e-11, 0.0, 5e-11, 2e-10]:
            s, c = np.ones((1, 1, 1)), np.zeros((1, 1, 1))
            horizon = 0.5 + offset
            rows.append(
                {
                    "case": "box_face_horizon",
                    "periodic": periodic,
                    "horizon": horizon,
                    **compare_case(
                        s, c, plus_x, periodic=periodic, maximum_distance=horizon
                    ),
                }
            )
    for raw_direction in [
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.37, 0.19],
    ]:
        direction = np.asarray(raw_direction)
        direction /= np.linalg.norm(direction)
        for offset in [-2e-10, 0.0, 5e-11, 2e-10]:
            n = 5
            s, c = np.zeros((n,) * 3), np.zeros((n,) * 3)
            s[1, 1, 1] = 1
            target = tuple(
                np.array([1, 1, 1]) + (direction == direction.max()).astype(int)
            )
            c[target] = 2
            horizon = float(0.5 / LD(direction.max())) + offset
            rows.append(
                {
                    "case": "opaque_cell_face_horizon",
                    "direction": direction.tolist(),
                    "horizon": horizon,
                    **compare_case(s, c, direction[None], maximum_distance=horizon),
                }
            )
    for n in [1, 2, 5]:
        s, c = np.zeros((n,) * 3), np.zeros((n,) * 3)
        s[(n // 2,) * 3] = 1
        rows.append(
            {
                "case": "periodic_empty_cycles",
                "n": n,
                **compare_case(
                    s, c, axes_and_diagonals(), periodic=True, maximum_distance=9.37 * n
                ),
            }
        )
    for epsilon in [1e-8, 1e-10, 1e-12, 1e-14]:
        for reflected in [False, True]:
            s, c = np.zeros((5,) * 3), np.zeros((5,) * 3)
            s[1, 1, 1] = 1
            c[1, 2, 1], c[2, 2, 1] = 100, 100
            direction = np.array([[1.0, 1.0 + epsilon, 0.0]])
            direction /= np.linalg.norm(direction)
            if reflected:
                s, c = s[::-1, ::-1, :].copy(), c[::-1, ::-1, :].copy()
                direction *= np.array([-1.0, -1.0, 1.0])
            rows.append(
                {
                    "case": "near_corner_opaque_sliver",
                    "direction_offset": epsilon,
                    "reflected": reflected,
                    **compare_case(s, c, direction, maximum_distance=3.0),
                }
            )
    rng = np.random.default_rng(9911)
    s, c = np.zeros((5,) * 3), np.ones((5,) * 3)
    selected = rng.choice(c.size, size=9, replace=False)
    s.ravel()[selected] = 0.01
    c.ravel()[selected] = 0
    many_directions = rng.normal(size=(513, 3))
    many_directions /= np.linalg.norm(many_directions, axis=1)[:, None]
    rows.append(
        {
            "case": "ray_store_block_boundary",
            "explicit_rays": 9 * 513,
            **compare_case(s, c, many_directions),
        }
    )
    return rows


def randomized_cases(count: int = 100, seed: int = 91821) -> list:
    """Compare random source/capacity maps including signed direction octants."""
    rng = np.random.default_rng(seed)
    rows = []
    for trial in range(count):
        n = int(rng.integers(2, 7))
        c = np.exp(rng.uniform(-3, 3, size=(n,) * 3))
        c[rng.random(c.shape) < 0.35] = 0
        s = np.zeros_like(c)
        selected = rng.choice(
            c.size, size=int(rng.integers(1, min(5, c.size))), replace=False
        )
        s.ravel()[selected] = np.exp(rng.uniform(-3, 4, size=len(selected)))
        random = rng.normal(size=(5, 3))
        random /= np.linalg.norm(random, axis=1)[:, None]
        dirs = np.concatenate([axes_and_diagonals(), random]) if trial % 2 else random
        periodic = bool(trial % 3 == 0)
        duration = float(rng.uniform(0.05, 2))
        horizon = float(rng.uniform(0.1, 2.5 * n))
        row = compare_case(
            s, c, dirs, periodic=periodic, duration=duration, maximum_distance=horizon
        )
        rows.append(
            {
                "trial": trial,
                "n": n,
                "sources": len(selected),
                "angles": len(dirs),
                "periodic": periodic,
                "duration": duration,
                "horizon": horizon,
                **row,
            }
        )
    return rows


def scale_and_tie_cases() -> list:
    """Probe unit scaling, exact/near event ties, and disparate local capacities."""
    rows = []
    dirs = axes_and_diagonals()[:6]
    n = 4
    base_source = np.zeros((n,) * 3)
    base_source[1, 1, 1] = 12
    base_source[2, 1, 1] = 12
    for tie_offset in [0.0, -1e-13, 1e-13, 1e-8]:
        base_capacity = np.full_like(base_source, 5.0)
        base_capacity[1, 1, 1] = 1
        base_capacity[2, 1, 1] = 1 + tie_offset
        for factor in [1e-120, 1.0, 1e120]:
            rows.append(  # noqa: PERF401 -- retain sequential named validation cases
                {
                    "case": "near_tie_and_unit_scale",
                    "factor": factor,
                    "tie_offset": tie_offset,
                    **compare_case(
                        base_source * factor,
                        base_capacity * factor,
                        dirs,
                        duration=0.73,
                        maximum_distance=6.73,
                    ),
                }
            )
    rng = np.random.default_rng(996)
    for trial in range(12):
        c = np.exp(rng.uniform(-25, 25, size=(3,) * 3))
        s = np.zeros_like(c)
        s[0, 0, 0] = 1e-3
        s[2, 2, 2] = 1e3
        c[0, 0, 0] = 0
        rows.append(
            {
                "case": "mixed_capacity_scales",
                "trial": trial,
                **compare_case(s, c, dirs, maximum_distance=6.8),
            }
        )
    return rows


def history_cases() -> dict:
    """Compare piecewise histories and constant-rate splitting independently."""
    import event_front

    n = 5
    dirs = axes_and_diagonals()[:6]
    capacity = np.full((n,) * 3, 100.0)
    capacity[1, 2, 2], capacity[2, 2, 2] = 6.0, 12.0
    first, second = np.zeros_like(capacity), np.zeros_like(capacity)
    first[1, 2, 2], second[2, 2, 2] = 6.0, 6.0
    histories = {
        "constant_one_step": [(first + second, 2.0)],
        "constant_split": [(first + second, t) for t in [0.37, 0.23, 0.41, 0.99]],
        "A_then_B": [(2 * first, 1.0), (2 * second, 1.0)],
        "B_then_A": [(2 * second, 1.0), (2 * first, 1.0)],
        "on_off_on": [
            (first + second, 0.31),
            (np.zeros_like(first), 0.22),
            (2 * second, 0.51),
            (first + 0.1 * second, 0.96),
        ],
    }
    fields, rows = {}, {}
    for name, history in histories.items():
        reference_residual = capacity.copy()
        actual_residual = capacity.copy()
        reference_total, actual_total = np.zeros_like(capacity), np.zeros_like(capacity)
        source_reference, source_actual = (
            np.zeros_like(capacity),
            np.zeros_like(capacity),
        )
        max_interval_error = 0.0
        for source, duration in history:
            ref = reference_solve(source, reference_residual, dirs, duration=duration)
            out = event_front.solve(
                source, actual_residual, ray_directions=dirs, duration=duration
            )
            max_interval_error = max(
                max_interval_error,
                float(np.max(np.abs(out["absorbed"] - ref["absorbed"]))),
            )
            reference_total += ref["absorbed"]
            actual_total += out["absorbed"]
            source_reference += ref["source_absorbed"]
            source_actual += out["source_absorbed"]
            reference_residual = np.maximum(capacity - reference_total, 0)
            actual_residual = np.maximum(capacity - actual_total, 0)
        error = float(np.max(np.abs(reference_total - actual_total)))
        rows[name] = {
            "passed": error <= 2e-10 and max_interval_error <= 2e-10,
            "max_cell_error": error,
            "max_interval_error": max_interval_error,
            "max_source_absorption_error": float(
                np.max(np.abs(source_reference - source_actual))
            ),
        }
        fields[name] = actual_total
    split_error = float(
        np.max(np.abs(fields["constant_one_step"] - fields["constant_split"]))
    )
    reversal_l1 = float(np.abs(fields["A_then_B"] - fields["B_then_A"]).sum())
    return {
        "histories": rows,
        "constant_split_max_cell_error": split_error,
        "history_reversal_atom_l1": reversal_l1,
        "history_reversal_expected_to_differ": True,
        "passed": all(row["passed"] for row in rows.values()) and split_error <= 2e-10,
    }


def horizon_error_cases() -> dict:
    """Verify the retained-budget bound under prefix-preserving path extension."""
    import event_front

    source = np.zeros((3,) * 3)
    capacity = np.zeros_like(source)
    source[0, 1, 1] = 2
    capacity[0, 1, 1], capacity[1, 1, 1] = 1, 2
    fixtures = [
        (
            "bound_saturating_chain",
            source,
            capacity,
            np.array([[1.0, 0.0, 0.0]]),
            True,
            0.25,
            1.1,
        )
    ]
    rng = np.random.default_rng(65152)
    for trial in range(24):
        n = int(rng.integers(3, 6))
        capacity = np.exp(rng.uniform(-2, 2, size=(n,) * 3))
        capacity[rng.random(capacity.shape) < 0.6] = 0
        source = np.zeros_like(capacity)
        active = rng.choice(capacity.size, size=3, replace=False)
        source.ravel()[active] = rng.uniform(2, 20, size=3)
        capacity.ravel()[active] = 0
        random = rng.normal(size=(5, 3))
        random /= np.linalg.norm(random, axis=1)[:, None]
        directions = np.concatenate([axes_and_diagonals(), random])
        short = float(rng.uniform(0.2, n))
        fixtures.append(
            (
                f"random_{trial}",
                source,
                capacity,
                directions,
                bool(trial % 2),
                short,
                2.3 * short + 0.1,
            )
        )
    rows = []
    for name, source, capacity, directions, periodic, short, long in fixtures:
        outputs = []
        reference_error = 0.0
        for horizon in (short, long):
            out = event_front.solve(
                source,
                capacity,
                ray_directions=directions,
                periodic=periodic,
                maximum_distance=horizon,
            )
            reference = reference_solve(
                source,
                capacity,
                directions,
                periodic=periodic,
                maximum_distance=horizon,
            )
            reference_error = max(
                reference_error,
                float(np.max(np.abs(out["absorbed"] - reference["absorbed"]))),
            )
            outputs.append(out)
        small, large = outputs
        emitted = float(source.sum())
        retained = float(small["source_retained"].sum())
        delta = large["absorbed"] - small["absorbed"]
        minimum = float(delta.min())
        increase = float(delta.sum())
        escaped_increase = float(
            large["source_escaped"].sum() - small["source_escaped"].sum()
        )
        tolerance = 2e-11 * emitted
        passed = (
            minimum >= -tolerance
            and increase <= retained + tolerance
            and escaped_increase >= -tolerance
            and reference_error <= tolerance
        )
        rows.append(
            {
                "case": name,
                "periodic": periodic,
                "short_horizon": short,
                "long_horizon": long,
                "passed": bool(passed),
                "short_retained_fraction": retained / emitted,
                "long_retained_fraction": float(large["source_retained"].sum())
                / emitted,
                "additional_absorption_fraction": increase / emitted,
                "atom_l1_fraction": float(np.abs(delta).sum()) / emitted,
                "minimum_cell_change_fraction": minimum / emitted,
                "escape_increase_fraction": escaped_increase / emitted,
                "bound_excess_fraction": max(0.0, increase - retained) / emitted,
                "reference_max_cell_error_fraction": reference_error / emitted,
            }
        )
    assert abs(rows[0]["short_retained_fraction"] - 0.5) < 1e-14
    assert abs(rows[0]["additional_absorption_fraction"] - 0.5) < 1e-14
    return {
        "cases": len(rows),
        "passed": all(row["passed"] for row in rows),
        "results": rows,
        "scope": "Fixed source history, no recombination, nested ray paths",
    }


def run_validation(count: int = 100) -> dict:
    """Run the finite reference comparison and retain every failed fixture label."""
    before = hashes()
    groups = {
        "special": special_cases(),
        "random": randomized_cases(count),
        "scales_and_ties": scale_and_tie_cases(),
    }
    history = history_cases()
    horizon_error = horizon_error_cases()
    after = hashes()
    summaries = {}
    for name, rows in groups.items():
        summaries[name] = {
            "cases": len(rows),
            "passed": sum(row["passed"] for row in rows),
            "maximum_cell_error_per_emitted_photon": max(
                row["max_cell_error_per_emitted_photon"] for row in rows
            ),
            "maximum_source_error_per_own_emitted_budget": max(
                row["max_source_error_per_own_emitted_budget"] for row in rows
            ),
            "failures": [row for row in rows if not row["passed"]],
        }
    return {
        "scope": "Independent geometric path and temporal reference for the finite ray/voxel model",
        "source_hashes_before": before,
        "source_hashes_after": after,
        "source_unchanged_during_run": before == after,
        "initial_failure_hashes": INITIAL_FAILURE_HASHES,
        "long_double_mantissa_bits": int(np.finfo(LD).nmant),
        "horizon_convention": "Opaque entry<horizon; open exit<=horizon; periodic exit never escapes",
        "groups": summaries,
        "history": history,
        "horizon_error_bound": horizon_error,
        "all_passed": before == after
        and all(group["cases"] == group["passed"] for group in summaries.values())
        and history["passed"]
        and horizon_error["passed"],
        "special_case_results": groups["special"],
        "initial_minimal_failures": [
            {
                "case": "weak_local_source",
                "source": 0.001,
                "capacity": 1e12,
                "old_cell_absorbed": 0.0009765625,
                "reference_absorbed": 0.001,
                "relative_emitted_photon_deficit": 0.0234375,
            },
            {
                "case": "transparent_n1_open_box",
                "horizon": 0.5 + 5e-11,
                "old_escaped": 0.0,
                "old_retained": 1.0,
                "reference_escaped": 1.0,
            },
            {
                "case": "near_corner_opaque_sliver",
                "n": 5,
                "source_cell": [1, 1, 1],
                "source_rate": 1,
                "raw_direction": [1.0, 1.0 + 1e-10, 0.0],
                "maximum_distance": 3.0,
                "opaque_cells_capacity_100": [[1, 2, 1], [2, 2, 1]],
                "old_absorber": [2, 2, 1],
                "reference_absorber": [1, 2, 1],
                "global_ledger_was_still_closed": True,
            },
        ],
    }


def main() -> None:
    """Write results even when a comparison fails, then signal that failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-cases", type=int, default=100)
    parser.add_argument(
        "--output", type=Path, default=HERE / "validation_event_reference_results.json"
    )
    args = parser.parse_args()
    result = run_validation(args.random_cases)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    summary = {
        name: {"cases": row["cases"], "passed": row["passed"]}
        for name, row in result["groups"].items()
    }
    print(  # noqa: T201
        json.dumps(
            {
                "all_passed": result["all_passed"],
                "groups": summary,
                "history_passed": result["history"]["passed"],
            }
        )
    )
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
