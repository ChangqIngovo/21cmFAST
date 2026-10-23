"""Independent small-graph audit of fixed-ray sharp-front event timing.

This deliberately uses an exhaustive event scan, not the production candidate's
priority queue or geometric DDA. Paths are prescribed ordered cell lists.
It verifies the temporal model and cannot verify 3D ray geometry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


class GraphFront:
    """Evolve constant-rate beams through opaque-until-full graph cells."""

    def __init__(self, capacity: list, paths: list, rates: list):
        self.capacity = np.asarray(capacity, dtype=float)
        self.paths = tuple(tuple(path) for path in paths)
        self.rates = np.asarray(rates, dtype=float)
        self.absorption = np.zeros((len(paths), len(capacity)))
        self.escaped = np.zeros(len(paths))
        self.cursor = np.zeros(len(paths), dtype=int)
        self.time = 0.0
        self.events = 0
        assert len(paths) == len(rates)
        assert np.all(self.capacity >= 0)
        assert np.all(self.rates >= 0)

    @property
    def absorbed(self) -> np.ndarray:
        """Return actual atom absorption summed across source beams."""
        return self.absorption.sum(axis=0)

    def _targets(self) -> np.ndarray:
        residual = self.capacity - self.absorbed
        tolerance = 64 * np.finfo(float).eps * max(1.0, self.capacity.max())
        targets = np.full(len(self.paths), -1, dtype=int)
        for beam, path in enumerate(self.paths):
            while self.cursor[beam] < len(path):
                cell = path[self.cursor[beam]]
                if residual[cell] > tolerance:
                    targets[beam] = cell
                    break
                self.cursor[beam] += 1
        return targets

    def advance(self, duration: float) -> None:
        """Advance the exact piecewise-linear temporal model by duration."""
        end = self.time + duration
        while self.time < end:
            targets = self._targets()
            active = targets >= 0
            rate = np.bincount(
                targets[active],
                weights=self.rates[active],
                minlength=self.capacity.size,
            ).astype(float)
            hit = rate > 0
            if np.any(hit):
                residual = self.capacity - self.absorbed
                until = np.full_like(rate, np.inf)
                until[hit] = residual[hit] / rate[hit]
                next_event = float(until.min())
            else:
                until = np.full_like(rate, np.inf)
                next_event = np.inf
            step = min(end - self.time, next_event)
            assert step > 0
            for beam in np.flatnonzero(active):
                self.absorption[beam, targets[beam]] += self.rates[beam] * step
            self.escaped[~active] += self.rates[~active] * step
            self.time += step
            if next_event <= step:
                # Exact simultaneous events exhaust together. Numerical
                # residuals are not force-filled: their bounded size is
                # exposed by the accounting checks below.
                self.events += int(np.count_nonzero(until == next_event))
        self.assert_ledger()

    def advance_batch(self, duration: float) -> None:
        """Apply the rejected full-budget synchronous hit-round update."""
        pending = self.rates * duration
        while np.any(pending > 0):
            targets = self._targets()
            escaped = targets < 0
            self.escaped[escaped] += pending[escaped]
            pending[escaped] = 0
            active = (targets >= 0) & (pending > 0)
            if not np.any(active):
                break
            offer = np.bincount(
                targets[active],
                weights=pending[active],
                minlength=self.capacity.size,
            )
            residual = self.capacity - self.absorbed
            fraction = np.zeros_like(offer)
            receives = offer > 0
            fraction[receives] = np.minimum(1, residual[receives] / offer[receives])
            for beam in np.flatnonzero(active):
                amount = pending[beam] * fraction[targets[beam]]
                self.absorption[beam, targets[beam]] += amount
                pending[beam] -= amount
        self.time += duration
        self.assert_ledger()

    def assert_ledger(self) -> None:
        """Check each beam, each target, and the complete photon count."""
        tolerance = 1e-11 * max(1.0, self.rates.sum() * self.time)
        np.testing.assert_allclose(
            self.absorption.sum(axis=1) + self.escaped,
            self.rates * self.time,
            rtol=0,
            atol=tolerance,
        )
        assert np.all(self.absorbed <= self.capacity + tolerance)
        assert np.all(self.absorption >= -tolerance)


def delayed_shared_cell() -> dict:
    """Compare the event solution with one and two full-budget batches."""
    fixture = ([1, 2, 100, 100], [[0, 1, 2], [1, 3]], [1, 1])
    exact = GraphFront(*fixture)
    exact.advance(2)
    batch = GraphFront(*fixture)
    batch.advance_batch(2)
    split = GraphFront(*fixture)
    split.advance_batch(1)
    split.advance_batch(1)
    np.testing.assert_allclose(exact.absorbed, [1, 2, 0.5, 0.5])
    np.testing.assert_allclose(batch.absorbed, [1, 2, 1, 0])
    np.testing.assert_allclose(split.absorbed, exact.absorbed)
    return {
        "cells": ["P", "C", "X", "Y"],
        "beam_paths": [["P", "C", "X"], ["C", "Y"]],
        "rates": [1, 1],
        "duration": 2,
        "capacities": fixture[0],
        "event_result": exact.absorbed.tolist(),
        "one_batch_result": batch.absorbed.tolist(),
        "two_batch_result": split.absorbed.tolist(),
        "one_batch_atom_l1_error": float(np.abs(batch.absorbed - exact.absorbed).sum()),
        "exact_absorption_by_beam_and_cell": exact.absorption.tolist(),
        "event_count": exact.events,
    }


def event_semigroup_and_ties() -> dict:
    """Test stateful time splitting, beam permutation and exact event ties."""
    fixture = ([1, 2, 100, 100], [[0, 1, 2], [1, 3]], [1, 1])
    whole = GraphFront(*fixture)
    whole.advance(2)
    split = GraphFront(*fixture)
    partitions = [0.37, 0.23, 0.41, 0.99]
    for duration in partitions:
        split.advance(duration)
    permuted = GraphFront(fixture[0], fixture[1][::-1], fixture[2][::-1])
    permuted.advance(2)
    ties = GraphFront([1, 1, 100, 100], [[0, 1, 2], [1, 3]], [1, 1])
    ties.advance(2)
    np.testing.assert_allclose(split.absorption, whole.absorption, rtol=0, atol=2e-14)
    np.testing.assert_allclose(permuted.absorption[::-1], whole.absorption)
    np.testing.assert_allclose(ties.absorbed, [1, 1, 1, 1])
    return {
        "partition_durations": partitions,
        "split_max_absorption_difference": float(
            np.max(np.abs(split.absorption - whole.absorption))
        ),
        "permutation_max_absorption_difference": float(
            np.max(np.abs(permuted.absorption[::-1] - whole.absorption))
        ),
        "simultaneous_event_result": ties.absorbed.tolist(),
        "simultaneous_event_count": ties.events,
    }


def six_axis_spatial_fixture() -> dict:
    """Embed the timing failure in two six-axis voxel-centered sources."""
    coordinates = [
        [0, 0, 0],
        [1, 0, 0],
        [2, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
        [1, 1, 0],
        [1, -1, 0],
        [1, 0, 1],
        [1, 0, -1],
    ]
    paths = [
        [0, 1, 2],
        [0, 3],
        [0, 4],
        [0, 5],
        [0, 6],
        [0, 7],
        [1, 2],
        [1, 0, 3],
        [1, 8],
        [1, 9],
        [1, 10],
        [1, 11],
    ]
    capacity = [6, 12] + [100] * 10
    exact = GraphFront(capacity, paths, [1] * 12)
    exact.advance(2)
    batch = GraphFront(capacity, paths, [1] * 12)
    batch.advance_batch(2)
    expected = [6, 12, 2 / 7, 8 / 7] + [1] * 4 + [1 / 7] * 4
    np.testing.assert_allclose(exact.absorbed, expected)
    np.testing.assert_allclose(batch.absorbed, [6, 12] + [1] * 6 + [0] * 4)
    return {
        "cell_coordinates": coordinates,
        "capacity": capacity,
        "source_positions": coordinates[:2],
        "total_rate_per_source": 6,
        "directions": "+x,-x,+y,-y,+z,-z with equal 1/6 weights",
        "duration": 2,
        "source_A_cell_opening_time": 1,
        "source_B_cell_opening_time": 13 / 7,
        "event_result": exact.absorbed.tolist(),
        "batch_result": batch.absorbed.tolist(),
        "batch_atom_l1_error": float(np.abs(batch.absorbed - exact.absorbed).sum()),
        "scope": "six-axis angular discretization; ordered paths are straight voxel rays",
    }


def batch_refinement() -> dict:
    """Demonstrate nonmonotonic refinement error in the rejected batch model."""
    fixture = ([0.91, 1.07, 100, 100], [[0, 1, 2], [1, 3]], [1, 1])
    exact = GraphFront(*fixture)
    exact.advance(2)
    results = []
    for count in [1, 2, 3, 4, 7, 16, 64, 256]:
        batch = GraphFront(*fixture)
        for _ in range(count):
            batch.advance_batch(2 / count)
        results.append(
            {
                "steps": count,
                "field": batch.absorbed.tolist(),
                "atom_l1_error": float(np.abs(batch.absorbed - exact.absorbed).sum()),
            }
        )
    assert results[2]["atom_l1_error"] > results[1]["atom_l1_error"]
    assert results[-1]["atom_l1_error"] < 1e-11
    return {"event_result": exact.absorbed.tolist(), "batch_refinement": results}


def weak_source_compression() -> dict:
    """Verify compact local absorption before a source-cell opening event."""
    weights = np.array([0.1, 0.2, 0.7])
    explicit = GraphFront([0.4, 100, 100, 100], [[0, 1], [0, 2], [0, 3]], weights)
    explicit.advance(1)
    activation_time = 0.4 / weights.sum()
    compressed_local = weights * activation_time
    compressed_downstream = weights * (1 - activation_time)
    np.testing.assert_allclose(explicit.absorption[:, 0], compressed_local)
    np.testing.assert_allclose(explicit.absorbed[1:], compressed_downstream)
    return {
        "angular_weights": weights.tolist(),
        "source_cell_capacity": 0.4,
        "source_cell_opening_time": float(activation_time),
        "implicit_per_angle_local_absorption": compressed_local.tolist(),
        "subsequent_downstream_absorption": compressed_downstream.tolist(),
        "scope": "opaque source cell; rays share that first absorber before activation",
    }


def random_graph_comparisons() -> dict:
    """Probe fixed-path monotonicity and temporal composition on small graphs."""
    rng = np.random.default_rng(70123)
    max_split_error = 0.0
    minimum_brightness_change = 0.0
    for _ in range(100):
        capacity = rng.uniform(0.1, 1.5, size=12)
        paths = [rng.choice(12, size=5, replace=False).tolist() for _ in range(6)]
        rates = rng.uniform(0.1, 1.5, size=6)
        low = GraphFront(capacity, paths, rates)
        low.advance(1.3)
        high_rates = rates.copy()
        high_rates[0] *= 2
        high = GraphFront(capacity, paths, high_rates)
        high.advance(1.3)
        split = GraphFront(capacity, paths, rates)
        for duration in [0.11, 0.43, 0.26, 0.5]:
            split.advance(duration)
        minimum_brightness_change = min(
            minimum_brightness_change, float(np.min(high.absorbed - low.absorbed))
        )
        max_split_error = max(
            max_split_error, float(np.max(np.abs(split.absorption - low.absorption)))
        )
        assert np.all(high.absorbed >= low.absorbed - 1e-12)
        np.testing.assert_allclose(split.absorption, low.absorption, rtol=0, atol=1e-12)
        assert low.events <= len(capacity)
    return {
        "seed": 70123,
        "fixtures": 100,
        "maximum_temporal_split_absorption_error": max_split_error,
        "minimum_brightness_change": minimum_brightness_change,
        "scope": "prescribed ordered paths; these do not all represent straight 3D rays",
    }


def main() -> None:
    """Execute the independent graph checks and save their results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("event_front_audit_results.json"),
    )
    args = parser.parse_args()
    probes = [
        delayed_shared_cell,
        event_semigroup_and_ties,
        six_axis_spatial_fixture,
        batch_refinement,
        weak_source_compression,
        random_graph_comparisons,
    ]
    results = {probe.__name__: probe() for probe in probes}
    results["all_assertions_passed"] = True
    results["scope"] = (
        "Temporal model audit; independent of spatial DDA and heap implementation"
    )
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"{len(probes)} event-front audits passed; wrote {args.output}")  # noqa: T201


if __name__ == "__main__":
    main()
