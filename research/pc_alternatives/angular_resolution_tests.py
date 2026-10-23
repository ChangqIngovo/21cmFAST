"""Resolution and frozen-source probes of active higher-order angular transport."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import angular_high_order
import numpy as np
from angular_hybrid import angular_directions
from angular_tests import budget_checks, sphere_voxel_reference


def resolution(count=96):
    """Scale radius at fixed R-to-flight ratio; report angular resolution fixed."""
    report = []
    for n, radius in ((24, 4.), (40, 8.), (64, 12.)):
        source = np.zeros((n,)*3)
        center = np.full(3, n//2)
        photons = 4*np.pi/3*radius**3
        source[tuple(center)] = photons
        target = sphere_voxel_reference(n, radius, center)
        directions, weights = angular_directions(count)
        flight = 4*radius
        max_l1 = np.max(abs(directions).sum(axis=1))
        steps = int(np.ceil(flight*max_l1/.45))
        result = angular_high_order.solve(source, np.ones_like(source), directions=directions,
                                          weights=weights, steps=steps,
                                          cfl=flight*max_l1/steps, periodic=False)
        budget_checks(result, np.ones_like(source))
        report.append({"grid": n, "radius_cells": radius, "fixed_directions": count,
                       "sphere_voxel_L1_over_photons": float(abs(result["absorbed"]-target).sum()/photons),
                       "reference_voxel_volume_relative_error": float(target.sum()/photons-1),
                       **result["diagnostics"]})
    return report


def frozen_fixture(path, count=48, flight=16.):
    """Inject saved sources as a toy flash, not a Qin20 history or time increment."""
    data = np.load(path)
    source, capacity = data["source"], data["capacity"]
    directions, weights = angular_directions(count)
    max_l1 = np.max(abs(directions).sum(axis=1))
    steps = int(np.ceil(flight*max_l1/.45))
    report = []
    for scale in (.1, 1.):
        result = angular_high_order.solve(source*scale, capacity, directions=directions,
                                          weights=weights, steps=steps,
                                          cfl=flight*max_l1/steps, periodic=True)
        budget_checks(result, capacity)
        report.append({"source_scale": scale, "grid": source.shape[0], "directions": count,
                       "input_photons_over_capacity": float(source.sum()*scale/capacity.sum()),
                       "source_cells_above_local_capacity": int((scale*source > capacity).sum()),
                       **result["diagnostics"]})
    return {"source_fixture": str(path),
            "scope": "Saved Qin20-derived source/capacity arrays reused as synthetic simultaneous photon increment; no redshift history, feedback, recombinations or cosmological validation",
            "cases": report}


def main():
    """Run optional resolution and native-fixture kernel exercises."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("angular_resolution_results.json"))
    parser.add_argument("--counts", type=int, default=96)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--fixture-only", action="store_true")
    args = parser.parse_args()
    report = {"scope": "Angular transport research, separate from production ES and global-budget modes"}
    if not args.fixture_only:
        report["resolution"] = resolution(args.counts)
    if args.fixture:
        report["frozen_fixture"] = frozen_fixture(args.fixture)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))  # noqa: T201


if __name__ == "__main__":
    main()
