"""Compare positive MUSCL angular transport to donor-cell at equal flight time."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import angular_high_order
import angular_hybrid
import angular_tests
import numpy as np


def beam_comparison():
    """Measure diagonal beam broadening and finite-screen leakage."""
    n = 48
    source = np.zeros((n,)*3)
    source[3, 3, 24] = 1
    capacity = np.zeros_like(source)
    x, y, _ = np.indices(source.shape)
    transverse = (x-y)/np.sqrt(2)
    capacity[(x+y == 24) & (abs(x-y) <= 4)] = 1e6
    directions = np.array([[1., 1., 0]])/np.sqrt(2)
    results = {}
    for name, module, cfl, steps in (("donor_cell", angular_hybrid, .8, 36),
                                      ("MUSCL_MC", angular_high_order, .4, 72)):
        free = module.solve(source, np.zeros_like(source), directions=directions,
                            cfl=cfl, steps=steps, periodic=False)
        wall = module.solve(source, capacity, directions=directions, cfl=cfl,
                            steps=steps, periodic=False)
        angular_tests.budget_checks(free, np.zeros_like(source))
        angular_tests.budget_checks(wall, capacity)
        results[name] = {
            "flight_distance_cells": free["diagnostics"]["elapsed_time"],
            "transverse_variance_cells_squared": float((free["radiation"][..., 0]*transverse**2).sum()),
            "screen_leak_fraction": float(wall["radiation"][x+y > 24].sum()),
            "free_stream_seconds": free["diagnostics"]["seconds"],
            "screen_seconds": wall["diagnostics"]["seconds"],
            "screen_ledger_residual": wall["diagnostics"]["ledger_residual"],
        }
    results["exact_collimated_values"] = {"transverse_variance_cells_squared": 0., "screen_leak_fraction": 0.}
    return results


def isolated_spheres(counts=(24, 48, 96), n=32, radius=6., flight=24.):
    """Compare angular/spatial accuracy at the same nominal physical time."""
    source = np.zeros((n,)*3)
    center = np.full(3, n//2)
    photons = 4*np.pi/3*radius**3
    source[tuple(center)] = photons
    target, coordinates, rad = angular_tests.radial_reference(n, photons, center)
    voxel_reference = angular_tests.sphere_voxel_reference(n, radius, center)
    results = []
    for count in counts:
        directions, weights = angular_hybrid.angular_directions(count)
        max_l1 = np.max(abs(directions).sum(axis=1))
        # Set equal end times exactly, while respecting each solver's CFL.
        for name, module, max_cfl in (("donor_cell", angular_hybrid, .85),
                                        ("MUSCL_MC", angular_high_order, .45)):
            steps = int(np.ceil(flight*max_l1/max_cfl))
            cfl = flight*max_l1/steps
            result = module.solve(source, np.ones_like(source), directions=directions,
                                  weights=weights, steps=steps, cfl=cfl, periodic=False)
            angular_tests.budget_checks(result, np.ones_like(source))
            ion = result["absorbed"]
            moment = np.einsum("pi,pj,p->ij", coordinates.reshape(-1, 3),
                               coordinates.reshape(-1, 3), ion.ravel())/ion.sum()
            eigenvalues = np.linalg.eigvalsh(moment)
            results.append({"method": name, "grid": n, "radius_cells": radius,
                            "directions": count, **result["diagnostics"],
                            "radial_ranking_L1_over_photons": float(abs(ion-target).sum()/photons),
                            "sphere_voxel_L1_over_photons": float(abs(ion-voxel_reference).sum()/photons),
                            "reference_voxel_volume_relative_error": float(voxel_reference.sum()/photons-1),
                            "ionization_outside_R_plus_2_fraction": float(ion[rad > radius+2].sum()/photons),
                            "moment_largest_to_smallest_axis_ratio": float(np.sqrt(eigenvalues[-1]/eigenvalues[0]))})
    return results


def saturation_and_symmetry():
    """Exercise nonuniform invariants using the high-order backend."""
    previous = angular_tests.solve
    angular_tests.solve = angular_high_order.solve
    try:
        result = angular_tests.invariants()
    finally:
        angular_tests.solve = previous
    rng = np.random.default_rng(86)
    source = rng.random((8,)*3)*.2
    source[3, 4, 5] += 100
    capacity = rng.uniform(.1, 2, source.shape)
    active = angular_high_order.solve(source, capacity, steps=16)
    dense = angular_high_order.solve(source, capacity, steps=16, active_cells=False)
    result["active_vs_dense_absorbed_max_difference"] = float(abs(active["absorbed"]-dense["absorbed"]).max())
    result["active_vs_dense_radiation_max_difference"] = float(abs(active["radiation"]-dense["radiation"]).max())
    assert result["active_vs_dense_absorbed_max_difference"] == 0
    assert result["active_vs_dense_radiation_max_difference"] == 0
    source = np.zeros((4,)*3)
    source[2, 2, 2] = 1000
    saturated = angular_high_order.solve(source, np.ones_like(source), steps=32)
    angular_tests.budget_checks(saturated, np.ones_like(source))
    assert np.min(saturated["absorbed"]) == 1
    result["full_saturation"] = saturated["diagnostics"]
    return result


def main():
    """Save error-versus-cost measurements without changing production code."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("angular_high_order_results.json"))
    parser.add_argument("--beam-only", action="store_true")
    parser.add_argument("--counts", type=int, nargs="+", default=(24, 48, 96))
    parser.add_argument("--grid", type=int, default=32)
    parser.add_argument("--radius", type=float, default=6.)
    parser.add_argument("--flight", type=float, default=24.)
    args = parser.parse_args()
    results = {"scope": "Toy finite-volume angular transport; no Qin20 production run",
               "streaming": "MC reconstruction, SSPRK2; cell absorption at both stages; joint convex radiation/ionization average",
               "invariants": saturation_and_symmetry(), "beam_and_screen": beam_comparison()}
    if not args.beam_only:
        results["sphere_error_cost_curve"] = isolated_spheres(args.counts, args.grid, args.radius, args.flight)
    args.output.write_text(json.dumps(results, indent=2)+"\n")
    print(json.dumps(results, indent=2))  # noqa: T201


if __name__ == "__main__":
    main()
