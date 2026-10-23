"""Falsifiable correctness, morphology and cost probes for angular_hybrid.

Run with ``python angular_tests.py --output angular_results.json``. These are
toy transport tests, never a claim of production 21cmFAST equivalence or speed.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from angular_hybrid import overlap_quadrature, solve
from scipy import fft


def budget_checks(result, capacity):
    """Assert global/angular ledgers and local positivity/capacity bounds."""
    d = result["diagnostics"]
    scale = max(d["emitted"], 1)
    assert abs(d["ledger_residual"]) / scale < 2e-12, d
    assert d["maximum_angle_residual"] / scale < 2e-12, d
    assert np.min(result["absorbed"]) >= 0
    assert np.max(result["absorbed"]-capacity) <= 2e-12
    assert np.min(result["radiation"]) >= -2e-15
    return d


def radial_reference(n, photon_count, center):
    """Exact-count radial ranking diagnostic; not exact sphere/voxel integration."""
    coordinates = np.moveaxis(np.indices((n,)*3), 0, -1)-center
    radius2 = (coordinates**2).sum(axis=-1)
    target = np.zeros_like(radius2, dtype=float)
    remaining = photon_count
    for shell in np.unique(radius2):
        mask = radius2 == shell
        take = min(remaining, mask.sum())
        target[mask] = take/mask.sum()
        remaining -= take
        if remaining <= 0:
            break
    return target, coordinates, np.sqrt(radius2)


def sphere_voxel_reference(n, radius, center, quadrature=24):
    """Integrate a sphere in boundary voxels, using midpoint x/y and exact z."""
    coordinates = np.moveaxis(np.indices((n,)*3), 0, -1)-center
    distance = np.linalg.norm(coordinates, axis=-1)
    half_diagonal = np.sqrt(3)/2
    result = np.zeros((n,)*3)
    result[distance < radius-half_diagonal] = 1
    boundary = abs(distance-radius) <= half_diagonal
    points = coordinates[boundary]
    offsets = (np.arange(quadrature)+.5)/quadrature-.5
    x = points[:, 0, None, None]+offsets[None, :, None]
    y = points[:, 1, None, None]+offsets[None, None, :]
    z = points[:, 2, None, None]
    height = np.sqrt(np.maximum(0, radius**2-x*x-y*y))
    length = np.maximum(0, np.minimum(height, z+.5)-np.maximum(-height, z-.5))
    result[boundary] = length.mean(axis=(1, 2))
    return result


def invariants():
    """Check simple limits, a nonuniform case, translation and threading."""
    n = 12
    rng = np.random.default_rng(418)
    cap = rng.uniform(.2, 2.1, (n,)*3)
    source = rng.exponential(.2, (n,)*3)*(rng.random((n,)*3) < .08)
    source[4, 5, 6] += 200
    result = solve(source, cap, steps=24, directions=24)
    report = {"nonuniform": budget_checks(result, cap)}
    zero = solve(np.zeros_like(cap), cap, steps=4)
    budget_checks(zero, cap)
    assert not np.any(zero["absorbed"])
    weak = np.zeros_like(cap)
    weak[5, 6, 7] = .3
    partial = solve(weak, np.ones_like(cap), steps=4)
    budget_checks(partial, np.ones_like(cap))
    assert np.allclose(partial["absorbed"], weak, rtol=0, atol=1e-15)
    report["weak_partial_ionization"] = float(partial["absorbed"][5, 6, 7])
    transparent = solve(source, np.zeros_like(cap), steps=12)
    budget_checks(transparent, np.zeros_like(cap))
    assert not np.any(transparent["absorbed"])
    report["transparent"] = transparent["diagnostics"]
    shift = (2, -3, 4)
    moved = solve(np.roll(source, shift, axis=(0, 1, 2)),
                  np.roll(cap, shift, axis=(0, 1, 2)), steps=24)
    translation_error = np.max(abs(moved["absorbed"]-
                                   np.roll(result["absorbed"], shift, axis=(0, 1, 2))))
    assert translation_error < 1e-14
    report["translation_max_cell_error"] = float(translation_error)
    single = solve(source, cap, steps=24, threads=1)
    report["threads_1_vs_4_max_cell_error"] = float(np.max(abs(single["absorbed"]-result["absorbed"])))
    assert report["threads_1_vs_4_max_cell_error"] == 0
    # A scalar source map has no source-list order. Split/coincident sources
    # combine linearly at injection; no independent per-source arrays are stored.
    report["source_order"] = "source contributions summed before transport; order-free to summation precision"
    report["all_assertions_passed"] = True
    return report


def beam_tests():
    """Compare axis walls, half spaces and a shadow sensitive to diffusion."""
    n = 32
    source = np.zeros((n,)*3)
    source[3, 16, 16] = 1
    capacity = np.zeros_like(source)
    capacity[15, :, :] = 1e6
    wall = solve(source, capacity, directions=np.array([[1., 0, 0]]),
                 steps=40, cfl=.9, periodic=False)
    budget_checks(wall, capacity)
    beyond = wall["radiation"][16:, ...].sum()+wall["absorbed"][16:, ...].sum()
    assert beyond == 0
    assert wall["absorbed"][15].sum() > .999999
    hemispheres = []
    source = np.zeros((n,)*3)
    source[13, 16, 16] = 1000
    for count in (12, 24, 48, 96):
        isotropic = solve(source, capacity, directions=count, steps=96, periodic=False)
        budget_checks(isotropic, capacity)
        absorbed_fraction = float(isotropic["absorbed"][15].sum()/1000)
        assert absorbed_fraction <= .5+1e-13
        hemispheres.append({"directions": count, "wall_absorbed_fraction": absorbed_fraction,
                            "hemisphere_upper_bound": .5,
                            "pending_fraction": isotropic["diagnostics"]["pending"]/1000,
                            "boundary": "vacuum escape; finite plane spans open box"})
    # A diagonal collimated pulse has zero transverse variance in exact free
    # streaming. Donor-cell stencil splits it across x/y paths, introducing a
    # calculable nonphysical transverse variance despite exact photon counts.
    source = np.zeros((48,)*3)
    source[3, 3, 24] = 1
    diagonal = np.array([[1., 1, 0]])/np.sqrt(2)
    steps = 24
    free = solve(source, np.zeros_like(source), directions=diagonal,
                 steps=steps, cfl=.8, periodic=False)
    budget_checks(free, np.zeros_like(source))
    x, y, _ = np.indices(source.shape)
    transverse = (x-y)/np.sqrt(2)
    measured = float((free["radiation"][..., 0]*transverse**2).sum())
    theory = steps*free["diagnostics"]["dt"]/np.sqrt(2)
    assert np.isclose(measured, theory, rtol=2e-14)
    # A small oblique screen blocking the exact ray allows numerically diffused
    # radiation around its edges: this is an intentionally failing shadow probe.
    capacity = np.zeros_like(source)
    plane = x+y == 24
    capacity[plane & (abs(x-y) <= 4)] = 1e6
    shadow = solve(source, capacity, directions=diagonal,
                   steps=36, cfl=.8, periodic=False)
    budget_checks(shadow, capacity)
    shadow_leak = float(shadow["radiation"][x+y > 24].sum())
    assert shadow_leak > .01
    return {
        "axis_aligned_infinite_wall": {**wall["diagnostics"], "photons_beyond_wall": float(beyond)},
        "isotropic_half_space_screen": hemispheres,
        "diagonal_beam": {"transverse_variance_cells_squared": measured,
                          "stencil_prediction": theory, "exact_transport_value": 0.,
                          "rms_transverse_width_cells": float(np.sqrt(measured)),
                          "elapsed_light_distance_cells": free["diagnostics"]["elapsed_time"]},
        "finite_oblique_screen": {"fraction_leaking_past_screen": shadow_leak,
                                  "exact_collimated_transport_leak": 0.,
                                  "conclusion": "Fails accurate shadows at this resolution because spatial diffusion bypasses screen edges"},
    }


def sphere_tests():
    """Measure finite-angle morphology and identical-source inversion symmetry."""
    report = []
    n, radius = 32, 6.
    source = np.zeros((n,)*3)
    center = np.full(3, n//2)
    photons = 4*np.pi/3*radius**3
    source[tuple(center)] = photons
    target, coord, rad = radial_reference(n, photons, center)
    for count in (12, 24, 48, 96):
        result = solve(source, np.ones_like(source), directions=count,
                       steps=72, periodic=False)
        d = budget_checks(result, np.ones_like(source))
        ion = result["absorbed"]
        moment = np.einsum("pi,pj,p->ij", coord.reshape(-1, 3),
                           coord.reshape(-1, 3), ion.ravel())/ion.sum()
        eig = np.linalg.eigvalsh(moment)
        report.append({**d, "radius_cells": radius,
                       "radial_ranking_L1_over_photons": float(abs(ion-target).sum()/photons),
                       "ionization_outside_R_plus_2_fraction": float(ion[rad > radius+2].sum()/photons),
                       "moment_largest_to_smallest_axis_ratio": float(np.sqrt(eig[-1]/eig[0])),
                       "partially_ionized_cells": int(((ion > 1e-6) & (ion < 1-1e-6)).sum())})
    # Identical sources symmetric about box origin; inversion must interchange
    # antipodal directions and leave scalar absorption unchanged.
    source = np.zeros((n,)*3)
    q = 4*np.pi/3*4**3
    source[13, 16, 16] = q
    source[19, 16, 16] = q
    pair = solve(source, np.ones_like(source), steps=72, directions=48)
    budget_checks(pair, np.ones_like(source))
    opposite = (-np.arange(n)) % n
    inversion = pair["absorbed"][np.ix_(opposite, opposite, opposite)]
    symmetry_error = float(abs(pair["absorbed"]-inversion).max())
    assert symmetry_error < 1e-12
    return {"single_sphere": report,
            "two_sources_separation_1p5_R": {**pair["diagnostics"], "inversion_max_cell_error": symmetry_error},
            "reference_caveat": "Radial ranking is an exact-count discrete shape diagnostic, not a voxel-integrated analytic sphere"}


def fft_filters(n, threads=4, filters=12, repeats=3):
    """Time one forward+12 scalar spherical filters/inverse FFTs, no source model."""
    data = np.random.default_rng(41).random((n,)*3)
    kx = 2*np.pi*fft.fftfreq(n)
    kz = 2*np.pi*fft.rfftfreq(n)
    magnitude = np.sqrt(kx[:, None, None]**2+kx[None, :, None]**2+kz[None, None, :]**2)
    kernels = []
    for radius in np.geomspace(1, n/4, filters):
        kr = magnitude*radius
        window = np.ones_like(kr)
        nz = kr != 0
        window[nz] = 3*(np.sin(kr[nz])-kr[nz]*np.cos(kr[nz]))/kr[nz]**3
        kernels.append(window)
    times = []
    for _ in range(repeats+1):
        started = time.perf_counter()
        transform = fft.rfftn(data, workers=threads)
        checksum = 0.
        for kernel in kernels:
            filtered = fft.irfftn(transform*kernel, s=data.shape, workers=threads)
            checksum += filtered[0, 0, 0]
        times.append(time.perf_counter()-started)
    return {"seconds_median": float(np.median(times[1:])), "seconds_samples": times[1:],
            "forward_FFTs": 1, "inverse_FFTs": filters, "threads": threads,
            "checksum": float(checksum), "kernel_construction_timed": False,
            "source_integrals_and_thermal_history_included": False}


def performance():
    """Measure explicit transport against a scalar-filter kernel microbenchmark."""
    cases = []
    for n in (32, 64):
        baseline = fft_filters(n)
        source = np.zeros((n,)*3)
        source[n//2, n//2, n//2] = 4*np.pi/3*(n/6)**3
        capacity = np.ones_like(source)
        for count in (24, 48):
            runs = [solve(source, capacity, directions=count, steps=n,
                          threads=4) for _ in range(2)]
            for result in runs:
                budget_checks(result, capacity)
            timing = float(np.median([r["diagnostics"]["seconds"] for r in runs]))
            cases.append({"grid": n, "directions": count, "transport_steps": n,
                          "transport_seconds_median": timing,
                          "transport_seconds_samples": [r["diagnostics"]["seconds"] for r in runs],
                          "scalar_filter_reference": baseline,
                          "ratio_to_scalar_filter_kernel_only": timing/baseline["seconds_median"],
                          "transport_working_bytes": runs[0]["diagnostics"]["working_array_bytes"],
                          "absorbed_fraction": runs[0]["diagnostics"]["absorbed"]/source.sum(),
                          "whole_21cmFAST_overhead": "Not measured"})
    return cases


def main():
    """Run the probes and save their results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("angular_results.json"))
    parser.add_argument("--skip-performance", action="store_true")
    args = parser.parse_args()
    results = {"scope": "Toy photon-counting discrete-ordinate transport; no Qin20 simulation and no production changes",
               "environment": {"python": platform.python_version(), "numpy": np.__version__,
                               "scipy": scipy.__version__, "machine": platform.machine()},
               "invariants": invariants(), "beams_and_screens": beam_tests(),
               "sphere_morphology": sphere_tests(), "one_eighth_cone": overlap_quadrature()}
    if not args.skip_performance:
        results["performance"] = performance()
    args.output.write_text(json.dumps(results, indent=2)+"\n")
    print(json.dumps({"output": str(args.output), "invariants_pass": results["invariants"]["all_assertions_passed"],  # noqa: T201
                      "screen_leak": results["beams_and_screens"]["finite_oblique_screen"]["fraction_leaking_past_screen"],
                      "performance": results.get("performance", [])}, indent=2))


if __name__ == "__main__":
    main()
