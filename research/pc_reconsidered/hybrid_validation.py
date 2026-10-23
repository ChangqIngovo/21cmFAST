"""Reproduce spatial/time validation of adaptive sharp-front quadrature.

The continuum sphere is an independent spatial oracle. The time check uses
constant source rates and static capacities, not cosmological histories. Native
fields, when supplied, are only a frozen-field transport fixture.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import hybrid
from compare_models import sphere_reference


def distance(a, b, photons):
    return float(np.abs(a - b).sum() / photons)


def partition(source, capacity, steps, **options):
    old = np.zeros_like(capacity)
    audits = []
    for _ in range(steps):
        result = hybrid.solve(source / steps, capacity, old, **options)
        old += result["absorbed"]
        audits.append(result["audit"])
    return old, {
        "steps": steps,
        "seconds": sum(a["seconds"] for a in audits),
        "mean_angles_per_step": [a["angles_mean"] for a in audits],
        "rays_materialized_per_step": [a["rays_materialized"] for a in audits],
        "retained": sum(a["retained"] for a in audits),
        "maximum_absolute_ledger_residual": max(abs(a["ledger_residual"]) for a in audits),
    }


def validate(native_path=None, native_max_packets=2_000_000):
    hybrid.library()  # Exclude compilation from every timing.
    report = {
        "scope": "Static opaque-cell sharp fronts, infinite c, no recombinations; not a native evolving reionization run",
        "error_definition": "sum(abs(new_ionized_atoms - reference)) / emitted_photons",
        "spatial": [], "temporal": [],
    }
    for radius in (4, 8, 12):
        n = 4 * radius
        capacity = np.ones((n,) * 3)
        source = np.zeros_like(capacity)
        location = (n // 2,) * 3
        source[location] = 4 * np.pi * radius**3 / 3
        photons = source.sum()
        oracle = sphere_reference(n, radius, location, 64)
        coarse = sphere_reference(n, radius, location, 32)
        adaptive = hybrid.solve(source, capacity, periodic=False)
        reference = hybrid.solve(source, capacity, min_angles=65536,
                                 max_angles=65536, periodic=False)
        report["spatial"].append({
            "radius_cells": radius, "grid": n,
            "oracle_volume_relative_error": float(oracle.sum() / photons - 1),
            "oracle_32_64_error": distance(oracle, coarse, photons),
            "adaptive_vs_continuum_error": distance(adaptive["absorbed"], oracle, photons),
            "reference65536_vs_continuum_error": distance(reference["absorbed"], oracle, photons),
            "adaptive_vs_reference_error": distance(adaptive["absorbed"], reference["absorbed"], photons),
            "adaptive_angles": adaptive["audit"]["angles_max"],
            "adaptive_seconds": adaptive["audit"]["seconds"],
            "reference_seconds": reference["audit"]["seconds"],
        })
    capacity = np.ones((28,) * 3)
    cases = {}
    source = np.zeros_like(capacity)
    source[14, 14, 14] = 4 * np.pi * 6**3 / 3
    cases["sphere_R6"] = source
    source = np.zeros_like(capacity)
    source[11, 14, 14] = source[17, 14, 14] = 4 * np.pi * 4**3 / 3
    cases["overlap_R4"] = source
    source = np.zeros_like(capacity)
    source[12:17, 12:17, 12:17] = 10.
    cases["compact_125_sources"] = source
    for name, source in cases.items():
        reference = hybrid.solve(source, capacity, min_angles=12288,
                                 max_angles=12288)["absorbed"]
        record = {"case": name, "results": {}}
        for label, options in (("fixed3072", {"min_angles": 3072, "max_angles": 3072}),
                               ("adaptive", {})):
            rows = []
            first = None
            for steps in (1, 2, 4, 8):
                field, audit = partition(source, capacity, steps, **options)
                if first is None:
                    first = field
                audit["error_vs_one_step"] = distance(field, first, source.sum())
                audit["error_vs_fixed12288"] = distance(field, reference, source.sum())
                rows.append(audit)
            record["results"][label] = rows
        report["temporal"].append(record)
    if native_path is not None:
        data = np.load(native_path)
        source, capacity = data["source"], data["capacity"]
        first = hybrid.solve(source / 2, capacity)
        remaining = capacity - first["absorbed"]
        levels, counts, _, _ = hybrid.estimate_angles(source / 2, capacity, remaining)
        active = source > 0
        transparent = active & (remaining <= 0)
        self_clearing = active & (source / 2 > remaining)
        upper = int(counts[levels[active]].sum())
        native = {
            "path": str(Path(native_path).resolve()),
            "scope": "Frozen fields with constant rates; deliberately excludes cosmological source/sink evolution",
            "photons_over_capacity": float(source.sum() / capacity.sum()),
            "first_half_audit": first["audit"],
            "second_half_predicted_packets_upper_bound": upper,
            "second_half_already_transparent_packets": int(counts[levels[transparent]].sum()),
            "second_half_self_clearing_packets": int(counts[levels[self_clearing]].sum()),
            "second_half_ran": upper <= native_max_packets,
            "packet_execution_limit": native_max_packets,
        }
        full_fixed, one = partition(source, capacity, 1, min_angles=768, max_angles=768)
        split_fixed, two = partition(source, capacity, 2, min_angles=768, max_angles=768)
        native["fixed768_time_partition_error"] = distance(full_fixed, split_fixed, source.sum())
        native["fixed768_one_step"] = one
        native["fixed768_two_steps"] = two
        if native["second_half_ran"]:
            second = hybrid.solve(source / 2, capacity, first["absorbed"])
            native["second_half_audit"] = second["audit"]
        else:
            native["second_half_skip_reason"] = "Predictor exceeds explicit packet allocation limit; no evolutionary speedup inferred"
        report["native_frozen_fixture"] = native
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".json"))
    parser.add_argument("--native", type=Path)
    parser.add_argument("--native-max-packets", type=int, default=2_000_000)
    args = parser.parse_args()
    result = validate(args.native, args.native_max_packets)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)
