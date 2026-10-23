"""Independently refine fixed-angle references for a saved native PC fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource

import numpy as np

from compare_models import (
    ALTERNATIVES, Case, array_hash, field_statistics, independent_ledger,
    sanitized, solve_event,
)


def run(filename, angles, maximum_distance, output, maximum_ray_gib=2.):
    filename = Path(filename)
    with np.load(filename) as stored:
        source = np.asarray(stored["source"], dtype=np.float64)
        capacity = np.asarray(stored["capacity"], dtype=np.float64)
        adaptive = np.asarray(stored["adaptive_pc_absorbed"], dtype=np.float64)
        retained = float(np.asarray(stored["adaptive_pc_source_retained"]).sum())
    case = Case("native-coeval", source, capacity, np.zeros_like(source), maximum_distance)
    metadata_file = filename.parent / "native_comparison.json"
    metadata = json.loads(metadata_file.read_text()) if metadata_file.exists() else {}
    report = {
        "scope": "Independent fixed-angle refinement of a saved restricted native L-INTEGRAL coeval; no recombinations or initial ionization; not full RT or exact Qin20",
        "reference_model": "constant emission over a unit interval, infinite c, static density, opaque remaining atoms, periodic box, finite source-relative distance limit",
        "input": str(filename), "input_sha256": hashlib.sha256(filename.read_bytes()).hexdigest(),
        "shape": list(source.shape), "maximum_distance_cells": maximum_distance,
        "input_hashes": {"source": array_hash(source), "capacity": array_hash(capacity),
                         "adaptive_absorbed": array_hash(adaptive)},
        "native_metadata": metadata,
        "timing_scope": "one worker; compilation excluded; shared-host wall times; independent reference only, not native generation or full simulation",
        "adaptive_ledger": independent_ledger(adaptive, retained, case),
        "adaptive_volume_mean_xHI": float(np.mean((capacity - adaptive) / capacity)),
        "requested_angles": list(angles), "maximum_ray_storage_estimate_gib": maximum_ray_gib,
        "memory_preflight": "Scale measured ray allocation by requested-angle ratio. This is a practical estimate, not a certified peak-memory bound.",
        "references": [],
    }
    previous = None
    for count in angles:
        if report["references"]:
            last = report["references"][-1]
            predicted = last["audit"].get("ray_allocated_bytes", 0) * count / last["angles"]
            if predicted > maximum_ray_gib * 2**30:
                report["skipped"] = {"angles": count, "predicted_ray_bytes": predicted,
                                     "reason": "Measured allocation extrapolation exceeds explicit practical budget"}
                break
        absorbed, remaining, seconds, audit = solve_event(case, count)
        row = {"angles": count, "seconds": seconds, "audit": audit,
               "ledger": independent_ledger(absorbed, remaining, case),
               "volume_mean_xHI": float(np.mean((capacity - absorbed) / capacity)),
               "adaptive_vs_reference": field_statistics(adaptive, absorbed, case),
               "process_peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 2**30}
        if previous is not None:
            row["previous_reference_vs_current"] = field_statistics(previous, absorbed, case)
        report["references"].append(row)
        previous = absorbed
        report["source_hashes"] = {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__), Path(__file__).with_name("compare_models.py"),
             ALTERNATIVES / "event_front.py", ALTERNATIVES / "event_front.cpp"]
        }
        output.write_text(json.dumps(sanitized(report), indent=2, allow_nan=False) + "\n")
        print(json.dumps({"angles": count, "seconds": seconds,
                          "ray_gib": audit.get("ray_allocated_bytes", 0) / 2**30,
                          "adaptive_L1_Q": row["adaptive_vs_reference"]["L1_absorption_over_emitted"],
                          "reference_change_L1_Q": row.get("previous_reference_vs_current", {}).get("L1_absorption_over_emitted")}), flush=True)
    output.write_text(json.dumps(sanitized(report), indent=2, allow_nan=False) + "\n")
    if not report["adaptive_ledger"]["passed"] or not all(r["ledger"]["passed"] for r in report["references"]):
        raise AssertionError("Independent photon ledger failed")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--angles", nargs="+", type=int, default=[192, 768])
    parser.add_argument("--maximum-distance", type=float, default=16.)
    parser.add_argument("--maximum-ray-gib", type=float, default=2.)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.angles or any(a < 1 for a in args.angles) or args.angles != sorted(set(args.angles)):
        parser.error("Provide strictly increasing positive angle counts")
    if not np.isfinite(args.maximum_ray_gib) or args.maximum_ray_gib <= 0:
        parser.error("Provide a finite positive practical ray-storage budget")
    run(args.input, args.angles, args.maximum_distance, args.output, args.maximum_ray_gib)


if __name__ == "__main__":
    main()
