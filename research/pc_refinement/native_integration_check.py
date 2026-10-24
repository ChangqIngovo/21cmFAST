#!/usr/bin/env python3
"""Validate the balanced profile through native 21cmFAST and a fixed-angle check.

The native driver enforces the same restricted no-recombination, ACG-only,
saturated-spin physics for every profile. It uses one common source/capacity
realization; reference angular changes are reported separately from model error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pc_reconsidered"))
sys.path.insert(0, str(HERE.parent / "pc_alternatives"))
from native_driver import run_native
from hybrid import library as old_library, solve as original_solve
from event_front import library as ref_library, solve as reference_solve


def stats(array, reference, photons):
    delta = array-reference
    return {
        "L1_absorption_over_emitted": float(np.abs(delta).sum()/photons),
        "maximum_cell_absorption_difference": float(np.abs(delta).max()),
        "field_correlation": float(np.corrcoef(array.ravel(), reference.ravel())[0, 1]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "native_integration.json")
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--reference-angles", nargs=2, type=int, default=[1536, 6144])
    args = parser.parse_args()
    result = run_native(output=args.directory, dimension=args.dimension, box_len=200.,
                        redshift=8., threads=args.threads, solver="balanced")
    source, capacity = result["source"].source, result["source"].capacity
    photons = float(source.sum(dtype=np.longdouble))
    distance = result["metadata"]["maximum_distance_cells"]
    old_library()
    ref_library()
    original = original_solve(source, capacity, periodic=True, maximum_distance=distance)
    references = [reference_solve(source, capacity, periodic=True, maximum_distance=distance,
                                 angles=k) for k in args.reference_angles]
    low, high = references
    report = {
        "scope": "Actual restricted 21cmFAST native integration, shared sources; not full Qin20 or time-evolving radiation",
        "native": result["metadata"],
        "reference_angles": args.reference_angles,
        "reference_model": "constant emission, static gas, infinite speed, opaque voxels",
        "reference_angular_change": stats(low["absorbed"], high["absorbed"], photons),
        "original_vs_reference": stats(original["absorbed"], high["absorbed"], photons),
        "balanced_vs_reference": stats(result["candidate"]["absorbed"], high["absorbed"], photons),
        "balanced_vs_original": stats(result["candidate"]["absorbed"], original["absorbed"], photons),
        "original_audit": original["audit"],
        "reference_audits": [r["audit"] for r in references],
        "source_hashes": {str(path.relative_to(HERE.parent.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in [Path(__file__), HERE / "upgraded.py", HERE.parent / "pc_reconsidered/native_driver.py"]},
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: report[name] for name in
                      ("reference_angular_change", "original_vs_reference", "balanced_vs_reference")}, indent=2))


if __name__ == "__main__":
    main()
