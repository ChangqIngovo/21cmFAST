"""Recheck the delivered incremental preset against fixed-angle saved fields.

The fixture is a frozen static second-half experiment, not cosmological time
evolution. Reference files must have been produced on the identical initial
state. The original adaptive map is recomputed, not taken from old timings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from sampling_native_compare import error
from speed_core import module
import upgraded
from audit_accuracy import strict_ledger


def run(fields, reference, reference_low, output, repeats=2):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    with np.load(fields) as saved:
        s, c, old = (saved[k] for k in ("source", "capacity", "initial"))
        rho = saved["density"] if "density" in saved else c
        previous = saved["baseline"] if "baseline" in saved else None
    horizon = s.shape[0]/2
    hashes = {k: hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()
              for k, a in (("source", s), ("capacity", c), ("initial", old))}

    def checked_reference(path, expected_angles):
        with np.load(path) as data:
            for key, expected in hashes.items():
                if key+"_sha256" not in data or str(data[key+"_sha256"].item()) != expected:
                    raise ValueError(f"{path}: reference {key} provenance missing or mismatched; regenerate it")
            if int(data["angles"]) != expected_angles or float(data["maximum_distance"]) != horizon:
                raise ValueError(f"{path}: reference angular count or horizon mismatched")
            return data["reference"]

    ref = checked_reference(reference, 3072)
    low = checked_reference(reference_low, 1536)
    if not (s.shape == c.shape == old.shape == ref.shape == low.shape):
        raise ValueError("Reference/fixture shapes differ")
    module("original")
    upgraded.warmup("incremental")
    profiles = ("original", "incremental")
    rows = {name: {"profile": name, "times": []} for name in profiles}
    report = {
        "scope": "same frozen native second-half source and initial state; static no-recombination test",
        "timing": "compilation excluded; sequential alternating order; shared host; transport serial",
        "reference": "fixed K3072 with K1536 crosscheck; same opaque-voxel approximation",
        "reference_1536_to_3072": error(low, ref, rho, float(s.sum())),
        "inputs": {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                   for p in (fields, reference, reference_low)},
        "array_hashes": hashes,
        "maximum_distance": horizon,
        "models": [],
    }
    for repeat in range(repeats):
        for name in (profiles if repeat % 2 == 0 else profiles[::-1]):
            start = time.perf_counter()
            if name == "original":
                result = module("original").solve(s, c, old, maximum_distance=horizon)
                if previous is not None:
                    np.testing.assert_array_equal(result["absorbed"], previous)
            else:
                result = upgraded.solve(s, c, old, profile=name, maximum_distance=horizon)
            elapsed = time.perf_counter() - start
            ledger = strict_ledger(result["absorbed"], result["unabsorbed"], s, c, old)
            assert ledger["passed"]
            row = rows[name]
            row["times"].append(elapsed)
            row.update(audit=result["audit"], strict_ledger=ledger,
                       vs_reference=error(result["absorbed"], ref, rho, float(s.sum())),
                       median_seconds=float(np.median(row["times"])))
            report["models"] = list(rows.values())
            Path(output).write_text(json.dumps(report, indent=2)+"\n")
            print(json.dumps({"profile": name, "repeat": repeat, "seconds": elapsed,
                              "vs_reference": row["vs_reference"]}), flush=True)
    report["code_hashes"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__), Path(upgraded.__file__),
                  Path(upgraded.sampling_solver.__file__),
                  Path(__file__).with_name("speed_core.cpp"))}
    Path(output).write_text(json.dumps(report, indent=2)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fields", "reference", "reference-low"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".json"))
    parser.add_argument("--repeats", type=int, default=2)
    run(**vars(parser.parse_args()))
