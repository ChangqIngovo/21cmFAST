"""Compare exact traversal optimizations on fixed geometry and native fixtures."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

from speed_core import module, solve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native32", type=Path)
    parser.add_argument("--native64", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--variants", nargs="+", default=["original", "index", "filtered"])
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("speed_results.json"))
    args = parser.parse_args()
    variants = args.variants
    for name in variants:
        module(name)
    source = np.zeros((32,)*3)
    source[16, 16, 16] = 4*np.pi*8**3/3
    cases = [("sphere_R8", source, np.ones_like(source), dict(maximum_distance=16.))]
    for name in ["native32", "native64"]:
        path = getattr(args, name)
        if path is not None:
            with np.load(path) as data:
                cases.append((name, data["source"], data["capacity"], dict(maximum_distance=16.)))
    report = {"scope": "Same photons, opacity, angular samples and source locations; exact arithmetic-path acceleration only",
              "timing": "one worker, sequential alternating order, compilation excluded; shared host wall-time noise remains",
              "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__), Path(__file__).with_name("speed_core.cpp"),
                                          Path(__file__).with_name("speed_core.py")]}, "cases": []}
    for name, source, capacity, options in cases:
        ref = solve(source, capacity, variant="original", **options)
        samples = {variant: [] for variant in variants}
        audits = {}
        for repeat in range(args.repeats):
            order = variants[repeat % len(variants):] + variants[:repeat % len(variants)]
            for variant in order:
                out = solve(source, capacity, variant=variant, **options)
                for key in ("absorbed", "source_absorbed", "source_escaped", "source_retained"):
                    if not np.array_equal(out[key], ref[key]):
                        raise AssertionError(f"{name}/{variant}: {key} differs from original")
                samples[variant].append(out["audit"]["transport_seconds"])
                audits[variant] = out["audit"]
        medians = {variant: statistics.median(values) for variant, values in samples.items()}
        record = {"name": name, "bitwise_equal_maps_and_source_ledgers": True,
                  "seconds_samples": samples, "median_seconds": medians,
                  "speedup": {v: medians["original"] / value for v, value in medians.items()},
                  "audits": audits}
        report["cases"].append(record)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in record.items() if k != "audits"}), flush=True)


if __name__ == "__main__":
    main()
