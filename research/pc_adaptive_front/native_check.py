#!/usr/bin/env python3
"""Validate the adaptive front through restricted native 21cmFAST outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pc_reconsidered"))
sys.path.insert(0, str(HERE.parent / "pc_refinement"))
sys.path.insert(0, str(HERE))
import adaptive
import front_subcells
import upgraded
from native_driver import run_native


def run(directory, output, dimension=32, threads=4, repeats=3):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    native = run_native(output=directory, dimension=dimension, box_len=200.,
                        redshift=8., threads=threads, solver="adaptive-front")
    source, capacity = native["source"].source, native["source"].capacity
    horizon = native["metadata"]["maximum_distance_cells"]
    options = dict(maximum_distance=horizon, periodic=True)
    front_subcells.library()
    upgraded.warmup("balanced")
    runs = {name: {"times": []} for name in ("balanced", "full_subcells", "adaptive_front")}
    models = {"balanced": lambda: upgraded.solve(source, capacity, profile="balanced", **options),
              "full_subcells": lambda: front_subcells.solve(source, capacity, parent_skip=True, **options),
              "adaptive_front": lambda: adaptive.solve(source, capacity, **options)}
    for model in models.values():
        model()
    maps = {}
    for k in range(repeats):
        names = tuple(models) if k % 2 == 0 else tuple(models)[::-1]
        for name in names:
            started = time.perf_counter()
            result = models[name]()
            runs[name]["times"].append(time.perf_counter()-started)
            runs[name]["audit"] = result["audit"]
            maps[name] = result["absorbed"]
    q = float(source.sum(dtype=np.longdouble))
    delta = maps["adaptive_front"]-maps["full_subcells"]
    dx = np.divide(abs(delta), capacity, out=np.zeros_like(delta), where=capacity > 0)
    equivalence = {"L1_over_emitted": float(abs(delta).sum()/q) if q else 0.,
                   "maximum_ionized_fraction_difference": float(dx.max())}
    assert equivalence["L1_over_emitted"] < 2e-10
    assert equivalence["maximum_ionized_fraction_difference"] < 2e-9
    for row in runs.values():
        row["median_seconds"] = float(np.median(row["times"]))
    import py21cmfast
    code_paths = [Path(__file__), Path(adaptive.__file__), HERE/"adaptive.cpp",
                  HERE.parent/"pc_reconsidered/native_driver.py",
                  HERE.parent/"pc_reconsidered/hybrid.py", HERE.parent/"pc_reconsidered/hybrid.cpp",
                  HERE.parent/"pc_refinement/front_subcells.py", HERE.parent/"pc_refinement/front_subcells.cpp",
                  HERE.parent/"pc_refinement/upgraded.py", HERE.parent/"pc_refinement/budgeted.py",
                  HERE.parent/"pc_refinement/speed_core.cpp"]
    package = Path(py21cmfast.__file__).resolve().parent
    report = {"scope": "Native 21cmFAST restricted coeval; adaptive-front and full eight-subcell equivalence",
              "timing": "One untimed solve per model, then alternating warmed calls; serial transport",
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "py21cmfast": py21cmfast.__version__, "native_threads": threads,
                              "binary_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                                for p in package.glob('*.so')}},
              "native": native["metadata"], "transport": runs, "equivalence": equivalence,
              "source_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
              "capacity_sha256": hashlib.sha256(capacity.tobytes()).hexdigest(),
              "code_hashes": {str(p.relative_to(HERE.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in code_paths}}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"equivalence": equivalence,
                      "seconds": {k: v["median_seconds"] for k,v in runs.items()}}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE/"native_results.json")
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    run(**vars(parser.parse_args()))
