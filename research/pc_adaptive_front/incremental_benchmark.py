#!/usr/bin/env python3
"""Compare front solvers on the same resolved second-half native state.

The first half is computed once with full subdivision. Its eight ionized atom
counts, rather than a coarse-cell average, initialize every second-half arm.
Angular counts are fixed to isolate geometry/data-structure changes.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"pc_refinement"))
sys.path.insert(0, str(HERE.parent/"pc_reconsidered"))
sys.path.insert(0, str(HERE))

import adaptive
import front_subcells
from benchmarks import check_equivalence, ledger
from compare_models import Case


def state_from_dense(capacity, ionized):
    """Lossless encoding of the full fine state in the public sparse schema."""
    c = np.ascontiguousarray(capacity, dtype=float)
    n = len(c)
    coarse = front_subcells._combine(ionized)
    children = np.ascontiguousarray(ionized.reshape(n,2,n,2,n,2)
                                    .transpose(0,2,4,1,3,5).reshape(-1,8))
    different = np.any(children != children[:, :1], axis=1)
    ids = np.flatnonzero(different).astype(np.int32)
    return {"ionized": coarse, "parent_ids": ids,
            "child_ionized": np.ascontiguousarray(children[ids]),
            "capacity_sha256": hashlib.sha256(c.tobytes()).hexdigest()}


def load_variant(path, compile_flag=()):
    if path is None:
        adaptive.library()
        return adaptive
    spec = importlib.util.spec_from_file_location("_incremental_"+path.stem, HERE/"adaptive.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = Path(tempfile.mkdtemp(prefix="pc_front_increment_"))/"variant.so"
    subprocess.run(["g++", "-O3", "-std=c++17", "-shared", "-fPIC", *compile_flag,
                    str(path), "-o", str(output)], check=True)
    lib = ct.CDLL(str(output))
    template = adaptive.library()
    for name in ("adaptive_evolve", "adaptive_state_count", "adaptive_copy_state",
                 "adaptive_copy_remaining", "adaptive_copy_fine_absorbed", "adaptive_copy_state_absorbed"):
        original = getattr(template, name)
        actual = getattr(lib, name)
        actual.argtypes, actual.restype = original.argtypes, original.restype
    module._library = lib
    return module


def run(native, output, angles=384, repeats=3, variant_cpp=None, compile_flag=()):
    if repeats < 1 or angles < 24 or angles % 24:
        raise ValueError("Use positive repeats and an angular count divisible by 24")
    with np.load(native) as data:
        source = np.asarray(data["source"], dtype=float)/2
        capacity = np.asarray(data["capacity"], dtype=float)
    options = dict(min_angles=angles, max_angles=angles, periodic=True,
                   maximum_distance=len(source)/2)
    front_subcells.library()
    choices = variant_cpp or [None]
    modules = {p.stem if p else "adaptive": load_variant(p, compile_flag) for p in choices}
    code_paths = [Path(__file__), HERE/"adaptive.py", HERE.parent/"pc_refinement/front_subcells.py",
                  HERE.parent/"pc_refinement/front_subcells.cpp"]
    code_paths += [p if p else HERE/"adaptive.cpp" for p in choices]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths}
    first = front_subcells.solve(source, capacity, subcells=2, parent_skip=True, **options)
    fine_initial = first["subcell_absorbed"]
    state = state_from_dense(capacity, fine_initial)
    case = Case("native-second-half-resolved", source, capacity, state["ionized"], len(source)/2)
    rows = {name: {"name": name, "seconds_samples": []} for name in ("full_subcells", *modules)}
    order = tuple(rows)
    last = {}
    for repeat in range(repeats):
        for name in order if repeat % 2 == 0 else order[::-1]:
            started = time.perf_counter()
            if name == "full_subcells":
                result = front_subcells.solve(source, capacity, initial_subcell_ionized=fine_initial,
                                              subcells=2, parent_skip=True, **options)
            else:
                result = modules[name].solve(source, capacity, initial_state=state, **options)
            elapsed = time.perf_counter()-started
            check = ledger(result, source, capacity, state["ionized"])
            assert check["passed"], (name, check)
            rows[name]["seconds_samples"].append(elapsed)
            rows[name].update(audit=result["audit"], independent_ledger=check,
                              median_seconds=float(np.median(rows[name]["seconds_samples"])))
            last[name] = result
    for name, module in modules.items():
        dense = module.solve(source, capacity, initial_state=state, return_dense=True, **options)
        rows[name]["vs_full_subcells"] = check_equivalence(dense, last["full_subcells"], case)
    report = {
        "scope": "Static native half-dose increment from identical resolved fine initial state; no cosmological evolution",
        "timing": "Compilation and initial-state generation excluded; sequential alternating repeats; dense debug output untimed",
        "fixture": str(native), "fixture_sha256": hashlib.sha256(Path(native).read_bytes()).hexdigest(),
        "angles": angles, "shape": list(source.shape), "maximum_distance": options["maximum_distance"],
        "variant_compile_flags": list(compile_flag),
        "initial_state": {"coarse_ionized_sha256": hashlib.sha256(state["ionized"].tobytes()).hexdigest(),
                          "fine_ionized_sha256": hashlib.sha256(fine_initial.tobytes()).hexdigest(),
                          "nonuniform_parents": len(state["parent_ids"])},
        "first_half_audit": first["audit"], "models": list(rows.values()), "source_hashes": hashes,
    }
    if any(hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p)] for p in code_paths):
        raise RuntimeError("Source changed during increment benchmark; rerun")
    Path(output).write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({row["name"]: {"seconds": row["median_seconds"],
                                    "L1_Q": row.get("vs_full_subcells", {}).get("L1_over_emitted", 0.)}
                      for row in report["models"]}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE/"incremental_results.json")
    parser.add_argument("--angles", type=int, default=384)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--variant-cpp", type=Path, nargs="+")
    parser.add_argument("--compile-flag", action="append", default=[],
                        help="Extra flag for variant compilation; otherwise the file's compile-time defaults apply")
    run(**vars(parser.parse_args()))
