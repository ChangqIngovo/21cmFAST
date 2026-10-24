"""Cached exact dynamic transparent-block skipping, compared with parent-only DDA."""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np

import adaptive

HERE = Path(__file__).resolve().parent
_modules = {}


def module(enabled=True):
    if enabled not in _modules:
        source = HERE / "hierarchy_cached.cpp"
        output = Path(tempfile.mkdtemp(prefix="pc_hierarchy_")) / "hierarchy.so"
        subprocess.run(["g++", "-O3", "-std=c++17", "-shared", "-fPIC",
                        f"-DPC_TRANSPARENT_HIERARCHY={int(enabled)}", str(source), "-o", str(output)], check=True)
        spec = importlib.util.spec_from_file_location(f"_pc_hierarchy_{enabled}", HERE / "adaptive.py")
        result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
        lib = ct.CDLL(str(output))
        base = adaptive.library()
        for name in ("adaptive_evolve", "adaptive_state_count", "adaptive_copy_state",
                     "adaptive_copy_remaining", "adaptive_copy_state_absorbed", "adaptive_copy_fine_absorbed"):
            getattr(lib, name).argtypes = getattr(base, name).argtypes
            getattr(lib, name).restype = getattr(base, name).restype
        for name in ("adaptive_hierarchy_jumps", "adaptive_hierarchy_faces_saved", "adaptive_hierarchy_bytes"):
            getattr(lib, name).argtypes = []
            getattr(lib, name).restype = ct.c_int64
        result._library = lib
        _modules[enabled] = result
    return _modules[enabled]


def solve(source, capacity, initial_ionized=None, *, enabled=True, **kwargs):
    model = module(enabled)
    result = model.solve(source, capacity, initial_ionized, **kwargs)
    result["audit"].update(
        hierarchy_enabled=bool(enabled),
        hierarchy_large_jumps=int(model._library.adaptive_hierarchy_jumps()),
        hierarchy_faces_saved_estimate=int(model._library.adaptive_hierarchy_faces_saved()),
        hierarchy_bytes=int(model._library.adaptive_hierarchy_bytes()))
    return result


def compare(source, capacity, initial=None, *, repeats=3, horizon=None, angles=None):
    options = dict(maximum_distance=source.shape[0]/2 if horizon is None else horizon)
    if angles is not None:
        options.update(min_angles=angles, max_angles=angles)
    module(False); module(True)
    rows = {False: {"enabled": False, "seconds": []}, True: {"enabled": True, "seconds": []}}
    maps = {}
    for repeat in range(repeats):
        for enabled in ([False, True] if repeat % 2 == 0 else [True, False]):
            started = time.perf_counter()
            result = solve(source, capacity, initial, enabled=enabled, **options)
            rows[enabled]["seconds"].append(time.perf_counter()-started)
            rows[enabled]["audit"] = result["audit"]
            maps[enabled] = result["absorbed"]
    delta = abs(maps[True]-maps[False])
    report = {"variants": list(rows.values()), "L1_over_emitted": float(delta.sum()/source.sum()),
              "maximum_count_difference": float(delta.max())}
    for row in report["variants"]:
        row["median_seconds"] = float(np.median(row["seconds"]))
    if report["L1_over_emitted"] > 2e-10:
        raise RuntimeError(report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native32", type=Path)
    parser.add_argument("--native64", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=HERE/"hierarchy_cached_results.json")
    args = parser.parse_args()
    records = []
    # A resolved initial cavity provides a controlled long-transparent-path case.
    n = 32
    c = np.ones((n,)*3)
    xyz = np.indices(c.shape)-n//2
    initial = ((xyz*xyz).sum(axis=0)<10**2).astype(float)
    s = np.zeros_like(c);s[n//2,n//2,n//2] = 1.
    records.append({"case": "cavity_R10", **compare(s,c,initial,repeats=args.repeats,angles=12288)})
    for name, path in (("native32", args.native32), ("native64", args.native64)):
        if path is None:
            continue
        with np.load(path) as data:
            s, c = data["source"], data["capacity"]
        records.append({"case": name, "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        **compare(s,c,repeats=args.repeats)})
        args.output.write_text(json.dumps({"cases": records},indent=2)+"\n")
        print(json.dumps(records[-1]), flush=True)
    result = {"scope": "Exact dynamic transparent hierarchy; same rays and gas cells",
              "timing": "alternating sequential warm calls, compilation excluded, shared host",
              "cases": records,
              "code_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (HERE/"hierarchy_cached.cpp", HERE/"adaptive.py", Path(__file__))}}
    args.output.write_text(json.dumps(result,indent=2)+"\n")


if __name__ == "__main__":
    main()
