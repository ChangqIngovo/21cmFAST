"""Compare equal-cost angular rules against unchanged event/analytic references."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "pc_reconsidered"
sys.path.insert(0, str(PREVIOUS))
from compare_models import Case, make_cases, sphere_reference, field_statistics, array_hash
import hybrid
from quadrature import METHODS, directions, solve, warmup


def finite_json(value):
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def run(output, native32=None, native64=None, repeats=3):
    warmup()
    hybrid.library()
    cases = make_cases(24)
    selected = [cases[name] for name in (
        "isolated", "equal-overlap", "unequal-overlap",
        "transparent-overlap-increment", "opaque-plane", "dense-correlated")]
    n, radius = 32, 8
    source = np.zeros((n,)*3)
    source[n//2, n//2, n//2] = 4*np.pi*radius**3/3
    selected.append(Case("isolated_R8", source, np.ones_like(source), np.zeros_like(source),
                         n/2, {"radius_cells": radius}, sphere_reference(n, radius, (n//2,)*3)))
    if native32:
        data = np.load(native32)
        selected.append(Case("native-frozen", data["source"], data["capacity"],
                             np.zeros_like(data["source"]), 16.,
                             {"fixture": str(native32), "source_model": "mass-normalized Qin20 E-INTEGRAL frozen source"}))
    if native64:
        data = np.load(native64)
        selected.append(Case("native64_L", data["source"], data["capacity"],
                             np.zeros_like(data["source"]), 16.,
                             {"fixture": str(native64), "source_model": "restricted native L-INTEGRAL"}))
    metadata = {}
    for name in ("comparison_final_results.json", "comparison_native_refined_results.json"):
        metadata.update({case["name"]: case for case in json.loads((PREVIOUS/name).read_text())["cases"]})
    report = {"scope": "Same count-conserving event equations and same per-source angle counts; only direction nodes change",
              "timing": "Single-worker kernel and setup; compilation excluded, three sequential repetitions; deterministic quadratures cached",
              "rules": list(METHODS), "cases": []}
    for case in selected:
        saved = PREVIOUS / "comparison_fields" / f"comparison_{case.name}.npz"
        if saved.exists() and case.name in metadata:
            data = np.load(saved)
            for name, values in (("source", case.source), ("capacity", case.capacity), ("initial", case.initial)):
                np.testing.assert_array_equal(values, data[name])
                assert array_hash(values) == metadata[case.name]["input_hashes"][name]
            reference = data["reference"]
            np.testing.assert_allclose(reference.sum(), metadata[case.name]["reference_audit"]["absorbed"], rtol=1e-15)
            reference_info = {"source": str(saved), "angles": metadata[case.name]["reference_angles"],
                              "angular_refinement_L1_Q": metadata[case.name]["reference_angular_change"]["L1_absorption_over_emitted"]}
        else:
            angles = metadata.get(case.name, {}).get("reference_angles",
                         768 if case.name == "native64_L" else 65536)
            ref = hybrid.solve(case.source, case.capacity, case.initial, min_angles=angles,
                               max_angles=angles, periodic=True, maximum_distance=case.maximum_distance)
            reference = ref["absorbed"]
            reference_info = {"source": "Fresh unchanged fixed-angle event solve", "angles": angles,
                              "audit": ref["audit"]}
        row = {"case": case.name, "shape": list(case.source.shape), "metadata": case.metadata,
               "reference": reference_info, "models": [],
               "source_sha256": array_hash(case.source), "capacity_sha256": array_hash(case.capacity)}
        if case.analytic is not None:
            row["reference_vs_analytic"] = field_statistics(reference, case.analytic, case)
        for method in METHODS:
            seconds = []
            outer = []
            for _ in range(repeats):
                started = time.perf_counter()
                result = solve(case.source, case.capacity, case.initial, method=method,
                               periodic=True, maximum_distance=case.maximum_distance)
                outer.append(time.perf_counter()-started)
                seconds.append(result["audit"]["seconds"])
            item = {"method": method, "seconds_median": float(np.median(seconds)),
                    "seconds_samples": seconds, "outer_seconds_samples": outer,
                    "audit": result["audit"], "vs_reference": field_statistics(result["absorbed"], reference, case)}
            if case.analytic is not None:
                item["vs_analytic"] = field_statistics(result["absorbed"], case.analytic, case)
            row["models"].append(item)
        report["cases"].append(finite_json(row))
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        print(json.dumps({"case": case.name, "models": [{"name": item["method"], "seconds": item["seconds_median"],
                         "L1_Q": item["vs_reference"]["L1_absorption_over_emitted"]} for item in row["models"]]}), flush=True)
    report["source_hashes"] = {str(path.relative_to(HERE.parent.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (HERE/"quadrature.py", HERE/"quadrature_weights.py",
                                            HERE/"quadrature_weighted.cpp", Path(__file__),
                                            PREVIOUS/"hybrid.py", PREVIOUS/"hybrid.cpp")}
    output.write_text(json.dumps(finite_json(report), indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE/"quadrature_results.json")
    parser.add_argument("--native32", type=Path)
    parser.add_argument("--native64", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    run(**vars(parser.parse_args()))
