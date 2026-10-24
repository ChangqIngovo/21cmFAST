"""Compare compact storage plus modest angular refinement on identical fields."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "pc_reconsidered"
sys.path.insert(0, str(OLD))
from compare_models import Case, field_statistics, make_cases
from event_front import solve as reference_solve
from audit_accuracy import strict_ledger
from budgeted import solve, warmup
from speed_core import module, solve as old_solve


def run(output, native64=None, repeats=3, reference_dir=None):
    warmup()
    module("original")
    cases = []
    reference_dir = Path(reference_dir) if reference_dir is not None else OLD / "comparison_fields"
    reference_provenance = {}
    toy = make_cases()
    for name in ("isolated", "equal-overlap", "transparent-overlap-increment", "dense-correlated"):
        reference_file = reference_dir / f"comparison_{name}.npz"
        with np.load(reference_file) as stored:
            saved = {key: stored[key] for key in stored.files}
        for key, values in (("source", toy[name].source), ("capacity", toy[name].capacity), ("initial", toy[name].initial)):
            np.testing.assert_array_equal(saved[key], values)
        reference_provenance[name] = {"path": str(reference_file.resolve()),
                                      "sha256": hashlib.sha256(reference_file.read_bytes()).hexdigest()}
        cases.append((toy[name], saved["reference"], "previous high-angle reference"))
    for name in ("native-dose-0.2", "native-dose-0.6", "native-frozen"):
        reference_file = reference_dir / f"comparison_{name}.npz"
        with np.load(reference_file) as stored:
            saved = {key: stored[key] for key in stored.files}
        reference_provenance[name] = {"path": str(reference_file.resolve()),
                                      "sha256": hashlib.sha256(reference_file.read_bytes()).hexdigest()}
        case = Case(name, saved["source"], saved["capacity"], saved["initial"], 16.)
        cases.append((case, saved["reference"], "supplied stored native frozen reference; quadrature specified by its generation manifest"))
    if native64:
        data = np.load(native64)
        c, s = data["capacity"], data["source"]
        case = Case("native64-L", s, c, np.zeros_like(c), 16.)
        ref = reference_solve(s, c, angles=768, periodic=True, maximum_distance=16.)
        cases.append((case, ref["absorbed"], "K768; separate 192-to-768 convergence report"))
    report = {"scope": "same opaque-voxel physics; compact ray memory used for extra angular samples",
              "timing": "compilation excluded; alternating sequential repeats; shared host",
              "reference_provenance": reference_provenance,
              "cases": []}
    profiles = [("original", 1), ("compact", 1), ("boost-1.25", 1.25),
                ("boost-4/3", 4/3), ("boost-1.5", 1.5)]
    for case, ref, reference in cases:
        rows = {name: {"profile": name, "angular_boost": factor, "times": []}
                for name, factor in profiles}
        for repetition in range(repeats):
            order = profiles if repetition % 2 == 0 else profiles[::-1]
            for name, factor in order:
                start = time.perf_counter()
                if name == "original":
                    out = old_solve(case.source, case.capacity, case.initial, variant="original",
                                    periodic=True, maximum_distance=case.maximum_distance)
                else:
                    out = solve(case.source, case.capacity, case.initial, angular_boost=factor,
                                periodic=True, maximum_distance=case.maximum_distance)
                row = rows[name]
                row["times"].append(time.perf_counter()-start)
                row["audit"] = out["audit"]
                row["vs_reference"] = field_statistics(out["absorbed"], ref, case)
                row["strict_ledger"] = strict_ledger(out["absorbed"], out["unabsorbed"],
                    case.source, case.capacity, case.initial)
                assert row["strict_ledger"]["passed"]
                if case.analytic is not None:
                    row["vs_continuum"] = field_statistics(out["absorbed"], case.analytic, case)
        for row in rows.values():
            row["median_seconds"] = float(np.median(row["times"]))
        record = {"name": case.name, "reference": reference,
                  "source_sha256": hashlib.sha256(case.source.tobytes()).hexdigest(),
                  "profiles": list(rows.values())}
        report["cases"].append(record)
        Path(output).write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps({"case": case.name, "profiles": [
            {"name": r["profile"], "seconds": r["median_seconds"],
             "L1_Q": r["vs_reference"]["L1_absorption_over_emitted"]} for r in rows.values()]}), flush=True)
    report["code_hashes"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (HERE/"budgeted.py", HERE/"speed_core.cpp", HERE/"speed_core.py", Path(__file__))}
    Path(output).write_text(json.dumps(report, indent=2)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE/"budgeted_results.json")
    parser.add_argument("--native64", type=Path)
    parser.add_argument("--reference-dir", type=Path,
                        help="Directory containing comparison_<case>.npz reference cubes")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    run(**vars(args))
