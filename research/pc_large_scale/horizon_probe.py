"""Fixed-angle, fixed-photon-budget horizon tests on an existing native fixture.

Each horizon starts from the original neutral gas. No final-state reemission is
used. The low-k statistic covers integer Fourier radii 1 <= |m| < 3 (92 full
modes); it is a small-box diagnostic, not the production 0.04--0.06 h/cMpc band.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy import fft

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pc_adaptive_front"))
import adaptive


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def total(a):
    return float(np.sum(a, dtype=np.longdouble))


def low_k_bias(capacity, absorbed):
    n = capacity.shape[0]
    modes = np.rint(fft.fftfreq(n) * n).astype(int)
    radius2 = modes[:, None, None]**2 + modes[None, :, None]**2 + modes[None, None, :]**2
    mask = (radius2 >= 1) & (radius2 < 9)
    neutral = capacity - absorbed
    fh = fft.fftn(neutral / neutral.mean() - 1, workers=1)
    fm = fft.fftn(capacity / capacity.mean() - 1, workers=1)
    return {"b_HI_squared": float(np.sum(abs(fh[mask])**2) / np.sum(abs(fm[mask])**2)),
            "full_fft_modes": int(mask.sum()), "integer_fourier_radius_band": [1, 3]}


def reemission_counterexample():
    """Source-only retained budgets do not preserve directional event history."""
    n = 12
    xyz = np.indices((n, n, n))
    capacity = 1 + .4*np.cos(2*np.pi*xyz[0]/n) + .2*np.sin(2*np.pi*xyz[1]/n)
    photons = np.zeros_like(capacity)
    photons[2, 3, 4] = .55*.9*total(capacity)
    photons[8, 7, 6] = .45*.9*total(capacity)
    options = dict(min_angles=384, max_angles=384, periodic=True)
    short = adaptive.solve(photons, capacity, maximum_distance=4, **options)
    direct = adaptive.solve(photons, capacity, maximum_distance=16, **options)
    reemitted = adaptive.solve(short["source_retained"], capacity,
                              initial_state=short["state"], maximum_distance=16, **options)
    wrong = short["absorbed"] + reemitted["absorbed"]
    return {"grid": n, "angles": 384, "dose": .9, "short_horizon_cells": 4,
            "long_horizon_cells": 16, "emitted": total(photons),
            "short_absorbed": total(short["absorbed"]),
            "direct_long_absorbed": total(direct["absorbed"]),
            "isotropic_reemission_total_absorbed": total(wrong),
            "reemission_vs_direct_L1_over_emitted": total(abs(wrong-direct["absorbed"]))/total(photons),
            "reemission_vs_direct_max_fraction_difference": float(np.max(abs(wrong-direct["absorbed"])/capacity)),
            "scope": "Counterexample to reemitting retained source totals from final gas state"}


def probe(fixture, angles, doses, horizon_fractions, output, include_reemission_counterexample=False):
    fixture = Path(fixture).resolve()
    with np.load(fixture) as arrays:
        source = np.array(arrays["source"], dtype=float)
        capacity = np.array(arrays["capacity"], dtype=float)
    adaptive.library()
    report = {"scope": "Static native fixture; fixed angular bank and fixed photon budget; independent reruns",
              "fixture": str(fixture), "fixture_sha256": digest(fixture),
              "shape": list(source.shape), "angles": angles or "adaptive, fixed reference horizon",
              "angular_reference_distance_cells": source.shape[0]/3,
              "code_sha256": {str(p.relative_to(HERE.parent)): digest(p) for p in
                              (Path(__file__).resolve(), HERE.parent / "pc_adaptive_front/adaptive.py",
                               HERE.parent / "pc_adaptive_front/adaptive.cpp")},
              "timing": "Single serial solve per configuration; compilation excluded; shared host",
              "cases": []}
    if include_reemission_counterexample:
        report["reemission_counterexample"] = reemission_counterexample()
        print(json.dumps(report["reemission_counterexample"]), flush=True)
    options = dict(min_angles=angles or 24, max_angles=angles or 12288,
                   angular_reference_distance=source.shape[0]/3, periodic=True)
    for dose in doses:
        photons = source * (dose * total(capacity) / total(source))
        emitted = total(photons)
        previous = None
        for fraction in horizon_fractions:
            started = time.perf_counter()
            result = adaptive.solve(photons, capacity,
                                    maximum_distance=source.shape[0] * fraction, **options)
            elapsed = time.perf_counter() - started
            absorbed = result["absorbed"]
            entry = {"dose": dose, "horizon_box_fraction": fraction,
                     "Q_mass": total(absorbed) / total(capacity),
                     "retained_fraction": total(result["source_retained"]) / emitted,
                     "ledger_relative_residual": abs(emitted - total(absorbed)
                         - total(result["source_retained"]) - total(result["source_escaped"])) / emitted,
                     "local_fraction_violation": float(np.max(np.maximum(absorbed-capacity, 0)/capacity)),
                     "angle_counts_sha256": hashlib.sha256(result["angle_counts"].tobytes()).hexdigest(),
                     "angles_min": int(result["angle_counts"][photons>0].min()),
                     "angles_max": int(result["angle_counts"][photons>0].max()),
                     "wall_seconds": elapsed, "solver_seconds": result["audit"]["seconds"],
                     "queries": result["audit"]["queries"],
                     "ray_bytes": result["audit"]["ray_bytes"],
                     "low_k": low_k_bias(capacity, absorbed)}
            if previous is not None:
                change = absorbed - previous["absorbed"]
                entry["versus_previous_horizon"] = {
                    "field_L1_over_emitted": total(abs(change)) / emitted,
                    "negative_count_over_emitted": total(np.maximum(-change, 0)) / emitted,
                    "max_fraction_difference": float(np.max(abs(change) / capacity)),
                    "b2_relative_change": entry["low_k"]["b_HI_squared"] / previous["b2"] - 1,
                    "previous_retained_fraction": previous["retained"]}
            report["cases"].append(entry)
            previous = {"absorbed": absorbed, "b2": entry["low_k"]["b_HI_squared"],
                        "retained": entry["retained_fraction"]}
            Path(output).write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(entry), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--angles", type=int, default=0,
                        help="0 keeps the source-adaptive angular bank fixed; positive selects uniform counts")
    parser.add_argument("--doses", nargs="+", type=float, default=[0.8, 0.9])
    parser.add_argument("--horizon-fractions", nargs="+", type=float, default=[1/3, 2/3, 4/3])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-reemission-counterexample", action="store_true")
    probe(**vars(parser.parse_args()))
