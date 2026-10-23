#!/usr/bin/env python3
"""Event-front cost and angular self-convergence on a frozen native source closure.

This is not native temporal/thermal/recombination integration. The native
cumulative source counts are reinterpreted as constant rates over duration one.
Recombination costs are excluded. Every worker starts in a fresh process so its
peak RSS is measured independently. Compilation is outside the solve timer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np


HERE = Path(__file__).resolve().parent


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def available_memory():
    """Use both host availability and the current cgroup headroom where exposed."""
    host = None
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            host = int(line.split()[1])*1024
            break
    cgroup = None
    limit = Path("/sys/fs/cgroup/memory.max")
    current = Path("/sys/fs/cgroup/memory.current")
    if limit.exists() and current.exists() and limit.read_text().strip() != "max":
        cgroup = int(limit.read_text())-int(current.read_text())
    values = [value for value in (host,cgroup) if value is not None]
    return dict(host_available_bytes=host,cgroup_headroom_bytes=cgroup,
                effective_available_bytes=min(values) if values else None)


def input_fields(path):
    with np.load(path) as data:
        fields = {name: data[name].astype(float) for name in (
            "source", "capacity", "density", "xray_fraction",
            "source_acg", "source_mcg", "recombinations_previous",
        )}
    s, c, rho, xe = (fields[name] for name in ("source", "capacity", "density", "xray_fraction"))
    assert s.shape == c.shape == rho.shape == xe.shape
    assert np.isfinite(s).all() and np.isfinite(c).all()
    assert np.min(s) >= 0 and np.min(c) >= 0 and np.min(rho) > 0
    assert np.max(np.abs(c-rho*(1-xe))) < 1e-12
    assert np.max(np.abs(s-fields["source_acg"]-fields["source_mcg"])) < 1e-12
    return fields


def worker(args):
    from event_front import library, solve

    fields = input_fields(args.inputs)
    s, c, rho, xe = (fields[name] for name in ("source", "capacity", "density", "xray_fraction"))
    library()
    baseline_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    start = time.perf_counter()
    result = solve(s, c, angles=args.angles, duration=1.0, periodic=True,
                   maximum_distance=s.shape[0]*np.sqrt(3))
    wall = time.perf_counter()-start
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    audit = result["audit"]
    count_scale = max(float(s.sum()), 1)
    assert abs(audit["ledger_residual"]) <= 1e-10*count_scale, audit
    assert audit["capacity_excess"] <= 1e-10*max(float(c.max()), 1), audit
    assert audit["negative_absorption"] <= 1e-12, audit
    assert audit["per_source_ledger_max"] <= 1e-9*max(float(s.max()), 1), audit
    assert audit["per_angle_ledger_max"] <= 1e-9*count_scale/args.angles, audit
    assert abs(audit["independent_absorption_ledger"]) <= 1e-10*count_scale, audit
    absorbed = result["absorbed"]
    xhi = 1-xe-absorbed/rho
    assert xhi.min() >= -1e-10 and xhi.max() <= 1+1e-10
    audit.update(solve_wall_seconds=wall, peak_rss_megabytes=peak_rss,
                 baseline_rss_megabytes=baseline_rss,
                 mean_xHI=float(xhi.mean()),
                 fully_exhausted_cells=int(np.count_nonzero(absorbed >= c-1e-12)),
                 source_nonzero_cells=int(np.count_nonzero(s)))
    np.savez_compressed(args.scratch / f"event_K{args.angles}.npz", absorbed=absorbed, xHI=xhi)
    write(args.scratch / f"event_K{args.angles}.json", audit)
    print(json.dumps(audit), flush=True)


def shells(field, boxlen):
    n = field.shape[0]
    kmodes = 2*np.pi*np.fft.fftfreq(n, d=boxlen/n)
    kx, ky, kz = np.meshgrid(kmodes, kmodes, kmodes, indexing="ij")
    k = np.sqrt(kx*kx+ky*ky+kz*kz)
    power = np.abs(np.fft.fftn(field-field.mean()))**2 * boxlen**3 / n**6
    edges = np.geomspace(2*np.pi/boxlen*.9999, np.pi*n/boxlen*1.0001, 13)
    rows = []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        mask = (k >= low) & (k < high)
        if mask.any():
            rows.append(dict(k_min=float(low), k_max=float(high), k_mean=float(k[mask].mean()),
                             full_fft_modes=int(mask.sum()), P_xHI=float(power[mask].mean())))
    return rows


def compare_maps(left, right, capacity, source):
    delta = left["absorbed"]-right["absorbed"]
    delta_x = left["xHI"]-right["xHI"]
    return dict(
        absorbed_L1_over_photons=float(np.abs(delta).sum()/source.sum()),
        absorbed_L1_over_capacity=float(np.abs(delta).sum()/capacity.sum()),
        xHI_mean_abs_difference=float(np.abs(delta_x).mean()),
        xHI_max_abs_difference=float(np.abs(delta_x).max()),
        xHI_rms_difference=float(np.sqrt(np.mean(delta_x**2))),
        xHI_rms_difference_over_reference_std=float(np.sqrt(np.mean(delta_x**2))/right["xHI"].std()),
        xHI_correlation=float(np.corrcoef(left["xHI"].ravel(),right["xHI"].ravel())[0,1]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path("/workspace/scratch/40b418003e4a/qin20_native_cost_grid32/source_capacity.npz"))
    parser.add_argument("--scratch", type=Path, default=Path("/workspace/scratch/40b418003e4a/event_native_grid32"))
    parser.add_argument("--output", type=Path, default=HERE / "event_native_benchmark.json")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--angles", type=int, default=48)
    parser.add_argument("--k768-max-seconds", type=float, default=20)
    parser.add_argument("--k768-max-rss-mib", type=float, default=900)
    parser.add_argument("--minimum-available-gib", type=float, default=2)
    parser.add_argument("--reference-scratch", type=Path)
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    fields = input_fields(args.inputs)
    prep = json.loads((args.inputs.parent / "preparation.json").read_text())
    native = json.loads((HERE / "native_benchmark.json").read_text())
    result = {
        "scope": "Actual frozen native-derived grid, new spatial source closure, constant-rate sharp-front toy evolution; angular self-convergence only",
        "input_file": str(args.inputs), "input_sha256": digest(args.inputs),
        "source_files_sha256": {name:digest(HERE/name) for name in ("event_front.py","event_front.cpp","characteristics.py")},
        "preparation": prep,
        "photons": float(fields["source"].sum()),
        "neutral_capacity": float(fields["capacity"].sum()),
        "previous_recombinations_not_charged": float(fields["recombinations_previous"].sum()),
        "source_density_capacity_identity_checks_passed": True,
        "physical_qualifications": [
            "Native cumulative sources treated as constant rate over an arbitrary duration of one; not native emissivity/time evolution",
            "Source closure changes native radius-conditioned source statistics",
            "No recombinations, no chemistry, infinite propagation speed, cell-centered point sources and sharp opaque voxels",
            "Periodic rays stop at n*sqrt(3) cell lengths and unabsorbed photons stay in the retained ledger",
            "Shared equal-weight Fibonacci direction grid is finite angular quadrature; different K grids are not nested",
            "Angular self-convergence is not proof of radiative-transfer physical accuracy or grid/subcell convergence",
            "Native no-PC denominator includes source/filter/ionization/Gamma/recombination; candidate does not yet replace this complete operator",
        ],
        "native_no_PC_seconds_median_four_threads": native["summary"]["no-photoncons"]["median_seconds"],
        "K768_guard_settings": {"maximum_projected_seconds":args.k768_max_seconds,
            "maximum_projected_peak_rss_MiB":args.k768_max_rss_mib,
            "minimum_available_GiB":args.minimum_available_gib},
        "runs": [], "convergence": [], "skipped": [],
    }
    for angles in (48,192,768):
        if angles == 768:
            prior = result["runs"][-1]
            predicted_seconds = 4.4*prior["solve_wall_seconds"]
            predicted_rss = prior["baseline_rss_megabytes"] + 4.4*(prior["peak_rss_megabytes"]-prior["baseline_rss_megabytes"])
            result["K768_projection"] = dict(seconds=predicted_seconds,peak_rss_megabytes=predicted_rss,
                                               method="4.4x measured192 cost and RSS increment; heuristic guard, not complexity proof")
            memory = available_memory()
            result["K768_memory_check"] = memory
            if predicted_seconds >= args.k768_max_seconds or predicted_rss >= args.k768_max_rss_mib:
                result["skipped"].append(dict(angles=768,reason="Projected solve exceeds configured time or RSS guard"))
                break
            if memory["effective_available_bytes"] is None or memory["effective_available_bytes"] < args.minimum_available_gib*1024**3:
                result["skipped"].append(dict(angles=768,reason="Insufficient verified available memory"))
                break
        environment = os.environ.copy()
        environment.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
        command = [sys.executable,str(Path(__file__).resolve()),"--worker","--angles",str(angles),
                   "--inputs",str(args.inputs),"--scratch",str(args.scratch)]
        try:
            run = subprocess.run(command,env=environment,capture_output=True,text=True,
                                 timeout=args.k768_max_seconds+10 if angles==768 else 60)
        except subprocess.TimeoutExpired:
            result["skipped"].append(dict(angles=angles,reason="Worker timeout including startup/build; no result asserted"))
            break
        if run.returncode != 0:
            result["skipped"].append(dict(angles=angles,reason="Worker failed",stderr=run.stderr[-4000:]))
            break
        row = json.loads((args.scratch/f"event_K{angles}.json").read_text())
        row["candidate_one_thread_over_native_four_threads"] = row["solve_wall_seconds"]/result["native_no_PC_seconds_median_four_threads"]
        result["runs"].append(row)
        print(json.dumps(row),flush=True)
        write(args.output,result)
    maps = {}
    for row in result["runs"]:
        count = row["angles"]
        with np.load(args.scratch/f"event_K{count}.npz") as data:
            maps[count] = {name:data[name] for name in ("absorbed","xHI")}
        if args.reference_scratch is not None:
            reference = args.reference_scratch/f"event_K{count}.npz"
            if reference.exists():
                with np.load(reference) as data:
                    row["reference_map_comparison"] = {
                        "file":str(reference),
                        "absorbed_bitwise_equal":bool(np.array_equal(maps[count]["absorbed"],data["absorbed"])),
                        "xHI_bitwise_equal":bool(np.array_equal(maps[count]["xHI"],data["xHI"])),
                        "max_abs_absorption_difference":float(np.max(np.abs(maps[count]["absorbed"]-data["absorbed"]))),
                    }
        row["spectrum"] = shells(maps[count]["xHI"],prep["box_mpc"])
    comparisons = list(zip(result["runs"][:-1],result["runs"][1:],strict=True))
    if len(result["runs"]) > 2:
        comparisons.append((result["runs"][0],result["runs"][-1]))
    for lower, higher in comparisons:
        low,high = lower["angles"],higher["angles"]
        comparison = {"lower_K":low,"reference_K":high,**compare_maps(maps[low],maps[high],fields["capacity"],fields["source"])}
        ratios = []
        num=den=0.
        for shell_low,shell_high in zip(lower["spectrum"],higher["spectrum"],strict=True):
            ratio = shell_low["P_xHI"]/shell_high["P_xHI"]
            modes=shell_high["full_fft_modes"]
            num+=modes*abs(shell_low["P_xHI"]-shell_high["P_xHI"])
            den+=modes*shell_high["P_xHI"]
            ratios.append(dict(k_mean=shell_high["k_mean"],full_fft_modes=modes,ratio_P_lower_over_reference=ratio))
        comparison["power_shells"] = ratios
        comparison["mode_weighted_P_absolute_difference_over_reference"] = num/den
        result["convergence"].append(comparison)
    write(args.output,result)


if __name__ == "__main__":
    main()
