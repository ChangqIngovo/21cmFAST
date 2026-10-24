#!/usr/bin/env python3
"""Portable, explicit reproduction of selected PC refinement experiments.

No raw cosmological or reference cube is shipped in git. This driver can build
the small native fixtures and high-angle references, or consume user-supplied
fixtures. Every command, code revision, fixture hash, exit status and log path
is retained in a new result directory. It never submits a cluster job itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STEPS = ("prepare-native32", "prepare-native64", "tests", "audit", "theory",
         "references", "speed", "front", "quadrature", "budgeted", "sampling",
         "incremental", "convergence", "faint", "density", "geometry",
         "validated-incremental", "native-integration")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            value.update(chunk)
    return value.hexdigest()


def fixture_provenance(fixtures):
    """Keep native snapshot/parameter metadata even if the input is moved later."""
    result = {}
    for name, path in fixtures.items():
        if not path.is_file():
            continue
        entry = {"path": str(path), "sha256": digest(path)}
        metadata_names = {"native32": ("preparation.json",),
                          "native64": ("native_comparison.json",)}.get(name, ())
        entry["metadata"] = []
        for filename in metadata_names:
            metadata = path.parent / filename
            if metadata.is_file():
                entry["metadata"].append({"path": str(metadata), "sha256": digest(metadata),
                                          "contents": json.loads(metadata.read_text())})
        if metadata_names and not entry["metadata"]:
            entry["provenance_note"] = "Native preparation metadata not supplied; fixture hash alone does not identify the source snapshot or physics settings."
        result[name] = entry
    return result


def command_plan(args):
    selected = set(args.benchmarks)
    out = args.output_dir.resolve()
    native32 = args.native32.resolve() if args.native32 else None
    native64 = args.native64.resolve() if args.native64 else None
    if "prepare-native32" in selected:
        if native32 is not None:
            raise ValueError("Choose --native32 or prepare-native32, not both")
        native32 = out / "native32/source_capacity.npz"
    if "prepare-native64" in selected:
        if native64 is not None:
            raise ValueError("Choose --native64 or prepare-native64, not both")
        native64 = out / "native64/native_fields.npz"
    reference_dir = args.reference_dir.resolve() if args.reference_dir else out / "comparison_fields"
    fields = args.increment_fields.resolve() if args.increment_fields else out / "incremental_fields.npz"
    reference = args.increment_reference.resolve() if args.increment_reference else out / "incremental_reference.npz"
    if "incremental" in selected and args.increment_fields:
        raise ValueError("Choose --increment-fields or incremental generation, not both")
    if "convergence" in selected and args.increment_reference:
        raise ValueError("Choose --increment-reference or convergence generation, not both")
    if selected & {"incremental"} and native32 is None:
        raise ValueError("incremental requires --native32 or prepare-native32")
    if selected & {"convergence", "faint", "density", "validated-incremental"} and not args.increment_fields and "incremental" not in selected:
        raise ValueError("Supply --increment-fields or select incremental")
    if "density" in selected and not args.increment_reference and "convergence" not in selected:
        raise ValueError("density requires --increment-reference or convergence")
    if selected & {"budgeted", "geometry"} and not args.reference_dir and "references" not in selected:
        raise ValueError("Supply --reference-dir or select references")
    if "budgeted" in selected and "references" in selected and native32 is None:
        raise ValueError("budgeted needs native32 reference cases; supply --native32 or prepare-native32")
    plan = []

    def add(name, script, *extra):
        plan.append((name, [sys.executable, str(script), *map(str, extra)]))

    if "prepare-native32" in selected:
        add("prepare-native32", HERE.parent / "pc_alternatives/native_benchmark.py", "prepare",
            "--grid", 32, "--box-mpc", 200, "--threads", args.threads,
            "--directory", out / "native32")
    if "prepare-native64" in selected:
        add("prepare-native64", HERE.parent / "pc_reconsidered/native_driver.py",
            "--dimension", 64, "--box-len", 200, "--redshift", 8,
            "--threads", args.threads, "--output", out / "native64")
    if "tests" in selected:
        plan.append(("tests", [sys.executable, "-m", "pytest", str(HERE), "-q", "--confcutdir", str(HERE)]))
    for name, script in (("audit", "audit_accuracy.py"), ("theory", "audit_theory.py")):
        if name in selected:
            extra = ["--native", native32] if name == "theory" and native32 is not None else []
            add(name, HERE / script, *extra, "--output", out / f"{name}.json")
    if "references" in selected:
        comparator = HERE.parent / "pc_reconsidered/compare_models.py"
        add("reference-toys", comparator, "--methods", "adaptive-event", "--reference-angles", 65536,
            "--dense-reference-angles", 3072, "--refinement", 4,
            "--fields-directory", reference_dir, "--output", out / "reference_toys.json")
        if native32 is not None:
            add("reference-native32", comparator, "--fixture", native32,
                "--fixture-dose-fractions", .2, .6, "--cases", "native-dose-0.2", "native-dose-0.6", "native-frozen",
                "--methods", "adaptive-event", "--dense-reference-angles", 768, "--refinement", 4,
                "--fields-directory", reference_dir, "--output", out / "reference_native32.json")
    for name, script in (("speed", "speed_benchmark.py"), ("quadrature", "bench_quadrature.py")):
        if name in selected:
            extra = []
            if native32 is not None:
                extra += ["--native32", native32]
            if native64 is not None:
                extra += ["--native64", native64]
            if name == "speed":
                extra += ["--variants", "original", "compact"]
            add(name, HERE / script, *extra, "--repeats", args.repeats, "--output", out / f"{name}.json")
    if "front" in selected:
        extra = ["--native", native32] if native32 is not None else []
        add("front", HERE / "front_benchmark.py", *extra, "--repeats", args.repeats, "--output", out / "front.json")
    if "budgeted" in selected:
        extra = ["--native64", native64] if native64 is not None else []
        add("budgeted", HERE / "budgeted_benchmark.py", "--reference-dir", reference_dir,
            *extra, "--repeats", args.repeats, "--output", out / "budgeted.json")
    if "sampling" in selected:
        extra = ["--native", native32] if native32 is not None else []
        add("sampling", HERE / "sampling_benchmark.py", *extra, "--output", out / "sampling.json")
    if "incremental" in selected:
        add("incremental", HERE / "sampling_native_compare.py", "--native", native32,
            "--fields", fields, "--output", out / "incremental.json")
    if "convergence" in selected:
        add("convergence", HERE / "sampling_convergence.py", "--fields", fields,
            "--angles", args.reference_angles, "--reference-output", reference,
            "--output", out / "convergence.json")
    if "faint" in selected:
        add("faint", HERE / "sampling_faint_benchmark.py", "--fields", fields,
            "--output", out / "faint.json")
    if "density" in selected:
        add("density", HERE / "sampling_density.py", "--fields", fields,
            "--reference", reference, "--output", out / "density.json")
    if "geometry" in selected:
        add("geometry", HERE / "sampling_geometry.py", "--reference-dir", reference_dir,
            "--output", out / "geometry.json")
    if "validated-incremental" in selected:
        for angles in (1536, 3072):
            add(f"validated-reference-{angles}", HERE / "sampling_convergence.py",
                "--fields", fields, "--angles", angles,
                "--reference-output", out / f"validated_reference{angles}.npz",
                "--output", out / f"validated_reference{angles}.json")
        add("validated-incremental", HERE / "incremental_validation.py", "--fields", fields,
            "--reference", out / "validated_reference3072.npz",
            "--reference-low", out / "validated_reference1536.npz",
            "--repeats", args.repeats, "--output", out / "validated_incremental.json")
    if "native-integration" in selected:
        add("native-integration", HERE / "native_integration_check.py",
            "--directory", out / "native_integration_fields", "--threads", args.threads,
            "--output", out / "native_integration.json")
    fixtures = {name: path for name, path in (("native32", native32), ("native64", native64),
                 ("increment_fields", fields), ("increment_reference", reference)) if path is not None}
    return plan, fixtures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", choices=STEPS, nargs="+", default=["tests", "audit", "speed"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--native32", type=Path)
    parser.add_argument("--native64", type=Path)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--increment-fields", type=Path)
    parser.add_argument("--increment-reference", type=Path)
    parser.add_argument("--reference-angles", type=int, default=3072)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.threads, args.repeats, args.reference_angles) < 1:
        parser.error("Thread, repetition, and angle counts must be positive")
    try:
        plan, fixtures = command_plan(args)
    except ValueError as exc:
        parser.error(str(exc))
    for name, command in plan:
        print(name + ": " + shlex.join(command), flush=True)
    if args.dry_run:
        return
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source_paths = [p for folder in (HERE, HERE.parent / "pc_reconsidered", HERE.parent / "pc_alternatives")
                    for pattern in ("*.py", "*.cpp") for p in sorted(folder.glob(pattern))]
    report = {"commit": revision, "python": sys.executable, "python_version": platform.python_version(),
              "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
              "source_hashes": {str(p.relative_to(ROOT)): digest(p) for p in source_paths},
              "threads": args.threads, "steps": [], "fixtures": fixture_provenance(fixtures)}
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    manifest = out / "reproduction.json"
    manifest.write_text(json.dumps(report, indent=2)+"\n")
    try:
        for name, command in plan:
            logfile = out / f"{name}.log"
            started = time.perf_counter()
            with logfile.open("w") as stream:
                completed = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            report["steps"].append({"name": name, "argv": command, "returncode": completed.returncode,
                                    "seconds": time.perf_counter()-started, "log": str(logfile)})
            report["fixtures"] = fixture_provenance(fixtures)
            manifest.write_text(json.dumps(report, indent=2)+"\n")
            if completed.returncode:
                raise RuntimeError(f"{name} failed; see {logfile}")
            print(f"Completed {name}; log: {logfile}", flush=True)
    except Exception:
        (out / "FAILED").touch()
        raise
    (out / "COMPLETED").touch()


if __name__ == "__main__":
    main()
