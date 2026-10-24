"""Exact data-layout/heap alternatives to the lazy eight-subcell kernel.

The event priority remains (time, node). A wider heap changes only its storage
layout; compact rays preserve every floating-point field and use wide integer
fallback when the full finite support is not representable.
"""
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

HERE = Path(__file__).resolve().parent
VARIANTS = ("original", "heap2-wide", "heap4-wide", "heap8-wide",
            "heap2-compact", "heap4-compact", "heap8-compact",
            "heap4-wide-aligned", "heap4-compact-aligned",
            "calendar1024-compact", "calendar4096-compact", "calendar16384-compact",
            "local-dda-compact", "local-dda-wide", "face-table-compact")
_modules = {}


def module(variant="heap4-compact"):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    if variant not in _modules:
        spec = importlib.util.spec_from_file_location("_pc_adaptive_"+variant.replace("-", "_"),
                                                     HERE/"adaptive.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        calendar = variant.startswith("calendar")
        local = variant.startswith("local-dda")
        table = variant.startswith("face-table")
        arity = 2 if variant == "original" else 4 if calendar or local or table else int(variant.split("-")[0][4:])
        compact = "-compact" in variant
        source = HERE/("lazy.cpp" if variant == "original" else "face_table.cpp" if table else "local_dda.cpp" if local else "calendar.cpp" if calendar else "optimized.cpp")
        buckets = int(variant.split("-")[0][8:]) if calendar else 4096
        output = Path(tempfile.mkdtemp(prefix="pc_front_opt_"))/"optimized.so"
        subprocess.run(["g++", "-O3", "-std=c++17", "-shared", "-fPIC",
            f"-DPC_HEAP_ARITY={arity}", f"-DPC_COMPACT_RAYS={int(compact)}",
            f"-DPC_ALIGN_NODES={int(variant.endswith('-aligned'))}",
            f"-DPC_CALENDAR_BUCKETS={buckets}",
            str(source), "-o", str(output)], check=True)
        lib = ct.CDLL(str(output))
        lib.adaptive_evolve.argtypes = [ct.c_int, ct.c_void_p, ct.c_void_p,
            ct.c_int, ct.c_void_p, ct.c_void_p,
            ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_void_p,
            ct.c_int, ct.c_void_p, ct.c_double, ct.c_double, ct.c_int,
            *([ct.c_void_p]*8)]
        lib.adaptive_evolve.restype = ct.c_int
        lib.adaptive_state_count.argtypes=[]
        lib.adaptive_state_count.restype=ct.c_int64
        for name, args in (("adaptive_copy_state", [ct.c_void_p, ct.c_void_p]),
                          ("adaptive_copy_remaining", [ct.c_void_p]),
                          ("adaptive_copy_state_absorbed", [ct.c_void_p]),
                          ("adaptive_copy_fine_absorbed", [ct.c_void_p])):
            getattr(lib, name).argtypes=args
            getattr(lib, name).restype=ct.c_int
        mod.library = lambda: lib
        if table:
            lib.adaptive_face_table_bytes.argtypes=[]
            lib.adaptive_face_table_bytes.restype=ct.c_int64
            original_solve=mod.solve

            def measured_solve(*args, **kwargs):
                result=original_solve(*args,**kwargs)
                result["audit"]["face_table_bytes"]=int(lib.adaptive_face_table_bytes())
                return result

            mod.solve=measured_solve
        mod.library()
        _modules[variant]=mod
    return _modules[variant]


def solve(source, capacity, initial_ionized=None, *, variant="heap4-compact", **kwargs):
    result=module(variant).solve(source,capacity,initial_ionized,**kwargs)
    result["audit"]["exact_optimization"]=variant
    return result


def run(output, native32=None, native64=None, repeats=3,
        variants=VARIANTS):
    if repeats<1:
        raise ValueError("repeats must be positive")
    for name in variants:
        module(name)
    rng=np.random.default_rng(91755)
    cap=np.exp(.7*rng.normal(size=(16,)*3))
    src=np.zeros_like(cap);src[6,8,8],src[10,8,8]=150,95
    cases=[("unequal_density_pair",src,cap,np.zeros_like(cap),8.)]
    for name,path in (("native32",native32),("native64",native64)):
        if path:
            with np.load(path) as data:
                s,c=data["source"],data["capacity"]
                old=data["initial"] if "initial" in data else np.zeros_like(c)
                cases.append((name,s,c,old,16.))
    report={"scope":"Exact lazy-eight-subcell heap/ray/DDA layouts; unchanged transport arithmetic",
            "timing":"compilation excluded; sequential alternating variants; shared host",
            "cases":[],"code_hashes":{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in (Path(__file__),HERE/"adaptive.py",HERE/"adaptive.cpp",HERE/"lazy.cpp",HERE/"optimized.cpp",HERE/"calendar.cpp",HERE/"local_dda.cpp",HERE/"face_table.cpp")}}
    for name,s,c,old,horizon in cases:
        reference=module("original").solve(s,c,old,maximum_distance=horizon,return_dense=True)
        rows={v:{"variant":v,"times":[]} for v in variants}
        for repeat in range(repeats):
            order=list(variants) if repeat%2==0 else list(variants)[::-1]
            for v in order:
                started=time.perf_counter()
                out=module(v).solve(s,c,old,maximum_distance=horizon)
                elapsed=time.perf_counter()-started
                for field in ("absorbed","source_absorbed","source_escaped","source_retained","angle_counts"):
                    np.testing.assert_array_equal(out[field],reference[field])
                for field in ("ionized","parent_ids","child_ionized"):
                    np.testing.assert_array_equal(out["state"][field],reference["state"][field])
                rows[v]["times"].append(elapsed)
                rows[v].update(median_seconds=float(np.median(rows[v]["times"])),
                               audit=out["audit"],bitwise_map_and_source_ledgers=True)
        record={"case":name,"source_sha256":hashlib.sha256(s.tobytes()).hexdigest(),
                "capacity_sha256":hashlib.sha256(c.tobytes()).hexdigest(),
                "initial_sha256":hashlib.sha256(old.tobytes()).hexdigest(),
                "maximum_distance":horizon,"variants":list(rows.values())}
        report["cases"].append(record)
        Path(output).write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps({"case":name,"results":[{"variant":r["variant"],"seconds":r["median_seconds"],
            "ray_bytes":r["audit"]["ray_bytes"]} for r in rows.values()]}),flush=True)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native32",type=Path)
    parser.add_argument("--native64",type=Path)
    parser.add_argument("--output",type=Path,default=HERE/"optimized_results.json")
    parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument("--variants",nargs="+",choices=VARIANTS,default=list(VARIANTS))
    run(**vars(parser.parse_args()))
