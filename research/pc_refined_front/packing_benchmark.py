#!/usr/bin/env python3
"""Compare ray storage implementations at exactly the same discrete model."""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent/'pc_large_scale'))
import refined
from pc_statistics import coarsen_counts, total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--baseline-cpp', type=Path, default=HERE/'reference_prepack.cpp')
    p.add_argument('--candidate-cpp', type=Path, default=HERE/'refined.cpp')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--angles', nargs='+', type=int, default=[384, 1536])
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--last-repeats', type=int)
    p.add_argument('--grid', type=int, default=16)
    args = p.parse_args()
    signatures = refined.library()
    library = Path(tempfile.mkdtemp(prefix='pc_prepack_'))/'baseline.so'
    subprocess.run(['g++', '-O3', '-std=c++17', '-shared', '-fPIC', str(args.baseline_cpp), '-o', str(library)], check=True)
    old = ct.CDLL(str(library))
    candidate = library.parent/'candidate.so'
    subprocess.run(['g++', '-O3', '-std=c++17', '-shared', '-fPIC', str(args.candidate_cpp), '-o', str(candidate)], check=True)
    new = ct.CDLL(str(candidate))
    for lib in (old, new):
        for name in ('refined_evolve', 'refined_state_count', 'refined_copy_state', 'refined_clear_template_cache'):
            getattr(lib, name).argtypes = getattr(signatures, name).argtypes
            getattr(lib, name).restype = getattr(signatures, name).restype
        for name in ('refined_set_contiguous_ray_budget', 'refined_last_ray_storage'):
            if hasattr(lib, name):
                getattr(lib, name).argtypes = getattr(signatures, name).argtypes
                getattr(lib, name).restype = getattr(signatures, name).restype
    with np.load(args.fixture) as d:
        source, capacity = (coarsen_counts(d[name], args.grid) for name in ('source', 'capacity'))
    source *= .9*total(capacity)/total(source)
    files = [args.baseline_cpp, args.candidate_cpp, HERE/'refined.py', Path(__file__)]
    hashes = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    report = dict(scope='Same sources, capacities, directions and arithmetic; ray representation and allocation only',
                  code_sha256=hashes, candidate_cpp=str(args.candidate_cpp),
                  fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
                  grid=args.grid, factor=4, dose=.9,
                  timing='Compilation excluded; explicit cold cache calls then warmup; alternating measured repeats', cases=[])
    for angles in args.angles:
        rows = {name: {'seconds_samples': []} for name in ('prepack', 'packed')}
        saved = {}
        def call(name):
            refined._library = old if name == 'prepack' else new
            return refined.solve(source, capacity, subcells_per_axis=4, angles=angles, maximum_distance=16*args.grid)
        for name in rows:
            refined._library = old if name == 'prepack' else new
            refined.clear_template_cache()
            started = time.perf_counter()
            out = call(name)
            rows[name]['cold_seconds'] = time.perf_counter()-started
            rows[name]['cold_audit'] = out['audit']
            call(name)
        repeats = args.last_repeats if angles == args.angles[-1] and args.last_repeats is not None else args.repeats
        if repeats < 1:
            raise ValueError('At least one measured repeat is required')
        for repeat in range(repeats):
            for name in (('prepack', 'packed') if repeat % 2 == 0 else ('packed', 'prepack')):
                started = time.perf_counter()
                out = call(name)
                rows[name]['seconds_samples'].append(time.perf_counter()-started)
                rows[name]['median_seconds'] = float(np.median(rows[name]['seconds_samples']))
                rows[name]['audit'] = out['audit']
                saved[name] = out
        checks = {key: bool(np.array_equal(saved['prepack'][key], saved['packed'][key]))
                  for key in ('absorbed', 'source_absorbed', 'source_escaped', 'source_retained')}
        checks.update({'state_'+key: bool(np.array_equal(saved['prepack']['state'][key], saved['packed']['state'][key]))
                       for key in ('ionized', 'cleared_bits', 'partial_ids', 'partial_ionized')})
        if not all(checks.values()):
            raise RuntimeError(f'Packing changed the discrete output: {checks}')
        row = dict(angles=angles, repeats=repeats, models=rows, bitwise_equal=checks,
                   packed_over_prepack_seconds=rows['packed']['median_seconds']/rows['prepack']['median_seconds'],
                   packed_over_prepack_ray_bytes=rows['packed']['audit']['ray_allocated_bytes']/rows['prepack']['audit']['ray_allocated_bytes'])
        report['cases'].append(row)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps(dict(angles=angles, seconds={k:v['median_seconds'] for k,v in rows.items()},
                              time_ratio=row['packed_over_prepack_seconds'], ray_bytes_ratio=row['packed_over_prepack_ray_bytes'],
                              bitwise_equal=checks)), flush=True)
    if hashes != {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}:
        raise RuntimeError('Code changed during packing comparison')
    refined._library = new


if __name__ == '__main__':
    main()
