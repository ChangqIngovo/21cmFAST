"""Compare front subdivision at identical source locations and angular counts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent/'pc_reconsidered'))
from front_subcells import solve, library
from hybrid import solve as original, library as original_library
from compare_models import make_cases


def statistics(candidate, reference, photons):
    return {'L1_over_Q': float(np.abs(candidate-reference).sum()/photons),
            'max_cell': float(np.abs(candidate-reference).max())}


def run(output, native=None, repeats=3):
    library(); original_library()
    source_files = (Path(__file__), HERE/'front_subcells.py', HERE/'front_subcells.cpp',
                    HERE.parent/'pc_reconsidered/hybrid.py', HERE.parent/'pc_reconsidered/hybrid.cpp')
    source_hashes = {str(p.relative_to(HERE.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    cases = make_cases(24)
    chosen = [cases[k] for k in ('isolated', 'equal-overlap', 'transparent-overlap-increment', 'opaque-plane')]
    result = {'scope': 'Same source points and total angular counts; refinement uniformly divides gas capacity; no chemistry.', 'cases': [],
              'source_hashes':source_hashes}
    for case in chosen:
        rows = {}
        fields = {}
        for name, factor in [('original-fibonacci', 0), ('balanced-coarse', 1), ('balanced-subcells', 2)]:
            elapsed = []
            for _ in range(repeats):
                kwargs = dict(periodic=True, maximum_distance=case.maximum_distance)
                out = original(case.source, case.capacity, case.initial, **kwargs) if not factor else solve(case.source, case.capacity, case.initial, subcells=factor, parent_skip=True, **kwargs)
                elapsed.append(out['audit']['seconds'])
            rows[name] = {'audit': out['audit'], 'seconds_samples': elapsed, 'median_seconds': float(np.median(elapsed))}
            fields[name] = out['absorbed']
            if case.analytic is not None:
                rows[name]['vs_analytic'] = statistics(out['absorbed'], case.analytic, case.source.sum())
        ref = solve(case.source, case.capacity, case.initial, subcells=2,
                    min_angles=49152, max_angles=49152, periodic=True,
                    maximum_distance=case.maximum_distance, parent_skip=True)
        for name in rows:
            rows[name]['vs_high_angle_subcells'] = statistics(fields[name], ref['absorbed'], case.source.sum())
        result['cases'].append({'name': case.name, 'models': rows, 'reference_audit': ref['audit'], 'metadata': case.metadata})
        Path(output).write_text(json.dumps(result, indent=2)+'\n')
        print(case.name, {k:(v['median_seconds'],v['vs_high_angle_subcells']['L1_over_Q']) for k,v in rows.items()}, flush=True)
    if native is not None:
        data = np.load(native)
        source, capacity = data['source'], data['capacity']
        rows = {}
        baseline = None
        for name, factor in [('original-fibonacci', 0), ('balanced-coarse', 1), ('balanced-subcells', 2)]:
            elapsed = []
            for _ in range(repeats):
                out = original(source, capacity, maximum_distance=len(source)/2) if not factor else solve(source, capacity, subcells=factor, maximum_distance=len(source)/2, parent_skip=True)
                elapsed.append(out['audit']['seconds'])
            if baseline is None:
                baseline = out['absorbed']
            rows[name] = {'audit':out['audit'], 'median_seconds':float(np.median(elapsed)), 'seconds_samples':elapsed,
                          'change_from_original':statistics(out['absorbed'], baseline, source.sum())}
        result['native'] = {'path':str(native), 'shape':list(source.shape), 'models':rows,
            'note':'Morphology changes here are not errors: no continuum oracle is available for this frozen native field.'}
        Path(output).write_text(json.dumps(result, indent=2)+'\n')
        print('native', {k:v['median_seconds'] for k,v in rows.items()},flush=True)
    final_hashes = {str(p.relative_to(HERE.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    if source_hashes != final_hashes:
        raise RuntimeError('Benchmark source files changed during execution; rerun the comparison')
    Path(output).write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=HERE/'front_results.json')
    p.add_argument('--native',type=Path)
    p.add_argument('--repeats',type=int,default=3)
    args=p.parse_args()
    run(**vars(args))
