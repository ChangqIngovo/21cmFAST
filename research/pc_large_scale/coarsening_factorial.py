#!/usr/bin/env python3
"""Isolate input smoothing on one fine mesh before comparing coarse transport.

The factorial has original, density-smoothed, source-smoothed, and both-smoothed
fine-grid inputs. Coarse counts are lifted uniformly to the fine grid, preserving
all block totals. Sources remain at fine-cell centres in these four solves;
the fifth solve instead uses the actual coarse cells and their source centres.
All spectra are measured after projection onto one common coarse mesh.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
sys.path.insert(0, str(RESEARCH/'pc_adaptive_front'))
sys.path.insert(0, str(RESEARCH/'pc_reconsidered'))
import adaptive
from diagnose_coarsening import compare_projection
from pc_statistics import coarsen_counts, spectrum, total


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def lift_counts(coarse, factor):
    """Uniformly split counts; this changes their spatial source distribution."""
    if type(factor) is not int or factor < 1:
        raise ValueError('Positive integer lifting factor required')
    return coarse.repeat(factor, 0).repeat(factor, 1).repeat(factor, 2)/factor**3


def run(fixture, coarse_grid, dose, angles, box_hMpc, band, horizon_boxes=4.):
    fixture = Path(fixture).resolve()
    deps = ['pc_adaptive_front/adaptive.py', 'pc_adaptive_front/adaptive.cpp',
            'pc_reconsidered/hybrid.py', 'pc_refinement/front_subcells.py',
            'pc_alternatives/characteristics.py', 'pc_large_scale/pc_statistics.py',
            'pc_large_scale/diagnose_coarsening.py', 'pc_large_scale/coarsening_factorial.py']
    hashes = {name: digest(RESEARCH/name) for name in deps}
    fixture_hash = digest(fixture)
    with np.load(fixture) as data:
        sf, cf = np.array(data['source']), np.array(data['capacity'])
    n = sf.shape[0]
    if not 0 < dose < 1 or coarse_grid >= n or n % coarse_grid:
        raise ValueError('Use a sub-unity photon dose and a proper divisor coarse grid')
    sc, cc = coarsen_counts(sf, coarse_grid), coarsen_counts(cf, coarse_grid)
    lifted_s, lifted_c = lift_counts(sc, n//coarse_grid), lift_counts(cc, n//coarse_grid)
    variants = {'original': (sf, cf), 'density_smooth': (sf, lifted_c),
                'source_smooth': (lifted_s, cf), 'both_smooth': (lifted_s, lifted_c),
                'coarse_solve': (sc, cc)}
    results, arrays = {}, {}
    adaptive.library()  # Compilation is excluded from solve timings.
    for count in (angles, 2*angles):
        names = list(variants) if count == angles else ['original', 'coarse_solve']
        for name in names:
            s, c = variants[name]
            photons = s*dose*total(c)/total(s)
            start = time.perf_counter()
            solved = adaptive.solve(photons, c, min_angles=count, max_angles=count,
                                    maximum_distance=horizon_boxes*len(s))
            seconds = time.perf_counter()-start
            absorbed = solved['absorbed']
            residual = (np.sum(photons, dtype=np.longdouble)-np.sum(absorbed, dtype=np.longdouble)
                        -np.sum(solved['source_retained'], dtype=np.longdouble)
                        -np.sum(solved['source_escaped'], dtype=np.longdouble))
            active = photons > 0
            source_residual = (solved['source_absorbed']+solved['source_retained']
                               +solved['source_escaped']-photons)
            source_error = float(np.max(abs(source_residual[active])/photons[active]))
            local_error = float(np.max(np.maximum(absorbed-c, 0)/c))
            global_error = float(abs(residual)/total(photons))
            if max(source_error, local_error, global_error) > 2e-9 or (absorbed < 0).any():
                raise RuntimeError('Independent photon accounting failed')
            a = coarsen_counts(absorbed, coarse_grid)
            cap = coarsen_counts(c, coarse_grid)
            stat = spectrum(cap, a, box_hMpc, np.geomspace(.01, 1., 24), band, 8.)
            key = f'{name}_K{count}'
            arrays[key] = absorbed
            results[key] = {'seconds': seconds, 'Q_mass': total(absorbed)/total(c),
                            'retained_fraction': total(solved['source_retained'])/total(photons),
                            'b2_common_mesh': stat['k0_band']['b_HI_squared'],
                            'full_fft_modes': stat['k0_band']['full_fft_modes'],
                            'global_relative_residual': global_error,
                            'maximum_source_relative_residual': source_error,
                            'maximum_local_fraction_violation': local_error}
            print(key, json.dumps(results[key]), flush=True)
    decompositions = {str(count): compare_projection(cf, arrays[f'original_K{count}'], cc,
                                arrays[f'coarse_solve_K{count}'], box_hMpc, band)
                      for count in (angles, 2*angles)}
    if fixture_hash != digest(fixture) or hashes != {name: digest(RESEARCH/name) for name in deps}:
        raise RuntimeError('Inputs or solver code changed during experiment')
    return {'scope': 'Frozen native fixture factorial; not cosmological evolution or continuum convergence',
            'fixture': str(fixture), 'fixture_sha256': fixture_hash, 'code_sha256': hashes,
            'config': {'fine_grid': n, 'coarse_grid': coarse_grid, 'dose': dose,
                       'angles': angles, 'angle_check': 2*angles, 'box_hMpc': box_hMpc,
                       'band_hMpc_inv': list(band), 'horizon_boxes': horizon_boxes},
            'models': results, 'projection_decomposition': decompositions,
            'interpretation': 'The four fine-grid cases isolate density/source smoothing and their interaction. The remaining direct-coarse change combines source-centre relocation and front spatial discretization; these are not separately identified.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--coarse-grid', type=int, default=16)
    p.add_argument('--dose', type=float, default=.8)
    p.add_argument('--angles', type=int, default=384)
    p.add_argument('--box-hMpc', type=float, default=135.6)
    p.add_argument('--band', type=float, nargs=2, default=[.04, .08])
    p.add_argument('--horizon-boxes', type=float, default=4.)
    args = p.parse_args()
    out = args.output
    result = run(args.fixture, args.coarse_grid, args.dose, args.angles,
                 args.box_hMpc, args.band, args.horizon_boxes)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
