"""Decompose late PC resolution differences using already measured spectra.

No grid interpolation or new transport. Cross-bias is relative to the density
field of each resolution. The matter-orthogonal residual is a spectral
decomposition, not a measurement of numerical noise or physical randomness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def stats(values):
    a = np.asarray(values, dtype=float)
    return {'mean': float(a.mean()), 'seed_sample_std': float(a.std(ddof=1)),
            'minimum': float(a.min()), 'maximum': float(a.max())}


def bias_terms(row, q):
    b = row['P_cross']/row['P_matter']
    b2 = row['P_HI']/row['P_matter']
    residual = b2-b*b
    return {'b_HI_cross': b, 'b_ionized_cross': (1-(1-q)*b)/q,
            'b_HI_squared': b2, 'matter_correlated_b2': b*b,
            'matter_orthogonal_b2': residual,
            'matter_orthogonal_fraction': residual/b2}


def analyze(path, projection_path, box_hMpc=500.):
    path, projection_path = Path(path), Path(projection_path)
    data = json.loads(path.read_text())
    projection = json.loads(projection_path.read_text())
    lookup = {(r['task']['seed'], r['task']['grid'], r['task']['dose']): r
              for r in data['case_results']
              if r['task']['distance_factor'] == 8 and r['task']['angle_factor'] == 1}
    seeds = sorted({k[0] for k in lookup})
    grids = sorted({k[1] for k in lookup}, reverse=True)
    doses = sorted({k[2] for k in lookup})
    output = {'scope': 'Supplied extended-distance, fixed-budget PC spectra; no raw grids',
              'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (path, projection_path)},
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'box_hMpc': box_hMpc, 'distance_factor': 8, 'angle_factor': 1,
              'maximum_actual_Q_minus_dose': max(abs(r['spectra']['Q_mass']-key[2]) for key, r in lookup.items()),
              'maximum_retained_fraction': max(r['ledger']['retained_fraction'] for r in lookup.values()),
              'seeds': seeds, 'comparisons': []}
    for dose in doses:
        for coarse in grids[1:]:
            seed_rows = []
            shell_rows = {}
            for seed in seeds:
                fine = lookup[seed, grids[0], dose]['spectra']
                low = lookup[seed, coarse, dose]['spectra']
                f = bias_terms(fine['k0_band'], fine['Q_mass'])
                c = bias_terms(low['k0_band'], low['Q_mass'])
                change = {k: c[k]-f[k] for k in ('b_HI_squared', 'matter_correlated_b2', 'matter_orthogonal_b2')}
                if not np.isclose(change['b_HI_squared'], change['matter_correlated_b2']+change['matter_orthogonal_b2'], rtol=1e-12, atol=1e-12):
                    raise RuntimeError('Power decomposition failed')
                seed_rows.append({'seed': seed, 'fine_Q': fine['Q_mass'], 'coarse_Q': low['Q_mass'],
                                  'fine': f, 'coarse': c, 'coarse_minus_fine': change,
                                  'ionized_crossbias_coarse_over_fine_minus_one': c['b_ionized_cross']/f['b_ionized_cross']-1,
                                  'native_fine_over_coarse_b2_minus_one': f['b_HI_squared']/c['b_HI_squared']-1})
                fs = {s['bin_index']: s for s in fine['shells']}
                for row in low['shells']:
                    if row['k_max'] > np.pi*coarse/box_hMpc:
                        continue  # Drop a shell truncated by the coarse Nyquist sphere.
                    ff = fs[row['bin_index']]
                    if row['full_fft_modes'] != ff['full_fft_modes'] or row['k_min'] != ff['k_min'] or row['k_max'] != ff['k_max']:
                        raise RuntimeError('Shells do not use the same Fourier modes')
                    bf, bc = bias_terms(ff, fine['Q_mass']), bias_terms(row, low['Q_mass'])
                    shell_rows.setdefault(row['bin_index'], []).append({
                        'k_hMpc_inv': row['k_hMpc_inv'], 'full_fft_modes': row['full_fft_modes'],
                        'fine_over_coarse_b2_minus_one': bf['b_HI_squared']/bc['b_HI_squared']-1,
                        'coarse_over_fine_ionized_crossbias_minus_one': bc['b_ionized_cross']/bf['b_ionized_cross']-1})
            mean_delta = {k: float(np.mean([r['coarse_minus_fine'][k] for r in seed_rows]))
                          for k in seed_rows[0]['coarse_minus_fine']}
            bands = {}
            for key in seed_rows[0]['fine']:
                bands[key] = {side: stats([r[side][key] for r in seed_rows]) for side in ('fine', 'coarse')}
            rows = [r for r in projection['rows'] if r['dose'] == dose and r['coarse_grid'] == coarse and r['model'] == 'pc']
            output['comparisons'].append({
                'dose': dose, 'fine_grid': grids[0], 'coarse_grid': coarse,
                'k0_band_terms': bands, 'k0_seed_terms': seed_rows,
                'k0_coarse_minus_fine_mean_b2': mean_delta,
                'fraction_of_mean_b2_change_in_correlated_component': mean_delta['matter_correlated_b2']/mean_delta['b_HI_squared'],
                'ionized_crossbias_relative_change': stats([r['ionized_crossbias_coarse_over_fine_minus_one'] for r in seed_rows]),
                'parent_original_common_mesh_b2_ratio': stats([r['input_coarsening_and_solver_factor'] for r in rows]),
                'parent_original_common_mesh_HI_correlation': stats([r['same_mesh_HI_cross_correlation'] for r in rows]),
                'shared_shells': [dict(bin_index=idx, k_hMpc_inv=rr[0]['k_hMpc_inv'], full_fft_modes=rr[0]['full_fft_modes'],
                                      fine_over_coarse_b2_minus_one=stats([r['fine_over_coarse_b2_minus_one'] for r in rr]),
                                      coarse_over_fine_ionized_crossbias_minus_one=stats([r['coarse_over_fine_ionized_crossbias_minus_one'] for r in rr]))
                                  for idx, rr in sorted(shell_rows.items())]})
    output['interpretation'] = [
        'delta_HI=(delta_m-Q*delta_ionized)/(1-Q); b_HI_cross=(1-Q*b_ionized_cross)/(1-Q).',
        'Matter-orthogonal power includes physical nonlinear structure and any discretization contribution; it is not identified as shot noise.',
        'The first shell has only six full FFT modes per seed. These spectra cannot establish the exact k->0 limit.',
        'Parent projection factors refer to original matched-mass maps, whereas main terms here refer to extended-distance fixed-budget maps.']
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary', type=Path, required=True)
    p.add_argument('--projection', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = analyze(args.summary, args.projection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
