#!/usr/bin/env python3
"""Separate output projection from changes induced by solving on coarse inputs.

A fine and coarse map have different cell windows and aliasing. Conservatively
project the *already solved* fine absorbed counts onto the coarse mesh first.
The projected fine and directly solved coarse maps then share exactly the same
atom capacities, pixel geometry and matter-power denominator. This does not
identify source smoothing versus transport error: both are in the second factor.
No source efficiency or ionization field is adjusted by this diagnostic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import fft

from pc_statistics import coarsen_counts, spectrum, total


def _ratio(x, y):
    return float(x/y) if x is not None and y is not None and y > 0 else None


def compare_projection(fine_capacity, fine_absorbed, coarse_capacity,
                       coarse_absorbed, box_hMpc, band, redshift=8.):
    """Exact multiplicative decomposition of the native fine/coarse bias ratio."""
    cf, af, cc, ac = map(lambda x: np.asarray(x, dtype=float),
                         (fine_capacity, fine_absorbed, coarse_capacity, coarse_absorbed))
    n = cc.shape[0]
    projected_c = coarsen_counts(cf, n)
    if (projected_c.shape != cc.shape or not np.allclose(projected_c, cc, rtol=2e-12, atol=0)):
        raise ValueError('Coarse capacity must be the conservative sum of fine capacity')
    if not 0 < band[0] < band[1] < np.pi*n/box_hMpc:
        raise ValueError('Comparison band must be below the coarse Nyquist frequency')
    projected_a = coarsen_counts(af, n)
    edges = np.geomspace(2*np.pi/box_hMpc*(1-1e-10),
                         np.pi*cf.shape[0]/box_hMpc*(1+1e-10), 33)
    stat = {}
    for name, cap, absorbed in [('fine_native', cf, af),
                                ('fine_projected', cc, projected_a),
                                ('coarse_solved', cc, ac)]:
        out = spectrum(cap, absorbed, box_hMpc, edges, band, redshift)
        if out['k0_band'] is None:
            raise ValueError('Comparison band has no Fourier modes')
        stat[name] = {'Q_mass': out['Q_mass'], **out['k0_band']}
    f, p, c = (stat[k]['b_HI_squared'] for k in stat)
    native = _ratio(f, c)
    projection = _ratio(f, p)
    solve = _ratio(p, c)
    # Cross-spectrum on the COMMON mesh measures phase agreement, which two
    # equal auto-spectra alone cannot establish. Hermitian multiplicities matter.
    h1, h2 = cc-projected_a, cc-ac
    g1, g2 = fft.rfftn(h1/h1.mean()-1), fft.rfftn(h2/h2.mean()-1)
    k = 2*np.pi*fft.fftfreq(n, d=box_hMpc/n)
    kz = 2*np.pi*fft.rfftfreq(n, d=box_hMpc/n)
    km = np.sqrt(k[:, None, None]**2+k[None, :, None]**2+kz[None, None, :]**2)
    w = np.full(kz.size, 2.)
    w[0] = 1
    if n % 2 == 0:
        w[-1] = 1
    weights = np.broadcast_to(w, km.shape)
    use = (km >= band[0]) & (km < band[1])
    cross = float(np.sum((g1[use]*g2[use].conj()).real*weights[use]))
    power1 = float(np.sum(abs(g1[use])**2*weights[use]))
    power2 = float(np.sum(abs(g2[use])**2*weights[use]))
    count_L1 = total(abs(projected_a-ac))
    return {
        'native_fine_over_coarse_b2': native,
        'output_projection_factor': projection,
        'input_coarsening_and_solver_factor': solve,
        'factorization_residual': (native-projection*solve
                                   if None not in (native, projection, solve) else None),
        'same_mesh_HI_cross_correlation': _ratio(cross, np.sqrt(power1*power2)),
        'same_mesh_absorbed_L1_over_atoms': count_L1/total(cc),
        'same_mesh_absorbed_L1_over_fine_absorbed': _ratio(count_L1, total(af)),
        'same_mesh_maximum_fraction_difference': float(np.max(abs(projected_a-ac)/cc)),
        'Q_mass_difference': stat['fine_projected']['Q_mass']-stat['coarse_solved']['Q_mass'],
        'spectra': stat,
        'interpretation': 'native ratio = output projection factor * input coarsening and solver factor; no deconvolution or field recalibration',
    }


def collect_projection(root, models=('pc', 'es_matched_mass'), verify=True):
    """Read a completed large-scale results directory; perform no transport."""
    root = Path(root)
    config = json.loads((root/'experiment.json').read_text())['config']
    tasks = json.loads((root/'tasks.json').read_text())
    lookup = {(t['seed'], t['grid'], t['dose']): t for t in tasks if t['variant']=='baseline'}
    verified = set()
    source_hashes = {}

    def load_fields(task):
        directory = root/'cases'/task['directory']
        path = directory/'fields.npz'
        if path not in verified:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if verify:
                receipt = json.loads((directory/'COMPLETED.json').read_text())
                if actual != receipt['files_sha256']['fields.npz']:
                    raise RuntimeError(f'Completed field changed: {path}')
            source_hashes[str(path.relative_to(root))] = actual
            verified.add(path)
        with np.load(path) as arrays:
            return {model: np.array(arrays[f'{model}_absorbed']) for model in models}

    rows = []
    for seed in config['seeds']:
        caps = {}
        for n in config['grids']:
            path = root/'fixtures'/f'seed{seed}'/f'n{n}'/'capacity.npy'
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if verify:
                receipt = json.loads((path.parent.parent/'COMPLETED.json').read_text())
                if actual != receipt['files_sha256'][str(path.relative_to(path.parent.parent))]:
                    raise RuntimeError(f'Completed capacity changed: {path}')
            source_hashes[str(path.relative_to(root))] = actual
            caps[n] = np.load(path)
        nf = config['native_grid']
        for dose in config['photon_doses']:
            fine = load_fields(lookup[seed, nf, dose])
            for nc in config['grids']:
                if nc == nf:
                    continue
                coarse = load_fields(lookup[seed, nc, dose])
                for model in models:
                    row = compare_projection(caps[nf], fine[model], caps[nc], coarse[model],
                                              config['box_hMpc'], config['k0_band_hMpc_inv'], config['redshift'])
                    row.update(seed=seed, dose=dose, model=model, fine_grid=nf, coarse_grid=nc,
                               mass_matched=abs(row['Q_mass_difference']) <= 2*config['mass_match_tolerance'])
                    rows.append(row)
    return {'scope': 'existing-map projection diagnostic; no new photon transport',
            'rows': rows, 'input_sha256': source_hashes,
            'tolerance_note': 'Same-mesh agreement isolates the combined effect of input coarsening and the solver; it is not a continuum-accuracy certificate.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Large-run results directory containing experiment.json')
    parser.add_argument('--output', type=Path, help='Defaults to ROOT/coarsening_diagnostic.json')
    args = parser.parse_args()
    result = collect_projection(args.root)
    output = args.output or args.root/'coarsening_diagnostic.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(output), 'comparisons': len(result['rows'])}))


if __name__ == '__main__':
    main()
