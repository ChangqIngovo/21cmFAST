"""Angular accuracy of native-derived fields, including phase and density weights.

This is a frozen no-recombination rate experiment. The brightness proxy uses
fixed native density and spin temperature, with no velocity mapping or coupled
thermal history. Its ratios test angular quadrature, not PC versus ES physics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from characteristics import directions
from event_front import library, solve
from event_native_benchmark import input_fields


HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def statistics(candidate, reference, boxlen):
    """Compare complete fields, auto power, cross power and residual power."""
    n = candidate.shape[0]
    difference = candidate - reference
    mean_a, mean_b = candidate.mean(), reference.mean()
    fa = np.fft.fftn(candidate - mean_a)
    fb = np.fft.fftn(reference - mean_b)
    pa, pb = np.abs(fa)**2, np.abs(fb)**2
    cross = (fa * fb.conj()).real
    error = np.abs(fa - fb)**2
    kmode = 2*np.pi*np.fft.fftfreq(n, boxlen/n)
    k = np.sqrt(sum(v*v for v in np.meshgrid(kmode, kmode, kmode, indexing="ij")))
    edges = np.geomspace(2*np.pi/boxlen*.9999, np.pi*n/boxlen*1.0001, 13)
    num = den = error_sum = 0.
    rows = []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        mask = (k >= low) & (k < high)
        modes = int(mask.sum())
        if not modes:
            continue
        a, b, ab, err = (float(array[mask].mean()) for array in (pa, pb, cross, error))
        rows.append(dict(k_cMpc_inv=float(k[mask].mean()), full_fft_modes=modes,
                         auto_power_ratio=a/b, cross_coherence=ab/np.sqrt(a*b),
                         residual_power_over_reference=err/b))
        num += modes*abs(a-b)
        den += modes*b
        error_sum += modes*err
    return dict(
        candidate_mean=float(mean_a), reference_mean=float(mean_b),
        mean_abs_difference=float(np.abs(difference).mean()),
        max_abs_difference=float(np.abs(difference).max()),
        rms_difference=float(np.sqrt(np.mean(difference**2))),
        correlation=float(np.corrcoef(candidate.ravel(), reference.ravel())[0, 1]),
        all_nonzero_modes_residual_power_over_reference=float(error.sum()/pb.sum()),
        shell_weighted_absolute_power_difference=float(num/den),
        shell_weighted_residual_power_over_reference=float(error_sum/den),
        max_shell_relative_auto_power_difference=float(max(abs(r['auto_power_ratio']-1) for r in rows)),
        min_shell_cross_coherence=float(min(r['cross_coherence'] for r in rows)),
        shells=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path('/workspace/scratch/40b418003e4a')
    parser.add_argument('--fixture', type=Path, default=root/'qin20_native_cost_grid32')
    parser.add_argument('--maps', type=Path, default=root/'event_native_grid32_optimized')
    parser.add_argument('--output', type=Path, default=HERE/'validation_native_observables.json')
    parser.add_argument('--regenerate-maps', action='store_true',
                        help='Regenerate K48/192/768 in --maps using the current solver, preserving historical maps elsewhere')
    parser.add_argument('--legacy-maps', type=Path,
                        help='Optional historical maps for measuring the numerical-fix regression')
    args = parser.parse_args()
    fields = input_fields(args.fixture/'source_capacity.npz')
    prep = json.loads((args.fixture/'preparation.json').read_text())
    old = json.loads((HERE/'event_native_optimized_benchmark.json').read_text())
    hashes = {name: digest(HERE/name) for name in ('event_front.py', 'event_front.cpp', 'characteristics.py')}
    map_manifest = args.maps/'validation_map_manifest.json'
    if args.regenerate_maps:
        args.maps.mkdir(parents=True, exist_ok=True)
        library()
        audits = {}
        for count in (48, 192, 768):
            run = solve(fields['source'], fields['capacity'], angles=count, periodic=True)
            np.savez_compressed(args.maps/f'event_K{count}.npz', absorbed=run['absorbed'],
                                xHI=1-fields['xray_fraction']-run['absorbed']/fields['density'])
            audits[str(count)] = run['audit']
            if args.legacy_maps is not None:
                legacy_file = args.legacy_maps/f'event_K{count}.npz'
                with np.load(legacy_file) as legacy:
                    delta = run['absorbed']-legacy['absorbed']
                    audits[str(count)]['numerical_fix_regression'] = dict(
                        legacy_map_sha256=digest(legacy_file),
                        legacy_solver_hashes=old['source_files_sha256'],
                        maximum_absorbed_change=float(np.max(np.abs(delta))),
                        absorption_L1_over_emitted=float(np.abs(delta).sum()/fields['source'].sum()))
        map_manifest.write_text(json.dumps(dict(source_files_sha256=hashes,
                                               input_sha256=digest(args.fixture/'source_capacity.npz'),
                                               audits=audits), indent=2)+'\n')
    provenance = json.loads(map_manifest.read_text()) if map_manifest.exists() else old
    if hashes != provenance['source_files_sha256']:
        raise RuntimeError('Cached maps belong to a different solver; regenerate and validate their provenance first')
    if provenance['input_sha256'] != digest(args.fixture/'source_capacity.npz'):
        raise RuntimeError('Cached maps do not match the supplied source fixture')
    rho, xe = fields['density'], fields['xray_fraction']
    spin_path = args.fixture/'frozen_native_step/spin.h5'
    with h5py.File(spin_path) as stream:
        spin = stream['TsBox/OutputFields/spin_temperature'][...].astype(float)
    z = float(prep['nodes'][-1])
    spin_weight = 1-2.7255*(1+z)/spin
    assert np.isfinite(spin_weight).all() and np.all(spin > 0)
    maps, absorbed_maps = {}, {}
    for count in (48, 192, 768):
        with np.load(args.maps/f'event_K{count}.npz') as data:
            maps[count] = data['xHI'].copy()
            absorbed_maps[count] = data['absorbed'].copy()
            np.testing.assert_allclose(maps[count], 1-xe-data['absorbed']/rho, atol=1e-15)
    result = dict(
        scope='Fixed native-derived 32^3 rate fixture: angular self-comparison only',
        input_sha256=digest(args.fixture/'source_capacity.npz'),
        map_provenance=provenance if map_manifest.exists() else {'record': str(HERE/'event_native_optimized_benchmark.json')},
        spin_sha256=digest(spin_path), source_files_sha256=hashes,
        redshift=z, box_cMpc=float(prep['box_mpc']),
        reference='K768 original Fibonacci directions; finite, not an RT truth field',
        fields={'xHI': 'neutral fraction', 'neutral_density': 'rho*xHI',
                'frozen_brightness_proxy': 'rho*xHI*(1-Tcmb/Ts), fixed Ts, no RSD, no mK prefactor'},
        rotation_seed=20261023,
        definitions={'power_summary': 'sum Nmodes*abs(Pcandidate-Pref) / sum Nmodes*Pref',
                     'residual_power': 'P(candidate-reference) / P(reference), includes phase differences',
                     'shell_limit': '0<k<=Cartesian Nyquist; all-mode residual also includes cube corners'},
        pairs=[], rotations=[], range_refinement=[],
    )

    def compare(left, right):
        return {name: statistics(left*weight, right*weight, prep['box_mpc']) for name, weight in
                [('xHI', 1.), ('neutral_density', rho), ('frozen_brightness_proxy', rho*spin_weight)]}

    for a, b in [(48, 192), (192, 768), (48, 768)]:
        result['pairs'].append(dict(candidate_K=a, reference_K=b, fields=compare(maps[a], maps[b])))
    rotations = Rotation.random(4, random_state=np.random.default_rng(result['rotation_seed'])).as_matrix()
    for count, number in [(48, 4), (192, 2)]:
        for i, matrix in enumerate(rotations[:number]):
            run = solve(fields['source'], fields['capacity'], periodic=True,
                        maximum_distance=rho.shape[0]*np.sqrt(3),
                        ray_directions=directions(count)@matrix.T)
            xhi = 1-xe-run['absorbed']/rho
            assert abs(run['audit']['ledger_residual']) < 1e-9*fields['source'].sum()
            result['rotations'].append(dict(K=count, rotation_index=i, matrix=matrix.tolist(),
                                           audit=run['audit'], fields=compare(xhi, maps[768])))
            args.output.write_text(json.dumps(result, indent=2)+'\n')
    for count in (48, 192):
        previous = absorbed_maps[count]
        base = provenance.get('audits', {}).get(str(count))
        if base is None:
            base = next(row for row in old['runs'] if row['angles'] == count)
        short_retained = base['retained']
        horizon = 2*rho.shape[0]*np.sqrt(3)
        run = solve(fields['source'], fields['capacity'], angles=count, periodic=True,
                    maximum_distance=horizon)
        delta = run['absorbed']-previous
        row = dict(K=count, distance_factor=2., maximum_distance=horizon,
                   short_retained=short_retained, audit=run['audit'],
                   maximum_absorption_decrease=float(np.maximum(-delta, 0).max()),
                   added_absorption=float(delta.sum()), absorption_L1=float(np.abs(delta).sum()),
                   absorption_L1_over_emitted=float(np.abs(delta).sum()/fields['source'].sum()),
                   frozen_brightness=statistics((1-xe-run['absorbed']/rho)*rho*spin_weight,
                                                maps[count]*rho*spin_weight, prep['box_mpc']))
        tolerance = 1e-11*fields['source'].sum()
        assert row['maximum_absorption_decrease'] <= tolerance
        assert row['absorption_L1'] <= short_retained+tolerance
        result['range_refinement'].append(row)
    result['summary'] = {}
    for count in (48, 192):
        samples = [r for r in result['rotations'] if r['K'] == count]
        result['summary'][f'K{count}_rotations_vs_original_K768'] = {
            name: {metric: max(r['fields'][name][metric] for r in samples) for metric in
                   ['mean_abs_difference', 'max_abs_difference', 'shell_weighted_absolute_power_difference',
                    'max_shell_relative_auto_power_difference', 'all_nonzero_modes_residual_power_over_reference']}
            for name in result['fields']}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
