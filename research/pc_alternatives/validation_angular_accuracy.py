"""Reproduce fixes for silent low-K/range acceptance in the research solver.

This is numerical convergence evidence, not a physical-accuracy certificate.
All source/capacity inputs are synthetic fixed-rate no-recombination controls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from angular_accuracy import AccuracyNotReached, mass_difference, solve_controlled
from characteristics import directions
from event_front import library, solve
from validation_geometry import sphere_inputs


def record(result, capacity):
    out = {"solver_audit": result["audit"], "controller": result.get("accuracy")}
    q = result['audit']['photons']
    out['relative_ledger_errors'] = {name: abs(result['audit'][name])/q if q else 0
                                   for name in ['ledger_residual', 'per_source_ledger_max',
                                                'per_angle_ledger_max', 'independent_absorption_ledger']}
    assert max(out['relative_ledger_errors'].values()) < 2e-11
    assert np.min(result['absorbed']) >= 0
    assert np.max(result['absorbed'] - capacity) <= 0
    return out


def no_escape_sphere():
    """Repeat the previously failing R8/K48 case in the same 128-cubed box."""
    n, radius = 128, 8
    s = np.zeros((n,)*3)
    c = np.ones_like(s)
    s[(n//2,)*3] = q = 4*np.pi*radius**3/3
    # Embed a local voxel-integrated sphere; avoids a needless large quadrature.
    _, _, small_reference, *_ = sphere_inputs(radius)
    reference = np.zeros_like(c)
    h = small_reference.shape[0]//2
    reference[n//2-h:n//2+h, n//2-h:n//2+h, n//2-h:n//2+h] = small_reference
    low = solve(s, c, angles=48)
    assert low['audit']['escaped'] == 0
    low_l1 = float(abs(low['absorbed'] - reference).sum()/q)
    try:
        solve_controlled(s, c, maximum_angles=48)
    except AccuracyNotReached as failure:
        low_rejected = failure.result['accuracy']['status'] == 'unresolved'
    else:
        raise AssertionError('K48 must not be silently accepted')
    repaired = solve_controlled(s, c, angular_rtol=.03)
    improved_l1 = float(abs(repaired['absorbed'] - reference).sum()/q)
    high = solve(s, c, angles=196608)
    outside_phase = Rotation.from_rotvec([.513, -.891, 1.209]).as_matrix()
    phase = solve(s, c, ray_directions=directions(49152) @ outside_phase.T)
    result = dict(radius_cells=radius, grid=n, photons=q, low_K48_rejected=low_rejected,
                  low_K48_continuum_atom_l1_over_Q=low_l1,
                  corrected_continuum_atom_l1_over_Q=improved_l1,
                  corrected_vs_K196608_atom_l1_over_Q=float(abs(repaired['absorbed']-high['absorbed']).sum()/q),
                  independent_rotation_atom_l1_over_Q=float(abs(repaired['absorbed']-phase['absorbed']).sum()/q),
                  old=record(low, c), corrected=record(repaired, c), high_K196608=record(high, c),
                  limitation='The continuum sphere error is a spatial opaque-voxel-front error after angular convergence; it remains larger than the controller angular tolerance.')
    assert improved_l1 < .06 < low_l1
    assert repaired['audit']['escaped'] == 0
    return result


def spatial_sequence():
    rows=[]
    for radius in [4, 8, 12, 16]:
        s, c, ref, *_ = sphere_inputs(radius)
        result = solve_controlled(s, c, angular_rtol=.03, maximum_angles=196608)
        rows.append(dict(radius_cells=radius, physical_cell_width_over_R=1/radius,
                         continuum_atom_l1_over_Q=float(abs(result['absorbed']-ref).sum()/s.sum()),
                         **record(result, c)))
    assert all(b['continuum_atom_l1_over_Q'] < a['continuum_atom_l1_over_Q']
               for a, b in zip(rows, rows[1:]))
    return dict(description='Same physical sphere, progressively smaller cells; each grid has its own accepted angular convergence checks.',
                rows=rows)


def multisource_heterogeneous():
    n=24
    x, y, z = np.indices((n,)*3)
    c = .25 + .06*x + .02*y + .015*z
    c[12:14, 5:19, 5:19] *= 5
    s = np.zeros_like(c)
    s[7, 10, 12], s[17, 14, 12] = 600, 450
    out = solve_controlled(s, c, angular_rtol=.04, maximum_angles=196608)
    phase = Rotation.from_rotvec([.394, 1.072, -.731]).as_matrix()
    reference = solve(s, c, ray_directions=directions(196608) @ phase.T)
    return dict(grid=n, source_cells=2, model='Two overlapping fronts; spatially varying capacity plus dense slab',
                result=record(out, c), independent_reference=record(reference, c),
                independent_high_K_differences={key: value/s.sum() for key, value in mass_difference(out, reference).items()},
                reference_limitation='Same discretized front model, independent quadrature phase at K196608; not continuum RT truth')


def shadow_and_range():
    s=np.zeros((12,)*3);c=np.zeros_like(s)
    s[2, 4, 6], s[3, 8, 5] = 2, 3
    c[7,:,:] = 1e6
    c[8:,:,:] = 1
    out=solve_controlled(s, c, angular_rtol=.1, maximum_angles=196608,
                         maximum_distance=.25, maximum_range=32, range_rtol=0)
    assert not np.any(out['absorbed'][8:])
    assert out['audit']['retained'] == 0
    return dict(grid=12, photons=float(s.sum()), shadow_absorption=float(out['absorbed'][8:].sum()),
                description='Two source cells on one side of an opaque plane; target gas behind plane must remain dark. Initial numerical range .25 is extended with a bound until no photons are range-limited.',
                result=record(out, c))


def blind_spot_and_resource_limits():
    s=np.zeros((96,)*3);c=np.zeros_like(s)
    s[1, 48, 48], c[94, 48, 48] = 1, 10
    out=solve_controlled(s, c, initial_angles=3, maximum_angles=48,
                         angular_rtol=1e-12, raise_unresolved=False)
    exact=float(np.arctan(.25/(92.5*np.sqrt(92.5**2+.5)))/np.pi)
    assert out['accuracy']['stop_reason'] == 'zero_absorption_coverage_unresolved'
    guarded=solve_controlled(np.ones((2,)*3), np.ones((2,)*3),
                             maximum_ray_bytes=1, raise_unresolved=False)
    assert guarded['absorbed'] is None and guarded['accuracy']['kernel_calls'] == 0
    return dict(missed_opaque_voxel=dict(grid=96, source=[1,48,48], absorber=[94,48,48],
                exact_continuum_absorbed_fraction=exact, quadrature_absorbed_fraction=0,
                controller=out['accuracy'],
                limitation='The zero-hit guard catches this concrete blind spot. Some absorption elsewhere can still hide an unsampled feature; empirical convergence is not a general angular certificate.'),
                resource_guard=guarded['accuracy'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(__file__).with_suffix('.json'))
    args=parser.parse_args()
    library()  # Exclude compiler startup from this validation's kernel timings.
    began=time.perf_counter()
    output = dict(scope='Synthetic numerical validation of controlled angular/range convergence; fixed-rate, infinite-c, opaque voxel-front, no recombination. No 21cmFAST production integration changed.',
                  algorithm='Refine by factor four; two fixed quadrature rotations; two consecutive levels below the photon-count L1 tolerance; strict unresolved failure by default; separate monotone retained-photon range bound.')
    output['no_escape_sphere'] = no_escape_sphere()
    output['spatial_sequence'] = spatial_sequence()
    output['multisource_heterogeneous'] = multisource_heterogeneous()
    output['shadow_and_range'] = shadow_and_range()
    output['blind_spot_and_resource_limits'] = blind_spot_and_resource_limits()
    output['wall_seconds'] = time.perf_counter()-began
    folder=Path(__file__).parent
    output['source_sha256']={name:hashlib.sha256((folder/name).read_bytes()).hexdigest()
                             for name in ['event_front.cpp', 'event_front.py', 'angular_accuracy.py',
                                          'validation_angular_accuracy.py']}
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({'output':str(args.output), 'wall_seconds':output['wall_seconds'],
                      'sphere_before':output['no_escape_sphere']['low_K48_continuum_atom_l1_over_Q'],
                      'sphere_after':output['no_escape_sphere']['corrected_continuum_atom_l1_over_Q']}, indent=2))


if __name__=='__main__':
    main()
