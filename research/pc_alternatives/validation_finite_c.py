"""Finite-c packet fixes: analytic delays, persistent state and separated refinements."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from angular_tests import sphere_voxel_reference
from finite_c_packets import FiniteCPackets
from validation_time_state import delayed_absorption, relativistic_radius

HERE = Path(__file__).resolve().parent


def beam_control(steps, observation=3.014):
    cap = np.zeros((8,)*3)
    cap[6, 1, 1] = 100
    rate = np.zeros_like(cap)
    rate[1, 1, 1] = 1
    solver = FiniteCPackets(cap, speed=2, ray_directions=[[1., 0, 0]])
    solver.schedule_constant(rate, 0, 1, emission_steps=steps)
    result = solver.advance(observation)
    exact = delayed_absorption(observation, 1, 4.5, 2, emission_stop=1)
    error = abs(result['audit']['absorbed']-exact)
    bound = .5/steps
    assert error <= bound+1e-14
    return dict(emission_steps=steps, emission_dt=1/steps,
                observation=observation, exact_absorbed=exact,
                packet_absorbed=result['audit']['absorbed'],
                absolute_error=error, exact_midpoint_boundary_error_bound=bound,
                audit=result['audit'])


def fixed_counterexamples():
    cap = np.zeros((8,)*3)
    cap[6, 1, 1] = 100
    rate = np.zeros_like(cap)
    rate[1, 1, 1] = 1
    delayed = FiniteCPackets(cap, speed=2, ray_directions=[[1., 0, 0]])
    delayed.schedule_constant(rate, 0, 1, emission_steps=32)
    rows = [delayed.advance(t)['audit'] for t in (1, 2, 3, 3.25)]
    capped = FiniteCPackets(cap, speed=2, maximum_distance=3,
                           ray_directions=[[1., 0, 0]])
    capped.schedule_flash((1, 1, 1), 1)
    for t in (1, 2, 3, 10):
        capped.advance(t)
    retained = capped.snapshot(include_packets=True)
    near_cap = np.zeros((4,)*3)
    near_cap[2, 1, 1] = 100
    near = FiniteCPackets(near_cap, speed=1, ray_directions=[[1., 0, 0]])
    near_rate = np.zeros_like(near_cap)
    near_rate[1, 1, 1] = 1
    near.schedule_constant(near_rate, 0, 1, emission_steps=64)
    return dict(source_off_arrivals=rows,
                former_c_dt_failure_fixed=near.advance(1)['audit'],
                cumulative_range_preserved=dict(audit=retained['audit'],
                                                 packets=retained['packets']))


def sphere(n, angles, steps):
    width = 4.5/n
    density, rate, speed, duration = 3/(4*np.pi), 1., 1., 1.
    cap = np.full((n,)*3, density*width**3)
    src = np.zeros_like(cap)
    center = (n//2,)*3
    src[center] = rate
    start = time.perf_counter()
    solver = FiniteCPackets(cap, speed=speed, cell_width=width, angles=angles)
    solver.schedule_constant(src, 0, duration, emission_steps=steps)
    result = solver.advance(duration)
    elapsed = time.perf_counter()-start
    radius = relativistic_radius(duration, rate, density, speed)
    # Independent continuous-sphere voxel-volume reference. Its expected atom
    # count is R^3 in these units. Arrival-face transport has O(dx) spatial error.
    ideal = sphere_voxel_reference(n, radius/width, center, quadrature=48)*cap.flat[0]
    ideal24 = sphere_voxel_reference(n, radius/width, center, quadrature=24)*cap.flat[0]
    absorbed = result['absorbed']
    effective_radius = (absorbed.sum()/(4*np.pi*density/3))**(1/3)
    coords = (np.indices(cap.shape)-np.asarray(center)[:, None, None, None])*width
    rcenter = np.sqrt((coords*coords).sum(axis=0))
    # Absorption happens at voxel entry: the furthest point represented by that
    # voxel is at most sqrt(3)*dx beyond the causal radius, not exactly at its centre.
    supercausal = absorbed[rcenter > speed*duration+np.sqrt(3)*width].sum()
    assert supercausal == 0
    row = dict(grid=n, cell_width=width, angles=angles, emission_steps=steps,
               emitted=rate*duration, analytic_radius=radius,
               effective_ionized_volume_radius=float(effective_radius),
               relative_radius_error=float(effective_radius/radius-1),
               atom_l1_vs_continuum_per_analytic_ionized=float(abs(absorbed-ideal).sum()/radius**3),
               reference_quadrature24_48_l1=float(abs(ideal-ideal24).sum()/radius**3),
               reference_atom_count_relative_error=float(ideal.sum()/radius**3-1),
               absorption_beyond_causal_radius_plus_voxel_diagonal=float(supercausal),
               seconds=elapsed, audit=result['audit'])
    return row, absorbed


def _sphere_job(config):
    row, field = sphere(*config)
    print(json.dumps(dict(completed_sphere=config, seconds=row['seconds'])), flush=True)
    return config, row, field


def refinements(workers=4):
    # Independent configurations run on separate processes, bounded to four.
    # Each configuration remains a serial packet reference.
    jobs = [(15, 768, 512)] + [(15, 768, s) for s in (8,16,32,64,128,256)]
    jobs += [(n, 3072, 128) for n in (9,15,21,33,45)]
    jobs += [(21, k, 128) for k in (48,192,768,12288)]
    jobs += [(45,12288,128)]
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for config, row, field in pool.map(_sphere_job, jobs):
            results[config] = (row, field)
    reference_row, reference = results[(15,768,512)]
    temporal = []
    for steps in (8,16,32,64,128,256):
        row, field = results[(15,768,steps)]
        row['atom_l1_vs_same_grid_K768_steps512_per_emitted'] = float(abs(field-reference).sum())
        temporal.append(row)
    spatial = [results[(n,3072,128)][0] for n in (9,15,21,33,45)]
    angle_reference_row, angle_reference = results[(21,12288,128)]
    angular = []
    for k in (48,192,768,3072):
        row, field = results[(21,k,128)]
        row = row.copy()
        row['atom_l1_vs_same_grid_steps128_K12288_per_emitted'] = float(abs(field-angle_reference).sum())
        angular.append(row)
    fine_row, fine_reference = results[(45,12288,128)]
    fine_low_row, fine_low = results[(45,3072,128)]
    fine_angular = dict(grid=45, emission_steps=128, low_angles=3072, high_angles=12288,
                        validation_workers=workers,
                        atom_l1_per_emitted=float(abs(fine_low-fine_reference).sum()),
                        low_relative_radius_error=fine_low_row['relative_radius_error'],
                        high_relative_radius_error=fine_row['relative_radius_error'],
                        high_angle_result=fine_row)
    return dict(validation_workers=workers, finest_grid_angular_check=fine_angular,
                temporal=dict(fixed_grid=15, fixed_angles=768,
                              reference=reference_row, rows=temporal),
                spatial=dict(fixed_angles=3072, fixed_emission_steps=128, rows=spatial),
                angular=dict(fixed_grid=21, fixed_emission_steps=128,
                             reference=angle_reference_row, rows=angular))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=HERE/'validation_finite_c.json')
    parser.add_argument('--workers', type=int, default=4, choices=range(1,5))
    args = parser.parse_args()
    result = dict(scope='Research finite-c packet reference, fixed gas/no recombination; no native Qin20 integration',
                  implementation_sha256=hashlib.sha256((HERE/'finite_c_packets.py').read_bytes()).hexdigest(),
                  model='Midpoint photon impulses, first opaque voxel entry absorption, permanent transparency, source-centred straight rays',
                  fixed_counterexamples=fixed_counterexamples(),
                  analytic_delay_temporal=[beam_control(n) for n in (8,16,32,64,128,256,512)],
                  refinement=refinements(args.workers),
                  limitations=['Emission quadrature is first-order bounded at arrival discontinuities; smooth midpoint quadrature alone does not imply second-order evolving fronts.',
                               'Same-time sharing uses exact equal long-double queue times; no tolerance coalesces near events.',
                               'Finite-c transport is causal at voxel entry; cell contents are lumped there, leaving an O(cell-width) spatial bias.',
                               'No recombination, finite opacity, hydrodynamic/advection, spectral, expanding-universe, or native feedback evolution.',
                               'Python packet reference is not a measured low-cost replacement for production 21cmFAST.'])
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),
                         spatial_radius_errors=[r['relative_radius_error'] for r in result['refinement']['spatial']['rows']],
                         temporal_l1=[r['atom_l1_vs_same_grid_K768_steps512_per_emitted'] for r in result['refinement']['temporal']['rows']],
                         angular_l1=[r['atom_l1_vs_same_grid_steps128_K12288_per_emitted'] for r in result['refinement']['angular']['rows']]), indent=2))


if __name__ == '__main__':
    main()
