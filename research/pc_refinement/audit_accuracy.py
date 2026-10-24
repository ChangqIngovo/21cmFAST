"""Independent accuracy gates for low-cost photon-conservation upgrades.

The analytic angular probes below do not invoke the transport implementation.
Their purpose is to distinguish count closure, angular integration, voxel-front
geometry, and the weighting implicit in a reported map metric.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'pc_reconsidered'
ALT = HERE.parent / 'pc_alternatives'
for directory in (OLD, ALT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
from characteristics import directions, trace


def angular_metrics(d):
    d = np.asarray(d, dtype=float)
    if d.ndim != 2 or d.shape[1] != 3 or not np.allclose(np.linalg.norm(d, axis=1), 1):
        raise ValueError('Unit direction vectors required')
    mean = d.mean(axis=0)
    second = np.einsum('ni,nj->ij', d, d) / len(d)
    return dict(dipole_norm=float(np.linalg.norm(mean)),
                quadrupole_frobenius=float(np.linalg.norm(second-np.eye(3)/3)),
                mean_direction=mean.tolist(), second_moment=second.tolist())


def analytic_cap_audit(d, *, threshold=3.5/12, normals=None):
    """Any plane at distance d, with range L, covers (1-d/L)/2 of the sphere.

    Rotating the plane normal should not change its exact fraction. This test
    directly integrates directions, avoiding voxelized oblique-plane ambiguity.
    """
    if normals is None:
        normals = np.vstack((np.eye(3), -np.eye(3), directions(127)))
    expected = (1-threshold)/2
    measured = (np.asarray(normals) @ np.asarray(d).T >= threshold).mean(axis=1)
    return dict(expected=expected, samples=len(normals),
                axis_fractions=measured[:6].tolist(),
                minimum=float(measured.min()), maximum=float(measured.max()),
                mean_absolute_fraction_error=float(abs(measured-expected).mean()),
                maximum_absolute_fraction_error=float(abs(measured-expected).max()))


def missed_cone(d, *, candidates=32768, safety=.75):
    """Construct a finite solid-angle aperture missed by every probe direction."""
    trial = directions(candidates)
    cosine = (trial @ np.asarray(d).T).max(axis=1)
    centre = trial[int(np.argmin(cosine))]
    nearest = float(np.arccos(np.clip(cosine.min(), -1, 1)))
    angle = safety*nearest
    fraction = float((1-np.cos(angle))/2)
    return dict(normal=centre, angle=angle, fraction=fraction,
                probe_hits=int(np.count_nonzero(np.asarray(d) @ centre >= np.cos(angle))),
                nearest_probe_angle=nearest)


def probe_alias_case(count=24, *, n=48, inner=5., outer=17.):
    """Transparent core plus an unsampled cone; all quantities have length units.

    The continuum cone is an exact angular oracle. The voxel trace is a separate
    check: near cells can make the aperture smaller/larger and are not silently
    treated as the same geometry.
    """
    probe = directions(count)
    cone = missed_cone(probe)
    centre = np.full(3, n//2+.5)
    coordinates = np.moveaxis(np.indices((n,)*3), 0, -1) + .5 - centre
    radius = np.linalg.norm(coordinates, axis=-1)
    cosine = np.divide(coordinates@cone['normal'], radius,
                       out=np.ones_like(radius), where=radius>0)
    mask = (radius <= inner) | ((radius <= outer) & (cosine >= np.cos(cone['angle'])))
    observed = trace(mask, [centre], probe, workers=1, maximum_distance=n/2)
    refined = trace(mask, [centre], directions(16384), workers=1, maximum_distance=n/2)
    f = cone['fraction']
    continuum_mean_r2 = (1-f)*inner**2 + f*outer**2
    result = dict(probe_count=count,
                  continuum_cone_fraction=f,
                  continuum_probe_cone_hits=cone['probe_hits'],
                  aperture_half_angle_radians=cone['angle'],
                  nearest_probe_angle_radians=cone['nearest_probe_angle'],
                  continuum_probe_mean_r2=inner**2,
                  continuum_true_mean_r2=continuum_mean_r2,
                  continuum_mean_r2_underestimate=float(1-inner**2/continuum_mean_r2),
                  voxel_probe_max_distance=float(observed['distance'].max()),
                  voxel_refined_max_distance=float(refined['distance'].max()),
                  voxel_probe_mean_r2=float((observed['distance']**2).mean()),
                  voxel_refined_mean_r2=float((refined['distance']**2).mean()),
                  voxel_probe_far_fraction=float((observed['distance'] > .8*outer).mean()),
                  voxel_refined_far_fraction=float((refined['distance'] > .8*outer).mean()))
    return result


def rotation_case(*, angles=None, n=24):
    """Cyclic coordinate rotation is an exact symmetry of this voxel problem."""
    from hybrid import solve, library
    c=np.ones((n,)*3)
    xyz=np.indices(c.shape)
    # The fields rotate together. Density anisotropy makes this a nontrivial
    # covariance test, not just a spherically symmetric output comparison.
    c *= 1 + .3*np.cos(2*np.pi*xyz[0]/n) + .2*np.sin(2*np.pi*xyz[2]/n)
    s=np.zeros_like(c)
    s[n//2-3,n//2,n//2]=160
    s[n//2+3,n//2+1,n//2-2]=95
    kwargs=dict(periodic=True,maximum_distance=n/2)
    if angles is not None:
        kwargs.update(min_angles=angles,max_angles=angles)
    library()
    start=time.perf_counter()
    a=solve(s,c,**kwargs)
    rotated=solve(s.transpose(1,2,0),c.transpose(1,2,0),**kwargs)
    b=rotated['absorbed'].transpose(2,0,1)
    return dict(angles=angles or 'adaptive',
                L1_over_emitted=float(abs(a['absorbed']-b).sum()/s.sum()),
                maximum_fraction_difference=float(np.max(abs(a['absorbed']-b)/c)),
                seconds_both=time.perf_counter()-start,
                angular_count_max=int(a['angle_counts'].max()),
                emitted=float(s.sum()),
                per_source_ledger=max(a['audit']['per_source_relative_ledger_max'],
                                      rotated['audit']['per_source_relative_ledger_max']))


def weighting_counterexample():
    """A small photon-count L1 does not control volume or void-cell xHII error."""
    capacity=np.ones((4,)*3)
    capacity[0,0,0]=1e-6
    reference=.2*capacity
    candidate=reference.copy()
    moved=.7e-6
    candidate[0,0,0]+=moved
    candidate[1,1,1]-=moved
    q=reference.sum()
    difference=candidate-reference
    return dict(photon_count_L1_over_Q=float(abs(difference).sum()/q),
                volume_mean_absolute_fraction_error=float(np.mean(abs(difference)/capacity)),
                maximum_fraction_error=float(np.max(abs(difference)/capacity)),
                global_count_difference=float(difference.sum()),
                capacity_bounds_preserved=bool(((candidate>=0)&(candidate<=capacity)).all()))


def run():
    rows=[]
    for k in (24,48,96,192,768,3072):
        d=directions(k)
        rows.append(dict(angles=k,**angular_metrics(d),cap=analytic_cap_audit(d)))
    return dict(scope='Independent diagnostic gates, not a new transport solver',
                angular_quadrature=rows,
                missed_probe_cones=[probe_alias_case(12),probe_alias_case(24)],
                rotation_covariance=[rotation_case(angles=k) for k in (24,96,384,None)],
                weighting_counterexample=weighting_counterexample(),
                acceptance=[
                    'Compare map error to a continuum oracle when one exists, not only a same-core high-angle solution.',
                    'Report count-weighted and volume/fraction errors separately; global count closure is not a local accuracy metric.',
                    'Repeat angular-aperture tests over many normals; a plane aligned with Fibonacci polar axis is not representative.',
                    'Check cubic rotation covariance of both fields; translations alone do not expose orientation bias.',
                    'Require reference angular change below claimed improvements; preserve source origins under subcell refinement.',
                    'Treat sparse first-hit radius probes as estimates: missed solid-angle features can bias surface-area predictions.',
                    'Benchmark setup, compilation separately, repeated kernel medians, materialized ray memory, and full coeval cost.',
                ])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE/'audit_accuracy_results.json')
    args=parser.parse_args()
    report=run()
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


def strict_ledger(absorbed, unabsorbed, source, capacity, initial=None, *, rtol=1e-9):
    """Audit local capacity in fractions, independently of total box luminosity.

    The existing comparison gate scales its per-cell allowance by global Q.
    Here a local excess is normalized by that cell's total atom capacity; the
    limit therefore does not loosen as the box or a distant source grows.
    """
    a=np.asarray(absorbed,dtype=float)
    s=np.asarray(source,dtype=float)
    c=np.asarray(capacity,dtype=float)
    old=np.zeros_like(c) if initial is None else np.asarray(initial,dtype=float)
    if not (a.shape==s.shape==c.shape==old.shape) or any(not np.isfinite(x).all() for x in (a,s,c,old)):
        raise ValueError('Matching finite grids required')
    if (s<0).any() or (c<0).any() or (old<0).any() or (old>c).any() or not np.isfinite(unabsorbed):
        raise ValueError('Physical source/atom inputs required')
    excess=np.maximum(a-(c-old),0)
    negative=np.maximum(-a,0)
    scale=np.maximum(c,np.finfo(float).tiny)
    local=float(np.max(np.maximum(excess,negative)/scale))
    q=float(s.sum(dtype=np.longdouble))
    residual=float(np.longdouble(q)-a.sum(dtype=np.longdouble)-np.longdouble(unabsorbed))
    global_relative=abs(residual)/max(q,np.finfo(float).tiny)
    passed=global_relative<=rtol and local<=rtol and unabsorbed>=-rtol*q
    return dict(passed=bool(passed),relative_count_residual=global_relative,
                maximum_local_fraction_violation=local)
