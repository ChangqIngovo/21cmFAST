"""Research-only sharp voxel-front packets with exact characteristic directions.

Count closure, no deflection and first opaque voxel are enforced. Cell-centred
point emitters, optically thick neutral cells, infinite-speed snapshot packets,
and synchronous absorption rounds are explicit approximations. This is not a
finite-time radiative transfer/chemistry integrator or native 21cmFAST mode.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from characteristics import directions, trace, clearance, library


def allocate(source, capacity, *, angles=192, workers=4, max_rounds=256,
             maximum_distance=np.inf, local_absorption=True):
    q=np.asarray(source,dtype=float)
    c=np.asarray(capacity,dtype=float).copy()
    if q.shape!=c.shape or q.ndim!=3 or (q<0).any() or (c<0).any():
        raise ValueError('Nonnegative cubic source/capacity fields required')
    if not np.isfinite(q).all() or not np.isfinite(c).all():
        raise ValueError('Finite counts required')
    library()
    started=time.perf_counter()
    # This assumes centred point emitters in an unresolved sharp-front cell;
    # it is not an escape-fraction theorem for a uniformly emitting voxel.
    here=np.minimum(q,c) if local_absorption else np.zeros_like(c)
    c-=here
    q_remote=q-here
    ids=np.flatnonzero(q_remote>0)
    absorbed=here.copy()
    if not len(ids):
        return dict(absorbed=absorbed, remaining=0.0,escaped=0.0,audit=dict(
            photons=float(q.sum()),absorbed=float(absorbed.sum()),remaining=0.0,
            escaped=0.0,ledger_residual=0.0,iterations=0,active_sources=0,
            angles=angles,seconds=time.perf_counter()-started,converged=True))
    positions=np.array(np.unravel_index(ids,q.shape)).T+0.5
    dirs=directions(angles)
    pending=np.repeat(q_remote.ravel()[ids,None]/angles,angles,axis=1)
    emitted=pending.copy()
    escaped=np.zeros_like(pending)
    retained=np.zeros_like(pending)
    consumed=np.zeros_like(pending)
    tol=1e-13*max(float(q.sum()),1)
    queries=0
    for iteration in range(1,max_rounds+1):
        # Geometry changes monotonically within this count-allocation call.
        # Rebuilding is exact here. An older clearance would also stay safe,
        # but could be inefficient as a previously opaque front opens.
        clear=clearance(c<=0)
        rays=trace(c<=0,positions,dirs,workers=workers,
                   maximum_distance=maximum_distance,precomputed_clearance=clear)
        hits=rays['hits']
        queries+=int(rays['visits'].sum())
        outside=hits==-1
        limited=hits==-2
        escaped[outside]+=pending[outside]
        retained[limited]+=pending[limited]
        pending[outside|limited]=0
        good=(hits>=0)&(pending>0)
        offers=np.bincount(hits[good],weights=pending[good],minlength=q.size)
        take=np.minimum(offers,c.ravel())
        alpha=np.divide(take,offers,out=np.zeros_like(offers),where=offers>0)
        accepted=np.zeros_like(pending)
        accepted[good]=pending[good]*alpha[hits[good]]
        consumed+=accepted
        pending-=accepted
        # Use actual shared packet debits, not independently rounded celltake.
        deposition=np.bincount(hits[good],weights=accepted[good],minlength=q.size)
        absorbed+=deposition.reshape(q.shape)
        c-=deposition.reshape(q.shape)
        c=np.maximum(c,0)
        # Cells with saturated offers must become traversable despite ulps.
        saturated=offers>=c.ravel()+deposition
        c.ravel()[saturated & (offers>0)]=0
        if pending.sum()<=tol:
            break
    residual=pending+retained
    total=float(q.sum())
    audit=dict(photons=total,absorbed=float(absorbed.sum()),
        remaining=float(residual.sum()),escaped=float(escaped.sum()),
        ledger_residual=total-float(absorbed.sum()+residual.sum()+escaped.sum()),
        packet_ledger_max_error=float(np.max(np.abs(emitted-consumed-escaped-residual))),
        capacity_excess=float(np.maximum(absorbed-capacity,0).max()),
        iterations=iteration,converged=bool(pending.sum()<=tol),
        active_sources=len(ids),angles=angles,packets=int(pending.size),
        geometry_queries=queries,seconds=time.perf_counter()-started,
        packet_arrays_megabytes=5*pending.nbytes/2**20,
        topology='open box',time_model='synchronous infinite-speed snapshot rounds')
    return dict(absorbed=absorbed,remaining=float(residual.sum()),
                escaped=float(escaped.sum()),audit=audit)


def sphere_probe(n=32,radius=5,angles=192,workers=4):
    shape=(n,)*3
    source=np.zeros(shape)
    source[(n//2,)*3]=4*np.pi*radius**3/3
    capacity=np.ones(shape)
    out=allocate(source,capacity,angles=angles,workers=workers)
    r2=sum((a-n//2)**2 for a in np.indices(shape))
    # Exact lattice-shell count sphere for this total budget. This comparator
    # isolates voxel-front anisotropy; it is not an analytic cell-volume sphere.
    unique,counts=np.unique(r2,return_counts=True)
    cumulative=np.cumsum(counts)
    pivot=int(np.searchsorted(cumulative,source.sum()))
    ideal=np.zeros(shape)
    ideal[r2<unique[pivot]]=1
    ideal[r2==unique[pivot]]=(source.sum()-ideal.sum())/counts[pivot]
    return dict(**out['audit'],n=n,radius=radius,
        sphere_l1_per_photon=float(np.abs(out['absorbed']-ideal).sum()/source.sum()),
        radius_centroid=((np.indices(shape)*out['absorbed']).reshape(3,-1).sum(axis=1)
                         /out['absorbed'].sum()-n//2).tolist())


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path('packet_front_results.json'))
    a=p.parse_args()
    rows=[sphere_probe(angles=k) for k in (48,192,768,3072)]
    results=dict(scope='Toy static sharp voxel front, not full native evolution',
                 sphere=rows,preliminary_timings=True)
    a.output.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))
