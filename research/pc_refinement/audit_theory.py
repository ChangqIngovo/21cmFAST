"""Test order and weak-source quadrature bounds in the sharp-front event model.

The theorem concerns the finite-direction, static, opaque-cell, constant-rate
model only. It does not bound errors caused by voxel geometry, chemistry or a
changed physical source budget. This script invokes the existing event C++ core
with explicit source-dependent quadratures, bypassing radius prediction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'pc_reconsidered'))
import hybrid
from characteristics import clearance, directions


def explicit_solve(source,capacity,initial,weak,*,strong_angles=96,weak_angles=96,
                   periodic=True,horizon=None,weak_rotation=False):
    s=np.ascontiguousarray(source,dtype=float)
    c=np.ascontiguousarray(capacity,dtype=float)
    initial=np.asarray(initial,dtype=float)
    rem=np.ascontiguousarray(c-initial)
    n=s.shape[0]
    if s.shape!=c.shape or s.shape!=initial.shape or s.shape!=np.shape(weak):
        raise ValueError('Matching source/capacity/initial/mask arrays required')
    if not all(np.isfinite(x).all() for x in (s,c,initial)) or (s<0).any() or (rem<0).any():
        raise ValueError('Physical finite counts required')
    if horizon is None: horizon=n/2
    lib=hybrid.library()
    started=time.perf_counter()
    counts=np.ascontiguousarray([strong_angles,weak_angles],dtype=np.int32)
    levels=np.ascontiguousarray(weak,dtype=np.int32)
    offset=np.ascontiguousarray([0,strong_angles],dtype=np.int32)
    dweak=directions(weak_angles)
    if weak_rotation:
        # Proper orthogonal rotation, kept fixed throughout this solve.
        angle=.713
        matrix=np.array([[np.cos(angle),-np.sin(angle),0],
                         [np.sin(angle),np.cos(angle),0],[0,0,1]])
        dweak=dweak@matrix.T
    dirs=np.ascontiguousarray(np.vstack((directions(strong_angles),dweak)))
    clear=clearance(rem<=0)
    out=[np.zeros_like(s) for _ in range(4)]
    byangle=[np.zeros(len(dirs)) for _ in range(3)]
    stats=np.zeros(6)
    status=lib.evolve_adaptive_events(n,s.ctypes.data,rem.ctypes.data,clear.ctypes.data,
        2,levels.ctypes.data,offset.ctypes.data,counts.ctypes.data,len(dirs),dirs.ctypes.data,
        1.,float(horizon),int(periodic),*[x.ctypes.data for x in out],
        *[x.ctypes.data for x in byangle],stats.ctypes.data)
    if status: raise RuntimeError(status)
    absorbed,src_abs,escaped,retained=out
    return dict(absorbed=absorbed,source_absorbed=src_abs,escaped=escaped,retained=retained,
                seconds=time.perf_counter()-started,rays=int(stats[1]),
                ray_bytes=int(lib.hybrid_last_ray_allocated_bytes()))


def audit_triplet(source,capacity,initial,weak,*,strong_angles=96,weak_angles=6,
                  periodic=True,horizon=None,keep_arrays=False):
    strong=np.where(weak,0.,source)
    base=explicit_solve(strong,capacity,initial,np.zeros_like(weak),
                        strong_angles=strong_angles,weak_angles=strong_angles,
                        periodic=periodic,horizon=horizon)
    fine=explicit_solve(source,capacity,initial,weak,strong_angles=strong_angles,
                        weak_angles=strong_angles,periodic=periodic,horizon=horizon)
    cheap=explicit_solve(source,capacity,initial,weak,strong_angles=strong_angles,
                         weak_angles=weak_angles,weak_rotation=True,
                         periodic=periodic,horizon=horizon)
    q=float(source.sum(dtype=np.longdouble))
    qw=float(source[weak].sum(dtype=np.longdouble))
    rows=[]
    for result in (fine,cheap):
        delta=result['absorbed']-base['absorbed']
        strong_abs=result['source_absorbed'][~weak]-base['source_absorbed'][~weak]
        rows.append(dict(minimum_absorbed_increment=float(delta.min()),
                          total_absorbed_increment=float(delta.sum()),
                          maximum_strong_absorption_increment=float(strong_abs.max(initial=0)),
                          weak_absorbed=float(result['source_absorbed'][weak].sum()),
                          count_residual=float(q-result['absorbed'].sum()-result['escaped'].sum()-result['retained'].sum())))
    error=float(np.sum(abs(fine['absorbed']-cheap['absorbed']),dtype=np.longdouble))
    atol=5e-11*max(q,1.)
    report=dict(photons=q,weak_photons=qw,weak_fraction=qw/q if q else 0.,
                source_count=int(np.count_nonzero(source)),weak_source_count=int(np.count_nonzero(weak&(source>0))),
                strong_angles=strong_angles,weak_angles=weak_angles,periodic=periodic,horizon=horizon,
                audits=rows,L1=error,L1_over_Q=error/q if q else 0.,
                bound_2Qweak=2*qw,bound_fraction=2*qw/q if q else 0.,
                fine_seconds=fine['seconds'],cheap_seconds=cheap['seconds'],
                fine_rays=fine['rays'],cheap_rays=cheap['rays'],
                fine_ray_bytes=fine['ray_bytes'],cheap_ray_bytes=cheap['ray_bytes'],
                passed=bool(all(r['minimum_absorbed_increment']>=-atol and
                                r['total_absorbed_increment']<=qw+atol and
                                r['maximum_strong_absorption_increment']<=atol
                                for r in rows) and error<=2*qw+atol))
    if keep_arrays: return report,(base,fine,cheap)
    return report


def tail_mask(source,fraction):
    flat=np.ravel(source)
    order=np.argsort(flat,kind='stable')
    total=float(flat.sum(dtype=np.longdouble))
    cumulative=np.cumsum(flat[order],dtype=np.longdouble)
    selected=order[cumulative<=fraction*total]
    result=np.zeros(flat.size,dtype=bool)
    result[selected]=True
    return result.reshape(source.shape)


def random_tests(count=48):
    rng=np.random.default_rng(182411)
    rows=[]
    for k in range(count):
        n=6 if k%3==0 else 8
        cap=np.exp(rng.normal(0,1,size=(n,)*3))
        source=np.exp(rng.normal(-1,2,size=cap.shape))
        source*=([.1,.5,1.5][k%3])*cap.sum()/source.sum()
        old=cap*np.where(rng.random(cap.shape)<.25,1.,rng.uniform(0,.6,cap.shape))
        weak=rng.random(cap.shape)<(.1 if k%2 else .6)
        # Include tiny weak budgets where a hidden response amplification is
        # easier to distinguish from a loose bound.
        if k%4==0: source[weak]*=1e-4
        rows.append(audit_triplet(source,cap,old,weak,strong_angles=48,
                                  weak_angles=[1,6,12][k%3],periodic=bool(k%2),
                                  horizon=[1.2,n/2,n*1.4][k%3]))
        if not rows[-1]['passed']: raise AssertionError((k,rows[-1]))
    return rows


def native_test(path):
    with np.load(path) as stored:
        source=stored['source'].astype(float)
        cap=stored['capacity'].astype(float)
    old=np.zeros_like(cap)
    zero=np.zeros_like(cap,dtype=bool)
    # First-half state creates a real shared cavity but uses a declared fixed
    # angle budget; this is not the costly cavity-radius predictor.
    half=explicit_solve(source/2,cap,old,zero,strong_angles=96,weak_angles=96,
                        periodic=True,horizon=cap.shape[0]/2)
    rows=[]
    for initial,name in ((old,'neutral'),(half['absorbed'],'after_half_dose')):
        current=source if name=='neutral' else source/2
        for fraction in (.001,.005,.01,.05):
            mask=tail_mask(current,fraction)
            result=audit_triplet(current,cap,initial,mask,strong_angles=96,weak_angles=6,
                                  periodic=True,horizon=cap.shape[0]/2)
            result['state']=name
            result['requested_weak_fraction']=fraction
            rows.append(result)
            if not result['passed']: raise AssertionError(result)
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native',type=Path)
    parser.add_argument('--output',type=Path,default=HERE/'audit_theory_results.json')
    args=parser.parse_args()
    out=dict(scope='Order and bounded quadrature perturbation in fixed opaque-cell transport',
             random=random_tests(),native=native_test(args.native) if args.native else [])
    args.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(random_cases=len(out['random']),passed=all(x['passed'] for x in out['random']),native=out['native']),indent=2))
