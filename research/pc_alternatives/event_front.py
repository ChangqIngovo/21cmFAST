"""Sparse exhaustion events for continuous source rates on a sharp voxel front.

This prototype preserves every source and angular rate, unlike batch allocation.
It uses straight characteristics and a lazy priority queue, not time-stepped
whole-grid streaming. Its physical closure is infinite light speed, centred
point emitters, optically thick neutral voxels, no recombinations, and constant
emissivity within a step. Native 21cmFAST source/thermal integration is pending.
"""
from __future__ import annotations
import argparse
import ctypes as ct
import json
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
from characteristics import clearance, directions

_library=None


def library():
    global _library
    if _library is None:
        here=Path(__file__).resolve().parent
        out=Path(tempfile.mkdtemp(prefix='pc_event_front_'))/'reference.so'
        subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',
                        str(here/'event_front.cpp'),'-o',str(out)],check=True)
        _library=ct.CDLL(str(out))
        _library.evolve_events.argtypes=[ct.c_int,ct.c_void_p,ct.c_void_p,
            ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_double,ct.c_double,ct.c_int,
            ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_void_p,
            ct.c_void_p,ct.c_void_p]
        _library.evolve_events.restype=ct.c_int
        for name in ('event_front_last_max_heap_nodes','event_front_last_heap_updates',
                     'event_front_last_ray_allocated_bytes'):
            if hasattr(_library,name):
                getattr(_library,name).argtypes=[]
                getattr(_library,name).restype=ct.c_int64
    return _library


def solve(source, capacity, *, angles=192, duration=1., periodic=False,
          maximum_distance=None, ray_directions=None):
    s=np.ascontiguousarray(source,dtype=np.float64)
    c=np.ascontiguousarray(capacity,dtype=np.float64)
    if s.shape!=c.shape or s.ndim!=3 or len(set(s.shape))!=1:
        raise ValueError('Cubic equally shaped arrays required')
    if (s<0).any() or (c<0).any() or not np.isfinite(s).all() or not np.isfinite(c).all():
        raise ValueError('Finite nonnegative rates/capacities required')
    n=s.shape[0]
    max_distance=float(n*np.sqrt(3) if maximum_distance is None else maximum_distance)
    if not np.isfinite(duration) or duration<=0:
        raise ValueError('Finite positive duration required')
    if not np.isfinite(max_distance) or max_distance<=0:
        raise ValueError('Finite positive maximum distance required')
    if n<1 or n**3>np.iinfo(np.int32).max:
        raise ValueError('Grid size exceeds this prototype index representation')
    if max_distance>np.iinfo(np.int32).max-n-2:
        raise ValueError('Maximum distance exceeds the unwrapped voxel index range')
    if ray_directions is None and (not isinstance(angles,(int,np.integer)) or angles<1):
        raise ValueError('Positive integer angle count required')
    dirs=np.ascontiguousarray(directions(angles) if ray_directions is None
                             else ray_directions,dtype=np.float64)
    if dirs.ndim!=2 or dirs.shape[1]!=3 or len(dirs)==0 or not np.isfinite(dirs).all():
        raise ValueError('Finite nonempty direction array with shape (K,3) required')
    angles=len(dirs)
    if not np.allclose(np.linalg.norm(dirs,axis=1),1,atol=1e-12,rtol=0):
        raise ValueError('Unit angular directions required')
    lib=library()
    started=time.perf_counter()
    dist=clearance(c<=0)
    absorbed=np.zeros_like(c);src_abs=np.zeros_like(c);src_escape=np.zeros_like(c)
    src_left=np.zeros_like(c)
    ang_abs=np.zeros(angles);ang_escape=np.zeros(angles);ang_left=np.zeros(angles)
    stats=np.zeros(6)
    args=[n,s.ctypes.data,c.ctypes.data,dist.ctypes.data,angles,dirs.ctypes.data,
          duration,max_distance,int(periodic),absorbed.ctypes.data,src_abs.ctypes.data,
          src_escape.ctypes.data,src_left.ctypes.data,ang_abs.ctypes.data,
          ang_escape.ctypes.data,ang_left.ctypes.data,stats.ctypes.data]
    status=lib.evolve_events(*args)
    if status: raise RuntimeError(f'Event integrator status {status}')
    total=float(s.sum()*duration)
    audit=dict(photons=total,absorbed=float(absorbed.sum()),
        escaped=float(src_escape.sum()),retained=float(src_left.sum()),
        ledger_residual=total-float(absorbed.sum()+src_escape.sum()+src_left.sum()),
        per_source_ledger_max=float(np.max(np.abs(s*duration-src_abs-src_escape-src_left))),
        per_angle_ledger_max=float(np.max(np.abs(total/angles-ang_abs-ang_escape-ang_left))),
        independent_absorption_ledger=float(absorbed.sum()-src_abs.sum()),
        capacity_excess=float(np.maximum(absorbed-c,0).max()),
        negative_absorption=float(np.maximum(-absorbed,0).max()),
        events=int(stats[0]),rays_materialized=int(stats[1]),queries=int(stats[2]),
        assignments=int(stats[3]),stale_queue_entries=int(stats[4]),
        ray_storage_megabytes=float(stats[1]*stats[5]/2**20),
        angles=angles,n=n,workers=1,seconds=time.perf_counter()-started,
        periodic=periodic,maximum_distance=max_distance,
        model='constant rates, infinite-c sharp opaque voxel front, no recombinations')
    for name,label in (
        ('event_front_last_max_heap_nodes','peak_heap_nodes'),
        ('event_front_last_heap_updates','heap_updates'),
        ('event_front_last_ray_allocated_bytes','ray_allocated_bytes')):
        if hasattr(lib,name):
            audit[label]=int(getattr(lib,name)())
    return dict(absorbed=absorbed,source_absorbed=src_abs,
                source_escaped=src_escape,source_retained=src_left,audit=audit)


def probes():
    rows=[]
    for k in [48,192,768,3072]:
        n=32;radius=5
        s=np.zeros((n,)*3);s[(n//2,)*3]=4*np.pi*radius**3/3
        c=np.ones_like(s)
        out=solve(s,c,angles=k)
        r2=sum((x-n//2)**2 for x in np.indices(s.shape))
        unique,count=np.unique(r2,return_counts=True)
        i=np.searchsorted(np.cumsum(count),s.sum())
        ideal=(r2<unique[i]).astype(float)
        ideal[r2==unique[i]]=(s.sum()-ideal.sum())/count[i]
        row=dict(**out['audit'],sphere_l1_per_photon=float(np.abs(out['absorbed']-ideal).sum()/s.sum()))
        # Reinitializing fixed straight rays each half step is valid in the
        # infinite-c model: only their first opaque cell can absorb.
        half=solve(s,c,angles=k,duration=.5)
        half2=solve(s,np.maximum(c-half['absorbed'],0),angles=k,duration=.5)
        row['time_partition_l1']=float(np.abs(out['absorbed']-half['absorbed']-half2['absorbed']).sum())
        row['time_partition_max']=float(np.max(np.abs(out['absorbed']-half['absorbed']-half2['absorbed'])))
        rows.append(row)
    # Off-axis screen. Initially empty surroundings, very large wall capacity.
    n=40;xyz=np.indices((n,)*3).astype(float)+.5
    wall=(np.abs(xyz[0]+xyz[1]-40)<1.2)&(np.abs(xyz[0]-xyz[1])<10)
    c=np.where(wall,1e6,0.)
    s=np.zeros_like(c);s[8,8,20]=1000
    screen=solve(s,c,ray_directions=np.array([[1,1,0.]])/np.sqrt(2))
    return dict(scope='Toy fixed-rate sharp-front validation; no native integration',
                sphere=rows,diagonal_screen=screen['audit'],preliminary_timings=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,
                                             default=Path('event_front_results.json'))
    a=p.parse_args();r=probes();a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))
