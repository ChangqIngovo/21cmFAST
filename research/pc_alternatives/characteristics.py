"""Frozen-front characteristic reference with certified transparent jumps.

This resolves the geometric first absorber; it is NOT a complete ionization
solver. Fixed-angle quadrature, source subcell placement and voxel front shape
remain approximations. No source packet is merged at a different position.
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
from scipy.ndimage import distance_transform_cdt

_library = None


def directions(count):
    """Equal-area Fibonacci sphere. Each direction has weight 1/count."""
    j = np.arange(count)
    mu = 1-2*(j+0.5)/count
    phase = j*(np.pi*(3-np.sqrt(5)))
    return np.column_stack((mu,np.sqrt(1-mu*mu)*np.cos(phase),
                            np.sqrt(1-mu*mu)*np.sin(phase)))


def clearance(transparent):
    # Opaque padding treats the open box edge as an obstacle for the purpose
    # of safe jump distance only; trace_first_hits still reports open escape.
    mask = np.asarray(transparent,dtype=bool)
    return np.ascontiguousarray(distance_transform_cdt(
        np.pad(mask,1,constant_values=False),metric='chessboard')[1:-1,1:-1,1:-1],
        dtype=np.int32)


def library():
    global _library
    if _library is None:
        here = Path(__file__).resolve().parent
        # Keep platform-specific binary out of the repository.
        out = Path(tempfile.mkdtemp(prefix='pc_characteristics_'))/'reference.so'
        subprocess.run(['gcc','-O3','-std=c99','-shared','-fPIC','-fopenmp',
                        str(here/'characteristics.c'),'-lm','-o',str(out)],check=True)
        _library = ct.CDLL(str(out))
        f = _library.trace_first_hits
        f.argtypes = [ct.c_int,ct.c_void_p,ct.c_int,ct.c_void_p,
                      ct.c_int,ct.c_void_p,ct.c_double,ct.c_int,ct.c_int,
                      ct.c_void_p,ct.c_void_p,ct.c_void_p]
        f.restype=ct.c_int
    return _library


def trace(mask, sources, ray_directions, *, maximum_distance=np.inf,
          skip=True, workers=4, precomputed_clearance=None):
    mask = np.asarray(mask,dtype=bool)
    if mask.ndim!=3 or len(set(mask.shape))!=1:
        raise ValueError('Cubic transparent boolean geometry required')
    s = np.ascontiguousarray(sources,dtype=np.float64)
    d = np.ascontiguousarray(ray_directions,dtype=np.float64)
    if s.ndim!=2 or s.shape[1]!=3 or d.ndim!=2 or d.shape[1]!=3:
        raise ValueError('Sources/directions must have shape (count,3)')
    if not np.isfinite(s).all() or not np.isfinite(d).all():
        raise ValueError('Finite positions and directions required')
    if not np.allclose(np.linalg.norm(d,axis=1),1,rtol=0,atol=1e-12):
        raise ValueError('Unit directions required')
    lib = library()
    start = time.perf_counter()
    dist = clearance(mask) if precomputed_clearance is None else precomputed_clearance
    dist = np.ascontiguousarray(dist,dtype=np.int32)
    prep = time.perf_counter()-start
    shape=(len(s),len(d))
    hits=np.empty(shape,dtype=np.int64)
    lengths=np.empty(shape)
    visits=np.empty(shape,dtype=np.int32)
    start=time.perf_counter()
    status=lib.trace_first_hits(mask.shape[0],dist.ctypes.data,len(s),s.ctypes.data,
                               len(d),d.ctypes.data,maximum_distance,int(skip),workers,
                               hits.ctypes.data,lengths.ctypes.data,visits.ctypes.data)
    if status or np.any(hits==-3):
        raise RuntimeError('Characteristic traversal failed')
    return dict(hits=hits,distance=lengths,visits=visits,audit=dict(
        rays=hits.size,workers=workers,skip=skip,
        clearance_seconds=prep,trace_seconds=time.perf_counter()-start,
        mean_geometry_queries=float(visits.mean()),
        hit_fraction=float(np.mean(hits>=0)),escape_fraction=float(np.mean(hits==-1)),
        range_limited_fraction=float(np.mean(hits==-2))))


def probes(workers=4):
    n=64;R=12;d=18
    xyz=np.indices((n,)*3).astype(float)+0.5
    a=np.array([23.5,32.5,32.5]);b=a+[d,0,0]
    ra=np.sqrt(sum((xyz[j]-a[j])**2 for j in range(3)))
    rb=np.sqrt(sum((xyz[j]-b[j])**2 for j in range(3)))
    mask=(ra<=R)|(rb<=R)
    clear=clearance(mask)
    data=[]
    for k in [48,192,768,3072,12288]:
        dirs=directions(k)
        result=trace(mask,[a],dirs,workers=workers,precomputed_clearance=clear)
        no_skip=trace(mask,[a],dirs,workers=workers,skip=False,
                      precomputed_clearance=clear)
        hit_positions=np.array(np.unravel_index(result['hits'][0],mask.shape)).T+0.5
        v=hit_positions-a
        # The input cone is the clean analytic statistic; classification using
        # voxel-centre directions has a separate spatial discretization error.
        analytic_cone=dirs[:,0]>=0.75
        centre_cone=v[:,0]>=0.75*np.linalg.norm(v,axis=1)
        data.append(dict(k=k,**result['audit'],
            isotropic_overlap_cone=float(analytic_cone.mean()),
            hit_centre_cone_fraction=float(centre_cone.mean()),
            analytic_expected=0.125,
            skip_hits_equal_ordinary_dda=bool(np.array_equal(result['hits'],no_skip['hits'])),
            no_skip_queries=float(no_skip['visits'].mean()),
            cone_hit_ray_distance_mean=float(result['distance'][0,analytic_cone].mean()),
            note='Voxel union differs from exact spheres; input cone weight is not a voxel shape accuracy test',
            no_skip_trace_seconds=no_skip['audit']['trace_seconds']))
    # Finite opaque diagonal sheet with a collimated beam known to cross it.
    n2=40
    grid=np.indices((n2,)*3).astype(float)+0.5
    wall=(np.abs(grid[0]+grid[1]-40)<1.2)&(np.abs(grid[0]-grid[1])<10)
    transparent=~wall
    beam=np.array([[1,1,0]],dtype=float)/np.sqrt(2)
    starts=np.array([[8.5,8.5,20.5],[9.5,8.5,20.5],[8.5,9.5,20.5]])
    shot=trace(transparent,starts,beam,workers=workers)
    one=trace(mask,[a],directions(3072),workers=1,precomputed_clearance=clear)
    four=trace(mask,[a],directions(3072),workers=4,precomputed_clearance=clear)
    return dict(scope='Frozen transparent front; exact per-ray geometry, finite angular quadrature',
        overlap=data,opaque_diagonal_sheet=dict(**shot['audit'],
                 all_rays_hit_sheet=bool(np.all(shot['hits']>=0)),
                 behind_sheet_leak_fraction=float(np.mean(shot['hits']<0))),
        threads_1_4_identical=bool(np.array_equal(one['hits'],four['hits'])),
        limitation='No evolving capacity competition, recombination, finite-time source history or native integration',
        preliminary_timings=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path('characteristics_results.json'))
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    results=probes(a.workers)
    a.output.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))
