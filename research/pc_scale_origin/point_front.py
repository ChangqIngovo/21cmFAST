"""Independent source-position / gas-resolution reference for static fronts.

Inputs are explicit point-source coordinates and photon budgets, independent of
the cubic gas grid. Coordinates and maximum_distance are in gas-cell units.
The solver uses the existing equal-weight octant directions, opaque gas cells,
constant simultaneous source rates and infinite propagation speed. It eagerly
allocates all rays and is a diagnostic reference, not a production replacement.
"""
from __future__ import annotations

import ctypes as ct
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'pc_refinement'))
from front_subcells import octant_directions

_library=None


def library():
    global _library
    if _library is None:
        source=HERE/'point_front.cpp'
        digest=hashlib.sha256(source.read_bytes()).hexdigest()[:12]
        output=Path(tempfile.mkdtemp(prefix=f'pc_points_{digest}_'))/'point_front.so'
        subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(source),'-o',str(output)],check=True)
        _library=ct.CDLL(str(output))
        _library.point_evolve.argtypes=[ct.c_int,ct.c_int,ct.c_int,*([ct.c_void_p]*4),ct.c_double,ct.c_int,*([ct.c_void_p]*5)]
        _library.point_evolve.restype=ct.c_int
    return _library


def solve(positions,photons,capacity,*,angles=96,maximum_distance=None,periodic=True):
    lib=library()
    start=time.perf_counter()
    p,q,c=(np.ascontiguousarray(a,dtype=float) for a in (positions,photons,capacity))
    if (c.ndim!=3 or len(set(c.shape))!=1 or c.shape[0]<2 or q.ndim!=1
            or p.shape!=(len(q),3) or any(not np.isfinite(a).all() for a in (p,q,c))
            or (c<0).any() or (q<0).any()):
        raise ValueError('Finite source positions, nonnegative source counts and cubic gas counts required')
    n=c.shape[0]
    if (p<0).any() or (p>=n).any():
        raise ValueError('Source coordinates must lie in [0, gas_grid_size)')
    if type(angles) is not int or angles<24 or angles%24:
        raise ValueError('angles must be a positive multiple of24')
    limit=np.iinfo(np.int32).max
    distance=n*np.sqrt(3.) if maximum_distance is None else float(maximum_distance)
    if (not np.isfinite(distance) or distance<=0 or distance>limit-n-2
            or c.size>limit or len(q)*angles>limit):
        raise ValueError('Distance, gas grid or ray count exceeds supported range')
    emitted=q.sum(dtype=np.longdouble)
    if not np.isfinite(emitted) or emitted>np.finfo(float).max:
        raise ValueError('Total photon count overflows')
    directions=octant_directions(angles)
    absorbed=np.zeros_like(c)
    source_outputs=[np.zeros_like(q) for _ in range(3)]
    stats=np.zeros(6)
    rc=lib.point_evolve(n,len(q),angles,p.ctypes.data,q.ctypes.data,c.ctypes.data,directions.ctypes.data,
                        distance,int(periodic),absorbed.ctypes.data,*[a.ctypes.data for a in source_outputs],stats.ctypes.data)
    if rc:raise RuntimeError(f'Explicit-position reference returned code{rc}')
    sa,escaped,retained=source_outputs
    residual=emitted-sum(a.sum(dtype=np.longdouble) for a in (absorbed,escaped,retained))
    source_delta=q-sa-escaped-retained
    active=q>0
    source_error=float(np.max(abs(source_delta[active])/q[active])) if active.any() else 0.
    scale=max(float(emitted),np.finfo(float).tiny)
    local_error=float(np.max(np.maximum(absorbed-c,0)/np.maximum(c,np.finfo(float).tiny)))
    if (abs(residual)/scale>2e-10 or source_error>2e-10 or local_error>2e-10
            or (absorbed<0).any() or any(not np.isfinite(a).all() for a in (absorbed,*source_outputs))):
        raise RuntimeError('Explicit-position photon accounting failed')
    audit=dict(scope='Dense explicit-source static opaque-cell reference',photons=float(emitted),
               absorbed=float(absorbed.sum(dtype=np.longdouble)),retained=float(retained.sum(dtype=np.longdouble)),
               escaped=float(escaped.sum(dtype=np.longdouble)),ledger_residual=float(residual),
               relative_ledger_residual=float(abs(residual)/scale),source_relative_error=source_error,
               maximum_local_fraction_violation=local_error,angles=angles,maximum_distance=distance,
               seconds=time.perf_counter()-start)
    audit.update({name:int(v) for name,v in zip(('events','queries','assignments','rays','ray_size','ray_bytes'),stats)})
    return dict(absorbed=absorbed,source_absorbed=sa,source_escaped=escaped,source_retained=retained,audit=audit)
