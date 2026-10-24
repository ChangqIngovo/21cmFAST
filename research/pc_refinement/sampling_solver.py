"""Probe-driven angular allocation with unchanged shared-event transport.

Only the quadrature predictor changes. A small set of directions measures the
initial first-absorber distance. In each probe direction the predicted front
radius is (distance**3 + isolated_radius**3)**(1/3), exact for a uniform spherical
cavity and otherwise a declared geometry heuristic. Thin angular openings and
collective clearing during the interval are not certified by these probes.
"""
from __future__ import annotations

import ctypes as ct
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'pc_reconsidered'))
import hybrid

_probe=None
_INT32_MAX=np.iinfo(np.int32).max


def _validate_probe_options(probe_angles,statistic,effective_density):
    if (not isinstance(probe_angles,(int,np.integer)) or isinstance(probe_angles,bool)
            or not 1<=probe_angles<=_INT32_MAX):
        raise ValueError('probe_angles must be a positive supported integer')
    if statistic not in ('rms','p90','max'):
        raise ValueError('statistic must be rms, p90, or max')
    if effective_density not in ('total','remaining'):
        raise ValueError('effective_density must be total or remaining')


def _validate_angular_options(min_angles,max_angles,rays_per_surface_cell):
    if any(not isinstance(k,(int,np.integer)) or isinstance(k,bool) or not 1<=k<=_INT32_MAX
           for k in (min_angles,max_angles)):
        raise ValueError('Angular limits must be positive supported integers')
    if min_angles>max_angles or not np.isfinite(rays_per_surface_cell) or rays_per_surface_cell<=0:
        raise ValueError('Ordered angular limits and positive finite surface sampling required')


def _validate_supplied_counts(angle_count_grid,shape):
    supplied=np.asarray(angle_count_grid)
    if (supplied.shape!=shape or not np.isfinite(supplied).all()
            or (supplied<1).any() or (supplied!=np.floor(supplied)).any()
            or (supplied>_INT32_MAX).any()):
        raise ValueError('angle_count_grid must contain positive supported integer counts on the source grid')
    return supplied


def probe_library():
    global _probe
    if _probe is None:
        source=HERE/'sampling_probe.cpp'
        output=Path(tempfile.mkdtemp(prefix='pc_sampling_'))/'probe.so'
        subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(source),'-o',str(output)],check=True)
        _probe=ct.CDLL(str(output))
        _probe.probe_distances.argtypes=[ct.c_int,ct.c_void_p,ct.c_int,ct.c_void_p,
            ct.c_int,ct.c_void_p,ct.c_double,ct.c_int,ct.c_void_p]
        _probe.probe_distances.restype=ct.c_int
    return _probe


def probe_radius(source,capacity,remaining,*,probe_angles=48,statistic='rms',
                 periodic=True,maximum_distance=None,effective_density='total'):
    """Predict final radius after local source exhaustion; not an error bound."""
    _validate_probe_options(probe_angles,statistic,effective_density)
    n=source.shape[0]
    horizon=n*np.sqrt(3.) if maximum_distance is None else float(maximum_distance)
    density_grid=capacity if effective_density=='total' else remaining
    positive=density_grid[density_grid>0]
    rho=float(np.quantile(positive,.1)) if positive.size else 1.
    isolated3=3*source/(4*np.pi*rho)
    radius=np.cbrt(isolated3)
    transparent=remaining<=0
    adjacent=transparent.copy()
    for axis in range(3):
        for side in (-1,1):
            shifted=np.roll(transparent,side,axis)
            if not periodic:
                index=[slice(None)]*3
                index[axis]=0 if side==1 else -1
                shifted[tuple(index)]=False
            adjacent |= shifted
    active=(source>0)&adjacent
    indices=np.ascontiguousarray(np.flatnonzero(active),dtype=np.int32)
    if len(indices):
        if len(indices)*probe_angles>_INT32_MAX:
            raise ValueError('Probe array exceeds supported native index range')
        directions=np.ascontiguousarray(hybrid.directions(probe_angles))
        distances=np.empty((len(indices),probe_angles))
        rem=np.ascontiguousarray(remaining,dtype=np.float64)
        code=probe_library().probe_distances(n,rem.ctypes.data,len(indices),indices.ctypes.data,
            probe_angles,directions.ctypes.data,horizon,int(periodic),distances.ctypes.data)
        if code:raise RuntimeError(f'Geometry probe failed {code}')
        front=np.cbrt(distances**3+isolated3.ravel()[indices,None])
        front=np.minimum(front,horizon)
        if statistic=='rms': values=np.sqrt(np.mean(front**2,axis=1))
        elif statistic=='max': values=front.max(axis=1)
        elif statistic=='p90': values=np.quantile(front,.9,axis=1)
        else:raise ValueError('statistic must be rms, p90, or max')
        radius.ravel()[indices]=values
    return np.minimum(radius,horizon),{'probe_sources':len(indices),'probe_angles':probe_angles,
        'probe_statistic':statistic,'probe_rays':len(indices)*probe_angles,
        'effective_density':effective_density,'density_reference':rho}


def count_levels(radius,source,*,min_angles=24,max_angles=12288,rays_per_surface_cell=4.):
    _validate_angular_options(min_angles,max_angles,rays_per_surface_cell)
    wanted=np.maximum(min_angles,np.ceil(4*np.pi*radius**2*rays_per_surface_cell))
    counts=[min_angles]
    while counts[-1]<max_angles:counts.append(min(2*counts[-1],max_angles))
    counts=np.asarray(counts,dtype=np.int32)
    levels=np.searchsorted(counts,wanted,side='left').clip(max=len(counts)-1)
    levels[source==0]=0
    return np.asarray(levels,dtype=np.int32),counts,wanted


def solve(source,capacity,initial_ionized=None,*,probe_angles=48,statistic='rms',
          min_angles=24,max_angles=12288,rays_per_surface_cell=4.,periodic=True,
          maximum_distance=None,angle_count_grid=None,effective_density='total',backend=None,
          dominant_source_fraction=None):
    """Run the existing core using the new predictor and the same angular sets."""
    core=hybrid if backend is None else backend
    lib=core.library()
    probe_library()
    started=time.perf_counter()
    s=np.ascontiguousarray(source,dtype=np.float64)
    c=np.ascontiguousarray(capacity,dtype=np.float64)
    old=np.zeros_like(c) if initial_ionized is None else np.asarray(initial_ionized,dtype=float)
    if s.shape!=c.shape or old.shape!=s.shape or s.ndim!=3 or len(set(s.shape))!=1 or s.size==0:
        raise ValueError('Matching cubic source, capacity and initial counts required')
    if not all(np.isfinite(a).all() for a in (s,c,old)) or (s<0).any() or (old<0).any() or (old>c).any():
        raise ValueError('Nonnegative finite source and bounded ionized atoms required')
    if not np.isfinite(s.sum(dtype=np.longdouble)) or float(s.sum(dtype=np.longdouble))==np.inf:
        raise ValueError('Aggregate photon budget overflows; rescale count units')
    rem=np.ascontiguousarray(c-old)
    n=s.shape[0]
    horizon=float(n*np.sqrt(3.) if maximum_distance is None else maximum_distance)
    if not np.isfinite(horizon) or horizon<=0 or horizon>_INT32_MAX-n-2:
        raise ValueError('Maximum distance outside supported positive voxel range')
    if n**3>_INT32_MAX:raise ValueError('Grid too large for native indices')
    _validate_probe_options(probe_angles,statistic,effective_density)
    _validate_angular_options(min_angles,max_angles,rays_per_surface_cell)
    supplied=None if angle_count_grid is None else _validate_supplied_counts(angle_count_grid,s.shape)
    if dominant_source_fraction is not None:
        if (not np.isfinite(dominant_source_fraction) or not 0<dominant_source_fraction<=1
                or angle_count_grid is not None):
            raise ValueError('dominant_source_fraction must lie in (0,1], with predicted angle counts')
    if periodic and not np.any(rem>0):
        result=core.solve(s,c,old,maximum_distance=horizon)
        result['audit'].update({'probe_sources':0,'probe_angles':probe_angles,'probe_rays':0,
                               'probe_statistic':statistic,'effective_density':effective_density,
                               'predictor_seconds':0.,'sampling_backend':core.__name__,
                               'per_angle_relative_ledger_max':0.,
                               'dominant_source_fraction':dominant_source_fraction,'dominant_sources':0})
        return result
    if angle_count_grid is None:
        radius,probes=probe_radius(s,c,rem,probe_angles=probe_angles,statistic=statistic,
                                  periodic=periodic,maximum_distance=horizon,effective_density=effective_density)
        levels,counts,wanted=count_levels(radius,s,min_angles=min_angles,max_angles=max_angles,
                                          rays_per_surface_cell=rays_per_surface_cell)
        probes['dominant_source_fraction']=dominant_source_fraction
        probes['dominant_sources']=0
        if dominant_source_fraction is not None:
            total_budget=s.sum(dtype=np.longdouble)
            # Compare in the source array's precision. Promoting a binary64
            # fraction before multiplying can exclude exact decimal ties such
            # as one hundred equal sources at a 1% threshold.
            threshold=np.nextafter(float(dominant_source_fraction)*float(total_budget),-np.inf)
            dominant=(s>0)&(s>=threshold)
            probes['dominant_sources']=int(dominant.sum())
            probes['maximum_source_photon_fraction']=float(s.max()/total_budget) if total_budget else 0.
            if dominant.any():
                # Only dominant sources enter the potentially expensive connected
                # component query. Guard the original surface-factor-4 angular
                # plan using its historical total-capacity density convention.
                # The main probe predictor still uses the selected convention.
                protected=np.where(dominant,s,0.)
                guard_levels,guard_counts,_,_=hybrid.estimate_angles(protected,c,rem,
                    min_angles=min_angles,max_angles=max_angles,rays_per_surface_cell=4.,
                    periodic=periodic,maximum_distance=horizon)
                desired=counts[levels]
                desired[dominant]=np.maximum(desired[dominant],guard_counts[guard_levels[dominant]])
                levels=np.searchsorted(counts,desired).astype(np.int32)
    else:
        counts=np.unique(supplied[s>0]).astype(np.int32) if (s>0).any() else np.array([24],dtype=np.int32)
        levels=np.searchsorted(counts,supplied).clip(max=len(counts)-1).astype(np.int32)
        radius=np.zeros_like(s)
        probes={'probe_sources':0,'probe_angles':0,'probe_rays':0,'probe_statistic':'explicit counts'}
    used=np.unique(levels[s>0]) if (s>0).any() else np.array([0])
    remap=np.zeros(len(counts),dtype=np.int32);remap[used]=np.arange(len(used))
    levels=np.ascontiguousarray(remap[levels],dtype=np.int32)
    counts=np.ascontiguousarray(counts[used],dtype=np.int32)
    if counts.sum(dtype=np.int64)>_INT32_MAX:
        raise ValueError('Concatenated quadratures exceed supported native index range')
    active=s>0
    if active.any() and np.any(s[active]/counts[levels[active]]<np.finfo(float).tiny):
        raise ValueError('Per-ray budget is subnormal; rescale units')
    offsets=np.ascontiguousarray(np.r_[0,np.cumsum(counts[:-1])],dtype=np.int32)
    directions=np.ascontiguousarray(np.concatenate([hybrid.directions(int(k)) for k in counts]))
    clearance=hybrid.clearance(rem<=0)
    arrays=[np.zeros_like(s) for _ in range(4)]
    angular=[np.zeros(len(directions)) for _ in range(3)]
    stats=np.zeros(6)
    predicted_seconds=time.perf_counter()-started
    rc=lib.evolve_adaptive_events(n,s.ctypes.data,rem.ctypes.data,clearance.ctypes.data,
        len(counts),levels.ctypes.data,offsets.ctypes.data,counts.ctypes.data,len(directions),
        directions.ctypes.data,1.,horizon,int(periodic),*[a.ctypes.data for a in arrays],
        *[a.ctypes.data for a in angular],stats.ctypes.data)
    if rc:raise RuntimeError(f'Event core returned {rc}')
    absorbed,src_abs,escaped,retained=arrays
    if not all(np.isfinite(a).all() for a in [*arrays,*angular]):
        raise RuntimeError('Nonfinite output from angular event core')
    total=float(s.sum(dtype=np.longdouble))
    ledger=total-float(sum(a.sum(dtype=np.longdouble) for a in (absorbed,escaped,retained)))
    per_source=s-src_abs-escaped-retained
    expected_angles=np.concatenate([
        np.full(int(k),np.sum(s[levels==lev],dtype=np.longdouble)/k)
        for lev,k in enumerate(counts)])
    angle_error=expected_angles-sum(angular)
    per_angle_relative=float(np.max(np.abs(angle_error)/np.maximum(expected_angles,np.finfo(float).tiny)))
    relative=float(np.max(np.abs(per_source[active])/s[active])) if active.any() else 0.
    local_tolerance=5e-12*rem
    if (abs(ledger)>5e-10*total or relative>5e-10 or per_angle_relative>5e-10
            or np.any(absorbed>rem+local_tolerance) or np.any(absorbed < -local_tolerance)):
        raise RuntimeError('Photon/capacity accounting failed')
    audit={**probes,'model':'probe-derived source-specific angle counts; unchanged event core',
           'photons':total,'absorbed':float(absorbed.sum()),'escaped':float(escaped.sum()),
           'retained':float(retained.sum()),'ledger_residual':ledger,'per_source_relative_ledger_max':relative,
           'per_angle_ledger_max':float(np.max(np.abs(angle_error))),
           'per_angle_relative_ledger_max':per_angle_relative,
           'capacity_excess':float(np.maximum(absorbed-rem,0).max()),
           'angles_mean':float(counts[levels[active]].mean()) if active.any() else 0.,
           'angles_max':int(counts[levels[active]].max()) if active.any() else 0,
           'angles_sum':int(counts[levels[active]].sum()),'rays_materialized':int(stats[1]),
           'ray_allocated_bytes':int(lib.hybrid_last_ray_allocated_bytes()),
           'predictor_seconds':predicted_seconds,'seconds':time.perf_counter()-started,
           'angular_accuracy_certified':False,'sampling_backend':core.__name__}
    return {'absorbed':absorbed,'unabsorbed':audit['escaped']+audit['retained'],
            'source_absorbed':src_abs,'source_escaped':escaped,'source_retained':retained,
            'angle_counts':counts[levels],'estimated_radius':radius,'audit':audit}
