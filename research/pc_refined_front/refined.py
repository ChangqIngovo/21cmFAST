"""Sparse front refinement with unchanged original cell-centred point sources.

Gas cells are divided into 2, 4 or 8 bins per axis. Only the eight fine cells
adjacent to the source receive local emission; other children are reached by
geometric transport. Untouched gas is implicit, cleared gas uses a bit mask,
and partially illuminated gas uses sparse states. Angular sampling is an
independent control and is never multiplied by the spatial refinement factor.

This is the static, simultaneous-emission, infinite-light-speed opaque-cell
model. A returned state preserves spatial ionization, not in-flight photons.
"""
from __future__ import annotations
import ctypes as ct
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
from threading import RLock
import time
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'pc_refinement'))
from front_subcells import octant_directions
_library=None
_lock=RLock()
_INTMAX=np.iinfo(np.int32).max
_BITCOUNTS=np.array([i.bit_count() for i in range(256)],dtype=np.uint8)


def library():
    global _library
    with _lock:
        if _library is None:
            source=HERE/'refined.cpp'
            digest=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
            out=Path(tempfile.mkdtemp(prefix=f'pc_refined_{digest}_'))/'refined.so'
            subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(source),'-o',str(out)],check=True)
            _library=ct.CDLL(str(out))
            _library.refined_evolve.argtypes=[ct.c_int,ct.c_int,*([ct.c_void_p]*6),ct.c_int64,
                                             *([ct.c_void_p]*3),ct.c_double,ct.c_int,ct.c_int,*([ct.c_void_p]*6)]
            _library.refined_evolve.restype=ct.c_int
            _library.refined_set_contiguous_ray_budget.argtypes=[ct.c_uint64]
            _library.refined_set_contiguous_ray_budget.restype=None
            _library.refined_last_ray_storage.argtypes=[]
            _library.refined_last_ray_storage.restype=ct.c_int
            _library.refined_clear_template_cache.argtypes=[]
            _library.refined_clear_template_cache.restype=None
            _library.refined_state_count.argtypes=[]
            _library.refined_state_count.restype=ct.c_int64
            _library.refined_copy_state.argtypes=[ct.c_void_p,ct.c_void_p]
            _library.refined_copy_state.restype=None
    return _library


def clear_template_cache():
    """Clear normalized isolated-parent templates for cold-cost measurements."""
    with _lock:library().refined_clear_template_cache()


def _geometry(ids,n,f):
    fn=n*f
    z=ids%fn;y=(ids//fn)%fn;x=ids//(fn*fn)
    return ((x//f)*n+y//f)*n+z//f,((x%f)*f+y%f)*f+z%f


def _coarse_ionized(c,f,cleared,ids,values):
    # Count set bits without constructing a dense floating-point fine grid.
    counts=np.empty(c.size,dtype=np.int64)
    byte_masks=cleared.view(np.uint8)
    for start in range(0,c.size,32768):
        counts[start:start+32768]=_BITCOUNTS[byte_masks[start:start+32768]].sum(axis=1)
    out=counts.astype(float)*(c.ravel()/f**3)
    if len(ids):
        parent,_=_geometry(ids,c.shape[0],f)
        np.add.at(out,parent,values)
    return out.reshape(c.shape)


def _dense(c,f,cleared,ids,values):
    n=c.shape[0];sub=f**3
    bits=np.unpackbits(cleared.view(np.uint8),axis=1,bitorder='little')[:,:sub]
    parent_major=bits*(c.ravel()[:,None]/sub)
    dense=parent_major.reshape(n,n,n,f,f,f).transpose(0,3,1,4,2,5).reshape((n*f,)*3)
    dense.ravel()[ids]=values
    return dense


def solve(source,capacity,*,subcells_per_axis=4,angles=384,angle_counts=None,
          maximum_distance=None,periodic=True,initial_state=None,return_dense=False,
          use_local_templates=True,contiguous_ray_bytes=1<<30):
    """Return incremental absorbed counts, per-source ledgers and resolved state.

    `maximum_distance` is in original coarse-cell units. `angle_counts`, if
    supplied, is a cubic array of direction counts (positive multiples of 24)
    and overrides `angles`. It allows the existing coarse angular predictor to
    be reused unchanged at every gas resolution. `state` must be carried with
    the same atom capacities and refinement factor. All photon inputs to each
    call are newly emitted budgets; escaped/retained photons are not re-emitted.
    `use_local_templates` exactly caches neutral isolated-parent evolution until
    the first foreign illumination or outgoing ray. `contiguous_ray_bytes` caps
    reserved contiguous ray capacity; larger cases use blocks and zero forces
    blocks. This controls storage choice, not total memory or resident usage.
    """
    if not isinstance(use_local_templates,(bool,np.bool_)):
        raise ValueError('use_local_templates must be boolean')
    if (isinstance(contiguous_ray_bytes,(bool,np.bool_))
            or not isinstance(contiguous_ray_bytes,(int,np.integer))
            or not 0<=contiguous_ray_bytes<=np.iinfo(np.uint64).max):
        raise ValueError('contiguous_ray_bytes must be a nonnegative supported integer byte budget')
    contiguous_ray_bytes=int(contiguous_ray_bytes)
    lib=library()
    started=time.perf_counter()
    s=np.ascontiguousarray(source,dtype=float);c=np.ascontiguousarray(capacity,dtype=float)
    f=subcells_per_axis
    if (isinstance(f,(bool,np.bool_)) or not isinstance(f,(int,np.integer)) or f not in (2,4,8)):
        raise ValueError('subcells_per_axis must be 2, 4 or 8')
    f=int(f)
    if (s.ndim!=3 or len(set(s.shape))!=1 or not s.size or c.shape!=s.shape
            or not np.isfinite(s).all() or not np.isfinite(c).all() or (s<0).any() or (c<0).any()):
        raise ValueError('Matching finite nonnegative cubic source and atom counts required')
    n=s.shape[0];fn=n*f;sub=f**3;words=(sub+63)//64
    photons=s.sum(dtype=np.longdouble)
    if s.size>_INTMAX or fn>_INTMAX or fn**3>np.iinfo(np.int64).max or not np.isfinite(photons) or photons>np.finfo(float).max:
        raise ValueError('Grid or aggregate photons exceed supported range')
    if ((c>0)&(c/sub==0)).any():raise ValueError('Fine atom counts underflow; rescale units')
    horizon=float(n*np.sqrt(3.) if maximum_distance is None else maximum_distance)
    if not np.isfinite(horizon) or horizon<=0 or f*(horizon+n+2)>_INTMAX:
        raise ValueError('Maximum distance exceeds positive voxel range')
    if angle_counts is None:
        if isinstance(angles,(bool,np.bool_)) or not isinstance(angles,(int,np.integer)) or angles<24 or angles%24 or angles>_INTMAX:
            raise ValueError('angles must be a supported positive multiple of 24')
        angular=np.full(s.shape,angles,dtype=np.int32)
    else:
        raw=np.asarray(angle_counts)
        if (raw.shape!=s.shape or not (np.issubdtype(raw.dtype,np.integer) or np.issubdtype(raw.dtype,np.floating)) or not np.isfinite(raw).all()
                or (raw<24).any() or (raw>_INTMAX).any() or (raw%24).any()):
            raise ValueError('angle_counts must match the source grid with positive multiples of 24')
        angular=np.ascontiguousarray(raw,dtype=np.int32)
    active=s>0
    if active.any() and (s[active]/angular[active]<np.finfo(float).tiny).any():
        raise ValueError('Per-ray photons underflow; rescale units')
    counts,levels=np.unique(angular,return_inverse=True)
    counts=np.ascontiguousarray(counts,dtype=np.int32);levels=np.ascontiguousarray(levels.reshape(s.shape),dtype=np.int32)
    if counts.sum(dtype=np.int64)>_INTMAX:raise ValueError('Angular bank too large')
    offsets=np.ascontiguousarray(np.r_[0,np.cumsum(counts[:-1],dtype=np.int64)],dtype=np.int32)
    directions=np.ascontiguousarray(np.concatenate([octant_directions(int(k)) for k in counts]))
    fingerprint=hashlib.sha256(memoryview(c)).hexdigest()
    cleared=np.zeros((s.size,words),dtype=np.uint64)
    ids=np.empty(0,dtype=np.int64);old=np.empty(0,dtype=float)
    if initial_state is not None:
        state=initial_state
        if state.get('version')!=1 or state.get('subcells_per_axis')!=f or state.get('capacity_sha256')!=fingerprint:
            raise ValueError('Carried state requires the same static capacity grid and refinement factor')
        if not {'cleared_bits','partial_ids','partial_ionized','ionized'} <= state.keys():
            raise ValueError('Incomplete resolved front state')
        raw_bits=np.asarray(state['cleared_bits'])
        raw_ids=np.asarray(state['partial_ids'])
        old=np.ascontiguousarray(state['partial_ionized'],dtype=float)
        if raw_bits.shape!=cleared.shape or raw_bits.dtype!=np.uint64:
            raise ValueError('Invalid cleared-cell bit mask')
        cleared=raw_bits.copy()
        if sub<64 and (cleared>>np.uint64(sub)).any():raise ValueError('Invalid padding bits')
        if (raw_ids.ndim!=1 or not (np.issubdtype(raw_ids.dtype,np.integer) or np.issubdtype(raw_ids.dtype,np.floating)) or not np.isfinite(raw_ids).all()
                or (raw_ids!=np.floor(raw_ids)).any() or (raw_ids<0).any() or (raw_ids>=fn**3).any()
                or len(np.unique(raw_ids))!=len(raw_ids)):
            raise ValueError('Partial fine indices must be unique valid integers')
        ids=np.ascontiguousarray(raw_ids,dtype=np.int64)
        if old.shape!=ids.shape or not np.isfinite(old).all() or (old<0).any():
            raise ValueError('Invalid partial ionization counts')
        parent,child=_geometry(ids,n,f)
        if (old>c.ravel()[parent]/sub).any() or ((cleared[parent,child//64]>>((child%64).astype(np.uint64)))&np.uint64(1)).any():
            raise ValueError('Partial states exceed capacities or duplicate cleared cells')
        coarse_old=_coarse_ionized(c,f,cleared,ids,old)
        provided=np.asarray(state['ionized'],dtype=float)
        if provided.shape!=c.shape or not np.isfinite(provided).all() or not np.allclose(provided,coarse_old,rtol=2e-12,atol=0):
            raise ValueError('Coarse and resolved state budgets disagree')
    else:coarse_old=np.zeros_like(c)
    outputs=[np.zeros_like(c) for _ in range(4)]
    fine=np.zeros((fn,)*3) if return_dense else None
    stats=np.zeros(20)
    storage_mode="none"
    preparation_seconds=time.perf_counter()-started
    full_word=np.uint64((1<<sub)-1) if sub<64 else np.uint64(2**64-1)
    transparent=(c.ravel()==0)|np.all(cleared==full_word,axis=1)
    if periodic and transparent.all():
        outputs[3][...]=s
        final_ids=ids;final_values=old
    else:
        with _lock:
            supports_budget=hasattr(lib,'refined_set_contiguous_ray_budget')
            if supports_budget:lib.refined_set_contiguous_ray_budget(contiguous_ray_bytes)
            rc=lib.refined_evolve(n,f,s.ctypes.data,c.ctypes.data,levels.ctypes.data,offsets.ctypes.data,
                counts.ctypes.data,directions.ctypes.data,len(ids),ids.ctypes.data,old.ctypes.data,
                cleared.ctypes.data,horizon*f,int(periodic),int(use_local_templates),*[a.ctypes.data for a in outputs],
                fine.ctypes.data if fine is not None else None,stats.ctypes.data)
            if rc:raise RuntimeError(f'Refined sparse core returned status {rc}')
            storage_mode=('contiguous' if lib.refined_last_ray_storage() else 'blocks') if supports_budget else 'legacy-unreported'
            count=lib.refined_state_count()
            if not 0<=count<=fn**3:raise RuntimeError('Invalid native state size')
            final_ids=np.empty(count,dtype=np.int64);final_values=np.empty(count)
            lib.refined_copy_state(final_ids.ctypes.data,final_values.ctypes.data)
    order=np.argsort(final_ids);final_ids=final_ids[order];final_values=final_values[order]
    a,sa,escaped,retained=outputs
    final_coarse=_coarse_ionized(c,f,cleared,final_ids,final_values)
    residual=photons-sum(v.sum(dtype=np.longdouble) for v in (a,escaped,retained))
    src_error=float(np.max(abs(s[active]-sa[active]-escaped[active]-retained[active])/s[active])) if active.any() else 0.
    local_error=float(np.max(np.maximum(a-(c-coarse_old),0)/np.maximum(c,np.finfo(float).tiny)))
    if (not all(np.isfinite(x).all() for x in (*outputs,final_values,final_coarse)) or (a<0).any()
            or abs(residual)>2e-10*max(float(photons),np.finfo(float).tiny) or src_error>2e-10 or local_error>2e-10):
        raise RuntimeError('Refined photon ledger failed')
    state=dict(version=1,subcells_per_axis=f,capacity_sha256=fingerprint,ionized=final_coarse,
               cleared_bits=cleared,partial_ids=final_ids,partial_ionized=final_values)
    audit=dict(model='sparse fixed-source refined opaque front',subcells_per_axis=f,photons=float(photons),
               absorbed=float(a.sum(dtype=np.longdouble)),escaped=float(escaped.sum(dtype=np.longdouble)),
               retained=float(retained.sum(dtype=np.longdouble)),ledger_residual=float(residual),
               source_relative_error=src_error,maximum_local_fraction_violation=local_error,
               maximum_distance=horizon,source_positions='original coarse centres',
               continuum_accuracy_certified=False,preparation_seconds=preparation_seconds)
    for name,v in zip(('events','queries','assignments','rays_materialized','ray_allocated_bytes',
                      'node_creations','peak_live_nodes','node_allocated_bytes','heap_allocated_bytes',
                      'hash_estimated_bytes','parent_skips','source_octant_launches'),stats):audit[name]=int(v)
    audit['use_local_templates']=bool(use_local_templates)
    for name,i in (('template_parents',12),('template_materialisations',13),('implicit_final_parents',14),('template_cache_misses',15),('implicit_source_directions',17)):
        audit[name]=int(stats[i])
    audit['template_build_seconds']=float(stats[16])
    audit['ray_storage']=storage_mode
    audit['contiguous_ray_budget_bytes']=contiguous_ray_bytes
    audit['ray_capacity_bytes_are_rss']=False
    audit['transport_ray_bytes']=int(stats[18])
    audit['source_rate_bytes']=int(stats[19])
    audit['clear_mask_bytes']=cleared.nbytes
    audit['ledger_workspace_bytes']=5*s.size*np.dtype(np.longdouble).itemsize
    audit['state_bytes']=sum(v.nbytes for v in state.values() if isinstance(v,np.ndarray))
    audit['hash_memory_is_estimate']=True
    audit['seconds']=time.perf_counter()-started
    result=dict(absorbed=a,source_absorbed=sa,source_escaped=escaped,source_retained=retained,
                unabsorbed=float(escaped.sum(dtype=np.longdouble)+retained.sum(dtype=np.longdouble)),
                state=state,angle_counts=angular,audit=audit)
    if return_dense:
        result['subcell_absorbed']=fine
        result['subcell_ionized']=_dense(c,f,cleared,final_ids,final_values)
    return result
