from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from optimize_core import module, solve


def raw(variant,source,remaining,directions,*,ids=None,children=None,horizon=5.,periodic=False):
    lib=module(variant).library()
    n=source.shape[0]
    levels=np.zeros_like(source,dtype=np.int32)
    counts=np.array([len(directions)],dtype=np.int32)
    offsets=np.array([0],dtype=np.int32)
    ids=np.empty(0,dtype=np.int32) if ids is None else np.ascontiguousarray(ids,dtype=np.int32)
    children=np.empty((0,8)) if children is None else np.ascontiguousarray(children,dtype=float)
    fields=[np.zeros_like(source) for _ in range(4)]
    angles=[np.zeros(len(directions)) for _ in range(3)];stats=np.zeros(16)
    status=lib.adaptive_evolve(n,source.ctypes.data,remaining.ctypes.data,len(ids),ids.ctypes.data,
        children.ctypes.data,1,levels.ctypes.data,offsets.ctypes.data,counts.ctypes.data,
        len(directions),directions.ctypes.data,1.,horizon,int(periodic),
        *[x.ctypes.data for x in fields],*[x.ctypes.data for x in angles],stats.ctypes.data)
    assert status==0
    fine=np.empty((source.size,8));lib.adaptive_copy_fine_absorbed(fine.ctypes.data)
    return fields+angles+[fine]


@pytest.mark.parametrize('variant',['local-dda-compact','face-table-compact'])
@pytest.mark.parametrize('offset',[0.,2**-45,2**-50,2**-52])
def test_internal_midplane_near_tie_keeps_positive_length_target(variant,offset):
    n=8;s=np.zeros((n,)*3);s[1,1,1]=1.
    remaining=np.zeros_like(s)
    target=(2,2,1);remaining[target]=1.
    remaining[3,3,1]=1e6
    children=np.zeros((1,8));children[0,4:6]=.5
    ids=np.array([np.ravel_multi_index(target,s.shape)],dtype=np.int32)
    direction=np.array([1.+offset,1.,0.]);direction/=np.linalg.norm(direction)
    signs=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    dirs=np.ascontiguousarray(signs*direction)
    # Trigger the face-table path without changing the source neighbourhood:
    # these tiny distant emitters add a negligible but explicitly counted rate.
    s[6,0:4,0:4]=1e-10
    expected=raw('original',s,remaining,dirs,ids=ids,children=children)
    actual=raw(variant,s,remaining,dirs,ids=ids,children=children)
    for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)
    assert actual[0][target] > .24 if offset else actual[0][target] < 1e-8


@pytest.mark.parametrize('variant',['local-dda-compact','face-table-compact'])
@pytest.mark.parametrize('horizon',[.5,np.nextafter(.5,0),np.nextafter(.5,1)])
def test_horizon_at_internal_and_open_faces(variant,horizon):
    n=4;s=np.zeros((n,)*3);s[3,1,1]=1
    s[0,:,:]=1e-10
    remaining=np.zeros_like(s)
    dirs=np.ascontiguousarray([[x,0.,0.] for x in (-1.,1.) for _ in range(4)])
    for periodic in (False,True):
        expected=raw('original',s,remaining,dirs,horizon=horizon,periodic=periodic)
        actual=raw(variant,s,remaining,dirs,horizon=horizon,periodic=periodic)
        for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)


def test_face_cache_many_sources_is_exact_and_bounded():
    rng=np.random.default_rng(20042)
    c=np.exp(rng.normal(0,.6,size=(8,)*3))
    s=np.exp(rng.normal(-1.,1.,size=c.shape));s*=.5*c.sum()/s.sum()
    old=.2*c
    options=dict(min_angles=96,max_angles=96,maximum_distance=4.,return_dense=True)
    expected=solve(s,c,old,variant='original',**options)
    actual=solve(s,c,old,variant='face-table-compact',**options)
    for key in ('absorbed','source_absorbed','source_escaped','source_retained','subcell_absorbed'):
        np.testing.assert_array_equal(actual[key],expected[key])
    assert 0<actual['audit']['face_table_bytes']<=64*1024**2


def test_face_cache_skips_single_source_precomputation():
    c=np.ones((4,)*3);s=np.zeros_like(c);s[2,2,2]=2
    actual=solve(s,c,variant='face-table-compact',min_angles=24,max_angles=24)
    assert actual['audit']['face_table_bytes']==0


@pytest.mark.parametrize('variant',['local-dda-compact','face-table-compact'])
def test_periodic_unwrapped_offsets_beyond_one_box(variant):
    n=6;s=np.zeros((n,)*3);s[5,2,2]=1
    s[1,0:4,0:4]=1e-10
    remaining=np.zeros_like(s);remaining[:,4,:]=1e3
    direction=np.array([1.,.2,.03]);direction/=np.linalg.norm(direction)
    signs=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    dirs=np.ascontiguousarray(signs*direction)
    expected=raw('original',s,remaining,dirs,horizon=16.,periodic=True)
    actual=raw(variant,s,remaining,dirs,horizon=16.,periodic=True)
    for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)


def test_face_cache_memory_cap_uses_exact_division_fallback():
    c=np.ones((4,)*3);s=np.zeros_like(c);s[:2,:,:]=2.
    options=dict(min_angles=96,max_angles=96,maximum_distance=20000.,
                 periodic=False,return_dense=True)
    expected=solve(s,c,variant='original',**options)
    actual=solve(s,c,variant='face-table-compact',**options)
    for key in ('absorbed','source_absorbed','source_escaped','source_retained','subcell_absorbed'):
        np.testing.assert_array_equal(actual[key],expected[key])
    assert actual['audit']['face_table_bytes']==0
