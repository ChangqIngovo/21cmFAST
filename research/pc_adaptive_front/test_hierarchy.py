"""Hierarchy skips are geometry optimizations, not an opacity approximation."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adaptive
from hierarchy_benchmark import module, solve


@pytest.mark.parametrize("n", [8, 12, 15])
@pytest.mark.parametrize("periodic", [False, True])
def test_random_partial_fields_match_parent_only(n, periodic):
    rng = np.random.default_rng(285+n)
    c = np.exp(rng.normal(0,.4,(n,)*3))
    old = c*rng.uniform(0,.6,c.shape)
    old[:n//2] = c[:n//2]
    s = np.exp(rng.normal(-3,2,c.shape))
    options = dict(min_angles=96,max_angles=96,maximum_distance=n,periodic=periodic,return_dense=True)
    a = adaptive.solve(s,c,old,**options)
    b = solve(s,c,old,**options)
    for key in ("absorbed","subcell_absorbed","source_absorbed","source_retained","source_escaped"):
        np.testing.assert_allclose(a[key],b[key],rtol=2e-10,atol=2e-12)


def test_large_initial_cavity_has_certified_jumps():
    c = np.ones((16,)*3)
    old = np.zeros_like(c);old[2:14,2:14,2:14]=1
    s = np.zeros_like(c);s[8,8,8]=1
    a = solve(s,c,old,enabled=False,min_angles=1536,max_angles=1536,return_dense=True)
    b = solve(s,c,old,enabled=True,min_angles=1536,max_angles=1536,return_dense=True)
    np.testing.assert_array_equal(a["subcell_absorbed"],b["subcell_absorbed"])
    assert b["audit"]["hierarchy_large_jumps"] > 0
    assert b["audit"]["queries"] < a["audit"]["queries"]


def test_dynamic_clearing_and_carried_state():
    c = np.ones((12,)*3)
    s = np.zeros_like(c);s[5,6,6]=180;s[7,6,6]=80
    states = {False:None, True:None}
    totals = {False:np.zeros_like(c), True:np.zeros_like(c)}
    for _ in range(4):
        for enabled in (False,True):
            out = solve(s/4,c,initial_state=states[enabled],enabled=enabled,min_angles=384,max_angles=384)
            states[enabled] = out["state"]
            totals[enabled] += out["absorbed"]
    np.testing.assert_allclose(totals[False],totals[True],atol=2e-12,rtol=0)


def test_near_corner_crossing_keeps_thin_opaque_cell():
    # Exercise custom directions directly through the C ABI. Near ties remain
    # distinct; the certification cannot round the cursor across the target.
    c = np.zeros((8,)*3);c[4,3,4]=1e6;c[4,4,4]=1e6
    s = np.zeros_like(c);s[1,1,1]=1
    n=8
    raw=[]
    for epsilon in (-1e-13,1e-13,-1e-15,1e-15):
        direction=np.array([1.,1.+epsilon,1.])
        direction/=np.linalg.norm(direction)
        # API bank is octant-contiguous; make one direction per octant.
        signs=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
        directions=np.ascontiguousarray(signs*direction)
        outputs=[]
        for enabled in (False,True):
            lib=module(enabled)._library
            level=np.zeros_like(c,dtype=np.int32);counts=np.array([8],dtype=np.int32);offset=np.array([0],dtype=np.int32)
            fields=[np.zeros_like(c) for _ in range(4)]
            angular=[np.zeros(8) for _ in range(3)];stats=np.zeros(16)
            status=lib.adaptive_evolve(n,s.ctypes.data,c.ctypes.data,0,0,0,1,
                level.ctypes.data,offset.ctypes.data,counts.ctypes.data,8,directions.ctypes.data,
                1.,10.,0,*[a.ctypes.data for a in fields],*[a.ctypes.data for a in angular],stats.ctypes.data)
            assert status==0
            outputs.append(fields[0])
        np.testing.assert_array_equal(*outputs)
        raw.append(outputs[1].sum())
    assert max(raw)>0
