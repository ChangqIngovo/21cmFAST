"""Independent adversarial comparisons against the full eight-subcell model."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent/'pc_refinement'))
import adaptive
import front_subcells


def fine_to_parent(fine):
    n=fine.shape[0]//2
    return fine.reshape(n,2,n,2,n,2).transpose(0,2,4,1,3,5).reshape(n**3,8)


def state_from_fine(fine,capacity):
    child=fine_to_parent(fine)
    old=child.sum(axis=1).reshape(capacity.shape)
    mixed=np.any(child!=child[:,0,None],axis=1)
    ids=np.flatnonzero(mixed).astype(np.int32)
    return dict(ionized=old,parent_ids=ids,child_ionized=child[ids],
                capacity_sha256=adaptive._capacity_hash(capacity))


def compare_fine_state(source,capacity,fine_initial,*,angles=24,periodic=True,horizon=None):
    if horizon is None:horizon=source.shape[0]/2
    state=state_from_fine(fine_initial,capacity)
    options=dict(min_angles=angles,max_angles=angles,periodic=periodic,maximum_distance=horizon)
    full=front_subcells.solve(source,capacity,initial_subcell_ionized=fine_initial,
                             parent_skip=True,**options)
    lazy=adaptive.solve(source,capacity,initial_state=state,return_dense=True,**options)
    q=float(source.sum(dtype=np.longdouble))
    delta=lazy['subcell_absorbed']-full['subcell_absorbed']
    scale=np.maximum(capacity,np.finfo(float).tiny)
    local=float(np.max(abs(lazy['absorbed']-full['absorbed'])/scale))
    err=float(abs(delta).sum()/max(q,np.finfo(float).tiny))
    for key in ('source_absorbed','source_escaped','source_retained'):
        np.testing.assert_allclose(lazy[key],full[key],rtol=2e-9,atol=2e-12)
    assert err<2e-10
    assert local<2e-9
    return dict(L1_fine_over_Q=err,maximum_parent_fraction_difference=local,
                promotions=lazy['audit']['promotions'],coarsenings=lazy['audit']['coarsenings'])


def test_near_transparent_mixed_state_survives_zero_source_step():
    c=np.ones((3,)*3);s=np.zeros_like(c)
    child=.125-np.arange(1,9)*1e-14
    old=np.zeros_like(c);old.ravel()[13]=child.sum()
    state=dict(ionized=old,parent_ids=np.array([13],dtype=np.int32),child_ionized=child[None,:],
               capacity_sha256=adaptive._capacity_hash(c))
    result=adaptive.solve(s,c,initial_state=state,min_angles=24,max_angles=24)
    assert result['absorbed'].sum()==0
    np.testing.assert_array_equal(result['state']['child_ionized'],child[None,:])


@pytest.mark.parametrize('periodic',[False,True])
def test_initially_clear_source_octants_keep_their_own_budgets(periodic):
    c=np.ones((8,)*3);s=np.zeros_like(c);s[4,4,4]=100
    fine=np.zeros((16,)*3)
    fine[8:10,8:10,8:10]=np.array([0,.125,.1,.125,.03,.125,0,.125]).reshape(2,2,2)
    compare_fine_state(s,c,fine,angles=96,periodic=periodic,horizon=6)


@pytest.mark.parametrize('seed',range(12))
def test_random_mixed_fronts_match_full_grid(seed):
    rng=np.random.default_rng(9001+seed)
    n=4+2*(seed%3)
    c=np.exp(rng.normal(0,1,size=(n,)*3))
    source=np.exp(rng.normal(-1,1.5,size=c.shape))
    source*=([.2,.8,2.][seed%3])*c.sum()/source.sum()
    finecap=c.repeat(2,0).repeat(2,1).repeat(2,2)/8
    fractions=rng.uniform(0,.95,size=finecap.shape)
    fractions[rng.random(finecap.shape)<.25]=1
    fine=finecap*fractions
    compare_fine_state(source,c,fine,angles=24 if seed%2 else 96,
                       periodic=bool(seed%2),horizon=[.3,n/2,2*n][seed%3])
