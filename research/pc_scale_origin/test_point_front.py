from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import point_front
from front_subcells import solve as existing


def centres(n):
    return np.indices((n,)*3).reshape(3,-1).T+.5


@pytest.mark.parametrize('factor',[1,2])
@pytest.mark.parametrize('periodic',[True,False])
def test_matches_existing_same_positions_same_quadrature(factor,periodic):
    n=6
    rng=np.random.default_rng(817)
    c=rng.uniform(.3,1.4,(n,)*3)
    s=np.zeros_like(c)
    s[1,2,3]=43.;s[4,3,2]=24.;s[0,0,0]=18.
    ref=existing(s,c,subcells=factor,min_angles=96,max_angles=96,
                 periodic=periodic,maximum_distance=2*n)
    active=s.ravel()>0
    gas=c.repeat(factor,0).repeat(factor,1).repeat(factor,2)/factor**3
    out=point_front.solve(centres(n)[active]*factor,s.ravel()[active],gas,
                          angles=96,periodic=periodic,maximum_distance=2*n*factor)
    actual=out['absorbed'].reshape(n,factor,n,factor,n,factor).sum(axis=(1,3,5))
    np.testing.assert_allclose(actual,ref['absorbed'],atol=1e-12,rtol=1e-11)
    for key in ('source_absorbed','source_retained','source_escaped'):
        np.testing.assert_allclose(out[key],ref[key].ravel()[active],atol=1e-12,rtol=1e-11)


def test_vertex_sources_emit_into_eight_geometric_octants():
    c=np.ones((4,)*3)
    out=point_front.solve([[2.,2.,2.]],[.8],c,angles=96)
    expected=np.zeros_like(c);expected[1:3,1:3,1:3]=.1
    np.testing.assert_allclose(out['absorbed'],expected,atol=1e-16)


def test_open_vertex_boundary_and_periodic_wrapping():
    c=np.ones((4,)*3)
    opened=point_front.solve([[0.,0.,0.]],[.8],c,angles=96,periodic=False)
    assert opened['source_escaped'][0]==pytest.approx(.7)
    assert opened['absorbed'][0,0,0]==pytest.approx(.1)
    wrapped=point_front.solve([[0.,0.,0.]],[.8],c,angles=96,periodic=True)
    assert np.count_nonzero(wrapped['absorbed'])==8
    np.testing.assert_allclose(wrapped['absorbed'][wrapped['absorbed']>0],.1)


def test_translation_reflection_and_source_permutation():
    n=8
    rng=np.random.default_rng(81)
    c=rng.uniform(.2,1.4,(n,)*3)
    p=np.array([[1.2,3.7,5.4],[4.8,2.1,1.8]])
    q=np.array([90.,70.])
    base=point_front.solve(p,q,c,angles=96,maximum_distance=2*n)
    trans=point_front.solve((p+2)%n,q,np.roll(c,2,axis=(0,1,2)),angles=96,maximum_distance=2*n)
    np.testing.assert_allclose(trans['absorbed'],np.roll(base['absorbed'],2,axis=(0,1,2)),atol=2e-12)
    reflected=point_front.solve(n-p,q,c[::-1,::-1,::-1],angles=96,maximum_distance=2*n)
    np.testing.assert_allclose(reflected['absorbed'],base['absorbed'][::-1,::-1,::-1],atol=2e-12)
    perm=point_front.solve(p[::-1],q[::-1],c,angles=96,maximum_distance=2*n)
    np.testing.assert_allclose(perm['absorbed'],base['absorbed'],atol=2e-12)


def test_uniform_partial_ionization_and_source_merging_at_same_point():
    c=np.ones((4,)*3)
    a=point_front.solve(centres(4),np.full(4**3,.3),c)
    np.testing.assert_allclose(a['absorbed'],.3,atol=1e-16)
    one=point_front.solve([[1.5,1.5,1.5]],[25.],c,maximum_distance=8)
    split=point_front.solve([[1.5,1.5,1.5],[1.5,1.5,1.5]],[10.,15.],c,maximum_distance=8)
    np.testing.assert_allclose(one['absorbed'],split['absorbed'],atol=1e-13)


def test_photons_retained_in_periodic_vacuum():
    c=np.zeros((4,)*3)
    out=point_front.solve([[1.5,1.5,1.5]],[10.],c,maximum_distance=4)
    assert out['source_retained'][0]==pytest.approx(10.)
    assert out['absorbed'].sum()==0


def test_open_escape_at_exact_horizon_is_not_counted_as_retained():
    c=np.zeros((2,)*3)
    pos=np.array([1.5,1.,1.],dtype=np.longdouble)
    directions=point_front.octant_directions(24).astype(np.longdouble)
    crossing=np.where(directions>0,(2-pos)/directions,-pos/directions).min(axis=1)
    assert (crossing==1).any()  # mu=0.5 gives exactly representable x-face times.
    escaped=np.count_nonzero(crossing<=1)
    result=point_front.solve([pos],[24.],c,angles=24,maximum_distance=1.,periodic=False)
    assert result['source_escaped'][0]==pytest.approx(escaped)
    assert result['source_retained'][0]==pytest.approx(24-escaped)


@pytest.mark.parametrize('p,q,c', [([[float('nan'),1,1]],[1],np.ones((4,)*3)),
    ([[4,1,1]],[1],np.ones((4,)*3)),([[1,1,1]],[-1],np.ones((4,)*3)),
    ([[1,1,1]],[1],np.full((4,)*3,-1.))])
def test_bad_inputs_rejected(p,q,c):
    with pytest.raises(ValueError):point_front.solve(p,q,c)
