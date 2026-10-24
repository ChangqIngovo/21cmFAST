from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import sampling_solver as new
import hybrid


def test_neutral_predictor_preserves_original_angles_and_map():
    c=np.ones((14,)*3);s=np.zeros_like(c);s[5,6,7]=123;s[8,6,7]=42
    before=hybrid.solve(s,c)
    after=new.solve(s,c)
    np.testing.assert_array_equal(before['angle_counts'],after['angle_counts'])
    np.testing.assert_array_equal(before['absorbed'],after['absorbed'])


def test_near_transparent_origin_has_continuous_counts():
    c=np.ones((20,)*3);xyz=np.indices(c.shape)
    bubble=sum((a-10)**2 for a in xyz)<6**2
    s=np.zeros_like(c);s[10,10,10]=.1
    counts=[]
    for residual in [0.,1e-8,.05]:
        old=bubble.astype(float);old[10,10,10]-=residual
        result=new.solve(s,c,old)
        counts.append(result['angle_counts'][10,10,10])
        assert abs(result['absorbed'].sum()-.1)<1e-11
    assert counts==[counts[0]]*3


def test_periodic_probe_translation():
    c=np.ones((16,)*3);s=np.zeros_like(c);s[3,3,3]=.1
    xyz=np.indices(c.shape);old=(sum((a-3)**2 for a in xyz)<3**2).astype(float)
    a=new.solve(s,c,old)
    shift=(12,11,10)
    b=new.solve(np.roll(s,shift,(0,1,2)),c,np.roll(old,shift,(0,1,2)))
    np.testing.assert_array_equal(np.roll(a['absorbed'],shift,(0,1,2)),b['absorbed'])
    np.testing.assert_array_equal(np.roll(a['estimated_radius'],shift,(0,1,2)),b['estimated_radius'])


def test_probe_cube_sum_growth_not_radius_sum():
    # Directional volume addition is exact in spherical uniform media. This test
    # verifies the implemented relation with measured, voxelized wall distances.
    c=np.ones((20,)*3);s=np.zeros_like(c);s[10,10,10]=4*np.pi*2**3/3
    xyz=np.indices(c.shape);old=(sum((a-10)**2 for a in xyz)<6**2).astype(float)
    r,_=new.probe_radius(s,c,c-old,statistic='max')
    # Max voxelized radius lies within sqrt(3)/2 of the continuum sphere.
    assert r[10,10,10] < np.cbrt((6+np.sqrt(3)/2)**3+2**3)
    assert r[10,10,10] < 6+2


def test_saturated_periodic_box_keeps_photons():
    c=np.ones((6,)*3);s=np.full_like(c,.5)
    result=new.solve(s,c,c)
    assert result['unabsorbed']==s.sum()
    assert not result['absorbed'].any()


def test_uniform_partial_state_uses_remaining_density():
    c=np.ones((20,)*3);s=np.zeros_like(c);s[10,10,10]=.01*4*np.pi*4**3/3
    state=new.solve(s,c,np.full_like(c,.99),effective_density='remaining')
    equivalent=new.solve(s,.01*c,effective_density='remaining')
    np.testing.assert_array_equal(state['angle_counts'],equivalent['angle_counts'])
    np.testing.assert_allclose(state['absorbed'],equivalent['absorbed'],atol=1e-13,rtol=0)


def test_dominant_source_keeps_original_cavity_resolution():
    c=np.ones((24,)*3);xyz=np.indices(c.shape);old=(sum((a-12)**2 for a in xyz)<6**2).astype(float)
    s=np.zeros_like(c);s[12,12,12]=.01
    original=hybrid.solve(s,c,old,maximum_distance=12)
    guarded=new.solve(s,c,old,maximum_distance=12,rays_per_surface_cell=16,
                      effective_density='remaining',dominant_source_fraction=.01)
    assert guarded['angle_counts'][12,12,12]>=original['angle_counts'][12,12,12]
    assert guarded['audit']['dominant_sources']==1


@pytest.mark.parametrize('transparent',[False,True])
@pytest.mark.parametrize('options',[
    {'min_angles':0},{'max_angles':2**31},{'probe_angles':2**31},
    {'maximum_distance':2**31},{'rays_per_surface_cell':float('nan')},
    {'effective_density':'invalid'},{'statistic':'invalid'},
    {'angle_count_grid':np.zeros((3,)*3)},
])
def test_invalid_options_rejected_before_transparent_shortcut(transparent,options):
    c=np.ones((3,)*3);s=np.zeros_like(c);s[1,1,1]=1
    with pytest.raises(ValueError):new.solve(s,c,c if transparent else None,**options)


def test_concatenated_directions_overflow_rejected_before_allocation():
    c=np.ones((3,)*3);s=np.zeros_like(c);s[0,0,0]=1;s[1,1,1]=1
    counts=np.ones(c.shape,dtype=np.int64)
    counts[0,0,0]=2**30;counts[1,1,1]=2**30+1
    with pytest.raises(ValueError,match='Concatenated'):new.solve(s,c,angle_count_grid=counts)


def test_subnormal_ray_budget_rejected():
    c=np.ones((3,)*3);s=np.zeros_like(c);s[1,1,1]=np.finfo(float).tiny
    with pytest.raises(ValueError,match='subnormal'):new.solve(s,c)


def test_per_angle_ledger_reported():
    c=np.ones((7,)*3);s=np.zeros_like(c);s[3,3,3]=70
    result=new.solve(s,c)
    assert result['audit']['per_angle_relative_ledger_max']<5e-10


def test_dominant_guard_includes_equal_one_percent_sources():
    c=np.ones((5,)*3);s=np.zeros_like(c);s.ravel()[:100]=1
    result=new.solve(s,c,dominant_source_fraction=.01)
    assert result['audit']['dominant_sources']==100


def test_dominant_guard_protects_historical_total_density_counts():
    c=np.ones((16,)*3);c[:4]=.001;old=np.zeros_like(c);old[:4]=c[:4]
    s=np.zeros_like(c);s[10,8,8]=10
    original=hybrid.solve(s,c,old,maximum_distance=8)
    result=new.solve(s,c,old,maximum_distance=8,rays_per_surface_cell=16,
                     effective_density='remaining',dominant_source_fraction=.01)
    np.testing.assert_array_equal(result['angle_counts'][s>0],original['angle_counts'][s>0])
