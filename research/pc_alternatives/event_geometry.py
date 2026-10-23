"""Independent spatial/invariance probes for the continuous event-front solver.

This validates the declared infinite-c opaque-voxel model and quantifies its
spatial/angle discretization error. It does not test recombinations, feedback,
finite-speed transfer, or a production native-21cmFAST integration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad

from angular_tests import sphere_voxel_reference
from characteristics import directions,trace
from event_front import solve


def compare(a,b):
    delta=np.abs(a-b)
    return dict(l1=float(delta.sum()),linf=float(delta.max()),
                l1_per_photon=float(delta.sum()/max(a.sum(),1e-300)))


def check(out,capacity):
    a=out['audit'];scale=max(a['photons'],1.)
    assert abs(a['ledger_residual'])<1e-10*scale,a
    assert abs(a['independent_absorption_ledger'])<1e-10*scale,a
    assert a['per_source_ledger_max']<1e-10*scale,a
    assert a['per_angle_ledger_max']<1e-10*scale,a
    assert np.all(out['absorbed']>=-1e-12)
    assert np.max(out['absorbed']-capacity)<1e-12
    return {k:v for k,v in a.items() if k!='seconds'}


def monotonicity_and_partition():
    n=16;s=np.zeros((n,)*3);c=np.ones_like(s);s[6,8,8]=s[9,8,8]=160
    s_more=s.copy();s_more[6,8,8]=240;k=3072
    base=solve(s,c,angles=k);more=solve(s_more,c,angles=k)
    half=solve(s,c,angles=k,duration=.5)
    half2=solve(s,np.maximum(c-half['absorbed'],0),angles=k,duration=.5)
    decrease=np.maximum(base['absorbed']-more['absorbed'],0)
    partition=compare(base['absorbed'],half['absorbed']+half2['absorbed'])
    assert decrease.max()<1e-11
    assert partition['linf']<1e-10
    fixture=dict(grid=n,angles=k,capacity=1,source_positions=[[6,8,8],[9,8,8]],
        source_rates_before=[160,160],source_rates_after=[240,160],duration=1,
        worst_pointwise_absorption_drop=float(decrease.max()),
        total_absorption_drop=float(decrease.sum()),target=[7,4,7],
        target_before=float(base['absorbed'][7,4,7]),target_after=float(more['absorbed'][7,4,7]),
        one_vs_two_half_steps=partition,
        audits=dict(base=check(base,c),more=check(more,c),half=check(half,c),
                    half2=check(half2,np.maximum(c-half['absorbed'],0))))
    rng=np.random.default_rng(774529);worst=0.;worst_partition=0.;random_audits=[]
    for trial in range(24):
        c=rng.uniform(.6,1.6,(16,)*3);s=np.zeros_like(c)
        positions=rng.integers(5,11,size=(4,3));rates=rng.uniform(20,120,size=4)
        for p,q in zip(positions,rates):s[tuple(p)]+=q
        s_more=s.copy();s_more[tuple(positions[0])]*=1.8
        base=solve(s,c,angles=384);more=solve(s_more,c,angles=384)
        a=solve(s,c,angles=384,duration=.37)
        b=solve(s,np.maximum(c-a['absorbed'],0),angles=384,duration=.63)
        worst=max(worst,float(np.maximum(base['absorbed']-more['absorbed'],0).max()))
        worst_partition=max(worst_partition,compare(base['absorbed'],a['absorbed']+b['absorbed'])['linf'])
        random_audits.extend([check(base,c),check(more,c),check(a,c),check(b,np.maximum(c-a['absorbed'],0))])
    assert worst<1e-10
    assert worst_partition<1e-10
    return dict(previous_scalar_failure_fixture=fixture,random_trials=24,random_seed=774529,
                random_worst_pointwise_drop=worst,random_partition_linf=worst_partition,
                random_per_source_ledger_max=max(x['per_source_ledger_max'] for x in random_audits),
                scope='Fixed gas, fixed spectrum/rates within step, no recombinations or feedback.')


def rigid_invariance():
    n=24;rng=np.random.default_rng(483);c=rng.uniform(.7,1.3,(n,)*3)
    s=np.zeros_like(c);s[10,10,12]=170;s[13,12,11]=90;s[11,14,13]=40
    d=directions(768);base=solve(s,c,ray_directions=d)
    shift=(1,2,-2);moved=solve(np.roll(s,shift,axis=(0,1,2)),np.roll(c,shift,axis=(0,1,2)),ray_directions=d)
    back=np.roll(moved['absorbed'],tuple(-x for x in shift),axis=(0,1,2))
    translations=compare(base['absorbed'],back)
    rotation=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
    rotated=solve(np.rot90(s,axes=(0,1)),np.rot90(c,axes=(0,1)),ray_directions=d@rotation.T)
    rotations=compare(base['absorbed'],np.rot90(rotated['absorbed'],k=-1,axes=(0,1)))
    fixed_dirs=solve(np.rot90(s,axes=(0,1)),np.rot90(c,axes=(0,1)),ray_directions=d)
    fixed_compare=compare(base['absorbed'],np.rot90(fixed_dirs['absorbed'],k=-1,axes=(0,1)))
    assert translations['linf']<1e-10
    assert rotations['linf']<1e-10
    assert base['audit']['escaped']==0 and moved['audit']['escaped']==0 and rotated['audit']['escaped']==0
    return dict(grid=n,angles=len(d),shift=list(shift),translation=translations,
                cubic_rotation_with_rotated_direction_set=rotations,
                cubic_rotation_with_fixed_direction_set=fixed_compare,
                audits=dict(base=check(base,c),translated=check(moved,np.roll(c,shift,axis=(0,1,2))),
                    rotated=check(rotated,np.rot90(c,axes=(0,1)))),
                interpretation='Direction sets must rotate with geometry for exact covariance; fixed finite quadrature has angular anisotropy.')


def two_preionized_spheres(fields_path=None):
    n=64;R=12;distance=18;A=np.array([20,32,32]);B=A+[distance,0,0]
    xyz=np.indices((n,)*3);a=xyz-A[:,None,None,None];b=xyz-B[:,None,None,None]
    rA=np.sqrt((a*a).sum(0));rB=np.sqrt((b*b).sum(0))
    c=((rA>R)&(rB>R)).astype(float);s=np.zeros_like(c);s[tuple(A)]=100
    k=12288;d=directions(k);event=solve(s,c,ray_directions=d);audit=check(event,c)
    # Independent ordinary cell-face DDA, with the optional clearance skip OFF.
    first=trace(c==0,[A+.5],d,skip=False,workers=1)
    assert np.all(first['hits']>=0)
    histogram=np.bincount(first['hits'][0],minlength=n**3).reshape(c.shape)*(100/k)
    assert histogram.max()<1,'This small increment should not exhaust any first-hit voxel'
    frozen_compare=compare(event['absorbed'],histogram)
    assert frozen_compare['linf']<1e-11
    input_cone=d[:,0]>=.75
    center_cone=(a[0]>0)&(a[0]**2>=.75**2*rA*rA)
    centroid=[float((event['absorbed']*a[i]).sum()/100) for i in range(3)]
    analytic_distance=np.full(k,float(R))
    analytic_distance[input_cone]=distance*d[input_cone,0]+np.sqrt(
        R*R-distance**2*(1-d[input_cone,0]**2))
    continuum_endpoint=analytic_distance[:,None]*d
    exact_centroid_x=.5*quad(lambda mu:(distance*mu+np.sqrt(R*R-distance**2*(1-mu*mu))-R)*mu,.75,1)[0]
    voxel_dist=first['distance'][0]
    result=dict(grid=n,radius=R,center_distance=distance,photons=100,angles=k,
        audit=audit,independent_frozen_DDA_spatial_comparison=frozen_compare,
        maximum_first_hit_voxel_demand=float(histogram.max()),
        emitted_direction_cone_fraction=float(input_cone.mean()),continuum_cone_fraction=.125,
        absorption_classified_by_voxel_center_cone=float(event['absorbed'][center_cone].sum()/100),
        event_absorption_centroid_cells=centroid,
        continuum_infinitesimal_centroid_x_cells=exact_centroid_x,
        angular_quadrature_continuum_centroid_cells=continuum_endpoint.mean(axis=0).tolist(),
        mean_absolute_first_hit_distance_error_cells=float(abs(voxel_dist-analytic_distance).mean()),
        cone_mean_absolute_first_hit_distance_error_cells=float(abs(voxel_dist[input_cone]-analytic_distance[input_cone]).mean()),
        cone_early_voxel_hit_by_more_than_two_cells_fraction=float(np.mean(voxel_dist[input_cone]<analytic_distance[input_cone]-2)),
        interpretation='The 12.5% input cone quota is preserved, but finite voxel surfaces shift where those rays first hit. The histogram comparison verifies spatial placement for the actual voxel geometry; it does not prove continuum shape accuracy.')
    if fields_path is not None:
        from hierarchy import radial_redistribute
        scalar=radial_redistribute(s,c,maximum_radius=31)
        metadata=dict(grid=n,radius=R,center_distance=distance,center_A=A.tolist(),center_B=B.tolist(),
            photons=100,duration=1,event_angles=k,
            event='Constant isotropic rate100 at centerA forduration1; infinite-c opaque voxel front; no initial first-hit cell exhausts.',
            scalar='SCRIPT-like reconstruction: independent source radial shells, equal-photon terminal shell then synchronous capacity-proportional excess relays; photon increment100.',
            source_positions='Integer array coordinates denote cell centers; transport internal physical coordinates add0.5.',
            transparent_mask='Union of cells whose centers lie within either radius12 sphere.',
            DDA='Independent ordinary non-skipping DDA frozen-opacity first-hit photon histogram.',
            event_audit=audit,scalar_audit=scalar['audit'])
        np.savez_compressed(fields_path,transparent=c==0,capacity=c,source=s,
            event_absorbed=event['absorbed'],frozen_dda_absorbed=histogram,
            script_like_absorbed=scalar['absorbed'],metadata=np.asarray(json.dumps(metadata)))
        result['field_archive']=str(fields_path)
    return result


def opaque_plane():
    n=32;c=np.ones((n,)*3);c[10,:,:]=1e6;s=np.zeros_like(c);s[8,16,16]=1000
    out=solve(s,c,angles=3072);a=out['absorbed'];fraction=float(a[10].sum()/1000)
    assert fraction<=.5+1e-12
    assert a[11:].sum()==0
    return dict(grid=n,source=[8,16,16],plane_x=10,photons=1000,plane_capacity_per_cell=1e6,
        absorbed_fraction_plane=fraction,isotropic_directional_upper_bound=.5,
        photons_behind_plane=float(a[11:].sum()),audit=check(out,c),
        scope='Opaque plane cannot exhaust; open boundaries prevent periodic return paths.')


def sphere_spatial_convergence():
    rows=[]
    for R in [4,8,12]:
        n=4*R+8;center=np.full(3,n//2);s=np.zeros((n,)*3);c=np.ones_like(s)
        photons=4*np.pi*R**3/3;s[tuple(center)]=photons;k=int(np.ceil(16*np.pi*R*R))
        out=solve(s,c,angles=k);reference=sphere_voxel_reference(n,R,center,quadrature=48)
        low_reference=sphere_voxel_reference(n,R,center,quadrature=24)
        a=np.moveaxis(np.indices(s.shape),0,-1)-center;rad=np.linalg.norm(a,axis=-1)
        row=dict(radius_cells=R,grid=n,angles=k,photons=photons,audit=check(out,c),
            sphere_voxel_volume_l1_per_photon=float(abs(out['absorbed']-reference).sum()/photons),
            reference_volume_relative_error=float(reference.sum()/photons-1),
            reference_quadrature24_48_l1_per_photon=float(abs(reference-low_reference).sum()/photons),
            photons_outside_radius_plus_one=float(out['absorbed'][rad>R+1].sum()/photons),
            photons_outside_radius_plus_two=float(out['absorbed'][rad>R+2].sum()/photons),
            centroid_cells=[float((out['absorbed']*a[...,i]).sum()/photons) for i in range(3)])
        rows.append(row)
    return dict(reference='Continuous analytic sphere intersected with voxels;48² midpoint xy quadrature with exact z intersection, compared to24².',
                angle_rule='ceil(16*pi*R_cells**2)',rows=rows,
                caveat='Joint angular/spatial refinement at one point-source phase; not a complete convergence proof for arbitrary source/density fields.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path(__file__).with_name('event_geometry_results.json'))
    p.add_argument('--fields',type=Path)
    args=p.parse_args();here=Path(__file__).parent
    hashes={name:hashlib.sha256((here/name).read_bytes()).hexdigest() for name in ['event_front.py','event_front.cpp','characteristics.c','angular_tests.py']}
    result=dict(scope='Independent geometry tests; declared infinite-c sharp opaque voxel model, no recombination; no performance benchmark.',
        source_sha256=hashes,monotonicity_and_partition=monotonicity_and_partition(),
        rigid_invariance=rigid_invariance(),overlap=two_preionized_spheres(args.fields),
        opaque_plane=opaque_plane(),
        sphere_convergence=sphere_spatial_convergence())
    after={name:hashlib.sha256((here/name).read_bytes()).hexdigest() for name in hashes}
    assert hashes==after,'Implementation changed during validation; rerun to pin one version'
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(args.output)


if __name__=='__main__':main()
