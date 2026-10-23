"""Separate angular/spatial validation of the read-only event-front kernel.

All numbers concern the declared infinite-c, no-recombination opaque-voxel
model. The comparison field is the voxel integral of a continuous sphere,
not a rank-ordered set of lattice sites. No native cosmology runs are needed.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.spatial.transform import Rotation

from angular_tests import sphere_voxel_reference
from characteristics import clearance,directions,trace
from event_front import solve


def independent_audit(out,capacity):
    d=out['audit'];Q=d['photons'];scale=max(Q,1.)
    for key in ['ledger_residual','per_source_ledger_max','per_angle_ledger_max','independent_absorption_ledger']:
        assert abs(d[key])<1e-10*scale,(key,d[key],Q)
    assert np.min(out['absorbed'])>=-1e-12
    assert np.max(out['absorbed']-capacity)<1e-12
    return {key:d[key] for key in ['photons','absorbed','escaped','retained','ledger_residual',
                                   'per_source_ledger_max','per_angle_ledger_max','capacity_excess']}


@lru_cache(maxsize=12)
def sphere_inputs(radius):
    n=4*radius+8;center=np.full(3,n//2);Q=4*np.pi*radius**3/3
    s=np.zeros((n,)*3);s[tuple(center)]=Q;c=np.ones_like(s)
    reference=sphere_voxel_reference(n,radius,center,quadrature=48)
    reference24=sphere_voxel_reference(n,radius,center,quadrature=24)
    coordinate=np.indices(s.shape)-center[:,None,None,None]
    rad=np.sqrt((coordinate*coordinate).sum(0))
    return s,c,reference,coordinate,rad,dict(
        volume_relative_error=float(reference.sum()/Q-1),
        quadrature24_48_l1_per_photon=float(abs(reference-reference24).sum()/Q))


def sphere_measure(radius,angles,rotation=None,return_field=False):
    s,c,reference,coord,rad,reference_audit=sphere_inputs(radius)
    d=directions(angles)
    if rotation is not None:d=d@np.asarray(rotation).T
    out=solve(s,c,ray_directions=d);a=out['absorbed'];Q=float(s.sum());absorbed=float(a.sum())
    centroid=np.array([(a*coord[i]).sum()/absorbed for i in range(3)])
    moment=np.array([[(a*coord[i]*coord[j]).sum()/absorbed for j in range(3)] for i in range(3)])
    central=moment-np.outer(centroid,centroid)
    anisotropy=float(np.sqrt(1.5)*np.linalg.norm(central/np.trace(central)-np.eye(3)/3))
    row=dict(radius_cells=radius,grid=s.shape[0],angles=angles,
        voxel_sphere_l1_per_photon=float(abs(a-reference).sum()/Q),
        reference=reference_audit,quadrupole_anisotropy=anisotropy,
        centroid_over_radius=(centroid/radius).tolist(),
        centroid_norm_over_radius=float(np.linalg.norm(centroid)/radius),
        rms_radius_equivalent_over_true=float(np.sqrt(5*np.trace(moment)/3)/radius),
        absorbed_outside_R_plus_1_fraction=float(a[rad>radius+1].sum()/Q),
        absorbed_outside_R_plus_2_fraction=float(a[rad>radius+2].sum()/Q),
        escaped_fraction=float(out['audit']['escaped']/Q),audit=independent_audit(out,c))
    row['l1_plus_unabsorbed_fraction']=row['voxel_sphere_l1_per_photon']+(out['audit']['escaped']+out['audit']['retained'])/Q
    return (row,a.copy()) if return_field else row


def angular_convergence():
    counts=[48,192,768,3072,12288,49152];rows=[]
    for R in [8,12]:
        group=[];fields=[]
        for K in counts:
            row,field=sphere_measure(R,K,return_field=True);group.append(row);fields.append(field)
        for row,a in zip(group,fields):
            row['l1_vs_same_grid_K49152_per_photon']=float(abs(a-fields[-1]).sum()/row['audit']['photons'])
        rows.extend(group)
    return dict(variable='angle count only',fixed_radius_cells=[8,12],rows=rows,
        interpretation='The remaining K49152 error is chiefly the finite opaque-voxel front error; this is checked by separate spatial refinement, not assumed zero.')


def spatial_convergence():
    rows=[sphere_measure(R,49152) for R in [4,8,12,16]]
    angular_floor_checks=[]
    for R in [12,16]:
        low,a=sphere_measure(R,12288,return_field=True);high,b=sphere_measure(R,49152,return_field=True)
        angular_floor_checks.append(dict(radius_cells=R,low_angles=12288,high_angles=49152,
            low_l1_error=low['voxel_sphere_l1_per_photon'],high_l1_error=high['voxel_sphere_l1_per_photon'],
            field_l1_between_angles_per_photon=float(abs(a-b).sum()/low['audit']['photons'])))
    return dict(variable='radius in cells only',fixed_angles=49152,rows=rows,
        high_angle_crosschecks=angular_floor_checks,
        physical_interpretation='Fixed physical sphere with cell width proportional to1/R; normalized errors are compared. Angular quadrature is held fixed.')


def enlarged_box_low_angle_control():
    """Check that poor K48 morphology is not explained away by open edges."""
    R=8;K=48;n=128;center=n//2;Q=4*np.pi*R**3/3
    s=np.zeros((n,)*3);c=np.ones_like(s);s[center,center,center]=Q
    out=solve(s,c,angles=K)
    _,_,reference,_,_,_=sphere_inputs(R);half=reference.shape[0]//2
    crop=out['absorbed'][center-half:center+half,center-half:center+half,center-half:center+half]
    outside=float(out['absorbed'].sum()-crop.sum())
    error=float((abs(crop-reference).sum()+outside)/Q)
    assert out['audit']['escaped']==0
    return dict(radius_cells=R,angles=K,grid=n,escaped_fraction=0.,
        voxel_sphere_l1_per_photon=error,absorbed_outside_original_40cubed_box_fraction=outside/Q,
        audit=independent_audit(out,c),
        purpose='Same R8 cell resolution/K48 but a128³ box prevents escape. The larger sphere-shape error proves angular under-resolution remains even without the finite-box leakage artifact.')


def phase_sensitivity(phases=8):
    rotations=Rotation.random(phases,random_state=np.random.default_rng(918721)).as_matrix()
    rows=[]
    for R in [8,12]:
        for K in [48,192,768,3072,12288]:
            measures=[];maps=[]
            for rotation in rotations:
                row,a=sphere_measure(R,K,rotation,return_field=True);measures.append(row);maps.append(a)
            Q=measures[0]['audit']['photons'];errors=np.array([x['voxel_sphere_l1_per_photon'] for x in measures])
            pair=max(float(abs(a-b).sum()/(2*Q)) for i,a in enumerate(maps) for b in maps[:i])
            rows.append(dict(radius_cells=R,angles=K,phases=phases,
                l1_error_min=float(errors.min()),l1_error_mean=float(errors.mean()),
                l1_error_max=float(errors.max()),l1_error_std=float(errors.std()),
                max_pairwise_half_l1_per_photon=pair,
                quadrupole_anisotropy_max=max(x['quadrupole_anisotropy'] for x in measures),
                centroid_norm_over_radius_max=max(x['centroid_norm_over_radius'] for x in measures),
                escaped_fraction_max=max(x['escaped_fraction'] for x in measures)))
    return dict(seed=918721,rotations=rotations.tolist(),rows=rows,
        meaning='Keep the physical sphere and grid fixed; rotate only the equal-weight quadrature. Exact isotropic physics is unchanged.',
        recommended_standard='Report mean and worst reference error plus max pairwise half-L1/Q over quadrature orientations; this measures sensitivity, not a proof of accuracy.')


@lru_cache(maxsize=8)
def bubble_inputs(radius):
    R=int(radius);n=4*R+8;A=np.array([R+4,n//2,n//2]);B=A+[3*R//2,0,0]
    xyz=np.indices((n,)*3)
    mask_a=((xyz-A[:,None,None,None])**2).sum(0)<=R*R
    mask_b=((xyz-B[:,None,None,None])**2).sum(0)<=R*R
    mask=mask_a|mask_b
    return A,B,mask_a,mask_b,mask,clearance(mask)


def continuous_union_exit(radius,dirs):
    R=float(radius);distance=1.5*R;mu=dirs[:,0];cone=mu>=.75
    lengths=np.full(len(dirs),R)
    lengths[cone]=distance*mu[cone]+np.sqrt(R*R-distance**2*(1-mu[cone]**2))
    return lengths,cone


def tagged_bubble_measure(radius,angles=49152,rotation=None,verify_event=False):
    """Label rays by their last transparent cell, never by absorber-center angle."""
    A,B,ma,mb,mask,clear=bubble_inputs(radius);d=directions(angles)
    if rotation is not None:d=d@np.asarray(rotation).T
    out=trace(mask,[A+.5],d,skip=False,workers=1,precomputed_clearance=clear)
    assert np.all(out['hits']>=0)
    lengths=out['distance'][0];exact,cone=continuous_union_exit(radius,d)
    physical_source=A+.5
    # The DDA disambiguation is1e-10 per face.1e-7 upstream is safely in the
    # last traversed voxel for these grids, apart from exact tangencies.
    before=physical_source+(lengths[:,None]-1e-7)*d
    previous=np.floor(before).astype(int);pidx=tuple(previous.T)
    assert np.all(mask[pidx]),'Failed to identify the upstream transparent cell'
    in_a=ma[pidx];in_b=mb[pidx]
    b_exclusive=in_b&~in_a;shared=in_a&in_b
    a_exclusive=in_a&~in_b
    # A B-exclusive last transparent cell establishes an actual far-B exit.
    # A shared last cell is reported separately, not silently called B-only.
    mismatch=b_exclusive^cone
    hit_centres=np.array(np.unravel_index(out['hits'][0],mask.shape)).T+.5
    endpoint=physical_source+lengths[:,None]*d
    continuum_centroid=(exact[:,None]*d).mean(0)/radius
    exact_centroid=.5*quad(lambda mu:(1.5*mu+np.sqrt(1-2.25*(1-mu*mu))-1)*mu,.75,1)[0]
    row=dict(radius_cells=radius,grid=mask.shape[0],angles=angles,
        analytic_cone_weight=float(cone.mean()),continuum_expected_cone_weight=.125,
        actual_B_exclusive_exit_weight=float(b_exclusive.mean()),
        shared_A_B_last_cell_weight=float(shared.mean()),
        B_exit_fraction_lower_bound=float(b_exclusive.mean()),
        B_exit_fraction_upper_bound=float(np.mean(b_exclusive|shared)),
        cone_footprint_symmetric_difference=float(mismatch.mean()),
        definite_footprint_mismatch_weight=float(np.mean((cone&a_exclusive)|(~cone&b_exclusive))),
        ambiguous_footprint_weight=float(shared.mean()),
        cone_ambiguous_last_cell_weight=float(np.mean(cone&shared)),
        false_negative_cone_weight=float(np.mean(cone&~b_exclusive)),
        false_positive_cone_weight=float(np.mean(~cone&b_exclusive)),
        false_negative_fraction_of_input_cone=float(np.mean(~b_exclusive[cone])),
        first_front_centroid_over_R=((endpoint-physical_source).mean(0)/radius).tolist(),
        absorber_cell_centroid_over_R=((hit_centres-physical_source).mean(0)/radius).tolist(),
        quadrature_continuum_centroid_over_R=continuum_centroid.tolist(),
        exact_continuum_centroid_x_over_R=exact_centroid,
        mean_absolute_front_distance_error_over_R=float(abs(lengths-exact).mean()/radius),
        mean_absolute_cone_front_distance_error_over_R=float(abs(lengths[cone]-exact[cone]).mean()/radius),
        front_distance_99pct_error_over_R=float(np.quantile(abs(lengths-exact),.99)/radius))
    if verify_event:
        # Infinitesimal increment: every initially opaque voxel needs1 atom,
        # while all emitted photons together total1, so none can be overfilled.
        c=(~mask).astype(float);s=np.zeros_like(c);s[tuple(A)]=1
        event=solve(s,c,ray_directions=d)
        histogram=np.bincount(out['hits'][0],minlength=mask.size).reshape(mask.shape)/angles
        row['event_vs_frozen_histogram_linf']=float(abs(event['absorbed']-histogram).max())
        row['maximum_first_hit_atom_demand']=float(histogram.max())
        row['event_audit']=independent_audit(event,c)
        assert row['event_vs_frozen_histogram_linf']<1e-12
    return row


def bubble_convergence():
    spatial=[tagged_bubble_measure(R,49152,verify_event=R==12) for R in [4,8,12,16,24,32]]
    rotations=Rotation.random(8,random_state=np.random.default_rng(22871)).as_matrix()
    angular=[]
    for K in [48,192,768,3072,12288]:
        r=[tagged_bubble_measure(12,K,rotation) for rotation in rotations]
        angular.append(dict(radius_cells=12,angles=K,phases=8,
            analytic_cone_weight_min=min(x['analytic_cone_weight'] for x in r),
            analytic_cone_weight_max=max(x['analytic_cone_weight'] for x in r),
            analytic_cone_max_absolute_quadrature_error=max(abs(x['analytic_cone_weight']-.125) for x in r),
            actual_B_exit_weight_mean=float(np.mean([x['actual_B_exclusive_exit_weight'] for x in r])),
            actual_B_exit_weight_min=min(x['actual_B_exclusive_exit_weight'] for x in r),
            actual_B_exit_weight_max=max(x['actual_B_exclusive_exit_weight'] for x in r),
            cone_footprint_symmetric_difference_mean=float(np.mean([x['cone_footprint_symmetric_difference'] for x in r])),
            cone_footprint_symmetric_difference_max=max(x['cone_footprint_symmetric_difference'] for x in r)))
    return dict(spatial_refinement_fixed_K49152=spatial,
        angular_refinement_at_R12=angular,rotation_seed=22871,
        interpretation='Actual ray tags use the final transparent voxel before the first opaque face. B-only is a confirmed B-front exit; shared A/B last cells have ambiguous lobe attribution and bound the total B-exit weight. The strict B-only symmetric difference treats shared cells as outside B-only; it is not a rigorous physical error bound. Definite mismatches and continuous first-face distances are reported separately.',
        limitation='Infinitesimal front increment in a frozen two-ball union; it does not validate bulk front evolution or recombination.')


def source_hashes():
    here=Path(__file__).parent
    return {f:hashlib.sha256((here/f).read_bytes()).hexdigest() for f in
            ['event_front.cpp','event_front.py','characteristics.c','characteristics.py','angular_tests.py']}


def plot_results(result,folder):
    """Standalone scientific figure; data stay in the companion JSON."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=Path(folder);folder.mkdir(exist_ok=True,parents=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'savefig.facecolor':'white','figure.facecolor':'white'})
    fig=plt.figure(figsize=(11.5,6.3))
    grid=fig.add_gridspec(2,2,width_ratios=[1.35,1],height_ratios=[2.3,1],
                          left=.075,right=.98,bottom=.21,top=.80,hspace=.14,wspace=.28)
    angular=fig.add_subplot(grid[0,0]);escape=fig.add_subplot(grid[1,0],sharex=angular)
    spatial=fig.add_subplot(grid[:,1]);colors={8:'#2563a6',12:'#ca7421'}
    counts=[48,192,768,3072,12288,49152]
    for R in [8,12]:
        rows=[x for x in result['angular']['rows'] if x['radius_cells']==R]
        angular.plot([x['angles'] for x in rows],[100*x['voxel_sphere_l1_per_photon'] for x in rows],
                     '-o',color=colors[R],label=f'R / cell width = {R}',markersize=4,lw=1.8)
        escape.plot([x['angles'] for x in rows],[100*x['escaped_fraction'] for x in rows],
                    '--o',color=colors[R],markersize=3,lw=1.4)
        phase=[x for x in result['angular_phase']['rows'] if x['radius_cells']==R]
        angular.fill_between([x['angles'] for x in phase],[100*x['l1_error_min'] for x in phase],
                             [100*x['l1_error_max'] for x in phase],color=colors[R],alpha=.14)
    control=result['enlarged_box_control']
    angular.plot(48,100*control['voxel_sphere_l1_per_photon'],marker='*',color=colors[8],ms=11)
    angular.annotate(f"Larger box, no escape:\n{100*control['voxel_sphere_l1_per_photon']:.1f}% error",
        xy=(48,100*control['voxel_sphere_l1_per_photon']),xytext=(130,155),fontsize=9,
        arrowprops={'arrowstyle':'-','color':'#777','lw':.8})
    angular.set_xscale('log');angular.set_ylim(0,180)
    angular.set_ylabel('Sphere morphology L1 / Q (%)')
    angular.set_title('A  Angular refinement at fixed cell resolution',loc='left',fontsize=11,pad=12)
    angular.tick_params(labelbottom=False);angular.legend(loc='upper right',frameon=False,fontsize=9)
    angular.grid(axis='y',alpha=.15)
    escape.set_xscale('log');escape.set_xticks(counts,[f'{x:,}' for x in counts],rotation=20)
    escape.set_ylim(-2,75);escape.set_yticks([0,25,50,75]);escape.set_ylabel('Escaped / Q (%)')
    escape.set_xlabel('Equal-weight ray directions K');escape.grid(axis='y',alpha=.15)
    rows=result['spatial']['rows'];r=np.array([x['radius_cells'] for x in rows]);err=np.array([100*x['voxel_sphere_l1_per_photon'] for x in rows])
    spatial.plot(r,err,'-o',color='#287e62',lw=2,ms=5,label='K = 49,152')
    spatial.plot(r,err[0]*r[0]/r,'--',color='#777',lw=1.2,label='1 / resolved radius guide')
    spatial.axhline(5,color='#999',ls=':',lw=1)
    spatial.text(15.8,5.25,'5% illustrative target',ha='right',color='#777',fontsize=8)
    for index,(x,y) in enumerate(zip(r,err)):
        offset=(9,9) if index==0 else ((-5,9) if index==len(r)-1 else (0,9))
        align='left' if index==0 else ('right' if index==len(r)-1 else 'center')
        spatial.annotate(f'{y:.2f}%',(x,y),textcoords='offset points',xytext=offset,ha=align,fontsize=9)
    spatial.set_xticks(r);spatial.set_ylim(0,14);spatial.set_xlabel('Sphere radius / cell width')
    spatial.set_ylabel('Sphere morphology L1 / Q (%)');spatial.grid(axis='y',alpha=.15)
    spatial.set_title('B  Spatial refinement at fixed K',loc='left',fontsize=11,pad=12)
    spatial.legend(loc='upper right',frameon=False,fontsize=8)
    fig.suptitle('Angular sampling and voxel-front errors are distinct',x=.075,ha='left',y=.965,fontsize=16)
    fig.text(.075,.905,'Toy infinite-c opaque-voxel front; continuous-sphere voxel integral reference — not an RT benchmark.',fontsize=10,color='#444')
    fig.text(.075,.105,'L1 / Q can exceed 100%. Low-K escape can make inside-box L1 deceptively smaller; the lower panel reports it explicitly.',fontsize=9,color='#444')
    fig.text(.075,.066,'Shading: eight quadrature rotations. The enlarged-box control isolates the K=48 error from escape. Reference quadrature checks are in JSON.',fontsize=9,color='#444')
    fig.text(.075,.029,f"Read-only kernel SHA256: {result['kernel_sha256']['event_front.cpp'][:16]}…   |   No native-model timing claims",fontsize=8,color='#666')
    targets=[folder/'validation_geometry_convergence.png',folder/'validation_geometry_convergence.pdf']
    fig.savefig(targets[0],dpi=200);fig.savefig(targets[1]);plt.close(fig)
    return [str(p) for p in targets]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path(__file__).with_suffix('.json'))
    parser.add_argument('--plot',action='store_true')
    parser.add_argument('--plot-dir',type=Path,default=Path(__file__).parent/'figures')
    args=parser.parse_args();hashes=source_hashes()
    result=dict(scope='Independent toy angular/spatial accuracy validation; core code read-only; no native-model cost claims.',
        kernel_sha256=hashes,angular=angular_convergence(),spatial=spatial_convergence(),
        enlarged_box_control=enlarged_box_low_angle_control(),
        angular_phase=phase_sensitivity(),overlap=bubble_convergence(),
        definitions=dict(shape_error='sum_voxels |absorbed - continuous_sphere_voxel_integral| / emitted_photons',
            quadrupole='sqrt(3/2) Frobenius norm of central second-moment tensor/trace minus I/3;0 for isotropic second moments.',
            orientation_distance='Half L1 difference between two quadrature orientations / emitted photons; cancellation-free moved-mass diagnostic.',
            l1_plus_unabsorbed='Inside-box morphology L1/Q plus escaped/Q plus retained/Q, reported to prevent large angular leakage from making the inside-box L1 deceptively smaller.',
            illustrative_target='5% atom-weighted L1 morphology error; a declared diagnostic threshold, not an externally established science requirement.'))
    assert hashes==source_hashes(),'Core changed during validation; rerun against one version'
    if args.plot:result['figures']=plot_results(result,args.plot_dir)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(args.output)


if __name__=='__main__':main()
