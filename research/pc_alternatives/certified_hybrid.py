"""Conservative toy acceptance certificates, not a production hybrid solver.

The shell extrema bound EVERY direction through a piecewise-constant density
grid: cells are included if their bounding cube intersects a radial shell.
This deliberately does not infer a bound from a finite set of sampled rays.
The certificate applies to one isolated isotropic source, static hydrogen,
no recombinations/scattering, and a sharp front.  A separate exact photon
ledger is required before accepting a photon-conserving spherical update.
"""
from __future__ import annotations

import json
import argparse
import hashlib
from pathlib import Path

import numpy as np


def shell_density_extrema(density,center,edges):
    """Rigorous density bounds for each spherical shell in a unit-cell grid.

    Density is constant within each closed unit cube centered on an integer
    coordinate. Cells touching a shell boundary are included conservatively.
    The sphere must fit within the domain; there is no periodic aliasing.
    """
    density=np.asarray(density,float);center=np.asarray(center,float);edges=np.asarray(edges,float)
    if density.ndim!=3 or np.any(density<0) or not np.isfinite(density).all():
        raise ValueError('Finite nonnegative density on a 3D grid is required')
    if center.shape!=(3,) or len(edges)<2 or edges[0]<0 or np.any(np.diff(edges)<=0):
        raise ValueError('Invalid source center or radial shell edges')
    if edges[-1]>np.min(np.minimum(center+.5,np.array(density.shape)-.5-center)):
        raise ValueError('Trial sphere must fit inside this nonperiodic certificate domain')
    delta=np.abs(np.indices(density.shape)-center[:,None,None,None])
    rmin=np.sqrt((np.maximum(delta-.5,0)**2).sum(0)).ravel()
    rmax=np.sqrt(((delta+.5)**2).sum(0)).ravel()
    lo=np.searchsorted(edges,rmin,side='left')-1
    hi=np.searchsorted(edges,rmax,side='right')-1
    valid=(rmin<=edges[-1])&(rmax>=edges[0])
    lo=np.maximum(lo[valid],0);hi=np.minimum(hi[valid],len(edges)-2)
    rho=density.ravel()[valid]
    minimum=np.full(len(edges)-1,np.inf);maximum=np.zeros(len(edges)-1)
    # For shell width comparable to a cell, each cube intersects O(1) shells.
    # Very fine shell spacing increases this loop accordingly.
    for offset in range(int(np.max(hi-lo))+1):
        shell=lo+offset;keep=shell<=hi
        np.minimum.at(minimum,shell[keep],rho[keep])
        np.maximum.at(maximum,shell[keep],rho[keep])
    if not np.isfinite(minimum).all():raise ValueError('A shell has no intersecting cells')
    return minimum,maximum


def certify_single_source_sphere(density,center,edges,photons,*,epsilon=.05,
                                exact_trial_photons=None,source_count=1):
    """Bound atom-L1 morphology error and cone allocation error by epsilon Q.

    For each direction, M(Omega)=integral rho r²dr. An isotropic source gives
    B=Q/(4*pi) photons per steradian. If |M/B-1|<=epsilon, the atom-weighted
    difference between this trial front and the exact radial front is at
    most epsilon*Q, and every cone obeys the same relative budget bound.
    Exact radial transport must exist inside the modeled domain. The source
    count and exact spherical ledger are external preconditions, not guessed.
    """
    if photons<=0 or not 0<=epsilon<1:raise ValueError('Positive budget and epsilon in [0,1) required')
    lower,upper=shell_density_extrema(density,center,edges)
    edges=np.asarray(edges,float);weights=np.diff(edges**3)/3
    m_lower=float(lower@weights);m_upper=float(upper@weights);B=photons/(4*np.pi)
    bound=max(abs(m_lower/B-1),abs(m_upper/B-1))
    geometry_pass=bool(source_count==1 and bound<=epsilon+1e-14)
    ledger_pass=bool(exact_trial_photons is not None and
                     abs(exact_trial_photons-photons)<=1e-12*photons)
    return dict(source_count=source_count,photons=photons,epsilon_requested=epsilon,
        all_direction_column_lower=m_lower,all_direction_column_upper=m_upper,
        isotropic_column_budget=B,relative_atom_l1_error_bound=bound,
        atom_l1_error_bound=bound*photons,cone_relative_budget_error_bound=bound,
        shell_density_min=lower.tolist(),shell_density_max=upper.tolist(),
        geometry_certificate_pass=geometry_pass,exact_trial_ledger_verified=ledger_pass,
        eligible_for_spherical_pc_update=bool(geometry_pass and ledger_pass),
        scope='Isolated isotropic source; static H; no recombinations, scattering or competing sources; sharp radial front.',
        certification='All intersecting cell cubes bound every direction; no sampled-ray interpolation assumption.')


def equal_sphere_visibility(radius,distance):
    """Exact all-source visibility classification for two equal-ball unions.

    This formula is for this analytic family only.  It is not a universal
    connected-component or coarse-mesh visibility classifier.
    """
    R=float(radius);d=float(distance)
    if R<=0 or d<0:raise ValueError('Invalid geometry')
    connected=d<=2*R
    all_source_visible=bool(connected and d<=np.sqrt(2)*R)
    r=dict(radius=R,center_distance=d,connected=connected,
        common_star_shape_for_both_centers=all_source_visible,
        rule='For this equal-two-ball family, d <= sqrt(2)*R is necessary and sufficient.',
        reason='Connectedness alone does not imply transparent source-to-boundary line segments.')
    if np.sqrt(2)*R<d<2*R:
        # Exposed Bsurface: nx > -d/(2R).  A has inward flux if nx < -R/d.
        nx=.5*(-d/(2*R)-R/d);normal=np.array([nx,np.sqrt(1-nx*nx),0.])
        point=np.array([d,0.,0.])+R*normal
        distance_a=float(np.linalg.norm(point))
        r.update(witness_B_outward_normal=normal.tolist(),witness_boundary_point=point.tolist(),
                 witness_outside_A=bool(distance_a>R),
                 source_A_outward_flux_numerator=float(point@normal),
                 neutral_gap_along_A_ray=distance_a-R)
    return r


def run_probes():
    n=32;center=(15.5,15.5,15.5);edges=np.arange(0,11,dtype=float)
    Q=4*np.pi*edges[-1]**3/3
    x=np.indices((n,)*3)[0]-center[0]
    # Reflection-odd perturbations integrate to zero in a centered sphere,
    # even as piecewise-constant cell densities; their trial ledger is exact.
    weak=1+.02*np.sin(x/5);strong=1+.8*np.sin(x/5)
    screen=np.ones((n,)*3);screen[18,:,:]=100
    uniform=certify_single_source_sphere(np.ones_like(weak),center,edges,Q,exact_trial_photons=Q)
    weak_r=certify_single_source_sphere(weak,center,edges,Q,exact_trial_photons=Q)
    strong_r=certify_single_source_sphere(strong,center,edges,Q,exact_trial_photons=Q)
    screen_r=certify_single_source_sphere(screen,center,edges,Q)
    missing_ledger=certify_single_source_sphere(weak,center,edges,Q)
    multiple=certify_single_source_sphere(weak,center,edges,Q,exact_trial_photons=Q,source_count=2)
    visibility=equal_sphere_visibility(12,18)
    assert uniform['eligible_for_spherical_pc_update']
    assert weak_r['eligible_for_spherical_pc_update']
    assert not strong_r['geometry_certificate_pass']
    assert not screen_r['geometry_certificate_pass']
    assert not missing_ledger['eligible_for_spherical_pc_update']
    assert not multiple['eligible_for_spherical_pc_update']
    assert visibility['connected'] and not visibility['common_star_shape_for_both_centers']
    assert visibility['source_A_outward_flux_numerator']<0
    return dict(uniform=uniform,weak_anisotropy=weak_r,strong_anisotropy=strong_r,opaque_screen=screen_r,
                missing_exact_ledger=missing_ledger,multiple_sources=multiple,
                two_sphere_visibility=visibility,
                compact_two_sphere_visibility=equal_sphere_visibility(12,12),
                next_step='Measure certified fraction on native additive source/demand fields, then transport all rejected domains with retained directions.',
                limitation='No solver, no 21cmFAST integration, no proof of a high acceptance fraction or low full-run cost.')


def native_geometry_probe(path):
    """Measure strict certificate opportunities on one frozen native box.

    A periodic 6-connected voxel boundary is used, without smoothing. A BFS
    unwraps components; winding components cannot be represented by one
    finite closed front. For each nonwinding component, all positive source
    cell centers must lie in every inward boundary-face half-space. This is
    a conservative visibility certificate for the voxel polyhedron, not a
    claim about an interpolated smooth front or future competing sources.
    """
    f=np.load(path);q=f['source'];c=f['capacity'];xhi=f['native_xHI'];n=q.shape[0]
    mask=xhi<1e-6;labels=np.full(q.shape,-1,int);unwrapped=np.zeros(q.shape+(3,),int)
    steps=[(axis,sign) for axis in range(3) for sign in (-1,1)]
    components=[]
    for flat in np.flatnonzero(mask):
        origin=tuple(np.unravel_index(flat,q.shape))
        if labels[origin]>=0:continue
        label=len(components);labels[origin]=label;unwrapped[origin]=origin
        queue=[origin];head=0;winding=False
        while head<len(queue):
            p=queue[head];head+=1
            for axis,sign in steps:
                t=list(p);t[axis]=(t[axis]+sign)%n;t=tuple(t)
                if not mask[t]:continue
                desired=unwrapped[p].copy();desired[axis]+=sign
                if labels[t]<0:
                    labels[t]=label;unwrapped[t]=desired;queue.append(t)
                elif np.any(unwrapped[t]!=desired):winding=True
        idx=tuple(np.asarray(queue).T);coords=unwrapped[idx];positive=q[idx]>0
        sourcecoords=coords[positive];smin=sourcecoords.min(0);smax=sourcecoords.max(0)
        minimum_dot=np.inf;faces=0
        for p in queue:
            for axis,sign in steps:
                t=list(p);t[axis]=(t[axis]+sign)%n;t=tuple(t)
                if mask[t]:continue
                face=unwrapped[p][axis]+sign*.5;faces+=1
                value=face-smax[axis] if sign>0 else smin[axis]-face
                minimum_dot=min(minimum_dot,float(value))
        visible=bool(not winding and minimum_dot>=0)
        components.append(dict(cells=len(queue),source_cells=int(np.count_nonzero(positive)),
            source_photons=float(q[idx].sum()),winding=winding,boundary_faces=faces,
            minimum_source_normal_distance=minimum_dot,
            common_visibility_certificate_pass=visible))
    total=float(q.sum());visible_photons=sum(z['source_photons'] for z in components if z['common_visibility_certificate_pass'])
    large_visible=sum(z['source_photons'] for z in components if z['common_visibility_certificate_pass'] and z['cells']>1)
    return dict(input_file=str(path),input_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        grid=n,total_cells=int(q.size),positive_source_cells=int(np.count_nonzero(q>0)),
        photons=total,source_above_local_capacity_cells=int(np.count_nonzero(q>c)),
        local_absorption_fraction=float(np.minimum(q,c).sum()/total),
        photons_needing_nonlocal_allocation_fraction=float(np.maximum(q-c,0).sum()/total),
        strict_isolated_single_source_whole_box=False,
        fully_ionized_cells=int(mask.sum()),component_count=len(components),
        winding_components=sum(z['winding'] for z in components),
        common_visibility_certified_components=sum(z['common_visibility_certificate_pass'] for z in components),
        all_source_photon_fraction_in_certified_components=visible_photons/total,
        all_source_photon_fraction_in_multicell_certified_components=large_visible/total,
        largest_components=sorted(components,key=lambda z:z['cells'],reverse=True)[:12],
        scope='Geometry-only current native ES fully ionized voxel components. Partial cells and sources outside components are not certified.',
        caveats=['Native ES morphology is not a photon-conserving reference.',
                 'Staircase voxel boundaries make this exact certificate stricter than a smoothed-front approximation.',
                 'Visibility is insufficient for finite-volume conservative front evolution and competition between components.',
                 'Positive conditional source values in all cells do not constitute individually resolved halos.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--native',type=Path);args=parser.parse_args()
    result=run_probes()
    if args.native is not None:result['native_geometry']=native_geometry_probe(args.native)
    path=Path(__file__).with_name('certified_hybrid_results.json')
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(path)
