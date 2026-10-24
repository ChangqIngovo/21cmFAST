"""Coarsen a small photon-budget subset with an absolute map-error certificate.

Within the static, irreversible, opaque-cell event model, positive beam-rate
addition is order preserving. Holding all other sources' directions and rates
fixed, changing the angular quadrature of a subset W changes the final absorbed
map by at most 2*sum(Q[W]) in L1. This certificate is relative to the supplied
fine-quadrature map, not continuum RT, and excludes feedback/recombination.
"""
from __future__ import annotations
import numpy as np
import sampling_solver
import hybrid


def allocate(source,full_counts,*,epsilon=1e-3,coarse_angles=24):
    """Select whole equal-score groups under a strict emitted-photon budget.

The greedy score Q/(K_full-K_coarse) estimates emitted photons spent per saved
ray. It is not an optimal knapsack solution or a measured transport cost.
Whole exact-score tie groups preserve translations of equal-valued source maps.
"""
    s=np.asarray(source,dtype=float);counts=np.asarray(full_counts)
    if (s.shape!=counts.shape or not np.isfinite(s).all() or (s<0).any()
            or not np.isfinite(counts).all() or (counts<1).any()
            or (counts!=np.floor(counts)).any() or (counts>np.iinfo(np.int32).max).any()):
        raise ValueError('Matching nonnegative photon counts and positive integer angle counts required')
    if not np.isfinite(epsilon) or not 0<=epsilon<=2:
        raise ValueError('epsilon must lie in [0,2]')
    if not isinstance(coarse_angles,(int,np.integer)) or isinstance(coarse_angles,bool) or coarse_angles<1:
        raise ValueError('coarse_angles must be a positive integer')
    total=s.sum(dtype=np.longdouble)
    if not np.isfinite(total) or total>np.finfo(float).max:raise ValueError('Total source count overflows')
    budget=np.longdouble(epsilon)*total/2
    candidate=np.flatnonzero((s>0)&(counts>coarse_angles))
    score=s.ravel()[candidate]/(counts.ravel()[candidate]-coarse_angles)
    order=np.argsort(score,kind='stable');candidate=candidate[order];score=score[order]
    bounds=np.r_[0,np.flatnonzero(np.diff(score)!=0)+1,len(candidate)]
    selected=np.zeros(s.size,dtype=bool);spent=np.longdouble(0)
    for start,end in zip(bounds[:-1],bounds[1:]):
        group=candidate[start:end]
        q=s.ravel()[group].sum(dtype=np.longdouble)
        if spent+q<=budget:
            selected[group]=True;spent+=q
    selected=selected.reshape(s.shape)
    reduced=np.asarray(counts,dtype=np.int32).copy();reduced[selected]=coarse_angles
    return {'angle_counts':reduced,'selected':selected,'full_counts':counts.copy(),
            'audit':{'epsilon_requested':float(epsilon),'selected_source_photons':float(spent),
                     'total_source_photons':float(total),
                     'L1_absorption_absolute_bound':float(2*spent),
                     'L1_over_emitted_bound':float(2*spent/total) if total else 0.,
                     'selected_sources':int(selected.sum()),
                     'predicted_ray_savings':int(np.sum(counts[selected]-coarse_angles,dtype=np.int64)),
                     'full_source_angle_sum':int(counts[s>0].sum(dtype=np.int64)),
                     'reduced_source_angle_sum':int(reduced[s>0].sum(dtype=np.int64)),
                     'scope':'relative to identical strong-source quadratures; static opaque shared-event model',
                     'actual_cost_savings_certified':False}}


def solve(source,capacity,initial_ionized=None,*,full_counts=None,epsilon=1e-3,
          coarse_angles=24,periodic=True,maximum_distance=None):
    s=np.asarray(source,dtype=float);c=np.asarray(capacity,dtype=float)
    old=np.zeros_like(c) if initial_ionized is None else np.asarray(initial_ionized,dtype=float)
    if full_counts is None:
        levels,counts,_,_=hybrid.estimate_angles(s,c,c-old,periodic=periodic,maximum_distance=maximum_distance)
        full_counts=counts[levels]
    allocation=allocate(s,full_counts,epsilon=epsilon,coarse_angles=coarse_angles)
    result=sampling_solver.solve(s,c,old,angle_count_grid=allocation['angle_counts'],
                                 periodic=periodic,maximum_distance=maximum_distance)
    result['audit']['coarsening']=allocation['audit']
    result['coarsened_sources']=allocation['selected']
    return result
