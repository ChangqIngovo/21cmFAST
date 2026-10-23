"""Reproducible probes for count conservation versus spatial transport.

Run with the NumPy/SciPy runtime specified in CONTRACT.md.  Timings are toy
microbenchmarks, not native SCRIPT or complete 21cmFAST runtimes.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy
from scipy import fft
from scipy.integrate import quad

from hierarchy import dyadic_pool, radial_redistribute, radial_offsets


def metrics(a, b):
    return dict(l1=float(np.abs(a-b).sum()), linf=float(np.abs(a-b).max()),
                l1_per_photon=float(np.abs(a-b).sum()/max(a.sum(), 1e-300)))


def periodic_displacements(n, center):
    x=np.indices((n,)*3)-np.array(center)[:,None,None,None]
    return (x+n//2)%n-n//2


def sphere_metrics(a, center):
    d=periodic_displacements(a.shape[0],center)
    d2=(d*d).sum(axis=0)
    means=np.bincount(d2.ravel(),weights=a.ravel())/np.maximum(np.bincount(d2.ravel()),1)
    residual=a-means[d2]
    return dict(shell_rms=float(np.sqrt(np.mean(residual**2))),
                shell_l1_per_photon=float(np.abs(residual).sum()/a.sum()),
                centroid_cells=[float((a*d[i]).sum()/a.sum()) for i in range(3)],
                occupied_max_radius=float(np.sqrt(d2[a>1e-12].max())))


def algorithms(n):
    return dict(script_like=lambda q,c,**kw: radial_redistribute(q,c,maximum_radius=n/2-1,**kw),
                dyadic=lambda q,c,**kw:dyadic_pool(q,c))


def basic_and_geometry():
    n=32;c=np.ones((n,)*3);p=(16,16,16)
    cases={}
    q=np.zeros_like(c);cases['zero']=(q.copy(),c)
    q[p]=.4;cases['weak']=(q.copy(),c)
    q[p]=1e6;cases['saturation']=(q.copy(),c)
    q[p]=257.;cases['isolated']=(q.copy(),c)
    q=np.zeros_like(c);q[13,16,16]=257;q[19,16,16]=257
    cases['two_equal']=(q.copy(),c)
    q[13,16,16]=83;q[19,16,16]=347
    cases['two_unequal']=(q.copy(),c)
    rng=np.random.default_rng(572)
    cases['nonuniform']=(q.copy(),np.exp(rng.normal(0,.5,c.shape)))
    result={}
    for name,solve in algorithms(n).items():
        result[name]={}
        for case,(q,c0) in cases.items():
            r=solve(q,c0); result[name][case]=dict(r['audit'])
            if case=='isolated':result[name][case].update(sphere_metrics(r['absorbed'],p))
        q,c0=cases['nonuniform']; a=solve(q,c0)['absorbed']
        shifts=(1,3,5)
        translated=solve(np.roll(q,shifts,axis=(0,1,2)),np.roll(c0,shifts,axis=(0,1,2)))['absorbed']
        result[name]['translation']=metrics(a,np.roll(translated,tuple(-s for s in shifts),axis=(0,1,2)))
        rotated=solve(np.rot90(q,axes=(0,1)),np.rot90(c0,axes=(0,1)))['absorbed']
        result[name]['rotation_90deg']=metrics(a,np.rot90(rotated,k=-1,axes=(0,1)))
        if name=='script_like':
            reverse=solve(q,c0,source_order=[1,0])['absorbed']
            result[name]['source_permutation']=metrics(a,reverse)
        else:
            result[name]['source_permutation']={'definition':'Array input contains no source-order loop'}
        first=solve(q*.5,c0)
        second=solve(q*.5+first['remaining'],c0-first['absorbed'])
        result[name]['time_partition']=metrics(a,first['absorbed']+second['absorbed'])
    return result


def screen_and_overlap():
    n=32;q=np.zeros((n,)*3);c=np.ones_like(q);q[8,16,16]=1000;c[10,:,:]=1e6
    result={'opaque_plane':dict(grid=n,source=[8,16,16],plane_x=10,
                plane_capacity_per_cell=1e6,photons=1000,
                isotropic_photon_fraction_upper_bound=.5,
                note='No photon path wraps around the periodic box in the physical half-space bound.')}
    for name,solve in dict(script_like=lambda q,c:radial_redistribute(q,c,maximum_radius=12),
                           capacity_shell=lambda q,c:radial_redistribute(q,c,maximum_radius=12,initial_shell='capacity'),
                           dyadic=dyadic_pool).items():
        r=solve(q,c);a=r['absorbed']
        result['opaque_plane'][name]=dict(audit=r['audit'],
            absorbed_fraction_plane=float(a[10].sum()/q.sum()),
            absorbed_fraction_behind=float(a[11:20].sum()/q.sum()),
            directional_bound_pass=bool(a[10].sum()<=.5*q.sum()+1e-10))
    n=64;R=12;A=(20,32,32);B=(38,32,32)
    dA=periodic_displacements(n,A);dB=periodic_displacements(n,B)
    r2A=(dA*dA).sum(0);r2B=(dB*dB).sum(0)
    c=((r2A>R*R)&(r2B>R*R)).astype(float);q=np.zeros_like(c);q[A]=100
    cone=(dA[0]>0)&(dA[0]**2 >= .75**2*r2A)
    result['overlap_cone']=dict(grid=n,radius=R,center_distance=18,photons=100,
        expected_continuum_fraction=.125,
        infinitesimal_continuum_centroid_x_cells=.5*quad(
            lambda mu:(18*mu+np.sqrt(R*R-18**2*(1-mu*mu))-R)*mu,.75,1)[0],
        definition='Photons added at A after both spheres already ionized; cone cos(theta)>=d/(2R).',
        note='Finite-grid angular discretization cannot explain a completely empty positive-solid-angle cone.')
    for name,solve in algorithms(n).items():
        r=solve(q,c);a=r['absorbed']
        result['overlap_cone'][name]=dict(audit=r['audit'],absorbed_cone_fraction=float(a[cone].sum()/q.sum()),
            centroid_cells=[float((a*dA[i]).sum()/q.sum()) for i in range(3)])
    return result


def monotonicity_search(trials=120):
    n=8;rng=np.random.default_rng(293041);result={}
    for name,solve in algorithms(n).items():
        worst=0.;fixture=None;capacity_total_worst=0.
        for trial in range(trials):
            c=np.exp(rng.normal(0,.4,(n,)*3));q=np.zeros_like(c)
            sites=rng.choice(n**3,4,replace=False);q.flat[sites]=rng.uniform(5,45,4)
            a=solve(q,c)['absorbed'];q2=q.copy();q2.flat[sites[0]]*=1.7
            a2=solve(q2,c)['absorbed'];decrease=np.maximum(a-a2,0)
            if decrease.max()>worst:
                worst=float(decrease.max());idx=np.unravel_index(decrease.argmax(),q.shape)
                fixture=dict(trial=trial,source_sites_flat=sites.tolist(),source_photons=q.flat[sites].tolist(),
                    changed_site_flat=int(sites[0]),changed_multiplier=1.7,
                    cell=list(map(int,idx)),before=float(a[idx]),after=float(a2[idx]),
                    total_decreased=float(decrease.sum()),
                    pointwise_pass=False)
            c2=c.copy();c2.flat[sites[1]]*=2
            a3=solve(q,c2)['absorbed'];capacity_total_worst=max(capacity_total_worst,float(a.sum()-a3.sum()))
        result[name]=dict(trials=trials,seed=293041,worst_pointwise_photon_decrease=worst,
            photon_monotonicity_pass=worst<=1e-9,fixture=fixture,
            worst_total_absorption_drop_with_increased_capacity=capacity_total_worst)
    return result


def uniform_monotonicity_fixture():
    """A small deterministic failure without screens or nonuniform density."""
    n=16;c=np.ones((n,)*3);q=np.zeros_like(c)
    A=(6,8,8);B=(9,8,8);target=(7,4,7);q[A]=q[B]=160
    solve=lambda q,c:radial_redistribute(q,c,maximum_radius=7)
    base=solve(q,c);q_more=q.copy();q_more[A]=240;more=solve(q_more,c)
    decrease=np.maximum(base['absorbed']-more['absorbed'],0)
    first=solve(q*.5,c);second=solve(q*.5+first['remaining'],c-first['absorbed'])
    return dict(grid=n,uniform_capacity=1,source_positions=[A,B],photons_before=[160,160],
        photons_after=[240,160],per_hop_radius=7,target=target,
        target_before=float(base['absorbed'][target]),target_after=float(more['absorbed'][target]),
        total_decreased_cells=float(decrease.sum()),max_decrease=float(decrease.max()),
        monotonicity_pass=bool(decrease.max()<1e-10),
        time_partition=metrics(base['absorbed'],first['absorbed']+second['absorbed']),
        audits=dict(base=base['audit'],more=more['audit'],first_half=first['audit'],second_half=second['audit']),
        scope='SCRIPT-like Python reconstruction; not a run of public SCRIPT Fortran; fixed gas, no recombination or feedback.')


def support_relay_search():
    n=16;c=np.ones((n,)*3);q=np.zeros_like(c)
    positions=[(7,8,8),(9,8,8)];q[positions[0]]=32;q[positions[1]]=32
    radius=2.5;r=radial_redistribute(q,c,maximum_radius=radius)
    distance=np.minimum.reduce([np.sqrt((periodic_displacements(n,p)**2).sum(0)) for p in positions])
    a=r['absorbed'];far=a[distance>radius+1e-12].sum()
    return dict(grid=n,source_positions=positions,per_hop_radius=radius,audit=r['audit'],
        absorbed_beyond_every_original_source_radius=float(far),
        maximum_original_source_distance=float(distance[a>1e-12].max()),
        source_radius_bound_pass=bool(far<1e-10))


@lru_cache(maxsize=16)
def filter_kernels(n):
    d=periodic_displacements(n,(0,0,0));r2=(d*d).sum(0)
    radii=2.0**np.arange(0,int(np.log2(n))-1)
    return tuple(fft.rfftn((r2<=r*r).astype(float),workers=4) for r in radii)


def es_filter_microbenchmark(q,c):
    """Only a common-cost FFT filtering comparator, not a 21cmFAST model."""
    qk=fft.rfftn(q,workers=4);ck=fft.rfftn(c,workers=4)
    ionized=np.zeros(q.shape,bool)
    for kernel in filter_kernels(q.shape[0]):
        ionized |= fft.irfftn(qk*kernel,s=q.shape,workers=4)>=fft.irfftn(ck*kernel,s=q.shape,workers=4)
    return ionized


def benchmark(sizes=(32,64,128),repeats=3):
    rows=[]
    for n in sizes:
        rng=np.random.default_rng(81);c=np.ones((n,)*3);q=np.zeros_like(c)
        sites=rng.choice(n**3,8,replace=False);q.flat[sites]=100
        radius=n/4;start=time.perf_counter();radial_offsets(n,radius);filter_kernels(n)
        setup=time.perf_counter()-start
        row=dict(grid=n,cells=n**3,active_sources=8,photons=800,
                 fft_workers=4,python_radial_workers=1,dyadic_workers=1,
                 per_hop_radius=radius,geometry_and_kernel_setup_s=setup)
        for name,solve in dict(script_like=lambda:radial_redistribute(q,c,maximum_radius=radius),
                               dyadic=lambda:dyadic_pool(q,c),
                               fft_filter=lambda:es_filter_microbenchmark(q,c)).items():
            durations=[]
            for _ in range(repeats):
                t=time.perf_counter();r=solve();durations.append(time.perf_counter()-t)
            row[name]=dict(median_s=float(np.median(durations)),min_s=min(durations),all_s=durations)
            if isinstance(r,dict):row[name]['audit']=r['audit']
        row['ratio_script_to_fft']=row['script_like']['median_s']/row['fft_filter']['median_s']
        row['ratio_dyadic_to_fft']=row['dyadic']['median_s']/row['fft_filter']['median_s']
        row['fft_count']=2+2*len(filter_kernels(n))
        rows.append(row)
    return dict(scope='Warm geometry toy routines; shared-host contention possible; not full 21cmFAST or optimized SCRIPT.',
                source_pattern='8 fixed-seed sparse sources,100 photons each,uniform capacity1',rows=rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path(__file__).with_name('hierarchy_results.json'))
    p.add_argument('--skip-benchmark',action='store_true');p.add_argument('--benchmark-only',action='store_true')
    p.add_argument('--sizes',nargs='+',type=int,default=[32,64,128]);p.add_argument('--repeats',type=int,default=3)
    args=p.parse_args()
    results=dict(environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                                 cpu_count=os.cpu_count(),fft_workers=4),
                 semantics='Inputs are nonnegative photon increments and current residual atom capacities.')
    if not args.benchmark_only:
        results.update(basic_and_geometry=basic_and_geometry(),physical_counterexamples=screen_and_overlap(),
                       monotonicity=monotonicity_search(),support_relay=support_relay_search(),
                       uniform_monotonicity_fixture=uniform_monotonicity_fixture())
    if not args.skip_benchmark:results['benchmark']=benchmark(tuple(args.sizes),args.repeats)
    args.output.write_text(json.dumps(results,indent=2)+'\n')
    print(args.output)


if __name__=='__main__':main()
