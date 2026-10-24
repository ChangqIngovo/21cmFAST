"""Independent randomized comparisons with an eager explicit-source reference."""
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'pc_scale_origin'))
import point_front
sys.path.insert(0,str(HERE))
import refined


def audit(output, cases=80):
    paths=[HERE/'refined.cpp',HERE/'refined.py',HERE.parent/'pc_scale_origin/point_front.cpp',HERE.parent/'pc_scale_origin/point_front.py',Path(__file__)]
    hashes={str(p.relative_to(HERE.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    refined.library();point_front.library()
    started=time.perf_counter();rows=[]
    for seed in range(cases):
        rng=np.random.default_rng(8137+seed)
        n=(2,3,4)[seed%3];f=(2,4,8)[(seed//3)%3];k=(24,48,96,192)[seed%4]
        cap=np.exp(rng.uniform(-5,5,(n,)*3));cap[rng.random(cap.shape)<.15]=0
        source=np.exp(rng.uniform(-5,5,cap.shape));source[rng.random(cap.shape)<.75]=0
        if not source.any():source.ravel()[0]=1
        dose=(.01,.3,.8,1.,1.2,2.)[seed%6]
        source*=dose*cap.sum()/source.sum()
        periodic=seed%2==0;horizon=(.2,1.,3.,16.)[seed%4]*n
        finecap=cap.repeat(f,0).repeat(f,1).repeat(f,2)/f**3
        kwargs=dict(subcells_per_axis=f,angles=k,maximum_distance=horizon,periodic=periodic,return_dense=True)
        old=None;maxfield=0.;maxledger=0.
        for step in range(2):
            emitted=source if step==0 else np.roll(source,1,axis=0)*.37
            solved=refined.solve(emitted,cap,initial_state=old,**kwargs)
            remaining=finecap if old is None else finecap-previous_fine
            active=emitted.ravel()>0
            positions=(np.indices(cap.shape).reshape(3,-1).T+.5)[active]*f
            ref=point_front.solve(positions,emitted.ravel()[active],remaining,angles=k,maximum_distance=horizon*f,periodic=periodic)
            scale=emitted.sum(dtype=np.longdouble)
            field=float(np.abs(solved['subcell_absorbed']-ref['absorbed']).sum(dtype=np.longdouble)/scale)
            ledger=max(float(np.abs(solved[name].ravel()[active]-ref[name]).sum(dtype=np.longdouble)/scale) for name in ('source_absorbed','source_escaped','source_retained'))
            state_delta=float(np.abs(solved['subcell_ionized']-(finecap-remaining+ref['absorbed'])).sum(dtype=np.longdouble)/cap.sum(dtype=np.longdouble))
            if max(field,ledger,state_delta)>2e-10:raise AssertionError((seed,step,field,ledger,state_delta))
            maxfield=max(maxfield,field);maxledger=max(maxledger,ledger)
            old=solved['state'];previous_fine=solved['subcell_ionized']
        rows.append(dict(seed=seed,n=n,refinement=f,angles=k,dose=dose,periodic=periodic,horizon=horizon,maximum_field_L1_over_emitted=maxfield,maximum_source_ledger_L1_over_emitted=maxledger))
    mixed_rows=[]
    for seed in range(60):
        rng=np.random.default_rng(18421+seed)
        n=3;f=(2,4,8)[seed%3]
        cap=np.exp(rng.uniform(-3,3,(n,)*3));cap[rng.random(cap.shape)<.1]=0
        source=np.exp(rng.uniform(-3,3,cap.shape));source[rng.random(cap.shape)<.3]=0
        source*=((.3,.8,1.2)[seed%3]*cap.sum()/source.sum())
        angles=rng.choice([24,48,96,192],size=cap.shape)
        horizon=(.1,1.,8.)[(seed//3)%3]*n
        states={False:None,True:None};worst=0.;source_worst=0.
        for step in range(2):
            emitted=source if step==0 else np.roll(source,1,axis=1)*.4
            solved={enabled:refined.solve(emitted,cap,subcells_per_axis=f,angle_counts=angles,
                                         maximum_distance=horizon,periodic=seed%2==0,
                                         use_local_templates=enabled,initial_state=states[enabled],
                                         return_dense=True) for enabled in (False,True)}
            scale=emitted.sum(dtype=np.longdouble)
            field=float(np.abs(solved[False]['subcell_ionized']-solved[True]['subcell_ionized']).sum(dtype=np.longdouble)/cap.sum(dtype=np.longdouble))
            ledger=max(float(np.abs(solved[False][name]-solved[True][name]).sum(dtype=np.longdouble)/scale) for name in ('source_absorbed','source_escaped','source_retained'))
            if max(field,ledger)>2e-10:raise AssertionError(('mixed',seed,step,field,ledger))
            worst=max(worst,field);source_worst=max(source_worst,ledger)
            states={enabled:r['state'] for enabled,r in solved.items()}
        mixed_rows.append(dict(seed=seed,refinement=f,horizon=horizon,periodic=seed%2==0,
                               maximum_template_field_L1_over_atoms=worst,
                               maximum_template_source_ledger_L1_over_emitted=source_worst))
    current={str(p.relative_to(HERE.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    report=dict(scope='Independent eager-point reference; two successive static increments with changed source positions; log-uniform densities/sources and vacuum cells',cases=cases,solves=2*cases,seconds=time.perf_counter()-started,source_hashes=hashes,code_unchanged=current==hashes,maximum_field_L1_over_emitted=max(r['maximum_field_L1_over_emitted'] for r in rows),maximum_source_ledger_L1_over_emitted=max(r['maximum_source_ledger_L1_over_emitted'] for r in rows),rows=rows, mixed_angle_template_cases=len(mixed_rows), mixed_angle_template_solves=4*len(mixed_rows), maximum_template_field_L1_over_atoms=max(r['maximum_template_field_L1_over_atoms'] for r in mixed_rows), maximum_template_source_ledger_L1_over_emitted=max(r['maximum_template_source_ledger_L1_over_emitted'] for r in mixed_rows),mixed_angle_rows=mixed_rows)
    Path(output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','source_hashes','mixed_angle_rows')}))
    if current!=hashes:raise RuntimeError('Code changed during audit; rerun stable revision')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=HERE/'audit_results.json');parser.add_argument('--cases',type=int,default=80);args=parser.parse_args();audit(args.output,args.cases)
