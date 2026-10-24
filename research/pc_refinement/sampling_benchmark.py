"""Reproduce probe-allocation accuracy and cost on cavity increments."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import sampling_solver as new
import hybrid


def stats(result,reference,q):
    return {**result['audit'],'L1_over_emitted':float(np.abs(result['absorbed']-reference).sum()/q)}


def run(native=None):
    hybrid.library();new.probe_library()
    report={'scope':'same static gas/shared event core; angular allocation only; no cosmological history',
            'cases':[]}
    c=np.ones((28,)*3);xyz=np.indices(c.shape)
    for label,q in [('tiny-cavity-increment',.01),('growing-cavity',4*np.pi*3**3/3)]:
        old=(sum((a-14)**2 for a in xyz)<7**2).astype(float)
        s=np.zeros_like(c);s[14,14,14]=q
        reference=hybrid.solve(s,c,old,min_angles=65536,max_angles=65536,maximum_distance=14)
        row={'case':label,'reference':reference['audit'],'variants':[]}
        baseline=hybrid.solve(s,c,old,maximum_distance=14)
        row['variants'].append({'name':'baseline',**stats(baseline,reference['absorbed'],q)})
        for statistic in ['rms','p90','max']:
            for surface in [4.,8.]:
                result=new.solve(s,c,old,maximum_distance=14,statistic=statistic,rays_per_surface_cell=surface)
                row['variants'].append({'name':f'probe-{statistic}-surface{surface}',**stats(result,reference['absorbed'],q)})
        report['cases'].append(row)
    if native:
        with np.load(native) as data:s,c=data['source'],data['capacity']
        q=float(s.sum()/2);s=s/2
        first=hybrid.solve(s,c,maximum_distance=16)
        old=first['absorbed'];remaining=c-old
        levels,counts,_,_=hybrid.estimate_angles(s,c,remaining,maximum_distance=16)
        row={'case':'native32-second-half','first_half':first['audit'],
             'baseline_predicted_all_source_angles':int(counts[levels[s>0]].sum()),
             'baseline_not_executed':True,'variants':[]}
        reference=hybrid.solve(s,c,old,min_angles=768,max_angles=768,maximum_distance=16)
        row['reference']=reference['audit']
        for statistic in ['rms','p90','max']:
            for surface in [4.,8.]:
                result=new.solve(s,c,old,maximum_distance=16,statistic=statistic,rays_per_surface_cell=surface)
                row['variants'].append({'name':f'probe-{statistic}-surface{surface}',**stats(result,reference['absorbed'],q)})
                print(row['variants'][-1]['name'],row['variants'][-1]['seconds'],row['variants'][-1]['L1_over_emitted'],flush=True)
        report['cases'].append(row)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--native',type=Path)
    p.add_argument('--output',type=Path,default=Path(__file__).with_suffix('.json'))
    a=p.parse_args();r=run(a.native);a.output.write_text(json.dumps(r,indent=2)+'\n')
