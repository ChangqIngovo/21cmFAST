"""Randomized independent fine-grid comparisons, including sparse carried states."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from test_adversarial import compare_fine_state

HERE=Path(__file__).resolve().parent


def run(output,count=48):
    rows=[]
    start=time.perf_counter()
    for case in range(count):
        rng=np.random.default_rng(31001+case)
        n=[4,6,8][case%3]
        c=np.exp(rng.normal(0,.7,size=(n,)*3))
        source=np.exp(rng.normal(-1,1.7,size=c.shape))
        source*=([.1,.5,1.5,3.][case%4])*c.sum()/source.sum()
        coarse_fraction=rng.uniform(0,.8,size=c.shape)
        coarse_fraction[rng.random(c.shape)<.2]=1
        finecap=c.repeat(2,0).repeat(2,1).repeat(2,2)/8
        fraction=coarse_fraction.repeat(2,0).repeat(2,1).repeat(2,2)
        if case%3:
            mixed_parents=rng.random(c.shape)<.25
            mixed=mixed_parents.repeat(2,0).repeat(2,1).repeat(2,2)
            random_fraction=rng.uniform(0,1,size=finecap.shape)
            random_fraction[rng.random(finecap.shape)<.3]=1
            fraction[mixed]=random_fraction[mixed]
        if case%7==0:
            fraction=1-(1-fraction)*1e-7
        initial=finecap*fraction
        result=compare_fine_state(source,c,initial,angles=24 if case%2 else 96,
                                  periodic=bool(case%2),horizon=[.3,n/2,2*n][case%3])
        rows.append(dict(case=case,seed=31001+case,n=n,periodic=bool(case%2),
                         photons_per_total_atom=float(source.sum()/c.sum()),**result))
    report={'scope':'Random fixed-gas full-eight-subcell versus adaptive comparisons; same angles and initial fine states',
            'cases':rows,'case_count':len(rows),'seconds':time.perf_counter()-start,
            'maximum_L1_fine_over_Q':max(r['L1_fine_over_Q'] for r in rows),
            'maximum_parent_fraction_difference':max(r['maximum_parent_fraction_difference'] for r in rows),
            'code_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (HERE/'adaptive.py',HERE/'adaptive.cpp',Path(__file__),HERE/'test_adversarial.py')}}
    output.write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE/'audit_fuzz.json')
    parser.add_argument('--count',type=int,default=48)
    args=parser.parse_args()
    report=run(args.output,args.count)
    print(json.dumps({k:v for k,v in report.items() if k not in ('cases','code_hashes')},indent=2))
