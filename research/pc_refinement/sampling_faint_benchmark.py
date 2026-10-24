"""Measure certified weak-source coarsening against the saved exact baseline."""
import argparse
import json
from pathlib import Path
import numpy as np
import faint_sources
import hybrid
from sampling_native_compare import error


def run(fields,output):
    with np.load(fields) as f:d={k:f[k] for k in f.files}
    hybrid.library()
    levels,counts,_,_=hybrid.estimate_angles(d['source'],d['capacity'],d['capacity']-d['initial'],maximum_distance=16)
    report={'scope':'weak-quadrature changes relative to identical original strong-source directions',
            'source_fields':str(fields),'rows':[]}
    for epsilon in [.001,.003,.01]:
        r=faint_sources.solve(d['source'],d['capacity'],d['initial'],full_counts=counts[levels],
                              epsilon=epsilon,maximum_distance=16)
        difference=error(r['absorbed'],d['baseline'],d['density'],d['source'].sum())
        assert difference['L1_over_emitted']<=r['audit']['coarsening']['L1_over_emitted_bound']+1e-11
        row={'epsilon':epsilon,'audit':r['audit'],'vs_original':difference}
        report['rows'].append(row);print(row,flush=True)
        output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fields',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path(__file__).with_suffix('.json'))
    a=p.parse_args();run(a.fields,a.output)
