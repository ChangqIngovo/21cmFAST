"""Test remaining-capacity angular radius against uniform partial ionization."""
import argparse
import json
from pathlib import Path
import numpy as np
import sampling_solver as new
import hybrid
from sampling_native_compare import error


def run(fields,reference_path,output):
    hybrid.library();new.probe_library()
    report={'scope':'remaining neutral capacity vs total capacity in angular predictor only','uniform_partial':{},'native':[]}
    c=np.ones((24,)*3);old=.99*c;s=np.zeros_like(c);s[12,12,12]=.01*4*np.pi*6**3/3
    ref=hybrid.solve(s,c,old,min_angles=65536,max_angles=65536)
    for mode in ['total','remaining']:
        result=new.solve(s,c,old,effective_density=mode)
        report['uniform_partial'][mode]={'audit':result['audit'],'vs_reference':error(result['absorbed'],ref['absorbed'],c,s.sum())}
    with np.load(fields) as f:data={k:f[k] for k in f.files}
    reference=np.load(reference_path)['reference']
    for surface in [4.,8.,16.]:
        result=new.solve(data['source'],data['capacity'],data['initial'],maximum_distance=16,
                         rays_per_surface_cell=surface,effective_density='remaining')
        row={'surface_factor':surface,'audit':result['audit'],
             'vs_reference3072':error(result['absorbed'],reference,data['density'],data['source'].sum())}
        report['native'].append(row);print(row,flush=True)
        output.write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fields',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path(__file__).with_suffix('.json'))
    a=p.parse_args();run(a.fields,a.reference,a.output)
