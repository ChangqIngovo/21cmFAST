"""Execute the original cavity allocation and probe alternatives on one fixture."""
import argparse
import json
from pathlib import Path
import numpy as np

import sampling_solver as new
import hybrid


def error(a,b,rho,photons):
    difference=(a-b)/rho
    return {'L1_over_emitted':float(np.abs(a-b).sum()/photons),
            'volume_mean_absolute_xHII_error':float(np.mean(np.abs(difference))),
            'maximum_absolute_xHII_error':float(np.max(np.abs(difference))),
            'delta_volume_mean_xHII':float(np.mean(difference))}


def run(path,output,fields):
    with np.load(path) as data:
        s,c=data['source']/2,data['capacity']
        rho=data['density'] if 'density' in data else c
    hybrid.library();new.probe_library()
    first=hybrid.solve(s,c,maximum_distance=16)
    old=first['absorbed']
    levels,counts,_,_=hybrid.estimate_angles(s,c,c-old,maximum_distance=16)
    estimate=int(counts[levels[s>0]].sum())*64
    if estimate>3*1024**3:raise RuntimeError('Predicted baseline ray allocation exceeds 3 GiB gate')
    report={'scope':'frozen native32 second-half; sameinitialstate/source; no recombination evolution',
            'baseline_predicted_ray_bytes':estimate,'models':[]}
    maps={'source':s,'capacity':c,'density':rho,'initial':old}
    reference=hybrid.solve(s,c,old,min_angles=768,max_angles=768,maximum_distance=16)
    maps['reference768']=reference['absorbed']
    report['reference']=reference['audit']
    baseline=hybrid.solve(s,c,old,maximum_distance=16)
    maps['baseline']=baseline['absorbed']
    for name,result in [('baseline',baseline),('probe_rms4',new.solve(s,c,old,maximum_distance=16)),
                         ('probe_rms8',new.solve(s,c,old,maximum_distance=16,rays_per_surface_cell=8.))]:
        maps[name]=result['absorbed']
        row={'name':name,'audit':result['audit'],
             'vs_fixed768':error(result['absorbed'],reference['absorbed'],rho,s.sum()),
             'vs_original_adaptive':error(result['absorbed'],baseline['absorbed'],rho,s.sum())}
        report['models'].append(row)
        print(name,result['audit']['seconds'],row['vs_fixed768'],flush=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    if fields:np.savez_compressed(fields,**maps)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--native',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path(__file__).with_suffix('.json'))
    parser.add_argument('--fields',type=Path)
    args=parser.parse_args();run(args.native,args.output,args.fields)
