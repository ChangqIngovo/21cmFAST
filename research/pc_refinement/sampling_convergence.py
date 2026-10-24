"""Increase angular reference for the matched native cavity comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from sampling_native_compare import error
import hybrid


def run(path,angles,output,reference_output):
    with np.load(path) as data:fields={k:data[k] for k in data.files}
    hybrid.library()
    horizon=fields['source'].shape[0]/2
    result=hybrid.solve(fields['source'],fields['capacity'],fields['initial'],
                        min_angles=angles,max_angles=angles,maximum_distance=horizon)
    q=fields['source'].sum();ref=result['absorbed'];rho=fields['density']
    report={'scope':'same frozen native32 cavity; reference-angle refinement only',
            'angles':angles,'audit':result['audit'],'comparisons':{}}
    hashes={k+'_sha256':hashlib.sha256(np.ascontiguousarray(fields[k],dtype=np.float64).tobytes()).hexdigest()
            for k in ('source','capacity','initial')}
    report.update(input_hashes=hashes,maximum_distance=horizon,
                  fixture_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
    for name in ['reference768','baseline','probe_rms4','probe_rms8']:
        report['comparisons'][name]=error(fields[name],ref,rho,q)
    output.write_text(json.dumps(report,indent=2)+'\n')
    if reference_output:
        np.savez_compressed(reference_output,reference=ref,angles=angles,
                            maximum_distance=horizon,**hashes)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fields',type=Path,required=True)
    p.add_argument('--angles',type=int,default=1536)
    p.add_argument('--output',type=Path,default=Path(__file__).with_suffix('.json'))
    p.add_argument('--reference-output',type=Path)
    a=p.parse_args();run(a.fields,a.angles,a.output,a.reference_output)
