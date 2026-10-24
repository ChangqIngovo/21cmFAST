"""Check the high-sampling probe profile against saved high-angle toy references."""
import argparse
import json
from pathlib import Path
import numpy as np
import sampling_solver as new
import hybrid
from compare_models import make_cases,field_statistics


def run(folder,output):
    hybrid.library();new.probe_library()
    report={'scope':'same saved high-angle reference fields; no new RT oracle','cases':[]}
    for name,case in make_cases(24).items():
        file=folder/f'comparison_{name}.npz'
        if not file.exists():raise FileNotFoundError(f'Missing required reference field {file}')
        with np.load(file) as data:
            assert np.array_equal(data['source'],case.source)
            assert np.array_equal(data['capacity'],case.capacity)
            assert np.array_equal(data['initial'],case.initial)
            reference=data['reference']
        row={'case':name,'variants':[]}
        for label,result in [('original',hybrid.solve(case.source,case.capacity,case.initial,
                                                      periodic=True,maximum_distance=case.maximum_distance)),
                             ('probe_rms16_remaining',new.solve(case.source,case.capacity,case.initial,
                                 periodic=True,maximum_distance=case.maximum_distance,
                                 rays_per_surface_cell=16,effective_density='remaining')),
                             ('probe_rms16_guarded',new.solve(case.source,case.capacity,case.initial,
                                 periodic=True,maximum_distance=case.maximum_distance,
                                 rays_per_surface_cell=16,effective_density='remaining',dominant_source_fraction=.01))]:
            row['variants'].append({'name':label,'audit':result['audit'],
                                     'vs_reference':field_statistics(result['absorbed'],reference,case)})
        report['cases'].append(row)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print([(x['case'],[(v['name'],v['audit']['seconds'],v['vs_reference']['L1_absorption_over_emitted']) for v in x['variants']]) for x in report['cases']])


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--reference-dir',type=Path,default=Path(__file__).resolve().parent.parent/'pc_reconsidered/comparison_fields')
    p.add_argument('--output',type=Path,default=Path(__file__).with_suffix('.json'))
    a=p.parse_args();run(a.reference_dir,a.output)
