"""Run independent numerical validations concurrently, with separate logs.

This does not parallelize either transport algorithm and does not launch a
cosmological simulation. Native SFRD extraction has its own fixture-dependent
driver, validation_source_history.py, and is deliberately not assumed here.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
DRIVERS=(
    'validation_angular_accuracy.py',
    'validation_coupled_opacity.py',
    'validation_finite_c.py',
    'validation_coupled_3d.py',
    'accuracy_acceptance.py',
)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--skip-unit-tests',action='store_true')
    args=parser.parse_args()
    if args.workers<1:
        parser.error('--workers must be positive')
    # Never mix fresh evidence with a partly completed previous run.
    args.output_dir.mkdir(parents=True,exist_ok=False)
    out=args.output_dir.resolve()
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    began=time.perf_counter()
    results=[]

    def execute(name, command):
        start=time.perf_counter()
        with (out/(name+'.log')).open('w') as stream:
            result=subprocess.run(command,cwd=HERE,env=env,stdout=stream,stderr=subprocess.STDOUT)
        return dict(name=name,returncode=result.returncode,seconds=time.perf_counter()-start,
                    command=command,log=name+'.log')

    if not args.skip_unit_tests:
        result=execute('pytest',[sys.executable,'-m','pytest','-q',str(HERE)])
        results.append(result)
        print(json.dumps(result),flush=True)
        if result['returncode']:
            (out/'FAILED').write_text('Unit regression failed; numerical drivers not started.\n')
            return 1
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(execute,Path(script).stem,
                            [sys.executable,str(HERE/script),'--output',str(out/(Path(script).stem+'.json'))])
                 for script in DRIVERS]
        for future in as_completed(futures):
            result=future.result(); results.append(result)
            print(json.dumps(result),flush=True)
    manifest=dict(scope=__doc__,python=sys.executable,concurrent_independent_jobs=args.workers,
                  elapsed_seconds=time.perf_counter()-began,results=results,
                  hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in HERE.iterdir() if p.suffix in ('.py','.cpp','.c')},
                  excluded=['Native SFRD fixture extraction and evolving Qin20 feedback',
                            'Full finite-c plus finite-opacity/recombination coupling',
                            'Multi-frequency thermal evolution, He chemistry and gas advection'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    passed=all(r['returncode']==0 for r in results)
    (out/('COMPLETED' if passed else 'FAILED')).write_text('Numerical suite finished.\n')
    return 0 if passed else 1


if __name__=='__main__':
    raise SystemExit(main())
