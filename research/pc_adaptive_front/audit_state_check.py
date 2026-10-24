"""Quantify why a mixed subcell state must survive time-step boundaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'pc_refinement'))
from front_subcells import solve


def run(output):
    capacity = np.ones((24,)*3)
    source = np.zeros_like(capacity)
    source[10, 12, 12], source[15, 12, 12] = 120., 45.
    options = dict(min_angles=1536, max_angles=1536, maximum_distance=10., parent_skip=True)
    one = solve(source, capacity, **options)
    first = solve(source/2, capacity, **options)
    preserved = solve(source/2, capacity,
                      initial_subcell_ionized=first['subcell_absorbed'], **options)
    averaged = solve(source/2, capacity, initial_ionized=first['absorbed'], **options)
    full_persistent = first['absorbed']+preserved['absorbed']
    full_averaged = first['absorbed']+averaged['absorbed']
    coarse_first = first['absorbed']
    fine_first = first['subcell_absorbed']
    reconstructed = coarse_first.repeat(2,0).repeat(2,1).repeat(2,2)/8
    mixed = np.abs(fine_first-reconstructed)>1e-14
    mixed_parents = mixed.reshape(24,2,24,2,24,2).any(axis=(1,3,5))
    q = float(source.sum())
    report = {
        'scope':'Fixed-grid, fixed-direction opaque-cell time subdivision; no cosmological evolution',
        'angles_per_source':1536,
        'source_photons':q,
        'first_half_mixed_parents':int(mixed_parents.sum()),
        'persistent_L1_over_Q':float(abs(full_persistent-one['absorbed']).sum()/q),
        'averaged_state_L1_over_Q':float(abs(full_averaged-one['absorbed']).sum()/q),
        'averaged_state_maximum_parent_ionization_error':float(abs(full_averaged-one['absorbed']).max()),
        'one_step_unabsorbed':one['unabsorbed'],
        'persistent_steps_unabsorbed':first['unabsorbed']+preserved['unabsorbed'],
        'averaged_steps_unabsorbed':first['unabsorbed']+averaged['unabsorbed'],
        'note':'Both continuations preserve photons; coarse averaging loses the location of remaining neutral atoms.',
    }
    assert report['persistent_L1_over_Q']<1e-10
    assert report['averaged_state_L1_over_Q']>1e-4
    output.write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE/'audit_state_check.json')
    args=parser.parse_args()
    print(json.dumps(run(args.output),indent=2))
