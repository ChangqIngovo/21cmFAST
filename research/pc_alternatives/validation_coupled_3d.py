"""Cross-module validation: 3D chords, time-dependent sources and chemistry.

Three oblique beams share heterogeneous fixed cells. The independent reference
integrates the simultaneous Beer--Lambert ODE with continuous linear rates;
the candidate integrates fresh emission per interval, then couples exact beam
flows and recombination. This is a finite-cell closure test, not a Qin20 run or
a claim of convergence to full multi-frequency cosmological RT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from coupled_opacity import solve
from ray_segments import build_paths
from source_history import RateHistory

HERE = Path(__file__).resolve().parent


def fixture():
    n, dx = 8, .125
    xyz = (np.indices((n,)*3) + .5) * dx
    density = 1 + .5 * xyz[0] + .2 * xyz[1] + .4 * (xyz[2] > .5)
    kappa = 12 * density
    origins = np.array([[0, .23, .31], [1, .64, .57], [.22, 0, .43]])
    directions = np.array([[1, .4, .2], [-1, -.4, -.2], [.3, 1, .1]])
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    paths, tau, geometry = build_paths(kappa, origins, directions, cell_width=dx)
    ids = np.unique(np.concatenate(paths)); remap = {int(v): i for i, v in enumerate(ids)}
    paths = [[remap[v] for v in path] for path in paths]
    atoms = density.ravel()[ids] * dx**3
    initial = .12 + .2 * xyz[0].ravel()[ids]
    alpha = .8 * density.ravel()[ids]
    # A declared jump at t=.46; no step is allowed to straddle it silently.
    times = np.array([0., .21, .46, .8])
    left = np.array([[.025, .002, .008], [.06, .012, .003], [.004, .03, .01]])
    right = np.array([[.06, .012, .003], [0., .03, .01], [.012, 0., .02]])
    history = RateHistory(times, left, right, clock='dimensionless fixed-gas test time')
    return dict(atoms=atoms, initial=initial, alpha=alpha, paths=paths, tau=tau,
                history=history, geometry=geometry, ids=ids, density=density,
                origins=origins, directions=directions, dx=dx)


def reference(data, *, method='DOP853'):
    atoms, paths, tau, hist = (data[k] for k in ('atoms', 'paths', 'tau', 'history'))
    n, b = len(atoms), len(paths)
    state = np.r_[data['initial'], np.zeros(2*n+b)]
    for i, (lo, hi) in enumerate(zip(hist.times[:-1], hist.times[1:])):
        # Independent time interpolation and simultaneous flux evaluation.
        def rhs(t, y):
            rates = hist.left[i] + (hist.right[i]-hist.left[i]) * ((t-lo)/(hi-lo))
            absorption = np.zeros(n); outgoing = np.zeros(b)
            x = y[:n]
            for beam, (path, depths) in enumerate(zip(paths, tau)):
                flux = rates[beam]
                for cell, depth in zip(path, depths):
                    loss = flux * (-np.expm1(-depth * (1-x[cell])))
                    absorption[cell] += loss
                    flux -= loss
                outgoing[beam] = flux
            recombined = atoms * data['alpha'] * x*x
            return np.r_[(absorption-recombined)/atoms, absorption, recombined, outgoing]
        result = solve_ivp(rhs, (lo, hi), state, method=method, rtol=2e-11, atol=2e-13)
        if not result.success:
            raise RuntimeError(result.message)
        state = result.y[:, -1]
    return dict(x=state[:n], absorbed=state[n:2*n], recombined=state[2*n:3*n], escaped=state[3*n:])


def candidate(data, subdivisions, *, rtol=2e-8, atol=2e-11):
    atoms, hist = data['atoms'], data['history']
    x = data['initial'].copy()
    absorbed = np.zeros_like(x); rec = np.zeros_like(x); escape = np.zeros(len(data['paths']))
    emitted = np.zeros(len(data['paths'])); flow_calls = 0
    for start, stop in zip(hist.times[:-1], hist.times[1:]):
        nodes = np.linspace(start, stop, subdivisions+1)
        for lo, hi in zip(nodes[:-1], nodes[1:]):
            increment = hist.integrate(lo, hi)
            rates = increment/(hi-lo)
            result = solve(atoms, x, data['alpha'], np.zeros_like(x), data['paths'], rates, hi-lo,
                           path_tau=data['tau'], rtol=rtol, atol=atol)
            x = result['x']; absorbed += result['absorbed']; rec += result['recombinations']
            escape += result['escaped_by_beam']; emitted += increment
            flow_calls += result['audit']['flow_calls']
    return dict(x=x, absorbed=absorbed, recombined=rec, escaped=escape, emitted=emitted,
                flow_calls=flow_calls, photon_ledger_error=float(abs(emitted.sum()-absorbed.sum()-escape.sum())),
                atom_ledger_max=float(np.abs(atoms*(x-data['initial'])-absorbed+rec).max()))


def run():
    data = fixture(); truth = reference(data); alternate = reference(data, method='Radau')
    q = data['history'].integrate(data['history'].times[0], data['history'].times[-1]).sum()
    rows = []
    for count in (1, 2, 4, 8, 16, 32):
        result = candidate(data, count)
        rows.append(dict(subdivisions_per_source_interval=count,
                         max_x_error=float(np.abs(result['x']-truth['x']).max()),
                         atom_l1_per_photon=float(np.sum(data['atoms']*np.abs(result['x']-truth['x']))/q),
                         escape_l1_per_photon=float(np.sum(np.abs(result['escaped']-truth['escaped']))/q),
                         photon_ledger_error=result['photon_ledger_error'],
                         atom_ledger_max=result['atom_ledger_max'], flow_calls=result['flow_calls']))
    # Axis permutation changes grid ordering but not lengths, optical depths or
    # the simultaneous physics. This also detects wrong per-cell tau reuse.
    perm = (2, 0, 1)
    _, depth_rot, _ = build_paths((12*data['density']).transpose(perm),
                                  data['origins'][:, perm], data['directions'][:, perm], cell_width=data['dx'])
    depth_difference = max(np.max(np.abs(np.array(a)-np.array(b))) for a, b in zip(data['tau'], depth_rot))
    return dict(scope=__doc__, cells=len(data['atoms']), beams=len(data['paths']), emitted=float(q),
                shared_cells=sum(sum(cell in p for p in data['paths'])>1 for cell in range(len(data['atoms']))),
                independent_DOP853_Radau_max_x=float(np.abs(truth['x']-alternate['x']).max()),
                axis_permutation_max_tau_difference=float(depth_difference), rows=rows,
                hashes={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                        for name in ('coupled_opacity.py','ray_segments.py','source_history.py',Path(__file__).name)},
                limitations=['fixed heterogeneous gas; no density advection',
                             'prescribed open straight paths; no angular convergence claim',
                             'infinite c, pure H, fixed temperature; not coupled to finite-c packets',
                             'source quadrature refinement is distinct from chemistry splitting tolerance'])


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,default=HERE/'validation_coupled_3d.json')
    args=parser.parse_args(); result=run(); args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
