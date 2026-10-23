"""Measured time, spatial and closure validation for coupled_opacity.py."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from coupled_opacity import instantaneous_flux, solve, solve_history
from validation_recombination import opacity_reference


def serializable(value):
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def temporal_cases():
    definitions = [
        ('recombining_screen', [1], [1], [1], [1], [[0]], [1], 2),
        ('shared_screen', [1, 3, 3], [1, 0, 0], [1, .4, .4], [1, 4, 4], [[0, 1], [0, 2]], [1, 2], 1.5),
        ('opposite_beams', [1, 2], [.8, .2], [.4, .1], [3, 4], [[0, 1], [1, 0]], [2, 1], 1),
        ('optically_thick_screen', [1], [1], [1], [1000], [[0]], [1], 2),
    ]
    output = []
    for name, *args in definitions:
        reference = opacity_reference(*args, method='Radau', rtol=1e-11, atol=1e-13)
        cross = opacity_reference(*args, method='DOP853', rtol=2e-12, atol=2e-14)
        fixed = []
        fixed_counts = (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096) if name == 'optically_thick_screen' else (8, 16, 32, 64, 128)
        for count in fixed_counts:
            result = solve(*args, steps=count)
            err = float(np.abs(result['x'] - reference['x']).max())
            row = dict(steps=count, x=result['x'], max_x_error=err,
                       absorption_l1_per_emitted=float(np.abs(result['absorbed'] - reference['absorbed']).sum() / result['emitted']),
                       audit=result['audit'])
            if fixed:
                row['observed_order'] = math.log2(fixed[-1]['max_x_error'] / err)
            fixed.append(row)
        adaptive = []
        for tolerance in (1e-4, 1e-6, 1e-8):
            result = solve(*args, rtol=tolerance, atol=tolerance * .001)
            adaptive.append(dict(rtol=tolerance, x=result['x'],
                max_x_error=float(np.abs(result['x'] - reference['x']).max()),
                absorption_l1_per_emitted=float(np.abs(result['absorbed'] - reference['absorbed']).sum() / result['emitted']),
                escaped_by_beam=result['escaped_by_beam'], audit=result['audit']))
        print('Validated temporal case:', name, flush=True)
        output.append(dict(name=name, reference=reference,
            independent_ode_crosscheck_max=float(np.abs(np.asarray(reference['x']) - cross['x']).max()),
            fixed=fixed, adaptive=adaptive))
    return output


def spatial_cases():
    # For one pure-photo beam, every prefix of a constant sigma/n slab has an
    # analytic neutral column. It gives exact cell averages, not centre samples.
    pure = []
    tau_total, fluence = 20., .7
    for count in (2, 4, 8, 16, 32, 64):
        atoms = np.full(count, 1 / count)
        result = solve(atoms, np.zeros(count), np.zeros(count), np.full(count, tau_total / count),
                       [list(range(count))], [1], fluence, steps=1)
        coord = np.arange(count + 1) / count
        with np.errstate(divide='ignore'):
            column = np.logaddexp(0, np.log(np.expm1(tau_total * coord)) - tau_total * fluence) / tau_total
        exact = 1 - np.diff(column) * count
        pure.append(dict(cells=count, max_cell_average_error=float(np.abs(result['x'] - exact).max()),
                         ionized_atoms=float(np.dot(atoms, result['x'])),
                         escaped=result['escaped_by_beam'][0], audit=result['audit']))

    # Independent vectorized method-of-lines slab reference with simultaneous
    # photo+R, then compare volume averages as grid spacing is halved.
    fine_n, tau_total, alpha, duration = 512, 20., .7, 1.5
    def fine_slab(n, rtol=2e-11, atol=2e-13):
        def rhs(_t, x):
            optical = tau_total / n * (1 - x)
            before = np.r_[0, np.cumsum(optical[:-1])]
            incoming = np.exp(-before)
            absorption = incoming * (-np.expm1(-optical))
            return absorption * n - alpha * x * x
        solution = solve_ivp(rhs, (0, duration), np.zeros(n), method='DOP853', rtol=rtol, atol=atol)
        assert solution.success
        return solution.y[:, -1], solution.nfev
    fine, evaluations = fine_slab(fine_n)
    finer, eval2 = fine_slab(2 * fine_n)
    fine_crosscheck = float(np.abs(fine - finer.reshape(fine_n, 2).mean(axis=1)).sum() / fine_n)
    rec = []
    for count in (4, 8, 16, 32, 64, 128):
        atoms = np.full(count, 1 / count)
        result = solve(atoms, np.zeros(count), np.full(count, alpha), np.full(count, tau_total / count),
                       [list(range(count))], [1], duration, rtol=1e-8, atol=1e-11)
        average = finer.reshape(count, 2 * fine_n // count).mean(axis=1)
        error = float(np.abs(result['x'] - average).sum() / count)
        row = dict(cells=count, l1_ionized_atoms_error=error,
                   mean_x=float(result['x'].mean()), audit=result['audit'])
        if rec:
            row['observed_spatial_order'] = math.log2(rec[-1]['l1_ionized_atoms_error'] / error)
        rec.append(row)
    return dict(pure_photo_exact_cell_averages=pure,
                recombining_slab=dict(tau_total=tau_total, alpha_n=alpha, duration=duration,
                    reference_cells=2 * fine_n, reference_method='independent simultaneous DOP853 method-of-lines',
                    reference_nfev=eval2, preceding_reference_nfev=evaluations,
                    reference_grid_change_l1=fine_crosscheck, rows=rec))


def history_and_order():
    cells = ([1, 3, 3], [1, 0, 0], [1, .4, .4], [1, 4, 4])
    paths, rates = [[0, 1], [0, 2]], [1, 2]
    rows = []
    for count in (8, 16, 32, 64, 128):
        a = solve(*cells, paths, rates, 1.5, steps=count)
        b = solve(*cells, paths[::-1], rates[::-1], 1.5, steps=count)
        rows.append(dict(steps=count, max_order_bias=float(np.abs(a['x'] - b['x']).max())))
    steps = [(.2, [1, 2]), (.5, [0, 0]), (.3, [2, .5])]
    h = solve_history(*cells, paths, steps, rtol=1e-8, atol=1e-11)
    reference_x = cells[1]
    reference_abs = np.zeros(3)
    for dt, rate in steps:
        ref = opacity_reference(cells[0], reference_x, cells[2], cells[3], paths, rate, dt)
        reference_x = ref['x']
        reference_abs += ref['absorbed']
    flux = instantaneous_flux([1, 2], [1, 1], [3, 4], [[0, 1], [1, 0]], [2, 1])
    return dict(order_bias=rows, source_on_off_on=dict(
        segments=steps, result=h, reference_x=reference_x,
        max_x_error=float(np.abs(h['x'] - reference_x).max()),
        absorption_l1_per_emitted=float(np.abs(h['absorbed'] - reference_abs).sum() / h['emitted'])),
        fully_ionized_Gamma=flux,
        Gamma_definition='sum_b F_in,b*(-expm1(-tau_b*(1-x)))/(N_H*(1-x)); neutral->0 limit sum_b F_in,b*tau_b/N_H')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_suffix('.json'))
    args = parser.parse_args()
    here = Path(__file__).parent
    result = dict(scope='Fixed-temperature pure-H finite-opacity acyclic-characteristic research solver; not native Qin20 coupled evolution',
                  code_sha256={name: hashlib.sha256((here / name).read_bytes()).hexdigest()
                               for name in ('coupled_opacity.py', 'validation_coupled_opacity.py')},
                  temporal=temporal_cases(), spatial=spatial_cases(), history=history_and_order())
    args.output.write_text(json.dumps(serializable(result), indent=2) + '\n')
    print(args.output)
    for case in result['temporal']:
        print(case['name'], 'adaptive fine xerror', case['adaptive'][-1]['max_x_error'],
              'fixed final order', case['fixed'][-1]['observed_order'])
    print('spatial', [(r['cells'], r['l1_ionized_atoms_error']) for r in result['spatial']['recombining_slab']['rows']])

if __name__ == '__main__':
    main()
