"""Collect final fixed-input accuracy, coarsening and runtime comparisons."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    names = ['sphere_results.json', 'native_results.json', 'native_pair8.json',
             'native_pair16.json', 'native_convergence.json']
    inputs = {name: json.loads((HERE/name).read_text()) for name in names}
    code_hashes = [report['code_sha256'] for report in inputs.values()]
    if any(h != code_hashes[0] for h in code_hashes):
        raise RuntimeError('Benchmark files do not use one fixed code revision')
    native8 = {}
    for name in ('native_results.json', 'native_pair8.json', 'native_convergence.json'):
        for case in inputs[name]['cases']:
            if case['name'] == 'native8_Q0.9':
                native8[case['angles']] = case
    pairs = []
    for case in inputs['native_pair16.json']['cases']:
        coarse = native8[case['angles']]
        row = dict(angles=case['angles'], measurement_grid=8,
                   changed_inputs='Conservative source and gas coarsening; source centres move with their original grids',
                   factors={})
        for factor, key in ((2, 'adaptive_f2'), (4, 'refined_f4')):
            a, b = case['models'][key], coarse['models'][key]
            ratio = a['statistics']['b_HI_squared']/b['statistics']['b_HI_squared']
            row['factors'][str(factor)] = dict(b2_source16=a['statistics']['b_HI_squared'],
                                               b2_source8=b['statistics']['b_HI_squared'],
                                               source16_over_source8_minus_one=ratio-1)
        row['relative_reduction_of_abs_grid_difference'] = 1-abs(
            row['factors']['4']['source16_over_source8_minus_one'])/abs(
            row['factors']['2']['source16_over_source8_minus_one'])
        pairs.append(row)
    sphere = inputs['sphere_results.json']['cases'][-1]
    e2 = sphere['models']['refined_f2']['vs_integrated_continuum_sphere']['L1_over_emitted']
    errors = {str(f): sphere['models'][f'refined_f{f}']['vs_integrated_continuum_sphere']['L1_over_emitted']
              for f in (2, 4, 8)}
    rows = [row for report in inputs.values() for case in report['cases'] for row in case['models'].values()]
    ledgers = [r['independent_ledger'] for r in rows]
    dense = [r['vs_dense_point_reference'] for r in rows if 'vs_dense_point_reference' in r]
    templates = [r['vs_template_disabled'] for r in rows if 'vs_template_disabled' in r]
    report = dict(
        scope='One frozen native32 realization plus an independent uniform sphere; common 8^3 measurement mesh for native comparisons',
        measurement=dict(box_hMpc=135.6, band_hMpc_inv=[.04, .08], grid=8, full_fft_modes=18),
        input_sha256={name: hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in names},
        code_sha256=code_hashes[0],
        sphere=dict(angles=sphere['angles'], integrated_L1_over_emitted=errors,
                    improvement_f4_over_f2=1-errors['4']/e2,
                    improvement_f8_over_f2=1-errors['8']/e2,
                    oracle=sphere['oracle']),
        original_grid_coarsening=pairs,
        fixed_source_front_refinement=inputs['native_convergence.json']['cases'],
        high_angle_checks=inputs['native_convergence.json']['angular_checks'],
        validation=dict(
            model_case_count=len(rows),
            maximum_relative_global_residual=max(r['relative_count_residual'] for r in ledgers),
            maximum_source_relative_residual=max(r['maximum_source_relative_residual'] for r in ledgers),
            maximum_local_fraction_violation=max(r['maximum_local_fraction_violation'] for r in ledgers),
            maximum_dense_reference_L1_over_emitted=max(r['L1_over_emitted'] for r in dense),
            maximum_template_control_L1_over_emitted=max(r['L1_over_emitted'] for r in templates)),
        cost_scope='Full solver call including Python preparation and returned state, serial kernel, compilation excluded; first cold-template solve and warmed timings are separate; this is not total 21cmFAST runtime')
    (HERE/'benchmark_summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: report[k] for k in ('sphere', 'original_grid_coarsening', 'validation')}, indent=2))


if __name__ == '__main__':
    main()
