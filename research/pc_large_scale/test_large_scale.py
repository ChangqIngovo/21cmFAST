"""Independent statistical and orchestration contracts for production outputs."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pc_statistics import coarsen_counts, es_map, es_threshold, input_band_statistics, match_mass, spectrum, total
from run_comparison import checked_receipt, finish, task_list, validate_config, match_pc_mass


def test_coarse_inputs_preserve_both_global_and_local_counts():
    a = np.arange(12**3, dtype=float).reshape(12, 12, 12)
    coarse = coarsen_counts(a, 4)
    assert total(coarse) == total(a)
    assert coarse[1, 2, 3] == a[3:6, 6:9, 9:12].sum()
    with pytest.raises(ValueError):
        coarsen_counts(a, 5)


@pytest.mark.parametrize('n', [9, 10])
def test_HI_bias_and_full_fft_mode_count_against_independent_fft(n):
    rng = np.random.default_rng(716)
    c = rng.uniform(.8, 1.2, (n,)*3)
    a = .4*c  # spatially constant neutral fraction => exactly unit HI bias
    L = 100.
    edges = np.linspace(0, np.pi*n/L*(1+1e-10), 12)
    row = spectrum(c, a, L, edges, [.05, .09], 8.)
    assert row['Q_mass'] == pytest.approx(.4)
    assert row['k0_band']['b_HI_squared'] == pytest.approx(1, abs=2e-14)
    k = 2*np.pi*np.fft.fftfreq(n, d=L/n)
    km = np.sqrt(k[:, None, None]**2+k[None, :, None]**2+k[None, None, :]**2)
    ph = abs(np.fft.fftn((c-a)/(c-a).mean()-1))**2*L**3/n**6
    for s in row['shells']:
        m = (km>0)&(km<=np.pi*n/L*(1+1e-12))&(km>=s['k_min'])&(km<s['k_max'])
        assert s['full_fft_modes'] == np.count_nonzero(m)
        assert s['P_HI'] == pytest.approx(ph[m].mean(), rel=2e-13)


def test_fractional_HI_power_is_not_raw_neutral_fraction_power():
    n = 16
    wave = np.cos(2*np.pi*np.arange(n)/n)[:,None,None]*np.ones((1,n,n))
    c = 1+.1*wave
    # neutral density = 0.5 * (1 + 0.3*cos); matter amplitude=0.1 => b^2=9.
    a = c-.5*(1+.3*wave)
    out = spectrum(c, a, 100., np.linspace(.001,.5,20), [.05,.07], 8.)
    assert out['k0_band']['b_HI_squared'] == pytest.approx(9.)
    inputs = input_band_statistics(1+.3*wave, c, 100., [.05,.07])
    assert inputs['source_bias_squared'] == pytest.approx(9.)
    assert inputs['P_source_matter_cross']/inputs['P_matter'] == pytest.approx(3.)


def test_uniform_sources_recover_partial_ionization_and_match_mass():
    s, c = np.full((8,)*3, .2), np.ones((8,)*3)
    threshold, _ = es_threshold(s, c, 4)
    np.testing.assert_allclose(es_map(s,c,threshold,1), s)
    a, match = match_mass(s, c, threshold, .7, tolerance=1e-8)
    assert match['matched']
    assert total(a)/total(c) == pytest.approx(.7, abs=1e-8)


def test_discontinuous_ES_target_is_reported_without_blending_fields():
    s, c = np.full((4,)*3, .1), np.ones((4,)*3)
    a, match = match_mass(s, c, np.full(s.shape,.5), .3, tolerance=1e-5)
    assert not match['matched']
    assert not match['interpolated_fields']
    np.testing.assert_array_equal(a, es_map(s,c,np.full(s.shape,.5),match['amplitude']))
    assert match['absolute_error'] > .2


def test_completed_receipt_rejects_corrupt_output(tmp_path):
    p = tmp_path/'result.json'
    p.write_text('{}')
    finish(tmp_path,['result.json'])
    assert checked_receipt(tmp_path)
    p.write_text('{"changed":true}')
    with pytest.raises(RuntimeError,match='changed'):
        checked_receipt(tmp_path)


def test_production_plan_has_independent_cases_and_physical_k0():
    c = json.loads((Path(__file__).parent/'default.json').read_text())
    validate_config(c)
    tasks = task_list(c)
    assert len(tasks) == 112
    assert len({t['directory'] for t in tasks}) == 112
    assert all(t['seed']==c['seeds'][0] for t in tasks if t['variant']!='baseline')
    c['grids'] = [100, 30]
    with pytest.raises(ValueError):
        validate_config(c)


def test_bias_undefined_for_fully_ionized_box():
    c = np.ones((8,)*3)
    with pytest.raises(ValueError,match='undefined'):
        spectrum(c,c,100.,[.01,.1,.3],[.04,.06],8.)


def test_pc_matching_keeps_original_budget_map_and_does_not_blend():
    s, c = np.ones((4,)*3)*.2, np.ones((4,)*3)
    def retained_solver(s, c, **kwargs):
        return {'absorbed': np.minimum(.75*s,c)}
    original, matched, audit = match_pc_mass(retained_solver,s,c,.6,{},1e-6)
    assert total(original['absorbed'])/total(c) == pytest.approx(.45)
    assert total(matched['absorbed'])/total(c) == pytest.approx(.6, abs=1e-6)
    assert audit['matched']
    np.testing.assert_array_equal(matched['absorbed'], retained_solver(s*audit['amplitude'],c)['absorbed'])
