"""One-history production integrity, model scope and phase restart tests."""
import json
from pathlib import Path

import numpy as np
import pytest

import production
from native_history import ARRAY_KEYS, array_hash, _write_fresh_history


def config():
    return dict(production.DEFAULT_CONFIG, dimension=4, highres_dimension=16,
                box_len=40., intervals=8, z_start=12., z_end=8., reference_z=8.,
                subcells_per_axis=2, angles=24, stride=4, maximum_distance=32.)


def fixture(path, cfg):
    n, t = cfg['dimension'], cfg['intervals']+1
    # A changing two-source field produces nontrivial morphology below overlap.
    rates = np.zeros((t,n,n,n), dtype=np.float64)
    rates[:,1,1,1] = np.linspace(0., 2., t)
    rates[:,2,2,2] = np.linspace(0., 1., t)
    history = dict(redshifts=np.linspace(cfg['z_start'],cfg['z_end'],t),
        times_seconds=np.arange(t,dtype=np.float64), rates=rates,
        capacity=np.ones((n,n,n)), native_xhi=np.ones(rates.shape,dtype=np.float32),
        metadata=dict(schema_version=1,density_evolution='frozen_at_reference_redshift',
            rates_definition='native_n_ion_times_H_over_t_STAR',
            reference_redshift=cfg['reference_z'], dimension=n,
            highres_dimension=cfg['highres_dimension'], box_len_cMpc=cfg['box_len'],
            box_hMpc=cfg['box_len']*.678, hlittle=.678, random_seed=cfg['seed'],
            intervals=cfg['intervals'], redshift_start=cfg['z_start'], redshift_end=cfg['z_end'],
            inputs={'simulation_options':dict(DIM=cfg['highres_dimension'],HII_DIM=n,BOX_LEN=cfg['box_len'])}))
    history['metadata']['array_sha256']={key:array_hash(history[key]) for key in ARRAY_KEYS}
    _write_fresh_history(path,history)
    return history


def test_default_production_is_within_kernel_index_limit():
    c=production.validate_config(production.DEFAULT_CONFIG)
    assert c['dimension']**3*c['angles'] <= production.INTMAX
    assert c['dimension']==128 and c['highres_dimension']==512


@pytest.mark.parametrize('change', [dict(dimension=256,highres_dimension=768,angles=768),
    dict(highres_dimension=15),dict(highres_dimension=18),dict(angles=25),
    dict(seed=2**32),dict(seed=True),dict(maximum_distance=float('inf')),
    dict(stride=3),dict(intervals=7),dict(z_end=13),dict(box_len=-1),dict(reference_z=7)])
def test_invalid_or_unsafe_production_config_rejected(change):
    with pytest.raises(ValueError):production.validate_config({**config(),**change})


def test_complete_history_and_resume_emit_no_extra_photons(tmp_path, monkeypatch):
    cfg=config();path=tmp_path/'fixture.npz';h=fixture(path,cfg);root=tmp_path/'run'
    result=production.run(root=root,config=cfg,fixture=path,threads=1)
    expected=sum(.5*(h['rates'][i]+h['rates'][i+1]) for i in range(cfg['intervals']))
    assert result['all_ledgers_passed'] and result['all_comparison_source_budgets_match']
    assert result['final_Q_mass_PC']==pytest.approx(float(expected.sum()/h['capacity'].sum()))
    assert result['final_PC_ledger']['emitted']==pytest.approx(float(expected.sum()))
    assert result['new_convergence_test_performed'] is False
    assert 'not coupled Qin20' in result['scope']
    assert (root/'COMPLETED').is_file()
    for name in ['history.csv','power_spectra.csv','evolution_comparison.png','evolution_comparison.pdf']:
        assert (root/name).stat().st_size>0
    raw=(root/'pc/summary.json').read_bytes()
    assert 'spectra' not in json.loads(raw)['checkpoints'][-1]
    monkeypatch.setattr(production,'run_history',lambda *a,**k:pytest.fail('PC rerun on complete resume'))
    monkeypatch.setattr(production,'run_controls',lambda *a,**k:pytest.fail('ES rerun on complete resume'))
    assert production.run(root=root,config=cfg,fixture=path,threads=1,resume=True)==result
    assert (root/'pc/summary.json').read_bytes()==raw


def test_fixture_config_mismatch_cannot_silently_change_production(tmp_path):
    cfg=config();path=tmp_path/'fixture.npz';fixture(path,cfg)
    with pytest.raises(ValueError,match='Fixture does not match.*seed'):
        production.run(root=tmp_path/'run',config={**cfg,'seed':2345},fixture=path,threads=1)
    assert not (tmp_path/'run/COMPLETED').exists()


def test_checkpoint_and_config_tampering_rejected_on_resume(tmp_path):
    cfg=config();path=tmp_path/'fixture.npz';fixture(path,cfg);root=tmp_path/'run'
    production.run(root=root,config=cfg,fixture=path,threads=1)
    with pytest.raises(RuntimeError,match='config, code, runtime or fixture changed'):
        production.run(root=root,config={**cfg,'angles':48},fixture=path,threads=1,resume=True)
    checkpoint=next((root/'pc/checkpoints').glob('*.npz'))
    with checkpoint.open('ab') as stream:stream.write(b'tampered')
    with pytest.raises(RuntimeError,match='pc artifact changed'):
        production.run(root=root,config=cfg,fixture=path,threads=1,resume=True)


def test_resume_after_es_interruption_preserves_attempt_and_skips_pc(tmp_path,monkeypatch):
    cfg=config();path=tmp_path/'fixture.npz';fixture(path,cfg);root=tmp_path/'run'
    real=production.run_controls
    def fail(history,output):
        output.mkdir();(output/'partial.txt').write_text('interrupted ES attempt')
        raise RuntimeError('simulated ES interruption')
    monkeypatch.setattr(production,'run_controls',fail)
    with pytest.raises(RuntimeError,match='simulated ES interruption'):
        production.run(root=root,config=cfg,fixture=path,threads=1)
    assert (root/'pc_receipt.json').exists() and not (root/'COMPLETED').exists()
    raw=(root/'pc/summary.json').read_bytes()
    monkeypatch.setattr(production,'run_controls',real)
    monkeypatch.setattr(production,'run_history',lambda *a,**k:pytest.fail('PC should not run twice'))
    result=production.run(root=root,config=cfg,fixture=path,threads=1,resume=True)
    assert result['completed']
    assert (root/'pc/summary.json').read_bytes()==raw
    assert len(list((root/'incomplete').glob('es_*/partial.txt')))==1


def test_resume_after_native_atomic_write_reuses_valid_fixture(tmp_path,monkeypatch):
    cfg=config();root=tmp_path/'run';calls=[]
    def fail_after_install(**kwargs):
        calls.append(kwargs);fixture(kwargs['output'],cfg)
        raise RuntimeError('interrupted after atomic fixture installation')
    monkeypatch.setattr(production,'build_native_history',fail_after_install)
    with pytest.raises(RuntimeError,match='atomic fixture installation'):
        production.run(root=root,config=cfg,threads=1)
    assert not (root/'fixture_receipt.json').exists()
    assert production.run(root=root,config=cfg,threads=1,resume=True)['completed']
    assert len(calls)==1


def test_resume_diagnostics_keeps_completed_nodes(tmp_path,monkeypatch):
    cfg=config();path=tmp_path/'fixture.npz';fixture(path,cfg);root=tmp_path/'run'
    real=production._diagnostic_node;calls=[]
    def fail_on_second(history,pc,es):
        calls.append(pc['node_index'])
        if len(calls)==2:raise RuntimeError('diagnostic interruption')
        return real(history,pc,es)
    monkeypatch.setattr(production,'_diagnostic_node',fail_on_second)
    with pytest.raises(RuntimeError,match='diagnostic interruption'):
        production.run(root=root,config=cfg,fixture=path,threads=1)
    first=(root/'diagnostics/node_00000.json').read_bytes()
    calls.clear()
    def track(history,pc,es):
        calls.append(pc['node_index']);return real(history,pc,es)
    monkeypatch.setattr(production,'_diagnostic_node',track)
    assert production.run(root=root,config=cfg,fixture=path,threads=1,resume=True)['completed']
    assert calls==[4,8]
    assert (root/'diagnostics/node_00000.json').read_bytes()==first
