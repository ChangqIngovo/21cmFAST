import numpy as np

from validation_coupled_3d import candidate, fixture, reference


def test_3d_chords_source_history_and_chemistry_converge_together():
    data=fixture(); ref=reference(data)
    a=candidate(data,4); b=candidate(data,32)
    err_a=np.abs(a['x']-ref['x']).max(); err_b=np.abs(b['x']-ref['x']).max()
    assert err_b < err_a / 8
    assert err_b < 1e-4
    assert b['photon_ledger_error'] < 1e-14
    assert b['atom_ledger_max'] < 1e-14
