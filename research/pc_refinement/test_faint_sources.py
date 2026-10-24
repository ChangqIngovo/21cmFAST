import numpy as np
import faint_sources
import sampling_solver


def test_zero_budget_keeps_exact_quadratures():
    s=np.ones((4,)*3);counts=np.full_like(s,192,dtype=int)
    a=faint_sources.allocate(s,counts,epsilon=0)
    np.testing.assert_array_equal(a['angle_counts'],counts)
    assert a['audit']['L1_over_emitted_bound']==0


def test_selected_photons_obey_requested_budget_and_strong_counts_unchanged():
    rng=np.random.default_rng(372);s=np.exp(rng.normal(size=(8,)*3));counts=rng.choice([24,96,384],size=s.shape)
    a=faint_sources.allocate(s,counts,epsilon=.03)
    assert a['audit']['selected_source_photons']<=.03*s.sum()/2
    np.testing.assert_array_equal(a['angle_counts'][~a['selected']],counts[~a['selected']])
    assert np.all(a['angle_counts'][a['selected']]==24)


def test_relative_map_certificate_and_positive_source_ordering():
    c=np.ones((12,)*3);s=np.zeros_like(c);s[6,6,6]=18;s[6,6,7]=.1;s[6,7,6]=.2;s[9,9,9]=3
    xyz=np.indices(c.shape);old=(sum((a-6)**2 for a in xyz)<3**2).astype(float)
    counts=np.full(c.shape,192,dtype=int)
    before=sampling_solver.solve(s,c,old,angle_count_grid=counts,maximum_distance=6)
    after=faint_sources.solve(s,c,old,full_counts=counts,epsilon=.03,maximum_distance=6)
    delta=float(np.abs(before['absorbed']-after['absorbed']).sum())
    assert delta <= after['audit']['coarsening']['L1_absorption_absolute_bound']+1e-11
    strong=s.copy();strong[after['coarsened_sources']]=0
    common=sampling_solver.solve(strong,c,old,angle_count_grid=counts,maximum_distance=6)
    assert np.min(before['absorbed']-common['absorbed'])>=-1e-12
    assert np.min(after['absorbed']-common['absorbed'])>=-1e-12


def test_equal_score_ties_are_translation_covariant():
    s=np.ones((8,)*3);s[1:3]=.01;counts=np.full(s.shape,192,dtype=int)
    a=faint_sources.allocate(s,counts,epsilon=.3)
    b=faint_sources.allocate(np.roll(s,3,0),np.roll(counts,3,0),epsilon=.3)
    np.testing.assert_array_equal(np.roll(a['angle_counts'],3,0),b['angle_counts'])
