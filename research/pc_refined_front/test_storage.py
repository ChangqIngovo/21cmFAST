"""Storage selection must preserve the discrete transport and carried state."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import refined


@pytest.mark.parametrize('templates', [False, True])
def test_reservation_switch_preserves_fields_and_resets_between_calls(templates):
    rng = np.random.default_rng(7129)
    capacity = rng.uniform(.2, 2., (4,)*3)
    source = rng.uniform(0, 1., capacity.shape)
    source *= .9*capacity.sum()/source.sum()
    options = dict(subcells_per_axis=4, angles=96, maximum_distance=12.,
                   use_local_templates=templates, return_dense=True)
    a = refined.solve(source, capacity, contiguous_ray_bytes=1 << 30, **options)
    b = refined.solve(source, capacity, contiguous_ray_bytes=0, **options)
    c = refined.solve(source, capacity, contiguous_ray_bytes=1 << 30, **options)
    assert a['audit']['ray_storage'] == c['audit']['ray_storage'] == 'contiguous'
    assert b['audit']['ray_storage'] == 'blocks'
    assert min(a['audit']['rays_materialized'], b['audit']['rays_materialized']) > 0
    for other in (b, c):
        for key in ('absorbed', 'source_absorbed', 'source_escaped',
                    'source_retained', 'subcell_absorbed', 'subcell_ionized'):
            np.testing.assert_array_equal(a[key], other[key])
        for key in ('ionized', 'cleared_bits', 'partial_ids', 'partial_ionized'):
            np.testing.assert_array_equal(a['state'][key], other['state'][key])


@pytest.mark.parametrize('budget', [-1, 1.5, True, 1 << 64])
def test_invalid_reservation_threshold(budget):
    with pytest.raises(ValueError, match='contiguous_ray_bytes'):
        refined.solve(np.ones((2,)*3), np.ones((2,)*3),
                      contiguous_ray_bytes=budget)
