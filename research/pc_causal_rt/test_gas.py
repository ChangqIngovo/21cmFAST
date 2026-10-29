"""Independent conservation and advection limits for native mass-packet motion."""
import numpy as np
import pytest

from research.pc_causal_rt.gas import deposit, remap_ions


def _lattice(n):
    return np.stack(np.meshgrid(*(np.arange(n, dtype=float),)*3, indexing="ij"), axis=-1).reshape(-1, 3)


def _reference(mass, positions, n):
    grid = np.zeros((n,)*3)
    for m, p in zip(mass, positions, strict=True):
        base = np.floor(p).astype(int)
        f = p - base
        for corner in np.ndindex(2, 2, 2):
            index = tuple((base + corner) % n)
            weight = np.prod([f[a] if corner[a] else 1-f[a] for a in range(3)])
            grid[index] += m * weight
    return grid


def test_periodic_cic_matches_independent_reference():
    rng = np.random.default_rng(12)
    mass, positions = rng.uniform(0, 2, 500), rng.uniform(-8, 12, (500, 3))
    result = deposit(mass, positions, 4)
    np.testing.assert_allclose(result, _reference(mass, positions, 4), rtol=3e-15, atol=3e-15)
    assert result.sum() == pytest.approx(mass.sum(), rel=3e-15)


def test_zero_motion_is_exact_identity_not_gather_scatter_diffusion():
    rng = np.random.default_rng(123)
    mass, positions = rng.uniform(.1, 2, 600), rng.uniform(0, 6, (600, 3))
    cap = deposit(mass, positions, 6)
    ions = rng.random(cap.shape) * cap
    result = remap_ions(ions, mass, positions, positions.copy(), 6)
    np.testing.assert_array_equal(result["ionized"], ions)
    np.testing.assert_array_equal(result["capacity"], cap)
    assert result["audit"]["identity_for_zero_displacement"]


def test_integer_translation_on_node_lattice_is_exact_shift():
    n = 5
    positions = _lattice(n)
    mass = np.linspace(.5, 1.5, n**3)
    ions = mass.reshape((n,)*3) * np.random.default_rng(2).random((n,)*3)
    result = remap_ions(ions, mass, positions, positions + [1, -2, 0], n)
    np.testing.assert_array_equal(result["ionized"], np.roll(ions, (1, -2, 0), axis=(0, 1, 2)))
    np.testing.assert_array_equal(result["capacity"], np.roll(mass.reshape((n,)*3), (1, -2, 0), axis=(0, 1, 2)))


def test_translating_broad_clouds_moves_interior_species_not_just_density_edges():
    n = 8
    p = _lattice(n)+np.array([.2, .3, .4])
    mass = np.random.default_rng(19).uniform(.7, 1.3, len(p))
    capacity = deposit(mass, p, n, cloud_width=3.)
    ions = capacity*np.random.default_rng(20).random(capacity.shape)
    result = remap_ions(ions, mass, p, p+[1, -1, 0], n, cloud_width=3.)
    np.testing.assert_allclose(result["ionized"], np.roll(ions, (1, -1, 0), axis=(0, 1, 2)),
                               rtol=5e-15, atol=5e-15)
    np.testing.assert_allclose(result["capacity"], np.roll(capacity, (1, -1, 0), axis=(0, 1, 2)),
                               rtol=5e-15, atol=5e-15)


@pytest.mark.parametrize("fraction", [0., .37, 1.])
def test_deforming_gas_preserves_uniform_species_fraction_and_counts(fraction):
    rng = np.random.default_rng(33)
    positions = rng.uniform(0, 8, (1300, 3))
    mass = rng.uniform(.01, 2, len(positions))
    after = positions + rng.normal(0, .3, positions.shape)
    cap = deposit(mass, positions, 8)
    result = remap_ions(fraction * cap, mass, positions, after, 8)
    np.testing.assert_allclose(result["ionized"], fraction * result["capacity"], rtol=1e-14, atol=1e-14)
    assert result["audit"]["ion_relative_residual"] < 2e-15
    assert result["audit"]["mass_relative_residual"] < 2e-15


def test_nonuniform_species_remains_bounded_and_conserved_through_many_moves():
    rng = np.random.default_rng(45)
    positions = _lattice(8) + .25
    mass = rng.uniform(.1, 3, len(positions))
    ions = rng.random((8,)*3) * deposit(mass, positions, 8)
    initial = ions.sum()
    for _ in range(20):
        after = positions + rng.normal(0, .07, positions.shape)
        result = remap_ions(ions, mass, positions, after, 8)
        ions, positions = result["ionized"], after
        assert np.min(ions) >= 0
        assert np.max(ions - result["capacity"]) <= 1e-13
    assert ions.sum() == pytest.approx(initial, rel=3e-14)


def test_refinement_count_units_and_sparse_vacuum():
    mass, positions = np.array([2.]), np.array([[1.2, 1.3, 1.4]])
    coarse = deposit(mass, positions, 4)
    fine = deposit(8*mass, 2*positions, 8)
    assert coarse.sum() == pytest.approx(fine.sum()/8)
    result = remap_ions(.4*fine, 8*mass, 2*positions, 2*positions+.2, 8)
    assert np.all(result["ionized"][result["capacity"] == 0] == 0)
    assert result["ionized"].sum() == pytest.approx(.4*fine.sum())


@pytest.mark.parametrize("factor", [1, 2, 4])
def test_fixed_cloud_refinement_exactly_preserves_coarse_density(factor):
    rng = np.random.default_rng(93)
    positions = rng.uniform(-4, 8, (300, 3))
    mass = rng.uniform(.2, 2, len(positions))
    coarse = deposit(mass, positions, 4)
    fine_positions = positions*factor+(factor-1)/2
    fine = deposit(mass*factor**3, fine_positions, 4*factor, cloud_width=factor)
    restricted = fine.reshape(4, factor, 4, factor, 4, factor).mean(axis=(1, 3, 5))
    np.testing.assert_allclose(restricted, coarse, rtol=6e-15, atol=6e-15)
    result = remap_ions(.6*fine, mass*factor**3, fine_positions,
                       fine_positions+np.array([.2, -.05, .1])*factor,
                       4*factor, cloud_width=factor)
    np.testing.assert_allclose(result["ionized"], .6*result["capacity"], rtol=6e-15, atol=6e-15)
    assert result["audit"]["ion_relative_residual"] < 2e-15


def test_zero_mass_inventory():
    mass, positions = np.zeros(1), np.array([[.1, .2, .3]])
    result = remap_ions(np.zeros((2,)*3), mass, positions, positions+.4, 2)
    np.testing.assert_array_equal(result["ionized"], 0)
    assert result["audit"]["ion_relative_residual"] == 0


@pytest.mark.parametrize("bad", [np.nan, -1., 1.01])
def test_invalid_species_is_rejected(bad):
    p = _lattice(3)
    ions = np.zeros((3,)*3)
    ions[0, 0, 0] = bad
    with pytest.raises(ValueError):
        remap_ions(ions, np.ones(len(p)), p, p+.1, 3)


def test_smooth_translation_converges_with_grid_refinement():
    errors = []
    for n in (8, 16, 32):
        p = _lattice(n)
        phase = 2*np.pi*np.arange(n)/n
        ions = np.broadcast_to((.5+.2*np.sin(phase))[:, None, None], (n,)*3).copy()
        cap = np.ones(n**3)
        elapsed = 0.
        while elapsed < .2 - 1e-14:
            dt = min(.2/n, .2-elapsed)
            new = p + [dt*n, 0, 0]
            ions = remap_ions(ions, cap, p, new, n)["ionized"]
            p, elapsed = new, elapsed+dt
        exact = np.broadcast_to((.5+.2*np.sin(phase-2*np.pi*.2))[:, None, None], (n,)*3)
        errors.append(float(np.mean(np.abs(ions-exact))))
    assert errors[1] < .65*errors[0]
    assert errors[2] < .65*errors[1]
