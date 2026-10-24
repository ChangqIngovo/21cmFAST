from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import upgraded
from speed_core import module


@pytest.mark.parametrize("profile", upgraded.PROFILES)
def test_local_absorption_and_transparent_reservoir(profile):
    c = np.ones((4,)*3)
    s = .1*c
    result = upgraded.solve(s, c, .8*c, profile=profile)
    np.testing.assert_allclose(result["absorbed"], s, rtol=2e-14, atol=0)
    result = upgraded.solve(s, c, c, profile=profile)
    assert result["absorbed"].sum() == 0
    np.testing.assert_array_equal(result["source_retained"], s)
    assert result["audit"]["solver_profile"] == profile


def test_balanced_partial_background_uses_remaining_atoms():
    c = np.ones((20,)*3)
    s = np.zeros_like(c)
    s[10, 10, 10] = .01*4*np.pi*4**3/3
    old = .99*c
    original = upgraded.solve(s, c, old, profile="compact")
    balanced = upgraded.solve(s, c, old)
    reference = module("original").solve(s, c, old, min_angles=24576, max_angles=24576)
    error = lambda x: abs(x["absorbed"]-reference["absorbed"]).sum()/s.sum()
    assert error(balanced) < .15*error(original)
    assert balanced["audit"]["effective_density"] == "remaining"


def test_incremental_compact_preserves_wide_kernel_map():
    c = np.ones((16,)*3)
    xyz = np.indices(c.shape)-8
    old = ((xyz*xyz).sum(axis=0) < 3**2).astype(float)
    s = np.zeros_like(c)
    s[8, 8, 8] = 25
    compact = upgraded.solve(s, c, old, profile="incremental", maximum_distance=7)
    wide = upgraded.sampling_solver.solve(s, c, old, maximum_distance=7,
        rays_per_surface_cell=16, effective_density="remaining", dominant_source_fraction=.01)
    for key in ("absorbed", "source_absorbed", "source_retained", "source_escaped", "angle_counts"):
        np.testing.assert_array_equal(compact[key], wide[key])


def test_invalid_profile_rejected():
    with pytest.raises(ValueError):
        upgraded.solve(np.zeros((2,)*3), np.ones((2,)*3), profile="magic")


@pytest.mark.parametrize("profile", ["balanced", "incremental"])
def test_constant_rates_fixed_directions_preserve_time_subdivision(profile):
    c = np.ones((16,)*3)
    s = np.zeros_like(c)
    s[6, 8, 8], s[10, 8, 8] = 150, 65
    options = dict(profile=profile, min_angles=768, max_angles=768,
                   maximum_distance=7)
    full = upgraded.solve(s, c, **options)
    old = np.zeros_like(c)
    retained = 0.
    for _ in range(4):
        step = upgraded.solve(s/4, c, old, **options)
        old += step["absorbed"]
        retained += step["unabsorbed"]
    np.testing.assert_allclose(retained, full["unabsorbed"], rtol=1e-10, atol=2e-12)
    np.testing.assert_allclose(old, full["absorbed"], rtol=1e-10, atol=2e-12)


@pytest.mark.parametrize("profile", upgraded.PROFILES)
def test_units_do_not_change_maps(profile):
    c = np.ones((8,)*3)
    s = np.zeros_like(c)
    s[4, 4, 4] = 20
    base = upgraded.solve(s, c, .1*c, profile=profile, maximum_distance=3)
    scale = 1e-12
    small = upgraded.solve(s*scale, c*scale, .1*c*scale, profile=profile, maximum_distance=3)
    np.testing.assert_allclose(small["absorbed"]/scale, base["absorbed"], rtol=5e-11, atol=2e-12)
