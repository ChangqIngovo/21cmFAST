"""Tests for the optional continuous-ray transport discretization."""

from pathlib import Path

import numpy as np
import pytest

from research.c2ray_teacher.conservative import angular_rule, patch_transport
from research.c2ray_teacher.geometry_audit import uniform_budget
from research.c2ray_teacher.probe_compare import cell_average
from research.c2ray_teacher.runner import validate_config


def test_solid_angles_cover_sphere_and_are_rotation_balanced() -> None:
    """Cube-face pixels tile the sky; their weights are geometrical, not fitted."""
    for order in (1, 2, 7, 16):
        directions, weights = angular_rule(order)
        assert len(weights) == 6 * order**2
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1, atol=2e-15)
        assert np.all(weights > 0)
        assert weights.sum() == pytest.approx(1, abs=4e-15)
        np.testing.assert_allclose(weights @ directions, 0, atol=1e-15)
        np.testing.assert_allclose(
            np.einsum("n,ni,nj->ij", weights, directions, directions),
            np.eye(3) / 3,
            atol=2e-15,
        )


def test_transport_patch_rejects_unrecognized_source(tmp_path: Path) -> None:
    """A numerical patch must never silently target a different upstream."""
    (tmp_path / "files_for_3D").mkdir()
    (tmp_path / "doric.f90").write_text("changed source\n")
    (tmp_path / "files_for_3D/evolve_source.F90").write_text("changed source\n")
    with pytest.raises(ValueError, match="anchor"):
        patch_transport(tmp_path, tmp_path / "numerics.patch")


@pytest.mark.parametrize("order", [0, -1, 1.5, True])
def test_invalid_angular_order(order: int) -> None:
    """Reject invalid angular sizes before allocating directions."""
    with pytest.raises(ValueError, match="positive integer"):
        angular_rule(order)


def test_exact_columns_do_not_remove_native_geometric_deficit() -> None:
    """A uniform exact-column case isolates source and external-cell geometry."""
    row = uniform_budget(1.0)
    assert -0.096 < row["native_geometry_residual"] < -0.094
    assert -0.034 < row["exact_source_cell_only_residual"] < -0.032
    assert row["unabsorbed_tail_upper_bound"] < 1e-8


def test_transport_is_explicit_and_native_stays_default() -> None:
    """Reject unknown modes and keep existing preparations reproducible."""
    assert validate_config({})["transport"] == "native"
    c = validate_config({"transport": "conservative_rays", "angular_order": 32})
    assert c["angular_order"] == 32
    with pytest.raises(ValueError, match="transport"):
        validate_config({"transport": "renormalize"})
    with pytest.raises(ValueError, match="angular_order"):
        validate_config({"angular_order": 0})


def test_incommensurate_projection_preserves_volume_integral() -> None:
    """Conservative overlap works when fine/coarse sizes have no integer ratio."""
    x = np.zeros((9, 9, 9))
    x[2, 3, 4] = 1
    y = cell_average(x, 5)
    assert np.min(y) >= 0
    assert y.mean() == pytest.approx(x.mean(), rel=2e-14)
    np.testing.assert_allclose(cell_average(np.ones_like(x), 5), 1, atol=2e-15)
