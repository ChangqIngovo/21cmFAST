"""Checks for the pinned hydrogen-only C2-Ray comparison adapter."""

import os
from pathlib import Path

import numpy as np
import pytest

from .c2ray_reference import PINNED_FILES, _safe_subbox_size, run, verify_upstream


def test_reject_changed_upstream(tmp_path: Path) -> None:
    """A directory without the pinned upstream bytes is not a reference."""
    with pytest.raises(ValueError, match="Pinned pyC2Ray"):
        verify_upstream(tmp_path)


@pytest.mark.parametrize("sigma", [0.0, -1.0, float("nan")])
def test_reject_invalid_cross_section(sigma: float, tmp_path: Path) -> None:
    """Opacity is explicit and must have physical units and a positive value."""
    a = np.ones((4, 4, 4))
    with pytest.raises(ValueError, match="sigma_cm2"):
        run(
            a,
            a,
            box_len=1.0,
            redshift=8.0,
            nH_mean_comoving_cm3=2e-7,
            sigma_cm2=sigma,
            workdir=tmp_path,
        )


def test_pinned_source_manifest() -> None:
    """The actual transport, chemistry and license are all pinned."""
    assert set(PINNED_FILES) == {
        "LICENSE",
        "src/c2ray/photorates.f90",
        "src/c2ray/raytracing.f90",
        "src/c2ray/chemistry.f90",
    }


@pytest.mark.parametrize(
    ("dimension", "requested", "expected"),
    [(8, 2, 2), (16, 2, 2), (8, 1, 8), (10, 2, 10), (9, 2, 2)],
)
def test_subboxes_reach_both_native_domain_edges(
    dimension: int, requested: int, expected: int
) -> None:
    """The upstream asymmetric even-grid bounds must not lose their last plane."""
    assert _safe_subbox_size(dimension, requested) == expected


@pytest.mark.skipif(
    not os.environ.get("PC_C2RAY_SOURCE"),
    reason="requires the pinned pyC2Ray source and Fortran compiler",
)
def test_actual_native_zero_and_weak_source(tmp_path: Path) -> None:
    """The compiled upstream kernel preserves a dark field and absorbs weak UV."""
    capacity = np.ones((8, 8, 8))
    source = np.zeros_like(capacity)
    common = {
        "box_len": 4.0,
        "redshift": 8.0,
        "nH_mean_comoving_cm3": 2e-7,
        "sigma_cm2": 2.7e-18,
        "exposure_steps": 8,
        "upstream": Path(os.environ["PC_C2RAY_SOURCE"]),
    }
    dark = run(source, capacity, workdir=tmp_path / "dark", **common)
    np.testing.assert_array_equal(dark["ionized"], source)
    source[4, 4, 4] = 0.2
    lit = run(source, capacity, workdir=tmp_path / "lit", **common)
    assert lit["audit"]["complete"]
    assert lit["audit"]["native_transport_algorithm_unchanged"]
    assert 0.19 < lit["ionized"].sum() < 0.21
    assert np.all((lit["ionized"] >= 0) & (lit["ionized"] <= capacity))
    source[1, 1, 1] = 0.1
    two = run(source, capacity, workdir=tmp_path / "two", **common)
    np.testing.assert_allclose(two["ionized"][1, 1, 1], 0.1, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(two["ionized"][4, 4, 4], 0.2, atol=1e-6, rtol=1e-6)
    changed_duration = run(
        source, capacity, workdir=tmp_path / "duration", exposure_seconds=1e12, **common
    )
    np.testing.assert_allclose(
        changed_duration["ionized"], two["ionized"], atol=2e-12, rtol=2e-12
    )
    full_cube = run(
        source,
        capacity,
        workdir=tmp_path / "full_cube",
        subbox_size=8,
        loss_fraction=0.0,
        **common,
    )
    np.testing.assert_allclose(
        full_cube["ionized"], two["ionized"], atol=2e-12, rtol=2e-12
    )
    assert two["audit"]["subbox_size_actual"] == 2
    assert two["audit"]["subbox_loss_fraction_requested"] == 1e-10
