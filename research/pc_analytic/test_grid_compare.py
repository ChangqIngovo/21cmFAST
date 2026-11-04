"""Exercise conservative grid restriction and provenance-aware grid comparisons."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from research.pc_analytic import grid_compare as grid
from research.pc_analytic.compare import _digest, _write_json


def _fixture(root: Path, n: int, *, sigma: float = 1.0) -> Path:
    """Write small self-contained completed outputs with genuine receipts."""
    root.mkdir(parents=True)
    identity = {
        "config": {
            "sigma_cm2": sigma,
            "brightness": "native",
            "analytic": {"directions": 96, "exposure_steps": 32},
            "c2ray": {"tolerance": 1e-6, "timeout": 1800},
            "coarsen_to": n,
        },
        "require_c2ray": True,
        "implementation_sha256": {name: name for name in grid.SCIENTIFIC_FILES},
        "script_provenance": {"commit": "script", "files": {}},
    }
    _write_json(root / "identity.json", identity)
    rng = np.random.default_rng(123)
    capacity8 = rng.uniform(0.5, 1.5, (8, 8, 8))
    source8 = rng.uniform(0.1, 0.4, (8, 8, 8))
    capacity = grid._block_mean(capacity8, n)
    source = grid._block_mean(source8, n)
    xhi8 = rng.uniform(0.1, 0.9, (8, 8, 8))
    field = grid._block_mean(capacity8 * xhi8, n) / capacity
    arrays = {"capacity": capacity, "source": source}
    for method in (*grid.METHODS, "native_es"):
        arrays[f"{method}_xHI"] = field
        arrays[f"{method}_brightness"] = grid._block_mean(capacity8 * xhi8, n)
    meta = {
        "seed": 222,
        "redshift": 7.5,
        "box_len": 256.0,
        "dimension": n,
        "native_dimension": 8,
        "receipt_sha256": "original",
        "files": {"native.h5": "native", "native.json": "json", "source.npz": "npz"},
        "archived_array_sha256": {
            "source": "originalsource",
            "capacity": "originalatoms",
        },
        "inputs": {"physics": "unchanged"},
        "nH_mean_comoving_cm3": 1e-7,
        "atoms_per_count_unit": 512 / n**3,
        "source_convention": "native",
        "count_conversion": "same",
        "density_definition": "same",
        "density_formula": "same",
        "brightness_mode": "native",
    }
    row = {
        "seed": 222,
        "redshift": 7.5,
        "input": meta,
        "complete": True,
        "three_methods_completed": True,
        "budgets": {
            method: {"residual": None if method == "c2ray" else 0.0}
            for method in grid.METHODS
        },
        "audits": {"c2ray": {"complete": True, "photon_residual": None}},
        "c2ray_provenance": {
            "identity": {
                "commit": "c2",
                "compiler_version": "GNU Fortran 12.2",
                "files": {},
                "driver_sha256": "driver",
                "flags": [],
            }
        },
    }
    folder = root / "seed222" / "z7p5"
    folder.mkdir(parents=True)
    np.savez(folder / "fields.npz", **arrays)
    _write_json(folder / "metrics.json", row)
    _write_json(
        folder / "receipt.json",
        {
            "identity_sha256": _digest(root / "identity.json"),
            "files": {
                name: _digest(folder / name) for name in ("metrics.json", "fields.npz")
            },
        },
    )
    report = {
        "identity": identity,
        "complete": True,
        "three_methods_completed": True,
        "snapshots": [row],
        "common_physics": {"recombinations": False},
        "brightness_definition": "native",
    }
    _write_json(root / "comparison_report.json", report)
    (root / "COMPLETED").write_text("complete\n")
    return root


def _rewrite(root: Path, mutate: Callable[[dict], None]) -> None:
    """Apply metadata changes and reseal valid receipts for mismatch tests."""
    report = json.loads((root / "comparison_report.json").read_text())
    mutate(report)
    _write_json(root / "identity.json", report["identity"])
    _write_json(root / "comparison_report.json", report)
    folder = root / "seed222" / "z7p5"
    _write_json(folder / "metrics.json", report["snapshots"][0])
    _write_json(
        folder / "receipt.json",
        {
            "identity_sha256": _digest(root / "identity.json"),
            "files": {
                name: _digest(folder / name) for name in ("metrics.json", "fields.npz")
            },
        },
    )


def test_atomic_restriction_preserves_neutral_atoms() -> None:
    """Use atom weights for xHI and volume weights for brightness."""
    capacity = np.ones((4, 4, 4))
    capacity[::2] = 9
    neutral = np.zeros_like(capacity)
    neutral[::2] = 1
    fields = {"capacity": capacity, "source": np.ones_like(capacity)}
    for method in grid.METHODS:
        fields[f"{method}_xHI"] = neutral
        fields[f"{method}_brightness"] = neutral
    result = grid.restrict_fields(fields, 2)
    np.testing.assert_allclose(result["analytic_xHI"], 0.9, rtol=0, atol=1e-15)
    np.testing.assert_allclose(result["analytic_brightness"], 0.5, rtol=0, atol=1e-15)
    assert np.sum(result["capacity"] * result["analytic_xHI"]) * 8 == pytest.approx(
        np.sum(capacity * neutral)
    )


def test_verified_grid_comparison(tmp_path: Path) -> None:
    """Report identical restricted fields and retain C2-Ray unknown residuals."""
    coarse = _fixture(tmp_path / "coarse", 4)
    fine = _fixture(tmp_path / "fine", 8)
    report = grid.run(coarse, fine, tmp_path / "output", redshifts=[7.5])
    row = report["snapshots"][0]
    assert report["convergence_certified"] is False
    assert row["native_es_restriction_control"]["field"]["MAE"] < 1e-15
    assert (tmp_path / "output" / "COMPLETED").is_file()
    assert row["budgets"]["fine"]["c2ray"]["residual"] is None
    for method in grid.METHODS:
        assert row["self_grid_changes"][method]["field"]["MAE"] < 1e-15
        spectrum = row["self_grid_changes"][method]["spectra_21cm"]
        np.testing.assert_allclose(
            spectrum["ratio_fine_restricted"], 1, rtol=0, atol=1e-13
        )
        np.testing.assert_allclose(spectrum["r_fine_restricted"], 1, rtol=0, atol=1e-13)
        assert max(spectrum["P_difference"]) < 1e-25
        assert (
            spectrum["modes"]
            == row["method_gaps"]["coarse"]["analytic_vs_c2ray"]["spectra_21cm"][
                "modes"
            ]
        )


def test_corrupt_completed_receipt_rejected(tmp_path: Path) -> None:
    """A completion marker does not override corrupted fields."""
    coarse = _fixture(tmp_path / "coarse", 4)
    fine = _fixture(tmp_path / "fine", 8)
    with (fine / "seed222/z7p5/fields.npz").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        grid.run(coarse, fine, tmp_path / "output")


def test_changed_physics_rejected(tmp_path: Path) -> None:
    """Grid studies may not silently change the absorption cross-section."""
    coarse = _fixture(tmp_path / "coarse", 4)
    fine = _fixture(tmp_path / "fine", 8, sigma=2)
    with pytest.raises(ValueError, match="settings"):
        grid.run(coarse, fine, tmp_path / "output")


def test_original_input_change_rejected(tmp_path: Path) -> None:
    """Identical count totals do not excuse different original native arrays."""
    coarse = _fixture(tmp_path / "coarse", 4)
    fine = _fixture(tmp_path / "fine", 8)
    _rewrite(
        fine,
        lambda report: report["snapshots"][0]["input"]["archived_array_sha256"].update(
            source="different"
        ),
    )
    with pytest.raises(ValueError, match="original input"):
        grid.run(coarse, fine, tmp_path / "output")


def test_operational_settings_can_change(tmp_path: Path) -> None:
    """Timeouts, thread counts and new analysis files do not change physics."""
    coarse = _fixture(tmp_path / "coarse", 4)
    fine = _fixture(tmp_path / "fine", 8)

    def mutate(report: dict) -> None:
        report["identity"]["config"]["c2ray"]["timeout"] = 72000
        report["identity"]["config"]["threads"] = 16
        report["identity"]["implementation_sha256"][
            "research/pc_analytic/grid_compare.py"
        ] = "new"

    _rewrite(fine, mutate)
    assert grid.run(coarse, fine, tmp_path / "output")["complete"]


def test_missing_requested_redshift_rejected(tmp_path: Path) -> None:
    """Explicit snapshot selection cannot silently become an intersection."""
    coarse = _fixture(tmp_path / "coarse", 4)
    fine = _fixture(tmp_path / "fine", 8)
    with pytest.raises(ValueError, match="redshift"):
        grid.run(coarse, fine, tmp_path / "output", redshifts=[8.0])


def test_equal_resolution_rejected(tmp_path: Path) -> None:
    """Require genuine nested grid refinement."""
    coarse = _fixture(tmp_path / "coarse", 4)
    fine = _fixture(tmp_path / "fine", 4)
    with pytest.raises(ValueError, match="finer"):
        grid.run(coarse, fine, tmp_path / "output")


def test_changed_kernel_rejected(tmp_path: Path) -> None:
    """Do not label simultaneous scientific code changes as grid sensitivity."""
    coarse = _fixture(tmp_path / "coarse", 4)
    fine = _fixture(tmp_path / "fine", 8)
    _rewrite(
        fine,
        lambda report: report["identity"]["implementation_sha256"].update(
            {grid.SCIENTIFIC_FILES[0]: "changed"}
        ),
    )
    with pytest.raises(ValueError, match="implementation changed"):
        grid.run(coarse, fine, tmp_path / "output")


def test_nonconservative_inputs_rejected(tmp_path: Path) -> None:
    """Verify actual input arrays even when metadata declares identical origins."""
    coarse = _fixture(tmp_path / "coarse", 4)
    fine = _fixture(tmp_path / "fine", 8)
    path = fine / "seed222/z7p5/fields.npz"
    with np.load(path) as archive:
        fields = dict(archive)
    fields["source"] *= 1.001
    np.savez(path, **fields)
    _rewrite(fine, lambda report: None)
    with pytest.raises(ValueError, match="Coarse source"):
        grid.run(coarse, fine, tmp_path / "output")


def test_physical_count_units_rejected(tmp_path: Path) -> None:
    """Grid mean densities alone cannot establish equal photon totals."""
    coarse = _fixture(tmp_path / "coarse", 4)
    fine = _fixture(tmp_path / "fine", 8)
    _rewrite(
        fine,
        lambda report: report["snapshots"][0]["input"].update(atoms_per_count_unit=0.5),
    )
    with pytest.raises(ValueError, match="Physical source totals"):
        grid.run(coarse, fine, tmp_path / "output")


def test_incomplete_method_rejected(tmp_path: Path) -> None:
    """Require all three methods to finish rather than treating absent as zero."""
    coarse = _fixture(tmp_path / "coarse", 4)
    fine = _fixture(tmp_path / "fine", 8)
    _rewrite(
        fine,
        lambda report: report["snapshots"][0].update(three_methods_completed=False),
    )
    with pytest.raises(ValueError, match="Snapshot is incomplete"):
        grid.run(coarse, fine, tmp_path / "output")


def test_zero_photon_input_is_defined(tmp_path: Path) -> None:
    """An initially neutral, unilluminated grid has valid zero source totals."""
    coarse = _fixture(tmp_path / "coarse", 4)
    fine = _fixture(tmp_path / "fine", 8)
    for root in (coarse, fine):
        path = root / "seed222/z7p5/fields.npz"
        with np.load(path) as archive:
            fields = dict(archive)
        fields["source"][:] = 0
        for method in (*grid.METHODS, "native_es"):
            fields[f"{method}_xHI"][:] = 1
            fields[f"{method}_brightness"] = fields["capacity"].copy()
        np.savez(path, **fields)
        _rewrite(root, lambda report: None)
    report = grid.run(coarse, fine, tmp_path / "output")
    audit = report["snapshots"][0]["input_audit"]["source"]
    assert audit["coarse_physical_count"] == 0
    assert audit["fine_physical_count"] == 0
    assert audit["physical_total_relative_difference"] is None


def test_changed_c2_compiler_rejected(tmp_path: Path) -> None:
    """Backend compiler changes must not masquerade as spatial refinement."""
    coarse = _fixture(tmp_path / "coarse", 4)
    fine = _fixture(tmp_path / "fine", 8)
    _rewrite(
        fine,
        lambda report: report["snapshots"][0]["c2ray_provenance"]["identity"].update(
            compiler_version="GNU Fortran 14"
        ),
    )
    with pytest.raises(ValueError, match="compiler version differs"):
        grid.run(coarse, fine, tmp_path / "output")
