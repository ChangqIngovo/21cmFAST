"""Tests for matched-input reporting and native snapshot validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from research.pc_analytic.compare import (
    _completed_snapshot,
    coarsen_snapshot,
    independent_budget,
    load_native_snapshot,
    pair_statistics,
    run_c2ray_cached,
    validate_physics,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(tmp_path: Path) -> Path:
    source = np.full((4, 4, 4), 0.25)
    capacity = np.ones_like(source)
    np.savez(
        tmp_path / "source.npz",
        source=source,
        capacity=capacity,
        native_es_xhi=1 - source,
        native_brightness=20 * (1 - source),
    )
    (tmp_path / "native.h5").write_bytes(
        b"fixture native h5: no native runtime required"
    )
    identity = {
        "inputs": {
            "astro_options": {
                "RECOMB_MODEL": "none",
                "CELL_RECOMB": False,
                "USE_TS_FLUCT": False,
                "USE_MCGS": False,
                "USE_REIONIZATION_PHOTOHEATING_FEEDBACK": False,
                "PHOTON_CONS_TYPE": "no-photoncons",
            },
            "matter_options": {"SOURCE_MODEL": "L-INTEGRAL"},
            "simulation_options": {"BOX_LEN": 40.0},
            "cosmo_params": {"hlittle": 0.6766, "OMb": 0.04897468, "Y_He": 0.24},
        },
        "redshift": 8.0,
    }
    record = {"redshift": 8.0, "seed": 111}
    arrays = {
        name: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        for name, value in {"source": source, "capacity": capacity}.items()
    }
    (tmp_path / "native.json").write_text(
        json.dumps(
            {
                "identity": identity,
                "array_sha256": arrays,
                "source": {"cumulative_budget_not_rate": True},
            }
        )
    )
    files = {
        name: _sha(tmp_path / name)
        for name in ("source.npz", "native.json", "native.h5")
    }
    (tmp_path / "receipt.json").write_text(
        json.dumps({"files": files, "identity": identity, "record": record})
    )
    return tmp_path


def test_snapshot_validates_unchanged_native_arrays(tmp_path: Path) -> None:
    """Verify snapshot validates unchanged native arrays."""
    data = load_native_snapshot(_snapshot(tmp_path), brightness="none")
    assert data["source"].shape == (4, 4, 4)
    assert data["metadata"]["source_per_atom"] == 0.25
    assert data["metadata"]["native_h5_reextracted"] is False
    assert data["metadata"]["nH_mean_comoving_cm3"] > 0


def test_tampered_snapshot_is_rejected(tmp_path: Path) -> None:
    """Verify tampered snapshot is rejected."""
    directory = _snapshot(tmp_path)
    with (directory / "source.npz").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_native_snapshot(directory, brightness="none")


def test_physics_mismatch_is_rejected(tmp_path: Path) -> None:
    """Verify physics mismatch is rejected."""
    directory = _snapshot(tmp_path)
    identity = json.loads((directory / "receipt.json").read_text())["identity"]
    identity["inputs"]["astro_options"]["RECOMB_MODEL"] = "inhomogeneous"
    with pytest.raises(ValueError, match="RECOMB_MODEL"):
        validate_physics(identity["inputs"])


def test_remaining_and_escape_are_not_counted_as_absorption() -> None:
    """Verify remaining and escape are not counted as absorption."""
    source = np.ones((4, 4, 4))
    capacity = np.ones_like(source)
    xhi = np.full_like(source, 0.25)
    budget = independent_budget(source, capacity, xhi, remaining=8.0, escaped=8.0)
    assert budget["absorbed_photons_from_field"] == 48.0
    assert budget["relative_photon_residual"] == 0.0
    missing = independent_budget(source, capacity, xhi, remaining=None, escaped=None)
    assert missing["relative_photon_residual"] is None
    assert missing["unaccounted_photons"] == 16.0


def test_spectra_zero_reference_masks_and_field_difference() -> None:
    """Verify spectra zero reference masks and field difference."""
    capacity = np.ones((4, 4, 4))
    fields = {
        "analytic": np.full_like(capacity, 0.7),
        "script": np.full_like(capacity, 0.6),
    }
    report = pair_statistics(fields, capacity, 40.0, brightness=None)
    pair = report["pairs"]["analytic_vs_script"]
    assert pair["MAE"] == pytest.approx(0.1)
    assert pair["same_mean_neutral_fraction"] is False
    assert all(x is None for x in report["spectra_xHI"]["ratio_analytic"])
    assert report["spectra_21cm"] is None


def test_explicit_coarsening_preserves_physical_counts(tmp_path: Path) -> None:
    """Verify explicit coarsening preserves physical counts."""
    directory = _snapshot(tmp_path)
    fine = load_native_snapshot(directory, brightness="none")
    coarse = load_native_snapshot(directory, brightness="none", coarsen_to=2)
    for name in ("source", "capacity"):
        before = fine[name].sum() * fine["metadata"]["atoms_per_count_unit"]
        after = coarse[name].sum() * coarse["metadata"]["atoms_per_count_unit"]
        assert after == pytest.approx(before, rel=2e-15)
    assert coarse["metadata"]["coarsening_factor"] == 2
    assert fine["source"].shape == (4, 4, 4)


def test_coarse_es_is_atom_weighted() -> None:
    """Verify coarse es is atom weighted."""
    capacity = np.arange(64).reshape(4, 4, 4) + 1.0
    xhi = np.arange(64).reshape(4, 4, 4) / 64.0
    fields = {
        "source": np.ones_like(capacity),
        "capacity": capacity,
        "native_es_xhi": xhi,
        "native_brightness": capacity * xhi,
    }
    coarse = coarsen_snapshot(fields, 2)
    assert np.sum(coarse["capacity"] * coarse["native_es_xhi"]) * 8 == pytest.approx(
        np.sum(capacity * xhi)
    )
    with pytest.raises(ValueError, match="coarsen_to"):
        coarsen_snapshot(fields, 3)


def test_unemitted_budget_is_not_photon_loss() -> None:
    """Keep incomplete source injection separate from photons in transit."""
    source = np.ones((4, 4, 4))
    capacity = np.ones_like(source)
    xhi = np.full_like(source, 0.75)
    budget = independent_budget(
        source, capacity, xhi, remaining=8.0, escaped=8.0, unemitted=32.0
    )
    assert budget["emitted_photons"] == 32.0
    assert budget["source_photon_budget"] == 64.0
    assert budget["relative_photon_residual"] == 0.0
    with pytest.raises(ValueError, match="unemitted"):
        independent_budget(
            source, capacity, xhi, remaining=0.0, escaped=0.0, unemitted=65.0
        )


def _c2ray_cache_fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple:
    """Provide a deterministic mock of the reference IO contract for cache tests."""
    from research.pc_analytic import c2ray_reference as backend

    calls = []
    monkeypatch.setattr(
        backend,
        "build",
        lambda *args, **kwargs: {"identity": {"test_compiler": "fixture"}},
    )

    def fake_run(
        source,
        capacity,
        *,
        workdir,
        threads=1,
        upstream=None,
        box_len=40.0,
        redshift=8.0,
        nH_mean_comoving_cm3=2e-7,
        sigma_cm2=2.7e-18,
        exposure_steps=32,
    ):
        calls.append({"threads": threads, "exposure_steps": exposure_steps})
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / "output.bin").write_bytes(source.tobytes() + capacity.tobytes())
        return {
            "ionized": source.copy(),
            "xHI": 1 - source / capacity,
            "audit": {
                "complete": True,
                "elapsed_seconds": 4.0,
                "output_sha256": _sha(workdir / "output.bin"),
            },
            "provenance": {"unit_test_fixture": True},
            "workdir": str(workdir),
        }

    monkeypatch.setattr(backend, "run", fake_run)
    source = np.full((4, 4, 4), 0.2)
    capacity = np.ones_like(source)
    args = {
        "physical": {
            "box_len": 40.0,
            "redshift": 8.0,
            "nH_mean_comoving_cm3": 2e-7,
            "sigma_cm2": 2.7e-18,
        },
        "settings": {"upstream": str(tmp_path / "upstream")},
        "threads": 4,
        "workdir": tmp_path / "local",
        "cache_directory": tmp_path / "cache",
    }
    return source, capacity, args, calls


def test_c2ray_cache_matches_arrays_settings_and_reports_timing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reuse exact native inputs and retain distinct cached execution timing."""
    source, capacity, args, calls = _c2ray_cache_fixture(monkeypatch, tmp_path)
    first = run_c2ray_cached(source, capacity, **args)
    again = run_c2ray_cached(source, capacity, **args)
    assert len(calls) == 1
    assert first["audit"]["cache_hit"] is False
    assert again["audit"]["cache_hit"] is True
    assert again["audit"]["elapsed_seconds"] == 0.0
    assert again["audit"]["cached_backend_elapsed_seconds"] == 4.0
    assert np.array_equal(first["ionized"], again["ionized"])
    explicit_default = dict(args)
    explicit_default["settings"] = args["settings"] | {"exposure_steps": 32}
    run_c2ray_cached(source, capacity, **explicit_default)
    assert len(calls) == 1
    run_c2ray_cached(source * 2, capacity, **args)
    assert len(calls) == 2
    changed = dict(args)
    changed["settings"] = args["settings"] | {"exposure_steps": 64}
    run_c2ray_cached(source, capacity, **changed)
    assert len(calls) == 3


@pytest.mark.parametrize("target", ["fields.npz", "metadata.json", "native_output"])
def test_c2ray_cache_rejects_tampering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, target: str
) -> None:
    """Reject altered cached arrays, metadata, and original native output bytes."""
    source, capacity, args, _ = _c2ray_cache_fixture(monkeypatch, tmp_path)
    result = run_c2ray_cached(source, capacity, **args)
    entry = Path(args["cache_directory"]) / result["audit"]["cache_key"]
    receipt = json.loads((entry / "receipt.json").read_text())
    path = (
        Path(receipt["native_output"]) if target == "native_output" else entry / target
    )
    with path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        run_c2ray_cached(source, capacity, **args)


def test_completed_snapshot_resume_verifies_receipts(tmp_path: Path) -> None:
    """Reuse an unchanged completed case and reject changed field bytes."""
    (tmp_path / "fields.npz").write_bytes(b"unit fixture field bytes")
    row = {"complete": True, "three_methods_completed": True, "seed": 222}
    (tmp_path / "metrics.json").write_text(json.dumps(row))
    receipt = {
        "identity_sha256": "same-run",
        "files": {
            name: _sha(tmp_path / name) for name in ("fields.npz", "metrics.json")
        },
    }
    (tmp_path / "receipt.json").write_text(json.dumps(receipt))
    assert _completed_snapshot(tmp_path, "same-run", require_c2ray=True) == row
    with pytest.raises(ValueError, match="identity"):
        _completed_snapshot(tmp_path, "other-run", require_c2ray=True)
    (tmp_path / "fields.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        _completed_snapshot(tmp_path, "same-run", require_c2ray=True)


def test_incomplete_snapshot_is_not_reused(tmp_path: Path) -> None:
    """A partial two-method run does not satisfy a required three-way comparison."""
    row = {"complete": True, "three_methods_completed": False}
    (tmp_path / "metrics.json").write_text(json.dumps(row))
    (tmp_path / "receipt.json").write_text(
        json.dumps(
            {
                "identity_sha256": "run",
                "files": {"metrics.json": _sha(tmp_path / "metrics.json")},
            }
        )
    )
    assert _completed_snapshot(tmp_path, "run", require_c2ray=True) is None
    assert _completed_snapshot(tmp_path, "run", require_c2ray=False) == row


def test_direct_c2ray_power_reference_is_not_limited_by_zero_script_power() -> None:
    """Measure analytic/C²-Ray auto and cross power independently of SCRIPT."""
    capacity = np.ones((4, 4, 4))
    wave = np.cos(np.arange(4)[:, None, None] * 2 * np.pi / 4) * capacity
    fields = {
        "analytic": 0.5 + 0.2 * wave,
        "c2ray": 0.5 + 0.1 * wave,
        "script": np.full_like(capacity, 0.5),
    }
    report = pair_statistics(
        fields,
        capacity,
        40.0,
        brightness={key: 10 * val for key, val in fields.items()},
    )
    assert all(val is None for val in report["spectra_21cm"]["ratio_analytic"])
    direct = report["spectra_21cm_vs_c2ray"]
    ratios = [val for val in direct["ratio_analytic"] if val is not None]
    correlations = [val for val in direct["r_analytic"] if val is not None]
    assert ratios == pytest.approx([4.0], rel=1e-12)
    assert correlations == pytest.approx([1.0], abs=1e-12)
