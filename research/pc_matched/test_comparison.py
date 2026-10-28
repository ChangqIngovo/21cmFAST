"""Independent photon and spectral audits for direct RTE comparisons."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from research.pc_evolution.native_history import ARRAY_KEYS, array_hash
from research.pc_matched.comparison import build_report


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value))


def _refine(value: np.ndarray, refinement: int) -> np.ndarray:
    for axis in range(value.ndim - 3, value.ndim):
        value = np.repeat(value, refinement, axis=axis)
    return value


def _fixture(root: Path) -> Path:
    n = 8
    x, y, _ = np.indices((n, n, n))
    capacity = 1 + 0.2 * np.cos(2 * np.pi * x / n) + 0.07 * np.cos(4 * np.pi * x / n)
    pattern = capacity * (1 + 0.1 * np.cos(2 * np.pi * y / n))
    history = {
        "capacity": capacity.astype(np.float64),
        "rates": np.stack([a * pattern for a in (0.1, 0.7, 1.1)]),
        "redshifts": np.array([10.0, 8.0, 6.0]),
        "times_seconds": np.array([0.0, 1.0, 2.0]),
        "native_xhi": np.stack(
            [np.ones_like(capacity), 0.6 + 0.0 * capacity, 0.05 + 0.0 * capacity]
        ).astype(np.float32),
    }
    history["metadata"] = {
        "schema_version": 1,
        "density_evolution": "frozen_at_reference_redshift",
        "rates_definition": "native_n_ion_times_H_over_t_STAR",
        "reference_redshift": 8.0,
        "random_seed": 111,
        "box_len_cMpc": 50.0,
        "capacity_units": "native effective atom counts per mean-density cell",
        "array_sha256": {name: array_hash(history[name]) for name in ARRAY_KEYS},
    }
    root.mkdir()
    fixture = root / "history.npz"
    np.savez_compressed(
        fixture, **(history | {"metadata": np.asarray(json.dumps(history["metadata"]))})
    )
    emitted = np.stack([np.zeros_like(capacity), 0.4 * pattern, 1.3 * pattern])
    ionized = np.stack([np.zeros_like(capacity), 0.75 * emitted[1], capacity])
    cases = []
    for index, (directions, refinement, cfl) in enumerate(
        ((6, 1, 0.4), (26, 1, 0.4), (26, 2, 0.4), (26, 1, 0.2), (26, 1, 0.1))
    ):
        case_ions = ionized.copy()
        if directions == 6:
            case_ions[1] *= 0.9
        fields = {
            "capacity": _refine(capacity, refinement),
            "ionized": _refine(case_ions, refinement),
            "radiation": _refine(emitted - case_ions, refinement),
            "source_emitted": _refine(emitted, refinement),
            "redshifts": history["redshifts"],
            "times_seconds": history["times_seconds"],
            "metadata": np.asarray(
                json.dumps(
                    {
                        "setting": {
                            "directions": directions,
                            "refinement": refinement,
                            "cfl": cfl,
                        },
                        "seed": 111,
                        "fixture_sha256": _digest(fixture),
                        "box_len_cMpc": 50.0,
                    }
                )
            ),
        }
        path = root / f"case{index}.npz"
        np.savez_compressed(path, **fields)
        cases.append(
            {
                "seed": 111,
                "directions": directions,
                "refinement": refinement,
                "cfl": cfl,
                "path": str(path),
                "sha256": _digest(path),
                "case_id": f"case{index}",
            }
        )
    _json(
        root / "matrix.json",
        {
            "configuration": {},
            "cases": cases,
            "fixtures": [
                {
                    "seed": 111,
                    "path": str(fixture),
                    "sha256": _digest(fixture),
                    "native_es_sidecar": None,
                    "native_es_sidecar_sha256": None,
                }
            ],
        },
    )
    return root


def _replace_case(root: Path, fields: dict, index: int = 2) -> None:
    path = root / f"case{index}.npz"
    np.savez_compressed(path, **fields)
    matrix = json.loads((root / "matrix.json").read_text())
    matrix["cases"][index]["sha256"] = _digest(path)
    _json(root / "matrix.json", matrix)


def test_report_audits_physical_radiation_and_exports_consistent_pairs(
    tmp_path: Path,
) -> None:
    """In-flight photons are valid radiation, and native density is never invented."""
    root = _fixture(tmp_path / "run")
    report = build_report(root, make_plot=False)
    assert report["all_photon_ledgers_passed"] is True
    assert report["convergence_certified"] is False
    assert report["maximum_inflight_fraction"] > 0.1
    case = report["cases"][0]
    assert case["native_es_context"]["power_available"] is False
    assert case["matched_es_comparison"][1]["power_error"]["mean"] > 0.01
    assert case["matched_es_comparison"][-1]["power_error"]["undefined"] > 0
    assert case["matched_es_comparison"][-1]["power_error"]["mean"] is None
    with np.load(root / "pairs" / "seed111.npz", allow_pickle=False) as pairs:
        np.testing.assert_array_equal(pairs["redshifts"], [10, 8, 6])
        assert pairs["xhi_pc"].shape == (3, 8, 8, 8)
        np.testing.assert_allclose(
            pairs["ionized_counts_pc"] + pairs["radiation_counts_pc"],
            pairs["cumulative_photons"],
            rtol=1e-14,
            atol=1e-14,
        )
        metadata = json.loads(str(pairs["metadata"].item()))
        assert metadata["convergence_certified"] is False
        assert metadata["native_es_is_matched_control"] is False
    spatial = next(
        row for row in case["refinement_comparisons"] if row["kind"] == "spatial"
    )
    assert spatial["all_primary_power_bins_passed"] is True


def test_rehashed_photon_loss_fails_independent_global_ledger(tmp_path: Path) -> None:
    """A result checksum does not certify the ion-plus-radiation inventory."""
    root = _fixture(tmp_path / "run")
    with np.load(root / "case2.npz") as archive:
        fields = {key: archive[key].copy() for key in archive.files}
    fields["radiation"][1] *= 0.9
    _replace_case(root, fields)
    with pytest.raises(ValueError, match="photon ledger"):
        build_report(root, make_plot=False)


def test_rehashed_source_change_fails_independent_source_integral(
    tmp_path: Path,
) -> None:
    """An internally balanced artificial budget cannot replace prescribed photons."""
    root = _fixture(tmp_path / "run")
    with np.load(root / "case2.npz") as archive:
        fields = {key: archive[key].copy() for key in archive.files}
    for key in ("source_emitted", "radiation", "ionized"):
        fields[key][1] *= 1.1
    _replace_case(root, fields)
    with pytest.raises(ValueError, match="source history"):
        build_report(root, make_plot=False)


def test_unphysical_initial_inventory_fails_even_if_counts_balance(
    tmp_path: Path,
) -> None:
    """The prescribed neutral, dark initial state is independently enforced."""
    root = _fixture(tmp_path / "run")
    with np.load(root / "case2.npz") as archive:
        fields = {key: archive[key].copy() for key in archive.files}
    fields["ionized"][0] = 0.1
    _replace_case(root, fields)
    with pytest.raises(ValueError, match="initial"):
        build_report(root, make_plot=False)


def _sidecar(root: Path) -> Path:
    with np.load(root / "history.npz", allow_pickle=False) as archive:
        history = {name: archive[name] for name in archive.files}
    original = json.loads(str(history.pop("metadata").item()))
    values = {
        "redshifts": history["redshifts"],
        "times_seconds": history["times_seconds"],
        "xhi_es": history["native_xhi"],
        "radius_es_cMpc": np.zeros_like(history["native_xhi"]),
        "density_es": np.stack([factor * history["capacity"] for factor in (1, 2, 3)]),
    }
    metadata = {
        "kind": "native-es-emulator-features",
        "random_seed": 111,
        "box_len_cMpc": 50.0,
        "history_array_sha256": original["array_sha256"],
        "array_sha256": {name: array_hash(value) for name, value in values.items()},
    }
    destination = root / "history_es.npz"
    np.savez_compressed(
        destination, **values, metadata=np.asarray(json.dumps(metadata))
    )
    matrix = json.loads((root / "matrix.json").read_text())
    matrix["fixtures"][0].update(
        native_es_sidecar=str(destination),
        native_es_sidecar_sha256=_digest(destination),
    )
    _json(root / "matrix.json", matrix)
    return destination


def test_native_spectra_use_verified_evolving_density_sidecar(tmp_path: Path) -> None:
    """Native context uses its own evolving gas, never the fixed comparison gas."""
    root = _fixture(tmp_path / "run")
    _sidecar(root)
    report = build_report(root, make_plot=False)
    native = report["cases"][0]["native_es_context"]
    assert native["power_available"] is True
    from research.pc_field_emulator.evaluation import shell_spectra

    with np.load(root / "pairs/seed111.npz") as pair:
        expected = shell_spectra(
            {
                "teacher": 27
                * np.sqrt(0.9)
                * pair["rho_native_context"][1]
                * pair["xhi_es_native_context"][1]
            },
            50,
        )
    np.testing.assert_allclose(
        native["snapshots"][1]["spectra"]["P_native"], expected["P_teacher"], rtol=1e-14
    )


def test_bad_native_sidecar_fails_instead_of_silent_fallback(tmp_path: Path) -> None:
    """A mismatched native sidecar is not treated as missing context."""
    root = _fixture(tmp_path / "run")
    sidecar = _sidecar(root)
    with np.load(sidecar) as archive:
        values = {name: archive[name] for name in archive.files}
    values["density_es"] *= 1.1
    np.savez_compressed(sidecar, **values)
    with pytest.raises(ValueError, match="sidecar"):
        build_report(root, make_plot=False)


def test_balanced_counts_cannot_hide_recombination_in_no_recombination_run(
    tmp_path: Path,
) -> None:
    """A closing global ledger does not excuse declining ion inventory."""
    root = _fixture(tmp_path / "run")
    with np.load(root / "case2.npz") as archive:
        fields = {key: archive[key].copy() for key in archive.files}
    fields["ionized"][2] = 0.2 * fields["capacity"]
    fields["radiation"][2] = fields["source_emitted"][2] - fields["ionized"][2]
    _replace_case(root, fields)
    with pytest.raises(ValueError, match="Ionized counts decrease"):
        build_report(root, make_plot=False)


def test_balanced_counts_cannot_ionize_more_atoms_than_exist(tmp_path: Path) -> None:
    """Local atom bounds remain independent of global photon closure."""
    root = _fixture(tmp_path / "run")
    with np.load(root / "case2.npz") as archive:
        fields = {key: archive[key].copy() for key in archive.files}
    fields["ionized"][2] = 1.1 * fields["capacity"]
    fields["radiation"][2] = fields["source_emitted"][2] - fields["ionized"][2]
    _replace_case(root, fields)
    with pytest.raises(ValueError, match="exceeds local atom capacity"):
        build_report(root, make_plot=False)
