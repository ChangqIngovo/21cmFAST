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


def _convergence_fixture(root: Path) -> Path:
    _fixture(root)
    with np.load(root / "history.npz") as archive:
        history = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(history.pop("metadata").item()))
    capacity = history["capacity"]
    times = np.linspace(0, 2, 9)
    pattern = history["rates"][0] / 0.1
    history.update(
        rates=np.stack([(0.08 + 0.32 * t + 0.1 * t**2) * pattern for t in times]),
        times_seconds=times,
        redshifts=10 - 2 * times,
        native_xhi=np.stack(
            [np.full_like(capacity, 1 - 0.45 * t) for t in times]
        ).astype(np.float32),
    )
    metadata["array_sha256"] = {key: array_hash(history[key]) for key in ARRAY_KEYS}
    fixture = root / "history.npz"
    np.savez_compressed(fixture, **history, metadata=np.asarray(json.dumps(metadata)))
    settings = [(384, f, 0.4, 1) for f in (1, 2, 4)]
    settings += [(d, 4, 0.4, 1) for d in (24, 96)]
    settings += [(384, 4, c, 1) for c in (0.2, 0.1)]
    settings += [(384, 4, 0.1, s) for s in (4, 2)]
    cases = []
    for i, (directions, refinement, cfl, stride) in enumerate(settings):
        nodes = np.arange(len(times))
        knots = np.arange(0, len(times), stride)
        coefficients = np.interp(
            times, times[knots], 0.08 + 0.32 * times[knots] + 0.1 * times[knots] ** 2
        )
        rates = coefficients[:, None, None, None] * pattern
        dt = np.diff(times)[:, None, None, None]
        emitted = np.concatenate(
            [
                np.zeros_like(rates[:1]),
                np.cumsum(0.5 * (rates[:-1] + rates[1:]) * dt, axis=0),
            ]
        )
        ionized = np.minimum(0.8 * emitted, capacity)
        setting = {
            "directions": directions,
            "refinement": refinement,
            "cfl": cfl,
            "source_stride": stride,
        }
        fields = {
            "capacity": _refine(capacity, refinement),
            "ionized": _refine(ionized, refinement),
            "radiation": _refine(emitted - ionized, refinement),
            "source_emitted": _refine(emitted, refinement),
            "redshifts": history["redshifts"][nodes],
            "times_seconds": times[nodes],
            "metadata": np.asarray(
                json.dumps(
                    {
                        "setting": setting,
                        "seed": 111,
                        "fixture_sha256": _digest(fixture),
                        "box_len_cMpc": 50.0,
                        "native_snapshot_indices": nodes.tolist(),
                        "source_knot_indices": knots.tolist(),
                    }
                )
            ),
        }
        path = root / f"case{i}.npz"
        np.savez_compressed(path, **fields)
        cases.append(
            dict(
                seed=111,
                **setting,
                path=str(path),
                sha256=_digest(path),
                case_id=f"case{i}",
            )
        )
    _json(
        root / "matrix.json",
        {
            "configuration": {"profile": "convergence"},
            "cases": cases,
            "fixtures": [
                {"seed": 111, "path": str(fixture), "sha256": _digest(fixture)}
            ],
        },
    )
    return root


def test_refinements_align_sampled_source_histories_by_physical_time(
    tmp_path: Path,
) -> None:
    """Sampling changes emission while background and comparison times stay fixed."""
    root = _convergence_fixture(tmp_path / "convergence")
    matrix = json.loads((root / "matrix.json").read_text())
    matrix["cases"].reverse()
    _json(root / "matrix.json", matrix)
    report = build_report(root, make_plot=False)
    case = report["cases"][0]
    assert case["selected_teacher"]["source_stride"] == 1
    assert case["selected_teacher"]["cfl"] == 0.1
    assert case["selected_teacher"]["refinement"] == 4
    refinements = case["refinement_comparisons"]
    assert len(refinements) == 8
    assert {row["kind"] for row in refinements} == {
        "spatial",
        "angular",
        "temporal",
        "source_history",
    }
    for row in refinements:
        keys = ("directions", "refinement", "cfl", "source_stride")
        assert sum(row["coarse"][key] != row["fine"][key] for key in keys) == 1
        assert all(
            len(r["spectra"]["modes"]) == len(r["primary_bins"])
            for r in row["checkpoints"]
        )
    source = next(
        row
        for row in refinements
        if row["kind"] == "source_history" and row["coarse"]["source_stride"] == 4
    )
    assert [row["node_index"] for row in source["checkpoints"]] == list(range(9))
    assert [row["time_seconds"] for row in source["checkpoints"]] == np.linspace(
        0, 2, 9
    ).tolist()
    assert source["checkpoints"][4]["field_MAE"] > 0
    assert report["summary"]["convergence_certified"] is False
    assert (
        case["self_convergence_summary"]["source_history"]["adjacent_comparisons"] == 2
    )


def test_false_subsampled_snapshot_indices_are_rejected(tmp_path: Path) -> None:
    """A matching array index cannot replace matching physical time and redshift."""
    root = _convergence_fixture(tmp_path / "convergence")
    with np.load(root / "case7.npz") as archive:
        fields = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(fields["metadata"].item()))
    metadata["native_snapshot_indices"] = [0, 1, 2]
    fields["metadata"] = np.asarray(json.dumps(metadata))
    _replace_case(root, fields, 7)
    with pytest.raises(ValueError, match="snapshot indices"):
        build_report(root, make_plot=False)


def test_matched_neutral_fraction_interpolates_fields_with_common_amplitude() -> None:
    """Different matching epochs must not inject a redshift-amplitude difference."""
    from research.pc_matched.comparison import _matched_mean_neutral

    capacity = np.ones((8, 8, 8))
    x, _, _ = np.indices(capacity.shape)
    shape = 0.1 * np.cos(2 * np.pi * x / 8)
    history = {
        "capacity": capacity,
        "redshifts": np.array([10.0, 8.0, 6.0]),
        "times_seconds": np.array([0.0, 1.0, 2.0]),
        "metadata": {"box_len_cMpc": 50.0},
    }

    def model(means: list[float], contrasts: tuple[float, ...] = (1, 1, 1)) -> dict:
        return {
            "fields": {
                i: {
                    "ionized": 1 - mean - contrasts[i] * shape,
                    "time_seconds": float(i),
                    "redshift": float(history["redshifts"][i]),
                }
                for i, mean in enumerate(means)
            }
        }

    rows = _matched_mean_neutral(
        model([0.9, 0.5, 0.3], (0, 4 / 3, 1)),
        model([0.8, 0.7, 0.4]),
        history,
        targets=(0.6, 0.2),
    )
    assert rows[0]["available"] is True
    assert rows[0]["field_MAE"] < 1e-14
    assert rows[0]["common_brightness_amplitude_mK"] == 27
    assert (
        rows[0]["comparison_interpolation"]["time_seconds"]
        != rows[0]["reference_interpolation"]["time_seconds"]
    )
    assert rows[0]["spectra"]["ratio_comparison"][0] == pytest.approx(1, abs=1e-14)
    assert rows[0]["is_training_label"] is False
    assert rows[1]["available"] is False
    assert (
        rows[1]["reason"]
        == "Target is not bracketed in both saved histories; no extrapolation"
    )


def test_workflow_source_history_hash_is_independently_verified(tmp_path: Path) -> None:
    """Real workflow hashes agree, while rehashing a false rate payload cannot pass."""
    from research.pc_evolution.native_history import load_history
    from research.pc_matched.workflow import source_history

    root = _convergence_fixture(tmp_path / "convergence")
    fixture = root / "history.npz"
    history = load_history(fixture)
    _, definition = source_history(history, 4, _digest(fixture))
    with np.load(root / "case7.npz") as archive:
        fields = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(fields["metadata"].item()))
    metadata.update(
        source_history=definition,
        source_history_sha256=definition["source_history_sha256"],
    )
    fields["metadata"] = np.asarray(json.dumps(metadata))
    _replace_case(root, fields, 7)
    assert build_report(root, make_plot=False)["all_photon_ledgers_passed"]
    definition["arrays_sha256"]["rates"] = "0" * 64
    payload = {
        key: value
        for key, value in definition.items()
        if key != "source_history_sha256"
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    definition["source_history_sha256"] = digest
    metadata["source_history_sha256"] = digest
    fields["metadata"] = np.asarray(json.dumps(metadata))
    _replace_case(root, fields, 7)
    with pytest.raises(ValueError, match="independently reconstructed rates"):
        build_report(root, make_plot=False)
