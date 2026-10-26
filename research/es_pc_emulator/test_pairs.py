"""Exercise source/label pairing, primary accounting and held-out IC integrity."""

import copy
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from research.c2ray_teacher.build import FIELD_NAMES, META_NAMES
from research.es_pc_emulator.pairs import (
    combine_manifests,
    export_pairs,
    file_hash,
    load_dataset,
    load_pair,
)
from research.pc_evolution.native_history import (
    ARRAY_KEYS,
    _write_fresh_history,
    array_hash,
    write_es_features,
)


def pair_fixture(root, *, seed=1234, split="train", refinement=3):
    """Create real native-format output with known asymmetric gas and labels."""
    root = Path(root)
    root.mkdir(parents=True)
    n = 2
    rho = 1 + np.arange(n**3, dtype=np.float64).reshape((n,) * 3) / 10
    history = {
        "redshifts": np.array([12.0, 10.0, 8.0]),
        "times_seconds": np.array([0.0, 1.0, 3.0]),
        "rates": np.broadcast_to(
            np.array([1.0, 3.0, 5.0])[:, None, None, None], (3, n, n, n)
        ).copy(),
        "capacity": rho,
        "native_xhi": np.full((3, n, n, n), 0.8, dtype=np.float32),
        "metadata": {
            "schema_version": 1,
            "density_evolution": "frozen_at_reference_redshift",
            "rates_definition": "native_n_ion_times_H_over_t_STAR",
            "reference_redshift": 8.0,
            "random_seed": seed,
            "initial_hires_density_sha256": hashlib.sha256(
                str(seed).encode()
            ).hexdigest(),
            "box_len_cMpc": 20.0,
            "native_extension_sha256": "compiled-test-runtime",
            "contextual_native_ES": "contextual evolving gas and cumulative native proxy",
        },
    }
    history["metadata"]["array_sha256"] = {
        key: array_hash(history[key]) for key in ARRAY_KEYS
    }
    history["native_xhi"][:, 0, 0, 0] = 0.0
    history["metadata"]["array_sha256"]["native_xhi"] = array_hash(
        history["native_xhi"]
    )
    source, es = root / "history.npz", root / "native_es.npz"
    _write_fresh_history(source, history)
    radius = np.zeros(history["rates"].shape, dtype=np.float32)
    radius[:, 0, 0, 0] = 2.0
    write_es_features(es, history, radius, np.broadcast_to(rho, history["rates"].shape))
    results = root / "teacher"
    (results / "science/native_diagnostics").mkdir(parents=True)
    fine_rho = rho
    for axis in range(3):
        fine_rho = np.repeat(fine_rho, refinement, axis=axis)
    fine_n = n * refinement
    xhii = np.arange(fine_n**3).reshape((fine_n,) * 3) / (fine_n**3 + 1)
    steps, records = [], []
    binary = results / "science/native_output.bin"
    previous = 0.0
    with binary.open("wb") as stream:
        stream.write(b"C2RTOUT1")
        stream.write(struct.pack("<4i", fine_n, 2, len(FIELD_NAMES), len(META_NAMES)))
        for i, (time, redshift) in enumerate(zip([1.0, 3.0], [10.0, 8.0], strict=True)):
            dt = time - previous
            emitted = [16.0, 64.0][i] * 27
            step = {
                "time_start_seconds": previous,
                "time_end_seconds": time,
                "redshift_end": redshift,
                "dt_seconds": dt,
                "emitted_photons": emitted,
                "proper_box_cm": 6.0,
                "redshift_density_evaluation": redshift,
            }
            steps.append(step)
            meta = dict.fromkeys(META_NAMES, 0.0)
            meta.update(
                time_seconds=time,
                dt_seconds=dt,
                emitted_photons=emitted,
                proper_box_cm=6.0,
                redshift_mid=redshift,
                native_photon_loss=emitted * 0.2,
            )
            np.asarray([meta[key] for key in META_NAMES], dtype="<f8").tofile(stream)
            for name in FIELD_NAMES:
                array = (
                    fine_rho
                    if name == "density_total_cm3"
                    else xhii
                    if name == "xHII"
                    else np.full_like(fine_rho, 100.0)
                    if name == "temperature_K"
                    else np.zeros_like(fine_rho)
                )
                np.asarray(array.transpose(2, 1, 0), dtype="<f8").tofile(stream)
            records.append(
                {
                    "schema_version": 1,
                    "step": i + 1,
                    "time_seconds": time,
                    "dt_seconds": dt,
                    "final_transport_iteration": 2,
                    "nominal_emitted_photons": emitted,
                    "table_emitted_photons": emitted,
                    "primary_absorptions_by_species": [
                        emitted * 0.7,
                        emitted * 0.1,
                        0.0,
                    ],
                    "secondary_ionizations_by_species": [emitted * 0.05, 0.0, 0.0],
                    "primary_absorptions": emitted * 0.8,
                    "secondary_ionizations": emitted * 0.05,
                    "boundary_outflow_photons": emitted * 0.2,
                    "native_boundary_outflow_photons": emitted * 0.2,
                    "lls_loss_photons": 0.0,
                    "transport_rate_product_ionizations": emitted * 0.85,
                    "postchemistry_rate_product_ionizations": emitted * 0.85,
                    "transport_fraction_max_abs_change": 0.0,
                    "transport_fraction_max_relative_change": 0.0,
                    "recorded_cell_visits": 8,
                }
            )
            previous = time
    manifest = {
        "dimension": fine_n,
        "native_dimension": n,
        "refinement": refinement,
        "step_count": 2,
        "source_count": n**3,
        "source_positions_sha256": array_hash(((np.argwhere(rho > 0) + 1) * refinement - (refinement - 1) // 2).astype(np.int32)),
        "box_len_cMpc": 20.0,
        "density_reference_redshift": 8.0,
        "native_array_sha256": history["metadata"]["array_sha256"],
        "native_metadata_sha256": hashlib.sha256(
            json.dumps(history["metadata"], sort_keys=True).encode()
        ).hexdigest(),
        "native_history_provenance": {
            key: history["metadata"][key]
            for key in ("random_seed", "initial_hires_density_sha256")
        },
        "transport": "conservative_rays",
        "angular_order": 16,
        "photon_normalization": "native_N_b0_H_plus_He_nuclei",
        "density_evolution": "frozen_native_overdensity_with_cosmological_dilution",
        "source_units": "escaped ionizing photons per second",
        "initial_state": {"xHII": 1e-6},
        "thermal_evolution": True,
        "mean_native_cell_nuclei": 27.0,
        "initial_total_nuclei": float(fine_rho.sum()),
        "helium_number_fraction": 0.08,
        "steps": steps,
        "source_integration": "exact trapezoidal integral",
    }
    summary = {
        "input_manifest": manifest,
        "native_output_sha256": file_hash(binary),
        "validation": {"state_checks_passed": True},
        "timing": {
            "sed_table_photons_over_nominal": 1.0,
            "sed_transport_photons_over_nominal": 1.0,
        },
    }
    (results / "summary.json").write_text(json.dumps(summary))
    (results / "science/native_diagnostics/transport_diagnostics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in records) + "\n"
    )
    (results / "COMPLETED").touch()
    return {
        "history": source,
        "es": es,
        "teacher": results,
        "output": root / "pairs",
        "split": split,
        "expected_seed": seed,
        "fine_xhii": xhii,
        "fine_rho": fine_rho,
    }


def export(fixture):
    """Call the public exporter with fixture paths only."""
    return export_pairs(
        fixture["history"],
        fixture["es"],
        fixture["teacher"],
        fixture["output"],
        fixture["split"],
        fixture["expected_seed"],
    )


def test_pairing_uses_integrated_rates_and_preserves_ionized_atoms(tmp_path):
    """Correct endpoints and density weighting preserve the known source/atom counts."""
    f = pair_fixture(tmp_path / "seed")
    manifest = export(f)
    assert len(manifest["samples"]) == 2
    first = load_pair(manifest["samples"][0])
    last = load_pair(manifest["samples"][1])
    np.testing.assert_array_equal(first["photons"], 2.0)
    np.testing.assert_array_equal(last["photons"], 10.0)
    # Nonuniform labels make an axis reversal or unweighted restriction detectable.
    assert np.sum(last["density"] * (1 - last["xhi_c2ray"])) * 27 == pytest.approx(
        np.sum(f["fine_rho"] * f["fine_xhii"]), rel=2e-14
    )
    assert last["metadata"]["convergence_certified"] is False
    assert last["metadata"]["scope"].startswith("ES-to-C2-Ray")


def test_existing_pair_output_and_sidecar_are_not_overwritten(tmp_path):
    """Completed datasets and exported native features remain immutable."""
    f = pair_fixture(tmp_path / "seed")
    export(f)
    with pytest.raises(FileExistsError):
        export(f)
    from research.pc_evolution.native_history import load_history

    h = load_history(f["history"])
    with pytest.raises(FileExistsError):
        write_es_features(
            f["es"], h, np.zeros_like(h["rates"]), np.ones_like(h["rates"])
        )


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (
            lambda s: s["input_manifest"]["native_array_sha256"].__setitem__(
                "rates", "wrong"
            ),
            "history hashes",
        ),
        (
            lambda s: s["input_manifest"]["native_history_provenance"].__setitem__(
                "random_seed", 77
            ),
            "identity",
        ),
        (
            lambda s: s["input_manifest"].__setitem__("transport", "native"),
            "conservative_rays",
        ),
        (
            lambda s: s["input_manifest"]["steps"][0].__setitem__(
                "time_end_seconds", 2.0
            ),
            "clocks",
        ),
        (
            lambda s: s["validation"].__setitem__("state_checks_passed", False),
            "state checks",
        ),
        (
            lambda s: s["timing"].pop("sed_transport_photons_over_nominal"),
            "active-band",
        ),
    ],
)
def test_wrong_teacher_identity_clock_or_physics_is_rejected(tmp_path, change, match):
    """A completion marker cannot override incompatible physical input identities."""
    f = pair_fixture(tmp_path / "seed")
    path = f["teacher"] / "summary.json"
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match=match):
        export(f)
    assert not f["output"].exists()


def test_missing_or_tampered_radius_sidecar_is_rejected(tmp_path):
    """Radius features must come from the recorded native diagnostic payload."""
    f = pair_fixture(tmp_path / "seed")
    with np.load(f["es"], allow_pickle=False) as archive:
        value = {key: np.array(archive[key]) for key in archive.files}
    value["radius_es_cMpc"][0, 0, 0, 0] += 1
    np.savez_compressed(f["es"], **value)
    with pytest.raises(ValueError, match="sidecar hash"):
        export(f)


def test_incomplete_or_tampered_native_payload_is_rejected(tmp_path):
    """Labels must be read from a completed hash-verified native binary."""
    f = pair_fixture(tmp_path / "seed")
    (f["teacher"] / "COMPLETED").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        export(f)
    (f["teacher"] / "COMPLETED").touch()
    with (f["teacher"] / "science/native_output.bin").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="hash changed"):
        export(f)


def test_primary_photon_closure_is_required_even_for_bounded_labels(tmp_path):
    """Bounded ionization alone cannot admit a transport ledger failure."""
    f = pair_fixture(tmp_path / "seed")
    path = f["teacher"] / "science/native_diagnostics/transport_diagnostics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["primary_absorptions"] *= 0.9
    rows[0]["primary_absorptions_by_species"] = [
        rows[0]["primary_absorptions"],
        0.0,
        0.0,
    ]
    rows[0]["transport_rate_product_ionizations"] = (
        rows[0]["primary_absorptions"] + rows[0]["secondary_ionizations"]
    )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="photon ledger"):
        export(f)


def test_independent_seed_split_roundtrip_and_tampering(tmp_path):
    """A complete independent split remains auditable after dataset assembly."""
    manifests = [
        export(pair_fixture(tmp_path / str(seed), seed=seed, split=split))
        for seed, split in [(11, "train"), (23, "validation"), (47, "test")]
    ]
    path = tmp_path / "dataset.json"
    combined = combine_manifests(manifests, path)
    assert len(load_dataset(path)["samples"]) == 6
    record = combined["samples"][0]
    with Path(record["path"]).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Pair file hash"):
        load_dataset(path)


def test_all_three_splits_are_required(tmp_path):
    """Missing validation or test data cannot silently reduce the experiment."""
    manifest = export(pair_fixture(tmp_path / "seed"))
    with pytest.raises(ValueError, match="nonempty train"):
        combine_manifests([manifest], tmp_path / "dataset.json")


def test_same_seed_cannot_leak_through_another_history_or_geometry(tmp_path):
    """Every realization of an initial-condition seed has exactly one split."""
    train = export(pair_fixture(tmp_path / "train", seed=11))
    validation = export(
        pair_fixture(tmp_path / "validation", seed=11, split="validation")
    )
    test = export(pair_fixture(tmp_path / "test", seed=47, split="test"))
    with pytest.raises(ValueError, match="leaks across"):
        combine_manifests([train, validation, test], tmp_path / "dataset.json")


def test_reused_ic_with_relabelled_seed_is_rejected(tmp_path):
    """Relabeling a record cannot manufacture an independent validation sample."""
    train = export(pair_fixture(tmp_path / "train", seed=11))
    validation = copy.deepcopy(train)
    validation["samples"][0]["seed"] = 23
    validation["samples"][0]["split"] = "validation"
    with pytest.raises(ValueError, match=r"record differs|leaks across"):
        combine_manifests([train, validation], tmp_path / "dataset.json")
