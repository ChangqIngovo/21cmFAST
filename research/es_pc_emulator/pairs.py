"""Audit and pair native ES inputs with completed C2-Ray ionization labels.

Native ES uses its original cumulative source proxy and evolving density;
C2-Ray uses the prescribed integrated-rate history and frozen overdensity.
This dataset trains an ES-to-RT emulator, not a measurement of photon-loss
effects alone. It never imposes an invalid photons-equals-HII projection on
the H/He/recombination teacher.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from research.c2ray_teacher.compare import load_run, restrict_fields
from research.c2ray_teacher.runner import read_transport_diagnostics, validate_output
from research.es_pc_emulator.initial_state import (
    initial_state_audit,
    require_initial_state,
    validate_policy,
)
from research.pc_evolution.native_history import array_hash, load_history

SCHEMA = 1
SPLITS = ("train", "validation", "test")
FIELDS = ("xhi_es", "radius_es_cMpc", "density", "photons", "xhi_c2ray")
ES_FIELDS = ("redshifts", "times_seconds", "xhi_es", "radius_es_cMpc", "density_es")


def file_hash(path):
    """Stream a file's SHA256 without reading the complete payload into memory."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(value):
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind not in "US":
        raise ValueError("Archive metadata must be a scalar JSON string")
    result = json.loads(str(raw.item()))
    if not isinstance(result, dict):
        raise TypeError("Archive metadata must be an object")
    return result


def _atomic(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink()


def _write_json(path, value):
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    _atomic(path, lambda stream: stream.write(payload))


def _positive_seed(value):
    if type(value) is not int or value <= 0:
        raise ValueError("Initial-condition seed must be a positive integer")
    return value


def _identity(value):
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("Initial-condition identity must be a SHA256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(
            "Initial-condition identity must be a SHA256 digest"
        ) from error
    return value


def _fields(fields):
    if any(name not in fields for name in FIELDS):
        raise ValueError("Pair archive is missing required input/target fields")
    shape = np.shape(fields["density"])
    if len(shape) != 3 or len(set(shape)) != 1 or shape[0] < 2:
        raise ValueError("Pair fields must use one cubic grid with at least two cells")
    for name in FIELDS:
        values = np.asarray(fields[name])
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(
                f"Pair field is nonfinite or has a mismatched grid: {name}"
            )
        if (values < 0).any():
            raise ValueError(f"Pair field is negative: {name}")
        if name.startswith("xhi") and (values > 1).any():
            raise ValueError(f"Pair fraction lies outside [0,1]: {name}")
    if (np.asarray(fields["density"]) <= 0).any():
        raise ValueError("Pair density must be strictly positive")


def _load_es(path, history):
    with np.load(path, allow_pickle=False) as archive:
        if not {*ES_FIELDS, "metadata"} <= set(archive.files):
            raise ValueError(
                "Missing native ES radius sidecar fields; regenerate with es_output"
            )
        result = {key: np.array(archive[key], copy=True) for key in ES_FIELDS}
        meta = _json(archive["metadata"])
    hm = history["metadata"]
    if (
        meta.get("schema_version") != SCHEMA
        or meta.get("kind") != "native-es-emulator-features"
        or meta.get("diagnostic_capability") != "STORE_ES_RADIUS/es_crossing_radius"
        or meta.get("radius_units") != "cMpc"
    ):
        raise ValueError(
            "Unsupported ES sidecar or missing native crossing-radius capability"
        )
    for key in ES_FIELDS:
        if meta.get("array_sha256", {}).get(key) != array_hash(result[key]):
            raise ValueError(f"Native ES sidecar hash mismatch: {key}")
    for key in ("random_seed", "initial_hires_density_sha256", "box_len_cMpc"):
        if meta.get(key) != hm.get(key):
            raise ValueError(f"Native ES and source history identity differs: {key}")
    if meta.get("history_array_sha256") != hm["array_sha256"]:
        raise ValueError("Native ES sidecar belongs to a different source history")
    for key in ("redshifts", "times_seconds"):
        if not np.array_equal(result[key], history[key]):
            raise ValueError("ES sidecar endpoint clock differs from source history")
    if not np.array_equal(result["xhi_es"], history["native_xhi"]):
        raise ValueError("ES sidecar neutral field differs from source history")
    shape = history["native_xhi"].shape
    for key in ("xhi_es", "radius_es_cMpc", "density_es"):
        values = result[key]
        if values.shape != shape or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Invalid native ES sidecar field: {key}")
    if (result["density_es"] <= 0).any():
        raise ValueError("Native ES gas density must be positive")
    if (result["radius_es_cMpc"] > hm["box_len_cMpc"]).any():
        raise ValueError("ES crossing radius exceeds its physical box length")
    if np.any((result["radius_es_cMpc"] > 0) & (result["xhi_es"] > 1e-6)):
        raise ValueError(
            "ES crossing radius is attached to a non-crossed native centre"
        )
    return result, meta


def _check_geometry(manifest, history):
    coarse = history["capacity"].shape[0]
    hm = history["metadata"]
    refinement = manifest.get("refinement")
    if (type(refinement) is not int or refinement < 1 or refinement % 2 == 0
            or manifest["dimension"] != coarse * refinement):
        raise ValueError("Teacher refinement must preserve original native source centres")
    if (manifest.get("native_dimension") != coarse
            or manifest.get("box_len_cMpc") != hm["box_len_cMpc"]
            or manifest.get("density_reference_redshift") != hm["reference_redshift"]):
        raise ValueError("Teacher and native ES geometry/density reference differ")
    active = np.any(history["rates"] > 0, axis=0)
    if not active.any():
        active[(coarse // 2,) * 3] = True
    positions = ((np.argwhere(active) + 1) * refinement - (refinement - 1) // 2).astype(np.int32)
    if (manifest.get("source_count") != len(positions)
            or manifest.get("source_positions_sha256") != array_hash(positions)):
        raise ValueError("Teacher source positions differ from original native cell centres")


def _teacher(directory, history):
    if (directory / "FAILED").exists():
        raise ValueError("Teacher has a FAILED marker")
    decoded = load_run(directory)
    summary = json.loads((directory / "summary.json").read_text())
    manifest = decoded["manifest"]
    if summary.get("validation", {}).get("state_checks_passed") is not True:
        raise ValueError("Teacher state checks did not pass")
    if manifest.get("transport") != "conservative_rays":
        raise ValueError("Training labels require conservative_rays transport")
    conventions = {
        "photon_normalization": "native_N_b0_H_plus_He_nuclei",
        "density_evolution": "frozen_native_overdensity_with_cosmological_dilution",
        "source_units": "escaped ionizing photons per second",
    }
    if any(manifest.get(key) != value for key, value in conventions.items()):
        raise ValueError("Teacher photon/density count convention is unsupported")
    if manifest.get("native_array_sha256") != history["metadata"]["array_sha256"]:
        raise ValueError("Teacher source/gas history hashes differ from ES inputs")
    metadata_hash = hashlib.sha256(
        json.dumps(history["metadata"], sort_keys=True).encode()
    ).hexdigest()
    if manifest.get("native_metadata_sha256") != metadata_hash:
        raise ValueError("Teacher source-history metadata differs from ES inputs")
    hm = history["metadata"]
    provenance = manifest.get("native_history_provenance", {})
    for key in ("random_seed", "initial_hires_density_sha256"):
        if provenance.get(key) != hm.get(key):
            raise ValueError(f"Teacher initial-condition identity differs: {key}")
    _check_geometry(manifest, history)
    # Use the native binary and raw independent counters, not a mutable states.npz
    # label copy or the secondary-inclusive rate-product estimate.
    from research.c2ray_teacher.build import read_output

    native = read_output(directory / "science/native_output.bin")
    validate_output(native, manifest)
    count_per_cell = manifest.get("mean_native_cell_nuclei")
    if (
        not isinstance(count_per_cell, (int, float))
        or isinstance(count_per_cell, bool)
        or not np.isfinite(count_per_cell)
        or count_per_cell <= 0
    ):
        raise ValueError("Teacher lacks a positive native-cell nuclei normalization")
    total_nuclei = float(history["capacity"].sum(dtype=np.longdouble)) * count_per_cell
    if not np.isclose(
        total_nuclei, manifest["initial_total_nuclei"], rtol=2e-12, atol=0
    ):
        raise ValueError("Teacher nuclei normalization differs from source history")
    total_rate = history["rates"].sum(axis=(1, 2, 3), dtype=np.longdouble).astype(float)
    times = history["times_seconds"]
    previous = 0.0
    for step, native_step in zip(manifest["steps"], native["steps"], strict=True):
        start, end = step.get("time_start_seconds"), step["time_end_seconds"]
        if not np.isclose(end, native_step["time_seconds"], rtol=2e-12, atol=1e-8):
            raise ValueError("Teacher/native saved endpoints have inconsistent clocks")
        if (
            not isinstance(start, (int, float))
            or start != previous
            or not start < end <= times[-1]
        ):
            raise ValueError(
                "Teacher transport intervals do not partition the source clock"
            )
        nodes = np.r_[start, times[(times > start) & (times < end)], end]
        values = np.interp(nodes, times, total_rate)
        emitted = (
            float(np.sum(0.5 * (values[:-1] + values[1:]) * np.diff(nodes)))
            * count_per_cell
        )
        if not np.isclose(emitted, step["emitted_photons"], rtol=2e-12, atol=1e-100):
            raise ValueError(
                "Teacher emitted-photon budget differs from source-rate integral"
            )
        previous = end
    if previous != times[-1]:
        raise ValueError("Teacher output does not cover the complete prescribed source history")
    timing = summary.get("timing", {})
    active_ratio = timing.get("sed_transport_photons_over_nominal")
    if active_ratio is None:
        raise ValueError("Teacher lacks measured active-band spectral normalization")
    diagnostics = read_transport_diagnostics(
        directory / "science/native_diagnostics/transport_diagnostics.jsonl",
        native,
        timing["sed_table_photons_over_nominal"],
        active_spectral_ratio=active_ratio,
    )
    for row in diagnostics["steps"]:
        residual = row["photon_residual_over_transported_emitted"]
        if residual is not None and abs(residual) > 1e-8:
            raise ValueError("Teacher primary absorption/outflow photon ledger failed")
        if residual is None and abs(row["photon_accounted"]) > 1e-100:
            raise ValueError(
                "Zero-emission teacher step has nonzero primary photon accounting"
            )
    return decoded, summary, diagnostics


def _physics(history, summary):
    hm, manifest = history["metadata"], summary["input_manifest"]
    inputs = hm.get("inputs", {})
    matter = dict(inputs.get("matter_options", {}))
    for key in ("STORE_ES_RADIUS", "MINIMIZE_MEMORY"):
        matter.pop(key, None)
    return {
        "native_source_inputs": {
            key: inputs.get(key)
            for key in ("astro_options", "astro_params", "cosmo_params")
        },
        "native_matter_options": matter,
        "rates_definition": hm["rates_definition"],
        "t_STAR": hm.get("t_STAR"),
        "reference_redshift": hm["reference_redshift"],
        "cosmology": manifest.get("cosmology"),
        "initial_state": manifest.get("initial_state"),
        "thermal_evolution": manifest.get("thermal_evolution"),
        "helium_number_fraction": manifest.get("helium_number_fraction"),
        "transport": manifest["transport"],
        "angular_order": manifest.get("angular_order"),
        "initial_redshift": float(history["redshifts"][0]),
        "spectral_table_over_nominal": summary["timing"][
            "sed_table_photons_over_nominal"
        ],
        "spectral_active_over_nominal": summary["timing"][
            "sed_transport_photons_over_nominal"
        ],
        "source_integration": manifest.get("source_integration"),
    }


def export_pairs(
    history_path, es_path, teacher_results, output, split, expected_seed=None,
    *, initial_state_config=None,
):
    """Export audited same-IC/same-endpoint ES/RT samples into a fresh directory."""
    if split not in SPLITS:
        raise ValueError("Split must be train, validation or test")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    history_path, es_path, directory = (
        Path(path).resolve() for path in (history_path, es_path, teacher_results)
    )
    history = load_history(history_path)
    hm = history["metadata"]
    seed = _positive_seed(hm.get("random_seed"))
    ic_hash = _identity(hm.get("initial_hires_density_sha256"))
    if expected_seed is not None and seed != _positive_seed(expected_seed):
        raise ValueError("Requested seed differs from native history")
    es, _ = _load_es(es_path, history)
    decoded, summary, diagnostics = _teacher(directory, history)
    manifest = decoded["manifest"]
    initial_policy = validate_policy(initial_state_config)
    initial_audit = initial_state_audit(
        history, es, initial_xhii=manifest["initial_state"]["xHII"],
        max_mass_weighted_error=initial_policy["max_mass_weighted_error"],
    )
    initial_audit["exploratory_mismatch_opt_in"] = initial_policy["allow_mismatch"]
    require_initial_state(initial_audit, allow_mismatch=initial_policy["allow_mismatch"])
    history_digest, es_digest = file_hash(history_path), file_hash(es_path)
    times, rates = history["times_seconds"], history["rates"]
    cumulative = np.zeros_like(rates, dtype=np.float64)
    for i in range(1, len(times)):
        cumulative[i] = cumulative[i - 1] + 0.5 * (rates[i - 1] + rates[i]) * (
            times[i] - times[i - 1]
        )
    matches, used = [], set()
    for step_index, step in enumerate(manifest["steps"]):
        found = np.flatnonzero(
            np.isclose(history["redshifts"], step["redshift_end"], rtol=0, atol=1e-8)
        )
        if len(found) > 1:
            raise ValueError("Ambiguous teacher/native redshift match")
        if len(found) == 0:
            continue
        index = int(found[0])
        if index in used or index == 0:
            raise ValueError("Repeated or initial-time teacher label")
        if not np.isclose(
            step["time_end_seconds"], times[index], rtol=2e-12, atol=1e-8
        ):
            raise ValueError("Teacher/native saved endpoints have inconsistent clocks")
        used.add(index)
        matches.append((step_index, index))
    if not matches:
        raise ValueError("No common saved endpoints; label interpolation is disabled")
    physics = _physics(history, summary)
    records = []
    for step_index, index in matches:
        fine = {
            key: decoded["fields"][key][step_index]
            for key in ("density_total_cm3", "xHII", "temperature_K")
        }
        coarse = restrict_fields(fine, len(history["capacity"]))
        normalized_density = (
            coarse["density_total_cm3"] / coarse["density_total_cm3"].mean()
        )
        normalized_capacity = history["capacity"] / history["capacity"].mean()
        if not np.allclose(
            normalized_density, normalized_capacity, rtol=2e-12, atol=1e-12
        ):
            raise ValueError("Teacher gas does not reproduce declared frozen density")
        fields = {
            "xhi_es": es["xhi_es"][index],
            "radius_es_cMpc": es["radius_es_cMpc"][index],
            "density": history["capacity"],
            "photons": cumulative[index],
            "xhi_c2ray": 1 - coarse["xHII"],
        }
        _fields(fields)
        record = {
            "seed": seed,
            "split": split,
            "snapshot_id": index,
            "redshift": float(history["redshifts"][index]),
            "time_seconds": float(times[index]),
            "box_len_cMpc": float(hm["box_len_cMpc"]),
            "dimension": len(history["capacity"]),
            "initial_conditions_sha256": ic_hash,
        }
        meta = record | {
            "schema_version": SCHEMA,
            "kind": "es-to-c2ray-pair",
            "density_definition": "1+delta at source-history reference redshift",
            "photon_definition": "cumulative trapezoidal escaped-source count per mean-nuclei native cell; no density reweighting",
            "radius_definition": "largest native ES crossing radius in cMpc; zero means no crossing",
            "target_definition": "nuclei-weighted C2-Ray HI fraction on original source grid",
            "scope": "ES-to-C2-Ray field emulation; ES source clock/gas and RT physics differ explicitly",
            "contextual_native_ES": hm.get("contextual_native_ES"),
            "physics": physics,
            "history_sha256": history_digest,
            "es_sidecar_sha256": es_digest,
            "teacher_native_output_sha256": summary["native_output_sha256"],
            "teacher_transport_primary_residual": diagnostics["steps"][step_index][
                "photon_residual_over_transported_emitted"
            ],
            "convergence_certified": False,
            "initial_state_audit": initial_audit,
            "array_sha256": {key: array_hash(value) for key, value in fields.items()},
        }
        path = output / f"snapshot_{index:04d}.npz"
        _atomic(
            path,
            lambda stream, f=fields, m=meta: np.savez_compressed(
                stream,
                **f,
                metadata=np.asarray(json.dumps(m, sort_keys=True, allow_nan=False)),
            ),
        )
        records.append(record | {"path": str(path), "sha256": file_hash(path)})
    if file_hash(history_path) != history_digest or file_hash(es_path) != es_digest:
        raise ValueError("Native history or ES sidecar changed during pair export")
    result = {
        "schema_version": SCHEMA,
        "kind": "es-to-c2ray-dataset",
        "samples": records,
        "split_policy": "all epochs/geometries of a seed and IC identity stay in one split",
        "physics": physics,
        "convergence_certified": False,
        "initial_state_compatible": initial_audit["compatible"],
        "scope": "Audited paired emulator data; transport closure does not certify spatial/time convergence",
    }
    _write_json(output / "manifest.json", result)
    return result


def load_pair(record):
    """Load a sample after verifying its file, fields, metadata and identity."""
    path = Path(record["path"])
    if file_hash(path) != record.get("sha256"):
        raise ValueError(f"Pair file hash changed: {path}")
    with np.load(path, allow_pickle=False) as archive:
        fields = {key: np.array(archive[key], copy=True) for key in FIELDS}
        meta = _json(archive["metadata"])
    if meta.get("schema_version") != SCHEMA or meta.get("kind") != "es-to-c2ray-pair":
        raise ValueError("Unsupported pair schema")
    for key in (
        "seed",
        "split",
        "snapshot_id",
        "redshift",
        "time_seconds",
        "box_len_cMpc",
        "dimension",
        "initial_conditions_sha256",
    ):
        if meta.get(key) != record.get(key):
            raise ValueError(f"Pair record differs from stored metadata: {key}")
    for key in FIELDS:
        if meta.get("array_sha256", {}).get(key) != array_hash(fields[key]):
            raise ValueError(f"Pair array hash changed: {key}")
    _fields(fields)
    if fields["density"].shape != (record["dimension"],) * 3:
        raise ValueError("Pair grid differs from its declared dimension")
    return fields | {"metadata": meta}


def _initial_compatibility(statuses):
    """Preserve a known failure even when another legacy sample is unaudited."""
    if any(value is not None and type(value) is not bool for value in statuses):
        raise ValueError("Initial-state compatibility must be boolean or unknown")
    if any(value is False for value in statuses):
        return False
    if any(value is None for value in statuses):
        return None
    return True


def _audit_manifest(manifest, *, require_splits):
    if (
        manifest.get("schema_version") != SCHEMA
        or manifest.get("kind") != "es-to-c2ray-dataset"
    ):
        raise ValueError("Unsupported ES-to-C2-Ray dataset manifest")
    records = manifest.get("samples")
    if not isinstance(records, list) or not records:
        raise ValueError("Dataset has no samples")
    seeds, ics, seen, splits = {}, {}, set(), set()
    initial_states = []
    for record in records:
        split = record.get("split")
        if split not in SPLITS:
            raise ValueError("Dataset split must be train, validation or test")
        splits.add(split)
        seed = _positive_seed(record.get("seed"))
        ic_hash = _identity(record.get("initial_conditions_sha256"))
        for mapping, key in ((seeds, seed), (ics, ic_hash)):
            if key in mapping and mapping[key] != split:
                raise ValueError(
                    "Initial-condition seed/identity leaks across data splits"
                )
            mapping[key] = split
        identity = (
            ic_hash,
            record.get("redshift"),
            record.get("dimension"),
            record.get("box_len_cMpc"),
        )
        if identity in seen:
            raise ValueError("Duplicate physical sample in dataset")
        seen.add(identity)
        fields = load_pair(record)
        initial_states.append(fields["metadata"].get("initial_state_audit", {}).get("compatible"))
        if fields["metadata"].get("physics") != manifest.get("physics"):
            raise ValueError("Dataset mixes incompatible source or RT physics")
    if require_splits and splits != set(SPLITS):
        raise ValueError("Training requires nonempty train, validation and test splits")
    compatibility = _initial_compatibility(initial_states)
    if manifest.get("initial_state_compatible") != compatibility:
        raise ValueError("Dataset initial-state status differs from paired metadata")
    return manifest


def load_dataset(manifest_path):
    """Verify all samples, physics compatibility and independent data splits."""
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text())
    for record in manifest.get("samples", []):
        if not Path(record["path"]).is_absolute():
            record["path"] = str((path.parent / record["path"]).resolve())
    return _audit_manifest(manifest, require_splits=True)


def combine_manifests(manifests, output_path):
    """Combine completed seed manifests without relaxing any split/integrity rule."""
    parts = []
    for value in manifests:
        if isinstance(value, (str, Path)):
            path = Path(value).resolve()
            part = json.loads(path.read_text())
            for record in part.get("samples", []):
                if not Path(record["path"]).is_absolute():
                    record["path"] = str((path.parent / record["path"]).resolve())
        else:
            part = value
        parts.append(_audit_manifest(part, require_splits=False))
    if not parts:
        raise ValueError("No completed seed manifests supplied")
    result = dict(parts[0])
    result["samples"] = [row for part in parts for row in part["samples"]]
    statuses = [part.get("initial_state_compatible") for part in parts]
    result["initial_state_compatible"] = _initial_compatibility(statuses)
    if any(part["physics"] != result["physics"] for part in parts):
        raise ValueError("Seed datasets use incompatible source or RT physics")
    _audit_manifest(result, require_splits=True)
    _write_json(output_path, result)
    return result
