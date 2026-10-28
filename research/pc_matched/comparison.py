"""Audit direct radiative-transfer histories and export matched ES/PC pairs.

The report compares common-time fields using identical prescribed photons and
fixed gas. Native ES remains a contextual control: its evolving gas and source
proxy differ. Photon closure, numerical refinement, and field accuracy are
reported separately. Exported pairs are not convergence-certified training data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from research.pc_evolution.native_history import array_hash, load_history
from research.pc_field_emulator.evaluation import shell_spectra
from research.pc_large_scale.pc_statistics import es_map, es_threshold

LEDGER_TOLERANCE = 2e-10
SOURCE_TOLERANCE = 1e-12
REFINEMENT_DIMENSIONS = {
    "angular": "directions",
    "spatial": "refinement",
    "temporal": "cfl",
    "source_history": "source_stride",
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _sum(value: np.ndarray) -> np.longdouble:
    return np.sum(value, dtype=np.longdouble)


def _resolve(root: Path, path: str) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _integrated_sources(history: dict) -> np.ndarray:
    rates = np.asarray(history["rates"], dtype=np.longdouble)
    dt = np.diff(history["times_seconds"]).astype(np.longdouble)
    increments = 0.5 * (rates[:-1] + rates[1:]) * dt[:, None, None, None]
    return np.concatenate([np.zeros_like(rates[:1]), np.cumsum(increments, axis=0)])


def _repeat(value: np.ndarray, refinement: int) -> np.ndarray:
    for axis in range(value.ndim - 3, value.ndim):
        value = np.repeat(value, refinement, axis=axis)
    return value


def _coarsen(value: np.ndarray, dimension: int) -> np.ndarray:
    refinement = value.shape[0] // dimension
    # Fine grids use mean-fine-cell count units; block means preserve original-cell units.
    return value.reshape((dimension, refinement) * 3).mean(
        axis=(1, 3, 5), dtype=np.float64
    )


def _audit_snapshot(
    ionized: np.ndarray,
    radiation: np.ndarray,
    source: np.ndarray,
    expected: np.ndarray,
    capacity: np.ndarray,
) -> dict:
    active = expected > 0
    source_error = (
        float(np.max(np.abs(source[active] / expected[active] - 1)))
        if active.any()
        else 0.0
    )
    if np.any(source[~active] != 0) or source_error > SOURCE_TOLERANCE:
        raise ValueError(
            "RTE photons differ from independently integrated source history"
        )
    total_source = _sum(source)
    total_ions, total_radiation = _sum(ionized), _sum(radiation)
    scale = max(total_source, np.longdouble(np.finfo(float).tiny))
    residual = float(abs(total_source - total_ions - total_radiation) / scale)
    if residual > LEDGER_TOLERANCE:
        raise ValueError("Independent ion-plus-radiation photon ledger does not close")
    return {
        "passed": True,
        "emitted": float(total_source),
        "ionized": float(total_ions),
        "radiation_in_flight": float(total_radiation),
        "inflight_fraction": float(total_radiation / scale),
        "global_fractional_residual": residual,
        "source_history_maximum_fractional_residual": source_error,
        "xHI_volume": float(np.mean(1 - ionized / capacity)),
        "xHI_mass": float((_sum(capacity) - total_ions) / _sum(capacity)),
    }


def _sampled_rates(history: dict, stride: int, nodes: np.ndarray) -> np.ndarray:
    knots = np.arange(0, len(history["times_seconds"]), stride)
    if knots[-1] != len(history["times_seconds"]) - 1 or not set(knots) <= set(nodes):
        raise ValueError(
            "Source history knots must include both endpoints and be saved"
        )
    times = history["times_seconds"]
    upper = np.minimum(
        np.searchsorted(times[knots], times[nodes], side="right"), len(knots) - 1
    )
    lower = np.maximum(upper - 1, 0)
    fraction = (times[nodes] - times[knots[lower]]) / (
        times[knots[upper]] - times[knots[lower]]
    )
    rates = history["rates"][knots]
    sampled = (1 - fraction[:, None, None, None]) * rates[lower] + fraction[
        :, None, None, None
    ] * rates[upper]
    return sampled


def _typed_array_hash(value: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _audit_source_identity(
    metadata: dict, history: dict, rates: np.ndarray, stride: int
) -> None:
    definition = metadata.get("source_history")
    if definition is None:
        return
    payload = {
        key: value
        for key, value in definition.items()
        if key != "source_history_sha256"
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    expected_hashes = {
        name: _typed_array_hash(value)
        for name, value in {
            "rates": rates,
            "capacity": history["capacity"],
            "times_seconds": history["times_seconds"],
            "redshifts": history["redshifts"],
        }.items()
    }
    if (
        definition.get("source_history_sha256") != digest
        or metadata.get("source_history_sha256") != digest
        or definition.get("parent_fixture_sha256") != history["_fixture_sha256"]
        or definition.get("source_stride") != stride
        or definition.get("source_knot_indices") != list(range(0, len(rates), stride))
        or definition.get("native_snapshot_indices") != list(range(len(rates)))
        or definition.get("arrays_sha256") != expected_hashes
    ):
        raise ValueError(
            "RTE source history identity differs from independently reconstructed rates"
        )


def _load_case(root: Path, task: dict, history: dict) -> dict:
    path = _resolve(root, task["path"])
    if _digest(path) != task["sha256"]:
        raise ValueError("RTE field checksum differs from matrix identity")
    task = dict(task, source_stride=task.get("source_stride", 1))
    refinement, stride = task["refinement"], task["source_stride"]
    if (
        type(refinement) is not int
        or refinement < 1
        or task["directions"] not in (6, 26, 24, 96, 384)
        or type(stride) is not int
        or stride < 1
        or not np.isfinite(task["cfl"])
        or not 0 < task["cfl"] <= 1
    ):
        raise ValueError("Invalid RTE case resolution or Courant factor")
    with np.load(path, allow_pickle=False) as archive:
        raw = {name: archive[name] for name in archive.files}
    metadata = json.loads(str(raw["metadata"].item()))
    setting = {key: task[key] for key in REFINEMENT_DIMENSIONS.values()}
    actual_setting = dict(metadata.get("setting", {}))
    actual_setting.setdefault("source_stride", metadata.get("source_stride", 1))
    if (
        actual_setting != setting
        or metadata.get("source_stride", stride) != stride
        or metadata.get("seed") != task["seed"]
        or metadata.get("fixture_sha256") != history["_fixture_sha256"]
        or metadata.get("box_len_cMpc") != history["metadata"]["box_len_cMpc"]
        or (
            "source_history_sha256" in task
            and task["source_history_sha256"] != metadata.get("source_history_sha256")
        )
    ):
        raise ValueError("RTE metadata differs from matrix settings or native fixture")
    times = history["times_seconds"]
    nodes = np.searchsorted(times, raw["times_seconds"])
    if (
        nodes.ndim != 1
        or len(nodes) < 2
        or nodes[0] != 0
        or nodes[-1] != len(times) - 1
        or np.any(np.diff(nodes) <= 0)
        or (
            "native_snapshot_indices" in metadata
            and metadata["native_snapshot_indices"] != nodes.tolist()
        )
        or (stride > 1 and "native_snapshot_indices" not in metadata)
        or (stride > 1 and not np.array_equal(nodes, np.arange(len(times))))
    ):
        raise ValueError(
            "RTE native snapshot indices do not match its physical timeline"
        )
    knots = np.arange(0, len(times), stride).tolist()
    if "source_knot_indices" in metadata and metadata["source_knot_indices"] != knots:
        raise ValueError("RTE source knot indices differ from declared source stride")
    sampled = _sampled_rates(history, stride, nodes)
    _audit_source_identity(metadata, history, sampled, stride)
    expected = _integrated_sources({"rates": sampled, "times_seconds": times[nodes]})
    capacity = _repeat(history["capacity"], refinement)
    shape = (len(nodes), *capacity.shape)
    if (
        not np.array_equal(raw["capacity"], capacity)
        or not np.array_equal(raw["redshifts"], history["redshifts"][nodes])
        or not np.array_equal(raw["times_seconds"], times[nodes])
    ):
        raise ValueError(
            "RTE capacity or physical times differ from native source fixture"
        )
    for name in ("ionized", "radiation", "source_emitted"):
        if (
            raw[name].shape != shape
            or np.iscomplexobj(raw[name])
            or not np.isfinite(raw[name]).all()
        ):
            raise ValueError("RTE counts must be finite matching real histories")
        if np.any(raw[name] < -LEDGER_TOLERANCE * capacity):
            raise ValueError("RTE photon ledger contains negative counts")
        if np.any(raw[name][0] != 0):
            raise ValueError(
                "RTE initial state must be neutral with zero radiation and emission"
            )
    if np.any(raw["ionized"] > (1 + LEDGER_TOLERANCE) * capacity):
        raise ValueError("RTE photon ledger exceeds local atom capacity")
    decrease = float(np.max(np.maximum(-np.diff(raw["ionized"], axis=0), 0) / capacity))
    if decrease > LEDGER_TOLERANCE:
        raise ValueError(
            "Ionized counts decrease despite fixed gas and no recombination"
        )
    fields, audits = {}, []
    for local, node in enumerate(nodes):
        redshift = history["redshifts"][node]
        fine_source = _repeat(expected[local], refinement)
        audit = _audit_snapshot(
            raw["ionized"][local],
            raw["radiation"][local],
            raw["source_emitted"][local],
            fine_source,
            capacity,
        )
        audit.update(
            node_index=int(node),
            redshift=float(redshift),
            time_seconds=float(times[node]),
            maximum_ionized_fraction_decrease=decrease,
            counts_normalization="Mean-fine-cell units; divide totals by refinement^3 for original-cell units",
        )
        fields[int(node)] = {
            name: _coarsen(raw[name][local], len(history["capacity"]))
            for name in ("ionized", "radiation", "source_emitted")
        }
        fields[int(node)].update(
            time_seconds=float(times[node]), redshift=float(redshift)
        )
        audits.append(audit)
    return {
        "task": task,
        "fields": fields,
        "audits": audits,
        "metadata": metadata,
        "native_snapshot_indices": nodes.tolist(),
        "source_knot_indices": knots,
    }


def _matched_control(history: dict, expected: np.ndarray) -> dict:
    capacity = history["capacity"]
    fields = {}
    for node, source in enumerate(expected):
        source = np.asarray(source, dtype=np.float64)
        threshold, filtering = es_threshold(source, capacity, len(capacity) / 2)
        fields[node] = {
            "ionized": es_map(source, capacity, threshold, 1.0),
            "source_emitted": source,
            "time_seconds": float(history["times_seconds"][node]),
            "redshift": float(history["redshifts"][node]),
        }
    return {
        "task": {"model": "matched-source-ES"},
        "fields": fields,
        "filtering": filtering,
    }


def _distribution(values: list[float | None]) -> dict:
    defined = np.asarray([x for x in values if x is not None], dtype=float)
    return {
        "count": len(defined),
        "undefined": len(values) - len(defined),
        "mean": float(defined.mean()) if len(defined) else None,
        "median": float(np.median(defined)) if len(defined) else None,
        "max": float(defined.max()) if len(defined) else None,
    }


def _neutral(fields: dict, capacity: np.ndarray) -> np.ndarray:
    # Bounds were audited before this roundoff-only adjustment; retain raw counts.
    return np.clip(1 - fields["ionized"] / capacity, 0, 1)


def _field_metrics(
    xnew: np.ndarray, xref: np.ndarray, history: dict, amplitude: float
) -> dict:
    capacity = history["capacity"]
    box = float(history["metadata"]["box_len_cMpc"])
    spectra = shell_spectra(
        {
            "teacher": amplitude * capacity * xref,
            "comparison": amplitude * capacity * xnew,
        },
        box,
    )
    # Preserve all populated NN-pilot primary shells; never replace by band means.
    k = np.asarray(spectra["k_cMpc_inv"])
    primary = (k >= 0.1) & (k <= 0.6 * np.pi * len(capacity) / box)
    if not primary.any():
        raise ValueError("No populated primary power bins")
    errors = [
        abs(r - 1) if r is not None else None for r in spectra["ratio_comparison"]
    ]
    selected = [v for v, keep in zip(errors, primary, strict=True) if keep]
    difference = np.abs(np.asarray(spectra["P_comparison"]) - spectra["P_teacher"])
    active = 0.1 <= float(xref.mean()) < 0.9
    return {
        "reference_xHI_volume": float(xref.mean()),
        "comparison_xHI_volume": float(xnew.mean()),
        "reference_xHI_mass": float(_sum(capacity * xref) / _sum(capacity)),
        "comparison_xHI_mass": float(_sum(capacity * xnew) / _sum(capacity)),
        "field_MAE": float(np.mean(abs(xnew - xref))),
        "mass_weighted_field_MAE": float(
            _sum(capacity * abs(xnew - xref)) / _sum(capacity)
        ),
        "primary_bins": primary.tolist(),
        "spectra": spectra,
        "power_error_per_bin": errors,
        "power_error": _distribution(selected),
        "absolute_power_error": _distribution(difference[primary].tolist()),
        "absolute_power_error_units": "mK^2 cMpc^3",
        "active_reference": active,
        "all_primary_bins_within_one_percent": bool(
            active and all(v is not None and v <= 0.01 for v in selected)
        ),
        "undefined_power_reason": "Numerically vanishing reference fluctuation power"
        if None in selected
        else None,
    }


def _physical_nodes(case: dict, history: dict) -> dict[float, int]:
    result = {}
    for node, field in case["fields"].items():
        time = float(field.get("time_seconds", history["times_seconds"][node]))
        redshift = float(field.get("redshift", history["redshifts"][node]))
        if (
            time != history["times_seconds"][node]
            or redshift != history["redshifts"][node]
            or time in result
        ):
            raise ValueError(
                "Comparison snapshot indices differ from physical time or redshift"
            )
        result[time] = node
    return result


def _compare(new: dict, reference: dict, history: dict) -> list[dict]:
    capacity = history["capacity"]
    new_nodes, ref_nodes = (
        _physical_nodes(new, history),
        _physical_nodes(reference, history),
    )
    rows = []
    for time in sorted(new_nodes.keys() & ref_nodes.keys()):
        node, ref_node = new_nodes[time], ref_nodes[time]
        if history["redshifts"][node] != history["redshifts"][ref_node]:
            raise ValueError("Common comparison times must have matching redshifts")
        amplitude = 27 * np.sqrt((1 + history["redshifts"][ref_node]) / 10)
        metrics = _field_metrics(
            _neutral(new["fields"][node], capacity),
            _neutral(reference["fields"][ref_node], capacity),
            history,
            amplitude,
        )
        rows.append(
            dict(
                metrics,
                node_index=ref_node,
                comparison_node_index=node,
                redshift=float(history["redshifts"][ref_node]),
                time_seconds=time,
            )
        )
    return rows


def _interpolate_neutral(
    case: dict, history: dict, target: float
) -> tuple[np.ndarray, dict] | None:
    capacity = history["capacity"]
    timed = _physical_nodes(case, history)
    nodes = [timed[t] for t in sorted(timed)]
    means = [float(_neutral(case["fields"][n], capacity).mean()) for n in nodes]
    # Select an exact saved endpoint when possible; otherwise the first time bracket.
    bracket = next(((i, i) for i, mean in enumerate(means) if mean == target), None)
    if bracket is None:
        bracket = next(
            (
                (i, i + 1)
                for i in range(len(nodes) - 1)
                if min(means[i : i + 2]) <= target <= max(means[i : i + 2])
                and means[i] != means[i + 1]
            ),
            None,
        )
    if bracket is None:
        return None
    lo, hi = bracket
    weight = 0.0 if lo == hi else (target - means[lo]) / (means[hi] - means[lo])
    left, right = nodes[lo], nodes[hi]
    xhi = (1 - weight) * _neutral(case["fields"][left], capacity) + weight * _neutral(
        case["fields"][right], capacity
    )
    return xhi, {
        "node_indices": [left, right],
        "upper_weight": float(weight),
        "endpoint_xHI_volume": [means[lo], means[hi]],
        "endpoint_times_seconds": [
            float(history["times_seconds"][n]) for n in (left, right)
        ],
        "endpoint_redshifts": [float(history["redshifts"][n]) for n in (left, right)],
        "time_seconds": float(
            (1 - weight) * history["times_seconds"][left]
            + weight * history["times_seconds"][right]
        ),
        "redshift_linear_diagnostic": float(
            (1 - weight) * history["redshifts"][left]
            + weight * history["redshifts"][right]
        ),
        "first_bracket_chosen": True,
    }


def _matched_mean_neutral(
    new: dict,
    reference: dict,
    history: dict,
    *,
    targets: tuple[float, ...] = (0.8, 0.6, 0.4, 0.2),
) -> list[dict]:
    rows = []
    for target in targets:
        a, b = (
            _interpolate_neutral(new, history, target),
            _interpolate_neutral(reference, history, target),
        )
        row = {
            "target_xHI_volume": target,
            "available": a is not None and b is not None,
            "is_training_label": False,
            "common_brightness_amplitude_mK": 27.0,
        }
        if a is None or b is None:
            row["reason"] = (
                "Target is not bracketed in both saved histories; no extrapolation"
            )
            row["comparison_target_covered"] = a is not None
            row["reference_target_covered"] = b is not None
        else:
            row.update(_field_metrics(a[0], b[0], history, 27.0))
            row.update(comparison_interpolation=a[1], reference_interpolation=b[1])
        rows.append(row)
    return rows


def _attach_native_density(root: Path, fixture: dict, history: dict) -> None:
    """Accept only a hashed native sidecar belonging to this exact source history."""
    declared = fixture.get("native_es_sidecar")
    if declared is None:
        return
    sidecar = _resolve(root, declared)
    if _digest(sidecar) != fixture.get("native_es_sidecar_sha256"):
        raise ValueError("Native ES sidecar file differs from matrix identity")
    with np.load(sidecar, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    meta = json.loads(str(values.pop("metadata").item()))
    required = {"redshifts", "times_seconds", "xhi_es", "radius_es_cMpc", "density_es"}
    if (
        set(values) != required
        or meta.get("kind") != "native-es-emulator-features"
        or meta.get("history_array_sha256") != history["metadata"]["array_sha256"]
        or meta.get("array_sha256")
        != {name: array_hash(value) for name, value in values.items()}
    ):
        raise ValueError("Native ES sidecar hashes differ from source history")
    for name in (
        "random_seed",
        "initial_hires_density_sha256",
        "native_extension_sha256",
        "box_len_cMpc",
    ):
        if meta.get(name) != history["metadata"].get(name):
            raise ValueError("Native ES sidecar provenance differs from source history")
    if (
        not np.array_equal(values["redshifts"], history["redshifts"])
        or not np.array_equal(values["times_seconds"], history["times_seconds"])
        or not np.array_equal(values["xhi_es"], history["native_xhi"])
    ):
        raise ValueError("Native ES sidecar fields or times differ from source history")
    density, radius = values["density_es"], values["radius_es_cMpc"]
    shape = history["native_xhi"].shape
    if (
        density.shape != shape
        or radius.shape != shape
        or not np.isfinite(density).all()
        or not np.isfinite(radius).all()
        or np.any(density <= 0)
        or np.any(radius < 0)
        or np.any((radius > 0) & (history["native_xhi"] > 1e-6))
    ):
        raise ValueError("Invalid native ES sidecar density or crossing radii")
    history["native_density"] = density
    history["_native_sidecar"] = {"path": str(sidecar), "sha256": _digest(sidecar)}


def _native_context(history: dict, nodes: list[int], reference: dict) -> dict:
    native = history["native_xhi"]
    density = history.get("native_density")
    if density is not None:
        valid = (
            density.shape == native.shape
            and np.isfinite(density).all()
            and np.all(density > 0)
            and "_native_sidecar" in history
        )
        if not valid:
            raise ValueError(
                "Native evolving-density context lacks matching shape/hash"
            )
    rows = []
    for node in nodes:
        row = {
            "node_index": node,
            "redshift": float(history["redshifts"][node]),
            "xHI_volume": float(native[node].mean(dtype=np.float64)),
        }
        if density is not None:
            amplitude = 27 * np.sqrt((1 + history["redshifts"][node]) / 10)
            row["spectra"] = shell_spectra(
                {
                    "teacher": amplitude
                    * history["capacity"]
                    * _neutral(reference["fields"][node], history["capacity"]),
                    "native": amplitude * density[node] * native[node],
                },
                history["metadata"]["box_len_cMpc"],
            )
        rows.append(row)
    return {
        "scope": "Native ES uses its own source proxy and evolving gas; contextual, not source-matched",
        "power_available": density is not None,
        "sidecar": history.get("_native_sidecar"),
        "power_unavailable_reason": None
        if density is not None
        else "Fixture does not retain native evolving density; fixed PC gas is not substituted",
        "snapshots": rows,
    }


def _refinements(cases: list[dict], history: dict) -> list[dict]:
    results = []
    for kind, key in REFINEMENT_DIMENSIONS.items():
        others = set(REFINEMENT_DIMENSIONS.values()) - {key}
        for fine in cases:
            ft = fine["task"]
            candidates = [
                coarse
                for coarse in cases
                if all(coarse["task"][other] == ft[other] for other in others)
                and (
                    coarse["task"][key] > ft[key]
                    if kind in ("temporal", "source_history")
                    else coarse["task"][key] < ft[key]
                )
            ]
            if not candidates:
                continue
            coarse = min(candidates, key=lambda item: abs(item["task"][key] - ft[key]))
            if (
                kind == "source_history"
                and coarse["fields"].keys() != fine["fields"].keys()
            ):
                raise ValueError(
                    "Source-only refinement requires identical saved background timelines"
                )
            # Fine is the reference denominator; compare the coarse approximation.
            rows = _compare(coarse, fine, history)
            active = [row for row in rows if row["active_reference"]]
            results.append(
                {
                    "kind": kind,
                    "varied_setting": key,
                    "alignment": "Exact shared physical times and redshifts; canonical native snapshot indices",
                    "coarse": coarse["task"],
                    "fine": ft,
                    "checkpoints": rows,
                    "active_checkpoints": len(active),
                    "active_power_error": _distribution(
                        [
                            value
                            for row in active
                            for value, use in zip(
                                row["power_error_per_bin"],
                                row["primary_bins"],
                                strict=True,
                            )
                            if use
                        ]
                    ),
                    "active_field_MAE": _distribution(
                        [row["field_MAE"] for row in active]
                    ),
                    "all_primary_power_bins_passed": bool(active)
                    and all(
                        row["all_primary_bins_within_one_percent"] for row in active
                    ),
                    "field_MAE_within_one_percentage_point": bool(active)
                    and all(row["field_MAE"] <= 0.01 for row in active),
                }
            )
    return results


def _self_summary(refinements: list[dict]) -> dict:
    result = {}
    for kind, key in REFINEMENT_DIMENSIONS.items():
        rows = [row for row in refinements if row["kind"] == kind]
        others = sorted(set(REFINEMENT_DIMENSIONS.values()) - {key})
        chains: dict[tuple, set] = {}
        for row in rows:
            group = tuple(row["fine"][other] for other in others)
            chains.setdefault(group, set()).update(
                (row["coarse"][key], row["fine"][key])
            )
        longest = max((len(levels) - 1 for levels in chains.values()), default=0)
        result[kind] = {
            "adjacent_comparisons": len(rows),
            "maximum_successive_refinement_steps": longest,
            "two_adjacent_steps_available": longest >= 2,
            "all_have_active_checkpoints": bool(rows)
            and all(row["active_checkpoints"] for row in rows),
            "all_primary_power_bins_passed": bool(rows)
            and all(row["all_primary_power_bins_passed"] for row in rows),
            "all_field_MAE_checks_passed": bool(rows)
            and all(row["field_MAE_within_one_percentage_point"] for row in rows),
            "worst_active_power_bin_error": max(
                (
                    row["active_power_error"]["max"]
                    for row in rows
                    if row["active_power_error"]["max"] is not None
                ),
                default=None,
            ),
            "worst_active_field_MAE": max(
                (
                    row["active_field_MAE"]["max"]
                    for row in rows
                    if row["active_field_MAE"]["max"] is not None
                ),
                default=None,
            ),
            "undefined_active_power_bins": sum(
                row["active_power_error"]["undefined"] for row in rows
            ),
        }
    return result


def _export_pair(
    root: Path, history: dict, finest: dict, control: dict, fixture: dict
) -> dict:
    nodes = sorted(finest["fields"].keys() & control["fields"].keys())
    capacity = history["capacity"]
    seed = fixture["seed"]
    metadata = {
        "schema_version": 1,
        "kind": "matched-ES-PC-diagnostic-pairs",
        "seed": seed,
        "box_len_cMpc": history["metadata"]["box_len_cMpc"],
        "fixture_sha256": fixture["sha256"],
        "teacher_task": finest["task"],
        "source_history": finest["metadata"].get("source_history"),
        "source_atom_convention": history["metadata"].get(
            "capacity_units", "native effective atom counts"
        ),
        "density_evolution": "frozen_at_reference_redshift",
        "reference_redshift": history["metadata"]["reference_redshift"],
        "physics": "Direct monoenergetic finite-volume S_N RTE at physical light speed; periodic fixed density contrast, native effective H+He counts with a single H-opacity absorber; no recombination or heating feedback",
        "native_es_is_matched_control": False,
        "convergence_certified": False,
        "scope": "Diagnostic paired fields; not automatically admitted as production training labels",
        "native_es_sidecar": history.get("_native_sidecar"),
    }
    arrays = {
        "redshifts": history["redshifts"][nodes],
        "times_seconds": history["times_seconds"][nodes],
        "rho": capacity,
        "node_indices": np.array(nodes),
        "xhi_pc": np.stack(
            [_neutral(finest["fields"][node], capacity) for node in nodes]
        ),
        "xhi_es_matched": np.stack(
            [_neutral(control["fields"][node], capacity) for node in nodes]
        ),
        "xhi_es_native_context": history["native_xhi"][nodes],
        "cumulative_photons": np.stack(
            [finest["fields"][node]["source_emitted"] for node in nodes]
        ),
        "emissivity": history["rates"][nodes],
        "radiation_counts_pc": np.stack(
            [finest["fields"][node]["radiation"] for node in nodes]
        ),
        "ionized_counts_pc": np.stack(
            [finest["fields"][node]["ionized"] for node in nodes]
        ),
    }
    if "native_density" in history:
        arrays["rho_native_context"] = history["native_density"][nodes]
    metadata["array_sha256"] = {
        name: array_hash(value) for name, value in arrays.items()
    }
    destination = root / "pairs" / f"seed{seed}.npz"
    destination.parent.mkdir(exist_ok=True)
    temporary = destination.with_suffix(".npz.partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream, **arrays, metadata=np.asarray(json.dumps(metadata, sort_keys=True))
        )
    temporary.replace(destination)
    return {
        "seed": seed,
        "path": str(destination.resolve()),
        "sha256": _digest(destination),
        "snapshots": len(nodes),
        "convergence_certified": False,
        "teacher_task": finest["task"],
    }


def _plot(root: Path, report: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(report["cases"]),
        3,
        figsize=(13, 3.5 * len(report["cases"])),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, case in zip(axes, report["cases"], strict=True):
        rows = case["matched_es_comparison"]
        z = [row["redshift"] for row in rows]
        axis[0].plot(z, [r["reference_xHI_volume"] for r in rows], label="PC")
        axis[0].plot(
            z, [r["comparison_xHI_volume"] for r in rows], label="Matched-source ES"
        )
        native = case["native_es_context"]["snapshots"]
        axis[0].plot(
            [r["redshift"] for r in native],
            [r["xHI_volume"] for r in native],
            linestyle=":",
            label="Native ES context",
        )
        axis[0].set_ylabel("Mean neutral fraction")
        axis[0].legend(fontsize=7)
        for row in rows:
            if row["active_reference"]:
                axis[1].scatter(
                    [row["redshift"]], [100 * row["field_MAE"]], color="C1", s=18
                )
                errors = [
                    100 * v
                    for v, use in zip(
                        row["power_error_per_bin"], row["primary_bins"], strict=True
                    )
                    if use and v is not None
                ]
                axis[2].scatter(
                    [row["redshift"]] * len(errors), errors, color="C1", s=18
                )
        axis[1].set_ylabel("Active-stage field MAE [pp]")
        axis[2].set_ylabel("Active-stage P21 bin error [%]")
        axis[2].set_yscale("symlog", linthresh=1)
        axis[2].axhline(1, color="0.4", linestyle=":")
        for panel in axis:
            panel.set_xlabel("Redshift")
            panel.invert_xaxis()
            panel.grid(alpha=0.2)
        axis[0].set_title(f"Seed {case['seed']}")
    figure.suptitle(
        "Direct finite-volume RTE and matched ES; fixed gas; convergence uncertified"
    )
    figure.savefig(root / "matched_comparison.png", dpi=160)
    plt.close(figure)


def _plot_self_convergence(root: Path, report: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        2 * len(report["cases"]),
        4,
        figsize=(16, 6 * len(report["cases"])),
        squeeze=False,
        constrained_layout=True,
    )
    labels = {
        "directions": "directions",
        "refinement": "grid refinement",
        "cfl": "CFL",
        "source_stride": "source stride",
    }
    for seed_index, case in enumerate(report["cases"]):
        for column, (kind, setting) in enumerate(REFINEMENT_DIMENSIONS.items()):
            field_axis, power_axis = axes[2 * seed_index : 2 * seed_index + 2, column]
            comparisons = [
                row for row in case["refinement_comparisons"] if row["kind"] == kind
            ]
            for number, row in enumerate(comparisons):
                active = [r for r in row["checkpoints"] if r["active_reference"]]
                label = f"{row['coarse'][setting]:g} → {row['fine'][setting]:g}"
                z = [r["redshift"] for r in active]
                field_axis.plot(
                    z,
                    [100 * r["field_MAE"] for r in active],
                    ".-",
                    label=label,
                    color=f"C{number}",
                )
                maxima = [
                    r["power_error"]["max"]
                    if r["power_error"]["undefined"] == 0
                    else None
                    for r in active
                ]
                power_axis.plot(
                    z,
                    [np.nan if v is None else 100 * v for v in maxima],
                    ".-",
                    label=label,
                    color=f"C{number}",
                )
            field_axis.set_title(f"Seed {case['seed']} · {labels[setting]}")
            field_axis.set_ylabel("Neutral-field MAE [pp]")
            power_axis.set_ylabel("Worst primary P21 bin [%]")
            for panel in (field_axis, power_axis):
                panel.axhline(1, color="0.4", linestyle=":", linewidth=1)
                panel.set_yscale("symlog", linthresh=1)
                panel.set_xlabel("Redshift (active reference only)")
                panel.invert_xaxis()
                panel.grid(alpha=0.2)
                if comparisons:
                    panel.legend(fontsize=8, title="Adjacent coarse → fine")
                else:
                    panel.text(
                        0.5,
                        0.5,
                        "No isolated refinement pair",
                        ha="center",
                        va="center",
                        transform=panel.transAxes,
                    )
    figure.suptitle(
        "RTE self-refinement: other settings held fixed; raw per-bin errors; no convergence certificate"
    )
    figure.savefig(root / "self_convergence.png", dpi=160)
    plt.close(figure)


def build_report(root: Path, *, make_plot: bool = True) -> dict[str, Any]:
    """Audit direct RTE histories and export source-matched diagnostic pairs.

    Parameters
    ----------
    root
        Directory containing ``matrix.json``, native fixtures and RTE field
        histories. Source-stride comparisons use the same complete background
        timeline and interpolate native rate knots in physical seconds.
    make_plot
        Save model-comparison and separate self-convergence figures.

    Returns
    -------
    dict
        Independent photon audits, per-bin power comparisons, separate
        refinement diagnostics and uncertified paired-field identities.

    Raises
    ------
    ValueError
        Input identities, source integrals, field bounds or photon ledgers fail.
    """
    root = Path(root).resolve()
    identity = json.loads((root / "matrix.json").read_text())
    tasks = identity["cases"]
    seeds = [fixture["seed"] for fixture in identity["fixtures"]]
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or len({task["case_id"] for task in tasks}) != len(tasks)
        or any(task["seed"] not in seeds for task in tasks)
    ):
        raise ValueError("Unique known fixture seeds and case identities are required")
    cases, pair_records = [], []
    for fixture in identity["fixtures"]:
        path = _resolve(root, fixture["path"])
        if _digest(path) != fixture["sha256"]:
            raise ValueError("Native fixture checksum changed")
        history = load_history(path)
        history["_fixture_sha256"] = fixture["sha256"]
        if history["metadata"].get("random_seed") != fixture["seed"]:
            raise ValueError("Native seed differs from matrix fixture identity")
        _attach_native_density(root, fixture, history)
        expected = _integrated_sources(history)
        pcs = [
            _load_case(root, task, history)
            for task in tasks
            if task["seed"] == fixture["seed"]
        ]
        if not pcs:
            raise ValueError("Each native fixture needs an RTE history")
        target = (
            max(row["task"]["refinement"] for row in pcs),
            max(row["task"]["directions"] for row in pcs),
        )
        selected = [
            row
            for row in pcs
            if tuple(row["task"][key] for key in ("refinement", "directions")) == target
            and row["task"]["source_stride"] == 1
        ]
        if not selected:
            raise ValueError(
                "No jointly finest spatial/angular RTE reference with source_stride=1 exists"
            )
        finest = min(selected, key=lambda row: row["task"]["cfl"])
        if identity.get("configuration", {}).get("profile") == "convergence" and finest[
            "task"
        ]["cfl"] != min(row["task"]["cfl"] for row in pcs):
            raise ValueError(
                "Convergence profile requires the jointly finest spatial/angular/temporal/source reference"
            )
        control = _matched_control(history, expected)
        comparisons = _compare(control, finest, history)
        for row in comparisons:
            node = row["node_index"]
            emitted = _sum(expected[node])
            unassigned = emitted - _sum(control["fields"][node]["ionized"])
            row.update(
                prescribed_emitted_photons=float(emitted),
                matched_es_emitted_minus_ionized=float(unassigned),
                matched_es_unassigned_fraction=float(unassigned / emitted)
                if emitted
                else None,
            )
        active = [row for row in comparisons if row["active_reference"]]
        raw_errors = [
            error
            for row in active
            for error, use in zip(
                row["power_error_per_bin"], row["primary_bins"], strict=True
            )
            if use
        ]
        refinements = _refinements(pcs, history)
        cases.append(
            {
                "seed": fixture["seed"],
                "fixture_sha256": fixture["sha256"],
                "selected_teacher": finest["task"],
                "photon_audits": [
                    {"task": row["task"], "checkpoints": row["audits"]} for row in pcs
                ],
                "matched_es_comparison": comparisons,
                "matched_mean_xHI_comparison": _matched_mean_neutral(
                    control, finest, history
                ),
                "matched_mean_xHI_scope": "Diagnostic only: bounded convex interpolation of neutral FIELDS between the first saved time bracket reaching each volume-mean xHI target; spectra are recomputed from interpolated fields, never interpolated. Both use the same frozen gas and constant brightness amplitude 27 mK to remove the different-epoch amplitude. No extrapolation; interpolation is an approximation, not an RTE solution or a training label.",
                "matched_es_budget_scope": "Signed unassigned ES count; no invented radiation reservoir. Finite-c RTE radiation remains a distinct physical inventory, so this is not a pure photon-loss attribution.",
                "active_matched_es_power_error": _distribution(raw_errors),
                "active_stage_covered": bool(active),
                "native_es_context": _native_context(
                    history, sorted(finest["fields"]), finest
                ),
                "refinement_comparisons": refinements,
                "self_convergence_summary": _self_summary(refinements),
            }
        )
        pair_records.append(_export_pair(root, history, finest, control, fixture))
    audits = [
        row
        for case in cases
        for history in case["photon_audits"]
        for row in history["checkpoints"]
    ]
    report = {
        "schema_version": 3,
        "profile": identity.get("configuration", {}).get("profile", "pilot"),
        "scope": "Direct monoenergetic finite-volume discrete-ordinate RTE at physical c; periodic fixed density contrast; native effective H+He atom counts with a single H-opacity absorber; no recombination or heating feedback",
        "signal": "Saturated spin temperature; no RSD; A=27 sqrt((1+z)/10) mK",
        "primary_k_definition": "All populated shells with 0.1 <= k [cMpc^-1] <= 0.6 analysis-grid Nyquist; no band-average substitution",
        "analysis_grid": "Native source grid; fine ion and radiation counts averaged in blocks to preserve original-cell count units",
        "active_stage_definition": "0.1 <= reference volume-mean xHI < 0.9",
        "ledger_scope": "Independent emitted = ionized + in-flight radiation at every saved endpoint, including snap0; periodic boundaries, zero initial photons and ions",
        "photon_ledger_tolerance": LEDGER_TOLERANCE,
        "source_history_tolerance": SOURCE_TOLERANCE,
        "radiation_scope": "Physical in-flight photons, not an unresolved numerical remainder; no requirement that radiation vanish while neutral gas remains",
        "all_photon_ledgers_passed": all(row["passed"] for row in audits),
        "maximum_global_photon_residual": max(
            row["global_fractional_residual"] for row in audits
        ),
        "maximum_source_history_residual": max(
            row["source_history_maximum_fractional_residual"] for row in audits
        ),
        "maximum_inflight_fraction": max(row["inflight_fraction"] for row in audits),
        "all_active_stages_covered": all(
            case["active_stage_covered"] for case in cases
        ),
        "convergence_certified": False,
        "convergence_reason": "Ledger closure and sampled adjacent-refinement gates are separate from certification. Reported checks do not establish an asymptotic limit, angular-orientation accuracy, source-grid/box convergence, or sub-percent physical fidelity; no automatic production-label certificate.",
        "source_history_refinement_scope": "Only source-rate knot spacing changes; linear interpolation in physical seconds restores every finest background time/redshift node. All source arms share the same background evolution.",
        "matrix_sha256": _digest(root / "matrix.json"),
        "comparison_source_sha256": _digest(Path(__file__)),
        "cases": cases,
        "pairs": pair_records,
    }
    report["summary"] = {
        "profile": report["profile"],
        "all_photon_ledgers_passed": report["all_photon_ledgers_passed"],
        "maximum_global_photon_residual": report["maximum_global_photon_residual"],
        "convergence_certified": False,
        "seeds": [
            {
                "seed": case["seed"],
                "selected_teacher": case["selected_teacher"],
                "same_time_ES_power_error": case["active_matched_es_power_error"],
                "self_convergence": case["self_convergence_summary"],
                "matched_mean_xHI": [
                    {
                        key: row[key]
                        for key in (
                            "target_xHI_volume",
                            "available",
                            "field_MAE",
                            "power_error",
                        )
                        if key in row
                    }
                    for row in case["matched_mean_xHI_comparison"]
                ],
            }
            for case in cases
        ],
    }
    _write_json(
        root / "pairs/manifest.json",
        {
            "kind": "matched-ES-direct-RTE-diagnostic-pairs",
            "convergence_certified": False,
            "scope": report["scope"],
            "cases": pair_records,
        },
    )
    _write_json(root / "matched_report.json", report)
    _write_json(root / "matched_summary.json", report["summary"])
    if make_plot:
        _plot(root, report)
        _plot_self_convergence(root, report)
    return report


def main(argv: list[str] | None = None) -> None:
    """Generate a report from a completed direct-RTE matrix.

    Parameters
    ----------
    argv
        Optional command-line arguments; defaults to the process arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    build_report(args.root)


if __name__ == "__main__":
    main()
