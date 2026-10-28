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


def _load_case(root: Path, task: dict, history: dict, expected: np.ndarray) -> dict:
    path = _resolve(root, task["path"])
    if _digest(path) != task["sha256"]:
        raise ValueError("RTE field checksum differs from matrix identity")
    refinement = task["refinement"]
    if (
        type(refinement) is not int
        or refinement < 1
        or task["directions"] not in (6, 26)
        or not np.isfinite(task["cfl"])
        or not 0 < task["cfl"] <= 1
    ):
        raise ValueError("Invalid RTE case resolution or Courant factor")
    with np.load(path, allow_pickle=False) as archive:
        raw = {name: archive[name] for name in archive.files}
    metadata = json.loads(str(raw["metadata"].item()))
    setting = {key: task[key] for key in ("directions", "refinement", "cfl")}
    if (
        metadata.get("setting") != setting
        or metadata.get("seed") != task["seed"]
        or metadata.get("fixture_sha256") != history["_fixture_sha256"]
        or metadata.get("box_len_cMpc") != history["metadata"]["box_len_cMpc"]
    ):
        raise ValueError("RTE metadata differs from matrix settings or native fixture")
    capacity = _repeat(history["capacity"], refinement)
    shape = (len(history["redshifts"]), *capacity.shape)
    if (
        not np.array_equal(raw["capacity"], capacity)
        or not np.array_equal(raw["redshifts"], history["redshifts"])
        or not np.array_equal(raw["times_seconds"], history["times_seconds"])
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
    for node, redshift in enumerate(history["redshifts"]):
        fine_source = _repeat(expected[node], refinement)
        audit = _audit_snapshot(
            raw["ionized"][node],
            raw["radiation"][node],
            raw["source_emitted"][node],
            fine_source,
            capacity,
        )
        audit.update(
            node_index=node,
            redshift=float(redshift),
            maximum_ionized_fraction_decrease=decrease,
            counts_normalization="Mean-fine-cell units; divide totals by refinement^3 for original-cell units",
        )
        fields[node] = {
            name: _coarsen(raw[name][node], len(history["capacity"]))
            for name in ("ionized", "radiation", "source_emitted")
        }
        audits.append(audit)
    return {
        "task": task,
        "fields": fields,
        "audits": audits,
        "metadata": metadata,
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


def _compare(new: dict, reference: dict, history: dict) -> list[dict]:
    capacity = history["capacity"]
    box = float(history["metadata"]["box_len_cMpc"])
    rows = []
    for node in sorted(new["fields"].keys() & reference["fields"].keys()):
        a, b = new["fields"][node], reference["fields"][node]
        xnew, xref = _neutral(a, capacity), _neutral(b, capacity)
        amplitude = 27 * np.sqrt((1 + history["redshifts"][node]) / 10)
        spectra = shell_spectra(
            {
                "teacher": amplitude * capacity * xref,
                "comparison": amplitude * capacity * xnew,
            },
            box,
        )
        # Match the NN pilot: 0.1 <= k <= 0.6 times the analysis Nyquist.
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
        spectrum = {
            key: value
            for key, value in spectra.items()
            if key
            in (
                "k_cMpc_inv",
                "modes",
                "P_teacher",
                "P_comparison",
                "ratio_comparison",
                "r_comparison",
                "P_cross_comparison",
            )
        }
        rows.append(
            {
                "node_index": node,
                "redshift": float(history["redshifts"][node]),
                "time_seconds": float(history["times_seconds"][node]),
                "reference_xHI_volume": float(xref.mean()),
                "comparison_xHI_volume": float(xnew.mean()),
                "reference_xHI_mass": float(_sum(capacity * xref) / _sum(capacity)),
                "comparison_xHI_mass": float(_sum(capacity * xnew) / _sum(capacity)),
                "field_MAE": float(np.mean(abs(xnew - xref))),
                "mass_weighted_field_MAE": float(
                    _sum(capacity * abs(xnew - xref)) / _sum(capacity)
                ),
                "primary_bins": primary.tolist(),
                "spectra": spectrum,
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
        )
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
    dimensions = {
        "angular": "directions",
        "spatial": "refinement",
        "temporal": "cfl",
    }
    for kind, key in dimensions.items():
        others = set(dimensions.values()) - {key}
        for fine in cases:
            ft = fine["task"]
            candidates = [
                coarse
                for coarse in cases
                if all(coarse["task"][other] == ft[other] for other in others)
                and (
                    coarse["task"][key] > ft[key]
                    if kind == "temporal"
                    else coarse["task"][key] < ft[key]
                )
            ]
            if not candidates:
                continue
            coarse = min(candidates, key=lambda item: abs(item["task"][key] - ft[key]))
            # Fine is the reference denominator; compare the coarse approximation.
            rows = _compare(coarse, fine, history)
            active = [row for row in rows if row["active_reference"]]
            results.append(
                {
                    "kind": kind,
                    "coarse": coarse["task"],
                    "fine": ft,
                    "checkpoints": rows,
                    "active_checkpoints": len(active),
                    "all_primary_power_bins_passed": bool(active)
                    and all(
                        row["all_primary_bins_within_one_percent"] for row in active
                    ),
                    "field_MAE_within_one_percentage_point": bool(active)
                    and all(row["field_MAE"] <= 0.01 for row in active),
                }
            )
    return results


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


def build_report(root: Path, *, make_plot: bool = True) -> dict[str, Any]:
    """Audit direct RTE histories and export source-matched diagnostic pairs.

    Parameters
    ----------
    root
        Directory containing ``matrix.json``, native fixtures and RTE field
        histories. All cases must retain the same complete endpoint timeline.
    make_plot
        Save the comparison figure alongside the JSON report.

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
            _load_case(root, task, history, expected)
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
        ]
        if not selected:
            raise ValueError("No jointly finest spatial/angular RTE reference exists")
        finest = min(selected, key=lambda row: row["task"]["cfl"])
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
        cases.append(
            {
                "seed": fixture["seed"],
                "fixture_sha256": fixture["sha256"],
                "selected_teacher": finest["task"],
                "photon_audits": [
                    {"task": row["task"], "checkpoints": row["audits"]} for row in pcs
                ],
                "matched_es_comparison": comparisons,
                "matched_es_budget_scope": "Signed unassigned ES count; no invented radiation reservoir. Finite-c RTE radiation remains a distinct physical inventory, so this is not a pure photon-loss attribution.",
                "active_matched_es_power_error": _distribution(raw_errors),
                "active_stage_covered": bool(active),
                "native_es_context": _native_context(
                    history, list(range(len(history["redshifts"]))), finest
                ),
                "refinement_comparisons": _refinements(pcs, history),
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
        "schema_version": 2,
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
        "convergence_reason": "Two angular/spatial settings do not establish successive-refinement convergence; field, all-bin P21 and temporal diagnostics are reported separately",
        "matrix_sha256": _digest(root / "matrix.json"),
        "comparison_source_sha256": _digest(Path(__file__)),
        "cases": cases,
        "pairs": pair_records,
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
    if make_plot:
        _plot(root, report)
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
