"""Measure native ES-to-RT residuals before fitting a neural network.

A mean/centred MSE decomposition diagnoses coherent history offsets without
changing any labels. It is not a matched-ionization morphology comparison.
Source-budget diagnostics and HII inventory errors are not photon closure.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from research.es_pc_emulator.initial_state import DEFAULT_POLICY
from research.es_pc_emulator.pairs import (
    _audit_manifest,
    _load_es,
    _write_json,
    file_hash,
    load_pair,
)
from research.pc_evolution.native_history import load_history

DEFAULT_INITIAL_TOLERANCE = DEFAULT_POLICY["max_mass_weighted_error"]


def residual_metrics(xhi_es, xhi_rt, density):
    """Return exact volume/mass MSE decompositions for the HII residual."""
    es, rt, rho = (np.asarray(a, dtype=np.float64) for a in (xhi_es, xhi_rt, density))
    if (es.shape != rt.shape or es.shape != rho.shape or not es.size
            or not all(np.isfinite(a).all() for a in (es, rt, rho))
            or np.any(rho <= 0) or np.any((es < 0) | (es > 1))
            or np.any((rt < 0) | (rt > 1))):
        raise ValueError("Finite matching fractions and strictly positive density required")
    error = rt - es  # xHII_ES - xHII_RT
    weights = rho / rho.sum(dtype=np.float64)
    bias = float(error.mean())
    mass_bias = float(np.sum(weights * error))
    mse = float(np.mean(error**2))
    mass_mse = float(np.sum(weights * error**2))
    return {
        "ES_Q_volume": float(np.mean(1 - es)),
        "RT_Q_volume": float(np.mean(1 - rt)),
        "ES_Q_mass_reference": float(np.sum(weights * (1 - es))),
        "RT_Q_mass_reference": float(np.sum(weights * (1 - rt))),
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(mse)),
        "mass_weighted_MAE": float(np.sum(weights * np.abs(error))),
        "mass_weighted_RMSE": float(np.sqrt(mass_mse)),
        "delta_Q_volume": bias,
        "delta_Q_mass": mass_bias,
        "centred_residual_RMSE": float(np.sqrt(np.mean((error - bias)**2))),
        "mass_centred_residual_RMSE": float(np.sqrt(np.sum(weights * (error - mass_bias)**2))),
        "mean_bias_fraction_of_MSE": bias**2 / mse if mse > 0 else 0.0,
        "mass_mean_bias_fraction_of_MSE": mass_bias**2 / mass_mse if mass_mse > 0 else 0.0,
    }


def _sources(paths, loader):
    result = {}
    for path in paths:
        path = Path(path).resolve()
        digest = file_hash(path)
        result[digest] = (path, loader(path))
    return result


def _context(pair, histories, sidecars, threshold):
    meta = pair["metadata"]
    result = {
        "history_available": False,
        "density_sidecar_available": False,
        "initial_state_mismatch": None,
        "initial_ES_Q_volume": None,
        "initial_ES_Q_mass_reference": None,
        "initial_teacher_xHII": None,
        "initial_delta_Q_mass": None,
        "initial_mass_weighted_MAE": None,
        "initial_state_compatible_at_export": None,
        "native_nion_proxy_over_reference_nuclei": None,
        "integrated_over_native_proxy": None,
        "density_reference_vs_native_RMSE": None,
        "ES_Q_mass_native_density": None,
    }
    saved = meta.get("initial_state_audit", {})
    if saved:
        mapping = {
            "initial_ES_Q_volume": "initial_es_Q_volume",
            "initial_ES_Q_mass_reference": "initial_es_Q_mass_reference_gas",
            "initial_teacher_xHII": "teacher_initial_xHII",
            "initial_delta_Q_mass": "initial_delta_Q_mass",
            "initial_mass_weighted_MAE": "mass_weighted_initial_MAE",
            "initial_state_compatible_at_export": "compatible",
        }
        result.update({key: saved.get(value) for key, value in mapping.items()})
        if result["initial_mass_weighted_MAE"] is not None:
            result["initial_state_mismatch"] = result["initial_mass_weighted_MAE"] > threshold
    history_entry = histories.get(meta["history_sha256"])
    if history_entry is None:
        return result
    history = history_entry[1]
    index = int(meta["snapshot_id"])
    if (history["metadata"].get("random_seed") != meta["seed"]
            or not np.isclose(history["redshifts"][index], meta["redshift"], rtol=0, atol=1e-8)):
        raise ValueError("Audit history does not match sample seed/redshift")
    rho = history["capacity"]
    initial = 1.0 - np.asarray(history["native_xhi"][0], dtype=np.float64)
    result.update(
        history_available=True,
        initial_ES_Q_volume=float(initial.mean()),
        initial_ES_Q_mass_reference=float(np.sum(rho * initial) / rho.sum()),
    )
    initial_state = meta.get("physics", {}).get("initial_state") or {}
    rt_initial = initial_state.get("xHII")
    if isinstance(rt_initial, (int, float)) and np.isfinite(rt_initial):
        difference = result["initial_ES_Q_mass_reference"] - rt_initial
        initial_mae = float(np.sum(rho * np.abs(initial - rt_initial)) / rho.sum())
        result.update(initial_teacher_xHII=float(rt_initial),
                      initial_delta_Q_mass=float(difference),
                      initial_mass_weighted_MAE=initial_mae,
                      initial_state_mismatch=initial_mae > threshold)
    provenance = history["metadata"].get("source_provenance", [])
    if index < len(provenance):
        proxy = provenance[index].get("native_n_ion_sum")
        if proxy is not None and np.isfinite(proxy) and proxy >= 0:
            result["native_nion_proxy_over_reference_nuclei"] = float(proxy / rho.sum())
            if proxy > 0:
                result["integrated_over_native_proxy"] = float(pair["photons"].sum() / proxy)
    sidecar = sidecars.get(meta["es_sidecar_sha256"])
    if sidecar is not None:
        es, _ = _load_es(sidecar[0], history)
        native_rho = es["density_es"][index]
        result.update(
            density_sidecar_available=True,
            density_reference_vs_native_RMSE=float(np.sqrt(np.mean((native_rho - rho)**2))),
            ES_Q_mass_native_density=float(np.sum(native_rho * (1 - pair["xhi_es"])) / native_rho.sum()),
        )
    return result


def _any_mismatch(values):
    if any(value is True for value in values):
        return True
    return None if any(value is None for value in values) else False


def _all_compatible(values):
    if any(value is False for value in values):
        return False
    return None if any(value is None for value in values) else True


def audit_dataset(dataset, output, *, histories=(), sidecars=(),
                  initial_tolerance=DEFAULT_INITIAL_TOLERANCE):
    """Audit a complete or single-seed pair manifest into a fresh report directory.

    Optional source histories and ES sidecars are selected by their stored
    SHA256 identities, never by seed alone. Missing context remains unknown.
    """
    if not np.isfinite(initial_tolerance) or initial_tolerance < 0:
        raise ValueError("Initial-state tolerance must be finite and nonnegative")
    path, output = Path(dataset).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads(path.read_text())
    for record in manifest.get("samples", []):
        if not Path(record["path"]).is_absolute():
            record["path"] = str((path.parent / record["path"]).resolve())
    _audit_manifest(manifest, require_splits=False)
    history_map = _sources(histories, load_history)
    sidecar_map = _sources(sidecars, lambda path: None)
    rows = []
    for record in manifest["samples"]:
        pair = load_pair(record)
        row = {key: record[key] for key in ("seed", "split", "snapshot_id", "redshift", "dimension", "box_len_cMpc")}
        row.update(residual_metrics(pair["xhi_es"], pair["xhi_c2ray"], pair["density"]))
        row.update(_context(pair, history_map, sidecar_map, initial_tolerance))
        row["integrated_emitted_over_reference_nuclei"] = float(pair["photons"].sum() / pair["density"].sum())
        row["convergence_certified"] = pair["metadata"].get("convergence_certified", False)
        rows.append(row)
    groups = []
    for split, seed in sorted({(row["split"], row["seed"]) for row in rows}):
        subset = [row for row in rows if row["split"] == split and row["seed"] == seed]
        groups.append({"split": split, "seed": seed, "snapshots": len(subset),
                       "mean_snapshot_MAE": float(np.mean([row["MAE"] for row in subset])),
                       "max_snapshot_MAE": max(row["MAE"] for row in subset),
                       "initial_state_mismatch": _any_mismatch([row["initial_state_mismatch"] for row in subset]),
                       "initial_state_compatible_at_export": _all_compatible([row["initial_state_compatible_at_export"] for row in subset])})
    report = {
        "schema_version": 1,
        "kind": "es-to-c2ray-residual-audit",
        "dataset_sha256": file_hash(path),
        "residual_convention": "xHII_ES - xHII_RT; MAE is an absolute fraction, not a relative percent",
        "decomposition": "MSE = mean residual squared + centred residual variance, also with reference-density weights; centring is diagnostic and does not alter the field",
        "budget_scope": "HII inventory relative to teacher, not complete photon closure; cumulative emitted photons and native n_ion proxy have different histories",
        "initial_tolerance": float(initial_tolerance),
        "initial_state_interpretation": "initial_state_mismatch uses this report's initial_tolerance; initial_state_compatible_at_export preserves the exporter policy. An already-ionized native ES field paired with a near-neutral RT start indicates omitted pre-start evolution, not a quantitative causal attribution",
        "groups": groups,
        "samples": rows,
    }
    output.mkdir(parents=True)
    _write_json(output / "summary.json", report)
    with (output / "samples.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report


def main():
    """Write the residual report from audited paired data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--history", dest="histories", action="append", type=Path, default=[])
    parser.add_argument("--es-sidecar", dest="sidecars", action="append", type=Path, default=[])
    parser.add_argument("--initial-tolerance", type=float, default=DEFAULT_INITIAL_TOLERANCE)
    result = audit_dataset(**vars(parser.parse_args()))
    print(json.dumps(result["groups"], indent=2))  # noqa: T201


if __name__ == "__main__":
    main()
