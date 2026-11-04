"""Compare nested grids using shared native inputs and conservative restriction.

This measures combined input-grid and transport-grid sensitivity. Two grid levels
neither certify convergence nor distinguish input substructure from transport
truncation. Every method is compared on the same coarse FFT grid without shifting
its mean neutral fraction or renormalizing its photon budget.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from research.pc_analytic.compare import _block_mean, _digest, _write_json
from research.pc_field_emulator.evaluation import field_metrics, shell_spectra

METHODS = ("analytic", "script", "c2ray")
SCIENTIFIC_FILES = tuple(
    "research/pc_analytic/" + name
    for name in (
        "absorption.py",
        "solver.py",
        "transport_kernel.cpp",
        "c2ray_reference.py",
        "c2ray_driver.F90",
        "compare.py",
    )
)
_OPERATIONAL_CONFIG = {
    "coarsen_to",
    "coarsening_scope",
    "output_dimension",
    "input_run",
    "snapshot_directories",
    "redshifts",
    "seeds",
    "threads",
    "mfp_rays",
    "script_timeout_seconds",
    "c2ray_cache_directory",
    "grid_reference_run",
    "grid_reference",
    "grid_reference_identity",
    "grid_reference_sha256",
}
_ORIGINAL_METADATA = (
    "receipt_sha256",
    "files",
    "archived_array_sha256",
    "native_dimension",
    "inputs",
    "box_len",
    "nH_mean_comoving_cm3",
    "source_convention",
    "count_conversion",
    "density_definition",
    "density_formula",
    "brightness_mode",
)


def _results(path: str | Path) -> Path:
    root = Path(path).resolve()
    return root if (root / "comparison_report.json").is_file() else root / "results"


def _science_config(identity: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(identity["config"])
    for name in _OPERATIONAL_CONFIG:
        config.pop(name, None)
    backend = config.get("c2ray", {})
    for name in ("timeout", "compiler", "upstream", "build_dir"):
        backend.pop(name, None)
    return config


def _selected_rows(
    report: dict[str, Any], redshifts: list[float] | None
) -> dict[tuple[int, float], dict[str, Any]]:
    rows: dict[tuple[int, float], dict[str, Any]] = {}
    for row in report["snapshots"]:
        key = (int(row["seed"]), float(row["redshift"]))
        if redshifts is not None and key[1] not in redshifts:
            continue
        if key in rows:
            raise ValueError(f"Duplicate snapshot: {key}")
        rows[key] = row
    if not rows or (
        redshifts is not None and set(redshifts) != {key[1] for key in rows}
    ):
        raise ValueError("Requested redshift selection is missing from a run")
    return rows


def _load_run(
    path: str | Path, redshifts: list[float] | None
) -> tuple[Path, dict[str, Any], dict[tuple[int, float], dict[str, Any]]]:
    root = _results(path)
    if not (root / "COMPLETED").is_file():
        raise ValueError(f"Run has no COMPLETED marker: {root}")
    report = json.loads((root / "comparison_report.json").read_text())
    identity = json.loads((root / "identity.json").read_text())
    if not report.get("complete") or not report.get("three_methods_completed"):
        raise ValueError(f"Three-method run is incomplete: {root}")
    if report["identity"] != identity:
        raise ValueError(f"Report and stored identity differ: {root}")
    rows = _selected_rows(report, redshifts)
    for key, row in rows.items():
        folder = root / f"seed{key[0]}" / ("z" + f"{key[1]:g}".replace(".", "p"))
        receipt = json.loads((folder / "receipt.json").read_text())
        if receipt["identity_sha256"] != _digest(root / "identity.json"):
            raise ValueError(f"Identity checksum mismatch: {folder}")
        for name in ("fields.npz", "metrics.json"):
            if receipt["files"].get(name) != _digest(folder / name):
                raise ValueError(f"Snapshot checksum mismatch: {folder / name}")
        if row != json.loads((folder / "metrics.json").read_text()):
            raise ValueError(f"Report differs from verified snapshot metrics: {folder}")
        if not row.get("complete") or not row.get("three_methods_completed"):
            raise ValueError(f"Snapshot is incomplete: {folder}")
    return root, report, rows


def _same_implementation(coarse: dict[str, Any], fine: dict[str, Any]) -> None:
    if _science_config(coarse["identity"]) != _science_config(fine["identity"]):
        raise ValueError("Scientific settings differ between resolutions")
    for name in SCIENTIFIC_FILES:
        first = coarse["identity"]["implementation_sha256"].get(name)
        second = fine["identity"]["implementation_sha256"].get(name)
        if first is None or second is None or first != second:
            raise ValueError(f"Scientific implementation changed: {name}")
    for name in ("common_physics", "brightness_definition"):
        if coarse.get(name) is None or coarse.get(name) != fine.get(name):
            raise ValueError(f"Scientific settings differ: {name}")
    if coarse["identity"].get("script_provenance") != fine["identity"].get(
        "script_provenance"
    ):
        raise ValueError("SCRIPT upstream implementation differs")


def _same_native_input(coarse: dict[str, Any], fine: dict[str, Any]) -> None:
    for name in _ORIGINAL_METADATA:
        if name not in coarse["input"] or coarse["input"][name] != fine["input"].get(
            name
        ):
            raise ValueError(f"Different original input: {name}")
    for name in ("commit", "files", "driver_sha256", "flags"):
        first = (coarse.get("c2ray_provenance") or {}).get("identity", {}).get(name)
        second = (fine.get("c2ray_provenance") or {}).get("identity", {}).get(name)
        if first is None or first != second:
            raise ValueError(f"C2-Ray scientific implementation differs: {name}")
    first_version = coarse["c2ray_provenance"]["identity"].get("compiler_version")
    second_version = fine["c2ray_provenance"]["identity"].get("compiler_version")
    if (
        first_version is not None
        and second_version is not None
        and first_version != second_version
    ):
        raise ValueError("C2-Ray compiler version differs between resolutions")


def _load_fields(root: Path, row: dict[str, Any]) -> dict[str, np.ndarray]:
    folder = (
        root / f"seed{row['seed']}" / ("z" + f"{row['redshift']:g}".replace(".", "p"))
    )
    required = ["source", "capacity"] + [
        f"{method}_{field}" for method in METHODS for field in ("xHI", "brightness")
    ]
    with np.load(folder / "fields.npz", allow_pickle=False) as archive:
        fields = {name: np.asarray(archive[name], dtype=float) for name in required}
        for name in ("native_es_xHI", "native_es_brightness"):
            if name in archive:
                fields[name] = np.asarray(archive[name], dtype=float)
    n = int(row["input"]["dimension"])
    for name, value in fields.items():
        if value.shape != (n, n, n) or not np.all(np.isfinite(value)):
            raise ValueError(f"Invalid cubic field: {name}")
        if name.endswith("xHI") and (
            np.min(value) < -1e-12 or np.max(value) > 1 + 1e-12
        ):
            raise ValueError(f"Neutral fraction out of bounds: {name}")
    if np.any(fields["source"] < 0) or np.any(fields["capacity"] <= 0):
        raise ValueError("Require nonnegative photons and positive atom capacities")
    return fields


def restrict_fields(
    fields: dict[str, np.ndarray], target: int
) -> dict[str, np.ndarray]:
    """Restrict native counts, neutral fractions and brightness to a nested grid.

    Parameters
    ----------
    fields
        Source and atom-capacity arrays in mean-density cell count units, plus
        neutral fractions and brightness fields for all three methods.
    target
        Coarse dimension, an integer divisor of the input dimension.

    Returns
    -------
    dict
        Block-mean counts, atom-weighted neutral fractions and volume-mean
        brightness temperatures. No field normalization is applied.
    """
    capacity = np.asarray(fields["capacity"], dtype=float)
    n = capacity.shape[0]
    if type(target) is not int or target < 2 or target > n or n % target:
        raise ValueError("Target dimension must divide the finer dimension")
    if capacity.shape != (n, n, n) or np.any(capacity <= 0):
        raise ValueError("Capacity must be a positive cubic array")
    restricted = {
        name: _block_mean(fields[name], target) for name in ("source", "capacity")
    }
    for method in (*METHODS, "native_es"):
        if f"{method}_xHI" not in fields or f"{method}_brightness" not in fields:
            continue
        restricted[f"{method}_xHI"] = (
            _block_mean(capacity * fields[f"{method}_xHI"], target)
            / restricted["capacity"]
        )
        restricted[f"{method}_brightness"] = _block_mean(
            fields[f"{method}_brightness"], target
        )
    return restricted


def _spectra(
    first: np.ndarray, reference: np.ndarray, box_len: float, label: str
) -> dict[str, Any]:
    result = shell_spectra(
        {"teacher": reference, label: first, "difference": first - reference}, box_len
    )
    result = {
        key.replace("teacher", "reference"): value for key, value in result.items()
    }
    result["below_half_coarse_nyquist"] = [
        k < 0.5 * result["k_max_cMpc_inv"] for k in result["k_cMpc_inv"]
    ]
    result["valid_ratio_bins"] = [
        value is not None for value in result[f"ratio_{label}"]
    ]
    result["reference_is_exact"] = False
    return result


def _means(fields: dict[str, np.ndarray], method: str) -> dict[str, float]:
    neutral = fields[f"{method}_xHI"]
    return {
        "xHI_volume": float(np.mean(neutral)),
        "xHI_mass": float(
            np.sum(fields["capacity"] * neutral, dtype=np.longdouble)
            / np.sum(fields["capacity"], dtype=np.longdouble)
        ),
        "brightness_volume_mK": float(np.mean(fields[f"{method}_brightness"])),
    }


def _field_pair(
    first: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    method: str,
    box_len: float,
) -> dict[str, Any]:
    return {
        "field": field_metrics(
            1 - first[f"{method}_xHI"],
            1 - reference[f"{method}_xHI"],
            reference["capacity"],
        ),
        "spectra_xHI": _spectra(
            first[f"{method}_xHI"],
            reference[f"{method}_xHI"],
            box_len,
            "fine_restricted",
        ),
        "spectra_21cm": _spectra(
            first[f"{method}_brightness"],
            reference[f"{method}_brightness"],
            box_len,
            "fine_restricted",
        ),
    }


def _method_gaps(fields: dict[str, np.ndarray], box_len: float) -> dict[str, Any]:
    return {
        f"analytic_vs_{reference}": {
            "field": field_metrics(
                1 - fields["analytic_xHI"],
                1 - fields[f"{reference}_xHI"],
                fields["capacity"],
            ),
            "spectra_xHI": _spectra(
                fields["analytic_xHI"], fields[f"{reference}_xHI"], box_len, "analytic"
            ),
            "spectra_21cm": _spectra(
                fields["analytic_brightness"],
                fields[f"{reference}_brightness"],
                box_len,
                "analytic",
            ),
        }
        for reference in ("script", "c2ray")
    }


def _input_audit(
    coarse: dict[str, np.ndarray],
    restricted: dict[str, np.ndarray],
    fine: dict[str, np.ndarray],
    coarse_row: dict[str, Any],
    fine_row: dict[str, Any],
) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for name in ("source", "capacity"):
        if not np.allclose(coarse[name], restricted[name], rtol=2e-12, atol=0):
            raise ValueError(
                f"Coarse {name} differs from conservatively restricted fine input"
            )
        totals = [
            float(
                np.sum(fields[name], dtype=np.longdouble)
                * row["input"]["atoms_per_count_unit"]
            )
            for fields, row in ((coarse, coarse_row), (fine, fine_row))
        ]
        if (
            not np.all(np.isfinite(totals))
            or min(totals) < 0
            or (name == "capacity" and min(totals) == 0)
        ):
            raise ValueError(f"Invalid physical {name} totals")
        if not np.isclose(*totals, rtol=2e-12, atol=0):
            raise ValueError(f"Physical {name} totals differ between resolutions")
        record[name] = {
            "coarse_physical_count": totals[0],
            "fine_physical_count": totals[1],
            "physical_total_relative_difference": (totals[1] - totals[0]) / totals[0]
            if totals[0]
            else None,
            "restricted_maximum_absolute_difference": float(
                np.max(np.abs(coarse[name] - restricted[name]))
            ),
        }
    record["passed"] = True
    return record


def _snapshot(
    coarse_root: Path,
    fine_root: Path,
    coarse_row: dict[str, Any],
    fine_row: dict[str, Any],
) -> dict[str, Any]:
    _same_native_input(coarse_row, fine_row)
    coarse = _load_fields(coarse_root, coarse_row)
    fine = _load_fields(fine_root, fine_row)
    nc, nf = coarse["source"].shape[0], fine["source"].shape[0]
    if nf <= nc or nf % nc:
        raise ValueError("The second run must have a strictly finer nested grid")
    restricted = restrict_fields(fine, nc)
    box_len = float(coarse_row["input"]["box_len"])
    if not np.isfinite(box_len) or box_len <= 0:
        raise ValueError("Box length must be finite and positive")
    result = {
        "seed": coarse_row["seed"],
        "redshift": coarse_row["redshift"],
        "coarse_dimension": nc,
        "fine_dimension": nf,
        "box_len_cMpc": box_len,
        "coarse_cell_cMpc": box_len / nc,
        "fine_cell_cMpc": box_len / nf,
        "half_coarse_nyquist_cMpc_inv": 0.5 * np.pi * nc / box_len,
        "input_audit": _input_audit(coarse, restricted, fine, coarse_row, fine_row),
        "means": {
            label: {method: _means(fields, method) for method in METHODS}
            for label, fields in (
                ("coarse", coarse),
                ("fine_native", fine),
                ("fine_restricted", restricted),
            )
        },
        "self_grid_changes": {
            method: _field_pair(restricted, coarse, method, box_len)
            for method in METHODS
        },
        "method_gaps": {
            label: _method_gaps(fields, box_len)
            for label, fields in (("coarse", coarse), ("fine_restricted", restricted))
        },
        "budgets": {"coarse": coarse_row["budgets"], "fine": fine_row["budgets"]},
        "audits": {"coarse": coarse_row["audits"], "fine": fine_row["audits"]},
        "original_input": {
            name: coarse_row["input"][name] for name in _ORIGINAL_METADATA
        },
    }
    if (
        "native_es_xHI" in coarse
        and "native_es_xHI" in restricted
        and "native_es_brightness" in coarse
        and "native_es_brightness" in restricted
    ):
        result["native_es_restriction_control"] = {
            "scope": "Archived native ES only undergoes conservative averaging; this checks restriction, not an ES solver rerun",
            **_field_pair(restricted, coarse, "native_es", box_len),
        }
    return result


def _plot(output: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(rows), 3, squeeze=False, figsize=(16, 4.6 * len(rows))
    )
    for row, panels in zip(rows, axes, strict=True):
        for method in METHODS:
            spectrum = row["self_grid_changes"][method]["spectra_21cm"]
            panels[0].plot(
                spectrum["k_cMpc_inv"],
                spectrum["ratio_fine_restricted"],
                marker="o",
                label=method,
            )
        for reference, panel in zip(("script", "c2ray"), panels[1:], strict=True):
            for label, style in (("coarse", "--"), ("fine_restricted", "-")):
                spectrum = row["method_gaps"][label][f"analytic_vs_{reference}"][
                    "spectra_21cm"
                ]
                panel.plot(
                    spectrum["k_cMpc_inv"],
                    spectrum["ratio_analytic"],
                    style,
                    marker="o",
                    label=label.replace("_", " "),
                )
            panel.set_ylabel(f"P analytic / P {reference}")
        panels[0].set_ylabel("P fine restricted / P coarse")
        for panel in panels:
            panel.axhline(1, color="0.5", linewidth=1)
            panel.axvline(
                row["half_coarse_nyquist_cMpc_inv"],
                color="0.5",
                linestyle=":",
                label="half coarse Nyquist",
            )
            panel.set_xlabel(r"$k$ [cMpc$^{-1}$]")
            panel.set_title(
                f"seed {row['seed']}, z={row['redshift']:g}, N={row['coarse_dimension']} to {row['fine_dimension']}"
            )
            panel.legend(fontsize=8)
            panel.grid(alpha=0.2)
    figure.suptitle(
        "Shared original inputs; atom-weighted xHI and volume-weighted brightness restriction"
    )
    figure.tight_layout()
    temporary = output / "grid_comparison.tmp.png"
    figure.savefig(temporary, dpi=170)
    temporary.replace(output / "grid_comparison.png")
    plt.close(figure)


def run(
    coarse_run: str | Path,
    fine_run: str | Path,
    output: str | Path,
    *,
    redshifts: list[float] | None = None,
) -> dict[str, Any]:
    """Audit and compare two completed shared-input nested-grid calculations.

    Parameters
    ----------
    coarse_run, fine_run
        Completed comparison run roots or their ``results`` directories.
    output
        New analysis directory. An existing identical analysis may be regenerated.
    redshifts
        Explicit snapshot selection. Without selection the complete seed/redshift
        sets must match; with selection every requested redshift must exist in
        both runs and the selected seed/redshift sets must match.

    Returns
    -------
    dict
        Verified input counts, independent budgets, grid-sensitive field and
        spectral comparisons, plus immutable hashes of the input reports.
    """
    if redshifts is not None and (
        not redshifts
        or not all(np.isfinite(z) and z >= 0 for z in redshifts)
        or len(set(redshifts)) != len(redshifts)
    ):
        raise ValueError("Require unique finite nonnegative redshifts")
    coarse_root, coarse_report, coarse_rows = _load_run(coarse_run, redshifts)
    fine_root, fine_report, fine_rows = _load_run(fine_run, redshifts)
    if coarse_rows.keys() != fine_rows.keys():
        raise ValueError("Selected seed/redshift sets differ between runs")
    _same_implementation(coarse_report, fine_report)
    output = Path(output).resolve()
    if any(
        output == root.parent or output.is_relative_to(root)
        for root in (coarse_root, fine_root)
    ):
        raise ValueError("Analysis output must not overwrite a comparison run")
    identity = {
        "coarse_results": str(coarse_root),
        "fine_results": str(fine_root),
        "report_sha256": {
            label: _digest(root / "comparison_report.json")
            for label, root in (("coarse", coarse_root), ("fine", fine_root))
        },
        "identity_sha256": {
            label: _digest(root / "identity.json")
            for label, root in (("coarse", coarse_root), ("fine", fine_root))
        },
        "redshifts": redshifts,
        "analysis_sha256": _digest(Path(__file__)),
        "analysis_dependency_sha256": {
            name: _digest(Path(__file__).resolve().parents[1] / name)
            for name in ("pc_field_emulator/evaluation.py", "pc_analytic/compare.py")
        },
    }
    report_path = output / "grid_report.json"
    if (
        report_path.exists()
        and json.loads(report_path.read_text())["identity"] != identity
    ):
        raise ValueError("Analysis inputs changed; use a fresh output directory")
    snapshots = [
        _snapshot(coarse_root, fine_root, coarse_rows[key], fine_rows[key])
        for key in sorted(coarse_rows)
    ]
    report = {
        "schema": 1,
        "identity": identity,
        "complete": True,
        "convergence_certified": False,
        "comparison_is_truth_error": False,
        "scope": "Combined sensitivity to input-grid and transport-grid resolution, including restored source and density substructure; two levels do not certify convergence",
        "restriction": "xHI weighted by fine atom capacity; brightness volume-averaged; same coarse FFT geometry for all primary comparisons; no mean matching or photon renormalization",
        "low_k_scope": "k strictly below half coarse Nyquist is a diagnostic mask, not an accuracy guarantee",
        "budget_scope": "Original independent field budgets retained; C2-Ray unknown escape/reservoir and strict residual remain null when unavailable",
        "snapshots": snapshots,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "COMPLETED").unlink(missing_ok=True)
    _plot(output, snapshots)
    _write_json(report_path, report)
    _write_json(
        output / "receipt.json",
        {
            "files": {
                name: _digest(output / name)
                for name in ("grid_report.json", "grid_comparison.png")
            }
        },
    )
    temporary = output / "COMPLETED.tmp"
    temporary.write_text("Grid comparison completed; convergence not certified.\n")
    temporary.replace(output / "COMPLETED")
    return report


def main() -> None:
    """Run the verified grid analysis from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse-run", required=True, type=Path)
    parser.add_argument("--fine-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--redshifts",
        help="Comma-separated explicit redshifts; default requires identical full snapshot sets",
    )
    args = parser.parse_args()
    redshifts = (
        None
        if args.redshifts is None
        else [float(value) for value in args.redshifts.split(",")]
    )
    run(args.coarse_run, args.fine_run, args.output, redshifts=redshifts)


if __name__ == "__main__":
    main()
