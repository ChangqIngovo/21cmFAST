"""Audit a frozen emulator's geometric symmetries and count-unit invariance.

Only the three inference arrays are used. Teacher ray quadratures and teacher
labels are not part of this test. Float32 feature tolerances are explicit;
independent photon accounting is checked separately at its tighter tolerance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from itertools import permutations, product
from pathlib import Path
from typing import Any

import numpy as np

from research.pc_analytic_emulator.benchmark import _model_hashes, independent_ledger
from research.pc_analytic_emulator.model import Emulator
from research.pc_analytic_emulator.predict import _read_inputs

FIELDS = ("xHI", "xHII", "raw_xHI", "latent_xHII")
COUNT_KEYS = ("predicted_absorbed_photons", "predicted_unabsorbed_photons")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _comparison(
    actual: dict,
    expected: dict,
    source: np.ndarray,
    capacity: np.ndarray,
    *,
    atol: float,
    rtol: float,
    count_unit: float = 1,
) -> dict:
    fields = {}
    for key in FIELDS:
        difference = np.asarray(actual[key]) - expected[key]
        fields[key] = {
            "maximum_absolute_difference": float(np.max(np.abs(difference))),
            "RMSE": float(np.sqrt(np.mean(difference**2))),
            "passed": bool(
                np.allclose(actual[key], expected[key], atol=atol, rtol=rtol)
            ),
        }
    count_normalizer = max(
        np.sum(source, dtype=np.longdouble) / count_unit,
        np.sum(capacity, dtype=np.longdouble) / count_unit,
        np.longdouble(np.finfo(float).tiny),
    )
    count_errors = {
        key: float(
            abs(np.longdouble(actual[key]) / count_unit - expected[key])
            / count_normalizer
        )
        for key in COUNT_KEYS
    }
    ledger = independent_ledger(
        source,
        capacity,
        actual["xHI"],
        actual["predicted_unabsorbed_photons"],
        actual["predicted_absorbed_photons"],
    )
    efficiency_error = abs(
        float(actual["absorption_efficiency"]) - expected["absorption_efficiency"]
    )
    return {
        "fields": fields,
        "count_error_over_max_source_or_atoms": count_errors,
        "absolute_efficiency_error": efficiency_error,
        "independent_photon_ledger": ledger,
        "passed": all(value["passed"] for value in fields.values())
        and max(count_errors.values()) <= 1e-10
        and efficiency_error <= 1e-10
        and ledger["passed"],
    }


def audit_model(
    model: Emulator,
    source: np.ndarray,
    capacity: np.ndarray,
    native_es_xHI: np.ndarray,
    redshift: float,
    box_len: float,
    *,
    atol: float = 1e-7,
    rtol: float = 1e-7,
    count_units: tuple[float, ...] = (1e-6, 1e-3, 1e3, 1e6),
) -> tuple[dict[str, Any], dict]:
    """Check 48 cubic transforms, periodic translations and count-unit changes.

    Returns an explicit pass/fail report and the unmodified original prediction.
    All transforms act on source, capacity and native ES together. Geometry and
    redshift remain fixed; common count scaling changes only inventory units.
    No model weights are changed, trained or selected.
    """
    if any(not np.isfinite(value) or value < 0 for value in (atol, rtol)):
        raise ValueError("Invariance tolerances must be finite and nonnegative")
    if not count_units or any(
        not np.isfinite(unit) or unit <= 0 for unit in count_units
    ):
        raise ValueError("Count-unit factors must be finite and strictly positive")
    arrays = tuple(
        np.asarray(a, dtype=float) for a in (source, capacity, native_es_xHI)
    )
    baseline = model.predict(*arrays, redshift, box_len)
    rows = []
    for order in permutations(range(3)):
        for reflected in product((False, True), repeat=3):
            axes = tuple(axis for axis, flip in enumerate(reflected) if flip)
            transformed = tuple(np.flip(a.transpose(order), axis=axes) for a in arrays)
            expected = dict(baseline)
            expected.update(
                {
                    key: np.flip(baseline[key].transpose(order), axis=axes)
                    for key in FIELDS
                }
            )
            actual = model.predict(*transformed, redshift, box_len)
            rows.append(
                {
                    "kind": "cube_symmetry",
                    "axis_permutation": list(order),
                    "reflected_axes": list(axes),
                    **_comparison(
                        actual, expected, *transformed[:2], atol=atol, rtol=rtol
                    ),
                }
            )
    for shift in ((1, -2, 3), (0, 0, 1), (-1, 0, 0)):
        transformed = tuple(np.roll(a, shift, axis=(0, 1, 2)) for a in arrays)
        expected = dict(baseline)
        expected.update(
            {key: np.roll(baseline[key], shift, axis=(0, 1, 2)) for key in FIELDS}
        )
        actual = model.predict(*transformed, redshift, box_len)
        rows.append(
            {
                "kind": "periodic_translation",
                "cell_shift": list(shift),
                **_comparison(actual, expected, *transformed[:2], atol=atol, rtol=rtol),
            }
        )
    for unit in count_units:
        scaled = arrays[0] * unit, arrays[1] * unit, arrays[2]
        actual = model.predict(*scaled, redshift, box_len)
        rows.append(
            {
                "kind": "count_units",
                "count_unit": unit,
                **_comparison(
                    actual, baseline, *scaled[:2], atol=atol, rtol=rtol, count_unit=unit
                ),
            }
        )
    report = {
        "passed": all(row["passed"] for row in rows),
        "dimension": arrays[0].shape[0],
        "redshift": redshift,
        "box_len_cMpc": box_len,
        "checks": len(rows),
        "field_atol": atol,
        "field_rtol": rtol,
        "budget_atol_over_max_inventory": 1e-10,
        "maximum_field_absolute_difference": max(
            value["maximum_absolute_difference"]
            for row in rows
            for value in row["fields"].values()
        ),
        "maximum_photon_residual": max(
            row["independent_photon_ledger"]["relative_photon_residual"] for row in rows
        ),
        "maximum_budget_error_over_inventory": max(
            value
            for row in rows
            for value in row["count_error_over_max_source_or_atoms"].values()
        ),
        "rows": rows,
    }
    return report, baseline


def run(
    model: str | Path,
    input_path: str | Path,
    output: str | Path,
    *,
    redshift: float,
    box_len: float,
    atol: float = 1e-7,
    rtol: float = 1e-7,
) -> dict:
    """Audit saved fitted weights against a native inference archive only."""
    model, input_path, output = (
        Path(path).resolve() for path in (model, input_path, output)
    )
    if output.exists() or output.is_relative_to(model) or output == input_path:
        raise ValueError("Audit output must be new and outside the fitted model")
    model_before, input_before = _model_hashes(model), _digest(input_path)
    output.mkdir(parents=True)
    manifest = {
        "model": str(model),
        "model_files_sha256": model_before,
        "input": str(input_path),
        "input_sha256": input_before,
        "audit_source_sha256": _digest(Path(__file__)),
        "redshift": redshift,
        "box_len_cMpc": box_len,
        "atol": atol,
        "rtol": rtol,
        "teacher_labels_used": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    arrays, _ = _read_inputs(input_path)
    fitted = Emulator.load(model)
    inputs = tuple(arrays[key] for key in ("source", "capacity", "native_es_xHI"))
    report, baseline = audit_model(
        fitted, *inputs, redshift, box_len, atol=atol, rtol=rtol
    )
    fitted.save(output / "roundtrip_model")
    restored = Emulator.load(output / "roundtrip_model")
    roundtrip = _comparison(
        restored.predict(*inputs, redshift, box_len),
        baseline,
        *inputs[:2],
        atol=0,
        rtol=0,
    )
    report.update(
        complete=True,
        roundtrip=roundtrip,
        passed=report["passed"] and roundtrip["passed"],
        teacher_labels_used=False,
        model_weights_unchanged=_model_hashes(model) == model_before,
        input_unchanged=_digest(input_path) == input_before,
        scope="Emulator invariance at fixed geometry; no assumption of exact teacher angular-quadrature symmetry",
    )
    if not report["model_weights_unchanged"] or not report["input_unchanged"]:
        raise ValueError(
            "Frozen model or inference input changed during symmetry audit"
        )
    np.savez_compressed(
        output / "baseline_prediction.npz",
        **{key: baseline[key] for key in (*FIELDS, *COUNT_KEYS)},
    )
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    hashes = {
        str(path.relative_to(output)): _digest(path)
        for path in output.rglob("*")
        if path.is_file()
    }
    (output / "receipt.json").write_text(
        json.dumps({"files_sha256": hashes}, indent=2) + "\n"
    )
    (output / "COMPLETED").write_text(
        "Symmetry audit completed; inspect report.passed\n"
    )
    return report


def main() -> None:
    """Check saved weights without accessing teacher labels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--redshift", type=float, required=True)
    parser.add_argument("--box-len", type=float, required=True)
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--rtol", type=float, default=1e-7)
    args = parser.parse_args()
    report = run(
        args.model,
        args.input,
        args.output,
        redshift=args.redshift,
        box_len=args.box_len,
        atol=args.atol,
        rtol=args.rtol,
    )
    sys.stdout.write(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "complete",
                    "passed",
                    "checks",
                    "maximum_field_absolute_difference",
                    "maximum_photon_residual",
                )
            }
        )
        + "\n"
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
