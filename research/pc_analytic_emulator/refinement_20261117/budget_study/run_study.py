"""Reproduce the rejected 12-candidate budget study using development data only.

This entry point audits the existing 164/28 training/validation cache, freezes
its configuration before fitting, and evaluates one scalar-selected budget
against the unchanged validation field objective. It never opens test archives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from research.pc_analytic_emulator.budget import (
    budget_error_metrics,
    load_budget,
)
from research.pc_analytic_emulator.model import Emulator
from research.pc_analytic_emulator.objectives import projected_objective
from research.pc_analytic_emulator.refinement_20261117.budget_study.anchor_budget import (
    AnchoredDeficit,
    fit,
)


def digest(path: Path) -> str:
    """Hash exact bytes for replay provenance."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    """Write finite JSON evidence in a fresh output directory."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def audit_cache(cache: Path) -> tuple[dict, dict]:
    """Verify native cache bytes and the original disjoint development split."""
    receipt = json.loads((cache / "receipt.json").read_text())
    for name in ("budget_arrays.npz", "features.json"):
        if digest(cache / name) != receipt["files"][name]:
            raise ValueError(f"Training cache hash mismatch: {name}")
    details = json.loads((cache / "features.json").read_text())
    train = {row["seed"] for row in details["training_snapshots"]}
    validation = {row["seed"] for row in details["validation_snapshots"]}
    if (
        details["test_labels_in_fit_cache"]
        or train != set(range(2101, 2113)) | set(range(2121, 2129))
        or validation != set(range(2113, 2117))
        or len(details["training_snapshots"]) != 164
        or len(details["validation_snapshots"]) != 28
    ):
        raise ValueError(
            "This reproduction requires the original 164/28 development split"
        )
    return details, receipt


def score_fields(training: Path, budget: AnchoredDeficit) -> dict:
    """Compare unchanged field weights with original and recursively replaced budgets."""
    cache = training / "fit_cache"
    details, receipt = audit_cache(cache)
    config = json.loads((training / "identity.json").read_text())["configuration"]
    objective = {
        key: config[key]
        for key in (
            "spectral_weight",
            "spectral_power_floor",
            "front_weight",
            "field_variance_floor",
            "k_fraction",
        )
    }
    hashes = {
        str(path.relative_to(training / "model")): digest(path)
        for path in (training / "model").rglob("*")
        if path.is_file()
    }
    reports = []
    for replace in (False, True):
        model = Emulator.load(training / "model")
        depth = 0
        if replace:
            layer = model
            while layer is not None:
                # In-memory callable substitution only: the production loader,
                # saved model kind and all field files remain unchanged.
                layer.budget_model.predict = budget.predict
                layer = layer.base_model
                depth += 1
        rows = []
        for row in details["validation_snapshots"]:
            # Relocate cache paths by filename when archives were copied.
            path = cache / Path(row["cache"]).name
            if digest(path) != receipt["files"][path.name]:
                raise ValueError("Validation frame failed its original receipt")
            with np.load(path, allow_pickle=False) as frame:
                prediction = model.predict(
                    frame["source"],
                    frame["capacity"],
                    frame["native_es_xHI"],
                    row["redshift"],
                    row["box_len"],
                )
                truth = 1 - frame["teacher_xHI"]
                residual = prediction["xHII"] - truth
                loss, _, components = projected_objective(
                    prediction["xHII"],
                    truth,
                    frame["capacity"],
                    row["box_len"],
                    **objective,
                )
                rows.append(
                    {
                        "seed": row["seed"],
                        "redshift": row["redshift"],
                        "field_MSE": float(np.mean(residual**2)),
                        "field_MAE": float(np.mean(abs(residual))),
                        "projected_loss": loss,
                        "objective": components,
                        "photon_ledger": prediction["photon_ledger"],
                    }
                )
        reports.append(
            {
                "kind": "recursive_candidate" if replace else "original_tabulated",
                "budget_layers_replaced": depth,
                "validation_objective": float(
                    np.mean([r["projected_loss"] for r in rows])
                ),
                "validation_MSE": float(np.mean([r["field_MSE"] for r in rows])),
                "validation_MAE": float(np.mean([r["field_MAE"] for r in rows])),
                "maximum_photon_residual": max(
                    r["photon_ledger"]["relative_photon_residual"] for r in rows
                ),
                "rows": rows,
            }
        )
    expected = json.loads((training / "summary.json").read_text())[
        "selected_validation_objective"
    ]
    np.testing.assert_allclose(
        reports[0]["validation_objective"], expected, rtol=1e-10, atol=1e-12
    )
    if any(digest(training / "model" / name) != sha for name, sha in hashes.items()):
        raise ValueError("Frozen field model changed during the study")
    return {
        "objective": objective,
        "field_model_hashes": hashes,
        "evaluations": reports,
    }


def main() -> None:
    """Freeze and execute the fixed development study, without deployment changes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-training-run", type=Path, required=True)
    parser.add_argument("--field-training-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    cache = args.budget_training_run.absolute() / "fit_cache"
    details, receipt = audit_cache(cache)
    specs = [
        {"name": f"{name}_r{ridge:g}", "feature_indices": indices, "ridge": ridge}
        for name, indices in (
            ("load_only", []),
            ("structure", [4, 5, 7, 8, 9]),
            ("multiscale_structure", list(range(4, 14))),
        )
        for ridge in (1e-5, 1e-4, 1e-3, 1e-2)
    ]
    module = Path(__file__).with_name("anchor_budget.py")
    protocol = {
        "schema": 1,
        "frozen_before_fit_utc": datetime.now(UTC).isoformat(),
        "cache": str(cache),
        "cache_hashes": receipt["files"],
        "runner_sha256": digest(Path(__file__)),
        "model_code_sha256": digest(module),
        "training_count": 164,
        "validation_count": 28,
        "native_feature_names": details["budget_feature_names"],
        "candidates": specs,
        "correction_bound": 3.0,
        "selection": "Scalar mixed budget validation loss; then unchanged full-field validation objective",
        "budget_metric": "mass_mse + 0.1*floored_relative_neutral_mse; neutral floor0.01",
        "blind_or_transfer_used": False,
    }
    write_json(output / "protocol.json", protocol)
    with np.load(cache / "budget_arrays.npz", allow_pickle=False) as saved:
        x, eta, scale = (
            saved[f"{key}_train"].copy() for key in ("global", "eta", "mass_scale")
        )
        xv, ev, sv = (
            saved[f"{key}_validation"].copy() for key in ("global", "eta", "mass_scale")
        )
    if x.shape != (164, 14) or xv.shape != (28, 14):
        raise ValueError("Unexpected native budget array dimensions")
    for inputs, s in ((x, scale), (xv, sv)):
        np.testing.assert_allclose(
            np.minimum(np.expm1(inputs[:, 0]), 1), s, rtol=0, atol=2e-15
        )
    table_path = args.budget_training_run / "budget/tabulated.npz"
    table = load_budget(table_path)
    rows = []
    predictions = {
        "teacher_efficiency": ev,
        "mass_scale": sv,
        "tabulated": table.predict(xv),
    }
    for spec in specs:
        model, optimization, _ = fit(
            x, eta, scale, spec["feature_indices"], spec["ridge"]
        )
        predicted = model.predict(xv)
        path = output / f"{spec['name']}.npz"
        model.save(path)
        np.testing.assert_array_equal(AnchoredDeficit.load(path).predict(xv), predicted)
        rows.append(
            {
                **spec,
                "optimization": optimization,
                "training_metrics": budget_error_metrics(model.predict(x), eta, scale),
                "validation_metrics": budget_error_metrics(predicted, ev, sv),
                "validation_by_redshift": {
                    str(z): budget_error_metrics(
                        predicted[xv[:, 1] == z], ev[xv[:, 1] == z], sv[xv[:, 1] == z]
                    )
                    for z in sorted(set(xv[:, 1]))
                },
                "model_file": path.name,
                "model_sha256": digest(path),
            }
        )
        predictions[spec["name"]] = predicted
    selected = min(rows, key=lambda row: row["validation_metrics"]["mixed_loss"])
    candidate = AnchoredDeficit.load(output / selected["model_file"])
    field = score_fields(args.field_training_run.absolute(), candidate)
    before, after = field["evaluations"]
    report = {
        "complete": True,
        "protocol_sha256": digest(output / "protocol.json"),
        "training_count": 164,
        "validation_count": 28,
        "blind_or_transfer_used": False,
        "scalar_selected_candidate": selected["name"],
        "reference": {
            "name": "tabulated",
            "model_sha256": digest(table_path),
            "validation_metrics": budget_error_metrics(table.predict(xv), ev, sv),
        },
        "candidates": rows,
        "field_validation": field,
        "field_objective_improved": after["validation_objective"]
        < before["validation_objective"],
        "deployment_changed": False,
    }
    write_json(output / "report.json", report)
    np.savez_compressed(output / "validation_predictions.npz", **predictions)
    write_json(
        output / "receipt.json",
        {
            "files": {
                p.name: digest(p)
                for p in output.iterdir()
                if p.is_file() and p.name != "receipt.json"
            }
        },
    )
    print(  # noqa: T201
        json.dumps(
            {
                "selected": selected["name"],
                "field_objective_improved": report["field_objective_improved"],
            }
        )
    )


if __name__ == "__main__":
    main()
