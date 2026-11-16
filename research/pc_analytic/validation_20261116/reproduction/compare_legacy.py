"""Compare validation-selected legacy and residual models on a sealed test seed."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from recompute import aggregate, compare_mfp, geometric_mfp, metrics, spectra, total

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Freeze the comparator using validation scores before reading test fields."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2108)
    args = parser.parse_args()
    training, teacher, source_root, output = (
        args.training,
        args.teacher,
        args.source,
        args.output,
    )
    output.mkdir(parents=True, exist_ok=True)
    identity = json.loads((training / "identity.json").read_text())
    if args.seed not in identity["configuration"]["test_seeds"] or args.seed in (
        identity["configuration"]["train_seeds"]
        + identity["configuration"]["validation_seeds"]
    ):
        raise ValueError("Paired comparison requires an exclusive held-out test seed")
    inference_paths = (
        "research/pc_analytic_emulator/model.py",
        "research/pc_analytic_emulator/budget.py",
        "research/pc_analytic_emulator/latent.py",
        "research/pc_analytic_emulator/projection.py",
        "research/pc_field_emulator/model.py",
        "research/pc_field_emulator/features.py",
    )
    for relative in inference_paths:
        if (
            hashlib.sha256((source_root / relative).read_bytes()).hexdigest()
            != identity["implementation_sha256"][relative]
        ):
            raise ValueError(
                "Legacy inference implementation differs from the frozen training run"
            )
    sys.path.insert(0, str(source_root))
    from research.pc_analytic_emulator.model import Emulator

    selection = json.loads((training / "model_selection.json").read_text())
    if selection["test_used_for_selection"]:
        raise ValueError("Test-informed selection is not a held-out comparison")
    candidates = [
        (
            next(
                row for row in trial["candidates"] if row["kind"] == "legacy_reference"
            )["validation_objective"],
            trial["model_seed"],
        )
        for trial in selection["trials"]
    ]
    legacy_objective, legacy_seed = min(candidates)
    model_path = (
        training / "trials" / f"seed{legacy_seed}" / "legacy_reference" / "model"
    )
    model = Emulator.load(model_path)
    result = {
        "legacy_selection": {
            "model_seed": legacy_seed,
            "validation_objective": legacy_objective,
            "criterion": "Minimum legacy-reference validation objective among all four initializations; held-out labels excluded",
            "model_path": str(model_path),
        },
        "final_selection": {
            "model_seed": selection["selected"]["model_seed"],
            "validation_objective": selection["selected"]["validation_objective"],
            "kind": selection["selected"]["selected_kind"],
        },
        "model_artifact_sha256": {
            str(path.relative_to(model_path)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in model_path.rglob("*")
            if path.is_file()
        },
        "scope": "Postselection held-out comparison; no model updates. Metrics use independent recomputation functions. Legacy inference uses verified frozen training implementation.",
    }
    inputs = json.loads(
        (teacher / "seeds" / f"seed{args.seed}" / "native_inputs.json").read_text()
    )
    cosmology = inputs["cosmo_params"]
    draws = np.random.default_rng(20261116).random((32768, 6))
    rows = []
    for path in sorted((training / "test").glob(f"seed{args.seed}_z*/predictions.npz")):
        metadata = json.loads((path.parent / "metrics.json").read_text())
        z, length = metadata["redshift"], metadata["box_len"]
        native_path = (
            teacher
            / "seeds"
            / f"seed{args.seed}"
            / "teacher"
            / f"z{z:.12g}".replace(".", "p")
            / "fields.npz"
        )
        with np.load(native_path) as data:
            source, atoms, native, truth = (
                np.array(data[name])
                for name in ("source", "capacity", "native_es_xHI", "teacher_xHI")
            )
        with np.load(path) as data:
            final = np.array(data["emulator_xHI"])
        predicted = model.predict(source, atoms, native, z, length)
        destination = output / "legacy_predictions" / path.parent.name
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination / "predictions.npz",
            xHI=predicted["xHI"],
            xHII=predicted["xHII"],
            predicted_unabsorbed_photons=np.asarray(
                predicted["predicted_unabsorbed_photons"]
            ),
            predicted_absorbed_photons=np.asarray(
                predicted["predicted_absorbed_photons"]
            ),
        )
        field_absorbed = total(atoms.astype(np.longdouble) * predicted["xHII"])
        residual = float(
            abs(
                total(source)
                - field_absorbed
                - np.longdouble(predicted["predicted_unabsorbed_photons"])
            )
            / total(source)
        )
        fields = {
            "teacher": truth,
            "emulator": final,
            "raw": predicted["xHI"],
            "native_es": native,
        }
        coefficient = (
            27
            * cosmology["OMb"]
            * cosmology["hlittle"] ** 2
            / 0.023
            * np.sqrt(
                0.15 / (cosmology["OMm"] * cosmology["hlittle"] ** 2) * (1 + z) / 10
            )
        )
        brightness = {
            name: coefficient * atoms * field for name, field in fields.items()
        }
        all_mfp = {}
        for phase in ("ionized", "neutral"):
            samples = {
                name: geometric_mfp(fields[name], length, phase, draws)
                for name in ("teacher", "emulator", "raw")
            }
            all_mfp[phase] = {
                "legacy": samples["raw"][0],
                "legacy_vs_teacher": compare_mfp(samples["raw"], samples["teacher"]),
                "final_vs_teacher": compare_mfp(
                    samples["emulator"], samples["teacher"]
                ),
            }
        rows.append(
            {
                "redshift": z,
                "teacher_xHI_mass": float(
                    total(atoms.astype(np.longdouble) * truth) / total(atoms)
                ),
                "field_metrics": {
                    name: metrics(field, truth, atoms) for name, field in fields.items()
                },
                "low_k_21cm": spectra(brightness, length, 0.5),
                "legacy_photon_residual": residual,
                "MFP": all_mfp,
                "maximum_field_difference_final_vs_legacy": float(
                    np.max(np.abs(final - predicted["xHI"]))
                ),
            }
        )
    rows.sort(key=lambda row: -row["redshift"])
    result.update(
        all_snapshots=aggregate(rows),
        active_snapshots=aggregate(
            [row for row in rows if 0.05 < row["teacher_xHI_mass"] < 0.95]
        ),
        rows=rows,
    )

    # Rename generic slots used only to share the independent aggregation code.
    def rename(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                (
                    "legacy" if key == "raw" else "final" if key == "emulator" else key
                ): rename(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [rename(item) for item in value]
        return value

    result = rename(result)
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output / "paired_legacy_comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    LOGGER.info(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "legacy_selection",
                    "final_selection",
                    "all_snapshots",
                    "active_snapshots",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
