"""Run a saved analytic-PC emulator on native arrays without teacher labels.

The input NPZ supplies cumulative native ``source``, ``capacity = 1 + density``
and ``native_es_xHI``. Their physical settings and count convention must match
the model's training inputs. This command predicts a coeval field and a global
unabsorbed-photon reservoir; it does not reconstruct local photon transport.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from research.pc_analytic_emulator.model import Emulator

INPUT_ARRAYS = ("source", "capacity", "native_es_xHI")
FIELD_ARRAYS = ("xHI", "xHII", "raw_xHI", "raw_xHII", "latent_xHII")
LEDGER_TOLERANCE = 1e-10


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _model_hashes(directory: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    root = directory.resolve()
    visited: set[Path] = set()

    def visit(folder: Path) -> None:
        resolved = folder.resolve()
        if not resolved.is_relative_to(root) or resolved in visited:
            raise ValueError("Inference requires a self-contained, acyclic model")
        visited.add(resolved)
        for name in ("model.json", "field_model.npz", "budget_model.npz"):
            path = folder / name
            if not path.resolve().is_relative_to(root) or not path.is_file():
                raise ValueError(f"Missing or external saved model artifact: {path}")
            hashes[str(path.relative_to(directory))] = _digest(path)
        record = json.loads((folder / "model.json").read_text())
        if record.get("schema") == 3 or (
            record.get("schema") == 4 and "base_model" in record
        ):
            if record.get("base_model") != "base_model":
                raise ValueError("Residual model requires its local base_model")
            visit(folder / "base_model")

    visit(directory)
    return hashes


def _read_inputs(path: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    with np.load(path, allow_pickle=False) as saved:
        names = {name: name for name in INPUT_ARRAYS}
        if "native_es_xhi" in saved.files:
            if "native_es_xHI" in saved.files:
                raise ValueError("Supply only one native ES neutral-fraction key")
            names["native_es_xHI"] = "native_es_xhi"
        missing = set(names.values()) - set(saved.files)
        if missing:
            raise ValueError(f"Inference input is missing arrays: {sorted(missing)}")
        arrays = {}
        for name in INPUT_ARRAYS:
            value = saved[names[name]]
            if value.dtype.kind not in "fiu":
                raise TypeError(f"{name} must contain real numeric values")
            arrays[name] = np.array(value, copy=True)
    shape = arrays["source"].shape
    if (
        len(shape) != 3
        or len(set(shape)) != 1
        or shape[0] < 2
        or any(value.shape != shape for value in arrays.values())
    ):
        raise ValueError("Inference arrays must share a cubic grid with dimension >=2")
    return arrays, names


def _physical_compatibility(
    model: Emulator,
    native_path: Path | None,
    arrays: dict[str, np.ndarray],
    redshift: float,
    box_len: float,
) -> dict[str, Any]:
    expected = model.metadata.get("teacher_physics")
    geometries = None
    if "training_geometries" in model.metadata:
        from research.pc_analytic_emulator.data import validate_geometries

        if expected is None:
            raise ValueError("Training geometries require recorded teacher_physics")
        geometries = validate_geometries(model.metadata["training_geometries"])
    if native_path is None:
        if expected is not None:
            raise ValueError(
                "A trained model requires --native-inputs to check its physical parameters"
            )
        return {
            "training_parameters_matched": False,
            "native_array_hashes_verified": False,
            "scope": "Model has no recorded training physics; compatibility cannot be independently checked from arrays",
        }
    native_hash = _digest(native_path)
    record = json.loads(native_path.read_text())
    if not isinstance(record, dict):
        raise TypeError("Native physical parameters must be a JSON object")
    if "identity" in record:
        identity = record["identity"]
        inputs = identity["inputs"]
        if "redshift" in identity and identity["redshift"] != redshift:
            raise ValueError("Native metadata redshift differs from inference")
    else:
        inputs = record
    from research.pc_analytic.compare import validate_physics

    validate_physics(inputs)
    simulation = inputs["simulation_options"]
    if (
        simulation.get("HII_DIM") != arrays["source"].shape[0]
        or simulation.get("BOX_LEN") != box_len
    ):
        raise ValueError("Native geometry differs from the input array or box length")
    common = copy.deepcopy(inputs)
    common.pop("random_seed", None)
    common["simulation_options"].pop("N_THREADS", None)
    comparison = copy.deepcopy(expected)
    geometry_report = {}
    if geometries is not None:
        from research.pc_analytic_emulator.data import (
            GEOMETRY_KEYS,
            validate_geometries,
        )

        geometry = validate_geometries(
            [{key: simulation[key] for key in GEOMETRY_KEYS}]
        )[0]
        if geometry not in geometries:
            raise ValueError(
                "Native geometry is outside the recorded training geometries"
            )
        for key in GEOMETRY_KEYS:
            common["simulation_options"].pop(key, None)
            comparison["simulation_options"].pop(key, None)
        geometry_report = {
            "training_geometries": geometries,
            "evaluation_geometry": geometry,
            "training_geometry_supported": True,
        }
    if expected is not None and common != comparison:
        raise ValueError("Native physical parameters differ from model training")
    hashes = record.get("array_sha256")
    if hashes is not None:
        for name, array in arrays.items():
            archived = "native_es_xhi" if name == "native_es_xHI" else name
            digest = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
            if hashes.get(archived) != digest:
                raise ValueError(
                    f"Inference input differs from native array checksum: {name}"
                )
    return {
        "path": str(native_path),
        "sha256": native_hash,
        "training_parameters_matched": expected is not None,
        "native_array_hashes_verified": hashes is not None,
        **geometry_report,
        "scope": (
            "Native physics matched exactly except random_seed, N_THREADS and explicitly recorded training geometry; current geometry belongs to the training support; archived array hashes checked when supplied"
            if geometries is not None
            else "Native parameter groups compared exactly except random_seed and N_THREADS; archived array hashes checked when supplied"
        ),
    }


def _saved_ledger(path: Path, inputs: dict[str, np.ndarray]) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as saved:
        if any(not np.array_equal(saved[name], inputs[name]) for name in INPUT_ARRAYS):
            raise FloatingPointError(
                "Saved inference inputs differ from their originals"
            )
        for name in FIELD_ARRAYS:
            value = saved[name]
            if value.shape != inputs["source"].shape or not np.isfinite(value).all():
                raise FloatingPointError("Invalid saved inference field")
            if name != "latent_xHII" and np.any((value < 0) | (value > 1)):
                raise FloatingPointError("Saved inference fractions violate bounds")
        if not np.array_equal(saved["xHI"], 1 - saved["xHII"]):
            raise FloatingPointError("Saved neutral and ionized fields disagree")
        source = np.sum(saved["source"], dtype=np.longdouble)
        atoms = np.sum(saved["capacity"], dtype=np.longdouble)
        absorbed = np.sum(
            np.multiply(saved["capacity"], saved["xHII"], dtype=np.longdouble)
        )
        neutral_derived = np.sum(
            np.multiply(saved["capacity"], 1 - saved["xHI"], dtype=np.longdouble)
        )
        target = np.longdouble(saved["predicted_absorbed_photons"])
        unabsorbed = np.longdouble(saved["predicted_unabsorbed_photons"])
    residual = source - absorbed - unabsorbed
    relative = abs(residual) / max(source, np.longdouble(np.finfo(float).tiny))
    if (
        not np.isfinite(target)
        or not np.isfinite(unabsorbed)
        or target < 0
        or unabsorbed < 0
        or target > min(source, atoms)
        or not np.isfinite(relative)
        or relative > LEDGER_TOLERANCE
        or abs(absorbed - target)
        > LEDGER_TOLERANCE * max(target, np.longdouble(np.finfo(float).tiny))
    ):
        raise FloatingPointError("Independent saved-field photon ledger failed")
    return {
        "source_photons": float(source),
        "atom_capacity": float(atoms),
        "predicted_absorbed_photons": float(target),
        "field_absorbed_photons": float(absorbed),
        "predicted_unabsorbed_photons": float(unabsorbed),
        "photon_residual": float(residual),
        "relative_photon_residual": float(relative),
        "relative_tolerance": LEDGER_TOLERANCE,
        "neutral_field_absorbed_photons": float(neutral_derived),
        "neutral_complement_rounding_difference": float(neutral_derived - absorbed),
        "bounded": True,
        "passed": True,
        "scope": "Recomputed from saved source, capacity, xHII and independently predicted unabsorbed count; no local transport or measured escape",
        "field_representation": "xHII retains weak ionization that may round away in the complementary xHI field",
    }


def run(
    model_directory: str | Path,
    input_path: str | Path,
    *,
    redshift: float,
    box_len: float,
    output: str | Path,
    native_inputs: str | Path | None = None,
) -> dict[str, Any]:
    """Predict one coeval and publish independently audited saved fields.

    Parameters
    ----------
    model_directory
        Self-contained model directory written by ``Emulator.save``.
    input_path
        Pickle-free NPZ containing source, capacity and native_es_xHI. Other
        arrays are ignored, including any teacher labels. The native archive's
        lowercase native_es_xhi key is also accepted.
    redshift, box_len
        Snapshot redshift and comoving box length in cMpc.
    output
        New directory outside the saved model. Existing paths are never replaced.
    native_inputs
        Full native InputParameters JSON or archived native.json. Required when
        the model records teacher_physics; all parameter groups except seed and
        thread count must match training. New models with explicit
        training_geometries accept any recorded training grid; other models
        require the original exact geometry. Archived array
        checksums, when present, must agree with the inference input arrays.

    Returns
    -------
    dict
        Input/model hashes, projection diagnostics and the field-derived ledger
        recomputed after writing and rereading fields.npz.

    Notes
    -----
    Input arrays alone cannot certify their physical parameter identity. A plain
    parameter JSON checks the declared settings; an archived native.json also
    binds them to its saved array checksums. No teacher field or photon reservoir
    is required at inference.
    """
    model_directory = Path(model_directory).resolve()
    input_path = Path(input_path).resolve()
    output = Path(output).resolve()
    if output == model_directory or output.is_relative_to(model_directory):
        raise ValueError("Inference output must be outside the saved model")
    if output.exists():
        raise FileExistsError(f"Inference output already exists: {output}")
    if not model_directory.is_dir() or not input_path.is_file():
        raise ValueError("Inference requires a saved model directory and input NPZ")
    if (
        isinstance(redshift, bool)
        or isinstance(box_len, bool)
        or not np.isfinite(redshift)
        or redshift < 0
        or not np.isfinite(box_len)
        or box_len <= 0
    ):
        raise ValueError("Require finite nonnegative redshift and positive box length")
    input_hash = _digest(input_path)
    model_hashes = _model_hashes(model_directory)
    inputs, names = _read_inputs(input_path)
    model = Emulator.load(model_directory)
    native_path = Path(native_inputs).resolve() if native_inputs is not None else None
    physics = _physical_compatibility(model, native_path, inputs, redshift, box_len)
    started = time.perf_counter()
    result = model.predict(
        inputs["source"], inputs["capacity"], inputs["native_es_xHI"], redshift, box_len
    )
    elapsed = time.perf_counter() - started
    if (
        _digest(input_path) != input_hash
        or _model_hashes(model_directory) != model_hashes
        or (native_path is not None and _digest(native_path) != physics["sha256"])
    ):
        raise ValueError("Model or input bytes changed during inference")
    output.mkdir(parents=True, exist_ok=False)
    arrays = inputs | {name: result[name] for name in FIELD_ARRAYS}
    for name in (
        "absorption_efficiency",
        "predicted_absorbed_photons",
        "predicted_unabsorbed_photons",
    ):
        arrays[name] = np.asarray(result[name], dtype=np.float64)
    np.savez_compressed(output / "fields.npz", **arrays)
    ledger = _saved_ledger(output / "fields.npz", inputs)
    report = {
        "schema": 1,
        "complete": True,
        "redshift": float(redshift),
        "box_len_cMpc": float(box_len),
        "dimension": inputs["source"].shape[0],
        "input": {
            "path": str(input_path),
            "sha256": input_hash,
            "arrays_used": list(names.values()),
            "source_convention": "Cumulative native n_ion; capacity=1+density; no extra density multiplier or time integral",
        },
        "model": {
            "path": str(model_directory),
            "files_sha256": model_hashes,
            "metadata": model.metadata,
        },
        "projection": result["projection"],
        "physical_compatibility": physics,
        "prediction_photon_ledger": result["photon_ledger"],
        "saved_field_photon_ledger": ledger,
        "prediction_seconds": elapsed,
        "timing_scope": "Feature construction and model inference including global projection; excludes model/input I/O and serialized-output audit",
        "files_sha256": {"fields.npz": _digest(output / "fields.npz")},
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    (output / "COMPLETED").write_text("Saved emulator field photon ledger verified.\n")
    return report


def main() -> None:
    """Run label-free coeval inference from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--redshift", required=True, type=float)
    parser.add_argument("--box-len", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--native-inputs", type=Path)
    args = parser.parse_args()
    report = run(
        args.model,
        args.input,
        redshift=args.redshift,
        box_len=args.box_len,
        output=args.output,
        native_inputs=args.native_inputs,
    )
    sys.stdout.write(
        json.dumps(
            {
                "complete": report["complete"],
                "report": str(args.output.resolve() / "report.json"),
                "relative_photon_residual": report["saved_field_photon_ledger"][
                    "relative_photon_residual"
                ],
            },
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
