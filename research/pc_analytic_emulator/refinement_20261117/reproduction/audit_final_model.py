"""Audit a frozen final model using validation native inputs and saved receipts.

No teacher fields are loaded for the symmetry, inference, or feature checks.
Saved validation scores are checked for selection consistency separately.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import logging
import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from research.pc_analytic_emulator import audit_symmetry, predict
from research.pc_analytic_emulator.benchmark import _model_hashes
from research.pc_analytic_emulator.data import GEOMETRY_KEYS, validate_geometries
from research.pc_analytic_emulator.model import Emulator, field_features
from research.pc_analytic_emulator.refinement_20261117.reproduction.diagnose_feature_scaling import (
    feature_statistics,
    levels,
)

LOGGER = logging.getLogger(__name__)
KERNEL_FILES = (
    "research/pc_analytic/absorption.py",
    "research/pc_analytic/solver.py",
    "research/pc_analytic/transport_kernel.cpp",
    "research/pc_reconsidered/native_driver.py",
)
INFERENCE_MODULES = (
    "research.pc_analytic_emulator.audit_symmetry",
    "research.pc_analytic_emulator.benchmark",
    "research.pc_analytic_emulator.data",
    "research.pc_analytic_emulator.predict",
    "research.pc_analytic_emulator.model",
    "research.pc_analytic_emulator.latent",
    "research.pc_analytic_emulator.budget",
    "research.pc_analytic_emulator.spatial",
    "research.pc_analytic_emulator.projection",
    "research.pc_analytic_emulator.fit",
    "research.pc_field_emulator.model",
    "research.pc_field_emulator.features",
    "research.pc_analytic.validation_20261116.reproduction.recompute",
)


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _python_comparison(frozen: Path, released: Path) -> dict:
    """Record byte identity separately from parsed Python equivalence."""
    return {
        "byte_identical": _digest(frozen) == _digest(released),
        "AST_identical": ast.dump(ast.parse(frozen.read_text()))
        == ast.dump(ast.parse(released.read_text())),
        "frozen_sha256": _digest(frozen),
        "released_sha256": _digest(released),
    }


def verify_sources(
    frozen_source: Path,
    manifest_path: Path,
    physics_reference: Path,
    *,
    released_source: Path | None = None,
) -> dict:
    """Verify the frozen source manifest and explicit physical-kernel reference."""
    frozen_source, physics_reference = (
        frozen_source.resolve(),
        physics_reference.resolve(),
    )
    expected = json.loads(manifest_path.read_text())
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Frozen source manifest must be a nonempty file-hash mapping")
    for name, digest in expected.items():
        path = (frozen_source / name).resolve()
        if (
            not path.is_relative_to(frozen_source)
            or not path.is_file()
            or _digest(path) != digest
        ):
            raise ValueError(f"Frozen source mismatch: {name}")
    if released_source is None:
        module = importlib.import_module("research.pc_analytic_emulator.model")
        released_source = Path(module.__file__).resolve().parents[2]
    released_source = released_source.resolve()
    kernel_names = set(KERNEL_FILES)
    native = released_source / "src/py21cmfast/src"
    if not native.is_dir():
        raise ValueError("Released source lacks native physical kernels")
    native_names = {
        str(path.relative_to(released_source))
        for path in native.rglob("*")
        if path.is_file() and path.suffix in {".c", ".h", ".cpp"}
    }
    reference_names = {
        str(path.relative_to(physics_reference))
        for path in (physics_reference / "src/py21cmfast/src").rglob("*")
        if path.is_file() and path.suffix in {".c", ".h", ".cpp"}
    }
    if native_names != reference_names:
        raise ValueError(
            "Released native kernel file set differs from supplied reference"
        )
    kernel_names.update(
        str(path.relative_to(released_source))
        for path in native.rglob("*")
        if path.is_file() and path.suffix in {".c", ".h", ".cpp"}
    )
    if len(kernel_names) == len(KERNEL_FILES):
        raise ValueError("No native physical C/C++ source was found")
    kernel_hashes = {}
    for name in sorted(kernel_names):
        current, reference = released_source / name, physics_reference / name
        if (
            not current.is_file()
            or not reference.is_file()
            or _digest(current) != _digest(reference)
        ):
            raise ValueError(f"Physical kernel differs from supplied reference: {name}")
        if name in KERNEL_FILES and _digest(frozen_source / name) != _digest(reference):
            raise ValueError(f"Archived training physical kernel changed: {name}")
        kernel_hashes[name] = _digest(current)
    loaded, python_comparisons = {}, {}
    for name in INFERENCE_MODULES:
        relative = name.replace(".", "/") + ".py"
        module = importlib.import_module(name)
        if relative not in expected:
            raise ValueError(
                f"Loaded inference module absent from manifest: {relative}"
            )
        comparison = _python_comparison(frozen_source / relative, Path(module.__file__))
        if not comparison["AST_identical"]:
            raise ValueError(
                f"Loaded inference implementation differs beyond formatting: {relative}"
            )
        loaded[relative] = comparison["released_sha256"]
        python_comparisons[relative] = comparison
    train_relative = "research/pc_analytic_emulator/train.py"
    training, released = (
        frozen_source / train_relative,
        released_source / train_relative,
    )
    train_comparison = _python_comparison(training, released)
    if not train_comparison["AST_identical"]:
        raise ValueError("Released training implementation differs beyond formatting")
    python_comparisons[train_relative] = train_comparison
    return {
        "passed": True,
        "frozen_source_file_count": len(expected),
        "source_manifest_sha256": _digest(manifest_path),
        "loaded_inference_source_sha256": loaded,
        "python_module_comparisons": python_comparisons,
        "formatted_python_files": sorted(
            name
            for name, comparison in python_comparisons.items()
            if not comparison["byte_identical"]
        ),
        "train_python_comparison": train_comparison,
        "python_comparison_scope": "Archived manifest bytes are verified exactly. Listed loaded inference modules and released train.py must have identical parsed Python ASTs to their archived versions; individual byte hashes and formatting-only differences are reported separately.",
        "unchanged_physical_kernel_sha256": kernel_hashes,
        "physical_reference_scope": "Released analytic absorption/transport and native source adapter plus all native C/C++ headers/sources are byte-identical to supplied teacher-source reference. Archived training analytic kernels also match. No claim about unlisted files.",
    }


def _native(path: Path) -> dict:
    metadata = path.with_name("native.json")
    record = json.loads(metadata.read_text())
    identity = record["identity"]
    simulation = identity["inputs"]["simulation_options"]
    geometry = validate_geometries([{key: simulation[key] for key in GEOMETRY_KEYS}])[0]
    z = identity["redshift"]
    if (
        isinstance(z, bool)
        or not isinstance(z, (int, float))
        or not np.isfinite(z)
        or z < 0
    ):
        raise ValueError("Native validation redshift is invalid")
    return {
        "source": path,
        "metadata": metadata,
        "geometry": geometry,
        "redshift": float(z),
        "source_sha256": _digest(path),
        "metadata_sha256": _digest(metadata),
    }


def feature_guard_audit(
    model: Emulator, inputs: dict, redshift: float, length: float
) -> list[dict]:
    """Measure unchanged heads and verify their declared normalization safeguards."""
    rows = []
    arrays = tuple(inputs[key] for key in ("source", "capacity", "native_es_xHI"))
    for depth, level in enumerate(levels(model)):
        features, names, _ = field_features(
            *arrays,
            redshift,
            length,
            radii_cMpc=level.radii_cMpc,
            feature_set=level.feature_set,
        )
        head = level.field_model
        metadata = head.metadata
        floor = metadata.get("field_feature_scale_floor", 0)
        # Legacy heads persist a vector under the older generic key.
        if not floor and metadata.get("feature_scale_floor") is not None:
            floor = float(np.min(metadata["feature_scale_floor"]))
        indices = metadata.get("constant_feature_indices", [])
        constants_enabled = metadata.get("zero_constant_features", False)
        constant_norm = (
            float(np.max(np.abs(head.weights[0][indices]))) if indices else 0.0
        )
        if not np.isfinite(floor) or floor < 0:
            raise ValueError("Saved feature normalization floor is invalid")
        floor_passed = bool(np.min(head.scale) >= floor)
        constants_passed = not constants_enabled or constant_norm == 0
        rows.append(
            {
                "depth": depth,
                "feature_count": len(names),
                "minimum_feature_scale": float(np.min(head.scale)),
                "all_feature_scales_at_least_0p01": bool(np.min(head.scale) >= 0.01),
                "declared_feature_scale_floor": floor,
                "declared_floor_satisfied": floor_passed,
                "zero_constant_features": constants_enabled,
                "constant_feature_indices": indices,
                "maximum_constant_weight_magnitude": constant_norm,
                "constant_rows_satisfied": constants_passed,
                "passed": floor_passed and constants_passed,
                **feature_statistics(features, names, head),
            }
        )
    return rows


def _exact_predictions(first: Path, second: Path) -> dict:
    keys = (
        *predict.FIELD_ARRAYS,
        "absorption_efficiency",
        "predicted_absorbed_photons",
        "predicted_unabsorbed_photons",
    )
    with (
        np.load(first, allow_pickle=False) as a,
        np.load(second, allow_pickle=False) as b,
    ):
        result = {key: bool(np.array_equal(a[key], b[key])) for key in keys}
    return {
        "passed": all(result.values()),
        "array_equal": result,
        "scope": "Exact numerical equality after model save/load; output ZIP byte identity is not required",
    }


def run(
    model: str | Path,
    selection: str | Path,
    validations: Mapping[str, str | Path],
    training_runs: Mapping[str, str | Path],
    output: str | Path,
    *,
    frozen_source: str | Path,
    source_hashes: str | Path,
    physics_reference: str | Path,
    reference_model: str | Path | None = None,
    expected_validation_count: int = 42,
) -> dict:
    """Audit frozen selection and supported geometries without opening blind labels."""
    from research.pc_analytic_emulator.refinement_20261117.reproduction.audit_selection import (
        audit_selection,
    )

    model, selection, output = (
        Path(path).resolve() for path in (model, selection, output)
    )
    if output.exists() or output.is_relative_to(model):
        raise ValueError("Final audit output must be new and outside the fitted model")
    if not validations or any(
        not re.fullmatch(r"[A-Za-z0-9_-]+", label) for label in validations
    ):
        raise ValueError("Validation labels must be nonempty public alphanumeric names")
    before, selection_hash = _model_hashes(model), _digest(selection)
    frozen_source, source_hashes, physics_reference = (
        Path(path).resolve()
        for path in (frozen_source, source_hashes, physics_reference)
    )
    source_report = verify_sources(frozen_source, source_hashes, physics_reference)
    selection_report = audit_selection(
        selection,
        model,
        {name: Path(path).resolve() for name, path in training_runs.items()},
        expected_validation_count=expected_validation_count,
    )
    fitted = Emulator.load(model)
    supported = fitted.metadata.get("training_geometries")
    if supported is None:
        simulation = fitted.metadata["teacher_physics"]["simulation_options"]
        supported = [{key: simulation[key] for key in GEOMETRY_KEYS}]
    supported = validate_geometries(supported)
    native_inputs = {
        label: _native(Path(path).resolve()) for label, path in validations.items()
    }
    covered = [
        entry["geometry"]
        for entry in native_inputs.values()
        if entry["geometry"] in supported
    ]
    if any(geometry not in covered for geometry in supported):
        raise ValueError(
            "Native validation inputs do not cover every recorded training geometry"
        )
    reference_path = Path(reference_model).resolve() if reference_model else None
    reference_hashes = _model_hashes(reference_path) if reference_path else None
    reference = Emulator.load(reference_path) if reference_path else None
    output.mkdir(parents=True)
    snapshots = {}
    for label, entry in native_inputs.items():
        if entry["geometry"] not in supported:
            snapshots[label] = {
                "audited": False,
                "geometry": entry["geometry"],
                "reason": "Outside selected model training support; this audit does not certify transfer",
            }
            continue
        path, z, length = (
            entry["source"],
            entry["redshift"],
            entry["geometry"]["BOX_LEN"],
        )
        folder = output / label
        sym = audit_symmetry.run(
            model,
            path,
            folder / "symmetry",
            redshift=z,
            box_len=length,
            atol=1e-7,
            rtol=1e-7,
        )
        inference = predict.run(
            model,
            path,
            redshift=z,
            box_len=length,
            output=folder / "inference",
            native_inputs=entry["metadata"],
        )
        restored = predict.run(
            folder / "symmetry/roundtrip_model",
            path,
            redshift=z,
            box_len=length,
            output=folder / "roundtrip_inference",
            native_inputs=entry["metadata"],
        )
        exact = _exact_predictions(
            folder / "inference/fields.npz", folder / "roundtrip_inference/fields.npz"
        )
        arrays, _ = predict._read_inputs(path)
        guards = feature_guard_audit(fitted, arrays, z, length)
        previous_features = (
            feature_guard_audit(reference, arrays, z, length)
            if reference is not None
            else None
        )
        snapshots[label] = {
            "audited": True,
            "geometry": entry["geometry"],
            "redshift": z,
            "input_sha256": entry["source_sha256"],
            "native_metadata_sha256": entry["metadata_sha256"],
            "symmetry": {
                key: sym[key]
                for key in (
                    "passed",
                    "checks",
                    "field_atol",
                    "field_rtol",
                    "budget_atol_over_max_inventory",
                    "maximum_field_absolute_difference",
                    "maximum_photon_residual",
                    "maximum_budget_error_over_inventory",
                )
            },
            "exact_model_roundtrip": exact,
            "saved_prediction_photon_ledger": inference["saved_field_photon_ledger"],
            "roundtrip_saved_photon_ledger": restored["saved_field_photon_ledger"],
            "selected_feature_guards": guards,
            "reference_feature_statistics": previous_features,
            "passed": sym["passed"]
            and exact["passed"]
            and all(row["passed"] for row in guards),
        }
        if (
            _digest(path) != entry["source_sha256"]
            or _digest(entry["metadata"]) != entry["metadata_sha256"]
        ):
            raise ValueError("Native validation inputs changed during final audit")
        LOGGER.info(
            "Final audit %s completed: passed=%s", label, snapshots[label]["passed"]
        )
    if _model_hashes(model) != before or _digest(selection) != selection_hash:
        raise ValueError("Final selection or selected weights changed during audit")
    if reference_path and _model_hashes(reference_path) != reference_hashes:
        raise ValueError("Reference model changed during audit")
    if verify_sources(frozen_source, source_hashes, physics_reference) != source_report:
        raise ValueError("Inference source or physical kernels changed during audit")
    report = {
        "schema": 1,
        "complete": True,
        "passed": selection_report["passed"]
        and all(row["passed"] for row in snapshots.values() if row["audited"]),
        "teacher_labels_used_for_symmetry_or_inference": False,
        "blind_data_accessed": False,
        "invariance_scope": "Periodic translations, all 48 cubic rotations/reflections, and common photon/atom count-unit scaling are mathematical symmetries of these emulator descriptors. Floating-point evaluation is tested separately at field atol=rtol=1e-7; no assumption is made about discrete teacher-ray quadrature symmetry.",
        "budget_scope": "Independent global field absorption plus separately predicted unabsorbed inventory; inventory invariance tolerance 1e-10, not a transport or escape-accuracy claim",
        "selection_sha256": selection_hash,
        "model_files_sha256": before,
        "reference_model_files_sha256": reference_hashes,
        "source_and_physical_kernels": source_report,
        "audit_source_sha256": {
            "audit_final_model.py": _digest(Path(__file__)),
            "audit_selection.py": _digest(
                Path(__file__).with_name("audit_selection.py")
            ),
        },
        "selection_and_checkpoints": selection_report,
        "snapshots": snapshots,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return report


def _mapping(values: list[str], parser) -> dict[str, Path]:
    result = {}
    for value in values:
        label, separator, path = value.partition("=")
        if not separator or not path or label in result:
            parser.error("Named inputs must use unique LABEL=PATH entries")
        result[label] = Path(path)
    return result


def main() -> None:
    """Audit only after the final selection record and weights have been frozen."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "model",
        "selection",
        "frozen-source",
        "source-hashes",
        "physics-reference",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--reference-model", type=Path)
    parser.add_argument(
        "--validation", action="append", required=True, metavar="LABEL=SOURCE_NPZ"
    )
    parser.add_argument(
        "--training-run", action="append", default=[], metavar="LABEL=RUN"
    )
    parser.add_argument("--expected-validation-count", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = run(
        args.model,
        args.selection,
        _mapping(args.validation, parser),
        _mapping(args.training_run, parser),
        args.output,
        frozen_source=args.frozen_source,
        source_hashes=args.source_hashes,
        physics_reference=args.physics_reference,
        reference_model=args.reference_model,
        expected_validation_count=args.expected_validation_count,
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
