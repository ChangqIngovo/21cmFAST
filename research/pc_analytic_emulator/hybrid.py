"""Use validation-approved surrogate routes and explicit analytic fallback.

This optional deployment preserves the analytic teacher's physical scope. A
route is selected using only exact grid geometry and redshift, never a teacher
field or a predicted photon ledger. Validation establishes empirical support,
not an error guarantee for another realization. Reports distinguish surrogate
predictions from newly solved analytic fields and retain measured reservoirs
for the latter.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from research.pc_analytic.compare import independent_budget, load_native_snapshot
from research.pc_analytic.solver import solve
from research.pc_analytic_emulator.benchmark import (
    _inference_source_hashes,
    _physics_compatibility,
)
from research.pc_analytic_emulator.data import GEOMETRY_KEYS, validate_geometries
from research.pc_analytic_emulator.model import Emulator
from research.pc_analytic_emulator.predict import _digest, _model_hashes
from research.pc_analytic_emulator.reliability import select_route

TEACHER_KEYS = {
    "sigma_cm2",
    "directions",
    "exposure_steps",
    "boundary",
    "transport_mode",
    "exhaustion_tolerance",
    "batch_tail_tolerance",
    "max_sweeps",
}


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    names = (
        "research/pc_analytic/solver.py",
        "research/pc_analytic/transport_kernel.cpp",
        "research/pc_analytic/absorption.py",
        "research/pc_analytic/compare.py",
        "research/pc_matched/workflow.py",
        "research/pc_matched/rt.py",
    )
    return _inference_source_hashes() | {name: _digest(root / name) for name in names}


def _teacher_settings(settings: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(settings, dict) or set(settings) != TEACHER_KEYS:
        raise ValueError(
            "Teacher settings must explicitly specify every recorded control"
        )
    if (
        settings["boundary"] != "source_periodic_cube"
        or settings["transport_mode"] != "characteristic"
    ):
        raise ValueError(
            "Hybrid deployment requires the recorded characteristic teacher geometry"
        )
    for key in ("directions", "exposure_steps", "max_sweeps"):
        if (
            isinstance(settings[key], bool)
            or not isinstance(settings[key], int)
            or settings[key] < 1
        ):
            raise ValueError("Teacher work counts must be positive integers")
    for key in ("sigma_cm2", "exhaustion_tolerance", "batch_tail_tolerance"):
        value = settings[key]
        if isinstance(value, bool) or not np.isfinite(value) or value < 0:
            raise ValueError("Teacher coefficients must be finite and nonnegative")
    return copy.deepcopy(settings)


def _teacher_identity(
    model: Emulator, settings: dict[str, Any], source_hashes: dict[str, str]
) -> set[str]:
    identity = model.metadata.get("teacher_identity")
    if not isinstance(identity, dict):
        raise TypeError("Deployment requires recorded teacher campaign identity")
    required = {
        "research/pc_analytic/solver.py",
        "research/pc_analytic/transport_kernel.cpp",
        "research/pc_analytic/absorption.py",
        "research/pc_analytic/compare.py",
        "research/pc_matched/workflow.py",
        "research/pc_matched/rt.py",
    }
    runtimes: set[str] = set()

    def check(record: dict[str, Any]) -> None:
        campaigns = record.get("campaigns")
        if campaigns is not None:
            if not isinstance(campaigns, list) or not campaigns:
                raise ValueError("Invalid teacher collection identity")
            for campaign in campaigns:
                check(campaign["identity"])
            return
        config = record.get("configuration", {})
        controls = config.get("analytic")
        if not isinstance(controls, dict):
            raise TypeError("Teacher identity lacks analytic controls")
        recorded_settings = {"sigma_cm2": config.get("sigma_cm2"), **controls}
        if recorded_settings != settings:
            raise ValueError("Fallback teacher settings differ from training campaigns")
        implementation = record.get("implementation_sha256", {})
        if any(implementation.get(name) != source_hashes[name] for name in required):
            raise ValueError(
                "Fallback physical implementation differs from teacher campaigns"
            )
        extension = record.get("native_runtime", {}).get("native_extension_sha256")
        if not isinstance(extension, str) or not extension:
            raise ValueError("Teacher identity lacks its native runtime checksum")
        runtimes.add(extension)

    check(identity)
    return runtimes


def freeze_deployment(
    models: dict[str, str | Path],
    policy: dict[str, Any],
    teacher_physics: dict[str, Any],
    teacher_settings: dict[str, Any],
    output: str | Path,
) -> dict[str, Any]:
    """Save self-contained models with frozen validation and teacher controls.

    Parameters
    ----------
    models
        Candidate IDs mapped to their saved, self-contained model directories.
    policy
        Validation-only policy returned by ``reliability.build_policy``.
    teacher_physics
        Recorded native parameter dictionary shared by training and fallback.
        Geometry, random seed and runtime thread count may differ at inference.
    teacher_settings
        Explicit cross-section and analytic solver controls. Their identity is
        supplied from the audited training campaign, not fitted to new inputs.
    output
        New package directory. Existing artifacts are never overwritten.

    Returns
    -------
    dict
        Manifest binding model bytes, policy, physical parameters, solver
        settings and current inference/teacher source code.
    """
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if not models or any(
        not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_]+", name)
        for name in models
    ):
        raise ValueError("Candidate IDs must be unique safe nonempty names")
    settings = _teacher_settings(teacher_settings)
    paths = {name: Path(path).resolve() for name, path in models.items()}
    if any(output.is_relative_to(path) for path in paths.values()):
        raise ValueError("Deployment output must be outside source model directories")
    hashes = {name: _model_hashes(path) for name, path in paths.items()}
    # A synthetic query validates the complete policy checksum and model map;
    # the returned route is deliberately ignored.
    select_route(policy, {"DIM": 2, "HII_DIM": 2, "BOX_LEN": 1.0}, 0.0, hashes)
    source_hashes = _source_hashes()
    runtimes: set[str] = set()
    for path in paths.values():
        model = Emulator.load(path)
        compatibility = _physics_compatibility(model, teacher_physics)
        if compatibility.get("physical_parameters_matched") is not True:
            raise ValueError("Every deployed model requires recorded teacher physics")
        runtimes.update(_teacher_identity(model, settings, source_hashes))
    output.mkdir(parents=True)
    for name, path in paths.items():
        for relative in hashes[name]:
            target = output / "models" / name / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path / relative, target)
    _write(output / "policy.json", policy)
    manifest = {
        "schema": 1,
        "models": {
            name: {"path": f"models/{name}", "files_sha256": hashes[name]}
            for name in paths
        },
        "policy_sha256": _digest(output / "policy.json"),
        "teacher_physics": copy.deepcopy(teacher_physics),
        "teacher_settings": settings,
        "native_extension_sha256": sorted(runtimes),
        "implementation_sha256": source_hashes,
        "scope": "Empirically validated coeval surrogate routes with fresh analytic teacher fallback; no independent transport-convergence certification",
    }
    _write(output / "deployment.json", manifest)
    _write(
        output / "receipt.json",
        {
            "files": {
                name: _digest(output / name)
                for name in ("deployment.json", "policy.json")
            }
        },
    )
    _load_deployment(output)
    return manifest


def _load_deployment(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, str]]]:
    receipt = json.loads((root / "receipt.json").read_text())
    if set(receipt.get("files", {})) != {"deployment.json", "policy.json"}:
        raise ValueError("Deployment receipt lacks required files")
    for name, digest in receipt["files"].items():
        if _digest(root / name) != digest:
            raise ValueError(f"Deployment checksum changed: {name}")
    record = json.loads((root / "deployment.json").read_text())
    if (
        record.get("schema") != 1
        or record.get("implementation_sha256") != _source_hashes()
    ):
        raise ValueError(
            "Deployment implementation changed; validation must be regenerated"
        )
    if record.get("policy_sha256") != _digest(root / "policy.json"):
        raise ValueError("Deployment policy checksum changed")
    _teacher_settings(record["teacher_settings"])
    hashes = {}
    for name, model in record["models"].items():
        if (
            not re.fullmatch(r"[A-Za-z0-9_]+", name)
            or model["path"] != f"models/{name}"
        ):
            raise ValueError("Unsafe deployment model path")
        path = (root / model["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Deployment model escapes its package")
        hashes[name] = _model_hashes(path)
        if hashes[name] != model["files_sha256"]:
            raise ValueError("Deployment model checksum changed")
    policy = json.loads((root / "policy.json").read_text())
    select_route(policy, {"DIM": 2, "HII_DIM": 2, "BOX_LEN": 1.0}, 0, hashes)
    return record, policy, hashes


def _field_ledger(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    source = np.sum(arrays["source"], dtype=np.longdouble)
    capacity = arrays["capacity"]
    xhi, xhii = arrays["xHI"], arrays["xHII"]
    if (
        not np.array_equal(xhi, 1 - xhii)
        or not np.isfinite(xhii).all()
        or np.any((xhii < 0) | (xhii > 1))
    ):
        raise FloatingPointError(
            "Saved hybrid ionization field violates bounds or complement"
        )
    absorbed = np.sum(
        np.multiply(capacity, xhii, dtype=np.longdouble), dtype=np.longdouble
    )
    unabsorbed = np.longdouble(arrays["unabsorbed_photons"])
    relative = abs(source - absorbed - unabsorbed) / max(source, np.finfo(float).tiny)
    if (
        not np.isfinite(unabsorbed)
        or unabsorbed < 0
        or not np.isfinite(relative)
        or relative > 1e-10
    ):
        raise FloatingPointError("Independent saved hybrid photon ledger failed")
    return {
        "source_photons": float(source),
        "absorbed_photons": float(absorbed),
        "unabsorbed_photons": float(unabsorbed),
        "relative_photon_residual": float(relative),
        "passed": True,
    }


def run(
    deployment: str | Path,
    native_directory: str | Path,
    output: str | Path,
    *,
    threads: int = 1,
) -> dict[str, Any]:
    """Predict or freshly solve one native coeval and audit the saved field.

    Parameters
    ----------
    deployment
        Frozen package produced by :func:`freeze_deployment`.
    native_directory
        Original receipt-backed native snapshot. Only source, atom and native
        ES arrays plus physical metadata are read; no teacher archive is used.
    output
        New result directory outside the model and native input packages.
    threads
        Positive analytic solver thread count. This controls execution only.

    Returns
    -------
    dict
        Route, method-specific photon accounting, elapsed computation time and
        hashes of saved arrays. The analytic route stores measured radiation,
        escaped and unemitted reservoirs separately.
    """
    deployment, native_directory, output = (
        Path(value).resolve() for value in (deployment, native_directory, output)
    )
    if output.exists():
        raise FileExistsError(output)
    if output.is_relative_to(deployment) or output.is_relative_to(native_directory):
        raise ValueError("Hybrid output must be outside model and native inputs")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer")
    manifest, policy, hashes = _load_deployment(deployment)
    manifest_hash = _digest(deployment / "deployment.json")
    native_hashes = {
        name: _digest(native_directory / name)
        for name in ("receipt.json", "native.json", "source.npz", "native.h5")
    }
    data = load_native_snapshot(native_directory, brightness="none")
    meta = data["metadata"]
    native_identity = json.loads((native_directory / "native.json").read_text())[
        "identity"
    ]
    if (
        native_identity.get("extension_sha256")
        not in manifest["native_extension_sha256"]
    ):
        raise ValueError("Native runtime differs from the teacher campaign runtime")
    physical_model = Emulator.load(
        deployment / next(iter(manifest["models"].values()))["path"]
    )
    physics = _physics_compatibility(physical_model, meta["inputs"])
    if physics.get("physical_parameters_matched") is not True:
        raise ValueError("Native physical parameters must match the deployed teacher")
    geometry = validate_geometries(
        [{name: meta["inputs"]["simulation_options"][name] for name in GEOMETRY_KEYS}]
    )[0]
    if geometry["HII_DIM"] != data["source"].shape[0] or (
        data["native_es_xhi"].shape != data["source"].shape
        or not np.isfinite(data["native_es_xhi"]).all()
        or np.any((data["native_es_xhi"] < 0) | (data["native_es_xhi"] > 1))
    ):
        raise ValueError("Native grid or ES fractions differ from the declared inputs")
    route = select_route(policy, geometry, meta["redshift"], hashes)
    arrays = {
        "source": data["source"],
        "capacity": data["capacity"],
        "native_es_xHI": data["native_es_xhi"],
    }
    input_array_hashes = {
        name: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        for name, value in arrays.items()
    }
    started = time.perf_counter()
    diagnostics: dict[str, Any] = {}
    if route["method"] == "emulator":
        model = Emulator.load(
            deployment / manifest["models"][route["candidate"]]["path"]
        )
        result = model.predict(
            arrays["source"],
            arrays["capacity"],
            arrays["native_es_xHI"],
            meta["redshift"],
            meta["box_len"],
        )
        arrays.update(
            xHI=result["xHI"],
            xHII=result["xHII"],
            unabsorbed_photons=np.asarray(result["predicted_unabsorbed_photons"]),
        )
        diagnostics.update(
            projection=result["projection"],
            unabsorbed_scope="Predicted global reservoir; no measured source-resolved escape",
        )
    elif route["method"] == "teacher":
        result = solve(
            arrays["source"],
            arrays["capacity"],
            redshift=meta["redshift"],
            box_len=meta["box_len"],
            nH_mean_comoving_cm3=meta["nH_mean_comoving_cm3"],
            threads=threads,
            **manifest["teacher_settings"],
        )
        if not result["audit"].get("complete") or not result["audit"].get(
            "ledger_passed"
        ):
            raise RuntimeError("Analytic fallback did not finish with a passing ledger")
        xhii = result["ionized"] / arrays["capacity"]
        arrays.update(
            xHI=1 - xhii,
            xHII=xhii,
            remaining_photons=result["remaining_radiation"].sum(axis=0),
            escaped_by_direction=result["escaped"],
            unemitted_source=result["unemitted_source"],
        )
        for name in ("remaining_photons", "escaped_by_direction", "unemitted_source"):
            if not np.isfinite(arrays[name]).all() or np.any(arrays[name] < 0):
                raise FloatingPointError("Invalid analytic photon reservoir")
        arrays["unabsorbed_photons"] = np.asarray(
            sum(
                np.sum(arrays[name], dtype=np.longdouble)
                for name in (
                    "remaining_photons",
                    "escaped_by_direction",
                    "unemitted_source",
                )
            ),
            dtype=np.longdouble,
        )
        diagnostics.update(
            solver_audit=result["audit"],
            solver_metadata=result["metadata"],
            unabsorbed_scope="Measured analytic radiation plus escaped and unemitted source reservoirs",
        )
    else:
        raise ValueError("Unknown deployment route")
    elapsed = time.perf_counter() - started
    if any(
        hashlib.sha256(np.ascontiguousarray(arrays[name]).tobytes()).hexdigest()
        != digest
        for name, digest in input_array_hashes.items()
    ):
        raise ValueError("Native input arrays changed during inference")
    if (
        _digest(deployment / "deployment.json") != manifest_hash
        or _load_deployment(deployment)[2] != hashes
        or any(
            _digest(native_directory / name) != digest
            for name, digest in native_hashes.items()
        )
    ):
        raise ValueError("Native input or deployment changed during inference")
    output.mkdir(parents=True)
    np.savez_compressed(output / "fields.npz", **arrays)
    with np.load(output / "fields.npz", allow_pickle=False) as saved:
        restored = {name: saved[name] for name in saved.files}
    for name in ("source", "capacity", "native_es_xHI"):
        if not np.array_equal(restored[name], arrays[name]):
            raise FloatingPointError("Saved hybrid inputs differ from their originals")
    ledger = _field_ledger(restored)
    if route["method"] == "teacher":
        independent = independent_budget(
            restored["source"],
            restored["capacity"],
            restored["xHI"],
            remaining=float(np.sum(restored["remaining_photons"], dtype=np.longdouble)),
            escaped=float(
                np.sum(restored["escaped_by_direction"], dtype=np.longdouble)
            ),
            unemitted=float(np.sum(restored["unemitted_source"], dtype=np.longdouble)),
        )
        if (
            not independent["bounded"]
            or not np.isfinite(independent["relative_photon_residual"])
            or independent["relative_photon_residual"] > 1e-10
        ):
            raise FloatingPointError("Saved analytic field reservoir audit failed")
        diagnostics["independent_neutral_field_budget"] = independent
    report = dict(
        schema=1,
        complete=True,
        route=route,
        geometry=geometry,
        redshift=meta["redshift"],
        seed=meta["seed"],
        physical_compatibility=physics,
        deployment_sha256=manifest_hash,
        native_files_sha256=native_hashes,
        photon_ledger=ledger,
        computation_seconds=elapsed,
        timing_scope="Route computation including model loading or analytic kernel compilation; excludes input audits and result I/O",
        files_sha256={"fields.npz": _digest(output / "fields.npz")},
        **diagnostics,
    )
    _write(output / "report.json", report)
    (output / "COMPLETED").write_text(
        "Saved hybrid field and photon ledger verified.\n"
    )
    return report


def main() -> None:
    """Run frozen validation routing on an original native snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", required=True, type=Path)
    parser.add_argument("--native-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    sys.stdout.write(
        json.dumps(
            run(
                args.deployment,
                args.native_directory,
                args.output,
                threads=args.threads,
            ),
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
