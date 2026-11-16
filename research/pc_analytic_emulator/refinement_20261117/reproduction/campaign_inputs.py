"""Prepare isolated, resumable native inputs for the refinement comparison.

Only identities, checksums, completeness and photon ledgers are inspected here.
Blind campaigns never enter either training/validation collection.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

COMMON_Z = (10, 9, 8, 7.5, 7, 6.5, 6)
PROFILES = {
    "train_resolution": (tuple(range(2241, 2245)), COMMON_Z, 32, 50.0),
    "train_volume": (tuple(range(2251, 2255)), COMMON_Z, 32, 256.0),
    "validation_resolution": ((2261, 2262), COMMON_Z, 32, 50.0),
    "validation_volume": ((2271, 2272), COMMON_Z, 32, 256.0),
    "blind_original": ((2201, 2202), COMMON_Z, 16, 50.0),
    "blind_resolution": ((2211, 2212), COMMON_Z, 32, 50.0),
    "blind_volume": ((2221, 2222), COMMON_Z, 32, 256.0),
    "blind_unseen_box": ((2231, 2232), COMMON_Z, 32, 100.0),
}
REUSED_PROFILES = {
    "original": (tuple(range(2101, 2109)), COMMON_Z, 16, 50.0),
    "old_external": (tuple(range(2109, 2113)), COMMON_Z, 16, 50.0),
    "old_train": (
        tuple(range(2121, 2129)),
        (10, 9, 8, 7.5, 7, 6.75, 6.5, 6.25, 6, 5.75),
        16,
        50.0,
    ),
    "old_validation": (tuple(range(2113, 2117)), COMMON_Z, 16, 50.0),
    "old_resolution": ((2131, 2132), (8, 7.5, 6.5, 6), 32, 50.0),
    "old_volume": ((2141, 2142), (8, 7.5, 6.5, 6), 32, 256.0),
}
ANALYTIC = {
    "directions": 96,
    "exposure_steps": 32,
    "boundary": "source_periodic_cube",
    "transport_mode": "characteristic",
    "exhaustion_tolerance": 1e-7,
    "batch_tail_tolerance": 0.0001,
    "max_sweeps": 10000,
}
SIGMA_CM2 = 2.7173135802469135e-18
SINGLE_TRAIN_SEEDS = (*range(2101, 2113), 2115, 2116, *range(2121, 2129))
MIXED_TRAIN_SEEDS = (
    *SINGLE_TRAIN_SEEDS,
    2131,
    2132,
    2141,
    2142,
    *range(2241, 2245),
    *range(2251, 2255),
)
VALIDATION_SEEDS = (2113, 2114, 2261, 2262, 2271, 2272)


def digest(path: str | Path) -> str:
    """Hash bytes without interpreting scientific arrays or manifests."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _freeze(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        if _json(path) != value:
            raise ValueError(f"Frozen record changed: {path}")
    else:
        _write(path, value)


def _git(source: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source), *arguments], text=True
    ).strip()


def native_environment() -> dict[str, str]:
    """Return the fixed two-thread teacher environment without import overrides."""
    environment = os.environ.copy()
    environment.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        PYTHONNOUSERSITE="1",
        MPLBACKEND="Agg",
        OMP_NUM_THREADS="2",
        OMP_THREAD_LIMIT="2",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OMP_WAIT_POLICY="PASSIVE",
        OMP_DYNAMIC="FALSE",
    )
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    return environment


def _probe_runtime(source: Path, executable: str) -> dict[str, Any]:
    probe = (
        "import json; import research.pc_analytic.campaign as c; "
        "print(json.dumps({'native_runtime':c._runtime_identity(),"
        "'implementation_sha256':c._implementation_hashes(),"
        "'campaign_path':c.__file__}))"
    )
    result = json.loads(
        subprocess.check_output(
            [executable, "-c", probe], cwd=source, text=True, env=native_environment()
        )
    )
    if (
        Path(result.pop("campaign_path")).resolve()
        != (source / "research/pc_analytic/campaign.py").resolve()
    ):
        raise ValueError("Native probe imported campaign code outside frozen source")
    return result


def _resolve_reuse(root: Path, names: tuple[str, ...]) -> Path:
    matches = [root / name for name in names if (root / name).exists()]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one reused campaign in {root}: {names}")
    return matches[0].resolve()


def _reference(name: str, root: Path) -> dict[str, Any]:
    return {
        "name": name,
        "path": str(root),
        "completion_sha256": digest(root / "completion.json"),
        "identity_sha256": digest(root / "identity.json"),
        "manifest_sha256": digest(root / "dataset_manifest.json"),
    }


def _completion(root: Path, expected: int) -> dict[str, Any]:
    receipt = _json(root / "completion.json")
    if digest(root / "identity.json") != receipt["identity_sha256"]:
        raise ValueError(f"Completed campaign identity changed: {root}")
    for name in ("dataset_manifest.json", "summary.json"):
        if digest(root / name) != receipt["files"].get(name):
            raise ValueError(f"Completed campaign file changed: {root / name}")
    summary = _json(root / "summary.json")
    if (
        summary.get("complete") is not True
        or summary.get("all_photon_ledgers_passed") is not True
        or summary.get("completed_snapshots") != expected
        or summary.get("requested_snapshots") != expected
        or summary.get("failed_seeds") != []
    ):
        raise ValueError(f"Campaign incomplete or photon ledger failed: {root}")
    return {
        "output": str(root),
        "complete": True,
        "completed_snapshots": expected,
        "requested_snapshots": expected,
        "all_photon_ledgers_passed": True,
        "completion_sha256": digest(root / "completion.json"),
        "identity_sha256": digest(root / "identity.json"),
        "manifest_sha256": digest(root / "dataset_manifest.json"),
    }


def _check_identity(
    root: Path, profile: tuple, runtime: dict[str, Any], template_hash: str
) -> None:
    identity = _json(root / "identity.json")
    config = identity["configuration"]
    seeds, redshifts, dimension, box_len = profile
    expected = {
        "dimension": dimension,
        "highres_dimension": 96,
        "box_len": box_len,
        "brightness": "native",
        "sigma_cm2": SIGMA_CM2,
        "analytic": ANALYTIC,
        "native_template_sha256": template_hash,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError(
            f"Reused/generated campaign physics or geometry differs: {root}"
        )
    if sorted(config["seeds"]) != sorted(seeds) or sorted(
        config["redshifts"]
    ) != sorted(redshifts):
        raise ValueError(f"Campaign seed/redshift membership differs: {root}")
    for key in ("native_runtime", "implementation_sha256"):
        if identity[key] != runtime[key]:
            raise ValueError(f"Campaign {key} differs from frozen runtime: {root}")


def prepare_inputs(
    source: str | Path,
    previous_campaign_root: str | Path,
    native_campaign_root: str | Path,
    output_root: str | Path,
    native_python: str | Path,
    native_template: str | Path,
    commit: str = "HEAD",
) -> dict[str, Any]:
    """Freeze source, native runtime, eight profiles and reused receipt identities.

    The executable path deliberately retains its venv symlink. Existing output
    records must match exactly; a changed source commit or input cannot resume.
    """
    source = Path(source).expanduser().resolve()
    previous = Path(previous_campaign_root).expanduser().resolve()
    native = Path(native_campaign_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    template = Path(native_template).expanduser().resolve()
    executable = str(Path(native_python).expanduser().absolute())
    reuse = {
        "original": _resolve_reuse(
            native, ("teacher_results_verified", "teacher_development_original")
        ),
        "old_external": _resolve_reuse(
            native, ("teacher_results_external", "teacher_development_external")
        ),
        "old_train": _resolve_reuse(previous, ("teacher_train", "teacher_train_new")),
        "old_validation": _resolve_reuse(previous, ("teacher_validation",)),
        "old_resolution": _resolve_reuse(previous, ("teacher_transfer_resolution",)),
        "old_volume": _resolve_reuse(previous, ("teacher_transfer_volume",)),
    }
    if output == source or any(output.is_relative_to(path) for path in reuse.values()):
        raise ValueError("Output must be separate from source and reused campaigns")
    output.mkdir(parents=True, exist_ok=True)
    frozen_source = output / "source"
    revision = _git(source, "rev-parse", f"{commit}^{{commit}}")
    if not frozen_source.exists():
        _git(source, "worktree", "add", "--detach", str(frozen_source), revision)
    if _git(frozen_source, "rev-parse", "HEAD") != revision:
        raise ValueError("Output source belongs to a different commit")
    if _git(frozen_source, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Frozen source has tracked modifications")
    copied_template = output / "native_template.json"
    if copied_template.exists():
        if digest(copied_template) != digest(template):
            raise ValueError("Frozen native template changed")
    else:
        shutil.copyfile(template, copied_template)
    template_hash = digest(copied_template)
    runtime = _probe_runtime(frozen_source, executable)
    reused = {}
    for name, root in reuse.items():
        profile = REUSED_PROFILES[name]
        _check_identity(root, profile, runtime, template_hash)
        _completion(root, len(profile[0]) * len(profile[1]))
        reused[name] = _reference(name, root)
    configs = {}
    for name, (seeds, redshifts, dimension, box_len) in PROFILES.items():
        config = {
            "dimension": dimension,
            "highres_dimension": 96,
            "box_len": box_len,
            "seeds": list(seeds),
            "redshifts": list(redshifts),
            "workers": 2,
            "threads": 2,
            "cpus": 4,
            "brightness": "native",
            "sigma_cm2": SIGMA_CM2,
            "analytic": dict(ANALYTIC),
            "native_template_json": str(copied_template),
            "native_template_sha256": template_hash,
            "native_extension_sha256": runtime["native_runtime"][
                "native_extension_sha256"
            ],
        }
        path = output / f"{name}_config.json"
        _freeze(path, config)
        configs[name] = {
            "path": str(path),
            "sha256": digest(path),
            "output": str(output / f"teacher_{name}"),
            "expected_snapshots": len(seeds) * len(redshifts),
        }
    protocol = {
        "schema": 1,
        "source_commit": revision,
        "source_path": str(frozen_source),
        "native_python": executable,
        "output_root": str(output),
        "previous_campaign_root": str(previous),
        "native_campaign_root": str(native),
        "protocol_path": str(output / "generation_protocol.json"),
        "reuse_roots": {name: str(root) for name, root in reuse.items()},
        "runtime": runtime,
        "implementation_sha256": runtime["implementation_sha256"],
        "native_runtime": runtime["native_runtime"],
        "native_template": str(copied_template),
        "template_sha256": template_hash,
        "campaign_configs": configs,
        "reused_campaigns": reused,
        "training_new_seeds": list(range(2241, 2245)) + list(range(2251, 2255)),
        "validation_new_seeds": [2261, 2262, 2271, 2272],
        "blind_seeds": [2201, 2202, 2211, 2212, 2221, 2222, 2231, 2232],
        "generator_sha256": digest(Path(__file__)),
        "evaluation_policy": "Blind outputs are integrity-audited only until final model and inference settings are frozen; no blind campaigns enter collections.",
    }
    _freeze(Path(protocol["protocol_path"]), protocol)
    audit_inputs(protocol)
    return protocol


def prepare_archived_inputs(
    source: str | Path,
    previous_campaign_root: str | Path,
    native_campaign_root: str | Path,
    generated_campaign_root: str | Path,
    output_root: str | Path,
    commit: str = "HEAD",
) -> dict[str, Any]:
    """Bind completed archives to a frozen training checkout without native imports.

    Every archived teacher must share its recorded runtime, implementation and
    template identity. Those historical identities need not match the machine
    or training checkout used now. Original archive receipts remain untouched.
    """
    source = Path(source).expanduser().resolve()
    previous = Path(previous_campaign_root).expanduser().resolve()
    native = Path(native_campaign_root).expanduser().resolve()
    generated = Path(generated_campaign_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    reuse = {
        "original": _resolve_reuse(
            native, ("teacher_results_verified", "teacher_development_original")
        ),
        "old_external": _resolve_reuse(
            native, ("teacher_results_external", "teacher_development_external")
        ),
        "old_train": _resolve_reuse(previous, ("teacher_train", "teacher_train_new")),
        "old_validation": _resolve_reuse(previous, ("teacher_validation",)),
        "old_resolution": _resolve_reuse(previous, ("teacher_transfer_resolution",)),
        "old_volume": _resolve_reuse(previous, ("teacher_transfer_volume",)),
    }
    archives = {name: (generated / f"teacher_{name}").resolve() for name in PROFILES}
    if output in (source, previous, native, generated) or any(
        output.is_relative_to(path) for path in (*reuse.values(), *archives.values())
    ):
        raise ValueError("Output must be separate from source and archived campaigns")
    # Validate historical provenance before creating the training checkout.
    identity = _json(reuse["original"] / "identity.json")
    runtime = {
        key: identity[key] for key in ("native_runtime", "implementation_sha256")
    }
    template_hash = identity["configuration"]["native_template_sha256"]
    reused, archived, configs = {}, {}, {}
    for name, root in {**reuse, **archives}.items():
        profile = REUSED_PROFILES[name] if name in reuse else PROFILES[name]
        _check_identity(root, profile, runtime, template_hash)
        count = len(profile[0]) * len(profile[1])
        _completion(root, count)
        reference = _reference(name, root)
        if name in reuse:
            reused[name] = reference
        else:
            archived[name] = reference
            configs[name] = {
                "path": str(root / "identity.json"),
                "sha256": reference["identity_sha256"],
                "configuration_source": "identity.json:configuration",
                "output": str(root),
                "expected_snapshots": count,
            }
    output.mkdir(parents=True, exist_ok=True)
    frozen_source = output / "source"
    revision = _git(source, "rev-parse", f"{commit}^{{commit}}")
    if not frozen_source.exists():
        _git(source, "worktree", "add", "--detach", str(frozen_source), revision)
    if _git(frozen_source, "rev-parse", "HEAD") != revision:
        raise ValueError("Output source belongs to a different commit")
    if _git(frozen_source, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Frozen source has tracked modifications")
    protocol = {
        "schema": 1,
        "inputs_mode": "archived",
        "source_commit": revision,
        "source_path": str(frozen_source),
        "native_python": None,
        "output_root": str(output),
        "previous_campaign_root": str(previous),
        "native_campaign_root": str(native),
        "generated_campaign_root": str(generated),
        "protocol_path": str(output / "generation_protocol.json"),
        "reuse_roots": {name: str(root) for name, root in reuse.items()},
        "runtime": runtime,
        "implementation_sha256": runtime["implementation_sha256"],
        "native_runtime": runtime["native_runtime"],
        "native_template": None,
        "template_sha256": template_hash,
        "campaign_configs": configs,
        "reused_campaigns": reused,
        "archived_campaigns": archived,
        "training_new_seeds": list(range(2241, 2245)) + list(range(2251, 2255)),
        "validation_new_seeds": [2261, 2262, 2271, 2272],
        "blind_seeds": [2201, 2202, 2211, 2212, 2221, 2222, 2231, 2232],
        "generator_sha256": digest(Path(__file__)),
        "evaluation_policy": "Blind outputs are integrity-audited only until final model and inference settings are frozen; no blind campaigns enter collections.",
    }
    _freeze(Path(protocol["protocol_path"]), protocol)
    audit_inputs(protocol)
    return protocol


def _audit_archived_inputs(protocol: dict[str, Any]) -> dict[str, Any]:
    """Audit immutable historical identities without importing a native runtime."""
    if set(protocol["reused_campaigns"]) != set(REUSED_PROFILES) or set(
        protocol["archived_campaigns"]
    ) != set(PROFILES):
        raise ValueError("Archived campaign membership differs")
    if set(protocol["campaign_configs"]) != set(PROFILES):
        raise ValueError("Archived configuration membership differs")
    if protocol["runtime"] != {
        "native_runtime": protocol["native_runtime"],
        "implementation_sha256": protocol["implementation_sha256"],
    }:
        raise ValueError("Archived runtime records disagree")
    for name, expected in {
        **protocol["reused_campaigns"],
        **protocol["archived_campaigns"],
    }.items():
        root = Path(expected["path"])
        if _reference(name, root) != expected:
            raise ValueError(f"Archived campaign receipt changed: {name}")
        profile = REUSED_PROFILES[name] if name in REUSED_PROFILES else PROFILES[name]
        count = len(profile[0]) * len(profile[1])
        _check_identity(root, profile, protocol["runtime"], protocol["template_sha256"])
        _completion(root, count)
        if name in PROFILES:
            spec = protocol["campaign_configs"][name]
            if spec != {
                "path": str(root / "identity.json"),
                "sha256": expected["identity_sha256"],
                "configuration_source": "identity.json:configuration",
                "output": str(root),
                "expected_snapshots": count,
            }:
                raise ValueError(f"Archived campaign configuration changed: {name}")
        elif protocol["reuse_roots"].get(name) != str(root):
            raise ValueError(f"Archived reuse path differs: {name}")
    return {
        "complete": True,
        "inputs_mode": "archived",
        "source_commit": protocol["source_commit"],
        "native_runtime": protocol["native_runtime"],
        "template_sha256": protocol["template_sha256"],
        "native_probe_performed": False,
        "archived_campaign_count": 14,
    }


def audit_inputs(protocol: dict[str, Any]) -> dict[str, Any]:
    """Verify frozen source/runtime/template/config identities without simulation."""
    if _json(Path(protocol["protocol_path"])) != protocol:
        raise ValueError("Generation protocol changed")
    if digest(Path(__file__)) != protocol["generator_sha256"]:
        raise ValueError("Generation helper changed")
    source = Path(protocol["source_path"])
    if _git(source, "rev-parse", "HEAD") != protocol["source_commit"] or _git(
        source, "status", "--porcelain", "--untracked-files=no"
    ):
        raise ValueError("Frozen source commit or tracked content changed")
    if protocol.get("inputs_mode") == "archived":
        return _audit_archived_inputs(protocol)
    for relative, expected in protocol["implementation_sha256"].items():
        if digest(source / relative) != expected:
            raise ValueError(f"Teacher implementation changed: {relative}")
    if _probe_runtime(source, protocol["native_python"]) != protocol["runtime"]:
        raise ValueError("Native runtime or imported teacher code changed")
    if digest(protocol["native_template"]) != protocol["template_sha256"]:
        raise ValueError("Native template changed")
    for spec in protocol["campaign_configs"].values():
        if digest(spec["path"]) != spec["sha256"]:
            raise ValueError("Frozen campaign configuration changed")
    return {
        "complete": True,
        "source_commit": protocol["source_commit"],
        "native_runtime": protocol["native_runtime"],
        "template_sha256": protocol["template_sha256"],
    }


def generate_inputs(
    protocol: dict[str, Any], profiles: list[str] | tuple[str, ...] | None = None
) -> list[dict[str, Any]]:
    """Generate requested profiles sequentially and return integrity-only records."""
    names = list(PROFILES) if profiles is None else list(profiles)
    if (
        isinstance(profiles, str)
        or not names
        or len(names) != len(set(names))
        or any(name not in PROFILES for name in names)
    ):
        raise ValueError("Select unique, known campaign profile names")
    if protocol.get("inputs_mode") == "archived":
        raise ValueError("Archived inputs are read-only; native generation is disabled")
    audit_inputs(protocol)
    output = Path(protocol["output_root"])
    status_path = output / "generation_status.json"
    protocol_hash = digest(protocol["protocol_path"])
    records = {}
    if status_path.exists():
        previous = _json(status_path)
        if previous["protocol_sha256"] != protocol_hash:
            raise ValueError("Generation status belongs to another protocol")
        for record in previous["campaigns"]:
            name = record["campaign"]
            spec = protocol["campaign_configs"][name]
            records[name] = {
                "campaign": name,
                **_completion(Path(spec["output"]), spec["expected_snapshots"]),
            }
    for name in names:
        audit_inputs(protocol)
        spec = protocol["campaign_configs"][name]
        command = [
            protocol["native_python"],
            "-m",
            "research.pc_analytic.campaign",
            "--config",
            spec["path"],
            "--output",
            spec["output"],
        ]
        with (output / f"{name}.log").open("a") as log:
            subprocess.run(
                command,
                cwd=protocol["source_path"],
                env=native_environment(),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        audit_inputs(protocol)
        root = Path(spec["output"])
        _check_identity(
            root, PROFILES[name], protocol["runtime"], protocol["template_sha256"]
        )
        records[name] = {
            "campaign": name,
            **_completion(root, spec["expected_snapshots"]),
        }
        _write(
            status_path,
            {
                "schema": 1,
                "protocol_sha256": protocol_hash,
                "complete": set(records) == set(PROFILES),
                "campaigns": [records[key] for key in PROFILES if key in records],
            },
        )
    return [records[name] for name in names]


def build_collections(protocol: dict[str, Any]) -> dict[str, Path]:
    """Write six-/ten-campaign development collections, excluding all blind data."""
    audit_inputs(protocol)
    references = {}
    for name, expected in protocol["reused_campaigns"].items():
        root = Path(expected["path"])
        if _reference(name, root) != expected:
            raise ValueError(f"Reused campaign receipt changed: {name}")
        profile = REUSED_PROFILES[name]
        _check_identity(root, profile, protocol["runtime"], protocol["template_sha256"])
        _completion(root, len(profile[0]) * len(profile[1]))
        references[name] = expected
    for name in (
        "validation_resolution",
        "validation_volume",
        "train_resolution",
        "train_volume",
    ):
        spec = protocol["campaign_configs"][name]
        root = Path(spec["output"])
        _check_identity(
            root, PROFILES[name], protocol["runtime"], protocol["template_sha256"]
        )
        _completion(root, spec["expected_snapshots"])
        references[name] = _reference(name, root)
    common = (
        "original",
        "old_external",
        "old_train",
        "old_validation",
        "validation_resolution",
        "validation_volume",
    )
    collections = {}
    for name, members, seeds, count in (
        ("guarded_single", common, SINGLE_TRAIN_SEEDS, 178),
        (
            "guarded_mixed",
            (
                *common,
                "old_resolution",
                "old_volume",
                "train_resolution",
                "train_volume",
            ),
            MIXED_TRAIN_SEEDS,
            250,
        ),
    ):
        destination = Path(protocol["output_root"]) / f"collection_{name}"
        destination.mkdir(exist_ok=True)
        _freeze(
            destination / "collection_manifest.json",
            {
                "schema": 1,
                "kind": "analytic-teacher-campaign-collection",
                "allow_mixed_geometry": True,
                "campaigns": [references[key] for key in members],
                "split_protocol_sha256": digest(protocol["protocol_path"]),
                "train_seeds": list(seeds),
                "validation_seeds": list(VALIDATION_SEEDS),
                "training_snapshots": count,
                "validation_snapshots": 42,
                "creation_scope": "Known development and specified validation only; no new blind campaigns included",
            },
        )
        collections[name] = destination
    return collections
