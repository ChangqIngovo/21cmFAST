"""Generate independent native-ES/C2-Ray paired histories for field emulation.

The native ES fields are contextual proxies: their cumulative source estimator
and evolving density differ from the teacher's explicitly integrated rates and
frozen density contrast.  This dataset learns ES-to-RT differences; it does not
isolate photon non-conservation as their only cause.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.es_pc_emulator.initial_state import (  # noqa: E402
    DEFAULT_POLICY,
    initial_state_audit,
    require_initial_state,
    validate_policy,
)

DEFAULT_CONFIG = {
    "schema_version": 1,
    "initial_state": dict(DEFAULT_POLICY),
    "teacher": {
        "dimension": 16,
        "highres_dimension": 48,
        "box_len": 50.0,
        "z_start": 20.0,
        "z_end": 5.0,
        "intervals": 30,
        "reference_z": 8.0,
        "refinement": 3,
        "time_substeps": 2,
        "stride": 1,
        "transport": "conservative_rays",
        "angular_order": 32,
    },
    "splits": {"train": [1234, 2345], "validation": [3456], "test": [4567]},
}
SCOPE = (
    "Native ES proxies to prescribed-source C2-Ray H+He/thermal labels; "
    "fixed gas-density contrast, no source feedback. Native ES source/"
    "density closure differs from teacher. No NN photon-conservation or "
    "cosmological convergence certificate is implied."
)


def write_json(path: Path, value: dict) -> None:
    """Atomically publish a JSON record owned by this workflow."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def digest(path: Path) -> str:
    """Hash a file without loading large payloads into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_config(value: dict) -> dict:
    """Require complete, disjoint seed splits and conservative teacher settings."""
    from research.c2ray_teacher.runner import validate_config as validate_teacher

    if not isinstance(value, dict):
        raise TypeError("Dataset configuration must be an object")
    if set(value) - set(DEFAULT_CONFIG):
        raise ValueError("Unknown dataset configuration keys")
    if value.get("schema_version", 1) != 1:
        raise ValueError("Unsupported dataset configuration schema")
    teacher = copy.deepcopy(DEFAULT_CONFIG["teacher"])
    updates = value.get("teacher", {})
    if not isinstance(updates, dict):
        raise TypeError("teacher must be an object")
    if set(updates) & {"seed", "history", "benchmark"}:
        raise ValueError("Seeds, histories and benchmark are controlled by the dataset workflow")
    teacher.update(updates)
    checked = validate_teacher(teacher)
    if checked["transport"] != "conservative_rays" or checked["stride"] != 1:
        raise ValueError("Dataset requires conservative_rays and stride=1")
    splits = copy.deepcopy(value.get("splits", DEFAULT_CONFIG["splits"]))
    if not isinstance(splits, dict) or set(splits) != {"train", "validation", "test"}:
        raise ValueError("Exactly train, validation and test splits are required")
    seen = set()
    for name, seeds in splits.items():
        if not isinstance(seeds, list) or not seeds:
            raise ValueError(f"Nonempty seed list required for {name}")
        for seed in seeds:
            if type(seed) is not int or seed <= 0:
                raise ValueError("Seeds must be positive integers")
            if seed in seen:
                raise ValueError("Every initial-condition seed belongs to exactly one split")
            seen.add(seed)
    # Seed and benchmark are set independently for every run below.
    checked.pop("seed")
    checked.pop("benchmark")
    return {"schema_version": 1, "teacher": checked, "splits": splits,
            "initial_state": validate_policy(value.get("initial_state"))}


def audit_initial_history(history_path, es_path, policy, initial_xhii):
    """Audit actual native initial fields before spending time on RT transport."""
    from research.es_pc_emulator.pairs import _load_es
    from research.pc_evolution.native_history import load_history

    history = load_history(history_path)
    es, _ = _load_es(es_path, history)
    return initial_state_audit(history, es, initial_xhii=initial_xhii,
                               max_mass_weighted_error=policy["max_mass_weighted_error"])


def runtime_fingerprint() -> dict:
    """Record the exact native package, interpreter and compiler runtime."""
    packages = {}
    native = None
    for name in ("numpy", "scipy", "h5py", "py21cmfast"):
        module = importlib.import_module(name)
        packages[name] = str(module.__version__)
        if name == "py21cmfast":
            native = Path(module.__file__).resolve().parent
    native_files = {
        str(path.relative_to(native)): digest(path)
        for path in sorted(native.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
        and path.suffix not in (".pyc", ".pyo")
    }
    if not any(name.endswith((".so", ".pyd", ".dylib")) for name in native_files):
        raise RuntimeError("The selected runtime has no compiled native extension")
    compilers = {}
    for name in ("gfortran", "mpifort"):
        found = shutil.which(name)
        if found is None:
            raise RuntimeError(f"Missing compiler: {name}")
        path = Path(found).resolve()
        compilers[name] = {
            "path": str(path), "sha256": digest(path),
            # OpenMPI selects its wrapper data from argv[0]. Invoking the
            # resolved opal_wrapper target loses the mpifort identity.
            "version": subprocess.check_output([found, "--version"], text=True),
        }
    return {
        "python": sys.executable, "python_version": platform.python_version(),
        "package_versions": packages, "native_package": str(native),
        "native_files_sha256": native_files, "compilers": compilers,
    }


def verify_prepared(run: Path, workers: int) -> dict:
    """Reject mutated configuration, runtime, allocation or prepared locations."""
    run = run.resolve()
    manifest = json.loads((run / "run_manifest.json").read_text())
    if manifest.get("experiment") != "es-to-c2ray-training-dataset":
        raise RuntimeError("Wrong preparation experiment")
    if manifest.get("schema_version") != 1 or manifest.get("run_directory") != str(run):
        raise RuntimeError("Prepared run moved or schema differs")
    if type(workers) is not int or manifest.get("requested_workers") != workers:
        raise RuntimeError("PBS allocation differs from preparation")
    config = validate_config(json.loads((run / "dataset_config.json").read_text()))
    if config != manifest.get("config"):
        raise RuntimeError("Prepared dataset configuration changed")
    if runtime_fingerprint() != manifest.get("runtime"):
        raise RuntimeError("Prepared interpreter, native package or compiler changed")
    source = Path(manifest["source_checkout"])
    if source != run / "source":
        raise RuntimeError("Prepared source location changed")
    subprocess.run(
        ["git", "-C", str(source), "diff", "--exit-code", manifest["commit"], "--"],
        check=True,
    )
    return manifest


def generation_identity(config: dict) -> dict:
    """Freeze all numerical source files and the runtime for durable continuation."""
    paths = [Path(__file__), Path(__file__).with_name("pairs.py"),
             Path(__file__).with_name("initial_state.py")]
    paths += list((ROOT / "research/c2ray_teacher").glob("*.py"))
    paths += list((ROOT / "research/c2ray_teacher").glob("*.F90"))
    paths += list((ROOT / "research/c2ray_teacher").glob("*.f90"))
    paths += [ROOT / "research/pc_evolution/native_history.py",
              ROOT / "research/pc_reconsidered/native_driver.py",
              ROOT / "research/c2ray_teacher/upstream_lock.json"]
    return {
        "schema_version": 1, "config": config, "runtime": runtime_fingerprint(),
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(paths)) if p.is_file()},
    }


def _seed_receipt(directory: Path, manifest_path: Path, *, seed: int, split: str,
                  smoke_passed: bool) -> dict:
    """Record only completely finished seed outputs for exact later reuse."""
    return {
        "schema_version": 1, "seed": seed, "split": split,
        "manifest": str(manifest_path), "smoke_passed": smoke_passed,
        "files_sha256": {str(p.relative_to(directory)): digest(p)
                         for p in sorted(manifest_path.parent.parent.rglob("*")) if p.is_file()},
    }


def _reuse_seed(directory: Path, *, seed: int, split: str) -> dict:
    """Rehash a completed seed and independently re-audit every paired sample."""
    from research.es_pc_emulator.pairs import load_pair

    receipt = json.loads((directory / "COMPLETED.json").read_text())
    if receipt.get("schema_version") != 1 or receipt.get("seed") != seed or receipt.get("split") != split:
        raise ValueError("Completed seed receipt differs from requested split")
    for name, expected in receipt.get("files_sha256", {}).items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file() or digest(path) != expected:
            raise ValueError(f"Completed seed output changed: {name}")
    if not receipt.get("files_sha256"):
        raise ValueError("Completed seed has no immutable output hashes")
    manifest_path = Path(receipt["manifest"]).resolve()
    if not manifest_path.is_relative_to(directory.resolve()):
        raise ValueError("Completed seed manifest escaped its original directory")
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("samples"):
        raise ValueError("Completed seed has no samples")
    for record in manifest["samples"]:
        if record.get("seed") != seed or record.get("split") != split:
            raise ValueError("Completed seed sample has changed split identity")
        load_pair(record)
    return receipt


def run_dataset(config: dict, output: Path, upstream: Path, workers: int = 32, *, resume: bool = False) -> dict:
    """Generate or exactly resume independent audited seed histories.

    Successful seeds are reused only after rehashing all saved outputs and
    re-auditing pairs. An interrupted seed restarts in a fresh attempt directory;
    its incomplete source history and native binary output are retained.
    """
    from research.c2ray_teacher.runner import parallel_layout
    from research.c2ray_teacher.runner import run as run_teacher
    from research.es_pc_emulator.pairs import (
        combine_manifests,
        export_pairs,
        load_dataset,
    )
    from research.pc_evolution.native_history import build_native_history

    config = validate_config(config)
    parallel_layout(workers)
    output, upstream = Path(output).resolve(), Path(upstream).resolve()
    if not upstream.is_dir():
        raise ValueError("Pinned C2-Ray source directory does not exist")
    identity = generation_identity(config)
    if resume:
        if not output.is_dir() or not (output / "generation_identity.json").is_file():
            raise ValueError("Resume requires an existing dataset generation identity")
        saved = json.loads((output / "generation_identity.json").read_text())
        if saved != identity or json.loads((output / "config.json").read_text()) != config:
            raise ValueError("Resume configuration, source code or runtime changed")
        if (output / "FAILED.json").is_file():
            index = 1
            while (output / f"FAILED_{index:04d}.json").exists():
                index += 1
            (output / "FAILED.json").rename(output / f"FAILED_{index:04d}.json")
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "config.json", config)
        write_json(output / "generation_identity.json", identity)
    report = {"schema_version": 1, "scope": SCOPE, "config": config, "cases": []}
    smoke_done = False
    try:
        for split in ("train", "validation", "test"):
            for seed in config["splits"][split]:
                directory = output / f"seed_{seed}"
                directory.mkdir(exist_ok=resume)
                if (directory / "COMPLETED.json").is_file():
                    receipt = _reuse_seed(directory, seed=seed, split=split)
                    manifest_path = Path(receipt["manifest"])
                    manifest = json.loads(manifest_path.read_text())
                    smoke_done |= receipt["smoke_passed"]
                    report["cases"].append({"seed": seed, "split": split,
                        "manifest": str(manifest_path), "samples": len(manifest["samples"]), "reused": True})
                    write_json(output / "progress.json", report)
                    continue
                attempt_index = 1
                while (directory / f"attempt_{attempt_index:04d}").exists():
                    attempt_index += 1
                attempt = directory / f"attempt_{attempt_index:04d}"
                attempt.mkdir()
                teacher = config["teacher"] | {"seed": seed}
                history_path, es_path = attempt / "native_history.npz", attempt / "native_es.npz"
                keys = (
                    "dimension", "box_len", "seed", "z_start", "z_end", "intervals",
                    "reference_z", "highres_dimension",
                )
                build_native_history(
                    output=history_path, es_output=es_path, threads=workers,
                    **{key: teacher[key] for key in keys},
                )
                initial_audit = audit_initial_history(
                    history_path, es_path, config["initial_state"], teacher["initial_xhii"],
                )
                write_json(attempt / "initial_state_audit.json", initial_audit)
                require_initial_state(initial_audit,
                                      allow_mismatch=config["initial_state"]["allow_mismatch"])
                # runner executes an actual compiled eight-source MPI/OpenMP
                # smoke test before its first science history. It is never mocked
                # or treated as a spatial-convergence certificate in production.
                summary = run_teacher(
                    teacher | {"history": str(history_path), "benchmark": not smoke_done},
                    attempt / "teacher", upstream, workers,
                )
                if not smoke_done and not summary.get("benchmarks", {}).get("passed"):
                    raise RuntimeError("Native C2-Ray smoke validation did not pass")
                performed_smoke = not smoke_done
                smoke_done = True
                manifest = export_pairs(
                    history_path, es_path, attempt / "teacher", attempt / "pairs",
                    split, expected_seed=seed,
                    initial_state_config=config["initial_state"],
                )
                manifest_path = attempt / "pairs/manifest.json"
                write_json(directory / "COMPLETED.json", _seed_receipt(
                    directory, manifest_path, seed=seed, split=split, smoke_passed=performed_smoke,
                ))
                report["cases"].append({
                    "seed": seed, "split": split,
                    "manifest": str(manifest_path),
                    "samples": len(manifest["samples"]), "reused": False,
                })
                write_json(output / "progress.json", report)
        dataset_path = output / "dataset_manifest.json"
        if dataset_path.exists():
            audited = load_dataset(dataset_path)
            expected = []
            for case in report["cases"]:
                expected.extend(json.loads(Path(case["manifest"]).read_text())["samples"])
            if audited["samples"] != expected:
                raise ValueError("Existing combined dataset differs from completed seed receipts")
        else:
            combine_manifests([row["manifest"] for row in report["cases"]], dataset_path)
            audited = load_dataset(dataset_path)
        report.update(dataset_manifest=str(dataset_path), samples=len(audited["samples"]))
        write_json(output / "summary.json", report)
        (output / "COMPLETED").write_text(
            "Paired dataset generation and integrity checks passed; spatial accuracy is not certified.\n"
        )
        return report
    except Exception as error:
        write_json(output / "FAILED.json", {"type": type(error).__name__, "message": str(error)})
        raise


def main() -> None:
    """Generate a new paired dataset from a pinned source configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    options = parser.parse_args()
    config = json.loads(options.config.read_text())
    report = run_dataset(config, options.output, options.upstream, options.workers, resume=options.resume)
    sys.stdout.write(json.dumps({"dataset_manifest": report["dataset_manifest"], "samples": report["samples"]}) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
