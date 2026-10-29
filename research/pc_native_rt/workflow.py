"""Native-budget coeval radiation-transfer comparisons.

Each redshift is an independent frozen-background RTE experiment. The original
native cumulative ``n_ion`` field and that coeval's gas density are used without
rate conversion, time integration, source rescaling, or density reweighting.
The propagation time is a relaxation coordinate, not cosmological elapsed time.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
from pathlib import Path
import sys
import tempfile
import uuid

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger(__name__)
SCOPE = "Independent native-budget coeval RTE; not a causal cosmological history"
DEFAULT_CONFIG = {
    "dimension": 8, "highres_dimension": None, "box_len": 50.0, "threads": 8,
    "seeds": [111, 222], "redshifts": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0],
    "settings": [{"directions": 96, "refinement": 1, "cfl": 0.4}],
    "photon_energy_ev": 18.0, "exhaustion_tolerance": 1e-6,
    "max_box_crossings": 20.0, "injection_box_crossings": 0.0,
    "max_steps": 1_000_000, "native_inputs_json": None,
}


def configuration(overrides: dict | None = None) -> dict:
    """Return validated independent defaults; reject unknown or unsafe settings."""
    overrides = {} if overrides is None else overrides
    if not isinstance(overrides, dict) or set(overrides) - set(DEFAULT_CONFIG):
        raise ValueError("Unknown native-RT configuration keys")
    cfg = json.loads(json.dumps(DEFAULT_CONFIG | overrides, allow_nan=False))
    for key, minimum in (("dimension", 2), ("threads", 1), ("max_steps", 1)):
        if type(cfg[key]) is not int or cfg[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    if cfg["threads"] > 32:
        raise ValueError("At most 32 threads supported")
    high = cfg["highres_dimension"]
    if high is not None and (type(high) is not int or high < cfg["dimension"]):
        raise ValueError("highres_dimension must be an integer >= dimension")
    for key in ("box_len", "photon_energy_ev", "exhaustion_tolerance", "max_box_crossings"):
        if isinstance(cfg[key], bool) or not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if cfg["photon_energy_ev"] < 13.6 or cfg["exhaustion_tolerance"] >= 1:
        raise ValueError("Ionizing photon energy and sub-unit exhaustion tolerance required")
    inject = cfg["injection_box_crossings"]
    if isinstance(inject, bool) or not np.isfinite(inject) or not 0 <= inject <= cfg["max_box_crossings"]:
        raise ValueError("Injection duration must be nonnegative and within the transport budget")
    if (not cfg["seeds"] or len(set(cfg["seeds"])) != len(cfg["seeds"])
            or any(type(x) is not int or x < 1 for x in cfg["seeds"])):
        raise ValueError("Seeds must be unique positive integers")
    zs = cfg["redshifts"]
    if (not zs or any(isinstance(z, bool) or not np.isfinite(z) or z < 0 for z in zs)
            or any(a <= b for a, b in zip(zs, zs[1:]))):
        raise ValueError("Redshifts must be finite, nonnegative, and strictly descending")
    settings = cfg["settings"]
    if not settings:
        raise ValueError("At least one RT numerical setting required")
    for row in settings:
        if set(row) != {"directions", "refinement", "cfl"}:
            raise ValueError("Each setting must have directions, refinement, and cfl")
        if type(row["directions"]) is not int or row["directions"] not in (6, 24, 26, 96, 384):
            raise ValueError("Unsupported angular direction count")
        if type(row["refinement"]) is not int or row["refinement"] < 1:
            raise ValueError("Refinement must be a positive integer")
        if isinstance(row["cfl"], bool) or not np.isfinite(row["cfl"]) or not 0 < row["cfl"] <= 1:
            raise ValueError("CFL must lie in (0,1]")
    if len({json.dumps(row, sort_keys=True) for row in settings}) != len(settings):
        raise ValueError("Duplicate RT settings")
    custom = cfg["native_inputs_json"]
    if custom is not None and (not isinstance(custom, str) or not custom or len(cfg["seeds"]) != 1):
        raise ValueError("Custom native inputs require a path and exactly one matching seed")
    return cfg


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _hash_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _hash_array(array) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def _write_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(path)


def _input_dict(inputs) -> dict:
    return json.loads(json.dumps(inputs.asdict(include_cosmo_tables="never", only_cstruct_params=False),
                                allow_nan=False))


def _native_inputs(config: dict, seed: int):
    from research.pc_reconsidered.native_driver import prepare_native, validate_inputs
    if config["native_inputs_json"] is None:
        inputs, differences = prepare_native(
            dimension=config["dimension"], box_len=config["box_len"], threads=config["threads"],
            random_seed=seed, highres_dimension=config["highres_dimension"])
        provenance = {"mode": "existing-restricted-native-baseline", "differences_from_Qin20": differences}
    else:
        from py21cmfast.input_serialization import deserialize_inputs
        from py21cmfast.wrapper.inputs import InputParameters
        path = Path(config["native_inputs_json"])
        raw = json.loads(path.read_text())
        expected = {"random_seed", "node_redshifts", "cosmo_params", "simulation_options",
                    "matter_options", "astro_params", "astro_options"}
        if set(raw) != expected:
            raise ValueError("Custom inputs must be a full snake-case InputParameters.asdict export")
        structs = deserialize_inputs({k: v for k, v in raw.items()
                                      if k not in ("random_seed", "node_redshifts")}, safe=True)
        inputs = InputParameters(**structs, random_seed=raw["random_seed"], node_redshifts=raw["node_redshifts"])
        if _input_dict(inputs) != raw:
            raise ValueError("Custom native inputs do not round-trip exactly; no silent normalization allowed")
        provenance = {"mode": "explicit-unmodified-native-inputs", "path": str(path.resolve()),
                      "sha256": _digest(path), "differences_from_supplied_inputs": {}}
    validate_inputs(inputs)
    high = config["highres_dimension"] or 3 * config["dimension"]
    checks = {
        "seed": inputs.random_seed == seed,
        "dimension": inputs.simulation_options.HII_DIM == config["dimension"],
        "highres_dimension": inputs.simulation_options.DIM == high,
        "box_len": inputs.simulation_options.BOX_LEN == config["box_len"],
        "threads": inputs.simulation_options.N_THREADS == config["threads"],
    }
    if not all(checks.values()):
        raise ValueError(f"Native inputs disagree with run configuration; inputs were not modified: {checks}")
    return inputs, provenance


def _source_hashes() -> dict:
    paths = [Path(__file__), Path(__file__).with_name("solver.py"), Path(__file__).with_name("report.py"),
             REPO / "research/pc_reconsidered/native_driver.py",
             REPO / "research/pc_field_emulator/evaluation.py",
             REPO / "research/pc_matched/workflow.py", REPO / "research/pc_matched/rt.py",
             REPO / "research/pc_matched/rt_kernel.cpp"]
    paths += [p for p in (REPO / "src/py21cmfast").rglob("*")
              if p.is_file() and p.suffix in (".py", ".c", ".h", ".toml") and p.name != "_version.py"]
    return {str(path.relative_to(REPO)): _digest(path) for path in sorted(set(paths))}


def _runtime_identity() -> dict:
    """Identify the imported native binary before any completed-run fast return."""
    import py21cmfast
    from py21cmfast import c_21cmfast
    extension = Path(c_21cmfast.__file__).resolve()
    return {"native_extension_sha256": _digest(extension),
            "native_package_version": str(py21cmfast.__version__),
            "python_version": list(sys.version_info[:3])}


def _receipt(folder: Path, filename: str, identity: dict | None = None):
    path = folder / filename
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if identity is not None and data["identity"] != identity:
        raise ValueError(f"Checkpoint identity changed: {folder}")
    for name, digest in data["files"].items():
        if not (folder / name).is_file() or _digest(folder / name) != digest:
            raise ValueError(f"Checkpoint artifact changed: {folder / name}")
    return data


def _archive_incomplete(root: Path, folder: Path) -> None:
    if folder.exists():
        archive = root / "incomplete"
        archive.mkdir(exist_ok=True)
        folder.rename(archive / f"{folder.name}_{uuid.uuid4().hex[:12]}")
    folder.mkdir(parents=True)


def _redshift_name(z: float) -> str:
    return f"z{float(z):.12g}".replace(".", "p")


def _native_snapshots(root: Path, config: dict) -> list[dict]:
    from py21cmfast import c_21cmfast
    from py21cmfast.drivers.coeval import generate_coeval
    from py21cmfast.drivers.single_field import compute_initial_conditions
    from py21cmfast.io.caching import OutputCache
    from research.pc_reconsidered.native_driver import from_native_coeval
    records = []
    for seed in config["seeds"]:
        inputs, provenance = _native_inputs(config, seed)
        input_dict = _input_dict(inputs)
        identity = {"inputs": input_dict, "provenance": provenance,
                    "extension_sha256": _digest(Path(c_21cmfast.__file__))}
        folders = {float(z): root / "native" / f"seed{seed}" / _redshift_name(z)
                   for z in config["redshifts"]}
        missing = []
        for z, folder in folders.items():
            receipt = _receipt(folder, "receipt.json", identity | {"redshift": z})
            if receipt is None:
                missing.append(z)
            else:
                records.append(receipt["record"])
        if not missing:
            continue
        with tempfile.TemporaryDirectory(prefix="native_rt_cache_", dir=root) as temporary:
            cache = OutputCache(Path(temporary))
            initial = compute_initial_conditions(inputs=inputs, regenerate=True, write=False, cache=cache)
            ic_hash = _hash_array(initial.get("hires_density"))
            prior = [r for r in records if r["seed"] == seed]
            if any(r["initial_conditions_sha256"] != ic_hash for r in prior):
                raise ValueError("Restarted native initial conditions differ from saved snapshots")
            for coeval, requested in generate_coeval(
                    inputs=inputs, initial_conditions=initial, out_redshifts=missing,
                    regenerate=True, write=False, cache=cache, progressbar=False):
                if not requested:
                    continue
                z = float(coeval.redshift)
                if z not in missing:
                    raise ValueError("Native driver returned an unexpected coeval")
                if _input_dict(coeval.inputs) != input_dict:
                    raise ValueError("Native driver altered configured inputs")
                extracted = from_native_coeval(coeval)
                if np.any(extracted.capacity <= 0):
                    raise ValueError("RT requires positive native capacity; no density clipping is applied")
                folder = folders[z]
                _archive_incomplete(root, folder)
                temporary_h5 = folder / "native.h5.tmp"
                coeval.save(temporary_h5)
                temporary_h5.replace(folder / "native.h5")
                arrays = {"source": extracted.source, "capacity": extracted.capacity,
                          "native_es_xhi": np.array(coeval.ionized_box.get("neutral_fraction")),
                          "native_brightness": np.array(coeval.brightness_temp)}
                metadata = {"identity": identity | {"redshift": z}, "source": extracted.metadata,
                            "initial_conditions_sha256": ic_hash, "scope": SCOPE,
                            "array_sha256": {k: _hash_array(v) for k, v in arrays.items()}}
                _write_npz(folder / "source.npz", **arrays)
                _write_json(folder / "native.json", metadata)
                record = {"seed": seed, "redshift": z, "h5_path": str((folder / "native.h5").relative_to(root)),
                          "fields_path": str((folder / "source.npz").relative_to(root)),
                          "metadata_path": str((folder / "native.json").relative_to(root)),
                          "source_sha256": metadata["array_sha256"]["source"],
                          "capacity_sha256": metadata["array_sha256"]["capacity"],
                          "inputs_sha256": _hash_json(input_dict), "initial_conditions_sha256": ic_hash}
                _write_json(folder / "receipt.json", {"identity": identity | {"redshift": z},
                            "files": {name: _digest(folder / name) for name in ("native.h5", "source.npz", "native.json")},
                            "record": record})
                records.append(record)
                LOGGER.info("Native coeval saved: seed=%s z=%g", seed, z)
    return sorted(records, key=lambda r: (r["seed"], -r["redshift"]))


def _repeat(array: np.ndarray, factor: int) -> np.ndarray:
    for axis in range(3):
        array = np.repeat(array, factor, axis=axis)
    return array


def _restrict(array: np.ndarray, dimension: int) -> np.ndarray:
    if array.ndim != 3 or len(set(array.shape)) != 1 or array.shape[0] % dimension:
        raise ValueError("Cannot conservatively restrict this field")
    factor = array.shape[0] // dimension
    return array.reshape(dimension, factor, dimension, factor, dimension, factor).mean(axis=(1, 3, 5))


def _run_case(root: Path, config: dict, native: dict, setting: dict) -> dict:
    from py21cmfast.drivers.coeval import Coeval
    from py21cmfast.drivers.single_field import brightness_temperature
    from research.pc_matched.workflow import opacity_parameters
    from research.pc_native_rt.solver import solve
    from research.pc_reconsidered.native_driver import _native_box, from_native_coeval
    case_id = (f"seed{native['seed']}_{_redshift_name(native['redshift'])}"
               f"_d{setting['directions']}_f{setting['refinement']}_cfl{setting['cfl']:g}")
    folder = root / "cases" / case_id
    identity = {"native": native, "setting": setting, "configuration": config}
    receipt = _receipt(folder, "receipt.json", identity)
    if receipt is not None:
        return receipt["record"]
    native_folder = (root / native["h5_path"]).parent
    _receipt(native_folder, "receipt.json")
    metadata = json.loads((root / native["metadata_path"]).read_text())
    with np.load(root / native["fields_path"], allow_pickle=False) as saved:
        arrays = {k: np.array(saved[k], copy=True) for k in saved.files}
    source, capacity = arrays["source"], arrays["capacity"]
    coeval = Coeval.from_file(root / native["h5_path"])
    extracted = from_native_coeval(coeval)
    inputs_before = _input_dict(coeval.inputs)
    if (inputs_before != metadata["identity"]["inputs"] or not np.array_equal(source, extracted.source)
            or not np.array_equal(capacity, extracted.capacity)):
        raise ValueError("Saved native source, density, or inputs changed on reload")
    physics = opacity_parameters({"inputs": inputs_before}, config["photon_energy_ev"])
    factor = setting["refinement"]
    result = solve(_repeat(source, factor), _repeat(capacity, factor),
                   redshift=native["redshift"], box_len_cMpc=config["box_len"],
                   photon_energy_ev=config["photon_energy_ev"], directions=setting["directions"],
                   cfl=setting["cfl"], threads=config["threads"],
                   hydrogen_density_z0_cm3=physics["hydrogen_density_z0_cm3"],
                   exhaustion_tolerance=config["exhaustion_tolerance"],
                   max_box_crossings=config["max_box_crossings"],
                   injection_box_crossings=config["injection_box_crossings"], max_steps=config["max_steps"])
    ionized = _restrict(result["ionized"], len(source))
    radiation = _restrict(result["radiation"].sum(axis=0), len(source))
    rt_box = _native_box(coeval, ionized, capacity)
    rt_brightness = brightness_temperature(ionized_box=rt_box, perturbed_field=coeval.perturbed_field,
                                          regenerate=True, write=False)
    rt_xhi = 1 - ionized / capacity
    total_source = source.sum(dtype=np.longdouble)
    total_capacity = capacity.sum(dtype=np.longdouble)
    total_ion = ionized.sum(dtype=np.longdouble)
    total_radiation = radiation.sum(dtype=np.longdouble)
    residual = abs(total_source - total_ion - total_radiation) / max(total_source, np.finfo(float).tiny)
    if residual > 2e-10:
        raise RuntimeError("Independent restricted-grid photon ledger failed")
    inputs_unchanged = _input_dict(coeval.inputs) == inputs_before and _input_dict(rt_box.inputs) == inputs_before
    extracted_after = from_native_coeval(coeval)
    fields_unchanged = (np.array_equal(extracted_after.source, source)
                        and np.array_equal(extracted_after.capacity, capacity)
                        and np.array_equal(coeval.ionized_box.get("neutral_fraction"), arrays["native_es_xhi"])
                        and np.array_equal(coeval.brightness_temp, arrays["native_brightness"]))
    if not inputs_unchanged or not fields_unchanged:
        raise RuntimeError("RT comparison modified native source, density, inputs, or original outputs")
    es_ions = np.sum(capacity * (1 - arrays["native_es_xhi"].astype(float)), dtype=np.longdouble)
    row = {"seed": native["seed"], "redshift": native["redshift"], "case_id": case_id,
           "setting": setting, "native": native, "fields_path": str((folder / "fields.npz").relative_to(root)),
           "source_total": float(total_source), "capacity_total": float(total_capacity),
           "native_es_xHI_volume": float(arrays["native_es_xhi"].mean(dtype=np.float64)),
           "native_es_xHI_mass": float(1 - es_ions / total_capacity),
           "rt_xHI_volume": float(rt_xhi.mean()), "rt_xHI_mass": float(1 - total_ion / total_capacity),
           "native_es_emitted_minus_ionized": float(total_source - es_ions),
           "rt_radiation_total": float(total_radiation), "ledger_relative_residual": float(residual),
           "solver_audit": result["audit"], "solver_metadata": result["metadata"],
           "native_inputs_unchanged": inputs_unchanged, "source_and_capacity_unchanged": fields_unchanged,
           "opacity": physics, "convergence_certified": False, "scope": SCOPE}
    _archive_incomplete(root, folder)
    _write_npz(folder / "fields.npz", **arrays, rt_xhi=rt_xhi, ionized=ionized, radiation=radiation,
               rt_brightness=np.array(rt_brightness.get("brightness_temp")), metadata=np.asarray(json.dumps(row, sort_keys=True)))
    _write_json(folder / "case.json", row)
    _write_json(folder / "receipt.json", {"identity": identity,
                "files": {name: _digest(folder / name) for name in ("fields.npz", "case.json")}, "record": row})
    LOGGER.info("RT coeval saved: %s, exhausted=%s", case_id, result["audit"]["converged"])
    return row


def run(root: str | Path, config: dict | None = None) -> dict:
    """Generate native coevals and solve resumable comparisons without changing them."""
    config = configuration(config)
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    custom = config["native_inputs_json"]
    identity = {"schema": 1, "configuration": config, "source_hashes": _source_hashes(),
                "native_runtime": _runtime_identity(),
                "custom_inputs_sha256": _digest(Path(custom)) if custom is not None else None, "scope": SCOPE}
    with (root / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = root / "manifest.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != identity:
                raise ValueError("Run inputs, executable sources, or native runtime changed; use a fresh output directory")
            if (root / "COMPLETED").exists():
                if _receipt(root, "completion.json", identity) is None:
                    raise ValueError("Completed run has no artifact receipt")
                return json.loads((root / "summary.json").read_text())
        else:
            if any(p.name != ".run.lock" for p in root.iterdir()):
                raise ValueError("Fresh run directory is not empty")
            _write_json(manifest, identity)
        native_records = _native_snapshots(root, config)
        rows = []
        for native in native_records:
            for setting in config["settings"]:
                rows.append(_run_case(root, config, native, setting))
        summary = {"schema": 1, "scope": SCOPE, "configuration": config,
                   "all_native_input_audits_passed": all(r["native_inputs_unchanged"] and r["source_and_capacity_unchanged"] for r in rows),
                   "all_photon_ledgers_passed": all(r["solver_audit"]["ledger_passed"] and r["ledger_relative_residual"] <= 2e-10 for r in rows),
                   "all_exhaustion_checks_passed": all(r["solver_audit"]["converged"] for r in rows),
                   "convergence_certified": False, "cases": rows}
        if _source_hashes() != identity["source_hashes"]:
            raise RuntimeError("Executable sources changed while running")
        _write_json(root / "summary.json", summary)
        from research.pc_native_rt.report import build_report
        build_report(root)
        if not summary["all_native_input_audits_passed"] or not summary["all_photon_ledgers_passed"]:
            raise RuntimeError("Native-input or photon-ledger audit failed")
        if summary["all_exhaustion_checks_passed"]:
            artifacts = [path for path in root.rglob("*") if path.is_file()
                         and "incomplete" not in path.relative_to(root).parts
                         and path.name not in (".run.lock", "COMPLETED", "completion.json")
                         and not path.name.endswith(".tmp")]
            _write_json(root / "completion.json", {"identity": identity,
                        "files": {str(path.relative_to(root)): _digest(path) for path in artifacts}})
            (root / "COMPLETED").write_text("Native equality, photon ledger and coeval exhaustion passed; numerical convergence uncertified.\n")
        else:
            (root / "COMPLETED").unlink(missing_ok=True)
            raise RuntimeError("Coeval relaxation budget exhausted before radiation/neutral exhaustion; inspect summary.json")
        return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = json.loads(args.config.read_text()) if args.config else None
    run(args.root, config)


if __name__ == "__main__":
    main()
