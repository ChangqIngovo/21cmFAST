"""Solve the radiation-transfer equation with prescribed native 21cmFAST sources.

Native 21cmFAST supplies prescribed sources and a contextual native ES history.
The controlled ES and explicit finite-volume radiation solver share integrated
sources, fixed gas, periodic boundaries and the no-recombination atom model.
Completion certifies successful audits, not converged morphology or NN accuracy.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import subprocess
import sys
import uuid
from itertools import pairwise
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG = {
    "profile": "pilot",
    "threads": 16,
    "dimension": 8,
    "box_len": 50.0,
    "seeds": [111, 222],
    "z_start": 20.0,
    "z_end": 4.0,
    "base_intervals": 10,
    "reference_z": 8.0,
    "directions": [6, 26],
    "refinements": [1, 2],
    "cfl_values": [0.4, 0.2, 0.1],
    "photon_energy_ev": 18.0,
    "memory_gb": 48.0,
    "source_strides": [1],
}
CONVERGENCE_CONFIG = DEFAULT_CONFIG | {
    "profile": "convergence", "threads": 8, "memory_gb": 24.0,
    "base_intervals": 40, "directions": [24, 96, 384],
    "refinements": [1, 2, 4], "source_strides": [4, 2, 1],
}
CHECKPOINT_CADENCE = 8


def configuration(profile: str = "pilot", **overrides: object) -> dict:
    """Return independent validated defaults for an explicitly selected profile.

    Parameters
    ----------
    profile : str
        ``pilot`` preserves the small exploratory matrix; ``convergence`` runs
        three levels on each independent numerical axis.
    **overrides : object
        Explicit settings replacing the selected profile's defaults.

    Returns
    -------
    dict
        Validated configuration with no shared mutable default lists.
    """
    if profile not in ("pilot", "convergence"):
        raise ValueError("Profile must be pilot or convergence")
    defaults = DEFAULT_CONFIG if profile == "pilot" else CONVERGENCE_CONFIG
    return validate_config(defaults | overrides)


def native_intervals(config: dict) -> int:
    """Return the finest native interval count (always four times base intervals).

    Parameters
    ----------
    config : dict
        Validated workflow configuration.

    Returns
    -------
    int
        Number of intervals generated once per seed.
    """
    return 4 * config["base_intervals"]


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_config(config: dict) -> dict:
    """Validate the bounded native generation and transport selection.

    Parameters
    ----------
    config : dict
        Complete run configuration, with keys matching ``DEFAULT_CONFIG``.

    Returns
    -------
    dict
        Independent JSON-compatible configuration.
    """
    if set(config) != set(DEFAULT_CONFIG):
        raise ValueError("Configuration keys differ from the matched workflow contract")
    result = json.loads(json.dumps(config, allow_nan=False))
    if result["profile"] not in ("pilot", "convergence"):
        raise ValueError("Profile must be pilot or convergence")
    for name, minimum in (("threads", 1), ("dimension", 4), ("base_intervals", 1)):
        if type(result[name]) is not int or result[name] < minimum:
            raise ValueError(f"{name} must be an integer at least {minimum}")
    if result["threads"] > 32:
        raise ValueError("At most 32 native threads are supported")
    for name in ("seeds", "directions", "refinements", "source_strides"):
        values = result[name]
        if not values or len(values) != len(set(values)) or any(type(v) is not int or v < 1 for v in values):
            raise ValueError(f"{name} must be unique positive integers")
    _validate_axes(result)
    for name in ("box_len", "z_start", "z_end", "reference_z", "photon_energy_ev"):
        if isinstance(result[name], bool) or not np.isfinite(result[name]):
            raise ValueError(f"{name} must be finite")
    if result["box_len"] <= 0 or not result["z_start"] > result["z_end"] >= 0:
        raise ValueError("Positive box length and descending redshift range required")
    if result["photon_energy_ev"] < 13.6:
        raise ValueError("Monochromatic photon energy must reach the H ionization threshold")
    if not np.isfinite(result["memory_gb"]) or result["memory_gb"] <= 0:
        raise ValueError("Memory budget must be positive and finite")
    if not result["z_end"] <= result["reference_z"] <= result["z_start"]:
        raise ValueError("Reference redshift must lie within the requested history")
    return result


def _validate_axes(result: dict) -> None:
    """Validate allowed, nested and adequately sampled convergence axes."""
    if any(v not in (6, 24, 26, 96, 384) for v in result["directions"]):
        raise ValueError("Supported discrete direction counts are 6, 24, 26, 96 and 384")
    if min(result["refinements"]) != 1:
        raise ValueError("Spatial refinements must include the original grid")
    if min(result["source_strides"]) != 1:
        raise ValueError("Source strides must include the finest native history")
    if any(native_intervals(result) % stride for stride in result["source_strides"]):
        raise ValueError("Every source stride must divide the finest native interval count")
    for name in ("refinements", "source_strides"):
        levels = sorted(result[name])
        if any(right % left for left, right in pairwise(levels)):
            raise ValueError(f"{name} must be nested integer multiples")
    cfl = result["cfl_values"]
    if not cfl or len(cfl) != len(set(cfl)) or any(isinstance(v, bool) or not np.isfinite(v) or not 0 < v <= 1 for v in cfl):
        raise ValueError("CFL values must be unique and in (0, 1]")
    if result["profile"] == "convergence":
        for name in ("directions", "refinements", "cfl_values", "source_strides"):
            if len(result[name]) < 3:
                raise ValueError(f"Convergence profile requires at least three {name} levels")
        if any(value not in (24, 96, 384) for value in result["directions"]):
            raise ValueError("Convergence angles must use the cubed-sphere 24/96/384 family")


def ensure_identity(root: Path, identity: dict) -> None:
    """Create or verify an immutable experiment identity.

    Parameters
    ----------
    root : pathlib.Path
        Run directory. An unrelated nonempty directory is rejected.
    identity : dict
        Configuration, fixture paths and executable-source hashes.
    """
    path = root / "experiment.json"
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError("Run identity changed; use a fresh output directory")
        return
    if any(p.name != ".run.lock" for p in root.iterdir()):
        raise FileExistsError("Unrecognized nonempty output directory")
    _write_json(path, identity)


def archive_partial(root: Path, path: Path) -> Path:
    """Preserve an incomplete phase under a unique archive name.

    Parameters
    ----------
    root : pathlib.Path
        Experiment root.
    path : pathlib.Path
        Existing incomplete directory or file.

    Returns
    -------
    pathlib.Path
        Archive destination.
    """
    directory = root / "incomplete"
    directory.mkdir(exist_ok=True)
    destination = directory / f"{path.name}_{uuid.uuid4().hex[:12]}"
    path.rename(destination)
    return destination


def write_completion(root: Path, files: list[Path]) -> None:
    """Record hashes of completed results before publishing the completion marker.

    Parameters
    ----------
    root : pathlib.Path
        Experiment directory.
    files : list of pathlib.Path
        Result files that must remain unchanged for a completed-run reuse.
    """
    _write_json(root / "completion.json", {
        "files": {str(path.resolve()): _digest(path) for path in files},
        "scope": "Completed computation and ledger audits; no convergence certificate",
    })
    (root / "COMPLETED").write_text("Matched-source comparison and independent audits completed.\n")


def verify_completion(root: Path) -> None:
    """Reject altered completed artifacts.

    Parameters
    ----------
    root : pathlib.Path
        Previously completed experiment directory.
    """
    receipt = json.loads((root / "completion.json").read_text())
    for name, expected in receipt["files"].items():
        path = Path(name)
        if not path.is_file() or _digest(path) != expected:
            raise ValueError(f"Completed artifact changed: {path}")


def _source_hashes() -> dict:
    result = {}
    paths = [p for p in (REPO / "research/pc_matched").iterdir()
             if p.is_file() and p.suffix in (".py", ".cpp") and not p.name.startswith("test_")]
    paths += [REPO / "research/pc_evolution/native_history.py",
              REPO / "research/pc_reconsidered/native_driver.py",
              REPO / "research/pc_reconsidered/source_budget.py",
              REPO / "research/pc_large_scale/pc_statistics.py",
              REPO / "research/pc_field_emulator/evaluation.py"]
    paths += [p for p in (REPO / "src/py21cmfast").rglob("*")
              if p.is_file() and p.suffix in (".py", ".c", ".h")
              and p.name != "_version.py"]
    result.update({str(p.relative_to(REPO)): _digest(p) for p in sorted(paths)})
    return result


def _check_fixture(path: Path, seed: int, config: dict) -> dict:
    from research.pc_evolution.native_history import load_history

    history = load_history(path)
    expected = np.linspace(config["z_start"], config["z_end"], native_intervals(config) + 1)
    meta = history["metadata"]
    checks = {
        "seed": meta.get("random_seed") == seed,
        "grid": history["capacity"].shape == (config["dimension"],) * 3,
        "timeline": np.array_equal(history["redshifts"], expected),
        "box": meta.get("box_len_cMpc") == config["box_len"],
        "reference": meta.get("reference_redshift") == config["reference_z"],
        "source": meta.get("source_model") == "L-INTEGRAL",
    }
    if not all(checks.values()):
        raise ValueError(f"Fixture selection mismatch: {path}: {checks}")
    sidecar = path.with_name(path.stem + "_es.npz")
    return {"seed": seed, "path": str(path), "sha256": _digest(path),
            "native_es_sidecar": str(sidecar) if sidecar.exists() else None,
            "native_es_sidecar_sha256": _digest(sidecar) if sidecar.exists() else None}


def _fixtures(root: Path, config: dict, supplied: list[Path]) -> list[Path]:
    from research.pc_evolution.workflow import worker_environment

    receipt = root / "fixtures.json"
    paths = supplied or [root / "fixtures" / f"seed{seed}.npz" for seed in config["seeds"]]
    if len(paths) != len(config["seeds"]):
        raise ValueError("Provide one fixture for every configured seed")
    if receipt.exists():
        current = [_check_fixture(p, s, config) for p, s in zip(paths, config["seeds"], strict=True)]
        if json.loads(receipt.read_text())["fixtures"] != current:
            raise ValueError("Fixture hashes changed after preparation")
        return paths
    for path, seed in zip(paths, config["seeds"], strict=True):
        sidecar = path.with_name(path.stem + "_es.npz")
        if supplied:
            _check_fixture(path, seed, config)
            continue
        path.parent.mkdir(exist_ok=True)
        if path.exists() and sidecar.exists():
            _check_fixture(path, seed, config)
            continue
        for partial in (path, sidecar):
            if partial.exists():
                archive_partial(root, partial)
        command = [sys.executable, str(REPO / "research/pc_evolution/native_history.py"),
                   "--output", str(path), "--es-output", str(sidecar),
                   "--dimension", str(config["dimension"]), "--box-len", str(config["box_len"]),
                   "--threads", str(config["threads"]), "--seed", str(seed),
                   "--z-start", str(config["z_start"]), "--z-end", str(config["z_end"]),
                   "--intervals", str(native_intervals(config)),
                   "--reference-z", str(config["reference_z"])]
        log = root / f"native_seed{seed}.log"
        LOGGER.info("Generating native source history for seed %s; log %s", seed, log)
        with log.open("a") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                           env=worker_environment(config["threads"]), check=True)
    records = [_check_fixture(p, s, config) for p, s in zip(paths, config["seeds"], strict=True)]
    _write_json(receipt, {"fixtures": records})
    return paths


def case_settings(config: dict) -> list[dict]:
    """Construct separate angular, spatial, temporal and source-history comparisons.

    Parameters
    ----------
    config : dict
        Validated experiment configuration.

    Returns
    -------
    list of dict
        Unique direction/refinement/CFL/source-stride combinations.
    """
    angular = max(config["directions"])
    cfl = max(config["cfl_values"])
    spatial = max(config["refinements"]) if config["profile"] == "convergence" else 1
    selections = [(d, spatial, cfl, 1) for d in sorted(config["directions"])]
    selections += [(angular, f, cfl, 1) for f in sorted(config["refinements"])]
    selections += [(angular, spatial, value, 1) for value in sorted(config["cfl_values"], reverse=True)]
    selections += [(angular, spatial, min(config["cfl_values"]), stride)
                   for stride in sorted(config["source_strides"], reverse=True)]
    return [{"directions": d, "refinement": f, "cfl": value, "source_stride": stride}
            for d, f, value, stride in dict.fromkeys(selections)]


def resource_estimate(config: dict) -> dict:
    """Estimate peak live RT array storage for the sequential case matrix.

    Parameters
    ----------
    config : dict
        Validated workflow configuration.

    Returns
    -------
    dict
        Matrix size and approximate array allocations in GiB. This is a memory
        planning estimate, not a wall-time prediction or a total RSS guarantee.
    """
    snapshots = native_intervals(config) + 1
    estimates = []
    for setting in case_settings(config):
        dimension = config["dimension"] * setting["refinement"]
        size = 8 * dimension**3 * (3 * snapshots + 5 * setting["directions"] + 12)
        estimates.append(setting | {"array_gib": size / 1024**3,
                                     "snapshots": snapshots, "grid_dimension": dimension})
    return {"profile": config["profile"], "cases_per_seed": len(estimates),
            "total_cases": len(estimates) * len(config["seeds"]),
            "finest_source_intervals": snapshots - 1, "finest_source_snapshots": snapshots,
            "max_array_gib": max(item["array_gib"] for item in estimates),
            "estimated_array_gib_by_case": estimates,
            "scope": "Conservative live RT arrays only; excludes native generation, libraries, filesystem caches and compression buffers"}


def _array_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def source_history(history: dict, stride: int, parent_sha256: str) -> tuple[dict, dict]:
    """Interpolate selected native rate knots on the unchanged finest time grid.

    Parameters
    ----------
    history : dict
        Finest native history. The capacity and physical-time/redshift arrays
        are reused unchanged by every source-history arm.
    stride : int
        Number of finest native intervals between retained source-rate knots.
    parent_sha256 : str
        SHA256 of the finest native fixture, preserving its exact provenance.

    Returns
    -------
    tuple of dict
        Derived history and its hashed definition. No derived history is
        presented as an independently generated native simulation.
    """
    times = np.asarray(history["times_seconds"], dtype=np.float64)
    rates = np.asarray(history["rates"], dtype=np.float64)
    if type(stride) is not int or stride < 1 or (len(times) - 1) % stride:
        raise ValueError("Source stride must divide the native interval count")
    if len(times) < 2 or np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("Source history requires increasing finite physical times")
    if rates.shape != (len(times), *history["capacity"].shape):
        raise ValueError("Source rate and capacity shapes do not match the native timeline")
    knots = np.arange(0, len(times), stride)
    sampled = rates.copy()
    for left, right in pairwise(knots):
        for node in range(left + 1, right):
            fraction = (times[node] - times[left]) / (times[right] - times[left])
            sampled[node] = (1 - fraction) * rates[left] + fraction * rates[right]
    definition = {
        "parent_fixture_sha256": parent_sha256, "source_stride": stride,
        "source_knot_indices": knots.tolist(), "native_snapshot_indices": list(range(len(times))),
        "interpolation": "Piecewise linear native source-rate knots in physical seconds, evaluated on unchanged finest time/redshift nodes",
        "background_timeline": "Identical finest native time/redshift nodes and per-interval z(t) interpolation in all arms",
        "arrays_sha256": {"rates": _array_hash(sampled), "capacity": _array_hash(history["capacity"]),
                          "times_seconds": _array_hash(times), "redshifts": _array_hash(history["redshifts"])},
        "array_hash_definition": "sha256(str(dtype).encode + json.dumps(list(shape)).encode + C-contiguous raw bytes)",
    }
    definition["source_history_sha256"] = hashlib.sha256(
        json.dumps(definition, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return history | {"rates": sampled}, definition


def opacity_parameters(metadata: dict, energy_ev: float) -> dict:
    """Derive explicit opacity parameters without altering native source counts.

    Parameters
    ----------
    metadata : dict
        Native history metadata containing the cosmological parameters.
    energy_ev : float
        Constant monochromatic photon energy in eV.

    Returns
    -------
    dict
        Opacity density and cross section, with their physical convention.

    Notes
    -----
    Native ``n_ion`` and capacity use effective H+He nuclei counts. For this
    H-only baseline every such effective atom has hydrogen opacity. This is
    an explicit effective-atom approximation, not a separate H/He chemistry
    calculation; no helium factor is silently applied to source counts.
    """
    cosmo = metadata.get("inputs", {}).get("cosmo_params", {})
    values = [cosmo.get(key) for key in ("OMb", "hlittle", "Y_He")]
    if any(not isinstance(v, (float, int)) or not np.isfinite(v) for v in values):
        raise ValueError("Native cosmology with OMb, hlittle and Y_He is required")
    omega_b, h, helium_mass = values
    if omega_b <= 0 or h <= 0 or not 0 <= helium_mass < 1:
        raise ValueError("Invalid native cosmology for effective-atom opacity")
    critical = 3 * (h * 3.2407e-18) ** 2 / (8 * np.pi * 6.67430e-8)
    density = critical * omega_b * (1 - 3 * helium_mass / 4) / 1.67262192369e-24
    return {"hydrogen_density_z0_cm3": density,
            "cross_section_cm2": 6.3e-18 * (13.6 / energy_ev) ** 3,
            "photon_energy_ev": energy_ev,
            "density_definition": "Native effective H+He nuclei density N_b0, each treated as an H absorber",
            "density_formula": "3*(h*3.2407e-18)^2/(8*pi*G)*OMb*(1-3*Y_He/4)/m_p",
            "cosmology": {key: cosmo[key] for key in ("OMb", "hlittle", "Y_He")},
            "G_cgs": 6.67430e-8, "proton_mass_g": 1.67262192369e-24,
            "source_count_rescaling": 1.0,
            "recombinations": False, "helium_chemistry": False,
            "thermal_evolution": False, "frequency_redshifting": False,
            "physical_light_speed": True, "periodic_boundaries": True}


def repeat_field(field: np.ndarray, refinement: int) -> np.ndarray:
    """Refine a piecewise-constant density or rate in mean-cell count units.

    Parameters
    ----------
    field : numpy.ndarray
        Cubic coarse field.
    refinement : int
        Number of fine cells on each coarse-cell axis.

    Returns
    -------
    numpy.ndarray
        Repeated field. Integrated counts are divided by refinement cubed when
        comparing with the original mean-cell count unit.
    """
    result = np.asarray(field, dtype=np.float64)
    for axis in range(3):
        result = np.repeat(result, refinement, axis=axis)
    return result


def _case_id(seed: int, setting: dict) -> str:
    base = f"seed{seed}_d{setting['directions']}_f{setting['refinement']}_cfl{setting['cfl']:g}"
    return f"{base}_s{setting.get('source_stride', 1)}"


def _case_result(folder: Path, identity: dict) -> dict:
    receipt = json.loads((folder / "receipt.json").read_text())
    if receipt["identity"] != identity:
        raise ValueError(f"Completed RT case identity changed: {folder}")
    fields = folder / "fields.npz"
    if _digest(fields) != receipt["fields_sha256"]:
        raise ValueError(f"Completed RT fields changed: {fields}")
    return identity["setting"] | {"seed": identity["seed"], "case_id": folder.name,
                                  "path": str(fields), "sha256": receipt["fields_sha256"],
                                  "source_history_sha256": identity["source_history"]["source_history_sha256"]}


def _save_checkpoint(folder: Path, identity: dict, node: int, photons: np.ndarray,
                     emitted: np.ndarray, ion_history: np.ndarray,
                     radiation_history: np.ndarray, emission_history: np.ndarray,
                     diagnostics: list[dict]) -> None:
    """Atomically commit a full directional state and all completed snapshots."""
    receipt_path = folder / "checkpoint.json"
    old = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
    filename = f"checkpoint_{node}_{uuid.uuid4().hex[:12]}.npz"
    checkpoint = folder / filename
    temporary = checkpoint.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        # Uncompressed arrays avoid repeated compression of the full directional
        # radiation inventory. Only the latest committed checkpoint is retained.
        np.savez(stream, photons=photons, emitted=emitted,
                 ionized=ion_history[:node + 1], radiation=radiation_history[:node + 1],
                 source_emitted=emission_history[:node + 1])
    temporary.replace(checkpoint)
    _write_json(receipt_path, {"identity": identity, "node": node,
                              "path": filename, "sha256": _digest(checkpoint),
                              "intervals": diagnostics})
    if old:
        (folder / old["path"]).unlink(missing_ok=True)


def _load_checkpoint(folder: Path, identity: dict, snapshots: int,
                     shape: tuple[int, ...]) -> tuple[int, dict, list[dict]]:
    """Validate a committed checkpoint before any solver state is reused."""
    receipt = json.loads((folder / "checkpoint.json").read_text())
    if receipt["identity"] != identity:
        raise ValueError(f"RT checkpoint identity changed: {folder}")
    filename = receipt["path"]
    if Path(filename).name != filename or not filename.startswith("checkpoint_"):
        raise ValueError("Invalid checkpoint path")
    path = folder / filename
    if not path.is_file() or _digest(path) != receipt["sha256"]:
        raise ValueError(f"RT checkpoint fields changed: {path}")
    node = receipt["node"]
    if type(node) is not int or not 1 <= node < snapshots or len(receipt["intervals"]) != node:
        raise ValueError("RT checkpoint node or interval history is invalid")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    expected = {"photons": (identity["setting"]["directions"], *shape), "emitted": shape,
                "ionized": (node + 1, *shape), "radiation": (node + 1, *shape),
                "source_emitted": (node + 1, *shape)}
    if set(arrays) != set(expected) or any(
        arrays[name].shape != dimensions or np.any(~np.isfinite(arrays[name]))
        for name, dimensions in expected.items()
    ):
        raise ValueError("RT checkpoint arrays are invalid")
    return node, arrays, receipt["intervals"]


def _run_case(root: Path, history: dict, fixture: dict, setting: dict,
              config: dict, source_hashes: dict) -> dict:
    from research.pc_matched.rt import evolve_interval

    seed = fixture["seed"]
    setting = {"source_stride": 1} | setting
    history, source_definition = source_history(history, setting["source_stride"], fixture["sha256"])
    folder = root / "cases" / _case_id(seed, setting)
    physics = opacity_parameters(history["metadata"], config["photon_energy_ev"])
    identity = {"seed": seed, "setting": setting, "physics": physics,
                "fixture_sha256": fixture["sha256"], "source_sha256": source_hashes,
                "box_len_cMpc": config["box_len"], "threads": config["threads"],
                "source_history": source_definition}
    if (folder / "COMPLETED").exists():
        return _case_result(folder, identity)
    has_checkpoint = (folder / "checkpoint.json").exists()
    if folder.exists() and not has_checkpoint:
        archive_partial(root, folder)
    folder.mkdir(parents=True, exist_ok=True)
    refinement = setting["refinement"]
    capacity = repeat_field(history["capacity"], refinement)
    snapshots = len(history["redshifts"])
    n = len(capacity)
    estimate = 8 * n**3 * (3 * snapshots + 5 * setting["directions"] + 12)
    if estimate > config["memory_gb"] * 1024**3:
        raise ValueError("Estimated RT array allocation exceeds the requested memory budget")
    ionized = np.zeros_like(capacity)
    photons = np.zeros((setting["directions"], *capacity.shape), dtype=np.float64)
    emitted = np.zeros_like(capacity, dtype=np.longdouble)
    ion_history = np.zeros((snapshots, *capacity.shape), dtype=np.float64)
    radiation_history = np.zeros_like(ion_history)
    emission_history = np.zeros_like(ion_history)
    interval_diagnostics = []
    start_node = 0
    if has_checkpoint:
        start_node, saved, interval_diagnostics = _load_checkpoint(folder, identity, snapshots, capacity.shape)
        photons, emitted = saved["photons"], saved["emitted"]
        ion_history[:start_node + 1] = saved["ionized"]
        radiation_history[:start_node + 1] = saved["radiation"]
        emission_history[:start_node + 1] = saved["source_emitted"]
        ionized = ion_history[start_node].copy()
        del saved
        LOGGER.info("%s: resuming after snapshot %s/%s", folder.name, start_node, snapshots - 1)
    for node in range(start_node + 1, snapshots):
        left = repeat_field(history["rates"][node - 1], refinement)
        right = repeat_field(history["rates"][node], refinement)
        duration = float(history["times_seconds"][node] - history["times_seconds"][node - 1])
        result = evolve_interval(
            capacity, ionized, photons, left, right, dt_seconds=duration,
            z_start=float(history["redshifts"][node - 1]), z_end=float(history["redshifts"][node]),
            box_len_cMpc=config["box_len"],
            hydrogen_density_z0_cm3=physics["hydrogen_density_z0_cm3"],
            cross_section_cm2=physics["cross_section_cm2"], cfl=setting["cfl"],
            threads=config["threads"],
        )
        ionized, photons = result["ionized"], result["photons"]
        emitted += 0.5 * (left.astype(np.longdouble) + right) * duration
        ion_history[node] = ionized
        radiation_history[node] = photons.sum(axis=0)
        emission_history[node] = emitted
        diagnostic = {"node": node, "redshift": float(history["redshifts"][node]),
                      "steps": int(result["steps"]), "ledger": result["ledger"]}
        interval_diagnostics.append(diagnostic)
        _write_json(folder / "progress.json", diagnostic)
        LOGGER.info("%s: snapshot %s/%s, z=%.3f, substeps=%s", folder.name, node,
                    snapshots - 1, history["redshifts"][node], result["steps"])
        if node == 1 or node % CHECKPOINT_CADENCE == 0 or node == snapshots - 1:
            _save_checkpoint(folder, identity, node, photons, emitted, ion_history,
                             radiation_history, emission_history, interval_diagnostics)
    metadata = identity | {
        "scope": "Direct monochromatic finite-volume discrete-ordinate RTE coupled to effective H ionization",
        "count_units": "Native effective mean-cell atom count; divide fine-grid sums by refinement^3 for coarse comparison",
        "initial_ionized": 0.0, "initial_radiation": 0.0,
        "radiation_inventory": "Physical in-flight radiation carried between intervals",
        "source_stride": setting["source_stride"],
        "source_knot_indices": source_definition["source_knot_indices"],
        "native_snapshot_indices": source_definition["native_snapshot_indices"],
        "source_history_sha256": source_definition["source_history_sha256"],
        "convergence_certified": False,
    }
    temporary = folder / "fields.npz.tmp"
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, capacity=capacity, ionized=ion_history,
                            radiation=radiation_history, source_emitted=emission_history,
                            redshifts=history["redshifts"], times_seconds=history["times_seconds"],
                            metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
    fields = folder / "fields.npz"
    temporary.replace(fields)
    _write_json(folder / "intervals.json", {"intervals": interval_diagnostics})
    _write_json(folder / "receipt.json", {"identity": identity, "fields_sha256": _digest(fields)})
    (folder / "COMPLETED").write_text("RT history completed; independent matrix audits follow.\n")
    for checkpoint in folder.glob("checkpoint_*.npz*"):
        checkpoint.unlink()
    (folder / "checkpoint.json").unlink(missing_ok=True)
    return _case_result(folder, identity)


def run(root: Path, *, fixtures: list[Path] | None = None, **settings: object) -> dict:
    """Generate native sources, execute the matched matrix and audit its results.

    Parameters
    ----------
    root : pathlib.Path
        Fresh output or an identical resumable run.
    fixtures : list of pathlib.Path, optional
        Existing native histories, ordered like the configured seeds.
    **settings : object
        Explicit profile and overrides of its defaults.

    Returns
    -------
    dict
        Independently audited comparison report.
    """
    from research.pc_evolution.native_history import load_history
    from research.pc_matched.comparison import build_report

    config = configuration(**settings)
    estimate = resource_estimate(config)
    if estimate["max_array_gib"] > config["memory_gb"]:
        raise ValueError("Estimated RT array allocation exceeds the requested memory budget")
    supplied = [Path(p).resolve() for p in fixtures or []]
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = {"schema": 2, "configuration": config, "fixtures": [str(p) for p in supplied],
                "source_sha256": _source_hashes(),
                "scope": "Direct RTE with native sources, fixed gas, effective H atoms and no recombinations"}
    with (root / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ensure_identity(root, identity)
        if (root / "COMPLETED").exists():
            verify_completion(root)
            return json.loads((root / "matched_report.json").read_text())
        paths = _fixtures(root, config, supplied)
        fixtures_record = json.loads((root / "fixtures.json").read_text())["fixtures"]
        matrix = {"configuration": config, "fixtures": fixtures_record, "cases": [],
                  "resource_estimate": estimate,
                  "source_history_design": "All arms share finest capacity/time/redshift nodes; selected emissivity knots are interpolated in physical time"}
        for path, fixture in zip(paths, fixtures_record, strict=True):
            history = load_history(path)
            for setting in case_settings(config):
                matrix["cases"].append(_run_case(root, history, fixture, setting, config,
                                                 identity["source_sha256"]))
                _write_json(root / "matrix.json", matrix)
        report = build_report(root)
        if _source_hashes() != identity["source_sha256"]:
            raise RuntimeError("Executable source changed during the run")
        artifacts = [p for p in root.rglob("*") if p.is_file()
                     and "incomplete" not in p.relative_to(root).parts
                     and p.name not in (".run.lock", "completion.json", "COMPLETED")]
        for path in paths:
            if not path.is_relative_to(root):
                artifacts.append(path)
                sidecar = path.with_name(path.stem + "_es.npz")
                if sidecar.exists():
                    artifacts.append(sidecar)
        write_completion(root, artifacts)
        return report


def main(argv: list[str] | None = None) -> None:
    """Execute a matched comparison from the command line.

    Parameters
    ----------
    argv : list of str, optional
        Explicit command-line arguments; otherwise read process arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "estimate"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--fixtures", type=Path, nargs="+")
    for key, default in DEFAULT_CONFIG.items():
        kwargs = {"default": argparse.SUPPRESS}
        if isinstance(default, list):
            kwargs.update(type=type(default[0]), nargs="+")
        else:
            kwargs["type"] = type(default)
        parser.add_argument("--" + key.replace("_", "-"), **kwargs)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if command == "estimate":
        args.pop("root")
        args.pop("fixtures")
        config = configuration(**args)
        sys.stdout.write(json.dumps({"configuration": config, "resource_estimate": resource_estimate(config)},
                                    indent=2, sort_keys=True) + "\n")
        return
    if args["root"] is None:
        parser.error("run requires --root")
    run(**args)


if __name__ == "__main__":
    main()
