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
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG = {
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
}


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
        raise ValueError("Configuration keys differ from the matched pilot contract")
    result = json.loads(json.dumps(config, allow_nan=False))
    for name, minimum in (("threads", 1), ("dimension", 4), ("base_intervals", 1)):
        if type(result[name]) is not int or result[name] < minimum:
            raise ValueError(f"{name} must be an integer at least {minimum}")
    if result["threads"] > 32:
        raise ValueError("At most 32 native threads are supported")
    for name in ("seeds", "directions", "refinements"):
        values = result[name]
        if not values or len(values) != len(set(values)) or any(type(v) is not int or v < 1 for v in values):
            raise ValueError(f"{name} must be unique positive integers")
    if any(v not in (6, 26) for v in result["directions"]):
        raise ValueError("Only 6 or 26 discrete directions are supported")
    if min(result["refinements"]) != 1:
        raise ValueError("Spatial refinements must include the original grid")
    cfl = result["cfl_values"]
    if not cfl or len(cfl) != len(set(cfl)) or any(isinstance(v, bool) or not np.isfinite(v) or not 0 < v <= 1 for v in cfl):
        raise ValueError("CFL values must be unique and in (0, 1]")
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
    expected = np.linspace(config["z_start"], config["z_end"], 4 * config["base_intervals"] + 1)
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
                   "--intervals", str(4 * config["base_intervals"]),
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
    """Construct separate angular, spatial and temporal comparisons.

    Parameters
    ----------
    config : dict
        Validated experiment configuration.

    Returns
    -------
    list of dict
        Unique direction/refinement/CFL combinations.
    """
    angular = max(config["directions"])
    cfl = max(config["cfl_values"])
    selections = [(d, 1, cfl) for d in sorted(config["directions"])]
    selections += [(angular, f, cfl) for f in sorted(config["refinements"])]
    selections += [(angular, 1, value) for value in sorted(config["cfl_values"], reverse=True)]
    return [{"directions": d, "refinement": f, "cfl": value}
            for d, f, value in dict.fromkeys(selections)]


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
    return f"seed{seed}_d{setting['directions']}_f{setting['refinement']}_cfl{setting['cfl']:g}"


def _case_result(folder: Path, identity: dict) -> dict:
    receipt = json.loads((folder / "receipt.json").read_text())
    if receipt["identity"] != identity:
        raise ValueError(f"Completed RT case identity changed: {folder}")
    fields = folder / "fields.npz"
    if _digest(fields) != receipt["fields_sha256"]:
        raise ValueError(f"Completed RT fields changed: {fields}")
    return identity["setting"] | {"seed": identity["seed"], "case_id": folder.name,
                                  "path": str(fields), "sha256": receipt["fields_sha256"]}


def _run_case(root: Path, history: dict, fixture: dict, setting: dict,
              config: dict, source_hashes: dict) -> dict:
    from research.pc_matched.rt import evolve_interval

    seed = fixture["seed"]
    folder = root / "cases" / _case_id(seed, setting)
    physics = opacity_parameters(history["metadata"], config["photon_energy_ev"])
    identity = {"seed": seed, "setting": setting, "physics": physics,
                "fixture_sha256": fixture["sha256"], "source_sha256": source_hashes,
                "box_len_cMpc": config["box_len"]}
    if (folder / "COMPLETED").exists():
        return _case_result(folder, identity)
    if folder.exists():
        archive_partial(root, folder)
    folder.mkdir(parents=True)
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
    source_history = np.zeros_like(ion_history)
    interval_diagnostics = []
    for node in range(1, snapshots):
        left = repeat_field(history["rates"][node - 1], refinement)
        right = repeat_field(history["rates"][node], refinement)
        duration = float(history["times_seconds"][node] - history["times_seconds"][node - 1])
        result = evolve_interval(
            capacity, ionized, photons, left, right, dt_seconds=duration,
            z_start=float(history["redshifts"][node - 1]), z_end=float(history["redshifts"][node]),
            box_len_cMpc=config["box_len"],
            hydrogen_density_z0_cm3=physics["hydrogen_density_z0_cm3"],
            cross_section_cm2=physics["cross_section_cm2"], cfl=setting["cfl"],
        )
        ionized, photons = result["ionized"], result["photons"]
        emitted += 0.5 * (left.astype(np.longdouble) + right) * duration
        ion_history[node] = ionized
        radiation_history[node] = photons.sum(axis=0)
        source_history[node] = emitted
        diagnostic = {"node": node, "redshift": float(history["redshifts"][node]),
                      "steps": int(result["steps"]), "ledger": result["ledger"]}
        interval_diagnostics.append(diagnostic)
        _write_json(folder / "progress.json", diagnostic)
        LOGGER.info("%s: snapshot %s/%s, z=%.3f, substeps=%s", folder.name, node,
                    snapshots - 1, history["redshifts"][node], result["steps"])
    metadata = identity | {
        "scope": "Direct monochromatic finite-volume discrete-ordinate RTE coupled to effective H ionization",
        "count_units": "Native effective mean-cell atom count; divide fine-grid sums by refinement^3 for coarse comparison",
        "initial_ionized": 0.0, "initial_radiation": 0.0,
        "radiation_inventory": "Physical in-flight radiation carried between intervals",
        "convergence_certified": False,
    }
    temporary = folder / "fields.npz.tmp"
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, capacity=capacity, ionized=ion_history,
                            radiation=radiation_history, source_emitted=source_history,
                            redshifts=history["redshifts"], times_seconds=history["times_seconds"],
                            metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
    fields = folder / "fields.npz"
    temporary.replace(fields)
    _write_json(folder / "intervals.json", {"intervals": interval_diagnostics})
    _write_json(folder / "receipt.json", {"identity": identity, "fields_sha256": _digest(fields)})
    (folder / "COMPLETED").write_text("RT history completed; independent matrix audits follow.\n")
    return _case_result(folder, identity)


def run(root: Path, *, fixtures: list[Path] | None = None, **configuration: object) -> dict:
    """Generate native sources, execute the matched matrix and audit its results.

    Parameters
    ----------
    root : pathlib.Path
        Fresh output or an identical resumable run.
    fixtures : list of pathlib.Path, optional
        Existing native histories, ordered like the configured seeds.
    **configuration : object
        Overrides of ``DEFAULT_CONFIG``.

    Returns
    -------
    dict
        Independently audited comparison report.
    """
    from research.pc_evolution.native_history import load_history
    from research.pc_matched.comparison import build_report

    config = validate_config(DEFAULT_CONFIG | configuration)
    supplied = [Path(p).resolve() for p in fixtures or []]
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = {"schema": 1, "configuration": config, "fixtures": [str(p) for p in supplied],
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
        matrix = {"configuration": config, "fixtures": fixtures_record, "cases": []}
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
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, nargs="+")
    for key, default in DEFAULT_CONFIG.items():
        kwargs = {"default": default}
        if isinstance(default, list):
            kwargs.update(type=type(default[0]), nargs="+")
        else:
            kwargs["type"] = type(default)
        parser.add_argument("--" + key.replace("_", "-"), **kwargs)
    args = vars(parser.parse_args(argv))
    args.pop("command")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(**args)


if __name__ == "__main__":
    main()
