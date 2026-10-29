"""Continuous finite-c RT with native source rates and conservative 2LPT gas.

Ionized atoms and directional photons persist through every proper-time step.
The native coeval ES is context; its cumulative proxy is not the integral of
the native instantaneous source rate. No global history or field is fitted.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import logging
from pathlib import Path
import time

import numpy as np

from research.pc_native_rt.workflow import (
    REPO, _digest, _hash_json, _receipt, _repeat, _restrict, _runtime_identity,
    _source_hashes as _native_source_hashes, _write_json, _write_npz,
)

LOGGER = logging.getLogger(__name__)
SCOPE = "Continuous finite-c effective-atom RT with native instantaneous sources and native 2LPT mass advection; current restricted no-recombination/no-thermal native configuration"


def refinement_ladder():
    """Five independent two-step refinements sharing one finest anchor."""
    anchor = dict(directions=384, refinement=4, cfl=.1, time_substeps=4, source_stride=1)
    rows = [anchor]
    for name, values in (("directions", (24, 96)), ("refinement", (1, 2)),
                         ("cfl", (.4, .2)), ("time_substeps", (1, 2)),
                         ("source_stride", (4, 2))):
        rows.extend(anchor | {name: value} for value in values)
    return sorted(rows, key=lambda r: (r["refinement"], r["directions"], -r["cfl"],
                                      r["time_substeps"], -r["source_stride"]))


DEFAULT_CONFIG = dict(
    dimension=8, highres_dimension=None, box_len=50., threads=8, seeds=[111, 222],
    z_start=25., z_end=5., source_intervals=160, checkpoint_stride=4,
    settings=refinement_ladder(), photon_energy_ev=18., native_inputs_json=None,
    max_steps=1_000_000, tolerance=.01,
)


def configuration(overrides=None):
    overrides = {} if overrides is None else overrides
    if not isinstance(overrides, dict) or set(overrides) - set(DEFAULT_CONFIG):
        raise ValueError("Unknown causal RT configuration keys")
    cfg = json.loads(json.dumps(DEFAULT_CONFIG | overrides, allow_nan=False))
    for name, minimum in (("dimension", 4), ("threads", 1), ("source_intervals", 1),
                          ("checkpoint_stride", 1), ("max_steps", 1)):
        if type(cfg[name]) is not int or cfg[name] < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if cfg["threads"] > 32:
        raise ValueError("At most 32 threads supported")
    high = cfg["highres_dimension"]
    if high is not None and (type(high) is not int or high < cfg["dimension"]):
        raise ValueError("highres_dimension must be an integer >= dimension")
    for name in ("box_len", "z_start", "z_end", "photon_energy_ev", "tolerance"):
        if isinstance(cfg[name], bool) or not np.isfinite(cfg[name]):
            raise ValueError(f"{name} must be finite")
    if not cfg["z_start"] > cfg["z_end"] >= 0 or cfg["box_len"] <= 0:
        raise ValueError("Positive box length and descending nonnegative redshifts required")
    if cfg["photon_energy_ev"] < 13.6 or not 0 < cfg["tolerance"] < 1:
        raise ValueError("Ionizing photon energy and fractional convergence tolerance required")
    if (not cfg["seeds"] or len(set(cfg["seeds"])) != len(cfg["seeds"])
            or any(type(s) is not int or s < 1 for s in cfg["seeds"])):
        raise ValueError("Seeds must be unique positive integers")
    names = {"directions", "refinement", "cfl", "time_substeps", "source_stride"}
    if not cfg["settings"]:
        raise ValueError("At least one numerical setting is required")
    for row in cfg["settings"]:
        if set(row) != names:
            raise ValueError("Each setting requires directions/refinement/cfl/time_substeps/source_stride")
        for key in names - {"cfl"}:
            if type(row[key]) is not int or row[key] < 1:
                raise ValueError("Numerical count settings must be positive integers")
        if row["directions"] not in (6, 24, 26, 96, 384):
            raise ValueError("Unsupported angular quadrature")
        if (isinstance(row["cfl"], bool) or not np.isfinite(row["cfl"])
                or not 0 < row["cfl"] <= .5):
            raise ValueError("MUSCL positivity requires CFL in (0,0.5]")
        if cfg["source_intervals"] % row["source_stride"]:
            raise ValueError("Each source stride must divide source_intervals")
    if len({_hash_json(row) for row in cfg["settings"]}) != len(cfg["settings"]):
        raise ValueError("Duplicate numerical settings")
    custom = cfg["native_inputs_json"]
    if custom is not None and (not isinstance(custom, str) or not custom or len(cfg["seeds"]) != 1):
        raise ValueError("Custom native inputs require a path and one matching seed")
    return cfg


def _source_hashes():
    paths = list(Path(__file__).parent.glob("*.py")) + list(Path(__file__).parent.glob("*.cpp"))
    paths += [REPO / "research/pc_evolution/native_history.py", REPO / "research/pc_large_scale/pc_statistics.py"]
    return _native_source_hashes() | {str(p.relative_to(REPO)): _digest(p) for p in paths}


def sample_rates(rates, times, stride):
    """Change only source knots; preserve the native background/time grid."""
    rates, times = np.asarray(rates, float), np.asarray(times, float)
    if (type(stride) is not int or stride < 1 or (len(times) - 1) % stride
            or rates.shape[0] != len(times) or not np.all(np.diff(times) > 0)):
        raise ValueError("Source stride must divide the ordered physical timeline")
    sampled = rates.copy()
    for left in range(0, len(times) - 1, stride):
        right = left + stride
        for i in range(left + 1, right):
            weight = (times[i] - times[left]) / (times[right] - times[left])
            sampled[i] = (1 - weight) * rates[left] + weight * rates[right]
    return sampled


def _positions(history, growth, factor):
    from .native_history import positions_at_growth
    return positions_at_growth(history["packets"], growth) * factor + (factor - 1) / 2


def advance_node(history, node, setting, config, ionized, photons, cumulative_source):
    """Strang gas motion / fixed-midpoint RT / gas motion, retaining both states.

    Source knots are interpolated in proper time; the native time/growth
    functions supply the background at each split boundary. Matter
    splitting is refined independently of the internal radiation CFL steps.
    All atom clouds retain their original native physical width at every f.
    """
    from .gas import remap_ions
    from .transport import evolve_interval
    from .native_history import background_at_time
    from research.pc_matched.workflow import opacity_parameters
    factor = setting["refinement"]
    n = config["dimension"] * factor
    times = history["times_seconds"]
    rates = history["sampled_rates"]
    duration = float(times[node] - times[node - 1])
    mass = history["packets"]["mass"] * factor**3
    physics = opacity_parameters(history["metadata"], config["photon_energy_ev"])
    gas_error, rt_error, ion_excess, steps = 0., 0., 0., 0
    emitted = cumulative_source.copy()
    for sub in range(setting["time_substeps"]):
        fractions = np.array([sub, sub + .5, sub + 1]) / setting["time_substeps"]
        background = [background_at_time(history, node, float(x)) for x in fractions]
        old, mid, new = (_positions(history, value["growth_factor"], factor) for value in background)
        first = remap_ions(ionized, mass, old, mid, n, cloud_width=factor)
        a, b = fractions[0], fractions[-1]
        left = (1 - a) * rates[node - 1] + a * rates[node]
        right = (1 - b) * rates[node - 1] + b * rates[node]
        dt = duration / setting["time_substeps"]
        result = evolve_interval(
            first["capacity"], first["ionized"], photons, _repeat(left, factor), _repeat(right, factor),
            dt_seconds=dt, z_start=background[0]["redshift"], z_end=background[-1]["redshift"],
            box_len_cMpc=config["box_len"], hydrogen_density_z0_cm3=physics["hydrogen_density_z0_cm3"],
            cross_section_cm2=physics["cross_section_cm2"], cfl=setting["cfl"],
            maximum_steps=config["max_steps"], threads=config["threads"],
        )
        last = remap_ions(result["ionized"], mass, mid, new, n, cloud_width=factor)
        ionized, photons = last["ionized"], result["photons"]
        emitted += (left.astype(np.longdouble) + right) * (np.longdouble(dt) / 2)
        for remap in (first, last):
            gas_error = max(gas_error, remap["audit"]["ion_relative_residual"],
                            remap["audit"]["mass_relative_residual"])
        rt_error = max(rt_error, result["ledger"]["relative_residual"])
        ion_excess = max(ion_excess, result["ledger"]["maximum_fraction_excess"])
        steps += result["steps"]
    audit = audit_state(last["capacity"], ionized, photons, emitted, factor,
                        history["packets"]["mass"].sum(dtype=np.longdouble))
    audit.update(maximum_gas_substep_residual=gas_error, maximum_rt_substep_residual=rt_error,
                 maximum_rt_substep_ion_fraction_excess=ion_excess,
                 radiation_steps=steps, matter_substeps=setting["time_substeps"])
    return dict(capacity=last["capacity"], ionized=ionized, photons=photons,
                cumulative_source=emitted, audit=audit)


def audit_state(capacity, ionized, photons, source, factor, atom_total):
    atoms = np.sum(capacity, dtype=np.longdouble) / factor**3
    ions = np.sum(ionized, dtype=np.longdouble) / factor**3
    radiation = np.sum(photons, dtype=np.longdouble) / factor**3
    emitted = np.sum(source, dtype=np.longdouble)
    tiny = np.longdouble(np.finfo(float).tiny)
    photon_error = float(abs(emitted - ions - radiation) / max(emitted, tiny))
    atom_error = float(abs(atoms - atom_total) / max(atom_total, tiny))
    excess = np.divide(ionized - capacity, capacity, out=np.zeros_like(capacity), where=capacity > 0)
    if (not all(np.isfinite(x).all() for x in (capacity, ionized, photons, source))
            or any((x < 0).any() for x in (capacity, ionized, photons, source))
            or np.any(ionized - capacity > 2e-11 * capacity)
            or photon_error > 2e-10 or atom_error > 2e-11):
        raise RuntimeError("Independent cumulative photon/atom ledger or species bound failed")
    return dict(photon_ledger_relative_residual=photon_error, atom_ledger_relative_residual=atom_error,
                total_emitted=float(emitted), total_ionized=float(ions), total_radiation=float(radiation),
                total_atoms=float(atoms), rt_xHI_mass=float(1 - ions / atoms),
                maximum_ion_fraction_excess=max(0., float(excess.max())), ion_bound_tolerance=2e-11,
                passed=True)


def _checkpoint(root, folder, history, node, setting, config, state):
    from py21cmfast.drivers.coeval import Coeval
    from py21cmfast.drivers.single_field import brightness_temperature
    from research.pc_reconsidered.native_driver import _native_box
    from research.pc_matched.comparison import es_threshold, es_map
    n = config["dimension"]
    capacity, ions = (_restrict(state[k], n) for k in ("capacity", "ionized"))
    radiation = _restrict(state["photons"].sum(axis=0), n)
    source = np.asarray(state["cumulative_source"], dtype=float)
    coeval = Coeval.from_file(Path(history["metadata"]["native_archive_root"]) / history["records"][node]["h5_path"])
    rt_box = _native_box(coeval, ions, capacity)
    brightness = brightness_temperature(ionized_box=rt_box, perturbed_field=coeval.perturbed_field,
                                        regenerate=True, write=False).get("brightness_temp")
    maximum_radius = min(n / 2, coeval.inputs.astro_params.R_BUBBLE_MAX * n / config["box_len"])
    threshold, filtering = es_threshold(source, capacity, maximum_radius)
    es_ions = es_map(source, capacity, threshold, 1.)
    es_box = _native_box(coeval, es_ions, capacity)
    matched_brightness = brightness_temperature(ionized_box=es_box, perturbed_field=coeval.perturbed_field,
                                                regenerate=True, write=False).get("brightness_temp")
    xhi = np.divide(capacity - ions, capacity, out=np.ones_like(capacity), where=capacity > 0)
    matched_xhi = np.divide(capacity - es_ions, capacity, out=np.ones_like(capacity), where=capacity > 0)
    path = folder / "checkpoints" / f"node{node:05d}.npz"
    _write_npz(path, capacity=capacity, ionized=ions, radiation=radiation,
               cumulative_source=source, rt_xhi=xhi, rt_brightness=np.asarray(brightness),
               native_xhi=history["native_xhi"][node], native_brightness=history["native_brightness"][node],
               native_capacity=history["native_capacity"][node], matched_es_xhi=matched_xhi,
               matched_es_brightness=np.asarray(matched_brightness))
    density_difference = float(np.max(np.abs(capacity - history["native_capacity"][node])))
    return dict(node=node, redshift=float(history["redshifts"][node]),
                time_seconds=float(history["times_seconds"][node]), fields_path=str(path.relative_to(root)),
                fields_sha256=_digest(path), rt_xHI_volume=float(xhi.mean()),
                matched_es_xHI_volume=float(matched_xhi.mean()),
                native_es_xHI_volume=float(history["native_xhi"][node].mean()),
                maximum_native_density_roundoff=density_difference, matched_es_filtering=filtering,
                **state["audit"])


def _case_id(seed, setting):
    return (f"seed{seed}_d{setting['directions']}_f{setting['refinement']}_cfl{setting['cfl']:g}"
            f"_t{setting['time_substeps']}_s{setting['source_stride']}")


def run_case(root, history, seed, setting, config, sources):
    from .gas import deposit
    folder = root / "cases" / _case_id(seed, setting)
    folder.mkdir(parents=True, exist_ok=True)
    history_digest = _digest(Path(history["metadata"]["native_archive_root"]) / "history.npz")
    identity = dict(configuration=config, setting=setting, seed=seed, history_sha256=history_digest,
                    source_hashes=sources, native_runtime=_runtime_identity())
    completed = _receipt(folder, "receipt.json", identity)
    if completed is not None:
        return completed["record"]
    sampled = sample_rates(history["rates"], history["times_seconds"], setting["source_stride"])
    history = history | {"sampled_rates": sampled}
    factor = setting["refinement"]
    n = config["dimension"] * factor
    restart = _receipt(folder, "restart.json", identity)
    checkpoints = []
    start_node = 0
    start_clock = time.monotonic()
    if restart:
        start_node = restart["node"]
        checkpoints = restart["checkpoints"]
        with np.load(folder / restart["state_file"], allow_pickle=False) as saved:
            state = {k: np.array(saved[k]) for k in ("capacity", "ionized", "photons", "cumulative_source")}
        audit_state(state["capacity"], state["ionized"], state["photons"], state["cumulative_source"],
                    factor, history["packets"]["mass"].sum(dtype=np.longdouble))
    else:
        capacity = deposit(history["packets"]["mass"] * factor**3,
                           _positions(history, history["growth_factors"][0], factor), n, cloud_width=factor)
        state = dict(capacity=capacity, ionized=np.zeros_like(capacity),
                     photons=np.zeros((setting["directions"], n, n, n)),
                     cumulative_source=np.zeros((config["dimension"],) * 3, dtype=np.longdouble))
        state["audit"] = audit_state(capacity, state["ionized"], state["photons"], state["cumulative_source"],
                                     factor, history["packets"]["mass"].sum(dtype=np.longdouble))
        checkpoints.append(_checkpoint(root, folder, history, 0, setting, config, state))
    for node in range(start_node + 1, len(history["redshifts"])):
        state = advance_node(history, node, setting, config, state["ionized"], state["photons"], state["cumulative_source"])
        LOGGER.info("%s node %d/%d z=%.3f xHI_mass=%.5f ledger=%.3g", folder.name, node,
                    len(history["redshifts"]) - 1, history["redshifts"][node],
                    state["audit"]["rt_xHI_mass"], state["audit"]["photon_ledger_relative_residual"])
        _write_json(folder / "progress.json", dict(node=node, audit=state["audit"]))
        if node % config["checkpoint_stride"] == 0 or node == len(history["redshifts"]) - 1:
            checkpoints.append(_checkpoint(root, folder, history, node, setting, config, state))
            state_file = f"state_{node:05d}.npz"
            _write_npz(folder / state_file, **{k: state[k] for k in ("capacity", "ionized", "photons", "cumulative_source")})
            files = {str((root / row["fields_path"]).relative_to(folder)): row["fields_sha256"] for row in checkpoints}
            files[state_file] = _digest(folder / state_file)
            _write_json(folder / "restart.json", dict(identity=identity, node=node, state_file=state_file,
                                                      files=files, checkpoints=checkpoints))
            for old in folder.glob("state_*.npz"):
                if old.name != state_file:
                    old.unlink()
    record = dict(case_id=folder.name, seed=seed, setting=setting, history_sha256=history_digest,
                  checkpoints=checkpoints, all_ledgers_passed=True, elapsed_seconds=time.monotonic() - start_clock,
                  scope=SCOPE, convergence_certified=False)
    _write_json(folder / "case.json", record)
    restart = _receipt(folder, "restart.json", identity)
    files = restart["files"] | {"restart.json": _digest(folder / "restart.json"), "case.json": _digest(folder / "case.json")}
    _write_json(folder / "receipt.json", dict(identity=identity, files=files, record=record))
    return record


def run(root, config=None):
    from .native_history import build_native_history
    from .report import build_report
    config = configuration(config)
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = dict(configuration=config, source_hashes=_source_hashes(), native_runtime=_runtime_identity(),
                    custom_inputs_sha256=_digest(Path(config["native_inputs_json"])) if config["native_inputs_json"] else None)
    with (root / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = root / "manifest.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != identity:
                raise ValueError("Inputs or executable changed; use a fresh causal RT directory")
            completed = _receipt(root, "completion.json", identity)
            if completed is not None and (root / "COMPLETED").exists():
                return json.loads((root / "summary.json").read_text())
        else:
            if any(p.name != ".run.lock" for p in root.iterdir()):
                raise ValueError("New causal RT directory is not empty")
            _write_json(manifest, identity)
        cases, histories = [], []
        for seed in config["seeds"]:
            history = build_native_history(root / "histories" / f"seed{seed}", config, seed)
            histories.append(dict(seed=seed, path=f"histories/seed{seed}/history.npz",
                                  sha256=_digest(root / "histories" / f"seed{seed}" / "history.npz"),
                                  metadata=history["metadata"]))
            for setting in config["settings"]:
                cases.append(run_case(root, history, seed, setting, config, identity["source_hashes"]))
        if _source_hashes() != identity["source_hashes"]:
            raise RuntimeError("Executable sources changed during calculation")
        summary = dict(schema=1, scope=SCOPE, configuration=config, histories=histories, cases=cases,
                       all_ledgers_passed=all(c["all_ledgers_passed"] for c in cases),
                       convergence_certified=False)
        _write_json(root / "summary.json", summary)
        report = build_report(root)
        summary["convergence_certified"] = report["convergence_certified"]
        _write_json(root / "summary.json", summary)
        artifacts = [p for p in root.rglob("*") if p.is_file()
                     and p.name not in (".run.lock", "completion.json", "COMPLETED") and not p.name.endswith(".tmp")]
        _write_json(root / "completion.json", dict(identity=identity,
                    files={str(p.relative_to(root)): _digest(p) for p in artifacts}))
        (root / "COMPLETED").write_text("Continuous evolution and photon/atom audits completed. Numerical certification is in causal_rt_report.json.\n")
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(args.root, json.loads(args.config.read_text()) if args.config else None)


if __name__ == "__main__":
    main()
