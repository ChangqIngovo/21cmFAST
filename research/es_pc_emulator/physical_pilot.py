"""Small native RT physical-control experiment, not production training data.

The density/source template is native 21cmFAST; explicit controlled transforms
then vary six physical properties. The ES prior is a separately implemented
periodic photon-budget excursion-set approximation, NOT a native coeval.
This diagnostic holds the density contrast fixed in time. Its posterior is
conditional on that approximation and on one specified IC realization.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from research.c2ray_teacher.compare import restrict_fields
from research.c2ray_teacher.data import prepare_history
from research.c2ray_teacher.runner import (
    _build,
    _launch,
    read_output,
    read_transport_diagnostics,
    validate_output,
)
from research.pc_evolution.native_history import (
    ARRAY_KEYS,
    _write_fresh_history,
    array_hash,
    build_native_history,
    load_history,
    validate_history,
)
from research.pc_field_emulator.evaluation import shell_spectra

ROOT = Path(__file__).resolve().parents[2]
MYR = 365.25 * 86400 * 1e6
PARAMETER_NAMES = [
    "log10_source_amplitude", "history_tilt", "source_concentration",
    "density_contrast_scale", "log10_Tbb_K", "log10_clumping",
]
BOUNDS = np.array([[np.log10(2), np.log10(8)], [-1, 1], [.7, 1.3],
                   [.7, 1.3], [np.log10(35000), np.log10(70000)], [0, np.log10(5)]])
FIDUCIAL = BOUNDS.mean(axis=1)
SED_LOCK = threading.Lock()


def write_json(path, value):
    """Write strict JSON for reproducible scientific records."""
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def integrated_history(rates, times):
    """Exact integral and first time moment of piecewise-linear source rates."""
    rates, times = np.asarray(rates), np.asarray(times)
    dt = np.diff(times).reshape((-1, 1, 1, 1))
    t0 = times[:-1].reshape((-1, 1, 1, 1))
    q0, q1 = rates[:-1], rates[1:]
    increments = .5 * (q0 + q1) * dt
    moment_increments = t0 * increments + dt**2 * (q0 / 6 + q1 / 3)
    photons = np.concatenate([np.zeros_like(rates[:1]), np.cumsum(increments, axis=0)])
    moments = np.concatenate([np.zeros_like(rates[:1]), np.cumsum(moment_increments, axis=0)])
    mean_time = np.divide(moments, photons, out=np.zeros_like(moments), where=photons > 0)
    return photons, mean_time


def transform_history(base, theta):
    """Preserve gas mass and per-source integrated photons under timing changes.

    Source concentration changes rate contrast at each endpoint while retaining
    that endpoint's total rate. Timing tilt is applied next, normalized for
    EACH source's final cumulative emission. Amplitude is applied last.
    """
    validate_history(base)
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (6,) or not np.isfinite(theta).all() or np.any(theta < BOUNDS[:, 0]) or np.any(theta > BOUNDS[:, 1]):
        raise ValueError("Controls must be six finite values within declared bounds")
    history = copy.deepcopy(base)
    rates = base["rates"].copy()
    totals = rates.sum(axis=(1, 2, 3), keepdims=True)
    mean = totals / base["capacity"].size
    relative = np.divide(rates, mean, out=np.zeros_like(rates), where=mean > 0)
    concentrated = relative ** theta[2]
    concentrated *= np.divide(totals, concentrated.sum(axis=(1, 2, 3), keepdims=True),
                              out=np.zeros_like(totals), where=totals > 0)
    before, _ = integrated_history(concentrated, base["times_seconds"])
    u = 2 * base["times_seconds"] / base["times_seconds"][-1] - 1
    tilted = concentrated * np.exp(theta[1] * u[:, None, None, None])
    after, _ = integrated_history(tilted, base["times_seconds"])
    tilted *= np.divide(before[-1], after[-1], out=np.ones_like(after[-1]), where=after[-1] > 0)
    history["rates"] = np.ascontiguousarray(tilted * 10**theta[0])
    rho = base["capacity"] ** theta[3]
    history["capacity"] = np.ascontiguousarray(rho * base["capacity"].sum() / rho.sum())
    history["metadata"].update(
        source_model="controlled_native_L-INTEGRAL",
        rates_definition="controlled_transform_of_native_rates",
        physical_pilot_transform={
            "parameter_names": PARAMETER_NAMES, "theta": theta.tolist(),
            "parent_array_sha256": base["metadata"]["array_sha256"],
            "definition": "endpoint-total-preserving source contrast; per-source-integral-preserving timing tilt; mass-preserving gas contrast",
            "native_xhi_is_untransformed_context_only": True,
            "astrophysically_self_consistent_source_population": False,
        },
    )
    history["metadata"]["array_sha256"] = {key: array_hash(history[key]) for key in ARRAY_KEYS}
    validate_history(history)
    return history


def budget_es(photons, density, box_len, hydrogen_fraction):
    """Centre-flagging real-space top-hat ES with a local partial-cell branch.

    Hydrogen-only no-recombination prior using actual prescribed cumulative
    photons. Its physics difference from H+He RT is deliberate and recorded.
    """
    n = len(density)
    dx = box_len / n
    d = np.minimum(np.arange(n), n - np.arange(n)) * dx
    distance = np.sqrt(d[:, None, None]**2 + d[None, :, None]**2 + d[None, None, :]**2)
    atoms = density * hydrogen_fraction
    ionized = np.minimum(photons / atoms, 1.)
    radius = np.zeros_like(density)
    flagged = np.zeros_like(density, dtype=bool)
    for r in np.geomspace(box_len / 2, dx, max(2, n)):
        window = (distance <= r).astype(float)
        window /= window.sum()
        kernel = np.fft.rfftn(window)
        smoothed_photons = np.fft.irfftn(np.fft.rfftn(photons) * kernel, s=density.shape, axes=(0, 1, 2)).real
        smoothed_atoms = np.fft.irfftn(np.fft.rfftn(atoms) * kernel, s=density.shape, axes=(0, 1, 2)).real
        crossing = (smoothed_photons >= smoothed_atoms) & ~flagged
        radius[crossing] = r
        ionized[crossing] = 1
        flagged |= crossing
    return 1 - ionized, radius


def design():
    """Predeclare independent parameter designs, causal controls and refinement runs."""
    cases = []
    def add(case_id, theta, split, seed=111, **extra):
        cases.append(dict(case_id=case_id, theta=np.asarray(theta).tolist(), split=split, seed=seed,
                          refinement=3, angular_order=2, time_substeps=1, **extra))
    add("fiducial", FIDUCIAL, "control")
    for j, name in enumerate(PARAMETER_NAMES):
        for side in (0, 1):
            theta = FIDUCIAL.copy()
            theta[j] = BOUNDS[j, side]
            add(f"oat_{j}_{side}", theta, "control", varied_parameter=name)
    for split, count, rng in (("train", 64, 9141), ("validation", 16, 9142), ("test", 16, 9143)):
        points = qmc.scale(qmc.LatinHypercube(6, seed=rng).random(count), BOUNDS[:, 0], BOUNDS[:, 1])
        for i, theta in enumerate(points):
            add(f"{split}_{i:03d}", theta, split)
    points = qmc.scale(qmc.LatinHypercube(6, seed=9144).random(16), BOUNDS[:, 0], BOUNDS[:, 1])
    for i, theta in enumerate(points):
        add(f"external_{i:03d}", theta, "external_test", seed=222 if i < 8 else 333)
    for name, key, value in (("angle4", "angular_order", 4), ("time2", "time_substeps", 2),
                              ("space5", "refinement", 5)):
        add(f"convergence_{name}", FIDUCIAL, "convergence")
        cases[-1][key] = value
    return cases


def prepare(root, upstream):
    """Freeze the design and generate the three native IC/source templates."""
    root.mkdir(parents=True, exist_ok=True)
    cases = design()
    configuration = {"schema_version": 1, "parameter_names": PARAMETER_NAMES,
                         "parameter_bounds": BOUNDS.tolist(), "cases": cases,
                         "metadata": {"dimension": 8, "box_len_cMpc": 50.,
                                   "main_IC_seed": 111, "native_threads": 1,
                                   "posterior_scope": "fixed IC, diagnostic physical-control priors, synthetic likelihood",
                                   "density_evolution": "frozen contrast at z8 with cosmic dilution",
                                   "prior": "periodic hydrogen-only photon-budget ES; NOT native21cmFAST",
                                   "convergence_certified": False}}
    plan = root / "design.json"
    if plan.exists() and json.loads(plan.read_text()) != configuration:
        raise ValueError("Existing design differs; use a new output directory")
    write_json(plan, configuration)
    for seed in (111, 222, 333):
        path = root / f"base_{seed}.npz"
        if not path.exists():
            build_native_history(output=path, dimension=8, highres_dimension=24, box_len=50., threads=1,
                                 seed=seed, z_start=20., z_end=7.9, intervals=8, reference_z=8.)
    write_json(root / "provenance.json", {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in list((ROOT / "research/c2ray_teacher").glob("*.py"))
                          + list((ROOT / "research/c2ray_teacher").glob("*.F90"))
                          + [Path(__file__), ROOT / "research/pc_evolution/native_history.py"]},
        "upstream": str(upstream),
    })


def calibrate_sed(root, base, theta, executable, workers):
    """Measure active/nominal SED photon normalization using a zero-source run.

    The primary experiment holds active transported emission fixed as spectral
    hardness changes. This explicit, input-only table calibration is not a fit
    to ionization labels or to an observed photon-loss residual.
    """
    temperature = float(10**theta[4])
    key = hashlib.sha256(np.float64(temperature).tobytes()).hexdigest()[:16]
    directory = root / "sed_calibration" / key
    with SED_LOCK:
        if (directory / "calibration.json").exists():
            return json.loads((directory / "calibration.json").read_text())["active_over_nominal"]
        directory.mkdir(parents=True, exist_ok=True)
        history = transform_history(base, theta)
        for name in ARRAY_KEYS:
            if name != "capacity":
                history[name] = history[name][:2].copy()
        history["rates"][:] = 0
        history["metadata"].update(intervals=1, redshift_end=float(history["redshifts"][-1]),
                                    normalization_calibration_only=True)
        history["metadata"]["array_sha256"] = {name: array_hash(history[name]) for name in ARRAY_KEYS}
        build_manifest = json.loads((executable.parent / "build_manifest.json").read_text())
        refinement = build_manifest["dimension"] // len(base["capacity"])
        binary = directory / f"input_{time.time_ns()}.bin"
        prepare_history(history, binary, refinement=refinement, blackbody_temperature_K=temperature)
        _, timing = _launch(executable, binary, directory / f"sed_{key}_{time.time_ns()}", workers)
        ratio = timing["sed_transport_photons_over_nominal"]
        if not np.isfinite(ratio) or ratio <= 0:
            raise ValueError("Invalid native active SED normalization")
        write_json(directory / "calibration.json", {
            "blackbody_temperature_K": temperature, "active_over_nominal": ratio,
            "method": "native initialization of a zero-source case; physical rates divided by this factor before transport",
            "timing": timing,
        })
        return ratio


def run_case(root, case, executable, workers):
    """Evolve one real RT history and retain fields, budgets and physical spectra."""
    folder = root / "cases" / case["case_id"]
    if (folder / "COMPLETED").exists():
        if json.loads((folder / "case.json").read_text()) != case:
            raise ValueError("Completed case configuration changed")
        return json.loads((folder / "result.json").read_text())
    if folder.exists():
        failed = root / "failed_attempts"
        failed.mkdir(exist_ok=True)
        folder.rename(failed / f'{case["case_id"]}_{time.time_ns()}')
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "case.json", case)
    history = transform_history(load_history(root / f'base_{case["seed"]}.npz'), case["theta"])
    _write_fresh_history(folder / "history.npz", history)
    sed_factor = calibrate_sed(root, load_history(root / f'base_{case["seed"]}.npz'),
                               case["theta"], executable, workers)
    nominal = copy.deepcopy(history)
    nominal["rates"] /= sed_factor
    nominal["metadata"]["physical_pilot_transform"]["active_over_nominal_sed_factor"] = sed_factor
    nominal["metadata"]["physical_pilot_transform"]["rate_normalization"] = "desired transported photon rate divided by measured active/nominal table factor"
    nominal["metadata"]["array_sha256"] = {name: array_hash(nominal[name]) for name in ARRAY_KEYS}
    _write_fresh_history(folder / "nominal_transport_history.npz", nominal)
    manifest = prepare_history(nominal, folder / "input.bin", refinement=case["refinement"],
                               time_substeps=case["time_substeps"],
                               blackbody_temperature_K=10**case["theta"][4],
                               clumping_factor=10**case["theta"][5])
    manifest.update(transport="conservative_rays", angular_order=case["angular_order"])
    write_json(folder / "input_manifest.json", manifest)
    state, timing = _launch(executable, folder / "input.bin", folder / f'run_{case["case_id"]}_{time.time_ns()}', workers)
    if not np.isclose(timing["sed_transport_photons_over_nominal"], sed_factor, rtol=1e-13, atol=0):
        raise ValueError("Source SED normalization changed between calibration and science")
    validation = validate_output(state, manifest)
    write_json(folder / "native_validation.json", validation)
    write_json(folder / "timing.json", timing)
    selected = []
    for t in history["times_seconds"][1:]:
        matches = [i for i, row in enumerate(manifest["steps"])
                   if np.isclose(row["time_end_seconds"], t, rtol=1e-12, atol=1.)]
        if len(matches) != 1:
            raise ValueError("Cannot pair native output to an exact source endpoint")
        selected.append(matches[0])
    photons, emission_time = integrated_history(history["rates"], history["times_seconds"])
    rho = history["capacity"]
    hydrogen_fraction = 1 - manifest["helium_number_fraction"]
    arrays = {"redshifts": history["redshifts"][1:], "density": rho,
              "photons": photons[1:], "emissivity": history["rates"][1:] * MYR,
              "emission_time_mean": emission_time[1:] / MYR}
    collected = {name: [] for name in ("xhi_rt", "temperature_rt", "xheii_rt", "xheiii_rt",
                                       "xhi_es", "radius_es_cMpc")}
    power, mean_xhi, fine_power = [], [], []
    for j, index in enumerate(selected):
        fine = {key: value[index] for key, value in state["fields"].items()}
        restricted = restrict_fields(fine, len(rho))
        xhi = 1 - restricted["xHII"]
        collected["xhi_rt"].append(xhi)
        collected["temperature_rt"].append(restricted["temperature_K"])
        ratio = case["refinement"]
        for field, output in (("xHeII", "xheii_rt"), ("xHeIII", "xheiii_rt")):
            weighted = (fine[field] * fine["density_total_cm3"]).reshape((8, ratio) * 3).mean(axis=(1, 3, 5))
            collected[output].append(weighted / restricted["density_total_cm3"])
        es, radius = budget_es(photons[j + 1], rho, 50., hydrogen_fraction)
        collected["xhi_es"].append(es)
        collected["radius_es_cMpc"].append(radius)
        amplitude = 27 * np.sqrt((1 + arrays["redshifts"][j]) / 10)
        spectra = shell_spectra({"teacher": amplitude * rho * xhi}, 50.)
        power.append(spectra["P_teacher"])
        mean_xhi.append(float(xhi.mean()))
        fine_rho = fine["density_total_cm3"] / fine["density_total_cm3"].mean() * rho.mean()
        fs = shell_spectra({"teacher": amplitude * fine_rho * (1 - fine["xHII"])}, 50.)
        fine_power.append(fs["P_teacher"])
    arrays.update({key: np.asarray(value) for key, value in collected.items()})
    metadata = case | {"parameter_names": PARAMETER_NAMES, "box_len_cMpc": 50.,
                       "field_prior": "hydrogen-only budget ES, ignores recombinations and helium",
                       "density_units": "mean reference nuclei per cell", "photons_units": "emitted photons per mean reference nucleus",
                       "emissivity_units": "emitted photons per mean reference nucleus per Myr",
                       "emission_time_mean_units": "Myr since z20", "seed_disjoint_primary_split": False,
                       "history_array_sha256": history["metadata"]["array_sha256"],
                       "native_nominal_array_sha256": nominal["metadata"]["array_sha256"],
                       "active_over_nominal_sed_factor": sed_factor,
                       "input_photons_are_active_transport_band": True,
                       "initial_conditions_sha256": history["metadata"]["initial_hires_density_sha256"],
                       "convergence_certified": False}
    np.savez_compressed(folder / "fields.npz", **arrays, metadata=np.asarray(json.dumps(metadata)))
    transported = state.get("transport_diagnostics", {}).get("cumulative", {})
    result = case | {"path": str(folder / "fields.npz"), "P21": power,
                     "k": spectra["k_cMpc_inv"], "redshifts": arrays["redshifts"].tolist(),
                     "xHI_mean": mean_xhi, "P21_fine": fine_power, "k_fine": fs["k_cMpc_inv"],
                     "timing_seconds": timing["seconds"], "transport_ledger": transported,
                     "sed_transport_photons_over_nominal": timing["sed_transport_photons_over_nominal"],
                     "fields_sha256": hashlib.sha256((folder / "fields.npz").read_bytes()).hexdigest()}
    write_json(folder / "result.json", result)
    (folder / "COMPLETED").write_text("passed native state/metadata/transport validations\n")
    print(json.dumps({"completed": case["case_id"], "seconds": timing["seconds"],  # noqa: T201
                      "final_xHI": mean_xhi[-1]}), flush=True)
    return result


def run(root, upstream, group, workers, parallel):
    """Run a selected design group with shared immutable native executables."""
    configuration = json.loads((root / "design.json").read_text())
    cases = configuration["cases"]
    if group != "all":
        groups = group.split(",")
        cases = [c for c in cases if c["split"] in groups or c["case_id"] in groups]
    if not cases:
        raise ValueError("No cases match requested group")
    executables = {}
    for case in cases:
        key = (case["refinement"], case["angular_order"])
        if key in executables:
            continue
        cache = root / f"build_r{key[0]}_a{key[1]}"
        if (cache / "build_manifest.json").exists():
            manifest = json.loads((cache / "build_manifest.json").read_text())
            if not manifest.get("runtime_physics_input_supported", False):
                raise ValueError("Cached binary predates explicit runtime physical controls")
            executable = cache / "c2ray_teacher"
            if not executable.exists():
                candidates = [p for p in cache.iterdir() if p.is_file() and p.name.startswith("c2ray")]
                if len(candidates) != 1:
                    raise ValueError("Cached executable cannot be identified")
                executable = candidates[0]
            if hashlib.sha256(executable.read_bytes()).hexdigest() != manifest["executable_sha256"]:
                raise ValueError("Cached binary hash differs")
        else:
            history = transform_history(load_history(root / "base_111.npz"), FIDUCIAL)
            probe = root / f"build_input_r{key[0]}_a{key[1]}.bin"
            manifest = prepare_history(history, probe, refinement=key[0])
            manifest.update(transport="conservative_rays", angular_order=key[1])
            executable = _build(upstream, cache, manifest)
        executables[key] = executable
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(run_case, root, case, executables[(case["refinement"], case["angular_order"])], workers): case for case in cases}
        errors = []
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:  # noqa: BLE001 - retain every failed native attempt, then raise
                errors.append({"case": futures[future]["case_id"], "error": repr(error)})
                print(json.dumps(errors[-1]), flush=True)  # noqa: T201
        if errors:
            write_json(root / f"errors_{int(time.time())}.json", errors)
            raise RuntimeError(f"{len(errors)} cases failed; retained for inspection")


def collect(root):
    """Collect only completed, explicitly identified native cases for analysis."""
    configuration = json.loads((root / "design.json").read_text())
    results = []
    for case in configuration["cases"]:
        path = root / "cases" / case["case_id"]
        if (path / "COMPLETED").exists():
            if json.loads((path / "case.json").read_text()) != case:
                raise ValueError("Completed native case differs from the frozen design")
            result = json.loads((path / "result.json").read_text())
            if hashlib.sha256((path / "fields.npz").read_bytes()).hexdigest() != result["fields_sha256"]:
                raise ValueError("Completed field artifact hash differs")
            native = list(path.glob("run_*/native_output.bin"))
            if len(native) != 1:
                raise ValueError("Completed case has ambiguous native outputs")
            initialization = json.loads((native[0].parent / "native_diagnostics/native_initialization.json").read_text())
            diagnostics = read_transport_diagnostics(
                native[0].parent / "native_diagnostics/transport_diagnostics.jsonl",
                read_output(native[0]), initialization["sed_table_photons_over_nominal"],
                active_spectral_ratio=initialization["sed_transport_photons_over_nominal"],
            )
            residuals = [row["photon_residual_over_transported_emitted"] for row in diagnostics["steps"]
                         if row["photon_residual_over_transported_emitted"] is not None]
            maximum = max(abs(value) for value in residuals)
            if not np.isfinite(maximum) or maximum > 1e-8:
                raise ValueError(f"Primary transport closure failed for {case['case_id']}: {maximum}")
            result["max_step_primary_transport_fractional_residual"] = maximum
            results.append(result)
    if not results:
        raise ValueError("No completed runs")
    manifest = configuration | {"cases": results}
    write_json(root / "dataset_manifest.json", manifest)
    np.savez_compressed(root / "summary.npz", parameter_names=PARAMETER_NAMES, parameter_bounds=BOUNDS,
                        theta=np.array([r["theta"] for r in results]), split=np.array([r["split"] for r in results]),
                        case_id=np.array([r["case_id"] for r in results]), seed=np.array([r["seed"] for r in results]),
                        k=np.array(results[0]["k"]), redshifts=np.array(results[0]["redshifts"]),
                        P21=np.array([r["P21"] for r in results]), xHI_mean=np.array([r["xHI_mean"] for r in results]),
                        metadata=np.asarray(json.dumps(configuration["metadata"])))
    print(json.dumps({"collected": len(results), "expected": len(configuration["cases"])}), flush=True)  # noqa: T201


def main():
    """Expose preparation, native evolution and collection as separate phases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "collect"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--group", default="all")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--parallel", type=int, default=3)
    args = parser.parse_args()
    root = args.output.resolve()
    if args.phase == "prepare":
        prepare(root, args.upstream)
    elif args.phase == "run":
        run(root, args.upstream, args.group, args.workers, args.parallel)
    else:
        collect(root)


if __name__ == "__main__":
    main()
