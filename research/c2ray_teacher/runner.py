"""Run the unchanged C2RayHHe backend on physical native 21cmFAST histories."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from importlib import import_module
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_backend = import_module("research.c2ray_teacher.build")
FIELD_NAMES = _backend.FIELD_NAMES
build_backend, read_output, sha256 = (
    _backend.build_backend,
    _backend.read_output,
    _backend.sha256,
)
prepare_history = import_module("research.c2ray_teacher.data").prepare_history
_native = import_module("research.pc_evolution.native_history")
ARRAY_KEYS, array_hash = _native.ARRAY_KEYS, _native.array_hash
build_native_history, load_history = _native.build_native_history, _native.load_history
native_time_interval = _native.native_time_interval

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG = {
    "dimension": 16,
    "highres_dimension": 48,
    "box_len": 50.0,
    "seed": 1234,
    "z_start": 20.0,
    "z_end": 5.0,
    "intervals": 30,
    "reference_z": 8.0,
    "refinement": 1,
    "stride": 1,
    "time_substeps": 1,
    "initial_temperature": 100.0,
    "initial_xhii": 1e-6,
    "initial_xheii": 1e-6,
    "initial_xheiii": 1e-12,
    "thermal": True,
    "benchmark": True,
}


def parallel_layout(workers: int) -> tuple[int, int]:
    """Return MPI ranks and OpenMP threads using every requested CPU.

    Parameters
    ----------
    workers : int
        CPU allocation between one and 32. Each MPI rank uses at most eight
        threads, matching the native source-octant OpenMP decomposition.
    """
    if type(workers) is not int or not 1 <= workers <= 32:
        raise ValueError("workers must be an integer between 1 and 32")
    threads = max(n for n in range(1, min(workers, 8) + 1) if workers % n == 0)
    return workers // threads, threads


def _validate_numbers(config: dict) -> None:
    integers = (
        "dimension",
        "highres_dimension",
        "seed",
        "intervals",
        "refinement",
        "stride",
        "time_substeps",
    )
    for name in integers:
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    names = (
        "box_len",
        "z_start",
        "z_end",
        "reference_z",
        "initial_temperature",
        "initial_xhii",
        "initial_xheii",
        "initial_xheiii",
    )
    for name in names:
        value = config[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
        ):
            raise ValueError(f"{name} must be a finite number")
    for name in ("thermal", "benchmark"):
        if type(config[name]) is not bool:
            raise ValueError(f"{name} must be boolean")
    if not config["thermal"]:
        raise ValueError(
            "thermal=True is required: unchanged upstream isothermal mode is unsupported"
        )


def validate_config(updates: dict) -> dict:
    """Validate configuration and return explicit defaults plus supplied values.

    Parameters
    ----------
    updates : dict
        Run settings, optionally including a path to an existing ``history``.
    """
    if not isinstance(updates, dict):
        raise TypeError("Configuration must be an object")
    unknown = set(updates) - set(DEFAULT_CONFIG) - {"history"}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    config = DEFAULT_CONFIG | updates
    _validate_numbers(config)
    if config["dimension"] < 16 or config["highres_dimension"] < config["dimension"]:
        raise ValueError("dimension >= 16 and highres_dimension >= dimension required")
    if config["refinement"] % 2 == 0:
        raise ValueError("refinement must be odd to preserve original source centres")
    if not config["z_start"] > config["z_end"] >= 0 or config["reference_z"] < 0:
        raise ValueError("Redshifts require z_start > z_end >= 0 and reference_z >= 0")
    if min(config["box_len"], config["initial_temperature"]) <= 0:
        raise ValueError("Box length and initial temperature must be positive")
    fractions = [config[k] for k in ("initial_xhii", "initial_xheii", "initial_xheiii")]
    if min(fractions) <= 0 or max(fractions) >= 1 or sum(fractions[1:]) >= 1:
        raise ValueError(
            "Initial ionization fractions require strictly positive populations of all species"
        )
    if "history" in config and not isinstance(config["history"], (str, Path)):
        raise ValueError("history must be an existing native-history path")
    return config


def _check_fields(output: dict, manifest: dict) -> None:
    shape = (manifest["step_count"],) + (manifest["dimension"],) * 3
    if output["dimension"] != manifest["dimension"] or len(output["steps"]) != shape[0]:
        raise ValueError("Native output dimension or step count differs from input")
    for name in FIELD_NAMES:
        if name not in output["fields"]:
            raise ValueError(f"Missing native field: {name}")
        values = output["fields"][name]
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(
                f"Native field has invalid shape or nonfinite values: {name}"
            )
        if name.startswith("x") and (values.min() < -1e-10 or values.max() > 1 + 1e-10):
            raise ValueError(f"Native fraction outside [0,1]: {name}")
        if name.startswith("Gamma_") and values.min() < 0:
            raise ValueError(f"Native photoionization rate is negative: {name}")
    fields = output["fields"]
    if min(fields["density_total_cm3"].min(), fields["temperature_K"].min()) <= 0:
        raise ValueError("Native density and temperature must be positive")
    if (fields["xHeII"] + fields["xHeIII"]).max() > 1 + 1e-10:
        raise ValueError("Native helium ionization fractions sum to more than one")


def _check_metadata(meta: dict, expected: dict, elapsed: float) -> None:
    if not np.isfinite(list(meta.values())).all():
        raise ValueError("Native step metadata is nonfinite")
    pairs = [
        ("dt_seconds", expected["dt_seconds"]),
        ("proper_box_cm", expected["proper_box_cm"]),
        ("redshift_mid", expected["redshift_density_evaluation"]),
        ("emitted_photons", expected["emitted_photons"]),
        ("time_seconds", elapsed),
    ]
    for name, value in pairs:
        if not np.isclose(meta[name], value, rtol=2e-12, atol=1e-12):
            raise ValueError(f"Native step metadata differs from input: {name}")


def _state_diagnostic(fields: dict, index: int, meta: dict, manifest: dict) -> dict:
    density = fields["density_total_cm3"][index]
    cell_volume = (meta["proper_box_cm"] / manifest["dimension"]) ** 3
    density_sum = float(density.sum(dtype=np.float64))
    nuclei = density_sum * cell_volume
    if not np.isclose(nuclei, manifest["initial_total_nuclei"], rtol=2e-12, atol=0):
        raise ValueError("Native total nuclei changed under cosmological dilution")
    he = manifest["helium_number_fraction"]
    absorption = (
        (1 - he) * fields["xHI_time_average"][index] * fields["Gamma_HI_s"][index]
    )
    absorption += he * fields["xHeI_time_average"][index] * fields["Gamma_HeI_s"][index]
    absorption += (
        he * fields["xHeII_time_average"][index] * fields["Gamma_HeII_s"][index]
    )
    neutral = 1 - fields["xHII"][index]
    return {
        "step": index,
        "redshift_end": manifest["steps"][index]["redshift_end"],
        "xHI_mass": float(np.sum(density * neutral) / density_sum),
        "xHI_volume": float(neutral.mean()),
        "temperature_volume_mean_K": float(fields["temperature_K"][index].mean()),
        "total_nuclei": nuclei,
        "photoabsorptions_time_average_estimate": float(
            np.sum(density * absorption) * cell_volume * meta["dt_seconds"]
        ),
        "native_diagnostics": {
            k: v for k, v in meta.items() if k.startswith("native_")
        },
    }


def validate_output(output: dict, manifest: dict) -> dict:
    """Check physical states and input/output units without claiming RT convergence.

    Parameters
    ----------
    output : dict
        Decoded binary output from ``build.read_output``.
    manifest : dict
        Physical input manifest returned by ``data.prepare_history``.
    """
    _check_fields(output, manifest)
    history = []
    elapsed = 0.0
    for index, (meta, expected) in enumerate(
        zip(output["steps"], manifest["steps"], strict=True)
    ):
        elapsed += expected["dt_seconds"]
        _check_metadata(meta, expected, elapsed)
        history.append(_state_diagnostic(output["fields"], index, meta, manifest))
    return {
        "state_checks_passed": True,
        "photon_conservation_certified": False,
        "spatial_convergence_certified": False,
        "temporal_convergence_certified": False,
        "history": history,
        "limitations": [
            "Native time-averaged rate products are absorption estimates, not an independent integrated ledger.",
            "Upstream thermal photonstatistics use global coefficient diagnostics; raw counts do not certify photon closure.",
            "The original 50000 K spectral tables have an approximately 1.5 percent source-normalization mismatch.",
            "Quasistatic transport: finite light-travel time and a carried escaped-photon background are absent.",
            "Gas overdensity is frozen at the reference redshift, with physical cosmological density dilution.",
            "Native ES coevals are contextual controls and do not share this integrated emission history.",
        ],
    }


def compare_parallel_outputs(
    reference: dict, candidate: dict, tolerance: float = 1e-8
) -> dict:
    """Compare every native field, detecting morphology as well as mean changes.

    Parameters
    ----------
    reference, candidate : dict
        Decoded outputs of the same input with different CPU layouts.
    tolerance : float
        Relative tolerance scaled by the reference field maximum. Ionization
        fractions additionally allow absolute roundoff up to ``1e-9``; density,
        temperature, rates and heating use only the relative tolerance.
    """
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Comparison tolerance must be finite and positive")
    diagnostics = {}
    for name in FIELD_NAMES:
        left, right = reference["fields"][name], candidate["fields"][name]
        if left.shape != right.shape or not np.isfinite(right).all():
            raise ValueError("Parallel comparison needs finite equal-shape fields")
        difference = float(np.max(np.abs(left - right)))
        scale = max(float(np.max(np.abs(left))), 1e-100)
        absolute_tolerance = 1e-9 if name.startswith("x") else 0.0
        diagnostics[name] = {
            "maximum_absolute_difference": difference,
            "maximum_normalized_difference": difference / scale,
            "absolute_tolerance": absolute_tolerance,
            "passed": difference <= absolute_tolerance + tolerance * scale,
        }
    return {
        "passed": all(row["passed"] for row in diagnostics.values()),
        "relative_tolerance": tolerance,
        "fields": diagnostics,
    }


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _build(upstream: Path, destination: Path, manifest: dict) -> Path:
    return build_backend(
        upstream,
        destination,
        dimension=manifest["dimension"],
        helium_number_fraction=manifest["helium_number_fraction"],
        omega_m=manifest["omega_m"],
        omega_b=manifest["omega_b"],
        h=manifest["h"],
        mpi=True,
        compiler=os.environ.get("MPIFC"),
    )


def _launch(
    executable: Path, binary: Path, destination: Path, workers: int
) -> tuple[dict, dict]:
    ranks, threads = parallel_layout(workers)
    destination.mkdir(parents=True, exist_ok=False)
    build_dir = executable.parent
    runtime = build_dir / f"case_{destination.name}"
    shutil.copytree(build_dir / "runtime", runtime)
    if any((runtime / "results").iterdir()):
        raise ValueError("Native case needs an empty results directory")
    launcher = shutil.which(os.environ.get("C2_MPIRUN", "mpirun"))
    if not launcher:
        raise RuntimeError("MPI launcher not found; load OpenMPI or set C2_MPIRUN")
    environment = os.environ.copy()
    environment.update(
        OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    if os.getuid() == 0:
        environment.update(
            OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1"
        )
    output = destination / "native_output.bin"
    command = [
        launcher,
        "-np",
        str(ranks),
        "--bind-to",
        "core",
        "--map-by",
        f"slot:PE={threads}",
        str(executable),
        str(binary),
        str(output),
    ]
    LOGGER.info(
        "native_start %s CPUs=%s MPI=%s OMP=%s",
        destination.name,
        workers,
        ranks,
        threads,
    )
    started = time.perf_counter()
    with (destination / "native.log").open("w") as stream:
        subprocess.run(
            command,
            cwd=runtime,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    elapsed = time.perf_counter() - started
    shutil.copytree(runtime / "results", destination / "native_diagnostics")
    _check_native_logs(destination)
    initialization = _native_initialization(destination / "native.log")
    LOGGER.info("native_complete %s seconds=%.3f", destination.name, elapsed)
    return read_output(output), {
        "workers": workers,
        "mpi_ranks": ranks,
        "omp_threads": threads,
        "seconds": elapsed,
        "command": command,
        "sed_table_photons_over_nominal": initialization[
            "sed_table_photons_over_nominal"
        ],
        "timing_scope": "MPI startup, table loading and native evolution; excludes compilation and input preparation",
    }


def _native_initialization(path: Path) -> dict:
    records = []
    for line in path.read_text(errors="replace").splitlines():
        if "native_initialized" not in line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Invalid native initialization JSON record") from exc
        if record.get("stage") == "native_initialized":
            records.append(record)
    if len(records) != 1:
        raise RuntimeError("Expected exactly one native initialization record")
    ratio = records[0].get("sed_table_photons_over_nominal")
    if (
        isinstance(ratio, bool)
        or not isinstance(ratio, (int, float))
        or not np.isfinite(ratio)
        or ratio <= 0
    ):
        raise RuntimeError(
            "Native spectral table normalization must be a positive finite number"
        )
    return records[0]


def _check_native_logs(destination: Path) -> None:
    paths = [
        destination / "native.log",
        *(destination / "native_diagnostics").rglob("*"),
    ]
    markers = (
        "Multiple sources not converging",
        "ERROR STOP",
        "SIGSEGV",
        "Fortran runtime error",
    )
    for path in paths:
        if not path.is_file():
            continue
        content = path.read_text(errors="replace")
        if any(marker in content for marker in markers):
            raise RuntimeError(f"Native solver reported failure in {path}")


def _benchmark_history(science: dict) -> dict:
    metadata = copy.deepcopy(science["metadata"])
    dt, _ = native_time_interval(10.001, 10.0, metadata["cosmology_for_clocks"])
    rates = np.zeros((2, 16, 16, 16), dtype=np.float64)
    for i in (4, 11):
        for j in (4, 11):
            for k in (4, 11):
                rates[:, i, j, k] = 1e-12
    history = {
        "redshifts": np.array([10.001, 10.0]),
        "times_seconds": np.array([0.0, dt]),
        "rates": rates,
        "capacity": np.ones((16, 16, 16)),
        "native_xhi": np.ones(rates.shape, dtype=np.float32),
        "metadata": metadata,
    }
    metadata.update(
        dimension=16,
        highres_dimension=48,
        box_len_cMpc=0.25,
        reference_redshift=10.0,
        scope="Controlled uniform 16^3, eight-source MPI benchmark",
    )
    metadata["array_sha256"] = {name: array_hash(history[name]) for name in ARRAY_KEYS}
    return history


def _benchmarks(science: dict, output: Path, upstream: Path, workers: int) -> dict:
    binary = output / "benchmark_input.bin"
    manifest = prepare_history(
        _benchmark_history(science), binary, thermal=True, initial_temperature=1e4
    )
    _write_json(output / "benchmark_input_manifest.json", manifest)
    executable = _build(upstream, output / "build_benchmark", manifest)
    results = {
        "scope": "Identical static-density, eight-source input at each CPU count",
        "repetitions": 1,
        "runs": [],
        "passed": True,
    }
    reference = None
    for count in sorted({1, min(8, workers), workers}):
        state, timing = _launch(
            executable, binary, output / f"benchmark_{count:02d}", count
        )
        checks = validate_output(state, manifest)
        reference = state if reference is None else reference
        comparison = compare_parallel_outputs(reference, state)
        results["runs"].append(
            timing
            | {
                "state_checks_passed": checks["state_checks_passed"],
                "comparison_to_one_cpu": comparison,
            }
        )
        results["passed"] &= comparison["passed"]
        _write_json(output / "benchmark.json", results)
        if not comparison["passed"]:
            raise RuntimeError(
                "MPI/OpenMP native fields differ beyond relative tolerance 1e-8 "
                "plus absolute ionization-fraction tolerance 1e-9"
            )
    return results


def _history(config: dict, output: Path, workers: int) -> dict:
    if "history" in config:
        return load_history(Path(config["history"]).expanduser().resolve())
    keys = (
        "dimension",
        "highres_dimension",
        "box_len",
        "seed",
        "z_start",
        "z_end",
        "intervals",
        "reference_z",
    )
    LOGGER.info(
        "native_history_start dimension=%s intervals=%s",
        config["dimension"],
        config["intervals"],
    )
    return build_native_history(
        output=output / "native_history.npz",
        threads=workers,
        **{key: config[key] for key in keys},
    )


def run(config: dict, output: Path, upstream: Path, workers: int = 32) -> dict:
    """Prepare inputs, verify CPU equivalence and evolve a fresh native RT run.

    Parameters
    ----------
    config : dict
        Input geometry, thermal settings and optional native history path.
    output : Path
        Fresh output directory; existing directories are never reused.
    upstream : Path
        Pinned unchanged C2RayHHe checkout verified by the build adapter.
    workers : int
        Number of allocated CPUs, default 32.
    """
    started = time.perf_counter()
    config = validate_config(config)
    parallel_layout(workers)
    output, upstream = Path(output).resolve(), Path(upstream).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if "history" in config:
        config["history"] = str(Path(config["history"]).expanduser().resolve())
    _write_json(output / "config.json", config | {"workers": workers})
    history = _history(config, output, workers)
    binary = output / "input.bin"
    keys = (
        "refinement",
        "stride",
        "time_substeps",
        "initial_temperature",
        "initial_xhii",
        "initial_xheii",
        "initial_xheiii",
        "thermal",
    )
    manifest = prepare_history(history, binary, **{key: config[key] for key in keys})
    _write_json(output / "input_manifest.json", manifest)
    benchmarks = (
        _benchmarks(history, output, upstream, workers)
        if config["benchmark"]
        else {"skipped": True}
    )
    LOGGER.info(
        "science_build dimension=%s steps=%s sources=%s",
        manifest["dimension"],
        manifest["step_count"],
        manifest["source_count"],
    )
    executable = _build(upstream, output / "build_science", manifest)
    state, timing = _launch(executable, binary, output / "science", workers)
    validation = validate_output(state, manifest)
    np.savez_compressed(
        output / "states.npz",
        **state["fields"],
        metadata=np.asarray(
            json.dumps({"steps": state["steps"], "input_manifest": manifest})
        ),
    )
    summary = {
        "config": config,
        "timing": timing,
        "benchmarks": benchmarks,
        "validation": validation,
        "input_manifest": manifest,
        "native_output_sha256": sha256(output / "science" / "native_output.bin"),
        "total_seconds": time.perf_counter() - started,
    }
    _write_json(output / "summary.json", summary)
    (output / "COMPLETED").write_text(
        "Native pilot and requested checks completed; convergence is not certified.\n"
    )
    LOGGER.info("complete %s total_seconds=%.3f", output, summary["total_seconds"])
    return summary


def main() -> None:
    """Parse the Gadi batch-job command line and execute the native pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout
    )
    run(json.loads(args.config.read_text()), args.output, args.upstream, args.workers)


if __name__ == "__main__":
    main()
