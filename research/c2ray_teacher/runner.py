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
    ionization_rate = (
        (1 - he) * fields["xHI_time_average"][index] * fields["Gamma_HI_s"][index]
    )
    ionization_rate += (
        he * fields["xHeI_time_average"][index] * fields["Gamma_HeI_s"][index]
    )
    ionization_rate += (
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
        "radiation_ionizations_time_average_estimate": float(
            np.sum(density * ionization_rate) * cell_volume * meta["dt_seconds"]
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
        "diagnostic_schema_version": 2,
        "state_checks_passed": True,
        "photon_conservation_certified": False,
        "primary_photon_absorption_count_available": False,
        "spatial_convergence_certified": False,
        "temporal_convergence_certified": False,
        "diagnostic_definitions": {
            "radiation_ionizations_time_average_estimate": {
                "quantity": "Radiation-driven primary plus secondary ionization events per transport step",
                "includes_secondary_ionizations": True,
                "is_primary_photon_absorption_count": False,
                "rates_and_fractions_are_iteration_staggered": True,
                "time_sampling": "Post-chemistry time-averaged fractions multiplied by effective rates from the preceding transport pass",
            },
            "Gamma_HI_s": "Effective HI ionization rate per neutral H atom, including photoelectron secondary ionizations, in s^-1",
            "Gamma_HeI_s": "Effective HeI ionization rate per neutral He atom, including photoelectron secondary ionizations, in s^-1",
            "Gamma_HeII_s": "Primary HeII photoionization rate per HeII ion in s^-1; the native secondary routine adds only HI and HeI ionizations",
        },
        "history": history,
        "limitations": [
            "Native effective radiation rates include secondary ionizations; their time-averaged products estimate ionization events, not primary photon absorptions.",
            "Exported average fractions follow the chemistry update while exported rates come from the preceding transport pass; the product is iteration-staggered and is not an exact time integral.",
            "Upstream thermal photonstatistics use global coefficient diagnostics; raw counts do not certify photon closure.",
            "The original 50000 K spectral tables have an approximately 1.5 percent source-normalization mismatch.",
            "Quasistatic transport: finite light-travel time and a carried escaped-photon background are absent.",
            "Gas overdensity is frozen at the reference redshift, with physical cosmological density dilution.",
            "Native ES coevals are contextual controls and do not share this integrated emission history.",
        ],
    }


def _compare_transport_counters(
    reference: dict, candidate: dict, tolerance: float
) -> dict:
    left = reference.get("transport_diagnostics")
    right = candidate.get("transport_diagnostics")
    if left is None and right is None:
        return {"available": False, "reason": "Neither output has independent counters"}
    if left is None or right is None or len(left["steps"]) != len(right["steps"]):
        raise ValueError(
            "Parallel comparison needs matching independent counter outputs"
        )
    budgets = np.asarray([row["nominal_emitted_photons"] for row in left["steps"]])
    scales = np.maximum(budgets, 1e-100)
    fields = {}
    for name in (
        "primary_absorptions_by_species",
        "secondary_ionizations_by_species",
        "primary_absorptions",
        "secondary_ionizations",
        "boundary_outflow_photons",
        "lls_loss_photons",
    ):
        a = np.asarray([row[name] for row in left["steps"]])
        b = np.asarray([row[name] for row in right["steps"]])
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(
                "Parallel counter comparison needs finite equal-shape counts"
            )
        difference = np.abs(a - b)
        scaled = difference / scales.reshape((len(scales),) + (1,) * (a.ndim - 1))
        maximum = float(np.max(scaled))
        fields[name] = {
            "maximum_absolute_difference": float(np.max(difference)),
            "maximum_difference_over_nominal_emitted": maximum,
            "passed": maximum <= tolerance,
        }
    return {
        "available": True,
        "passed": all(row["passed"] for row in fields.values()),
        "relative_to_nominal_emission_tolerance": tolerance,
        "excluded": "Original native boundary counter; independently tallied boundary flux is checked",
        "fields": fields,
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
    transport = _compare_transport_counters(reference, candidate, tolerance)
    return {
        "passed": all(row["passed"] for row in diagnostics.values())
        and transport.get("passed", True),
        "relative_tolerance": tolerance,
        "fields": diagnostics,
        "transport_counters": transport,
    }


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _transport_timing(
    record: dict, native: dict, index: int, spectral_ratio: float
) -> dict:
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        raise ValueError("Transport diagnostic schema is unsupported")
    if type(record.get("step")) is not int or record["step"] != index + 1:
        raise ValueError("Transport diagnostic step index differs from native output")
    for name in ("final_transport_iteration", "recorded_cell_visits"):
        if type(record.get(name)) is not int or record[name] < (
            2 if name.startswith("final") else 0
        ):
            raise ValueError(f"Transport diagnostic has invalid {name}")
    expected = {
        "time_seconds": native["time_seconds"],
        "dt_seconds": native["dt_seconds"],
        "nominal_emitted_photons": native["emitted_photons"],
        "table_emitted_photons": native["emitted_photons"] * spectral_ratio,
    }
    for name, value in expected.items():
        measured = record.get(name)
        if (
            isinstance(measured, bool)
            or not isinstance(measured, (float, int))
            or not np.isfinite(measured)
            or not np.isclose(measured, value, rtol=2e-12, atol=1e-12)
        ):
            raise ValueError(
                f"Transport diagnostic disagrees with native output: {name}"
            )
    return expected


def read_transport_diagnostics(
    path: Path,
    output: dict,
    spectral_ratio: float,
    *,
    active_spectral_ratio: float | None = None,
) -> dict:
    """Read independent final-pass counters without imposing photon closure.

    Parameters
    ----------
    path : Path
        JSON-lines sidecar written by read-only native instrumentation.
    output : dict
        Decoded native binary containing the corresponding step clock/budget.
    spectral_ratio : float
        Measured upstream table photons divided by nominal source photons.
    active_spectral_ratio : float or None
        Measured photons in the native transport loop's frequency band divided
        by nominal source photons. Absent legacy telemetry remains unavailable;
        the full-table normalization is never substituted for this measurement.
    """
    if not np.isfinite(spectral_ratio) or spectral_ratio <= 0:
        raise ValueError("Transport spectral ratio must be finite and positive")
    if active_spectral_ratio is not None and (
        isinstance(active_spectral_ratio, bool)
        or not isinstance(active_spectral_ratio, (int, float))
        or not np.isfinite(active_spectral_ratio)
        or not 0 < active_spectral_ratio <= spectral_ratio
    ):
        raise ValueError(
            "Transport active spectral ratio must be positive and no larger than the full table"
        )
    try:
        records = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Transport diagnostic file is missing or invalid") from exc
    if len(records) != len(output["steps"]):
        raise ValueError("Transport diagnostic step count differs from native output")
    rows = []
    counts = (
        "primary_absorptions",
        "secondary_ionizations",
        "boundary_outflow_photons",
        "native_boundary_outflow_photons",
        "lls_loss_photons",
        "transport_rate_product_ionizations",
        "postchemistry_rate_product_ionizations",
    )
    for index, (record, native) in enumerate(
        zip(records, output["steps"], strict=True)
    ):
        expected = _transport_timing(record, native, index, spectral_ratio)
        scale = max(expected["table_emitted_photons"], 1e-100)
        for name in (
            *counts,
            "transport_fraction_max_abs_change",
            "transport_fraction_max_relative_change",
        ):
            value = record.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
            ):
                raise ValueError(
                    f"Transport diagnostic has nonfinite or missing {name}"
                )
            floor = -1e-12 * scale if name in counts else 0
            if value < floor:
                raise ValueError(f"Transport diagnostic has negative {name}")
        for kind, total in (
            ("primary_absorptions", "primary_absorptions"),
            ("secondary_ionizations", "secondary_ionizations"),
        ):
            values = np.asarray(record.get(f"{kind}_by_species"), dtype=float)
            if (
                values.shape != (3,)
                or not np.isfinite(values).all()
                or np.min(values) < -1e-12 * scale
            ):
                raise ValueError(
                    f"Transport diagnostic has invalid {kind} species counts"
                )
            if abs(float(values.sum()) - record[total]) > 2e-10 * scale:
                raise ValueError(
                    f"Transport diagnostic {kind} species counts disagree with total"
                )
        direct_events = record["primary_absorptions"] + record["secondary_ionizations"]
        if abs(
            record["transport_rate_product_ionizations"] - direct_events
        ) > 2e-9 * max(scale, abs(direct_events)):
            raise ValueError(
                "Transport direct counters disagree with the pre-chemistry rate product"
            )
        for sidecar, key in (
            ("native_boundary_outflow_photons", "native_photon_loss"),
            ("lls_loss_photons", "native_LLS_loss"),
        ):
            if key in native and abs(record[sidecar] - native[key]) > 2e-10 * scale:
                raise ValueError(f"Transport {sidecar} differs from native diagnostic")
        photon_accounted = (
            record["primary_absorptions"]
            + record["boundary_outflow_photons"]
            + record["lls_loss_photons"]
        )
        row = dict(record)
        transported_emitted = (
            expected["nominal_emitted_photons"] * active_spectral_ratio
            if active_spectral_ratio is not None
            else None
        )
        row.update(
            photon_accounted=photon_accounted,
            transported_emitted_photons=transported_emitted,
            photon_residual_over_transported_emitted=(
                photon_accounted / transported_emitted - 1
                if transported_emitted
                else None
            ),
            photon_residual_over_nominal_emitted=(
                photon_accounted / expected["nominal_emitted_photons"] - 1
            )
            if expected["nominal_emitted_photons"]
            else None,
            photon_residual_over_table_emitted=(
                photon_accounted / expected["table_emitted_photons"] - 1
            )
            if expected["table_emitted_photons"]
            else None,
            postchemistry_rate_product_difference=record[
                "postchemistry_rate_product_ionizations"
            ]
            - direct_events,
            boundary_minus_native_outflow_photons=record["boundary_outflow_photons"]
            - record["native_boundary_outflow_photons"],
        )
        rows.append(row)
    cumulative = {
        name: float(sum(row[name] for row in rows))
        for name in (
            *counts,
            "nominal_emitted_photons",
            "table_emitted_photons",
            "photon_accounted",
        )
    }
    for budget in ("nominal", "table"):
        total = cumulative[f"{budget}_emitted_photons"]
        cumulative[f"photon_residual_over_{budget}_emitted"] = (
            cumulative["photon_accounted"] / total - 1 if total else None
        )
        cumulative[f"boundary_outflow_over_{budget}_emitted"] = (
            cumulative["boundary_outflow_photons"] / total if total else None
        )
    transported_total = (
        sum(row["transported_emitted_photons"] for row in rows)
        if active_spectral_ratio is not None
        else None
    )
    cumulative["transported_emitted_photons"] = transported_total
    cumulative["photon_residual_over_transported_emitted"] = (
        cumulative["photon_accounted"] / transported_total - 1
        if transported_total
        else None
    )
    return {
        "schema_version": 1,
        "counter_integrity_passed": True,
        "primary_photon_absorption_count_available": True,
        "photon_conservation_certified": False,
        "scope": "Direct primary absorption and secondary-event tallies from the final converged transport pass; secondary events are excluded from photon closure",
        "normalization": "Nominal and full spectral-table emission are reported; measured transport-band emission is separate when available. No source renormalization or residual correction is applied",
        "steps": rows,
        "cumulative": cumulative,
    }


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
    state = read_output(output)
    build_manifest = json.loads((build_dir / "build_manifest.json").read_text())
    diagnostic_path = destination / "native_diagnostics" / "transport_diagnostics.jsonl"
    if build_manifest.get("diagnostics_enabled", False) or diagnostic_path.exists():
        state["transport_diagnostics"] = read_transport_diagnostics(
            diagnostic_path,
            state,
            initialization["sed_table_photons_over_nominal"],
            active_spectral_ratio=initialization["sed_transport_photons_over_nominal"],
        )
    return state, {
        "workers": workers,
        "mpi_ranks": ranks,
        "omp_threads": threads,
        "seconds": elapsed,
        "command": command,
        "sed_table_photons_over_nominal": initialization[
            "sed_table_photons_over_nominal"
        ],
        "sed_transport_photons_over_nominal": initialization[
            "sed_transport_photons_over_nominal"
        ],
        "sed_omitted_tail_photons_over_nominal": initialization[
            "sed_omitted_tail_photons_over_nominal"
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
    record = records[0]
    active = record.get("sed_transport_photons_over_nominal")
    tail = record.get("sed_omitted_tail_photons_over_nominal")
    if (active is not None or tail is not None) and (
        any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            for value in (active, tail)
        )
        or not 0 < active <= ratio
        or tail < 0
        or not np.isclose(active + tail, ratio, rtol=2e-12, atol=1e-15)
    ):
        raise RuntimeError(
            "Native spectral band normalizations are invalid or inconsistent"
        )
    record["sed_transport_photons_over_nominal"] = active
    record["sed_omitted_tail_photons_over_nominal"] = tail
    return record


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
                "transport_diagnostics": state.get("transport_diagnostics"),
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
    manifest["native_history_provenance"] = {
        key: history["metadata"].get(key)
        for key in (
            "initial_hires_density_sha256",
            "random_seed",
            "dimension",
            "highres_dimension",
            "box_len_cMpc",
            "reference_redshift",
        )
    }
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
    transport = state.get("transport_diagnostics")
    if transport is not None:
        validation["primary_photon_absorption_count_available"] = True
        validation["transport_diagnostics"] = transport
    np.savez_compressed(
        output / "states.npz",
        **state["fields"],
        metadata=np.asarray(
            json.dumps(
                {
                    "steps": state["steps"],
                    "input_manifest": manifest,
                    "diagnostic_schema_version": validation[
                        "diagnostic_schema_version"
                    ],
                    "diagnostic_definitions": validation["diagnostic_definitions"],
                    "transport_diagnostics": transport,
                }
            )
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
