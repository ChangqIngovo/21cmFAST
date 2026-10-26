"""Measure spatial dependence of a fixed opaque two-source C2-Ray problem.

This controlled static-gas test uses physical source positions unchanged under
odd mesh refinement. It reports photon counters without certifying convergence
or imposing global closure. It does not generate a cosmological source field.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
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
build_backend, read_output = _backend.build_backend, _backend.read_output

BASE_DIMENSION = 16
BASE_POSITIONS = np.array([[5, 8, 8], [10, 8, 8]], dtype=np.int32)
CONFIGURATION = {
    "proper_box_cm": 5e22,
    "density_total_H_plus_He_nuclei_cm3": 1e-3,
    "helium_number_fraction": 0.074,
    "initial_temperature_K": 100.0,
    "initial_xHII": 1e-6,
    "initial_xHeII": 1e-6,
    "initial_xHeIII": 1e-6,
    "redshift_fixed": 9.0,
    "dt_seconds": 3e11,
    "source_rates_photons_s": [5e48, 5e48],
    "second_interval_source_rates_photons_s": [0.0, 0.0],
    "blackbody_temperature_K": 50000,
    "omega_m": 0.3,
    "omega_b": 0.048,
    "h": 0.7,
    "random_seed": None,
    "source_positions": "Fixed deterministic point sources, no random sampling",
    "gas_evolution": "Uniform proper density and proper box held fixed",
}


def validate_refinements(refinements: list[int]) -> list[int]:
    """Validate unique odd factors that preserve the original source centres."""
    if not refinements or any(
        type(q) is not int or q < 1 or q % 2 == 0 for q in refinements
    ):
        raise ValueError("refinements must be nonempty odd positive integers")
    if len(set(refinements)) != len(refinements):
        raise ValueError("refinements must be unique")
    if BASE_DIMENSION * max(refinements) > np.iinfo(np.int32).max:
        raise ValueError("Refined dimension exceeds the native int32 format")
    return sorted(refinements)


def source_positions(refinement: int) -> np.ndarray:
    """Map coarse one-based integer centres onto an odd refined grid."""
    validate_refinements([refinement])
    return BASE_POSITIONS.astype(np.int64) * refinement - (refinement - 1) // 2


def _write_input(path: Path, refinement: int) -> dict:
    n = BASE_DIMENSION * refinement
    positions = source_positions(refinement)
    fractional = (positions - 0.5) / n
    if not np.array_equal(fractional, (BASE_POSITIONS - 0.5) / BASE_DIMENSION):
        raise RuntimeError("Refinement moved physical source centres")
    c = CONFIGURATION
    density = np.full(n**3, c["density_total_H_plus_He_nuclei_cm3"], dtype="<f8")
    with path.open("xb") as stream:
        stream.write(b"C2RTHHE1")
        stream.write(struct.pack("<4i", n, 2, len(positions), 1))
        np.array(
            [
                c["redshift_fixed"],
                c["proper_box_cm"],
                c["initial_temperature_K"],
                c["initial_xHII"],
                c["initial_xHeII"],
                c["initial_xHeIII"],
            ],
            dtype="<f8",
        ).tofile(stream)
        positions.astype("<i4").tofile(stream)
        density.tofile(stream)
        for rates in (
            c["source_rates_photons_s"],
            c["second_interval_source_rates_photons_s"],
        ):
            np.array(
                [c["dt_seconds"], c["redshift_fixed"], c["proper_box_cm"]], dtype="<f8"
            ).tofile(stream)
            density.tofile(stream)
            np.array(rates, dtype="<f8").tofile(stream)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "refinement": refinement,
        "dimension": n,
        "cell_width_proper_cm": c["proper_box_cm"] / n,
        "source_positions_1based": positions.tolist(),
        "source_positions_fractional": fractional.tolist(),
        "source_positions_proper_cm": (fractional * c["proper_box_cm"]).tolist(),
        "input_sha256": digest,
        "configuration": c,
    }


def _report(case: Path, fixture: dict, seconds: float) -> dict:
    state = read_output(case / "output.bin")
    rows = [
        json.loads(line)
        for line in (case / "build/runtime/results/transport_diagnostics.jsonl")
        .read_text()
        .splitlines()
    ]
    if len(rows) != 2 or [row["step"] for row in rows] != [1, 2]:
        raise ValueError("Expected both complete emission and dark intervals")
    data = rows[0]
    emitted = data["table_emitted_photons"]
    primary, secondary = data["primary_absorptions"], data["secondary_ionizations"]
    if emitted <= 0 or not np.isfinite(emitted):
        raise ValueError("Invalid table-emitted budget")
    return {
        "refinement": fixture["refinement"],
        "dimension": fixture["dimension"],
        "cell_width_proper_cm": fixture["cell_width_proper_cm"],
        "native_seconds": seconds,
        "timing_scope": "Table initialization and two RT intervals; excludes build and input generation",
        "final_transport_iteration": data["final_transport_iteration"],
        "recorded_cell_visits": data["recorded_cell_visits"],
        "primary_over_table": primary / emitted,
        "secondary_over_table": secondary / emitted,
        "independent_boundary_over_table": data["boundary_outflow_photons"] / emitted,
        "native_boundary_over_table": data["native_boundary_outflow_photons"] / emitted,
        "primary_plus_independent_boundary_residual": (
            primary + data["boundary_outflow_photons"] + data["lls_loss_photons"]
        )
        / emitted
        - 1,
        "transport_identity_relative_error": data["transport_rate_product_ionizations"]
        / (primary + secondary)
        - 1,
        "mean_xHII": float(state["fields"]["xHII"][0].mean()),
        "mean_temperature_K": float(state["fields"]["temperature_K"][0].mean()),
        "final_dark_interval_mean_xHII": float(state["fields"]["xHII"][1].mean()),
        "field_scaling": "Dimensionless H/He species fractions; temperature in K; uniform-density volume and mass means coincide",
    }


def run_probe(
    upstream: Path,
    output: Path,
    refinements: list[int],
    threads: int = 4,
    compiler: str | None = None,
    transport: str = "native",
    angular_order: int = 16,
) -> dict:
    """Build and run a fixed physical problem at several mesh resolutions.

    Parameters
    ----------
    upstream : Path
        Checkout or archive of the pinned C2-Ray upstream revision.
    output : Path
        Fresh directory for complete inputs, builds, logs and native outputs.
    refinements : list of int
        Unique odd multiples of the original 16 cells per axis.
    threads : int
        OpenMP threads for each native executable, without MPI processes.
    compiler : str, optional
        GNU Fortran compiler path; otherwise resolved from the environment.
    transport : str
        Native short characteristics or opt-in conservative rays.
    angular_order : int
        Angular pixels per side of each cube face in the ray mode.
    """
    refinements = validate_refinements(refinements)
    if type(threads) is not int or not 1 <= threads <= 32:
        raise ValueError("threads must be an integer from 1 through 32")
    upstream, output = Path(upstream).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "scope": "Controlled opaque two-source spatial probe, not a cosmological simulation",
        "convergence_certified": False,
        "configuration": CONFIGURATION,
        "threads": threads,
        "refinements": refinements,
        "transport": transport,
        "angular_order": angular_order if transport == "conservative_rays" else None,
        "cases": [],
    }
    for q in refinements:
        # Three input density grids and 13 output fields for each of two steps.
        # Include build files and a margin before allocating a high-resolution case.
        required = (29 * 8 * (BASE_DIMENSION * q) ** 3 * 11) // 10 + 16 * 1024**2
        available = shutil.disk_usage(output).free
        if available < required:
            raise OSError(
                f"Spatial probe q={q} needs at least {required} free bytes; "
                f"only {available} available at {output}"
            )
        case = output / f"q{q}"
        case.mkdir()
        fixture = _write_input(case / "input.bin", q)
        (case / "fixture.json").write_text(json.dumps(fixture, indent=2) + "\n")
        executable = build_backend(
            upstream,
            case / "build",
            dimension=BASE_DIMENSION * q,
            helium_number_fraction=CONFIGURATION["helium_number_fraction"],
            omega_m=CONFIGURATION["omega_m"],
            omega_b=CONFIGURATION["omega_b"],
            h=CONFIGURATION["h"],
            mpi=False,
            compiler=compiler,
            diagnostics=True,
            transport=transport,
            angular_order=angular_order,
        )
        started = time.perf_counter()
        with (case / "native.log").open("w") as stream:
            subprocess.run(
                [str(executable), str(case / "input.bin"), str(case / "output.bin")],
                cwd=case / "build/runtime",
                env=os.environ | {"OMP_NUM_THREADS": str(threads)},
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        result = _report(case, fixture, time.perf_counter() - started)
        report["cases"].append(result)
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        sys.stdout.write(json.dumps(result) + "\n")
        sys.stdout.flush()
    (output / "COMPLETED").write_text(
        "Requested spatial probes completed; convergence is not certified.\n"
    )
    return report


def main() -> None:
    """Run the controlled spatial probe from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refinements", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--compiler")
    parser.add_argument(
        "--transport", choices=("native", "conservative_rays"), default="native"
    )
    parser.add_argument("--angular-order", type=int, default=16)
    run_probe(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
