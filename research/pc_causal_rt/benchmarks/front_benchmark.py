"""Measure finite-volume convergence against an exact plane ionization front.

Notes
-----
The benchmark uses a source-free travelling solution of the same finite-light-
speed radiation and absorption equations as the production kernel. Initial
states and references are analytic cell averages. Errors are measured only in
a causal window that the periodic trailing discontinuity cannot reach.

The inverse front width is varied to separate smooth-field convergence from
unresolved, optically thick fronts. This diagnostic does not alter production
physical parameters or certify a cosmological power spectrum.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)


def _digest(path: Path) -> str:
    """Return the byte-level SHA256 digest of a recorded input."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _logistic_averages(n: int, aL: float, location: float) -> np.ndarray:
    """Integrate a logistic front over each finite-volume cell."""
    edges = np.linspace(0.0, 1.0, n + 1)
    averages = (
        np.logaddexp(0.0, -aL * (edges[:-1] - location))
        - np.logaddexp(0.0, -aL * (edges[1:] - location))
    ) / (aL / n)
    return np.broadcast_to(averages[:, None, None], (n, n, n)).copy()


def benchmark(repo: Path, module_path: Path) -> dict[str, object]:
    """Compare a selected transport implementation with analytic front averages.

    Parameters
    ----------
    repo : pathlib.Path
        Repository supplying the recorded angular quadrature and physical
        constants in ``research.pc_matched.rt``.
    module_path : pathlib.Path
        Transport Python module exposing ``evolve_interval``. Its companion
        ``transport_kernel.cpp`` must be in the same directory.

    Returns
    -------
    dict of str to object
        JSON-compatible parameters, implementation hashes, analytic reference
        definitions and errors for ten spatial, temporal and opacity settings.

    Notes
    -----
    The reference is exact in the recorded front window. The global translated
    smooth-wave ion integral is diagnostic only because it does not include
    the periodic trailing discontinuity outside that window.
    """
    repo, module_path = repo.resolve(), module_path.resolve()
    before = (
        _digest(module_path),
        _digest(module_path.with_name("transport_kernel.cpp")),
    )
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location("front_transport", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load transport module: {module_path}")
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    length, z, density, p0, ct_fraction = 50.0, 7.0, 1.9e-7, 1.0, 0.25
    duration = (
        ct_fraction * length * transport.MPC_CM / (transport.LIGHT_SPEED_CM_S * (1 + z))
    )
    speed_fraction = p0 / (1 + p0)
    settings = [(n, aL, 0.2) for aL in (16.0, 1e4) for n in (16, 32, 64)]
    settings += [(32, 1e4, cfl) for cfl in (0.4, 0.1, 0.05)]
    settings += [(32, 1e6, 0.2)]
    results = []
    for n, aL, cfl in settings:
        capacity = np.ones((n, n, n))
        ions = _logistic_averages(n, aL, 0.5)
        exact = _logistic_averages(n, aL, 0.5 + ct_fraction * speed_fraction)
        photons = np.zeros((6, n, n, n))
        vectors, _ = transport.directions(6)
        beam = np.flatnonzero(np.all(vectors == [1, 0, 0], axis=1))[0]
        photons[beam] = p0 * ions
        rate = np.zeros_like(capacity)
        sigma = aL / (density * (1 + z) ** 2 * (1 + p0) * length * transport.MPC_CM)
        start = time.monotonic()
        evolved = transport.evolve_interval(
            capacity,
            ions,
            photons,
            rate,
            rate,
            dt_seconds=duration,
            z_start=z,
            z_end=z,
            box_len_cMpc=length,
            hydrogen_density_z0_cm3=density,
            cross_section_cm2=sigma,
            cfl=cfl,
            threads=2,
        )
        centres = (np.arange(n) + 0.5) / n
        selected = (centres >= 0.3) & (centres <= 0.9)
        delta = evolved["ionized"][selected] - exact[selected]
        row = {
            "dimension": n,
            "front_inverse_width_times_box": aL,
            "cfl": cfl,
            "elapsed_seconds": time.monotonic() - start,
            "radiation_steps": evolved["steps"],
            "front_window_cells": int(selected.sum()),
            "front_window_ion_MAE": float(np.abs(delta).mean()),
            "front_window_maximum_ion_error": float(np.abs(delta).max()),
            "front_window_signed_integral_error_over_box": float(delta.sum() / n**3),
            "ionized_added_box_fraction": float(np.mean(evolved["ionized"] - ions)),
            "global_translated_wave_added_box_fraction_diagnostic": float(
                np.mean(exact - ions)
            ),
            "maximum_ion_fraction_excess": evolved["ledger"]["maximum_fraction_excess"],
            "photon_ledger_relative_residual": evolved["ledger"]["relative_residual"],
        }
        if aL >= 1e4:
            row["sharp_front_added_length_bias_over_box"] = float(
                np.mean(evolved["ionized"] - exact)
            )
        results.append(row)
        LOGGER.info("%s", json.dumps(row))
    after = (
        _digest(module_path),
        _digest(module_path.with_name("transport_kernel.cpp")),
    )
    if before != after:
        raise RuntimeError(
            "Transport implementation changed during the front benchmark"
        )
    compiler = shutil.which("g++")
    if compiler is None:
        raise RuntimeError("g++ is required to identify the transport compiler")
    return {
        "schema": 1,
        "equations": "d_t P + c_comoving d_x P = -kappa*N*P; d_t N = -kappa*N*P; I+N=1",
        "exact_front": "I=1/(1+exp[a*(x-x0-v*t)]); P=P0*I; v=c_comoving*P0/(1+P0); a=kappa*(1+P0)/c_comoving",
        "initialization_and_reference": "Analytic finite-volume cell averages using the logaddexp antiderivative, not samples at cell centres",
        "comparison_domain": "Only cells centred in x/L=[0.3,0.9]; their left edges exceed c_comoving*t/L=0.25, excluding the periodic trailing discontinuity",
        "global_integral_scope": "Translated smooth-wave global integral is diagnostic only; it is not the full periodic solution outside the causal comparison window. Sharp aL>=1e4 tails are negligible.",
        "parameters": {
            "box_len_cMpc": length,
            "redshift": z,
            "hydrogen_density_z0_cm3": density,
            "initial_photons_per_atom_behind_front": p0,
            "c_times_t_over_comoving_box": ct_fraction,
            "duration_seconds": duration,
            "angular_directions": 6,
            "occupied_direction": [1, 0, 0],
            "threads": 2,
            "source_rate": 0,
            "recombinations": False,
            "boundaries": "periodic",
        },
        "identity": {
            "transport_python_sha256": _digest(module_path),
            "transport_kernel_sha256": _digest(
                module_path.with_name("transport_kernel.cpp")
            ),
            "angular_rules_sha256": _digest(repo / "research/pc_matched/rt.py"),
            "benchmark_driver_sha256": _digest(__file__),
            "benchmark_driver": str(Path(__file__).resolve()),
            "compiler": compiler,
            "compiler_sha256": _digest(compiler),
            "compiler_version": subprocess.check_output(
                [compiler, "--version"], text=True
            ).splitlines()[0],
        },
        "cases": results,
    }


def main() -> None:
    """Run the front benchmark and write its requested JSON report."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--transport-module", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = benchmark(args.repo.resolve(), args.transport_module.resolve())
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
