"""Run pinned pyC2Ray CPU transport on the same frozen hydrogen inputs.

The upstream Fortran sources remain byte-identical. ``GREY_NOTABLES`` selects
their monochromatic rate routine. The external driver sums individual-source
rates before calling the author's ``doric`` chemistry routine with recombination
and collisional coefficients set to zero. No thermal evolution is present.

This reference traces at most one periodically wrapped, source-centred cube.
The author's subbox optimization stops opaque source traces at an explicit
native loss threshold. Its raw traced-surface counter includes both subbox
truncation and outer-domain outflow. Photons are not recycled, and this counter
is not called an independently measured escape flux or used to enforce closure.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

UPSTREAM_REPOSITORY = "https://github.com/cosmic-reionization/pyC2Ray"
UPSTREAM_COMMIT = "5befb722128c638ce94026bab0cd95fb797b0ee7"
PINNED_FILES = {
    "LICENSE": "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986",
    "src/c2ray/photorates.f90": "3e0d4af0048e1db0570d88ff43eaa1bda2146e444bf20ace0fd2937480478403",
    "src/c2ray/raytracing.f90": "91ebb56b0bbad4ec7aebba0658a2b67128ac8bc91a121515421185415fac2fdb",
    "src/c2ray/chemistry.f90": "468aca4ae7e62810ce545ab47858fcdb5c3fbaa5148bbca19768c053cf5767f5",
}
MPC_CM = 3.0856775814913673e24
HERE = Path(__file__).resolve().parent


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_subbox_size(dimension: int, requested: int) -> int:
    """Retain both asymmetric even-grid domain edges with the native AND loop."""
    size = min(dimension, requested)
    left = dimension // 2
    right = dimension // 2 - 1 + dimension % 2
    if (left + size - 1) // size != (right + size - 1) // size:
        return dimension
    return size


def verify_upstream(upstream: str | Path) -> dict[str, str]:
    """Verify the actual upstream transport, chemistry and license bytes.

    Parameters
    ----------
    upstream : str or Path
        Root of the pinned source checkout or minimal verified source archive.

    Returns
    -------
    dict
        Verified relative paths and SHA-256 digests.
    """
    upstream = Path(upstream)
    for relative, expected in PINNED_FILES.items():
        path = upstream / relative
        if not path.is_file() or _digest(path) != expected:
            raise ValueError(f"Pinned pyC2Ray source mismatch: {relative}")
    return dict(PINNED_FILES)


def fetch_upstream(destination: str | Path) -> Path:
    """Download and verify the minimal pinned source set on a login host.

    Parameters
    ----------
    destination : str or Path
        Directory in which to retain the four original upstream files.

    Returns
    -------
    Path
        Verified source root. Compute jobs can use this without network access.
    """
    destination = Path(destination).resolve()
    base = "https://raw.githubusercontent.com/cosmic-reionization/pyC2Ray/"
    for relative, expected in PINNED_FILES.items():
        target = destination / relative
        if target.is_file():
            if _digest(target) != expected:
                raise ValueError(f"Pinned pyC2Ray source mismatch: {relative}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(
            f"{base}{UPSTREAM_COMMIT}/{relative}", timeout=60
        ) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Downloaded pyC2Ray checksum mismatch: {relative}")
        target.write_bytes(content)
    verify_upstream(destination)
    return destination


def build(
    upstream: str | Path,
    output: str | Path,
    compiler: str | None = None,
) -> dict[str, Any]:
    """Compile the unchanged CPU sources and the external comparison driver.

    Parameters
    ----------
    upstream : str or Path
        Verified upstream source root.
    output : str or Path
        Build directory outside the frozen research source tree.
    compiler : str, optional
        GNU Fortran executable. Defaults to ``FC`` then ``gfortran``.

    Returns
    -------
    dict
        Executable path, checksums, compiler arguments and physical scope.
    """
    upstream = Path(upstream).resolve()
    hashes = verify_upstream(upstream)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    compiler = (
        compiler
        or os.environ.get("FC")
        or shutil.which("gfortran")
        or shutil.which("gfortran-13")
    )
    if not compiler:
        raise RuntimeError(
            "GNU Fortran is required for the actual C2-Ray reference; set FC"
        )
    compiler_path = shutil.which(compiler)
    if not compiler_path:
        raise RuntimeError(f"Fortran compiler not found: {compiler}")
    version = subprocess.run(
        [compiler_path, "--version"], check=True, text=True, capture_output=True
    ).stdout.splitlines()[0]
    flags = [
        "-O3",
        "-cpp",
        "-DUSE_SUBBOX",
        "-DGREY_NOTABLES",
        "-ffree-line-length-none",
    ]
    flags += shlex.split(os.environ.get("FFLAGS", ""))
    identity = {
        "repository": UPSTREAM_REPOSITORY,
        "commit": UPSTREAM_COMMIT,
        "files": hashes,
        "driver_sha256": _digest(HERE / "c2ray_driver.F90"),
        "compiler": str(compiler_path),
        "compiler_version": version,
        "flags": flags,
    }
    manifest_path = output / "build.json"
    executable = output / "c2ray_reference"
    if manifest_path.is_file() and executable.is_file():
        previous = json.loads(manifest_path.read_text())
        if previous.get("identity") == identity and previous.get(
            "executable_sha256"
        ) == _digest(executable):
            return previous
    sources = []
    for relative in PINNED_FILES:
        if relative == "LICENSE":
            shutil.copyfile(upstream / relative, output / "UPSTREAM_LICENSE")
        else:
            copied = output / Path(relative).name
            shutil.copyfile(upstream / relative, copied)
            sources.append(copied)
    driver = output / "driver.F90"
    shutil.copyfile(HERE / "c2ray_driver.F90", driver)
    command = [
        compiler_path,
        *flags,
        *(str(p) for p in sources),
        str(driver),
        "-o",
        str(executable),
    ]
    result = subprocess.run(
        command, cwd=output, text=True, capture_output=True, check=False
    )
    (output / "compiler.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"C2-Ray compilation failed; see {output / 'compiler.log'}")
    manifest = {
        "identity": identity,
        "executable": str(executable),
        "executable_sha256": _digest(executable),
        "native_transport_algorithm_unchanged": True,
        "upstream_sources_modified": False,
        "license": "GPL-3.0",
        "source_summation": "One upstream source call per emitter; sum rates before shared chemistry",
        "iteration": "External damped time-averaged-opacity iteration with explicit convergence",
        "zero_rate_chemistry": "Exact unchanged-state branch avoids native all-zero-rate 0/0",
        "opacity_neutral_fraction_floor": 1e-14,
        "native_ionized_fraction_floor": 1e-14,
        "boundary": "source_centred_periodic_cube_without_recycling",
        "cpu_parallelism": "serial upstream CPU kernel",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def run(
    source: np.ndarray,
    capacity: np.ndarray,
    *,
    box_len: float,
    redshift: float,
    nH_mean_comoving_cm3: float,
    sigma_cm2: float,
    exposure_steps: int = 32,
    workdir: str | Path | None = None,
    upstream: str | Path | None = None,
    compiler: str | None = None,
    threads: int = 1,
    exposure_seconds: float = 1e13,
    tolerance: float = 1e-7,
    max_iterations: int = 500,
    timeout: float = 1800,
    subbox_size: int = 2,
    loss_fraction: float = 1e-10,
) -> dict[str, Any]:
    """Run monochromatic, non-recombining C2-Ray on the supplied arrays.

    Parameters
    ----------
    source, capacity : ndarray
        Cubic photon and atom counts in the same dimensionless normalization.
        Capacity is density relative to the declared mean hydrogen density.
    box_len : float
        Comoving side length in Mpc.
    redshift : float
        Fixed snapshot redshift, used only for conversion to proper density.
    nH_mean_comoving_cm3 : float
        Mean comoving hydrogen number density in cm^-3.
    sigma_cm2 : float
        Explicit monochromatic hydrogen absorption cross section in cm^2.
    exposure_steps : int
        Equal steps during a constant-emission exposure.
    workdir : str or Path, optional
        Fresh run directory retaining binary input, output and native log.
    upstream : str or Path, optional
        Pinned pyC2Ray root, or the ``PC_C2RAY_SOURCE`` environment setting.
    compiler : str, optional
        GNU Fortran executable; passed to :func:`build`.
    threads : int
        Recorded requested allocation. The upstream CPU kernel is serial.
    exposure_seconds : float
        Exposure duration. With the disabled time-dependent microphysics,
        only emitted count and relative source history affect this experiment.
    tolerance : float
        Maximum absolute change of the time-averaged and final fractions.
    max_iterations : int
        Maximum shared-opacity iterations per exposure step.
    timeout : float
        Native subprocess timeout in seconds.
    subbox_size : int
        Requested upstream subbox expansion in cells. An increment that would
        omit an asymmetric domain edge falls back to a single full-cube trace.
    loss_fraction : float
        Native traced-surface loss threshold relative to each source rate.
        Use zero and ``subbox_size >= dimension`` for the full-cube control.

    Returns
    -------
    dict
        Ionized count field, neutral fraction, independent field budget,
        native diagnostic counters and explicit reference provenance.
    """
    source = np.asarray(source, dtype=np.float64)
    capacity = np.asarray(capacity, dtype=np.float64)
    values = {
        "box_len": box_len,
        "nH_mean_comoving_cm3": nH_mean_comoving_cm3,
        "sigma_cm2": sigma_cm2,
        "exposure_seconds": exposure_seconds,
        "tolerance": tolerance,
        "timeout": timeout,
    }
    for name, value in values.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(redshift) or redshift < 0:
        raise ValueError("redshift must be finite and nonnegative")
    if not np.isfinite(loss_fraction) or not 0 <= loss_fraction < 1:
        raise ValueError("loss_fraction must be finite and in [0, 1)")
    if (
        source.ndim != 3
        or len(set(source.shape)) != 1
        or source.shape[0] < 3
        or capacity.shape != source.shape
    ):
        raise ValueError(
            "C2-Ray inputs must be matching cubic arrays with dimension >=3"
        )
    if not np.all(np.isfinite(source)) or np.any(source < 0):
        raise ValueError("source must be finite and nonnegative")
    if not np.all(np.isfinite(capacity)) or np.any(capacity <= 0):
        raise ValueError("capacity must be finite and positive")
    for name, value in {
        "exposure_steps": exposure_steps,
        "threads": threads,
        "max_iterations": max_iterations,
        "subbox_size": subbox_size,
    }.items():
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    upstream = upstream or os.environ.get("PC_C2RAY_SOURCE")
    if upstream is None:
        raise RuntimeError(
            "Set PC_C2RAY_SOURCE or upstream to a pinned pyC2Ray source archive"
        )
    workdir = Path(workdir or tempfile.mkdtemp(prefix="pc_c2ray_")).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    output_path = workdir / "output.bin"
    if output_path.exists():
        raise FileExistsError(f"C2-Ray output already exists: {output_path}")
    manifest = build(upstream, workdir / "build", compiler)
    n = source.shape[0]
    actual_subbox_size = _safe_subbox_size(n, int(subbox_size))
    cell_comoving_cm = box_len * MPC_CM / n
    atom_unit = nH_mean_comoving_cm3 * cell_comoving_cm**3
    dr = cell_comoving_cm / (1.0 + redshift)
    density = capacity * nH_mean_comoving_cm3 * (1.0 + redshift) ** 3
    indices = np.argwhere(source > 0)
    if len(indices):
        rates = source[tuple(indices.T)] * atom_unit / exposure_seconds
    else:
        indices = np.array([[0, 0, 0]])
        rates = np.zeros(1)
    input_path = workdir / "input.bin"
    with input_path.open("wb") as stream:
        stream.write(b"PC2REF02")
        stream.write(
            np.array(
                [n, exposure_steps, len(indices), max_iterations], dtype="<i4"
            ).tobytes()
        )
        stream.write(
            np.array(
                [dr, sigma_cm2, exposure_seconds, tolerance], dtype="<f8"
            ).tobytes()
        )
        stream.write(np.array([actual_subbox_size], dtype="<i4").tobytes())
        stream.write(np.array([loss_fraction], dtype="<f8").tobytes())
        stream.write(np.asarray(density, dtype="<f8").tobytes(order="F"))
        stream.write(np.asarray(indices.T + 1, dtype="<i4").tobytes(order="F"))
        stream.write(np.asarray(rates, dtype="<f8").tobytes())
    started = time.perf_counter()
    with (workdir / "native.log").open("w") as log:
        result = subprocess.run(
            [manifest["executable"], str(input_path), str(output_path)],
            cwd=workdir,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    elapsed = time.perf_counter() - started
    if result.returncode:
        raise RuntimeError(f"C2-Ray reference failed; see {workdir / 'native.log'}")
    with output_path.open("rb") as stream:
        magic = stream.read(8)
        header = np.frombuffer(stream.read(16), dtype="<i4")
        stats = np.frombuffer(stream.read(64), dtype="<f8")
        xhii = (
            np.frombuffer(stream.read(8 * n**3), dtype="<f8")
            .reshape(source.shape, order="F")
            .copy()
        )
        gamma = (
            np.frombuffer(stream.read(8 * n**3), dtype="<f8")
            .reshape(source.shape, order="F")
            .copy()
        )
        trailing = stream.read(1)
    if (
        magic != b"PC2OUT01"
        or header[0] != n
        or header[1] != exposure_steps
        or header[3] != 1
        or trailing
    ):
        raise RuntimeError("Malformed or incomplete C2-Ray output")
    if not np.all(np.isfinite(xhii)) or np.any(xhii < 0) or np.any(xhii > 1):
        raise RuntimeError("C2-Ray returned an invalid ionization field")
    ionized = capacity * xhii
    photon_total = float(source.sum(dtype=np.float64))
    ionized_total = float(ionized.sum(dtype=np.float64))
    native_outflow = float(stats[3] / atom_unit)
    primary = float(stats[4] / atom_unit)
    scale = max(photon_total, float(capacity.sum()) * 1e-30)
    audit = {
        "complete": True,
        "elapsed_seconds": elapsed,
        "photon_total": photon_total,
        "ionized_total": ionized_total,
        "native_outflow_photons": native_outflow,
        "native_traced_surface_loss_photons": native_outflow,
        "native_outflow_counter_scope": "Final source subbox surface; includes truncation and outer-domain outflow",
        "native_primary_rate_integral": primary,
        "native_outflow_fraction": native_outflow / scale,
        "emitted_minus_field_ions_relative": (photon_total - ionized_total) / scale,
        "native_counter_residual_relative": (
            ionized_total + native_outflow - photon_total
        )
        / scale,
        "field_vs_native_rate_relative": (ionized_total - primary) / scale,
        "escaped_photons": None,
        "photon_residual": None,
        "independent_escape_available": False,
        "native_transport_algorithm_unchanged": True,
        "boundary": manifest["boundary"],
        "boundary_matches_closed_periodic": False,
        "exposure_steps": int(exposure_steps),
        "exposure_seconds": exposure_seconds,
        "iterations": int(header[2]),
        "last_iteration_change": float(stats[5]),
        "tolerance": tolerance,
        "requested_threads": int(threads),
        "actual_threads": 1,
        "subbox_size_requested": int(subbox_size),
        "subbox_size_actual": actual_subbox_size,
        "subbox_loss_fraction_requested": loss_fraction,
        "subbox_loss_fraction_native_float32": float(np.float32(loss_fraction)),
        "initial_xHII": 0.0,
        "recombinations": False,
        "collisional_ionization": False,
        "thermal_evolution": False,
        "helium": False,
        "sigma_cm2": sigma_cm2,
        "atom_unit": atom_unit,
        "input_sha256": _digest(input_path),
        "output_sha256": _digest(output_path),
    }
    (workdir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return {
        "ionized": ionized,
        "xHI": 1.0 - xhii,
        "photoionization_rate_s": gamma,
        "audit": audit,
        "provenance": manifest,
        "workdir": str(workdir),
    }
