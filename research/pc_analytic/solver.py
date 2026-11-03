"""Frozen coeval directional transport with analytic column absorption.

Each source emits simultaneous dose batches. The default characteristic
transport tracks an exact common voxel phase per angular bin and applies the
column response once per completed chord. It has no interpolation diffusion.
The shared cell inventory and finite angular sampling remain approximations
to unresolved 3D illumination. No physical light-crossing clock is inferred.
The older interpolated scheme is retained only as an explicit diagnostic.
"""

from __future__ import annotations

import ctypes
import hashlib
import math
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from research.pc_matched.rt import MPC_CM
from research.pc_matched.rt import directions as angular_quadrature

_LIBRARY = None
_IDENTITY = None
_LOCK = threading.Lock()


def _library() -> ctypes.CDLL:
    global _LIBRARY, _IDENTITY
    with _LOCK:
        if _LIBRARY is not None:
            return _LIBRARY
        source = Path(__file__).with_name("transport_kernel.cpp")
        compiler = shutil.which(os.environ.get("CXX", "g++"))
        if compiler is None:
            raise RuntimeError("Analytic transport requires a C++/OpenMP compiler")
        version = subprocess.run(
            [compiler, "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
        flags = ["-O3", "-std=c++17", "-shared", "-fPIC", "-fopenmp"]
        directory = Path(tempfile.mkdtemp(prefix="pc_analytic_kernel_"))
        binary = directory / "transport.so"
        subprocess.run(
            [compiler, *flags, str(source), "-o", str(binary)],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
        _IDENTITY = {
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "compiler": compiler,
            "compiler_version": version,
            "flags": flags,
        }
        library = ctypes.CDLL(str(binary))
        library.analytic_transport.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        library.analytic_transport.restype = ctypes.c_int
        library.analytic_characteristics.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        library.analytic_characteristics.restype = ctypes.c_int
        library.analytic_shared_cell.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_double,
            ctypes.c_void_p,
        ]
        library.analytic_shared_cell.restype = ctypes.c_int
        _LIBRARY = library
        return library


def build_identity() -> dict[str, Any]:
    """Return the compiled kernel's source and compiler provenance.

    Returns
    -------
    dict
        Source SHA256, compiler path/version and build flags.
    """
    _library()
    return dict(_IDENTITY)


def _angles(
    count: int, rotation: np.ndarray | None, emission_weights: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    vectors, weights = angular_quadrature(count)
    if rotation is not None:
        rotation = np.asarray(rotation, dtype=float)
        if (
            rotation.shape != (3, 3)
            or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12, rtol=0)
            or np.linalg.det(rotation) < 0
        ):
            raise ValueError("rotation must be a proper orthogonal 3x3 matrix")
        vectors = np.ascontiguousarray(vectors @ rotation.T)
    if emission_weights is not None:
        weights = np.asarray(emission_weights, dtype=float)
        if (
            weights.shape != (count,)
            or not np.isfinite(weights).all()
            or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1, atol=1e-14, rtol=0)
        ):
            raise ValueError("emission_weights must be nonnegative and sum to one")
    return vectors, weights


def _boundary_code(boundary: str, transport_mode: str) -> int:
    codes = {"periodic": 0, "open": 1, "source_periodic_cube": 2}
    if boundary not in codes:
        raise ValueError(
            "boundary must be 'periodic', 'open' or 'source_periodic_cube'"
        )
    if transport_mode not in ("characteristic", "interpolated"):
        raise ValueError("transport_mode must be 'characteristic' or 'interpolated'")
    if boundary == "source_periodic_cube" and transport_mode != "characteristic":
        raise ValueError("source_periodic_cube requires characteristic transport")
    return codes[boundary]


def solve(
    source: np.ndarray,
    capacity: np.ndarray,
    *,
    box_len: float,
    redshift: float,
    nH_mean_comoving_cm3: float,
    sigma_cm2: float,
    directions: int = 24,
    exposure_steps: int = 16,
    threads: int = 1,
    boundary: str = "periodic",
    transport_mode: str = "characteristic",
    transport_fraction: float = 0.5,
    exhaustion_tolerance: float = 1e-7,
    batch_tail_tolerance: float = 1e-4,
    max_sweeps: int = 10000,
    initial_ionized: np.ndarray | None = None,
    emission_weights: np.ndarray | None = None,
    rotation: np.ndarray | None = None,
) -> dict[str, Any]:
    """Transport normalized photon and atom counts through one frozen snapshot.

    ``capacity=1`` is one cell's mean cosmological hydrogen inventory, so the
    optical depth per normalized atom count and unit-cell chord is
    ``sigma * nH_mean_comoving * (1+z)^2 * dx_comoving``.
    ``source`` uses the identical count unit. There is no extra density factor
    on sources. If the grid is refined, normalized source/capacity arrays must
    refer to the mean inventory of the NEW cell (not the original base cell).

    ``emission_weights`` is an optional normalized angular distribution for
    controlled beam tests. The default is an isotropic positive quadrature.
    ``rotation`` rotates that quadrature without altering its weights.
    Every photon left after termination is retained. A sweep limit returns
    ``audit['complete']=False`` and the un-emitted source budget separately.

    Parameters
    ----------
    source, capacity : ndarray
        Photon and atom counts in common mean-hydrogen-cell units, shape (N,N,N).
    box_len : float
        Comoving box side length in Mpc.
    redshift : float
        Frozen redshift used to convert comoving lengths and density.
    nH_mean_comoving_cm3 : float
        Physical mean hydrogen density at z=0, in cm^-3.
    sigma_cm2 : float
        Monochromatic photoionization cross section, in cm^2.
    directions : int, optional
        Positive cubically symmetric quadrature size: 6, 24, 26, 96 or 384.
    exposure_steps : int, optional
        Number of simultaneous source-dose batches.
    threads : int, optional
        OpenMP worker count.
    boundary : {'periodic', 'open', 'source_periodic_cube'}, optional
        Global periodic recycling, fixed open box, or one source-centred periodic
        cube followed by escape. The latter uses integer offsets from -N//2
        through N//2-1+N%2, matching the C2-Ray tracing domain.
    transport_mode : {'characteristic', 'interpolated'}, optional
        Exact integer voxel crossings with full-chord absorption (default), or
        the previous diffusive translation scheme for diagnostic comparisons.
    transport_fraction : float, optional
        Legacy interpolated-scheme translation path, in cell widths, in (0,1].
        Ignored by characteristic transport, which advances between voxel faces.
    exhaustion_tolerance : float, optional
        Final limiting-inventory tolerance relative to the smaller initial budget.
    batch_tail_tolerance : float, optional
        Legacy interpolated-scheme intermediate tolerance. Radiation is never dropped.
        A looser intermediate tolerance prevents an equal-inventory asymptotic tail
        from blocking the next source increment; it is a numerical control.
    max_sweeps : int, optional
        Total sweep limit; unfinished results are explicitly marked incomplete.
    initial_ionized : ndarray, optional
        Initial ionized counts in the same units and shape as capacity.
    emission_weights : ndarray, optional
        Angular emission weights for controlled anisotropic beam tests.
    rotation : ndarray, optional
        Proper orthogonal (3,3) matrix rotating the angular quadrature.

    Returns
    -------
    dict
        Ionized counts, remaining angular radiation, escaped and unemitted
        photons, independent budget audit, and numerical/physical metadata.
    """
    for name, array in (("source", source), ("capacity", capacity)):
        if np.iscomplexobj(array):
            raise ValueError(f"{name} must be real")
    source = np.ascontiguousarray(source, dtype=float)
    capacity = np.ascontiguousarray(capacity, dtype=float)
    if (
        source.ndim != 3
        or source.shape != capacity.shape
        or len(set(source.shape)) != 1
        or source.shape[0] < 2
    ):
        raise ValueError(
            "Source and capacity must have identical cubic grids with N >= 2"
        )
    for name, array in (("source", source), ("capacity", capacity)):
        if not np.isfinite(array).all() or np.any(array < 0):
            raise ValueError(f"{name} must contain finite nonnegative counts")
        if array.sum(dtype=np.longdouble) > np.finfo(float).max:
            raise ValueError(f"{name} total exceeds the supported audit range")
    for name, value in (
        ("threads", threads),
        ("exposure_steps", exposure_steps),
        ("max_sweeps", max_sweeps),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if threads > 256 or exposure_steps > 2**31 - 1 or max_sweeps > 2**63 - 1:
        raise ValueError("Requested work parameters exceed supported ranges")
    parameters = (
        box_len,
        redshift,
        nH_mean_comoving_cm3,
        sigma_cm2,
        transport_fraction,
        exhaustion_tolerance,
        batch_tail_tolerance,
    )
    if not all(math.isfinite(x) for x in parameters):
        raise ValueError("Physical and transport parameters must be finite")
    if box_len <= 0 or redshift < 0 or nH_mean_comoving_cm3 <= 0 or sigma_cm2 < 0:
        raise ValueError("Invalid box size, redshift, mean density or cross section")
    if not 0 < transport_fraction <= 1 or not 0 < exhaustion_tolerance < 1:
        raise ValueError("Transport fraction must lie in (0,1] and tolerance in (0,1)")
    if not exhaustion_tolerance <= batch_tail_tolerance < 1:
        raise ValueError("batch_tail_tolerance must be >= exhaustion_tolerance and < 1")
    boundary_code = _boundary_code(boundary, transport_mode)
    vectors, weights = _angles(directions, rotation, emission_weights)
    angular_source = np.ascontiguousarray(weights[:, None, None, None] * source)
    ions = (
        np.zeros_like(capacity)
        if initial_ionized is None
        else np.array(initial_ionized, dtype=float, order="C", copy=True)
    )
    if (
        ions.shape != capacity.shape
        or not np.isfinite(ions).all()
        or np.any(ions < 0)
        or np.any(ions > capacity)
    ):
        raise ValueError("initial_ionized must lie between zero and capacity")
    initial_ions = ions.copy()
    radiation = np.zeros_like(angular_source)
    escaped = np.zeros(directions)
    statistics = np.zeros(6)
    dx = box_len / source.shape[0] * MPC_CM
    coefficient = sigma_cm2 * nH_mean_comoving_cm3 * (1 + redshift) ** 2 * dx
    if not math.isfinite(coefficient):
        raise ValueError("Physical opacity exceeds the floating-point range")
    library = _library()
    started = time.perf_counter()
    arguments = (
        source.shape[0],
        directions,
        threads,
        angular_source.ctypes.data,
        capacity.ctypes.data,
        vectors.ctypes.data,
    )
    outputs = (
        ions.ctypes.data,
        radiation.ctypes.data,
        escaped.ctypes.data,
        statistics.ctypes.data,
    )
    if transport_mode == "characteristic":
        status = library.analytic_characteristics(
            *arguments,
            coefficient,
            exposure_steps,
            boundary_code,
            exhaustion_tolerance,
            max_sweeps,
            *outputs,
        )
    else:
        status = library.analytic_transport(
            *arguments,
            coefficient * transport_fraction,
            transport_fraction,
            exposure_steps,
            int(boundary == "periodic"),
            exhaustion_tolerance,
            batch_tail_tolerance,
            max_sweeps,
            *outputs,
        )
    elapsed = time.perf_counter() - started
    if status not in (0, 1):
        raise RuntimeError("Analytic transport kernel failed")
    prescribed = source.sum(dtype=np.longdouble)
    emitted = np.longdouble(statistics[2])
    new_ions = np.sum(ions.astype(np.longdouble) - initial_ions)
    stored = radiation.sum(dtype=np.longdouble)
    outflow = escaped.sum(dtype=np.longdouble)
    residual = emitted - new_ions - stored - outflow
    relative = float(residual / max(prescribed, np.longdouble(np.finfo(float).tiny)))
    bounded = bool(
        np.isfinite(ions).all()
        and np.isfinite(radiation).all()
        and np.all(ions >= 0)
        and np.all(ions <= capacity)
        and np.all(radiation >= 0)
        and np.all(escaped >= 0)
    )
    ledger_passed = bounded and abs(relative) < 2e-10
    if not ledger_passed:
        raise FloatingPointError(
            f"Independent analytic photon ledger failed: {relative:.6g}"
        )
    fraction = float(statistics[1]) / exposure_steps
    unresolved = (
        min(stored, np.sum(capacity - ions, dtype=np.longdouble))
        if boundary == "periodic"
        else stored
    )
    audit = {
        "method": "analytic-column-" + transport_mode,
        "complete": status == 0,
        "termination": (
            "transparent_medium"
            if sigma_cm2 == 0 and boundary == "periodic"
            else "complete"
        )
        if status == 0
        else "sweep_limit",
        "ledger_passed": ledger_passed,
        "relative_photon_residual": relative,
        "photon_total": float(prescribed),
        "emitted": float(emitted),
        "absorbed": float(new_ions),
        "initial_ionized": float(initial_ions.sum(dtype=np.longdouble)),
        "ionized_total": float(ions.sum(dtype=np.longdouble)),
        "remaining_photons": float(stored),
        "escaped_photons": float(outflow),
        "unemitted_photons": float(prescribed * (1 - fraction)),
        "atom_total": float(capacity.sum(dtype=np.longdouble)),
        "remaining_neutral": float(np.sum(capacity - ions, dtype=np.longdouble)),
        "unresolved_photons": float(unresolved),
        "elapsed_seconds": elapsed,
        "sweeps": int(statistics[0]),
        "completed_exposure_batches": int(statistics[1]),
        "exposure_steps": exposure_steps,
        "directions": directions,
        "exhaustion_tolerance": exhaustion_tolerance,
        "batch_tail_tolerance": batch_tail_tolerance,
        "maximum_retained_batch_photons": float(statistics[5]),
        "threads": threads,
        "boundary": boundary,
        "transport_mode": transport_mode,
        "retained_photons_are_propagated_in_later_batches": transport_mode
        == "interpolated",
        "spatial_angular_convergence_certified": False,
        "source_normalization_residual": float(
            (emitted + prescribed * (1 - fraction) - prescribed)
            / max(prescribed, np.longdouble(np.finfo(float).tiny))
        ),
    }
    metadata = {
        "box_len_cMpc": float(box_len),
        "redshift": float(redshift),
        "nH_mean_comoving_cm3": float(nH_mean_comoving_cm3),
        "sigma_cm2": float(sigma_cm2),
        "transport_fraction": float(transport_fraction),
        "optical_depth_per_normalized_atom_per_cell_chord": float(coefficient),
        "clock": "quasistatic simultaneous source-dose batches; no cosmic elapsed time",
        "source_geometry": "cell-centred angular emission; initial half chord to first voxel face",
        "local_closure": "shared atom inventory; exact equal-chord update, coupled population ODE for unequal simultaneous chords",
        "transport": (
            "exact integer voxel crossings; one absorption update per completed chord"
            if transport_mode == "characteristic"
            else "legacy positive trilinear translation; spatial diffusion and split-column error"
        ),
        "transport_fraction_applies": transport_mode == "interpolated",
        "batch_tail_tolerance_applies": transport_mode == "interpolated",
        "characteristic_tails": "per-batch tolerance tails retained without resetting their ray phase or propagating them again",
        "direction_vectors": vectors.tolist(),
        "direction_weights": weights.tolist(),
        "limitations": "No recombination, helium, heating, finite light speed or unresolved directional column structure; memory O(directions*cells)",
    }
    return {
        "ionized": ions,
        "remaining_radiation": radiation,
        "escaped": escaped,
        "unemitted_source": source * (1 - fraction),
        "audit": audit,
        "metadata": metadata,
    }
