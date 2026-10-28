"""Direct monochromatic finite-volume radiative transfer in fixed gas.

The solver evolves directional photon number and hydrogen ionization together.
It uses the physical light speed, periodic upwind fluxes, finite photoionization
opacity, and an analytic local photon/neutral reaction update. Photon number is
stored in comoving effective mean-atom-cell units, so cosmic dilution enters
opacity through ``(1+z)**3`` and streaming through ``(1+z)``. Cosmological photon
redshifting, frequency evolution, helium, recombinations, and heating are absent.
This is a controlled RTE baseline, not a production multifrequency RT model.
"""

from __future__ import annotations

import ctypes
import hashlib
import subprocess
import tempfile
from functools import lru_cache
from itertools import product
from pathlib import Path
from threading import RLock

import numpy as np

LIGHT_SPEED_CM_S = 2.99792458e10
MPC_CM = 3.0856775814913673e24
DEFAULT_CROSS_SECTION_CM2 = 6.3e-18 * (13.6 / 18.0) ** 3
SUPPORTED_DIRECTION_COUNTS = (6, 24, 26, 96, 384)
_LOCK = RLock()


def _cubed_sphere(count: int) -> tuple[np.ndarray, np.ndarray]:
    """Integrate cube-face cell solid angles and sample their centre directions."""
    side = int(np.sqrt(count // 6))
    edges = np.linspace(-1.0, 1.0, side + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    u, v = np.meshgrid(edges, edges, indexing="ij")
    primitive = np.arctan2(u * v, np.sqrt(1 + u * u + v * v))
    solid_angles = (primitive[1:, 1:] - primitive[:-1, 1:]
                    - primitive[1:, :-1] + primitive[:-1, :-1]).ravel()
    face_u, face_v = np.meshgrid(centres, centres, indexing="ij")
    vectors = []
    for normal_axis, sign in product(range(3), (-1, 1)):
        face = np.empty((side * side, 3))
        tangential = [axis for axis in range(3) if axis != normal_axis]
        face[:, normal_axis] = sign
        face[:, tangential[0]] = face_u.ravel()
        face[:, tangential[1]] = face_v.ravel()
        vectors.append(face)
    unit_vectors = np.concatenate(vectors)
    unit_vectors /= np.linalg.norm(unit_vectors, axis=1, keepdims=True)
    weights = np.tile(solid_angles / (4 * np.pi), 6)
    weights /= weights.sum()
    return np.ascontiguousarray(unit_vectors), np.ascontiguousarray(weights)


def directions(count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a positive angular quadrature with cubic symmetry.

    Parameters
    ----------
    count : {6, 24, 26, 96, 384}
        Six Cartesian directions, or all normalized nonzero vectors in
        ``{-1, 0, 1}**3`` for 26. Both retain their equal weights. The refinable
        24/96/384 sequence uses 2/4/8 cells per side of each cube face, unit
        directions through face-cell centres, and exact rectangular solid-angle
        weights. Face rectangles cover the sphere without gaps or overlaps.

    Returns
    -------
    vectors, weights : numpy.ndarray
        Unit vectors ``[count, 3]`` and positive weights summing to one.
        Exact first/second moments follow from cubic symmetry. Higher angular
        moments and transport solutions require angular-convergence checks.
    """
    if type(count) is not int or count not in SUPPORTED_DIRECTION_COUNTS:
        raise ValueError(f"Angular direction count must be one of {SUPPORTED_DIRECTION_COUNTS}")
    if count not in (6, 26):
        return _cubed_sphere(count)
    vectors = np.array([v for v in product((-1, 0, 1), repeat=3)
                        if any(v) and (count == 26 or sum(abs(x) for x in v) == 1)], dtype=float)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.ascontiguousarray(vectors), np.full(count, 1 / count)


@lru_cache(maxsize=1)
def _library() -> ctypes.CDLL:
    source = Path(__file__).with_name("rt_kernel.cpp")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    folder = Path(tempfile.mkdtemp(prefix=f"direct_rt_{digest}_"))
    binary = folder / "rt_kernel.so"
    subprocess.run(["g++", "-O3", "-std=c++17", "-shared", "-fPIC", "-fopenmp",
                    str(source), "-o", str(binary)], check=True, capture_output=True)
    library = ctypes.CDLL(str(binary))
    library.rt_evolve.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                 *([ctypes.c_void_p] * 7), *([ctypes.c_double] * 7),
                                 ctypes.c_int64, ctypes.POINTER(ctypes.c_int64)]
    library.rt_evolve.restype = ctypes.c_int
    return library


def _array(value: np.ndarray, name: str) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real and nonnegative")
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return np.ascontiguousarray(result)


def _validate_scalars(
    dt_seconds: float, z_start: float, z_end: float, box_len_cMpc: float,
    hydrogen_density_z0_cm3: float, cross_section_cm2: float,
    cfl: float, maximum_steps: int, threads: int,
) -> None:
    values = (dt_seconds, z_start, z_end, box_len_cMpc,
              hydrogen_density_z0_cm3, cross_section_cm2, cfl)
    if not all(np.isfinite(v) for v in values):
        raise ValueError("Transport parameters must be finite")
    if dt_seconds <= 0 or box_len_cMpc <= 0 or hydrogen_density_z0_cm3 <= 0:
        raise ValueError("Duration, box length and hydrogen density must be positive")
    if not z_start >= z_end >= 0 or cross_section_cm2 < 0:
        raise ValueError("Redshifts must be nonincreasing/nonnegative and cross section nonnegative")
    if not 0 < cfl <= 1:
        raise ValueError("CFL number must lie in (0, 1]")
    if type(maximum_steps) is not int or not 1 <= maximum_steps <= np.iinfo(np.int64).max:
        raise ValueError("Maximum steps must be a positive int64")
    if type(threads) is not int or not 1 <= threads <= 32:
        raise ValueError("Threads must be an integer between 1 and 32")


def evolve_interval(
    capacity: np.ndarray, ionized: np.ndarray, photons: np.ndarray,
    rate_start: np.ndarray, rate_end: np.ndarray, *,
    dt_seconds: float, z_start: float, z_end: float, box_len_cMpc: float,
    hydrogen_density_z0_cm3: float = 1.9e-7,
    cross_section_cm2: float = DEFAULT_CROSS_SECTION_CM2,
    cfl: float = 0.4, maximum_steps: int = 1_000_000, threads: int = 1,
) -> dict:
    """Evolve a finite-speed radiative-transfer interval without editing inputs.

    Parameters
    ----------
    capacity, ionized : numpy.ndarray
        Total and ionized effective hydrogen counts per mean-atom cell, cubic
        ``[N,N,N]`` arrays. Capacity is held fixed during the calculation.
    photons : numpy.ndarray
        Carried directional photon counts ``[D,N,N,N]`` in the same units,
        with ``D`` in ``{6,24,26,96,384}`` (see :func:`directions`).
    rate_start, rate_end : numpy.ndarray
        Nonnegative photon emissivities per cell per second at interval ends.
        Their linear time interpolation is integrated exactly in each substep.
    dt_seconds : float
        Proper elapsed time in seconds.
    z_start, z_end : float
        Endpoint redshifts, linearly interpolated in proper time within this
        interval. Equal redshifts give a static-scale-factor test.
    box_len_cMpc : float
        Comoving periodic box side length.
    hydrogen_density_z0_cm3 : float, optional
        Proper mean absorber density at redshift zero, defining count units.
    cross_section_cm2 : float, optional
        Monochromatic H photoionization cross section; default is 18-eV hydrogen.
        Zero disables absorption for transparent-transport validation.
    cfl : float, optional
        Conservative streaming Courant number in ``(0,1]``.
    maximum_steps : int, optional
        Hard substep budget; exceeding it raises instead of returning a partial
        interval or silently reducing the physical speed of light.
    threads : int, optional
        OpenMP worker count in ``[1,32]``; default one. Parallel work partitions
        independent cells or angles without changing each cell's summation order.

    Returns
    -------
    dict
        Updated ``ionized`` and ``photons``, substep count, angular vectors and
        weights, and an independently recomputed global photon/ion ledger.
        The explicit radiation reservoir is carried into the next interval.

    Notes
    -----
    Streaming uses a positive conservative first-order upwind finite volume
    scheme. Source/reaction/transport splitting is symmetric, with exact local
    coupled photon-neutral absorption. Overall transport remains first order
    and requires resolution and time-step convergence checks.
    """
    _validate_scalars(dt_seconds, z_start, z_end, box_len_cMpc,
                      hydrogen_density_z0_cm3, cross_section_cm2, cfl, maximum_steps, threads)
    capacity = _array(capacity, "capacity")
    initial_ionized = _array(ionized, "ionized")
    initial_photons = _array(photons, "photons")
    first, last = _array(rate_start, "rate_start"), _array(rate_end, "rate_end")
    if (capacity.ndim != 3 or len(set(capacity.shape)) != 1 or capacity.shape[0] < 2
            or np.any(capacity <= 0)):
        raise ValueError("Capacity must be a strictly positive cubic grid with N >= 2")
    if (initial_ionized.shape != capacity.shape or first.shape != capacity.shape
            or last.shape != capacity.shape or np.any(initial_ionized > capacity)):
        raise ValueError("Ionized counts and source grids must match capacity and ion bounds")
    if initial_photons.ndim != 4 or initial_photons.shape[1:] != capacity.shape:
        raise ValueError("Radiation must have shape [directions,N,N,N]")
    vectors, weights = directions(initial_photons.shape[0])
    updated_ions, updated_photons = initial_ionized.copy(), initial_photons.copy()
    used_steps = ctypes.c_int64()
    arrays = (capacity, updated_ions, updated_photons, first, last, vectors, weights)
    with _LOCK:
        library = _library()
    status = library.rt_evolve(
        capacity.shape[0], len(vectors), threads, *[a.ctypes.data for a in arrays],
        dt_seconds, z_start, z_end, box_len_cMpc / capacity.shape[0] * MPC_CM,
        hydrogen_density_z0_cm3, cross_section_cm2, cfl,
        maximum_steps, ctypes.byref(used_steps),
    )
    if status:
        reasons = {1: "substep budget exceeded", 2: "invalid time increment",
                   3: "invalid absorption state", 4: "CFL positivity violated",
                   5: "allocation/kernel exception"}
        raise RuntimeError(f"Direct RT failed: {reasons.get(status, status)}")
    emitted = np.sum((first.astype(np.longdouble) + last) * (np.longdouble(dt_seconds) / 2))
    incoming = initial_photons.sum(dtype=np.longdouble)
    outgoing = updated_photons.sum(dtype=np.longdouble)
    change = np.sum(updated_ions.astype(np.longdouble) - initial_ionized)
    residual = incoming + emitted - outgoing - change
    scale = max(incoming + emitted, np.longdouble(np.finfo(float).tiny))
    upper = float(np.max((updated_ions - capacity) / capacity))
    lower = float(min(updated_ions.min(), updated_photons.min()))
    relative = float(abs(residual) / scale)
    finite = np.isfinite(updated_ions).all() and np.isfinite(updated_photons).all()
    passed = bool(finite and relative <= 2e-10 and upper <= 2e-13 and lower >= 0)
    ledger = {"incoming_radiation": float(incoming), "emitted": float(emitted),
              "ionized_increment": float(change), "outgoing_radiation": float(outgoing),
              "absolute_residual": float(residual), "relative_residual": relative,
              "maximum_fraction_excess": max(upper, 0.0), "minimum_count": lower,
              "passed": passed, "boundary": "periodic; no boundary photon loss"}
    if not passed:
        raise RuntimeError(f"Direct RT photon/ion ledger or bounds failed: {ledger}")
    return {"ionized": updated_ions, "photons": updated_photons,
            "steps": used_steps.value, "ledger": ledger,
            "directions": vectors, "weights": weights}
