"""Second-order, positive photon-number transport with carried radiation/ions.

This finite-interval kernel evolves the same effective single-absorber equation
as ``pc_matched.rt``, using physical c, exact coupled absorption, and an explicit
radiation reservoir. MUSCL/SSPRK2 replaces donor-cell streaming. Gas capacity is
held fixed within one call; the history driver is responsible for conservative
gas/ion advection between calls. This is not multifrequency H/He chemistry.
"""

from __future__ import annotations

import ctypes
from functools import lru_cache
import hashlib
from pathlib import Path
import subprocess
import tempfile
from threading import RLock

import numpy as np

from research.pc_matched.rt import (
    DEFAULT_CROSS_SECTION_CM2, LIGHT_SPEED_CM_S, MPC_CM, directions,
)

_LOCK = RLock()
ION_BOUND_TOLERANCE = 2e-11


@lru_cache(maxsize=1)
def _library():
    source = Path(__file__).with_name("transport_kernel.cpp")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    folder = Path(tempfile.mkdtemp(prefix=f"causal_rt_{digest}_"))
    binary = folder / "transport_kernel.so"
    subprocess.run(["g++", "-O3", "-std=c++17", "-shared", "-fPIC", "-fopenmp",
                    str(source), "-o", str(binary)], check=True, capture_output=True)
    library = ctypes.CDLL(str(binary))
    library.causal_rt_evolve.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                       *([ctypes.c_void_p] * 7), *([ctypes.c_double] * 7),
                                       ctypes.c_int64, ctypes.POINTER(ctypes.c_int64)]
    library.causal_rt_evolve.restype = ctypes.c_int
    return library


def _array(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real and nonnegative")
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return np.ascontiguousarray(result)


def evolve_interval(
    capacity: np.ndarray, ionized: np.ndarray, photons: np.ndarray,
    rate_start: np.ndarray, rate_end: np.ndarray, *,
    dt_seconds: float, z_start: float, z_end: float, box_len_cMpc: float,
    hydrogen_density_z0_cm3: float = 1.9e-7,
    cross_section_cm2: float = DEFAULT_CROSS_SECTION_CM2,
    cfl: float = 0.4, maximum_steps: int = 1_000_000, threads: int = 1,
) -> dict:
    """Advance a specified proper-time interval without changing input arrays.

    Counts are in mean-atom-cell units; rates are counts per second. Radiation
    has shape ``[directions,N,N,N]`` and the other fields are cubic ``[N,N,N]``.
    Vacuum capacity is allowed. Endpoint rates and redshifts interpolate
    linearly in proper time. Integrated photon injection uses exact trapezoids.
    A fractional ion excess up to 2e-11 from accumulated floating-point error
    in conservative remapping is retained without clipping. Such cells have
    zero remaining absorption capacity. This bound is shared with the gas
    remapper; the returned ledger records both its tolerance and measured excess.

    MUSCL with monotonized-central slopes and SSPRK2 is second order in smooth
    regions. A multidimensional Courant sum <=0.5 preserves nonnegative cell
    averages in both Euler stages. Limiting near extrema/discontinuities lowers
    the local order; numerical convergence must still be measured. Reaction and
    injection use symmetric splitting, with the original exact local coupled
    photon/neutral reaction. Redshifting of photon frequency, recombination,
    helium chemistry and heating are not added by this kernel.

    ``ledger.relative_residual`` is normalized by initial ions plus incoming
    and emitted radiation, so a nearly exhausted radiation tail does not make
    otherwise tiny floating-point ion increments an ill-conditioned relative
    error. The history driver must independently audit its cumulative budget.
    """
    scalars = (dt_seconds, z_start, z_end, box_len_cMpc,
               hydrogen_density_z0_cm3, cross_section_cm2, cfl)
    if not all(np.isfinite(x) for x in scalars):
        raise ValueError("Transport parameters must be finite")
    if dt_seconds <= 0 or box_len_cMpc <= 0 or hydrogen_density_z0_cm3 <= 0:
        raise ValueError("Duration, box length and absorber density must be positive")
    if not z_start >= z_end >= 0 or cross_section_cm2 < 0:
        raise ValueError("Redshifts must be nonincreasing/nonnegative and cross section nonnegative")
    if not 0 < cfl <= 0.5:
        raise ValueError("MUSCL positivity requires CFL in (0,0.5]")
    if type(maximum_steps) is not int or not 1 <= maximum_steps <= np.iinfo(np.int64).max:
        raise ValueError("Maximum steps must be a positive int64")
    if type(threads) is not int or not 1 <= threads <= 32:
        raise ValueError("Threads must be an integer between 1 and 32")
    capacity = _array(capacity, "capacity")
    old_ions, old_photons = _array(ionized, "ionized"), _array(photons, "photons")
    first, last = _array(rate_start, "rate_start"), _array(rate_end, "rate_end")
    if capacity.ndim != 3 or len(set(capacity.shape)) != 1 or len(capacity) < 2:
        raise ValueError("Capacity must be a nonnegative cubic grid with N >= 2")
    if (old_ions.shape != capacity.shape or first.shape != capacity.shape
            or last.shape != capacity.shape
            or np.any(old_ions - capacity > ION_BOUND_TOLERANCE * capacity)):
        raise ValueError("Ionized counts and rates must match capacity and ion bounds")
    if old_photons.ndim != 4 or old_photons.shape[1:] != capacity.shape:
        raise ValueError("Radiation must have shape [directions,N,N,N]")
    vectors, weights = directions(old_photons.shape[0])
    ions, radiation = old_ions.copy(), old_photons.copy()
    arrays = capacity, ions, radiation, first, last, vectors, weights
    used = ctypes.c_int64()
    with _LOCK:
        library = _library()
    status = library.causal_rt_evolve(
        len(capacity), len(weights), threads, *[a.ctypes.data for a in arrays],
        dt_seconds, z_start, z_end, box_len_cMpc / len(capacity) * MPC_CM,
        hydrogen_density_z0_cm3, cross_section_cm2, cfl,
        maximum_steps, ctypes.byref(used),
    )
    if status:
        reason = {1: "substep budget exceeded", 2: "invalid time increment",
                  3: "invalid absorption state", 4: "MUSCL positivity/CFL violated",
                  5: "allocation/kernel exception"}.get(status, str(status))
        raise RuntimeError(f"Causal RT failed: {reason}")
    emitted = np.sum((first.astype(np.longdouble) + last) * (np.longdouble(dt_seconds) / 2))
    incoming = old_photons.sum(dtype=np.longdouble)
    outgoing = radiation.sum(dtype=np.longdouble)
    initial_ions = old_ions.sum(dtype=np.longdouble)
    change = np.sum(ions.astype(np.longdouble) - old_ions)
    residual = incoming + emitted - outgoing - change
    scale = max(initial_ions + incoming + emitted, np.longdouble(np.finfo(float).tiny))
    relative = float(abs(residual) / scale)
    excess = np.divide(ions - capacity, capacity, out=np.zeros_like(capacity), where=capacity > 0)
    upper = max(float(excess.max()), 0.)
    lower = min(float(ions.min()), float(radiation.min()))
    finite = np.isfinite(ions).all() and np.isfinite(radiation).all()
    passed = bool(finite and relative <= 2e-10 and upper <= ION_BOUND_TOLERANCE and lower >= 0
                  and np.all(ions[capacity == 0] == 0))
    ledger = {"incoming_radiation": float(incoming), "emitted": float(emitted),
              "initial_ions": float(initial_ions), "ionized_increment": float(change),
              "outgoing_radiation": float(outgoing), "absolute_residual": float(residual),
              "normalization": float(scale), "relative_residual": relative,
              "maximum_fraction_excess": upper, "bound_tolerance": ION_BOUND_TOLERANCE,
              "minimum_count": lower, "passed": passed,
              "boundary": "periodic; no boundary photon loss"}
    if not passed:
        raise RuntimeError(f"Causal RT photon/ion ledger or bounds failed: {ledger}")
    return {"ionized": ions, "photons": radiation, "steps": used.value,
            "ledger": ledger, "directions": vectors, "weights": weights,
            "transport": "MUSCL-MC + SSPRK2, symmetric exact reaction/source splitting"}
