"""Direct RT of one native cumulative photon budget in a frozen snapshot.

The propagation clock is an exposure experiment, not cosmic time. A pulse
(the default), or a constant source of specified duration, injects the supplied
budget exactly once. No excursion-set operation or global ionization rescaling
is used. Remaining photons are retained, including any post-ionization excess.
"""

from __future__ import annotations

import ctypes

import numpy as np

from research.pc_matched import rt


LEDGER_TOLERANCE = 2e-10


def _nonnegative_grid(value: np.ndarray, name: str) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real, finite and nonnegative")
    array = np.ascontiguousarray(value, dtype=np.float64)
    if (array.ndim != 3 or len(set(array.shape)) != 1 or array.shape[0] < 2
            or not np.isfinite(array).all() or np.any(array < 0)):
        raise ValueError(f"{name} must be a finite nonnegative cubic grid with N >= 2")
    if array.sum(dtype=np.longdouble) > np.finfo(float).max:
        raise ValueError(f"{name} total exceeds the supported count range")
    return array


def _kernel_interval(capacity, ionized, radiation, rate, *, vectors, weights,
                     dt_seconds, redshift, box_len_cMpc, density, sigma,
                     cfl, threads, maximum_steps, cumulative_budget):
    """Use the same kernel for vacuum cells or a nearly exhausted inventory.

    The historical Python wrapper requires strictly positive capacity, whereas
    its exact reaction kernel supports zero capacity. Its local ledger also
    divides by the remaining radiation, becoming ill-conditioned when double
    precision ion increments are tiny. Audit the unmodified state against the
    cumulative budget instead; do not fill, remove or rescale any counts.
    """
    ions, photons = ionized.copy(), radiation.copy()
    arrays = (capacity, ions, photons, rate, rate, vectors, weights)
    used = ctypes.c_int64()
    with rt._LOCK:
        library = rt._library()
    status = library.rt_evolve(
        capacity.shape[0], len(weights), threads,
        *[array.ctypes.data for array in arrays],
        dt_seconds, redshift, redshift,
        box_len_cMpc / capacity.shape[0] * rt.MPC_CM,
        density, sigma, cfl, maximum_steps, ctypes.byref(used),
    )
    if status:
        raise RuntimeError(f"Direct RT kernel failed with status {status}")
    incoming = radiation.sum(dtype=np.longdouble)
    emitted = rate.sum(dtype=np.longdouble) * np.longdouble(dt_seconds)
    outgoing = photons.sum(dtype=np.longdouble)
    change = np.sum(ions.astype(np.longdouble) - ionized)
    residual = abs(incoming + emitted - outgoing - change)
    relative = residual / max(incoming + emitted, cumulative_budget,
                              np.longdouble(np.finfo(float).tiny))
    bounded = (np.isfinite(ions).all() and np.isfinite(photons).all()
               and np.all(ions >= 0) and np.all(photons >= 0)
               and np.all(ions <= capacity * (1 + 2e-13)))
    if not bounded or relative > LEDGER_TOLERANCE:
        raise RuntimeError("Direct RT interval failed its independent ledger/bounds")
    return {"ionized": ions, "photons": photons, "steps": used.value}


def solve(
    source: np.ndarray, capacity: np.ndarray, *, redshift: float,
    box_len_cMpc: float, hydrogen_density_z0_cm3: float = 1.9e-7,
    photon_energy_ev: float = 18.0, directions: int = 96, cfl: float = 0.4,
    threads: int = 1, exhaustion_tolerance: float = 1e-6,
    max_box_crossings: float = 20.0, injection_box_crossings: float = 0.0,
    max_steps: int = 1_000_000,
) -> dict:
    """Transport a supplied native photon budget through a frozen gas grid.

    ``source`` and ``capacity`` contain nonnegative counts in identical
    effective mean-atom-cell units. The caller supplies any spatial refinement.
    Zero-capacity vacuum cells are supported without a density floor. The
    default pulse initializes isotropic angular radiation as ``weights*source``.
    Positive ``injection_box_crossings`` instead emits the same budget at a
    constant rate over that many proper box light-crossing times.

    Convergence requires completed injection and
    ``min(radiation, neutral)/min(total_source, total_capacity) <= tolerance``.
    This is exhaustion of the limiting inventory, not spatial/angular
    convergence. Equal source and capacity may approach exhaustion slowly;
    explicit time/step limits return ``audit.converged=False`` and the actual
    unfinished state. A failed photon ledger or invalid state raises an error.

    Returns ``ionized[N,N,N]``, angular ``radiation[D,N,N,N]``, ``audit`` and
    ``metadata``. No stored radiation is discarded, even after ionization.
    """
    source = _nonnegative_grid(source, "source")
    capacity = _nonnegative_grid(capacity, "capacity")
    if source.shape != capacity.shape:
        raise ValueError("source and capacity must have the same shape")
    scalars = (redshift, box_len_cMpc, hydrogen_density_z0_cm3, photon_energy_ev,
               cfl, exhaustion_tolerance, max_box_crossings, injection_box_crossings)
    if not all(np.isfinite(value) for value in scalars):
        raise ValueError("All solver parameters must be finite")
    if redshift < 0 or box_len_cMpc <= 0 or hydrogen_density_z0_cm3 <= 0:
        raise ValueError("Redshift must be nonnegative; box length and density positive")
    if photon_energy_ev < 13.6:
        raise ValueError("Photons must be above the hydrogen ionization threshold")
    if not 0 < cfl <= 1 or not 0 < exhaustion_tolerance < 1:
        raise ValueError("CFL must lie in (0,1] and exhaustion tolerance in (0,1)")
    if max_box_crossings < 0 or injection_box_crossings < 0:
        raise ValueError("Propagation and injection durations must be nonnegative")
    if type(threads) is not int or not 1 <= threads <= 32:
        raise ValueError("threads must be an integer between 1 and 32")
    if type(max_steps) is not int or not 1 <= max_steps <= np.iinfo(np.int64).max:
        raise ValueError("max_steps must be a positive int64")
    vectors, weights = rt.directions(directions)
    crossing = box_len_cMpc * rt.MPC_CM / ((1 + redshift) * rt.LIGHT_SPEED_CM_S)
    maximum_time = max_box_crossings * crossing
    injection_time = injection_box_crossings * crossing
    if not all(np.isfinite(value) for value in (crossing, maximum_time, injection_time)):
        raise ValueError("Propagation or injection duration exceeds the supported range")
    sigma = 6.3e-18 * (13.6 / photon_energy_ev) ** 3
    total_source = source.sum(dtype=np.longdouble)
    total_capacity = capacity.sum(dtype=np.longdouble)
    normalization = max(min(total_source, total_capacity),
                        np.longdouble(np.finfo(float).tiny))
    zero = np.zeros_like(capacity)
    ions = zero.copy()
    pulse = injection_time == 0
    photons = (weights[:, None, None, None] * source if pulse
               else np.zeros((directions, *capacity.shape)))
    photons = np.ascontiguousarray(photons)
    emission_rate = np.ascontiguousarray(source / injection_time) if not pulse else zero
    if not np.isfinite(emission_rate).all():
        raise ValueError("Injection rate exceeds the supported count range")
    stable_dt = cfl * crossing / (capacity.shape[0] * np.abs(vectors).sum(axis=1).max())
    vacuum = bool(np.any(capacity == 0))
    elapsed, steps = 0.0, 0
    status = "unstarted"

    def inventory():
        radiation_total = photons.sum(dtype=np.longdouble)
        ionized_total = ions.sum(dtype=np.longdouble)
        neutral_total = np.sum(capacity.astype(np.longdouble) - ions)
        exhausted = min(radiation_total, neutral_total) / normalization
        return radiation_total, ionized_total, neutral_total, exhausted

    while True:
        radiation_total, ionized_total, neutral_total, exhaustion = inventory()
        injection_complete = pulse or elapsed >= injection_time or total_source == 0
        if injection_complete and exhaustion <= exhaustion_tolerance:
            status = "exhausted"
            break
        if steps >= max_steps:
            status = "max_steps_reached"
            break
        if elapsed >= maximum_time:
            status = "max_box_crossings_reached"
            break
        injecting = not pulse and elapsed < injection_time
        duration = min(0.25 * crossing, maximum_time - elapsed,
                       stable_dt * (max_steps - steps) * (1 - 1e-12))
        if injecting:
            duration = min(duration, injection_time - elapsed)
        if duration <= 0 or elapsed + duration == elapsed:
            status = "no_time_progress"
            break
        rate = emission_rate if injecting else zero
        if vacuum or (not injecting and radiation_total < 1e-4 * total_source):
            result = _kernel_interval(
                capacity, ions, photons, rate, vectors=vectors, weights=weights,
                dt_seconds=duration, redshift=redshift, box_len_cMpc=box_len_cMpc,
                density=hydrogen_density_z0_cm3, sigma=sigma, cfl=cfl,
                threads=threads, maximum_steps=max_steps - steps,
                cumulative_budget=total_source,
            )
        else:
            result = rt.evolve_interval(
                capacity, ions, photons, rate, rate, dt_seconds=duration,
                z_start=redshift, z_end=redshift, box_len_cMpc=box_len_cMpc,
                hydrogen_density_z0_cm3=hydrogen_density_z0_cm3,
                cross_section_cm2=sigma, cfl=cfl, threads=threads,
                maximum_steps=max_steps - steps,
            )
        ions, photons = result["ionized"], result["photons"]
        elapsed += duration
        steps += result["steps"]

    emitted_fraction = 1.0 if pulse else min(elapsed / injection_time, 1.0)
    emitted = total_source * np.longdouble(emitted_fraction)
    residual = emitted - ionized_total - radiation_total
    relative = abs(residual) / max(emitted, np.longdouble(np.finfo(float).tiny))
    bounds_passed = (np.isfinite(ions).all() and np.isfinite(photons).all()
                     and np.all(ions >= 0) and np.all(photons >= 0)
                     and np.all(ions <= capacity * (1 + 2e-13)))
    ledger_passed = bool(bounds_passed and relative <= LEDGER_TOLERANCE)
    if not ledger_passed:
        raise RuntimeError(f"Independent coeval RT ledger/bounds failed: residual={float(relative):.6g}")
    converged = status == "exhausted"
    return {
        "ionized": ions,
        "radiation": photons,
        "audit": {
            "passed": bool(ledger_passed and converged),
            "ledger_passed": ledger_passed, "converged": converged, "status": status,
            "source_injection_complete": bool(injection_complete),
            "prescribed_source": float(total_source), "emitted": float(emitted),
            "ionized": float(ionized_total), "radiation_in_flight": float(radiation_total),
            "remaining_neutral": float(neutral_total), "total_capacity": float(total_capacity),
            "absolute_residual": float(residual), "relative_residual": float(relative),
            "exhaustion_fraction": float(exhaustion),
            "exhaustion_tolerance": float(exhaustion_tolerance),
            "ledger_tolerance": LEDGER_TOLERANCE,
            "spatial_angular_convergence_certified": False,
        },
        "metadata": {
            "clock": "Frozen-snapshot exposure time; not cosmic elapsed time",
            "injection": "pulse" if pulse else "constant_rate",
            "injection_box_crossings": float(injection_box_crossings),
            "pseudo_time_seconds": elapsed, "box_crossings": elapsed / crossing,
            "steps": steps, "redshift": float(redshift), "box_len_cMpc": float(box_len_cMpc),
            "hydrogen_density_z0_cm3": float(hydrogen_density_z0_cm3),
            "photon_energy_ev": float(photon_energy_ev), "cross_section_cm2": sigma,
            "directions": directions, "cfl": cfl, "threads": threads,
            "boundary": "periodic; no boundary photon loss",
            "radiation_layout": "angular counts [direction,x,y,z]",
            "max_box_crossings": float(max_box_crossings), "max_steps": max_steps,
        },
    }
