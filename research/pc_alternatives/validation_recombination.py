"""Independent recombination/UVB closure probes; no production code is changed.

The graph-event solver has prescribed paths and is independent of event_front's
C++ heap/DDA. Units are dimensionless. Pure hydrogen, fixed density/temperature,
case-B local recombination, infinite light speed and constant sources are used.
The finite-opacity reference is a well-mixed-cell ODE, not continuum-resolved RT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from event_front_audit import GraphFront
from scipy.integrate import solve_ivp


def debt_events(atoms, initial, alpha_n, paths, rates, duration, steps):
    """Explicit recombination debit then a fresh sharp-front solve each step.

    This is the charitable version of adding R*dt to capacity: recombinations also
    decrease the existing ionized state. Omitting that debit fails even with Q=0.
    The cell x is interpreted as a homogeneous ionized fraction for the x**2 sink.
    Using opaque-until-full transport for that same x is the closure under test.
    """
    atoms, initial, alpha_n = map(np.asarray, (atoms, initial, alpha_n))
    x = initial.astype(float).copy()
    recombinations = np.zeros_like(x)
    absorbed = np.zeros_like(x)
    escaped = np.zeros(len(rates))
    dt = duration / steps
    for _ in range(steps):
        debit = atoms * alpha_n * x * x * dt
        if np.any(debit > atoms * x + 1e-13):
            raise ValueError("Explicit recombination step would make x negative")
        x -= debit / atoms
        solver = GraphFront((atoms * (1 - x)).tolist(), paths, rates)
        solver.advance(dt)
        x += solver.absorbed / atoms
        recombinations += debit
        absorbed += solver.absorbed
        escaped += solver.escaped
    emitted = float(sum(rates) * duration)
    return {
        "x": x.tolist(),
        "absorbed": absorbed.tolist(),
        "recombinations": recombinations.tolist(),
        "escaped_by_beam": escaped.tolist(),
        "emitted": emitted,
        "photon_ledger_error": float(abs(emitted - absorbed.sum() - escaped.sum())),
        "atom_ledger_max_error": float(
            np.max(np.abs(atoms * (x - initial) - absorbed + recombinations))
        ),
    }


def opacity_reference(
    atoms,
    initial,
    alpha_n,
    tau_neutral,
    paths,
    rates,
    duration,
    *,
    method="Radau",
    rtol=3e-11,
    atol=3e-13,
):
    """Independent coupled ODE with instantaneous Beer-Lambert attenuation.

    For each beam and cell: A=F_in*(1-exp[-tau0*(1-x)]), F_out=F_in-A.
    N*dx/dt=sum(A)-N*alpha_n*x**2. Rates are photon counts/time, not Gamma.
    The equivalent finite-volume Gamma is A/[N*(1-x)] with its finite x->1 limit.
    Integrated absorption, recombination and escape are separate ODE components.
    """
    atoms, initial, alpha_n, tau_neutral = (
        np.asarray(a, dtype=float) for a in (atoms, initial, alpha_n, tau_neutral)
    )
    count = atoms.size

    def rhs(_t, state):
        x = state[:count]
        tau = tau_neutral * np.maximum(1 - x, 0)
        absorption = np.zeros(count)
        escape = np.zeros(len(rates))
        for b, path in enumerate(paths):
            flux = rates[b]
            for cell in path:
                loss = flux * (-np.expm1(-tau[cell]))
                absorption[cell] += loss
                flux -= loss
            escape[b] = flux
        recombination = atoms * alpha_n * x * x
        return np.concatenate(
            ((absorption - recombination) / atoms, absorption, recombination, escape)
        )

    initial_state = np.r_[initial, np.zeros(2 * count + len(rates))]
    solution = solve_ivp(
        rhs, (0, duration), initial_state, method=method, rtol=rtol, atol=atol
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    state = solution.y[:, -1]
    x = state[:count]
    absorbed = state[count : 2 * count]
    recombined = state[2 * count : 3 * count]
    escaped = state[3 * count :]
    emitted = float(sum(rates) * duration)
    return {
        "x": x.tolist(),
        "absorbed": absorbed.tolist(),
        "recombinations": recombined.tolist(),
        "escaped_by_beam": escaped.tolist(),
        "emitted": emitted,
        "method": method,
        "nfev": solution.nfev,
        "photon_ledger_error": float(abs(emitted - absorbed.sum() - escaped.sum())),
        "atom_ledger_max_error": float(
            np.max(np.abs(atoms * (x - initial) - absorbed + recombined))
        ),
        "trajectory_x_min": float(solution.y[:count].min()),
        "trajectory_x_max": float(solution.y[:count].max()),
    }


def stromgren_probe():
    """Fully ionized volume closure: dN/dt=Q-N/t_rec; NOT parcel x**2."""
    total_atoms, rate, trec, duration = 10.0, 1.0, 1.0, 2.0
    exact = rate * trec * (-math.expm1(-duration / trec))
    rows = []
    for steps in (4, 8, 16, 32, 64, 128):
        dt, ionized, recombined, absorbed = duration / steps, 0.0, 0.0, 0.0
        for _ in range(steps):
            debit = ionized * dt / trec
            solver = GraphFront([total_atoms - ionized + debit], [[0]], [rate])
            solver.advance(dt)
            ionized += solver.absorbed[0] - debit
            recombined += debit
            absorbed += solver.absorbed[0]
        rows.append(
            {
                "steps": steps,
                "dt_over_trec": dt / trec,
                "N_ionized": ionized,
                "relative_volume_error": ionized / exact - 1,
                "relative_radius_error": (ionized / exact) ** (1 / 3) - 1,
                "atom_ledger_error": abs(ionized + recombined - absorbed),
            }
        )
    for i in range(1, len(rows)):
        rows[i]["observed_volume_order"] = math.log2(
            rows[i - 1]["relative_volume_error"] / rows[i]["relative_volume_error"]
        )
    # An exact constant-coefficient aggregate update is cheap, but cannot decide
    # the spatial sharing of photons between overlapping bubbles.
    analytic_update = 0.0
    for _ in range(4):
        dt = duration / 4
        analytic_update = analytic_update * math.exp(-dt / trec) + rate * trec * (
            -math.expm1(-dt / trec)
        )
    return {
        "model": "N is fully ionized volume times n_H; alpha*n_H=1/t_rec",
        "Q": rate,
        "t_rec": trec,
        "duration": duration,
        "exact_N": exact,
        "rows": rows,
        "exact_frozen_coefficient_update_error": abs(analytic_update - exact),
        "counterexample_to_large_explicit_steps": {
            "source_off_N0": 1.0,
            "dt_over_trec": 2.0,
            "explicit_N1": -1.0,
            "interpretation": "No positivity guarantee for explicit R_old*dt debit",
        },
    }


def source_off_probe():
    """Compare source-off recombination to its exact nonequilibrium solution."""
    duration, initial, alpha = 2.0, 1.0, 1.0
    exact = initial / (1 + alpha * initial * duration)
    rows = []
    for steps in (4, 8, 16, 32, 64, 128):
        result = debt_events([1], [initial], [alpha], [[0]], [0], duration, steps)
        x = result["x"][0]
        exact_substeps = initial
        for _ in range(steps):
            exact_substeps /= 1 + alpha * exact_substeps * duration / steps
        rows.append(
            {
                "steps": steps,
                "x": x,
                "absolute_error": abs(x - exact),
                "exact_recombination_substep_error": abs(exact_substeps - exact),
                "atom_ledger_error": result["atom_ledger_max_error"],
            }
        )
    return {
        "equation": "dx/dt=-alpha*n_H*x^2; fixed parcel, pure hydrogen, constant T",
        "duration": duration,
        "exact_x": exact,
        "rows": rows,
        "Gamma_zero_equilibrium_table_adapter_x": 1.0,
        "Gamma_zero_adapter_absolute_error": 1 - exact,
        "scope": (
            "The Gamma=0 result describes reusing the native equilibrium sink table "
            "as a nonequilibrium recombination RHS; it is not a claim that the full "
            "native ES simulation follows this toy evolution."
        ),
    }


def local_chemistry_step(x, incoming_rate, atoms, alpha_n, tau_neutral, dt):
    """Bounded Strang step for cells whose incoming photon flux is prescribed.

    Recombination x'=x/(1+a*x*h) and the photo-only finite-depth flow are exact.
    This is a cheap chemistry building block, not a multi-cell radiation solver:
    the attenuated incoming rates in an actual network still require coupling.
    """
    x, incoming_rate, atoms, alpha_n, tau_neutral = np.broadcast_arrays(
        np.asarray(x, dtype=float), incoming_rate, atoms, alpha_n, tau_neutral
    )
    half = x / (1 + alpha_n * x * dt / 2)
    recombined = atoms * (x - half)
    a = tau_neutral * (1 - half)
    # log(expm1(a)) = a + log(1-exp(-a)); stable even for tau0 >> 1.
    with np.errstate(divide="ignore", invalid="ignore"):
        log_expm1 = a + np.log(-np.expm1(-a))
        neutral = np.logaddexp(0, log_expm1 - tau_neutral * incoming_rate * dt / atoms)
        neutral /= tau_neutral
    photo = np.where(tau_neutral > 0, 1 - neutral, half)
    absorbed = atoms * (photo - half)
    end = photo / (1 + alpha_n * photo * dt / 2)
    recombined += atoms * (photo - end)
    escaped = incoming_rate * dt - absorbed
    return end, absorbed, recombined, escaped


def local_kernel_convergence():
    """Measure convergence against independent Radau and DOP853 integration."""
    reference = opacity_reference([1], [1], [1], [1], [[0]], [1], 2)
    independent = opacity_reference(
        [1], [1], [1], [1], [[0]], [1], 2, method="DOP853", rtol=1e-12, atol=1e-14
    )
    rows = []
    for steps in (4, 8, 16, 32, 64, 128):
        x, absorption, recombination, escape = np.array(1.0), 0.0, 0.0, 0.0
        for _ in range(steps):
            x, absorbed, rec, escaped = local_chemistry_step(x, 1, 1, 1, 1, 2 / steps)
            absorption += float(absorbed)
            recombination += float(rec)
            escape += float(escaped)
        rows.append(
            {
                "steps": steps,
                "x": float(x),
                "absolute_x_error": abs(float(x) - reference["x"][0]),
                "absorption_error": abs(absorption - reference["absorbed"][0]),
                "photon_ledger_error": abs(2 - absorption - escape),
                "atom_ledger_error": abs(float(x) - 1 - absorption + recombination),
            }
        )
    for i in range(1, len(rows)):
        rows[i]["observed_order"] = math.log2(
            rows[i - 1]["absolute_x_error"] / rows[i]["absolute_x_error"]
        )
    return {
        "reference": reference,
        "rows": rows,
        "Radau_DOP853_max_difference": max(
            abs(reference[k][0] - independent[k][0])
            for k in ("x", "absorbed", "recombinations", "escaped_by_beam")
        ),
        "scope": "Prescribed incoming flux, one homogeneous cell; no source/UVB iteration",
    }


def finite_opacity_probe():
    """Expose the sharp-front opacity closure separately from time truncation."""
    cases = []
    for optical_depth in (1.0, 10.0, 1000.0):
        ref = opacity_reference([1], [1], [1], [optical_depth], [[0]], [1], 2)
        rows = []
        for steps in (4, 16, 64, 256):
            result = debt_events([1], [1], [1], [[0]], [1], 2, steps)
            rows.append(
                {
                    "steps": steps,
                    **result,
                    "absolute_x_error_vs_finite_opacity": abs(
                        result["x"][0] - ref["x"][0]
                    ),
                    "escape_error_vs_finite_opacity": abs(
                        result["escaped_by_beam"][0] - ref["escaped_by_beam"][0]
                    ),
                }
            )
        cases.append(
            {"tau_neutral": optical_depth, "reference": ref, "debt_event_rows": rows}
        )
    atoms, initial, alpha, tau = [1, 3, 3], [1, 0, 0], [1, 0.4, 0.4], [1, 4, 4]
    paths, rates, duration = [[0, 1], [0, 2]], [1, 2], 1.5
    reference = opacity_reference(atoms, initial, alpha, tau, paths, rates, duration)
    overlap = []
    for steps in (8, 16, 32, 64, 128, 256):
        result = debt_events(atoms, initial, alpha, paths, rates, duration, steps)
        overlap.append(
            {
                "steps": steps,
                **result,
                "x_max_error_vs_finite_opacity": float(
                    np.max(np.abs(np.asarray(result["x"]) - reference["x"]))
                ),
                "absorption_L1_error_per_emitted_photon": float(
                    np.abs(np.asarray(result["absorbed"]) - reference["absorbed"]).sum()
                    / reference["emitted"]
                ),
            }
        )
    return {
        "single_preionized_cell": cases,
        "two_beams_shared_screen": {
            "atoms": atoms,
            "initial_x": initial,
            "alpha_n": alpha,
            "tau_neutral": tau,
            "paths": paths,
            "rates": rates,
            "duration": duration,
            "reference": reference,
            "debt_event_rows": overlap,
            "meaning": "Two beams overlap in cell0 then illuminate different target cells",
        },
    }


def reopened_screen_probe():
    """Expose deliberate stale-cursor reuse; fresh event_front.solve rebuilds it."""
    cached = GraphFront([0, 10], [[0, 1]], [1])
    cached.advance(0.1)
    before = cached.absorbed.copy()
    # One recombination time with no source: x_screen=1/(1+1)=1/2.
    cached.capacity[0] = 0.5
    cached.advance(0.2)
    stale_increment = cached.absorbed - before
    rebuilt = GraphFront([0.5, 9.9], [[0, 1]], [1])
    rebuilt.advance(0.2)
    return {
        "source_off_duration_over_initial_trec": 1.0,
        "screen_x_after_off": 0.5,
        "restart_photons": 0.2,
        "stale_cursor_absorption_increment": stale_increment.tolist(),
        "rebuilt_absorption_increment": rebuilt.absorbed.tolist(),
        "wrong_downstream_photons": float(stale_increment[1]),
        "scope": "Fresh solve rebuilds paths. Future stateful reuse must invalidate opacity caches.",
    }


def chemistry_microbenchmark():
    """Vectorized independent-cell kernel only; excludes transport and feedback."""
    rng = np.random.default_rng(8135)
    rows = []
    for size in (32, 64):
        shape = (size, size, size)
        initial = rng.uniform(0.1, 1, shape)
        flux = rng.uniform(0.1, 2, shape)
        local_chemistry_step(initial, flux, 1.0, 1.0, 1.0, 0.05)
        times = []
        for _ in range(7):
            start = time.perf_counter()
            local_chemistry_step(initial, flux, 1.0, 1.0, 1.0, 0.05)
            times.append(time.perf_counter() - start)
        rows.append(
            {
                "grid": size,
                "cells": size**3,
                "median_seconds": float(np.median(times)),
                "minimum_seconds": min(times),
                "repeats": len(times),
            }
        )
    return {
        "scope": "NumPy independent chemistry cells with prescribed flux; no ray/UVB solve",
        "rows": rows,
    }


def native_code_evidence(repo):
    """Record the native source files audited for Qin20 feedback conventions."""
    files = [
        "src/py21cmfast/templates/Qin20.toml",
        "src/py21cmfast/src/IonisationBox.c",
        "src/py21cmfast/src/recombinations.c",
        "src/py21cmfast/src/thermochem.c",
    ]
    return {
        "files_sha256": {
            name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
            for name in files
        },
        "Qin20": {
            "RECOMB_MODEL": "inhomogeneous",
            "CELL_RECOMB": False,
            "USE_REIONIZATION_PHOTOHEATING_FEEDBACK": True,
            "USE_TS_FLUCT": True,
        },
        "native_ES_Gamma": "First barrier crossing: R * gamma_prefactor * filtered emissivity",
        "native_sink": "MHR density PDF; equilibrium HI with self-shielding; case-B T=1e4 K",
        "native_Gamma_zero_table_rate": 0,
        "coupling": "Previous Gamma and z_reion enter source photoheating threshold M_turn",
        "native_order": "previous cumulative recombinations enter ES; current Gamma/x enter next cumulative recombinations",
    }


def main():
    """Run the independent validation suite and save numerical evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_suffix(".json")
    )
    args = parser.parse_args()
    result = {
        "scope": "Independent fixed-density pure-H toy audit; not a Qin20 cosmological run",
        "probe_files_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in (
                "validation_recombination.py",
                "test_validation_recombination.py",
                "event_front_audit.py",
            )
        },
        "native_code": native_code_evidence(Path(__file__).resolve().parents[2]),
        "stromgren": stromgren_probe(),
        "source_off": source_off_probe(),
        "finite_opacity": finite_opacity_probe(),
        "bounded_local_chemistry": local_kernel_convergence(),
        "reopened_screen": reopened_screen_probe(),
        "chemistry_cost": chemistry_microbenchmark(),
        "references": [
            "https://arxiv.org/abs/astro-ph/0508416",
            "https://arxiv.org/abs/1402.2298",
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(  # noqa: T201
        json.dumps(
            {
                "output": str(args.output),
                "single_cell_debt_error": result["finite_opacity"][
                    "single_preionized_cell"
                ][0]["debt_event_rows"][-1]["absolute_x_error_vs_finite_opacity"],
                "local_kernel_last_order": result["bounded_local_chemistry"]["rows"][
                    -1
                ]["observed_order"],
                "chemistry_cost": result["chemistry_cost"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
