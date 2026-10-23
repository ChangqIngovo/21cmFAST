#!/usr/bin/env python3
"""Audit event_front's time/horizon semantics without modifying the solver.

The packet and analytic functions here are small independent reference problems,
not a replacement transport implementation.  Counterexamples are expected results:
the current event solver explicitly adopts infinite c and does not return state.
Run with Python + numpy; report inputs are saved native Qin20 TOML/benchmark JSON.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import tomllib

import numpy as np

import event_front

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
DEFAULT_INPUTS = WORKSPACE / "qin20_validation/native_paired_z8/es_inputs.toml"
# Match src/py21cmfast/src/Constants.c and Constants.h, rather than silently
# substituting Astropy constants/cosmography. These clocks are slightly different.
MPC_KM = 3.08567758e19
MYR_S = 31556925.9747 * 1e6
C_KM_S = 299792.458
NATIVE_H0_PER_H = 3.2407e-18


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def delayed_absorption(time, rate, distance, speed, emission_stop=np.inf):
    """Exact count at an unsaturating opaque target after a transparent path.

    A constant source turns on at t=0 and off at emission_stop.  Its photons reach
    the target after distance/speed.  Spatial opacity is static; no scattering.
    """
    if min(time, rate, distance) < 0 or speed <= 0 or emission_stop < 0:
        raise ValueError("Nonnegative times/counts and positive speed required")
    return rate * max(0.0, min(time - distance / speed, emission_stop))


def relativistic_radius(time, rate=1.0, density=3 / (4 * np.pi), speed=1.0):
    """Exact no-recombination, static, uniform sharp-front radius by bisection.

    rate*(t-R/c) = (4*pi/3)*density*R**3.  Integrating the relativistic
    jump condition in Iliev+2006 section 2.10 (eq.3) gives this equation.
    """
    if time < 0 or rate <= 0 or density <= 0 or speed <= 0:
        raise ValueError("Positive rate/density/speed and nonnegative time required")
    upper = min(speed * time, (3 * rate * time / (4 * np.pi * density)) ** (1 / 3))
    lower = 0.0
    for _ in range(90):
        radius = (lower + upper) / 2
        required_time = radius / speed + 4 * np.pi * density * radius**3 / (3 * rate)
        if required_time > time:
            upper = radius
        else:
            lower = radius
    return (lower + upper) / 2


@dataclass(frozen=True)
class Flash:
    """One +x delta pulse; retains position and total travel, even after source off."""

    x: float
    photons: float
    travelled: float = 0.0
    time: float = 0.0
    absorbed: float = 0.0
    exhausted_horizon: bool = False


def advance_flash(state, duration, speed, target, maximum_distance=np.inf):
    """Exact 1-D oracle, with target absorption only strictly before a range cap.

    A range cap is an unresolved stopping condition, not a photon sink or a clock.
    Time advances while an unresolved packet remains; a later extension of its
    cap would require deciding the intended physical model, not just this oracle.
    """
    if duration < 0 or speed <= 0 or maximum_distance < state.travelled:
        raise ValueError("Invalid transport interval/range")
    if state.photons == 0 or state.exhausted_horizon:
        return replace(state, time=state.time + duration)
    distance = min(speed * duration, maximum_distance - state.travelled)
    to_target = target - state.x
    if 0 <= to_target <= speed * duration and to_target < maximum_distance - state.travelled:
        return Flash(target, 0, state.travelled + to_target, state.time + duration,
                     state.absorbed + state.photons)
    return replace(state, x=state.x + distance, travelled=state.travelled + distance,
                   time=state.time + duration,
                   exhausted_horizon=(state.travelled + distance >= maximum_distance))


def audit_subset(result):
    keys = ("photons", "absorbed", "escaped", "retained", "ledger_residual")
    return {key: result["audit"][key] for key in keys}


def counterexamples():
    shape = (8,) * 3
    source = np.zeros(shape)
    source[1, 1, 1] = 1
    capacity = np.zeros(shape)
    capacity[6, 1, 1] = 100
    beam = np.array([[1.0, 0, 0]])
    original_source = source.copy()
    restarted = []
    for _ in range(3):
        out = event_front.solve(source, capacity, ray_directions=beam, maximum_distance=2)
        restarted.append(audit_subset(out))
        capacity = np.maximum(capacity - out["absorbed"], 0)
        # A proposed adapter, NOT current production behavior: interpreting the
        # source-indexed residual as freshly emitted isotropic/beam source counts.
        source = out["source_retained"]
    pulse = Flash(x=1.5, photons=1)
    physical_flash = []
    for _ in range(3):
        pulse = advance_flash(pulse, duration=1, speed=2, target=6)
        physical_flash.append(asdict(pulse))

    two_dirs = np.array([[1., 0, 0], [0, 1., 0]])
    cap = np.zeros(shape)
    cap[6, 1, 1] = cap[1, 2, 1] = 100
    first = event_front.solve(original_source, cap, ray_directions=two_dirs, maximum_distance=2)
    second = event_front.solve(first["source_retained"], cap - first["absorbed"],
                               ray_directions=two_dirs, maximum_distance=2)

    cap = np.zeros(shape)
    cap[6, 1, 1] = 100
    finite = []
    for duration in (1., 3., 10.):
        out = event_front.solve(original_source, cap, ray_directions=beam,
                                maximum_distance=7, duration=duration)
        exact = delayed_absorption(duration, 1, 4.5, 2)
        finite.append(dict(duration=duration, distance=4.5, speed=2,
                           event_absorbed=out["audit"]["absorbed"],
                           finite_c_exact_absorbed=exact,
                           finite_c_exact_in_flight=duration - exact,
                           **{"audit": audit_subset(out)}))

    # Even c*dt as a search cutoff is insufficient: photons emitted late in the
    # step have not had dt to travel. Target begins at 0.5 cell, c=1, dt=1.
    near_cap = np.zeros(shape)
    near_cap[2, 1, 1] = 100
    near = event_front.solve(original_source, near_cap, ray_directions=beam,
                            maximum_distance=1, duration=1)

    # Valid infinite-c partition, including passage through a cell that exhausts.
    cap = np.zeros(shape)
    cap[2, 1, 1] = 0.2
    cap[6, 1, 1] = 100
    whole = event_front.solve(original_source, cap, ray_directions=beam, duration=1)
    half = event_front.solve(original_source, cap, ray_directions=beam, duration=.5)
    half2 = event_front.solve(original_source, cap - half["absorbed"],
                             ray_directions=beam, duration=.5)

    # A physical source turns off after one time unit; its last photons still
    # arrive until t=3.25. A zero-source restart cannot represent those arrivals.
    off_cap = np.zeros(shape)
    off_cap[6, 1, 1] = 100
    zero_step = event_front.solve(np.zeros(shape), off_cap, ray_directions=beam, duration=1)
    flash_at2 = delayed_absorption(2, 1, 4.5, 2, emission_stop=1)
    flash_at3 = delayed_absorption(3, 1, 4.5, 2, emission_stop=1)

    # A per-emitter total-distance cap must not be reset at each step.
    capped = Flash(1.5, 1)
    reset = Flash(1.5, 1)
    for _ in range(3):
        capped = advance_flash(capped, 1, 2, 6, maximum_distance=3)
        reset = advance_flash(replace(reset, travelled=0, exhausted_horizon=False),
                              1, 2, 6, maximum_distance=3)

    # Counts at successive snapshots are cumulative, not new source increments.
    large = np.zeros(shape)
    large[1, 1, 1] = 100
    prev = event_front.solve(original_source * 3, large, ray_directions=beam)
    bad_next = event_front.solve(original_source * 5, large - prev["absorbed"],
                                ray_directions=beam)
    good_next = event_front.solve(original_source * 2, large - prev["absorbed"],
                                 ray_directions=beam)

    return {
        "source_restart_stalls": {"meaning": "Hypothetical adapter re-emits retained at original source",
            "restart_steps": restarted, "total_external_emission": 1,
            "naive_sum_solver_photons": sum(r["photons"] for r in restarted),
            "correct_global_ledger": "initial_emission = total_absorption + current_reservoir; reinjected carry is not new emission",
            "finite_c_flash_reference": physical_flash,
            "finite_c_same_t0_to_t1_emission_absorbed_at_t3": delayed_absorption(3, 1, 4.5, 2, emission_stop=1),
            "flash_reference_scope": "Independent delta-flash state oracle; event_front itself accepts a constant rate, not a delta flash",
            "exact_flash_arrival_time": 2.25},
        "re_isotropizing_carry_changes_direction": {
            "first_absorbed_near_y": float(first["absorbed"][1, 2, 1]),
            "first_retained_x_only": first["audit"]["retained"],
            "restart_spurious_extra_y_absorption": float(second["absorbed"][1, 2, 1]),
            "correct_extra_y_absorption": 0,
            "wrong_fraction_of_carried_count": float(second["absorbed"].sum() / first["audit"]["retained"]),
            "restart_audit": audit_subset(second)},
        "constant_source_finite_c": finite,
        "distance_cutoff_is_not_causal_transport": {
            "duration": 1, "speed": 1, "maximum_distance_equals_c_dt": 1,
            "target_entry_distance": .5, "event_absorbed": near["audit"]["absorbed"],
            "finite_c_exact_absorbed": .5, "finite_c_exact_in_flight": .5},
        "source_off_arrivals": {"physical_emission_interval": [0, 1],
            "physical_target_arrival_interval": [2.25, 3.25],
            "exact_absorption_increment_t2_to_t3": flash_at3 - flash_at2,
            "zero_source_restart_absorption_increment": zero_step["audit"]["absorbed"]},
        "range_budget_must_persist": {"absolute_maximum_distance": 3,
            "target_distance": 4.5, "correct": asdict(capped),
            "wrong_reset_distance_every_step": asdict(reset)},
        "infinite_c_no_retained_restart_is_valid": {
            "split_vs_whole_l1": float(np.abs(whole["absorbed"] - half["absorbed"] - half2["absorbed"]).sum()),
            "total_absorption": whole["audit"]["absorbed"],
            "scope": "Fixed source geometry/rates, unchanged directions, monotone capacity, no finite-c state"},
        "cumulative_source_is_not_increment": {
            "previous_cumulative": 3, "next_cumulative": 5,
            "wrong_absorbed_total": prev["audit"]["absorbed"] + bad_next["audit"]["absorbed"],
            "correct_increment_absorbed_total": prev["audit"]["absorbed"] + good_next["audit"]["absorbed"],
            "scope": "Scalar bookkeeping illustration; spatial native source localization requires separate validation"},
        "uniform_relativistic_front_reference": [
            dict(time=t, radius_finite_c=relativistic_radius(t), radius_infinite_c=t**(1/3),
                 radius_infinite_over_finite=t**(1/3) / relativistic_radius(t),
                 volume_infinite_over_finite=t / relativistic_radius(t)**3)
            for t in (.1, 1., 10., 100.)],
    }


def native_dtdz_seconds(z, cosmo):
    """Exact algebraic simplification of cosmology.c:dtdz (before C float rounding).

    Do not silently normalize matter+Lambda to unity: retain its explicit square
    root as in the native expression. Default OMl is the Python property 1-OMm.
    """
    om = cosmo["OMm"]
    oml = cosmo.get("OMl", 1 - om)
    h0_s = cosmo["hlittle"] * NATIVE_H0_PER_H
    return -np.sqrt(om + oml) / (h0_s * (1 + z) * np.sqrt(om * (1 + z)**3 + oml))


def cosmic_interval(z_early, z_late, cosmo, clock="hubble"):
    """Integrate one of two distinct native clocks; default is native hubble().

    hubble()/t_hubble() includes OMr and sets the SFR timescale; dtdz()/drdz()
    omits OMr and is used in native sink/thermal updates. Keep this discrepancy
    explicit instead of confusing it with quadrature error or Astropy cosmography.
    """
    if not z_early > z_late >= 0:
        raise ValueError("Require z_early > z_late >= 0")
    om = cosmo["OMm"]
    oml = cosmo.get("OMl", 1 - om)
    h0_s = cosmo["hlittle"] * NATIVE_H0_PER_H
    x, w = np.polynomial.legendre.leggauss(64)
    z = (z_early + z_late) / 2 + x * (z_early - z_late) / 2
    if clock == "hubble":
        e = np.sqrt(om * (1 + z)**3 + cosmo.get("OMr", 0) * (1 + z)**4 + oml)
        dt_dz_abs = 1 / (h0_s * (1 + z) * e)
    elif clock == "dtdz":
        dt_dz_abs = -native_dtdz_seconds(z, cosmo)
    else:
        raise ValueError("Unknown clock")
    dt = (z_early - z_late) / 2 * np.sum(w * dt_dz_abs)
    distance = C_KM_S / MPC_KM * (z_early - z_late) / 2 * np.sum(w * (1 + z) * dt_dz_abs)
    return float(dt / MYR_S), float(distance)


def interval_row(z_early, z_late, cosmo, box, grid, bubble_max):
    dt, light_horizon = cosmic_interval(z_early, z_late, cosmo)
    dtdz_dt, dtdz_horizon = cosmic_interval(z_early, z_late, cosmo, clock="dtdz")
    endpoint_dt = -native_dtdz_seconds(z_late, cosmo) * (z_early - z_late) / MYR_S
    scales = {"cell": box / grid, "R_BUBBLE_MAX": bubble_max,
              "box": box, "event_default_diagonal": np.sqrt(3) * box}
    return dict(z_early=float(z_early), z_late=float(z_late), grid=int(grid), box_cMpc=float(box),
        dt_Myr=dt, true_light_horizon_cMpc=light_horizon,
        native_dtdz_integrated_Myr=dtdz_dt, native_drdz_horizon_cMpc=dtdz_horizon,
        native_recombination_endpoint_dt_Myr=float(endpoint_dt),
        scales={name: dict(distance_cMpc=float(distance),
            light_crossing_at_z_late_Myr=float(distance / (1 + z_late) * MPC_KM / C_KM_S / MYR_S),
            distance_over_interval_light_horizon=float(distance / light_horizon))
            for name, distance in scales.items()})


def native_times(inputs_path):
    inputs_path = Path(inputs_path)
    data = tomllib.loads(inputs_path.read_text())
    cosmo = data["CosmoParams"]
    sim = data["SimulationOptions"]
    box, radius = sim["BOX_LEN"], data["AstroParams"]["R_BUBBLE_MAX"]
    nodes = np.asarray(data["node_redshifts"])
    if not np.all(np.diff(nodes) < 0):
        raise ValueError("Saved native nodes must be in decreasing redshift order")
    benchmark_path = HERE / "native_benchmark.json"
    bench = json.loads(benchmark_path.read_text())["preparation"]
    coarse = bench["nodes"]
    # All intervals from the actual saved input; selected rows are convenient
    # summaries. The 64^3 grid is the runner's declared default, not an executed run.
    all_rows = [interval_row(a, b, cosmo, box, sim["HII_DIM"], radius)
                for a, b in zip(nodes[:-1], nodes[1:], strict=True)]
    selected = []
    for target_z in (20., 15., 10., 8., 6.):
        i = int(np.argmin(abs(nodes[1:] - target_z)))
        selected.append(interval_row(nodes[i], nodes[i+1], cosmo, box, 64, radius))
    return dict(
        inputs_path=str(inputs_path), inputs_sha256=hash_file(inputs_path),
        node_count=len(nodes), saved_grid=int(sim["HII_DIM"]),
        saved_box_cMpc=float(box), node_redshifts=nodes.tolist(),
        cosmology={key: cosmo[key] for key in ("hlittle", "OMm", "OMb", "OMr", "OMn")},
        clock="Main dt/horizon: integrate native hubble(z), including OMr; also report native dtdz/drdz (omits OMr) and endpoint |dtdz(z_late)|*dz used for recombinations",
        constants={"native_H0_per_h_s_inv": NATIVE_H0_PER_H,
                   "native_seconds_per_year": MYR_S / 1e6, "native_km_per_Mpc": MPC_KM},
        OMl_from_native_property=1-cosmo["OMm"],
        nominal_ZPRIME_STEP_FACTOR=sim["ZPRIME_STEP_FACTOR"],
        minimum_dt_Myr=min(r["dt_Myr"] for r in all_rows),
        maximum_dt_Myr=max(r["dt_Myr"] for r in all_rows),
        minimum_interval_light_horizon_cMpc=min(r["true_light_horizon_cMpc"] for r in all_rows),
        maximum_interval_light_horizon_cMpc=max(r["true_light_horizon_cMpc"] for r in all_rows),
        selected_default_runner_grid64_intervals=selected,
        grid64_scope="Scale calculation at tools/gadi_qin20_compare.py default grid=64, same saved physical parameters and node sequence; not a grid64 run result",
        frozen_cost_fixture={"source": str(benchmark_path), "sha256": hash_file(benchmark_path),
            "scope": bench["scope"], "grid": bench["grid"], "nodes": coarse,
            "intervals": [interval_row(a, b, cosmo, bench["box_mpc"], bench["grid"], radius)
                          for a, b in zip(coarse[:-1], coarse[1:], strict=True)]},
        conclusion="A search distance or one solver call is not a physical time step. Ratios alone neither validate nor rule out infinite-c for Qin20: need source/opacity evolution and finite-c convergence on matched histories.")


def render_report(report):
    c = report["counterexamples"]
    native = report["native_time_scales"]
    lines = [
        "# Time, finite range, and persistent radiation state audit", "",
        "This audit does not change the event solver. Its constant-rate, infinite-c, no-recombination model can close every photon ledger while disagreeing with causal transport. The counterexamples below are expected limitations, not failed numerical assertions.", "",
        "## What retained currently means", "",
        "`event_front.cpp::advance` adds `ray.rate * (duration-now)` to a source-indexed ledger when a geometric search hits `maximum_distance`. The Python API returns that count and no ray positions, directions, emission/arrival times or traveled-distance state. Consequently `source_retained` is unresolved flux under a search cutoff, not a resumable finite-c photon population. Within the declared infinite-c model, total counts remain accounted for. Silently discarding this ledger loses photons; calling it absorbed is also incorrect.", "",
        "## Exact counterexamples", "",
        "| Probe | Current/proposed restart | Independent answer |",
        "|---|---:|---:|",
        "| Unit source, target 4.5 cells away, c=2, duration=1 | 1 absorbed | 0 absorbed; 1 in flight |",
        "| Same source, duration=3 | 3 absorbed | 0.75 absorbed; 2.25 in flight |",
        "| Set cutoff=c*dt=1, target 0.5 cell away | 1 absorbed | 0.5 absorbed; 0.5 in flight |",
        "| Source emits 1 photon over t=0..1; restart its retained count at origin, cutoff=2 | After three intervals: 0 absorbed; 1 retained | At c=2, the same emission gives 0.75 absorbed by t=3 |",
        "| Re-isotropize 0.5 carried photons known to move +x | 0.25 spuriously goes to +y | 0 to +y |",
        "| Source emits at t=0..1; target delay=2.25; interval t=2..3 | Zero-source restart adds 0 | In-flight radiation adds 0.75 |",
        "| Move a packet but reset its total range budget each step (Rmax=3) | Absorbs at distance 4.5 | Must remain unresolved at distance 3 |",
        "| Feed cumulative source budgets 3 then 5 as fresh counts | Total absorption 8 | Correct incremental total 5 |", "",
        "The source-restart examples test hypothetical adapters; the existing prototype does not implement such an adapter. Reinjected carry must not be counted as new physical emission. Ledger closure by itself cannot detect the direction error or early-arrival error. A separate one-dimensional delta-flash state oracle confirms exact split-step continuation: the entire flash reaches distance 4.5 at t=2.25 for c=2. This flash is not represented as an event_front constant-rate input.", "",
        f"A valid control splits the same infinite-c fixed-rate step in half, retaining gas capacity and unchanged directions: allocation L1 difference = {c['infinite_c_no_retained_restart_is_valid']['split_vs_whole_l1']:.3g}. This control does not justify finite-c restarts or reuse of cumulative source counts.", "",
        "For a uniform static medium with no recombinations, the independent relativistic sharp-front solution obeys `Qdot*(t-R/c) = 4*pi*n_H*R^3/3`. It follows by integrating the relativistic jump condition in [Iliev et al. 2006, §2.10, eq.3](https://arxiv.org/abs/astro-ph/0603199), following [Shapiro et al. 2006](https://arxiv.org/abs/astro-ph/0507677). For dimensionless Qdot=c=1 and n_H=3/(4*pi), t=1 gives R=0.68233, whereas infinite-c gives R=1: a 46.6% radius excess. This deliberately relativistic case is a falsification test, not an estimate of Qin20 error.", "",
        "## Qin20 clock and scale audit", "",
        f"Read the actual saved native input `{native['inputs_path']}` ({native['node_count']} nodes, saved {native['saved_grid']}³ grid, box {native['saved_box_cMpc']:g} cMpc). It uses h={native['cosmology']['hlittle']}, Ωm={native['cosmology']['OMm']}, and ZPRIME_STEP_FACTOR={native['nominal_ZPRIME_STEP_FACTOR']}. Main elapsed times/horizons below integrate native `hubble(z)`, which includes radiation and sets native SFR timescales. ΩΛ uses the actual Python property 1−Ωm; constants match native Constants.c/h. The default 64³ grid changes the cell scale only. These are calculated scales, not a new simulation.", "",
        "The exact comoving distance light travels between nodes is `c/H0 * integral(dz/E(z))`; proper light crossing estimates use `L_com/(1+z_late)/c`. A fixed search radius, R_BUBBLE_MAX, and the time-dependent causal horizon have different meanings. In particular R_BUBBLE_MAX is an ES cutoff parameter, not proof that photons propagate that full distance.", "",
        "| Late z | Node dt (Myr) | Light horizon (cMpc) | 64³ cell crossing (Myr) | 50 cMpc crossing (Myr) | 50 cMpc / horizon | Default event diagonal / horizon |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in native["selected_default_runner_grid64_intervals"]:
        s = row["scales"]
        lines.append(f"| {row['z_late']:.3f} | {row['dt_Myr']:.3f} | {row['true_light_horizon_cMpc']:.3f} | {s['cell']['light_crossing_at_z_late_Myr']:.3f} | {s['R_BUBBLE_MAX']['light_crossing_at_z_late_Myr']:.3f} | {s['R_BUBBLE_MAX']['distance_over_interval_light_horizon']:.3f} | {s['event_default_diagonal']['distance_over_interval_light_horizon']:.3f} |")
    last = native["frozen_cost_fixture"]["intervals"][-1]
    lines += ["", f"The 32³ frozen cost fixture instead uses nodes 35→12→8→7.5. Its last interval spans {last['dt_Myr']:.3f} Myr with the native hubble clock, with light horizon {last['true_light_horizon_cMpc']:.3f} cMpc. Native `dtdz` omits radiation: integrating that separate expression gives {last['native_dtdz_integrated_Myr']:.3f} Myr. Native recombinations actually use the endpoint approximation `abs(dtdz(z_late))*dz`, giving {last['native_recombination_endpoint_dt_Myr']:.3f} Myr. These are distinct definitions, not quadrature error or different Astropy parameters. The candidate integration must explicitly choose and consistently apply a clock; this audit changes none of them.", "",
        "The frozen source grid is cumulative and is reinterpreted as rates for duration=1 in the microbenchmark; it has no conversion to a physical duration. Its speed measurements therefore do not validate temporal evolution.", "",
        "The default 50 cMpc crossing time is comparable to a native evolution interval; the event default diagonal, sqrt(3)*200 cMpc, is several intervals. This prevents claiming a causal approximation is automatically negligible. It also does not prove large physical errors: relevant absorption distances, source variability, and front speeds can be much smaller. Quasistatic validity requires (path/c)/t_source ≪1, (path/c)/t_opacity ≪1, and front speeds ≪c on the paths that carry substantial flux. Those histories and distances have not yet been measured for the candidate. One frozen field cannot supply them.", "",
        "## Minimum correct state and acceptance gates", "",
        "A finite-c continuous-beam extension can retain piecewise-constant rate segments instead of individual photons. It needs their original emission intervals, current positions and directions, current time or delayed arrival intervals, spectral group, integrated remaining counts, and cumulative traveled distance (including periodic wraps) if an absolute range limit is used. Source attribution is optional for physics but required for a per-source ledger. Gas ionized stock and recombination/thermal history must persist, with fresh source increments distinct from stored photons. Arrival, source-on/source-off, and cell-exhaustion events all change incident rates; only an exhaustion queue is insufficient.", "",
        "An unresolved numerical cutoff is neither physical absorption nor boundary escape. Store the stopped trajectory and remaining time profile, or reject a run when its unresolved fraction exceeds a declared error tolerance. A periodic ray is not escaped merely because it wraps. Empty-space certificates need invalidation when recombinations or changing density restore opacity.", "",
        "Required acceptance tests (not yet passed by an integrated candidate):", "",
        "1. Exact transparent-path step/flash arrival and source-off continuation, including split intervals and checkpoint/resume; total emitted = absorbed + escaped + in-flight + explicitly unresolved, without recounting carry.",
        "2. Preserve direction/subcell position, periodic path length, and absolute range across resume; the +x/+y and range-reset counterexamples must vanish.",
        "3. Recover the relativistic uniform-front radius and the nonrelativistic limit; check no absorption outside the causal light cone, and converge interval size and angular/grid resolution.",
        "4. Derive nonnegative native physical source increments with a validated spatial closure; prevent repeated spending of cumulative emission, and include previous ionization plus recombination sinks exactly once.",
        "5. Evolve matched Qin20 histories with finite c and infinite c, then compare xHI, source/atom ledger, ionization times and P21 at matched z and matched xHI. Report actual source/opacity variability and distance distributions. A chosen science tolerance (for example 1% mean xHI and 5% P21 over stated resolved k bands) is a project acceptance criterion, not a theorem or literature guarantee.", "",
        "[FlexRT, §3.1](https://arxiv.org/html/2409.04521v2) gives primary precedent for preserving packet positions/directions between light-speed steps; its §5.4 demonstrates why finite-c versus infinite-c can change early front positions. Its method is not identical to this candidate.", "",
        f"Generated from event_front.py SHA256 `{report['provenance']['event_front_py_sha256']}` and event_front.cpp SHA256 `{report['provenance']['event_front_cpp_sha256']}`. JSON records the numerical counterexamples, input hashes, and scale calculations. No production code was modified.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--json", type=Path, default=HERE / "validation_time_state.json")
    parser.add_argument("--markdown", type=Path, default=HERE / "validation_time_state.md")
    args = parser.parse_args()
    report = dict(scope="Independent exact time/state counterexamples; no production changes or new Qin20 evolution",
        provenance={"event_front_py_sha256": hash_file(HERE / "event_front.py"),
                    "event_front_cpp_sha256": hash_file(HERE / "event_front.cpp")},
        counterexamples=counterexamples(), native_time_scales=native_times(args.inputs))
    args.json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    args.markdown.write_text(render_report(report))
    print(json.dumps({"json": str(args.json), "markdown": str(args.markdown),
                      "scope": report["scope"]}))


if __name__ == "__main__":
    main()
