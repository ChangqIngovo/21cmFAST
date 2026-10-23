#!/usr/bin/env python3
"""Validate the instantaneous-rate time closure, separately from transport.

Native samples hold z=8 feedback fixed while evaluating the actual 21cmFAST
SFRD and perturbed density at increasing physical times. This measures source
quadrature, not a converged feedback/density/ionization simulation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np
from scipy.optimize import brentq

from source_history import RateHistory, SourceLedger
from validation_native_rates import (
    constants_from_source, helper_module, mean_baryon_density, time_interval, uv_rates,
)


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analytic_checks():
    records = []
    exact = np.expm1(2.) / 2
    previous_error = None
    for steps in (2, 4, 8, 16, 32, 64):
        times = np.linspace(0, 1, steps + 1)
        history = RateHistory.from_knots(times, np.exp(2 * times)[:, None])
        actual = history.integrate(0, 1)[0]
        error = abs(actual / exact - 1)
        records.append(dict(intervals=steps, relative_integral_error=error,
                            observed_order=None if previous_error is None else float(np.log2(previous_error / error))))
        previous_error = error
    rng = np.random.default_rng(1729)
    partition_errors = []
    for _ in range(64):
        times = np.r_[0., np.sort(rng.uniform(0, 1, 8)), 1.]
        left = rng.random((9, 3)) * 1e4
        right = rng.random((9, 3)) * 1e4
        history = RateHistory(times, left, right)
        ledger = SourceLedger(history, np.ones(3))
        for stop in np.r_[np.sort(rng.uniform(0, 1, 17)), 1.]:
            ledger.advance(stop)
        direct = history.integrate(0, 1)
        partition_errors.append(float(np.max(np.abs(ledger.emitted - direct) / direct)))
    # Endpoints alone cannot identify a source switch; it has to be a knot with
    # separate left/right values, or the caller has chosen a smoothed model.
    switched = RateHistory([0., .375, 1.], [[2.], [0.]], [[2.], [0.]])
    ledger = SourceLedger(switched, [1.])
    first = float(ledger.advance(.25)[0])
    restored = SourceLedger.from_checkpoint(switched, [1.], ledger.checkpoint())
    rest = float(restored.advance(1.)[0])
    return dict(exponential_quadrature=records,
                random_piecewise_linear_with_jumps_cases=64,
                maximum_partition_relative_error=max(partition_errors),
                source_switch=dict(turnoff_time=.375, rate=2., exact_emission=.75,
                                   before_checkpoint=first, fresh_after_checkpoint=rest,
                                   final_emission=float(restored.emitted[0])),
                interpretation="Second-order quadrature for a smooth source; subdivision exact within the declared interpolation up to rounding. No bound on unsampled source variability.")


def extract_native(fixture, old_cache, cache, intervals):
    from py21cmfast.drivers.single_field import compute_emissivity_fields, perturb_field
    from py21cmfast.io.h5 import read_output_struct

    helpers = helper_module()
    current, previous, frozen = helpers.load_frozen_step(fixture)
    inputs = current.inputs.evolve_input_structs(N_THREADS=4)
    old_spin = read_output_struct(old_cache / "spin_previous.h5")
    old_spin.load_all()
    old_spin = helpers.copy_output(old_spin, inputs)
    old_ion = helpers.copy_output(previous.ionized_box, inputs)
    params = inputs.asdict(include_cosmo_tables="never", only_cstruct_params=False)
    cosmo, astro = params["cosmo_params"], params["astro_params"]
    rho_b = mean_baryon_density(cosmo, constants_from_source())
    start_redshift = float(previous.ionized_box.redshift) - 1e-6
    end_redshift = float(current.redshift)
    clock = time_interval(start_redshift, end_redshift, cosmo)
    duration = clock["integrated_seconds"]
    times = np.linspace(0, duration, intervals + 1)
    redshifts, acg, mcg = [], [], []
    before = {str(p): digest(p) for p in [*fixture.glob("*.h5"), old_cache / "spin_previous.h5"]}
    started = time.perf_counter()
    for index, t in enumerate(times):
        if index == 0:
            z = start_redshift
        elif index == intervals:
            z = end_redshift
        else:
            z = brentq(lambda trial: time_interval(start_redshift, trial, cosmo)["integrated_seconds"] - t,
                       end_redshift, start_redshift - 1e-12, xtol=1e-12)
        density = perturb_field(redshift=z, inputs=inputs,
            initial_conditions=helpers.copy_initial_conditions(frozen, inputs),
            regenerate=True, write=False)
        emission = compute_emissivity_fields(redshift=z, inputs=inputs,
            initial_conditions=helpers.copy_initial_conditions(frozen, inputs),
            perturbed_field=density, previous_spin_temp=old_spin,
            previous_ionize_box=old_ion, regenerate=True, write=False)
        rates = uv_rates(emission.get("sfrd_acg"), emission.get("sfrd_mcg"), astro=astro, rho_b=rho_b)
        redshifts.append(float(z))
        acg.append(rates["acg"])
        mcg.append(rates["mcg"])
        print(json.dumps(dict(sample=index, intervals=intervals, z=z,
                              acg_rate=float(rates["acg"].sum()), mcg_rate=float(rates["mcg"].sum()))), flush=True)
    if before != {name: digest(name) for name in before}:
        raise RuntimeError("Frozen native inputs changed during source extraction")
    cache.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(cache / "rate_samples.npz", times=times,
                        redshifts=redshifts, acg=acg, mcg=mcg)
    metadata = dict(intervals=intervals, feedback_redshift=float(previous.ionized_box.redshift),
                    start_redshift=start_redshift, end_redshift=end_redshift,
                    seconds=time.perf_counter() - started, threads=4,
                    clock=clock, fixture_sha256=before,
                    grid=int(inputs.simulation_options.HII_DIM),
                    rate_shape=list(np.asarray(acg).shape),
                    source_unit="effective photons per mean-nucleon cell per second; native SFRD includes density",
                    policy="Hold z8 Ts and ionization feedback for every sample; use actual native perturbed density at each sampled z. No ionization, chemistry or feedback evolution between samples.",
                    model_choice="Instantaneous SFRD integration; do not renormalize to legacy cumulative delta nion",
                    native_source_sha256={name: digest(HERE.parents[1] / "src/py21cmfast/src" / name)
                        for name in ("EmissivityFields.c", "scaling_relations.c", "Constants.c", "Constants.h", "hmf.c")})
    (cache / "extraction.json").write_text(json.dumps(metadata, indent=2) + "\n")


def native_checks(fixture, old_cache, cache, extract):
    if not (cache / "rate_samples.npz").exists():
        if not extract:
            raise RuntimeError("Native time-refined samples absent; use --extract")
        extract_native(fixture, old_cache, cache, intervals=32)
    metadata = json.loads((cache / "extraction.json").read_text())
    with np.load(cache / "rate_samples.npz") as data:
        times, acg, mcg = data["times"], data["acg"], data["mcg"]
    maximum_intervals = len(times) - 1
    if maximum_intervals != 32:
        raise ValueError("Expected 32-interval nested native sample fixture")
    duration = float(times[-1] - times[0])
    combined = acg + mcg
    species = {}
    for name, rates in (("acg", acg), ("mcg", mcg), ("total", combined)):
        fine_history = RateHistory.from_knots(times, rates, clock=metadata["clock"]["definition"])
        fine = fine_history.integrate(0, duration)
        simpson = (duration / maximum_intervals / 3) * (
            rates[0] + rates[-1] + 4 * rates[1:-1:2].sum(axis=0) + 2 * rates[2:-1:2].sum(axis=0))
        rows = []
        for count in (1, 2, 4, 8, 16, 32):
            stride = maximum_intervals // count
            history = RateHistory.from_knots(times[::stride], rates[::stride], clock=fine_history.clock)
            integral = history.integrate(0, duration)
            row = dict(intervals=count, total_emission=float(integral.sum()),
                       minimum_cell_emission=float(integral.min()),
                       relative_total_difference_to_32=float(abs(integral.sum() / fine.sum() - 1)),
                       atom_l1_difference_to_32_over_fine_total=float(np.abs(integral - fine).sum() / fine.sum()),
                       atom_l1_difference_to_simpson32_over_simpson_total=float(np.abs(integral - simpson).sum() / simpson.sum()))
            rows.append(row)
        species[name] = dict(rows=rows, simpson_32_total=float(simpson.sum()),
                            trapezoid_32_total=float(fine.sum()),
                            simpson_vs_trapezoid32_total_fraction=float(simpson.sum() / fine.sum() - 1),
                            final_rule_difference_is_error_estimate_not_bound=True)
    old = json.loads((HERE / "validation_native_rates.json").read_text())
    legacy = old["legacy_cumulative_increment"]
    with h5py.File(fixture / "density.h5") as handle:
        atoms = 1 + handle["PerturbedField/OutputFields/density"][...].astype(float)
    history = RateHistory.from_knots(times, combined, clock=metadata["clock"]["definition"])
    ledger = SourceLedger(history, atoms)
    consumed = []
    for step, t in enumerate(times[1:], start=1):
        consumed.append(ledger.advance(t))
        if step == 13:
            ledger = SourceLedger.from_checkpoint(history, atoms, ledger.checkpoint())
    direct = history.integrate(times[0], times[-1])
    partition_l1 = float(np.abs(ledger.emitted - direct).sum() / direct.sum())
    density_change_rejected = False
    try:
        ledger.advance(times[-1], atoms=atoms * 1.001)
    except ValueError:
        density_change_rejected = True
    return dict(extraction=metadata, sample_file_sha256=digest(cache / "rate_samples.npz"),
                species=species, legacy_cumulative_delta=legacy,
                instantaneous_simpson32_over_legacy_delta=species["total"]["simpson_32_total"] / legacy,
                source_models_differ=True,
                start_offset_note="New integral starts at z=7.999999, the actual left evaluation; older legacy delta is z=8 to 7.5. This 107-year omission is reported, not silently integrated with the z=8 timestamp.",
                persistent_ledger=dict(checkpoint_after_interval=13, final_time_seconds=ledger.time,
                    relative_l1_partition_error=partition_l1,
                    repeat_final_advance_emission=float(ledger.advance(times[-1]).sum()),
                    changed_gas_rejected=density_change_rejected),
                limitations=["Holding feedback fixed isolates source quadrature; it does not establish chronological feedback accuracy",
                    "The finest source sampling and Simpson rule are numerical references, not rigorous bounds on unsampled variability",
                    "SourceLedger freezes transport atoms and rejects density changes; conservative gas advection is still required for evolving density",
                    "Effective stellar-yield photon units are retained; a physical H/He chemistry conversion is not implemented",
                    "The source integral cannot be used as fresh cumulative emission repeatedly; transport must also preserve photons in flight"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=WORKSPACE / "qin20_native_cost_grid32/frozen_native_step")
    parser.add_argument("--old-cache", type=Path, default=WORKSPACE / "validation_native_rates_grid32")
    parser.add_argument("--cache", type=Path, default=WORKSPACE / "validation_source_history_grid32")
    parser.add_argument("--output", type=Path, default=HERE / "validation_source_history.json")
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()
    result = dict(scope="Source time interpolation and fresh-emission ledger; no production replacement or coupled reionization accuracy claim",
                  implementation_sha256=digest(HERE / "source_history.py"),
                  validation_sha256=digest(Path(__file__)),
                  analytic=analytic_checks(),
                  native=native_checks(args.fixture, args.old_cache, args.cache, args.extract))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(analytic=result["analytic"], native_total=result["native"]["species"]["total"],
                         native_ledger=result["native"]["persistent_ledger"]), indent=2))


if __name__ == "__main__":
    main()
