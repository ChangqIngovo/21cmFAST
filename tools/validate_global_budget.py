#!/usr/bin/env python3
"""Run an independent, paired full-Qin20 native ES/global-budget validation.

The IC arrays are generated once and copied into both arms. Every arm then
follows its own source, X-ray, recombination and feedback history. This is a
small-box integration test, not a spatial or time-resolution convergence test.
"""

from __future__ import annotations

# ruff: noqa: T201 -- this validation command reports its independent audits.
import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def digest_arrays(arrays):
    """Hash named arrays including their shapes and precision."""
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        value = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def copy_initial_conditions(values, inputs):
    """Construct an independent initial-condition object from frozen arrays."""
    from py21cmfast.wrapper.outputs import InitialConditions

    copied = InitialConditions.new(inputs=inputs)
    for name, value in values.items():
        copied.set(name, value.copy())
    return copied


def box_digest(box):
    """Fingerprint the prior state to detect unintended mutation."""
    arrays = {
        name: box.get(name)
        for name, array in box.arrays.items()
        if array.state.is_computed
    }
    # Scalar source history is also an input to the following step.
    h = hashlib.sha256(digest_arrays(arrays).encode())
    for name in ("nion_unconditional_acg", "nion_unconditional_mcg"):
        h.update(str((name, getattr(box, name))).encode())
    return h.hexdigest()


def copy_output(box, inputs):
    """Copy frozen numerical inputs while changing only execution settings."""
    kwargs = {"redshift": box.redshift} if hasattr(box, "redshift") else {}
    copied = type(box).new(inputs=inputs, **kwargs)
    for name, array in box.arrays.items():
        if array.state.is_computed:
            copied.set(name, box.get(name).copy())
    for name in box._struct.primitive_fields:
        copied.set(name, getattr(box, name))
    return copied


def save_frozen_step(coeval, previous, frozen_initial, directory):
    """Persist every native input needed to repeat the final ionization step."""
    from py21cmfast.io.h5 import write_output_to_hdf5

    directory.mkdir(parents=True, exist_ok=False)
    objects = {
        "initial": copy_initial_conditions(frozen_initial, coeval.inputs),
        "density": coeval.perturbed_field,
        "spin": coeval.ts_box,
        "previous_density": previous.perturbed_field,
        "previous_ionized": previous.ionized_box,
    }
    for name, box in objects.items():
        write_output_to_hdf5(box, directory / f"{name}.h5")


def load_frozen_step(directory):
    """Reload the reproducible input fixture without evolving history again."""
    from py21cmfast.io.h5 import read_output_struct

    objects = {}
    for name in ("initial", "density", "spin", "previous_density", "previous_ionized"):
        box = read_output_struct(directory / f"{name}.h5")
        box.load_all()
        objects[name] = box
    coeval = SimpleNamespace(
        inputs=objects["density"].inputs,
        redshift=objects["density"].redshift,
        perturbed_field=objects["density"],
        ts_box=objects["spin"],
    )
    previous = SimpleNamespace(
        perturbed_field=objects["previous_density"],
        ionized_box=objects["previous_ionized"],
    )
    frozen = {
        name: objects["initial"].get(name).copy()
        for name, array in objects["initial"].arrays.items()
        if array.state.is_computed
    }
    return coeval, previous, frozen


def compare_threads(coeval, previous, frozen_initial):
    """Repeat the active final ionization step with identical fields and 1/4 threads."""
    from py21cmfast.drivers.single_field import compute_ionization_field

    results = {}
    for threads in (1, 4):
        inputs = coeval.inputs.evolve_input_structs(N_THREADS=threads)
        previous_ionized = copy_output(previous.ionized_box, inputs)
        previous_density = copy_output(previous.perturbed_field, inputs)
        density = copy_output(coeval.perturbed_field, inputs)
        spin = copy_output(coeval.ts_box, inputs)
        before = box_digest(previous_ionized)
        box = compute_ionization_field(
            inputs=inputs,
            initial_conditions=copy_initial_conditions(frozen_initial, inputs),
            perturbed_field=density,
            previous_perturbed_field=previous_density,
            previous_ionized_box=previous_ionized,
            spin_temp=spin,
            regenerate=True,
            write=False,
        )
        assert box_digest(previous_ionized) == before, (
            "Thread comparison mutated prior state"
        )
        proxy = SimpleNamespace(
            inputs=inputs,
            redshift=coeval.redshift,
            perturbed_field=density,
            ts_box=spin,
            ionized_box=box,
        )
        results[threads] = (
            box,
            independently_audit(
                proxy,
                np.sum(
                    previous_ionized.get("cumulative_recombinations"),
                    dtype=np.longdouble,
                ),
            ),
        )
    differences = {}
    for name in (
        "neutral_fraction",
        "ionisation_rate_G12",
        "cumulative_recombinations",
        "mean_free_path",
        "z_reion",
        "kinetic_temperature",
    ):
        one = results[1][0].get(name)
        four = results[4][0].get(name)
        differences[name] = float(np.max(np.abs(one.astype(float) - four)))
        assert np.allclose(one, four, rtol=2e-5, atol=2e-6), (name, differences[name])
    assert results[1][1]["recombinations_from_array"] > 0, "Inactive recombination test"
    return {
        "scope": "same physical current/prior fields, independent native ionization calls",
        "threads": [1, 4],
        "maximum_absolute_field_differences": differences,
        "audits": {str(n): value[1] for n, value in results.items()},
    }


def independently_audit(coeval, previous_recombinations=None):
    """Recompute the declared model ledger from output fields, not its residual."""
    ionized = coeval.ionized_box
    rho = 1 + np.asarray(coeval.perturbed_field.get("density"), dtype=np.longdouble)
    xhi = np.asarray(ionized.get("neutral_fraction"), dtype=np.longdouble)
    xe = np.asarray(coeval.ts_box.get("xray_ionised_fraction"), dtype=np.longdouble)
    recomb = np.asarray(ionized.get("cumulative_recombinations"), dtype=np.longdouble)
    gamma = np.asarray(ionized.get("ionisation_rate_G12"), dtype=np.longdouble)
    if not all(np.isfinite(x).all() for x in (rho, xhi, xe, recomb, gamma)):
        raise AssertionError("Non-finite native output")
    if np.any(rho < 0) or np.any(xhi < 0) or np.any(xhi > 1):
        raise AssertionError("Invalid density or neutral fraction")
    q = np.longdouble(xhi.size) * (
        np.longdouble(ionized.nion_unconditional_acg)
        + np.longdouble(ionized.nion_unconditional_mcg)
    )
    uv = np.sum(rho * np.maximum(1 - xhi - xe, 0), dtype=np.longdouble)
    ionized_total = np.sum(rho * (1 - xhi), dtype=np.longdouble)
    xray_stock = np.sum(rho * xe, dtype=np.longdouble)
    rec = np.sum(recomb, dtype=np.longdouble)
    remaining = np.longdouble(ionized.pc_remaining_photons)
    residual = q - uv - rec - remaining
    row = {
        "redshift": float(coeval.redshift),
        "volume_xHI": float(xhi.mean()),
        "mass_xHI": float(np.sum(rho * xhi) / np.sum(rho)),
        "photons_from_unconditional_source": float(q),
        "uv_ionized_stock_from_arrays": float(uv),
        "recombinations_from_array": float(rec),
        "xray_stock_from_arrays": float(xray_stock),
        "Gamma12_mean": float(gamma.mean()),
        "pending_photons": float(remaining),
        "independent_residual": float(residual),
        "pc_status": int(ionized.pc_status),
        "pc_evaluations": int(ionized.pc_evaluations),
        "declared_tolerance": float(ionized.pc_count_tolerance),
    }
    if coeval.inputs.astro_options.PHOTON_CONS_TYPE == "global-budget":
        tol = np.longdouble(ionized.pc_count_tolerance)
        scalar_roundoff = 128 * np.finfo(np.float64).eps * max(float(q), float(rec), 1)
        assert remaining >= 0, row
        assert row["pc_status"] in (1, 2, 3), row
        assert abs(residual) <= tol + scalar_roundoff, row
        assert abs(q - ionized.pc_photons) <= scalar_roundoff, row
        assert abs(uv - ionized.pc_ionized_atoms) <= scalar_roundoff, row
        assert abs(rec - ionized.pc_recombinations) <= scalar_roundoff, row
        assert abs(ionized_total - ionized.pc_ionized_atoms_total) <= scalar_roundoff, (
            row
        )
        assert abs(xray_stock - ionized.pc_xray_ionized_atoms) <= scalar_roundoff, row
        assert abs(residual - ionized.pc_budget_residual) <= scalar_roundoff, row
        if previous_recombinations is not None:
            assert (
                abs(previous_recombinations - ionized.pc_recombination_previous)
                <= scalar_roundoff
            ), row
        row["pending_fraction"] = float(remaining / q) if q > 0 else 0.0
        row["absorption_complete_within_tolerance"] = bool(remaining <= tol)
        row["uv_per_cell_below_float32_epsilon"] = bool(
            q / xhi.size < np.finfo(np.float32).eps
        )
        row["xray_baseline_roundoff"] = float(ionized.pc_xray_baseline_roundoff)
    return row


def pending_summary(records):
    """Distinguish ledger closure from actually absorbed source photons."""
    groups = {
        "all_nodes": records,
        "early_nodes_z_ge_25": [row for row in records if row["redshift"] >= 25],
        "underresolved_status_3": [row for row in records if row["pc_status"] == 3],
        "mean_UV_below_float32_epsilon": [
            row for row in records if row["uv_per_cell_below_float32_epsilon"]
        ],
        "mean_UV_above_float32_epsilon": [
            row for row in records if not row["uv_per_cell_below_float32_epsilon"]
        ],
        "late_nodes_z_le_12": [row for row in records if row["redshift"] <= 12],
    }
    result = {}
    for name, rows in groups.items():
        result[name] = {
            "nodes": len(rows),
            "maximum_pending_fraction": max(
                (row["pending_fraction"] for row in rows), default=None
            ),
            "status_3_nodes": sum(row["pc_status"] == 3 for row in rows),
            "all_absorption_complete_within_tolerance": all(
                row["absorption_complete_within_tolerance"] for row in rows
            )
            if rows
            else None,
        }
    return result


def main():
    """Run and audit both Qin20 evolutions with the requested numerical setup."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--box-mpc", type=float, default=200)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--redshift", type=float, default=8)
    parser.add_argument(
        "--node-redshifts",
        type=float,
        nargs="+",
        help="Optional numerical node grid for a separate coarse integration smoke test",
    )
    parser.add_argument(
        "--frozen-step",
        type=Path,
        help="Replay a saved native fixture at one and four threads without evolution",
    )
    parser.add_argument("--output", type=Path, default=Path("global_budget_validation"))
    args = parser.parse_args()
    if args.frozen_step is not None:
        args.output.mkdir(parents=True, exist_ok=False)
        result = compare_threads(*load_frozen_step(args.frozen_step))
        (args.output / "thread_comparison.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        (args.output / "COMPLETED").touch()
        print(json.dumps(result, indent=2))
        return
    if args.grid < 2 or args.threads < 1 or args.box_mpc < 150:
        parser.error("Require grid>=2, threads>=1, BOX_LEN>=150 for unchanged Qin20")

    from py21cmfast._templates import write_template
    from py21cmfast.drivers.coeval import generate_coeval
    from py21cmfast.drivers.single_field import compute_initial_conditions
    from py21cmfast.io.caching import OutputCache
    from py21cmfast.wrapper.inputs import InputParameters

    args.output.mkdir(parents=True, exist_ok=False)
    base = InputParameters.from_template(
        "Qin20",
        random_seed=args.seed,
        HII_DIM=args.grid,
        DIM=3 * args.grid,
        HIRES_TO_LOWRES_FACTOR=None,
        BOX_LEN=args.box_mpc,
        N_THREADS=args.threads,
        PHOTON_CONS_TYPE="no-photoncons",
    )
    if args.node_redshifts is not None:
        base = base.clone(node_redshifts=tuple(args.node_redshifts))
    pc = base.evolve_input_structs(PHOTON_CONS_TYPE="global-budget")
    options = [
        p.asdict(include_cosmo_tables="never", only_cstruct_params=False)
        for p in (base, pc)
    ]
    for d in options:
        d["astro_options"].pop("PHOTON_CONS_TYPE")
    assert options[0] == options[1], "The paired physical configurations differ"
    initial = compute_initial_conditions(inputs=base, regenerate=True, write=False)
    frozen = {
        name: initial.get(name).copy()
        for name, array in initial.arrays.items()
        if array.state.is_computed
    }
    ic_digest = digest_arrays(frozen)
    selected = tuple(
        sorted(
            {z for z in base.node_redshifts if z >= args.redshift} | {args.redshift},
            reverse=True,
        )
    )
    reports = {}
    thread_comparison = None
    for name, inputs in (("es", base), ("pc", pc)):
        write_template(inputs, args.output / f"{name}_inputs.toml", mode="full")
        copied = copy_initial_conditions(frozen, inputs)
        assert digest_arrays({n: copied.get(n) for n in frozen}) == ic_digest
        records = []
        previous_box = None
        previous_digest = None
        previous_rec = None
        previous_coeval = None
        started = time.monotonic()
        for coeval, _ in generate_coeval(
            inputs=inputs,
            initial_conditions=copied,
            out_redshifts=selected,
            regenerate=True,
            write=False,
            cache=OutputCache(args.output / f"{name}_cache"),
            progressbar=False,
        ):
            if previous_box is not None:
                assert box_digest(previous_box) == previous_digest, (
                    "Previous ionization state changed during the next step"
                )
            row = independently_audit(coeval, previous_rec)
            row["elapsed_seconds"] = time.monotonic() - started
            records.append(row)
            print(json.dumps({"arm": name, **row}), flush=True)
            if name == "pc" and coeval.redshift == args.redshift:
                save_frozen_step(
                    coeval, previous_coeval, frozen, args.output / "frozen_native_step"
                )
                thread_comparison = compare_threads(coeval, previous_coeval, frozen)
            previous_coeval = coeval
            previous_box = coeval.ionized_box
            previous_digest = box_digest(previous_box)
            previous_rec = row["recombinations_from_array"]
            if coeval.redshift == args.redshift:
                np.savez_compressed(
                    args.output / f"{name}_final.npz",
                    density=coeval.perturbed_field.get("density"),
                    xHI=coeval.ionized_box.get("neutral_fraction"),
                    xray_ionised_fraction=coeval.ts_box.get("xray_ionised_fraction"),
                    cumulative_recombinations=coeval.ionized_box.get(
                        "cumulative_recombinations"
                    ),
                    Gamma12=coeval.ionized_box.get("ionisation_rate_G12"),
                    brightness_temperature=coeval.brightness_temperature.get(
                        "brightness_temp"
                    ),
                )
            reports[name] = {
                "records": records,
                "elapsed_seconds": row["elapsed_seconds"],
            }
            if name == "pc":
                reports[name]["pending_summary"] = pending_summary(records)
            (args.output / "report.json").write_text(
                json.dumps(
                    {
                        "scope": "paired full-Qin20 integration; frozen IC arrays, separately evolved histories",
                        "initial_conditions_sha256": ic_digest,
                        "only_physical_option_difference": "PHOTON_CONS_TYPE",
                        "grid": args.grid,
                        "box_mpc": args.box_mpc,
                        "threads": args.threads,
                        "custom_node_redshifts": args.node_redshifts,
                        "source_sink_ledger": "declared UV effective model; excludes current X-ray/electron baseline and charges current native recombinations",
                        "arms": reports,
                        "frozen_step_thread_comparison": thread_comparison,
                    },
                    indent=2,
                )
                + "\n"
            )
        assert digest_arrays(frozen) == ic_digest
    (args.output / "COMPLETED").touch()


if __name__ == "__main__":
    main()
