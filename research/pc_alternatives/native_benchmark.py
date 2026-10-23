#!/usr/bin/env python3
"""Prepare a small native Qin20 fixture, then separately time frozen native steps.

Preparation evolves a deliberately coarse numerical history. Measurement reuses
identical frozen arrays for no-photoncons and global-budget, 4 threads by default.
Neither a coarse fixture nor a frozen one-step timing is a full science run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def helpers():
    spec = importlib.util.spec_from_file_location("frozen_native_helpers", ROOT / "tools/validate_global_budget.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clean(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, default=clean) + "\n")


def physical_differences(template, selected):
    one = template.asdict(include_cosmo_tables="never", only_cstruct_params=False)
    two = selected.asdict(include_cosmo_tables="never", only_cstruct_params=False)
    differences = {}
    for group in ("astro_options", "astro_params", "matter_options", "cosmo_params"):
        differences[group] = {
            key: {"template": one[group].get(key), "selected": value}
            for key, value in two[group].items()
            if value != one[group].get(key)
        }
    return differences


def prepare(args):
    from py21cmfast._templates import write_template
    from py21cmfast.drivers.coeval import generate_coeval
    from py21cmfast.drivers.single_field import compute_initial_conditions
    from py21cmfast.io.caching import OutputCache
    from py21cmfast.wrapper.inputs import InputParameters

    h = helpers()
    args.directory.mkdir(parents=True, exist_ok=False)
    template = InputParameters.from_template("Qin20", random_seed=args.seed)
    inputs = InputParameters.from_template(
        "Qin20", random_seed=args.seed, HII_DIM=args.grid, DIM=3 * args.grid,
        HIRES_TO_LOWRES_FACTOR=None, BOX_LEN=args.box_mpc, N_THREADS=args.threads,
        PHOTON_CONS_TYPE="no-photoncons",
    ).clone(node_redshifts=tuple(args.nodes))
    differences = physical_differences(template, inputs)
    for group, changed in differences.items():
        assert not (set(changed) - ({"PHOTON_CONS_TYPE"} if group == "astro_options" else set())), changed
    write_template(inputs, args.directory / "inputs.toml", mode="full")
    info = {
        "scope": "Numerical coarse-history fixture for frozen native-cost comparison, not converged science validation",
        "grid": args.grid, "dim": 3 * args.grid, "box_mpc": args.box_mpc,
        "seed": args.seed, "threads": args.threads, "nodes": args.nodes,
        "physical_differences_from_template": differences,
        "source_model": inputs.matter_options.SOURCE_MODEL,
        "power_spectrum": inputs.matter_options.POWER_SPECTRUM,
        "streaming_velocity_model": inputs.matter_options.V_CB_MODEL,
        "progress": [],
    }
    write(args.directory / "preparation.json", info)
    initial = compute_initial_conditions(inputs=inputs, regenerate=True, write=False)
    frozen = {
        name: initial.get(name).copy()
        for name, array in initial.arrays.items() if array.state.is_computed
    }
    info["initial_conditions_sha256"] = h.digest_arrays(frozen)
    previous = None
    started = time.monotonic()
    for coeval, _ in generate_coeval(
        inputs=inputs, initial_conditions=initial, out_redshifts=tuple(args.nodes),
        regenerate=True, write=False, cache=OutputCache(args.directory / "cache"),
        progressbar=False,
    ):
        row = {"redshift": float(coeval.redshift), "preparation_elapsed_seconds": time.monotonic() - started,
               "mean_xHI": float(coeval.ionized_box.get("neutral_fraction").mean())}
        info["progress"].append(row)
        write(args.directory / "preparation.json", info)
        print(json.dumps(row), flush=True)
        if coeval.redshift == min(args.nodes):
            assert previous is not None
            h.save_frozen_step(coeval, previous, frozen, args.directory / "frozen_native_step")
            density = 1 + coeval.perturbed_field.get("density").astype(float)
            xe = coeval.ts_box.get("xray_ionised_fraction").astype(float)
            source = np.zeros_like(density)
            species = {}
            for name in ("acg", "mcg"):
                values = coeval.ionized_box.get(f"nion_conditional_filtered_{name}")[0].astype(float)
                target = float(getattr(coeval.ionized_box, f"nion_unconditional_{name}")) * density.size
                raw = density * values
                if float(raw.min()) < 0 or float(raw.sum()) <= 0:
                    raise RuntimeError("Native cumulative source cannot define a positive packet closure")
                contribution = raw * target / raw.sum()
                species[name] = contribution
                source += contribution
            np.savez_compressed(args.directory / "source_capacity.npz", source=source,
                capacity=density * (1 - xe), density=density, xray_fraction=xe,
                source_acg=species["acg"], source_mcg=species["mcg"],
                recombinations_previous=previous.ionized_box.get("cumulative_recombinations"),
                native_xHI=coeval.ionized_box.get("neutral_fraction"))
            info["source_closure"] = "Cumulative cell conditional E-INTEGRAL source, density weighted and separately normalized to each species unconditional budget; NEW spatial closure"
            info["source_sum"] = float(source.sum())
            info["neutral_capacity_sum"] = float(np.sum(density * (1 - xe)))
            write(args.directory / "preparation.json", info)
        previous = coeval
    assert (args.directory / "frozen_native_step/initial.h5").is_file()
    (args.directory / "PREPARED").touch()


def measure(args):
    from py21cmfast.drivers.single_field import compute_ionization_field

    h = helpers()
    assert (args.directory / "PREPARED").is_file()
    coeval, previous, initial = h.load_frozen_step(args.directory / "frozen_native_step")
    inputs_by_mode = {
        mode: coeval.inputs.evolve_input_structs(N_THREADS=args.threads, PHOTON_CONS_TYPE=mode)
        for mode in ("no-photoncons", "global-budget")
    }
    modes = tuple(inputs_by_mode)
    result = {
        "scope": "Same frozen native Qin20 single-ionization-step; copies and fixture I/O excluded; warmup separate",
        "included": ["native ionization wrapper and native setup", "conditional ACG/MCG source integrals and history at all ES radii", "native FFT filtering and ES thresholding", "UV background and native recombination update", "global-budget cached threshold solve when selected"],
        "excluded": ["initial-condition generation", "density and velocity perturbation", "spin-temperature and X-ray evolution", "brightness-temperature calculation", "fixture loading and independent input copies", "source-packet closure extraction and alternative transport"],
        "interpretation": "Frozen operator timing, not total coeval cost. Small sample counts and host noise do not establish a PC speedup.",
        "preparation": json.loads((args.directory / "preparation.json").read_text()),
        "threads": args.threads, "warmup": [], "records": [],
        "history_note": "Both modes receive identical no-PC prior source/ionization history; this is an operator-cost comparison, not separately evolved science histories",
    }

    def once(mode, phase, iteration):
        inputs = inputs_by_mode[mode]
        prev = h.copy_output(previous.ionized_box, inputs)
        kwargs = dict(inputs=inputs,
            initial_conditions=h.copy_initial_conditions(initial, inputs),
            perturbed_field=h.copy_output(coeval.perturbed_field, inputs),
            previous_perturbed_field=h.copy_output(previous.perturbed_field, inputs),
            previous_ionized_box=prev, spin_temp=h.copy_output(coeval.ts_box, inputs),
            regenerate=True, write=False)
        before = h.box_digest(prev)
        start = time.perf_counter()
        box = compute_ionization_field(**kwargs)
        elapsed = time.perf_counter() - start
        assert h.box_digest(prev) == before
        row = {"mode": mode, "iteration": iteration, "elapsed_seconds": elapsed,
               "mean_xHI": float(box.get("neutral_fraction").mean()),
               "pc_evaluations": int(getattr(box, "pc_evaluations", 0)),
               "source_acg": float(box.nion_unconditional_acg),
               "source_mcg": float(box.nion_unconditional_mcg)}
        result[phase].append(row)
        print(json.dumps({"phase": phase, **row}), flush=True)
        write(args.output, result)

    for repetition in range(args.warmup):
        for mode in modes:
            once(mode, "warmup", repetition)
    for repetition in range(args.repeats):
        for mode in modes if repetition % 2 == 0 else modes[::-1]:
            once(mode, "records", repetition)
    result["summary"] = {}
    for mode in modes:
        times = [r["elapsed_seconds"] for r in result["records"] if r["mode"] == mode]
        result["summary"][mode] = {
            "median_seconds": statistics.median(times), "minimum_seconds": min(times),
            "maximum_seconds": max(times), "samples": len(times),
        }
    result["summary"]["pc_over_es_median"] = result["summary"]["global-budget"]["median_seconds"] / result["summary"]["no-photoncons"]["median_seconds"]
    write(args.output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--grid", type=int, default=32)
    prep.add_argument("--box-mpc", type=float, default=200)
    prep.add_argument("--seed", type=int, default=1234)
    prep.add_argument("--threads", type=int, default=4)
    prep.add_argument("--nodes", nargs="+", type=float, default=[35, 12, 8, 7.5])
    prep.add_argument("--directory", type=Path, required=True)
    bench = sub.add_parser("measure")
    bench.add_argument("--directory", type=Path, required=True)
    bench.add_argument("--threads", type=int, default=4)
    bench.add_argument("--warmup", type=int, default=1)
    bench.add_argument("--repeats", type=int, default=2)
    bench.add_argument("--output", type=Path, default=Path(__file__).with_name("native_benchmark.json"))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args)
    else:
        measure(args)


if __name__ == "__main__":
    main()
