#!/usr/bin/env python3
"""Run the exact Qin20 physics template with and without global-budget PC.

The only physical option changed between the two models is PHOTON_CONS_TYPE.
Numerical grid/box/seed/thread settings are explicit. Both evolutions use one
computed initial-condition realization; their later feedback histories evolve
independently. Brightness-temperature spectra use the native thermal result.
"""

from __future__ import annotations

# ruff: noqa: T201 -- this command-line runner reports progress on stdout.
import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy import fft

PC_MODE = "global-budget"
AUDIT_FIELDS = (
    "pc_photons",
    "pc_ionized_atoms",
    "pc_recombinations",
    "pc_recombination_previous",
    "pc_remaining_photons",
    "pc_budget_residual",
    "pc_amplitude_low",
    "pc_amplitude_high",
    "pc_boundary_weight",
    "pc_count_tolerance",
    "pc_evaluations",
    "pc_status",
    "pc_ionized_atoms_total",
    "pc_xray_ionized_atoms",
    "pc_xray_baseline_roundoff",
)


def sha256_array(array):
    """Hash one contiguous native array without converting its precision."""
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def inputs_for(config, mode):
    """Keep the Qin20 template intact apart from declared numerical settings."""
    from py21cmfast.wrapper.inputs import InputParameters

    inputs = InputParameters.from_template(
        "Qin20",
        random_seed=config["seed"],
        HII_DIM=config["grid"],
        DIM=3 * config["grid"],
        HIRES_TO_LOWRES_FACTOR=None,
        BOX_LEN=config["box_mpc"],
        N_THREADS=config["threads"],
        PHOTON_CONS_TYPE=PC_MODE if mode == "PC" else "no-photoncons",
    )
    return inputs.with_logspaced_redshifts(zmin=config["zmin"])


def check_pair(es, pc):
    """Reject a pair that differs by more than its PC option."""
    a = es.asdict(include_cosmo_tables="never")
    b = pc.asdict(include_cosmo_tables="never")
    b["astro_options"]["PHOTON_CONS_TYPE"] = a["astro_options"]["PHOTON_CONS_TYPE"]
    if a != b:
        raise RuntimeError("PC and ES inputs differ beyond PHOTON_CONS_TYPE")
    expected = {
        "USE_MCGS": True,
        "USE_TS_FLUCT": True,
        "RECOMB_MODEL": "inhomogeneous",
        "CELL_RECOMB": False,
        "HII_FILTER": "sharp-k",
        "USE_EXP_FILTER": False,
        "USE_REIONIZATION_PHOTOHEATING_FEEDBACK": True,
    }
    for key, value in expected.items():
        if getattr(es.astro_options, key) != value:
            raise RuntimeError(f"Qin20 template changed: {key}")
    if es.matter_options.SOURCE_MODEL != "E-INTEGRAL":
        raise RuntimeError("Qin20 requires its Eulerian source model")


def prepare(output, args):
    """Materialize a reviewable input pair before PBS submission."""
    from packaging.version import Version

    from py21cmfast import __version__
    from py21cmfast._templates import write_template

    if Version(__version__).major < 4:
        raise RuntimeError(
            "21cmFAST has missing release-tag version metadata; native HDF5 cache "
            "compatibility requires the correct version. Fetch release tags and rebuild."
        )
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Output directory is nonempty; use a fresh directory")
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "template": "Qin20",
        "pc_mode": PC_MODE,
        "grid": args.grid,
        "box_mpc": args.box_mpc,
        "threads": args.threads,
        "seed": args.seed,
        "zmin": args.zmin,
        "save_redshifts": [15.0, 12.0, 10.0, 8.0, 7.0, 6.0, args.zmin],
    }
    es, pc = (inputs_for(config, mode) for mode in ("ES", "PC"))
    check_pair(es, pc)
    config["node_redshifts"] = list(es.node_redshifts)
    config["versions"] = {
        name: version(name) for name in ("21cmFAST", "numpy", "scipy", "h5py")
    }
    config["native_version"] = __version__
    config["python"] = platform.python_version()
    config["executable"] = os.path.realpath(os.sys.executable)
    config["scope"] = (
        "Qin20 physics with declared numerical validation volume/resolution; not a reproduction of the paper's numerical volume/resolution"
    )
    try:
        config["commit"] = subprocess.check_output(
            [
                "git",
                "-C",
                str(Path(__file__).resolve().parent.parent),
                "rev-parse",
                "HEAD",
            ],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        config["commit"] = None
    write_template(es, output / "Qin20_ES.toml", mode="full")
    write_template(pc, output / "Qin20_PC.toml", mode="full")
    (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: config[k]
                for k in (
                    "template",
                    "pc_mode",
                    "grid",
                    "box_mpc",
                    "threads",
                    "seed",
                    "zmin",
                )
            },
            indent=2,
        ),
        flush=True,
    )
    return config


def power_spectrum(field, box_mpc, threads, bins=24):
    """Periodic, mean-subtracted P(k), with full-FFT mode-count weights."""
    values = np.asarray(field, dtype=np.float64)
    n = values.shape[0]
    if values.shape != (n, n, n):
        raise ValueError("Power spectra require a cubic grid")
    if not np.isfinite(values).all():
        raise FloatingPointError("Power-spectrum field is nonfinite")
    transformed = fft.rfftn(values - values.mean(), workers=threads)
    kxy = 2 * np.pi * fft.fftfreq(n, d=box_mpc / n)
    kz = 2 * np.pi * fft.rfftfreq(n, d=box_mpc / n)
    kmag = np.sqrt(
        kxy[:, None, None] ** 2 + kxy[None, :, None] ** 2 + kz[None, None, :] ** 2
    )
    weight_z = np.full(kz.size, 2.0)
    weight_z[0] = 1
    if n % 2 == 0:
        weight_z[-1] = 1
    weights = np.broadcast_to(weight_z, kmag.shape)
    kmin, knyquist = 2 * np.pi / box_mpc, np.pi * n / box_mpc
    edges = np.geomspace(kmin * (1 - 1e-10), knyquist * (1 + 1e-10), bins + 1)
    selected = (kmag > 0) & (kmag <= knyquist * (1 + 1e-10))
    index = np.digitize(kmag[selected], edges) - 1
    power = abs(transformed[selected]) ** 2 * box_mpc**3 / n**6
    w = weights[selected]
    modes = np.bincount(index, weights=w, minlength=bins)
    kvals = np.bincount(index, weights=w * kmag[selected], minlength=bins)
    ps = np.bincount(index, weights=w * power, minlength=bins)
    keep = modes > 0
    return kvals[keep] / modes[keep], modes[keep].astype(int), ps[keep] / modes[keep]


def row_for(coeval, elapsed, mode):
    """Record native fields and the explicit PC count ledger at every node."""
    neutral = np.asarray(coeval.neutral_fraction, dtype=np.float64)
    density = np.asarray(coeval.density, dtype=np.float64)
    rho = 1 + density
    if (
        np.any(rho < 0)
        or not np.isfinite(rho).all()
        or not np.isfinite(neutral).all()
        or np.any(neutral < 0)
        or np.any(neutral > 1)
    ):
        raise FloatingPointError("Nonphysical or nonfinite native grid")
    atoms = rho.sum(dtype=np.float64)
    ionized = (rho * (1 - neutral)).sum(dtype=np.float64)
    rec = coeval.ionized_box.get("cumulative_recombinations")
    xr = coeval.ts_box.get("xray_ionised_fraction")
    brightness = np.asarray(coeval.brightness_temp)
    if (
        not np.isfinite(rec).all()
        or np.any(rec < 0)
        or not np.isfinite(xr).all()
        or np.any(xr < 0)
        or np.any(xr > 1)
    ):
        raise FloatingPointError("Nonphysical recombination or X-ray field")
    if not np.isfinite(brightness).all():
        raise FloatingPointError("Nonfinite native brightness-temperature grid")
    row = {
        "photon_conservation_mode": PC_MODE if mode == "PC" else "no-photoncons",
        "redshift": float(coeval.redshift),
        "elapsed_seconds": elapsed,
        "cell_count": int(neutral.size),
        "xHI_volume": float(neutral.mean()),
        "xHI_mass": float(np.sum(rho * neutral) / atoms),
        "ionized_atoms": float(ionized),
        "atom_count": float(atoms),
        "cumulative_recombinations": float(np.sum(rec, dtype=np.float64)),
        "xray_ionized_atoms": float(np.sum(rho * xr, dtype=np.float64)),
        "mean_brightness_mK": float(np.mean(brightness)),
        "nion_unconditional_acg": float(coeval.ionized_box.nion_unconditional_acg),
        "nion_unconditional_mcg": float(coeval.ionized_box.nion_unconditional_mcg),
    }
    for key in AUDIT_FIELDS:
        value = getattr(coeval.ionized_box, key, None)
        if value is None:
            raise RuntimeError(f"Installed branch lacks native PC audit field {key}")
        row[key] = float(value)
    row["pc_absorption_margin"] = (
        row["pc_photons"] - row["pc_ionized_atoms"] - row["pc_recombinations"]
    )
    row["grid_uv_ionized_atoms"] = float(
        np.sum(rho * np.maximum(1 - neutral - xr, 0), dtype=np.float64)
    )
    row["model_uv_photons"] = neutral.size * (
        row["nion_unconditional_acg"] + row["nion_unconditional_mcg"]
    )
    row["grid_uv_budget_margin"] = (
        row["model_uv_photons"]
        - row["grid_uv_ionized_atoms"]
        - row["cumulative_recombinations"]
    )
    row["grid_uv_closed_residual"] = (
        row["grid_uv_budget_margin"] - row["pc_remaining_photons"]
    )
    return row


def validate_pc_row(row):
    """Independently require nonnegative spending and closure of the saved grids."""
    if row["photon_conservation_mode"] != PC_MODE:
        raise RuntimeError("PC history is missing its global-budget method identity")
    if not all(
        np.isfinite(value)
        for key, value in row.items()
        if key != "photon_conservation_mode"
    ):
        raise FloatingPointError("Nonfinite PC history diagnostic")
    if row["pc_status"] not in (1, 2, 3):
        raise RuntimeError("PC run did not execute an enabled global-budget state")
    for key in (
        "pc_photons",
        "pc_ionized_atoms",
        "pc_recombinations",
        "pc_remaining_photons",
        "pc_count_tolerance",
    ):
        if row[key] < 0:
            raise FloatingPointError(f"Negative PC diagnostic: {key}")
    # A boundary's float32 absorption tolerance is not an allowance for ledger
    # reconstruction error. These are sums of the exact same saved numbers.
    # The N*eps bound also covers each thread's ordinary double summation.
    summation_factor = 16 * max(row["cell_count"], 1) * np.finfo(float).eps
    scale = max(
        row["pc_photons"],
        row["grid_uv_ionized_atoms"] + row["cumulative_recombinations"],
        row["pc_remaining_photons"],
        np.finfo(float).tiny,
    )
    tolerance = summation_factor * scale
    pairs = (
        ("pc_photons", "model_uv_photons"),
        ("pc_ionized_atoms", "grid_uv_ionized_atoms"),
        ("pc_recombinations", "cumulative_recombinations"),
        ("pc_ionized_atoms_total", "ionized_atoms"),
        ("pc_xray_ionized_atoms", "xray_ionized_atoms"),
    )
    for native, independent in pairs:
        # X-ray and total-ionization sums can be much larger than the UV budget
        # at early times. Bound their summation error using their own magnitude.
        comparison_tolerance = summation_factor * max(
            abs(row[native]), abs(row[independent]), np.finfo(float).tiny
        )
        if abs(row[native] - row[independent]) > comparison_tolerance:
            raise FloatingPointError(
                f"PC diagnostic disagrees with saved grids: {native}"
            )
    spending = row["grid_uv_ionized_atoms"] + row["cumulative_recombinations"]
    if spending > row["model_uv_photons"] + tolerance:
        raise FloatingPointError("PC grids overspend the physical UV photon budget")
    if (
        abs(row["grid_uv_closed_residual"]) > tolerance
        or abs(row["pc_budget_residual"]) > tolerance
    ):
        raise FloatingPointError(
            "PC photon ledger does not close within double summation precision"
        )
    if (
        row["pc_status"] == 1
        and row["pc_remaining_photons"] > row["pc_count_tolerance"] + tolerance
    ):
        raise FloatingPointError(
            "Calibrated PC status has an unresolved absorption remainder"
        )
    return tolerance


def save_snapshot(output, coeval, config):
    """Save comparison fields and spectra without changing the native result."""
    tag = f"z{coeval.redshift:.8f}"
    np.savez_compressed(
        output / f"fields_{tag}.npz",
        redshift=coeval.redshift,
        density=coeval.density,
        xHI=coeval.neutral_fraction,
        brightness_mK=coeval.brightness_temp,
        xray_ionised_fraction=coeval.ts_box.get("xray_ionised_fraction"),
        cumulative_recombinations=coeval.ionized_box.get("cumulative_recombinations"),
        ionisation_rate_G12=coeval.ionized_box.get("ionisation_rate_G12"),
    )
    k, modes, p21 = power_spectrum(
        coeval.brightness_temp, config["box_mpc"], config["threads"]
    )
    kx, modes_x, px = power_spectrum(
        coeval.neutral_fraction, config["box_mpc"], config["threads"]
    )
    if not (np.array_equal(k, kx) and np.array_equal(modes, modes_x)):
        raise RuntimeError("Inconsistent power-spectrum bins")
    with (output / f"power_{tag}.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "k_cMpc_inv",
                "full_fft_modes",
                "P21_mK2_cMpc3",
                "Delta2_21_mK2",
                "P_xHI_cMpc3",
                "Delta2_xHI",
            ]
        )
        for kval, count, a, b in zip(k, modes, p21, px, strict=True):
            factor = kval**3 / (2 * np.pi**2)
            writer.writerow([kval, count, a, factor * a, b, factor * b])


def compare_outputs(output):
    """Write same-redshift histories and shell-by-shell PC/ES ratios."""
    histories = {}
    for mode in ("ES", "PC"):
        with (output / mode / "history.csv").open() as stream:
            histories[mode] = list(csv.DictReader(stream))
    es, pc = histories["ES"], histories["PC"]
    if len(es) != len(pc):
        raise RuntimeError("PC and ES histories have unequal lengths")
    fields = [
        "redshift",
        "xHI_volume_ES",
        "xHI_volume_PC",
        "delta_xHI_volume",
        "xHI_mass_ES",
        "xHI_mass_PC",
        "mean_brightness_ES_mK",
        "mean_brightness_PC_mK",
        *[f"PC_{key}" for key in AUDIT_FIELDS],
    ]
    with (output / "history_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for a, b in zip(es, pc, strict=True):
            if a["redshift"] != b["redshift"]:
                raise RuntimeError("PC and ES histories do not use the same redshifts")
            row = {
                "redshift": a["redshift"],
                "xHI_volume_ES": a["xHI_volume"],
                "xHI_volume_PC": b["xHI_volume"],
                "delta_xHI_volume": float(b["xHI_volume"]) - float(a["xHI_volume"]),
                "xHI_mass_ES": a["xHI_mass"],
                "xHI_mass_PC": b["xHI_mass"],
                "mean_brightness_ES_mK": a["mean_brightness_mK"],
                "mean_brightness_PC_mK": b["mean_brightness_mK"],
            }
            row.update({f"PC_{key}": b[key] for key in AUDIT_FIELDS})
            writer.writerow(row)
    for pc_path in sorted((output / "PC").glob("power_z*.csv")):
        es_path = output / "ES" / pc_path.name
        a, b = (
            np.genfromtxt(es_path, names=True, delimiter=","),
            np.genfromtxt(pc_path, names=True, delimiter=","),
        )
        if not np.array_equal(a["k_cMpc_inv"], b["k_cMpc_inv"]):
            raise RuntimeError("Power-spectrum wavenumber bins differ")
        with (output / ("comparison_" + pc_path.name)).open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "k_cMpc_inv",
                    "full_fft_modes",
                    "Delta2_21_ES_mK2",
                    "Delta2_21_PC_mK2",
                    "ratio_21_PC_ES",
                    "Delta2_xHI_ES",
                    "Delta2_xHI_PC",
                ]
            )
            for ae, bp in zip(np.atleast_1d(a), np.atleast_1d(b), strict=True):
                ratio = (
                    bp["Delta2_21_mK2"] / ae["Delta2_21_mK2"]
                    if ae["Delta2_21_mK2"]
                    else np.nan
                )
                writer.writerow(
                    [
                        ae["k_cMpc_inv"],
                        ae["full_fft_modes"],
                        ae["Delta2_21_mK2"],
                        bp["Delta2_21_mK2"],
                        ratio,
                        ae["Delta2_xHI"],
                        bp["Delta2_xHI"],
                    ]
                )


def run_prepared(output, threads):
    """Evolve the two models from the same actual initial conditions."""
    from py21cmfast import __version__
    from py21cmfast.drivers.coeval import generate_coeval
    from py21cmfast.drivers.single_field import compute_initial_conditions
    from py21cmfast.io.caching import CacheConfig, OutputCache

    config = json.loads((output / "run_config.json").read_text())
    if threads != config["threads"]:
        raise ValueError("PBS_NCPUS differs from the prepared N_THREADS")
    if __version__ != config["native_version"]:
        raise RuntimeError("Native 21cmFAST version changed after preparation")
    for name, expected in config["versions"].items():
        if version(name) != expected:
            raise RuntimeError(f"Package version changed after preparation: {name}")
    es, pc = (inputs_for(config, mode) for mode in ("ES", "PC"))
    check_pair(es, pc)
    if list(es.node_redshifts) != config["node_redshifts"]:
        raise RuntimeError("Node redshifts changed after preparation")
    cache = OutputCache(output / "cache")
    initial = compute_initial_conditions(
        inputs=es, cache=cache, write=True, regenerate=False
    )
    initial_hash = {
        name: sha256_array(initial.get(name))
        for name, array in initial.arrays.items()
        if array.state.is_computed
    }
    (output / "initial_conditions_sha256.json").write_text(
        json.dumps(initial_hash, indent=2) + "\n"
    )
    nodes = np.asarray(es.node_redshifts)
    outputs = tuple(
        sorted(
            {float(nodes[np.argmin(abs(nodes - z))]) for z in config["save_redshifts"]},
            reverse=True,
        )
    )
    cache_write = CacheConfig.last_step_only(
        initial_conditions=True, ionized_box=True, spin_temp=True, brightness_temp=True
    )
    for mode, inputs in (("ES", es), ("PC", pc)):
        mode_dir = output / mode
        if mode_dir.exists() and any(mode_dir.iterdir()):
            raise RuntimeError(f"Nonempty {mode} output directory; prepare a fresh run")
        mode_dir.mkdir(exist_ok=True)
        for name, expected in initial_hash.items():
            if sha256_array(initial.get(name)) != expected:
                raise RuntimeError(f"Shared initial-condition field changed: {name}")
        started = time.monotonic()
        history_stream = (mode_dir / "history.csv").open("w", newline="")
        writer = None
        status_counts = {}
        maximum_grid_residual = 0.0
        try:
            for coeval, is_output in generate_coeval(
                inputs=inputs,
                out_redshifts=outputs,
                initial_conditions=initial,
                cache=cache,
                write=cache_write,
                regenerate=False,
                progressbar=False,
            ):
                row = row_for(coeval, time.monotonic() - started, mode)
                row["grid_sum_tolerance"] = 0.0
                if mode == "PC":
                    row["grid_sum_tolerance"] = validate_pc_row(row)
                    status = str(int(row["pc_status"]))
                    status_counts[status] = status_counts.get(status, 0) + 1
                    maximum_grid_residual = max(
                        maximum_grid_residual, abs(row["grid_uv_closed_residual"])
                    )
                if writer is None:
                    writer = csv.DictWriter(history_stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                history_stream.flush()
                print(
                    f"{mode} z={row['redshift']:.6f} xHI_V={row['xHI_volume']:.7f} xHI_M={row['xHI_mass']:.7f} pc_residual={row['pc_budget_residual']:.6g} pc_status={int(row['pc_status'])} pending={row['pc_remaining_photons']:.6g}",
                    flush=True,
                )
                if is_output:
                    save_snapshot(mode_dir, coeval, config)
        finally:
            history_stream.close()
        if mode == "PC":
            summary = {
                "status_counts": status_counts,
                "maximum_grid_closed_residual": maximum_grid_residual,
                "unresolved_boundary_nodes": status_counts.get("3", 0),
            }
            (mode_dir / "validation_summary.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
        (mode_dir / "COMPLETED").write_text("\n")
    compare_outputs(output)
    (output / "COMPLETED").write_text("\n")


def main():
    """Parse the preparation/run mode and execute it."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run-prepared", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--box-mpc", type=float, default=200)
    parser.add_argument("--zmin", type=float, default=5.5)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if (
        args.grid < 4
        or args.threads < 1
        or args.box_mpc < 150
        or not 0 <= args.zmin < 35
    ):
        parser.error("Require grid>=4, threads>=1, box>=150cMpc and 0<=zmin<35")
    if not args.run_prepared:
        prepare(args.output, args)
    if not args.prepare:
        run_prepared(args.output, args.threads)


if __name__ == "__main__":
    main()
