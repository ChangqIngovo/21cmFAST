#!/usr/bin/env python3
"""Audit native Qin20 E-INTEGRAL source additivity using frozen HDF5 fields.

This is a read-only research diagnostic, not an alternative ionization solver.
It needs numpy and h5py; no native extension is executed unless --next-step is
requested. Every count is in units of the mean number of H atoms per grid cell.
The raw native conditional source arrays are cumulative source-history estimates,
not a set of already localized photon packets.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import h5py
import numpy as np


def scalar(value):
    return value.item() if hasattr(value, "item") else value


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rms(value):
    return float(np.sqrt(np.mean(np.asarray(value) ** 2)))


def compare(reference, candidate):
    residual = candidate - reference
    std = float(np.std(reference))
    return {
        "rms_error_over_native_mean": rms(residual) / float(np.mean(reference)),
        "rms_error_over_native_std": rms(residual) / std if std else None,
        "correlation": float(np.corrcoef(reference.ravel(), candidate.ravel())[0, 1]),
        "max_abs_error_over_native_mean": float(np.max(np.abs(residual)))
        / float(np.mean(reference)),
    }


def filter_grid(value, mask):
    return np.fft.ifftn(np.fft.fftn(value) * mask).real


def forbidden_modes(value, mask):
    """A necessary condition for value = W_R * ANY source is zero rejected power."""
    ft = np.fft.fftn(value)
    fluctuation_power = float(np.sum(np.abs(ft) ** 2) - abs(ft.flat[0]) ** 2)
    bad_power = float(np.sum(np.abs(ft[~mask]) ** 2))
    forbidden_field = np.fft.ifftn(ft * ~mask).real
    return {
        "fraction_of_fluctuation_power": bad_power / fluctuation_power
        if fluctuation_power > 0
        else None,
        "rms_over_mean": rms(forbidden_field) / float(np.mean(value)),
        "rms_over_float32_relative_epsilon": rms(forbidden_field)
        / (float(np.mean(value)) * np.finfo(np.float32).eps),
    }


def load_fixture(directory):
    with h5py.File(directory / "previous_ionized.h5") as handle:
        group = handle["IonizedBox"]
        inputs = group["InputParameters"]
        output = group["OutputFields"]
        arrays = {
            species: output[f"nion_conditional_filtered_{species}"][...].astype(float)
            for species in ("acg", "mcg")
        }
        means = {
            species: float(output.attrs[f"nion_unconditional_{species}"])
            for species in ("acg", "mcg")
        }
        metadata = {
            "redshift": float(group.attrs["redshift"]),
            "seed": int(inputs.attrs["random_seed"]),
            "saved_version": str(inputs.attrs["21cmFAST-version"]),
            "node_redshifts": inputs["node_redshifts"][...].tolist(),
            "simulation_options": {
                key: scalar(value) for key, value in inputs["simulation_options"].attrs.items()
            },
            "astro_options": {
                key: scalar(value) for key, value in inputs["astro_options"].attrs.items()
            },
            "matter_options": {
                key: scalar(value) for key, value in inputs["matter_options"].attrs.items()
            },
        }
        astro = dict(inputs["astro_params"].attrs)
        density_recombinations = output["cumulative_recombinations"][...].astype(float)
    with h5py.File(directory / "previous_density.h5") as handle:
        group = handle["PerturbedField"]
        assert float(group.attrs["redshift"]) == metadata["redshift"]
        density = group["OutputFields/density"][...].astype(float)
    return metadata, astro, arrays, means, density, density_recombinations


def audit(directory):
    metadata, astro, arrays, means, density, rec = load_fixture(directory)
    sim = metadata["simulation_options"]
    assert metadata["matter_options"]["SOURCE_MODEL"] == "E-INTEGRAL"
    assert metadata["astro_options"]["HII_FILTER"] == "sharp-k"
    assert sim["NON_CUBIC_FACTOR"] == 1.0
    size, boxlen = sim["HII_DIM"], sim["BOX_LEN"]
    rho = 1.0 + density
    assert float(np.min(rho)) > 0
    nr = arrays["acg"].shape[0]
    l_factor = 0.620350491
    minimum_radius = max(float(astro["R_BUBBLE_MIN"]), l_factor * boxlen / size)
    maximum_radius = min(float(astro["R_BUBBLE_MAX"]), l_factor * boxlen)
    radii = np.minimum(
        minimum_radius * float(astro["DELTA_R_HII_FACTOR"]) ** np.arange(nr),
        maximum_radius,
    )
    kvals = 2 * np.pi * np.fft.fftfreq(size, d=boxlen / size)
    kx, ky, kz = np.meshgrid(kvals, kvals, kvals, indexing="ij")
    kmag = np.sqrt(kx * kx + ky * ky + kz * kz)
    # The R_index==0 native step deliberately does not filter the cell fields.
    masks = [np.ones_like(kmag, dtype=bool)] + [
        kmag * radius * 0.413566994 <= 1 for radius in radii[1:]
    ]
    corrected = {
        species: values * (means[species] / values.mean(axis=(1, 2, 3)))[:, None, None, None]
        for species, values in arrays.items()
    }
    q = corrected["acg"] + corrected["mcg"]
    budget = size**3 * sum(means.values())
    source_uncalibrated = rho * q[0]
    source_factor = budget / float(source_uncalibrated.sum())
    source_packets = source_uncalibrated * source_factor
    species_budgets = {}
    source_packets_by_species = np.zeros_like(rho)
    for species, fields in corrected.items():
        raw = rho * fields[0]
        species_budget = size**3 * means[species]
        factor = species_budget / float(raw.sum())
        source_packets_by_species += factor * raw
        species_budgets[species] = {
            "global_budget": species_budget,
            "raw_density_weighted_local_sum": float(raw.sum()),
            "renormalization_factor_to_species_budget": factor,
        }
    per_scale = []
    for index, (radius, mask) in enumerate(zip(radii, masks, strict=True)):
        rho_r = 1 + filter_grid(density, mask)
        if float(rho_r.min()) <= 0:
            raise RuntimeError("Fixture requires density clipping; extend audit before use")
        source_r = rho_r * q[index]
        packet_prediction = filter_grid(source_packets, mask) / rho_r
        species_packet_prediction = filter_grid(source_packets_by_species, mask) / rho_r
        raw_count_prediction = filter_grid(source_uncalibrated, mask) / rho_r
        volume_prediction = filter_grid(q[0], mask)
        impulse = np.fft.ifftn(mask).real
        nonzero_modes = mask.copy()
        nonzero_modes.flat[0] = False
        # Shared low-k coefficients must equal those of the same source grid.
        ft_local = np.fft.fftn(source_uncalibrated)
        ft_native = np.fft.fftn(source_r)
        lowk_den = float(np.sum(np.abs(ft_native[nonzero_modes]) ** 2))
        lowk_diff = float(np.sum(np.abs((ft_local - ft_native)[nonzero_modes]) ** 2))
        per_scale.append(
            {
                "index": index,
                "radius_cMpc": float(radius),
                "is_unfiltered_cell_step": index == 0,
                "retained_full_fft_modes_including_dc": int(mask.sum()),
                "native_q_mean": float(q[index].mean()),
                "native_q_std": float(q[index].std()),
                "density_min": float(rho_r.min()),
                "native_density_weighted_source_sum": float(source_r.sum()),
                "native_density_weighted_source_sum_over_budget": float(source_r.sum() / budget),
                "raw_mean_acg": float(arrays["acg"][index].mean()),
                "raw_mean_mcg": float(arrays["mcg"][index].mean()),
                "volume_grid_prediction": compare(q[index], volume_prediction),
                "density_weighted_grid_prediction": compare(q[index], raw_count_prediction),
                "budget_normalized_packet_prediction": compare(q[index], packet_prediction),
                "species_budget_normalized_packet_prediction": compare(q[index], species_packet_prediction),
                "q_forbidden_modes": forbidden_modes(q[index], mask),
                "density_weighted_source_forbidden_modes": forbidden_modes(source_r, mask),
                "shared_lowk_source_coefficient_relative_l2_error": np.sqrt(lowk_diff / lowk_den)
                if lowk_den > 0
                else None,
                "sharp_k_realspace_kernel_minimum": float(impulse.min()),
                "sharp_k_realspace_kernel_negative_mass": float(-impulse[impulse < 0].sum()),
                "filtered_packet_grid_minimum": float(filter_grid(source_packets, mask).min()),
                "sharp_k_cutoff_minimum_distance_from_mode": float(np.min(np.abs(kmag * radius * 0.413566994 - 1))) if index else None,
            }
        )
    rpositive = per_scale[1:]
    result = {
        "scope": "Saved small-grid Qin20 global-budget history, not science-resolution or RT validation",
        "fixture_directory": str(directory.resolve()),
        "files_sha256": {
            name: sha256(directory / name)
            for name in ("previous_ionized.h5", "previous_density.h5")
        },
        "metadata": metadata,
        "photon_source_convention": "effective photons per mean H atom; one mean-density cell has unit capacity",
        "conditional_mean_correction": "species-specific volume mean; source floors omitted here because this is a stored-source additivity audit",
        "native_unconditional": means,
        "global_budget": budget,
        "local_count_interpretation": "S0=(1+delta_cell)*(q_ACG,cell+q_MCG,cell); proposed closure, not a native packet field",
        "rho_minimum": float(rho.min()),
        "rho_mean": float(rho.mean()),
        "raw_local_packet_sum": float(source_uncalibrated.sum()),
        "local_packet_renormalization_factor_to_budget": source_factor,
        "normalized_local_packet_sum": float(source_packets.sum()),
        "normalized_local_packet_minimum": float(source_packets.min()),
        "per_species_local_packet_normalizations": species_budgets,
        "species_normalized_packet_sum": float(source_packets_by_species.sum()),
        "stored_cumulative_recombination_sum": float(rec.sum()),
        "summary": {
            "max_volume_grid_rms_error_over_native_std": max(s["volume_grid_prediction"]["rms_error_over_native_std"] for s in rpositive),
            "max_count_grid_rms_error_over_native_std": max(s["density_weighted_grid_prediction"]["rms_error_over_native_std"] for s in rpositive),
            "max_budget_normalized_packet_rms_error_over_native_std": max(s["budget_normalized_packet_prediction"]["rms_error_over_native_std"] for s in rpositive),
            "max_species_budget_normalized_packet_rms_error_over_native_std": max(s["species_budget_normalized_packet_prediction"]["rms_error_over_native_std"] for s in rpositive),
            "max_q_forbidden_fluctuation_power_fraction": max(s["q_forbidden_modes"]["fraction_of_fluctuation_power"] for s in rpositive),
            "max_count_forbidden_fluctuation_power_fraction": max(s["density_weighted_source_forbidden_modes"]["fraction_of_fluctuation_power"] for s in rpositive),
            "max_sharp_k_kernel_negative_mass": max(s["sharp_k_realspace_kernel_negative_mass"] for s in rpositive),
        },
        "scales": per_scale,
    }
    # An exactly constructed convolution must pass the same support diagnostic.
    positive_source = 1.0 + np.arange(size**3).reshape((size,) * 3) / size**3
    filtered_example = filter_grid(positive_source, masks[-1])
    control = forbidden_modes(filtered_example, masks[-1])
    assert control["fraction_of_fluctuation_power"] < 1e-25
    assert np.max(np.abs(filter_grid(q[0], masks[0]) - q[0])) < 1e-12
    assert abs(float(source_packets_by_species.sum()) - budget) < 1e-12 * budget
    assert min(s["sharp_k_cutoff_minimum_distance_from_mode"] for s in rpositive) > 1e-5
    result["sanity_control_exact_convolution_forbidden_modes"] = control
    return result, q[0], rho


def audit_next_step(directory, prior_q, prior_rho):
    """Optional native frozen one-step call; no source mutation or history rerun."""
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("native_fixture_helpers", root / "tools/validate_global_budget.py")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    from py21cmfast.drivers.single_field import compute_ionization_field

    coeval, previous, initial_values = helpers.load_frozen_step(directory)
    inputs = coeval.inputs.evolve_input_structs(N_THREADS=4)
    previous_box = helpers.copy_output(previous.ionized_box, inputs)
    before = helpers.box_digest(previous_box)
    start = time.perf_counter()
    box = compute_ionization_field(
        inputs=inputs,
        initial_conditions=helpers.copy_initial_conditions(initial_values, inputs),
        perturbed_field=helpers.copy_output(coeval.perturbed_field, inputs),
        previous_perturbed_field=helpers.copy_output(previous.perturbed_field, inputs),
        previous_ionized_box=previous_box,
        spin_temp=helpers.copy_output(coeval.ts_box, inputs),
        regenerate=True,
        write=False,
    )
    elapsed = time.perf_counter() - start
    assert helpers.box_digest(previous_box) == before
    qnew = np.zeros_like(prior_q)
    for species in ("acg", "mcg"):
        name = f"nion_conditional_filtered_{species}"
        values = box.get(name).astype(float)[0]
        target = getattr(box, f"nion_unconditional_{species}")
        qnew += values * target / values.mean()
    newrho = 1 + coeval.perturbed_field.get("density").astype(float)
    s_old = prior_rho * prior_q
    s_new = newrho * qnew
    s_old *= prior_q.sum() / s_old.sum()
    s_new *= qnew.sum() / s_new.sum()
    increments = s_new - s_old
    return {
        "redshift_before": float(previous_box.redshift),
        "redshift_after": float(box.redshift),
        "native_one_step_seconds": elapsed,
        "timing_caveat": "single frozen step in this process; neither full runtime nor paired speed benchmark",
        "input_history_unchanged": helpers.box_digest(previous_box) == before,
        "cellwise_normalized_cumulative_packet_increment_min": float(increments.min()),
        "cellwise_normalized_cumulative_packet_increment_negative_cells": int((increments < 0).sum()),
        "cellwise_normalized_cumulative_packet_increment_negative_total": float(-increments[increments < 0].sum()),
        "global_packet_increment": float(increments.sum()),
        "inference": "Nonnegative increments in one fixture would not prove Eulerian cumulative maps always yield nonnegative emissivity increments; advection and changed mean normalization can break that property.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=Path("/workspace/scratch/40b418003e4a/qin20_validation/final_guard_coarse/frozen_native_step"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("native_source_results.json"))
    parser.add_argument("--next-step", action="store_true")
    args = parser.parse_args()
    result, prior_q, prior_rho = audit(args.fixture)
    if args.next_step:
        result["optional_next_step"] = audit_next_step(args.fixture, prior_q, prior_rho)
    args.output.write_text(json.dumps(result, indent=2, default=scalar) + "\n")
    print(json.dumps({"output": str(args.output), "summary": result["summary"], "local_packet_factor": result["local_packet_renormalization_factor_to_budget"], "next_step": result.get("optional_next_step")}, indent=2, default=scalar))


if __name__ == "__main__":
    main()
