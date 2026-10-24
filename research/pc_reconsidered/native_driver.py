#!/usr/bin/env python3
"""Run the spatial-PC research comparison on native 21cmFAST sources.

This is a deliberately restricted native integration, NOT the Qin20 template:
L-INTEGRAL, ACG only, no recombinations, no X-rays/spin evolution, and no UV
feedback.  Native initial conditions, displacement, emissivity, density and
brightness-temperature functions are used.  Three distinct outputs are kept:
native ES; an ES control on one additive source field with positive ball filters;
and the source-adaptive angular candidate on that same source field.
The default balanced research profile uses the compact kernel and denser angular
sampling; original, compact, incremental, adaptive-front and refined-front
remain explicit CLI choices.

The native n_ion field is a cumulative *effective* photon budget per mean-baryon
cell in Lagrangian-source mode.  It is already an extensive Eulerian source
field: multiplying it by 1+delta again is wrong.  Capacity uses 1+delta, in the
matching native count convention.  This driver does not interpret n_ion as a
rate or difference maps from different redshifts.  Each coeval is an independent
cumulative-budget construction in frozen gas, not a time-evolving RT solution.

No candidate IonizedBox is written into the native cache, whose input identity
does not contain this research solver.  The explicit output archive has its own
metadata, arrays, and independent double/float32 count audits.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

SOLVERS = ("original", "balanced", "incremental", "compact", "adaptive-front", "refined-front")


def _spatial_solver(profile):
    """Select an explicitly named research kernel without changing native physics."""
    if profile not in SOLVERS:
        raise ValueError(f"Unknown spatial solver {profile!r}; choose one of {SOLVERS}")
    if profile == "original":
        from hybrid import library, solve
        return solve, library
    if profile == "adaptive-front":
        directory = str(Path(__file__).resolve().parents[1] / "pc_adaptive_front")
        if directory not in sys.path:
            sys.path.insert(0, directory)
        from adaptive import library, solve

        def adaptive_selected(source, capacity, **kwargs):
            result = solve(source, capacity, **kwargs)
            result["audit"]["solver_profile"] = profile
            return result

        return adaptive_selected, library
    if profile == "refined-front":
        directory = str(Path(__file__).resolve().parents[1] / "pc_refined_front")
        if directory not in sys.path:
            sys.path.insert(0, directory)
        from refined import library, solve
        from hybrid import estimate_angles

        def refined_selected(source, capacity, *, min_angles=24, max_angles=12288,
                             rays_per_surface_cell=4., periodic=True,
                             maximum_distance=None):
            horizon = (len(source)*np.sqrt(3.) if maximum_distance is None
                       else float(maximum_distance))
            # Use the original source/density mesh for the angular predictor,
            # so front refinement does not also change angular resolution.
            levels, counts, radius, _ = estimate_angles(
                source, capacity, capacity, min_angles=min_angles,
                max_angles=max_angles, rays_per_surface_cell=rays_per_surface_cell,
                periodic=periodic, maximum_distance=horizon)
            result = solve(source, capacity, subcells_per_axis=4,
                           angle_counts=counts[levels], periodic=periodic,
                           maximum_distance=horizon)
            result["estimated_radius"] = radius
            result["audit"]["solver_profile"] = profile
            result["audit"]["angular_predictor"] = "original coarse source/density grid"
            return result

        return refined_selected, library
    refinement = str(Path(__file__).resolve().parents[1] / "pc_refinement")
    if refinement not in sys.path:
        sys.path.insert(0, refinement)
    from upgraded import solve, warmup

    def selected(source, capacity, **kwargs):
        return solve(source, capacity, profile=profile, **kwargs)

    return selected, lambda: warmup(profile=profile)


@dataclass
class NativeSources:
    source: np.ndarray
    capacity: np.ndarray
    metadata: dict


def validate_inputs(inputs):
    """Reject physical modules whose state cannot be finalized in this adapter."""
    a, m, s = inputs.astro_options, inputs.matter_options, inputs.simulation_options
    errors = []
    if m.SOURCE_MODEL != "L-INTEGRAL":
        errors.append("SOURCE_MODEL must be L-INTEGRAL")
    for flag in ("USE_MCGS", "USE_TS_FLUCT", "USE_REIONIZATION_PHOTOHEATING_FEEDBACK"):
        if getattr(a, flag):
            errors.append(f"{flag} must be False")
    if a.RECOMB_MODEL != "none":
        errors.append("RECOMB_MODEL must be none")
    if a.PHOTON_CONS_TYPE != "no-photoncons":
        errors.append("PHOTON_CONS_TYPE must be no-photoncons for the unmodified native control")
    if not m.MINIMIZE_MEMORY:
        errors.append("MINIMIZE_MEMORY must be True (no unfinalized kinetic-temperature field)")
    if s.NON_CUBIC_FACTOR != 1:
        errors.append("Only cubic grids are supported")
    if errors:
        raise ValueError("Restricted native PC adapter: " + "; ".join(errors))


def _hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def from_native_coeval(coeval):
    """Extract additive L-INTEGRAL counts without density reweighting or clipping."""
    validate_inputs(coeval.inputs)
    source = np.array(coeval.emissivity_fields.get("n_ion"), dtype=np.float64, copy=True)
    capacity = 1 + np.array(coeval.perturbed_field.get("density"), dtype=np.float64, copy=True)
    if source.shape != capacity.shape or source.ndim != 3 or len(set(source.shape)) != 1:
        raise ValueError("Native source and density must be matching cubic grids")
    if (not np.isfinite(source).all() or not np.isfinite(capacity).all()
            or (source < 0).any() or (capacity < 0).any()):
        raise ValueError("Invalid native source/density; adapter does not clip physical deficits")
    total = float(source.sum(dtype=np.longdouble))
    analytic_total = source.size * (coeval.ionized_box.nion_unconditional_acg
                                    + coeval.ionized_box.nion_unconditional_mcg)
    metadata = {
        "source_model": "L-INTEGRAL",
        "source_definition": "native EmissivityFields.n_ion, no extra density factor",
        "capacity_definition": "1 + native evolved density contrast",
        "count_units": "native effective photon/baryon count per mean-density cell",
        "cumulative_budget_not_rate": True,
        "source_total": total,
        "native_reported_unconditional_total": float(analytic_total),
        "source_minus_native_reported_total": total - analytic_total,
        "capacity_total": float(capacity.sum(dtype=np.longdouble)),
        "source_sha256_float64": _hash(source),
        "capacity_sha256_float64": _hash(capacity),
    }
    source.flags.writeable = False
    capacity.flags.writeable = False
    return NativeSources(source, capacity, metadata)


def update_reionization_redshift(xhi, redshift, previous=None):
    """Native first-full-ionization convention; metadata only, not time transport.

    Previous labels may be retained when reporting a sequence of coevals.  This
    does not turn repeated cumulative source maps into emitted increments.
    """
    xhi = np.asarray(xhi)
    if not np.isfinite(redshift) or redshift < 0 or not np.isfinite(xhi).all():
        raise ValueError("Finite nonnegative redshift and finite neutral fractions required")
    if (xhi < 0).any() or (xhi > 1).any():
        raise ValueError("Neutral fractions must lie in [0, 1]")
    result = np.full(xhi.shape, -1., dtype=np.float32)
    if previous is not None:
        previous = np.asarray(previous)
        if previous.shape != xhi.shape or not np.isfinite(previous).all():
            raise ValueError("Previous reionization labels must match the current grid")
        if ((previous != -1) & (previous < redshift)).any():
            raise ValueError("Previous ionization cannot be later than the current snapshot")
        result[:] = previous
    result[(result < 0) & (xhi <= 1e-6)] = redshift
    return result


def _native_box(coeval, absorbed, capacity):
    """Make only the fields used by the deliberately restricted native modules."""
    absorbed = np.asarray(absorbed, dtype=float)
    if absorbed.shape != capacity.shape or not np.isfinite(absorbed).all():
        raise ValueError("Invalid absorption field")
    # Relative to the local atom count, including voids. A unit-scale floor would
    # silently allow large fractional overionization in low-density cells.
    tolerance = 5e-10 * capacity
    if (absorbed < -tolerance).any() or (absorbed > capacity + tolerance).any():
        raise ValueError("Absorption violates native atom capacity")
    from py21cmfast.wrapper.outputs import IonizedBox
    fraction = np.divide(absorbed, capacity, out=np.zeros_like(absorbed), where=capacity > 0)
    # Only sub-tolerance floating-point bound errors are removed here. The double
    # precision absorption and its independent ledger are saved without clipping.
    xhi = np.asarray(np.clip(1 - fraction, 0, 1), dtype=np.float32)
    box = IonizedBox.new(inputs=coeval.inputs, redshift=coeval.redshift)
    box.set("neutral_fraction", xhi)
    box.set("z_reion", update_reionization_redshift(xhi, coeval.redshift))
    # Native no-recombination mode does not estimate a UV photoionization rate.
    # The strict feedback guard above prevents this zero from entering feedback.
    box.set("ionisation_rate_G12", np.zeros_like(xhi))
    for name in ("log10_mturn_ave_acg", "log10_mturn_ave_mcg",
                 "nion_unconditional_acg", "nion_unconditional_mcg"):
        box.set(name, getattr(coeval.ionized_box, name))
    return box


def compare_native_coeval(coeval, *, min_angles=24, max_angles=12288,
                         rays_per_surface_cell=None, maximum_distance=None,
                         workers=1, solver="balanced"):
    """Return actual native IonizedBox/BrightnessTemp objects for the two arms."""
    from py21cmfast.drivers.single_field import brightness_temperature
    from fast_candidates import source_consistent_es

    source = from_native_coeval(coeval)
    solve, warmup = _spatial_solver(solver)
    n = source.source.shape[0]
    cell_mpc = coeval.inputs.simulation_options.BOX_LEN / n
    if maximum_distance is None:
        maximum_distance = min(n / 2, coeval.inputs.astro_params.R_BUBBLE_MAX / cell_mpc)
    warmup_started = time.perf_counter()
    warmup()
    warmup_seconds = time.perf_counter() - warmup_started
    started = time.perf_counter()
    control = source_consistent_es(source.source, source.capacity,
                                   max_radius=maximum_distance, workers=workers)
    spatial_parameters = dict(min_angles=min_angles, max_angles=max_angles,
                              periodic=True, maximum_distance=maximum_distance)
    if rays_per_surface_cell is not None:
        spatial_parameters["rays_per_surface_cell"] = rays_per_surface_cell
    candidate = solve(source.source, source.capacity, **spatial_parameters)
    boxes = {}
    brightness = {}
    audits = {}
    for name, solution in (("additive_es", control), ("adaptive_pc", candidate)):
        box = _native_box(coeval, solution["absorbed"], source.capacity)
        boxes[name] = box
        brightness[name] = brightness_temperature(
            ionized_box=box, perturbed_field=coeval.perturbed_field,
            regenerate=True, write=False,
        )
        rounded = source.capacity * (1 - box.get("neutral_fraction").astype(float))
        audits[name] = {
            "double_absorbed": float(solution["absorbed"].sum(dtype=np.longdouble)),
            "stored_float32_absorbed": float(rounded.sum(dtype=np.longdouble)),
            "storage_count_difference": float(np.sum(rounded - solution["absorbed"], dtype=np.longdouble)),
            "volume_neutral_fraction": float(np.mean(box.get("neutral_fraction"), dtype=np.float64)),
            "mass_neutral_fraction": float(1 - rounded.sum() / source.capacity.sum()),
            "brightness_mean_mK": float(np.mean(brightness[name].get("brightness_temp"), dtype=np.float64)),
            "brightness_std_mK": float(np.std(brightness[name].get("brightness_temp"), dtype=np.float64)),
            "solver": solution["audit"],
        }
    native_atoms = source.capacity * (1 - coeval.ionized_box.get("neutral_fraction").astype(float))
    audits["native_es"] = {
        "absorbed": float(native_atoms.sum(dtype=np.longdouble)),
        "photon_residual": source.metadata["source_total"] - float(native_atoms.sum(dtype=np.longdouble)),
        "volume_neutral_fraction": float(np.mean(coeval.ionized_box.get("neutral_fraction"), dtype=np.float64)),
        "brightness_mean_mK": float(np.mean(coeval.brightness_temp, dtype=np.float64)),
        "brightness_std_mK": float(np.std(coeval.brightness_temp, dtype=np.float64)),
    }
    metadata = {
        "scope": "21cmFAST-native restricted coeval integration; not Qin20 exact or coupled time evolution",
        "solver_profile": solver,
        "solver_parameters": {**spatial_parameters,
                              "surface_sampling_override": rays_per_surface_cell},
        "solver_warmup_seconds": warmup_seconds,
        "comparison_timing_scope": "matched control, spatial candidate and native brightness; solver compilation/warmup excluded",
        "redshift": float(coeval.redshift),
        "source": source.metadata,
        "maximum_distance_cells": float(maximum_distance),
        "maximum_distance_cMpc": float(maximum_distance * cell_mpc),
        "matched_control": "same additive sources/capacity; positive discrete ball filters, center-cell ES",
        "native_es_control_is_different_filter": coeval.inputs.astro_options.HII_FILTER,
        "brightness_scope": "native brightness_temperature with USE_TS_FLUCT=False: saturated spin-temperature approximation",
        "limitations": [
            "ACG only; no MCG, X-ray ionization/heating, recombinations, or photoheating feedback",
            "independent cumulative-budget coevals; no emitted-rate integration, gas advection, or finite light speed",
            "native cumulative n_ion source convention retained; not an absolute H/He radiation ledger",
            "adaptive angle predictor is not an angular error certificate",
            "finite-distance photons are retained separately; no unresolved photon is counted as ionization",
            "candidate outputs are separate archives, never native ES cache entries",
        ],
        "audits": audits,
        "comparison_seconds": time.perf_counter() - started,
    }
    return dict(source=source, boxes=boxes, brightness=brightness,
                control=control, candidate=candidate, metadata=metadata)


def prepare_native(*, dimension=32, box_len=200., threads=4, random_seed=1234):
    """Construct documented restrictions from a Qin20 parameter starting point."""
    from py21cmfast.wrapper.inputs import InputParameters
    base = InputParameters.from_template(
        "Qin20", random_seed=random_seed, HII_DIM=dimension, DIM=3 * dimension,
        HIRES_TO_LOWRES_FACTOR=None, BOX_LEN=box_len, N_THREADS=threads,
        R_BUBBLE_MAX=min(50., box_len / 3),
    )
    inputs = base.evolve_input_structs(
        SOURCE_MODEL="L-INTEGRAL", USE_MCGS=False, USE_TS_FLUCT=False,
        RECOMB_MODEL="none", USE_REIONIZATION_PHOTOHEATING_FEEDBACK=False,
        PHOTON_CONS_TYPE="no-photoncons", MINIMIZE_MEMORY=True, V_CB_MODEL="NONE",
    ).clone(node_redshifts=())
    before = base.asdict(include_cosmo_tables="never", only_cstruct_params=False)
    after = inputs.asdict(include_cosmo_tables="never", only_cstruct_params=False)
    differences = {}
    for group, values in after.items():
        if isinstance(values, dict):
            for name, value in values.items():
                if before[group].get(name) != value:
                    differences[f"{group}.{name}"] = {"before": before[group].get(name), "after": value}
    differences["node_redshifts"] = {"before": list(base.node_redshifts), "after": []}
    validate_inputs(inputs)
    return inputs, differences


def run_native(*, output, redshift=8., dimension=32, box_len=200., threads=4,
               random_seed=1234, min_angles=24, max_angles=12288,
               rays_per_surface_cell=None, solver="balanced"):
    """Generate ICs and native sources, run PC/control, then native 21-cm outputs."""
    from py21cmfast import run_coeval
    from py21cmfast.io.caching import OutputCache
    if solver not in SOLVERS:
        raise ValueError(f"Unknown spatial solver {solver!r}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "native_comparison.json").exists():
        raise FileExistsError("Use a fresh output directory to retain the prior experiment")
    inputs, differences = prepare_native(dimension=dimension, box_len=box_len,
                                         threads=threads, random_seed=random_seed)
    started = time.perf_counter()
    coeval = run_coeval(inputs=inputs, out_redshifts=redshift, regenerate=True,
                        write=False, cache=OutputCache(output / "native_cache"))[0]
    native_seconds = time.perf_counter() - started
    result = compare_native_coeval(coeval, min_angles=min_angles, max_angles=max_angles,
                                   rays_per_surface_cell=rays_per_surface_cell, workers=threads,
                                   solver=solver)
    result["metadata"]["input_differences_from_Qin20_with_same_geometry"] = differences
    result["metadata"]["inputs"] = inputs.asdict(include_cosmo_tables="never", only_cstruct_params=False)
    result["metadata"]["native_generation_seconds"] = native_seconds
    arrays = {"source": result["source"].source, "capacity": result["source"].capacity,
              "native_es_xhi": coeval.ionized_box.get("neutral_fraction"),
              "native_es_brightness_mK": np.asarray(coeval.brightness_temp),
              "adaptive_pc_absorbed": result["candidate"]["absorbed"],
              "additive_es_absorbed": result["control"]["absorbed"],
              "adaptive_pc_source_retained": result["candidate"]["source_retained"]}
    for name, box in result["boxes"].items():
        arrays[f"{name}_xhi"] = box.get("neutral_fraction")
        arrays[f"{name}_z_reion"] = box.get("z_reion")
        arrays[f"{name}_brightness_mK"] = result["brightness"][name].get("brightness_temp")
    np.savez_compressed(output / "native_fields.npz", **arrays)
    (output / "native_comparison.json").write_text(json.dumps(result["metadata"], indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--redshift", type=float, default=8.)
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--box-len", type=float, default=200.)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=1234)
    parser.add_argument("--min-angles", type=int, default=24)
    parser.add_argument("--max-angles", type=int, default=12288)
    parser.add_argument("--solver", choices=SOLVERS, default="balanced",
                        help="Research spatial kernel; every choice retains the restricted native physics")
    parser.add_argument("--rays-per-surface-cell", type=float, default=None,
                        help="Override the selected profile's surface sampling parameter")
    result = run_native(**vars(parser.parse_args()))
    print(json.dumps({key: result["metadata"][key] for key in
                      ("scope", "solver_profile", "solver_parameters", "audits")}, indent=2))
