"""Validated accuracy/cost presets for the 21cmFAST research PC adapter.

This remains the static, infinite-light-speed, opaque-voxel event model. Its
angular allocation is approximate; photon closure is not an accuracy bound.
Inputs are photons and atoms per cell, not ionized fractions or emissivities.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import budgeted
import sampling_solver
from speed_core import module

PROFILES = ("balanced", "incremental", "compact")


def warmup(profile="balanced"):
    """Compile before steady-state timing; transport itself remains serial."""
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}")
    module("compact")
    if profile == "balanced":
        budgeted.warmup()
    elif profile == "incremental":
        sampling_solver.probe_library()


def solve(source, capacity, initial_ionized=None, *, profile="balanced", **options):
    """Return absorption and separate source escape/retention ledgers.

    balanced: conservative initial-cavity bound, remaining-neutral density,
        25% more angular samples, compact rays. Recommended for fresh coevals.
    incremental: 48 geometry probes, RMS predicted front radius, remaining-
        neutral density, surface factor 16, compact rays. Recommended for
        increments in existing cavities. Sources contributing >=1% of emitted
        photons retain at least the original conservative cavity resolution;
        sparse narrow openings can still be missed for other sources.
    compact: original model and angular counts, only smaller ray records.

    No profile propagates a retained radiation reservoir into the next step.
    Such a reservoir needs its angular state, not just a scalar photon count.
    """
    warmup(profile)
    if profile == "balanced":
        options.setdefault("angular_boost", 1.25)
        options.setdefault("effective_density", "remaining")
        result = budgeted.solve(source, capacity, initial_ionized, **options)
    elif profile == "incremental":
        options.setdefault("probe_angles", 48)
        options.setdefault("statistic", "rms")
        options.setdefault("rays_per_surface_cell", 16.)
        options.setdefault("effective_density", "remaining")
        options.setdefault("dominant_source_fraction", .01)
        result = sampling_solver.solve(source, capacity, initial_ionized,
                                       backend=module("compact"), **options)
    else:
        result = module("compact").solve(source, capacity, initial_ionized, **options)

    # Independent local and global checks. A large total Q cannot excuse a
    # material ionized-fraction error in a low-density cell.
    c = np.asarray(capacity, dtype=float)
    old = np.zeros_like(c) if initial_ionized is None else np.asarray(initial_ionized, dtype=float)
    a = result["absorbed"]
    violation = np.maximum(np.maximum(a - (c - old), 0.), np.maximum(-a, 0.))
    local = float(np.max(violation / np.maximum(c, np.finfo(float).tiny)))
    q = np.asarray(source, dtype=float).sum(dtype=np.longdouble)
    residual = q - a.sum(dtype=np.longdouble) - np.longdouble(result["unabsorbed"])
    if (not np.isfinite(residual) or local > 2e-10
            or abs(residual) / max(q, np.finfo(float).tiny) > 2e-9):
        raise RuntimeError("Independent local/global photon ledger failed")
    result["audit"].update(
        solver_profile=profile,
        maximum_local_fraction_violation=local,
        independent_global_residual=float(residual),
        ray_storage="48-byte compact rays; 64-byte fallback for wide indices",
        physical_scope="static gas, constant emission, infinite c, opaque voxels; no recombinations",
        continuum_accuracy_certified=False,
    )
    return result
