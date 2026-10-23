"""Convergence-controlled quadrature for the research sharp-front solver.

This controller repairs silent acceptance of an arbitrary low direction count.
It does not change the optically thick, infinite-c, no-recombination voxel
model. Angular convergence is empirical, NOT a rigorous true-error bound or a
continuum accuracy certificate. The range bound has a separate monotonicity
proof for precisely this fixed-history model. No quadrature maps are averaged.
"""
from __future__ import annotations

from itertools import combinations
import math
import time

import numpy as np
from scipy.spatial.transform import Rotation

from characteristics import directions
from event_front import solve


class AccuracyNotReached(RuntimeError):
    """The numerical checks were unresolved within the supplied limits."""

    def __init__(self, result):
        self.result = result
        state = result["accuracy"]
        super().__init__(
            f"Angular/range convergence unresolved at K={state['selected_angles']}; "
            f"angular={state['angular_convergence_observed']}, "
            f"range={state['range_bound_satisfied']}, reason={state['stop_reason']}. "
            "Inspect exception.result['accuracy']; no accuracy certificate was issued."
        )


def default_rotations():
    """Two deterministic non-axis rotations plus the original phase."""
    return Rotation.from_rotvec([[0.731, 0.413, 1.137], [-0.913, 0.677, 0.319]]).as_matrix()


def mass_difference(left, right):
    """Absorbed-map L1 and source-category L1, both in photon-count units.

    The category norm detects escape/retention/source-allocation changes even
    when maps agree. Both norms must pass; they are not summed (double count).
    """
    field = float(np.sum(np.abs(left["absorbed"] - right["absorbed"]), dtype=np.longdouble))
    categories = float(sum(
        np.sum(np.abs(left[name] - right[name]), dtype=np.longdouble)
        for name in ("source_absorbed", "source_escaped", "source_retained")
    ))
    return {"absorbed_l1": field, "source_category_l1": categories,
            "maximum": max(field, categories)}


def range_atom_l1_bound(result, capacity):
    """Bound absorbed-map change on extending the SAME ray paths.

    Fixed rates/history, static capacities, no recombination and infinite-c
    are essential. Open boundaries are allowed. This does not bound escape
    versus retention in a fully ionized box, relative power, feedback errors,
    finite-c errors, or errors from a different angular quadrature.
    """
    remaining = float(np.sum(
        np.maximum(np.asarray(capacity, dtype=np.longdouble)
                   - result["absorbed"].astype(np.longdouble), 0), dtype=np.longdouble
    ))
    return min(max(float(result["audit"]["retained"]), 0.0), remaining)


def _tolerance(relative, absolute, total, name):
    if not all(math.isfinite(float(x)) and x >= 0 for x in (relative, absolute)):
        raise ValueError(f"{name} tolerances must be finite and nonnegative")
    return float(absolute + relative * total)


def solve_controlled(
    source, capacity, *, angular_rtol=0.03, angular_atol=0.0,
    range_rtol=1e-3, range_atol=0.0, initial_angles=48, maximum_angles=49152,
    refinement_factor=4, required_successes=2, rotations=None,
    duration=1.0, periodic=False, maximum_distance=None, maximum_range=None,
    maximum_ray_bytes=512 * 1024**2, raise_unresolved=True,
):
    """Refine direction count/range until declared numerical checks pass.

    Acceptance needs ``required_successes`` consecutive angular levels with:
    (1) every phase-matched coarse/fine difference and (2) every pairwise
    rotated-quadrature difference below ``angular_atol + angular_rtol * Q``.
    At least two nontrivial rotations are mandatory. A separate atom-L1 range
    bound must be below ``range_atol + range_rtol * Q`` for every phase.
    These tests are empirical, not extrapolated true-error estimates. Features
    missed by every quadrature and biases common to all phases can survive.
    An all-zero absorption map with positive source/capacity is rejected unless
    the maximum possible absorbed photon count is already below tolerance.
    This catches one concrete quadrature blind spot, not all missed features.

    ``maximum_range`` is a numerical cutoff, not a physical mean-free path or
    c*dt. Do not use this API to extend a physical cutoff. For that use solve
    directly with a clearly changed physical model. Unresolved checks raise
    AccuracyNotReached by default; its result attribute holds the latest map.
    ``raise_unresolved=False`` returns status='unresolved', never silent success.
    A conservative angular-storage limit is checked before each refinement.
    It counts every positive source (including rays possibly kept lazy), plus
    direction storage/construction allowance, so it can reject a case that
    would in practice fit. It is NOT a total-memory bound:
    grid/heap arrays and retained phase maps are additional. ``None`` explicitly
    disables this resource limit. If the first solve is refused, returned array
    fields are None rather than a fabricated zero map.
    """
    started = time.perf_counter()
    s, c = (np.asarray(a, dtype=float) for a in (source, capacity))
    if s.shape != c.shape or s.ndim != 3 or len(set(s.shape)) != 1 or s.size == 0:
        raise ValueError("Nonempty equally shaped cubic source/capacity arrays required")
    if not np.isfinite(s).all() or not np.isfinite(c).all() or (s < 0).any() or (c < 0).any():
        raise ValueError("Finite nonnegative source rates and capacities required")
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Finite positive duration required")
    total = float(np.sum(s, dtype=np.longdouble) * duration)
    if not math.isfinite(total):
        raise ValueError("Total emitted photon count overflows float64")
    for name, value in (("initial_angles", initial_angles), ("maximum_angles", maximum_angles),
                        ("refinement_factor", refinement_factor), ("required_successes", required_successes)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if maximum_angles < initial_angles or refinement_factor < 2 or required_successes < 2:
        raise ValueError("Maximum angles >= initial, refinement >= 2, consecutive successes >= 2 required")
    if maximum_angles > np.iinfo(np.int32).max:
        raise ValueError("Direction count exceeds the C int index representation")
    if maximum_ray_bytes is not None and (
        isinstance(maximum_ray_bytes, (bool, np.bool_))
        or not isinstance(maximum_ray_bytes, (int, np.integer)) or maximum_ray_bytes < 1
    ):
        raise ValueError("maximum_ray_bytes must be a positive integer or None")
    angular_tol = _tolerance(angular_rtol, angular_atol, total, "Angular")
    range_tol = _tolerance(range_rtol, range_atol, total, "Range")
    if not np.isfinite(angular_tol + range_tol):
        raise ValueError("Tolerance overflow")
    rots = default_rotations() if rotations is None else np.asarray(rotations, dtype=float)
    if rots.ndim != 3 or rots.shape[1:] != (3, 3) or len(rots) < 2 or not np.isfinite(rots).all():
        raise ValueError("At least two finite 3x3 rotation matrices required")
    if (not np.allclose(rots @ np.swapaxes(rots, -1, -2), np.eye(3), atol=1e-12, rtol=0)
            or not np.allclose(np.linalg.det(rots), 1, atol=1e-12, rtol=0)):
        raise ValueError("Proper orthogonal rotation matrices required")
    for a, b in combinations([np.eye(3), *rots], 2):
        if np.allclose(a, b, rtol=0, atol=1e-6):
            raise ValueError("Distinct nontrivial rotations required")
    rotations_all = np.concatenate([np.eye(3)[None, ...], rots])
    distance = float(s.shape[0] * np.sqrt(3) if maximum_distance is None else maximum_distance)
    range_cap = float(8 * distance if maximum_range is None else maximum_range)
    if not (math.isfinite(distance) and math.isfinite(range_cap) and 0 < distance <= range_cap):
        raise ValueError("Finite positive maximum_distance <= maximum_range required")
    if range_cap > np.iinfo(np.int32).max - s.shape[0] - 2:
        raise ValueError("Range exceeds supported unwrapped voxel coordinates")

    history, old, streak, kernel_calls = [], None, 0, 0
    resource_checks = []
    source_cells = int(np.count_nonzero(s))
    maximum_absorbable = float(min(np.sum(c, dtype=np.longdouble), np.longdouble(total)))
    angles = int(initial_angles)
    selected = None
    selected_angles, phase_ranges = None, []
    accepted, range_ok = False, False
    stop_reason = "maximum_angles"
    while True:
        maximum_rays = source_cells * angles
        ray_bytes = ((maximum_rays + 4095) // 4096) * 4096 * 64
        direction_bytes = angles * 64
        angular_bytes = ray_bytes + direction_bytes
        resource_checks.append(dict(angles=angles, maximum_materialized_rays=maximum_rays,
                                    conservative_ray_bytes=ray_bytes,
                                    direction_workspace_allowance_bytes=direction_bytes,
                                    estimated_angular_storage_bytes=angular_bytes))
        if maximum_ray_bytes is not None and angular_bytes > maximum_ray_bytes:
            stop_reason = "ray_storage_limit"
            break
        phase_results, phase_ranges, range_attempts = [], [], []
        for phase, rotation in enumerate(rotations_all):
            ray_directions = np.ascontiguousarray(directions(angles) @ rotation.T)
            local_distance = distance
            attempts = []
            while True:
                result = solve(s, c, ray_directions=ray_directions, duration=duration,
                               periodic=periodic, maximum_distance=local_distance)
                kernel_calls += 1
                bound = range_atom_l1_bound(result, c)
                attempts.append({"maximum_distance": local_distance, "atom_l1_bound": bound,
                                 "retained": result["audit"]["retained"],
                                 "kernel_seconds": result["audit"]["seconds"]})
                if bound <= range_tol or local_distance >= range_cap:
                    break
                local_distance = min(2 * local_distance, range_cap)
            phase_results.append(result)
            phase_ranges.append(bound)
            range_attempts.append(attempts)
            distance = max(distance, local_distance)
        phase_pairs = [dict(left=i, right=j, **mass_difference(phase_results[i], phase_results[j]))
                       for i, j in combinations(range(len(phase_results)), 2)]
        refinement = ([] if old is None else [dict(phase=i, **mass_difference(now, previous))
                                               for i, (now, previous) in enumerate(zip(phase_results, old))])
        phase_error = max(row["maximum"] for row in phase_pairs)
        refinement_error = None if old is None else max(row["maximum"] for row in refinement)
        range_ok = max(phase_ranges) <= range_tol
        zero_absorption_unresolved = bool(
            total > 0 and 2 * maximum_absorbable > angular_tol
            and all(not np.any(item["absorbed"]) for item in phase_results)
        )
        level_ok = bool(old is not None and phase_error <= angular_tol
                        and refinement_error <= angular_tol and range_ok
                        and not zero_absorption_unresolved)
        streak = streak + 1 if level_ok else 0
        selected = phase_results[0]
        selected_angles = angles
        history.append(dict(angles=angles, phase_comparisons=phase_pairs,
                            refinement_comparisons=refinement, phase_maximum=phase_error,
                            refinement_maximum=refinement_error, range_atom_l1_bounds=phase_ranges,
                            range_attempts=range_attempts, range_satisfied=range_ok,
                            zero_absorption_coverage_unresolved=zero_absorption_unresolved,
                            level_satisfied=level_ok, consecutive_successes=streak))
        accepted = streak >= required_successes
        if accepted or angles >= maximum_angles:
            stop_reason = ("empirical_checks_satisfied" if accepted else
                           "zero_absorption_coverage_unresolved" if zero_absorption_unresolved else
                           "maximum_angles")
            break
        old = phase_results
        angles = min(angles * refinement_factor, maximum_angles)

    if selected is None:
        selected = dict(absorbed=None, source_absorbed=None, source_escaped=None,
                        source_retained=None, audit=dict(photons=total,
                        model="constant rates, infinite-c sharp opaque voxel front, no recombinations"))
    selected["accuracy"] = dict(
        status="empirically_converged" if accepted else "unresolved",
        stop_reason=stop_reason,
        angular_convergence_observed=bool(accepted), range_bound_satisfied=bool(range_ok),
        angular_tolerance_photons=angular_tol, range_tolerance_photons=range_tol,
        selected_angles=selected_angles, selected_phase=0,
        selected_range=selected["audit"].get("maximum_distance"),
        selected_range_atom_l1_bound=phase_ranges[0] if phase_ranges else None,
        required_successes=required_successes,
        resource_checks=resource_checks, maximum_ray_bytes=maximum_ray_bytes,
        resource_limit_scope="Conservative single-kernel rays plus direction workspace; grid/heap arrays and cached maps are additional",
        rotations=rots.tolist(), history=history, kernel_calls=kernel_calls,
        total_seconds=time.perf_counter() - started,
        angular_guarantee="Empirical phase/refinement agreement only; no rigorous angular true-error bound",
        range_guarantee="Atom-L1 bound under fixed-rate infinite-c static-capacity no-recombination model",
        spatial_or_physical_accuracy_certified=False,
        retained_photons_are_resumable=False,
        returned_map="Unmodified phase-0 high-resolution solver map; no smoothing or phase average",
        model=selected["audit"]["model"],
    )
    if not accepted and raise_unresolved:
        raise AccuracyNotReached(selected)
    return selected
