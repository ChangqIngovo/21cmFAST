"""Compare count-consistent PC candidates on identical physical toy inputs.

The angular reference is the previously independently audited continuous-rate
opaque-voxel event model.  It is *not* radiative-transfer truth: it assumes an
infinite light speed, no recombinations, point emitters at cell centres, and
fully opaque remaining atoms.  A second angular quadrature measures its finite
angle uncertainty.  An isolated source additionally has a continuum sphere
reference integrated over voxel volumes, exposing the event model's own spatial
discretization error.

All public source/capacity/initial arrays are photon or atom counts per cell.
The source is emitted at constant rate over a unit interval.  No candidate may
renormalize these inputs or count an unabsorbed remainder as ionized atoms.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
ALTERNATIVES = HERE.parent / "pc_alternatives"
if str(ALTERNATIVES) not in sys.path:
    sys.path.insert(0, str(ALTERNATIVES))


@dataclass
class Case:
    name: str
    source: np.ndarray
    capacity: np.ndarray
    initial: np.ndarray
    maximum_distance: float
    metadata: dict = field(default_factory=dict)
    analytic: np.ndarray | None = None
    analytic_audit: dict | None = None

    @property
    def remaining_capacity(self):
        return self.capacity - self.initial


def sphere_reference(n, radius, centre, quadrature=64):
    """Integrate a continuous sphere by midpoint xy and exact z intersections."""
    coordinates = np.moveaxis(np.indices((n,) * 3), 0, -1) - np.asarray(centre)
    distances = np.linalg.norm(coordinates, axis=-1)
    half_diagonal = np.sqrt(3) / 2
    answer = np.zeros((n,) * 3, dtype=np.float64)
    answer[distances < radius - half_diagonal] = 1
    boundary = abs(distances - radius) <= half_diagonal
    points = coordinates[boundary]
    offsets = (np.arange(quadrature) + 0.5) / quadrature - 0.5
    x = points[:, 0, None, None] + offsets[None, :, None]
    y = points[:, 1, None, None] + offsets[None, None, :]
    z = points[:, 2, None, None]
    height = np.sqrt(np.maximum(0, radius**2 - x * x - y * y))
    length = np.maximum(0, np.minimum(height, z + 0.5) - np.maximum(-height, z - 0.5))
    answer[boundary] = length.mean(axis=(1, 2))
    return answer


def make_cases(n=24):
    """Fixed geometries include limits, resolved fronts, opacity, and many sources."""
    if n < 24 or n % 2:
        raise ValueError("Use an even grid of at least 24 cells")
    shape = (n,) * 3
    centre = np.full(3, n // 2)
    coordinates = np.indices(shape)
    zero = np.zeros(shape)
    uniform = np.ones(shape)
    max_distance = n / 2
    radius = 4.0
    photons = 4 * np.pi * radius**3 / 3
    cases = {}

    rng = np.random.default_rng(190681)
    density = rng.uniform(0.3, 2.0, shape)
    weak = density * rng.uniform(0, 0.25, shape)
    cases["all-partial"] = Case(
        "all-partial", weak, density, zero.copy(), max_distance,
        {"oracle": "Every cell's own photons fit locally; no correction is needed."},
        weak.copy(), {"analytic_kind": "exact local absorption", "error": 0.0},
    )

    single = zero.copy()
    single[tuple(centre)] = photons
    sphere = sphere_reference(n, radius, centre, 64)
    coarse = sphere_reference(n, radius, centre, 32)
    cases["isolated"] = Case(
        "isolated", single, uniform.copy(), zero.copy(), max_distance,
        {"radius_cells": radius, "centre": centre.tolist()}, sphere,
        {"analytic_kind": "continuous sphere integrated over voxels",
         "quadrature": 64, "volume_relative_error": float(sphere.sum() / photons - 1),
         "quadrature32_64_l1_over_Q": float(abs(sphere - coarse).sum() / photons)},
    )

    left = centre + np.array([-3, 0, 0])
    right = centre + np.array([3, 0, 0])
    pair = zero.copy()
    pair[tuple(left)] = photons
    pair[tuple(right)] = photons
    cases["equal-overlap"] = Case(
        "equal-overlap", pair, uniform.copy(), zero.copy(), max_distance,
        {"isolated_radius_cells": radius, "separation_cells": 6,
         "sources": [left.tolist(), right.tolist()]},
    )

    unequal = zero.copy()
    unequal[tuple(left)] = photons
    unequal[tuple(right)] = 4 * np.pi * 2.5**3 / 3
    cases["unequal-overlap"] = Case(
        "unequal-overlap", unequal, uniform.copy(), zero.copy(), max_distance,
        {"isolated_radii_cells": [radius, 2.5], "separation_cells": 6,
         "sources": [left.tolist(), right.tolist()]},
    )

    ball_a = sum((coordinates[d] - left[d])**2 for d in range(3)) <= radius**2
    ball_b = sum((coordinates[d] - right[d])**2 for d in range(3)) <= radius**2
    initial = (ball_a | ball_b).astype(float)
    increment = zero.copy()
    increment[tuple(left)] = 0.01
    cases["transparent-overlap-increment"] = Case(
        "transparent-overlap-increment", increment, uniform.copy(), initial, max_distance,
        {"radius_cells": radius, "separation_cells": 6,
         "emitter": left.tolist(), "other_centre": right.tolist(),
         "continuum_direction_fraction_through_overlap": 0.125,
         "warning": "Absorber-centre cone fractions are not ray-labelled exit fractions; finite voxels alter the geometric footprint."},
    )

    # A complete yz plane blocks all +x rays that reach it.  The finite travel
    # range is less than the distance for a negative-x ray to wrap to the plane.
    # This deliberately tests angular budgets: allocating all photons to the
    # only available absorber violates the isotropic hemisphere bound.
    slab_cap = zero.copy()
    slab_index = n // 2
    slab_cap[slab_index] = 1e6
    slab_source = zero.copy()
    slab_source[slab_index - 4, n // 2, n // 2] = 1.0
    cases["opaque-plane"] = Case(
        "opaque-plane", slab_source, slab_cap, zero.copy(), max_distance,
        {"source_x": slab_index - 4, "plane_x": slab_index,
         "analytic_absorbed_fraction": (1 - 3.5 / max_distance) / 2,
         "analytic_absorbed_fraction_description": "isotropic rays intersect the near face x=plane_x-0.5 before the range limit",
         "isotropic_hemisphere_upper_bound": 0.5,
         "warning": "The remainder is radiation outside the selected angular/length support; it must not be silently reassigned."},
    )

    rng = np.random.default_rng(130912)
    gauss = gaussian_filter(rng.normal(size=shape), 1.2, mode="wrap")
    gauss = (gauss - gauss.mean()) / gauss.std()
    rho = np.exp(0.7 * gauss)
    rho /= rho.mean()
    sources = np.exp(2.3 * gauss + 0.8 * rng.normal(size=shape))
    sources *= 0.35 * rho.sum() / sources.sum()
    cases["dense-correlated"] = Case(
        "dense-correlated", sources, rho, zero.copy(), max_distance,
        {"seed": 130912, "mean_photons_per_atom": 0.35,
         "source_model": "correlated lognormal toy field; not a Qin20 cosmological realization"},
    )
    return cases


def array_hash(array):
    contiguous = np.ascontiguousarray(array, dtype=np.float64)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def fixture_cases(filename, dose_fractions=()):
    """Load real-source frozen arrays; optional doses are not new redshifts."""
    filename = Path(filename)
    with np.load(filename) as stored:
        source = np.asarray(stored["source"], dtype=np.float64)
        capacity = np.asarray(stored["capacity"], dtype=np.float64)
        initial = np.asarray(stored["initial"], dtype=np.float64) if "initial" in stored else np.zeros_like(source)
    if source.ndim != 3 or len(set(source.shape)) != 1 or source.shape != capacity.shape:
        raise ValueError("Fixture source and capacity must be matching cubic arrays")
    if np.any(source < 0) or np.any(capacity < initial) or np.any(initial < 0):
        raise ValueError("Fixture contains negative or overfilled counts")
    if not all(np.isfinite(a).all() for a in (source, capacity, initial)):
        raise ValueError("Fixture contains non-finite values")
    total = float(source.sum())
    atoms = float(capacity.sum())
    if total <= 0 or atoms <= 0:
        raise ValueError("Fixture must contain photons and atoms")
    metadata = {"fixture": str(filename), "fixture_sha256": hashlib.sha256(filename.read_bytes()).hexdigest(),
                "source_model": "supplied frozen native source field; no evolution rerun",
                "source_dose_multiplier": 1., "photons_per_total_atom": total / atoms}
    preparation = filename.parent / "preparation.json"
    if preparation.exists():
        metadata["fixture_preparation"] = json.loads(preparation.read_text())
    case = Case("native-frozen", source, capacity, initial, source.shape[0] / 2, metadata)
    result = {case.name: case}
    for dose in dose_fractions:
        if not np.isfinite(dose) or dose <= 0:
            raise ValueError("Dose fractions must be finite and positive")
        name = f"native-dose-{dose:g}"
        multiplier = dose * atoms / total
        result[name] = replace(case, name=name, source=source * multiplier,
                              metadata={**metadata, "source_dose_multiplier": multiplier,
                                        "photons_per_total_atom": dose,
                                        "dose_warning": "Artificial source-amplitude rescaling at fixed density; not a cosmological redshift."})
    return result


def finite_scalar(value):
    return float(value) if np.isfinite(value) else None


def power_statistics(candidate, reference):
    """Neutral-count statistics; only wavelengths >=4 cells and >=20 modes."""
    n = candidate.shape[0]
    fa = np.fft.fftn(candidate - candidate.mean())
    fb = np.fft.fftn(reference - reference.mean())
    pa, pb = abs(fa)**2, abs(fb)**2
    cross = (fa * fb.conj()).real
    residual = abs(fa - fb)**2
    k1 = np.fft.fftfreq(n) * n
    modes = np.sqrt(sum(v * v for v in np.meshgrid(k1, k1, k1, indexing="ij")))
    edges = np.linspace(0.5, n / 4 + 0.5, max(4, n // 4 + 1))
    rows = []
    numerator = denominator = residual_sum = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (modes >= low) & (modes < high) & (modes <= n / 4)
        count = int(mask.sum())
        if count < 20:
            continue
        a, b, ab, err = (float(v[mask].mean()) for v in (pa, pb, cross, residual))
        if b <= 1e-25:
            continue
        rows.append({"k_cell_inv": float(2 * np.pi / n * modes[mask].mean()),
                     "full_fft_modes": count, "auto_power_ratio": a / b,
                     "cross_coherence": finite_scalar(ab / np.sqrt(a * b)) if a > 0 else None,
                     "residual_power_over_reference": err / b})
        numerator += count * abs(a - b)
        denominator += count * b
        residual_sum += count * err
    return {"definition": "full-FFT mode-count-weighted shell-mean powers of remaining neutral counts; k<=Nyquist/2; shells with >=20 modes",
            "weighted_absolute_power_difference": numerator / denominator if denominator else None,
            "weighted_residual_power": residual_sum / denominator if denominator else None,
            "shells": rows}


def independent_ledger(absorbed, unabsorbed, case):
    """Reconstruct accounting from arrays rather than candidate diagnostics."""
    a = np.asarray(absorbed, dtype=np.float64)
    if a.shape != case.capacity.shape or not np.isfinite(a).all():
        raise AssertionError("Candidate returned invalid absorbed field")
    q = float(case.source.sum(dtype=np.float64))
    residual = q - float(a.sum(dtype=np.float64)) - float(unabsorbed)
    capacity_excess = float(np.maximum(a - case.remaining_capacity, 0).max())
    negative = float(np.maximum(-a, 0).max())
    scale = max(q, 1.0)
    valid = abs(residual) < 2e-9 * scale and capacity_excess < 2e-10 * scale and negative < 2e-10 * scale and unabsorbed >= -2e-10 * scale
    return {"emitted": q, "absorbed": float(a.sum()), "unabsorbed": float(unabsorbed),
            "unabsorbed_fraction": float(unabsorbed / q) if q else 0.0,
            "residual": residual, "relative_residual": residual / q if q else 0.0,
            "maximum_capacity_excess": capacity_excess, "maximum_negative_absorption": negative,
            "passed": bool(valid)}


def field_statistics(absorbed, reference, case):
    """Compare absorbed photons; include neutral-count statistics separately."""
    q = float(case.source.sum())
    delta = absorbed - reference
    cap = case.capacity
    valid = cap > 0
    dx = np.zeros_like(cap)
    dx[valid] = delta[valid] / cap[valid]
    variance_a = np.std(absorbed)
    variance_b = np.std(reference)
    corr = float(np.corrcoef(absorbed.ravel(), reference.ravel())[0, 1]) if variance_a and variance_b else None
    out = {"L1_absorption_over_emitted": float(abs(delta).sum() / q) if q else 0.0,
           "maximum_absorption_difference": float(abs(delta).max()),
           "volume_mean_absolute_ionized_fraction_difference": float(abs(dx[valid]).mean()) if valid.any() else 0.0,
           "maximum_ionized_fraction_difference": float(abs(dx).max()),
           "absorption_field_correlation": corr,
           "neutral_count_power": power_statistics(case.remaining_capacity - absorbed, case.remaining_capacity - reference)}
    source_weight = float(case.source.sum())
    if np.count_nonzero(case.source) <= 2 and source_weight > 0:
        xyz = np.indices(cap.shape)
        # Sources and fronts in these fixtures are central; no periodic cut
        # crosses the region for which this ordinary centroid is interpreted.
        origin = np.array([(xyz[d] * case.source).sum() / source_weight for d in range(3)])
        total = float(absorbed.sum())
        rtotal = float(reference.sum())
        c = np.array([((xyz[d] - origin[d]) * absorbed).sum() / total for d in range(3)]) if total else np.zeros(3)
        r = np.array([((xyz[d] - origin[d]) * reference).sum() / rtotal for d in range(3)]) if rtotal else np.zeros(3)
        out.update(centroid_offset_cells=c.tolist(), reference_centroid_offset_cells=r.tolist(),
                   centroid_difference_cells=float(np.linalg.norm(c - r)))
    if case.name == "transparent-overlap-increment":
        emitter = np.asarray(case.metadata["emitter"])
        xyz = np.indices(cap.shape) - emitter[:, None, None, None]
        distance = np.sqrt((xyz * xyz).sum(axis=0))
        mu = np.divide(xyz[0], distance, out=np.zeros_like(distance), where=distance > 0)
        cone = mu >= 0.75
        out["absorber_centre_cone_fraction"] = float(absorbed[cone].sum() / q)
        out["reference_absorber_centre_cone_fraction"] = float(reference[cone].sum() / q)
        out["cone_statistic_warning"] = "Spatial footprint diagnostic only; not the exact 12.5% ray-label conservation test."
    if case.name == "opaque-plane":
        frac = float(absorbed.sum() / q)
        out["absorbed_fraction"] = frac
        out["analytic_absorbed_fraction"] = case.metadata["analytic_absorbed_fraction"]
        out["isotropic_hemisphere_bound_violated"] = frac > 0.5 + 1e-8
    return out


def solve_event(case, angles):
    """Compile once before timing and use exactly the residual atom capacity."""
    event = importlib.import_module("event_front")
    event.library()
    start = time.perf_counter()
    out = event.solve(case.source, case.remaining_capacity, angles=angles,
                      periodic=True, maximum_distance=case.maximum_distance)
    elapsed = time.perf_counter() - start
    audit = out["audit"]
    return out["absorbed"], audit["escaped"] + audit["retained"], elapsed, audit


def solve_candidate(name, case):
    """Use a common physical interface; keep candidate-specific diagnostics."""
    if name == "adaptive-event":
        module = importlib.import_module("hybrid")
        if hasattr(module, "library"):
            module.library()
        start = time.perf_counter()
        out = module.solve(case.source, case.capacity, case.initial,
                           periodic=True, maximum_distance=case.maximum_distance)
    else:
        module = importlib.import_module("fast_candidates")
        start = time.perf_counter()
        if name == "es-control":
            out = module.source_consistent_es(case.source, case.capacity, case.initial,
                                              workers=1, max_radius=case.maximum_distance)
            # The deliberately nonconserving baseline does not provide a
            # radiation reservoir. Its missing/excess photons must fail the
            # independent ledger, not be hidden in a manufactured remainder.
            out["unabsorbed"] = 0.
        else:
            out = module.solve(case.source, case.capacity, case.initial,
                               method=name, workers=1, max_radius=case.maximum_distance)
    elapsed = time.perf_counter() - start
    audit = out.get("diagnostics", out.get("audit", {}))
    return np.asarray(out["absorbed"]), float(out["unabsorbed"]), elapsed, audit


def sanitized(value):
    """Write strict JSON with finite scalars and without bulky candidate arrays."""
    if isinstance(value, dict):
        return {str(k): sanitized(v) for k, v in value.items() if not isinstance(v, np.ndarray)}
    if isinstance(value, (list, tuple)):
        return [sanitized(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return finite_scalar(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def run_comparison(cases, methods, *, reference_angles=16384, dense_reference_angles=192,
                   refinement=4, repeats=1, fields_directory=None):
    """Measure each reference refinement and every candidate, recording failures."""
    report = {"scope": "Frozen source/capacity experiments, including optional supplied native field; no thermal, feedback or recombination evolution",
              "reference": "independently audited event-front model, with explicit angular refinement; not full RT",
              "software": {"python": platform.python_version(), "numpy": np.__version__,
                           "platform": platform.platform(), "cpu_count": os.cpu_count()},
              "timing": "one worker; event compilation excluded; candidate repetitions sequential; end-to-end 21cmFAST overhead is not measured",
              "cases": []}
    if fields_directory is not None:
        fields_directory = Path(fields_directory)
        fields_directory.mkdir(parents=True, exist_ok=True)
    for case in cases:
        base = dense_reference_angles if np.count_nonzero(case.source) > 100 else reference_angles
        low, low_left, low_time, low_audit = solve_event(case, base)
        ref, left, ref_time, ref_audit = solve_event(case, base * refinement)
        rec = {"name": case.name, "shape": list(case.source.shape), "metadata": case.metadata,
               "input_hashes": {"source": array_hash(case.source), "capacity": array_hash(case.capacity), "initial": array_hash(case.initial)},
               "initial_ionized_atoms": float(case.initial.sum()),
               "maximum_distance_cells": case.maximum_distance,
               "reference_base_angles": base, "reference_angles": base * refinement,
               "reference_seconds": ref_time, "reference_low_seconds": low_time,
               "reference_audit": ref_audit, "reference_ledger": independent_ledger(ref, left, case),
               "reference_angular_change": field_statistics(low, ref, case),
               "reference_low_remainder": low_left,
               "models": []}
        if case.analytic is not None:
            rec["analytic_reference"] = case.analytic_audit
            rec["event_vs_analytic"] = field_statistics(ref, case.analytic, case)
        fields = {"source": case.source, "capacity": case.capacity, "initial": case.initial,
                  "reference": ref, "reference_low": low}
        if case.analytic is not None:
            fields["analytic"] = case.analytic
        for method in methods:
            try:
                # Warm adaptive compilation before recorded measurements.
                if method == "adaptive-event":
                    module = importlib.import_module("hybrid")
                    warm = getattr(module, "library", None)
                    if callable(warm):
                        warm()
                elapsed = []
                for _ in range(repeats):
                    a, rem, seconds, diagnostics = solve_candidate(method, case)
                    elapsed.append(seconds)
                row = {"method": method, "seconds": float(np.median(elapsed)), "seconds_samples": elapsed,
                       "seconds_over_ref": float(np.median(elapsed) / ref_time),
                       "ledger": independent_ledger(a, rem, case),
                       "vs_event_reference": field_statistics(a, ref, case),
                       "diagnostics": diagnostics}
                if case.analytic is not None:
                    row["vs_analytic"] = field_statistics(a, case.analytic, case)
                fields[method.replace("-", "_")] = a
            except Exception as exc:
                row = {"method": method, "error": f"{type(exc).__name__}: {exc}"}
            rec["models"].append(sanitized(row))
        if fields_directory is not None:
            file = fields_directory / f"comparison_{case.name}.npz"
            np.savez_compressed(file, **fields)
            rec["field_file"] = str(file)
        report["cases"].append(sanitized(rec))
        print(json.dumps({"case": case.name, "reference_s": ref_time,
                          "angular_L1_Q": rec["reference_angular_change"]["L1_absorption_over_emitted"],
                          "models": [{"method": r["method"], "seconds": r.get("seconds"),
                                      "L1_Q": r.get("vs_event_reference", {}).get("L1_absorption_over_emitted"),
                                      "error": r.get("error")} for r in rec["models"]]}), flush=True)
    source_files = [Path(__file__), ALTERNATIVES / "event_front.py", ALTERNATIVES / "event_front.cpp"]
    source_files.extend(p for p in [HERE / "fast_candidates.py", HERE / "hybrid.py", HERE / "hybrid.cpp"] if p.exists())
    report["source_hashes"] = {str(p.relative_to(HERE.parent.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    return sanitized(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=24)
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--methods", nargs="*", default=["es-control", "es-global", "es-front", "radial-shell", "voronoi", "adaptive-event"])
    parser.add_argument("--reference-angles", type=int, default=16384)
    parser.add_argument("--dense-reference-angles", type=int, default=192)
    parser.add_argument("--refinement", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path, default=HERE / "comparison_results.json")
    parser.add_argument("--fields-directory", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--fixture-dose-fractions", type=float, nargs="*", default=[])
    args = parser.parse_args()
    available = make_cases(args.grid)
    if args.fixture:
        available.update(fixture_cases(args.fixture, args.fixture_dose_fractions))
    selected = args.cases or list(available)
    report = run_comparison([available[name] for name in selected], args.methods,
                            reference_angles=args.reference_angles,
                            dense_reference_angles=args.dense_reference_angles,
                            refinement=args.refinement, repeats=args.repeats,
                            fields_directory=args.fields_directory)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
