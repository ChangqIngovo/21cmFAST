"""Research-only positive-kernel source/capacity matching on a periodic grid.

The implicit edge flows obey source and target budgets. They do not contain
line-of-sight optical depths or fixed angular photon budgets.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import fft


def periodic_distance(shape):
    """Return cell-centre distances from the periodic origin."""
    axes = [np.minimum(np.arange(n), n - np.arange(n)) for n in shape]
    return np.sqrt(
        sum(
            a.reshape((1,) * k + (-1,) + (1,) * (len(shape) - k - 1)) ** 2
            for k, a in enumerate(axes)
        )
    )


def kernel(shape, radius, *, inner_radius=-1.0, inverse_square=False):
    """Construct a normalized, nonnegative real-space ball or shell."""
    distance = periodic_distance(shape)
    weight = ((distance <= radius) & (distance > inner_radius)).astype(float)
    if inverse_square:
        weight /= np.maximum(distance, 0.5) ** 2
    total = weight.sum()
    if total == 0:
        raise ValueError("Empty convolution support")
    return weight / total


@dataclass
class Convolver:
    """Periodic convolution; retain diagnostic count of actual FFT work."""

    weights: np.ndarray
    workers: int = 4

    def __post_init__(self):
        self.spectrum = fft.rfftn(self.weights, workers=self.workers)
        self.calls = 0

    def __call__(self, values):
        self.calls += 1
        result = fft.irfftn(
            fft.rfftn(values, workers=self.workers) * self.spectrum,
            s=values.shape,
            workers=self.workers,
        )
        # The exact operator is nonnegative. FFT roundoff is separately audited.
        return np.maximum(result, 0.0)


def allocation_round(photons, capacity, convolve):
    """Return accepted target counts and source debits from one common flow.

    In exact arithmetic F_ij = K_ij u_i c_j a_j, u_i=p_i/(Kc)_i,
    a_j=min(1,c_j/[c_j(Ku)_j]). Thus each row sum <= p_i and each
    column sum <= c_j. Symmetric K allows the same FFT for the adjoint.
    """
    norm = convolve(capacity)
    # A zero true denominator can have a small positive FFT residue. Never
    # amplify it into a huge source coefficient. This guard can leave extremely
    # weak/high-dynamic-range interactions unresolved, which is reported.
    floor = 128 * np.finfo(float).eps * np.max(capacity, initial=0.0)
    coefficient = np.divide(
        photons, norm, out=np.zeros_like(photons), where=norm > floor
    )
    offer = capacity * convolve(coefficient)
    acceptance = np.minimum(
        1.0,
        np.divide(capacity, offer, out=np.ones_like(capacity), where=offer > 0),
    )
    # min(offer, capacity) is algebraically offer*acceptance but prevents one-ulp
    # negative remaining capacities after a fully accepted target.
    absorbed = np.minimum(offer, capacity)
    debit_raw = coefficient * convolve(capacity * acceptance)
    debit = np.minimum(photons, np.maximum(debit_raw, 0.0))
    unresolved = float(np.sum(photons[(photons > 0) & (norm <= floor)]))
    return (
        absorbed,
        debit,
        {
            "sum_mismatch": float(np.sum(absorbed) - np.sum(debit)),
            "max_source_overdraw_before_roundoff_clip": float(
                np.max(np.maximum(debit_raw - photons, 0), initial=0)
            ),
            "guarded_source_photons": unresolved,
        },
    )


def allocate(
    source,
    capacity,
    radii,
    *,
    workers=4,
    rounds_per_scale=2,
    local_first=True,
    inverse_square=False,
    shells=False,
):
    """Consume available atoms using ascending scales and simultaneous matching.

    Inputs are available photon and atom counts, not rates. Every source is
    processed simultaneously, so source enumeration has no role. Each round
    debits its actual source field; no global efficiency correction is applied.
    """
    started = time.perf_counter()
    source = np.asarray(source, dtype=float)
    capacity = np.asarray(capacity, dtype=float)
    if source.shape != capacity.shape or source.ndim != 3:
        raise ValueError("Require equally shaped 3D arrays")
    if not np.isfinite(source).all() or not np.isfinite(capacity).all():
        raise ValueError("Nonfinite counts")
    if np.any(source < 0) or np.any(capacity < 0):
        raise ValueError("Negative counts")
    if any(b <= a for a, b in zip(radii[:-1], radii[1:])):
        raise ValueError("Radii must increase strictly")
    if not isinstance(rounds_per_scale, int) or rounds_per_scale < 1:
        raise ValueError("rounds_per_scale must be a positive integer")
    photons, available = source.copy(), capacity.copy()
    absorbed = np.zeros_like(source)
    if local_first:
        here = np.minimum(photons, available)
        photons -= here
        available -= here
        absorbed += here
        del here
    calls = 0
    records = []
    inner = -1.0
    for radius in radii:
        if photons.sum() <= 1e-13 * max(source.sum(), np.finfo(float).tiny):
            break
        weights = kernel(
            source.shape,
            radius,
            inner_radius=inner if shells else -1,
            inverse_square=inverse_square,
        )
        convolve = Convolver(weights, workers)
        for iteration in range(rounds_per_scale):
            taken, debit, audit = allocation_round(photons, available, convolve)
            photons = np.maximum(photons - debit, 0.0)
            available = np.maximum(available - taken, 0.0)
            absorbed += taken
            records.append(
                {
                    "radius": float(radius),
                    "iteration": iteration,
                    "absorbed": float(taken.sum()),
                    **audit,
                }
            )
            if debit.sum() <= 1e-14 * max(source.sum(), np.finfo(float).tiny):
                break
        calls += 2 * convolve.calls + 1
        inner = radius
    consumed = source - photons
    audit = {
        "source": float(source.sum()),
        "absorbed": float(absorbed.sum()),
        "remaining": float(photons.sum()),
        "ledger_residual": float(source.sum() - photons.sum() - absorbed.sum()),
        "max_source_overdraw": float(np.max(np.maximum(consumed - source, 0))),
        "max_capacity_overdraw": float(np.max(np.maximum(absorbed - capacity, 0))),
        "fft_calls": calls,
        "rounds": len(records),
        "seconds": time.perf_counter() - started,
        "records": records,
    }
    return {
        "absorbed": absorbed,
        "remaining_source": photons,
        "source_debit": consumed,
        "available": available,
        "audit": audit,
    }


def benchmark_adapter(source, capacity, radii, *, workers=4, rounds_per_scale=2):
    """Stable research adapter; audit seconds includes all allocation work.

    Source is an explicitly supplied additive count field. This does not claim
    to reconstruct native radius-conditioned Qin20 emissivity statistics.
    """
    return allocate(
        source, capacity, radii, workers=workers, rounds_per_scale=rounds_per_scale
    )


def es_filter_baseline(source, capacity, radii, workers=4):
    """Two-field ES filter microbenchmark with shared forward transforms.

    It excludes conditional-source integrals, thermal histories and allocation.
    It is not a timing for the full native 21cmFAST solver.
    """
    started = time.perf_counter()
    fs, fc = fft.rfftn(source, workers=workers), fft.rfftn(capacity, workers=workers)
    selected = np.zeros(source.shape, dtype=bool)
    for radius in radii:
        kh = fft.rfftn(kernel(source.shape, radius), workers=workers)
        sp = fft.irfftn(fs * kh, s=source.shape, workers=workers)
        cp = fft.irfftn(fc * kh, s=source.shape, workers=workers)
        selected |= sp >= cp
    return {
        "seconds": time.perf_counter() - started,
        "fft_calls": 2 + 3 * len(radii),
        "fully_selected": int(selected.sum()),
    }


def make_probe_results(workers=4):
    """Measure morphology and explicit failure cases on reproducible fixtures."""
    n = 32
    centre = np.array([n // 2] * 3)
    axes = np.indices((n,) * 3)
    distance = np.sqrt(np.sum((axes - centre[:, None, None, None]) ** 2, axis=0))
    capacity = np.ones((n,) * 3)
    source = np.zeros_like(capacity)
    radius = 5
    budget = float(np.count_nonzero(distance <= radius))
    source[tuple(centre)] = budget
    isolated = {}
    for label, radii in {
        "geometric": [1, 2, 4, 8],
        "unit_steps": list(range(1, 9)),
        "native_like_spacing_1p1": list(1.1 ** np.arange(30)),
    }.items():
        run = allocate(source, capacity, radii, workers=workers)
        isolated[label] = {
            **{k: v for k, v in run["audit"].items() if k != "records"},
            "ideal_sphere_count": budget,
            "counts_outside_ideal_sphere": float(
                run["absorbed"][distance > radius].sum()
            ),
            "L1_difference_from_ideal_sphere": float(
                np.abs(run["absorbed"] - (distance <= radius)).sum()
            ),
        }

    # A deliberately isolated collinear ray fixture: opaque foreground at x=2,
    # and downstream target at x=3. The broad radius connects both in one step.
    source.fill(0)
    capacity.fill(0)
    source[8, 16, 16] = 1.0
    capacity[10, 16, 16] = 100.0
    capacity[11, 16, 16] = 100.0
    wall = allocate(source, capacity, [4], workers=workers, local_first=False)
    shadow = {
        "foreground_absorbed": float(wall["absorbed"][10, 16, 16]),
        "downstream_absorbed": float(wall["absorbed"][11, 16, 16]),
        "foreground_capacity": 100.0,
        "ledger_residual": wall["audit"]["ledger_residual"],
        "interpretation": "Fails opaque-screen order: a target behind an unsaturated screen receives photons.",
    }

    # Two directions at equal radius must receive equal isotropic photon shares
    # for equal solid angle; matching instead directs budget toward greater capacity.
    source.fill(0)
    capacity.fill(0)
    source[16, 16, 16] = 1.0
    capacity[18, 16, 16] = 100.0
    capacity[14, 16, 16] = 1.0
    run = allocate(source, capacity, [1, 2], workers=workers, local_first=False)
    anisotropy = {
        "dense_direction": float(run["absorbed"][18, 16, 16]),
        "dilute_direction": float(run["absorbed"][14, 16, 16]),
        "ratio": float(run["absorbed"][18, 16, 16] / run["absorbed"][14, 16, 16]),
        "interpretation": "Equal-distance equal-solid-angle targets receive 100:1, not an isotropic source angular quota; thin shells do not fix this.",
    }

    # Finite support and zero-overlap numerical floor.
    source.fill(0)
    capacity.fill(0)
    source[1, 1, 1] = 2.0
    capacity[16, 16, 16] = 3.0
    run = allocate(source, capacity, [1, 2, 4], workers=workers, local_first=False)
    unsupported = {k: v for k, v in run["audit"].items() if k != "records"}

    # Two overlapping source supports; this checks shared capacities, not RT.
    overlap = {}
    for name, ratio in (("equal", 1.0), ("unequal", 0.25)):
        source.fill(0)
        capacity.fill(1)
        source[12, 16, 16] = budget
        source[20, 16, 16] = budget * ratio
        run = allocate(source, capacity, list(range(1, 13)), workers=workers)
        overlap[name] = {k: v for k, v in run["audit"].items() if k != "records"}

    # FFT denominator guard deliberately retains unresolved extreme dynamic range.
    source.fill(0)
    capacity.fill(0)
    source[1, 1, 1] = 0.5
    capacity[2, 1, 1] = 1.0
    capacity[16, 16, 16] = 1e15
    run = allocate(source, capacity, [1], workers=workers, local_first=False)
    dynamic_range = {k: v for k, v in run["audit"].items() if k != "records"}

    # Demonstrate why the native sharp-k window cannot be a probability kernel.
    frequencies = [fft.fftfreq(n) * n for _ in range(3)]
    wave = np.sqrt(
        frequencies[0][:, None, None] ** 2
        + frequencies[1][None, :, None] ** 2
        + frequencies[2][None, None, :] ** 2
    )
    sharp_real = fft.ifftn((wave <= 4).astype(float)).real
    return {
        "isolated_source": isolated,
        "opaque_screen": shadow,
        "equal_angle_density_bias": anisotropy,
        "unreachable": unsupported,
        "overlapping_sources": overlap,
        "extreme_dynamic_range": dynamic_range,
        "sharp_k_real_kernel": {
            "minimum": float(sharp_real.min()),
            "negative_weight_sum": float(-sharp_real[sharp_real < 0].sum()),
            "total_weight": float(sharp_real.sum()),
        },
    }


def monotonicity_probes(workers=1):
    """Recreate fixed-seed failures without depending on external fixture files."""
    result = {}
    rng = np.random.default_rng(20261024)
    for _ in range(26):
        c = np.exp(rng.normal(0, 1.5, (5,) * 3))
        p = rng.exponential(25, c.shape) * (rng.random(c.shape) < 0.08)
        q = p.copy()
        added_at = tuple(rng.integers(0, 5, 3))
        q[added_at] += rng.uniform(1, 20)
    refinement = {}
    for rounds in (1, 2, 4, 16):
        a = allocate(p, c, [1, 2, 3], workers=workers, rounds_per_scale=rounds)
        b = allocate(q, c, [1, 2, 3], workers=workers, rounds_per_scale=rounds)
        difference = a["absorbed"] - b["absorbed"]
        location = np.unravel_index(difference.argmax(), c.shape)
        refinement[str(rounds)] = {
            "maximum_target_decrease": float(difference.max()),
            "target": [int(v) for v in location],
            "before": float(a["absorbed"][location]),
            "after": float(b["absorbed"][location]),
            "ledger_before": a["audit"]["ledger_residual"],
            "ledger_after": b["audit"]["ledger_residual"],
        }
    result["source_increase"] = {
        "seed": 20261024,
        "trial": 25,
        "added_at": [int(v) for v in added_at],
        "added": float((q - p).sum()),
        "rounds_per_scale_refinement": refinement,
    }
    rng = np.random.default_rng(20261025)
    for _ in range(22):
        c = np.exp(rng.normal(0, 1.5, (5,) * 3))
        p = rng.exponential(25, c.shape) * (rng.random(c.shape) < 0.08)
        factor = rng.uniform(1.01, 2)
    a = allocate(p, c, [1, 2, 3], workers=workers)
    b = allocate(p, c * factor, [1, 2, 3], workers=workers)
    difference = b["absorbed"] / (factor * c) - a["absorbed"] / c
    location = np.unravel_index(difference.argmax(), c.shape)
    result["uniform_capacity_increase"] = {
        "seed": 20261025,
        "trial": 21,
        "capacity_scale": float(factor),
        "maximum_ionized_fraction_increase": float(difference.max()),
        "target": [int(v) for v in location],
        "before": float(a["absorbed"][location] / c[location]),
        "after": float(b["absorbed"][location] / (factor * c[location])),
    }
    return result


def benchmark(sizes=(32, 64, 128), workers=4):
    """Compare actual allocation and filter times with the same threads/arrays."""
    results = []
    for n in sizes:
        rng = np.random.default_rng(1701 + n)
        capacity = np.exp(rng.normal(0, 0.5, (n,) * 3))
        source = np.zeros_like(capacity)
        locations = rng.random(source.shape) < 0.02
        source[locations] = np.exp(rng.normal(0, 1, int(locations.sum())))
        source *= 0.3 * capacity.sum() / source.sum()
        radii = [1, 2, 4, 8, 16][: int(np.log2(n))]
        entries = []
        for _ in range(3):
            baseline = es_filter_baseline(source, capacity, radii, workers)
            run = allocate(source, capacity, radii, workers=workers, rounds_per_scale=2)
            entries.append(
                {
                    "baseline": baseline,
                    "allocation": {
                        k: v for k, v in run["audit"].items() if k != "records"
                    },
                }
            )
        ratios = [
            r["allocation"]["seconds"] / r["baseline"]["seconds"] for r in entries
        ]
        results.append(
            {
                "grid": n,
                "threads": workers,
                "source_active_fraction": float(locations.mean()),
                "photons_over_capacity": 0.3,
                "radii": radii,
                "median_time_ratio_to_twofield_ES_filter": float(np.median(ratios)),
                "runs": entries,
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("fft_results.json"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()
    result = {
        "scope": "Research-only nonnegative matching; no radiative transfer guarantee",
        "probes": make_probe_results(args.workers),
        "monotonicity": monotonicity_probes(),
        "benchmark": [] if args.skip_benchmark else benchmark(workers=args.workers),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
