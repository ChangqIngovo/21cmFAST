"""Geometry certificates, not a photoionization solver.

Fixed Cartesian, non-periodic, piecewise-constant opacity; no emission/scattering
along an already emitted characteristic. All opacities remain frozen. A packet
has an exact subcell position and direction. Unit cell width is used.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def levels(field: np.ndarray) -> list[np.ndarray]:
    n = field.shape[0]
    assert field.shape == (n, n, n) and n > 0 and n & (n - 1) == 0
    result = [field]
    while n > 1:
        field = field.reshape(n // 2, 2, n // 2, 2, n // 2, 2).max(axis=(1, 3, 5))
        result.append(field)
        n //= 2
    return result


def _inside_cell(point: np.ndarray, direction: np.ndarray, n: int):
    # A face point belongs to its downstream cell. nextafter only determines the
    # integer cell; it never changes the geometric position or flight length.
    probe = np.where(direction < 0, np.nextafter(point, -np.inf), point)
    cell = np.floor(probe).astype(int)
    return cell if np.all((cell >= 0) & (cell < n)) else None


def _exit_length(point, direction, lower, upper):
    lengths = np.full(3, np.inf)
    positive, negative = direction > 0, direction < 0
    lengths[positive] = (upper[positive] - point[positive]) / direction[positive]
    lengths[negative] = (lower[negative] - point[negative]) / direction[negative]
    return float(lengths.min())


def trace(opacity, origin, direction, distance, *, skip_threshold=None):
    """Pure-absorption unit packet; optionally skip max-kappa-bounded blocks."""
    opacity = np.asarray(opacity, dtype=float)
    assert np.isfinite(opacity).all() and (opacity >= 0).all()
    n = opacity.shape[0]
    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    tree = levels(opacity) if skip_threshold is not None else None
    absorbed = np.zeros_like(opacity)
    weight, flight = 1.0, 0.0
    segments, jumps, skipped_tau_bound = [], 0, 0.0
    for _ in range(10_000):
        point = origin + direction * flight
        cell = _inside_cell(point, direction, n)
        if cell is None or flight >= distance:
            break
        selected = 0
        if tree is not None:
            for level in range(len(tree) - 1, -1, -1):
                if tree[level][tuple(cell // (1 << level))] <= skip_threshold:
                    selected = level
                    break
        width = 1 << selected
        lower = (cell // width) * width
        upper = lower + width
        segment = min(_exit_length(point, direction, lower, upper), distance - flight)
        if segment <= 1e-13:
            # The computed intersection can fall one ulp on the upstream side.
            # Move only the parameter, by one ulp; no finite-length path is omitted.
            flight = np.nextafter(flight, np.inf)
            continue
        kappa_bound = float(tree[selected][tuple(cell // width)]) if tree is not None else None
        skipped = tree is not None and kappa_bound <= skip_threshold
        if skipped:
            skipped_tau_bound += kappa_bound * segment
            jumps += 1
        else:
            loss = weight * -math.expm1(-opacity[tuple(cell)] * segment)
            absorbed[tuple(cell)] += loss
            weight -= loss
        segments.append((tuple(int(i) for i in cell), segment, selected, skipped))
        flight += segment
    else:
        raise RuntimeError("Traversal failed to progress")
    return {
        "absorbed": absorbed,
        "remaining": weight,
        "distance": flight,
        "segments": segments,
        "jumps": jumps,
        "skipped_tau_bound": skipped_tau_bound,
        "final_position": origin + direction * flight,
    }


def test_zero_skip():
    n = 32
    opacity = np.zeros((n, n, n))
    opacity[22:24, 8:27, 4:25] = 0.7
    cases = []
    for origin, direction in [((.5, 3.5, 4.5), (1, .37, .21)),
                              ((30.5, 28.5, 27.5), (-1, -.6, -.4)),
                              ((1.5, 1.5, 1.5), (1, 1, 1))]:
        reference = trace(opacity, origin, direction, 100)
        skipped = trace(opacity, origin, direction, 100, skip_threshold=0)
        error = np.max(np.abs(reference["absorbed"] - skipped["absorbed"]))
        assert error < 1e-12
        assert abs(reference["remaining"] - skipped["remaining"]) < 1e-12
        assert np.allclose(reference["final_position"], skipped["final_position"], atol=1e-12)
        cases.append({"direction": direction, "max_absorption_error": float(error),
                      "fine_segments": len(reference["segments"]),
                      "accelerated_segments": len(skipped["segments"]),
                      "zero_block_jumps": skipped["jumps"]})
    return cases


def test_translation_cache():
    opacity = np.zeros((32, 32, 32))
    origin = np.array((1.5, 1.5, 1.5))
    shift = np.array((4, 6, 3))
    errors = []
    for direction in [(1, .37, .21), (-.17, .73, 1), (1, 1, 1)]:
        # Keep both paths within the domain; the offset geometry is then identical.
        direction = np.abs(direction)
        a = trace(opacity, origin, direction, 12)
        b = trace(opacity, origin + shift, direction, 12)
        assert len(a["segments"]) == len(b["segments"])
        max_error = 0.0
        for sa, sb in zip(a["segments"], b["segments"]):
            assert np.array_equal(np.array(sa[0]) + shift, sb[0])
            max_error = max(max_error, abs(sa[1] - sb[1]))
        assert max_error < 1e-12
        errors.append(max_error)
    return {"max_segment_length_error": float(max(errors)), "directions": len(errors)}


def test_merge_counterexample():
    opacity = np.zeros((8, 8, 8))
    opacity[4, 1, 0] = 100.0
    direction = (1, .2, 0)
    # Both origins occupy voxel (0,0,0), with identical direction and time.
    # Their parallel trajectories encounter different downstream cells.
    low = trace(opacity, (.5, .1, .5), direction, 6)
    high = trace(opacity, (.5, .9, .5), direction, 6)
    merged = trace(opacity, (.5, .5, .5), direction, 6)
    true_absorption = .5 * (low["absorbed"] + high["absorbed"])
    error = abs(float(true_absorption.sum()) - float(merged["absorbed"].sum()))
    assert error > .49
    return {"separate_absorbed": float(true_absorption.sum()),
            "merged_absorbed": float(merged["absorbed"].sum()),
            "absolute_error": error,
            "lesson": "Same cell and direction do not certify identical characteristics"}


def test_bounded_skip():
    opacity = np.full((32, 32, 32), 1e-5)
    opacity[22:24, 8:27, 4:25] = .7
    origin, direction = (.5, 3.5, 4.5), (1, .37, .21)
    a = trace(opacity, origin, direction, 100)
    b = trace(opacity, origin, direction, 100, skip_threshold=1e-5)
    total_variation_l1 = float(np.abs(a["absorbed"] - b["absorbed"]).sum()
                               + abs(a["remaining"] - b["remaining"]))
    l1_bound = 2 * -math.expm1(-b["skipped_tau_bound"])
    assert total_variation_l1 <= l1_bound + 1e-12
    assert abs(b["absorbed"].sum() + b["remaining"] - 1) < 1e-12
    return {"absorption_plus_residual_L1_error": total_variation_l1,
            "proved_frozen_opacity_L1_bound": l1_bound,
            "skipped_optical_depth_upper_bound": b["skipped_tau_bound"],
            "note": "Not a bound for nonlinear evolving chemistry"}


def main():
    result = {"scope": __doc__.strip(), "zero_opacity_skips": test_zero_skip(),
              "translation_cache": test_translation_cache(),
              "merge_counterexample": test_merge_counterexample(),
              "bounded_skip": test_bounded_skip()}
    output = Path(__file__).with_name("characteristic_certificates_results.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
