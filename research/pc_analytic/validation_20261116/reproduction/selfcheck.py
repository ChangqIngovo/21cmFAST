"""Check independent FFT and geometric ray diagnostics against exact identities."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from recompute import digest, geometric_mfp, spectra


def fft_check() -> float:
    """Compare full complex FFT to independent multiplicity-weighted real FFT."""
    rng = np.random.default_rng(444)
    maximum = 0.0
    for n in (7, 8, 16):
        length = 50.0
        teacher, prediction = rng.normal(size=(2, n, n, n))
        measured = spectra({"teacher": teacher, "prediction": prediction}, length, 0.5)
        xy, z = np.fft.fftfreq(n) * n, np.fft.rfftfreq(n) * n
        norm = np.sqrt(
            xy[:, None, None] ** 2 + xy[None, :, None] ** 2 + z[None, None, :] ** 2
        )
        chosen = (norm > 0) & (norm < n / 4)
        labels = np.floor(norm[chosen] + 0.5).astype(int)
        multiplicity = np.full((n, n, n // 2 + 1), 2.0)
        multiplicity[:, :, 0] = 1
        if n % 2 == 0:
            multiplicity[:, :, -1] = 1
        weights = multiplicity[chosen]
        counts = np.bincount(labels, weights=weights)
        occupied = np.flatnonzero(counts)
        for name, field in (("teacher", teacher), ("prediction", prediction)):
            transform = np.fft.rfftn(field - np.mean(field))[chosen]
            power = (
                np.bincount(labels, weights=weights * np.abs(transform) ** 2)[occupied]
                / counts[occupied]
                * length**3
                / n**6
            )
            expected = (
                measured["P_teacher"]
                if name == "teacher"
                else measured["comparisons"][name]["P"]
            )
            np.testing.assert_allclose(power, expected, rtol=1e-13, atol=1e-13)
            maximum = max(maximum, float(np.max(np.abs(power - expected))))
        np.testing.assert_array_equal(counts[occupied], measured["modes"])
    return maximum


def main() -> None:
    """Run exact slab, endpoint and optional C++ DDA cross-checks."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, help="Optional repository for C++ tracer comparison"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(4321)
    draws = rng.random((1024, 6))
    draws[:, 4] = 0  # Exactly negative z; permitted uniform-draw endpoint.
    field = np.ones((8, 8, 8))
    field[:, :, :4] = 0
    _, distances, capped = geometric_mfp(field, 16, "ionized", draws)
    selected = np.flatnonzero(field < 0.5)
    indices = selected[(draws[:, 0] * len(selected)).astype(int)]
    cells = np.array(np.unravel_index(indices, field.shape)).T
    expected = 2 * (cells[:, 2] + draws[:, 3])
    np.testing.assert_array_equal(distances, expected)
    assert not capped.any()
    full = geometric_mfp(np.zeros_like(field), 16, "ionized", draws)
    empty = geometric_mfp(np.zeros_like(field), 16, "neutral", draws)
    assert full[0]["status"] == "all_censored"
    assert np.all(full[1] == 16)
    assert empty[0]["status"] == "empty_phase"
    assert len(empty[1]) == 0
    result = {
        "complete": True,
        "axis_aligned_uniform_slab_rays": len(draws),
        "axis_aligned_slab_maximum_distance_error": float(
            np.max(np.abs(distances - expected))
        ),
        "full_and_empty_phase_tests_passed": True,
        "full_complex_vs_real_fft_maximum_absolute_error": fft_check(),
        "script_sha256": digest(Path(__file__)),
        "recompute_sha256": digest(Path(__file__).with_name("recompute.py")),
    }
    if args.source:
        sys.path.insert(0, str(args.source))
        from research.pc_coeval.mfp import measure_mfp

        random_field = rng.random((8, 8, 8))
        random_draws = rng.random((2048, 6))
        maximum = 0.0
        for phase in ("ionized", "neutral"):
            independent = geometric_mfp(random_field, 50, phase, random_draws)
            kernel = measure_mfp(
                random_field, 50, phase=phase, draws=random_draws, threads=1
            )
            np.testing.assert_allclose(
                independent[1], kernel["distance_cMpc"], rtol=1e-12, atol=1e-12
            )
            np.testing.assert_array_equal(independent[2], kernel["censored"])
            maximum = max(
                maximum, float(np.max(np.abs(independent[1] - kernel["distance_cMpc"])))
            )
        result["randomized_kernel_comparison_rays_per_phase"] = len(random_draws)
        result["randomized_cpp_DDA_maximum_distance_error_cMpc"] = maximum
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    logging.getLogger(__name__).info(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
