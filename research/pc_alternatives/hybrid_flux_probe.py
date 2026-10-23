"""Research-only exact-face vacuum flux and a certified monopole error probe.

This evaluates incident flux in a separately certified common-visible cavity.
It does not evolve a front, infer visibility from ES labels, or apply opacity.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import fft


def rectangular_face_kernel(shape, axis):
    """Fraction of an isotropic source crossing a positive-axis unit face.

    Index offset k corresponds to face centre k+1/2 along its normal, and k
    along either tangent. The offset support covers a zero-padded source box.
    Signed solid angle is used: a vector normal flux may be negative.
    """
    offset = []
    for dimension, n in enumerate(shape):
        indices = np.arange(2 * n)
        values = np.where(indices < n, indices, indices - 2 * n).astype(float)
        offset.append(values.reshape((1,) * dimension + (-1,) + (1,) * (2 - dimension)))
    normal = offset[axis] + 0.5
    tangent = [offset[d] for d in range(3) if d != axis]

    def corner(u, v):
        return np.arctan(u * v / (normal * np.sqrt(u * u + v * v + normal * normal)))

    return (
        corner(tangent[0] + 0.5, tangent[1] + 0.5)
        - corner(tangent[0] - 0.5, tangent[1] + 0.5)
        - corner(tangent[0] + 0.5, tangent[1] - 0.5)
        + corner(tangent[0] - 0.5, tangent[1] - 0.5)
    ) / (4 * np.pi)


class IsolatedFaceFlux:
    """Three staggered convolutions with cached exact solid-angle kernels.

    A source is a point at a cell centre. Rates produce face-integrated photon
    rates; photon increments produce integrated counts. Both have the same
    geometric operator. All sources in the array contribute, without opacity.
    """

    def __init__(self, shape, workers=4):
        self.shape = tuple(shape)
        self.padded_shape = tuple(2 * n for n in shape)
        self.workers = workers
        self.spectra = [
            fft.rfftn(rectangular_face_kernel(shape, axis), workers=workers)
            for axis in range(3)
        ]

    def __call__(self, source):
        started = time.perf_counter()
        source = np.asarray(source, dtype=float)
        if (
            source.shape != self.shape
            or np.any(source < 0)
            or not np.isfinite(source).all()
        ):
            raise ValueError(
                "Require finite nonnegative counts with the prepared shape"
            )
        padded = np.zeros(self.padded_shape)
        padded[tuple(slice(0, n) for n in self.shape)] = source
        spectrum = fft.rfftn(padded, workers=self.workers)
        faces = []
        for axis, weights in enumerate(self.spectra):
            field = fft.irfftn(
                spectrum * weights, s=self.padded_shape, workers=self.workers
            )
            indices = [np.arange(n) for n in self.shape]
            indices[axis] = np.r_[self.padded_shape[axis] - 1, indices[axis]]
            faces.append(field[np.ix_(*indices)])
        divergence = sum(np.diff(face, axis=axis) for axis, face in enumerate(faces))
        return {
            "faces": faces,
            "divergence": divergence,
            "audit": {
                "source_total": float(source.sum()),
                "divergence_total": float(divergence.sum()),
                "max_cell_divergence_error": float(np.max(np.abs(divergence - source))),
                "l1_cell_divergence_error": float(np.abs(divergence - source).sum()),
                "fft_calls_cached": 4,
                "kernel_fft_calls_setup": 3,
                "seconds_cached": time.perf_counter() - started,
            },
        }


def boundary_stats(faces, cavity):
    """Integrate outward flux and report negative incident flux at boundary faces."""
    total, negative, minimum, face_count = 0.0, 0.0, np.inf, 0
    for axis, flux in enumerate(faces):
        pads = [(0, 0)] * 3
        pads[axis] = (1, 1)
        padded = np.pad(cavity.astype(np.int8), pads)
        incidence = -np.diff(padded, axis=axis)
        outward = flux[incidence != 0] * incidence[incidence != 0]
        total += float(outward.sum())
        negative += float(-outward[outward < 0].sum())
        minimum = min(minimum, float(outward.min(initial=np.inf)))
        face_count += outward.size
    return {
        "net_outward_flux": total,
        "negative_outward_flux": negative,
        "minimum_boundary_face_flux": minimum,
        "boundary_face_count": face_count,
    }


def common_visibility_box_certificate(source, cavity):
    """Check every voxel-boundary plane against the source bounding box.

    A positive margin and source containment certify one outward crossing per
    source ray for a watertight represented cavity. Manifold topology is an
    additional caller assumption; this does not repair jagged ES geometry.
    """
    points = np.argwhere(source > 0)
    if len(points) == 0:
        return {"certified": True, "source_count": 0, "minimum_plane_margin": None}
    contained = bool(np.all(cavity[tuple(points.T)]))
    centre = (points.min(axis=0) + points.max(axis=0)) * 0.5
    half_size = (points.max(axis=0) - points.min(axis=0)) * 0.5
    minimum = np.inf
    for axis in range(3):
        pads = [(0, 0)] * 3
        pads[axis] = (1, 1)
        incidence = -np.diff(np.pad(cavity.astype(np.int8), pads), axis=axis)
        at = np.nonzero(incidence)
        sign = incidence[at]
        margin = sign * (at[axis] - 0.5 - centre[axis]) - half_size[axis]
        minimum = min(minimum, float(margin.min(initial=np.inf)))
    return {
        "certified": contained and minimum > 0,
        "source_count": len(points),
        "source_containment": contained,
        "source_box_centre": centre.tolist(),
        "source_box_half_size": half_size.tolist(),
        "minimum_plane_margin": minimum,
    }


def direct_flux(targets, sources, luminosity):
    """Direct vacuum point-source vector flux for the small multipole test."""
    result = np.zeros_like(targets)
    for point, rate in zip(sources, luminosity, strict=True):
        offset = targets - point
        result += (
            rate * offset / (4 * np.pi * np.linalg.norm(offset, axis=1)[:, None] ** 3)
        )
    return result


def monopole_probe():
    """Check a uniform rigorous centroid bound, not a statistical error estimate."""
    rng = np.random.default_rng(41271)
    source = rng.normal(size=(64, 3))
    luminosity = rng.uniform(0.1, 2, 64)
    source -= np.average(source, axis=0, weights=luminosity)
    source /= np.linalg.norm(source, axis=1).max()
    second_moment = float(np.sum(luminosity * np.sum(source * source, axis=1)))
    direction = rng.normal(size=(2048, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    total = luminosity.sum()
    results = []
    for opening in (0.05, 0.1, 0.2, 0.4):
        distance = 1 / opening
        targets = direction * distance
        exact = direct_flux(targets, source, luminosity)
        mono = total * direction / (4 * np.pi * distance**2)
        error = np.linalg.norm(exact - mono, axis=1)
        bound = 3 * second_moment / (4 * np.pi * (distance - 1) ** 4)
        results.append(
            {
                "a_over_distance": opening,
                "max_absolute_error": float(error.max()),
                "uniform_absolute_bound": float(bound),
                "max_error_over_bound": float(error.max() / bound),
                "max_error_over_monopole_magnitude": float(
                    error.max() / (total / (4 * np.pi * distance**2))
                ),
                "uniform_bound_over_monopole_magnitude": float(
                    bound / (total / (4 * np.pi * distance**2))
                ),
            }
        )
    return {
        "source_count": 64,
        "directions_tested": 2048,
        "centroid_residual": np.average(source, axis=0, weights=luminosity).tolist(),
        "results": results,
    }


def validation(workers=1):
    """Small exact-flux probes; no claim of a tested finite-step front update."""
    shape = (16, 16, 16)
    evaluator = IsolatedFaceFlux(shape, workers)
    source = np.zeros(shape)
    source[7, 7, 7] = 1
    single = evaluator(source)
    cube = np.zeros(shape, dtype=bool)
    cube[3:13, 3:13, 3:13] = True
    single_stats = {**single["audit"], **boundary_stats(single["faces"], cube)}
    single_stats["common_visibility"] = common_visibility_box_certificate(source, cube)
    rng = np.random.default_rng(924)
    source.fill(0)
    source[4:12, 4:12, 4:12] = rng.exponential(size=(8,) * 3)
    multiple = evaluator(source)
    multiple_stats = {**multiple["audit"], **boundary_stats(multiple["faces"], cube)}
    multiple_stats["common_visibility"] = common_visibility_box_certificate(
        source, cube
    )

    # Different cavity labels alone do not remove another cavity's sources.
    left = np.zeros(shape, dtype=bool)
    left[1:7, 4:12, 4:12] = True
    source.fill(0)
    source[3, 7, 7] = 1
    own = evaluator(source)
    source[12, 7, 7] = 1
    mixed = evaluator(source)
    contamination = {
        "own_source": boundary_stats(own["faces"], left),
        "global_all_sources": boundary_stats(mixed["faces"], left),
        "max_face_difference": float(
            max(
                np.max(np.abs(a - b))
                for a, b in zip(own["faces"], mixed["faces"], strict=True)
            )
        ),
        "meaning": "An outside source has zero net flux through this closed cavity but changes incoming/outgoing face flux; labels cannot repair it.",
    }
    return {
        "scope": "Static vacuum face-flux evaluator for an already certified common-visible cavity; no front evolution",
        "single_source_convex_cube": single_stats,
        "multiple_source_convex_cube": multiple_stats,
        "outside_source_contamination": contamination,
        "multipole": monopole_probe(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("hybrid_flux_results.json"))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    result = validation(args.workers)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
