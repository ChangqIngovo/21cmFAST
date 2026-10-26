"""Isolate the native Cartesian-to-spherical geometry error analytically.

Uniform monochromatic opacity gives exact incoming columns. Therefore this
experiment contains neither column interpolation, chemistry nor multiple-source
competition. The reference absorbed fraction is one in the infinite medium;
the finite cube used here truncates at least 18 optical depths from the source.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss


def uniform_budget(cell_optical_depth: float) -> dict:
    """Evaluate native geometric absorption using exact homogeneous columns.

    Parameters
    ----------
    cell_optical_depth : float
        Monochromatic absorption coefficient times Cartesian cell width.
    """
    tau = float(cell_optical_depth)
    if not np.isfinite(tau) or tau < 0.1:
        raise ValueError("cell optical depth must be finite and at least 0.1")
    extent = max(12, int(np.ceil(18 / tau)))
    line = np.arange(-extent, extent + 1, dtype=float)
    # Slice the cube to keep memory bounded when cells become optically thin.
    y, z = np.meshgrid(line, line, indexing="ij")
    external = 0.0
    for x in line:
        radius = np.sqrt(x * x + y * y + z * z)
        major = np.maximum(np.abs(x), np.maximum(np.abs(y), np.abs(z)))
        mask = major > 0
        radius, major = radius[mask], major[mask]
        path = radius / major
        external += np.sum(
            np.exp(-tau * (radius - path / 2))
            * (-np.expm1(-tau * path))
            / (4 * np.pi * radius**2 * path)
        )
    native_source = -np.expm1(-tau / 2)
    nodes, w = leggauss(96)
    a, b = np.meshgrid(nodes, nodes, indexing="ij")
    jacobian = (1 + a * a + b * b) ** -1.5
    exact_source = (
        6
        * np.sum(
            w[:, None]
            * w[None, :]
            * jacobian
            * (-np.expm1(-tau / 2 * np.sqrt(1 + a * a + b * b)))
        )
        / (4 * np.pi)
    )
    return {
        "cell_optical_depth": tau,
        "native_geometry_residual": float(native_source + external - 1),
        "exact_source_cell_only_residual": float(exact_source + external - 1),
        "unabsorbed_tail_upper_bound": float(np.exp(-tau * (extent + 0.5))),
        "reference": "Unit absorption in an infinite uniform medium",
        "columns": "Analytic, no short-characteristic interpolation",
    }


def main() -> None:
    """Write reproducible geometry-only results to a fresh JSON file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [uniform_budget(tau) for tau in (16, 8, 4, 2, 1, 0.5, 0.25)]
    with args.output.open("x") as stream:
        json.dump({"scope": __doc__, "cases": rows}, stream, indent=2)


if __name__ == "__main__":
    main()
