"""Training-only photon-load tables conditioned on native box geometry.

Mixed-geometry data can have different absorption efficiencies at the same
redshift and photon load. Tables therefore average independent realizations
within each geometry and epoch, never across resolutions. They are empirical
predictors: endpoint extrapolation does not impose physical full absorption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class GeometryTabulatedBudgetModel:
    """Piecewise-linear efficiencies with separate training geometry tables.

    Parameters
    ----------
    geometries
        Distinct pairs of log1p cell width and log1p box length, in cMpc.
    offsets
        Integer boundaries of each geometry's segment in the flat tables.
    knots, efficiencies
        Photon-load coordinates and bounded efficiencies. Knots increase
        strictly within each segment. Every value is derived from training.
    feature_count
        Native budget feature width, with geometry in columns two and three.
    fallback_knots, fallback_efficiencies
        Original geometry-agnostic training table for explicit unsupported
        geometry fallback, with the same redshift grouping as the legacy table.

    Notes
    -----
    A training geometry uses only its own table. A new geometry uses the
    original geometry-agnostic training table. This explicit fallback avoids
    assigning unvalidated geometric interpolation weights; its transfer
    accuracy is still not certified.
    Each table uses constant endpoint extrapolation in photon load. Exact
    full absorption occurs only where the learned table itself has eta=1.
    """

    geometries: np.ndarray
    offsets: np.ndarray
    knots: np.ndarray
    efficiencies: np.ndarray
    feature_count: int
    fallback_knots: np.ndarray
    fallback_efficiencies: np.ndarray
    output_mode: str = field(default="efficiency", init=False)
    budget_kind: str = field(default="geometry_tabulated", init=False)

    def __post_init__(self) -> None:
        """Copy and validate immutable tables without fitting parameters."""
        raw_offsets = np.asarray(self.offsets)
        if raw_offsets.dtype.kind not in "iu":
            raise ValueError("Geometry table offsets must be integers")
        self.geometries = np.array(self.geometries, dtype=np.float64, copy=True)
        self.offsets = np.array(raw_offsets, dtype=np.int64, copy=True)
        self.knots = np.array(self.knots, dtype=np.float64, copy=True)
        self.efficiencies = np.array(self.efficiencies, dtype=np.float64, copy=True)
        self.fallback_knots = np.array(self.fallback_knots, dtype=np.float64, copy=True)
        self.fallback_efficiencies = np.array(
            self.fallback_efficiencies, dtype=np.float64, copy=True
        )
        if (
            self.fallback_knots.ndim != 1
            or not self.fallback_knots.size
            or self.fallback_efficiencies.shape != self.fallback_knots.shape
            or not np.isfinite(self.fallback_knots).all()
            or not np.isfinite(self.fallback_efficiencies).all()
            or np.any(self.fallback_knots < 0)
            or np.any(np.diff(self.fallback_knots) <= 0)
            or np.any(
                (self.fallback_efficiencies < 0) | (self.fallback_efficiencies > 1)
            )
            or type(self.feature_count) is not int
            or self.feature_count < 4
            or self.geometries.ndim != 2
            or not len(self.geometries)
            or self.geometries.shape[1] != 2
            or self.offsets.shape != (len(self.geometries) + 1,)
            or self.offsets[0] != 0
            or self.knots.ndim != 1
            or self.efficiencies.shape != self.knots.shape
            or self.offsets[-1] != len(self.knots)
            or np.any(np.diff(self.offsets) <= 0)
            or any(
                not np.isfinite(value).all()
                for value in (self.geometries, self.knots, self.efficiencies)
            )
            or np.any(self.geometries < 0)
            or len(np.unique(self.geometries, axis=0)) != len(self.geometries)
            or np.any(self.knots < 0)
            or np.any((self.efficiencies < 0) | (self.efficiencies > 1))
        ):
            raise ValueError("Invalid geometry-conditioned absorption table")
        for start, end in zip(self.offsets[:-1], self.offsets[1:], strict=True):
            if np.any(np.diff(self.knots[start:end]) <= 0):
                raise ValueError("Each geometry requires strictly increasing knots")
        for value in (
            self.geometries,
            self.offsets,
            self.knots,
            self.efficiencies,
            self.fallback_knots,
            self.fallback_efficiencies,
        ):
            value.setflags(write=False)

    @classmethod
    def from_training(
        cls, features: np.ndarray, efficiencies: np.ndarray
    ) -> GeometryTabulatedBudgetModel:
        """Fit means within geometry and epoch using training samples only.

        Parameters
        ----------
        features
            Native budget matrix: load, redshift, log1p cell width, log1p box.
        efficiencies
            Observed absorbed/min(source, atoms), one bounded training label
            per snapshot. Validation and test arrays are not accepted here.

        Returns
        -------
        GeometryTabulatedBudgetModel
            Immutable means for distinct training geometries and epochs.
        """
        x, y = np.asarray(features), np.asarray(efficiencies)
        if (
            x.dtype.kind not in "fiu"
            or y.dtype.kind not in "fiu"
            or x.ndim != 2
            or not len(x)
            or x.shape[1] < 4
            or y.shape != (len(x),)
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
            or np.any(x[:, :4] < 0)
            or np.any((y < 0) | (y > 1))
        ):
            raise ValueError("Invalid geometry-budget training arrays")
        x, y = x.astype(np.float64), y.astype(np.float64)
        geometries = np.unique(x[:, 2:4], axis=0)
        knots, values, offsets = [], [], [0]
        for geometry in geometries:
            selected = np.all(x[:, 2:4] == geometry, axis=1)
            groups = [selected & (x[:, 1] == z) for z in np.unique(x[selected, 1])]
            coords = np.array([np.mean(x[group, 0]) for group in groups])
            labels = np.array([np.mean(y[group]) for group in groups])
            unique = np.unique(coords)
            knots.extend(unique)
            values.extend(np.mean(labels[coords == knot]) for knot in unique)
            offsets.append(len(knots))
        groups = [x[:, 1] == z for z in np.unique(x[:, 1])]
        coordinates = np.array([np.mean(x[group, 0]) for group in groups])
        labels = np.array([np.mean(y[group]) for group in groups])
        fallback_knots = np.unique(coordinates)
        fallback_values = np.array(
            [np.mean(labels[coordinates == knot]) for knot in fallback_knots]
        )
        return cls(
            geometries,
            np.array(offsets),
            np.array(knots),
            np.array(values),
            x.shape[1],
            fallback_knots,
            fallback_values,
        )

    def predict(
        self, features: np.ndarray, *, baseline: np.ndarray | None = None
    ) -> np.ndarray:
        """Interpolate learned efficiencies from native inputs only.

        Parameters
        ----------
        features
            Native budget features, without any teacher field or reservoir.
        baseline
            Unused compatibility parameter.

        Returns
        -------
        ndarray
            One efficiency in [0,1] per input row. Native redshift is not used
            during prediction; photon load controls interpolation in each table.
        """
        x = np.asarray(features)
        if (
            x.dtype.kind not in "fiu"
            or x.ndim != 2
            or x.shape[1] != self.feature_count
            or not np.isfinite(x).all()
            or np.any(x[:, :4] < 0)
        ):
            raise ValueError("Invalid native geometry-budget features")
        result = np.empty(len(x), dtype=np.float64)
        for row, query in enumerate(x.astype(np.float64)):
            exact = np.flatnonzero(np.all(self.geometries == query[2:4], axis=1))
            if len(exact):
                index = int(exact[0])
                start, end = self.offsets[index : index + 2]
                result[row] = np.interp(
                    query[0], self.knots[start:end], self.efficiencies[start:end]
                )
            else:
                result[row] = np.interp(
                    query[0], self.fallback_knots, self.fallback_efficiencies
                )
        return result

    def save(self, path: str | Path) -> None:
        """Save immutable geometry and efficiency tables as pickle-free NPZ.

        Parameters
        ----------
        path
            Destination file in an existing directory.
        """
        np.savez_compressed(
            path,
            schema=2,
            kind="geometry_tabulated_efficiency",
            geometries=self.geometries,
            offsets=self.offsets,
            knots=self.knots,
            efficiencies=self.efficiencies,
            feature_count=self.feature_count,
            fallback_knots=self.fallback_knots,
            fallback_efficiencies=self.fallback_efficiencies,
        )

    @classmethod
    def load(cls, path: str | Path) -> GeometryTabulatedBudgetModel:
        """Load exact table parameters without fitting or data access.

        Parameters
        ----------
        path
            Archive written by ``save``.

        Returns
        -------
        GeometryTabulatedBudgetModel
            Restored predictor with validated parameters.
        """
        with np.load(path, allow_pickle=False) as saved:
            if (
                int(saved["schema"]) not in (1, 2)
                or str(saved["kind"]) != "geometry_tabulated_efficiency"
                or "fallback_knots" not in saved
                or "fallback_efficiencies" not in saved
            ):
                raise ValueError("Unsupported geometry-budget schema")
            return cls(
                saved["geometries"],
                saved["offsets"],
                saved["knots"],
                saved["efficiencies"],
                int(saved["feature_count"]),
                saved["fallback_knots"],
                saved["fallback_efficiencies"],
            )
