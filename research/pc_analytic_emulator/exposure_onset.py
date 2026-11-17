"""Learn when cells become transparent from offline dose histories.

Inference receives native coeval arrays and a frozen regression tree ensemble.
The fitted onset initializes the same exposure pooling layers for every snapshot.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter1d

from research.pc_analytic_emulator.exposure_features import exposure_features

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

FEATURE_COUNT = 114


def _octants(values: np.ndarray, radius: int) -> np.ndarray:
    views = [values]
    left, right = -(radius // 2), (radius - 1) // 2
    for axis in range(3):
        views = [
            uniform_filter1d(view, radius, axis=axis, mode="wrap", origin=offset)
            for view in views
            for offset in (left, right)
        ]
    return np.stack(views)


def _cone_descriptors(
    photons: np.ndarray,
    atoms: np.ndarray,
    excess: np.ndarray,
    missing: np.ndarray,
) -> list[np.ndarray]:
    size = len(photons)
    features = []
    peak = np.zeros((8, *photons.shape))
    longest = np.zeros_like(peak)
    early = np.zeros_like(peak)
    for radius in (2, 3, 4, 6, size // 2):
        p, a, e, d = (
            _octants(values, radius) for values in (photons, atoms, excess, missing)
        )
        ratio = p / a
        spill = e / (d + 0.15)
        peak = np.maximum(peak, ratio)
        longest = np.where(ratio > 1, radius / size, longest)
        early = np.maximum(early, np.maximum(ratio - 1, 0) / radius)
        features.extend(
            (
                np.log1p(ratio.max(axis=0)),
                np.log1p(ratio.min(axis=0)),
                np.log1p(ratio.std(axis=0)),
                np.maximum(p - a, 0).mean(axis=0),
                np.log1p(spill.mean(axis=0)),
                np.log1p(spill.max(axis=0)),
            )
        )
    features.extend(
        (
            np.log1p(peak.mean(axis=0)),
            np.log1p(peak.max(axis=0)),
            np.log1p(peak.std(axis=0)),
            longest.mean(axis=0),
            longest.max(axis=0),
            early.mean(axis=0),
        )
    )
    return features


def onset_features(
    source: np.ndarray,
    capacity: np.ndarray,
    native_es_xhi: np.ndarray,
    redshift: float,
    box_len: float,
) -> tuple[NDArray[np.float32], np.ndarray, np.ndarray]:
    """Build per-cell onset features and two deterministic initial schedules.

    Parameters
    ----------
    source, capacity, native_es_xhi
        Equal cubic shapes with nonnegative photon counts, positive atom
        capacities, and bounded native excursion-set neutral fractions.
        Photon and atom counts share the same unit.
    redshift, box_len
        Snapshot redshift and box side in comoving Mpc.

    Returns
    -------
    features, local, early
        Feature rows in C-order, approximate saturation from local sources
        alone, and the onset after two exposure-pooling layers.
    """
    if any(np.iscomplexobj(v) for v in (source, capacity, native_es_xhi)):
        raise ValueError("Native onset features require real grids")
    photons, atoms, es = (
        np.asarray(values, dtype=np.float64)
        for values in (source, capacity, native_es_xhi)
    )
    if (
        photons.ndim != 3
        or len(set(photons.shape)) != 1
        or len(photons) < 14
        or photons.shape != atoms.shape
        or photons.shape != es.shape
        or any(not np.isfinite(v).all() for v in (photons, atoms, es))
        or np.any(photons < 0)
        or np.any(atoms <= 0)
        or np.any((es < 0) | (es > 1))
        or not np.isfinite(redshift)
        or redshift < 0
        or not np.isfinite(box_len)
        or box_len <= 0
    ):
        raise ValueError("Invalid native grids or geometry for onset features")

    mean = np.mean(atoms)
    p, a = photons / mean, atoms / mean
    excess = np.maximum(p - a, 0)
    missing = np.maximum(a - p, 0)
    ratio = p / a
    columns = [
        np.log(a),
        np.log1p(p),
        np.log1p(ratio),
        es,
        np.maximum(1 - ratio, 0) - es,
    ]
    for sigma in (0.6, 1.0, 1.6, 2.5, 4.0, 6.0):
        near_p, near_a, near_excess, near_missing, near_es = [
            gaussian_filter(values, sigma, mode="wrap")
            for values in (p, a, excess, missing, es)
        ]
        columns.extend(
            (
                np.log1p(near_excess / a),
                np.log1p(near_p / near_a),
                np.log(near_a),
                np.log1p(near_missing),
                near_es,
                gaussian_filter(np.maximum(near_p - near_a, 0), sigma / 2, mode="wrap"),
            )
        )
    for radius in (5.0, 12.0, 25.0):
        sigma = radius / (box_len / len(p))
        columns.extend(
            (
                np.log1p(gaussian_filter(excess, sigma, mode="wrap")),
                np.log1p(
                    gaussian_filter(p, sigma, mode="wrap")
                    / gaussian_filter(a, sigma, mode="wrap")
                ),
            )
        )
    columns.extend(_cone_descriptors(p, a, excess, missing))
    indicators = (
        (redshift - 8.0) / 2.0,
        np.log(box_len / 50.0),
        np.log(box_len / len(p) / 3.125),
        np.log1p(np.mean(p)),
        np.mean(es * a),
        np.mean(excess),
        np.mean(missing),
        np.mean(es),
        np.std(a),
        np.std(p),
    )
    columns.extend(np.full_like(a, indicator) for indicator in indicators)

    local = np.divide(a, p, out=np.ones_like(p), where=p > a)
    columns.append(local)
    schedule = local
    for _ in range(2):
        _, near, far, schedule, _ = exposure_features(
            p, a, depth=1, gains=(1.0, 1.0), onset=schedule
        )
        dose = np.log1p(near + far)
        columns.extend((schedule, dose, np.log1p(near), np.log1p(far)))
        for sigma in (0.8, 1.6, 3.2):
            columns.extend(
                (
                    gaussian_filter(schedule, sigma, mode="wrap"),
                    gaussian_filter(dose, sigma, mode="wrap"),
                )
            )
    features = np.column_stack([column.ravel() for column in columns]).astype(
        np.float32
    )
    if features.shape[1] != FEATURE_COUNT or not np.isfinite(features).all():
        raise FloatingPointError("Onset feature count or numeric range changed")
    return features, local, schedule


@dataclass
class OnsetHead:
    """Portable array representation of learned saturation-time residual trees.

    Parameters
    ----------
    roots, feature, threshold, left, right, leaves, value
        Packed tree structure and numeric split/leaf parameters.
    intercept
        Constant regression baseline before adding the learned trees.
    metadata
        Training provenance and cohort identifiers.
    """

    roots: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    left: np.ndarray
    right: np.ndarray
    leaves: np.ndarray
    value: np.ndarray
    intercept: float
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """Validate packed node shapes and forward-only pointers."""
        count = len(self.value)
        packed = (
            self.feature,
            self.threshold,
            self.left,
            self.right,
            self.leaves,
            self.value,
        )
        if (
            not count
            or self.roots.ndim != 1
            or not len(self.roots)
            or any(
                values.shape != (count,) or not np.isfinite(values).all()
                for values in packed
            )
            or not np.isfinite(self.intercept)
            or self.leaves.dtype != np.bool_
            or any(
                not np.issubdtype(values.dtype, np.integer)
                for values in (self.roots, self.feature, self.left, self.right)
            )
            or np.any((self.roots < 0) | (self.roots >= count))
        ):
            raise ValueError("Invalid packed onset regression trees")
        branch = ~self.leaves
        current = np.flatnonzero(branch)
        # Exported nodes are parent-first. Strictly larger child indices
        # make cycles impossible even in a malformed model archive.
        if (
            np.any((self.feature[branch] < 0) | (self.feature[branch] >= FEATURE_COUNT))
            or np.any((self.left[branch] <= current) | (self.left[branch] >= count))
            or np.any((self.right[branch] <= current) | (self.right[branch] >= count))
        ):
            raise ValueError("Onset regression node offsets are invalid")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict the learned onset residual for each finite feature row."""
        if (
            features.ndim != 2
            or features.shape[1] != FEATURE_COUNT
            or not np.isfinite(features).all()
        ):
            raise ValueError("Expected a finite matrix of onset features")
        result = np.full(len(features), self.intercept, dtype=np.float64)
        rows = np.arange(len(features))
        for root in self.roots:
            node = np.full(len(features), root, dtype=np.int32)
            while True:
                active = ~self.leaves[node]
                if not np.any(active):
                    break
                branch = node[active]
                go_left = (
                    features[rows[active], self.feature[branch]]
                    <= self.threshold[branch]
                )
                node[active] = np.where(go_left, self.left[branch], self.right[branch])
            result += self.value[node]
        return result

    def initialize(
        self,
        source: np.ndarray,
        capacity: np.ndarray,
        native_es_xhi: np.ndarray,
        redshift: float,
        box_len: float,
    ) -> np.ndarray:
        """Estimate a bounded schedule for the subsequent exposure layers."""
        features, local, early = onset_features(
            source, capacity, native_es_xhi, redshift, box_len
        )
        return np.clip(early + self.predict(features).reshape(early.shape), 0, local)

    def save(self, path: Path) -> None:
        """Serialize the trees and JSON provenance to a new NPZ file."""
        with path.open("xb") as stream:
            np.savez_compressed(
                stream,
                roots=self.roots,
                feature=self.feature,
                threshold=self.threshold,
                left=self.left,
                right=self.right,
                leaves=self.leaves,
                value=self.value,
                intercept=np.asarray(self.intercept),
                metadata=np.asarray(json.dumps(self.metadata, allow_nan=False)),
                schema=np.asarray(1),
            )

    @classmethod
    def load(cls, path: Path) -> OnsetHead:
        """Load frozen numeric trees without Python object deserialization."""
        with np.load(path, allow_pickle=False) as data:
            if data["schema"].item() != 1:
                raise ValueError("Unsupported onset model format")
            return cls(
                data["roots"],
                data["feature"],
                data["threshold"],
                data["left"],
                data["right"],
                data["leaves"],
                data["value"],
                float(data["intercept"].item()),
                json.loads(data["metadata"].item()),
            )


def _pack_model(fitted: Any, metadata: dict[str, Any]) -> OnsetHead:
    roots = []
    cursor = 0
    columns: dict[str, list[np.ndarray]] = {
        n: [] for n in ("feature", "threshold", "left", "right", "leaves", "value")
    }
    for stage in fitted._predictors:
        nodes = stage[0].nodes
        roots.append(cursor)
        for name, value in (
            ("feature", nodes["feature_idx"]),
            ("threshold", nodes["num_threshold"]),
            ("left", nodes["left"] + cursor),
            ("right", nodes["right"] + cursor),
            ("leaves", nodes["is_leaf"].astype(bool)),
            ("value", nodes["value"]),
        ):
            columns[name].append(value)
        cursor += len(nodes)
    return OnsetHead(
        roots=np.asarray(roots, dtype=np.int32),
        **{name: np.concatenate(items) for name, items in columns.items()},
        intercept=float(fitted._baseline_prediction.item()),
        metadata=metadata,
    )


def fit_onset(
    observations: Sequence[
        tuple[np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray]
    ],
    *,
    seeds: Sequence[int],
    sample_cells: int = 4096,
    steps: int = 220,
) -> OnsetHead:
    """Fit bounded schedule residuals using offline observed or modeled saturation times.

    Parameters
    ----------
    observations
        Tuples of native photons, atom capacities, native ES neutral fraction,
        redshift, box size and observed or approximated training onset. The
        last cube is a supervised target, unavailable at inference.
    seeds
        One realization seed per observation for reproducibility.
    sample_cells
        Maximum uniformly sampled cells per training snapshot.
    steps
        Number of supervised histogram gradient boosting iterations.

    Returns
    -------
    OnsetHead
        Pickle-free inference tree ensemble.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    if (
        not observations
        or len(observations) != len(seeds)
        or min(sample_cells, steps) <= 0
    ):
        raise ValueError(
            "Provide labeled training snapshots, seeds and positive controls"
        )
    random = np.random.default_rng(28)
    features, residuals, weights = [], [], []
    for source, capacity, es, z, box, onset in observations:
        x, _, early = onset_features(source, capacity, es, z, box)
        if (
            onset.shape != early.shape
            or not np.isfinite(onset).all()
            or np.any((onset < 0) | (onset > 1))
        ):
            raise ValueError(
                "Offline onset labels must lie in [0,1] and match the grid"
            )
        target = onset.ravel() - early.ravel()
        sample = random.choice(
            len(target), min(len(target), sample_cells), replace=False
        )
        features.append(x[sample])
        residuals.append(target[sample])
        weights.append((1 + 2 * np.abs(target[sample])) * sample_cells / len(sample))
    estimator = HistGradientBoostingRegressor(
        max_iter=steps,
        max_leaf_nodes=27,
        learning_rate=0.095,
        l2_regularization=1.5,
        min_samples_leaf=100,
        early_stopping=False,
        random_state=29,
    ).fit(
        np.concatenate(features),
        np.concatenate(residuals),
        sample_weight=np.concatenate(weights),
    )
    model = _pack_model(
        estimator,
        {
            "training_seeds": sorted(set(seeds)),
            "training_snapshots": len(seeds),
            "fitting_iterations": steps,
            "cells_per_snapshot": sample_cells,
        },
    )
    expected = estimator.predict(features[0])
    reproduced = model.predict(features[0])
    if not np.allclose(expected, reproduced, rtol=0.0, atol=4e-15):
        raise RuntimeError("Packed onset trees changed supervised predictions")
    return model
