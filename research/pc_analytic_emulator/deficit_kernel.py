"""Empirical kernel regression of photon absorption deficit from native inputs.

This optional estimator learns ``d = 1 - absorbed/min(photons, atoms)``. It adds
no transport closure: neither photon surplus nor a particular redshift forces
the predicted deficit to zero. Training-label floors regularize the logarithm;
they are not lower bounds on predicted deficits.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.special import expit

NATIVE_PREFIX = (
    "log1p_photons_per_atom",
    "redshift",
    "log1p_dx_cMpc",
    "log1p_box_cMpc",
    "log1p_density_std_over_mean",
    "log1p_source_std_over_mean_capacity",
    "native_es_Q_volume",
    "native_es_Q_mass",
    "native_es_xHII_std",
    "native_es_front_fraction",
)


def _real(value: np.ndarray, name: str) -> np.ndarray:
    """Reject complex, object and nonfinite inputs before conversion."""
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numeric values")
    return np.array(array, dtype=np.float64, copy=True)


def _native(value: np.ndarray, width: int) -> np.ndarray:
    """Validate the declared native-feature order and physical ranges."""
    array = _real(value, "Native features")
    if (
        array.ndim != 2
        or not len(array)
        or array.shape[1] != width
        or width < len(NATIVE_PREFIX)
        or np.any(array < 0)
        or np.any(array[:, 6:10] > 1 + 32 * np.finfo(float).eps)
    ):
        raise ValueError("Invalid native budget feature shape or physical range")
    return array


def _kernel(left: np.ndarray, right: np.ndarray, bandwidth: float) -> np.ndarray:
    """Evaluate an RBF using RMS distance in standardized native coordinates."""
    difference = left[:, None, :] - right[None, :, :]
    with np.errstate(over="ignore", invalid="raise"):
        distance2 = np.mean(difference * difference, axis=2)
    return np.exp(-0.5 * distance2 / bandwidth**2)


@dataclass
class KernelDeficitBudgetModel:
    """Store a training-only RBF fit to log or logit absorption deficit.

    Parameters
    ----------
    training_features, coefficients
        Standardized training coordinates and their fitted kernel coefficients.
    mean, scale
        Normalization computed from training snapshots only.
    feature_indices, feature_names
        Selected column indices and full ordered native feature schema.
    target_offset
        Mean transformed training target; also the distant-query prior.
    bandwidth, ridge
        Positive RBF bandwidth and diagonal kernel regularization.
    transform, target_floor
        ``log`` or ``logit`` with a training-only endpoint regularizer.
    """

    training_features: np.ndarray
    coefficients: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    feature_indices: np.ndarray
    feature_names: tuple[str, ...]
    target_offset: float
    bandwidth: float
    ridge: float
    transform: str = "logit"
    target_floor: float = 1e-10
    output_mode: str = field(default="efficiency", init=False)
    budget_kind: str = field(default="kernel_deficit", init=False)

    def __post_init__(self) -> None:
        """Validate and freeze the serialized empirical fit."""
        self.feature_names = tuple(self.feature_names)
        indices = np.asarray(self.feature_indices)
        if indices.dtype.kind not in "iu" or indices.ndim != 1:
            raise ValueError("Feature indices must be a one-dimensional integer array")
        self.feature_indices = np.array(indices, dtype=np.int64, copy=True)
        width = len(self.feature_indices)
        for name in ("training_features", "coefficients", "mean", "scale"):
            setattr(self, name, _real(getattr(self, name), name))
        if (
            self.feature_names[:10] != NATIVE_PREFIX
            or len(set(self.feature_names)) != len(self.feature_names)
            or not width
            or 0 not in self.feature_indices
            or len(np.unique(self.feature_indices)) != width
            or np.any(self.feature_indices < 0)
            or np.any(self.feature_indices >= len(self.feature_names))
            or self.training_features.ndim != 2
            or not len(self.training_features)
            or self.training_features.shape[1] != width
            or self.coefficients.shape != (len(self.training_features),)
            or self.mean.shape != (width,)
            or self.scale.shape != (width,)
            or np.any(self.scale <= 0)
            or not np.isfinite(
                [self.target_offset, self.bandwidth, self.ridge, self.target_floor]
            ).all()
            or self.bandwidth <= 0
            or self.ridge <= 0
            or not 0 < self.target_floor < 0.5
            or self.transform not in ("log", "logit")
        ):
            raise ValueError("Invalid kernel deficit model or native feature schema")
        for value in (
            self.training_features,
            self.coefficients,
            self.mean,
            self.scale,
            self.feature_indices,
        ):
            value.setflags(write=False)

    @classmethod
    def from_training(
        cls,
        features: np.ndarray,
        efficiencies: np.ndarray,
        *,
        feature_names: Sequence[str],
        feature_indices: Sequence[int],
        bandwidth: float,
        ridge: float,
        transform: str = "logit",
        target_floor: float = 1e-10,
    ) -> KernelDeficitBudgetModel:
        """Fit only supplied training rows; no validation labels are accepted.

        Parameters
        ----------
        features, efficiencies
            Native global features and absorbed/min(photons, atoms) labels.
        feature_names, feature_indices
            Exact input schema and selected inference-only native coordinates.
        bandwidth, ridge, transform, target_floor
            Fixed kernel and target transformation settings.

        Returns
        -------
        KernelDeficitBudgetModel
            Bounded empirical efficiency predictor with training normalization.
        """
        x = _native(features, len(feature_names))
        eta = _real(efficiencies, "Training efficiency")
        indices = np.asarray(feature_indices)
        if (
            eta.shape != (len(x),)
            or np.any((eta < 0) | (eta > 1))
            or indices.ndim != 1
            or indices.dtype.kind not in "iu"
            or not len(indices)
            or np.any(indices < 0)
            or np.any(indices >= x.shape[1])
            or not np.isfinite([bandwidth, ridge, target_floor]).all()
            or bandwidth <= 0
            or ridge <= 0
            or not 0 < target_floor < 0.5
            or transform not in ("log", "logit")
        ):
            raise ValueError("Invalid kernel training targets or controls")
        selected = x[:, indices]
        mean, scale = np.mean(selected, axis=0), np.std(selected, axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        coordinates = (selected - mean) / scale
        target = np.clip(1 - eta, target_floor, 1 - target_floor)
        transformed = np.log(target)
        if transform == "logit":
            transformed -= np.log1p(-target)
        offset = float(np.mean(transformed))
        matrix = _kernel(coordinates, coordinates, bandwidth)
        matrix.flat[:: len(matrix) + 1] += ridge
        coefficients = np.linalg.solve(matrix, transformed - offset)
        return cls(
            coordinates,
            coefficients,
            mean,
            scale,
            indices,
            tuple(feature_names),
            offset,
            bandwidth,
            ridge,
            transform,
            target_floor,
        )

    def predict_deficit(self, features: np.ndarray) -> np.ndarray:
        """Predict d in [0, 1] without a photon-load or redshift closure rule."""
        x = _native(features, len(self.feature_names))
        with np.errstate(over="raise", invalid="raise"):
            coordinates = (x[:, self.feature_indices] - self.mean) / self.scale
        transformed = (
            self.target_offset
            + _kernel(coordinates, self.training_features, self.bandwidth)
            @ self.coefficients
        )
        if self.transform == "logit":
            return expit(transformed)
        return np.exp(np.minimum(transformed, 0.0))

    def predict(
        self, features: np.ndarray, *, baseline: np.ndarray | None = None
    ) -> np.ndarray:
        """Return eta=1-d; absorbed atoms remain eta*min(source, capacity)."""
        return 1 - self.predict_deficit(features)

    def save(self, path: str | Path) -> None:
        """Save all inference state without pickle or future fitting."""
        np.savez_compressed(
            path,
            schema=np.asarray(1),
            kind=np.asarray(self.budget_kind),
            training_features=self.training_features,
            coefficients=self.coefficients,
            mean=self.mean,
            scale=self.scale,
            feature_indices=self.feature_indices,
            feature_names=np.asarray(self.feature_names),
            target_offset=np.asarray(self.target_offset),
            bandwidth=np.asarray(self.bandwidth),
            ridge=np.asarray(self.ridge),
            transform=np.asarray(self.transform),
            target_floor=np.asarray(self.target_floor),
        )

    @classmethod
    def load(cls, path: str | Path) -> KernelDeficitBudgetModel:
        """Restore the exact kernel and normalizer from a validated NPZ archive."""
        with np.load(path, allow_pickle=False) as saved:
            if int(saved["schema"]) != 1 or str(saved["kind"]) != "kernel_deficit":
                raise ValueError("Unsupported kernel deficit model schema")
            return cls(
                saved["training_features"],
                saved["coefficients"],
                saved["mean"],
                saved["scale"],
                saved["feature_indices"],
                tuple(str(name) for name in saved["feature_names"]),
                float(saved["target_offset"]),
                float(saved["bandwidth"]),
                float(saved["ridge"]),
                str(saved["transform"]),
                float(saved["target_floor"]),
            )
