"""Unbounded residual scores for differentiable bounded-field projection.

The latent score is not a physical ionized fraction. Keeping it unbounded lets
a neutral cell remain at zero after a positive global projection shift. Only
``predict`` clips the raw diagnostic field; ``latent_predict`` supplies the
scores used by the photon-budget projector and its training derivative.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

_SCHEMA = 1
_KIND = "analytic-emulator-latent-linear"


def _numeric(values: ArrayLike, name: str) -> NDArray[np.float64]:
    result = np.asarray(values)
    if result.dtype.kind not in "fiu":
        raise TypeError(f"{name} must contain real numeric values")
    result = np.asarray(result, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite values")
    return result


def _metadata_json(metadata: dict[str, Any]) -> str:
    if not isinstance(metadata, dict):
        raise TypeError("Latent metadata must be a JSON object")

    def check_keys(value: Any) -> None:
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise TypeError("Latent metadata object keys must be strings")
            for child in value.values():
                check_keys(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                check_keys(child)

    check_keys(metadata)
    return json.dumps(metadata, allow_nan=False, sort_keys=True)


def _integer_scalar(value: np.ndarray, name: str) -> int:
    if value.shape != () or value.dtype.kind not in "iu":
        raise ValueError(f"Stored {name} must be an integer scalar")
    return int(value)


def _text_scalar(value: np.ndarray, name: str) -> str:
    if value.shape != () or value.dtype.kind != "U":
        raise ValueError(f"Stored {name} must be a Unicode scalar")
    return str(value)


@dataclass
class LatentFieldModel:
    """Represent a tanh network with a linear residual output.

    Parameters
    ----------
    weights, biases
        Dense-layer parameters, including one scalar linear output layer.
        Parameters remain mutable float64 arrays for the training optimizer.
    mean, scale
        Feature means and strictly positive scales fitted on training inputs.
    metadata
        JSON-serializable training provenance and checkpoint diagnostics.
    output_mode
        Must be ``"latent_linear"``; a bounded output would change the model.
    """

    weights: list[NDArray[np.float64]]
    biases: list[NDArray[np.float64]]
    mean: NDArray[np.float64]
    scale: NDArray[np.float64]
    metadata: dict[str, Any]
    output_mode: str = "latent_linear"

    def __post_init__(self) -> None:
        """Validate the linear-output contract and fitted parameter shapes."""
        self._validate()

    def _validate(self) -> None:
        if self.output_mode != "latent_linear":
            raise ValueError("Latent field model requires latent_linear output")
        if (
            not isinstance(self.weights, list)
            or not isinstance(self.biases, list)
            or not 1 <= len(self.weights) <= 32
            or len(self.weights) != len(self.biases)
        ):
            raise ValueError("Latent network requires matching nonempty layer lists")
        self.mean = _numeric(self.mean, "Feature mean")
        self.scale = _numeric(self.scale, "Feature scale")
        if (
            self.mean.ndim != 1
            or not self.mean.size
            or self.scale.shape != self.mean.shape
            or np.any(self.scale <= 0)
        ):
            raise ValueError("Invalid latent feature standardizer")
        previous = self.mean.size
        for index in range(len(self.weights)):
            weight = _numeric(self.weights[index], "Layer weight")
            bias = _numeric(self.biases[index], "Layer bias")
            if (
                weight.ndim != 2
                or weight.shape[0] != previous
                or weight.shape[1] < 1
                or bias.shape != (weight.shape[1],)
            ):
                raise ValueError("Inconsistent latent layer shapes")
            self.weights[index], self.biases[index] = weight, bias
            previous = weight.shape[1]
        if previous != 1:
            raise ValueError("Latent network requires one scalar output")
        _metadata_json(self.metadata)

    def _features(self, features: ArrayLike) -> NDArray[np.float64]:
        values = _numeric(features, "Prediction features")
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError(
                "Prediction features require a nonempty two-dimensional matrix"
            )
        if values.shape[1] != self.mean.size:
            raise ValueError("Prediction features do not match the fitted network")
        return values

    def _forward(self, standardized: ArrayLike) -> list[NDArray[np.float64]]:
        values = [self._features(standardized)]
        with np.errstate(over="raise", invalid="raise"):
            for weight, bias in zip(self.weights[:-1], self.biases[:-1], strict=True):
                values.append(np.tanh(values[-1] @ weight + bias))
            values.append(values[-1] @ self.weights[-1] + self.biases[-1])
        if not np.isfinite(values[-1]).all():
            raise FloatingPointError("Latent network produced nonfinite residuals")
        return values

    def latent_predict(
        self,
        features: ArrayLike,
        *,
        baseline: ArrayLike | None = None,
        chunk_size: int = 262144,
    ) -> NDArray[np.float64]:
        """Predict unbounded baseline-plus-residual scores for global projection.

        Parameters
        ----------
        features
            Nonempty samples-by-features matrix in original feature units.
        baseline
            Required input-only native ES ionized fraction for every sample.
        chunk_size
            Positive maximum number of samples evaluated in one matrix batch.

        Returns
        -------
        ndarray
            Finite unbounded latent scores, with no sigmoid, tanh or clipping
            applied to the final scalar residual.
        """
        self._validate()
        features = self._features(features)
        if baseline is None:
            raise ValueError("Latent prediction requires an input-only baseline")
        base = _numeric(baseline, "Prediction baseline")
        if base.shape != (len(features),) or np.any((base < 0) | (base > 1)):
            raise ValueError(
                "Prediction baseline must contain one fraction in [0,1] per sample"
            )
        if (
            isinstance(chunk_size, bool)
            or not isinstance(chunk_size, (int, np.integer))
            or chunk_size < 1
        ):
            raise ValueError("chunk_size must be a positive integer")
        result = np.empty(len(features), dtype=np.float64)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for start in range(0, len(features), chunk_size):
                end = min(start + chunk_size, len(features))
                standardized = (features[start:end] - self.mean) / self.scale
                result[start:end] = (
                    base[start:end] + self._forward(standardized)[-1][:, 0]
                )
        if not np.isfinite(result).all():
            raise FloatingPointError("Latent prediction produced nonfinite scores")
        return result

    def predict(
        self,
        features: ArrayLike,
        *,
        baseline: ArrayLike | None = None,
        chunk_size: int = 262144,
    ) -> NDArray[np.float64]:
        """Return clipped raw fractions for diagnostics, before photon projection.

        Parameters
        ----------
        features
            Nonempty samples-by-features matrix in original feature units.
        baseline
            Required input-only native ES ionized fraction for every sample.
        chunk_size
            Positive maximum number of samples evaluated in one matrix batch.

        Returns
        -------
        ndarray
            Raw ionized fractions in [0,1]. A budget projector must instead use
            :meth:`latent_predict` to retain the complete unbounded scores.
        """
        return np.clip(
            self.latent_predict(features, baseline=baseline, chunk_size=chunk_size),
            0,
            1,
        )

    def save(self, path: str | Path) -> None:
        """Save validated parameters in an explicitly tagged pickle-free archive.

        Parameters
        ----------
        path
            Destination file whose parent directory already exists.
        """
        self._validate()
        arrays = {
            "schema": np.array(_SCHEMA, dtype=np.int64),
            "kind": np.array(_KIND),
            "output_mode": np.array(self.output_mode),
            "layer_count": np.array(len(self.weights), dtype=np.int64),
            "mean": self.mean,
            "scale": self.scale,
            "metadata_json": np.array(_metadata_json(self.metadata)),
        }
        for index, (weight, bias) in enumerate(
            zip(self.weights, self.biases, strict=True)
        ):
            arrays[f"weight_{index}"] = weight
            arrays[f"bias_{index}"] = bias
        with Path(path).open("wb") as stream:
            np.savez_compressed(stream, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> LatentFieldModel:
        """Load only the exact latent-network schema and validate every parameter.

        Parameters
        ----------
        path
            Pickle-free NPZ archive written by :meth:`save`.

        Returns
        -------
        LatentFieldModel
            Restored network with unchanged training feature standardization.
        """
        with np.load(path, allow_pickle=False) as saved:
            required = {
                "schema",
                "kind",
                "output_mode",
                "layer_count",
                "mean",
                "scale",
                "metadata_json",
            }
            if not required.issubset(saved.files):
                raise ValueError("Incomplete latent model archive")
            if (
                _integer_scalar(saved["schema"], "schema") != _SCHEMA
                or _text_scalar(saved["kind"], "kind") != _KIND
                or _text_scalar(saved["output_mode"], "output_mode") != "latent_linear"
            ):
                raise ValueError("Unsupported latent model schema or output mode")
            count = _integer_scalar(saved["layer_count"], "layer_count")
            if not 1 <= count <= 32:
                raise ValueError("Invalid stored latent layer count")
            required.update(
                f"{kind}_{index}"
                for kind in ("weight", "bias")
                for index in range(count)
            )
            if len(saved.files) != len(required) or set(saved.files) != required:
                raise ValueError("Unexpected or missing latent model archive fields")
            return cls(
                weights=[
                    _numeric(saved[f"weight_{index}"], "Stored weight")
                    for index in range(count)
                ],
                biases=[
                    _numeric(saved[f"bias_{index}"], "Stored bias")
                    for index in range(count)
                ],
                mean=_numeric(saved["mean"], "Stored mean"),
                scale=_numeric(saved["scale"], "Stored scale"),
                metadata=json.loads(
                    _text_scalar(saved["metadata_json"], "metadata_json")
                ),
            )
