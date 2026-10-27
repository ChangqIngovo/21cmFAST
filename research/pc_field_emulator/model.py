"""Small NumPy neural field emulator with an explicit global count projection.

The neural network predicts a coarse ionized fraction. The optional projection
enforces one supplied box-wide ionized-atom count; it does not enforce individual
source budgets, photon directions, or physical reachability. Its target must be
constructed from input photons and the experiment's declared sinks, never from
the held-out ionization field.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _sigmoid(values: np.ndarray) -> np.ndarray:
    magnitude = np.exp(-np.abs(values))
    return np.where(values >= 0, 1.0 / (1.0 + magnitude), magnitude / (1.0 + magnitude))


def _weighted_count(capacity: np.ndarray, fraction: np.ndarray) -> np.longdouble:
    products = np.multiply(capacity, fraction, dtype=np.longdouble)
    return np.sum(products, dtype=np.longdouble)


def _features(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty samples-by-features matrix")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains nonfinite values")
    return result


def _target(values: np.ndarray, count: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (count,):
        raise ValueError(f"{name} must have one fraction per sample")
    if not np.all(np.isfinite(result)) or np.any((result < 0) | (result > 1)):
        raise ValueError(f"{name} must contain finite fractions in [0, 1]")
    return result


def _sample_weights(values: np.ndarray | None, count: int, name: str) -> np.ndarray:
    if values is None:
        return np.ones(count, dtype=np.float64)
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (count,):
        raise ValueError(f"{name} must have one weight per sample")
    if not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    mean = float(np.mean(result))
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError(f"{name} must have positive total weight")
    return result / mean


def _residual_controls(
    baseline: np.ndarray | None,
    residual_scale: np.ndarray | None,
    count: int,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    if baseline is None:
        raise ValueError(f"{name} requires an input-only baseline")
    base = _target(baseline, count, f"{name} baseline")
    gate = (
        np.ones(count, dtype=np.float64)
        if residual_scale is None
        else _target(residual_scale, count, f"{name} residual scale")
    )
    return base, gate


def _residual_fraction(
    amplitude: np.ndarray, baseline: np.ndarray, residual_scale: np.ndarray
) -> np.ndarray:
    available = np.where(amplitude >= 0, 1.0 - baseline, baseline)
    return baseline + residual_scale * available * amplitude


def _residual_slope(
    amplitude: np.ndarray,
    baseline: np.ndarray,
    residual_scale: np.ndarray,
    error: np.ndarray,
) -> np.ndarray:
    # At zero residual the output has a kink. Select the feasible branch in
    # the loss-decreasing direction; saturated baselines can then move inward.
    increasing = (amplitude > 0) | ((amplitude == 0) & (error <= 0))
    available = np.where(increasing, 1.0 - baseline, baseline)
    return residual_scale * available * (1.0 - amplitude**2)


def _fit_standardizer(features, floor):
    """Fit feature units on training inputs with optional physical scale floors."""
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    if floor is not None:
        floor = np.asarray(floor, dtype=np.float64)
        if (floor.shape != scale.shape or not np.isfinite(floor).all() or np.any(floor < 0)):
            raise ValueError("Feature scale floor must be finite, nonnegative and match the feature count")
        scale = np.maximum(scale, floor)
    scale[scale == 0] = 1.0
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
        raise ValueError("Feature range overflowed standardization")
    return mean, scale, floor


@dataclass
class MLPModel:
    """Portable tanh network with direct or bounded residual output.

    Parameters
    ----------
    weights, biases : list of ndarray
        Learned dense-layer parameters, including the scalar output layer.
    mean, scale : ndarray
        Feature standardization fitted on training samples only.
    metadata : dict
        Training configuration and selected validation epoch.
    output_mode : {"sigmoid", "residual"}
        Direct fraction or a bounded correction to an input-only baseline.
        Residual models require the same baseline convention at inference.
    """

    weights: list[np.ndarray]
    biases: list[np.ndarray]
    mean: np.ndarray
    scale: np.ndarray
    metadata: dict
    output_mode: str = "sigmoid"

    def _forward(self, standardized: np.ndarray) -> list[np.ndarray]:
        values = [standardized]
        for weight, bias in zip(self.weights[:-1], self.biases[:-1], strict=True):
            values.append(np.tanh(values[-1] @ weight + bias))
        activation = _sigmoid if self.output_mode == "sigmoid" else np.tanh
        values.append(activation(values[-1] @ self.weights[-1] + self.biases[-1]))
        return values

    def predict(
        self,
        features: np.ndarray,
        *,
        baseline: np.ndarray | None = None,
        residual_scale: np.ndarray | None = None,
        chunk_size: int = 262144,
    ) -> np.ndarray:
        """Predict ionized fractions without fitting or updating any parameters.

        Parameters
        ----------
        features : ndarray
            Samples-by-features matrix in the original feature units.
        baseline : ndarray, optional
            Input-only fraction per sample, required for residual models.
        residual_scale : ndarray, optional
            Per-sample residual amplitude in [0, 1]; defaults to one. A zero
            gate preserves the baseline exactly, including fraction endpoints.
            The gate must be derived from inputs, never target ionization.
        chunk_size : int
            Maximum samples per dense network evaluation.

        Returns
        -------
        ndarray
            One ionized fraction in [0, 1] per sample.
        """
        features = _features(features, "Prediction features")
        if features.shape[1] != self.mean.size:
            raise ValueError("Prediction features do not match the fitted network")
        if self.output_mode == "residual":
            baseline, residual_scale = _residual_controls(
                baseline, residual_scale, len(features), "Prediction"
            )
        elif self.output_mode != "sigmoid":
            raise ValueError("Unsupported neural output mode")
        elif baseline is not None or residual_scale is not None:
            raise ValueError("A sigmoid model does not accept residual controls")
        if (
            isinstance(chunk_size, bool)
            or not isinstance(chunk_size, (int, np.integer))
            or chunk_size < 1
        ):
            raise ValueError("chunk_size must be a positive integer")
        predicted = np.empty(features.shape[0], dtype=np.float64)
        for start in range(0, features.shape[0], chunk_size):
            end = min(start + chunk_size, features.shape[0])
            standardized = (features[start:end] - self.mean) / self.scale
            amplitude = self._forward(standardized)[-1][:, 0]
            predicted[start:end] = (
                amplitude
                if self.output_mode == "sigmoid"
                else _residual_fraction(
                    amplitude, baseline[start:end], residual_scale[start:end]
                )
            )
        if not np.all(np.isfinite(predicted)):
            raise FloatingPointError("Network inference produced nonfinite fractions")
        return predicted

    def save(self, path: str | Path) -> None:
        """Save parameters and JSON metadata in a pickle-free NPZ archive.

        Parameters
        ----------
        path : str or Path
            Output file; its parent directory must already exist.
        """
        arrays = {
            "schema": np.array(2, dtype=np.int64),
            "output_mode": np.array(self.output_mode),
            "mean": self.mean,
            "scale": self.scale,
            "layer_count": np.array(len(self.weights), dtype=np.int64),
            "metadata_json": np.array(json.dumps(self.metadata, sort_keys=True)),
        }
        for index, (weight, bias) in enumerate(
            zip(self.weights, self.biases, strict=True)
        ):
            arrays[f"weight_{index}"] = weight
            arrays[f"bias_{index}"] = bias
        with Path(path).open("wb") as stream:
            np.savez_compressed(stream, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> MLPModel:
        """Load and validate a pickle-free model archive.

        Parameters
        ----------
        path : str or Path
            Archive written by :meth:`save`.

        Returns
        -------
        MLPModel
            Validated model ready for inference.
        """
        with np.load(path, allow_pickle=False) as archive:
            schema = int(archive["schema"])
            if schema not in (1, 2):
                raise ValueError("Unsupported neural model schema")
            output_mode = "sigmoid" if schema == 1 else str(archive["output_mode"])
            if output_mode not in ("sigmoid", "residual"):
                raise ValueError("Unsupported neural output mode")
            layer_count = int(archive["layer_count"])
            if not 1 <= layer_count <= 32:
                raise ValueError("Invalid neural layer count")
            mean = np.asarray(archive["mean"], dtype=np.float64)
            scale = np.asarray(archive["scale"], dtype=np.float64)
            weights = [
                np.asarray(archive[f"weight_{i}"], dtype=np.float64)
                for i in range(layer_count)
            ]
            biases = [
                np.asarray(archive[f"bias_{i}"], dtype=np.float64)
                for i in range(layer_count)
            ]
            metadata = json.loads(str(archive["metadata_json"]))
        if mean.ndim != 1 or mean.size == 0 or scale.shape != mean.shape:
            raise ValueError("Invalid neural feature standardizer")
        if (
            not np.all(np.isfinite(mean))
            or not np.all(np.isfinite(scale))
            or np.any(scale <= 0)
        ):
            raise ValueError("Nonfinite or nonpositive feature standardizer")
        previous_size = mean.size
        for weight, bias in zip(weights, biases, strict=True):
            if (
                weight.ndim != 2
                or weight.shape[0] != previous_size
                or weight.shape[1] == 0
            ):
                raise ValueError("Inconsistent neural weight shapes")
            if bias.shape != (weight.shape[1],):
                raise ValueError("Inconsistent neural bias shapes")
            if not np.all(np.isfinite(weight)) or not np.all(np.isfinite(bias)):
                raise ValueError("Nonfinite neural parameters")
            previous_size = weight.shape[1]
        if previous_size != 1 or not isinstance(metadata, dict):
            raise ValueError("Invalid neural output or metadata")
        return cls(
            weights=weights,
            biases=biases,
            mean=mean,
            scale=scale,
            metadata=metadata,
            output_mode=output_mode,
        )


def _validate_training_options(
    epochs: int,
    batch_size: int,
    patience: int,
    hidden: tuple[int, ...],
    learning_rate: float,
    min_delta: float,
) -> None:
    for name, value in [
        ("epochs", epochs),
        ("batch_size", batch_size),
        ("patience", patience),
    ]:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer")
    if any(
        isinstance(width, bool) or not isinstance(width, (int, np.integer)) or width < 1
        for width in hidden
    ):
        raise ValueError("Hidden layer widths must be positive integers")
    if (
        not np.isfinite(learning_rate)
        or learning_rate <= 0
        or not np.isfinite(min_delta)
        or min_delta < 0
    ):
        raise ValueError("Invalid learning rate or early-stopping tolerance")


def train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    seed: int = 1234,
    hidden: tuple[int, ...] = (32, 32),
    epochs: int = 100,
    batch_size: int = 2048,
    learning_rate: float = 1e-3,
    patience: int = 12,
    min_delta: float = 1e-7,
    on_epoch: Callable[[dict], None] | None = None,
    baseline_train: np.ndarray | None = None,
    baseline_validation: np.ndarray | None = None,
    residual_scale_train: np.ndarray | None = None,
    residual_scale_validation: np.ndarray | None = None,
    sample_weight_train: np.ndarray | None = None,
    sample_weight_validation: np.ndarray | None = None,
    feature_scale_floor: np.ndarray | None = None,
) -> tuple[MLPModel, list[dict]]:
    """Fit a small network using Adam and validation-based early stopping.

    Parameters
    ----------
    x_train, x_validation : ndarray
        Disjoint samples-by-features matrices. Split independent realizations
        before calling this function; it does not create the science split.
    y_train, y_validation : ndarray
        Ionized-fraction targets, one per sample.
    seed : int
        Seed for initialization and minibatch permutations.
    hidden : tuple of int
        Hidden-layer widths. An empty tuple fits sigmoid linear regression.
    epochs, batch_size, patience : int
        Training limit, minibatch size and nonimproving-epoch stopping limit.
    learning_rate : float
        Adam step size.
    min_delta : float
        Minimum absolute validation-loss improvement for selecting a new model.
    on_epoch : callable, optional
        Callback receiving one JSON-compatible training diagnostic per epoch.
    baseline_train, baseline_validation : ndarray, optional
        Input-only baseline fractions. Supplying both enables a bounded tanh
        residual model initialized to zero correction. Supplying neither
        retains the direct sigmoid model.
    residual_scale_train, residual_scale_validation : ndarray, optional
        Input-only residual gates in [0, 1], one per sample. Zero gates preserve
        the corresponding baseline exactly; omitted gates are one.
    sample_weight_train, sample_weight_validation : ndarray, optional
        Finite nonnegative loss weights with positive totals. Each array is
        normalized to mean one. Training uses weighted MSE, and validation
        uses its own weighted MSE for model selection. Both unweighted MSE and
        the selection loss are retained in the epoch history.
    feature_scale_floor : ndarray, optional
        Nonnegative scale lower bounds, one per input column, derived from
        training inputs only. These can keep nearly constant smoothed fields
        on the scale of their unsmoothed physical channel. Omitting this
        preserves independent per-column standardization.

    Returns
    -------
    model : MLPModel
        Parameters from the best validation epoch, not the final epoch. The
        initial zero correction is also a candidate for residual models and
        is recorded as best_epoch=0 when no fitted epoch improves it.
    history : list of dict
        Training/validation MSE and loss for epochs with optimizer updates.
        The initial residual candidate is recorded separately in metadata,
        so history length and completed_epochs count actual fitted epochs.
    """
    x_train = _features(x_train, "Training features")
    x_validation = _features(x_validation, "Validation features")
    if x_train.shape[1] != x_validation.shape[1]:
        raise ValueError("Training and validation feature counts differ")
    y_train = _target(y_train, len(x_train), "Training targets")
    y_validation = _target(y_validation, len(x_validation), "Validation targets")
    if (baseline_train is None) != (baseline_validation is None):
        raise ValueError(
            "Residual mode requires both training and validation baselines"
        )
    residual = baseline_train is not None
    if residual:
        baseline_train, residual_scale_train = _residual_controls(
            baseline_train, residual_scale_train, len(x_train), "Training"
        )
        baseline_validation, residual_scale_validation = _residual_controls(
            baseline_validation,
            residual_scale_validation,
            len(x_validation),
            "Validation",
        )
    elif residual_scale_train is not None or residual_scale_validation is not None:
        raise ValueError("Residual scale requires a baseline")
    training_weights = _sample_weights(
        sample_weight_train, len(x_train), "Training weights"
    )
    validation_weights = _sample_weights(
        sample_weight_validation, len(x_validation), "Validation weights"
    )
    _validate_training_options(
        epochs, batch_size, patience, hidden, learning_rate, min_delta
    )
    mean, scale, feature_scale_floor = _fit_standardizer(x_train, feature_scale_floor)
    standardized = (x_train - mean) / scale
    validation_standardized = (x_validation - mean) / scale
    rng = np.random.default_rng(seed)
    widths = (x_train.shape[1], *hidden, 1)
    weights = [
        rng.normal(scale=np.sqrt(2.0 / (left + right)), size=(left, right))
        for left, right in itertools.pairwise(widths)
    ]
    biases = [np.zeros(width, dtype=np.float64) for width in widths[1:]]
    if residual:
        weights[-1].fill(0.0)
    else:
        prior = float(np.clip(y_train.mean(), 1e-4, 1 - 1e-4))
        biases[-1][0] = np.log(prior) - np.log1p(-prior)
    metadata = {
        "schema": 2,
        "hidden_activation": "tanh",
        "output_activation": "bounded baseline plus tanh residual"
        if residual
        else "sigmoid",
        "optimizer": "Adam",
        "loss": "weighted ionized-fraction MSE",
        "sample_weighted": sample_weight_train is not None,
        "validation_weighted": sample_weight_validation is not None,
        "residual_zero_derivative": "loss-decreasing one-sided branch"
        if residual
        else None,
        "standardizer_scope": "training samples only",
        "feature_scale_floor": (feature_scale_floor.tolist()
                                if feature_scale_floor is not None else None),
        "seed": int(seed),
        "hidden": [int(width) for width in hidden],
        "requested_epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "patience": int(patience),
        "min_delta": float(min_delta),
        "training_samples": len(x_train),
        "validation_samples": len(x_validation),
    }
    model = MLPModel(
        weights,
        biases,
        mean,
        scale,
        metadata,
        output_mode="residual" if residual else "sigmoid",
    )
    parameters = [*weights, *biases]
    first_moment = [np.zeros_like(parameter) for parameter in parameters]
    second_moment = [np.zeros_like(parameter) for parameter in parameters]
    best_parameters = [parameter.copy() for parameter in parameters]
    best_loss = float("inf")
    best_epoch, stalled, iteration = 0, 0, 0
    best_validation_mse = float("inf")
    if residual:
        initial_train_error = (baseline_train - y_train) ** 2
        initial_validation_error = (baseline_validation - y_validation) ** 2
        best_loss = float(np.mean(validation_weights * initial_validation_error))
        best_validation_mse = float(np.mean(initial_validation_error))
        metadata["initial_baseline_metrics"] = {
            "epoch": 0,
            "training_mse": float(np.mean(initial_train_error)),
            "validation_mse": best_validation_mse,
            "training_loss": float(np.mean(training_weights * initial_train_error)),
            "validation_loss": best_loss,
        }
    history = []
    for epoch in range(1, epochs + 1):
        permutation = rng.permutation(len(standardized))
        for start in range(0, len(standardized), batch_size):
            indices = permutation[start : start + batch_size]
            activations = model._forward(standardized[indices])
            amplitude = activations[-1][:, 0]
            prediction = (
                _residual_fraction(
                    amplitude, baseline_train[indices], residual_scale_train[indices]
                )
                if residual
                else amplitude
            )
            error = prediction - y_train[indices]
            slope = (
                _residual_slope(
                    amplitude,
                    baseline_train[indices],
                    residual_scale_train[indices],
                    error,
                )
                if residual
                else prediction * (1.0 - prediction)
            )
            delta = (2.0 / len(indices) * training_weights[indices] * error * slope)[
                :, None
            ]
            weight_gradients = [np.empty_like(weight) for weight in weights]
            bias_gradients = [np.empty_like(bias) for bias in biases]
            for layer in range(len(weights) - 1, -1, -1):
                weight_gradients[layer] = activations[layer].T @ delta
                bias_gradients[layer] = delta.sum(axis=0)
                if layer:
                    delta = (delta @ weights[layer].T) * (1.0 - activations[layer] ** 2)
            iteration += 1
            gradients = [*weight_gradients, *bias_gradients]
            for index, (parameter, gradient) in enumerate(
                zip(parameters, gradients, strict=True)
            ):
                first_moment[index] *= 0.9
                first_moment[index] += 0.1 * gradient
                second_moment[index] *= 0.999
                second_moment[index] += 0.001 * gradient**2
                corrected_first = first_moment[index] / (1.0 - 0.9**iteration)
                corrected_second = second_moment[index] / (1.0 - 0.999**iteration)
                parameter -= (
                    learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)
                )
        training_prediction = model._forward(standardized)[-1][:, 0]
        validation_prediction = model._forward(validation_standardized)[-1][:, 0]
        if residual:
            training_prediction = _residual_fraction(
                training_prediction, baseline_train, residual_scale_train
            )
            validation_prediction = _residual_fraction(
                validation_prediction, baseline_validation, residual_scale_validation
            )
        training_error = (training_prediction - y_train) ** 2
        validation_error = (validation_prediction - y_validation) ** 2
        training_mse = float(np.mean(training_error))
        validation_mse = float(np.mean(validation_error))
        training_loss = float(np.mean(training_weights * training_error))
        validation_loss = float(np.mean(validation_weights * validation_error))
        if not np.isfinite(training_loss) or not np.isfinite(validation_loss):
            raise FloatingPointError("Neural optimization produced a nonfinite loss")
        improved = validation_loss < best_loss - min_delta
        if improved:
            best_loss, best_epoch, stalled = validation_loss, epoch, 0
            best_validation_mse = validation_mse
            best_parameters = [parameter.copy() for parameter in parameters]
        else:
            stalled += 1
        record = {
            "epoch": epoch,
            "training_mse": training_mse,
            "validation_mse": validation_mse,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "selected": improved,
        }
        history.append(record)
        if on_epoch is not None:
            on_epoch(record.copy())
        if stalled >= patience:
            break
    for parameter, best in zip(parameters, best_parameters, strict=True):
        parameter[...] = best
    metadata.update(
        best_epoch=best_epoch,
        best_validation_mse=best_validation_mse,
        best_validation_loss=best_loss,
        completed_epochs=len(history),
    )
    return model, history


def project_global_budget(
    predicted: np.ndarray,
    capacity: np.ndarray,
    target: float,
    *,
    rtol: float = 1e-11,
    atol: float = 1e-12,
) -> tuple[np.ndarray, dict]:
    """Shift predicted log-odds to satisfy a supplied global ionized-atom count.

    Parameters
    ----------
    predicted : ndarray
        Predicted fractions in [0, 1]. Exact endpoints receive finite log-odds
        so an all-zero or all-one neural field can still be corrected.
    capacity : ndarray
        Nonnegative atom counts with the same shape. Zero-capacity cells have
        output fraction zero by convention.
    target : float
        Desired total ionized atoms in [0, sum(capacity)]. Obtain this from
        input source budgets and known physical sinks, not a test label.
    rtol, atol : float
        Relative-to-target and absolute count tolerances.

    Returns
    -------
    corrected : ndarray
        Fractions with the original shape and corrected global atom count.
    diagnostics : dict
        Target, raw and corrected counts, residual, and log-odds adjustment.
        This is a global constraint only; it is not a transport solver.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    capacity = np.asarray(capacity, dtype=np.float64)
    if predicted.shape != capacity.shape or predicted.size == 0:
        raise ValueError("Prediction and capacity must have the same nonempty shape")
    if not np.all(np.isfinite(predicted)) or np.any((predicted < 0) | (predicted > 1)):
        raise ValueError("Predicted fractions must be finite and in [0, 1]")
    if not np.all(np.isfinite(capacity)) or np.any(capacity < 0):
        raise ValueError("Atom capacities must be finite and nonnegative")
    target = float(target)
    if not np.isfinite(target) or target < 0:
        raise ValueError("Target ionized count must be finite and nonnegative")
    if not np.isfinite(rtol) or not np.isfinite(atol) or rtol < 0 or atol < 0:
        raise ValueError("Projection tolerances must be finite and nonnegative")
    total = float(np.sum(capacity, dtype=np.longdouble))
    if not np.isfinite(total) or target > total:
        raise ValueError("Target ionized count exceeds the atom capacity")
    raw_count = float(_weighted_count(capacity, predicted))
    positive = capacity > 0
    result = np.zeros_like(predicted)
    shift = 0.0
    iterations = 0
    tolerance = atol + rtol * target
    target_extended = np.longdouble(target)
    if target == total:
        result[positive] = 1.0
    elif target > 0:
        fractions = np.clip(predicted[positive], 1e-12, 1 - 1e-12)
        logits = np.log(fractions) - np.log1p(-fractions)
        cell_capacity = capacity[positive]
        lower, upper = -64.0, 64.0
        for _ in range(16):
            if (
                _weighted_count(cell_capacity, _sigmoid(logits + lower))
                <= target_extended
            ):
                break
            lower *= 2
        else:
            raise FloatingPointError(
                "Global projection could not bracket its lower count"
            )
        for _ in range(16):
            if (
                _weighted_count(cell_capacity, _sigmoid(logits + upper))
                >= target_extended
            ):
                break
            upper *= 2
        else:
            raise FloatingPointError(
                "Global projection could not bracket its upper count"
            )
        for iteration in range(160):
            iterations = iteration + 1
            shift = (lower + upper) / 2
            trial = _sigmoid(logits + shift)
            count = _weighted_count(cell_capacity, trial)
            if abs(count - target_extended) <= tolerance:
                break
            if count < target_extended:
                lower = shift
            else:
                upper = shift
        result[positive] = trial
    corrected_extended = _weighted_count(capacity, result)
    corrected = float(corrected_extended)
    residual = float(corrected_extended - target_extended)
    rounding_allowance = 16 * np.finfo(float).eps * max(target, corrected)
    if abs(residual) > tolerance + rounding_allowance:
        raise FloatingPointError(
            "Global projection failed its requested count tolerance"
        )
    return result, {
        "scope": "global ionized-atom count only; no source or direction constraints",
        "target_count": target,
        "atom_capacity": total,
        "raw_count": raw_count,
        "corrected_count": corrected,
        "count_residual": residual,
        "relative_count_residual": residual / target if target else 0.0,
        "log_odds_shift": float(shift),
        "iterations": iterations,
        "prediction_endpoint_clip": 1e-12,
    }
