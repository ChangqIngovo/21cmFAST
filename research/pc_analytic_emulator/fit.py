"""Fit latent whole-grid fields through a fixed global photon constraint.

Each optimizer update uses one complete training snapshot. Feature units are
fitted using all training cells, while checkpoints and independent trials use
the same postprojection validation objective. The separately fitted absorption
model remains frozen throughout this optimization.
"""

from __future__ import annotations

import hashlib
import itertools
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import fields, is_dataclass
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from research.pc_analytic_emulator.budget import predict_efficiency
from research.pc_analytic_emulator.latent import LatentFieldModel
from research.pc_analytic_emulator.objectives import projected_objective
from research.pc_analytic_emulator.projection import (
    capacity_l2_vjp,
    project_capacity_l2,
)

if TYPE_CHECKING:
    from research.pc_analytic_emulator.model import Emulator

FrameRecord = str | Path | dict[str, Any]
FRAME_CACHE_MAX_BYTES = 512 * 1024**2
OBJECTIVE_DEFAULTS = {
    "spectral_weight": 0.1,
    "spectral_power_floor": 1e-4,
    "front_weight": 1.0,
    "field_variance_floor": 0.01,
    "k_fraction": 0.5,
    "raw_weight": 0.05,
}


def _file_identity(path: Path) -> tuple[int, int, int, int, int]:
    status = path.stat()
    return (
        status.st_dev,
        status.st_ino,
        status.st_mtime_ns,
        status.st_ctime_ns,
        status.st_size,
    )


class _FrameCache:
    def __init__(self, max_bytes: int) -> None:
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError(
                "Frame cache limit must be a nonnegative integer byte count"
            )
        self.max_bytes = max_bytes
        self.nbytes = 0
        self.entries: OrderedDict[Path | tuple, tuple[tuple, dict[str, Any], int]] = (
            OrderedDict()
        )
        self.frozen_bases: dict[int, tuple[Emulator, str]] = {}

    def _remember(
        self, key: Path | tuple, identity: tuple, frame: dict[str, Any]
    ) -> None:
        size = sum(
            value.nbytes for value in frame.values() if isinstance(value, np.ndarray)
        )
        if size <= self.max_bytes:
            while self.entries and self.nbytes + size > self.max_bytes:
                _, removed = self.entries.popitem(last=False)
                self.nbytes -= removed[2]
            self.entries[key] = (identity, frame, size)
            self.nbytes += size

    def load(self, path: Path) -> dict[str, Any]:
        identity = _file_identity(path)
        old = self.entries.pop(path, None)
        if old is not None:
            if old[0] == identity:
                self.entries[path] = old
                return old[1]
            self.nbytes -= old[2]
        frame = _read_frame(path)
        if _file_identity(path) != identity:
            raise ValueError(f"Snapshot cache changed while being read: {path}")
        self._remember(path, identity, frame)
        return frame

    def baseline(self, frame: dict[str, Any], base_model: Emulator) -> np.ndarray:
        digest = _inference_digest(base_model)
        previous = self.frozen_bases.setdefault(id(base_model), (base_model, digest))
        if previous[1] != digest:
            raise ValueError("Frozen base model changed during residual fitting")
        path, identity = frame["cache_path"], frame["cache_identity"]
        redshift = _base_redshift(frame)
        key = ("base_prediction", path, digest, redshift)
        old = self.entries.pop(key, None)
        if old is not None:
            if old[0] == identity:
                self.entries[key] = old
                return old[1]["baseline"]
            self.nbytes -= old[2]
        baseline = _predict_base(frame, base_model)
        if _file_identity(path) != identity:
            raise ValueError(f"Snapshot cache changed during base prediction: {path}")
        self._remember(key, identity, {"baseline": baseline})
        return baseline

    def clear(self) -> None:
        self.entries.clear()
        self.frozen_bases.clear()
        self.nbytes = 0


def _inference_digest(model: Any) -> str:
    """Hash inference parameters, including semantic metadata only."""
    digest = hashlib.sha256()
    visiting: set[int] = set()

    def add(value: Any) -> None:
        if isinstance(value, np.ndarray):
            digest.update(b"array")
            digest.update(value.dtype.str.encode())
            digest.update(str(value.shape).encode())
            digest.update(np.ascontiguousarray(value).tobytes())
        elif is_dataclass(value) and not isinstance(value, type):
            if id(value) in visiting:
                raise ValueError("Frozen base inference parameters contain a cycle")
            visiting.add(id(value))
            digest.update(type(value).__qualname__.encode())
            for item in fields(value):
                if item.name != "metadata":
                    digest.update(item.name.encode())
                    add(getattr(value, item.name))
                elif "budget_parameterization" in getattr(value, item.name):
                    # This persisted tag determines whether sigmoid output is
                    # eta or 1-eta. Other metadata records provenance only.
                    digest.update(b"budget_parameterization")
                    add(getattr(value, item.name)["budget_parameterization"])
            visiting.remove(id(value))
        elif isinstance(value, dict):
            digest.update(b"mapping")
            for key in sorted(value):
                add(key)
                add(value[key])
        elif isinstance(value, (list, tuple)):
            digest.update(f"sequence:{len(value)}".encode())
            for item in value:
                add(item)
        elif value is None or isinstance(value, (str, int, float, bool, np.generic)):
            digest.update(f"{type(value).__name__}:{value!r}".encode())
        else:
            raise TypeError(f"Unsupported frozen inference parameter: {type(value)}")

    add(model)
    return digest.hexdigest()


_ACTIVE_FRAME_CACHE: ContextVar[_FrameCache | None] = ContextVar(
    "analytic_emulator_frame_cache", default=None
)


@contextmanager
def _frame_cache_scope(max_bytes: int = FRAME_CACHE_MAX_BYTES) -> Iterator[_FrameCache]:
    cache = _FrameCache(max_bytes)
    token = _ACTIVE_FRAME_CACHE.set(cache)
    try:
        yield cache
    finally:
        cache.clear()
        _ACTIVE_FRAME_CACHE.reset(token)


def _cache_during_fit(function: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with _frame_cache_scope(
            kwargs.get("frame_cache_max_bytes", FRAME_CACHE_MAX_BYTES)
        ):
            return function(*args, **kwargs)

    return wrapped


def _options(config: dict[str, Any] | None) -> dict[str, float]:
    values = dict(OBJECTIVE_DEFAULTS)
    if config is not None:
        if set(config) - set(values):
            raise ValueError("Unknown projected-objective controls")
        values.update(config)
    for name, value in values.items():
        if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value):
            raise ValueError(f"{name} must be a finite numeric value")
        if value < 0:
            raise ValueError(f"{name} must be nonnegative")
    if values["field_variance_floor"] <= 0 or not 0 < values["k_fraction"] <= 1:
        raise ValueError("Positive variance floor and k_fraction in (0, 1] required")
    return {name: float(value) for name, value in values.items()}


def _read_frame(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as saved:
        frame = {
            name: np.asarray(saved[name], dtype=np.float64) for name in saved.files
        }
    required = {
        "features",
        "baseline",
        "teacher_xHI",
        "capacity",
        "source",
        "global_features",
        "box_len",
    }
    if not required <= frame.keys():
        raise ValueError(f"Incomplete snapshot feature cache: {path}")
    if any(not np.all(np.isfinite(frame[name])) for name in required):
        raise ValueError(f"Snapshot feature cache contains nonfinite values: {path}")
    shape = frame["teacher_xHI"].shape
    if len(shape) != 3 or len(set(shape)) != 1 or shape[0] < 2:
        raise ValueError("Teacher target must be a nonempty cubic snapshot")
    size = frame["teacher_xHI"].size
    if frame["features"].ndim != 2 or frame["features"].shape[0] != size:
        raise ValueError("Snapshot features must have one row per cell")
    if frame["features"].shape[1] < 1 or frame["baseline"].shape != (size,):
        raise ValueError("Snapshot feature or baseline dimensions are invalid")
    for name in ("teacher_xHI", "baseline"):
        if np.any((frame[name] < 0) | (frame[name] > 1)):
            raise ValueError(f"Snapshot {name} must be bounded in [0, 1]")
    for name in ("capacity", "source"):
        if frame[name].shape != shape or np.any(frame[name] < 0):
            raise ValueError(f"Snapshot {name} must be nonnegative on the target grid")
    if np.sum(frame["capacity"], dtype=np.longdouble) <= 0:
        raise ValueError("Snapshot must have positive total atom capacity")
    if frame["global_features"].ndim != 2 or frame["global_features"].shape[0] != 1:
        raise ValueError("Budget features require one row per snapshot")
    if frame["box_len"].ndim != 0 or float(frame["box_len"]) <= 0:
        raise ValueError("Snapshot box_len must be a positive scalar")
    frame["box_len"] = float(frame["box_len"])
    for value in frame.values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    return frame


def _load_frame(record: FrameRecord) -> dict[str, Any]:
    path = Path(record["cache"] if isinstance(record, dict) else record).resolve()
    identity = _file_identity(path)
    cache = _ACTIVE_FRAME_CACHE.get()
    frame = dict(cache.load(path) if cache is not None else _read_frame(path))
    if _file_identity(path) != identity:
        raise ValueError(f"Snapshot cache changed while being loaded: {path}")
    frame["cache_path"], frame["cache_identity"] = path, identity
    frame["record"] = (
        {name: record[name] for name in ("seed", "redshift") if name in record}
        if isinstance(record, dict)
        else {"cache": str(path)}
    )
    return frame


def _base_redshift(frame: dict[str, Any]) -> float:
    value = frame["record"].get("redshift")
    if (
        isinstance(value, bool)
        or value is None
        or not np.isscalar(value)
        or not np.isfinite(value)
        or value < 0
    ):
        raise ValueError("Warm-start frame records require a finite native redshift")
    return float(value)


def _predict_base(frame: dict[str, Any], base_model: Emulator) -> np.ndarray:
    if "native_es_xHI" not in frame:
        raise ValueError("Warm-start frames require the original native_es_xHI input")
    prediction = base_model.predict(
        frame["source"],
        frame["capacity"],
        frame["native_es_xHI"],
        _base_redshift(frame),
        frame["box_len"],
    )
    baseline = np.array(prediction["xHII"], dtype=np.float64, copy=True)
    if (
        baseline.shape != frame["capacity"].shape
        or not np.isfinite(baseline).all()
        or np.any((baseline < 0) | (baseline > 1))
    ):
        raise ValueError("Frozen base must predict a bounded field on the native grid")
    baseline = baseline.ravel()
    baseline.setflags(write=False)
    return baseline


def _warm_frame(frame: dict[str, Any], base_model: Emulator | None) -> dict[str, Any]:
    if base_model is None:
        return frame
    cache = _ACTIVE_FRAME_CACHE.get()
    baseline = (
        cache.baseline(frame, base_model)
        if cache is not None
        else _predict_base(frame, base_model)
    )
    return dict(frame, baseline=baseline)


def _normalizer(frames: Sequence[FrameRecord]) -> tuple[np.ndarray, np.ndarray, int]:
    if not frames:
        raise ValueError("At least one training snapshot is required")
    mean, second, count = None, None, 0
    for record in frames:
        features = _load_frame(record)["features"]
        batch_count = len(features)
        batch_mean = np.mean(features, axis=0)
        batch_second = np.sum((features - batch_mean) ** 2, axis=0)
        if mean is None:
            mean, second, count = batch_mean, batch_second, batch_count
            continue
        if mean.shape != batch_mean.shape:
            raise ValueError("Feature counts differ between training snapshots")
        difference = batch_mean - mean
        total = count + batch_count
        second += batch_second + difference**2 * (count * batch_count / total)
        mean += difference * (batch_count / total)
        count = total
    scale = np.sqrt(np.maximum(second / count, 0))
    scale[scale == 0] = 1.0
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise FloatingPointError("Whole-grid feature normalization overflowed")
    return mean, scale, count


def _initialize(
    frames: Sequence[FrameRecord], *, seed: int, hidden: tuple[int, ...]
) -> LatentFieldModel:
    mean, scale, count = _normalizer(frames)
    widths = (len(mean), *hidden, 1)
    random = np.random.default_rng(seed)
    weights = [
        random.normal(scale=np.sqrt(2 / (left + right)), size=(left, right))
        for left, right in itertools.pairwise(widths)
    ]
    biases = [np.zeros(width) for width in widths[1:]]
    weights[-1].fill(0)
    return LatentFieldModel(
        weights,
        biases,
        mean,
        scale,
        {
            "training_cells": count,
            "training_snapshots": len(frames),
            "standardizer_scope": "all training cells only; streaming snapshot moments",
            "initialization": "zero final layer gives the unmodified ES field",
        },
    )


def _target_count(frame: dict[str, Any], budget_model: Any) -> tuple[float, float]:
    values = predict_efficiency(budget_model, frame["global_features"])
    if values.shape != (1,) or not np.isfinite(values).all() or not 0 <= values[0] <= 1:
        raise ValueError("Frozen budget model must predict one efficiency in [0, 1]")
    source = np.sum(frame["source"], dtype=np.longdouble)
    capacity = np.sum(frame["capacity"], dtype=np.longdouble)
    limit = min(source, capacity)
    target = float(limit * values[0])
    if not np.isfinite(target) or not np.isfinite(float(source)):
        raise ValueError("Photon count overflowed the supported inference units")
    if np.longdouble(target) > limit:
        target = float(np.nextafter(target, 0.0))
    return target, float(values[0])


def _loss_gradient(
    latent: np.ndarray,
    frame: dict[str, Any],
    budget_model: Any,
    options: dict[str, float],
) -> tuple[dict[str, Any], np.ndarray]:
    truth = 1 - frame["teacher_xHI"]
    target, efficiency = _target_count(frame, budget_model)
    constrained, projection = project_capacity_l2(latent, frame["capacity"], target)
    projected_loss, projected_gradient, components = projected_objective(
        constrained,
        truth,
        frame["capacity"],
        frame["box_len"],
        **{name: value for name, value in options.items() if name != "raw_weight"},
    )
    gradient, _ = capacity_l2_vjp(constrained, frame["capacity"], projected_gradient)
    scale = max(float(np.var(truth)), options["field_variance_floor"])
    difference = latent - truth
    raw_mse = float(np.mean(difference**2))
    auxiliary = options["raw_weight"] * raw_mse / scale
    gradient += 2 * options["raw_weight"] * difference / (truth.size * scale)
    diagnostics = {
        **frame["record"],
        "projected_loss": float(projected_loss),
        "raw_auxiliary_loss": auxiliary,
        "optimization_loss": float(projected_loss) + auxiliary,
        "raw_mse": raw_mse,
        "projected_mse": float(np.mean((constrained - truth) ** 2)),
        "field_mse": float(np.mean((constrained - truth) ** 2)),
        "predicted_absorption_efficiency": efficiency,
        "projection": projection,
        "objective": components,
    }
    if (
        not np.isfinite(diagnostics["optimization_loss"])
        or not np.isfinite(gradient).all()
    ):
        raise FloatingPointError("Projected whole-grid objective is nonfinite")
    return diagnostics, gradient


def _frame_gradient(
    model: LatentFieldModel,
    frame: dict[str, Any],
    budget_model: Any,
    objective_config: dict[str, Any] | None,
    *,
    base_model: Emulator | None = None,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    frame = _warm_frame(frame, base_model)
    options = _options(objective_config)
    activations = model._forward((frame["features"] - model.mean) / model.scale)
    latent = (frame["baseline"] + activations[-1][:, 0]).reshape(
        frame["capacity"].shape
    )
    diagnostics, gradient = _loss_gradient(latent, frame, budget_model, options)
    delta = gradient.reshape(-1, 1)
    weight_gradients = [np.empty_like(weight) for weight in model.weights]
    bias_gradients = [np.empty_like(bias) for bias in model.biases]
    for layer in range(len(model.weights) - 1, -1, -1):
        weight_gradients[layer] = activations[layer].T @ delta
        bias_gradients[layer] = np.sum(delta, axis=0)
        if layer:
            delta = (delta @ model.weights[layer].T) * (1 - activations[layer] ** 2)
    return diagnostics, [*weight_gradients, *bias_gradients]


def score_field(
    model: LatentFieldModel,
    frames: Sequence[FrameRecord],
    budget_model: Any,
    objective_config: dict[str, Any] | None = None,
    *,
    base_model: Emulator | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    """Evaluate the identical projected objective used to select checkpoints.

    Parameters
    ----------
    model
        Latent residual field network, with a training-only standardizer.
    frames
        NPZ paths or records containing an absolute ``cache`` path. Each file
        stores features, baseline, teacher_xHI, capacity, source,
        global_features and scalar box_len for one full cube.
    budget_model
        Frozen input-only efficiency predictor accepting ``baseline=ones(1)``.
    objective_config
        Projected objective controls and optional raw_weight. The auxiliary raw
        loss is reported but never included in the returned selection score.
    base_model
        Optional frozen deployed emulator. Its full input-only prediction,
        including its own photon projection, replaces the native ES baseline.
        Warm-start frame records must include redshift and their cache must
        retain native_es_xHI. The original native features are unchanged.

    Returns
    -------
    score : float
        Mean projected objective with equal weight per snapshot.
    diagnostics : list of dict
        Per-snapshot projected, auxiliary, budget and projection diagnostics.
    """
    if not frames:
        raise ValueError("At least one scoring snapshot is required")
    options = _options(objective_config)
    diagnostics = []
    for record in frames:
        frame = _warm_frame(_load_frame(record), base_model)
        latent = model.latent_predict(
            frame["features"], baseline=frame["baseline"]
        ).reshape(frame["capacity"].shape)
        metrics, _ = _loss_gradient(latent, frame, budget_model, options)
        diagnostics.append(metrics)
    return float(np.mean([row["projected_loss"] for row in diagnostics])), diagnostics


def _validate_fit_options(
    seed: int,
    hidden: tuple[int, ...],
    epochs: int,
    patience: int,
    learning_rate: float,
    min_delta: float,
) -> None:
    if type(seed) is not int or seed < 0:
        raise ValueError("Initialization seed must be a nonnegative integer")
    if any(
        type(value) is not int or value < 1 for value in (epochs, patience, *hidden)
    ):
        raise ValueError("Epoch, patience and hidden widths must be positive integers")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive")
    if not np.isfinite(min_delta) or min_delta < 0:
        raise ValueError("Selection min_delta must be finite and nonnegative")


def _adam_step(
    parameters: list[np.ndarray],
    gradients: list[np.ndarray],
    first: list[np.ndarray],
    second: list[np.ndarray],
    iteration: int,
    learning_rate: float,
) -> None:
    for parameter, gradient, moment, square in zip(
        parameters, gradients, first, second, strict=True
    ):
        if not np.isfinite(gradient).all():
            raise FloatingPointError("Latent parameter gradient is nonfinite")
        moment *= 0.9
        moment += 0.1 * gradient
        square *= 0.999
        square += 0.001 * gradient**2
        numerator = moment / (1 - 0.9**iteration)
        denominator = np.sqrt(square / (1 - 0.999**iteration)) + 1e-8
        parameter -= learning_rate * numerator / denominator


@_cache_during_fit
def fit_field(
    training_frames: Sequence[FrameRecord],
    validation_frames: Sequence[FrameRecord],
    budget_model: Any,
    *,
    seed: int,
    hidden: tuple[int, ...] = (32, 32),
    epochs: int = 80,
    learning_rate: float = 1e-3,
    patience: int = 20,
    min_delta: float = 1e-9,
    objective_config: dict[str, Any] | None = None,
    on_epoch: Callable[[dict[str, Any]], None] | None = None,
    base_model: Emulator | None = None,
    frame_cache_max_bytes: int = FRAME_CACHE_MAX_BYTES,
) -> tuple[LatentFieldModel, list[dict[str, Any]]]:
    """Train full snapshots with differentiation through a fixed photon budget.

    Parameters
    ----------
    training_frames, validation_frames
        Disjoint snapshot cache records accepted by :func:`score_field`.
        Training reads one complete cube at a time and visits every training
        cube once per epoch in a reproducibly shuffled order. A read-only LRU
        cache uses the declared per-fit bound, checks file metadata on every
        access and is cleared on return or failure.
    budget_model
        Absorption predictor selected before field fitting; never updated here.
    seed
        Nonnegative initialization and snapshot permutation seed.
    hidden
        Hidden tanh layer widths; the final scalar output is linear.
    epochs, patience
        Maximum epochs and consecutive nonimproving validation epochs.
    learning_rate
        Adam learning rate, with one update per complete training snapshot.
    min_delta
        Minimum improvement in the projected validation objective.
    objective_config
        Spectral/front weights, variance floor, k fraction and raw_weight.
        The normalized latent-to-teacher auxiliary MSE affects optimization
        only. Checkpoint selection uses the projected objective alone.
    on_epoch
        Optional callback receiving the completed epoch diagnostics.
    base_model
        Optional complete frozen emulator supplying the residual baseline.
        Only the newly initialized residual network is optimized. Its zero
        output retains the base prediction when the absorption budget matches.
        Base predictions share the bounded frame cache; changing base inference
        parameters during fitting raises an error.
    frame_cache_max_bytes
        Nonnegative per-worker array cache cap in bytes; default 512 MiB.
        Zero disables storage. Frozen-base arrays share the cap. Activations,
        optimizer state and interpreter/library memory are additional.

    Returns
    -------
    model : LatentFieldModel
        Parameters of the best projected validation epoch. The zero-correction
        ES or frozen-model initialization remains eligible at epoch zero.
    history : list of dict
        Losses for each epoch containing optimizer updates.
    """
    _validate_fit_options(seed, hidden, epochs, patience, learning_rate, min_delta)
    options = _options(objective_config)
    if not training_frames or not validation_frames:
        raise ValueError("Training and validation each require at least one snapshot")
    model = _initialize(training_frames, seed=seed, hidden=hidden)
    base_digest = _inference_digest(base_model) if base_model is not None else None
    model.metadata.update(
        training_method="warm_started_v3"
        if base_model is not None
        else "projected_multiscale_v2",
        optimizer="Adam; one whole-snapshot update",
        projection_mode="capacity_l2",
        objective_config=options,
        selection_objective="equal-snapshot mean postprojection objective; raw auxiliary excluded",
        raw_auxiliary="variance-normalized latent-to-teacher MSE; optimization only",
        seed=seed,
        hidden=list(hidden),
        requested_epochs=epochs,
        learning_rate=learning_rate,
        patience=patience,
        min_delta=min_delta,
        validation_snapshots=len(validation_frames),
        budget_model_frozen=True,
        base_model_frozen=base_model is not None,
        base_model_inference_sha256=base_digest,
        baseline_scope="full frozen emulator prediction from native inputs"
        if base_model is not None
        else "native ES input field",
        frame_cache_max_bytes=frame_cache_max_bytes,
        frame_cache_scope="per fit; read-only LRU; resolved-path file metadata checked per access",
    )
    if base_model is not None:
        model.metadata["initialization"] = (
            "zero final residual layer retains the full frozen emulator baseline"
        )
    parameters = [*model.weights, *model.biases]
    moments = [np.zeros_like(parameter) for parameter in parameters]
    squares = [np.zeros_like(parameter) for parameter in parameters]
    best_parameters = [parameter.copy() for parameter in parameters]
    initial_train, _ = score_field(
        model, training_frames, budget_model, options, base_model=base_model
    )
    best_loss, initial_validation = score_field(
        model, validation_frames, budget_model, options, base_model=base_model
    )
    best_mse = float(np.mean([row["projected_mse"] for row in initial_validation]))
    model.metadata["initial_baseline_metrics"] = {
        "epoch": 0,
        "training_loss": initial_train,
        "validation_loss": best_loss,
        "validation_mse": best_mse,
    }
    random = np.random.default_rng(seed)
    best_epoch, stalled, iteration = 0, 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        for index in random.permutation(len(training_frames)):
            frame = _load_frame(training_frames[index])
            _, gradients = _frame_gradient(
                model, frame, budget_model, options, base_model=base_model
            )
            iteration += 1
            _adam_step(
                parameters, gradients, moments, squares, iteration, learning_rate
            )
        training_loss, training_metrics = score_field(
            model, training_frames, budget_model, options, base_model=base_model
        )
        validation_loss, validation_metrics = score_field(
            model, validation_frames, budget_model, options, base_model=base_model
        )
        validation_mse = float(
            np.mean([row["projected_mse"] for row in validation_metrics])
        )
        improved = validation_loss < best_loss - min_delta
        if improved:
            best_loss, best_mse, best_epoch, stalled = (
                validation_loss,
                validation_mse,
                epoch,
                0,
            )
            best_parameters = [parameter.copy() for parameter in parameters]
        else:
            stalled += 1
        row = {
            "epoch": epoch,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "training_mse": float(
                np.mean([entry["projected_mse"] for entry in training_metrics])
            ),
            "validation_mse": validation_mse,
            "training_optimization_loss": float(
                np.mean([entry["optimization_loss"] for entry in training_metrics])
            ),
            "validation_raw_auxiliary_loss": float(
                np.mean([entry["raw_auxiliary_loss"] for entry in validation_metrics])
            ),
            "selected": improved,
            "snapshot_updates": len(training_frames),
        }
        history.append(row)
        if on_epoch is not None:
            on_epoch(row.copy())
        if stalled >= patience:
            break
    for parameter, best in zip(parameters, best_parameters, strict=True):
        parameter[...] = best
    model.metadata.update(
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        best_validation_mse=best_mse,
        completed_epochs=len(history),
        optimizer_updates=iteration,
    )
    if base_model is not None and _inference_digest(base_model) != base_digest:
        raise ValueError("Frozen base model changed during residual fitting")
    return model, history
