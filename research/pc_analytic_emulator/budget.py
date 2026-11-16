"""Select an absorbed-count predictor from independent validation snapshots.

All candidates predict absorption efficiency from native inputs. The tabulated
candidate interpolates training-snapshot means and adds no transport physics.
Selection minimizes error in absorbed atoms divided by total atom capacity,
not an efficiency error detached from its physical photon supply.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from research.pc_field_emulator.model import MLPModel, train


@dataclass
class TabulatedBudgetModel:
    """Interpolate training efficiencies against log(1 + photons/atoms).

    Parameters
    ----------
    knots, efficiencies
        Strictly increasing photon-load coordinates and bounded efficiencies.
        Coordinates are means over training seeds at each training redshift.
    feature_count
        Required number of columns in the native input feature matrix.
    """

    knots: np.ndarray
    efficiencies: np.ndarray
    feature_count: int
    output_mode: str = field(default="efficiency", init=False)
    budget_kind: str = field(default="tabulated", init=False)

    def __post_init__(self) -> None:
        """Copy and validate the independent training table."""
        self.knots = np.array(self.knots, dtype=np.float64, copy=True)
        self.efficiencies = np.array(self.efficiencies, dtype=np.float64, copy=True)
        if (
            self.knots.ndim != 1
            or not self.knots.size
            or self.knots.shape != self.efficiencies.shape
            or not np.isfinite(self.knots).all()
            or np.any(self.knots < 0)
            or np.any(np.diff(self.knots) <= 0)
            or not np.isfinite(self.efficiencies).all()
            or np.any((self.efficiencies < 0) | (self.efficiencies > 1))
            or type(self.feature_count) is not int
            or self.feature_count < 2
        ):
            raise ValueError("Invalid training-only absorption table")

    def predict(
        self, features: np.ndarray, *, baseline: np.ndarray | None = None
    ) -> np.ndarray:
        """Predict efficiencies, using constant extrapolation outside the table.

        Parameters
        ----------
        features
            Native budget features. Column zero is log(1 + photons/atoms).
        baseline
            Unused compatibility argument; no teacher information is accepted.

        Returns
        -------
        ndarray
            Bounded efficiency predictions, one per snapshot.
        """
        values = np.asarray(features, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[1] != self.feature_count
            or not np.isfinite(values).all()
            or np.any(values[:, 0] < 0)
        ):
            raise ValueError("Invalid native budget feature matrix")
        return np.interp(values[:, 0], self.knots, self.efficiencies)

    def save(self, path: str | Path) -> None:
        """Save a pickle-free absorption table.

        Parameters
        ----------
        path
            Destination NPZ file in an existing directory.
        """
        np.savez_compressed(
            path,
            schema=np.asarray(1),
            kind=np.asarray("tabulated_efficiency"),
            knots=self.knots,
            efficiencies=self.efficiencies,
            feature_count=np.asarray(self.feature_count),
        )

    @classmethod
    def load(cls, path: str | Path) -> TabulatedBudgetModel:
        """Read and validate an independently fitted absorption table.

        Parameters
        ----------
        path
            Archive written by :meth:`save`.

        Returns
        -------
        TabulatedBudgetModel
            Validated inference predictor.
        """
        with np.load(path, allow_pickle=False) as saved:
            if (
                int(saved["schema"]) != 1
                or str(saved["kind"]) != "tabulated_efficiency"
            ):
                raise ValueError("Unsupported absorption table schema")
            return cls(
                saved["knots"], saved["efficiencies"], int(saved["feature_count"])
            )


@dataclass
class LocalDeficitBudgetModel:
    """Interpolate training absorption deficits from native box structure.

    Parameters
    ----------
    training_features
        Training-only standardized photon load, density/source scatter and
        native ES morphology coordinates, with one row per training snapshot.
    deficits
        Training labels ``1 - absorbed/min(source, capacity)`` in [0, 1].
    mean, scale
        Training-only feature standardizer for the selected coordinates.
    feature_indices
        Indices of native input coordinates, with no teacher-derived columns.
    feature_count
        Number of columns in the original native budget feature matrix.
    neighbors
        Positive number of nearest training snapshots used for interpolation.

    Notes
    -----
    Positive inverse-distance weights form a convex average of the observed
    deficits. This can return exact zero when all selected deficits are zero,
    but does not impose zero escape at any redshift or photon load. Outside
    the training support the same local averaging rule applies; accuracy there
    is unvalidated. This estimator is empirical, not a transport closure law.
    """

    training_features: np.ndarray
    deficits: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    feature_indices: np.ndarray
    feature_count: int
    neighbors: int = 3
    output_mode: str = field(default="efficiency", init=False)
    budget_kind: str = field(default="local_deficit", init=False)

    def __post_init__(self) -> None:
        """Copy and validate the native-input table and bounded labels."""
        self.training_features = np.array(
            self.training_features, dtype=np.float64, copy=True
        )
        self.deficits = np.array(self.deficits, dtype=np.float64, copy=True)
        self.mean = np.array(self.mean, dtype=np.float64, copy=True)
        self.scale = np.array(self.scale, dtype=np.float64, copy=True)
        indices = np.asarray(self.feature_indices)
        if (
            indices.ndim != 1
            or not np.issubdtype(indices.dtype, np.integer)
            or not indices.size
        ):
            raise ValueError("Local budget feature indices must be nonempty integers")
        self.feature_indices = np.array(indices, dtype=np.int64, copy=True)
        width = len(self.feature_indices)
        if (
            type(self.feature_count) is not int
            or self.feature_count < 10
            or type(self.neighbors) is not int
            or self.neighbors < 1
            or np.any(self.feature_indices < 0)
            or np.any(self.feature_indices >= self.feature_count)
            or len(np.unique(self.feature_indices)) != width
            or self.training_features.ndim != 2
            or not self.training_features.shape[0]
            or self.training_features.shape[1] != width
            or self.deficits.shape != (len(self.training_features),)
            or self.mean.shape != (width,)
            or self.scale.shape != (width,)
            or any(
                not np.isfinite(value).all()
                for value in (
                    self.training_features,
                    self.deficits,
                    self.mean,
                    self.scale,
                )
            )
            or np.any(self.scale <= 0)
            or np.any((self.deficits < 0) | (self.deficits > 1))
        ):
            raise ValueError("Invalid native-input local absorption table")
        for value in (
            self.training_features,
            self.deficits,
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
        neighbors: int = 3,
    ) -> LocalDeficitBudgetModel:
        """Fit coordinates using training snapshots only.

        Parameters
        ----------
        features
            Native global features in the order of ``model.budget_features``.
            The first ten fixed native columns are required; smoothing columns
            may follow and are not used in the neighborhood distance.
        efficiencies
            Bounded supervised training absorption efficiencies.
        neighbors
            Positive number of nearest neighbors; clipped to training count
            only when evaluating a query.

        Returns
        -------
        LocalDeficitBudgetModel
            Empirical interpolation model with a fixed training normalizer.
        """
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(efficiencies, dtype=np.float64)
        if (
            x.ndim != 2
            or not len(x)
            or x.shape[1] < 10
            or y.shape != (len(x),)
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
            or np.any(x[:, :2] < 0)
            or np.any((y < 0) | (y > 1))
        ):
            raise ValueError("Invalid native-input training absorption data")
        indices = np.asarray([0, 4, 5, 6, 7, 8, 9], dtype=np.int64)
        coordinates = x[:, indices]
        mean = np.mean(coordinates, axis=0)
        scale = np.std(coordinates, axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        return cls(
            (coordinates - mean) / scale,
            1 - y,
            mean,
            scale,
            indices,
            x.shape[1],
            neighbors,
        )

    def predict(
        self, features: np.ndarray, *, baseline: np.ndarray | None = None
    ) -> np.ndarray:
        """Predict bounded efficiencies from a convex local deficit average.

        Parameters
        ----------
        features
            Native global features only, in their original order and units.
        baseline
            Unused compatibility argument; no teacher information is accepted.

        Returns
        -------
        ndarray
            Efficiencies in [0, 1], one per input snapshot. Exactly duplicate
            native coordinates receive their mean training deficit.
        """
        x = np.asarray(features, dtype=np.float64)
        if (
            x.ndim != 2
            or x.shape[1] != self.feature_count
            or not np.isfinite(x).all()
            or np.any(x[:, :2] < 0)
        ):
            raise ValueError("Invalid native budget feature matrix")
        result = np.empty(len(x), dtype=np.float64)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            coordinates = (x[:, self.feature_indices] - self.mean) / self.scale
            for index, query in enumerate(coordinates):
                distance = np.sum((self.training_features - query) ** 2, axis=1)
                exact = distance == 0
                if np.any(exact):
                    deficit = float(np.mean(self.deficits[exact]))
                else:
                    nearest = np.argsort(distance, kind="stable")[
                        : min(self.neighbors, len(distance))
                    ]
                    # Relative weights avoid overflow arbitrarily close to a
                    # training point. distance is squared Euclidean distance.
                    weights = distance[nearest[0]] / distance[nearest]
                    deficit = float(
                        np.dot(weights, self.deficits[nearest]) / np.sum(weights)
                    )
                result[index] = np.clip(1 - deficit, 0, 1)
        return result

    def save(self, path: str | Path) -> None:
        """Save an immutable, pickle-free local deficit predictor.

        Parameters
        ----------
        path
            Destination NPZ file in an existing directory.
        """
        np.savez_compressed(
            path,
            schema=np.asarray(1),
            kind=np.asarray("local_deficit"),
            training_features=self.training_features,
            deficits=self.deficits,
            mean=self.mean,
            scale=self.scale,
            feature_indices=self.feature_indices,
            feature_count=np.asarray(self.feature_count),
            neighbors=np.asarray(self.neighbors),
        )

    @classmethod
    def load(cls, path: str | Path) -> LocalDeficitBudgetModel:
        """Restore training coordinates and labels without fitting.

        Parameters
        ----------
        path
            Archive written by :meth:`save`.

        Returns
        -------
        LocalDeficitBudgetModel
            Restored bounded predictor with its original normalizer.
        """
        with np.load(path, allow_pickle=False) as saved:
            if int(saved["schema"]) != 1 or str(saved["kind"]) != "local_deficit":
                raise ValueError("Unsupported local absorption table schema")
            return cls(
                saved["training_features"],
                saved["deficits"],
                saved["mean"],
                saved["scale"],
                saved["feature_indices"],
                int(saved["feature_count"]),
                int(saved["neighbors"]),
            )


def constant_efficiency_model(feature_count: int) -> MLPModel:
    """Construct the eta=1 reference without fitting any data.

    Parameters
    ----------
    feature_count
        Number of native budget feature columns.

    Returns
    -------
    MLPModel
        Zero-residual model whose inference baseline is one.
    """
    if type(feature_count) is not int or feature_count < 1:
        raise ValueError("A positive feature count is required")
    return MLPModel(
        weights=[np.zeros((feature_count, 1))],
        biases=[np.zeros(1)],
        mean=np.zeros(feature_count),
        scale=np.ones(feature_count),
        metadata={"kind": "constant_efficiency", "efficiency": 1.0},
        output_mode="residual",
    )


def load_budget(
    path: str | Path,
) -> MLPModel | TabulatedBudgetModel | LocalDeficitBudgetModel:
    """Restore a tagged table or a residual/direct-sigmoid budget network.

    Parameters
    ----------
    path
        Pickle-free budget archive saved by either supported predictor.

    Returns
    -------
    MLPModel, TabulatedBudgetModel or LocalDeficitBudgetModel
        Restored predictor with its original training normalization/table.
    """
    with np.load(path, allow_pickle=False) as saved:
        kind = str(saved["kind"]) if "kind" in saved else None
    if kind == "tabulated_efficiency":
        return TabulatedBudgetModel.load(path)
    if kind == "local_deficit":
        return LocalDeficitBudgetModel.load(path)
    return MLPModel.load(path)


def predict_efficiency(
    model: MLPModel | TabulatedBudgetModel | LocalDeficitBudgetModel,
    features: np.ndarray,
) -> np.ndarray:
    """Apply the saved budget head's output convention consistently.

    Parameters
    ----------
    model
        Fitted absorption predictor. Residual networks use an eta=1 baseline;
        sigmoid networks and empirical tables predict efficiencies directly.
    features
        Native input features, with one row per snapshot and no teacher data.

    Returns
    -------
    ndarray
        Finite efficiencies in [0, 1], with one value per input snapshot.
    """
    inputs = np.asarray(features, dtype=np.float64)
    if inputs.ndim != 2 or not len(inputs):
        raise ValueError("Budget prediction requires a nonempty feature matrix")
    if model.output_mode == "residual":
        values = model.predict(inputs, baseline=np.ones(len(inputs)))
    elif model.output_mode in ("sigmoid", "efficiency"):
        values = model.predict(inputs)
    else:
        raise ValueError("Unsupported budget model output mode")
    values = np.asarray(values, dtype=np.float64)
    if (
        values.shape != (len(inputs),)
        or not np.isfinite(values).all()
        or np.any((values < 0) | (values > 1))
    ):
        raise ValueError("Budget model must predict one efficiency in [0, 1] per row")
    return values


def _arrays(
    features: np.ndarray, eta: np.ndarray, mass_scale: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, scale = (
        np.asarray(value, dtype=np.float64) for value in (features, eta, mass_scale)
    )
    if (
        x.ndim != 2
        or x.shape[1] < 2
        or not x.shape[0]
        or y.shape != (len(x),)
        or scale.shape != y.shape
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or not np.isfinite(scale).all()
        or np.any(x[:, :2] < 0)
        or np.any((y < 0) | (y > 1))
        or np.any((scale < 0) | (scale > 1))
    ):
        raise ValueError("Invalid supervised photon-budget data")
    return x, y, scale


def _fit_base_candidates(
    x_train: np.ndarray,
    eta_train: np.ndarray,
    mass_scale_train: np.ndarray,
    x_validation: np.ndarray,
    eta_validation: np.ndarray,
    mass_scale_validation: np.ndarray,
    *,
    seed: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    hidden: tuple[int, ...] = (8,),
) -> tuple[dict[str, MLPModel | TabulatedBudgetModel], dict[str, Any]]:
    """Fit budget candidates on training data and select by validation mass error.

    Parameters
    ----------
    x_train, x_validation
        Native budget feature matrices. Columns zero and one are photon load
        and redshift; neither matrix contains teacher fields or reservoirs.
    eta_train, eta_validation
        Supervised absorbed fraction of min(source, capacity).
    mass_scale_train, mass_scale_validation
        min(source, capacity) divided by atom capacity, one per snapshot.
    seed, epochs, patience, learning_rate, hidden
        MLP initialization, optimization and architecture controls.

    Returns
    -------
    tuple
        Selected predictor and JSON-safe candidate errors/provenance. Constant
        eta=1 is eligible but never forced on finite-escape teacher data.
    """
    xt, yt, st = _arrays(x_train, eta_train, mass_scale_train)
    xv, yv, sv = _arrays(x_validation, eta_validation, mass_scale_validation)
    if xt.shape[1] != xv.shape[1]:
        raise ValueError("Training and validation budget features differ")
    reference = constant_efficiency_model(xt.shape[1])
    # Average across independent training realizations at each epoch before
    # interpolating. Validation/test labels never define knots or table values.
    groups = [xt[:, 1] == redshift for redshift in np.unique(xt[:, 1])]
    coordinates = np.array([np.mean(xt[group, 0]) for group in groups])
    values = np.array([np.mean(yt[group]) for group in groups])
    knots = np.unique(coordinates)
    values = np.array([np.mean(values[coordinates == knot]) for knot in knots])
    table = TabulatedBudgetModel(knots, values, xt.shape[1])
    candidates: list[tuple[str, MLPModel | TabulatedBudgetModel]] = [
        ("constant_efficiency", reference),
        ("tabulated", table),
    ]
    history: list[dict[str, Any]] = []
    if np.any(st > 0) and np.any(sv > 0):
        network, history = train(
            xt,
            yt,
            xv,
            yv,
            seed=seed,
            hidden=hidden,
            epochs=epochs,
            patience=patience,
            learning_rate=learning_rate,
            batch_size=len(xt),
            min_delta=1e-14,
            baseline_train=np.ones(len(xt)),
            baseline_validation=np.ones(len(xv)),
            sample_weight_train=st**2,
            sample_weight_validation=sv**2,
        )
        candidates.append(("mlp", network))
    records = []
    for name, model in candidates:
        predicted = predict_efficiency(model, xv)
        error = sv * (predicted - yv)
        records.append(
            {
                "kind": name,
                "validation_mass_mse": float(np.mean(error**2)),
                "validation_mass_MAE": float(np.mean(np.abs(error))),
                "validation_maximum_absolute_mass_error": float(np.max(np.abs(error))),
                "validation_efficiencies": predicted.tolist(),
            }
        )
    selected = min(
        range(len(records)),
        key=lambda index: (records[index]["validation_mass_mse"], index),
    )
    return dict(candidates), {
        "selected_kind": records[selected]["kind"],
        "selection": "Validation MSE of absorbed atoms / total atom capacity",
        "candidates": records,
        "network_history": history,
        "training_snapshots": len(xt),
        "validation_snapshots": len(xv),
        "table_knots_log1p_photons_per_atom": table.knots.tolist(),
        "table_efficiencies": table.efficiencies.tolist(),
        "table_extrapolation": "constant endpoint efficiency; no extrapolation accuracy is certified",
        "test_used": False,
    }


def fit_budget(
    x_train: np.ndarray,
    eta_train: np.ndarray,
    mass_scale_train: np.ndarray,
    x_validation: np.ndarray,
    eta_validation: np.ndarray,
    mass_scale_validation: np.ndarray,
    *,
    seed: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    hidden: tuple[int, ...] = (8,),
) -> tuple[MLPModel | TabulatedBudgetModel, dict[str, Any]]:
    """Preserve v2 budget fitting and its validation-mass-error selection.

    Parameters
    ----------
    x_train, x_validation
        Native global input features, with photon load and redshift first.
    eta_train, eta_validation
        Supervised efficiencies absorbed/min(source, capacity).
    mass_scale_train, mass_scale_validation
        min(source, capacity)/total atom capacity for each snapshot.
    seed, epochs, patience, learning_rate, hidden
        Original MLP initialization, optimization and architecture controls.

    Returns
    -------
    tuple
        Original selected predictor and JSON-safe candidate diagnostics.
        V3 local estimators are not fitted or selected on this path.
    """
    candidates, report = _fit_base_candidates(
        x_train,
        eta_train,
        mass_scale_train,
        x_validation,
        eta_validation,
        mass_scale_validation,
        seed=seed,
        epochs=epochs,
        patience=patience,
        learning_rate=learning_rate,
        hidden=hidden,
    )
    return candidates[report["selected_kind"]], report


def fit_budget_v3(
    x_train: np.ndarray,
    eta_train: np.ndarray,
    mass_scale_train: np.ndarray,
    x_validation: np.ndarray,
    eta_validation: np.ndarray,
    mass_scale_validation: np.ndarray,
    *,
    seed: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    hidden: tuple[int, ...] = (8,),
) -> tuple[
    dict[str, MLPModel | TabulatedBudgetModel | LocalDeficitBudgetModel],
    dict[str, Any],
]:
    """Retain budget candidates for controlled fixed-field comparisons.

    Parameters
    ----------
    x_train, x_validation
        Native global features in ``model.budget_features`` order. The first
        ten native columns are required; teacher fields are never inputs.
    eta_train, eta_validation
        Supervised efficiencies absorbed/min(source, capacity).
    mass_scale_train, mass_scale_validation
        min(source, capacity)/total atom capacity for each snapshot.
    seed, epochs, patience, learning_rate, hidden
        Original MLP initialization, optimization and architecture controls.

    Returns
    -------
    tuple
        All candidate models and a validation budget-error ranking. A direct
        sigmoid network avoids the positive-residual eta=1 plateau while the
        original residual network remains eligible. No model
        is discarded: the caller compares them with fixed field predictors
        and projections before residual fitting. Local tables and normalizers
        depend only on training snapshots; validation selects among models.
    """
    xt, yt, st = _arrays(x_train, eta_train, mass_scale_train)
    xv, yv, sv = _arrays(x_validation, eta_validation, mass_scale_validation)
    if xt.shape[1] < 10 or xv.shape[1] != xt.shape[1]:
        raise ValueError("V3 budget fitting requires complete native global features")
    candidates, report = _fit_base_candidates(
        xt,
        yt,
        mass_scale_train,
        xv,
        yv,
        sv,
        seed=seed,
        epochs=epochs,
        patience=patience,
        learning_rate=learning_rate,
        hidden=hidden,
    )
    models = dict(candidates)
    sigmoid_history: list[dict[str, Any]] = []
    if np.any(st > 0) and np.any(sv > 0):
        # No baseline: train() uses a direct sigmoid, with its initial output
        # bias derived only from the clipped mean training efficiency. Keep
        # the original optimizer, weights and validation selection criterion.
        smooth, sigmoid_history = train(
            xt,
            yt,
            xv,
            yv,
            seed=seed,
            hidden=hidden,
            epochs=epochs,
            patience=patience,
            learning_rate=learning_rate,
            batch_size=len(xt),
            min_delta=1e-14,
            sample_weight_train=st**2,
            sample_weight_validation=sv**2,
        )
        smooth.metadata.update(
            budget_parameterization="direct_sigmoid_efficiency",
            initial_efficiency_prior=float(np.clip(yt.mean(), 1e-4, 1 - 1e-4)),
            initial_efficiency_prior_scope="Training efficiency labels only",
        )
        models["sigmoid_mlp"] = smooth
        predicted = predict_efficiency(smooth, xv)
        error = sv * (predicted - yv)
        report["candidates"].append(
            {
                "kind": "sigmoid_mlp",
                "validation_mass_mse": float(np.mean(error**2)),
                "validation_mass_MAE": float(np.mean(np.abs(error))),
                "validation_maximum_absolute_mass_error": float(np.max(np.abs(error))),
                "validation_efficiencies": predicted.tolist(),
            }
        )
    for neighbors in (3, 5):
        name = f"local_k{neighbors}"
        model = LocalDeficitBudgetModel.from_training(xt, yt, neighbors=neighbors)
        models[name] = model
        predicted = predict_efficiency(model, xv)
        error = sv * (predicted - yv)
        report["candidates"].append(
            {
                "kind": name,
                "validation_mass_mse": float(np.mean(error**2)),
                "validation_mass_MAE": float(np.mean(np.abs(error))),
                "validation_maximum_absolute_mass_error": float(np.max(np.abs(error))),
                "validation_efficiencies": predicted.tolist(),
            }
        )
    ranked = sorted(report["candidates"], key=lambda item: item["validation_mass_mse"])
    report.update(
        selected_kind=ranked[0]["kind"],
        selection="Budget-error ranking only; all candidates retained for fixed-field validation",
        retained_candidate_count=len(models),
        ranking=[item["kind"] for item in ranked],
        sigmoid_network_history=sigmoid_history,
        sigmoid_fit_scope="Training native features and efficiency labels; weighted validation MSE chooses checkpoint",
        local_feature_indices=[0, 4, 5, 6, 7, 8, 9],
        local_fit_scope="Training native features and efficiencies only; no validation/test samples in normalizer or neighbor table",
        local_extrapolation="Convex nearest-training-deficit average; no out-of-support accuracy is certified",
        local_zero_deficit_scope="Exact zero only from zero local training deficits; no forced photon-load or redshift threshold",
    )
    return models, report
