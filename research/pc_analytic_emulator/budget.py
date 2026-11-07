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


def load_budget(path: str | Path) -> MLPModel | TabulatedBudgetModel:
    """Restore a tagged table or legacy-format residual budget network.

    Parameters
    ----------
    path
        Pickle-free budget archive saved by either supported predictor.

    Returns
    -------
    MLPModel or TabulatedBudgetModel
        Restored predictor with its original training normalization/table.
    """
    with np.load(path, allow_pickle=False) as saved:
        tabulated = "kind" in saved and str(saved["kind"]) == "tabulated_efficiency"
    return TabulatedBudgetModel.load(path) if tabulated else MLPModel.load(path)


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
        predicted = model.predict(xv, baseline=np.ones(len(xv)))
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
    return candidates[selected][1], {
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
