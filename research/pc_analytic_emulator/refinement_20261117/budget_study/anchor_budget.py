"""Bounded empirical native-structure correction to a training-only budget table.

This development estimator changes no transport or absorption physics. It fits
the same absolute-plus-relative neutral mass objective used for model selection.
The finite log-odds correction is an empirical regularizer, not a closure law.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit


@dataclass
class AnchoredDeficit:
    """Training-only table plus a bounded native-structure log-odds correction."""

    knots: np.ndarray
    efficiencies: np.ndarray
    trend: np.ndarray
    indices: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    feature_count: int = 14
    correction_bound: float = 3.0
    base_logit_floor: float = 1e-12
    output_mode: str = "efficiency"
    budget_kind: str = "anchored_deficit_development"

    def validate(self):
        """Reject malformed coefficients, coordinates and numerical controls."""
        if (
            np.asarray(self.indices).dtype.kind not in "iu"
            or self.indices.ndim != 1
            or not np.isfinite(self.correction_bound)
            or not np.isfinite(self.base_logit_floor)
        ):
            raise ValueError("Invalid bounded budget controls")
        if (
            self.feature_count != 14
            or self.knots.ndim != 1
            or np.any(np.diff(self.knots) <= 0)
            or self.efficiencies.shape != self.knots.shape
            or self.trend.shape != (len(self.knots), self.feature_count)
            or self.mean.shape != self.scale.shape
            or self.mean.shape != (3 + len(self.indices),)
            or self.coefficients.shape != (4 + len(self.indices),)
            or np.any(self.scale <= 0)
            or np.any((self.efficiencies < 0) | (self.efficiencies > 1))
            or np.any((self.indices < 0) | (self.indices >= self.feature_count))
            or len(set(self.indices.tolist())) != len(self.indices)
            or self.correction_bound <= 0
            or not 0 < self.base_logit_floor < 0.5
        ):
            raise ValueError("Invalid bounded budget parameters")
        for array in (
            self.knots,
            self.efficiencies,
            self.trend,
            self.mean,
            self.scale,
            self.coefficients,
        ):
            if not np.isfinite(array).all():
                raise ValueError("Nonfinite budget parameter")

    def features(self, x):
        """Remove the training-only load trend from native structure summaries."""
        values = np.asarray(x)
        if (
            values.dtype.kind not in "fiu"
            or values.ndim != 2
            or values.shape[1] != self.feature_count
            or not np.isfinite(values).all()
            or np.any(values[:, 0] < 0)
        ):
            raise ValueError("Invalid native budget matrix")
        load = values[:, 0]
        pieces = [load, load**2, load**3]
        pieces.extend(
            values[:, index] - np.interp(load, self.knots, self.trend[:, index])
            for index in self.indices
        )
        return np.column_stack(pieces)

    def design(self, x):
        """Apply the frozen training standardizer and add an intercept."""
        f = (self.features(x) - self.mean) / self.scale
        return np.column_stack([np.ones(len(f)), f])

    def logits(self, x):
        """Convert the empirical base deficit to finite log odds."""
        base_d = 1 - np.interp(np.asarray(x)[:, 0], self.knots, self.efficiencies)
        return logit(np.clip(base_d, self.base_logit_floor, 1 - self.base_logit_floor))

    def predict_deficit(self, x):
        """Predict a bounded deficit without a forced full-absorption endpoint."""
        h = self.design(x) @ self.coefficients
        correction = self.correction_bound * np.tanh(h / self.correction_bound)
        return expit(self.logits(x) + correction)

    def predict(self, x, *, baseline=None):
        """Return absorption efficiency for native-only input coordinates."""
        return 1 - self.predict_deficit(x)

    def save(self, path):
        """Write a pickle-free development model without deployment hooks."""
        self.validate()
        np.savez_compressed(
            path,
            schema=1,
            kind="anchored_deficit_development",
            **{
                key: value
                for key, value in vars(self).items()
                if key not in ("output_mode", "budget_kind")
            },
        )

    @classmethod
    def load(cls, path):
        """Restore a validated development model from exact saved parameters."""
        with np.load(path, allow_pickle=False) as a:
            if (
                int(a["schema"]) != 1
                or str(a["kind"]) != "anchored_deficit_development"
            ):
                raise ValueError("Unexpected budget schema")
            model = cls(
                **{
                    k: np.array(a[k])
                    for k in (
                        "knots",
                        "efficiencies",
                        "trend",
                        "indices",
                        "mean",
                        "scale",
                        "coefficients",
                    )
                },
                feature_count=int(a["feature_count"]),
                correction_bound=float(a["correction_bound"]),
                base_logit_floor=float(a["base_logit_floor"]),
            )
        model.validate()
        return model


def fit(x, eta, mass_scale, indices, ridge, *, correction_bound=3.0):
    """Fit only on training native features and bounded absorption labels."""
    if (
        isinstance(ridge, bool)
        or not np.isfinite(ridge)
        or ridge <= 0
        or isinstance(correction_bound, bool)
        or not np.isfinite(correction_bound)
        or correction_bound <= 0
    ):
        raise ValueError("Budget regularization must be finite and positive")
    if any(type(index) is not int or not 0 <= index < 14 for index in indices):
        raise ValueError("Native feature indices must be integers in [0,14)")
    if len(set(indices)) != len(indices):
        raise ValueError("Native feature indices must be unique")
    if any(np.asarray(a).dtype.kind not in "fiu" for a in (x, eta, mass_scale)):
        raise ValueError("Training inputs must contain real numeric values")
    x, eta, mass_scale = (np.asarray(a, dtype=float) for a in (x, eta, mass_scale))
    if (
        x.ndim != 2
        or x.shape[1] != 14
        or eta.shape != (len(x),)
        or mass_scale.shape != eta.shape
        or np.any((eta < 0) | (eta > 1))
        or np.any((mass_scale < 0) | (mass_scale > 1))
        or not np.isfinite(x).all()
        or not np.isfinite(eta).all()
        or not np.isfinite(mass_scale).all()
        or ridge <= 0
    ):
        raise ValueError("Invalid training-only budget inputs")
    groups = [x[:, 1] == z for z in np.unique(x[:, 1])]
    knots = np.array([x[g, 0].mean() for g in groups])
    order = np.argsort(knots)
    model = AnchoredDeficit(
        knots[order],
        np.array([eta[g].mean() for g in groups])[order],
        np.array([x[g].mean(axis=0) for g in groups])[order],
        np.asarray(indices, dtype=int),
        np.zeros(3 + len(indices)),
        np.ones(3 + len(indices)),
        np.zeros(4 + len(indices)),
        correction_bound=correction_bound,
    )
    initial = model.features(x)
    model.mean = initial.mean(axis=0)
    model.scale = np.maximum(initial.std(axis=0), 1e-10)
    design, logits = model.design(x), model.logits(x)
    target = 1 - eta
    neutral = 1 - mass_scale * eta
    weights = mass_scale**2 * (1 + 0.1 / np.maximum(neutral, 0.01) ** 2)

    def loss_gradient(beta):
        h = design @ beta
        tanh = np.tanh(h / correction_bound)
        pred = expit(logits + correction_bound * tanh)
        residual = pred - target
        loss = np.mean(weights * residual**2) + ridge * np.mean(beta**2)
        grad = design.T @ (
            2 / len(x) * weights * residual * pred * (1 - pred) * (1 - tanh**2)
        )
        grad += 2 * ridge * beta / len(beta)
        return loss, grad

    result = minimize(
        loss_gradient,
        model.coefficients,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-10, "maxls": 50},
    )
    if not result.success:
        raise RuntimeError(f"Budget optimization did not converge: {result.message}")
    model.coefficients = np.array(result.x)
    model.validate()
    return (
        model,
        {
            "success": bool(result.success),
            "message": str(result.message),
            "iterations": int(result.nit),
            "regularized_training_loss": float(result.fun),
            "gradient_maximum": float(np.max(np.abs(result.jac))),
        },
        loss_gradient,
    )
