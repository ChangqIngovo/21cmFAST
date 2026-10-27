"""Small RT-summary neural emulator and explicitly conditional mock inference.

This is a parameter-identifiability diagnostic, not a claim of a calibrated
cosmological posterior or a one-percent field emulator. The mock is a held-out
native RT run. Its assumed noise and the independent calibration residual
covariance are both retained in the likelihood and output record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from scipy.ndimage import gaussian_filter
from scipy.stats import norm, rankdata
from sklearn import __version__ as sklearn_version
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network._multilayer_perceptron import _pack

PARAMETER_LABELS = {
    "log10_source_amplitude": r"$\log_{10} A_\gamma$",
    "history_tilt": r"$\beta_{\rm history}$",
    "source_concentration": r"$p_{\rm source}$",
    "density_contrast_scale": r"$s_\rho$",
    "log10_Tbb_K": r"$\log_{10}(T_{\rm BB}/{\rm K})$",
    "log10_clumping": r"$\log_{10} C$",
}


@lru_cache(maxsize=1)
def multioutput_gradient_factor():
    """Probe installed sklearn on synthetic data before touching pilot labels.

    Return one for a correct upstream gradient or three for the known
    three-output normalization error. Fail rather than silently adapting any
    unrecognized implementation. A future upstream fix therefore does not get
    corrected twice. The finite differences include all weights, biases and L2.
    """
    rng = np.random.default_rng(817)
    x, y = rng.normal(size=(7, 2)), rng.normal(size=(7, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        probe = MLPRegressor(hidden_layer_sizes=(3,), activation="tanh", solver="lbfgs",
                             alpha=0.2, max_iter=1, random_state=4).fit(x, y)
    packed = _pack(probe.coefs_, probe.intercepts_)
    packed = packed + rng.normal(scale=0.1, size=packed.shape)
    arguments = (x, y, None,
                 [x] + [np.empty((len(x), w.shape[1])) for w in probe.coefs_],
                 [np.empty((len(x), w.shape[1])) for w in probe.coefs_],
                 [np.empty_like(w) for w in probe.coefs_],
                 [np.empty_like(b) for b in probe.intercepts_])
    loss, gradient = probe._loss_grad_lbfgs(packed, *arguments)
    expected_loss = (0.5 * np.mean((y - arguments[3][-1]) ** 2)
                     + 0.5 * probe.alpha * sum(np.sum(w ** 2) for w in probe.coefs_) / len(x))
    if not np.isclose(loss, expected_loss, rtol=1e-10, atol=1e-12):
        raise RuntimeError(f"Unrecognized sklearn {sklearn_version} squared-loss normalization")
    penalty_gradient = _pack([probe.alpha * w / len(x) for w in probe.coefs_],
                             [np.zeros_like(b) for b in probe.intercepts_])
    finite_difference = np.empty_like(gradient)
    for index in range(len(packed)):
        plus, minus = packed.copy(), packed.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        finite_difference[index] = (probe._loss_grad_lbfgs(plus, *arguments)[0]
                                    - probe._loss_grad_lbfgs(minus, *arguments)[0]) / 2e-6
    for factor in (1, 3):
        adjusted = (gradient - penalty_gradient) / factor + penalty_gradient
        if np.allclose(adjusted, finite_difference, rtol=2e-5, atol=1e-8):
            return factor
    raise RuntimeError(f"Unrecognized sklearn {sklearn_version} multi-output gradient; finite-difference probe failed")


class MeanOutputMLPRegressor(MLPRegressor):
    """Keep the multi-output squared-loss value and its gradient consistent.

    Scikit-learn 1.8 averages squared loss over outputs but its backpropagation
    sums their data gradients. A synthetic runtime probe detects this behavior
    and leaves a corrected upstream implementation unchanged. When needed,
    divide the data term alone by the output count; L2 already has the correct
    normalization. Tests also check sample weights and single-output fits.
    """

    def _backprop(self, X, y, sample_weight, activations, deltas, coef_grads,
                  intercept_grads):
        loss, coef_grads, intercept_grads = super()._backprop(
            X, y, sample_weight, activations, deltas, coef_grads, intercept_grads)
        if self.loss != "squared_error":
            raise ValueError("The pilot gradient correction requires squared error")
        if multioutput_gradient_factor() == 1:
            return loss, coef_grads, intercept_grads
        outputs = y.shape[1]
        weight_sum = len(X) if sample_weight is None else sample_weight.sum()
        for index, coefficient in enumerate(self.coefs_):
            penalty_gradient = self.alpha * coefficient / weight_sum
            coef_grads[index] = ((coef_grads[index] - penalty_gradient) / outputs
                                 + penalty_gradient)
            intercept_grads[index] = intercept_grads[index] / outputs
        return loss, coef_grads, intercept_grads


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _strings(values):
    return np.asarray(values).astype(str)


def load_dataset(path):
    """Read the root pilot's summary archive without loading simulation cubes."""
    with np.load(path, allow_pickle=False) as archive:
        required = ("theta", "split", "case_id", "seed", "parameter_names",
                    "parameter_bounds", "k", "redshifts", "P21", "xHI_mean")
        missing = set(required) - set(archive.files)
        if missing:
            raise ValueError(f"Missing dataset arrays: {sorted(missing)}")
        data = {key: np.array(archive[key]) for key in required}
    for name in ("split", "case_id", "parameter_names"):
        data[name] = _strings(data[name])
    n, d = data["theta"].shape
    nz, nk = len(data["redshifts"]), len(data["k"])
    shapes = {"split": (n,), "case_id": (n,), "seed": (n,),
              "parameter_names": (d,), "parameter_bounds": (d, 2),
              "P21": (n, nz, nk), "xHI_mean": (n, nz)}
    for key, shape in shapes.items():
        if data[key].shape != shape:
            raise ValueError(f"{key} has shape {data[key].shape}, expected {shape}")
    for key in ("theta", "parameter_bounds", "k", "redshifts", "P21", "xHI_mean"):
        if not np.isfinite(data[key]).all():
            raise ValueError(f"Nonfinite {key}")
    bounds = data["parameter_bounds"]
    if np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("Parameter bounds must have positive width")
    main = np.isin(data["split"], ("train", "validation", "test"))
    if (np.any(data["theta"][main] < bounds[:, 0] - 1e-12)
            or np.any(data["theta"][main] > bounds[:, 1] + 1e-12)):
        raise ValueError("Main inference parameters fall outside the declared prior")
    if len(set(data["case_id"].tolist())) != n:
        raise ValueError("Case IDs must be unique")
    if np.any(data["P21"] < 0) or np.any((data["xHI_mean"] < 0) | (data["xHI_mean"] > 1)):
        raise ValueError("Power or neutral fractions outside physical bounds")
    return data


def observables(data, *, dimension=8, box_len=50.0, last_redshifts=3,
                k_nyquist_fraction=0.6, power_floor=1e-12):
    """Use predeclared late redshifts and modes safely below the grid Nyquist."""
    if (dimension < 2 or box_len <= 0 or not 0 < k_nyquist_fraction < 1
            or last_redshifts < 1 or power_floor <= 0):
        raise ValueError("Invalid observable configuration")
    z_order = np.argsort(data["redshifts"])[::-1]
    z_index = z_order[-min(last_redshifts, len(z_order)):]
    k_max = k_nyquist_fraction * np.pi * dimension / box_len
    k_index = np.flatnonzero((data["k"] > 0) & (data["k"] <= k_max))
    if not len(k_index):
        raise ValueError("No power-spectrum modes below the selected cutoff")
    power = data["P21"][:, z_index][:, :, k_index]
    log_power = np.log(np.maximum(power, power_floor)).reshape(len(power), -1)
    neutral = data["xHI_mean"][:, z_index]
    y = np.column_stack((log_power, neutral))
    labels = [f"lnP(z={data['redshifts'][zi]:.6g},k={data['k'][ki]:.6g})"
              for zi in z_index for ki in k_index]
    labels.extend(f"xHI_mean(z={data['redshifts'][zi]:.6g})" for zi in z_index)
    return y, {
        "labels": labels, "power_columns": int(log_power.shape[1]),
        "redshift_indices": z_index.tolist(), "k_indices": k_index.tolist(),
        "redshifts": data["redshifts"][z_index].tolist(),
        "k_cMpc_inv": data["k"][k_index].tolist(),
        "dimension": dimension, "box_len_cMpc": box_len,
        "k_cut_cMpc_inv": float(k_max), "k_nyquist_fraction": k_nyquist_fraction,
        "power_floor": power_floor,
        "floored_power_values": int(np.count_nonzero(power <= power_floor)),
        "power_definition": "native RT saturated-spin-temperature, no-RSD 21-cm power",
    }


def split_indices(data, *, require_test=True):
    """Keep mock tests and external ICs out of selection and error calibration."""
    def indices(label):
        aliases = ("external_test", "external") if label == "external" else (label,)
        ids = np.flatnonzero(np.isin(data["split"], aliases))
        return ids[np.argsort(data["case_id"][ids])]
    train, validation, test = (indices(label) for label in ("train", "validation", "test"))
    if len(train) < 8 or len(validation) < 4 or (require_test and len(test) < 1):
        raise ValueError("Need at least 8 train, 4 validation and 1 test histories")
    midpoint = len(validation) // 2
    result = {"train": train, "selection": validation[:midpoint],
              "calibration": validation[midpoint:], "test": test,
              "external": indices("external")}
    main_seeds = np.unique(data["seed"][np.concatenate((train, validation, test))])
    if len(main_seeds) != 1:
        raise ValueError("This conditional pilot requires one fixed IC for main inference")
    return result


@dataclass
class SummaryEmulator:
    """Small ensemble predicting all log-power and global-history outputs."""

    models: list
    bounds: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray

    def normalized(self, theta):
        """Map declared prior coordinates to the fixed interval [-1, 1]."""
        return 2 * (np.asarray(theta) - self.bounds[:, 0]) / np.diff(self.bounds, axis=1).ravel() - 1

    def predict(self, theta):
        """Return the ensemble mean in natural-log-power and fraction units."""
        x = self.normalized(np.atleast_2d(theta))
        values = np.asarray([model.predict(x) for model in self.models])
        if values.ndim == 2:
            values = values[:, :, None]
        return values.mean(axis=0) * self.y_scale + self.y_mean

    def save(self, path):
        """Write portable tanh/linear network weights without a pickle."""
        arrays = {"bounds": self.bounds, "y_mean": self.y_mean, "y_scale": self.y_scale,
                  "n_models": np.asarray(len(self.models))}
        for i, model in enumerate(self.models):
            arrays[f"model_{i}_layers"] = np.asarray(len(model.coefs_))
            for j, (weight, bias) in enumerate(zip(model.coefs_, model.intercepts_, strict=True)):
                arrays[f"model_{i}_weight_{j}"] = weight
                arrays[f"model_{i}_bias_{j}"] = bias
        np.savez_compressed(path, **arrays)


def fit_emulator(theta, y, bounds, split, *, mock_sigma,
                 candidates=(((12,), 0.1), ((24,), 0.1), ((16, 16), 0.1),
                             ((12,), 1.0), ((24,), 1.0), ((16, 16), 1.0)),
                 model_seeds=(11, 23), max_iter=1000):
    """Select using half the validation cases; calibrate uncertainty elsewhere."""
    train, selection = split["train"], split["selection"]
    y_mean = y[train].mean(axis=0)
    y_scale = np.maximum(y[train].std(axis=0), mock_sigma)
    x = 2 * (theta - bounds[:, 0]) / np.diff(bounds, axis=1).ravel() - 1
    standardized = (y - y_mean) / y_scale
    trials, selected, best = [], None, np.inf
    for hidden, alpha in candidates:
        models, notes = [], []
        for seed in model_seeds:
            model = MeanOutputMLPRegressor(
                hidden_layer_sizes=hidden, activation="tanh", solver="lbfgs",
                alpha=alpha, random_state=seed, max_iter=max_iter,
                max_fun=max(15000, 20 * max_iter), tol=1e-8)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(x[train], standardized[train])
            models.append(model)
            notes.extend(str(w.message) for w in caught)
        candidate = SummaryEmulator(models, bounds, y_mean, y_scale)
        residual = candidate.predict(theta[selection]) - y[selection]
        score = float(np.mean((residual / mock_sigma) ** 2))
        trials.append({"hidden": list(hidden), "alpha": alpha, "model_seeds": list(model_seeds),
                       "selection_mean_squared_error_in_mock_noise_units": score,
                       "iterations": [int(m.n_iter_) for m in models], "warnings": notes})
        if score < best:
            best, selected = score, candidate
    return selected, {"trials": trials, "selected_trial": int(np.argmin([
        r["selection_mean_squared_error_in_mock_noise_units"] for r in trials])),
        "selection_uses_test": False, "standardizer_uses_training_only": True,
        "optimizer_objective": "mean squared error over samples and outputs plus L2; multi-output gradient verified by synthetic finite differences",
        "sklearn_version": sklearn_version,
        "base_gradient_factor_in_three_output_probe": multioutput_gradient_factor(),
        "multioutput_gradient_correction_applied": multioutput_gradient_factor() != 1}


def calibration_covariance(residual, *, shrinkage=0.5):
    """Uncentered second moment retains emulator bias as prediction uncertainty."""
    residual = np.asarray(residual, dtype=float)
    if residual.ndim != 2 or len(residual) < 2 or not np.isfinite(residual).all():
        raise ValueError("At least two finite calibration residual vectors are required")
    if not 0 <= shrinkage <= 1:
        raise ValueError("Shrinkage must be in [0,1]")
    second = residual.T @ residual / len(residual)
    return (1 - shrinkage) * second + shrinkage * np.diag(np.diag(second))


def numerical_sensitivity(data, y, power_columns):
    """Construct a heuristic stress matrix from actual fixed-physics refinements.

    Three changes at one fiducial point are neither draws from an error
    distribution nor a convergence certificate. The matrix is used only to show
    posterior sensitivity to the measured numerical shifts.
    """
    names = ("convergence_angle4", "convergence_time2", "convergence_space5")
    lookup = {str(name): i for i, name in enumerate(data["case_id"])}
    missing = [name for name in ("fiducial", *names) if name not in lookup]
    record = {
        "physical_posterior_validated": False,
        "interpretation": "single-fiducial numerical sensitivity stress, NOT calibrated numerical-error covariance and NOT a cure for nonconvergence",
        "required_cases": ["fiducial", *names], "missing_cases": missing,
        "refinement_threshold_relative_power": 0.003,
        "fiducial_refinement_threshold_passed": False,
        "construction": "equal-weight uncentered second moment of observable shifts, 50% diagonal shrinkage",
    }
    if missing:
        record["reason"] = "Incomplete numerical controls; no stress posterior constructed"
        return None, record
    fid = lookup["fiducial"]
    indices = np.array([lookup[name] for name in names])
    if (not np.all(data["seed"][indices] == data["seed"][fid])
            or not np.allclose(data["theta"][indices], data["theta"][fid], rtol=0, atol=1e-12)):
        raise ValueError("Numerical sensitivity controls must share fiducial physics and IC")
    if data["split"][fid] != "control" or not np.all(data["split"][indices] == "convergence"):
        raise ValueError("Numerical controls must be independently labelled convergence cases")
    shifts = y[indices] - y[fid]
    relative = np.expm1(shifts[:, :power_columns])
    matrix = calibration_covariance(shifts)
    largest = float(np.max(np.abs(relative)))
    record.update({
        "fiducial_case_id": "fiducial", "conditional_IC_seed": int(data["seed"][fid]),
        "observable_shifts": shifts.tolist(),
        "cases": {name: {"power_relative_shifts": relative[j].tolist(),
                         "maximum_absolute_relative_power_shift": float(np.max(np.abs(relative[j]))),
                         "global_xHI_shifts": shifts[j, power_columns:].tolist()}
                  for j, name in enumerate(names)},
        "largest_absolute_relative_power_shift": largest,
        "fiducial_refinement_threshold_passed": largest <= 0.003,
        "reason": ("Measured numerical changes exceed the 0.3% teacher budget; the physical posterior is not validated"
                   if largest > 0.003 else "Single-point refinement agreement alone does not certify the physical posterior"),
    })
    return matrix, record


def gaussian_log_probability(theta, emulator, observed, covariance, bounds, columns):
    """Gaussian mock likelihood with bounded uniform parameter-coordinate prior."""
    theta = np.atleast_2d(theta)
    out = np.full(len(theta), -np.inf)
    inside = np.all((theta >= bounds[:, 0]) & (theta <= bounds[:, 1]), axis=1)
    if inside.any():
        factor = np.linalg.cholesky(covariance)
        residual = emulator.predict(theta[inside])[:, columns] - observed
        white = solve_triangular(factor, residual.T, lower=True).T
        out[inside] = -0.5 * np.sum(white**2, axis=1)
        out[inside] -= np.log(np.diag(factor)).sum() + len(observed) * np.log(2 * np.pi) / 2
    return out


def metropolis(log_probability, bounds, *, chains=8, burn=3000, steps=10000,
               thin=5, seed=721):
    """Independent adaptive-Metropolis chains; adaptation stops before retention."""
    bounds = np.asarray(bounds, dtype=float)
    d = len(bounds)
    if chains < 4 or burn < 100 or steps < 100 or thin < 1:
        raise ValueError("Need >=4 chains and >=100 burn/production steps")
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-0.8, 0.8, size=(chains, d))
    center, width = bounds.mean(axis=1), np.diff(bounds, axis=1).ravel() / 2
    def physical(values):
        return center + width * values
    logp = np.asarray(log_probability(physical(positions)))
    scales = np.full(chains, 0.15)
    roots = np.broadcast_to(np.eye(d), (chains, d, d)).copy()
    running_mean = positions.copy()
    running_m2 = np.zeros((chains, d, d))
    block_accept = np.zeros(chains)
    accept = np.zeros(chains, dtype=int)
    retained = []
    for iteration in range(burn + steps):
        # The isotropic mixture preserves the ability to move along a direction
        # whose empirical covariance is temporarily small during burn-in.
        isotropic = rng.random(chains) < 0.05
        draw = rng.normal(size=(chains, d))
        delta = np.einsum("nij,nj->ni", roots, draw) * scales[:, None]
        delta[isotropic] = 0.08 * draw[isotropic]
        proposal = positions + delta
        proposed_logp = np.asarray(log_probability(physical(proposal)))
        accepted = np.log(rng.random(chains)) < proposed_logp - logp
        positions[accepted], logp[accepted] = proposal[accepted], proposed_logp[accepted]
        if iteration < burn:
            block_accept += accepted
            count = iteration + 2
            difference = positions - running_mean
            running_mean += difference / count
            running_m2 += np.einsum("ni,nj->nij", difference, positions - running_mean)
            if (iteration + 1) % 100 == 0:
                scales *= np.exp(np.clip(block_accept / 100 - 0.25, -0.2, 0.2))
                scales = np.clip(scales, 0.01, 3.0)
                block_accept[:] = 0
                if count >= 300:
                    covariance = running_m2 / (count - 1)
                    roots = np.linalg.cholesky(covariance + np.eye(d)[None] * 1e-3)
        else:
            accept += accepted
            if (iteration - burn) % thin == 0:
                retained.append(physical(positions).copy())
    samples = np.asarray(retained).transpose(1, 0, 2)
    return samples, {"algorithm": "independent adaptive Metropolis; adaptation only in burn",
                     "chains": chains, "burn_steps": burn, "production_steps": steps,
                     "thin": thin, "random_seed": seed,
                     "acceptance_fraction": (accept / steps).tolist(),
                     "final_normalized_proposal_scale": scales.tolist()}


def _rhat(samples):
    _m, n = samples.shape
    within = np.var(samples, axis=1, ddof=1).mean()
    between = n * np.var(samples.mean(axis=1), ddof=1)
    if within <= 0:
        return 1.0 if between <= 0 else float("inf")
    return float(np.sqrt(((n - 1) / n * within + between / n) / within))


def _ess(samples):
    m, n = samples.shape
    centered = samples - samples.mean(axis=1, keepdims=True)
    transform = np.fft.rfft(centered, n=2 * n, axis=1)
    acov = np.fft.irfft(transform * transform.conj(), n=2 * n, axis=1)[:, :n] / n
    within = np.var(samples, axis=1, ddof=1).mean()
    between = n * np.var(samples.mean(axis=1), ddof=1)
    variance = (n - 1) / n * within + between / n
    if variance <= 0:
        return float(m * n)
    rho = 1 - (within - acov.mean(axis=0)) / variance
    rho[0] = 1
    paired = []
    for i in range(0, n - 1, 2):
        pair = rho[i] + rho[i + 1]
        if pair < 0:
            break
        paired.append(min(pair, paired[-1] if paired else pair))
    tau = max(-1 + 2 * sum(paired), 1.0)
    return float(min(m * n / tau, m * n))


def chain_diagnostics(samples):
    """Rank-normalized/folded split R-hat and bulk ESS for independent chains."""
    samples = np.asarray(samples, dtype=float)
    m, n, d = samples.shape
    half = n // 2
    if m < 2 or half < 10 or not np.isfinite(samples).all():
        raise ValueError("Too few or nonfinite independent-chain samples")
    split = np.concatenate((samples[:, :half], samples[:, -half:]), axis=0)
    rhats, effective = [], []
    for j in range(d):
        x = split[:, :, j]
        if np.ptp(x) == 0:
            raise ValueError("No posterior exploration in a parameter")
        rank = rankdata(x.ravel()).reshape(x.shape)
        z = norm.ppf((rank - 0.375) / (x.size + 0.25))
        folded = np.abs(x - np.median(x))
        folded_rank = rankdata(folded.ravel()).reshape(x.shape)
        folded_z = norm.ppf((folded_rank - 0.375) / (x.size + 0.25))
        rhats.append(max(_rhat(z), _rhat(folded_z)))
        effective.append(_ess(z))
    return {"rank_normalized_split_rhat": rhats, "bulk_ess": effective,
            "passed": bool(max(rhats) < 1.01 and min(effective) >= 400),
            "thresholds": {"rhat_below": 1.01, "bulk_ess_at_least": 400}}


def posterior_summary(samples, truth, bounds, names):
    """Summarize injection recovery and contraction relative to the prior."""
    flat = samples.reshape(-1, samples.shape[-1])
    quantiles = np.quantile(flat, (0.025, 0.16, 0.5, 0.84, 0.975), axis=0)
    result = {}
    for i, name in enumerate(names):
        q = quantiles[:, i]
        result[str(name)] = {
            "truth": float(truth[i]), "q025_q16_median_q84_q975": q.tolist(),
            "truth_inside_95_percent_marginal": bool(q[0] <= truth[i] <= q[-1]),
            "truth_posterior_quantile": float(np.mean(flat[:, i] <= truth[i])),
            "posterior_95_width_over_prior_95_width": float(
                (q[-1] - q[0]) / (0.95 * (bounds[i, 1] - bounds[i, 0]))),
        }
    return result


def corner_plot(samples, truth, bounds, names, path, *, title, comparison=None):
    """Draw one- and two-dimensional posterior marginals with the injected truth."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    flat = samples.reshape(-1, samples.shape[-1])
    layers = [(flat, "#256d85")]
    if comparison is not None:
        layers.append((comparison.reshape(-1, comparison.shape[-1]), "#c48231"))
    d = flat.shape[1]
    fig, axes = plt.subplots(d, d, figsize=(2.05 * d, 2.05 * d), squeeze=False)
    for row in range(d):
        for col in range(d):
            ax = axes[row, col]
            if row < col:
                ax.set_visible(False)
                continue
            ax.set_xlim(bounds[col])
            if row == col:
                for number, (values, color) in enumerate(layers):
                    ax.hist(values[:, col], bins=35, range=tuple(bounds[col]), density=True,
                            color=color, alpha=0.65 if number == 0 else 1.0,
                            histtype="bar" if number == 0 else "step", linewidth=1.4)
                ax.axhline(1 / np.diff(bounds[col])[0], color="0.5", lw=0.8, ls=":")
                ax.axvline(truth[col], color="#bb4b35", lw=1.4, ls="--")
                q = np.quantile(flat[:, col], (0.16, 0.5, 0.84))
                ax.set_title(f"Median {q[1]:.3g}\n68% [{q[0]:.3g}, {q[2]:.3g}]", fontsize=8)
                ax.set_yticks([])
            else:
                for values, color in layers:
                    hist, xedge, yedge = np.histogram2d(values[:, col], values[:, row], bins=35,
                                                      range=[bounds[col], bounds[row]])
                    hist = gaussian_filter(hist, 1.0)
                    ordered = np.sort(hist.ravel())[::-1]
                    cumulative = np.cumsum(ordered) / ordered.sum()
                    levels = sorted({float(ordered[min(np.searchsorted(cumulative, p),
                                                           len(ordered) - 1)])
                                        for p in (0.95, 0.68)})
                    x, yy = (xedge[:-1] + xedge[1:]) / 2, (yedge[:-1] + yedge[1:]) / 2
                    if levels and max(hist.ravel()) > min(levels):
                        ax.contour(x, yy, hist.T, levels=levels, colors=color, linewidths=1.1)
                ax.axvline(truth[col], color="#bb4b35", lw=0.8, ls="--")
                ax.axhline(truth[row], color="#bb4b35", lw=0.8, ls="--")
                ax.plot(truth[col], truth[row], "+", color="#bb4b35", ms=6)
                ax.set_ylim(bounds[row])
            ax.tick_params(labelsize=7)
            if row != d - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(PARAMETER_LABELS.get(str(names[col]), str(names[col])), fontsize=10)
            if col == 0 and row > 0:
                ax.set_ylabel(PARAMETER_LABELS.get(str(names[row]), str(names[row])), fontsize=10)
            elif row != col:
                ax.set_yticklabels([])
    fig.suptitle(title + "\nDashed: injected truth; contours: 68% / 95%; dotted: uniform prior", fontsize=12)
    if comparison is not None:
        from matplotlib.lines import Line2D
        fig.legend(handles=[Line2D([0], [0], color="#256d85", label="Fixed discretization"),
                            Line2D([0], [0], color="#c48231", label="Numerical sensitivity stress")],
                   loc="upper right", bbox_to_anchor=(0.94, 0.86), frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.0, w_pad=0.5)
    fig.savefig(path, dpi=180)
    fig.savefig(Path(path).with_suffix(".pdf"))
    plt.close(fig)


def _error_metrics(prediction, truth, power_columns):
    relative = np.expm1(np.clip(prediction[:, :power_columns] - truth[:, :power_columns], -700, 700))
    global_error = prediction[:, power_columns:] - truth[:, power_columns:]
    return {"histories": len(truth), "power_relative_error_median_abs": float(np.median(np.abs(relative))),
            "power_relative_error_p95_abs": float(np.quantile(np.abs(relative), 0.95)),
            "power_relative_error_max_abs": float(np.max(np.abs(relative))),
            "global_xHI_MAE": float(np.mean(np.abs(global_error))),
            "predicted_global_xHI_outside_physical_range": int(np.count_nonzero(
                (prediction[:, power_columns:] < 0) | (prediction[:, power_columns:] > 1))),
            "per_history_power_median_abs": np.median(np.abs(relative), axis=1).tolist(),
            "per_observable_log_power_RMSE": np.sqrt(np.mean(
                (prediction[:, :power_columns] - truth[:, :power_columns])**2, axis=0)).tolist()}


def fit_forecast(dataset, output, *, dimension=8, box_len=50.0, last_redshifts=3,
                 power_noise=0.10, neutral_noise=0.02, max_iter=1000):
    """Assess training size on validation only, before any final-test inference."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    data = load_dataset(dataset)
    split = split_indices(data, require_test=False)
    y, info = observables(data, dimension=dimension, box_len=box_len,
                          last_redshifts=last_redshifts)
    n_power = info["power_columns"]
    if power_noise <= 0 or neutral_noise <= 0:
        raise ValueError("Mock noise scales must be positive")
    sigma = np.r_[np.full(n_power, np.log1p(power_noise)),
                  np.full(y.shape[1] - n_power, neutral_noise)]
    model, selected = fit_emulator(data["theta"], y, data["parameter_bounds"], split,
                                   mock_sigma=sigma, max_iter=max_iter)
    model.save(output / "summary_emulator.npz")
    report = {
        "stage": "training and selection-half forecast; calibration and final tests not evaluated",
        "dataset_sha256": hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
        "physical_posterior_validated": False, "observables": info,
        "selection": selected, "accuracy": {}, "case_ids": {},
        "untouched_calibration_case_ids": data["case_id"][split["calibration"]].tolist(),
    }
    for name in ("train", "selection"):
        ids = split[name]
        pred = model.predict(data["theta"][ids])
        report["accuracy"][name] = _error_metrics(pred, y[ids], n_power)
        report["case_ids"][name] = data["case_id"][ids].tolist()
    _, report["numerical_sensitivity"] = numerical_sensitivity(data, y, n_power)
    _json(output / "forecast.json", report)
    (output / "COMPLETED").write_text("Selection-only forecast finished; no calibration, posterior, or final-test evaluation.\n")
    print(json.dumps({"output": str(output.resolve()), "selection_accuracy": report["accuracy"]["selection"]}, indent=2))  # noqa: T201
    return report


def run_inference(dataset, output, *, dimension=8, box_len=50.0, last_redshifts=3,
                  power_noise=0.10, neutral_noise=0.02, mock_seed=20261027,
                  chains=8, burn=3000, steps=30000, thin=5, max_iter=1000):
    """Train the pilot summary emulator, sample two mock posteriors, and report."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    data = load_dataset(dataset)
    split = split_indices(data)
    y, observable_info = observables(data, dimension=dimension, box_len=box_len,
                                    last_redshifts=last_redshifts)
    n_power = observable_info["power_columns"]
    if power_noise <= 0 or neutral_noise <= 0:
        raise ValueError("Mock noise scales must be positive")
    noise_sigma = np.r_[np.full(n_power, np.log1p(power_noise)),
                        np.full(y.shape[1] - n_power, neutral_noise)]
    mock_index = int(split["test"][0])  # Declared before fitting or test evaluation.
    provenance = {
        "dataset": str(Path(dataset).resolve()),
        "dataset_sha256": hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
        "experiment": "small native-RT summary neural emulator and conditional mock inference",
        "parameter_names": data["parameter_names"].tolist(),
        "parameter_bounds": data["parameter_bounds"].tolist(),
        "prior": "independent uniform in the listed parameter coordinates; log-named coordinates imply log-uniform physical priors",
        "conditional_IC_seed": int(data["seed"][mock_index]),
        "case_ids": {key: data["case_id"][value].tolist() for key, value in split.items()},
        "mock_case_id": str(data["case_id"][mock_index]),
        "mock_choice": "lexicographically first test case, fixed before emulator fitting",
        "mock_seed": mock_seed, "mock_kind": "seeded noisy simulation injection, not observations",
        "noise_sigma": noise_sigma.tolist(), "power_noise_ln_sigma": float(np.log1p(power_noise)),
        "global_neutral_fraction_absolute_noise_sigma": neutral_noise,
        "covariance_scope": "assumed mock measurement noise plus fixed-IC emulator calibration error; no observational or cosmic-variance claim",
        "observables": observable_info,
    }
    _json(output / "experiment.json", provenance)
    model, selection_report = fit_emulator(data["theta"], y, data["parameter_bounds"], split,
                                           mock_sigma=noise_sigma, max_iter=max_iter)
    model.save(output / "summary_emulator.npz")
    _json(output / "model_selection.json", selection_report)
    prediction = model.predict(data["theta"])
    calibration = prediction[split["calibration"]] - y[split["calibration"]]
    emulator_covariance = calibration_covariance(calibration)
    covariance = emulator_covariance + np.diag(noise_sigma**2)
    stress_covariance, numerical_report = numerical_sensitivity(data, y, n_power)
    _json(output / "numerical_sensitivity.json", numerical_report)
    noisy_mock = y[mock_index] + np.random.default_rng(mock_seed).normal(size=y.shape[1]) * noise_sigma
    np.savez_compressed(output / "prediction_and_covariance.npz", prediction=prediction, truth=y,
                        calibration_residual=calibration, emulator_covariance=emulator_covariance,
                        total_covariance=covariance, noise_sigma=noise_sigma,
                        noisy_mock=noisy_mock, mock_truth=y[mock_index], theta=data["theta"],
                        case_id=data["case_id"], split=data["split"],
                        numerical_sensitivity_covariance=(stress_covariance if stress_covariance is not None
                                                          else np.empty((0, 0))))
    report = {"selection": selection_report, "accuracy": {}, "posteriors": {},
              "physical_posterior_validated": False, "numerical_sensitivity": numerical_report,
              "emulator_uncertainty": {
                  "calibration_histories": len(calibration), "diagonal_shrinkage": 0.5,
                  "uses_uncentered_residual_second_moment": True,
                  "sigma_over_assumed_mock_sigma": (np.sqrt(np.diag(emulator_covariance))
                                                     / noise_sigma).tolist(),
                  "emulator_fraction_of_total_variance": (np.diag(emulator_covariance)
                                                          / np.diag(covariance)).tolist()},
              "interpretation": [
                  "A wide posterior diagnoses identifiability under this mock likelihood, not whether a physical input can be omitted.",
                  "The emulator calibration covariance uses only the reserved half of validation histories.",
                  "This is one injection recovery diagnostic, not simulation-based calibration or a one-percent accuracy demonstration.",
                  "All main inference cases use the same IC. External-IC accuracy is reported separately and is not absorbed into mock noise.",
                  "The numerical-stress posterior uses a heuristic matrix from actual single-fiducial refinement shifts; it is not a calibrated physical posterior."]}
    for name, ids in split.items():
        if len(ids):
            report["accuracy"][name] = _error_metrics(prediction[ids], y[ids], n_power)
            report["accuracy"][name]["case_ids"] = data["case_id"][ids].tolist()
    native_candidates = {"parameter_names": data["parameter_names"].tolist(),
                         "conditional_IC_seed": int(data["seed"][mock_index]),
                         "native_validation_completed": False, "posteriors": {}}
    variants = [("power_only", np.arange(n_power), covariance),
                ("power_and_global", np.arange(y.shape[1]), covariance)]
    if stress_covariance is not None:
        variants.append(("power_only_numerical_stress", np.arange(n_power), covariance + stress_covariance))
    saved_samples = {}
    for number, (name, columns, full_covariance) in enumerate(variants):
        case_covariance = full_covariance[np.ix_(columns, columns)]
        observed = noisy_mock[columns]
        def log_probability(theta, observed=observed, case_covariance=case_covariance,
                            columns=columns):
            return gaussian_log_probability(theta, model, observed, case_covariance,
                                            data["parameter_bounds"], columns)
        samples, sampling = metropolis(log_probability, data["parameter_bounds"], chains=chains,
                                       burn=burn, steps=steps, thin=thin, seed=mock_seed + 11 + number)
        diagnostic = chain_diagnostics(samples)
        saved_samples[name] = samples
        parameter_report = posterior_summary(samples, data["theta"][mock_index],
                                              data["parameter_bounds"], data["parameter_names"])
        flat = samples.reshape(-1, samples.shape[-1])
        draw_logp = log_probability(flat)
        candidate_vectors = {"coordinatewise_median": np.median(flat, axis=0),
                             "highest_likelihood_retained_draw": flat[np.argmax(draw_logp)]}
        prior_width = np.diff(data["parameter_bounds"], axis=1).ravel()
        native_candidates["posteriors"][name] = {}
        for kind, vector in candidate_vectors.items():
            distance = np.linalg.norm((data["theta"] - vector) / prior_width, axis=1)
            native_candidates["posteriors"][name][kind] = {
                "theta": vector.tolist(),
                "parameters": dict(zip(data["parameter_names"].tolist(), vector.tolist(), strict=True)),
                "neural_log_likelihood": float(log_probability(vector)[0]),
                "nearest_existing_case_id": str(data["case_id"][np.argmin(distance)]),
                "nearest_existing_design_distance_in_prior_widths": float(np.min(distance)),
            }
        rng = np.random.default_rng(mock_seed + 100 + number)
        chosen = flat[rng.choice(len(flat), size=min(2000, len(flat)), replace=False)]
        predicted_all = model.predict(chosen)
        predicted = predicted_all[:, columns]
        predicted_global = predicted_all[:, n_power:]
        replicated = predicted + rng.multivariate_normal(np.zeros(len(columns)), case_covariance,
                                                         size=len(chosen))
        low, high = np.quantile(replicated, (0.025, 0.975), axis=0)
        predictive = {
            "labels": [observable_info["labels"][i] for i in columns],
            "observed": observed.tolist(), "median": np.median(replicated, axis=0).tolist(),
            "q025": low.tolist(), "q975": high.tolist(),
            "observed_inside_95_percent": ((observed >= low) & (observed <= high)).tolist(),
            "neural_log_likelihood_at_injected_parameters": float(
                log_probability(data["theta"][mock_index])[0]),
            "largest_neural_log_likelihood_among_predictive_draws": float(
                np.max(log_probability(chosen))),
            "forward_global_xHI_range": [float(predicted_global.min()), float(predicted_global.max())],
            "posterior_draw_fraction_with_unphysical_global_xHI_prediction": float(np.mean(
                np.any((predicted_global < 0) | (predicted_global > 1), axis=1))),
        }
        np.savez_compressed(output / f"posterior_{name}.npz", chains=samples,
                            truth_parameters=data["theta"][mock_index], bounds=data["parameter_bounds"],
                            parameter_names=data["parameter_names"], observable_columns=columns)
        corner_plot(samples, data["theta"][mock_index], data["parameter_bounds"],
                    data["parameter_names"], output / f"corner_{name}.png",
                    title=(("Numerical sensitivity stress (not calibrated uncertainty)"
                            if name.endswith("numerical_stress") else
                            "Conditional RT pilot: " + name.replace("_", " "))
                           + f"\nIC {data['seed'][mock_index]} | {100 * power_noise:g}% power mock noise + emulator error"))
        report["posteriors"][name] = {"sampling": sampling, "diagnostics": diagnostic,
                                      "parameters": parameter_report, "posterior_predictive": predictive}
    if "power_only_numerical_stress" in saved_samples:
        corner_plot(saved_samples["power_only"], data["theta"][mock_index], data["parameter_bounds"],
                    data["parameter_names"], output / "corner_numerical_stress_comparison.png",
                    title="Conditional power posterior versus numerical sensitivity stress\nPhysical posterior not validated; diagonal intervals refer to fixed discretization",
                    comparison=saved_samples["power_only_numerical_stress"])
    report["all_chains_passed_diagnostics"] = all(
        value["diagnostics"]["passed"] for value in report["posteriors"].values())
    _json(output / "posterior_predictive_native_candidates.json", native_candidates)
    _json(output / "summary.json", report)
    (output / "COMPLETED").write_text("Inference finished; inspect mixing and predictive diagnostics.\n")
    print(json.dumps({"output": str(output.resolve()), "test_accuracy": report["accuracy"]["test"],  # noqa: T201
                      "all_chains_passed_diagnostics": report["all_chains_passed_diagnostics"]}, indent=2))
    return report


def main():
    """Run the explicitly bounded pilot inference from a summary archive."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dimension", type=int, default=8)
    parser.add_argument("--box-len", type=float, default=50.0)
    parser.add_argument("--last-redshifts", type=int, default=3)
    parser.add_argument("--power-noise", type=float, default=0.10)
    parser.add_argument("--neutral-noise", type=float, default=0.02)
    parser.add_argument("--mock-seed", type=int, default=20261027)
    parser.add_argument("--chains", type=int, default=8)
    parser.add_argument("--burn", type=int, default=3000)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--thin", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--fit-only", action="store_true",
                        help="Use training and validation only to decide training size; no final-test evaluation")
    arguments = vars(parser.parse_args())
    if arguments.pop("fit_only"):
        for key in ("mock_seed", "chains", "burn", "steps", "thin"):
            arguments.pop(key)
        fit_forecast(**arguments)
    else:
        run_inference(**arguments)


if __name__ == "__main__":
    main()
