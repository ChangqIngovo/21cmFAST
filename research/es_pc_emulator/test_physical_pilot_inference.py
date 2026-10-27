"""Statistical and data-separation checks for the small mock-inference pilot."""

import warnings

import numpy as np
import pytest
from sklearn.neural_network._multilayer_perceptron import _pack

from . import physical_pilot_inference as inference
from .physical_pilot_inference import (
    calibration_covariance,
    chain_diagnostics,
    fit_emulator,
    gaussian_log_probability,
    load_dataset,
    metropolis,
    numerical_sensitivity,
    observables,
    posterior_summary,
    split_indices,
)


def _data():
    rng = np.random.default_rng(19)
    n, d, nz, nk = 24, 3, 4, 4
    return {
        "theta": rng.uniform(-1, 1, size=(n, d)),
        "split": np.array(["train"] * 12 + ["validation"] * 6 + ["test"] * 4
                          + ["external_test"] * 2),
        "case_id": np.array([f"case{i:03d}" for i in range(n)]),
        "seed": np.array([111] * 22 + [222, 333]),
        "parameter_names": np.array(["a", "b", "c"]),
        "parameter_bounds": np.tile([-1.0, 1.0], (d, 1)),
        "k": np.array([0.15, 0.27, 0.39, 0.49]),
        "redshifts": np.array([12.0, 10.0, 9.0, 8.0]),
        "P21": np.exp(rng.normal(size=(n, nz, nk))),
        "xHI_mean": rng.uniform(0.1, 0.9, size=(n, nz)),
    }


def test_installed_sklearn_gradient_probe_recognizes_normalization():
    """The guard accepts only a verified correct or recognized factor gradient."""
    assert inference.multioutput_gradient_factor() in (1, 3)


@pytest.mark.parametrize("outputs", [1, 3, 9])
@pytest.mark.parametrize("weighted", [False, True])
def test_all_mlp_gradients_match_finite_differences_with_l2(outputs, weighted):
    """L-BFGS receives the actual objective derivative for every weight/bias."""
    rng = np.random.default_rng(77)
    x = rng.normal(size=(7, 2))
    y = rng.normal(size=(7, outputs))
    sample_weight = rng.uniform(0.5, 1.5, len(x)) if weighted else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = inference.MeanOutputMLPRegressor(
            hidden_layer_sizes=(3, 2), activation="tanh", solver="lbfgs",
            alpha=0.2, max_iter=1, random_state=4,
        ).fit(x, y.ravel() if outputs == 1 else y, sample_weight=sample_weight)
    packed = _pack(model.coefs_, model.intercepts_)
    packed = packed + rng.normal(scale=0.1, size=packed.shape)
    arguments = (
        x, y, sample_weight,
        [x] + [np.empty((len(x), w.shape[1])) for w in model.coefs_],
        [np.empty((len(x), w.shape[1])) for w in model.coefs_],
        [np.empty_like(w) for w in model.coefs_],
        [np.empty_like(b) for b in model.intercepts_],
    )
    _, gradient = model._loss_grad_lbfgs(packed, *arguments)
    finite_difference = np.empty_like(gradient)
    for index in range(len(packed)):
        plus, minus = packed.copy(), packed.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        finite_difference[index] = (
            model._loss_grad_lbfgs(plus, *arguments)[0]
            - model._loss_grad_lbfgs(minus, *arguments)[0]
        ) / 2e-6
    np.testing.assert_allclose(gradient, finite_difference, rtol=2e-5, atol=1e-8)


def test_data_split_reserves_independent_calibration_and_test(tmp_path):
    """Neither final tests nor unseen ICs enter model selection or calibration."""
    data = _data()
    path = tmp_path / "pilot.npz"
    np.savez(path, **data)
    data = load_dataset(path)
    split = split_indices(data)
    assert len(split["selection"]) == len(split["calibration"]) == 3
    combined = np.concatenate(list(split.values()))
    assert len(np.unique(combined)) == len(combined)
    assert list(data["seed"][split["external"]]) == [222, 333]
    data["split"][split["external"][0]] = "external"
    assert list(data["seed"][split_indices(data)["external"]]) == [222, 333]
    data["seed"][split["test"][0]] = 222
    with pytest.raises(ValueError, match="one fixed IC"):
        split_indices(data)


def test_invalid_input_rejected_before_fitting(tmp_path):
    """Nonfinite native summary data fail before neural fitting."""
    data = _data()
    data["P21"][0, 0, 0] = np.nan
    path = tmp_path / "bad.npz"
    np.savez(path, **data)
    with pytest.raises(ValueError, match="Nonfinite P21"):
        load_dataset(path)


def test_observable_cut_excludes_high_k_and_uses_late_snapshots():
    """Near-Nyquist modes and early snapshots do not enter the declared mock."""
    data = _data()
    y, info = observables(data)
    assert info["k_indices"] == [0, 1]
    assert info["redshift_indices"] == [1, 2, 3]
    assert info["power_columns"] == 6
    assert y.shape == (24, 9)
    np.testing.assert_allclose(y[:, :6], np.log(data["P21"][:, 1:, :2]).reshape(24, -1))
    np.testing.assert_allclose(y[:, 6:], data["xHI_mean"][:, 1:])


def test_uncertainty_retains_constant_emulator_bias():
    """A biased emulator must not appear perfectly precise after mean removal."""
    residual = np.tile([0.1, -0.2], (8, 1))
    covariance = calibration_covariance(residual)
    np.testing.assert_allclose(np.diag(covariance), [0.01, 0.04])
    assert np.linalg.eigvalsh(covariance).min() > 0
    with pytest.raises(ValueError, match="Shrinkage"):
        calibration_covariance(residual, shrinkage=1.1)


def test_numerical_sensitivity_uses_actual_refinement_shifts():
    """Large fixed-physics numerical shifts block physical validation explicitly."""
    data = _data()
    n = len(data["theta"])
    data["theta"] = np.vstack((data["theta"], np.tile(data["theta"][0], (4, 1))))
    data["seed"] = np.r_[data["seed"], [111] * 4]
    data["case_id"] = np.r_[data["case_id"], ["fiducial", "convergence_angle4",
                                              "convergence_time2", "convergence_space5"]]
    data["split"] = np.r_[data["split"], ["control", "convergence", "convergence", "convergence"]]
    y = np.zeros((n + 4, 9))
    y[-3:, :6] = np.log([1.2, 1.3, 1.4])[:, None]
    y[-3:, 6:] = np.array([0.01, 0.02, 0.03])[:, None]
    matrix, report = numerical_sensitivity(data, y, 6)
    np.testing.assert_allclose(matrix, calibration_covariance(y[-3:]))
    assert report["largest_absolute_relative_power_shift"] == pytest.approx(0.4)
    assert not report["physical_posterior_validated"]
    assert not report["fiducial_refinement_threshold_passed"]
    assert "NOT calibrated" in report["interpretation"]
    data["theta"][-1, 0] += 0.01
    with pytest.raises(ValueError, match="share fiducial physics"):
        numerical_sensitivity(data, y, 6)


def test_missing_numerical_controls_cannot_create_a_stress_covariance():
    """Missing refinement runs remain an explicit unknown, never zero error."""
    data = _data()
    matrix, report = numerical_sensitivity(data, np.zeros((24, 9)), 6)
    assert matrix is None
    assert not report["physical_posterior_validated"]
    assert len(report["missing_cases"]) == 4


def test_likelihood_uses_covariance_and_rejects_outside_prior():
    """Correlated Gaussian likelihood ratios agree with their analytic values."""
    class Identity:
        def predict(self, x):
            return x
    covariance = np.array([[0.04, 0.01], [0.01, 0.09]])
    bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    observed = np.array([0.1, -0.2])
    values = np.array([observed, [0.3, 0.0], [2.0, 0.0]])
    result = gaussian_log_probability(values, Identity(), observed, covariance, bounds, [0, 1])
    delta = values[1] - observed
    np.testing.assert_allclose(result[1] - result[0], -0.5 * delta @ np.linalg.inv(covariance) @ delta)
    assert result[-1] == -np.inf


def test_independent_sampler_recovers_correlated_gaussian():
    """The actual sampler recovers a correlated target with bounded support."""
    truth = np.array([0.1, -0.2])
    covariance = np.array([[0.03, 0.018], [0.018, 0.06]])
    inverse = np.linalg.inv(covariance)
    bounds = np.tile([-1.0, 1.0], (2, 1))
    def target(theta):
        delta = theta - truth
        logp = -0.5 * np.einsum("ni,ij,nj->n", delta, inverse, delta)
        logp[np.any(np.abs(theta) > 1, axis=1)] = -np.inf
        return logp
    samples, info = metropolis(target, bounds, chains=4, burn=1500, steps=6000, thin=3, seed=41)
    flat = samples.reshape(-1, 2)
    np.testing.assert_allclose(flat.mean(axis=0), truth, atol=0.025)
    np.testing.assert_allclose(np.cov(flat.T), covariance, atol=0.012)
    assert np.all(np.abs(flat) <= 1)
    diagnostic = chain_diagnostics(samples)
    assert diagnostic["thresholds"] == {"rhat_below": 1.01, "bulk_ess_at_least": 400}
    assert max(diagnostic["rank_normalized_split_rhat"]) < 1.05
    assert min(diagnostic["bulk_ess"]) > 100
    assert all(0 < x < 1 for x in info["acceptance_fraction"])


def test_diagnostics_detect_chains_in_different_regions():
    """A narrow marginal in incompatible chain regions must fail mixing."""
    rng = np.random.default_rng(2)
    samples = rng.normal(0, 0.05, size=(4, 1000, 2))
    samples[2:] += 1.0
    diagnostic = chain_diagnostics(samples)
    assert not diagnostic["passed"]
    assert min(diagnostic["rank_normalized_split_rhat"]) > 1.1


def test_constant_chains_are_not_reported_as_converged():
    """A parameter never explored by any chain cannot receive a good R-hat."""
    with pytest.raises(ValueError, match="No posterior exploration"):
        chain_diagnostics(np.ones((4, 100, 2)))


def test_wide_posterior_report_compares_to_prior_width():
    """A prior-dominated result is identified rather than called informative."""
    rng = np.random.default_rng(17)
    samples = rng.uniform(-1, 1, size=(4, 2000, 1))
    result = posterior_summary(samples, [0.0], np.array([[-1.0, 1.0]]), ["a"])["a"]
    assert result["truth_inside_95_percent_marginal"]
    assert abs(result["posterior_95_width_over_prior_95_width"] - 1) < 0.03


def test_neural_fit_does_not_use_test_or_calibration_labels():
    """Changing held-out labels cannot change fitted network predictions."""
    data = _data()
    split = split_indices(data)
    x = data["theta"]
    y = np.column_stack((x[:, 0] + 0.4 * x[:, 1], np.sin(x[:, 2])))
    options = {"mock_sigma": np.array([0.1, 0.1]), "candidates": (((8,), 0.1),),
               "model_seeds": (11,), "max_iter": 300}
    first, _ = fit_emulator(x, y, data["parameter_bounds"], split, **options)
    changed = y.copy()
    changed[np.r_[split["test"], split["calibration"], split["external"]]] += 1000
    second, _ = fit_emulator(x, changed, data["parameter_bounds"], split, **options)
    np.testing.assert_array_equal(first.predict(x), second.predict(x))


def test_forecast_works_before_tests_exist_and_does_not_inspect_calibration(tmp_path, monkeypatch):
    """Training-size forecasts preserve the independent uncertainty-calibration set."""
    data = _data()
    keep = data["split"] != "test"
    for name in ("theta", "split", "case_id", "seed", "P21", "xHI_mean"):
        data[name] = data[name][keep]
    original_fit = inference.fit_emulator
    def quick_fit(*args, **kwargs):
        kwargs.update(candidates=(((8,), 0.1),), model_seeds=(11,), max_iter=100)
        return original_fit(*args, **kwargs)
    monkeypatch.setattr(inference, "fit_emulator", quick_fit)
    first_path = tmp_path / "first.npz"
    np.savez(first_path, **data)
    first = inference.fit_forecast(first_path, tmp_path / "first")
    assert set(first["accuracy"]) == {"train", "selection"}
    calibration = split_indices(data, require_test=False)["calibration"]
    data["P21"][calibration] *= 10000
    data["xHI_mean"][calibration] = 1 - data["xHI_mean"][calibration]
    second_path = tmp_path / "second.npz"
    np.savez(second_path, **data)
    second = inference.fit_forecast(second_path, tmp_path / "second")
    assert first["accuracy"] == second["accuracy"]
