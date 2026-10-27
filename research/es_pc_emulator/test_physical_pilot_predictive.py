"""Portable forward prediction and independent native-check diagnostics."""

import json

import numpy as np
import pytest
from sklearn.neural_network import MLPRegressor

from research.pc_field_emulator.evaluation import shell_spectra

from .physical_pilot_inference import SummaryEmulator, gaussian_log_probability
from .physical_pilot_predictive import (
    PortableEmulator,
    compare_observables,
    native_observables,
    run,
    verify_native_case,
    verify_summary,
)


def test_saved_numpy_forward_matches_sklearn_ensemble(tmp_path):
    """Retained networks must evaluate identically outside the training process."""
    rng = np.random.default_rng(421)
    x = rng.uniform(-1, 1, (24, 2))
    y = np.column_stack((np.sin(x[:, 0]), x[:, 0] * x[:, 1], x[:, 1]**2))
    models = [MLPRegressor(hidden_layer_sizes=(4, 3), activation="tanh", solver="lbfgs",
                           max_iter=1000, random_state=seed).fit(x, y) for seed in (1, 2)]
    ensemble = SummaryEmulator(models, np.array([[1., 3.], [-5., 7.]]),
                              np.array([10., 5., .4]), np.array([2., 4., .3]))
    path = tmp_path / "network.npz"
    ensemble.save(path)
    portable = PortableEmulator.load(path)
    theta = rng.uniform(ensemble.bounds[:, 0], ensemble.bounds[:, 1], (31, 2))
    np.testing.assert_allclose(portable.predict(theta), ensemble.predict(theta), rtol=1e-14, atol=1e-14)
    with pytest.raises(ValueError, match="coordinates"):
        portable.predict([1., np.nan])
    with np.load(path) as saved:
        arrays = {key: saved[key] for key in saved.files}
    arrays["model_0_bias_0"] = np.zeros(100)
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="layer"):
        PortableEmulator.load(path)


def test_native_observables_keep_exact_inference_order():
    """Direct RT comparisons use the saved z/k axes, not a new bin selection."""
    result = {"redshifts": [12., 10., 8.], "k": [.1, .2, .3],
              "P21": np.arange(1., 10.).reshape(3, 3), "xHI_mean": [.9, .7, .3]}
    info = {"redshift_indices": [1, 2], "k_indices": [0, 2],
            "redshifts": [10., 8.], "k_cMpc_inv": [.1, .3], "power_floor": 1e-12,
            "power_columns": 4}
    np.testing.assert_allclose(native_observables(result, info), np.r_[np.log([4., 6., 7., 9.]), .7, .3])
    result["redshifts"][1] = 9.5
    with pytest.raises(ValueError, match="axes"):
        native_observables(result, info)


def test_comparison_uses_fractional_power_and_original_gaussian_likelihood():
    """Log-space residuals convert to exact relative P errors for reporting."""
    native = np.array([np.log(10.), np.log(5.), .4])
    predicted = native + np.array([np.log(1.2), np.log(.8), .03])
    observed = native + np.array([.01, -.02, .01])
    covariance = np.array([[.04, .01, 0], [.01, .09, 0], [0, 0, .0004]])
    report = compare_observables(predicted, native, observed, covariance, 2)
    np.testing.assert_allclose(report["power_relative_error"], [.2, -.2])
    assert report["global_xHI_MAE"] == pytest.approx(.03)

    class Constant:
        def __init__(self, value):
            self.value = value

        def predict(self, theta):
            return np.broadcast_to(self.value, (len(theta), len(self.value)))

    for name, count in (("power_only", 2), ("power_and_global", 3)):
        for label, value in (("neural", predicted), ("direct_native", native)):
            logp = gaussian_log_probability([.5], Constant(value), observed[:count],
                                            covariance[:count, :count], np.array([[0., 1.]]),
                                            np.arange(count))[0]
            assert report["likelihood_comparisons"][name][label] == pytest.approx(logp)


def test_native_check_rejects_tampered_fields_before_acceptance(tmp_path):
    """A completed marker does not replace a payload-integrity check."""
    folder = tmp_path / "cases" / "predictive_00"
    folder.mkdir(parents=True)
    case = {"case_id": "predictive_00"}
    (folder / "case.json").write_text(json.dumps(case))
    (folder / "fields.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="field hash"):
        verify_native_case(tmp_path, case, {"fields_sha256": "incorrect"})


def test_native_check_requires_completed_inference(tmp_path):
    """Posterior-region checks cannot race an unfinished or changing fit."""
    with pytest.raises(ValueError, match="Finish and freeze"):
        run(tmp_path, tmp_path / "inference")


def test_cached_statistics_must_match_verified_field_payload():
    """Changing only result.json must not falsify the native forward check."""
    rng = np.random.default_rng(12)
    fields = {"redshifts": np.array([9., 8.]), "density": rng.uniform(.4, 2., (4, 4, 4)),
              "xhi_rt": rng.uniform(.1, .9, (2, 4, 4, 4))}
    spectra = [shell_spectra({"teacher": 27 * np.sqrt((1 + z) / 10) * fields["density"] * neutral}, 50.)
               for z, neutral in zip(fields["redshifts"], fields["xhi_rt"], strict=True)]
    result = {"redshifts": fields["redshifts"].tolist(), "k": spectra[0]["k_cMpc_inv"],
              "P21": [s["P_teacher"] for s in spectra],
              "xHI_mean": fields["xhi_rt"].mean(axis=(1, 2, 3)).tolist()}
    verify_summary(fields, result, 50.)
    result["P21"][0][0] *= 2
    with pytest.raises(ValueError, match="summaries"):
        verify_summary(fields, result, 50.)
