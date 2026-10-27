"""Direct native checks at three predeclared posterior-region points.

These runs test interpolation of the fixed-discretization teacher. They are
excluded from fitting, model selection, uncertainty calibration and the original
test set. They do not certify numerical convergence or posterior coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from research.c2ray_teacher.runner import read_output, read_transport_diagnostics
from research.es_pc_emulator.physical_pilot import (
    BOUNDS,
    PARAMETER_NAMES,
    run_case,
    write_json,
)
from research.pc_field_emulator.evaluation import shell_spectra

POINTS = (
    ("power_only", "coordinatewise_median"),
    ("power_only", "highest_likelihood_retained_draw"),
    ("power_and_global", "highest_likelihood_retained_draw"),
)


@dataclass
class PortableEmulator:
    """Evaluate the saved tanh/linear summary ensemble without unpickling."""

    bounds: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray
    models: list

    @classmethod
    def load(cls, path):
        """Check dimensions and finite values before evaluating saved weights."""
        with np.load(path, allow_pickle=False) as saved:
            bounds, mean, scale = (saved[key].copy() for key in ("bounds", "y_mean", "y_scale"))
            count = int(saved["n_models"])
            if (bounds.ndim != 2 or bounds.shape[1] != 2 or mean.ndim != 1
                    or scale.shape != mean.shape or count < 1
                    or not all(np.isfinite(x).all() for x in (bounds, mean, scale))
                    or np.any(bounds[:, 1] <= bounds[:, 0]) or np.any(scale <= 0)):
                raise ValueError("Invalid saved emulator dimensions or normalization")
            models = []
            for i in range(count):
                layers, width = [], len(bounds)
                n_layers = int(saved[f"model_{i}_layers"])
                if n_layers < 1:
                    raise ValueError("Saved model has no layers")
                for j in range(n_layers):
                    weight, bias = saved[f"model_{i}_weight_{j}"].copy(), saved[f"model_{i}_bias_{j}"].copy()
                    if (weight.ndim != 2 or weight.shape[0] != width or bias.shape != (weight.shape[1],)
                            or not np.isfinite(weight).all() or not np.isfinite(bias).all()):
                        raise ValueError("Invalid saved model layer")
                    layers.append((weight, bias))
                    width = len(bias)
                if width != len(mean):
                    raise ValueError("Saved model output size differs from normalization")
                models.append(layers)
        return cls(bounds, mean, scale, models)

    def predict(self, theta):
        """Return physical ln-power and global-neutral-fraction observables."""
        theta = np.atleast_2d(theta)
        if theta.ndim != 2 or theta.shape[1] != len(self.bounds) or not np.isfinite(theta).all():
            raise ValueError("Invalid prediction coordinates")
        normalized = 2 * (theta - self.bounds[:, 0]) / np.diff(self.bounds, axis=1).ravel() - 1
        predictions = []
        for layers in self.models:
            values = normalized
            for index, (weight, bias) in enumerate(layers):
                values = values @ weight + bias
                if index < len(layers) - 1:
                    values = np.tanh(values)
            predictions.append(values)
        return np.mean(predictions, axis=0) * self.y_scale + self.y_mean


def native_observables(result, information):
    """Use exactly the redshift/k ordering saved with the original posterior."""
    zi, ki = information["redshift_indices"], information["k_indices"]
    redshifts, wavenumbers = np.asarray(result["redshifts"]), np.asarray(result["k"])
    if (not np.allclose(redshifts[zi], information["redshifts"], rtol=0, atol=1e-10)
            or not np.allclose(wavenumbers[ki], information["k_cMpc_inv"], rtol=0, atol=1e-10)):
        raise ValueError("Native check axes differ from the inference observables")
    power = np.asarray(result["P21"])[np.ix_(zi, ki)]
    neutral = np.asarray(result["xHI_mean"])[zi]
    if (not np.isfinite(power).all() or np.any(power < 0) or not np.isfinite(neutral).all()
            or np.any((neutral < 0) | (neutral > 1))):
        raise ValueError("Invalid native posterior-check observables")
    if power.size != information["power_columns"]:
        raise ValueError("Native check has a different number of power bins")
    return np.r_[np.log(np.maximum(power, information["power_floor"])).ravel(), neutral]


def compare_observables(predicted, native, observed, covariance, power_columns):
    """Compare forward means using the same saved approximate likelihood."""
    predicted, native, observed = (np.asarray(x, dtype=float) for x in (predicted, native, observed))
    covariance = np.asarray(covariance, dtype=float)
    if (predicted.ndim != 1 or native.shape != predicted.shape or observed.shape != predicted.shape
            or covariance.shape != (len(native), len(native))
            or not all(np.isfinite(x).all() for x in (predicted, native, observed, covariance))):
        raise ValueError("Invalid forward comparison shapes or values")
    relative = np.expm1(predicted[:power_columns] - native[:power_columns])
    result = {
        "predicted_observables": predicted.tolist(), "native_observables": native.tolist(),
        "power_relative_error": relative.tolist(),
        "power_mean_absolute_relative_error": float(np.mean(np.abs(relative))),
        "power_max_absolute_relative_error": float(np.max(np.abs(relative))),
        "global_xHI_error": (predicted[power_columns:] - native[power_columns:]).tolist(),
        "global_xHI_MAE": float(np.mean(np.abs(predicted[power_columns:] - native[power_columns:]))),
        "likelihood_comparisons": {},
    }
    for name, count in (("power_only", power_columns), ("power_and_global", len(native))):
        factor = np.linalg.cholesky(covariance[:count, :count])
        normalization = -np.log(np.diag(factor)).sum() - count * np.log(2 * np.pi) / 2
        values = {}
        for label, mean in (("neural", predicted), ("direct_native", native)):
            whitened = np.linalg.solve(factor, mean[:count] - observed[:count])
            values[label] = float(normalization - .5 * np.sum(whitened**2))
        values["direct_native_minus_neural"] = values["direct_native"] - values["neural"]
        result["likelihood_comparisons"][name] = values
    return result


def verify_summary(fields, result, box_len):
    """Bind cached summary statistics to the verified field payload itself."""
    power, means = [], []
    for redshift, neutral in zip(fields["redshifts"], fields["xhi_rt"], strict=True):
        brightness = 27 * np.sqrt((1 + redshift) / 10) * fields["density"] * neutral
        spectrum = shell_spectra({"teacher": brightness}, box_len)
        power.append(spectrum["P_teacher"])
        means.append(float(neutral.mean()))
    if (not np.array_equal(fields["redshifts"], result["redshifts"])
            or not np.allclose(spectrum["k_cMpc_inv"], result["k"], rtol=1e-13, atol=0)
            or not np.allclose(power, result["P21"], rtol=1e-12, atol=0)
            or not np.allclose(means, result["xHI_mean"], rtol=1e-13, atol=0)):
        raise ValueError("Cached native summaries differ from the verified fields")


def verify_native_case(root, case, result):
    """Recheck retained fields, configuration and every primary photon ledger."""
    folder = root / "cases" / case["case_id"]
    if json.loads((folder / "case.json").read_text()) != case:
        raise ValueError("Posterior-check case configuration changed")
    if hashlib.sha256((folder / "fields.npz").read_bytes()).hexdigest() != result["fields_sha256"]:
        raise ValueError("Posterior-check field hash differs")
    with np.load(folder / "fields.npz", allow_pickle=False) as fields:
        metadata = json.loads(str(fields["metadata"].item()))
        if any(metadata.get(key) != value or result.get(key) != value for key, value in case.items()):
            raise ValueError("Posterior-check field/result metadata differs from its case")
        verify_summary(fields, result, metadata["box_len_cMpc"])
    native = list(folder.glob("run_*/native_output.bin"))
    if len(native) != 1:
        raise ValueError("Ambiguous posterior-check native output")
    initialization = json.loads((native[0].parent / "native_diagnostics/native_initialization.json").read_text())
    diagnostics = read_transport_diagnostics(
        native[0].parent / "native_diagnostics/transport_diagnostics.jsonl", read_output(native[0]),
        initialization["sed_table_photons_over_nominal"],
        active_spectral_ratio=initialization["sed_transport_photons_over_nominal"],
    )
    residuals = [row["photon_residual_over_transported_emitted"] for row in diagnostics["steps"]
                 if row["photon_residual_over_transported_emitted"] is not None]
    maximum = float(max(abs(value) for value in residuals))
    if not np.isfinite(maximum) or maximum > 1e-8:
        raise ValueError("Posterior-check primary transport ledger does not close")
    return maximum


def run(root, inference, *, workers=2, parallel=2, case_prefix="predictive_"):
    """Evolve three frozen posterior points without adding them to any fit split."""
    root, inference = Path(root).resolve(), Path(inference).resolve()
    if not (inference / "COMPLETED").exists():
        raise ValueError("Finish and freeze inference before native posterior checks")
    if workers < 1 or parallel < 1:
        raise ValueError("Workers and parallelism must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_]{1,64}", case_prefix):
        raise ValueError("Case prefix must contain only letters, digits or underscores")
    paths = [inference / name for name in ("posterior_predictive_native_candidates.json", "experiment.json",
                                          "summary_emulator.npz", "prediction_and_covariance.npz")]
    candidates, experiment = (json.loads(path.read_text()) for path in paths[:2])
    if candidates["parameter_names"] != PARAMETER_NAMES or experiment["parameter_names"] != PARAMETER_NAMES:
        raise ValueError("Native and inferred parameter definitions differ")
    model = PortableEmulator.load(paths[2])
    if not np.array_equal(model.bounds, BOUNDS):
        raise ValueError("Native and inferred parameter bounds differ")
    with np.load(paths[3], allow_pickle=False) as saved:
        observed, covariance = saved["noisy_mock"].copy(), saved["total_covariance"].copy()
    build = root / "build_r3_a2"
    executable = build / "c2ray_teacher"
    build_manifest = json.loads((build / "build_manifest.json").read_text())
    if (not build_manifest.get("runtime_physics_input_supported")
            or build_manifest["dimension"] != 24 or build_manifest["angular_order"] != 2
            or hashlib.sha256(executable.read_bytes()).hexdigest() != build_manifest["executable_sha256"]):
        raise ValueError("Cached native executable differs from the declared teacher")
    seed = candidates["conditional_IC_seed"]
    if seed != experiment["conditional_IC_seed"]:
        raise ValueError("Posterior-check IC differs from inference")
    planned = []
    frozen_ids = {case["case_id"] for case in json.loads((root / "design.json").read_text())["cases"]}
    for i, (posterior, kind) in enumerate(POINTS):
        vector = np.asarray(candidates["posteriors"][posterior][kind]["theta"])
        if vector.shape != (6,) or not np.isfinite(vector).all() or np.any(vector < BOUNDS[:, 0]) or np.any(vector > BOUNDS[:, 1]):
            raise ValueError("Posterior check lies outside native control bounds")
        case = {"case_id": f"{case_prefix}{i:02d}", "theta": vector.tolist(), "split": "posterior_predictive_check",
                "seed": seed, "refinement": 3, "angular_order": 2, "time_substeps": 1,
                "posterior": posterior, "point_definition": kind}
        if case["case_id"] in frozen_ids:
            raise ValueError("Posterior-check case must not be in the original design")
        planned.append(case)
    results = []
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = [pool.submit(run_case, root, case, executable, workers) for case in planned]
        for case, future in zip(planned, futures, strict=True):
            native = future.result()
            ledger = verify_native_case(root, case, native)
            comparison = compare_observables(model.predict(case["theta"])[0],
                                            native_observables(native, experiment["observables"]),
                                            observed, covariance, experiment["observables"]["power_columns"])
            results.append(case | comparison | {"fields_sha256": native["fields_sha256"],
                                                 "max_step_primary_transport_fractional_residual": ledger})
    report = {
        "schema_version": 1, "native_validation_completed": True, "physical_posterior_validated": False,
        "input_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "native_executable_sha256": build_manifest["executable_sha256"],
        "scope": "Three predeclared posterior-region interpolation checks against the same unconverged discrete native teacher; excluded from all fitting and calibration",
        "likelihood_scope": "Both means use the original approximate covariance, including emulator calibration error; differences isolate forward-mean changes, not a recalibrated posterior",
        "points": results,
    }
    write_json(inference / "native_predictive_checks.json", report)
    print(json.dumps({"native_check_cases": len(results), "max_power_errors": [row["power_max_absolute_relative_error"] for row in results]}), flush=True)  # noqa: T201
    return report


def main():
    """Run optional direct posterior-region RT checks after the small inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--inference", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--case-prefix", default="predictive_",
                        help="Use a new prefix when checking a different frozen inference")
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
