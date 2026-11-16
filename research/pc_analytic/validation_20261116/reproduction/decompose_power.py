"""Separate mean-brightness scaling from Fourier contrast and phase errors.

These are algebraic descriptions of saved predictions, not replacements for
model inference or unique causal fractions of prediction error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np


def decompose_power(
    teacher: np.ndarray, predicted: np.ndarray, box_len: float
) -> dict[str, Any]:
    """Attribute a shared-grid brightness spectrum using two exact identities.

    Parameters
    ----------
    teacher, predicted
        Finite, real, cubic brightness fields with identical units and geometry.
    box_len
        Positive comoving box length in cMpc.

    Returns
    -------
    dict
        Mean-brightness factors, full-FFT shell powers and signed coherent
        responses. Individual modes satisfy 0 < k < half Nyquist. Vanishing
        reference powers or nonpositive means yield undefined relative terms.
    """
    reference, estimate = np.asarray(teacher), np.asarray(predicted)
    if (
        np.iscomplexobj(reference)
        or np.iscomplexobj(estimate)
        or reference.ndim != 3
        or len(set(reference.shape)) != 1
        or reference.shape[0] < 4
        or reference.shape != estimate.shape
        or not np.isfinite(reference).all()
        or not np.isfinite(estimate).all()
        or not np.isfinite(box_len)
        or box_len <= 0
    ):
        raise ValueError("Finite real matching cubes and positive box length required")
    reference, estimate = reference.astype(float), estimate.astype(float)
    means = float(reference.mean()), float(estimate.mean())
    factor = (means[1] / means[0]) ** 2 if means[0] > 0 else None
    n = reference.shape[0]
    component = np.fft.fftfreq(n) * n
    norm = np.sqrt(
        component[:, None, None] ** 2
        + component[None, :, None] ** 2
        + component[None, None, :] ** 2
    )
    band = (norm > 0) & (norm < n / 4)
    labels = np.floor(norm[band] + 0.5).astype(int)
    counts = np.bincount(labels)
    occupied = np.flatnonzero(counts)
    normalization = float(box_len) ** 3 / n**6

    def average(values: np.ndarray) -> np.ndarray:
        return np.bincount(labels, weights=values)[occupied] / counts[occupied]

    ft = np.fft.fftn(reference - means[0])[band]
    fp = np.fft.fftn(estimate - means[1])[band]
    pt, pp = (
        average(abs(ft) ** 2) * normalization,
        average(abs(fp) ** 2) * normalization,
    )
    cross = average(np.real(fp * ft.conj())) * normalization
    power_floor = max(
        float(np.max(pt, initial=0)) * 1e-12,
        np.finfo(float).eps ** 2
        * float(box_len) ** 3
        * max(float(np.mean(reference**2)), 1),
    )
    predicted_floor = max(
        float(np.max(pp, initial=0)) * 1e-12,
        np.finfo(float).eps ** 2
        * float(box_len) ** 3
        * max(float(np.mean(estimate**2)), 1),
    )
    rows = []
    for j, label in enumerate(occupied):
        valid = pt[j] > power_floor
        ratio = float(pp[j] / pt[j]) if valid else None
        response = float(cross[j] / pt[j]) if valid else None
        correlation = (
            float(np.clip(cross[j] / np.sqrt(pt[j] * pp[j]), -1, 1))
            if valid and pp[j] > predicted_floor
            else None
        )
        selected = labels == label
        # Direct residual evaluation avoids cancellation for an exact rescaling.
        orthogonal = (
            float(
                np.mean(abs(fp[selected] - response * ft[selected]) ** 2)
                * normalization
            )
            if valid
            else None
        )
        residual = float(np.mean(abs(fp[selected] - ft[selected]) ** 2) * normalization)
        contrast = (
            ratio / factor if valid and factor is not None and factor > 0 else None
        )
        if valid:
            np.testing.assert_allclose(
                ratio, response**2 + orthogonal / pt[j], rtol=2e-12, atol=2e-12
            )
            np.testing.assert_allclose(
                residual / pt[j],
                (response - 1) ** 2 + orthogonal / pt[j],
                rtol=2e-12,
                atol=2e-12,
            )
        rows.append(
            {
                "k_cMpc_inv": float(norm[band][selected].mean() * 2 * np.pi / box_len),
                "modes": int(counts[label]),
                "P_teacher": float(pt[j]),
                "P_predicted": float(pp[j]),
                "P_cross": float(cross[j]),
                "P_field_residual": residual,
                "power_ratio": ratio,
                "fractional_contrast_power_ratio": contrast,
                "coherent_transfer": response,
                "cross_correlation": correlation,
                "orthogonal_power_over_teacher": orthogonal / float(pt[j])
                if valid
                else None,
                "field_residual_power_over_teacher": residual / float(pt[j])
                if valid
                else None,
            }
        )
    ratios = [row["power_ratio"] for row in rows if row["power_ratio"] is not None]
    contrasts = [
        row["fractional_contrast_power_ratio"]
        for row in rows
        if row["fractional_contrast_power_ratio"] is not None
    ]
    return {
        "mean_teacher": means[0],
        "mean_predicted": means[1],
        "mean_factor_squared": factor,
        "defined_shells": len(ratios),
        "mean_absolute_power_ratio_error": float(np.mean(np.abs(np.array(ratios) - 1)))
        if ratios
        else None,
        "mean_absolute_fractional_contrast_power_error": float(
            np.mean(np.abs(np.array(contrasts) - 1))
        )
        if contrasts
        else None,
        "shells": rows,
    }


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    """Analyze unchanged saved external predictions without fitting any model."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()
    if not (args.training / "COMPLETED").is_file():
        raise ValueError("A completed, selected training run is required")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for seed in args.seeds:
        seed_root = args.teacher / "seeds" / f"seed{seed}"
        settings = json.loads((seed_root / "native_inputs.json").read_text())
        if settings["astro_options"]["USE_TS_FLUCT"] is not False:
            raise ValueError("Brightness decomposition requires the no-Ts run")
        cosmology = settings["cosmo_params"]
        summary = json.loads((seed_root / "summary.json").read_text())
        for snapshot in summary["snapshots"]:
            z, length = snapshot["redshift"], snapshot["box_len"]
            ztag = f"z{z:.12g}".replace(".", "p")
            teacher_path = seed_root / "teacher" / ztag / "fields.npz"
            prediction_path = (
                args.training
                / "external_test"
                / "test"
                / f"seed{seed}_{ztag}"
                / "predictions.npz"
            )
            with np.load(teacher_path, allow_pickle=False) as data:
                atoms = np.array(data["capacity"], dtype=float)
                truth = np.array(data["teacher_xHI"], dtype=float)
            with np.load(prediction_path, allow_pickle=False) as data:
                np.testing.assert_array_equal(truth, data["teacher_xHI"])
                fields = {
                    name: np.array(data[key], dtype=float)
                    for name, key in (
                        ("constrained", "emulator_xHI"),
                        ("raw", "raw_xHI"),
                        ("native_es", "native_es_xHI"),
                    )
                }
            coefficient = (
                27
                * cosmology["OMb"]
                * cosmology["hlittle"] ** 2
                / 0.023
                * np.sqrt(
                    0.15 / (cosmology["OMm"] * cosmology["hlittle"] ** 2) * (1 + z) / 10
                )
            )
            neutral_mass = float(np.sum(atoms * truth) / np.sum(atoms))
            rows.append(
                {
                    "seed": seed,
                    "redshift": z,
                    "teacher_xHI_mass": neutral_mass,
                    "active": 0.05 < neutral_mass < 0.95,
                    "predictions": {
                        name: decompose_power(
                            coefficient * atoms * truth,
                            coefficient * atoms * field,
                            length,
                        )
                        for name, field in fields.items()
                    },
                    "input_sha256": {
                        str(teacher_path): _digest(teacher_path),
                        str(prediction_path): _digest(prediction_path),
                    },
                }
            )
    report = {
        "complete": True,
        "scope": "Algebraic diagnostic of existing saved predictions; no fitting or selection",
        "brightness": "native no-Ts coefficient times capacity*xHI; P units mK^2 cMpc^3",
        "band": "Individual Fourier modes 0 < k < 0.5 k_Nyquist; equal shell weighting",
        "mean_identity": "Ppred/Pteacher=(meanBpred/meanBteacher)^2 * P_fractional_contrast_pred/P_fractional_contrast_teacher",
        "cross_identity": "Ppred/Pteacher=a^2+Pperp/Pteacher, a=real(Pcross)/Pteacher",
        "interpretation": "Mean-normalized contrast retains density-ionization correlations; this is not a pure morphology isolation. Rescaled fields are not constrained physical solutions.",
        "script_sha256": _digest(Path(__file__)),
        "training_identity_sha256": _digest(args.training / "identity.json"),
        "snapshot_count": len(rows),
        "rows": rows,
    }
    destination = args.output / "power_attribution.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    logging.getLogger(__name__).info(
        json.dumps(
            {"complete": True, "snapshots": len(rows), "report": str(destination)}
        )
    )


if __name__ == "__main__":
    main()
