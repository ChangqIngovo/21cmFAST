"""Recompute held-out diagnostics from saved fields without training helpers."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)


def digest(path: Path) -> str:
    """Return the SHA256 digest of an input or reproduction file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def total(values: np.ndarray) -> np.longdouble:
    """Sum physical inventories in extended precision."""
    return np.sum(np.asarray(values, dtype=np.longdouble), dtype=np.longdouble)


def metrics(field: np.ndarray, truth: np.ndarray, atoms: np.ndarray) -> dict:
    """Measure neutral-field errors and signed ionized-fraction offsets."""
    difference = np.asarray(field, dtype=float) - truth
    absolute = np.abs(difference)
    centred = field - np.mean(field)
    reference = truth - np.mean(truth)
    norm = np.sqrt(np.sum(centred**2) * np.sum(reference**2))
    return {
        "MAE": float(np.mean(absolute)),
        "RMSE": float(np.sqrt(np.mean(difference**2))),
        "maximum_absolute_error": float(np.max(absolute)),
        "p95_absolute_error": float(np.quantile(absolute, 0.95)),
        "p99_absolute_error": float(np.quantile(absolute, 0.99)),
        "delta_Q_mass": float(
            -total(atoms.astype(np.longdouble) * difference) / total(atoms)
        ),
        "delta_Q_volume": float(-np.mean(difference)),
        "xHI_mass": float(total(atoms.astype(np.longdouble) * field) / total(atoms)),
        "xHI_volume": float(np.mean(field)),
        "field_correlation": float(np.sum(centred * reference) / norm)
        if norm > 0
        else None,
    }


def spectra(
    fields: dict[str, np.ndarray], length: float, fraction: float = 1.0
) -> dict:
    """Measure full-complex-FFT auto and cross spectra on a fixed mode band."""
    n = fields["teacher"].shape[0]
    component = np.fft.fftfreq(n) * n
    norm = np.sqrt(
        component[:, None, None] ** 2
        + component[None, :, None] ** 2
        + component[None, None, :] ** 2
    )
    cutoff = fraction * n / 2
    eligible = (norm > 0) & ((norm < cutoff) if fraction < 1 else (norm <= cutoff))
    labels = np.floor(norm[eligible] + 0.5).astype(int)
    if not len(labels):
        return {
            "k_cMpc_inv": [],
            "modes": [],
            "maximum_included_mode_k_cMpc_inv": None,
            "mode_selection": "No nonzero mode lies within the requested band",
            "P_teacher": [],
            "Delta2_teacher": [],
            "comparisons": {
                name: {
                    "P": [],
                    "power_ratio": [],
                    "cross_correlation": [],
                    "defined_shells": 0,
                    "mean_absolute_power_ratio_error": None,
                    "maximum_absolute_power_ratio_error": None,
                    "minimum_cross_correlation": None,
                }
                for name in fields
                if name != "teacher"
            },
        }
    counts = np.bincount(labels)
    occupied = np.flatnonzero(counts)
    k = (
        np.bincount(labels, weights=norm[eligible])[occupied]
        / counts[occupied]
        * 2
        * np.pi
        / length
    )
    normalizer = length**3 / n**6
    transforms = {
        name: np.fft.fftn(field - np.mean(field))[eligible]
        for name, field in fields.items()
    }
    powers = {
        name: np.bincount(labels, weights=np.abs(transform) ** 2)[occupied]
        / counts[occupied]
        * normalizer
        for name, transform in transforms.items()
    }
    floors = {
        name: max(
            float(np.max(power)) * 1e-12,
            np.finfo(float).eps ** 2
            * length**3
            * max(float(np.mean(fields[name] ** 2)), 1),
        )
        for name, power in powers.items()
    }
    valid = powers["teacher"] > floors["teacher"]
    result = {
        "k_cMpc_inv": k.tolist(),
        "modes": counts[occupied].astype(int).tolist(),
        "maximum_included_mode_k_cMpc_inv": float(
            norm[eligible].max() * 2 * np.pi / length
        ),
        "mode_selection": "0 < |k| < 0.5 k_Nyquist"
        if fraction < 1
        else "0 < |k| <= k_Nyquist",
        "normalization": "Full complex FFT; P=V/N^6 times shell mean squared amplitude; shell label=floor(|k|/k_fundamental+0.5)",
        "P_teacher": powers["teacher"].tolist(),
        "Delta2_teacher": (k**3 * powers["teacher"] / (2 * np.pi**2)).tolist(),
        "comparisons": {},
    }
    for name, power in powers.items():
        if name == "teacher":
            continue
        cross = (
            np.bincount(
                labels, weights=np.real(transforms[name] * transforms["teacher"].conj())
            )[occupied]
            / counts[occupied]
            * normalizer
        )
        ratios = np.divide(
            power, powers["teacher"], out=np.zeros_like(power), where=valid
        )
        both = valid & (power > floors[name])
        correlation = np.divide(
            cross,
            np.sqrt(power * powers["teacher"]),
            out=np.zeros_like(power),
            where=both,
        )
        result["comparisons"][name] = {
            "P": power.tolist(),
            "power_ratio": [
                float(value) if ok else None
                for value, ok in zip(ratios, valid, strict=True)
            ],
            "cross_correlation": [
                float(np.clip(value, -1, 1)) if ok else None
                for value, ok in zip(correlation, both, strict=True)
            ],
            "defined_shells": int(valid.sum()),
            "mean_absolute_power_ratio_error": float(np.mean(np.abs(ratios[valid] - 1)))
            if np.any(valid)
            else None,
            "maximum_absolute_power_ratio_error": float(
                np.max(np.abs(ratios[valid] - 1))
            )
            if np.any(valid)
            else None,
            "minimum_cross_correlation": float(np.min(correlation[both]))
            if np.any(both)
            else None,
        }
    return result


def geometric_mfp(
    field: np.ndarray, length: float, phase: str, draws: np.ndarray
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Independent vectorized voxel DDA, with common uniform random variates."""
    mask = field < 0.5 if phase == "ionized" else field >= 0.5
    selected = np.flatnonzero(mask)
    rays, n = len(draws), field.shape[0]
    dx = length / n
    if not len(selected):
        return (
            {
                "status": "empty_phase",
                "phase_volume_fraction": 0,
                "restricted_mean_cMpc": None,
                "cap_fraction": None,
            },
            np.array([]),
            np.array([], dtype=bool),
        )
    indices = selected[
        np.minimum((draws[:, 0] * len(selected)).astype(int), len(selected) - 1)
    ]
    cells = np.array(np.unravel_index(indices, mask.shape)).T
    mu, phi = 2 * draws[:, 4] - 1, 2 * np.pi * draws[:, 5]
    transverse = np.sqrt(1 - mu**2)
    directions = np.column_stack(
        [transverse * np.cos(phi), transverse * np.sin(phi), mu]
    )
    step = np.sign(directions).astype(int)
    distance = np.full(rays, length, dtype=np.float64)
    censored = np.ones(rays, dtype=bool)
    nonzero = directions != 0
    intervals = np.divide(
        dx, np.abs(directions), out=np.full_like(directions, np.inf), where=nonzero
    )
    faces = np.divide(
        np.where(directions > 0, 1 - draws[:, 1:4], draws[:, 1:4]) * dx,
        np.abs(directions),
        out=np.full_like(directions, np.inf),
        where=nonzero,
    )
    live = np.arange(rays)
    for _ in range(int(np.ceil(np.sqrt(3) * n)) + 4):
        if not len(live):
            break
        next_time = np.min(faces[live], axis=1)
        inside_cap = next_time < length
        live, next_time = live[inside_cap], next_time[inside_cap]
        if not len(live):
            break
        crossing = faces[live] <= next_time[:, None] + 16 * np.finfo(
            float
        ).eps * np.maximum(1, np.abs(next_time[:, None]))
        cells[live] = (cells[live] + crossing * step[live]) % n
        faces[live] += np.where(crossing, intervals[live], 0)
        hit = ~mask[tuple(cells[live].T)]
        distance[live[hit]] = next_time[hit]
        censored[live[hit]] = False
        live = live[~hit]
    if len(live) and np.any(np.min(faces[live], axis=1) < length):
        raise RuntimeError("Independent geometric ray traversal did not finish")
    uncensored = np.sort(distance[~censored])
    quantiles = {}
    for quantile in (0.16, 0.5, 0.84):
        index = max(0, int(np.ceil(quantile * rays)) - 1)
        quantiles[str(quantile)] = (
            float(uncensored[index]) if index < len(uncensored) else None
        )
    return (
        {
            "status": "all_censored" if censored.all() else "measured",
            "phase_volume_fraction": float(np.mean(mask)),
            "restricted_mean_cMpc": float(np.mean(distance)),
            "restricted_mean_MC_standard_error_cMpc": float(
                np.std(distance, ddof=1) / np.sqrt(rays)
            ),
            "cap_fraction": float(np.mean(censored)),
            "quantiles_cMpc": quantiles,
            "below_one_cell_fraction": float(np.mean((distance < dx) & ~censored)),
        },
        distance,
        censored,
    )


def compare_mfp(predicted: tuple, teacher: tuple) -> dict:
    """Compare capped ray distributions using paired common random variates."""
    first, second = predicted[1], teacher[1]
    if not len(first) or not len(second):
        return {"available": False, "reason": "empty phase"}
    a, b = np.sort(first), np.sort(second)
    points = np.unique(np.concatenate([a, b]))
    difference = np.searchsorted(a, points, side="right") / len(a) - np.searchsorted(
        b, points, side="right"
    ) / len(b)
    paired = first - second
    return {
        "available": True,
        "restricted_mean_ratio": float(np.mean(first) / np.mean(second)),
        "restricted_mean_difference_cMpc": float(np.mean(paired)),
        "paired_difference_MC_standard_error_cMpc": float(
            np.std(paired, ddof=1) / np.sqrt(len(first))
        ),
        "restricted_CDF_maximum_difference": float(np.max(np.abs(difference))),
        "restricted_wasserstein_distance_cMpc": float(
            np.sum(np.abs(difference[:-1]) * np.diff(points))
        ),
        "cap_fraction_difference": float(np.mean(predicted[2]) - np.mean(teacher[2])),
    }


def aggregate(rows: list[dict]) -> dict:
    """Average field and power diagnostics with equal snapshot weights."""
    if not rows:
        return {"count": 0, "metrics": None}
    methods = ("emulator", "raw", "native_es")
    result = {
        "count": len(rows),
        "redshifts": [row["redshift"] for row in rows],
        "metrics": {},
    }
    for method in methods:
        losses = [row["field_metrics"][method] for row in rows]
        result["metrics"][method] = {
            **{
                name: float(np.mean([loss[name] for loss in losses]))
                for name in ("MAE", "RMSE")
            },
            "mean_absolute_delta_Q_mass": float(
                np.mean([abs(loss["delta_Q_mass"]) for loss in losses])
            ),
            "maximum_absolute_delta_Q_mass": float(
                np.max([abs(loss["delta_Q_mass"]) for loss in losses])
            ),
        }
        ps = [
            row["low_k_21cm"]["comparisons"][method]["mean_absolute_power_ratio_error"]
            for row in rows
        ]
        ps = [value for value in ps if value is not None]
        result["metrics"][method]["mean_snapshot_low_k_power_ratio_error"] = (
            float(np.mean(ps)) if ps else None
        )
        result["metrics"][method]["defined_power_snapshots"] = len(ps)
    return result


def main() -> None:
    """Audit complete saved held-out results and write their independent report."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2108)
    parser.add_argument("--rays", type=int, default=32768)
    args = parser.parse_args()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.training / "summary.json").read_text())
    if not summary["complete"] or args.seed not in summary["split_seeds"]["test"]:
        raise ValueError("Requested seed is not a completed held-out test")
    if (
        args.seed
        in summary["split_seeds"]["train"] + summary["split_seeds"]["validation"]
    ):
        raise ValueError("Held-out seed overlaps the fitted data")
    if summary["test_used_for_selection"]:
        raise ValueError("Training reports test-based model selection")
    inputs = json.loads(
        (args.teacher / "seeds" / f"seed{args.seed}" / "native_inputs.json").read_text()
    )
    cosmology = inputs["cosmo_params"]
    if inputs["astro_options"]["USE_TS_FLUCT"] is not False:
        raise ValueError("Brightness expression assumes saturated spin temperature")
    draws = np.random.default_rng(20261116).random((args.rays, 6))
    rows = []
    for path in sorted(
        (args.training / "test").glob(f"seed{args.seed}_z*/predictions.npz")
    ):
        record = json.loads((path.parent / "metrics.json").read_text())
        redshift, length = record["redshift"], record["box_len"]
        teacher_path = (
            args.teacher
            / "seeds"
            / f"seed{args.seed}"
            / "teacher"
            / f"z{redshift:.12g}".replace(".", "p")
            / "fields.npz"
        )
        with np.load(path, allow_pickle=False) as archive:
            saved = {name: np.array(archive[name]) for name in archive.files}
        with np.load(teacher_path, allow_pickle=False) as archive:
            physical = {name: np.array(archive[name]) for name in archive.files}
        for name in ("teacher_xHI", "native_es_xHI"):
            np.testing.assert_array_equal(saved[name], physical[name])
        fields = {
            "teacher": physical["teacher_xHI"],
            "emulator": saved["emulator_xHI"],
            "raw": saved["raw_xHI"],
            "native_es": physical["native_es_xHI"],
        }
        atoms, source = physical["capacity"], physical["source"]
        for field in (*fields.values(), saved["emulator_xHII"]):
            if (
                field.shape != atoms.shape
                or not np.isfinite(field).all()
                or np.any((field < 0) | (field > 1))
            ):
                raise ValueError(
                    "Saved ionization fields must be finite, bounded and share a grid"
                )
        np.testing.assert_allclose(
            saved["emulator_xHII"] + fields["emulator"],
            1,
            rtol=0,
            atol=4 * np.finfo(float).eps,
        )
        for name in (
            "source",
            "capacity",
            "ionized",
            "remaining_photons",
            "escaped_by_direction",
            "unemitted_source",
        ):
            if not np.isfinite(physical[name]).all() or np.any(physical[name] < 0):
                raise ValueError(f"Negative or nonfinite physical inventory {name}")
        emitted, capacity = total(source), total(atoms)
        teacher_absorbed = total(
            atoms.astype(np.longdouble) * (1 - fields["teacher"].astype(np.longdouble))
        )
        teacher_remaining = sum(
            total(physical[name])
            for name in (
                "remaining_photons",
                "escaped_by_direction",
                "unemitted_source",
            )
        )
        predicted_absorbed = total(atoms.astype(np.longdouble) * saved["emulator_xHII"])
        neutral_field_absorbed = total(
            atoms.astype(np.longdouble) * (1 - fields["emulator"].astype(np.longdouble))
        )
        unabsorbed = np.longdouble(saved["predicted_unabsorbed_photons"])
        if unabsorbed < 0 or not np.isfinite(unabsorbed):
            raise ValueError(
                "Invalid independently predicted unabsorbed photon reservoir"
            )
        denominator = max(emitted, np.finfo(float).tiny)
        ledger = {
            "source_photons": float(emitted),
            "teacher_field_absorbed": float(teacher_absorbed),
            "teacher_stored_unabsorbed": float(teacher_remaining),
            "teacher_relative_residual": float(
                abs(emitted - teacher_absorbed - teacher_remaining) / denominator
            ),
            "emulator_field_absorbed": float(predicted_absorbed),
            "emulator_predicted_unabsorbed": float(unabsorbed),
            "emulator_relative_residual": float(
                abs(emitted - predicted_absorbed - unabsorbed) / denominator
            ),
            "emulator_absorbed_target_relative_residual": float(
                abs(
                    predicted_absorbed
                    - np.longdouble(saved["predicted_absorbed_photons"])
                )
                / denominator
            ),
            "emulator_xHI_reconstructed_relative_residual": float(
                abs(emitted - neutral_field_absorbed - unabsorbed) / denominator
            ),
            "absorption_error_over_source": float(
                (predicted_absorbed - teacher_absorbed) / denominator
            ),
            "native_ES_absorption_error_over_source": float(
                (
                    total(
                        atoms.astype(np.longdouble)
                        * (1 - fields["native_es"].astype(np.longdouble))
                    )
                    - teacher_absorbed
                )
                / denominator
            ),
        }
        if (
            max(
                ledger["teacher_relative_residual"],
                ledger["emulator_relative_residual"],
            )
            > 1e-10
        ):
            raise ValueError("Independent field photon accounting failed")
        coefficient = (
            27
            * cosmology["OMb"]
            * cosmology["hlittle"] ** 2
            / 0.023
            * np.sqrt(
                0.15
                / (cosmology["OMm"] * cosmology["hlittle"] ** 2)
                * (1 + redshift)
                / 10
            )
        )
        brightness = {
            name: coefficient * atoms * field for name, field in fields.items()
        }
        np.testing.assert_allclose(
            brightness["teacher"], physical["teacher_brightness"], rtol=3e-6, atol=2e-6
        )
        np.testing.assert_allclose(
            brightness["native_es"], physical["native_brightness"], rtol=3e-6, atol=2e-6
        )
        calculated = {
            name: metrics(field, fields["teacher"], atoms)
            for name, field in fields.items()
        }
        maximum_report_difference = 0.0
        for name, reported_name in (
            ("emulator", "constrained"),
            ("raw", "raw"),
            ("native_es", "native_es"),
        ):
            for key in ("MAE", "RMSE", "delta_Q_mass", "delta_Q_volume"):
                difference = abs(
                    calculated[name][key] - record["field_metrics"][reported_name][key]
                )
                maximum_report_difference = max(maximum_report_difference, difference)
                reference = record["evaluation"]["metrics"][reported_name][key]
                if abs(calculated[name][key] - reference) > 1e-12:
                    raise ValueError(
                        f"Independent metric mismatch {name} {key}: {difference}"
                    )
        measured = {}
        for phase in ("ionized", "neutral"):
            samples = {
                name: geometric_mfp(fields[name], length, phase, draws)
                for name in ("teacher", "emulator", "native_es")
            }
            measured[phase] = {
                "fields": {name: sample[0] for name, sample in samples.items()},
                "comparisons": {
                    name: compare_mfp(samples[name], samples["teacher"])
                    for name in ("emulator", "native_es")
                },
            }
        rows.append(
            {
                "seed": args.seed,
                "redshift": redshift,
                "dimension": atoms.shape[0],
                "box_len_cMpc": length,
                "teacher_xHI_mass": float(
                    total(atoms.astype(np.longdouble) * fields["teacher"]) / capacity
                ),
                "input_sha256": {
                    str(path): digest(path),
                    str(teacher_path): digest(teacher_path),
                },
                "field_metrics": calculated,
                "budget": ledger,
                "low_k_21cm": spectra(brightness, length, 0.5),
                "all_resolved_21cm": spectra(brightness, length),
                "MFP": measured,
                "maximum_metric_difference_from_training_report": maximum_report_difference,
            }
        )
        LOGGER.info(
            f"Verified seed={args.seed} z={redshift:g}: ledger={ledger['emulator_relative_residual']:.3g}, MAE={calculated['emulator']['MAE']:.5g}",
        )
    rows.sort(key=lambda row: -row["redshift"])
    if len(rows) != summary["sample_counts"]["test"]:
        raise ValueError(
            "Independent audit did not cover the complete requested test set"
        )
    result = {
        "complete": True,
        "scope": "Independent postselection audit of saved genuine held-out fields; no fitting or model selection. No emulator training/evaluation helper imported. Full-complex-FFT spectra and independent vectorized first-exit ray traversal.",
        "teacher_root": str(args.teacher.resolve()),
        "training_root": str(args.training.resolve()),
        "selected_kind": summary["selected_kind"],
        "selected_model_seed": summary["selected_model_seed"],
        "split_seeds": summary["split_seeds"],
        "script_sha256": digest(Path(__file__)),
        "training_summary_sha256": digest(args.training / "summary.json"),
        "MFP_settings": {
            "threshold_xHI": 0.5,
            "rays_per_field_phase": args.rays,
            "seed": 20261116,
            "draws_sha256": hashlib.sha256(draws.tobytes()).hexdigest(),
            "cap": "one box length",
            "phase_origin_sampling": "same uniform voxel-rank, subvoxel-position and direction variates; origins differ for unequal phase masks",
            "interpretation": "geometric first-exit lengths with cap censoring, not physical absorption mean-free-path; same redshift, potentially different neutral fractions",
        },
        "all_snapshots": aggregate(rows),
        "active_snapshots": aggregate(
            [row for row in rows if 0.05 < row["teacher_xHI_mass"] < 0.95]
        ),
        "active_criterion": "0.05 < teacher mass-weighted xHI < 0.95",
        "maximum_teacher_photon_residual": max(
            row["budget"]["teacher_relative_residual"] for row in rows
        ),
        "maximum_emulator_photon_residual": max(
            row["budget"]["emulator_relative_residual"] for row in rows
        ),
        "rows": rows,
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output / "independent_test_summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    LOGGER.info(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "all_snapshots",
                    "active_snapshots",
                    "maximum_teacher_photon_residual",
                    "maximum_emulator_photon_residual",
                    "elapsed_seconds",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
