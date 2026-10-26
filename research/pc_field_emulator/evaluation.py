"""Compare PC field emulation using local errors and physical 21-cm spectra.

Brightness uses the declared fixed-gas, saturated-spin-temperature, no-RSD
approximation: 27 sqrt((1+z)/10) * capacity * (1-xHII) mK. Capacity must therefore
be in units of the mean cosmic atom count per original cell, as in the teacher.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _cube(value: ArrayLike, name: str) -> NDArray[np.float64]:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 3 or len(set(result.shape)) != 1 or result.shape[0] < 2:
        raise ValueError(
            f"{name} must be a cubic grid with at least two cells per axis"
        )
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _fraction(value: ArrayLike, name: str) -> NDArray[np.float64]:
    result = _cube(value, name)
    if np.any(result < -1e-12) or np.any(result > 1 + 1e-12):
        raise ValueError(f"{name} must be an ionized fraction in [0, 1]")
    return np.clip(result, 0, 1)


def field_metrics(
    predicted: ArrayLike,
    teacher: ArrayLike,
    capacity: ArrayLike,
) -> dict[str, float | int | None]:
    """Measure coarse-cell errors and mass-weighted ionization.

    Parameters
    ----------
    predicted, teacher
        Predicted and reference ionized fractions on the same cubic grid.
    capacity
        Nonnegative fixed atom counts in each cell.

    Returns
    -------
    dict
        JSON-safe means, errors, and morphology diagnostics. Partial cells have
        0.01 < teacher < 0.99. Front cells touch, by a periodic six-neighbor
        connection, a cell on the other side of the teacher's xHII=0.5 boundary.
        Neutral cells have teacher xHII < 0.5; their IoU and MAE diagnose late
        islands even when ionized-volume overlap is close to one. Undefined
        correlations, empty masks and empty IoU unions return None.
    """
    pred = _fraction(predicted, "Prediction")
    truth = _fraction(teacher, "Teacher")
    rho = _cube(capacity, "Capacity")
    if pred.shape != truth.shape or pred.shape != rho.shape:
        raise ValueError("All fields must share a grid")
    total = float(np.sum(rho))
    if np.any(rho < 0) or total <= 0:
        raise ValueError("Capacity must be nonnegative with positive total")
    residual = pred - truth
    absolute = np.abs(residual)
    partial = (truth > 0.01) & (truth < 0.99)
    truth_ionized = truth >= 0.5
    pred_ionized = pred >= 0.5
    front = np.zeros(truth.shape, dtype=bool)
    for axis in range(3):
        for offset in (-1, 1):
            front |= truth_ionized != np.roll(truth_ionized, offset, axis=axis)
    union = int(np.count_nonzero(truth_ionized | pred_ionized))
    neutral_union = int(np.count_nonzero(~truth_ionized | ~pred_ionized))
    centered_pred = pred - np.mean(pred)
    centered_truth = truth - np.mean(truth)
    denominator = float(np.sqrt(np.mean(centered_pred**2) * np.mean(centered_truth**2)))
    correlation = (
        float(np.clip(np.mean(centered_pred * centered_truth) / denominator, -1, 1))
        if denominator > np.finfo(float).eps ** 2
        else None
    )
    return {
        "Q_volume": float(np.mean(pred)),
        "Q_mass": float(np.sum(rho * pred) / total),
        "ionized_atom_count": float(np.sum(rho * pred)),
        "delta_Q_volume": float(np.mean(residual)),
        "delta_Q_mass": float(np.sum(rho * residual) / total),
        "MAE": float(np.mean(absolute)),
        "mass_weighted_MAE": float(np.sum(rho * absolute) / total),
        "RMSE": float(np.sqrt(np.mean(residual**2))),
        "maximum_absolute_error": float(np.max(absolute)),
        "partial_cell_count": int(np.count_nonzero(partial)),
        "partial_cell_MAE": float(np.mean(absolute[partial]))
        if np.any(partial)
        else None,
        "front_cell_count": int(np.count_nonzero(front)),
        "front_MAE": float(np.mean(absolute[front])) if np.any(front) else None,
        "ionized_IoU": (
            float(np.count_nonzero(truth_ionized & pred_ionized) / union)
            if union
            else None
        ),
        "neutral_IoU": (
            float(np.count_nonzero(~truth_ionized & ~pred_ionized) / neutral_union)
            if neutral_union
            else None
        ),
        "neutral_cell_count": int(np.count_nonzero(~truth_ionized)),
        "neutral_cell_MAE": float(np.mean(absolute[~truth_ionized]))
        if np.any(~truth_ionized)
        else None,
        "field_correlation": correlation,
    }


def _shell_geometry(
    n: int, box_len: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fundamental = 2 * np.pi / box_len
    components = np.fft.fftfreq(n) * n
    last = np.fft.rfftfreq(n) * n
    norm = np.sqrt(
        components[:, None, None] ** 2
        + components[None, :, None] ** 2
        + last[None, None, :] ** 2
    )
    eligible = (norm > 0) & (norm <= n / 2 + 1e-12)
    labels = np.floor(norm[eligible] + 0.5).astype(np.int64)
    last_weights = np.full(last.size, 2, dtype=np.int64)
    last_weights[0] = 1
    if n % 2 == 0:
        last_weights[-1] = 1
    weights = np.broadcast_to(last_weights, norm.shape)[eligible]
    counts = np.bincount(labels, weights=weights)
    occupied = np.flatnonzero(counts > 0)
    k = np.bincount(labels, weights=weights * norm[eligible] * fundamental)[occupied]
    k /= counts[occupied]
    return eligible, labels, weights, occupied, k


def shell_spectra(
    fields: Mapping[str, ArrayLike],
    box_len: float,
) -> dict[str, Any]:
    """Calculate physical auto/cross spectra in fundamental-width shells.

    Parameters
    ----------
    fields
        Named real cubic fields, including ``teacher``. All grids must match.
        Each field's own mean is subtracted before the FFT.
    box_len
        Periodic comoving box length in cMpc.

    Returns
    -------
    dict
        JSON-safe k [cMpc^-1], full-FFT mode counts, P [field-unit^2 cMpc^3],
        Delta2 [field-unit^2], power ratios and teacher cross-correlations.
        The normalization is P=V/N^6 times the shell mean of |FFT|^2. The
        real-FFT conjugates are counted with weights one or two. Only modes
        with 0 < |k| <= pi*N/box_len enter; shell widths equal 2*pi/box_len.
        Ratios for numerically vanishing reference power are None.
    """
    if "teacher" not in fields or not np.isfinite(box_len) or box_len <= 0:
        raise ValueError("A teacher field and positive finite box length are required")
    truth = _cube(fields["teacher"], "Teacher")
    n = truth.shape[0]
    eligible, labels, weights, occupied, k = _shell_geometry(n, box_len)
    counts = np.bincount(labels, weights=weights)[occupied]
    normalization = box_len**3 / n**6
    fourier: dict[str, np.ndarray] = {}
    powers: dict[str, np.ndarray] = {}
    floors: dict[str, float] = {}
    result: dict[str, Any] = {
        "k_cMpc_inv": k.tolist(),
        "modes": counts.astype(int).tolist(),
        "normalization": "P=V/N^6 * mean(|FFT(field-mean)|^2)",
        "k_max_cMpc_inv": float(np.pi * n / box_len),
        "shell_width_cMpc_inv": float(2 * np.pi / box_len),
    }
    for name, value in fields.items():
        field = _cube(value, name)
        if field.shape != truth.shape:
            raise ValueError("All spectrum fields must share a grid")
        transformed = np.fft.rfftn(field - np.mean(field))[eligible]
        power = np.bincount(labels, weights=weights * np.abs(transformed) ** 2)[
            occupied
        ]
        power *= normalization / counts
        fourier[name], powers[name] = transformed, power
        floors[name] = max(
            float(np.max(power)) * 1e-12,
            np.finfo(float).eps ** 2 * box_len**3 * max(float(np.mean(field**2)), 1.0),
        )
        result[f"P_{name}"] = power.tolist()
        result[f"Delta2_{name}"] = (k**3 * power / (2 * np.pi**2)).tolist()
    reference = powers["teacher"]
    for name, transformed in fourier.items():
        if name == "teacher":
            continue
        cross = (
            np.bincount(
                labels,
                weights=weights * np.real(transformed * fourier["teacher"].conj()),
            )[occupied]
            * normalization
            / counts
        )
        valid_reference = reference > floors["teacher"]
        valid_pair = valid_reference & (powers[name] > floors[name])
        ratios = np.divide(
            powers[name], reference, out=np.zeros_like(reference), where=valid_reference
        )
        denominator = np.sqrt(powers[name] * reference)
        correlation = np.divide(
            cross, denominator, out=np.zeros_like(cross), where=valid_pair
        )
        result[f"P_cross_{name}"] = cross.tolist()
        result[f"ratio_{name}"] = [
            float(x) if valid else None
            for x, valid in zip(ratios, valid_reference, strict=True)
        ]
        result[f"r_{name}"] = [
            float(np.clip(x, -1, 1)) if valid else None
            for x, valid in zip(correlation, valid_pair, strict=True)
        ]
    return result


def evaluate_fields(
    capacity: ArrayLike,
    teacher: ArrayLike,
    raw: ArrayLike,
    projected: ArrayLike,
    redshift: float,
    box_len: float,
    *,
    baseline: ArrayLike | None = None,
) -> dict[str, Any]:
    """Evaluate ionization fields and their saturated-spin 21-cm spectra.

    Parameters
    ----------
    capacity
        Fixed atom capacity in mean-cosmic-atom-count-per-cell units.
    teacher, raw, projected
        Reference, unconstrained network and global-budget-projected xHII grids.
    redshift, box_len
        Snapshot redshift and comoving box length in cMpc.
    baseline
        Optional simple baseline ionized fraction on the same grid.

    Returns
    -------
    dict
        Metrics by field, spectra, and the brightness-temperature convention.
    """
    if not np.isfinite(redshift) or redshift < 0:
        raise ValueError("Redshift must be finite and nonnegative")
    rho = _cube(capacity, "Capacity")
    fields = {"teacher": teacher, "raw": raw, "projected": projected}
    if baseline is not None:
        fields["baseline"] = baseline
    metrics = {
        name: field_metrics(value, teacher, rho) for name, value in fields.items()
    }
    temperature = 27 * np.sqrt((1 + redshift) / 10)
    brightness = {
        name: temperature * rho * (1 - np.asarray(value))
        for name, value in fields.items()
    }
    return {
        "metrics": metrics,
        "spectra": shell_spectra(brightness, box_len),
        "brightness_convention": "27*sqrt((1+z)/10)*capacity*(1-xHII) mK; saturated Ts, no RSD",
        "redshift": float(redshift),
        "box_len_cMpc": float(box_len),
    }


def plot_comparison(
    capacity: ArrayLike,
    teacher: ArrayLike,
    raw: ArrayLike,
    projected: ArrayLike,
    redshift: float,
    box_len: float,
    output: Path,
    report: dict[str, Any] | None = None,
) -> None:
    """Save central slices, 21-cm spectra and field cross-correlations.

    Parameters
    ----------
    capacity, teacher, raw, projected
        Matching fixed-gas capacities and coarse ionized-fraction fields.
    redshift, box_len
        Snapshot redshift and comoving box size in cMpc.
    output
        Destination figure path, e.g. a PNG or PDF.
    report
        Optional existing output from evaluate_fields to avoid repeated FFTs.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if report is None:
        report = evaluate_fields(capacity, teacher, raw, projected, redshift, box_len)
    truth = np.asarray(teacher)
    central = truth.shape[2] // 2
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    extent = (0, box_len, 0, box_len)
    for axis, value, title in zip(
        axes[0, :3],
        (teacher, raw, projected),
        ("PC teacher", "Network", "Budget projected"),
        strict=True,
    ):
        im = axis.imshow(
            np.asarray(value)[:, :, central].T,
            origin="lower",
            extent=extent,
            vmin=0,
            vmax=1,
        )
        axis.set(title=title, xlabel="cMpc", ylabel="cMpc")
    fig.colorbar(im, ax=list(axes[0, :3]), label=r"$x_{\rm HII}$", shrink=0.8)
    residual = np.asarray(projected) - truth
    limit = max(float(np.max(np.abs(residual[:, :, central]))), 1e-5)
    im = axes[0, 3].imshow(
        residual[:, :, central].T,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    axes[0, 3].set(title="Projected - teacher", xlabel="cMpc", ylabel="cMpc")
    fig.colorbar(im, ax=axes[0, 3], label=r"$\Delta x_{\rm HII}$", shrink=0.8)
    spectra = report["spectra"]
    k = np.asarray(spectra["k_cMpc_inv"])
    colors = {
        "teacher": "black",
        "raw": "tab:orange",
        "projected": "tab:blue",
        "baseline": "tab:green",
    }
    for name in report["metrics"]:
        power = np.asarray(spectra[f"Delta2_{name}"])
        positive = power > 0
        axes[1, 0].loglog(k[positive], power[positive], color=colors[name], label=name)
        if name != "teacher":
            ratios = [np.nan if x is None else x for x in spectra[f"ratio_{name}"]]
            correlation = [np.nan if x is None else x for x in spectra[f"r_{name}"]]
            axes[1, 1].semilogx(k, ratios, color=colors[name], label=name)
            axes[1, 2].semilogx(k, correlation, color=colors[name], label=name)
    axes[1, 0].set(xlabel=r"$k$ [cMpc$^{-1}$]", ylabel=r"$\Delta^2_{21}$ [mK$^2$]")
    axes[1, 0].legend(fontsize="small")
    axes[1, 1].axhline(1, color="0.5", linestyle="--")
    axes[1, 1].set(xlabel=r"$k$ [cMpc$^{-1}$]", ylabel="Power / teacher")
    axes[1, 2].set(xlabel=r"$k$ [cMpc$^{-1}$]", ylabel=r"$r(k)$", ylim=(-1.05, 1.05))
    indices = np.linspace(0, truth.size - 1, min(truth.size, 30000), dtype=int)
    axes[1, 3].hexbin(
        truth.ravel()[indices],
        np.asarray(projected).ravel()[indices],
        gridsize=40,
        extent=(0, 1, 0, 1),
        mincnt=1,
        bins="log",
    )
    axes[1, 3].plot([0, 1], [0, 1], color="red", linewidth=1)
    axes[1, 3].set(
        xlabel="Teacher xHII", ylabel="Projected xHII", xlim=(0, 1), ylim=(0, 1)
    )
    fig.suptitle(f"Existing-PC field emulation: z={redshift:g}, L={box_len:g} cMpc")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
