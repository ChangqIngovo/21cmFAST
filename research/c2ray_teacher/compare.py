"""Compare native RT runs at common saved epochs and explicit physical scales.

Only identical prescribed source/gas histories admit paired transport errors.
Different ICs, source grids or boxes receive statistical comparisons, without
claiming that matching random seeds produce matched physical Fourier phases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_backend = import_module("research.c2ray_teacher.build")
FIELDS = ("density_total_cm3", "xHII", "temperature_K")
DIAGNOSTIC_TOLERANCES = {
    "mean_xHII_absolute": 0.005,
    "mean_temperature_relative": 0.05,
    "power_relative": 0.05,
    "minimum_full_fft_modes": 20,
}


def common_endpoints(left: dict, right: dict, *, paired: bool) -> list[tuple[int, int]]:
    """Match saved redshift endpoints without interpolation.

    Parameters
    ----------
    left, right : dict
        Physical input manifests containing the saved step endpoints.
    paired : bool
        Require the same elapsed time when matching identical source histories.
    """
    matches = []
    right_z = np.asarray([step["redshift_end"] for step in right["steps"]])
    for i, step in enumerate(left["steps"]):
        found = np.flatnonzero(
            np.isclose(right_z, step["redshift_end"], rtol=0, atol=1e-8)
        )
        if len(found) > 1:
            raise ValueError("Saved endpoints are ambiguous within redshift tolerance")
        if len(found) == 1:
            j = int(found[0])
            if paired and not np.isclose(
                step["time_end_seconds"],
                right["steps"][j]["time_end_seconds"],
                rtol=2e-12,
                atol=1e-8,
            ):
                raise ValueError(
                    "Identical source histories have inconsistent endpoint clocks"
                )
            matches.append((i, j))
    if not matches:
        raise ValueError("Runs have no common saved redshift endpoints")
    return matches


def restrict_fields(
    fields: dict[str, np.ndarray], dimension: int
) -> dict[str, np.ndarray]:
    """Restrict fields while preserving nuclei and ionized nuclei.

    Parameters
    ----------
    fields : dict
        One epoch's density, HII fraction and temperature on a cubic grid.
    dimension : int
        Coarse comparison dimension, an integer divisor of the input dimension.
        Temperature is density weighted, not an internal-energy reconstruction.
    """
    density = np.asarray(fields["density_total_cm3"])
    n = density.shape[0]
    if density.shape != (n,) * 3 or dimension < 1 or n % dimension:
        raise ValueError(
            "Restriction requires matching cubic grids and integer factors"
        )
    if any(
        np.asarray(fields[name]).shape != density.shape
        or not np.isfinite(fields[name]).all()
        for name in FIELDS
    ):
        raise ValueError("Restriction requires finite fields on one cubic grid")
    if density.min() <= 0:
        raise ValueError("Restriction requires positive gas density")
    ratio = n // dimension
    shape = (dimension, ratio) * 3

    def average(array: np.ndarray) -> np.ndarray:
        return array.reshape(shape).mean(axis=(1, 3, 5))

    coarse_density = average(density)
    return {
        "density_total_cm3": coarse_density,
        "xHII": average(density * fields["xHII"]) / coarse_density,
        "temperature_K": average(density * fields["temperature_K"]) / coarse_density,
    }


def _fourier_data(values: np.ndarray, length: float, edges: np.ndarray) -> tuple:
    n = len(values)
    if values.shape != (n,) * 3 or not np.isfinite(values).all() or length <= 0:
        raise ValueError(
            "Power spectra require finite cubic fields and positive box length"
        )
    frequency = 2 * np.pi * np.fft.fftfreq(n, d=length / n)
    wave = np.sqrt(
        frequency[:, None, None] ** 2
        + frequency[None, :, None] ** 2
        + frequency[None, None, :] ** 2
    )
    labels = np.searchsorted(edges, wave.ravel(), side="right") - 1
    valid = (wave.ravel() > 0) & (labels >= 0) & (labels < len(edges) - 1)
    transform = np.fft.fftn(values - values.mean()).ravel()
    return wave.ravel()[valid], labels[valid], transform[valid], length**3 / n**6


def neutral_spectrum(values: np.ndarray, length: float, edges: np.ndarray) -> dict:
    """Measure absolute neutral-fraction fluctuations with physical FFT units.

    Parameters
    ----------
    values : ndarray
        Neutral fraction field. Its mean is subtracted, without dividing by it.
    length : float
        Comoving box length in cMpc.
    edges : ndarray
        Common increasing physical wavenumber bin edges in inverse cMpc.
    """
    edges = np.asarray(edges, dtype=float)
    if (
        edges.ndim != 1
        or len(edges) < 2
        or not np.isfinite(edges).all()
        or not np.all(np.diff(edges) > 0)
    ):
        raise ValueError("Power bin edges must be finite and strictly increasing")
    wave, labels, fourier, normalization = _fourier_data(values, length, edges)
    bins = len(edges) - 1
    counts = np.bincount(labels, minlength=bins)
    powers = np.bincount(
        labels, weights=np.abs(fourier) ** 2 * normalization, minlength=bins
    )
    waves = np.bincount(labels, weights=wave, minlength=bins)
    k = [float(waves[i] / count) if count else None for i, count in enumerate(counts)]
    p = [float(powers[i] / count) if count else None for i, count in enumerate(counts)]
    return {
        "bin_edges_cMpc_inv": edges.tolist(),
        "k_cMpc_inv": k,
        "modes": counts.tolist(),
        "power_cMpc3": p,
        "delta2": [
            None if ki is None else float(ki**3 * pi / (2 * np.pi**2))
            for ki, pi in zip(k, p, strict=True)
        ],
        "mode_convention": "Full complex FFT modes including conjugate partners; counts are not independent-error estimates",
    }


def _paired(left: dict, right: dict) -> bool:
    hashes = left.get("native_array_sha256", {})
    required = {"rates", "capacity", "redshifts", "times_seconds"}
    return (
        required <= set(hashes)
        and hashes == right.get("native_array_sha256")
        and all(
            left.get(name) == right.get(name)
            for name in (
                "box_len_cMpc",
                "native_dimension",
                "cosmology",
                "initial_state",
                "thermal_evolution",
            )
        )
    )


def _statistics(fields: dict[str, np.ndarray]) -> dict:
    rho, ionized, temperature = (fields[name] for name in FIELDS)
    return {
        "xHI_volume": float(np.mean(1 - ionized)),
        "xHI_mass": float(np.sum(rho * (1 - ionized)) / rho.sum()),
        "temperature_volume_mean_K": float(temperature.mean()),
        "temperature_nuclei_weighted_K": float(np.sum(rho * temperature) / rho.sum()),
    }


def _errors(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> dict:
    result = {}
    for name in ("xHII", "temperature_K"):
        difference = right[name] - left[name]
        result[name] = {
            "mean_absolute": float(np.mean(np.abs(difference))),
            "rms": float(np.sqrt(np.mean(difference**2))),
            "maximum_absolute": float(np.max(np.abs(difference))),
        }
    return result


def _cross_correlation(
    left: np.ndarray, right: np.ndarray, length: float, edges: np.ndarray
) -> list:
    _, labels, a, _ = _fourier_data(left, length, edges)
    _, _, b, _ = _fourier_data(right, length, edges)
    bins = len(edges) - 1
    cross = np.bincount(labels, weights=np.real(a * np.conj(b)), minlength=bins)
    auto_a = np.bincount(labels, weights=np.abs(a) ** 2, minlength=bins)
    auto_b = np.bincount(labels, weights=np.abs(b) ** 2, minlength=bins)
    denominator = np.sqrt(auto_a * auto_b)
    return [float(cross[i] / d) if d > 0 else None for i, d in enumerate(denominator)]


def _difference_flags(row: dict) -> dict:
    left, right = row["reference_statistics"], row["candidate_statistics"]
    delta_ionized = abs(right["xHI_volume"] - left["xHI_volume"])
    delta_temperature = abs(
        right["temperature_volume_mean_K"] / left["temperature_volume_mean_K"] - 1
    )
    spectrum = row["neutral_fraction_power"]
    trusted = []
    if "candidate_over_reference" in spectrum:
        for index, ratio in enumerate(spectrum["candidate_over_reference"]):
            modes = min(
                spectrum["reference"]["modes"][index],
                spectrum["candidate"]["modes"][index],
            )
            if (
                ratio is not None
                and modes >= DIAGNOSTIC_TOLERANCES["minimum_full_fft_modes"]
            ):
                trusted.append(abs(ratio - 1))
    return {
        "mean_xHII_absolute_difference": delta_ionized,
        "mean_xHII_within_tolerance": delta_ionized
        <= DIAGNOSTIC_TOLERANCES["mean_xHII_absolute"],
        "mean_temperature_relative_difference": delta_temperature,
        "mean_temperature_within_tolerance": delta_temperature
        <= DIAGNOSTIC_TOLERANCES["mean_temperature_relative"],
        "trusted_power_bins": len(trusted),
        "maximum_trusted_power_relative_difference": max(trusted) if trusted else None,
        "power_within_tolerance": all(
            value <= DIAGNOSTIC_TOLERANCES["power_relative"] for value in trusted
        )
        if trusted
        else None,
    }


def compare_states(
    left: dict,
    right: dict,
    *,
    power_bins: int = 12,
    diagnostic_thresholds: bool = False,
) -> dict:
    """Compare two decoded runs without declaring convergence from one pair.

    Parameters
    ----------
    left, right : dict
        Dictionaries containing ``manifest`` and memory-mapped ``fields``.
    power_bins : int
        Number of common physical logarithmic bins below half the smaller
        Nyquist frequency. Empty bins remain explicit and receive no ratio.
    diagnostic_thresholds : bool
        Add exploratory difference flags for global means and adequately
        sampled power bins. These do not certify convergence from one pair.
    """
    if type(power_bins) is not int or power_bins < 1:
        raise ValueError("power_bins must be a positive integer")
    lm, rm = left["manifest"], right["manifest"]
    paired = _paired(lm, rm)
    endpoints = common_endpoints(lm, rm, paired=paired)
    dimensions = (lm["dimension"], rm["dimension"])
    comparison_dimension = min(dimensions) if paired else None
    if paired and (
        max(dimensions) % comparison_dimension
        or (max(dimensions) // comparison_dimension) % 2 != 1
    ):
        raise ValueError(
            "Paired source-preserving spatial comparison requires an odd integer refinement"
        )
    lengths = (lm["box_len_cMpc"], rm["box_len_cMpc"])
    k_min = max(2 * np.pi / length for length in lengths)
    effective_n = (comparison_dimension,) * 2 if paired else dimensions
    k_max = min(
        np.pi * n / length / 2 for n, length in zip(effective_n, lengths, strict=True)
    )
    edges = (
        np.geomspace(k_min * (1 - 1e-12), k_max * (1 + 1e-12), power_bins + 1)
        if k_max > k_min
        else None
    )
    snapshots = []
    for i, j in endpoints:
        a = {name: np.asarray(left["fields"][name][i]) for name in FIELDS}
        b = {name: np.asarray(right["fields"][name][j]) for name in FIELDS}
        row = {
            "reference_step": i,
            "candidate_step": j,
            "redshift": lm["steps"][i]["redshift_end"],
            "reference_statistics": _statistics(a),
            "candidate_statistics": _statistics(b),
        }
        if paired:
            a, b = (restrict_fields(field, comparison_dimension) for field in (a, b))
            row["paired_errors"] = _errors(a, b)
        if edges is not None:
            a_hi, b_hi = 1 - a["xHII"], 1 - b["xHII"]
            pa, pb = (
                neutral_spectrum(a_hi, lengths[0], edges),
                neutral_spectrum(b_hi, lengths[1], edges),
            )
            power = {
                "reference": pa,
                "candidate": pb,
                "candidate_over_reference": [
                    float(y / x) if x is not None and y is not None and x > 0 else None
                    for x, y in zip(pa["power_cMpc3"], pb["power_cMpc3"], strict=True)
                ],
            }
            if paired:
                power["cross_correlation"] = _cross_correlation(
                    a_hi, b_hi, lengths[0], edges
                )
            row["neutral_fraction_power"] = power
        else:
            row["neutral_fraction_power"] = {
                "unavailable": "No common physical band below half the smaller Nyquist frequency"
            }
        if diagnostic_thresholds:
            row["exploratory_difference_flags"] = _difference_flags(row)
        snapshots.append(row)
    provenance = [m.get("native_history_provenance", {}) for m in (lm, rm)]
    hashes = [item.get("initial_hires_density_sha256") for item in provenance]
    return {
        "schema_version": 1,
        "comparison_kind": "paired_transport" if paired else "statistical_only",
        "convergence_certified": False,
        "diagnostic_tolerances": DIAGNOSTIC_TOLERANCES
        if diagnostic_thresholds
        else None,
        "pairing_reason": "Exact native source/gas history hashes and physical initial settings match"
        if paired
        else "Source/gas history or physical box differs; no paired cell errors or cross-correlation are computed",
        "shared_hires_ic_hash": bool(
            hashes[0] and hashes[0] == hashes[1] and lengths[0] == lengths[1]
        ),
        "native_history_provenance": provenance,
        "comparison_dimension": comparison_dimension,
        "restriction": "Nuclei-weighted ionization and temperature; ionized-atom count conserved"
        if paired
        else None,
        "common_power_k_min_cMpc_inv": k_min,
        "common_power_k_max_cMpc_inv": k_max,
        "power_definition": "Absolute neutral-fraction fluctuations, mean removed; no saturated-Ts or RSD brightness approximation",
        "snapshots": snapshots,
    }


def load_run(directory: Path) -> dict:
    """Load a complete run, verify its native payload and memory-map fields.

    Parameters
    ----------
    directory : Path
        Results directory containing summary.json and science/native_output.bin.
    """
    directory = Path(directory).resolve()
    if not (directory / "COMPLETED").is_file():
        raise ValueError(f"Run is incomplete: {directory}")
    summary = json.loads((directory / "summary.json").read_text())
    binary = directory / "science" / "native_output.bin"
    with binary.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != summary["native_output_sha256"]:
        raise ValueError(f"Native output hash changed: {binary}")
    state = _backend.read_output(binary)
    manifest = summary["input_manifest"]
    if (
        state["dimension"] != manifest["dimension"]
        or len(state["steps"]) != manifest["step_count"]
    ):
        raise ValueError("Native output geometry differs from summary")
    return {
        "manifest": manifest,
        "fields": state["fields"],
        "directory": str(directory),
    }


def main() -> None:
    """Write a fresh JSON comparison of two completed native runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--power-bins", type=int, default=12)
    parser.add_argument("--diagnostic-thresholds", action="store_true")
    args = parser.parse_args()
    result = compare_states(
        load_run(args.reference),
        load_run(args.candidate),
        power_bins=args.power_bins,
        diagnostic_thresholds=args.diagnostic_thresholds,
    )
    result.update(
        reference=str(args.reference.resolve()), candidate=str(args.candidate.resolve())
    )
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
