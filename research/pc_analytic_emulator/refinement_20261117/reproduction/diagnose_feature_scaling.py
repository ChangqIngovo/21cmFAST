"""Diagnose geometry-dependent feature saturation using temporary model copies.

This is a regression/development diagnostic. Counterfactual normalizers are
not fitted models, and their scores must not be presented as blind validation.
All original model files and audited teacher inputs remain unchanged.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import re
from collections.abc import Iterator, Mapping
from pathlib import Path

import numpy as np

from research.pc_analytic.validation_20261116.reproduction import recompute
from research.pc_analytic_emulator.benchmark import (
    _model_hashes,
    _physics_compatibility,
    independent_ledger,
)
from research.pc_analytic_emulator.data import TeacherSample, load_dataset
from research.pc_analytic_emulator.model import Emulator, field_features

LOGGER = logging.getLogger(__name__)
VARIANTS = (
    "original",
    "dx_mean_recursive",
    "density50_floor_recursive",
    "density25_50_floor_recursive",
    "dx_mean_density25_50_floor_recursive",
    "density25_50_floor_top_only",
)


def levels(model: Emulator) -> Iterator[Emulator]:
    """Visit every deployed field head, rejecting cyclic frozen-base graphs."""
    seen = set()
    while model is not None:
        if id(model) in seen:
            raise ValueError("Frozen model has a recursive cycle")
        seen.add(id(model))
        yield model
        model = model.base_model


def _names(model: Emulator) -> list[str]:
    # Column names are independent of geometry/values; no teacher is consulted.
    ones = np.ones((2, 2, 2))
    names = field_features(
        ones,
        ones,
        ones,
        0,
        50,
        radii_cMpc=model.radii_cMpc,
        feature_set=model.feature_set,
    )[1]
    recorded = model.metadata.get("field_feature_names")
    if recorded is not None and recorded != names:
        raise ValueError("Recorded feature names differ from inference definitions")
    return names


def intervene(model: Emulator, variant: str, floor: float = 0.01) -> Emulator:
    """Apply one declared counterfactual to a deep copy, without changing budgets.

    Zeroing a first-layer input row is equivalent to replacing that standardized
    coordinate by zero. Density interventions only increase the standardizer
    scales of the explicitly named broad density columns.
    """
    if variant not in VARIANTS:
        raise ValueError("Unknown feature intervention")
    if isinstance(floor, bool) or not np.isfinite(floor) or floor <= 0:
        raise ValueError("Density scale floor must be finite and positive")
    list(levels(model))
    changed = copy.deepcopy(model)
    for depth, level in enumerate(levels(changed)):
        if variant == "density25_50_floor_top_only" and depth:
            continue
        names = _names(level)
        if variant in {"dx_mean_recursive", "dx_mean_density25_50_floor_recursive"}:
            level.field_model.weights[0][names.index("log1p_dx_cMpc")] = 0
        radii = (
            (50,)
            if variant == "density50_floor_recursive"
            else (
                (25, 50)
                if variant
                in {
                    "density25_50_floor_recursive",
                    "dx_mean_density25_50_floor_recursive",
                    "density25_50_floor_top_only",
                }
                else ()
            )
        )
        for radius in radii:
            name = f"log1p_density_sigma{radius}_cMpc"
            if name not in names:
                raise ValueError(f"Intervention requires feature {name}")
            index = names.index(name)
            level.field_model.scale[index] = max(level.field_model.scale[index], floor)
    return changed


def feature_statistics(features: np.ndarray, names: list[str], head) -> dict:
    """Measure standardized feature magnitudes and first-layer saturation."""
    values = np.asarray(features, dtype=float)
    if values.ndim != 2 or not len(values) or values.shape[1] != len(names):
        raise ValueError("Feature matrix and names have incompatible dimensions")
    if not np.isfinite(values).all() or len(head.mean) != len(names):
        raise ValueError("Features are nonfinite or incompatible with the fitted head")
    standardized = (values - head.mean) / head.scale
    first = standardized @ head.weights[0] + head.biases[0]
    rows = []
    for index, name in enumerate(names):
        column = values[:, index]
        magnitudes = np.abs(standardized[:, index])
        rows.append(
            {
                "name": name,
                "training_mean": float(head.mean[index]),
                "training_scale": float(head.scale[index]),
                "input_min": float(column.min()),
                "input_max": float(column.max()),
                "input_std": float(column.std()),
                "max_abs_z": float(magnitudes.max()),
                "p95_abs_z": float(np.quantile(magnitudes, 0.95)),
            }
        )
    return {
        "first_hidden_abs_pre_activation_gt3_fraction": float(
            np.mean(np.abs(first) > 3)
        ),
        "first_hidden_abs_pre_activation_gt10_fraction": float(
            np.mean(np.abs(first) > 10)
        ),
        "all_coordinates": rows,
        "top_coordinates": sorted(
            rows, key=lambda item: item["max_abs_z"], reverse=True
        )[:8],
    }


def _gaussian_diagnostic(
    atoms: np.ndarray, length: float, radii=(25.0, 50.0)
) -> list[dict]:
    """Compare the defined four-sigma kernel with a spectral periodic Gaussian."""
    from scipy.ndimage import gaussian_filter

    n = len(atoms)
    wave = np.fft.fftfreq(n, d=length / n) * 2 * np.pi
    square = (
        wave[:, None, None] ** 2 + wave[None, :, None] ** 2 + wave[None, None, :] ** 2
    )
    transformed = np.fft.fftn(atoms)
    rows = []
    for radius in radii:
        truncated = gaussian_filter(atoms, radius / (length / n), mode="wrap")
        spectral = np.fft.ifftn(transformed * np.exp(-0.5 * radius**2 * square)).real
        rows.append(
            {
                "radius_cMpc": radius,
                "truncated_log1p_density_std": float(
                    np.std(np.log1p(truncated / atoms.mean()))
                ),
                "spectral_log1p_density_std": float(
                    np.std(np.log1p(spectral / atoms.mean()))
                ),
                "maximum_density_difference": float(
                    np.max(np.abs(truncated - spectral))
                ),
            }
        )
    return rows


def analyze_sample(
    sample: TeacherSample, label: str, model: Emulator, floor: float
) -> dict:
    """Recompute field and power diagnostics from common native inputs."""
    source, atoms, native, truth = (
        sample.arrays[key]
        for key in ("source", "capacity", "native_es_xHI", "teacher_xHI")
    )
    row = {
        "dataset": label,
        "seed": sample.seed,
        "redshift": sample.redshift,
        "dimension": sample.dimension,
        "box_len_cMpc": sample.box_len,
        "teacher_xHI_mass": float(
            recompute.total(atoms * truth) / recompute.total(atoms)
        ),
        "features_by_level": [],
        "interventions": {},
        "gaussian_kernel_comparison": _gaussian_diagnostic(atoms, sample.box_len),
    }
    for depth, level in enumerate(levels(model)):
        features, names, _ = field_features(
            source,
            atoms,
            native,
            sample.redshift,
            sample.box_len,
            radii_cMpc=level.radii_cMpc,
            feature_set=level.feature_set,
        )
        row["features_by_level"].append(
            {"depth": depth, **feature_statistics(features, names, level.field_model)}
        )
    brightness = {"teacher": atoms * truth}
    targets = []
    for variant in VARIANTS:
        candidate = intervene(model, variant, floor)
        prediction = candidate.predict(
            source, atoms, native, sample.redshift, sample.box_len
        )
        field = prediction["xHI"]
        target, unabsorbed = (
            prediction[key]
            for key in ("predicted_absorbed_photons", "predicted_unabsorbed_photons")
        )
        targets.append(target)
        row["interventions"][variant] = {
            **recompute.metrics(field, truth, atoms),
            "predicted_absorbed_photons": target,
            "predicted_unabsorbed_photons": unabsorbed,
            "independent_photon_ledger": independent_ledger(
                source, atoms, field, unabsorbed, target
            ),
        }
        brightness[variant] = atoms * field
    if len(set(targets)) != 1:
        raise RuntimeError(
            "A feature-only intervention unexpectedly changed the final photon target"
        )
    power = recompute.spectra(brightness, sample.box_len, 0.5)
    row["low_k_band"] = {
        key: power[key] for key in ("k_cMpc_inv", "modes", "mode_selection")
    }
    for variant in VARIANTS:
        row["interventions"][variant]["low_k_power"] = power["comparisons"][variant]
    return row


def _mean(values: list[float | None]) -> dict:
    defined = [value for value in values if value is not None]
    return {
        "mean": float(np.mean(defined)) if defined else None,
        "defined_count": len(defined),
        "undefined_count": len(values) - len(defined),
    }


def summarize(rows: list[dict]) -> dict:
    """Average snapshots equally, with undefined endpoint power excluded explicitly."""
    result = {}
    for label in sorted({row["dataset"] for row in rows}):
        selected = [row for row in rows if row["dataset"] == label]
        groups = {
            "all": selected,
            "early": [row for row in selected if row["teacher_xHI_mass"] >= 0.95],
            "active": [
                row for row in selected if 0.05 < row["teacher_xHI_mass"] < 0.95
            ],
            "late": [row for row in selected if row["teacher_xHI_mass"] <= 0.05],
        }
        result[label] = {}
        for group, chosen in groups.items():
            variants = {}
            for variant in VARIANTS:
                values = [row["interventions"][variant] for row in chosen]
                variants[variant] = {
                    "MAE": _mean([value["MAE"] for value in values]),
                    "absolute_delta_Q_mass": _mean(
                        [abs(value["delta_Q_mass"]) for value in values]
                    ),
                    "low_k_mean_absolute_power_ratio_error": _mean(
                        [
                            value["low_k_power"]["mean_absolute_power_ratio_error"]
                            for value in values
                        ]
                    ),
                }
            result[label][group] = {"count": len(chosen), "variants": variants}
    return result


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _public_files(root: Path, files: Mapping[str, str]) -> dict[str, str]:
    public = {}
    for index, (raw, digest) in enumerate(sorted(files.items())):
        path = Path(raw).resolve()
        # Relocated collection members may live outside root. Keep their names
        # and digests without publishing account-specific parent directories.
        name = (
            str(path.relative_to(root))
            if path.is_relative_to(root)
            else f"external_{index}/{path.name}"
        )
        public[name] = digest
    return public


def run(
    datasets: Mapping[str, str | Path],
    model: str | Path,
    output: str | Path,
    *,
    seeds: list[int] | None = None,
    density_scale_floor: float = 0.01,
) -> dict:
    """Audit selected regression campaigns and emit a portable path-free report."""
    if not datasets or any(
        not re.fullmatch(r"[A-Za-z0-9_-]+", label) for label in datasets
    ):
        raise ValueError("Dataset labels must be nonempty public alphanumeric names")
    model_path, output = Path(model).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Diagnostic output already exists")
    original = _model_hashes(model_path)
    fitted = Emulator.load(model_path)
    intervene(fitted, "original", density_scale_floor)
    output.mkdir(parents=True)
    rows, inputs, compatibility, source_inputs = [], {}, {}, {}
    requested = set(seeds) if seeds is not None else None
    found = set()
    for label, path in datasets.items():
        dataset = load_dataset(path)
        compatibility[label] = _physics_compatibility(fitted, dataset.physics)
        inputs[label] = _public_files(dataset.root, dataset.file_hashes)
        source_inputs.update(dataset.file_hashes)
        for sample in dataset.samples:
            if requested is not None and sample.seed not in requested:
                continue
            found.add(sample.seed)
            rows.append(analyze_sample(sample, label, fitted, density_scale_floor))
            LOGGER.info("%s seed%d z=%g complete", label, sample.seed, sample.redshift)
    if not rows or (requested is not None and requested != found):
        raise ValueError(
            "Requested regression seeds were missing or no samples were selected"
        )
    if _model_hashes(model_path) != original:
        raise RuntimeError("Original fitted model files changed during diagnosis")
    if any(_digest(Path(path)) != digest for path, digest in source_inputs.items()):
        raise RuntimeError("Audited teacher inputs changed during diagnosis")
    report = {
        "schema": 1,
        "complete": True,
        "scope": "Copied-model counterfactuals on regression/development data; no fitting, model selection, or new blind evaluation",
        "density_scale_floor": density_scale_floor,
        "intervention_definition": "dx_mean zeros the first-layer dx row, equivalent to zero standardized dx. Density interventions raise only named density-column scales; recursive changes cover every frozen base. Final budget heads remain unchanged.",
        "brightness": "capacity*xHI, proportional to saturated-spin-temperature brightness with a common factor; no RSD or spin-temperature evolution",
        "gaussian_comparison_scope": "Defined four-sigma spatial Gaussian versus periodic spectral exp(-sigma^2*k^2/2); diagnostic only, neither replaces deployed features",
        "averaging": "Equal snapshot weight, equal defined shell weight within each snapshot; zero teacher-power ratios remain undefined",
        "model_files_before_sha256": original,
        "model_files_after_sha256": _model_hashes(model_path),
        "model_files_unchanged": True,
        "input_files_unchanged": True,
        "input_files_sha256": inputs,
        "physical_compatibility": compatibility,
        "diagnostic_source_sha256": _digest(Path(__file__)),
        "summaries": summarize(rows),
        "snapshots": rows,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return report


def main() -> None:
    """Evaluate explicitly named development datasets with copied models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", action="append", required=True, metavar="LABEL=CAMPAIGN"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--density-scale-floor", type=float, default=0.01)
    args = parser.parse_args()
    datasets = {}
    for value in args.dataset:
        label, separator, path = value.partition("=")
        if not separator or not path or label in datasets:
            parser.error("Each --dataset requires a unique LABEL=CAMPAIGN")
        datasets[label] = Path(path)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = run(
        datasets,
        args.model,
        args.output,
        seeds=args.seeds,
        density_scale_floor=args.density_scale_floor,
    )
    LOGGER.info("Diagnostic complete: %d snapshots", len(report["snapshots"]))


if __name__ == "__main__":
    main()
