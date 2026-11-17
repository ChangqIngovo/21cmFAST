"""Supervised native-input emulator of coeval photon absorption.

A fixed stack of donor-centred exposure layers produces spatial features.
The fitted field head corrects finite-exposure effects and a separate learned
head estimates the unabsorbed photon count before global budget projection.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from research.pc_analytic.absorption import absorbed
from research.pc_analytic_emulator.exposure_features import exposure_features
from research.pc_analytic_emulator.exposure_onset import OnsetHead
from research.pc_analytic_emulator.predict import (
    FIELD_ARRAYS,
    _digest,
    _physical_compatibility,
    _read_inputs,
    _saved_ledger,
)
from research.pc_analytic_emulator.projection import project_capacity_l2

FIELD_TERMS = 10
BUDGET_TERMS = 5


def implementation_hashes() -> dict[str, str]:
    """Bind weights to the numerical code used to produce predictions."""
    here = Path(__file__).parent
    return {
        name: _digest(here / name)
        for name in (
            "exposure_emulator.py",
            "exposure_features.cpp",
            "exposure_features.py",
            "exposure_onset.py",
            "projection.py",
        )
    }


@dataclass(frozen=True)
class FieldInputs:
    """Source-only dose features, field bounds and global reservoir features."""

    upper: np.ndarray
    latent: np.ndarray
    basis: np.ndarray
    reservoir: np.ndarray
    load: float


def field_inputs(
    source: np.ndarray,
    atoms: np.ndarray,
    near: np.ndarray,
    far: np.ndarray,
    onset: np.ndarray,
    escape: float,
    opacity: float,
) -> FieldInputs:
    """Build the learned-head inputs from normalized native counts."""
    upper = 1 - absorbed(source, atoms, opacity / 2) / atoms
    local = source / atoms
    external = near + far
    latent = 1 - local - external
    front = np.clip(latent, -1, 1)
    dose = np.clip(external, 0, 2)
    before = np.divide(1, local, out=np.ones_like(local), where=local > 1)
    load = float(np.mean(source))
    minimum_escape = max(load - 1.0, 0.0)
    neutral_mass = np.mean(atoms * np.clip(latent, 0, upper))
    excess_escape = max(escape - minimum_escape, 0.0)
    reservoir = np.array(
        [
            excess_escape,
            excess_escape * (np.clip(load, 0, 2) - 0.75),
            max(neutral_mass - max(1 - load, 0), 0),
            excess_escape * (np.mean(onset) - 0.65),
            excess_escape * np.mean(latent > 0),
        ]
    )
    basis = np.stack(
        [
            dose,
            dose * np.clip(load - 0.65, -1, 1),
            np.clip(near - far, -2, 2),
            dose * (atoms - 1) / (atoms + 1),
            dose * np.clip(local - 0.5, -1, 1),
            dose * front,
            dose * front**2,
            dose * (gaussian_filter(front, 0.75, mode="wrap") - front),
            dose * np.clip(before - onset, 0, 1),
            dose * (onset - 1),
        ],
        axis=0,
    ).reshape(FIELD_TERMS, -1)
    return FieldInputs(upper, latent, basis, reservoir, load)


def _project(
    latent: np.ndarray, upper: np.ndarray, atoms: np.ndarray, absorbed_count: float
) -> tuple[np.ndarray, dict]:
    initial_ionized = 1 - upper
    local = np.sum(atoms * initial_ionized, dtype=np.longdouble)
    total = np.sum(atoms, dtype=np.longdouble)
    if np.longdouble(absorbed_count) <= local:
        corrected = (
            initial_ionized * float(absorbed_count / local)
            if local
            else initial_ionized
        )
        return corrected, {"method": "scale_local_floor"}
    if np.longdouble(absorbed_count) >= total:
        return np.ones_like(atoms), {"method": "saturated"}
    normalized = np.divide(
        upper - latent, upper, out=np.zeros_like(upper), where=upper > 0
    )
    corrected, projection = project_capacity_l2(
        normalized,
        atoms * upper,
        float(np.longdouble(absorbed_count) - local),
        rtol=1e-13,
        atol=0.0,
    )
    return np.clip(initial_ionized + upper * corrected, 0, 1), {
        **projection,
        "method": "local_absorption_bounded_projection",
    }


@dataclass(frozen=True)
class ExposureEmulator:
    """Fitted spatial and reservoir coefficients for supported native grids."""

    metadata: dict[str, Any]
    field_weights: np.ndarray
    budget_weights: np.ndarray
    onset_head: OnsetHead | None = None

    @classmethod
    def load(cls, directory: Path) -> ExposureEmulator:
        """Load checked numeric model weights without deserializing objects."""
        with (directory / "model.json").open() as stream:
            metadata = json.load(stream)
        if (
            metadata.get("schema") != 1
            or metadata.get("method") != "learned_coeval_exposure"
            or type(metadata.get("depth")) is not int
            or metadata["depth"] <= 0
            or metadata.get("weights_sha256") != _digest(directory / "weights.npz")
        ):
            raise ValueError("Invalid exposure emulator manifest or weights")
        if "implementation_sha256" in metadata and (
            metadata["implementation_sha256"] != implementation_hashes()
        ):
            raise ValueError("Frozen exposure emulator implementation changed")
        with np.load(directory / "weights.npz", allow_pickle=False) as saved:
            field, budget = saved["field"], saved["budget"]
        n = len(metadata["training_geometries"])
        if (
            field.shape != (n, FIELD_TERMS)
            or budget.shape != (n, BUDGET_TERMS)
            or not np.isfinite(field).all()
            or not np.isfinite(budget).all()
        ):
            raise ValueError("Invalid exposure emulator coefficient arrays")
        onset_head = None
        if onset_hash := metadata.get("onset_sha256"):
            path = directory / "onset.npz"
            if _digest(path) != onset_hash:
                raise ValueError(
                    "Exposure onset weights differ from the model manifest"
                )
            onset_head = OnsetHead.load(path)
        return cls(metadata, field, budget, onset_head)

    def predict(
        self,
        source: np.ndarray,
        atoms: np.ndarray,
        redshift: float,
        box_len: float,
        *,
        native_es_xHI: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Predict the neutral field and global reservoir from native counts."""
        if not np.isfinite(redshift) or redshift < 0:
            raise ValueError("Require a finite nonnegative redshift")
        matching = [
            index
            for index, record in enumerate(self.metadata["training_geometries"])
            if (record["HII_DIM"], record["BOX_LEN"]) == (len(source), box_len)
        ]
        if len(matching) != 1:
            raise ValueError("Geometry is outside the exposure emulator training set")
        index = matching[0]
        initial = None
        if self.onset_head is not None:
            if native_es_xHI is None:
                raise ValueError("The onset head requires the native ES grid")
            initial = self.onset_head.initialize(
                source, atoms, native_es_xHI, redshift, box_len
            )
        _, near, far, onset, escape = exposure_features(
            source, atoms, depth=self.metadata["depth"], gains=(1.0, 1.0), onset=initial
        )
        mean = atoms.mean()
        features = field_inputs(
            source / mean,
            atoms / mean,
            near,
            far,
            onset,
            escape,
            self.metadata["opacity_prefactor"]
            * (1 + redshift) ** 2
            * box_len
            / len(source),
        )
        return predict_fields(
            features,
            source,
            atoms,
            self.field_weights[index],
            self.budget_weights[index],
        )


def predict_fields(
    features: FieldInputs,
    source: np.ndarray,
    atoms: np.ndarray,
    field_weights: np.ndarray,
    budget_weights: np.ndarray,
) -> dict[str, Any]:
    """Evaluate both learned heads, then impose their predicted global budget."""
    latent_hi = features.latent + (field_weights @ features.basis).reshape(atoms.shape)
    raw = np.clip(latent_hi, 0, features.upper)
    atom_count = np.sum(atoms, dtype=np.longdouble)
    emitted = np.sum(source, dtype=np.longdouble)
    maximum = min(atom_count, emitted)
    local = min(np.sum(atoms * (1 - features.upper), dtype=np.longdouble), maximum)
    excess = max(float(budget_weights @ features.reservoir), 0.0)
    target = float(max(maximum - excess * atom_count, local))
    if np.longdouble(target) > maximum:
        target = float(np.nextafter(target, 0.0))
    unabsorbed = float(emitted - target)
    corrected, projection = _project(latent_hi, features.upper, atoms, target)
    return {
        "xHI": 1 - corrected,
        "xHII": corrected,
        "raw_xHII": 1 - raw,
        "raw_xHI": raw,
        "latent_xHII": 1 - latent_hi,
        "predicted_absorbed_photons": target,
        "predicted_unabsorbed_photons": unabsorbed,
        "absorption_efficiency": target / float(maximum) if maximum else 1.0,
        "projection": projection,
    }


def run(
    model: Path, snapshot: Path, output: Path, *, redshift: float, box_len: float
) -> dict:
    """Save a pure model prediction for a native snapshot and audit it on disk."""
    if output.exists():
        raise FileExistsError(output)
    input_path = snapshot / "source.npz"
    inputs, _ = _read_inputs(input_path)
    emulator = ExposureEmulator.load(model)
    physics = _physical_compatibility(
        emulator, snapshot / "native.json", inputs, redshift, box_len
    )
    started = time.perf_counter()
    result = emulator.predict(
        inputs["source"],
        inputs["capacity"],
        redshift,
        box_len,
        native_es_xHI=inputs["native_es_xHI"],
    )
    seconds = time.perf_counter() - started
    output.mkdir(parents=True, exist_ok=False)
    arrays = inputs | {name: result[name] for name in FIELD_ARRAYS}
    for name in (
        "predicted_absorbed_photons",
        "predicted_unabsorbed_photons",
        "absorption_efficiency",
    ):
        arrays[name] = np.asarray(result[name], dtype=np.float64)
    path = output / "fields.npz"
    np.savez_compressed(path, **arrays)
    ledger = _saved_ledger(path, inputs)
    report = {
        "complete": True,
        "method": emulator.metadata["method"],
        "redshift": redshift,
        "box_len_cMpc": box_len,
        "model_sha256": _digest(model / "model.json"),
        "field_sha256": _digest(path),
        "native_physics": physics,
        "inference_seconds": seconds,
        "projection": result["projection"],
        "saved_field_photon_ledger": ledger,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def main() -> None:
    """Run the trained emulator from an archived native snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "snapshot", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--redshift", type=float, required=True)
    parser.add_argument("--box-len", type=float, required=True)
    args = parser.parse_args()
    run(
        args.model,
        args.snapshot,
        args.output,
        redshift=args.redshift,
        box_len=args.box_len,
    )


if __name__ == "__main__":
    main()
