"""Learn coeval ES residuals with an independently predicted global budget.

Both inference heads use source, capacity, native ES and geometry only. A
separate scalar head predicts absorbed photons; a bounded field projection
enforces that count. The complementary predicted unabsorbed reservoir is set
before auditing the field, and is not an independently measured escape count.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter

from research.pc_analytic_emulator.projection import project_capacity_l2
from research.pc_field_emulator.features import build_features
from research.pc_field_emulator.model import MLPModel, project_global_budget

if TYPE_CHECKING:
    from research.pc_analytic_emulator.budget import TabulatedBudgetModel
    from research.pc_analytic_emulator.latent import LatentFieldModel

DEFAULT_RADII = (6.25, 12.5, 25.0, 50.0)


def _arrays(
    source: ArrayLike, capacity: ArrayLike, native_es_xHI: ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    photons = np.asarray(source, dtype=np.float64)
    atoms = np.asarray(capacity, dtype=np.float64)
    neutral = np.asarray(native_es_xHI, dtype=np.float64)
    if (
        photons.ndim != 3
        or len(set(photons.shape)) != 1
        or photons.shape[0] < 2
        or photons.shape != atoms.shape
        or photons.shape != neutral.shape
    ):
        raise ValueError("Source, capacity and native ES require the same cubic grid")
    if not np.isfinite(photons).all() or np.any(photons < 0):
        raise ValueError("Source photons must be finite and nonnegative")
    if not np.isfinite(atoms).all() or np.any(atoms <= 0):
        raise ValueError("Native atom capacity must be finite and strictly positive")
    if not np.isfinite(neutral).all() or np.any((neutral < 0) | (neutral > 1)):
        raise ValueError("The native ES neutral fraction must be finite and bounded")
    return photons, atoms, neutral


def field_features(
    source: ArrayLike,
    capacity: ArrayLike,
    native_es_xHI: ArrayLike,
    redshift: float,
    box_len: float,
    *,
    radii_cMpc: Sequence[float] = DEFAULT_RADII,
) -> tuple[NDArray[np.float32], list[str], NDArray[np.float64]]:
    """Build input-only local and smoothed features for the field head.

    Parameters
    ----------
    source, capacity, native_es_xHI
        Native cumulative photons, atom capacities and ES neutral fractions.
    redshift, box_len
        Snapshot redshift and comoving box length in cMpc.
    radii_cMpc
        Physical Gaussian standard deviations, with periodic filtering.

    Returns
    -------
    tuple
        Per-cell feature matrix, ordered names and flattened ES xHII baseline.
        No teacher field or photon reservoir is accepted as an input.
    """
    photons, atoms, neutral = _arrays(source, capacity, native_es_xHI)
    features, names = build_features(
        atoms, photons, redshift, box_len, radii_cMpc=radii_cMpc
    )
    ionized = 1 - neutral
    columns = [ionized.ravel()]
    names.append("native_es_xHII")
    dx = box_len / photons.shape[0]
    for radius in radii_cMpc:
        columns.append(gaussian_filter(ionized, radius / dx, mode="wrap").ravel())
        names.append(f"native_es_xHII_sigma{radius:g}_cMpc")
    features = np.column_stack((features, *columns)).astype(np.float32)
    return features, names, ionized.ravel()


def budget_features(
    source: ArrayLike,
    capacity: ArrayLike,
    native_es_xHI: ArrayLike,
    redshift: float,
    box_len: float,
    *,
    radii_cMpc: Sequence[float] = DEFAULT_RADII,
) -> tuple[NDArray[np.float64], list[str]]:
    """Summarize inference inputs for the box absorption-efficiency head.

    Parameters
    ----------
    source, capacity, native_es_xHI
        Native photon, atom and ES neutral fields, without teacher information.
    redshift, box_len
        Snapshot redshift and box length in cMpc.
    radii_cMpc
        Physical Gaussian widths used for source-structure statistics.

    Returns
    -------
    tuple
        A one-row feature matrix and its ordered feature names.
    """
    photons, atoms, neutral = _arrays(source, capacity, native_es_xHI)
    radii = tuple(float(radius) for radius in radii_cMpc)
    if (
        not np.isfinite(redshift)
        or redshift < 0
        or not np.isfinite(box_len)
        or box_len <= 0
        or any(not np.isfinite(radius) or radius <= 0 for radius in radii)
        or len(set(radii)) != len(radii)
    ):
        raise ValueError("Invalid redshift, box length or Gaussian radii")
    atom_mean = float(np.mean(atoms))
    source_mean = float(np.mean(photons))
    ionized = 1 - neutral
    phase = ionized >= 0.5
    front = np.zeros_like(phase)
    for axis in range(3):
        front |= phase != np.roll(phase, 1, axis=axis)
        front |= phase != np.roll(phase, -1, axis=axis)
    dx = box_len / atoms.shape[0]
    values = [
        np.log1p(source_mean / atom_mean),
        redshift,
        np.log1p(dx),
        np.log1p(box_len),
        np.log1p(np.std(atoms) / atom_mean),
        np.log1p(np.std(photons) / atom_mean),
        np.mean(ionized),
        np.mean(atoms * ionized) / atom_mean,
        np.std(ionized),
        np.mean(front),
    ]
    names = [
        "log1p_photons_per_atom",
        "redshift",
        "log1p_dx_cMpc",
        "log1p_box_cMpc",
        "log1p_density_std_over_mean",
        "log1p_source_std_over_mean_capacity",
        "native_es_Q_volume",
        "native_es_Q_mass",
        "native_es_xHII_std",
        "native_es_front_fraction",
    ]
    for radius in radii:
        smoothed = gaussian_filter(photons, radius / dx, mode="wrap")
        values.append(np.log1p(np.std(smoothed) / atom_mean))
        names.append(f"log1p_source_std_sigma{radius:g}_cMpc")
    features = np.array([values], dtype=np.float64)
    if not np.isfinite(features).all():
        raise ValueError("Budget feature construction overflowed")
    return features, names


def absorption_efficiency(
    source: ArrayLike, capacity: ArrayLike, teacher_xHI: ArrayLike
) -> float:
    """Construct a supervised scalar target without assuming zero escape.

    Parameters
    ----------
    source, capacity, teacher_xHI
        Photon budget, atom capacity and teacher neutral fraction. This
        function is for training-label construction, never inference features.

    Returns
    -------
    float
        Field-derived absorbed atoms divided by min(total photons, total atoms).
        A photon-free neutral box has efficiency one by convention.
    """
    photons, atoms, neutral = _arrays(source, capacity, teacher_xHI)
    total_photons = np.sum(photons, dtype=np.longdouble)
    total_atoms = np.sum(atoms, dtype=np.longdouble)
    absorbed = np.sum(
        np.multiply(atoms, 1 - neutral, dtype=np.longdouble), dtype=np.longdouble
    )
    available = min(total_photons, total_atoms)
    if available == 0:
        if absorbed != 0:
            raise ValueError("Teacher absorption exceeds its photon budget")
        return 1.0
    efficiency = float(absorbed / available)
    if efficiency < -1e-12 or efficiency > 1 + 1e-10:
        raise ValueError("Teacher absorption exceeds its available photons/atoms")
    return float(np.clip(efficiency, 0, 1))


@dataclass
class Emulator:
    """Couple an ES-residual field predictor to an independent global budget.

    Parameters
    ----------
    field_model, budget_model
        The field uses native ES xHII plus a bounded MLP or unbounded latent
        correction. The absorption head is a residual MLP around efficiency
        one or a training-only interpolation table.
    radii_cMpc
        Physical smoothing scales used identically at training and inference.
    metadata
        Training provenance, feature names and model-selection diagnostics.
    projection_mode
        ``log_odds`` preserves the original bounded-MLP behavior.
        ``capacity_l2`` minimizes capacity-weighted squared field adjustment
        and also supports unbounded latent corrections.
    """

    field_model: MLPModel | LatentFieldModel
    budget_model: MLPModel | TabulatedBudgetModel
    radii_cMpc: tuple[float, ...] = DEFAULT_RADII
    metadata: dict[str, Any] = field(default_factory=dict)
    projection_mode: str = "log_odds"

    def _model_kinds(self) -> tuple[str, str]:
        if self.projection_mode not in ("log_odds", "capacity_l2"):
            raise ValueError("Unsupported emulator projection mode")
        field_mode, budget_mode = (
            self.field_model.output_mode,
            self.budget_model.output_mode,
        )
        if field_mode not in ("residual", "latent_linear"):
            raise ValueError("Unsupported field model output mode")
        if field_mode == "latent_linear" and self.projection_mode != "capacity_l2":
            raise ValueError("Latent field models require capacity_l2 projection")
        if budget_mode not in ("residual", "efficiency"):
            raise ValueError("Unsupported budget model output mode")
        return (
            "latent_linear" if field_mode == "latent_linear" else "mlp",
            "tabulated" if budget_mode == "efficiency" else "mlp",
        )

    def predict(
        self,
        source: ArrayLike,
        capacity: ArrayLike,
        native_es_xHI: ArrayLike,
        redshift: float,
        box_len: float,
    ) -> dict[str, Any]:
        """Predict a bounded field and audit its independent output reservoirs.

        Parameters
        ----------
        source, capacity, native_es_xHI
            Inference-time native photon, atom and ES neutral fields.
        redshift, box_len
            Snapshot redshift and box size in cMpc.

        Returns
        -------
        dict
            Raw/constrained fields, predicted absorbed/unabsorbed counts and
            their field-derived ledger. The unabsorbed reservoir combines
            retained and escaped photons without resolving their transport.
        """
        photons, atoms, _ = _arrays(source, capacity, native_es_xHI)
        features, _, baseline = field_features(
            photons,
            atoms,
            native_es_xHI,
            redshift,
            box_len,
            radii_cMpc=self.radii_cMpc,
        )
        box_features, _ = budget_features(
            photons,
            atoms,
            native_es_xHI,
            redshift,
            box_len,
            radii_cMpc=self.radii_cMpc,
        )
        field_kind, budget_kind = self._model_kinds()
        if field_kind == "latent_linear":
            latent = self.field_model.latent_predict(
                features, baseline=baseline
            ).reshape(atoms.shape)
            raw = np.clip(latent, 0, 1)
        else:
            raw = self.field_model.predict(features, baseline=baseline).reshape(
                atoms.shape
            )
            latent = raw
        efficiency = float(
            self.budget_model.predict(box_features)[0]
            if budget_kind == "tabulated"
            else self.budget_model.predict(box_features, baseline=np.ones(1))[0]
        )
        if not np.isfinite(efficiency) or not 0 <= efficiency <= 1:
            raise ValueError("Predicted absorption efficiency must lie in [0,1]")
        total_source = np.sum(photons, dtype=np.longdouble)
        total_capacity = np.sum(atoms, dtype=np.longdouble)
        limit = min(total_source, total_capacity)
        target = float(limit * efficiency)
        if not np.isfinite(target) or not np.isfinite(float(total_source)):
            raise ValueError("Photon count overflowed the supported inference units")
        if np.longdouble(target) > limit:
            target = float(np.nextafter(target, 0.0))
        # This reservoir is fixed by the model BEFORE projecting/auditing the
        # field; defining it from field absorption would make the ledger tautological.
        unabsorbed = float(total_source - np.longdouble(target))
        if total_source == 0:
            raw = np.zeros_like(raw)
            latent = raw
        if self.projection_mode == "capacity_l2":
            corrected, projection = project_capacity_l2(
                latent, atoms, target, rtol=1e-13, atol=0.0
            )
        else:
            corrected, projection = project_global_budget(
                raw, atoms, target, rtol=1e-13, atol=0.0
            )
        projection.update(method=self.projection_mode, field_kind=field_kind)
        field_absorbed = np.sum(
            np.multiply(atoms, corrected, dtype=np.longdouble), dtype=np.longdouble
        )
        residual = float(total_source - field_absorbed - np.longdouble(unabsorbed))
        relative = abs(residual) / max(float(total_source), np.finfo(float).tiny)
        bounded = bool(
            np.isfinite(corrected).all() and np.all((corrected >= 0) & (corrected <= 1))
        )
        if not bounded or unabsorbed < 0 or relative > 1e-10:
            raise FloatingPointError("Independent predicted-field photon ledger failed")
        ledger = {
            "source_photons": float(total_source),
            "atom_capacity": float(total_capacity),
            "predicted_absorbed_photons": target,
            "field_absorbed_photons": float(field_absorbed),
            "predicted_unabsorbed_photons": unabsorbed,
            "photon_residual": residual,
            "relative_photon_residual": relative,
            "bounded": bounded,
            "passed": True,
            "reservoir_definition": "source minus independently predicted absorbed count, fixed before field projection",
            "scope": "global predicted absorbed plus unabsorbed budget; no local/source-wise transport or measured escape",
        }
        return {
            "raw_xHI": 1 - raw,
            "xHI": 1 - corrected,
            "raw_xHII": raw,
            "latent_xHII": latent,
            "xHII": corrected,
            "absorption_efficiency": efficiency,
            "predicted_absorbed_photons": target,
            "predicted_unabsorbed_photons": unabsorbed,
            "photon_ledger": ledger,
            "projection": projection,
        }

    def save(self, directory: str | Path) -> None:
        """Save both fitted heads and their common inference configuration.

        Parameters
        ----------
        directory
            Destination containing two pickle-free NPZ models and model.json.
        """
        field_kind, budget_kind = self._model_kinds()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.field_model.save(directory / "field_model.npz")
        self.budget_model.save(directory / "budget_model.npz")
        (directory / "model.json").write_text(
            json.dumps(
                {
                    "schema": 2,
                    "method": "native-ES-residual-with-learned-global-absorption",
                    "radii_cMpc": list(self.radii_cMpc),
                    "metadata": self.metadata,
                    "projection_mode": self.projection_mode,
                    "field_kind": field_kind,
                    "budget_kind": budget_kind,
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )

    @classmethod
    def load(cls, directory: str | Path) -> Emulator:
        """Restore a fitted emulator without refitting its standardizers.

        Parameters
        ----------
        directory
            Directory written by :meth:`save`.

        Returns
        -------
        Emulator
            Restored field and scalar models with shared inference settings.
        """
        directory = Path(directory)
        record = json.loads((directory / "model.json").read_text())
        if (
            record.get("schema") not in (1, 2)
            or record.get("method")
            != "native-ES-residual-with-learned-global-absorption"
        ):
            raise ValueError("Unsupported analytic emulator schema")
        radii = tuple(float(value) for value in record["radii_cMpc"])
        if any(not np.isfinite(value) or value <= 0 for value in radii) or len(
            set(radii)
        ) != len(radii):
            raise ValueError("Saved emulator has invalid smoothing scales")
        if not isinstance(record["metadata"], dict):
            raise TypeError("Saved emulator metadata must be an object")
        if record["schema"] == 1:
            projection_mode, field_kind, budget_kind = "log_odds", "mlp", "mlp"
        else:
            projection_mode = record["projection_mode"]
            field_kind, budget_kind = record["field_kind"], record["budget_kind"]
        if field_kind == "latent_linear":
            from research.pc_analytic_emulator.latent import LatentFieldModel

            field_model = LatentFieldModel.load(directory / "field_model.npz")
        elif field_kind == "mlp":
            field_model = MLPModel.load(directory / "field_model.npz")
        else:
            raise ValueError("Unsupported saved field model kind")
        if budget_kind == "tabulated":
            from research.pc_analytic_emulator.budget import TabulatedBudgetModel

            budget_model = TabulatedBudgetModel.load(directory / "budget_model.npz")
        elif budget_kind == "mlp":
            budget_model = MLPModel.load(directory / "budget_model.npz")
        else:
            raise ValueError("Unsupported saved budget model kind")
        result = cls(
            field_model,
            budget_model,
            radii,
            record["metadata"],
            projection_mode,
        )
        result._model_kinds()
        return result
