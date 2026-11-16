"""Check physical symmetries of complete deployed emulator predictions.

These input transformations preserve the coeval problem. They exercise the
features, budget, frozen base and final projection together rather than only
checking a network or photon ledger in isolation.
"""

from collections.abc import Callable

import numpy as np
import pytest

from research.pc_analytic_emulator.latent import LatentFieldModel
from research.pc_analytic_emulator.model import (
    Emulator,
    budget_features,
    field_features,
)
from research.pc_analytic_emulator.projection import project_capacity_l2
from research.pc_field_emulator.model import MLPModel


def _problem(kind: str) -> tuple[Emulator, tuple[np.ndarray, ...]]:
    random = np.random.default_rng(8351)
    capacity = random.lognormal(0, 0.6, (6, 6, 6))
    source = random.lognormal(-0.8, 0.9, capacity.shape)
    native = random.uniform(0, 1, capacity.shape)
    native.ravel()[::13] = 0
    native.ravel()[::17] = 1
    features, _, _ = field_features(source, capacity, native, 7.5, 96)
    global_features, _ = budget_features(source, capacity, native, 7.5, 96)

    def network(width: int, bias: float) -> MLPModel:
        return MLPModel(
            [random.normal(0, 0.03, (width, 5)), random.normal(0, 0.04, (5, 1))],
            [random.normal(0, 0.03, 5), np.array([bias])],
            np.zeros(width),
            np.ones(width),
            {},
            output_mode="residual",
        )

    base = Emulator(
        network(features.shape[1], 0.03),
        network(global_features.shape[1], -0.3),
        projection_mode="log_odds" if kind == "legacy" else "capacity_l2",
    )
    if kind == "residual":
        residual = LatentFieldModel(
            [
                random.normal(0, 0.03, (features.shape[1], 4)),
                random.normal(0, 0.1, (4, 1)),
            ],
            [np.zeros(4), np.array([0.04])],
            np.zeros(features.shape[1]),
            np.ones(features.shape[1]),
            {},
        )
        base = Emulator(
            residual,
            base.budget_model,
            projection_mode="capacity_l2",
            base_model=base,
        )
    return base, (source, capacity, native)


@pytest.mark.parametrize("kind", ["legacy", "capacity_l2", "residual"])
@pytest.mark.parametrize(
    "transform",
    [
        pytest.param(lambda x: np.roll(x, (1, -2, 3), (0, 1, 2)), id="translation"),
        pytest.param(lambda x: np.rot90(x, axes=(0, 2)), id="rotation"),
        pytest.param(lambda x: np.flip(x, axis=1), id="reflection"),
    ],
)
def test_complete_prediction_respects_periodic_cube_symmetries(
    kind: str, transform: Callable[[np.ndarray], np.ndarray]
) -> None:
    """A transformed physical cube must produce the correspondingly moved field."""
    model, arrays = _problem(kind)
    original = model.predict(*arrays, 7.5, 96)
    moved = model.predict(*(transform(array) for array in arrays), 7.5, 96)
    for name in ("xHI", "xHII", "raw_xHI", "latent_xHII"):
        np.testing.assert_allclose(
            moved[name], transform(original[name]), rtol=0, atol=2e-12
        )
    assert moved["absorption_efficiency"] == pytest.approx(
        original["absorption_efficiency"], rel=2e-13, abs=0
    )
    assert moved["photon_ledger"]["passed"]


@pytest.mark.parametrize("kind", ["legacy", "capacity_l2", "residual"])
@pytest.mark.parametrize("count_unit", [1e-18, 1e18])
def test_common_photon_and_atom_count_units_do_not_change_ionization(
    kind: str, count_unit: float
) -> None:
    """Only counts and reservoirs scale under a common change of count units."""
    model, (source, capacity, native) = _problem(kind)
    original = model.predict(source, capacity, native, 7.5, 96)
    rescaled = model.predict(
        source * count_unit, capacity * count_unit, native, 7.5, 96
    )
    for name in ("xHI", "xHII", "raw_xHI"):
        np.testing.assert_allclose(rescaled[name], original[name], rtol=0, atol=2e-12)
    for name in ("predicted_absorbed_photons", "predicted_unabsorbed_photons"):
        assert rescaled[name] / count_unit == pytest.approx(original[name], rel=2e-12)
    assert rescaled["photon_ledger"]["relative_photon_residual"] < 1e-10


def test_projection_handles_large_density_contrast_and_weak_sources() -> None:
    """Audit heterogeneous capacities and targets far below the atom supply."""
    random = np.random.default_rng(351)
    for index in range(80):
        size = int(random.integers(2, 80))
        capacity = 10.0 ** random.uniform(-12, 12, size)
        latent = random.normal(0.5, 2, size)
        fraction = (
            10.0 ** random.uniform(-15, -0.001)
            if index % 2
            else random.uniform(0.01, 0.99)
        )
        target = float(np.sum(capacity, dtype=np.longdouble)) * fraction
        corrected, _ = project_capacity_l2(latent, capacity, target)
        absorbed = np.sum(
            np.multiply(capacity, corrected, dtype=np.longdouble), dtype=np.longdouble
        )
        assert np.all((corrected >= 0) & (corrected <= 1))
        assert abs(float(absorbed - target)) <= 2e-13 * target
