"""Check that spatial probes keep the physical problem fixed."""

import numpy as np
import pytest

from research.c2ray_teacher.spatial_probe import (
    BASE_DIMENSION,
    BASE_POSITIONS,
    source_positions,
    validate_refinements,
)


@pytest.mark.parametrize("q", [1, 3, 5, 9])
def test_odd_refinement_preserves_physical_source_centres(q: int) -> None:
    """A mesh test must not move or split the original point sources."""
    positions = source_positions(q)
    np.testing.assert_array_equal(
        (positions - 0.5) / (BASE_DIMENSION * q),
        (BASE_POSITIONS - 0.5) / BASE_DIMENSION,
    )
    assert np.all(positions >= 1)
    assert np.all(positions <= BASE_DIMENSION * q)


@pytest.mark.parametrize("factors", [[], [0], [-1], [2], [True], [1.5], [1, 1]])
def test_invalid_refinement_is_rejected(factors: list) -> None:
    """Reject shifts, duplicate cases and malformed refinement requests."""
    with pytest.raises(ValueError, match="refinements"):
        validate_refinements(factors)
