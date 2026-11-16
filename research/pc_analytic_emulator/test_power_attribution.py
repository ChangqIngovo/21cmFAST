"""Check algebraic power attribution on fields with known Fourier content."""

import numpy as np
import pytest

from research.pc_analytic.validation_20261116.reproduction.decompose_power import (
    decompose_power,
)


def _waves() -> tuple[np.ndarray, np.ndarray]:
    index = np.arange(16) * (2 * np.pi / 16)
    x = np.broadcast_to(np.sin(index)[:, None, None], (16, 16, 16))
    y = np.broadcast_to(np.sin(index)[None, :, None], (16, 16, 16))
    return 0.4 + 0.1 * x, 0.4 + 0.1 * y


def test_uniform_amplitude_error_has_exact_mean_factor() -> None:
    """A constant brightness rescaling changes no normalized contrast."""
    teacher, _ = _waves()
    result = decompose_power(teacher, 2 * teacher, 50.0)
    shell = result["shells"][0]
    assert result["mean_factor_squared"] == pytest.approx(4)
    assert shell["power_ratio"] == pytest.approx(4)
    assert shell["fractional_contrast_power_ratio"] == pytest.approx(1)
    assert shell["coherent_transfer"] == pytest.approx(2)
    assert shell["orthogonal_power_over_teacher"] < 1e-25


def test_equal_power_can_hide_orthogonal_structure() -> None:
    """Orthogonal waves have equal auto-power but no cross-power."""
    teacher, predicted = _waves()
    result = decompose_power(teacher, predicted, 50.0)
    shell = result["shells"][0]
    assert result["mean_factor_squared"] == pytest.approx(1)
    assert shell["power_ratio"] == pytest.approx(1)
    assert abs(shell["cross_correlation"]) < 1e-14
    assert shell["orthogonal_power_over_teacher"] == pytest.approx(1)
    assert shell["field_residual_power_over_teacher"] == pytest.approx(2)


def test_negative_coherent_response_keeps_its_sign() -> None:
    """An inverted fluctuation keeps auto-power but has negative response."""
    teacher, _ = _waves()
    shell = decompose_power(teacher, 0.8 - teacher, 50.0)["shells"][0]
    assert shell["power_ratio"] == pytest.approx(1)
    assert shell["coherent_transfer"] == pytest.approx(-1)
    assert shell["cross_correlation"] == pytest.approx(-1)
    assert shell["field_residual_power_over_teacher"] == pytest.approx(4)


def test_zero_prediction_keeps_power_error_and_undefined_contrast() -> None:
    """A fully ionized prediction has zero power and no fractional contrast."""
    teacher, _ = _waves()
    result = decompose_power(teacher, np.zeros_like(teacher), 50.0)
    assert result["mean_factor_squared"] == 0
    assert result["mean_absolute_power_ratio_error"] == pytest.approx(1)
    assert result["mean_absolute_fractional_contrast_power_error"] is None
    shell = result["shells"][0]
    assert shell["power_ratio"] == 0
    assert shell["cross_correlation"] is None
    assert shell["field_residual_power_over_teacher"] == pytest.approx(1)


def test_no_reference_fluctuations_are_undefined() -> None:
    """Constant reference fields do not define a relative power spectrum."""
    teacher, _ = _waves()
    result = decompose_power(np.ones_like(teacher), teacher, 50.0)
    assert result["defined_shells"] == 0
    assert result["mean_absolute_power_ratio_error"] is None


@pytest.mark.parametrize("length", [0.0, np.nan])
def test_reject_invalid_geometry(length: float) -> None:
    """An invalid physical box cannot define a power normalization."""
    teacher, _ = _waves()
    with pytest.raises(ValueError, match="positive box length"):
        decompose_power(teacher, teacher, length)
