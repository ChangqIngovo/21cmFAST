"""Scientific decomposition checks without RT runs or a training sweep."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from research.es_pc_emulator.small_pilot_model import shell_power  # noqa: E402
from research.es_pc_emulator.spectral_components import (  # noqa: E402
    component_loss,
    decompose_power,
)
from research.pc_field_emulator.evaluation import shell_spectra  # noqa: E402


@pytest.mark.parametrize("n", [5, 8])
def test_exact_signed_closure_and_estimator_normalization(n):
    """Even/odd shells and batch amplitudes agree with independent NumPy FFTs."""
    rng = np.random.default_rng(17)
    rho = torch.tensor(.5+rng.random((2, n, n, n)), dtype=torch.float64)
    neutral = torch.tensor(.1+.8*rng.random((2, n, n, n)), dtype=torch.float64)
    amplitude = torch.tensor([27., 31.], dtype=torch.float64)
    result = decompose_power(rho, neutral, amplitude, 50.)
    _, expected, _ = shell_power(amplitude[:, None, None, None]*rho*neutral, 50.)
    torch.testing.assert_close(result["total"], expected, rtol=1e-12, atol=1e-10)
    for i in range(2):
        reference = shell_spectra({"teacher": (amplitude[i]*rho[i]*neutral[i]).numpy()}, 50.)
        np.testing.assert_allclose(result["total"][i], reference["P_teacher"], rtol=1e-12)
        unbatched = decompose_power(rho[i], neutral[i], float(amplitude[i]), 50.)
        torch.testing.assert_close(unbatched["terms"], result["terms"][i])
        np.testing.assert_allclose(result["k"], reference["k_cMpc_inv"], rtol=1e-12)


def test_anticorrelated_cross_term_retains_sign_and_correct_coefficient():
    """A neutral field linear in minus-density has negative density cross power."""
    rng = np.random.default_rng(3)
    rho = torch.tensor(.8+.4*rng.random((8, 8, 8)), dtype=torch.float64)
    neutral = .5-.3*(rho-rho.mean())
    result = decompose_power(rho, neutral, 27., 50.)
    _, density_power, _ = shell_power(rho, 50.)
    torch.testing.assert_close(result["density_neutral_cross"], -.3*density_power)
    torch.testing.assert_close(result["neutral_power"], .09*density_power)
    torch.testing.assert_close(result["terms"][2],
                               2*27**2*neutral.mean()*rho.mean()*(-.3*density_power))
    assert torch.all(result["terms"][2] < 0)
    scale = torch.ones_like(result["terms"])
    assert component_loss(result["terms"], result["terms"], scale).item() == 0
    flipped = result["terms"].clone()
    flipped[2] *= -1
    assert component_loss(flipped, result["terms"], scale, indices=(2,)).item() > 0


def test_component_loss_gradient_matches_finite_difference():
    """Signed asinh errors propagate through the FFT and agree with a derivative."""
    rng = np.random.default_rng(73)
    rho = torch.tensor(.5+rng.random((5, 5, 5)), dtype=torch.float64)
    truth = torch.tensor(.2+.5*rng.random((5, 5, 5)), dtype=torch.float64)
    pattern = torch.tensor(rng.normal(size=(5, 5, 5)), dtype=torch.float64)
    target = decompose_power(rho, truth, 27., 50.)["terms"].detach()
    scale = target.abs().clamp_min(1.)

    def loss(delta):
        predicted = decompose_power(rho, truth+delta*pattern, 27., 50.)["terms"]
        return component_loss(predicted, target, scale, indices=(1, 2, 3, 4, 5))

    delta = torch.tensor(.01, dtype=torch.float64, requires_grad=True)
    loss(delta).backward()
    epsilon = 1e-6
    numeric = (loss(delta.detach()+epsilon)-loss(delta.detach()-epsilon))/(2*epsilon)
    torch.testing.assert_close(delta.grad, numeric, rtol=1e-6, atol=1e-8)
    assert torch.isfinite(delta.grad)
    assert delta.grad.abs() > 1e-8


@pytest.mark.parametrize("bad_scale", [0., -1., float("nan"), float("inf")])
def test_invalid_component_normalization_is_rejected(bad_scale):
    """Invalid normalizations cannot silently remove or explode components."""
    value = torch.zeros((2, 6, 3))
    with pytest.raises(ValueError, match="scale"):
        component_loss(value, value, torch.full((6, 3), bad_scale))


def test_loss_requires_six_channels_and_valid_selection():
    """Loss rejects an empty selection and scales with an extra batch axis."""
    value = torch.zeros((2, 6, 3))
    with pytest.raises(ValueError, match="indices"):
        component_loss(value, value, torch.ones(6, 3), indices=())
    with pytest.raises(ValueError, match="scale"):
        component_loss(value, value, torch.ones(2, 6, 3))
