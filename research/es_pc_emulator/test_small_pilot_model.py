"""Small scientific software checks; no native simulations or training runs."""

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from research.es_pc_emulator.small_pilot_model import (  # noqa: E402
    FieldNet,
    build_inputs,
    power_loss,
    shell_power,
    thermal_targets,
)
from research.es_pc_emulator.test_physical_pilot_training import make_case  # noqa: E402
from research.pc_field_emulator.evaluation import shell_spectra  # noqa: E402


def test_history_is_causal_and_has_no_rt_inputs():
    """Future sources and RT labels cannot affect the current input."""
    case = make_case()
    case["photons"] = np.cumsum(case["photons"], axis=0)
    reference, names = build_inputs(case, 0, history_steps=3)
    changed = copy.deepcopy(case)
    for name in ("xhi_rt", "temperature_rt", "xheii_rt", "xheiii_rt"):
        changed.pop(name)
    for name in ("photons", "emissivity", "emission_time_mean", "radius_es_cMpc", "xhi_es"):
        changed[name][1] = 999
    changed["metadata"]["theta"][:4] = [200, 300, 400, 500]
    actual, actual_names = build_inputs(changed, 0, history_steps=3)
    np.testing.assert_array_equal(actual, reference)
    assert names == actual_names
    assert len(reference) == 34 + 4 * 3
    assert "xhi_es:smooth_cMpc=6.25" in names
    np.testing.assert_allclose(reference[names.index("log1p_density:global_mean")],
                               np.log1p(case["density"]).mean(), rtol=1e-6)
    assert actual[names.index("history_0:available")].min() == 1
    assert actual[names.index("history_1:available")].max() == 0
    last, _ = build_inputs(case, 1, history_steps=3)
    np.testing.assert_allclose(last[names.index("history_0:log1p_interval_photons")],
                               np.log1p(case["photons"][1] - case["photons"][0]), rtol=1e-6)


@pytest.mark.parametrize("n", [5, 8])
def test_torch_spectra_match_existing_estimator(n):
    """Even and odd grids retain exact mode weighting and normalization."""
    value = np.random.default_rng(73).normal(size=(n, n, n)) + 21
    expected = shell_spectra({"teacher": value}, 50.)
    k, power, valid = shell_power(torch.tensor(value, dtype=torch.float64), 50.)
    np.testing.assert_allclose(k.numpy(), expected["k_cMpc_inv"], rtol=1e-12)
    np.testing.assert_allclose(power.numpy(), expected["P_teacher"], rtol=1e-12)
    assert valid.all()
    _, zeros, mask = shell_power(torch.ones((n, n, n), dtype=torch.float64), 50.)
    assert torch.count_nonzero(zeros) == 0
    assert not mask.any()


def test_power_loss_has_gradient_through_brightness_field():
    """The spectral objective backpropagates into the physical field."""
    rng = np.random.default_rng(4)
    rho = torch.tensor(.5 + rng.random((4, 4, 4)), dtype=torch.float64)
    xhi = torch.tensor(.2 + .5 * rng.random((4, 4, 4)), dtype=torch.float64, requires_grad=True)
    _, predicted, _ = shell_power(27 * rho * xhi, 50.)
    _, target, _ = shell_power(27 * rho * xhi.detach() * .9, 50.)
    power_loss(predicted, target).backward()
    assert torch.isfinite(xhi.grad).all()
    assert torch.count_nonzero(xhi.grad) > 0


def test_periodic_network_and_auxiliary_targets():
    """The network respects periodic translations and the helium simplex."""
    torch.manual_seed(15)
    case = make_case()
    inputs, _ = build_inputs(case, 0)
    model = FieldNet(len(inputs), width=3, wide=True, auxiliary=True)
    inputs = torch.from_numpy(inputs)[None]
    baseline = torch.tensor(case["xhi_es"][0], dtype=torch.float32)[None]
    original, aux = model(inputs, baseline)
    torch.testing.assert_close(original, baseline, rtol=0, atol=0)
    # Exercise a nonzero field residual, rather than only the identity baseline.
    with torch.no_grad():
        model.residual.weight.fill_(.1)
    original, aux = model(inputs, baseline)
    shifted, shifted_aux = model(torch.roll(inputs, 1, -1), torch.roll(baseline, 1, -1))
    torch.testing.assert_close(shifted, torch.roll(original, 1, -1))
    torch.testing.assert_close(shifted_aux, torch.roll(aux, 1, -1))
    assert ((original >= 0) & (original <= 1)).all()
    torch.testing.assert_close(aux[:, 1:].sum(dim=1), torch.ones_like(original))
    target = thermal_targets(case, 0)
    assert target.shape == (4, 4, 4, 4)
    np.testing.assert_allclose(target[0], 4.)
    np.testing.assert_allclose(target[1:].sum(axis=0), 1., rtol=1e-6)


def test_zero_head_can_leave_a_fully_neutral_prior():
    """Zero initialization does not trap saturated ES priors."""
    model = FieldNet(2, width=2)
    baseline = torch.ones((1, 2, 2, 2))
    prediction, _ = model(torch.ones((1, 2, 2, 2, 2)), baseline)
    (prediction - .5).square().mean().backward()
    assert model.residual.bias.grad.abs().item() > 0


@pytest.mark.parametrize(("prior", "bias", "target"), [(0., -2., .01), (1., 2., .99)])
def test_logit_residual_recovers_from_wrong_side_at_es_boundary(prior, bias, target):
    """A nonzero head must retain the corrective gradient at either ES endpoint."""
    model = FieldNet(2, width=2, output_transform="logit", prior_epsilon=1e-4)
    with torch.no_grad():
        model.residual.bias.fill_(bias)
    baseline = torch.full((1, 2, 2, 2), prior)
    prediction, _ = model(torch.ones((1, 2, 2, 2, 2)), baseline)
    (prediction - target).square().mean().backward()
    gradient = model.residual.bias.grad.item()
    assert np.isfinite(gradient)
    assert abs(gradient) > 0
    assert gradient < 0 if prior == 0 else gradient > 0
    assert ((prediction > 0) & (prediction < 1)).all()


def test_logit_residual_prior_regularization_is_not_an_output_floor():
    """Tiny residual neutral fractions remain representable below prior epsilon."""
    model = FieldNet(2, width=2, output_transform="logit", prior_epsilon=1e-4)
    inputs = torch.ones((1, 2, 2, 2, 2))
    baseline = torch.zeros((1, 2, 2, 2))
    prediction, _ = model(inputs, baseline)
    torch.testing.assert_close(prediction, torch.full_like(baseline, 1e-4), rtol=1e-6, atol=0)
    with torch.no_grad():
        model.residual.bias.fill_(-3.)
    prediction, _ = model(inputs, baseline)
    assert prediction.min() > 0
    assert prediction.max() < 1e-4


@pytest.mark.parametrize("options", [{"output_transform": "unknown"},
    {"prior_epsilon": 0.}, {"prior_epsilon": .5}, {"prior_epsilon": float("nan")},
    {"prior_epsilon": 1e-20}])
def test_output_transform_rejects_invalid_configuration(options):
    """Invalid transform settings fail before fitting or inference."""
    with pytest.raises(ValueError, match=r"output transform|Prior epsilon"):
        FieldNet(2, **options)
