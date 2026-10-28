"""Regression checks for phase weighting and low-signal spectral objectives."""

import numpy as np
import pytest
import torch

from research.es_pc_emulator.history_objective import (
    denominator_floor,
    relative_errors,
    stage_weights,
    weighted_rows,
)


def test_stages_have_equal_total_weight_and_duplication_does_not_change_score():
    """Repeating neutral snapshots cannot drown out a rarer EoR stage."""
    means = torch.tensor([.995, .995, .7, .005])
    weights, counts = stage_weights(means)
    assert counts == [1, 0, 0, 1, 0, 2]
    assert float(weights.mean()) == pytest.approx(1.)
    score = weighted_rows(torch.tensor([1., 1., 4., 7.]), weights)
    repeated = torch.tensor([.995]*8+[.7, .005])
    other, _ = stage_weights(repeated)
    assert float(score) == pytest.approx(4.)
    assert float(weighted_rows(torch.tensor([1.]*8+[4., 7.]), other)) == pytest.approx(4.)
    edges, counts = stage_weights(torch.tensor([0., .01, .1, .5, .9, .99, 1.]))
    assert counts == [1, 1, 1, 1, 1, 2]
    assert torch.isfinite(edges).all()


def test_floor_uses_only_supplied_training_powers_and_preserves_low_signal_gradients():
    """A tiny target has finite absolute-scale loss without ignoring the bin."""
    train = torch.tensor([[1., 4.], [3., 8.]], dtype=torch.float64)
    floor = denominator_floor(train, torch.ones(2), .01, 1e-8)
    torch.testing.assert_close(floor, .01*train.square().mean(0).sqrt())
    prediction = torch.tensor([[1e-5, 1e-6]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[0., 1e-12]], dtype=torch.float64)
    error = relative_errors(prediction, target, floor)
    error.mean().backward()
    assert torch.isfinite(error).all()
    assert torch.all(prediction.grad > 0)
    expected = (prediction.detach()-target).abs()/floor
    torch.testing.assert_close(error, expected)
    assert np.isfinite(error.detach().numpy()).all()


def test_valid_power_uses_exact_percentage_error_and_checks_invalid_arguments():
    """Above the declared floor, the objective is ordinary relative error."""
    pred, truth = torch.tensor([[3., 1.]]), torch.tensor([[2., 2.]])
    torch.testing.assert_close(relative_errors(pred, truth, torch.tensor([.01, .01])),
                               torch.tensor([[.5, .5]]))
    with pytest.raises(ValueError, match="Stage means"):
        stage_weights(torch.tensor([float("nan")]))
    with pytest.raises(ValueError, match="Stage means"):
        stage_weights(torch.tensor([1.1]))
    with pytest.raises(ValueError, match="training powers"):
        denominator_floor(truth, torch.ones(1), 0., 1e-8)
    with pytest.raises(ValueError, match="Matched powers"):
        relative_errors(pred, truth, torch.zeros(2))
