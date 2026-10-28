"""Stage-balanced supervision with a training-derived low-power denominator.

The regularized relative error retains weak-signal bins, but measures their
absolute residual relative to a fixed training scale. It is not a claim of
percentage accuracy below that scale. Teacher stages are supervision only.
"""

from __future__ import annotations

import numpy as np
import torch

STAGE_EDGES = (0., .01, .1, .5, .9, .99, 1.)
SELECTION_METRIC = "stage-balanced mean abs(P_pred-P_RT)/max(P_RT, training_RMS_floor)"


def stage_weights(means: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
    """Assign equal total weight to each occupied neutral-fraction stage.

    Parameters
    ----------
    means : torch.Tensor
        One volume-averaged teacher neutral fraction per snapshot.

    Returns
    -------
    weights : torch.Tensor
        Per-snapshot weights with mean one over the complete supplied split.
    counts : list of int
        Counts in the six fixed stages; the last stage includes one.
    """
    if (means.ndim != 1 or not means.numel() or not torch.isfinite(means).all()
            or torch.any((means < 0) | (means > 1))):
        raise ValueError("Stage means must be a nonempty finite vector in [0,1]")
    boundaries = torch.tensor(STAGE_EDGES[1:-1], device=means.device, dtype=means.dtype)
    indices = torch.bucketize(means.contiguous(), boundaries, right=True)
    counts = torch.bincount(indices, minlength=6)
    occupied = int((counts > 0).sum())
    weights = len(means)/(occupied*counts[indices].to(means.dtype))
    return weights, counts.tolist()


def weighted_rows(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Average per-row errors using globally normalized snapshot weights.

    Parameters
    ----------
    values : torch.Tensor
        Errors with snapshots on the leading axis; other axes are averaged.
    weights : torch.Tensor
        Weights normalized on the full split, not renormalized per minibatch.

    Returns
    -------
    torch.Tensor
        Scalar weighted mean. Minibatch estimates preserve global stage weights.
    """
    if values.ndim < 1 or weights.shape != (len(values),) or not len(values):
        raise ValueError("Weights must match a nonempty leading snapshot axis")
    return (values.reshape(len(values), -1).mean(-1)*weights).mean()


def denominator_floor(powers: torch.Tensor, weights: torch.Tensor,
                      fraction: float, absolute_floor: float) -> torch.Tensor:
    """Construct a fixed per-mode floor using training-label RMS powers only.

    Parameters
    ----------
    powers : torch.Tensor
        Training powers shaped ``[snapshot, mode]``.
    weights : torch.Tensor
        Training stage weights with mean one.
    fraction : float
        Positive fraction of each mode's stage-weighted RMS reference power.
    absolute_floor : float
        Positive minimum floor, in the same power units.

    Returns
    -------
    torch.Tensor
        Detached positive per-mode denominator floors.
    """
    if (powers.ndim != 2 or weights.shape != (len(powers),) or not len(powers)
            or not np.isfinite(fraction) or fraction <= 0
            or not np.isfinite(absolute_floor) or absolute_floor <= 0
            or not torch.isfinite(powers).all() or torch.any(powers < 0)
            or not torch.isfinite(weights).all() or torch.any(weights <= 0)):
        raise ValueError("Finite nonnegative training powers and positive weights/floors required")
    rms = (powers.detach().double().square()*weights.detach().double()[:, None]).mean(0).sqrt()
    return (fraction*rms).clamp_min(absolute_floor).to(powers.dtype)


def relative_errors(predicted: torch.Tensor, target: torch.Tensor,
                    floor: torch.Tensor) -> torch.Tensor:
    """Return absolute relative errors with a fixed low-signal denominator.

    Parameters
    ----------
    predicted, target : torch.Tensor
        Matched power spectra, with modes on the final axis.
    floor : torch.Tensor
        Strictly positive per-mode floors derived only from training labels.

    Returns
    -------
    torch.Tensor
        Unreduced dimensionless errors; weak-signal bins remain included.
    """
    floor = torch.as_tensor(floor, device=predicted.device, dtype=predicted.dtype).detach()
    if (predicted.shape != target.shape or floor.shape != predicted.shape[-1:]
            or not torch.isfinite(floor).all() or torch.any(floor <= 0)):
        raise ValueError("Matched powers and positive finite per-mode floors required")
    return (predicted-target).abs()/torch.maximum(target, floor)
