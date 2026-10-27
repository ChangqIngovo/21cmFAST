"""Exact signed decomposition of saturated-temperature, no-RSD P21.

The six contributions retain their signs. Their large cancellations make
unsigned/logarithmic cross-spectrum objectives inappropriate.
"""

from __future__ import annotations

import numpy as np
import torch

from research.es_pc_emulator.small_pilot_model import _geometry

TERM_NAMES = (
    "density_auto", "neutral_auto", "density_neutral_cross",
    "product_auto", "density_product_cross", "neutral_product_cross",
)


def decompose_power(density, xhi, amplitude, box_len):
    """Return exact six signed shell powers and their sum.

    Inputs are matched tensors [...,N,N,N]. ``amplitude`` is a positive scalar
    or a tensor matching the leading dimensions. Terms have shape [...,6,S]
    and order PDD, PXX, 2PDX, PNN, 2PDN, 2PXN, where
    D=A*xbar*drho, X=A*rhobar*dx, N=A*centered(drho*dx).
    ``total`` is the sum, so near complete cancellation its floating-point
    precision is less robust than an FFT of the brightness field itself.
    """
    if (not isinstance(density, torch.Tensor) or not isinstance(xhi, torch.Tensor)
            or density.shape != xhi.shape or density.ndim < 3
            or not density.is_floating_point() or not xhi.is_floating_point()
            or density.dtype != xhi.dtype or density.device != xhi.device):
        raise ValueError("Density and xHI require matched floating-point tensors")
    n = xhi.shape[-1]
    if (n < 2 or tuple(xhi.shape[-3:]) != (n, n, n)
            or not np.isfinite(box_len) or box_len <= 0):
        raise ValueError("Cubic grids and a positive finite box length are required")
    if (not torch.isfinite(density).all() or not torch.isfinite(xhi).all()
            or torch.any(density < 0) or torch.any((xhi < 0) | (xhi > 1))):
        raise ValueError("Density must be nonnegative and xHI must be in [0,1]")
    amplitude = torch.as_tensor(amplitude, dtype=xhi.dtype, device=xhi.device)
    if (amplitude.shape not in (torch.Size(), xhi.shape[:-3])
            or not torch.isfinite(amplitude).all() or torch.any(amplitude <= 0)):
        raise ValueError("Amplitude must be positive and scalar or match batch dimensions")
    axes = (-3, -2, -1)
    rm = density.mean(dim=axes, keepdim=True)
    xm = xhi.mean(dim=axes, keepdim=True)
    dr, dx = density-rm, xhi-xm
    product = dr*dx
    fields = torch.stack((dr, dx, product-product.mean(dim=axes, keepdim=True)), dim=-4)
    eligible, labels, weights, occupied, k = _geometry(n, float(box_len))
    selected = torch.as_tensor(eligible, device=xhi.device)
    labels = torch.as_tensor(labels, device=xhi.device)
    weights = torch.as_tensor(weights, dtype=xhi.dtype, device=xhi.device)
    occupied = torch.as_tensor(occupied, device=xhi.device)
    fft = torch.fft.rfftn(fields, dim=axes)[..., selected]
    pairs = ((0, 0), (1, 1), (0, 1), (2, 2), (0, 2), (1, 2))
    products = torch.stack([(fft[..., i, :] * fft[..., j, :].conj()).real
                            for i, j in pairs], dim=-2) * weights
    shell_count = int(labels.max())+1
    sums = torch.zeros((*products.shape[:-1], shell_count), dtype=xhi.dtype, device=xhi.device)
    sums = sums.scatter_add(-1, labels.expand_as(products), products)
    counts = torch.zeros(shell_count, dtype=xhi.dtype, device=xhi.device).scatter_add(0, labels, weights)
    spectra = sums[..., occupied] / counts[occupied] * (box_len**3 / n**6)
    rm, xm = rm[..., 0, 0, 0], xm[..., 0, 0, 0]
    factors = torch.stack((xm.square(), rm.square(), 2*xm*rm,
                           torch.ones_like(xm), 2*xm, 2*rm), dim=-1)
    terms = spectra * factors[..., None] * amplitude.square()[..., None, None]
    return {"k": torch.as_tensor(k, dtype=xhi.dtype, device=xhi.device),
            "terms": terms, "total": terms.sum(dim=-2),
            "neutral_power": spectra[..., 1, :],
            "density_neutral_cross": spectra[..., 2, :]}


def component_loss(predicted, target, scale, indices=None):
    """Compare selected signed spectra using fixed, train-derived asinh scales.

    ``scale`` must be [6,S], strictly positive, and is detached. Zero-valued
    and negative cross spectra are retained; no logarithm or absolute-value
    transform discards their signs.
    """
    if (predicted.shape != target.shape or predicted.ndim < 2
            or predicted.shape[-2] != 6):
        raise ValueError("Matched component arrays [...,6,S] are required")
    scale = torch.as_tensor(scale, dtype=predicted.dtype, device=predicted.device).detach()
    if (scale.shape != predicted.shape[-2:] or not torch.isfinite(scale).all()
            or torch.any(scale <= 0)):
        raise ValueError("Component scale must be finite, positive and shaped [6,S]")
    if not torch.isfinite(predicted).all() or not torch.isfinite(target).all():
        raise ValueError("Component spectra must be finite")
    if indices is None:
        indices = tuple(range(6))
    indices = tuple(indices)
    if (not indices or len(set(indices)) != len(indices)
            or any(type(i) is not int or not 0 <= i < 6 for i in indices)):
        raise ValueError("Component indices must be distinct integers from zero to five")
    index = torch.tensor(indices, dtype=torch.long, device=predicted.device)
    residual = (torch.asinh(predicted / scale) - torch.asinh(target / scale))
    return residual.index_select(-2, index).square().mean()
