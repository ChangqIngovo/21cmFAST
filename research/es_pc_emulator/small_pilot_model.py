"""Causal physical inputs, periodic field networks and differentiable P21.

RT thermal/helium fields are auxiliary training targets, never model inputs.
Spectra use the existing pilot's saturated-Ts, no-RSD signal convention.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch import nn

from research.pc_field_emulator.evaluation import _shell_geometry


def _field(case, name, snapshot, shape):
    value = np.asarray(case[name], dtype=np.float64)
    value = value[snapshot] if value.ndim == 4 else value
    if value.shape != shape or not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError(f"Invalid nonnegative field: {name}")
    return value


def build_inputs(case, snapshot, history_steps=0):
    """Return [channel,x,y,z] inputs available at this snapshot or earlier.

    Each historical interval contributes its integrated photon count, endpoint
    emissivity, endpoint redshift and an availability mask. The first stored
    interval starts with zero cumulative photons, as in the pilot generator.
    The six baseline spatial fields retain the previous pilot's three
    Gaussian-smoothed scales and global means in every ablation arm.
    """
    if type(history_steps) is not int or history_steps < 0:
        raise ValueError("history_steps must be a nonnegative integer")
    redshifts = np.asarray(case["redshifts"], dtype=float)
    if not 0 <= snapshot < len(redshifts):
        raise ValueError("Snapshot is outside the stored history")
    xhi = np.asarray(case["xhi_es"][snapshot], dtype=np.float64)
    if (xhi.ndim != 3 or len(set(xhi.shape)) != 1 or xhi.shape[0] < 2
            or not np.isfinite(xhi).all() or np.any((xhi < 0) | (xhi > 1))):
        raise ValueError("Invalid cubic ES prior")
    shape = xhi.shape
    channels = {"xhi_es": xhi}
    for name in ("radius_es_cMpc", "density", "photons", "emissivity", "emission_time_mean"):
        channels[f"log1p_{name}"] = np.log1p(_field(case, name, snapshot, shape))
    box = float(case["metadata"]["box_len_cMpc"])
    if not np.isfinite(box) or box <= 0:
        raise ValueError("Invalid box length")
    for name, value in tuple(channels.items()):
        for radius in (6.25, 12.5, 25.):
            channels[f"{name}:smooth_cMpc={radius:g}"] = gaussian_filter(
                value, radius / (box / shape[0]), mode="wrap")
        channels[f"{name}:global_mean"] = np.full(shape, value.mean())
    parameters = dict(zip(case["metadata"]["parameter_names"],
                          case["metadata"]["theta"], strict=True))
    for name, value in (("redshift", redshifts[snapshot]),
                        ("log1p_dx_cMpc", np.log1p(box / shape[0])),
                        ("log10_Tbb_K", parameters["log10_Tbb_K"]),
                        ("log10_clumping", parameters["log10_clumping"])):
        channels[name] = np.full(shape, float(value))
    for lag in range(history_steps):
        index = snapshot - lag
        if index >= 0:
            cumulative = _field(case, "photons", index, shape)
            previous = _field(case, "photons", index - 1, shape) if index else np.zeros(shape)
            interval = cumulative - previous
            tolerance = 1e-7 * max(float(np.max(cumulative)), 1.)
            if np.min(interval) < -tolerance:
                raise ValueError("Cumulative photons decrease in time")
            photons = np.log1p(np.maximum(interval, 0))
            emissivity = np.log1p(_field(case, "emissivity", index, shape))
            redshift, available = float(redshifts[index]), 1.
        else:
            photons = emissivity = np.zeros(shape)
            redshift = available = 0.
        channels[f"history_{lag}:log1p_interval_photons"] = photons
        channels[f"history_{lag}:log1p_emissivity"] = emissivity
        channels[f"history_{lag}:redshift"] = np.full(shape, redshift)
        channels[f"history_{lag}:available"] = np.full(shape, available)
    result = np.stack(list(channels.values())).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite physical inputs")
    return result, list(channels)


def thermal_targets(case, snapshot):
    """Auxiliary labels [log10(TK/K), xHeI, xHeII, xHeIII]."""
    shape = np.asarray(case["xhi_es"][snapshot]).shape
    temperature = _field(case, "temperature_rt", snapshot, shape)
    heii = _field(case, "xheii_rt", snapshot, shape)
    heiii = _field(case, "xheiii_rt", snapshot, shape)
    if np.any(temperature <= 0) or np.any(heii + heiii > 1 + 1e-6):
        raise ValueError("Invalid thermal or helium target")
    helium = np.stack((np.maximum(1 - heii - heiii, 0), heii, heiii))
    helium /= helium.sum(axis=0, keepdims=True)
    return np.concatenate((np.log10(temperature)[None], helium)).astype(np.float32)


class PeriodicConv3d(nn.Module):
    """Periodic convolution, including dilation larger than tiny test grids."""

    def __init__(self, incoming, outgoing, dilation):
        super().__init__()
        self.padding = dilation
        self.conv = nn.Conv3d(incoming, outgoing, 3, dilation=dilation)

    def forward(self, value):
        """Pad each spatial axis periodically and apply the convolution."""
        for axis in (-3, -2, -1):
            n = value.shape[axis]
            indices = torch.arange(-self.padding, n + self.padding, device=value.device) % n
            value = value.index_select(axis, indices)
        return self.conv(value)


class FieldNet(nn.Module):
    """Bounded ES residual with optional gas-temperature/helium supervision.

    The local arm uses one periodic 3-D convolution followed by two pointwise
    layers (a 3-cell receptive field). The wide arm uses three periodic 3-D
    convolutions with dilations 1,2,4 (a nominal 15-cell receptive field).
    """

    def __init__(self, in_channels, width=16, wide=False, auxiliary=False):
        super().__init__()
        if in_channels < 1 or width < 1:
            raise ValueError("Channel counts must be positive")
        layers = []
        for dilation in (1, 2, 4):
            layers.extend((PeriodicConv3d(in_channels, width, dilation) if wide or dilation == 1
                           else nn.Conv3d(in_channels, width, 1), nn.SiLU()))
            in_channels = width
        self.trunk = nn.Sequential(*layers)
        self.residual = nn.Conv3d(width, 1, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        self.auxiliary = nn.Conv3d(width, 4, 1) if auxiliary else None

    def forward(self, inputs, xhi_es):
        """Predict neutral fraction and optional auxiliary states."""
        if inputs.ndim != 5 or xhi_es.shape != (inputs.shape[0], *inputs.shape[-3:]):
            raise ValueError("Expected inputs [B,C,N,N,N] and ES [B,N,N,N]")
        hidden = self.trunk(inputs)
        residual = torch.tanh(self.residual(hidden)[:, 0])
        # Both clamp derivatives are one at zero: the zero-initialized head
        # can learn either direction even for a fully neutral/ionized prior.
        xhi = (xhi_es + (1 - xhi_es) * residual.clamp_min(0)
               + xhi_es * residual.clamp_max(0))
        aux = None
        if self.auxiliary is not None:
            raw = self.auxiliary(hidden)
            aux = torch.cat((raw[:, :1], torch.softmax(raw[:, 1:], dim=1)), dim=1)
        return xhi, aux


@lru_cache(maxsize=16)
def _geometry(n, box_len):
    return _shell_geometry(n, box_len)


def shell_power(field, box_len):
    """Differentiable power in the exact shells of evaluation.shell_spectra.

    Accepts [...,N,N,N]; returns k[S], P[...,S], valid[...,S]. P has units of
    field-unit squared times cMpc cubed. Valid masks numerically zero power;
    it does not silently remove bins from training losses.
    """
    n = field.shape[-1]
    if (field.ndim < 3 or tuple(field.shape[-3:]) != (n, n, n) or n < 2
            or not np.isfinite(box_len) or box_len <= 0):
        raise ValueError("A cubic field and positive finite box length are required")
    if not field.is_floating_point():
        raise ValueError("Spectrum inputs must be floating point")
    eligible, labels, weights, occupied, k = _geometry(n, float(box_len))
    device, dtype = field.device, field.dtype
    selected = torch.as_tensor(eligible, device=device)
    labels = torch.as_tensor(labels, device=device)
    weights = torch.as_tensor(weights, device=device, dtype=dtype)
    occupied = torch.as_tensor(occupied, device=device)
    transformed = torch.fft.rfftn(field - field.mean(dim=(-3, -2, -1), keepdim=True),
                                  dim=(-3, -2, -1))[..., selected]
    values = weights * transformed.abs().square()
    shell_count = int(labels.max()) + 1
    powers = torch.zeros((*values.shape[:-1], shell_count), device=device, dtype=dtype)
    powers = powers.scatter_add(-1, labels.expand_as(values), values)
    counts = torch.zeros(shell_count, device=device, dtype=dtype).scatter_add(0, labels, weights)
    power = powers[..., occupied] / counts[occupied] * (box_len**3 / n**6)
    floor = torch.maximum(power.amax(dim=-1, keepdim=True) * 1e-12,
                          np.finfo(float).eps**2 * box_len**3
                          * field.square().mean(dim=(-3, -2, -1)).clamp_min(1)[..., None])
    return (torch.as_tensor(k, device=device, dtype=dtype), power,
            torch.isfinite(power) & (power > floor))


def power_loss(predicted, target, floor=1e-8):
    """Mean squared natural-log power error with an explicit additive floor."""
    if predicted.shape != target.shape or not np.isfinite(floor) or floor <= 0:
        raise ValueError("Matched powers and a positive finite floor are required")
    return (torch.log(predicted + floor) - torch.log(target + floor)).square().mean()
