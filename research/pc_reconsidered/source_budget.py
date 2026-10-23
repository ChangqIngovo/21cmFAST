"""Additive source counts with an explicit spatial and temporal convention.

Native E-INTEGRAL ``nion`` is used as photons per local baryon in the ES
barrier, but its mean correction is a volume mean.  Multiplying that field by
the local atom capacity therefore does not generally preserve the native
unconditional source total.  ``build_mass_normalized_sources`` makes this
choice explicit: the spatial pattern comes from the local conditional field,
and each species' supplied *total count* is preserved in the mass measure.

This is a source adapter, not a photon transport or ionization solver.  ES
filters must all operate on the returned same additive count field.  A
cumulative source snapshot is never interpreted as newly emitted photons.
The temporal helpers either integrate positive instantaneous rates, or
explicitly impose a nondecreasing global cumulative history and distribute
its increment with an independently supplied instantaneous rate shape.
The latter is a calibrated source-history approximation, not the integral of
the unscaled rates.  Neither helper advects ionized gas or old photons.

All counts use one fixed unit, for example the native mean baryon content of
a cell; ``rho`` is the atom capacity in that unit.  No hydrogen/helium yield
conversion is inferred here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


class DecreasingCumulativeBudget(ValueError):
    """A declared cumulative emission history would remove emitted photons."""


@dataclass(frozen=True)
class SourceBudget:
    """Independent count fields; all arrays are read-only float64 copies."""

    by_species: dict[str, np.ndarray]
    total: np.ndarray
    diagnostics: dict


def _array(value, name):
    original = np.asarray(value)
    if np.iscomplexobj(original):
        raise ValueError(f"{name} must be real")
    result = np.array(original, dtype=np.float64, copy=True)
    if result.size == 0 or not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError(f"{name} must be nonempty, finite and nonnegative")
    return result


def _scalar(value, name):
    result = np.asarray(value)
    if result.ndim or np.iscomplexobj(result):
        raise ValueError(f"{name} must be a real scalar")
    result = np.longdouble(result)
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _float_field(value, name):
    value = np.asarray(value, dtype=np.longdouble)
    if not np.isfinite(value).all() or np.any(value > np.finfo(float).max):
        raise OverflowError(f"{name} is outside the float64 count range")
    result = np.asarray(value, dtype=np.float64)
    if np.any((value != 0) & (result == 0)):
        raise FloatingPointError(f"{name} contains nonzero counts below float64 range")
    result.flags.writeable = False
    return result


def _same_species(fields, totals):
    if not isinstance(fields, Mapping) or not fields:
        raise ValueError("A nonempty mapping of source species is required")
    if set(fields) != set(totals):
        raise ValueError("Source fields and target counts must have the same species")
    if any(not isinstance(name, str) or not name for name in fields):
        raise ValueError("Species names must be nonempty strings")


def _normalize(weights, target, name):
    """Normalize positive count weights without dividing by density again."""
    weights = np.asarray(weights, dtype=np.longdouble)
    total = np.sum(weights, dtype=np.longdouble)
    if not np.isfinite(total) or np.any(weights < 0):
        raise OverflowError(f"Invalid {name} count weights")
    if target > 0 and total == 0:
        raise ValueError(f"Positive {name} target has no source support")
    if target == 0:
        values = np.zeros_like(weights)
    else:
        # This order avoids overflow in target/total when the ratio is large.
        values = (weights / total) * target
    counts = _float_field(values, f"{name} normalized source")
    residual = target - np.sum(counts, dtype=np.longdouble)
    return counts, total, residual


def build_mass_normalized_sources(rho, eta_by_species, target_totals):
    """Construct S_i=A rho_i eta_i with sum(S_i)=target for each species.

    ``eta_by_species`` contains the native *smallest-scale* conditional source
    pattern before or after its volume mean correction; a uniform species
    factor cancels. ``target_totals`` are absolute counts, not per-cell means.
    For a Qin20 unconditional target use Ncell*nion_unconditional separately
    for ACGs and MCGs.  Density zero has zero photon support in this closure.

    These are cumulative coeval source maps when eta and targets are
    cumulative. They cannot be subtracted cellwise to obtain positive emission
    without an additional time/advection model; use the audit below.
    """
    rho = _array(rho, "Atom capacity")
    _same_species(eta_by_species, target_totals)
    by_species, diagnostic = {}, {}
    for name, value in eta_by_species.items():
        eta = _array(value, f"{name} photons per local baryon")
        if eta.shape != rho.shape:
            raise ValueError(f"{name} source and atom-capacity shapes differ")
        target = _scalar(target_totals[name], f"{name} target count")
        weights = rho.astype(np.longdouble) * eta.astype(np.longdouble)
        counts, raw_mass, residual = _normalize(weights, target, name)
        raw_volume = np.sum(eta, dtype=np.longdouble)
        # This is the photon count implied by the native volume normalization,
        # when the same eta is interpreted as a per-local-baryon ratio.
        native_count = target * raw_mass / raw_volume if raw_volume else 0
        by_species[name] = counts
        diagnostic[name] = {
            "target_count": float(target),
            "raw_ratio_sum": float(raw_volume),
            "raw_mass_weighted_count": float(raw_mass),
            "raw_density_covariance_sum": float(raw_mass - raw_volume),
            "count_implied_by_volume_normalization": float(native_count),
            "volume_normalization_count_excess": float(native_count - target),
            "normalization_factor": float(target / raw_mass) if raw_mass else None,
            "delivered_count": float(np.sum(counts, dtype=np.longdouble)),
            "roundoff_residual": float(residual),
        }
    total = _float_field(sum((x.astype(np.longdouble) for x in by_species.values()),
                            np.zeros(rho.shape, dtype=np.longdouble)), "Total source")
    return SourceBudget(by_species, total, {
        "convention": "species mass-normalized local conditional source",
        "mean_atom_capacity": float(np.mean(rho, dtype=np.longdouble)),
        "species": diagnostic,
        "delivered_total": float(np.sum(total, dtype=np.longdouble)),
        "target_total": float(sum(_scalar(x, "Target") for x in target_totals.values())),
    })


def audit_cumulative_map_change(previous, current):
    """Return the signed change; negative entries are retained and reported.

    Moving or renormalizing a cumulative Eulerian source map can give negative
    entries even when global emission increases.  This diagnostic makes no
    claim that its positive part is emitted radiation.  Clipping the negative
    part would introduce the reported ``negative_total`` as extra photons.
    """
    old = _array(previous, "Previous cumulative map")
    new = _array(current, "Current cumulative map")
    if old.shape != new.shape:
        raise ValueError("Cumulative maps must share a grid")
    delta = new.astype(np.longdouble) - old.astype(np.longdouble)
    positive = np.sum(delta[delta > 0], dtype=np.longdouble)
    negative = -np.sum(delta[delta < 0], dtype=np.longdouble)
    return delta, {
        "net_change": float(np.sum(delta, dtype=np.longdouble)),
        "positive_total": float(positive),
        "negative_total": float(negative),
        "negative_cells": int(np.count_nonzero(delta < 0)),
        "minimum_change": float(np.min(delta)),
        "can_be_nonnegative_emission_without_remapping": bool(np.all(delta >= 0)),
        "extra_photons_if_negative_changes_are_clipped": float(negative),
    }


def integrate_rate_step(left_rate, right_rate, dt_seconds):
    """Exact integral of the declared linear-in-time endpoint rate model.

    Rates already carry photons/cell/time; no density factor is applied.
    Endpoint quadrature is an approximation to a nonlinear physical history.
    For a source switch, supply the appropriate one-sided endpoint rates.
    """
    left = _array(left_rate, "Left instantaneous rate")
    right = _array(right_rate, "Right instantaneous rate")
    if left.shape != right.shape:
        raise ValueError("Endpoint rates must share a grid")
    duration = _scalar(dt_seconds, "Time interval in seconds")
    result = duration * (left.astype(np.longdouble) / 2 +
                         right.astype(np.longdouble) / 2)
    return _float_field(result, "Emitted photon increment")


def allocate_global_history_step(previous_totals, current_totals,
                                 integrated_rate_shapes):
    """Place fresh global Delta Q using positive instantaneous emission weights.

    The caller chooses to retain an externally specified cumulative history,
    e.g. native Qin20 unconditional nion.  Endpoint rate integrals supply only
    its spatial distribution.  The required rate rescaling is returned so this
    source-history approximation is observable.  A decreasing species budget
    is rejected, even when another species increases; no emitted photons may
    be removed or transferred between populations.  Old source maps are not
    subtracted, rebuilt, or assigned as new emission.
    """
    _same_species(integrated_rate_shapes, current_totals)
    _same_species(integrated_rate_shapes, previous_totals)
    by_species, diagnostic = {}, {}
    shape = None
    for name, value in integrated_rate_shapes.items():
        old = _scalar(previous_totals[name], f"{name} previous cumulative count")
        new = _scalar(current_totals[name], f"{name} current cumulative count")
        if new < old:
            raise DecreasingCumulativeBudget(
                f"{name} cumulative photons decrease by {float(old-new):.17g}; "
                "a cumulative source proxy cannot be used as positive emission")
        weights = _array(value, f"{name} integrated instantaneous rate shape")
        if shape is not None and weights.shape != shape:
            raise ValueError("All source species must share a grid")
        shape = weights.shape
        increment = new - old
        counts, raw, residual = _normalize(weights, increment, name)
        by_species[name] = counts
        diagnostic[name] = {
            "previous_cumulative_count": float(old),
            "current_cumulative_count": float(new),
            "emitted_increment": float(increment),
            "unscaled_rate_integral": float(raw),
            "rate_rescaling_factor": float(increment / raw) if raw else None,
            "roundoff_residual": float(residual),
        }
    total = _float_field(sum((x.astype(np.longdouble) for x in by_species.values()),
                            np.zeros(shape, dtype=np.longdouble)), "Total emission increment")
    return SourceBudget(by_species, total, {
        "convention": "imposed monotone global history; instantaneous rate spatial weights",
        "species": diagnostic,
        "emitted_total": float(np.sum(total, dtype=np.longdouble)),
    })
