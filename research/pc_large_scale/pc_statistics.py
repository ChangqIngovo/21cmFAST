"""Count-preserving resolution tests and the Choudhury & Paranjape HI statistic.

P_HI is the power of rho_HI / mean(rho_HI), not x_HI and not unnormalised
brightness. Distances here are h^-1 cMpc, so k is h cMpc^-1. The ES comparator
filters one additive source and capacity field using the native sharp-k window;
it is separately labelled, not substituted into native 21cmFAST caches.
"""
from __future__ import annotations

import numpy as np
from scipy import fft


def total(array):
    return float(np.sum(array, dtype=np.longdouble))


def input_band_statistics(source, capacity, box_hMpc, band):
    """Expose input smoothing/aliasing before attributing changes to ionization."""
    n = source.shape[0]
    fs = fft.rfftn(source/np.mean(source)-1)
    fm = fft.rfftn(capacity/np.mean(capacity)-1)
    k = 2*np.pi*fft.fftfreq(n, d=box_hMpc/n)
    kz = 2*np.pi*fft.rfftfreq(n, d=box_hMpc/n)
    km = np.sqrt(k[:,None,None]**2+k[None,:,None]**2+kz[None,None,:]**2)
    w = np.full(kz.size, 2.)
    w[0] = 1
    if n % 2 == 0:
        w[-1] = 1
    weights = np.broadcast_to(w, km.shape)
    mask = (km >= band[0]) & (km < band[1])
    modes = int(weights[mask].sum())
    if not modes:
        raise ValueError('Empty input spectrum band')
    norm = box_hMpc**3/n**6/modes
    ps = float(np.sum(abs(fs[mask])**2*weights[mask])*norm)
    pm = float(np.sum(abs(fm[mask])**2*weights[mask])*norm)
    cross = float(np.sum((fs[mask]*fm[mask].conj()).real*weights[mask])*norm)
    return {'P_source': ps, 'P_matter': pm, 'P_source_matter_cross': cross,
            'source_bias_squared': ps/pm if pm else None, 'full_fft_modes': modes}


def coarsen_counts(array, n):
    a = np.asarray(array, dtype=np.float64)
    if a.ndim != 3 or len(set(a.shape)) != 1 or n < 2 or a.shape[0] % n:
        raise ValueError("Coarse grid must divide the original cubic grid")
    f = a.shape[0] // n
    # Counts retain the original fine-cell unit at every resolution.
    return np.ascontiguousarray(a.reshape(n, f, n, f, n, f).sum(axis=(1, 3, 5)))


def es_threshold(source, capacity, max_radius_cells, workers=1):
    """Double-precision native-window centre-cell ES on fixed additive fields.

The unfiltered cell is always available; filtered radii grow by 1.1 from the
equal-volume cell radius, as in the native L-INTEGRAL implementation. Negative
sharp-k lobes in sources are zeroed; absorber floors are recorded explicitly.
"""
    s, c = np.asarray(source, float), np.asarray(capacity, float)
    if s.shape != c.shape or s.ndim != 3 or len(set(s.shape)) != 1:
        raise ValueError("Matching cubic count fields required")
    if not np.isfinite(s).all() or not np.isfinite(c).all() or (s < 0).any() or (c <= 0).any():
        raise ValueError("Finite nonnegative sources and positive capacities required")
    if not np.isfinite(max_radius_cells) or max_radius_cells <= 0:
        raise ValueError("Positive finite filter radius required")
    n = s.shape[0]
    k = 2*np.pi*fft.fftfreq(n)
    kz = 2*np.pi*fft.rfftfreq(n)
    kmag = np.sqrt(k[:, None, None]**2+k[None, :, None]**2+kz[None, None, :]**2)
    sf, cf = fft.rfftn(s, workers=workers), fft.rfftn(c, workers=workers)
    threshold = np.divide(c, s, out=np.full(s.shape, np.inf), where=s > 0)
    radius = (3/(4*np.pi))**(1/3)*1.1
    radii = []
    while radius <= max_radius_cells*(1+1e-12):
        radii.append(radius)
        radius *= 1.1
    floor_count = 0
    for radius in radii:
        window = kmag*radius*0.413566994 <= 1  # src/filtering.c sharp_k_filter
        sr = fft.irfftn(sf*window, s=s.shape, workers=workers)
        cr = fft.irfftn(cf*window, s=c.shape, workers=workers)
        floor = c.mean()*1e-7
        floor_count += int(np.count_nonzero(cr < floor))
        cr = np.maximum(cr, floor)
        threshold = np.minimum(threshold, np.divide(
            cr, sr, out=np.full(s.shape, np.inf), where=sr > 0))
    return threshold, {"window": "sharp-k, kR*0.413566994 <= 1",
                       "filtered_radii_cells": radii, "absorber_floor_count": floor_count,
                       "scope": "source-matched ES; not a native IonizedBox call"}


def es_map(source, capacity, threshold, amplitude):
    if not np.isfinite(amplitude) or amplitude < 0:
        raise ValueError("Amplitude must be nonnegative and finite")
    a = np.minimum(source*amplitude, capacity)
    a[amplitude > threshold] = capacity[amplitude > threshold]
    return a


def match_mass(source, capacity, threshold, target, tolerance=2e-4):
    """Closest actual ES map. Never interpolate across a barrier discontinuity."""
    if not 0 < target < 1 or not np.any(source > 0):
        raise ValueError("Nonzero sources and target mass fraction in (0,1) required")
    C = total(capacity)
    lo, hi = 0., target*C/total(source)
    evaluate = lambda amp: total(es_map(source, capacity, threshold, amp))/C
    while evaluate(hi) < target:
        hi *= 2
        if not np.isfinite(hi):
            raise RuntimeError("Cannot bracket requested ionized mass")
    best = (float('inf'), hi)
    for _ in range(80):
        mid = (lo+hi)/2
        value = evaluate(mid)
        error = abs(value-target)
        if error < best[0]:
            best = error, mid
        if error <= tolerance/4 or mid == lo or mid == hi:
            break
        if value < target:
            lo = mid
        else:
            hi = mid
    for amp in (lo, hi):
        error = abs(evaluate(amp)-target)
        if error < best[0]:
            best = error, amp
    absorbed = es_map(source, capacity, threshold, best[1])
    return absorbed, {"amplitude": best[1], "target_Q_mass": target,
                      "actual_Q_mass": total(absorbed)/C, "absolute_error": best[0],
                      "matched": bool(best[0] <= tolerance),
                      "bracket_amplitudes": [lo, hi],
                      "bracket_Q_mass": [evaluate(lo), evaluate(hi)],
                      "interpolated_fields": False}


def spectrum(capacity, absorbed, box_hMpc, edges, band, redshift, workers=1):
    c, a = np.asarray(capacity, float), np.asarray(absorbed, float)
    if c.shape != a.shape or c.ndim != 3 or len(set(c.shape)) != 1:
        raise ValueError("Matching cubic fields required")
    if (not np.isfinite(c).all() or not np.isfinite(a).all() or (c <= 0).any()
            or (a < -1e-10*c).any() or (a > (1+1e-10)*c).any()):
        raise ValueError("Invalid density or absorption field")
    n = c.shape[0]
    rho = c/c.mean()
    hi = (c-a)/c.mean()
    neutral_mean = hi.mean()
    if neutral_mean <= 0:
        raise ValueError("HI bias is undefined in a fully ionized box")
    fh = fft.rfftn(hi/neutral_mean-1, workers=workers)
    fm = fft.rfftn(rho-1, workers=workers)
    norm = box_hMpc**3/n**6
    powers = {"P_HI": abs(fh)**2*norm, "P_matter": abs(fm)**2*norm,
              "P_cross": (fh*fm.conj()).real*norm}
    k = 2*np.pi*fft.fftfreq(n, d=box_hMpc/n)
    kz = 2*np.pi*fft.rfftfreq(n, d=box_hMpc/n)
    kmag = np.sqrt(k[:, None, None]**2+k[None, :, None]**2+kz[None, None, :]**2)
    wz = np.full(kz.size, 2.)
    wz[0] = 1
    if n % 2 == 0:
        wz[-1] = 1
    weights = np.broadcast_to(wz, kmag.shape)
    nyquist = np.pi*n/box_hMpc
    keep = (kmag > 0) & (kmag <= nyquist*(1+1e-12))
    bins = np.searchsorted(edges, kmag, side='right')-1
    tb = 22*np.sqrt((1+redshift)/7)  # paper convention, saturated Ts, no RSD

    def reduce(mask):
        w = weights[mask]
        modes = int(w.sum())
        if not modes:
            return None
        row = {"k_hMpc_inv": float(np.sum(kmag[mask]*w)/modes), "full_fft_modes": modes}
        for name, values in powers.items():
            row[name] = float(np.sum(values[mask]*w)/modes)
        row['b_HI_squared'] = row['P_HI']/row['P_matter'] if row['P_matter'] > 0 else None
        row['r_HI_matter'] = (row['P_cross']/np.sqrt(row['P_HI']*row['P_matter'])
                              if row['P_HI']*row['P_matter'] > 0 else None)
        row['P_21_mK2_hMpc3'] = tb**2*neutral_mean**2*row['P_HI']
        row['Delta2_21_mK2'] = row['k_hMpc_inv']**3*row['P_21_mK2_hMpc3']/(2*np.pi**2)
        return row

    shells = []
    for i in range(len(edges)-1):
        row = reduce(keep & (bins == i))
        if row is not None:
            row.update(bin_index=i, k_min=float(edges[i]), k_max=float(edges[i+1]))
            shells.append(row)
    k0 = reduce(keep & (kmag >= band[0]) & (kmag < band[1]))
    return {"shells": shells, "k0_band": k0, "Q_mass": total(a)/total(c),
            "Q_volume": float(np.mean(a/c)), "mean_brightness_mK": tb*float(neutral_mean),
            "fractional_HI_definition": "(capacity-absorbed)/mean(capacity-absorbed)-1",
            "P_units": "(h^-1 cMpc)^3", "no_RSD": True,
            "k0_estimator": "ratio of mode-weighted powers, not mean of mode ratios"}
