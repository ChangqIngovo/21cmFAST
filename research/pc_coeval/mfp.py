"""Geometric mean-free-path distributions of ionized bubbles / neutral islands.

Uniform phase-volume origins and isotropic rays measure distance to the first
opposite-phase voxel. This is the MFP bubble-size statistic of Mesinger &
Furlanetto (2007), not an optical-depth / physical absorption mean free path.
It measures the supplied voxel field: increasing rays does not test grid
convergence. Periodic rays still inside the phase at a finite cap are censored.
"""
from __future__ import annotations

import ctypes
import hashlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading

import numpy as np

_LIBRARY = None
_BUILD_IDENTITY = None
_BUILD_LOCK = threading.Lock()


def _library():
    global _LIBRARY, _BUILD_IDENTITY
    with _BUILD_LOCK:
        if _LIBRARY is not None:
            return _LIBRARY
        source = Path(__file__).with_name("mfp_kernel.cpp")
        compiler = shutil.which(os.environ.get("CXX", "g++"))
        if compiler is None:
            raise RuntimeError("MFP measurement requires a C++ compiler with OpenMP (CXX or g++)")
        compiler = str(Path(compiler).resolve())
        version = subprocess.run([compiler, "--version"], check=True,
                                 capture_output=True, text=True, timeout=30).stdout
        flags = ["-O3", "-std=c++17", "-shared", "-fPIC", "-fopenmp"]
        identity = {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "compiler": compiler, "compiler_version": version, "flags": flags}
        directory = Path(tempfile.mkdtemp(prefix="pc_mfp_kernel_"))
        output = directory / "mfp.so"
        subprocess.run([compiler, *flags, str(source), "-o", str(output)], check=True,
                       capture_output=True, text=True, timeout=180)
        lib = ctypes.CDLL(str(output))
        lib.trace_mfp.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int64,
                                 ctypes.c_void_p, ctypes.c_void_p,
                                 ctypes.c_double, ctypes.c_double, ctypes.c_int,
                                 ctypes.c_void_p, ctypes.c_void_p]
        lib.trace_mfp.restype = ctypes.c_int
        _LIBRARY, _BUILD_IDENTITY = lib, identity
        return lib


def build_identity():
    _library()
    return dict(_BUILD_IDENTITY)


def _positive_float(value, name):
    if isinstance(value, bool) or not np.isscalar(value) or not math.isfinite(value) or value<=0:
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def _integer(value, name, minimum=1):
    if type(value) is not int or not minimum<=value<2**31:
        raise ValueError(f"{name} must be an integer in [{minimum}, 2**31)")
    return value


def make_draws(ray_count=100_000, seed=20261103):
    """Common random numbers: voxel rank, xyz subvoxel position, mu and azimuth.

    Rank is mapped into each field's own selected-phase list. Different fields
    therefore share random variates, not generally the same physical origins.
    """
    _integer(ray_count, "ray_count")
    return np.random.default_rng(seed).random((ray_count, 6))


def trace_rays(mask, box_len, origins, directions, *, max_distance=None, threads=1):
    """Return exact voxel first-exit distances and cap-censor flags for given rays.

    Origins must lie in the selected phase; coordinates are periodic. Direction
    vectors are normalized internally. Distances and coordinates use cMpc.
    """
    mask = np.asarray(mask)
    if mask.dtype != np.bool_ or mask.ndim!=3 or len(set(mask.shape))!=1 or mask.shape[0]<1:
        raise ValueError("mask must be a nonempty cubic boolean array")
    if mask.shape[0]>2048:
        raise ValueError("mask dimension exceeds supported range")
    box_len = _positive_float(box_len, "box_len")
    cap = box_len if max_distance is None else _positive_float(max_distance, "max_distance")
    _integer(threads, "threads")
    origins = np.asarray(origins, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    if origins.ndim!=2 or origins.shape[1]!=3 or origins.shape!=directions.shape:
        raise ValueError("origins and directions must have identical (rays, 3) shapes")
    if not np.isfinite(origins).all() or not np.isfinite(directions).all():
        raise ValueError("ray coordinates and directions must be finite")
    norm = np.linalg.norm(directions, axis=1)
    if np.any(norm==0) or not np.isfinite(norm).all():
        raise ValueError("directions must have finite nonzero norm")
    origins = np.ascontiguousarray(np.mod(origins, box_len))
    directions = np.ascontiguousarray(directions/norm[:, None])
    cell_size = box_len/mask.shape[0]
    cells = np.minimum(np.floor(origins/cell_size).astype(np.int64), mask.shape[0]-1)
    if not mask[tuple(cells.T)].all():
        raise ValueError("every ray origin must lie in the selected phase")
    distance = np.full(len(origins), cap, dtype=np.float64)
    censored = np.ones(len(origins), dtype=np.uint8)
    if len(origins) and not mask.all():
        data = np.ascontiguousarray(mask, dtype=np.uint8)
        status = _library().trace_mfp(mask.shape[0], data.ctypes.data, len(origins),
                                     origins.ctypes.data, directions.ctypes.data,
                                     cell_size, cap, threads,
                                     distance.ctypes.data, censored.ctypes.data)
        if status:
            raise RuntimeError(f"MFP ray kernel failed: status {status}")
    return distance, censored.astype(bool)


def _summary(distance, censored, cap, cell_size, bins):
    n = len(distance)
    # First bin starts at zero so subcell paths are retained. A log-spaced
    # histogram is a plotting convention, not removal of near-grid paths.
    edges = np.r_[0., np.geomspace(min(cell_size/100, cap/100), cap, bins)]
    if not n:
        return {"sample_count": 0, "status": "empty_phase", "restricted_mean_cMpc": None,
                "restricted_mean_standard_error_cMpc": None, "batch_restricted_mean_standard_error_cMpc": None,
                "censored_fraction": None,
                "uncensored_mean_cMpc": None, "quantiles_cMpc": {"0.16": None, "0.5": None, "0.84": None},
                "histogram_edges_cMpc": edges.tolist(), "histogram_probability": [0.]*(len(edges)-1),
                "histogram_density_per_cMpc": [0.]*(len(edges)-1), "boundary_cdf_at_edges": [0.]*len(edges),
                "cdf_dkw_95_halfwidth": None, "first_half_vs_all_restricted_cdf_max_difference": None,
                "batch_restricted_means_cMpc": [], "batch_censored_fractions": [],
                "restricted_mean_first_half_cMpc": None, "rays_below_one_cell_fraction": None,
                "censored_count": 0, "boundary_hit_count": 0}
    hist = np.histogram(distance[~censored], bins=edges)[0]/n
    sorted_dist = np.sort(distance)
    actual = np.sort(distance[~censored])
    quantiles = {}
    for q in (0.16, 0.5, 0.84):
        index = max(0, math.ceil(q*n)-1)
        quantiles[str(q)] = float(sorted_dist[index]) if index<len(actual) else None
    batch_count = min(16, max(1, n//128))
    batches = np.array_split(np.arange(n), batch_count)
    means = [float(np.mean(distance[b])) for b in batches]
    censored_batches = [float(np.mean(censored[b])) for b in batches]
    batch_se = float(np.std(means, ddof=1)/np.sqrt(batch_count)) if batch_count>1 else None
    se = float(np.std(distance, ddof=1)/np.sqrt(n)) if n>1 else None
    nhalf = n//2
    if nhalf:
        half = np.sort(distance[:nhalf])
        points = np.unique(sorted_dist)
        half_difference = float(np.max(np.abs(np.searchsorted(half, points, side="right")/nhalf
                                    -np.searchsorted(sorted_dist, points, side="right")/n)))
        half_mean = float(np.mean(half))
    else:
        half_difference, half_mean = None, None
    return {"sample_count": n, "status": "all_censored" if censored.all() else "measured",
            "restricted_mean_cMpc": float(np.mean(distance)),
            "restricted_mean_standard_error_cMpc": se,
            "batch_restricted_mean_standard_error_cMpc": batch_se,
            "censored_fraction": float(np.mean(censored)),
            "uncensored_mean_cMpc": float(np.mean(distance)) if not censored.any() else None,
            "quantiles_cMpc": quantiles,
            "histogram_edges_cMpc": edges.tolist(), "histogram_probability": hist.tolist(),
            "histogram_density_per_cMpc": (hist/np.diff(edges)).tolist(),
            "boundary_cdf_at_edges": (np.searchsorted(actual, edges, side="right")/n).tolist(),
            "cdf_dkw_95_halfwidth": float(np.sqrt(np.log(40)/(2*n))),
            "first_half_vs_all_restricted_cdf_max_difference": half_difference,
            "batch_restricted_means_cMpc": means, "batch_censored_fractions": censored_batches,
            "restricted_mean_first_half_cMpc": half_mean,
            "rays_below_one_cell_fraction": float(np.mean((distance<cell_size)&~censored)),
            "censored_count": int(censored.sum()), "boundary_hit_count": int((~censored).sum())}


def measure_mfp(xhi, box_len, *, phase="ionized", threshold=0.5, draws=None,
                ray_count=100_000, seed=20261103, max_distance=None, threads=1, bins=48):
    """Measure phase morphology; return raw rays and a JSON-safe summary.

    ``restricted_mean`` is E[min(first exit, cap)], not the mean of uncensored
    rays. The uncensored mean is reported only if every ray hits a boundary.
    Quantiles beyond the observed boundary fraction are null. Monte Carlo
    uncertainty is conditional on this one field and does not cover cosmic
    variance, threshold choice, subgrid physics or spatial resolution.
    """
    xhi = np.asarray(xhi)
    if xhi.ndim!=3 or len(set(xhi.shape))!=1 or xhi.shape[0]<1:
        raise ValueError("xhi must be a nonempty cubic 3D array")
    if not np.isfinite(xhi).all() or np.any(xhi<0) or np.any(xhi>1):
        raise ValueError("xhi must contain finite neutral fractions in [0, 1]")
    if isinstance(threshold, bool) or not math.isfinite(threshold) or not 0<=threshold<=1:
        raise ValueError("threshold must be finite and in [0, 1]")
    if phase not in ("ionized", "neutral"):
        raise ValueError("phase must be 'ionized' or 'neutral'")
    _integer(bins, "bins", minimum=2)
    _integer(threads, "threads")
    box_len = _positive_float(box_len, "box_len")
    cap = box_len if max_distance is None else _positive_float(max_distance, "max_distance")
    mask = xhi<float(threshold) if phase=="ionized" else xhi>=float(threshold)
    selected = np.flatnonzero(mask)
    if draws is None:
        draws = make_draws(ray_count, seed)
    draws = np.asarray(draws, dtype=np.float64)
    if draws.ndim!=2 or draws.shape[1]!=6 or len(draws)<1:
        raise ValueError("draws must have shape (positive ray count, 6)")
    if not np.isfinite(draws).all() or np.any(draws<0) or np.any(draws>=1):
        raise ValueError("draws must be finite uniform variates in [0, 1)")
    cell_size = box_len/xhi.shape[0]
    if len(selected):
        rank = np.minimum((draws[:, 0]*len(selected)).astype(np.int64), len(selected)-1)
        origins = (np.array(np.unravel_index(selected[rank], xhi.shape)).T+draws[:, 1:4])*cell_size
        mu = 2*draws[:, 4]-1
        phi = 2*np.pi*draws[:, 5]
        transverse = np.sqrt(np.maximum(0., 1-mu*mu))
        directions = np.column_stack((transverse*np.cos(phi), transverse*np.sin(phi), mu))
        distance, censored = trace_rays(mask, box_len, origins, directions,
                                        max_distance=cap, threads=threads)
    else:
        distance, censored = np.empty(0, dtype=np.float64), np.empty(0, dtype=bool)
    summary = _summary(distance, censored, cap, cell_size, bins)
    summary.update({"method": "periodic-voxel-geometric-first-exit-MFP", "phase": phase,
                    "threshold_xHI": float(threshold), "phase_rule": "xHI < threshold" if phase=="ionized" else "xHI >= threshold",
                    "selected_voxel_count": len(selected), "phase_volume_fraction": len(selected)/mask.size,
                    "grid_dimension": xhi.shape[0], "cell_size_cMpc": cell_size,
                    "box_len_cMpc": box_len, "max_distance_cMpc": cap, "requested_rays": len(draws),
                    "periodic": True, "isotropic": True, "uniform_phase_volume_origins": True,
                    "phase_mask_sha256": hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest(),
                    "random_draws_sha256": hashlib.sha256(np.ascontiguousarray(draws).tobytes()).hexdigest(),
                    "common_random_numbers": "Same rank/subvoxel/direction draws; origins differ between unequal phase masks.",
                    "interpretation": "Geometric bubble/island first-exit lengths, not optical absorption MFP or bubble radii.",
                    "uncertainty_scope": "Monte Carlo sampling on this fixed field; does not certify grid convergence.",
                    "censoring": "Rays still in the phase at max_distance remain right-censored; histogram integrates to 1-censored_fraction."})
    return {"distance_cMpc": distance, "censored": censored, "summary": summary}


def compare_mfp(first, second):
    """Two geometric samples: descriptive distances, no independent-sample p-value.

    A shared random sequence is allowed; the paired restricted-mean uncertainty
    uses the raw ray ordering. A CDF distance is a descriptive discrepancy, not
    a hypothesis-test significance or a measure of field-location agreement.
    """
    a, b = first["distance_cMpc"], second["distance_cMpc"]
    if not len(a) or not len(b):
        return {"available": False, "reason": "At least one selected phase is empty"}
    for key in ("max_distance_cMpc", "phase", "threshold_xHI", "box_len_cMpc"):
        if first["summary"][key]!=second["summary"][key]:
            raise ValueError(f"MFP comparisons require identical {key}")
    sa, sb = np.sort(a), np.sort(b)
    points = np.unique(np.r_[sa, sb])
    cdfa = np.searchsorted(sa, points, side="right")/len(a)
    cdfb = np.searchsorted(sb, points, side="right")/len(b)
    wasserstein = float(np.sum(np.abs(cdfa[:-1]-cdfb[:-1])*np.diff(points)))
    paired = len(a)==len(b) and first["summary"]["random_draws_sha256"]==second["summary"]["random_draws_sha256"]
    difference = a-b if paired else None
    paired_se = float(np.std(difference, ddof=1)/np.sqrt(len(a))) if paired and len(a)>1 else None
    mean_b = float(np.mean(b))
    return {"available": True, "restricted_cdf_max_difference": float(np.max(np.abs(cdfa-cdfb))),
            "restricted_wasserstein1_cMpc": wasserstein,
            "restricted_mean_difference_cMpc": float(np.mean(a)-mean_b),
            "restricted_mean_ratio": float(np.mean(a)/mean_b) if mean_b else None,
            "paired_restricted_mean_difference_standard_error_cMpc": paired_se,
            "common_random_variates": paired,
            "censored_fraction_difference": float(np.mean(first["censored"])-np.mean(second["censored"])),
            "interpretation": "Comparison of capped first-exit distributions; not field-location agreement or a grid-convergence test."}
