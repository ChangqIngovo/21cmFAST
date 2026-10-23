"""Fast photon-allocation candidates on one additive physical source field.

These are research comparators, not production ionization models.  Counts are
photons / hydrogen atoms in cells, not volume fractions.  Inputs describe one
emission increment and a frozen density/initial ionization state on a periodic
Cartesian grid.  Recombination, source feedback and finite light travel times
are outside these kernels.

``es-global`` closes the global count by varying an ES amplitude.  ``es-front``
preserves local absorption and ranks residual atom capacity by an ES statistic.
Neither global method proves a per-source allocation.  ``radial-shell`` retains
original source positions and debits simultaneous accepted source-target edges.
``voronoi`` fills nearest-active-source domains in radial order, sharing exact
distance ties equally.  Both preserve source/cell budgets and distance support,
but neither preserves angular budgets or line-of-sight opacity.  These
limitations are deliberate benchmark targets.
"""

from __future__ import annotations

from functools import lru_cache
from time import perf_counter

import numpy as np
from scipy import fft
from scipy.spatial import cKDTree


def _inputs(source, capacity, initial_ionized):
    source = np.asarray(source, dtype=np.float64)
    capacity = np.asarray(capacity, dtype=np.float64)
    if source.ndim != 3 or source.shape != capacity.shape:
        raise ValueError("source and capacity must have the same 3D shape")
    initial = (np.zeros_like(capacity) if initial_ionized is None else
               np.asarray(initial_ionized, dtype=np.float64))
    if initial.shape != capacity.shape:
        raise ValueError("initial_ionized must be atom counts on the same grid")
    for name, values in (("source", source), ("capacity", capacity),
                         ("initial_ionized", initial)):
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"{name} must contain finite nonnegative counts")
        if not np.isfinite(values.sum()):
            raise ValueError(f"{name} total is not representable")
    if np.any(initial > capacity):
        raise ValueError("initial_ionized exceeds the atom capacity")
    return source, capacity, initial, capacity - initial


@lru_cache(maxsize=12)
def _distance_squared(shape):
    axes = [np.minimum(np.arange(n), n - np.arange(n)).astype(np.float64)
            for n in shape]
    return sum(a.reshape((1,) * i + (-1,) + (1,) * (2-i))**2
               for i, a in enumerate(axes))


def _radii(shape, radii, max_radius):
    maximum = float(min(shape) / 2 if max_radius is None else max_radius)
    if not np.isfinite(maximum) or maximum < 0:
        raise ValueError("max_radius must be finite and nonnegative")
    if radii is None:
        radii = [] if maximum < 1 else list(1.3**np.arange(
            int(np.floor(np.log(maximum) / np.log(1.3))) + 1))
        if maximum > 0 and (not radii or radii[-1] < maximum):
            radii.append(maximum)
    radii = np.asarray(radii, dtype=np.float64)
    if (radii.ndim != 1 or not np.isfinite(radii).all() or
            np.any(radii <= 0) or np.any(np.diff(radii) <= 0) or
            np.any(radii > maximum*(1+1e-12))):
        raise ValueError("radii must strictly increase in (0, max_radius]")
    return radii, maximum


def _thresholds(source, available, radii, workers):
    """Minimum ES crossing amplitude from additive count fields.

    Positive real-space ball kernels are used.  Native sharp-k conditional
    source histories are not reinterpreted as additive source count fields.
    The small roundoff floor prevents amplification of FFT leakage into empty
    regions; unresolved interactions are retained as photons, not absorbed.
    """
    shape = source.shape
    distance2 = _distance_squared(shape)
    sf = fft.rfftn(source, workers=workers)
    cf = fft.rfftn(available, workers=workers)
    threshold = np.full(shape, np.inf)
    crossing_radius = np.full(shape, np.inf)
    transforms = 2
    for radius in radii:
        kernel = (distance2 <= radius*radius + 1e-12).astype(float)
        kernel /= kernel.sum()
        kf = fft.rfftn(kernel, workers=workers)
        sr = np.maximum(fft.irfftn(sf*kf, s=shape, workers=workers), 0)
        cr = np.maximum(fft.irfftn(cf*kf, s=shape, workers=workers), 0)
        floor = 128*np.finfo(float).eps*max(float(sr.max()), np.finfo(float).tiny)
        ratio = np.divide(cr, sr, out=np.full(shape, np.inf), where=sr > floor)
        # Exact lattice symmetries otherwise acquire arbitrary sub-ulp crossing
        # order from FFT noise.  Keep 40 mantissa bits for the threshold only;
        # source/atom counts and global closure retain full double precision.
        # This is a declared ~1e-12 relative tie convention, not spatial smoothing.
        finite = np.isfinite(ratio)
        mantissa, exponent = np.frexp(ratio[finite])
        ratio[finite] = np.ldexp(np.rint(mantissa*2.**40)/2.**40, exponent)
        better = ratio < threshold
        threshold[better] = ratio[better]
        crossing_radius[better] = radius
        transforms += 3
    return threshold, crossing_radius, transforms


def _map_at_amplitude(source, available, thresholds, amplitude):
    absorbed = np.minimum(amplitude*source, available)
    absorbed[amplitude > thresholds] = available[amplitude > thresholds]
    return absorbed


def _finish(source, capacity, initial, available, absorbed, remainder,
            started, method, diagnostics=None, remaining_source=None):
    q = np.sum(source, dtype=np.longdouble)
    a = np.sum(absorbed, dtype=np.longdouble)
    reservoir = np.longdouble(remainder)
    residual = q-a-reservoir
    scale = max(float(q), np.finfo(float).tiny)
    tolerance = 5e-10*scale
    if (np.any(absorbed < 0) or np.max(absorbed-available, initial=0) > tolerance
            or remainder < -tolerance or abs(residual) > tolerance):
        raise FloatingPointError({"method": method, "source": float(q),
                                  "absorbed": float(a), "remaining": float(reservoir),
                                  "residual": float(residual)})
    audit = dict(method=method, photons=float(q), absorbed=float(a),
                 unabsorbed=float(reservoir), ledger_residual=float(residual),
                 relative_residual=float(residual/scale),
                 max_capacity_excess=float(np.max(absorbed-available, initial=0)),
                 periodic=True, counts_are_increments=True,
                 seconds=perf_counter()-started)
    audit.update(diagnostics or {})
    return dict(absorbed=absorbed, ionized=initial+absorbed,
                available=available-absorbed, unabsorbed=float(reservoir),
                remaining_source=remaining_source, diagnostics=audit, audit=audit)


def source_consistent_es(source, capacity, initial_ionized=None, *, radii=None,
                         max_radius=None, workers=1):
    """Uncorrected center-cell ES control using the same additive sources.

    Its returned ``photon_residual`` may have either sign.  This comparator
    deliberately does not claim photon conservation or a nonnegative reservoir.
    """
    started = perf_counter()
    source, capacity, initial, available = _inputs(source, capacity, initial_ionized)
    radii, maximum = _radii(source.shape, radii, max_radius)
    threshold, _, transforms = _thresholds(source, available, radii, workers)
    absorbed = _map_at_amplitude(source, available, threshold, 1.)
    residual = float(np.sum(source, dtype=np.longdouble)-
                     np.sum(absorbed, dtype=np.longdouble))
    audit = dict(method="es-control", photons=float(source.sum()),
                 absorbed=float(absorbed.sum()), photon_residual=residual,
                 fft_calls=transforms, max_radius=maximum,
                 seconds=perf_counter()-started)
    return dict(absorbed=absorbed, ionized=initial+absorbed, diagnostics=audit, audit=audit)


def _es_global(source, capacity, initial, available, *, radii, maximum,
               workers, tolerance, started):
    threshold, _, transforms = _thresholds(source, available, radii, workers)
    q = float(source.sum())
    reachable = (source > 0) | np.isfinite(threshold)
    target = min(q, float(available[reachable].sum()))
    tol = tolerance*max(q, np.finfo(float).tiny)
    lo, hi = 0., 1.
    low = np.zeros_like(source)
    high = _map_at_amplitude(source, available, threshold, hi)
    evaluations = 1
    while high.sum() < target-tol and hi < 2.**80:
        lo, low = hi, high
        hi *= 2
        high = _map_at_amplitude(source, available, threshold, hi)
        evaluations += 1
    if high.sum() < target-tol:
        raise FloatingPointError("ES amplitude search could not reach its finite support")
    if abs(high.sum()-target) <= tol:
        low, lo = high, hi
    else:
        for _ in range(100):
            if target-low.sum() <= tol:
                break
            mid = .5*(lo+hi)
            if mid in (lo, hi):
                break
            trial = _map_at_amplitude(source, available, threshold, mid)
            evaluations += 1
            if trial.sum() <= target:
                lo, low = mid, trial
            else:
                hi, high = mid, trial
    weight = 0.
    if target-low.sum() > tol and high.sum() > low.sum():
        weight = (target-low.sum())/(high.sum()-low.sum())
        absorbed = low + weight*(high-low)
    else:
        absorbed = low
    remaining = max(0., float(np.sum(source, dtype=np.longdouble)-
                              np.sum(absorbed, dtype=np.longdouble)))
    return _finish(source, capacity, initial, available, absorbed, remaining,
                   started, "es-global", dict(
                       fft_calls=transforms, evaluations=evaluations,
                       amplitude_low=lo, amplitude_high=hi, boundary_weight=weight,
                       threshold_mantissa_bits=40,
                       max_radius=maximum, source_attribution=False,
                       limitation="global count only; ES rank does not certify source-target budgets"))


def _es_front(source, capacity, initial, available, *, radii, maximum,
              workers, tolerance, started):
    local = np.minimum(source, available)
    photons = source-local
    residual_capacity = available-local
    budget = float(photons.sum())
    if budget == 0 or not np.any(residual_capacity):
        return _finish(source, capacity, initial, available, local, budget,
                       started, "es-front", dict(fft_calls=0, evaluations=0,
                           source_attribution=budget == 0, max_radius=maximum),
                       remaining_source=photons)
    threshold, _, transforms = _thresholds(photons, residual_capacity, radii, workers)
    candidates = (residual_capacity > 0) & np.isfinite(threshold)
    scores, weights = threshold[candidates], residual_capacity[candidates]
    moved = min(budget, float(weights.sum()))
    additional = np.zeros_like(source)
    if moved == weights.sum():
        additional[candidates] = weights
    elif moved > 0:
        order = np.argsort(scores, kind="stable")
        cumulative = np.cumsum(weights[order])
        pivot = scores[order[min(np.searchsorted(cumulative, moved), len(order)-1)]]
        # FFT noise must not select a preferred direction in a tied lattice shell.
        tie = max(1e-10*abs(pivot), 256*np.finfo(float).eps)
        inner = candidates & (threshold < pivot-tie)
        boundary = candidates & (np.abs(threshold-pivot) <= tie)
        additional[inner] = residual_capacity[inner]
        left = moved-additional.sum()
        shell = residual_capacity[boundary].sum()
        if left < 0 or left > shell*(1+1e-10) or shell <= 0:
            raise FloatingPointError("ES rank boundary did not bracket its budget")
        additional[boundary] = residual_capacity[boundary]*min(1., left/shell)
    absorbed = local+additional
    remaining = max(0., float(np.sum(source, dtype=np.longdouble)-
                              np.sum(absorbed, dtype=np.longdouble)))
    return _finish(source, capacity, initial, available, absorbed, remaining,
                   started, "es-front", dict(
                       fft_calls=transforms, evaluations=1, max_radius=maximum,
                       source_attribution=False, local_absorption_preserved=True,
                       limitation="residual photons pooled globally; finite ES support is not an edge allocation"))


@lru_cache(maxsize=12)
def _offsets(shape, maximum):
    axes = [np.where(np.arange(n) <= n//2, np.arange(n), np.arange(n)-n)
            for n in shape]
    offsets = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    d2 = np.sum(offsets**2, axis=1)
    selected = d2 <= maximum*maximum+1e-12
    offsets, d2 = offsets[selected], d2[selected]
    order = np.argsort(d2, kind="stable")
    return offsets[order], d2[order]


def _radial_shell(source, capacity, initial, available, *, maximum,
                  tolerance, max_iterations, started):
    """Original-source anchored synchronous radial proposals.

    Each source proposes a nested spherical fill of currently available atoms.
    At a contested cell every proposal is multiplied by the SAME acceptance
    fraction.  Rejected photons remain with the original emitter.  Unlike
    overlap-cell relay methods this preserves emitter-to-target distance bounds;
    it still freely reallocates a source's angular shares.
    Accepted edges are not revisited, so finite-radius capacity constraints can
    strand photons even when a different allocation would consume them.
    """
    local = np.minimum(source, available)
    absorbed = local.copy()
    remaining_grid = source-local
    active = np.flatnonzero(remaining_grid)
    photons = remaining_grid.ravel()[active].copy()
    original_budget = photons.copy()
    positions = np.column_stack(np.unravel_index(active, source.shape))
    space = available-local
    offsets, distance2 = _offsets(source.shape, maximum)
    q = float(source.sum())
    tol = tolerance*max(q, np.finfo(float).tiny)
    iterations, offers, accepted_edges = 0, 0, 0
    for iteration in range(max_iterations):
        if photons.sum() <= tol or space.sum() <= 0 or not len(active):
            break
        pool = np.zeros_like(space)
        proposal = []
        for k, (position, budget) in enumerate(zip(positions, photons)):
            # Leave negligible numerical tails explicitly at their source;
            # subtracting a shell debit at that scale cannot be relative-accurate.
            if budget <= tolerance*max(original_budget[k], np.finfo(float).tiny):
                continue
            coordinates = (offsets+position) % np.array(source.shape)
            indexes = np.ravel_multi_index(coordinates.T, source.shape)
            demand = space.ravel()[indexes]
            total = float(demand.sum())
            if total <= 0:
                continue
            offered = min(float(budget), total)
            if offered >= total:
                values = demand.copy()
            else:
                cumulative = np.cumsum(demand)
                pivot = min(np.searchsorted(cumulative, offered), len(distance2)-1)
                radius2 = distance2[pivot]
                inner, shell = distance2 < radius2, distance2 == radius2
                values = np.zeros_like(demand)
                values[inner] = demand[inner]
                left = offered-values.sum()
                total_shell = demand[shell].sum()
                if total_shell > 0:
                    values[shell] = demand[shell]*min(1., max(0., left/total_shell))
            keep = values > 0
            indexes, values = indexes[keep], values[keep]
            np.add.at(pool.ravel(), indexes, values)
            proposal.append((k, indexes, values))
            offers += 1
        if not proposal:
            break
        acceptance = np.divide(space, pool, out=np.ones_like(space),
                               where=pool > space)
        taken = np.minimum(pool, space)
        if taken.sum() <= 0:
            break
        for k, indexes, values in proposal:
            debit = float(np.sum(values*acceptance.ravel()[indexes]))
            if debit > photons[k]*(1+1e-12):
                raise FloatingPointError("Radial proposal overdrew an original source")
            photons[k] = max(0., photons[k]-debit)
            accepted_edges += len(indexes)
        absorbed += taken
        space = np.maximum(space-taken, 0.)
        iterations = iteration+1
    remaining_grid.fill(0.)
    remaining_grid.ravel()[active] = photons
    remaining = float(photons.sum())
    return _finish(source, capacity, initial, available, absorbed, remaining,
                   started, "radial-shell", dict(
                       fft_calls=0, iterations=iterations, radial_offers=offers,
                       accepted_edges=accepted_edges, max_radius=maximum,
                       max_iterations_reached=iterations >= max_iterations,
                       source_attribution=True, original_source_positions=True,
                       maximum_absorption_certified=False,
                       limitation="spherical capacity search has no angular budget or foreground opacity"),
                   remaining_source=remaining_grid)


def _voronoi(source, capacity, initial, available, *, maximum, workers,
             tolerance, max_iterations, started):
    """Capacity-limited nearest-active-source allocation on a periodic grid.

    Available cells are assigned to their nearest active emitters.  Ties split
    the atom capacity equally across ALL equally near sources.  Each emitter
    fills its assigned cells in radial order up to its budget.  Exhausted sources
    retire and the partition is recomputed.  This is a geometric approximation,
    not a solution of angular transport or simultaneous continuous flux sharing.
    Accepted edges are not revisited; no maximum-absorption claim is made.
    """
    local = np.minimum(source, available)
    absorbed = local.copy()
    space = available-local
    origin_flat = np.flatnonzero(source-local)
    positions = np.column_stack(np.unravel_index(origin_flat, source.shape)).astype(float)
    photons = (source-local).ravel()[origin_flat].copy()
    original_budget = photons.copy()
    q = float(source.sum())
    tol = tolerance*max(q, np.finfo(float).tiny)
    dimensions = np.array(source.shape, dtype=float)
    iterations, tree_queries, edges_total, tie_targets = 0, 0, 0, 0
    for iteration in range(max_iterations):
        active = np.flatnonzero(photons > tolerance*np.maximum(original_budget,
                                                               np.finfo(float).tiny))
        targets = np.flatnonzero(space)
        if not len(active) or not len(targets) or photons.sum() <= tol:
            break
        centers = positions[active]
        target_positions = np.column_stack(np.unravel_index(targets, source.shape)).astype(float)
        tree = cKDTree(centers, boxsize=dimensions)
        k = min(8, len(active))
        distance, nearest = tree.query(target_positions, k=k, workers=workers)
        tree_queries += 1
        if k == 1:
            distance, nearest = distance[:, None], nearest[:, None]
        # All coordinates are integer cell centers: squared periodic distances
        # are integers.  This exact tie criterion is translation/axis symmetric.
        d2 = np.rint(distance*distance)
        best2 = d2[:, 0]
        within = best2 <= maximum*maximum+1e-12
        ties = (d2 == best2[:, None]) & within[:, None]
        crowded = (ties[:, -1] & within) if len(active) > k else np.zeros(len(targets), bool)
        ties[crowded] = False
        target_edge, column = np.nonzero(ties)
        owner_edge = nearest[target_edge, column]
        count = ties.sum(axis=1)
        weight_edge = space.ravel()[targets[target_edge]] / count[target_edge]
        radius_edge = best2[target_edge]
        tie_targets += int(np.count_nonzero(count > 1))
        if np.any(crowded):
            extras_t, extras_s, extras_w, extras_d = [], [], [], []
            for t in np.flatnonzero(crowded):
                owner = np.asarray(tree.query_ball_point(
                    target_positions[t], np.sqrt(best2[t])+1e-9), dtype=int)
                displacement = np.abs(centers[owner]-target_positions[t])
                displacement = np.minimum(displacement, dimensions-displacement)
                exact2 = np.sum(displacement*displacement, axis=1)
                owner = owner[exact2 == best2[t]]
                extras_t.extend([t]*len(owner))
                extras_s.extend(owner)
                extras_w.extend([space.ravel()[targets[t]]/len(owner)]*len(owner))
                extras_d.extend([best2[t]]*len(owner))
            tie_targets += int(np.count_nonzero(crowded))
            target_edge = np.concatenate((target_edge, np.asarray(extras_t, dtype=int)))
            owner_edge = np.concatenate((owner_edge, np.asarray(extras_s, dtype=int)))
            weight_edge = np.concatenate((weight_edge, np.asarray(extras_w)))
            radius_edge = np.concatenate((radius_edge, np.asarray(extras_d)))
        if not len(owner_edge):
            break
        order = np.lexsort((radius_edge, owner_edge))
        owner_edge, target_edge, weight_edge, radius_edge = (
            a[order] for a in (owner_edge, target_edge, weight_edge, radius_edge))
        boundaries = np.r_[0, 1+np.flatnonzero(np.diff(owner_edge)), len(owner_edge)]
        accepted = np.zeros(len(owner_edge))
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            owner = active[owner_edge[left]]
            budget = photons[owner]
            weights = weight_edge[left:right]
            total = weights.sum()
            if budget >= total:
                accepted[left:right] = weights
                debit = total
            else:
                cum = np.cumsum(weights)
                pivot = min(np.searchsorted(cum, budget), len(weights)-1)
                cut = radius_edge[left+pivot]
                inner = radius_edge[left:right] < cut
                shell = radius_edge[left:right] == cut
                portion = accepted[left:right]
                portion[inner] = weights[inner]
                remaining = budget-portion.sum()
                portion[shell] = weights[shell]*max(0., min(1., remaining/weights[shell].sum()))
                debit = portion.sum()
            if debit > photons[owner]*(1+1e-12):
                raise FloatingPointError("Voronoi allocation overdrew an original source")
            photons[owner] = max(0., photons[owner]-debit)
        taken = np.bincount(targets[target_edge], weights=accepted,
                            minlength=space.size).reshape(space.shape)
        taken = np.minimum(taken, space)
        if taken.sum() <= 0:
            break
        absorbed += taken
        space = np.maximum(space-taken, 0.)
        iterations = iteration+1
        edges_total += len(owner_edge)
    remaining_grid = np.zeros_like(source)
    remaining_grid.ravel()[origin_flat] = photons
    return _finish(source, capacity, initial, available, absorbed, float(photons.sum()),
                   started, "voronoi", dict(
                       fft_calls=0, iterations=iterations, tree_queries=tree_queries,
                       allocation_edges=edges_total, tied_target_queries=tie_targets,
                       max_radius=maximum, source_attribution=True,
                       maximum_absorption_certified=False,
                       max_iterations_reached=iterations >= max_iterations,
                       nearest_source_ties="equal capacity split across every tied source",
                       limitation="nearest-source domains replace angular flux competition; no foreground opacity"),
                   remaining_source=remaining_grid)


def solve(source, capacity, initial_ionized=None, *, method="es-global",
          radii=None, max_radius=None, workers=1, max_iterations=64,
          tolerance=1e-11):
    """Allocate a photon increment on a frozen 3D periodic atom field.

    Parameters ``capacity`` and ``initial_ionized`` are TOTAL atom counts and
    already ionized atom counts respectively.  Returned ``absorbed`` is the new
    absorption, and ``ionized`` includes the initial state.  No leftover photons
    are treated as absorbed.  Radial-shell and Voronoi return per-source
    leftovers.  The two ES closures generally identify only a global reservoir.
    """
    started = perf_counter()
    source, capacity, initial, available = _inputs(source, capacity, initial_ionized)
    if not np.isfinite(tolerance) or tolerance <= 0 or tolerance >= 1:
        raise ValueError("tolerance must lie in (0,1)")
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    radii, maximum = _radii(source.shape, radii, max_radius)
    if method == "es-local":
        method = "es-front"
    if method not in ("es-global", "es-front", "radial-shell", "voronoi"):
        raise ValueError("Unknown candidate method")
    if source.sum() == 0 or not np.any(available):
        return _finish(source, capacity, initial, available, np.zeros_like(source),
                       float(source.sum()), started, method,
                       dict(fft_calls=0, iterations=0), remaining_source=source.copy())
    if method == "es-global":
        return _es_global(source, capacity, initial, available, radii=radii,
                          maximum=maximum, workers=workers,
                          tolerance=tolerance, started=started)
    if method == "es-front":
        return _es_front(source, capacity, initial, available, radii=radii,
                         maximum=maximum, workers=workers,
                         tolerance=tolerance, started=started)
    if method == "voronoi":
        return _voronoi(source, capacity, initial, available, maximum=maximum,
                        workers=workers, tolerance=tolerance,
                        max_iterations=max_iterations, started=started)
    return _radial_shell(source, capacity, initial, available, maximum=maximum,
                         tolerance=tolerance, max_iterations=max_iterations,
                         started=started)
