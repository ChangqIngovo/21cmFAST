"""Research-only local photon allocation closures; no transport claim.

`radial_redistribute` is a transparent Python prototype of independent radial
bubbles followed by synchronous overlap redistribution, following the broad
SCRIPT construction. It is not the optimized public SCRIPT implementation.
`dyadic_pool` is a nonoverlapping-hierarchy control. Both take photon increments
and current available atom capacity, and return an explicit unspent reservoir.
"""
from __future__ import annotations

from functools import lru_cache
import numpy as np


def validate(source, capacity):
    """Copy nonnegative cubic count fields."""
    q = np.array(source, dtype=np.float64, copy=True)
    c = np.array(capacity, dtype=np.float64, copy=True)
    if (q.ndim != 3 or len(set(q.shape)) != 1 or c.shape != q.shape
            or not np.isfinite(q).all() or not np.isfinite(c).all()
            or np.any(q < 0) or np.any(c < 0)):
        raise ValueError('Expected finite nonnegative cubic count fields')
    return q, c


@lru_cache(maxsize=32)
def radial_offsets(n, maximum_radius):
    """Lattice offsets sorted in distance shells, each periodic cell once."""
    axis = np.arange(n)
    axis = np.where(axis <= n//2, axis, axis-n)
    offsets = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), axis=-1).reshape(-1, 3)
    d2 = np.sum(offsets**2, axis=1)
    keep = d2 <= maximum_radius**2 + 1e-12
    offsets, d2 = offsets[keep], d2[keep]
    order = np.argsort(d2, kind='stable')
    offsets, d2 = offsets[order], d2[order]
    return offsets, d2


def radial_offer(position, photons, available, maximum_radius, terminal_mode='capacity'):
    """Give a photon packet to increasing-distance shells with tied-shell fairness."""
    n = available.shape[0]
    offsets, d2 = radial_offsets(n, maximum_radius)
    coords = (offsets + np.asarray(position)) % n
    indexes = np.ravel_multi_index(coords.T, available.shape)
    demand = available.ravel()[indexes]
    total = float(demand.sum())
    allocated = min(float(photons), total)
    result = np.zeros_like(demand)
    if allocated > 0:
        if allocated >= total:
            result = demand.copy()
        else:
            cumulative = np.cumsum(demand)
            pivot = min(np.searchsorted(cumulative, allocated), len(d2)-1)
            radius2 = d2[pivot]
            inner = d2 < radius2
            shell = d2 == radius2
            result[inner] = demand[inner]
            left = allocated - demand[inner].sum()
            shell_total = demand[shell].sum()
            if terminal_mode == 'equal':
                # Public SCRIPT stage one uses equal photons per tied cell;
                # these provisional claims may exceed an individual capacity.
                result[shell] = left / np.count_nonzero(shell)
            elif shell_total > 0:
                result[shell] = demand[shell] * (left/shell_total)
    # Roundoff must not create a negative photon reservoir.
    return indexes, result, max(0.0, float(photons)-float(result.sum()))


def audit(source, capacity, absorbed, remaining):
    """Independent nonnegative-count accounting."""
    q = np.sum(source, dtype=np.longdouble)
    a = np.sum(absorbed, dtype=np.longdouble)
    p = np.sum(remaining, dtype=np.longdouble)
    residual = q-a-p
    tol = 3e-11 * max(float(q), 1.0)
    if (abs(residual) > tol or np.any(absorbed < -tol)
            or np.any(absorbed-capacity > tol) or np.any(remaining < -tol)):
        raise FloatingPointError((float(q), float(a), float(p), float(residual)))
    return dict(photons=float(q), absorbed=float(a), remaining=float(p),
                residual=float(residual), relative_residual=float(residual/max(q, 1e-300)),
                capacity_violation=float(np.maximum(absorbed-capacity, 0).max()))


def radial_redistribute(source, capacity, *, maximum_radius=None, max_iterations=512,
                        source_order=None, initial_shell='equal'):
    """Independent radial bubbles and synchronous excess redistribution.

    The first stage spends each emitter's own photons independently. After
    overlap, excess is relayed from overlap cells, losing its emitter/direction
    labels. `maximum_radius` limits each relay, NOT total emitter-to-target
    distance. The final reservoir is positioned at its current relay location.
    This is the physical limitation tested by the accompanying probes.
    """
    q, c = validate(source, capacity)
    if initial_shell not in ('equal', 'capacity'):
        raise ValueError('initial_shell must be equal or capacity')
    n = q.shape[0]
    maximum_radius = float(n/2 if maximum_radius is None else maximum_radius)
    if maximum_radius < 0 or maximum_radius > np.sqrt(3)*(n//2):
        raise ValueError('Invalid periodic support radius')
    accepted = np.minimum(q, c)
    excess = np.maximum(q-c, 0)
    initial_emitters = np.flatnonzero(excess)
    if source_order is not None:
        initial_emitters = initial_emitters[np.asarray(source_order)]
    pool = accepted.copy()
    pending = np.zeros_like(q)
    offers = 0
    # Independent bubbles see each source's original surrounding atom field.
    for flat in initial_emitters:
        position = np.unravel_index(flat, q.shape)
        available = c.copy()
        available[position] = 0
        idx, add, left = radial_offer(position, excess[position], available, maximum_radius,
                                     terminal_mode=initial_shell)
        np.add.at(pool.ravel(), idx, add)
        pending[position] += left
        offers += 1
    # Simultaneous relays make the mathematical map independent of source order.
    iterations = 0
    while iterations < max_iterations:
        absorbed = np.minimum(pool, c)
        excess = np.maximum(pool-c, 0)
        if excess.sum() <= 1e-13 * max(q.sum(), 1.0):
            pending += excess
            pool = absorbed
            break
        available = c-absorbed
        emitters = np.flatnonzero(excess)
        if not np.any(available > 0):
            pending += excess
            pool = absorbed
            break
        next_pool = absorbed.copy()
        new_pending = np.zeros_like(q)
        for flat in emitters:
            position = np.unravel_index(flat, q.shape)
            idx, add, left = radial_offer(position, excess[position], available, maximum_radius)
            np.add.at(next_pool.ravel(), idx, add)
            new_pending[position] += left
            offers += 1
        pending += new_pending
        iterations += 1
        if np.array_equal(next_pool, absorbed):
            pool = absorbed
            break
        pool = next_pool
    absorbed = np.minimum(pool, c)
    pending += np.maximum(pool-c, 0)
    result = dict(absorbed=absorbed, remaining=pending, neutral=c-absorbed)
    result['audit'] = audit(q, c, absorbed, pending)
    result['audit'].update(iterations=iterations, radial_offers=offers,
                           maximum_radius=maximum_radius,
                           initial_shell=initial_shell,
                           relay_count_exhausted=iterations >= max_iterations,
                           source_labels_after_overlap=False)
    return result


def blocks(a, width):
    """View disjoint cubic nodes as rows."""
    n = a.shape[0]
    m = n//width
    return a.reshape(m,width,m,width,m,width).transpose(0,2,4,1,3,5).reshape(m**3,width**3)


def unblocks(a, n, width):
    """Recover the native cubic layout."""
    m=n//width
    return a.reshape(m,m,m,width,width,width).transpose(0,3,1,4,2,5).reshape(n,n,n)


def dyadic_pool(source, capacity, *, shift=(0,0,0), max_width=None):
    """Pool unspent photons only among nested disjoint cubic nodes.

    Exact node budgets remove overlapping accounting. The geometry depends on
    the hierarchy origin and has no directional conservation or optical depth.
    """
    q0,c0 = validate(source,capacity)
    n=q0.shape[0]
    if n & (n-1):
        raise ValueError('Dyadic control requires power-of-two side length')
    if max_width is None:
        max_width=n
    q=np.roll(q0,shift,axis=(0,1,2))
    c=np.roll(c0,shift,axis=(0,1,2))
    absorbed=np.zeros_like(c)
    width=1
    levels=0
    while width<=max_width:
        qb=blocks(q,width).copy()
        cb=blocks(c,width).copy()
        qt=qb.sum(axis=1)
        ct=cb.sum(axis=1)
        moved=np.minimum(qt,ct)
        fq=np.divide(moved,qt,out=np.zeros_like(qt),where=qt>0)
        fc=np.divide(moved,ct,out=np.zeros_like(ct),where=ct>0)
        dq=qb*fq[:,None]
        dc=cb*fc[:,None]
        q=unblocks(qb-dq,n,width)
        c=unblocks(cb-dc,n,width)
        absorbed+=unblocks(dc,n,width)
        width*=2
        levels+=1
    inverse=tuple(-v for v in shift)
    absorbed,q,c=(np.roll(a,inverse,axis=(0,1,2)) for a in (absorbed,q,c))
    result=dict(absorbed=absorbed,remaining=q,neutral=c)
    result['audit']=audit(q0,c0,absorbed,q)
    result['audit'].update(levels=levels,max_width=max_width,shift=list(shift))
    return result
