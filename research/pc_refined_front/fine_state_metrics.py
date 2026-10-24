"""Compare resolved gas states without constructing a dense fine grid."""
from __future__ import annotations

import numpy as np

_POPCOUNT = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)


def _geometry(ids, n, f):
    nf = n*f
    z = ids % nf
    y = (ids//nf) % nf
    x = ids//(nf*nf)
    parent = ((x//f)*n+y//f)*n+z//f
    child = ((x % f)*f+y % f)*f+z % f
    return parent, child


def _validate(state, capacity, f):
    sub, n = f**3, capacity.shape[0]
    bits = np.asarray(state['cleared_bits'])
    ids = np.asarray(state['partial_ids'])
    values = np.asarray(state['partial_ionized'], dtype=float)
    if bits.dtype != np.uint64 or bits.shape != (capacity.size, (sub+63)//64):
        raise ValueError('Invalid cleared-cell bit mask')
    if sub < 64 and (bits >> np.uint64(sub)).any():
        raise ValueError('Cleared-cell mask has nonzero padding bits')
    if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or (ids < 0).any()
            or (ids >= (n*f)**3).any() or values.shape != ids.shape
            or not np.isfinite(values).all() or (values < 0).any()):
        raise ValueError('Invalid partial fine-cell counts')
    order = np.argsort(ids)
    ids, values = ids[order].astype(np.int64), values[order]
    if len(ids) > 1 and (np.diff(ids) <= 0).any():
        raise ValueError('Partial fine-cell IDs must be unique')
    parent, child = _geometry(ids, n, f)
    if ((values > capacity.ravel()[parent]/sub).any()
            or ((bits[parent, child//64] >> (child % 64).astype(np.uint64)) & np.uint64(1)).any()):
        raise ValueError('Partial counts exceed capacity or overlap a cleared cell')
    return bits, ids, values


def fine_state_metrics(capacity, subcells_per_axis, new_state, old_state, emitted):
    """Exact L1 on identical fine meshes from masks and sparse partial cells.

    A cleared/neutral disagreement contributes one full subcell capacity.
    Remove all cells listed as partial in either state from this mask term,
    then add their actual absolute count differences. No parent averaging or
    fine floating-point grid is used.
    """
    c = np.asarray(capacity, dtype=float)
    f = subcells_per_axis
    if (isinstance(f, (bool, np.bool_)) or not isinstance(f, (int, np.integer))
            or f not in (2, 4, 8) or c.ndim != 3 or len(set(c.shape)) != 1
            or not c.size or not np.isfinite(c).all() or (c <= 0).any()
            or not np.isfinite(emitted) or emitted <= 0):
        raise ValueError('Positive cubic capacities, supported refinement and emitted budget required')
    f, sub, n = int(f), int(f)**3, c.shape[0]
    bn, ni, nv = _validate(new_state, c, f)
    bo, oi, ov = _validate(old_state, c, f)
    ids = np.union1d(ni, oi)
    parent, child = _geometry(ids, n, f)
    mask = np.left_shift(np.uint64(1), (child % 64).astype(np.uint64))
    cn = (bn[parent, child//64] & mask) != 0
    co = (bo[parent, child//64] & mask) != 0
    cell_capacity = c.ravel()[parent]/sub
    an, ao = cell_capacity*cn, cell_capacity*co
    an[np.searchsorted(ids, ni)] = nv
    ao[np.searchsorted(ids, oi)] = ov
    partial_delta = abs(an-ao)
    xor = np.bitwise_xor(bn, bo)
    # Several partial IDs can share a parent word: .at is required here.
    np.bitwise_and.at(xor, (parent, child//64), np.bitwise_not(mask))
    counts = np.empty(c.size, dtype=np.int64)
    byte_masks = np.ascontiguousarray(xor).view(np.uint8)
    for start in range(0, c.size, 32768):
        counts[start:start+32768] = _POPCOUNT[byte_masks[start:start+32768]].sum(axis=1)
    full_l1 = np.sum(counts.astype(np.longdouble)*c.ravel()/sub, dtype=np.longdouble)
    partial_l1 = np.sum(partial_delta, dtype=np.longdouble)
    fraction_delta = partial_delta/cell_capacity
    return dict(subcell_L1_over_emitted=float((full_l1+partial_l1)/emitted),
                volume_mean_absolute_ionized_fraction_difference=float(
                    (np.sum(counts, dtype=np.longdouble)+np.sum(fraction_delta, dtype=np.longdouble))/(c.size*sub)),
                maximum_ionized_fraction_difference=max(
                    1. if counts.any() else 0., float(fraction_delta.max()) if ids.size else 0.),
                fine_grid=n*f,
                scope='Resolved subcell counts on the identical gas mesh, before parent averaging')
