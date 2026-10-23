"""Geometric path adapter for the finite-opacity research solver.

All positions and lengths use the same physical length unit.  Unlike assigning
one optical depth to a cell for every beam, this adapter supplies the actual
chord length for each ray/cell intersection.  Gas is piecewise constant.  Open
boundaries only: periodic loops require a different coupled transport solve.
This is a geometry adapter, not an ionization or source model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Segments:
    cells: tuple[int, ...]
    lengths: tuple[float, ...]
    entries: tuple[float, ...]
    distance: float
    terminal: str


def trace_segments(shape, origin, direction, *, cell_width=1., maximum_distance=None):
    """Enumerate positive-length voxel chords; no geometric nudges or tie bands.

    Coordinates start at the lower corner of the grid.  An origin on an internal
    face belongs to the cell immediately ahead of the ray; zero-length touches
    are excluded.  Exact corner crossings advance all exactly tied axes.
    """
    shape_arr = np.asarray(shape)
    if (shape_arr.shape != (3,) or not np.issubdtype(shape_arr.dtype, np.integer)
            or np.any(shape_arr < 1)):
        raise ValueError("shape must contain three positive integers")
    if not np.isfinite(cell_width) or cell_width <= 0:
        raise ValueError("cell_width must be finite and positive")
    start = np.asarray(origin, dtype=np.longdouble)
    velocity = np.asarray(direction, dtype=np.longdouble)
    if start.shape != (3,) or velocity.shape != (3,):
        raise ValueError("origin and direction must be 3-vectors")
    if not np.isfinite(start).all() or not np.isfinite(velocity).all():
        raise ValueError("finite origin and direction required")
    if not np.isclose(np.linalg.norm(velocity), 1, rtol=0, atol=1e-12):
        raise ValueError("unit direction required")
    width = np.longdouble(cell_width)
    extent = shape_arr.astype(np.longdouble) * width
    if np.any(start < 0) or np.any(start > extent):
        raise ValueError("origin must be inside or on the box")
    horizon = np.longdouble(np.inf if maximum_distance is None else maximum_distance)
    if horizon <= 0 or np.isnan(horizon) or (maximum_distance is not None and not np.isfinite(horizon)):
        raise ValueError("maximum_distance must be finite and positive when specified")
    scaled = start / width
    cell = np.floor(scaled).astype(np.int64)
    # Direction-aware ownership at exact grid planes needs no displacement.
    cell[(scaled == np.floor(scaled)) & (velocity < 0)] -= 1
    step = np.sign(velocity).astype(np.int64)
    plane = cell.copy()
    plane[step > 0] += 1
    moving = step != 0
    crossing = np.full(3, np.longdouble(np.inf))
    crossing[moving] = (plane[moving] * width - start[moving]) / velocity[moving]
    entries, lengths, cells = [], [], []
    entered = np.longdouble(0)
    # An open straight line crosses at most sum(shape) internal/boundary planes.
    for _ in range(int(shape_arr.sum()) + 4):
        if np.any(cell < 0) or np.any(cell >= shape_arr):
            return Segments(tuple(cells), tuple(lengths), tuple(entries), float(entered), "escaped")
        if entered >= horizon:
            return Segments(tuple(cells), tuple(lengths), tuple(entries), float(horizon), "range_limited")
        next_crossing = crossing.min()
        end = min(next_crossing, horizon)
        if end > entered:
            cells.append(int(np.ravel_multi_index(tuple(cell), tuple(shape_arr))))
            entries.append(float(entered))
            lengths.append(float(end - entered))
        if next_crossing > horizon:
            return Segments(tuple(cells), tuple(lengths), tuple(entries), float(horizon), "range_limited")
        tied = crossing == next_crossing
        cell[tied] += step[tied]
        plane[tied] += step[tied]
        crossing[tied] = (plane[tied] * width - start[tied]) / velocity[tied]
        entered = next_crossing
    raise RuntimeError("open ray exceeded the plane-crossing bound")


def build_paths(neutral_absorption_coefficient, origins, directions, *, cell_width=1., maximum_distance=None):
    """Return cell paths and per-intersection neutral optical depths.

    ``neutral_absorption_coefficient`` is sigma*n_H, in inverse length, at x=0.
    ``origins`` and ``directions`` contain one entry per beam; the caller assigns
    rates/solid-angle weights.  Flattening follows NumPy C ordering.
    """
    coefficient = np.asarray(neutral_absorption_coefficient, dtype=float)
    origins, directions = np.asarray(origins), np.asarray(directions)
    if coefficient.ndim != 3 or not np.isfinite(coefficient).all() or (coefficient < 0).any():
        raise ValueError("finite nonnegative 3D absorption coefficient required")
    if origins.ndim != 2 or origins.shape[1:] != (3,) or origins.shape != directions.shape:
        raise ValueError("one origin and direction per beam required")
    paths, path_tau, geometry = [], [], []
    for origin, direction in zip(origins, directions):
        segments = trace_segments(coefficient.shape, origin, direction,
                                  cell_width=cell_width, maximum_distance=maximum_distance)
        paths.append(list(segments.cells))
        path_tau.append((coefficient.ravel()[list(segments.cells)] * segments.lengths).tolist())
        geometry.append(segments)
    return paths, path_tau, geometry
