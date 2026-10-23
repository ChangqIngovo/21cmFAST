"""Finite-c, persistent, straight-ray packet reference for a fixed voxel gas.

This is a research reference, not a production 21cmFAST replacement. Neutral
cells absorb incident photon impulses at their entry face until their capacity
is exhausted; transparent cells remain transparent (NO recombinations). Sources
are at cell centres. Constant rates are integrated by midpoint emission packets,
so emission-time refinement is separate from voxel and angular convergence.

The queue stores the original emission time, source, direction and cumulative
path, including after a source switches off. Observing at extra times does not
resample photons. A numerical range stop is explicitly unresolved radiation,
never fresh source luminosity, escape, or absorbed radiation. Use a new solver
if the gas, opacity law, directions, speed or range policy changes.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
import math

import numpy as np

from characteristics import directions

LD = np.longdouble
FLOAT_MAX = LD(np.finfo(float).max)


@dataclass(frozen=True)
class _Path:
    cells: np.ndarray
    entries: np.ndarray
    terminal_distance: np.longdouble
    terminal_kind: str


@dataclass
class _Packet:
    uid: int
    source: int
    angle: int
    emission_time: np.longdouble
    photons: np.longdouble
    cursor: int = 0
    status: str = "in_flight"


class FiniteCPackets:
    """Monotone-opacity packet evolution with exact queue times for each packet.

    All lengths, speeds, times, photon counts and capacities must use a single
    consistent unit system. ``advance`` takes an absolute observation time.
    The only accepted gas evolution is absorption: changing capacities between
    calls, recombination, scattering, redshifting and density advection are not
    implemented. ``capacity`` is copied to prevent an external array mutation
    from silently changing the opacity of already scheduled rays.
    """

    def __init__(self, capacity, *, speed=1.0, cell_width=1.0,
                 ray_directions=None, angles=192, ray_weights=None,
                 periodic=False, maximum_distance=None, max_packets=2_000_000):
        cap = np.asarray(capacity, dtype=float)
        if cap.ndim != 3 or len(set(cap.shape)) != 1 or cap.shape[0] < 1:
            raise ValueError("A nonempty cubic capacity field is required")
        if not np.isfinite(cap).all() or np.any(cap < 0):
            raise ValueError("Finite nonnegative capacities are required")
        if not np.isfinite(speed) or speed <= 0 or speed > FLOAT_MAX:
            raise ValueError("Finite positive propagation speed is required")
        if (not np.isfinite(cell_width) or cell_width <= 0
                or LD(cell_width)*cap.shape[0] > FLOAT_MAX):
            raise ValueError("Finite positive cell width is required")
        if ray_directions is None:
            if not isinstance(angles, (int, np.integer)) or angles < 1:
                raise ValueError("A positive integer angle count is required")
            dirs = directions(angles)
        else:
            dirs = np.asarray(ray_directions, dtype=float)
        if dirs.ndim != 2 or dirs.shape[1] != 3 or len(dirs) < 1:
            raise ValueError("Directions must have shape (K, 3)")
        if (not np.isfinite(dirs).all() or
                not np.allclose(np.linalg.norm(dirs, axis=1), 1,
                                atol=1e-12, rtol=0)):
            raise ValueError("Finite unit directions are required")
        if ray_weights is None:
            weights = np.ones(len(dirs), dtype=LD) / len(dirs)
        else:
            weights = np.asarray(ray_weights, dtype=LD)
            if (weights.shape != (len(dirs),) or not np.isfinite(weights).all()
                    or np.any(weights <= 0)
                    or not np.isclose(weights.sum(), 1, atol=1e-12, rtol=0)):
                raise ValueError("Positive angular weights summing to one are required")
            weights = weights / weights.sum()
        if maximum_distance is None:
            maximum_distance = math.inf
        if (maximum_distance <= 0 or np.isnan(maximum_distance)
                or (np.isfinite(maximum_distance) and maximum_distance > FLOAT_MAX)):
            raise ValueError("Positive range is required")
        if periodic and not np.isfinite(maximum_distance):
            raise ValueError("Periodic transport requires a finite cumulative range")
        if not isinstance(max_packets, (int, np.integer)) or max_packets < 1:
            raise ValueError("Positive integer max_packets required")
        self.max_packets = int(max_packets)
        self._scheduled_packet_count = 0
        self.n = cap.shape[0]
        self.shape = cap.shape
        self.speed, self.cell_width = LD(speed), LD(cell_width)
        self.directions = dirs.copy()
        self.weights = weights.copy()
        self.periodic = bool(periodic)
        self.maximum_distance = LD(maximum_distance)
        self._remaining = cap.ravel().astype(LD).copy()
        self._remaining_compensation = np.zeros(cap.size, dtype=LD)
        self._initial_capacity = self._remaining.copy()
        self._absorbed = np.zeros(cap.size, dtype=LD)
        self._source_emitted = np.zeros(cap.size, dtype=LD)
        self._source_absorbed = np.zeros(cap.size, dtype=LD)
        self._source_escaped = np.zeros(cap.size, dtype=LD)
        self._source_retained = np.zeros(cap.size, dtype=LD)
        self._queue = []
        self._sequence = itertools.count()
        self._packet_ids = itertools.count()
        self._active = {}
        self._stopped = {}
        self._paths = {}
        self.time = LD(0)
        self.processed_arrivals = 0
        self.materialized_packets = 0
        self.scheduled_photons = LD(0)

    def _source_index(self, source_cell):
        cell = np.asarray(source_cell)
        if cell.shape != (3,) or not np.issubdtype(cell.dtype, np.integer):
            raise ValueError("source_cell must contain three integer grid indices")
        if np.any(cell < 0) or np.any(cell >= self.n):
            raise ValueError("Source lies outside the grid")
        return int(np.ravel_multi_index(tuple(cell), self.shape))

    def _push(self, time, kind, data):
        heapq.heappush(self._queue, (LD(time), next(self._sequence), kind, data))

    def _check_emission_budget(self, additional):
        total = self.scheduled_photons + LD(additional)
        if not np.isfinite(total) or total > FLOAT_MAX:
            raise ValueError("Scheduled photon budget exceeds finite float64 output range")

    def schedule_flash(self, source_cell, photons, *, emission_time=0.0):
        """Register one isotropic (or supplied angular-weight) delta emission."""
        source = self._source_index(source_cell)
        if not np.isfinite(photons) or photons < 0:
            raise ValueError("Finite nonnegative photon count is required")
        if (not np.isfinite(emission_time) or LD(emission_time) < self.time
                or emission_time > FLOAT_MAX):
            raise ValueError("Cannot schedule emission before the current solver time")
        if photons:
            self._check_emission_budget(photons)
            if self._scheduled_packet_count + len(self.directions) > self.max_packets:
                raise ValueError("Packet reference budget exceeded; reduce the problem or explicitly raise max_packets")
            self._scheduled_packet_count += len(self.directions)
            self._push(emission_time, "emission", (source, LD(photons)))
            self.scheduled_photons += LD(photons)

    def schedule_constant(self, source, start, stop, *, emission_steps):
        """Midpoint quadrature of a constant rate over [start, stop].

        Each source emits rate*(stop-start)/emission_steps at each midpoint.
        Create the emission schedule independently of observation/output times;
        subdividing output alone must never change the source quadrature.
        Overlapping schedules mean additive emissions and are intentionally
        permitted. A cumulative photon field is NOT a rate field.
        """
        rates = np.asarray(source, dtype=float)
        if rates.shape != self.shape or not np.isfinite(rates).all() or np.any(rates < 0):
            raise ValueError("Finite nonnegative source rates of matching shape are required")
        if (not np.isfinite(start) or not np.isfinite(stop)
                or LD(start) < self.time or stop <= start or stop > FLOAT_MAX):
            raise ValueError("An increasing source interval starting at/after current time is required")
        if not isinstance(emission_steps, (int, np.integer)) or emission_steps < 1:
            raise ValueError("Positive integer emission_steps required")
        start, stop = LD(start), LD(stop)
        dt = (stop - start) / emission_steps
        positive = np.flatnonzero(rates.ravel())
        self._check_emission_budget(rates.astype(LD).sum() * (stop-start))
        count = len(positive) * int(emission_steps) * len(self.directions)
        if self._scheduled_packet_count + count > self.max_packets:
            raise ValueError("Packet reference budget exceeded; reduce the problem or explicitly raise max_packets")
        self._scheduled_packet_count += count
        for step in range(emission_steps):
            stamp = start + (LD(step) + LD('.5')) * dt
            for source_index in positive:
                amount = LD(rates.ravel()[source_index]) * dt
                self._push(stamp, "emission", (int(source_index), amount))
                self.scheduled_photons += amount

    def _path(self, source, angle):
        key = (source, angle)
        if key in self._paths:
            return self._paths[key]
        cursor = np.array(np.unravel_index(source, self.shape), dtype=np.int64)
        origin = (cursor.astype(LD) + LD('.5')) * self.cell_width
        direction = self.directions[angle].astype(LD)
        step = np.sign(direction).astype(np.int64)
        distance = LD(0)
        cells, entries = [], []
        terminal_kind = "range_retained"
        # Infinite open rays leave a finite cube. Periodic rays have a finite
        # user range, but reject unreasonable traversal before memory exhaustion.
        for _ in range(1_000_000):
            if distance >= self.maximum_distance:
                break
            cells.append(int(np.ravel_multi_index(tuple(cursor % self.n), self.shape)))
            entries.append(distance)
            plane = (cursor + (step > 0)).astype(LD) * self.cell_width
            crossing = np.full(3, np.inf, dtype=LD)
            moving = step != 0
            crossing[moving] = (plane[moving] - origin[moving]) / direction[moving]
            nxt = crossing.min()
            next_cursor = cursor.copy()
            next_cursor[crossing == nxt] += step[crossing == nxt]
            exits = (not self.periodic and
                     (np.any(next_cursor < 0) or np.any(next_cursor >= self.n)))
            if exits and nxt <= self.maximum_distance:
                distance, terminal_kind = nxt, "escaped"
                break
            if nxt >= self.maximum_distance:
                distance = self.maximum_distance
                break
            if nxt <= distance:
                raise RuntimeError("DDA failed to make strictly positive progress")
            cursor, distance = next_cursor, nxt
        else:
            raise RuntimeError("Path exceeds one million crossed voxels")
        path = _Path(np.asarray(cells, dtype=np.int64),
                     np.asarray(entries, dtype=LD), distance, terminal_kind)
        self._paths[key] = path
        return path

    def _schedule_next(self, packet):
        path = self._path(packet.source, packet.angle)
        # Transparency is permanent in this model. Skipping cells that are
        # already empty cannot miss a future absorber; recombination would make
        # this optimization invalid and is deliberately unsupported.
        while (packet.cursor < len(path.cells)
               and self._remaining[path.cells[packet.cursor]] <= 0):
            packet.cursor += 1
        if packet.cursor < len(path.cells):
            time = packet.emission_time + path.entries[packet.cursor] / self.speed
            self._push(time, "arrival", packet)
        else:
            time = packet.emission_time + path.terminal_distance / self.speed
            self._push(time, path.terminal_kind, packet)

    def _born(self, source, photons, time):
        self._source_emitted[source] += photons
        packets = []
        # Last beam receives the rounding residual, preserving emission count.
        assigned = LD(0)
        for angle in range(len(self.directions)):
            amount = (photons - assigned if angle == len(self.directions)-1
                      else photons * self.weights[angle])
            assigned += amount
            packet = _Packet(next(self._packet_ids), source, angle, time, amount)
            self._active[packet.uid] = packet
            packets.append(packet)
        self.materialized_packets += len(packets)
        return packets

    def advance(self, until):
        """Process emissions and arrivals up to an absolute time, inclusively."""
        if not np.isfinite(until) or LD(until) < self.time or until > FLOAT_MAX:
            raise ValueError("Finite nondecreasing absolute observation time required")
        until = LD(until)
        while self._queue and self._queue[0][0] <= until:
            stamp = self._queue[0][0]
            groups = {}
            # Equal internal long-double event times are genuinely simultaneous.
            # No broad tolerance merges nearby, physically ordered arrivals.
            while self._queue and self._queue[0][0] == stamp:
                _, _, kind, item = heapq.heappop(self._queue)
                if kind == "emission":
                    source, photons = item
                    for packet in self._born(source, photons, stamp):
                        groups.setdefault(source, []).append(packet)
                elif kind == "arrival":
                    path = self._path(item.source, item.angle)
                    cell = int(path.cells[item.cursor])
                    groups.setdefault(cell, []).append(item)
                    self.processed_arrivals += 1
                else:
                    item.status = kind
                    del self._active[item.uid]
                    if kind == "escaped":
                        self._source_escaped[item.source] += item.photons
                    else:
                        self._source_retained[item.source] += item.photons
                        self._stopped[item.uid] = item
            for cell, packets in groups.items():
                total = sum((p.photons for p in packets), LD(0))
                loss = min(self._remaining[cell], total)
                fraction = loss / total if total else LD(0)
                spent = LD(0)
                # Consume shared capacity proportionally, rather than giving
                # whichever source happened to enter the heap first priority.
                for number, packet in enumerate(packets):
                    absorbed = (loss-spent if number == len(packets)-1
                                else packet.photons * fraction)
                    spent += absorbed
                    packet.photons -= absorbed
                    self._source_absorbed[packet.source] += absorbed
                    if packet.photons <= 0:
                        packet.photons = LD(0)
                        packet.status = "absorbed"
                        del self._active[packet.uid]
                    else:
                        packet.cursor += 1
                        self._schedule_next(packet)
                self._absorbed[cell] += spent
                # Kahan subtraction preserves weak consumption against large
                # remaining capacities; absorption is stored directly as well.
                delta = -spent - self._remaining_compensation[cell]
                updated = self._remaining[cell] + delta
                self._remaining_compensation[cell] = (updated-self._remaining[cell])-delta
                self._remaining[cell] = max(LD(0), updated)
        self.time = until
        return self.snapshot()

    def snapshot(self, *, include_packets=False):
        """Independent atom/source ledgers, plus optional full radiation state."""
        inflight = np.zeros(self.n**3, dtype=LD)
        for packet in self._active.values():
            inflight[packet.source] += packet.photons
        accounted = (self._source_absorbed + self._source_escaped
                     + self._source_retained + inflight)
        def field(a):
            return np.asarray(a, dtype=float).reshape(self.shape).copy()
        audit = dict(time=float(self.time), emitted=float(self._source_emitted.sum()),
                     absorbed=float(self._absorbed.sum()),
                     escaped=float(self._source_escaped.sum()),
                     range_retained=float(self._source_retained.sum()),
                     in_flight=float(inflight.sum()),
                     ledger_error=float(self._source_emitted.sum()-accounted.sum()),
                     per_source_ledger_max=float(np.max(np.abs(self._source_emitted-accounted))),
                     absorption_ledger_error=float(self._absorbed.sum()-self._source_absorbed.sum()),
                     capacity_excess=float(np.maximum(self._absorbed-self._initial_capacity, 0).max()),
                     materialized_packets=self.materialized_packets,
                     active_packets=len(self._active), range_stopped_packets=len(self._stopped),
                     processed_arrivals=self.processed_arrivals, cached_paths=len(self._paths),
                     workers=1, packet_budget=self.max_packets,
                     model="finite-c impulsive emission, sharp opaque voxel front, fixed gas, no recombination")
        result = dict(absorbed=field(self._absorbed), remaining_capacity=field(self._remaining),
                      source_emitted=field(self._source_emitted),
                      source_absorbed=field(self._source_absorbed),
                      source_escaped=field(self._source_escaped),
                      source_range_retained=field(self._source_retained),
                      source_in_flight=field(inflight), audit=audit)
        if include_packets:
            states = []
            for packet in itertools.chain(self._active.values(), self._stopped.values()):
                path = self._path(packet.source, packet.angle)
                travelled = min(max(LD(0), (self.time-packet.emission_time)*self.speed),
                                path.terminal_distance)
                origin = (np.array(np.unravel_index(packet.source, self.shape), dtype=LD)
                          + LD('.5')) * self.cell_width
                position = origin + travelled * self.directions[packet.angle]
                states.append(dict(source=[int(i) for i in np.unravel_index(packet.source, self.shape)],
                                   direction=self.directions[packet.angle].tolist(),
                                   emission_time=float(packet.emission_time),
                                   photons=float(packet.photons), travelled=float(travelled),
                                   position=np.asarray(position, dtype=float).tolist(),
                                   status=packet.status))
            result["packets"] = states
        return result
