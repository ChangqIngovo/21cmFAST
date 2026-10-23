"""Positive instantaneous-rate histories and persistent emission bookkeeping.

This research adapter chooses a new temporal closure: integrate instantaneous
escaped SFRD-derived photon *rates*.  Legacy cumulative nion is neither a rate
nor an additional emission.  The production global-budget path is unchanged.

Times are finite, increasing offsets in one explicitly named clock.  Rates are
linear within each supplied interval, with independent endpoint values so that
declared source switches can be represented without smearing a discontinuity.
No information between samples is invented beyond that interpolation model.
"""

from __future__ import annotations

import hashlib

import numpy as np


def _array(value, name):
    result = np.array(value, dtype=np.float64, copy=True)
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _float64(value, name):
    """Reject overflow instead of silently emitting an invalid photon budget."""
    value = np.asarray(value, dtype=np.longdouble)
    if not np.isfinite(value).all() or np.any(value > np.finfo(float).max):
        raise OverflowError(f"{name} exceeds the finite float64 budget range")
    result = np.asarray(value, dtype=np.float64)
    # Nonzero increments that cannot be represented must not disappear silently.
    if np.any((value != 0) & (result == 0)):
        raise FloatingPointError(f"{name} is below the float64 budget range")
    return result


def _add_compensated(high, low, increment):
    summed = high + increment
    correction = np.where(
        np.abs(high) >= np.abs(increment),
        (high - summed) + increment,
        (increment - summed) + high,
    )
    if not np.isfinite(summed).all():
        raise OverflowError("Cumulative emission exceeds float64 range")
    return summed, low + correction


class RateHistory:
    """Piecewise linear nonnegative rate field with explicitly declared jumps.

    ``times`` has N+1 entries; ``left_rates`` and ``right_rates`` have N
    fields.  A discontinuity at times[k] is represented by different values of
    right_rates[k-1] and left_rates[k].  Values exactly at a jump do not affect
    its time integral.  Absolute cosmological epochs should first be converted
    to offsets, avoiding subtraction of nearly identical large timestamps.
    """

    def __init__(self, times, left_rates, right_rates, *, clock="relative seconds"):
        self.times = np.array(times, dtype=np.float64, copy=True)
        if (self.times.ndim != 1 or self.times.size < 2
                or not np.isfinite(self.times).all()
                or np.any(np.diff(self.times) <= 0)):
            raise ValueError("At least two strictly increasing finite times required")
        self.left = _array(left_rates, "Left rates")
        self.right = _array(right_rates, "Right rates")
        if (self.left.shape != self.right.shape or self.left.ndim < 1
                or self.left.shape[0] != self.times.size - 1):
            raise ValueError("One matching left/right rate field per time interval required")
        if not isinstance(clock, str) or not clock.strip():
            raise ValueError("An explicit nonempty clock label is required")
        self.clock = clock
        self.shape = self.left.shape[1:]
        digest = hashlib.sha256(clock.encode())
        digest.update(str(self.left.shape).encode())
        for value in (self.times, self.left, self.right):
            digest.update(value.tobytes(order="C"))
            value.flags.writeable = False
        self.fingerprint = digest.hexdigest()

    @classmethod
    def from_knots(cls, times, rates, *, clock="relative seconds"):
        rates = _array(rates, "Knot rates")
        if rates.ndim < 1 or len(rates) != len(times):
            raise ValueError("One rate field per knot required")
        return cls(times, rates[:-1], rates[1:], clock=clock)

    def _time(self, value):
        value = float(value)
        if not np.isfinite(value) or not self.times[0] <= value <= self.times[-1]:
            raise ValueError("Time lies outside the declared rate history")
        return value

    def rate_at(self, time, *, side="right"):
        time = self._time(time)
        if side not in ("left", "right"):
            raise ValueError("side must be left or right")
        index = int(np.searchsorted(self.times, time, side=side) - 1)
        index = min(max(index, 0), len(self.left) - 1)
        a, b = self.times[index:index + 2].astype(np.longdouble)
        fraction = (np.longdouble(time) - a) / (b - a)
        value = (1 - fraction) * self.left[index].astype(np.longdouble)
        value += fraction * self.right[index].astype(np.longdouble)
        return _float64(value, "Interpolated rate")

    def integrate(self, start, stop):
        """Exact nonnegative integral of the declared linear interpolation.

        This is not an error bound on an unknown physical source between knots.
        Long-double products avoid an overflowing sum of endpoints when the
        duration is small.  Returned increments are not cumulative differences.
        """
        start, stop = self._time(start), self._time(stop)
        if stop < start:
            raise ValueError("Source time cannot run backwards")
        result = np.zeros(self.shape, dtype=np.longdouble)
        if stop == start:
            return result.astype(float)
        first = max(0, np.searchsorted(self.times, start, side="right") - 1)
        last = min(len(self.left), np.searchsorted(self.times, stop, side="left"))
        for index in range(first, last):
            a, b = self.times[index:index + 2].astype(np.longdouble)
            lo, hi = max(np.longdouble(start), a), min(np.longdouble(stop), b)
            f0, f1 = (lo - a) / (b - a), (hi - a) / (b - a)
            left = self.left[index].astype(np.longdouble)
            right = self.right[index].astype(np.longdouble)
            average_fraction = (f0 + f1) / 2
            result += (hi - lo) * ((1 - average_fraction) * left + average_fraction * right)
        return _float64(result, "Emission increment")

    def append(self, times, left_rates, right_rates):
        """Add a future segment when its source/feedback sample becomes known.

        The existing history is immutable.  The first new timestamp must equal
        its last endpoint; a source jump there is allowed and explicitly stored.
        """
        extension = RateHistory(times, left_rates, right_rates, clock=self.clock)
        if extension.times[0] != self.times[-1] or extension.shape != self.shape:
            raise ValueError("Extension must start at the existing endpoint with the same grid")
        return RateHistory(np.r_[self.times, extension.times[1:]],
                           np.concatenate((self.left, extension.left)),
                           np.concatenate((self.right, extension.right)), clock=self.clock)


class SourceLedger:
    """Persistent emission cursor for fixed gas, independent of transport state.

    This is an *emission* ledger, not an ionization or in-flight-photon ledger.
    Transport must separately retain its own state.  The atom field is frozen;
    changing density requires an explicit conservative gas-remapping method.
    The two compensated cumulative arrays are saved separately so a small new
    emission does not disappear by subtracting two large cumulative totals.
    """

    def __init__(self, history, atoms, *, start=None):
        if not isinstance(history, RateHistory):
            raise TypeError("A RateHistory is required")
        self.history = history
        self.atoms = _array(atoms, "Atom field")
        if self.atoms.shape != history.shape:
            raise ValueError("Atom field and source field shapes must agree")
        self.atoms.flags.writeable = False
        self.atoms_sha256 = hashlib.sha256(self.atoms.tobytes()).hexdigest()
        self.start = history._time(history.times[0] if start is None else start)
        self.time = self.start
        self.emitted_high = np.zeros(history.shape)
        self.emitted_low = np.zeros(history.shape)

    def advance(self, stop, *, atoms=None):
        """Return fresh emission over [current time, stop], then advance cursor."""
        if atoms is not None:
            checked = _array(atoms, "Atom field")
            if checked.shape != self.atoms.shape or not np.array_equal(checked, self.atoms):
                raise ValueError("Changing gas requires conservative advection/remapping; fixed-gas ledger rejected it")
        stop = self.history._time(stop)
        increment = self.history.integrate(self.time, stop)
        high, low = _add_compensated(self.emitted_high, self.emitted_low, increment)
        self.emitted_high, self.emitted_low = high, low
        self.time = stop
        return increment

    def extend_history(self, history):
        """Accept newly available future rates without changing past emission."""
        old = self.history
        size = len(old.left)
        if (not isinstance(history, RateHistory) or history.shape != old.shape
                or history.clock != old.clock or len(history.left) < size
                or not np.array_equal(history.times[:size + 1], old.times)
                or not np.array_equal(history.left[:size], old.left)
                or not np.array_equal(history.right[:size], old.right)):
            raise ValueError("A source-history extension must preserve the complete existing prefix")
        self.history = history

    @property
    def emitted(self):
        return self.emitted_high.astype(np.longdouble) + self.emitted_low.astype(np.longdouble)

    def checkpoint(self):
        return dict(version=1, history_sha256=self.history.fingerprint,
                    atoms_sha256=self.atoms_sha256, shape=list(self.history.shape),
                    clock=self.history.clock, start=self.start, time=self.time,
                    emitted_high=self.emitted_high.tolist(), emitted_low=self.emitted_low.tolist())

    @classmethod
    def from_checkpoint(cls, history, atoms, state):
        ledger = cls(history, atoms, start=state["start"])
        if (state["version"] != 1 or state["history_sha256"] != history.fingerprint
                or state["atoms_sha256"] != ledger.atoms_sha256
                or state["shape"] != list(history.shape) or state["clock"] != history.clock):
            raise ValueError("Checkpoint source history, gas, shape or clock does not match")
        ledger.time = history._time(state["time"])
        if ledger.time < ledger.start:
            raise ValueError("Checkpoint precedes its emission start")
        high = _array(state["emitted_high"], "Cumulative high part")
        low = np.asarray(state["emitted_low"], dtype=float)
        if high.shape != history.shape or low.shape != history.shape or not np.isfinite(low).all():
            raise ValueError("Invalid cumulative checkpoint arrays")
        # A content-valid checkpoint also has to agree with the declared source
        # integral. This catches accidental edits, not malicious forgery.
        expected = history.integrate(ledger.start, ledger.time)
        actual = high.astype(np.longdouble) + low.astype(np.longdouble)
        tolerance = 16 * np.finfo(float).eps * np.maximum(expected, np.finfo(float).tiny)
        if np.any(np.abs(actual - expected) > tolerance):
            raise ValueError("Checkpoint cumulative emission does not match the source history")
        ledger.emitted_high, ledger.emitted_low = high.copy(), low.copy()
        return ledger


def native_sfrd_history(times, sfrd_acg, sfrd_mcg, *, astro, rho_b,
                        clock="native dtdz matter+Lambda; relative seconds"):
    """Qin20 instantaneous rate closure, continuous linear interpolation in time.

    SFRD arrays have shape (number of times, *source grid). Native SFRD units
    are Msun/s/cMpc^3 and already contain Eulerian density. Returned effective
    rates use the native stellar-yield normalization; no H/He reinterpretation
    or global rescaling to legacy cumulative nion is performed.
    """
    from validation_native_rates import uv_rates
    rates = uv_rates(sfrd_acg, sfrd_mcg, astro=astro, rho_b=rho_b)
    return RateHistory.from_knots(times, rates["acg"] + rates["mcg"], clock=clock)
