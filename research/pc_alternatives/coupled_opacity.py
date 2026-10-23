"""Conservative finite-opacity coupling on prescribed non-cyclic characteristics.

Research implementation: fixed cells, pure H, fixed temperature, case-B local
recombination, infinite c. A cell is homogeneous, not opaque-until-complete.
Each beam's entire photo-only flow is exact in incident fluence. Symmetric
operator splitting couples beams sharing cells and recombinations; adaptive
step doubling controls this splitting error. No native Qin20 feedback is used.
"""
from __future__ import annotations

import math
import time
import numpy as np


def _prepare(atoms, initial, alpha_n, tau_neutral, paths, rates, path_tau=None):
    atoms, x, alpha, tau = [np.asarray(v, dtype=float).copy() for v in
                           (atoms, initial, alpha_n, tau_neutral)]
    if atoms.ndim != 1 or not atoms.size or any(v.shape != atoms.shape for v in (x, alpha, tau)):
        raise ValueError('Nonempty equally shaped one-dimensional cell arrays required')
    if not all(np.isfinite(v).all() for v in (atoms, x, alpha, tau)):
        raise ValueError('Finite cell values required')
    if not np.isfinite(np.sum(atoms, dtype=np.longdouble)) or np.sum(atoms, dtype=np.longdouble) > np.finfo(float).max:
        raise ValueError('Aggregate atoms exceed supported floating-point range')
    if np.any(atoms <= 0) or np.any(alpha < 0) or np.any(tau < 0) or np.any((x < 0) | (x > 1)):
        raise ValueError('Positive atoms, nonnegative opacity/recombination, and 0<=x<=1 required')
    rates = np.asarray(rates, dtype=float).copy()
    if rates.shape != (len(paths),) or not np.isfinite(rates).all() or np.any(rates < 0):
        raise ValueError('One finite nonnegative rate per path required')
    checked = []
    for path in paths:
        p = np.asarray(path)
        if p.ndim != 1 or (p.size and (not np.issubdtype(p.dtype, np.integer)
                        or np.any((p < 0) | (p >= atoms.size)))):
            raise ValueError('Paths must contain valid integer cell indices')
        p = p.astype(np.int64)
        if len(set(p.tolist())) != p.size:
            raise ValueError('Repeated cells in one beam path are unsupported: exact single-beam flow requires an acyclic path')
        checked.append(p)
    if path_tau is None:
        depths = [tau[p].copy() for p in checked]
    else:
        if len(path_tau) != len(paths):
            raise ValueError('One optical-depth array per beam path required')
        depths = [np.asarray(t, dtype=float).copy() for t in path_tau]
        if any(t.shape != p.shape or not np.isfinite(t).all() or np.any(t < 0)
               for t, p in zip(depths, checked)):
            raise ValueError('Finite nonnegative path_tau matching every path entry required')
    return atoms, x, alpha, tau, checked, rates, depths


def _photo_loss(neutral, fluence, atoms, tau):
    """Exact absorbed count, evaluated without subtracting near-equal neutral states."""
    if neutral <= 0 or fluence <= 0 or tau <= 0:
        return 0.0
    a = tau * neutral
    b = tau * (fluence / atoms)
    if not math.isfinite(a) or not math.isfinite(b):
        # Products may overflow for otherwise finite input; long double retains
        # the logarithmic formula and the saturation limit on usual platforms.
        a, b = np.longdouble(tau) * neutral, np.longdouble(tau) * fluence / atoms
    p = (-np.expm1(-a)) * (-np.expm1(-b))
    if p < 0.9:
        change = -np.log1p(-p)
    else:
        change = -np.logaddexp(-a, np.log(-np.expm1(-a)) - b)
    loss = float(atoms * (change / tau))
    return min(max(loss, 0.0), fluence, atoms * neutral)


def _beam_flow(x, atoms, path, depths, fluence, absorption):
    """Exact photo-only triangular beam flow; return transmitted fluence."""
    incoming = fluence
    absorbed_total = 0.0
    for cell, tau in zip(path, depths):
        loss = _photo_loss(1 - x[cell], incoming, atoms[cell], tau)
        x[cell] = min(1.0, x[cell] + loss / atoms[cell])
        absorption[cell] += loss
        incoming -= loss
        absorbed_total += loss
        if incoming <= 0:
            break
    return incoming, absorbed_total


def _step(x0, atoms, alpha, paths, rates, depths, dt):
    """Positive symmetric composition, including separate integral ledgers."""
    x = x0.copy()
    absorbed = np.zeros_like(x)
    recombined = np.zeros_like(x)
    escape = np.zeros(len(paths))
    beam_absorbed = np.zeros(len(paths))

    def rec(h):
        # The decrement is evaluated directly, including small recombination.
        # Extended exponent range avoids overflow in alpha*x*h; direct state
        # division retains the tiny x after a very strong recombination step.
        u = alpha.astype(np.longdouble) * x * h
        decrement = np.asarray(x * (u / (1 + u)), dtype=float)
        recombined[:] += atoms * decrement
        x[:] = np.asarray(x / (1 + u), dtype=float)

    rec(dt / 2)
    active = np.flatnonzero(rates > 0)
    if active.size:
        # palindromic Strang across the beam operators. The last operator can
        # combine its two adjacent half steps into one full step.
        schedule = [(b, dt / 2) for b in active[:-1]]
        schedule += [(active[-1], dt)]
        schedule += [(b, dt / 2) for b in active[-2::-1]]
        for b, h in schedule:
            fluence = rates[b] * h
            transmitted, loss = _beam_flow(x, atoms, paths[b], depths[b], fluence, absorbed)
            escape[b] += transmitted
            beam_absorbed[b] += loss
    rec(dt / 2)
    return x, absorbed, recombined, escape, beam_absorbed


def instantaneous_flux(atoms, x, tau_neutral, paths, rates, *, path_tau=None):
    """Beer-Lambert absorption and effective cell-averaged Gamma at a fixed state.

    Gamma=A/[N_H*(1-x)] is evaluated at x=1 by the analytic zero-neutral limit,
    summing F_in*tau0/N_H over all beams. It is not the native ES Gamma model.
    """
    atoms, x, _, _, paths, rates, depths = _prepare(
        atoms, x, np.zeros_like(atoms, dtype=float), tau_neutral, paths, rates, path_tau
    )
    absorbed = np.zeros_like(x)
    incoming_total = np.zeros_like(x)
    gamma = np.zeros_like(x)
    outgoing = np.zeros(len(paths))
    for b, (path, tau_entries) in enumerate(zip(paths, depths)):
        flux = rates[b]
        for cell, tau in zip(path, tau_entries):
            neutral = 1 - x[cell]
            optical = tau * neutral
            fraction = -np.expm1(-optical)
            loss = flux * fraction
            absorbed[cell] += loss
            incoming_total[cell] += flux
            if tau > 0 and flux > 0:
                ratio = np.longdouble(tau) if optical == 0 else np.longdouble(tau) * fraction / optical
                contribution = np.longdouble(flux) * ratio / atoms[cell]
                if contribution > np.finfo(float).max - gamma[cell]:
                    raise FloatingPointError('Gamma exceeds supported floating-point range')
                gamma[cell] += float(contribution)
            flux -= loss
        outgoing[b] = flux
    return dict(absorption_rate=absorbed, incoming_rate=incoming_total,
                Gamma=gamma, outgoing_rate=outgoing)


def solve(atoms, initial, alpha_n, tau_neutral, paths, rates, duration, *,
          path_tau=None, steps=None, rtol=1e-6, atol=1e-9, initial_step=None,
          max_steps=100000, require_resolved_state=True):
    """Evolve one constant-source interval; returns a bounded state and ledgers.

    steps selects a fixed number of second-order steps for convergence tests.
    Otherwise a one-step/two-half-step error estimate controls x and all photon
    integrals; the accepted solution is the two-half-step result, without an
    extrapolation that could break positivity. Tolerances control a local
    estimate, not a rigorous global error bound. Failure raises RuntimeError.
    The default also rejects a state whose count increment is not representable
    accurately in float64 x; require_resolved_state=False is diagnostic only.
    ionized_atom_increment preserves the directly accumulated count increment.
    """
    atoms, x, alpha, tau, paths, rates, depths = _prepare(
        atoms, initial, alpha_n, tau_neutral, paths, rates, path_tau
    )
    initial = x.copy()
    duration = float(duration)
    if not np.isfinite(duration) or duration < 0:
        raise ValueError('Finite nonnegative duration required')
    if not np.isfinite(rates.sum() * duration):
        raise ValueError('Total emitted photons must be representable')
    if steps is not None and (not isinstance(steps, (int, np.integer)) or steps < 1):
        raise ValueError('steps must be a positive integer')
    if not (np.isfinite(rtol) and np.isfinite(atol) and rtol > 0 and atol > 0):
        raise ValueError('Positive finite tolerances required')
    if not isinstance(max_steps, (int, np.integer)) or max_steps < 1:
        raise ValueError('max_steps must be a positive integer')
    if initial_step is not None and (not np.isfinite(initial_step) or initial_step <= 0):
        raise ValueError('initial_step must be positive and finite')
    start = time.perf_counter()
    absorbed, recombined = np.zeros_like(x), np.zeros_like(x)
    escape, beam_absorbed = np.zeros(len(paths)), np.zeros(len(paths))
    accepted, rejected, calls, largest_error = 0, 0, 0, 0.0
    t = 0.0
    h = duration / steps if steps is not None else min(duration, initial_step or duration)
    # Scale integral errors by characteristic counts, not very small cumulative
    # integrals at early times. State error remains controlled independently.
    total = float(rates.sum() * duration)
    integral_scale = max(total, float(atoms.sum()), np.finfo(float).tiny)
    while t < duration:
        if accepted + rejected >= max_steps:
            raise RuntimeError('Accuracy unresolved: max_steps exhausted')
        h = min(h, duration - t)
        if t + h == t or h <= 0:
            raise RuntimeError('Accuracy unresolved: time step underflow')
        coarse = _step(x, atoms, alpha, paths, rates, depths, h)
        calls += 1
        if steps is None:
            first = _step(x, atoms, alpha, paths, rates, depths, h / 2)
            second = _step(first[0], atoms, alpha, paths, rates, depths, h / 2)
            fine = (second[0],) + tuple(first[k] + second[k] for k in range(1, 5))
            calls += 2
            state_error = np.max(np.abs(fine[0] - coarse[0]) /
                                 (atol + rtol * np.maximum(np.abs(x), np.abs(fine[0]))))
            integral_error = max((np.abs(fine[k] - coarse[k]).sum() /
                                (atol * integral_scale + rtol * max(
                                    np.abs(fine[k]).sum(), np.abs(coarse[k]).sum())))
                                 for k in range(1, 5))
            error = max(state_error, integral_error) / 3
            if not np.isfinite(error):
                raise RuntimeError('Non-finite splitting error estimate')
            factor = 2.0 if error == 0 else min(2.0, max(0.1, 0.9 * error ** (-1 / 3)))
            if error > 1:
                rejected += 1
                h *= factor
                continue
            chosen = fine
            largest_error = max(largest_error, float(error))
        else:
            chosen, factor = coarse, 1.0
        x = chosen[0]
        absorbed += chosen[1]
        recombined += chosen[2]
        escape += chosen[3]
        beam_absorbed += chosen[4]
        t += h
        accepted += 1
        if steps is not None:
            # Avoid an artificial tiny last step caused by summing dt.
            t = duration if accepted == steps else duration * accepted / steps
        h *= factor
    flux = instantaneous_flux(atoms, x, tau, paths, rates, path_tau=depths)
    state_discrepancy = np.abs(atoms * (x - initial) - absorbed + recombined)
    processed = absorbed + recombined
    representation_ratio = float(np.max(np.divide(state_discrepancy, processed,
        out=np.zeros_like(processed), where=processed > 0)))
    state_resolved = representation_ratio <= max(4 * rtol, 1e-8)
    if require_resolved_state and not state_resolved:
        raise FloatingPointError('Ionization change is not resolved in the floating-point state; rescale/model state with extra precision')
    audit = dict(
        model='infinite-c, fixed homogeneous H cells, finite opacity, fixed T, case-B local recombination',
        accepted_steps=accepted, rejected_steps=rejected, flow_calls=calls,
        largest_accepted_error_estimate=largest_error, rtol=rtol, atol=atol,
        adaptive=steps is None, seconds=time.perf_counter() - start,
        photon_ledger_error=abs(total - absorbed.sum() - escape.sum()),
        per_beam_ledger_max=float(np.max(np.abs(rates * duration - beam_absorbed - escape), initial=0)),
        absorption_crosscheck=abs(absorbed.sum() - beam_absorbed.sum()),
        atom_ledger_max_error=float(state_discrepancy.max()),
        state_representation_relative_error=representation_ratio,
        state_precision_resolved=state_resolved,
        x_min=float(x.min()), x_max=float(x.max()),
    )
    return dict(x=x, absorbed=absorbed, recombinations=recombined,
                escaped_by_beam=escape, beam_absorbed=beam_absorbed,
                emitted=total, ionized_atom_increment=absorbed - recombined, **flux, audit=audit)


def solve_history(atoms, initial, alpha_n, tau_neutral, paths, segments, **kwargs):
    """Persist the ionized gas state across explicit (duration, rates) intervals.

    Boundaries are source changes, not cumulative budgets. Radiation is
    quasistatic: this API deliberately does not claim finite-c in-flight state.
    """
    x = np.asarray(initial, dtype=float).copy()
    outputs = []
    for duration, rates in segments:
        result = solve(atoms, x, alpha_n, tau_neutral, paths, rates, duration, **kwargs)
        x = result['x'].copy()
        outputs.append(result)
    if not outputs:
        raise ValueError('At least one source interval required')
    absorbed = sum((v['absorbed'] for v in outputs), np.zeros_like(x))
    recombined = sum((v['recombinations'] for v in outputs), np.zeros_like(x))
    escape = sum((v['escaped_by_beam'] for v in outputs), np.zeros(len(paths)))
    emitted = sum(v['emitted'] for v in outputs)
    return dict(x=x, absorbed=absorbed, recombinations=recombined,
                escaped_by_beam=escape, emitted=emitted, intervals=outputs,
                photon_ledger_error=abs(emitted - absorbed.sum() - escape.sum()),
                atom_ledger_max_error=float(np.max(np.abs(
                    np.asarray(atoms) * (x - np.asarray(initial)) - absorbed + recombined))))
