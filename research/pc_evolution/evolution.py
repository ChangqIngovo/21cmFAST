"""Carried refined-front evolution in fixed gas with positive rate histories.

Every time-resolution arm consumes the same finest-node trapezoidal photon
increments. Coarser arms group those increments, preserving every source's
budget while testing the temporal distribution of emission. A separate
diagnostic compares this budget to coarse endpoint quadrature.

The gas, source cell centres, angular bank and horizon remain fixed. This is
not a coupled Qin20 history: there is no gas advection, recombination,
photoheating feedback, spin-temperature evolution or in-flight photon state.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

HERE = Path(__file__).resolve().parent
for directory in ("pc_refined_front", "pc_reconsidered"):
    path = str(HERE.parent / directory)
    if path not in sys.path:
        sys.path.insert(0, path)
from refined import solve
from source_budget import integrate_rate_step


def _sum(array):
    return np.sum(array, dtype=np.longdouble)


def _metadata(value):
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, str):
        value = json.loads(value)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("History metadata must be a mapping or a JSON object")
    return value


def validate_history(history):
    """Validate without copying or constructing a time-by-fine-grid array."""
    if any(np.iscomplexobj(history[key]) for key in ("capacity", "times_seconds", "redshifts")):
        raise ValueError("Capacity and history coordinates must be real")
    capacity = np.ascontiguousarray(history["capacity"], dtype=float)
    rates = history["rates"]
    times = np.asarray(history["times_seconds"], dtype=float)
    redshifts = np.asarray(history["redshifts"], dtype=float)
    if (capacity.ndim != 3 or len(set(capacity.shape)) != 1 or not capacity.size
            or not np.isfinite(capacity).all() or (capacity <= 0).any()):
        raise ValueError("Positive finite cubic atom capacities required")
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or times[0] != 0 or (np.diff(times) <= 0).any()):
        raise ValueError("Times must start at zero and increase strictly")
    if (redshifts.shape != times.shape or not np.isfinite(redshifts).all()
            or (redshifts < 0).any() or (np.diff(redshifts) >= 0).any()):
        raise ValueError("Matching strictly decreasing nonnegative redshifts required")
    if np.shape(rates) != (len(times), *capacity.shape):
        raise ValueError("Rate history must have shape (time, n, n, n)")
    for index in range(len(times)):
        rate = np.asarray(rates[index])
        if np.iscomplexobj(rate) or not np.isfinite(rate).all() or (rate < 0).any():
            raise ValueError("Source rates must be finite real nonnegative increments per time")
    metadata = _metadata(history.get("metadata", {}))
    return capacity, rates, times, redshifts, metadata


def checkpoint_state(fields):
    """Reconstruct a solver state from a serialized checkpoint, without edits."""
    return dict(version=int(np.asarray(fields["state_version"]).item()),
                subcells_per_axis=int(np.asarray(fields["subcells_per_axis"]).item()),
                capacity_sha256=str(np.asarray(fields["capacity_sha256"]).item()),
                ionized=np.array(fields["ionized"], copy=True),
                cleared_bits=np.array(fields["cleared_bits"], copy=True),
                partial_ids=np.array(fields["partial_ids"], copy=True),
                partial_ionized=np.array(fields["partial_ionized"], copy=True))


def read_checkpoint(path):
    """Load arrays with pickle disabled; pass checkpoint_state(result) to solve."""
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _ledger(state, capacity, emitted, source_absorbed, escaped, retained):
    total_emitted = _sum(emitted)
    ionized = np.asarray(state["ionized"])
    total_ionized = _sum(ionized)
    residual = total_emitted - total_ionized - _sum(escaped) - _sum(retained)
    source_residual = emitted - source_absorbed - escaped - retained
    active = emitted > 0
    maximum_source_relative = (float(np.max(np.abs(source_residual[active]) / emitted[active]))
                               if active.any() else 0.)
    scale = max(total_emitted, np.longdouble(np.finfo(float).tiny))
    absorbed_disagreement = _sum(source_absorbed) - total_ionized
    fraction_violation = max(float(np.max(np.maximum(-ionized, 0) / capacity)),
                             float(np.max(np.maximum(ionized-capacity, 0) / capacity)))
    passed = (abs(residual) <= 2e-10*scale and maximum_source_relative <= 2e-10
              and abs(absorbed_disagreement) <= 2e-10*scale
              and fraction_violation <= 2e-10)
    return dict(emitted=float(total_emitted), ionized=float(total_ionized),
                source_absorbed=float(_sum(source_absorbed)),
                escaped=float(_sum(escaped)), retained=float(_sum(retained)),
                absolute_residual=float(residual), relative_residual=float(abs(residual)/scale),
                source_max_relative_residual=maximum_source_relative,
                source_absorbed_vs_grid_relative=float(abs(absorbed_disagreement)/scale),
                maximum_fraction_violation=fraction_violation,
                retained_fraction=float(_sum(retained)/scale), passed=bool(passed))


def _write_checkpoint(path, fields):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **fields)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _atomic_json(path, value):
    """Publish one complete JSON record; uncommitted temporary files are ignored."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent,
                                         prefix=".evolution-", suffix=".json.partial",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(value, indent=2, allow_nan=False)+"\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        # Persist the directory entry as well as the file contents.
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _history_identity(capacity, rates, times, redshifts, metadata, settings):
    """Hash source slices without materializing a second full rate history."""
    arrays = {}
    for name, value in (("capacity", capacity), ("rates", rates),
                        ("times_seconds", times), ("redshifts", redshifts)):
        digest = hashlib.sha256()
        digest.update(_json_bytes(dict(shape=list(np.shape(value)),
                                       dtype=np.asarray(value[0]).dtype.str)))
        for row in value:
            digest.update(np.ascontiguousarray(row).tobytes())
        arrays[name] = digest.hexdigest()
    files = (HERE/"evolution.py", HERE.parent/"pc_refined_front/refined.py",
             HERE.parent/"pc_refined_front/refined.cpp",
             HERE.parent/"pc_refinement/front_subcells.py",
             HERE.parent/"pc_reconsidered/source_budget.py")
    return dict(version=1, arrays=arrays, metadata=metadata, settings=settings,
                code_sha256={str(path.relative_to(HERE.parent)): _file_digest(path)
                             for path in files},
                runtime=dict(python=sys.version, numpy=np.__version__,
                             longdouble_dtype=np.dtype(np.longdouble).str,
                             longdouble_mantissa=int(np.finfo(np.longdouble).nmant)))


def _write_progress(path, payload):
    _atomic_json(path, dict(version=1,
                           sha256=hashlib.sha256(_json_bytes(payload)).hexdigest(),
                           payload=payload))


def _read_progress(path, identity):
    record = json.loads(path.read_text())
    payload = record.get("payload")
    if (record.get("version") != 1 or not isinstance(payload, dict)
            or record.get("sha256") != hashlib.sha256(_json_bytes(payload)).hexdigest()):
        raise ValueError("Evolution progress journal checksum mismatch")
    if payload.get("identity") != identity:
        raise ValueError("Evolution resume input, code, runtime or settings changed")
    return payload


def _recover_initial_checkpoint(directory, identity):
    """Recognize the sole crash window before the first journal exists.

    No interval has run yet. Rebuilding the neutral initialization is safe
    only when its published checkpoint identifies this exact history and no
    later published result exists. Unrelated files are left untouched.
    """
    path = directory/"checkpoints/node_00000.npz"
    if not path.is_file():
        raise FileNotFoundError("Evolution resume requires a durable progress.json or initial checkpoint")
    if ((directory/"summary.json").exists()
            or any(other != path for other in path.parent.glob("node_*.npz"))):
        raise ValueError("Missing evolution journal for existing noninitial results")
    fields = read_checkpoint(path)
    fingerprint = hashlib.sha256(_json_bytes(identity)).hexdigest()
    if ("evolution_identity_sha256" not in fields
            or str(fields["evolution_identity_sha256"].item()) != fingerprint
            or int(fields["node_index"]) != 0 or float(fields["time_seconds"]) != 0):
        raise ValueError("Unjournaled initial checkpoint belongs to a different history")
    for key in ("ionized", "cleared_bits", "partial_ids", "partial_ionized",
                "source_emitted", "source_absorbed", "source_escaped", "source_retained",
                "source_coarse_endpoint_emitted", "summed_interval_quadrature_l1",
                "signed_interval_quadrature_difference"):
        if np.any(fields[key] != 0):
            raise ValueError("Unjournaled initial checkpoint is not a neutral initial state")


def run_history(history, *, subcells_per_axis=4, angles=384, stride=1,
                maximum_distance=None, output=None, resume=False):
    """Run an initially neutral, fixed-capacity history and return a report.

    ``history`` is a mapping (an open npz archive is accepted) containing
    redshifts[T], times_seconds[T], rates[T,n,n,n], capacity[n,n,n], and optional
    metadata (mapping or JSON string). Rates are photons/cell/second in exactly
    the capacity count unit. Times start at zero; redshifts decrease.

    stride is 1, 2 or 4. All arms integrate the SAME fine-node rates, grouping
    adjacent increments into a piecewise constant average emission rate. This
    tests transport/history ordering at matched per-source budgets. Source
    quadrature error is reported separately and is not removed by relabeling.

    Checkpoints are saved at fine-node multiples of four and the last node.
    If output is supplied the report is JSON-serializable, summary.json is
    written there, and checkpoint entries point to sparse .npz files. Without
    output those entries instead contain an in-memory ``fields`` mapping.
    No dense fine field is constructed. The returned state stores ionized
    gas only; escaped and retained photons are accumulated, never re-emitted.
    ``resume=True`` restores the last atomically journaled checkpoint in
    ``output`` after verifying inputs, code, settings and all checkpoint hashes.
    Work after that checkpoint is repeated. A completed identical run is read
    back without invoking the transport solver.
    """
    if isinstance(stride, (bool, np.bool_)) or stride not in (1, 2, 4):
        raise ValueError("Time stride must be 1, 2 or 4")
    if not isinstance(stride, (int, np.integer)):
        raise ValueError("Time stride must be an integer")
    if (isinstance(angles, (bool, np.bool_)) or not isinstance(angles, (int, np.integer))
            or angles < 24 or angles % 24 or angles > np.iinfo(np.int32).max):
        raise ValueError("Frozen angle count must be a positive multiple of 24")
    if (isinstance(subcells_per_axis, (bool, np.bool_))
            or not isinstance(subcells_per_axis, (int, np.integer))
            or subcells_per_axis not in (2, 4, 8)):
        raise ValueError("Subcells per axis must be 2, 4 or 8")
    capacity, rates, times, redshifts, metadata = validate_history(history)
    horizon = float(8*len(capacity) if maximum_distance is None else maximum_distance)
    if not np.isfinite(horizon) or horizon <= 0:
        raise ValueError("Positive finite transport horizon required")
    if not isinstance(resume, (bool, np.bool_)):
        raise ValueError("resume must be boolean")
    directory = Path(output).resolve() if output is not None else None
    if resume and directory is None:
        raise ValueError("Evolution resume requires an output directory")
    identity = None
    progress = None
    if directory is not None:
        identity = _history_identity(capacity, rates, times, redshifts, metadata,
                                     dict(subcells_per_axis=int(subcells_per_axis),
                                          angles=int(angles), stride=int(stride),
                                          maximum_distance=horizon, periodic=True))
        if resume:
            if not (directory/"progress.json").is_file():
                _recover_initial_checkpoint(directory, identity)
            else:
                progress = _read_progress(directory/"progress.json", identity)
        elif directory.exists() and any(directory.iterdir()):
            raise FileExistsError("Evolution output must be a fresh or empty directory")
        if progress is not None:
            checkpoints = progress.get("checkpoints", [])
            if not checkpoints:
                raise ValueError("Evolution progress has no durable checkpoint")
            expected = [0, *[i for i in range(1, len(times))
                              if i % 4 == 0 or i == len(times)-1]]
            if [row["node_index"] for row in checkpoints] != expected[:len(checkpoints)]:
                raise ValueError("Evolution checkpoint sequence changed")
            for row in checkpoints:
                path = directory/"checkpoints"/f"node_{row['node_index']:05d}.npz"
                if row["path"] != str(path) or _file_digest(path) != row["sha256"]:
                    raise ValueError("Evolution checkpoint path or checksum changed")
            if progress.get("status") == "complete":
                if _file_digest(directory/"summary.json") != progress.get("summary_sha256"):
                    raise ValueError("Evolution completed summary checksum changed")
                return json.loads((directory/"summary.json").read_text())
            if progress.get("status") != "running":
                raise ValueError("Evolution progress has an unsupported status")
        (directory/"checkpoints").mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    angles = int(angles)
    # Explicit uniform bank stays unchanged even when an emitter switches off.
    angular = np.full(capacity.shape, angles, dtype=np.int32)
    options = dict(subcells_per_axis=subcells_per_axis, angle_counts=angular,
                   maximum_distance=horizon, periodic=True)
    identity_sha256 = hashlib.sha256(_json_bytes(identity)).hexdigest() if identity else None
    elapsed_before = 0.
    start_node = 0
    if progress is None:
        state = solve(np.zeros_like(capacity), capacity, **options)["state"]
        cumulative = {name: np.zeros(capacity.shape, dtype=np.longdouble)
                      for name in ("emitted", "absorbed", "escaped", "retained")}
        coarse_endpoint_emitted = np.zeros(capacity.shape, dtype=np.longdouble)
        steps, checkpoints = [], []
        quadrature_l1 = np.longdouble(0)
        quadrature_signed = np.longdouble(0)
    else:
        steps, checkpoints = progress["steps"], progress["checkpoints"]
        fields = read_checkpoint(checkpoints[-1]["path"])
        if str(fields["evolution_identity_sha256"].item()) != identity_sha256:
            raise ValueError("Evolution checkpoint belongs to a different history")
        start_node = int(fields["node_index"])
        if (start_node != checkpoints[-1]["node_index"]
                or float(fields["time_seconds"]) != times[start_node]
                or float(fields["redshift"]) != redshifts[start_node]):
            raise ValueError("Evolution checkpoint coordinates changed")
        expected_steps = [(left, min(left+stride, len(times)-1))
                          for left in range(0, start_node, stride)]
        if [(row["left_node"], row["right_node"]) for row in steps] != expected_steps:
            raise ValueError("Evolution journal step sequence changed")
        state = checkpoint_state(fields)
        cumulative = {name: np.array(fields["source_"+name], dtype=np.longdouble, copy=True)
                      for name in ("emitted", "absorbed", "escaped", "retained")}
        coarse_endpoint_emitted = np.array(fields["source_coarse_endpoint_emitted"],
                                           dtype=np.longdouble, copy=True)
        quadrature_l1 = np.longdouble(fields["summed_interval_quadrature_l1"].item())
        quadrature_signed = np.longdouble(fields["signed_interval_quadrature_difference"].item())
        restored_ledger = _ledger(state, capacity, cumulative["emitted"], cumulative["absorbed"],
                                  cumulative["escaped"], cumulative["retained"])
        if restored_ledger != checkpoints[-1]["ledger"] or not restored_ledger["passed"]:
            raise ValueError("Evolution restored photon ledger changed")
        elapsed_before = float(progress["elapsed_seconds"])

    def journal(status="running", **extra):
        if directory is not None:
            _write_progress(directory/"progress.json",
                            dict(identity=identity, status=status, steps=steps,
                                 checkpoints=checkpoints,
                                 elapsed_seconds=elapsed_before+time.perf_counter()-started,
                                 **extra))

    def checkpoint(index):
        ledger = _ledger(state, capacity, cumulative["emitted"], cumulative["absorbed"],
                         cumulative["escaped"], cumulative["retained"])
        fields = {name: state[name] for name in ("ionized", "cleared_bits", "partial_ids", "partial_ionized")}
        fields.update(state_version=np.array(state["version"]),
                      subcells_per_axis=np.array(state["subcells_per_axis"]),
                      capacity_sha256=np.array(state["capacity_sha256"]),
                      node_index=np.array(index), time_seconds=np.array(times[index]),
                      redshift=np.array(redshifts[index]))
        fields.update({"source_"+name: value for name, value in cumulative.items()})
        fields["source_coarse_endpoint_emitted"] = coarse_endpoint_emitted
        fields["summed_interval_quadrature_l1"] = np.array(quadrature_l1)
        fields["signed_interval_quadrature_difference"] = np.array(quadrature_signed)
        if identity_sha256:
            fields["evolution_identity_sha256"] = np.array(identity_sha256)
        emitted = _sum(cumulative["emitted"])
        normalization = max(emitted, np.longdouble(np.finfo(float).tiny))
        cumulative_quadrature_delta = coarse_endpoint_emitted-cumulative["emitted"]
        source_quadrature = dict(
            fine_emitted=float(emitted),
            coarse_endpoint_emitted=float(_sum(coarse_endpoint_emitted)),
            cumulative_photon_dose=float(emitted/_sum(capacity)),
            cumulative_L1_over_emitted=float(_sum(abs(cumulative_quadrature_delta))/normalization),
            signed_cumulative_difference_over_emitted=float(_sum(cumulative_quadrature_delta)/normalization))
        row = dict(node_index=index, time_seconds=float(times[index]),
                   redshift=float(redshifts[index]), ledger=ledger,
                   Q_mass=float(_sum(state["ionized"])/_sum(capacity)),
                   Q_volume=float(np.mean(state["ionized"]/capacity)),
                   xHI_mass=float(_sum(capacity-state["ionized"])/_sum(capacity)),
                   xHI_volume=float(np.mean(1-state["ionized"]/capacity)),
                   source_quadrature=source_quadrature)
        if directory is None:
            row["fields"] = {name: np.array(value, copy=True) for name, value in fields.items()}
        else:
            path = directory/"checkpoints"/f"node_{index:05d}.npz"
            if path.exists():
                # A crash can publish a checkpoint before publishing its journal.
                # Replace only that exact, identifiable checkpoint, never an
                # unrelated user file or an already journaled checkpoint.
                with np.load(path, allow_pickle=False) as orphan:
                    if ("evolution_identity_sha256" not in orphan
                            or str(orphan["evolution_identity_sha256"].item()) != identity_sha256
                            or int(orphan["node_index"]) != index):
                        raise FileExistsError(f"Unrecognized checkpoint already exists: {path}")
            row.update(path=str(path), sha256=_write_checkpoint(path, fields))
        checkpoints.append(row)
        journal()

    if progress is None:
        checkpoint(0)
    for left in range(start_node, len(times)-1, stride):
        right = min(left+stride, len(times)-1)
        grouped = np.zeros(capacity.shape, dtype=np.longdouble)
        for node in range(left, right):
            grouped += integrate_rate_step(rates[node], rates[node+1], times[node+1]-times[node])
        # Native kernel accepts float64; retain and report the one conversion.
        increment = np.asarray(grouped, dtype=float)
        if not np.isfinite(increment).all():
            raise OverflowError("Grouped photon increments exceed float64")
        if ((grouped != 0) & (increment == 0)).any():
            raise FloatingPointError("Grouped photon increments underflow float64")
        coarse = integrate_rate_step(rates[left], rates[right], times[right]-times[left])
        coarse_endpoint_emitted += coarse
        qdelta = coarse.astype(np.longdouble)-grouped
        quadrature_l1 += _sum(abs(qdelta))
        quadrature_signed += _sum(qdelta)
        previous = state["ionized"]
        result = solve(increment, capacity, initial_state=state, **options)
        state = result["state"]
        cumulative["emitted"] += increment
        for name in ("absorbed", "escaped", "retained"):
            cumulative[name] += result["source_"+name]
        ledger = _ledger(state, capacity, cumulative["emitted"], cumulative["absorbed"],
                         cumulative["escaped"], cumulative["retained"])
        deposited_error = _sum(abs(state["ionized"].astype(np.longdouble)-previous-result["absorbed"]))
        monotonic_error = float(np.max(np.maximum(previous-state["ionized"], 0)/capacity))
        scale = max(_sum(increment), _sum(cumulative["emitted"]), np.longdouble(np.finfo(float).tiny))
        ledger["state_increment_relative_residual"] = float(deposited_error/scale)
        ledger["monotonic_fraction_violation"] = monotonic_error
        ledger["passed"] &= bool(deposited_error <= 2e-10*scale and monotonic_error <= 2e-10)
        row = dict(left_node=left, right_node=right, redshift=float(redshifts[right]),
                   dt_seconds=float(times[right]-times[left]),
                   emitted_increment=float(_sum(increment)),
                   grouping_roundoff_absolute=float(_sum(increment)-_sum(grouped)),
                   coarse_endpoint_quadrature_signed_difference=float(_sum(qdelta)),
                   coarse_endpoint_quadrature_L1_difference=float(_sum(abs(qdelta))),
                   ledger=ledger, audit=result["audit"])
        steps.append(row)
        if not ledger["passed"]:
            raise RuntimeError(f"Independent evolution ledger failed at node {right}")
        if right % 4 == 0 or right == len(times)-1:
            checkpoint(right)
    emitted = _sum(cumulative["emitted"])
    denom = max(emitted, np.longdouble(np.finfo(float).tiny))
    report = dict(scope="Sequential positive-source evolution in fixed native gas; not coupled Qin20",
                  source_integration="Finest endpoint trapezoids grouped with matched per-source budgets",
                  stride=int(stride), subcells_per_axis=int(subcells_per_axis),
                  angles=angles, maximum_distance=horizon, fine_source_nodes=len(times),
                  steps=steps, checkpoints=checkpoints,
                  all_ledgers_passed=all(row["ledger"]["passed"] for row in steps),
                  capacity_sha256=hashlib.sha256(memoryview(capacity)).hexdigest(),
                  metadata=metadata,
                  source_quadrature=dict(
                      scope="Coarse endpoint trapezoid minus the common finest-node budget",
                      summed_interval_L1_over_emitted=float(quadrature_l1/denom),
                      signed_total_difference_over_emitted=float(quadrature_signed/denom),
                      checkpoints=[dict(node_index=row["node_index"], redshift=row["redshift"],
                                        Q_mass=row["Q_mass"], **row["source_quadrature"])
                                   for row in checkpoints]),
                  final_ledger=checkpoints[-1]["ledger"],
                  seconds=elapsed_before+time.perf_counter()-started,
                  timing_scope="Includes solver compilation if needed, checkpoints and diagnostics; resumes sum committed elapsed time and exclude discarded work",
                  continuum_accuracy_certified=False)
    if directory is not None:
        path = directory/"summary.json"
        _atomic_json(path, report)
        journal("complete", summary_sha256=_file_digest(path))
    return report
