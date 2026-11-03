"""Native cumulative stellar budgets on persistent Lagrangian source packets.

The unchanged native emissivity function is evaluated on a temporary copy of
the initial conditions with only source displacements zeroed. This exposes
the source packet counts before native CIC deposition. The original packet
trajectories then reproduce every archived native Eulerian source field.
Emission follows nonnegative increments, with strict rejection by default or an
explicit bounded causal envelope for slightly decreasing coeval proxies. Raw
native counts remain separately archived. Moving Eulerian maps are not differenced.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = 1
NATIVE_DEPOSITION_TOLERANCE = 1e-6
ARRAYS = ("cumulative", "times_seconds", "redshifts", "growth_factors", "native_totals")
PACKET_ARRAYS = ("q", "v1", "v2")
ENVELOPE_GLOBAL_LIMIT = 1e-5
ENVELOPE_PACKET_LIMIT = 1e-3


def validate_source_policy(policy):
    if policy not in ("strict", "bounded_envelope"):
        raise ValueError("source_monotonicity must be strict or bounded_envelope")
    return policy


def _budget_arrays(budget):
    return ARRAYS + (("native_cumulative",) if "native_cumulative" in budget else ())


def _monotonicity_audit(native, emitted, policy):
    """Audit a causal running maximum, using only one time slice of workspace.

    A declining coeval proxy cannot withdraw previously emitted photons. The
    envelope emits nothing until the proxy exceeds its previous maximum; this
    avoids counting a rebound twice, unlike clipping individual increments.
    Bounds apply both to the whole box and to each individual source packet.
    """
    validate_source_policy(policy)
    native, emitted = np.asarray(native), np.asarray(emitted)
    if (native.shape != emitted.shape or native.ndim != 4 or len(native) < 2
            or not np.isfinite(native).all() or np.any(native < 0)):
        raise ValueError("Invalid raw native source history")
    peak = native[0].copy()
    affected = np.zeros(peak.shape, dtype=bool)
    negative_variation = np.longdouble(0)
    negative_intervals = 0
    max_packet = 0.
    rows = []
    for node, raw in enumerate(native):
        if node:
            delta = raw - native[node - 1]
            negative = delta < 0
            negative_intervals += bool(np.any(negative))
            affected |= negative
            negative_variation -= delta[negative].sum(dtype=np.longdouble)
            if policy == "strict" and np.any(negative):
                raise ValueError(f"Native cumulative source decreases at node {node}; "
                                 "strict policy does not repair source histories")
            np.maximum(peak, raw, out=peak)
        if not np.array_equal(emitted[node], peak):
            raise ValueError("Emission history differs from the causal running maximum")
        extra = emitted[node] - raw
        packet_fraction = np.divide(extra, peak, out=np.zeros_like(extra, dtype=float), where=peak > 0)
        max_packet = max(max_packet, float(packet_fraction.max()))
        raw_total = raw.sum(dtype=np.longdouble)
        added = extra.sum(dtype=np.longdouble)
        rows.append({"node": node, "native_total": float(raw_total),
                     "emitted_total": float(emitted[node].sum(dtype=np.longdouble)),
                     "added_photons": float(added),
                     "added_fraction": float(added / max(raw_total, np.finfo(float).tiny))})
    negative_fraction = float(negative_variation / max(rows[-1]["native_total"], np.finfo(float).tiny))
    max_global = max(row["added_fraction"] for row in rows)
    if policy == "bounded_envelope" and (
            max_global > ENVELOPE_GLOBAL_LIMIT or negative_fraction > ENVELOPE_GLOBAL_LIMIT
            or max_packet > ENVELOPE_PACKET_LIMIT):
        raise ValueError("Source envelope correction exceeds its explicit bound: "
                         f"global={max_global:.6g}, negative_variation={negative_fraction:.6g}, "
                         f"packet={max_packet:.6g}")
    return {"policy": policy, "changed": bool(np.any(affected)),
            "affected_packets": int(np.count_nonzero(affected)),
            "affected_packet_fraction": float(np.mean(affected)),
            "intervals_with_negative_increments": int(negative_intervals),
            "total_negative_variation": float(negative_variation),
            "negative_variation_over_final_native_total": negative_fraction,
            "max_added_photons_over_native_total": max_global,
            "final_added_photons_over_native_total": rows[-1]["added_fraction"],
            "max_packet_added_fraction": max_packet,
            "limits": {"global_fraction": ENVELOPE_GLOBAL_LIMIT,
                       "packet_fraction": ENVELOPE_PACKET_LIMIT},
            "per_node": rows}


def prepare_emission_history(native, policy="strict"):
    """Return an audited emission history without modifying native input bytes."""
    validate_source_policy(policy)
    native = np.asarray(native)
    emitted = (np.maximum.accumulate(native, axis=0) if policy == "bounded_envelope"
               else native)
    return emitted, _monotonicity_audit(native, emitted, policy)


def _digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _array_hash(value):
    return hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast("B")).hexdigest()


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _archive_file(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Native archive path escapes its root")
    return path


def _verify_receipt(root, receipt, names):
    for name in names:
        if name not in receipt["files"] or _digest(_archive_file(root, name)) != receipt["files"][name]:
            raise ValueError(f"Native source archive checksum failed: {root / name}")


def _validate_budget(budget, check_hashes=True):
    metadata = budget["metadata"]
    n = budget["packets"]["coarse_dimension"]
    q = np.asarray(budget["cumulative"])
    times, z, growth = (np.asarray(budget[k]) for k in ("times_seconds", "redshifts", "growth_factors"))
    if metadata.get("schema") != SCHEMA or type(n) is not int or n < 2:
        raise ValueError("Invalid cumulative native source schema/grid")
    if (times.ndim != 1 or len(times) < 2 or times[0] != 0 or not np.isfinite(times).all()
            or not np.all(np.diff(times) > 0) or z.shape != times.shape
            or not np.isfinite(z).all() or np.any(z < 0) or not np.all(np.diff(z) < 0)
            or growth.shape != times.shape or not np.isfinite(growth).all()
            or not np.all(np.diff(growth) > 0)):
        raise ValueError("Invalid native source timeline")
    if q.shape != (len(times), n, n, n) or not np.isfinite(q).all() or np.any(q < 0):
        raise ValueError("Native cumulative packet budgets must be finite and nonnegative")
    for node in range(1, len(q)):
        delta = q[node] - q[node - 1]
        if np.any(delta < 0):
            i, j, k = np.unravel_index(np.argmin(delta), delta.shape)
            raise ValueError("Native cumulative source decreases on a persistent packet: "
                             f"node={node}, packet=({i},{j},{k}), delta={delta[i, j, k]:.17g}; "
                             "a nonnegative emitted-photon history cannot represent these inputs; no clipping is allowed")
    totals = np.asarray(budget["native_totals"])
    if totals.shape != times.shape or not np.isfinite(totals).all() or np.any(totals < 0):
        raise ValueError("Invalid archived native source totals")
    policy = validate_source_policy(metadata.get("emission_policy", "strict"))
    if policy == "bounded_envelope":
        if "native_cumulative" not in budget:
            raise ValueError("Bounded source history requires the original native cumulative array")
        native = np.asarray(budget["native_cumulative"])
        audit = _monotonicity_audit(native, q, policy)
        if metadata.get("monotonicity_audit") != audit:
            raise ValueError("Recorded source correction differs from its independent audit")
    else:
        if "native_cumulative" in budget:
            raise ValueError("Unexpected repaired source array under strict policy")
        native = q
        if "monotonicity_audit" in metadata and metadata["monotonicity_audit"] != _monotonicity_audit(native, q, policy):
            raise ValueError("Recorded source correction differs from its independent audit")
    counts = native.sum(axis=(1, 2, 3), dtype=np.longdouble)
    scale = np.maximum(totals, np.finfo(float).tiny)
    if np.any(np.abs(counts - totals) / scale > NATIVE_DEPOSITION_TOLERANCE):
        raise ValueError("Packet source budget differs from the native coeval total")
    initial_growth = budget["packets"]["growth_initial"]
    if not np.isfinite(initial_growth) or not 0 < initial_growth < growth[0]:
        raise ValueError("Invalid source initial growth factor")
    for name in PACKET_ARRAYS:
        a = budget["packets"][name]
        if a.shape != (n**3, 3) or not np.isfinite(a).all():
            raise ValueError(f"Invalid native source packet {name}")
    if check_hashes:
        hashes = metadata["array_sha256"]
        for name in _budget_arrays(budget):
            if _array_hash(budget[name]) != hashes[name]:
                raise ValueError(f"Source-budget array checksum failed: {name}")
        for name in PACKET_ARRAYS:
            if _array_hash(budget["packets"][name]) != hashes[f"packet_{name}"]:
                raise ValueError(f"Source-budget packet checksum failed: {name}")
    return budget


def source_positions(budget, growth, factor=1):
    """Original native source trajectories in the requested grid coordinates."""
    if type(factor) is not int or factor < 1 or not np.isfinite(growth) or growth <= 0:
        raise ValueError("Positive integer refinement and positive finite growth required")
    packets = budget["packets"]
    initial = packets["growth_initial"]
    position = (packets["q"] + (growth - initial) * packets["v1"]
                + 3 / 7 * (growth**2 - initial**2) * packets["v2"])
    return np.ascontiguousarray(position * factor + (factor - 1) / 2)


def sampled_cumulative(budget, times, stride):
    """Interpolate cumulative packet counts between retained proper-time knots."""
    times = np.asarray(times, dtype=float)
    if (not np.array_equal(times, budget["times_seconds"]) or type(stride) is not int
            or stride < 1 or (len(times) - 1) % stride):
        raise ValueError("Source stride must divide the original native timeline")
    q = np.asarray(budget["cumulative"], dtype=float)
    if any(np.any(q[node] < q[node - 1]) for node in range(1, len(q))):
        raise ValueError("Native cumulative source decreases; no clipping is allowed")
    sampled = q.copy()
    for left in range(0, len(times) - 1, stride):
        right = left + stride
        for node in range(left + 1, right):
            weight = (times[node] - times[left]) / (times[right] - times[left])
            sampled[node] = (1 - weight) * q[left] + weight * q[right]
    return sampled


def source_rate(budget, sampled, node, midgrowth, dimensionfactor=1):
    """Deposit positive packet emission at midpoint positions in fine-cell units.

All Q arrays use native mean-cell count units. The RT fine-grid convention
multiplies counts by f^3, so the sum divided by f^3 recovers physical coarse
counts independently of refinement. Source clouds retain one native-cell width.
"""
    from .gas import deposit
    if type(node) is not int or not 1 <= node < len(budget["times_seconds"]):
        raise ValueError("Source rate requires a valid right-endpoint node")
    if (np.shape(sampled) != np.shape(budget["cumulative"])
            or not np.isfinite(sampled[node - 1:node + 1]).all()):
        raise ValueError("Sampled cumulative sources must match the native source grid")
    delta = np.asarray(sampled[node], float) - sampled[node - 1]
    if np.any(delta < 0):
        raise ValueError("Negative packet emission increment; no clipping is allowed")
    positions = source_positions(budget, midgrowth, dimensionfactor)
    n = budget["packets"]["coarse_dimension"] * dimensionfactor
    duration = budget["times_seconds"][node] - budget["times_seconds"][node - 1]
    return deposit(delta.ravel() * dimensionfactor**3 / duration, positions, n,
                   cloud_width=dimensionfactor)


def source_initial_map(budget, factor=1):
    """Deposit the pre-start cumulative budget once, with no discarded photons."""
    from .gas import deposit
    positions = source_positions(budget, budget["growth_factors"][0], factor)
    return deposit(np.asarray(budget["cumulative"][0]).ravel() * factor**3,
                   positions, budget["packets"]["coarse_dimension"] * factor,
                   cloud_width=factor)


def _save_budget(path, budget):
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: budget[name] for name in _budget_arrays(budget)}
    arrays.update({f"packet_{name}": budget["packets"][name] for name in PACKET_ARRAYS})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays,
                            metadata=np.asarray(json.dumps(budget["metadata"], sort_keys=True, allow_nan=False)))
    temporary.replace(path)
    _write_json(path.with_suffix(".receipt.json"), {
        "schema": SCHEMA, "filename": path.name, "sha256": _digest(path),
        "identity": budget["metadata"]["identity"],
    })


def load_source_budget(path):
    path = Path(path).resolve()
    receipt = json.loads(path.with_suffix(".receipt.json").read_text())
    if receipt.get("schema") != SCHEMA or receipt["filename"] != path.name or receipt["sha256"] != _digest(path):
        raise ValueError("Saved source-budget file checksum failed")
    with np.load(path, allow_pickle=False) as saved:
        budget = {name: np.array(saved[name], copy=True) for name in ARRAYS}
        if "native_cumulative" in saved:
            budget["native_cumulative"] = np.array(saved["native_cumulative"], copy=True)
        metadata = json.loads(str(saved["metadata"].item()))
        budget["packets"] = {name: np.array(saved[f"packet_{name}"], copy=True) for name in PACKET_ARRAYS}
    if metadata["identity"] != receipt["identity"]:
        raise ValueError("Saved source-budget identity differs from receipt")
    budget["metadata"] = metadata
    budget["packets"].update(growth_initial=metadata["growth_initial"], coarse_dimension=metadata["dimension"])
    budget.update(path=str(path), sha256=receipt["sha256"])
    return _validate_budget(budget)


def build_source_budget(history, output_path, *, policy="strict"):
    """Export native cumulative source packets without editing input archives."""
    from py21cmfast.drivers.coeval import Coeval
    from py21cmfast.drivers.single_field import compute_emissivity_fields
    from py21cmfast.wrapper.outputs import InitialConditions
    from .native_history import validate_history, validate_native_gas_inputs
    from .gas import deposit
    from research.pc_native_rt.workflow import _input_dict, _runtime_identity
    from research.pc_reconsidered.native_driver import from_native_coeval

    validate_source_policy(policy)
    validate_history(history)
    root = Path(history.get("_archive_root", history["metadata"]["native_archive_root"])).resolve()
    output_path = Path(output_path).resolve()
    top = json.loads((root / "history_receipt.json").read_text())
    _verify_receipt(root, top, ("history.npz", "history_manifest.json"))
    with np.load(root / "history.npz", allow_pickle=False) as stored_history:
        if json.loads(str(stored_history["metadata"].item())) != history["metadata"]:
            raise ValueError("Supplied native history differs from its original archive")
    if len(history["records"]) != len(history["redshifts"]):
        raise ValueError("Native source history must have one record for every node")
    if json.loads((root / "history_manifest.json").read_text()) != top["identity"]:
        raise ValueError("Native history identity differs from its receipt")
    if output_path.is_relative_to(root) and str(output_path.relative_to(root)) in top["files"]:
        raise ValueError("Source-budget output may not replace a native input artifact")
    runtime = _runtime_identity()
    if runtime["native_extension_sha256"] != top["identity"]["native_runtime"]["native_extension_sha256"]:
        raise ValueError("Native source export requires the original archived native binary")
    identity = {"schema": SCHEMA, "history_sha256": top["files"]["history.npz"],
                "history_receipt_sha256": _digest(root / "history_receipt.json"),
                "native_runtime": runtime, "exporter_sha256": _digest(__file__),
                "emission_policy": policy}
    # Verify all source receipts even on a cached export, so changing an input
    # source does not become invisible behind an existing source-budget file.
    for record in history["records"]:
        child_relative = str(Path(record["fields_path"]).with_name("receipt.json"))
        _verify_receipt(root, top, (child_relative, record["fields_path"], record["metadata_path"], record["h5_path"]))
    if output_path.exists():
        previous = load_source_budget(output_path)
        if previous["metadata"]["identity"] != identity:
            raise ValueError("Source inputs or exporter changed; use a fresh source-budget output")
        return previous

    first = Coeval.from_file(_archive_file(root, history["records"][0]["h5_path"]))
    validate_native_gas_inputs(first.inputs)
    original = first.initial_conditions
    source_names = ("lowres_density",) + tuple(f"lowres_v{axis}{suffix}" for suffix in ("", "_2LPT") for axis in "xyz")
    initial_fields = {name: np.array(original.get(name), copy=True) for name in source_names}
    initial_hashes = {name: _array_hash(value) for name, value in initial_fields.items()}
    unshifted = InitialConditions.new(inputs=first.inputs)
    for name in original.arrays:
        try:
            value = np.array(original.get(name), copy=True)
        except ValueError:
            continue
        if name in source_names and name != "lowres_density":
            value[:] = 0
        unshifted.set(name, value)
    n = int(first.inputs.simulation_options.HII_DIM)
    box = float(first.inputs.simulation_options.BOX_LEN)
    packets = {"q": np.indices((n,) * 3, dtype=float).reshape(3, -1).T.copy(),
               "growth_initial": history["packets"]["growth_initial"], "coarse_dimension": n}
    for key, suffix in (("v1", ""), ("v2", "_2LPT")):
        packets[key] = np.ascontiguousarray(np.stack([initial_fields[f"lowres_v{a}{suffix}"].ravel()
                                                    for a in "xyz"], axis=1).astype(float) * (n / box))
    cumulative = np.empty((len(history["redshifts"]), n, n, n), dtype=float)
    native_totals = np.empty(len(cumulative))
    budget = {"cumulative": cumulative, "native_totals": native_totals, "packets": packets,
              **{name: np.array(history[name], copy=True) for name in ("times_seconds", "redshifts", "growth_factors")}}
    audits = []
    for node, record in enumerate(history["records"]):
        field_path = _archive_file(root, record["fields_path"])
        receipt = json.loads((field_path.parent / "receipt.json").read_text())
        _verify_receipt(field_path.parent, receipt, ("native.h5", "native.json", "source.npz"))
        if (receipt["record"] != record or record["seed"] != history["metadata"]["seed"]
                or record["redshift"] != float(history["redshifts"][node])
                or receipt["identity"]["inputs"] != history["metadata"]["inputs"]
                or receipt["identity"]["extension_sha256"] != runtime["native_extension_sha256"]):
            raise ValueError("Native source snapshot record, time or identity changed")
        coeval = Coeval.from_file(_archive_file(root, record["h5_path"]))
        validate_native_gas_inputs(coeval.inputs)
        if _input_dict(coeval.inputs) != receipt["identity"]["inputs"]:
            raise ValueError("Native source HDF5 inputs differ from their receipt")
        for name in source_names:
            if _array_hash(coeval.initial_conditions.get(name)) != initial_hashes[name]:
                raise ValueError(f"Native source initial conditions changed: {name}")
        extracted = from_native_coeval(coeval)
        with np.load(field_path, allow_pickle=False) as stored:
            target = np.array(stored["source"], copy=True)
            if not np.array_equal(extracted.source, target) or not np.array_equal(extracted.capacity, stored["capacity"]):
                raise ValueError("Native source HDF5 and archived arrays disagree")
        source_meta = json.loads(_archive_file(root, record["metadata_path"]).read_text())
        if (_array_hash(target) != record["source_sha256"]
                or _array_hash(target) != source_meta["array_sha256"]["source"]):
            raise ValueError("Original native source bytes changed")
        export = compute_emissivity_fields(redshift=record["redshift"], inputs=coeval.inputs,
            initial_conditions=unshifted, perturbed_field=coeval.perturbed_field,
            regenerate=True, write=False)
        cumulative[node] = np.asarray(export.get("n_ion"), dtype=float)
        target_total = target.sum(dtype=np.longdouble)
        native_totals[node] = target_total
        reconstructed = deposit(cumulative[node].ravel(),
                                source_positions(budget, budget["growth_factors"][node]), n)
        difference = np.abs(reconstructed - target)
        scale = max(target_total, np.longdouble(np.finfo(float).tiny))
        error = float(difference.sum(dtype=np.longdouble) / scale)
        total_error = float(abs(cumulative[node].sum(dtype=np.longdouble) - target_total) / scale)
        if max(error, total_error) > NATIVE_DEPOSITION_TOLERANCE:
            raise ValueError(f"Native source packet export does not reconstruct n_ion at node {node}: {error=}, {total_error=}")
        audits.append({"node": node, "redshift": record["redshift"], "passed": True,
                       "deposition_L1_relative_error": error, "total_relative_error": total_error,
                       "maximum_absolute_deposition_error": float(difference.max()),
                       "native_source_sha256": record["source_sha256"]})
    emitted, monotonicity_audit = prepare_emission_history(cumulative, policy)
    if policy == "bounded_envelope":
        budget["native_cumulative"] = cumulative
        budget["cumulative"] = emitted
    metadata = {"schema": SCHEMA, "seed": history["metadata"]["seed"], "dimension": n,
                "growth_initial": packets["growth_initial"], "native_archive_root": str(root),
                "archive_path": str(output_path), "history_sha256": identity["history_sha256"],
                "identity": identity, "initial_source_field_sha256": initial_hashes,
                "source_convention": "native cumulative L-INTEGRAL packet n_ion; emission is its nonnegative proper-time increment along original 2LPT source trajectories",
                "extraction": "unchanged native emissivity with a temporary zero-displacement initial-condition copy; original displacement restored by independently audited CIC deposition",
                "initial_budget": "initial Q_p is injected once at the first node; subsequent intervals inject only packet increments",
                "count_units": "native mean-cell effective photon count; fine RT arrays multiply counts by refinement cubed",
                "native_float_deposition_tolerance": NATIVE_DEPOSITION_TOLERANCE,
                "export_audits": audits,
                "emission_policy": policy,
                "monotonicity_audit": monotonicity_audit,
                "array_sha256": {name: _array_hash(budget[name]) for name in _budget_arrays(budget)}
                    | {f"packet_{name}": _array_hash(packets[name]) for name in PACKET_ARRAYS}}
    budget["metadata"] = metadata
    if policy == "bounded_envelope":
        metadata["source_convention"] = (
            "bounded causal running maximum of native cumulative L-INTEGRAL packet n_ion; "
            "original native_cumulative is retained separately; emission pauses during "
            "declines and resumes only above the previous maximum")
    _validate_budget(budget)
    _save_budget(output_path, budget)
    return load_source_budget(output_path)
