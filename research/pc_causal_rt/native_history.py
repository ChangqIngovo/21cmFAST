"""Native source endpoints and conservative native 2LPT atom trajectories.

The emitted source is the native instantaneous escaped SFR photon rate.  Its
time integral is deliberately not forced to equal the native coeval ``n_ion``
proxy.  Atom packets use the very same initial masses and displacement lookup
as ``map_mass.c::move_grid_masses``.  Their double-precision CIC density is
audited against every native density snapshot, allowing only native float/FFT
roundoff.  No ions, atoms or photons are reset when the density evolves.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from research.pc_evolution.native_history import native_time_interval, rate_from_nion


SCHEMA = 1
DENSITY_ABSOLUTE_TOLERANCE = 3e-6
ARRAY_NAMES = ("redshifts", "times_seconds", "growth_factors", "rates",
               "native_capacity", "native_xhi", "native_brightness")
PACKET_ARRAY_NAMES = ("mass", "q", "v1", "v2")


def _hash_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json_write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_native_gas_inputs(inputs):
    """Reject native density operations not represented by the packet flow."""
    from research.pc_reconsidered.native_driver import validate_inputs
    validate_inputs(inputs)
    matter, simulation = inputs.matter_options, inputs.simulation_options
    if matter.PERTURB_ALGORITHM != "2LPT":
        raise ValueError("Causal native gas requires the native 2LPT displacement model")
    if matter.PERTURB_ON_HIGH_RES or matter.SMOOTH_EVOLVED_DENSITY_FIELD:
        raise ValueError("Causal packet gas does not support native density filtering/high-res perturbation")
    if simulation.NON_CUBIC_FACTOR != 1:
        raise ValueError("Causal packet gas requires a cubic domain")
    if inputs.astro_options.HALO_SCALING_RELATIONS_MEDIAN:
        raise ValueError("Instantaneous n_ion-to-SFR conversion requires mean integral scaling relations")


def packets_from_initial(initial, inputs, growth_initial):
    """Extract fixed mean-coarse-cell atom masses and native 2LPT trajectories.

    Integer coordinates are CIC grid nodes, exactly as in native map_mass.c.
    The high-resolution density packets sample low-resolution displacements by
    native ``floor(i * N_low/N_high + 0.5)`` nearest-node indexing, not trilinear
    velocity interpolation.  v1 and v2 are expressed in coarse-cell lengths.
    """
    n = int(inputs.simulation_options.HII_DIM)
    high = int(inputs.simulation_options.DIM)
    box_len = float(inputs.simulation_options.BOX_LEN)
    if n < 2 or high < n or not np.isfinite(growth_initial) or growth_initial <= 0:
        raise ValueError("Invalid packet mesh or initial growth factor")
    density = np.array(initial.get("hires_density"), dtype=float, copy=True)
    if density.shape != (high,) * 3 or not np.isfinite(density).all():
        raise ValueError("Invalid native initial density packet grid")
    initial_mass = 1 + growth_initial * density
    if (initial_mass < 0).any():
        raise ValueError("Native initial atom packet has negative mass; no clipping is permitted")
    indices = np.indices(density.shape, dtype=np.int64).reshape(3, -1).T
    ratio = n / high
    sampled = np.floor(indices * ratio + .5).astype(np.int64) % n
    index_tuple = tuple(sampled.T)
    vectors = []
    for suffix in ("", "_2LPT"):
        vector = []
        for axis in "xyz":
            field = np.asarray(initial.get(f"lowres_v{axis}{suffix}"))
            if field.shape != (n,) * 3 or not np.isfinite(field).all():
                raise ValueError("Missing or invalid native low-resolution displacement field")
            vector.append(field[index_tuple].astype(float) * (n / box_len))
        vectors.append(np.ascontiguousarray(np.stack(vector, axis=1)))
    return {"mass": np.ascontiguousarray(initial_mass.ravel() * ratio**3),
            "q": np.ascontiguousarray(indices * ratio), "v1": vectors[0], "v2": vectors[1],
            "growth_initial": float(growth_initial), "coarse_dimension": n}


def positions_at_growth(packets, growth):
    """Unwrapped periodic-box positions at native linear growth factor D."""
    if not np.isfinite(growth) or growth <= 0:
        raise ValueError("Growth must be finite and positive")
    initial = packets["growth_initial"]
    d2 = -3 / 7 * (growth**2 - initial**2)
    return np.ascontiguousarray(packets["q"] + (growth - initial) * packets["v1"]
                               - d2 * packets["v2"])


def _native_age_seconds(redshift, cosmology):
    """Analytic primitive of native |dtdz|, with zero age at infinite z."""
    h0 = float(cosmology["hlittle"]) * 3.2407e-18
    om, ol = float(cosmology["OMm"]), float(cosmology["OMl"])
    if (cosmology.get("OMk", 0.) != 0 or cosmology.get("wl", -1.) != -1
            or not np.isfinite([redshift, h0, om, ol]).all()
            or redshift < 0 or h0 <= 0 or om <= 0 or ol < 0):
        raise ValueError("Native background clock requires finite flat matter-Lambda cosmology")
    normalization = 2 * np.sqrt(om + ol) / (3 * h0)
    if ol == 0:
        return normalization / np.sqrt(om) * (1 + redshift)**-1.5
    return normalization / np.sqrt(ol) * np.arcsinh(np.sqrt(ol / om) * (1 + redshift)**-1.5)


def _redshift_at_time_fraction(old_redshift, new_redshift, fraction, cosmology):
    """Invert native cosmic age, rather than interpolating redshift in time."""
    if not 0 <= fraction <= 1 or not old_redshift > new_redshift >= 0:
        raise ValueError("Require an interval fraction in [0,1] and descending redshifts")
    if fraction == 0:
        return float(old_redshift)
    if fraction == 1:
        return float(new_redshift)
    old_age = _native_age_seconds(old_redshift, cosmology)
    new_age = _native_age_seconds(new_redshift, cosmology)
    age = old_age + fraction * (new_age - old_age)
    h0 = float(cosmology["hlittle"]) * 3.2407e-18
    om, ol = float(cosmology["OMm"]), float(cosmology["OMl"])
    normalization = 2 * np.sqrt(om + ol) / (3 * h0)
    if ol == 0:
        return float((normalization / (np.sqrt(om) * age))**(2 / 3) - 1)
    return float((np.sqrt(ol / om) / np.sinh(age * np.sqrt(ol) / normalization))**(2 / 3) - 1)


@lru_cache(maxsize=16)
def _inputs_from_serialized(serialized):
    from py21cmfast.input_serialization import deserialize_inputs
    from py21cmfast.wrapper.inputs import InputParameters
    raw = json.loads(serialized)
    structs = deserialize_inputs({key: value for key, value in raw.items()
                                  if key not in ("random_seed", "node_redshifts")}, safe=True)
    inputs = InputParameters(**structs, random_seed=raw["random_seed"], node_redshifts=raw["node_redshifts"])
    restored = json.loads(json.dumps(inputs.asdict(include_cosmo_tables="never", only_cstruct_params=False), allow_nan=False))
    if restored != raw:
        raise ValueError("Archived native inputs do not round-trip for background growth")
    return inputs


@lru_cache(maxsize=32768)
def _cached_native_background(serialized, cosmology_serialized, old_z, new_z, fraction):
    from py21cmfast.wrapper.cfuncs import get_growth_factor
    cosmology = json.loads(cosmology_serialized)
    redshift = _redshift_at_time_fraction(old_z, new_z, fraction, cosmology)
    inputs = _inputs_from_serialized(serialized)
    # ComputePerturbedField takes a float redshift, while native dicke uses double.
    growth = get_growth_factor(inputs=inputs, redshift=float(np.float32(redshift)))
    return redshift, float(growth)


def background_at_time(history, node, fraction):
    """Exact native-clock background in interval [node-1,node], in proper time.

    Returns redshift, growth_factor and time_seconds.  Growth is evaluated by
    the native function with its perturbed-field redshift precision; it is not
    interpolated between source nodes.  Repeated substep coordinates are cached.
    """
    if (type(node) is not int or not 1 <= node < len(history["redshifts"])
            or not np.isfinite(fraction) or not 0 <= fraction <= 1):
        raise ValueError("Native background requires a right-endpoint node and fraction in [0,1]")
    fraction = float(fraction)
    times = history["times_seconds"]
    if fraction in (0., 1.):
        index = node - 1 + int(fraction)
        return {"redshift": float(history["redshifts"][index]),
                "growth_factor": float(history["growth_factors"][index]),
                "time_seconds": float(times[index])}
    meta = history["metadata"]
    redshift, growth = _cached_native_background(
        json.dumps(meta["inputs"], sort_keys=True, allow_nan=False),
        json.dumps(meta["cosmology"], sort_keys=True, allow_nan=False),
        float(history["redshifts"][node - 1]), float(history["redshifts"][node]), fraction,
    )
    return {"redshift": redshift, "growth_factor": growth,
            "time_seconds": float(times[node - 1] + fraction * (times[node] - times[node - 1]))}


def validate_history(history):
    """Validate loaded values and their exact stored hashes."""
    z = history["redshifts"]
    times = history["times_seconds"]
    growth = history["growth_factors"]
    meta = history["metadata"]
    if meta.get("schema") != SCHEMA:
        raise ValueError("Unsupported causal native history schema")
    if (z.ndim != 1 or len(z) < 2 or times.shape != z.shape or growth.shape != z.shape
            or not np.isfinite(z).all() or (z < 0).any() or not (np.diff(z) < 0).all()
            or times[0] != 0 or not np.isfinite(times).all() or not (np.diff(times) > 0).all()
            or not np.isfinite(growth).all() or not (np.diff(growth) > 0).all()):
        raise ValueError("Causal history requires ordered physical time and native growth")
    n = int(meta["dimension"])
    shape = (len(z), n, n, n)
    for name in ("rates", "native_capacity", "native_xhi", "native_brightness"):
        value = history[name]
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid native endpoint field {name}")
    if ((history["rates"] < 0).any() or (history["native_capacity"] <= 0).any()
            or ((history["native_xhi"] < 0) | (history["native_xhi"] > 1)).any()):
        raise ValueError("Native source/capacity/fraction violates bounds")
    packets = history["packets"]
    count = packets["mass"].size
    if (packets["mass"].shape != (count,) or not np.isfinite(packets["mass"]).all()
            or (packets["mass"] < 0).any() or packets["mass"].sum() <= 0
            or packets["coarse_dimension"] != n
            or not np.isfinite(packets["growth_initial"]) or packets["growth_initial"] <= 0
            or packets["growth_initial"] >= growth[0]):
        raise ValueError("Invalid conserved packet atom inventory")
    for name in ("q", "v1", "v2"):
        if packets[name].shape != (count, 3) or not np.isfinite(packets[name]).all():
            raise ValueError(f"Invalid native packet {name}")
    hashes = meta["array_sha256"]
    for name in ARRAY_NAMES:
        if hashes[name] != _hash_array(history[name]):
            raise ValueError(f"Native history hash mismatch: {name}")
    for name in PACKET_ARRAY_NAMES:
        if hashes[f"packet_{name}"] != _hash_array(packets[name]):
            raise ValueError(f"Native history packet hash mismatch: {name}")
    return history


def _save(path, history):
    arrays = {name: history[name] for name in ARRAY_NAMES}
    arrays.update({f"packet_{name}": history["packets"][name] for name in PACKET_ARRAY_NAMES})
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays,
                            metadata=np.asarray(json.dumps(history["metadata"], sort_keys=True)))
    temporary.replace(path)


def load_history(path):
    with np.load(path, allow_pickle=False) as saved:
        result = {name: np.array(saved[name], copy=True) for name in ARRAY_NAMES}
        metadata = json.loads(str(saved["metadata"].item()))
        result["packets"] = {name: np.array(saved[f"packet_{name}"], copy=True)
                             for name in PACKET_ARRAY_NAMES}
    result["metadata"] = metadata
    result["packets"].update(growth_initial=metadata["growth_initial"],
                             coarse_dimension=metadata["dimension"])
    result["records"] = metadata["records"]
    return validate_history(result)


def build_native_history(root, config, seed):
    """Archive native coevals and return native rates plus conservative atoms.

    root is a dedicated per-seed history directory.  Existing individual native
    snapshots resume through their input/runtime/field-hash receipts.  The
    completed history additionally hashes all native archives and endpoint
    arrays; changed executable/source identities require a fresh directory.
    """
    from py21cmfast.drivers.coeval import Coeval
    from py21cmfast.wrapper.cfuncs import get_growth_factor
    from research.pc_native_rt.workflow import (
        _native_inputs, _native_snapshots, _runtime_identity, _source_hashes,
        _receipt, configuration,
    )
    from research.pc_causal_rt.gas import deposit

    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    start, end = float(config.get("z_start", 25)), float(config.get("z_end", 5))
    intervals = config.get("source_intervals", 80)
    if type(intervals) is not int or intervals < 1 or not np.isfinite([start, end]).all() or not start > end >= 0:
        raise ValueError("Native history needs positive interval count and descending nonnegative redshifts")
    redshifts = np.linspace(start, end, intervals + 1, dtype=float)
    native_config = configuration({k: config[k] for k in (
        "dimension", "highres_dimension", "box_len", "threads", "native_inputs_json") if k in config}
        | {"seeds": [seed], "redshifts": redshifts.tolist()})
    inputs, provenance = _native_inputs(native_config, seed)
    validate_native_gas_inputs(inputs)
    identity = {"schema": SCHEMA, "native_configuration": native_config,
                "native_runtime": _runtime_identity(), "provenance": provenance,
                "source_sha256": _source_hashes() | {"research/pc_causal_rt/native_history.py": _file_hash(__file__)}}
    manifest_path = root / "history_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != identity:
            raise ValueError("Causal native history inputs or executable changed; use a fresh directory")
        receipt = _receipt(root, "history_receipt.json", identity)
        if receipt is not None:
            return load_history(root / "history.npz")
    else:
        _json_write(manifest_path, identity)

    records = _native_snapshots(root, native_config)
    if len(records) != len(redshifts) or [r["redshift"] for r in records] != redshifts.tolist():
        raise RuntimeError("Native generation did not return exactly the requested source nodes")
    first = Coeval.from_file(root / records[0]["h5_path"])
    growth_initial = get_growth_factor(inputs=inputs, redshift=float(inputs.simulation_options.INITIAL_REDSHIFT))
    packets = packets_from_initial(first.initial_conditions, inputs, growth_initial)
    growth = np.array([get_growth_factor(inputs=inputs, redshift=float(np.float32(z))) for z in redshifts])
    cosmology = {key: float(getattr(inputs.cosmo_params, key))
                 for key in ("hlittle", "OMm", "OMr", "OMl", "OMk", "wl")}
    times = np.zeros_like(redshifts)
    for i in range(intervals):
        duration, _ = native_time_interval(redshifts[i], redshifts[i + 1], cosmology)
        times[i + 1] = times[i] + duration
    n = native_config["dimension"]
    fields = {key: np.empty((len(redshifts), n, n, n), dtype=float)
              for key in ("rates", "native_capacity", "native_xhi", "native_brightness")}
    density_audits = []
    for i, record in enumerate(records):
        with np.load(root / record["fields_path"], allow_pickle=False) as saved:
            native_source = saved["source"]
            fields["rates"][i] = rate_from_nion(native_source, record["redshift"],
                                                t_star=inputs.astro_params.t_STAR, cosmo=cosmology)
            fields["native_capacity"][i] = saved["capacity"]
            fields["native_xhi"][i] = saved["native_es_xhi"]
            fields["native_brightness"][i] = saved["native_brightness"]
        capacity = deposit(packets["mass"], positions_at_growth(packets, growth[i]), n)
        error = float(np.max(np.abs(capacity - fields["native_capacity"][i])))
        if not np.isfinite(error) or error > DENSITY_ABSOLUTE_TOLERANCE:
            raise ValueError(f"Native density is not the conservative packet density at z={record['redshift']}: error={error}")
        density_audits.append({"redshift": record["redshift"], "maximum_absolute_density_difference": error,
                               "native_roundoff_tolerance": DENSITY_ABSOLUTE_TOLERANCE,
                               "packet_atom_total": float(packets["mass"].sum(dtype=np.longdouble)),
                               "native_atom_total": float(fields["native_capacity"][i].sum(dtype=np.longdouble))})

    result = {"redshifts": redshifts, "times_seconds": times, "growth_factors": growth,
              **fields, "packets": packets, "records": records}
    metadata = {"schema": SCHEMA, "seed": seed, "dimension": n,
                "highres_dimension": int(inputs.simulation_options.DIM),
                "box_len_cMpc": native_config["box_len"], "growth_initial": growth_initial,
                "cosmology": cosmology, "inputs": first.inputs.asdict(include_cosmo_tables="never", only_cstruct_params=False),
                "records": records, "native_archive_root": str(root), "density_audits": density_audits,
                "source_definition": "native instantaneous n_ion * H(z) / t_STAR; no density multiplier or cumulative-grid differencing",
                "density_definition": "conserved native initial mass packets with native 2LPT trajectories and CIC",
                "density_native_float_roundoff_only": True,
                "time_clock": "integrated native matter-Lambda dtdz; analytic cosmic-age inverse and native growth at substeps; rates interpolated in proper time",
                "initial_state": "neutral atoms and zero radiation at z_start; no pre-z_start emission supplied",
                "native_ES_comparison": "context only: native coeval n_ion is not the time-integrated emissivity",
                "optional_physics": "current restricted ACG/no-recombination/no-thermal/no-photoheating-feedback native baseline",
                "array_sha256": {name: _hash_array(result[name]) for name in ARRAY_NAMES}
                    | {f"packet_{name}": _hash_array(packets[name]) for name in PACKET_ARRAY_NAMES}}
    result["metadata"] = metadata
    validate_history(result)
    _save(root / "history.npz", result)
    paths = [path for path in (root / "native").rglob("*") if path.is_file()]
    paths += [root / "history.npz", manifest_path]
    _json_write(root / "history_receipt.json", {"identity": identity,
                "files": {str(path.relative_to(root)): _file_hash(path) for path in paths}})
    # Return the same JSON-normalized metadata on fresh and cached builds.
    # Native input tuples otherwise differ from their archived JSON lists.
    return load_history(root / "history.npz")
