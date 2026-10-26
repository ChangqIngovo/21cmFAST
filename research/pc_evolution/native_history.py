#!/usr/bin/env python3
"""Native evolving source fixtures for a fixed-gas temporal PC experiment.

Every endpoint is generated from the same native initial conditions.  The
restricted, deterministic L-INTEGRAL source prescription has
``weighted_SFRD / rho_b = n_ion / (t_STAR * t_hubble)``.  This identity supplies
an instantaneous escaped effective photon rate; n_ion is neither differenced
between snapshots nor itself interpreted as a rate.  The density used by the
transport experiment is explicitly frozen at ``reference_z``.

The contextual native ES fields use their own evolving native density and
cumulative n_ion prescription.  They are not a matched-rate non-PC control.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
from scipy.integrate import quad


ROOT = Path(__file__).resolve().parents[2]
ARRAY_KEYS = ("redshifts", "times_seconds", "rates", "capacity", "native_xhi")
SCHEMA_VERSION = 1


def require_es_radius_capability():
    """Reject a stale compiled runtime before exporting ES-radius features."""
    from py21cmfast import c_21cmfast

    ffi = c_21cmfast.ffi
    required = (("MatterOptions", "STORE_ES_RADIUS"),
                ("IonizedBox", "es_crossing_radius"))
    for kind, name in required:
        if name not in dict(ffi.typeof(kind).fields):
            raise RuntimeError(
                "Native ES-radius diagnostic is unavailable; rebuild this checkout's "
                "compiled py21cmfast extension before generating training pairs")


def write_es_features(path, history, radius, density):
    """Write a separate, hashed native ES feature archive without changing history schema.

    The radius is the native largest crossing scale in cMpc, zero when no full
    crossing occurs. Native ES density and its cumulative-source convention
    differ from the prescribed C2-Ray history; these are contextual emulator
    inputs rather than an isolated photon-loss control.
    """
    validate_history(history)
    radius, density = np.asarray(radius), np.asarray(density)
    shape = history["native_xhi"].shape
    if (radius.shape != shape or density.shape != shape
            or not np.isfinite(radius).all() or (radius < 0).any()
            or not np.isfinite(density).all() or (density <= 0).any()):
        raise ValueError("ES radius and density must be finite matching nonnegative/positive fields")
    if np.any((radius > 0) & (history["native_xhi"] > 1e-6)):
        raise ValueError("An ES crossing radius requires a fully ionized native centre")
    meta = history["metadata"]
    arrays = {"redshifts": history["redshifts"], "times_seconds": history["times_seconds"],
              "xhi_es": history["native_xhi"], "radius_es_cMpc": radius, "density_es": density}
    metadata = {
        "schema_version": 1,
        "kind": "native-es-emulator-features",
        "radius_definition": "largest native centre-cell ES crossing; zero when no full crossing",
        "radius_units": "cMpc",
        "diagnostic_capability": "STORE_ES_RADIUS/es_crossing_radius",
        "random_seed": meta["random_seed"],
        "initial_hires_density_sha256": meta["initial_hires_density_sha256"],
        "box_len_cMpc": meta["box_len_cMpc"],
        "history_array_sha256": meta["array_sha256"],
        "native_extension_sha256": meta["native_extension_sha256"],
        "contextual_native_ES": meta["contextual_native_ES"],
        "array_sha256": {key: array_hash(value) for key, value in arrays.items()},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays, metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink()


def array_hash(array):
    """Hash the exact contiguous numeric payload, without pickle or metadata."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def native_hubble(redshift, cosmo):
    """Native radiation-inclusive H(z), in s^-1 (cosmology.c::hubble).

    The native function accepts a float redshift; mirror this rounding before
    evaluating its expression in double precision.
    """
    z = float(np.float32(redshift))
    values = [cosmo[k] for k in ("hlittle", "OMm", "OMr", "OMl")]
    if (not np.isfinite(z) or z < 0 or not np.isfinite(values).all()
            or values[0] <= 0 or min(values[1:]) < 0):
        raise ValueError("Finite nonnegative redshift/densities and positive H0 required")
    h = cosmo["hlittle"] * 3.2407e-18 * np.sqrt(
        cosmo["OMm"] * (1 + z)**3 + cosmo["OMr"] * (1 + z)**4 + cosmo["OMl"])
    if h <= 0 or not np.isfinite(h):
        raise ValueError("Native Hubble rate must be finite and positive")
    return float(h)


def rate_from_nion(nion, redshift, *, t_star, cosmo):
    """Instantaneous deterministic L-INTEGRAL rate per mean-baryon cell."""
    source = np.asarray(nion, dtype=np.float64)
    if not np.isfinite(t_star) or t_star <= 0:
        raise ValueError("Positive finite t_STAR required")
    if not np.isfinite(source).all() or np.any(source < 0):
        raise ValueError("Native n_ion must be finite and nonnegative; no clipping")
    rate = source * (native_hubble(redshift, cosmo) / t_star)
    if not np.isfinite(rate).all():
        raise ValueError("Native source rate overflow")
    return np.ascontiguousarray(rate, dtype=np.float64)


def native_time_interval(z_old, z_new, cosmo):
    """Integrate native |dtdz| (matter+Lambda), distinct from the SFR clock.

    This is the continuous double-precision form of cosmology.c::dtdz.  The
    integral is evaluated in seconds rather than using a redshift endpoint
    approximation.  Native hubble() additionally includes radiation.
    """
    if not np.isfinite([z_old, z_new]).all() or not z_old > z_new >= 0:
        raise ValueError("Time integration requires decreasing nonnegative redshift")
    if cosmo.get("OMk", 0.) != 0 or cosmo.get("wl", -1.) != -1:
        raise ValueError("Native history clock requires flat LambdaCDM")
    h0 = float(cosmo["hlittle"]) * 3.2407e-18
    om, ol = float(cosmo["OMm"]), float(cosmo["OMl"])
    if not np.isfinite([h0, om, ol]).all() or h0 <= 0 or om <= 0 or ol < 0:
        raise ValueError("Invalid native cosmology for the time interval")

    def clock(z):
        return np.sqrt(om + ol) / ((1 + z) * h0 * np.sqrt(om * (1 + z)**3 + ol))

    value, error = quad(clock, z_new, z_old, epsabs=1., epsrel=1e-11)
    return float(value), float(error)


def _metadata_dict(value):
    if isinstance(value, dict):
        return value
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind not in "US":
        raise ValueError("Metadata must be a scalar JSON string or decoded dictionary")
    try:
        metadata = json.loads(str(raw.item()))
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid native-history JSON metadata") from exc
    if not isinstance(metadata, dict):
        raise ValueError("Native-history metadata must decode to an object")
    return metadata


def validate_history(history, *, verify_hashes=True):
    """Validate a history mapping and return its decoded metadata.

    Numeric arrays are inspected without coercing their dtype.  Shape, time
    order, bounds and exact stored payload hashes are all checked.
    """
    missing = set(ARRAY_KEYS + ("metadata",)) - set(history)
    if missing:
        raise ValueError(f"Missing native-history fields: {sorted(missing)}")
    arrays = {key: np.asarray(history[key]) for key in ARRAY_KEYS}
    z, times, rates, capacity, xhi = (arrays[k] for k in ARRAY_KEYS)
    if any(arrays[k].dtype != np.dtype("float64") for k in ARRAY_KEYS[:-1]):
        raise ValueError("Native-history times, redshifts, rates and capacity must be float64")
    if xhi.dtype != np.dtype("float32"):
        raise ValueError("Contextual native_xhi must be float32")
    if (z.ndim != 1 or len(z) < 2 or times.shape != z.shape
            or not np.isfinite(z).all() or (z < 0).any() or not (np.diff(z) < 0).all()
            or not np.isfinite(times).all() or times[0] != 0
            or not (np.diff(times) > 0).all()):
        raise ValueError("History needs descending redshifts and increasing times starting at zero")
    if (capacity.ndim != 3 or len(set(capacity.shape)) != 1 or capacity.size == 0
            or not np.isfinite(capacity).all() or (capacity <= 0).any()
            or capacity.sum(dtype=np.longdouble) <= 0):
        raise ValueError("Fixed capacity must be a strictly positive nonempty cubic atom field")
    shape = (len(z),) + capacity.shape
    if rates.shape != shape or not np.isfinite(rates).all() or (rates < 0).any():
        raise ValueError("Instantaneous rates must be finite nonnegative endpoint fields")
    if xhi.shape != shape or not np.isfinite(xhi).all() or ((xhi < 0) | (xhi > 1)).any():
        raise ValueError("Contextual native neutral fractions must lie in [0, 1]")
    metadata = _metadata_dict(history["metadata"])
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported native-history schema version")
    if metadata.get("density_evolution") != "frozen_at_reference_redshift":
        raise ValueError("This fixture requires an explicitly frozen gas-density field")
    if metadata.get("rates_definition") != "native_n_ion_times_H_over_t_STAR":
        raise ValueError("Unrecognized instantaneous source-rate convention")
    reference_z = metadata.get("reference_redshift")
    if (not isinstance(reference_z, (int, float)) or isinstance(reference_z, bool)
            or not np.isfinite(reference_z) or reference_z < 0):
        raise ValueError("A finite nonnegative density reference redshift is required")
    if verify_hashes:
        hashes = metadata.get("array_sha256", {})
        for name, value in arrays.items():
            if hashes.get(name) != array_hash(value):
                raise ValueError(f"Native-history array hash mismatch: {name}")
    return metadata


def load_history(path, *, verify_hashes=True):
    """Load a non-pickle NPZ as arrays plus a decoded ``metadata`` dictionary."""
    with np.load(path, allow_pickle=False) as data:
        result = {key: np.array(data[key], copy=True) for key in data.files}
    result["metadata"] = validate_history(result, verify_hashes=verify_hashes)
    return result


def _write_fresh_history(path, history):
    """Install a complete NPZ atomically, without replacing an existing path."""
    validate_history(history)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise FileExistsError(f"Use a fresh history output: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **{key: history[key] for key in ARRAY_KEYS},
                                metadata=np.asarray(json.dumps(history["metadata"], sort_keys=True)))
            stream.flush()
            os.fsync(stream.fileno())
        # An exclusive hard link avoids the check/replace race of os.replace.
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def _positive_integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def build_native_history(*, output, dimension=32, box_len=200., threads=32, seed=1234,
                         z_start=25., z_end=5., intervals=320, reference_z=8.,
                         highres_dimension=None, es_output=None):
    """Generate and save native endpoint rates and one explicit frozen gas grid.

    ``intervals + 1`` endpoints are uniformly spaced in redshift; their physical
    times follow the integrated native matter+Lambda clock.  A non-endpoint
    reference redshift is separately generated from the same IC object.
    ``highres_dimension`` selects native IC DIM independently of the source
    grid; its default remains three times ``dimension``.
    """
    dimension = _positive_integer(dimension, "dimension", 2)
    highres_dimension = (3 * dimension if highres_dimension is None else
                        _positive_integer(highres_dimension, "highres_dimension", dimension))
    threads = _positive_integer(threads, "threads")
    intervals = _positive_integer(intervals, "intervals")
    seed = _positive_integer(seed, "seed")
    if not np.isfinite([box_len, z_start, z_end, reference_z]).all() or not (
            box_len > 0 and z_start > z_end >= 0 and reference_z >= 0):
        raise ValueError("Positive box length and descending nonnegative redshift range required")
    output = Path(output)
    if os.path.lexists(output):
        raise FileExistsError(f"Use a fresh history output: {output}")
    if es_output is not None:
        es_output = Path(es_output)
        if es_output.resolve() == output.resolve() or os.path.lexists(es_output):
            raise FileExistsError(f"Use a distinct fresh ES feature output: {es_output}")
        require_es_radius_capability()
    output.parent.mkdir(parents=True, exist_ok=True)

    native_dir = str(ROOT / "research/pc_reconsidered")
    if native_dir not in sys.path:
        sys.path.insert(0, native_dir)
    from native_driver import prepare_native, from_native_coeval
    import py21cmfast
    from py21cmfast import c_21cmfast
    from py21cmfast.drivers.coeval import generate_coeval
    from py21cmfast.drivers.single_field import compute_initial_conditions
    from py21cmfast.io.caching import OutputCache

    inputs, differences = prepare_native(dimension=dimension, box_len=box_len,
                                         threads=threads, random_seed=seed,
                                         highres_dimension=highres_dimension)
    if es_output is not None:
        inputs = inputs.evolve_input_structs(STORE_ES_RADIUS=True)
        differences["matter_options.STORE_ES_RADIUS"] = {"before": False, "after": True}
    if inputs.astro_options.HALO_SCALING_RELATIONS_MEDIAN:
        raise ValueError("Native rate conversion requires the mean-scaling integral convention")
    cosmo = {key: float(getattr(inputs.cosmo_params, key))
             for key in ("hlittle", "OMm", "OMr", "OMl", "OMk", "wl")}
    t_star = float(inputs.astro_params.t_STAR)
    redshifts = np.linspace(float(z_start), float(z_end), intervals + 1, dtype=np.float64)
    if not (np.diff(redshifts) < 0).all():
        raise ValueError("Requested redshift spacing is not numerically distinct")
    times = np.zeros_like(redshifts)
    interval_errors = []
    for i in range(intervals):
        duration, error = native_time_interval(redshifts[i], redshifts[i + 1], cosmo)
        times[i + 1] = times[i] + duration
        interval_errors.append(error)
    rates = np.empty((len(redshifts), dimension, dimension, dimension), dtype=np.float64)
    native_xhi = np.empty(rates.shape, dtype=np.float32)
    radius = np.empty(rates.shape, dtype=np.float32) if es_output is not None else None
    native_density = np.empty(rates.shape, dtype=np.float64) if es_output is not None else None
    endpoint_index = {float(z): i for i, z in enumerate(redshifts)}
    generated = sorted(set(endpoint_index) | {float(reference_z)}, reverse=True)
    provenance = [None] * len(redshifts)
    capacity = None
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="pc_native_history_", dir=output.parent) as temporary:
        cache = OutputCache(Path(temporary) / "cache")
        initial = compute_initial_conditions(inputs=inputs, write=False, regenerate=True, cache=cache)
        initial_density_sha = array_hash(initial.get("hires_density"))
        for coeval, requested in generate_coeval(inputs=inputs, initial_conditions=initial,
                out_redshifts=generated, regenerate=True, write=False, cache=cache, progressbar=False):
            z = float(coeval.redshift)
            if not requested:
                continue
            extracted = from_native_coeval(coeval)
            if z == float(reference_z):
                capacity = np.array(extracted.capacity, dtype=np.float64, copy=True)
            if z in endpoint_index:
                i = endpoint_index[z]
                rates[i] = rate_from_nion(extracted.source, z, t_star=t_star, cosmo=cosmo)
                native_xhi[i] = coeval.ionized_box.get("neutral_fraction")
                if es_output is not None:
                    radius[i] = coeval.ionized_box.get("es_crossing_radius")
                    native_density[i] = extracted.capacity
                provenance[i] = {
                    "redshift": z, "native_n_ion_sha256_float64": array_hash(extracted.source),
                    "native_density_capacity_sha256_float64": array_hash(extracted.capacity),
                    "native_n_ion_sum": float(extracted.source.sum(dtype=np.longdouble)),
                    "rate_sum_per_second": float(rates[i].sum(dtype=np.longdouble)),
                    "native_sfr_timescale_seconds": t_star / native_hubble(z, cosmo),
                    "native_xhi_volume_mean": float(native_xhi[i].mean(dtype=np.float64)),
                }
                stage = "native_source_endpoint"
            else:
                stage = "native_density_reference_only"
            print(json.dumps({"stage": stage, "redshift": z,
                              "elapsed_seconds": time.perf_counter() - started}), flush=True)
    if capacity is None or any(row is None for row in provenance):
        raise RuntimeError("Native driver did not return every requested endpoint/reference field")

    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                           text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    code_paths = [Path(__file__), ROOT / "research/pc_reconsidered/native_driver.py",
                  ROOT / "src/py21cmfast/src/scaling_relations.c",
                  ROOT / "src/py21cmfast/src/map_mass.c",
                  ROOT / "src/py21cmfast/src/EmissivityFields.c",
                  ROOT / "src/py21cmfast/src/cosmology.c"]
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "scope": "Native 21cmFAST time-varying sources for fixed-gas carried-state validation",
        "density_evolution": "frozen_at_reference_redshift",
        "reference_redshift": float(reference_z),
        "reference_redshift_is_source_endpoint": float(reference_z) in endpoint_index,
        "rates_definition": "native_n_ion_times_H_over_t_STAR",
        "rate_units": "native effective escaped photons per mean-baryon cell per second",
        "capacity_definition": "1 + native evolved density contrast at reference_redshift",
        "capacity_units": "native effective atom counts per mean-density cell",
        "density_weighting_applied_to_n_ion": False,
        "local_cumulative_n_ion_differenced": False,
        "native_n_ion_is_a_rate": False,
        "source_model": "L-INTEGRAL",
        "source_components": ["ACG"],
        "t_STAR": t_star,
        "rate_clock": "native hubble(): matter + radiation + Lambda; float32 redshift argument",
        "time_clock": "integrated continuous native dtdz: matter + Lambda",
        "time_origin": "zero at first supplied redshift; no emission before z_start supplied",
        "time_interval_quadrature_errors_seconds": interval_errors,
        "redshift_sampling": "uniform redshift endpoints; actual native time intervals",
        "contextual_native_ES": "independent cumulative n_ion coevals with evolving native density; not a matched integrated-rate control",
        "same_initial_conditions_for_all_nodes": True,
        "initial_hires_density_sha256": initial_density_sha,
        "random_seed": seed,
        "dimension": dimension,
        "highres_dimension": highres_dimension,
        "box_len_cMpc": float(box_len),
        "hlittle": cosmo["hlittle"],
        "box_hMpc": float(box_len) * cosmo["hlittle"],
        "native_threads": threads,
        "redshift_start": float(z_start), "redshift_end": float(z_end), "intervals": intervals,
        "source_provenance": provenance,
        "physical_restrictions": [
            "Qin20 astrophysical starting parameters with L-INTEGRAL ACG sources",
            "No MCG, recombinations, X-ray/spin-temperature evolution or UV photoheating feedback",
            "No photon-conservation correction in native source/control generation",
            "Native source density/displacements evolve; transport gas density is frozen",
            "Effective native baryon-count convention, not a full H/He radiation ledger",
            "Native ES outputs are contextual and do not use the integrated source budget",
            "No pre-z_start photons or ionized initial state are supplied by this fixture",
        ],
        "input_differences_from_Qin20_with_same_geometry": differences,
        "inputs": inputs.asdict(include_cosmo_tables="never", only_cstruct_params=False),
        "cosmology_for_clocks": cosmo,
        "py21cmfast_version": py21cmfast.__version__, "git_revision": revision,
        "native_extension_path": c_21cmfast.__file__,
        "native_extension_sha256": hashlib.sha256(Path(c_21cmfast.__file__).read_bytes()).hexdigest(),
        "source_code_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in code_paths},
        "native_generation_seconds": time.perf_counter() - started,
    }
    result = dict(redshifts=redshifts, times_seconds=times, rates=rates,
                  capacity=capacity, native_xhi=native_xhi, metadata=metadata)
    metadata["array_sha256"] = {key: array_hash(result[key]) for key in ARRAY_KEYS}
    _write_fresh_history(output, result)
    if es_output is not None:
        write_es_features(es_output, result, radius, native_density)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--es-output", type=Path, default=None,
                        help="Optional separate native ES xHI/crossing-radius feature archive")
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--highres-dimension", type=int, default=None,
                        help="Native IC DIM; defaults to 3 * dimension")
    parser.add_argument("--box-len", type=float, default=200.)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--z-start", type=float, default=25.)
    parser.add_argument("--z-end", type=float, default=5.)
    parser.add_argument("--intervals", type=int, default=320)
    parser.add_argument("--reference-z", type=float, default=8.)
    options = vars(parser.parse_args())
    result = build_native_history(**options)
    print(json.dumps({"output": str(options["output"]), "endpoints": len(result["redshifts"]),
                      "scope": result["metadata"]["scope"],
                      "native_generation_seconds": result["metadata"]["native_generation_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
