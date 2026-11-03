"""Compare analytic absorption, original SCRIPT and native C²-Ray on shared inputs.

Native snapshots are checksum-verified before execution. Photon budgets and
boundary treatments are reported separately from field and power differences.
The C²-Ray result is an independent numerical reference, not an exact field.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import inspect
import itertools
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from research.pc_field_emulator.evaluation import field_metrics, shell_spectra

LOGGER = logging.getLogger(__name__)
MPC_CM = 3.0856775814913673e24


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(record), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def validate_physics(inputs: dict[str, Any]) -> None:
    """Require the restricted native source physics used by all three methods.

    Parameters
    ----------
    inputs
        Archived native input dictionaries from the snapshot receipt.
    """
    options = inputs["astro_options"]
    checks = {
        "RECOMB_MODEL": "none",
        "CELL_RECOMB": False,
        "USE_TS_FLUCT": False,
        "USE_MCGS": False,
        "USE_REIONIZATION_PHOTOHEATING_FEEDBACK": False,
        "PHOTON_CONS_TYPE": "no-photoncons",
    }
    for name, expected in checks.items():
        if options.get(name) != expected:
            raise ValueError(
                f"Matched frozen-H comparison requires {name}={expected!r}"
            )
    if inputs["matter_options"].get("SOURCE_MODEL") != "L-INTEGRAL":
        raise ValueError("Matched native snapshots require SOURCE_MODEL=L-INTEGRAL")


def _block_mean(value: np.ndarray, target: int) -> np.ndarray:
    n = value.shape[0]
    factor = n // target
    return value.reshape(target, factor, target, factor, target, factor).mean(
        axis=(1, 3, 5)
    )


def coarsen_snapshot(
    fields: dict[str, np.ndarray], target: int
) -> dict[str, np.ndarray]:
    """Restrict shared native inputs without changing their physical total counts.

    Parameters
    ----------
    fields
        Fine native source, capacity, neutral-fraction and brightness arrays.
    target
        Explicit target dimension dividing the input dimension exactly.

    Returns
    -------
    dict
        Source and capacity in mean-density coarse-cell units, atom-weighted
        native ES neutral fraction and volume-averaged native brightness.
    """
    n = fields["source"].shape[0]
    if type(target) is not int or target < 2 or target > n or n % target:
        raise ValueError(
            "coarsen_to must be an integer >=2 dividing the native dimension"
        )
    if target == n:
        return dict(fields)
    capacity = _block_mean(fields["capacity"], target)
    return {
        "source": _block_mean(fields["source"], target),
        "capacity": capacity,
        "native_es_xhi": _block_mean(
            fields["capacity"] * fields["native_es_xhi"], target
        )
        / capacity,
        "native_brightness": _block_mean(fields["native_brightness"], target),
    }


def load_native_snapshot(
    directory: str | Path, *, brightness: str = "native", coarsen_to: int | None = None
) -> dict[str, Any]:
    """Load the original cumulative photon and atom arrays with provenance checks.

    Parameters
    ----------
    directory
        Snapshot folder containing source.npz, native.h5, native.json and receipt.json.
    brightness
        ``native`` additionally re-extracts H5 arrays with the archived runtime;
        ``none`` skips brightness; ``saturated_proxy`` explicitly uses a common
        high-spin-temperature approximate brightness for portable diagnostics.
    coarsen_to
        Explicit common coarse grid; never inferred from available memory or cost.

    Returns
    -------
    dict
        Shared arrays, metadata and an optional native coeval object.
    """
    if brightness not in {"native", "none", "saturated_proxy"}:
        raise ValueError("brightness must be native, none, or saturated_proxy")
    directory = Path(directory).resolve()
    receipt = json.loads((directory / "receipt.json").read_text())
    for name in ("source.npz", "native.json", "native.h5"):
        if _digest(directory / name) != receipt["files"][name]:
            raise ValueError(f"Native snapshot checksum failed: {directory / name}")
    metadata = json.loads((directory / "native.json").read_text())
    if metadata["identity"] != receipt["identity"]:
        raise ValueError("Native metadata and receipt identity differ")
    inputs = receipt["identity"]["inputs"]
    validate_physics(inputs)
    if metadata.get("source", {}).get("cumulative_budget_not_rate") is not True:
        raise ValueError(
            "Snapshot must identify cumulative native photon counts, not an emissivity rate"
        )
    with np.load(directory / "source.npz", allow_pickle=False) as saved:
        fields = {
            name: np.array(saved[name])
            for name in ("source", "capacity", "native_es_xhi", "native_brightness")
        }
    for name, values in fields.items():
        expected = metadata.get("array_sha256", {}).get(name)
        if expected is not None and _array_digest(values) != expected:
            raise ValueError(f"Native array checksum failed: {name}")
    source, capacity = fields["source"], fields["capacity"]
    if (
        source.shape != capacity.shape
        or source.ndim != 3
        or len(set(source.shape)) != 1
    ):
        raise ValueError("Native source and capacity must share a cubic grid")
    if not np.isfinite(source).all() or (source < 0).any():
        raise ValueError("Native source must be finite and nonnegative")
    if not np.isfinite(capacity).all() or (capacity <= 0).any():
        raise ValueError("Native capacity must be finite and strictly positive")
    record = receipt["record"]
    redshift = float(record["redshift"])
    if redshift != float(receipt["identity"]["redshift"]):
        raise ValueError("Snapshot redshift mismatch")
    from research.pc_matched.workflow import opacity_parameters

    # Preserve the established native effective H+He-nuclei count convention:
    # every effective atom has H opacity, with no independent He chemistry.
    # The helper's spectral value is unused; compare config specifies sigma.
    opacity = opacity_parameters({"inputs": inputs}, energy_ev=13.6)
    n_h0 = opacity["hydrogen_density_z0_cm3"]
    box_len = float(inputs["simulation_options"]["BOX_LEN"])
    native = None
    if brightness == "native":
        from py21cmfast.drivers.coeval import Coeval
        from research.pc_native_rt.workflow import _input_dict, _runtime_identity
        from research.pc_reconsidered.native_driver import from_native_coeval

        runtime = _runtime_identity()
        if (
            runtime["native_extension_sha256"]
            != receipt["identity"]["extension_sha256"]
        ):
            raise ValueError(
                "Native brightness requires the snapshot's original native extension"
            )
        native = Coeval.from_file(directory / "native.h5")
        if _input_dict(native.inputs) != inputs:
            raise ValueError("Native H5 input parameters differ from receipt")
        extracted = from_native_coeval(native)
        for name, actual in (
            ("source", extracted.source),
            ("capacity", extracted.capacity),
            ("native_es_xhi", native.ionized_box.get("neutral_fraction")),
            ("native_brightness", native.brightness_temp),
        ):
            if not np.array_equal(fields[name], actual):
                raise ValueError(f"Native H5 re-extraction differs: {name}")
    archived_hashes = {name: _array_digest(value) for name, value in fields.items()}
    native_capacity = capacity
    native_dimension = source.shape[0]
    if coarsen_to is not None:
        fields = coarsen_snapshot(fields, coarsen_to)
        source, capacity = fields["source"], fields["capacity"]
    fields.update(
        {
            "native": native,
            "_native_capacity": native_capacity,
            "metadata": {
                "directory": str(directory),
                "receipt_sha256": _digest(directory / "receipt.json"),
                "files": {
                    name: receipt["files"][name]
                    for name in ("source.npz", "native.json", "native.h5")
                },
                "archived_array_sha256": archived_hashes,
                "array_sha256": {
                    name: _array_digest(value) for name, value in fields.items()
                },
                "native_dimension": native_dimension,
                "coarsening_factor": native_dimension // source.shape[0],
                "coarsening": "explicit conservative block means in coarse-cell count units; native ES atom-weighted, brightness volume-averaged"
                if coarsen_to is not None
                else "none",
                "resolution_scope": "Shared coarse-input transport test; not a rerun of native source physics or a native-grid convergence certificate"
                if coarsen_to is not None
                else "original archived native source grid",
                "seed": int(record["seed"]),
                "redshift": redshift,
                "box_len": box_len,
                "dimension": source.shape[0],
                "inputs": inputs,
                "nH_mean_comoving_cm3": float(n_h0),
                "atoms_per_count_unit": float(
                    n_h0 * (box_len / source.shape[0] * MPC_CM) ** 3
                ),
                "source_per_atom": float(
                    np.sum(source, dtype=np.longdouble)
                    / np.sum(capacity, dtype=np.longdouble)
                ),
                "source_convention": "raw cumulative native n_ion; no second density factor, time integral or source rescaling",
                "count_conversion": "source and capacity multiplied by the same native effective H+He-nuclei atoms-per-base-cell factor; every effective atom has H opacity",
                "density_definition": opacity["density_definition"],
                "density_formula": opacity["density_formula"],
                "native_h5_reextracted": native is not None,
                "brightness_mode": brightness,
            },
        }
    )
    return fields


def independent_budget(
    source: np.ndarray,
    capacity: np.ndarray,
    xhi: np.ndarray,
    *,
    remaining: float | None,
    escaped: float | None,
    unemitted: float = 0.0,
) -> dict[str, Any]:
    """Audit output fields against independently supplied radiation reservoirs.

    Parameters
    ----------
    source, capacity, xhi
        Shared photon counts, atom counts and the output neutral fraction.
    remaining, escaped
        Measured unabsorbed inventories in the same count units; ``None`` means
        unavailable, never zero inferred from the field difference.
    unemitted
        Source budget not yet injected by an incomplete analytic calculation.

    Returns
    -------
    dict
        Field-derived absorption and a residual only when both inventories exist.
    """
    emitted = np.sum(source, dtype=np.longdouble)
    atoms = np.sum(capacity, dtype=np.longdouble)
    absorbed = np.sum(capacity * (1.0 - xhi), dtype=np.longdouble)
    if not np.isfinite(unemitted) or unemitted < 0 or unemitted > emitted:
        raise ValueError("unemitted must be finite and within the source photon budget")
    residual = (
        None
        if remaining is None or escaped is None
        else float(emitted - absorbed - remaining - escaped - unemitted)
    )
    return {
        "source_photon_budget": float(emitted),
        "emitted_photons": float(emitted) - float(unemitted),
        "unemitted_photons": float(unemitted),
        "atom_capacity": float(atoms),
        "absorbed_photons_from_field": float(absorbed),
        "remaining_photons": remaining,
        "escaped_photons": escaped,
        "unaccounted_photons": float(emitted - unemitted - absorbed),
        "photon_residual": residual,
        "relative_photon_residual": None
        if residual is None
        else abs(residual) / max(float(emitted), np.finfo(float).tiny),
        "xHI_mass": float(np.sum(capacity * xhi, dtype=np.longdouble) / atoms),
        "xHI_volume": float(np.mean(xhi)),
        "bounded": bool(
            np.isfinite(xhi).all()
            and (xhi >= -1e-12).all()
            and (xhi <= 1 + 1e-12).all()
        ),
    }


def pair_statistics(
    fields: dict[str, np.ndarray],
    capacity: np.ndarray,
    box_len: float,
    *,
    brightness: dict[str, np.ndarray] | None,
) -> dict[str, Any]:
    """Measure same-snapshot field differences and common-binning power spectra.

    Parameters
    ----------
    fields
        Neutral-fraction arrays by method, including ``script``.
    capacity, box_len
        Common atom weights and comoving box length in cMpc.
    brightness
        Common-definition brightness fields, or ``None`` to skip 21-cm power.

    Returns
    -------
    dict
        Pair metrics and auto/cross spectra with zero-power validity masks.
    """
    pairs = {}
    names = [
        name for name in ("analytic", "script", "c2ray", "native_es") if name in fields
    ]
    for first, second in itertools.combinations(names, 2):
        metrics = field_metrics(1 - fields[first], 1 - fields[second], capacity)
        metrics["same_mean_neutral_fraction"] = abs(metrics["delta_Q_mass"]) <= 1e-4
        metrics["comparison"] = (
            "same redshift and source budget; no field renormalization or matched-xHI interpolation"
        )
        pairs[f"{first}_vs_{second}"] = metrics

    def spectrum(
        values: dict[str, np.ndarray], reference: str = "script"
    ) -> dict[str, Any]:
        converted = {
            ("teacher" if key == reference else key): val for key, val in values.items()
        }
        result = shell_spectra(converted, box_len)
        result = {key.replace("teacher", reference): val for key, val in result.items()}
        result["reference"] = f"{reference}; reference is not assumed exact"
        result["below_half_nyquist"] = [
            k <= 0.5 * result["k_max_cMpc_inv"] for k in result["k_cMpc_inv"]
        ]
        result["valid_reference_bins"] = [
            value is not None for value in result["ratio_analytic"]
        ]
        return result

    return {
        "pairs": pairs,
        "spectra_xHI": spectrum(fields),
        "spectra_21cm": None if brightness is None else spectrum(brightness),
        "spectra_xHI_vs_c2ray": spectrum(fields, "c2ray")
        if "c2ray" in fields
        else None,
        "spectra_21cm_vs_c2ray": spectrum(brightness, "c2ray")
        if brightness is not None and "c2ray" in brightness
        else None,
    }


def run_c2ray_cached(
    source: np.ndarray,
    capacity: np.ndarray,
    *,
    physical: dict[str, float],
    settings: dict[str, Any],
    threads: int,
    workdir: Path,
    cache_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Reuse only byte-verified C²-Ray results with identical inputs and build.

    Parameters
    ----------
    source, capacity
        Exact common normalized count arrays passed to the reference backend.
    physical, settings
        Shared physical parameters and C²-Ray-specific keyword arguments.
    threads
        Requested allocation, retained in the result identity.
    workdir
        Fresh native output path when no shared cache is requested.
    cache_directory
        Optional persistent cache outside the frozen source tree. The key
        includes input bytes, all settings, backend source and build identity.

    Returns
    -------
    dict
        Native backend result, with an explicit cache-hit and timing record.
    """
    from research.pc_analytic import c2ray_reference as backend

    if cache_directory is None:
        result = backend.run(
            source, capacity, **physical, threads=threads, workdir=workdir, **settings
        )
        result["audit"]["cache_hit"] = False
        return result
    import os

    started = time.perf_counter()
    root = Path(cache_directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    upstream = settings.get("upstream") or os.environ.get("PC_C2RAY_SOURCE")
    if upstream is None:
        raise RuntimeError("C²-Ray cache requires upstream or PC_C2RAY_SOURCE")
    with (root / "build.lock").open("a") as build_lock:
        fcntl.flock(build_lock, fcntl.LOCK_EX)
        build = backend.build(upstream, root / "build", settings.get("compiler"))
    bound = inspect.signature(backend.run).bind(
        source, capacity, **physical, threads=threads, workdir=workdir, **settings
    )
    bound.apply_defaults()
    effective_parameters = {
        name: value
        for name, value in bound.arguments.items()
        if name not in {"source", "capacity", "workdir"}
    }
    effective_parameters["upstream"] = str(Path(upstream).resolve())
    identity = {
        "source_sha256": _array_digest(source),
        "capacity_sha256": _array_digest(capacity),
        "shape": list(source.shape),
        "source_dtype": str(source.dtype),
        "capacity_dtype": str(capacity.dtype),
        "physical": physical,
        "effective_parameters": effective_parameters,
        "backend_sha256": _digest(Path(backend.__file__)),
        "build_identity": build["identity"],
    }
    key = hashlib.sha256(
        json.dumps(_jsonable(identity), sort_keys=True).encode()
    ).hexdigest()
    entry = root / key
    entry.mkdir(exist_ok=True)
    with (entry / "result.lock").open("a") as result_lock:
        fcntl.flock(result_lock, fcntl.LOCK_EX)
        receipt_path = entry / "receipt.json"
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text())
            if receipt["identity"] != _jsonable(identity):
                raise ValueError("C²-Ray cache identity changed")
            for name, expected in receipt["files"].items():
                if _digest(entry / name) != expected:
                    raise ValueError(f"C²-Ray cache checksum failed: {name}")
            native_output = Path(receipt["native_output"])
            if _digest(native_output) != receipt["native_output_sha256"]:
                raise ValueError("C²-Ray cached native output checksum failed")
            with np.load(entry / "fields.npz", allow_pickle=False) as saved:
                result = {name: np.array(saved[name]) for name in saved.files}
            result.update(json.loads((entry / "metadata.json").read_text()))
            result["audit"]["cache_hit"] = True
            result["audit"]["cached_backend_elapsed_seconds"] = result["audit"][
                "elapsed_seconds"
            ]
            result["audit"]["elapsed_seconds"] = 0.0
            result["audit"]["native_execution_seconds_this_call"] = 0.0
            result["audit"]["cache_lookup_seconds"] = time.perf_counter() - started
            result["audit"]["cache_key"] = key
            return result
        # A failed attempt is preserved, never overwritten or accepted as a
        # completed reference. Only a successful backend writes a receipt.
        native_dir = Path(tempfile.mkdtemp(prefix="native-", dir=entry))
        result = backend.run(
            source,
            capacity,
            **physical,
            threads=threads,
            workdir=native_dir,
            **settings,
        )
        if not result["audit"].get("complete", False):
            raise RuntimeError("Cannot cache an incomplete C²-Ray result")
        arrays = {
            name: value
            for name, value in result.items()
            if isinstance(value, np.ndarray)
        }
        metadata = {name: value for name, value in result.items() if name not in arrays}
        np.savez_compressed(entry / "fields.npz", **arrays)
        _write_json(entry / "metadata.json", metadata)
        native_output = native_dir / "output.bin"
        output_hash = _digest(native_output)
        if result["audit"].get("output_sha256") != output_hash:
            raise ValueError("C²-Ray completed output differs from its audit")
        _write_json(
            receipt_path,
            {
                "identity": identity,
                "files": {
                    name: _digest(entry / name)
                    for name in ("fields.npz", "metadata.json")
                },
                "native_output": str(native_output),
                "native_output_sha256": output_hash,
            },
        )
        result["audit"]["cache_hit"] = False
        result["audit"]["native_execution_seconds_this_call"] = result["audit"][
            "elapsed_seconds"
        ]
        result["audit"]["cache_key"] = key
        return result


def _completed_snapshot(
    folder: Path, identity_sha256: str, *, require_c2ray: bool
) -> dict[str, Any] | None:
    receipt_path = folder / "receipt.json"
    if not receipt_path.is_file():
        return None
    receipt = json.loads(receipt_path.read_text())
    if receipt["identity_sha256"] != identity_sha256:
        raise ValueError("Saved snapshot belongs to a different run identity")
    for name, expected in receipt["files"].items():
        if _digest(folder / name) != expected:
            raise ValueError(f"Saved comparison checksum failed: {name}")
    row = json.loads((folder / "metrics.json").read_text())
    if not row["complete"] or (require_c2ray and not row["three_methods_completed"]):
        return None
    return row


def _snapshot_directories(config: dict[str, Any]) -> list[Path]:
    if "snapshot_directories" in config:
        return [Path(value).resolve() for value in config["snapshot_directories"]]
    from research.pc_coeval.run import discover

    _, _, snapshots = discover(config["input_run"], config["redshifts"])
    seeds = config.get("seeds")
    return [
        Path(s["receipt_path"]).parent
        for s in snapshots
        if seeds is None or s["record"]["seed"] in seeds
    ]


def _brightness(
    data: dict[str, Any], fields: dict[str, np.ndarray], mode: str
) -> dict[str, np.ndarray] | None:
    if mode == "none":
        return None
    if mode == "saturated_proxy":
        z = data["metadata"]["redshift"]
        return {
            name: 27.0 * np.sqrt((1.0 + z) / 10.0) * data["capacity"] * xhi
            for name, xhi in fields.items()
        }
    from research.pc_coeval.run import native_brightness

    output = {}
    factor = data["metadata"]["coarsening_factor"]
    for name, xhi in fields.items():
        if name == "native_es":
            output[name] = data["native_brightness"]
            continue
        expanded = xhi
        if factor > 1:
            for axis in range(3):
                expanded = np.repeat(expanded, factor, axis=axis)
        rho = data["_native_capacity"]
        value = native_brightness(data["native"], rho * (1.0 - expanded), rho)
        output[name] = _block_mean(value, xhi.shape[0]) if factor > 1 else value
    return output


def _mfp(
    fields: dict[str, np.ndarray], box_len: float, rays: int, threads: int
) -> dict[str, Any]:
    from research.pc_coeval.mfp import compare_mfp, make_draws, measure_mfp

    draws = make_draws(rays, seed=20261103)
    result = {}
    for phase in ("ionized", "neutral"):
        samples = {
            name: measure_mfp(value, box_len, phase=phase, draws=draws, threads=threads)
            for name, value in fields.items()
        }
        result[phase] = {
            "methods": {name: sample["summary"] for name, sample in samples.items()},
            "analytic_vs_script": compare_mfp(samples["analytic"], samples["script"]),
        }
        if "c2ray" in samples:
            result[phase]["analytic_vs_c2ray"] = compare_mfp(
                samples["analytic"], samples["c2ray"]
            )
    return result


def _plot(output: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for seed in sorted({row["seed"] for row in rows}):
        selected = sorted(
            (row for row in rows if row["seed"] == seed),
            key=lambda row: -row["redshift"],
        )
        redshifts = [row["redshift"] for row in selected]
        for method, style in (
            ("analytic", "-o"),
            ("script", "--"),
            ("c2ray", "-s"),
            ("native_es", ":"),
        ):
            if all(method in row["budgets"] for row in selected):
                axes[0].plot(
                    redshifts,
                    [row["budgets"][method]["xHI_mass"] for row in selected],
                    style,
                    label=f"{method}, seed {seed}",
                )
        for name in ("analytic_vs_script", "analytic_vs_c2ray"):
            if all(name in row["pairs"] for row in selected):
                axes[1].plot(
                    redshifts,
                    [row["pairs"][name]["MAE"] for row in selected],
                    "o-",
                    label=f"{name}, {seed}",
                )
        midpoint = min(
            selected, key=lambda row: abs(row["budgets"]["script"]["xHI_mass"] - 0.5)
        )
        for reference, suffix in (("SCRIPT", ""), ("C2-Ray", "_vs_c2ray")):
            spectra = midpoint.get("spectra_21cm" + suffix) or midpoint.get(
                "spectra_xHI" + suffix
            )
            if spectra is not None:
                axes[2].plot(
                    spectra["k_cMpc_inv"],
                    spectra["ratio_analytic"],
                    label=f"analytic/{reference}, z={midpoint['redshift']:g}, {seed}",
                )
    axes[0].set(xlabel="Redshift", ylabel="Mass-weighted neutral fraction")
    axes[1].set(xlabel="Redshift", ylabel="Neutral-field MAE")
    kind = (
        "21-cm"
        if all(row["spectra_21cm"] is not None for row in rows)
        else "neutral-field"
    )
    axes[2].set(xlabel=r"$k$ [cMpc$^{-1}$]", ylabel=f"{kind} power ratio")
    axes[2].axhline(1, color="gray", linewidth=0.8)
    for ax in axes[:2]:
        ax.invert_xaxis()
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    fig.suptitle(
        "Shared native source/atom arrays; boundary and numerical settings recorded in report"
    )
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=170)
    plt.close(fig)


def run(
    config: dict[str, Any], output: str | Path, *, require_c2ray: bool = True
) -> dict[str, Any]:
    """Execute and report the same-input comparison without changing archived data.

    Parameters
    ----------
    config
        Snapshot selection, declared cross-section and backend settings.
    output
        Fresh output directory, or one with an identical saved identity.
    require_c2ray
        Raise if the real C²-Ray backend cannot complete; never replace it by a
        homemade reference or silently mark a two-method run as three-method.

    Returns
    -------
    dict
        Full comparison report including provenance and explicit model differences.
    """
    from research.pc_analytic.solver import solve
    from research.pc_script_reference import kernel as script_kernel

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    directories = _snapshot_directories(config)
    if not directories or len(set(directories)) != len(directories):
        raise ValueError("Require at least one unique native snapshot directory")
    sigma = float(config["sigma_cm2"])
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma_cm2 must be explicitly supplied, finite and positive")
    threads = int(config.get("threads", 1))
    mode = config.get("brightness", "native")
    sources = {
        str(path.relative_to(Path(__file__).resolve().parents[2])): _digest(path)
        for path in sorted(Path(__file__).parent.glob("*"))
        if path.suffix in {".py", ".cpp", ".f90", ".F90"}
    }
    identity = {
        "config": config,
        "require_c2ray": require_c2ray,
        "implementation_sha256": sources,
        "snapshot_receipt_sha256": {
            str(path): _digest(path / "receipt.json") for path in directories
        },
        "script_provenance": script_kernel.provenance(),
    }
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError(
            "Run inputs or implementation changed; use a fresh output directory"
        )
    _write_json(identity_path, identity)
    (output / "COMPLETED").unlink(missing_ok=True)
    report = {
        "schema": 1,
        "identity": identity,
        "snapshots": [],
        "complete": False,
        "convergence_certified": False,
        "comparison_is_truth_error": False,
        "common_physics": {
            "gas": "frozen native effective H+He nuclei, each with H opacity; no independent He chemistry",
            "initial_xHII": 0.0,
            "recombinations": False,
            "collisional_ionization": False,
            "thermal_evolution": False,
            "photoheating_feedback": False,
            "sigma_cm2": sigma,
            "source_schedule": "constant source proportions; cumulative dose refined by exposure_steps",
        },
        "method_differences": {
            "analytic": "directional absorption and shared neutral depletion; dose and angular discretization",
            "script": "original instantaneous periodic photon allocation; no spectrum, optical-depth attenuation, or dose-time evolution",
            "c2ray": "native short-characteristics reference; see backend provenance and escape diagnostics",
            "boundary": "SCRIPT closes a periodic photon box. Analytic source_periodic_cube matches C2-Ray's intended one-source-centred-domain boundary; analytic periodic recycles photons. C2-Ray subbox truncation is a separate numerical control. Raw loss counters are retained, not treated as measured escape or renormalized away.",
        },
        "brightness_definition": {
            "native": "original archived native brightness_temperature; identical gas, no Ts evolution or RSD",
            "none": "21-cm power not evaluated",
            "saturated_proxy": "27 sqrt((1+z)/10) (1+delta) xHI mK; explicit approximate common diagnostic",
        }[mode],
        "timing_scope": "Backend call including backend-specific build and IO if not cached; elapsed times are not pure-kernel speedups",
        "grid_caveat": "Single-grid results do not establish spatial convergence; half-Nyquist is only a diagnostic mask",
    }
    _write_json(output / "comparison_report.json", report)
    for directory in directories:
        started = time.perf_counter()
        data = load_native_snapshot(
            directory, brightness=mode, coarsen_to=config.get("coarsen_to")
        )
        meta, source, capacity = data["metadata"], data["source"], data["capacity"]
        folder = (
            output
            / f"seed{meta['seed']}"
            / ("z" + f"{meta['redshift']:g}".replace(".", "p"))
        )
        folder.mkdir(parents=True, exist_ok=True)
        reused = _completed_snapshot(
            folder, _digest(identity_path), require_c2ray=require_c2ray
        )
        if reused is not None:
            report["snapshots"].append(reused)
            _write_json(output / "comparison_report.json", report)
            LOGGER.info("reuse completed seed=%s z=%g", meta["seed"], meta["redshift"])
            continue
        LOGGER.info(
            "compare seed=%s z=%g N=%s", meta["seed"], meta["redshift"], source.shape[0]
        )
        physical = {
            "box_len": meta["box_len"],
            "redshift": meta["redshift"],
            "nH_mean_comoving_cm3": meta["nH_mean_comoving_cm3"],
            "sigma_cm2": sigma,
        }
        kwargs = dict(config.get("analytic", {}))
        if kwargs.get("initial_ionized") is not None:
            raise ValueError(
                "Three-way comparison requires a fully neutral initial state"
            )
        analytic = solve(source, capacity, **physical, threads=threads, **kwargs)
        script_qi, script_audit = script_kernel.run(
            source,
            capacity,
            build_dir=output / "_build" / "script",
            timeout_seconds=float(config.get("script_timeout_seconds", 300)),
        )
        fields = {
            "analytic": 1.0 - np.divide(analytic["ionized"], capacity),
            "script": 1.0 - script_qi,
            "native_es": data["native_es_xhi"],
        }
        audits = {"analytic": analytic["audit"], "script": script_audit}
        c2 = None
        try:
            c2_kwargs = dict(config.get("c2ray", {}))
            c2 = run_c2ray_cached(
                source,
                capacity,
                physical=physical,
                settings=c2_kwargs,
                threads=threads,
                workdir=folder / "c2ray",
                cache_directory=config.get("c2ray_cache_directory"),
            )
            fields["c2ray"] = 1.0 - np.divide(c2["ionized"], capacity)
            audits["c2ray"] = c2["audit"]
        except (ImportError, RuntimeError, FileNotFoundError) as error:
            audits["c2ray"] = {
                "complete": False,
                "status": "unavailable_or_failed",
                "reason": str(error),
            }
            if require_c2ray:
                _write_json(
                    folder / "failure.json", {"metadata": meta, "audits": audits}
                )
                raise
        radiation = analytic.get("remaining_radiation")
        remaining = (
            None if radiation is None else float(np.sum(radiation, dtype=np.longdouble))
        )
        escaped_raw = analytic.get("escaped")
        escaped = (
            None
            if escaped_raw is None
            else float(np.sum(escaped_raw, dtype=np.longdouble))
        )
        unemitted = float(
            np.sum(analytic.get("unemitted_source", 0.0), dtype=np.longdouble)
        )
        budgets = {
            "analytic": independent_budget(
                source,
                capacity,
                fields["analytic"],
                remaining=remaining,
                escaped=escaped,
                unemitted=unemitted,
            ),
            "script": independent_budget(
                source,
                capacity,
                fields["script"],
                remaining=script_audit["surplus_photons"],
                escaped=0.0,
            ),
            "native_es": independent_budget(
                source, capacity, fields["native_es"], remaining=None, escaped=None
            ),
        }
        budgets["script"]["remaining_scope"] = (
            "global surplus inferred from input totals; original kernel does not return a photon reservoir"
        )
        if c2 is not None:
            escaped_c2 = c2.get("escaped_photons", c2["audit"].get("escaped_photons"))
            remaining_c2 = c2.get(
                "remaining_photons", c2["audit"].get("remaining_photons")
            )
            budgets["c2ray"] = independent_budget(
                source,
                capacity,
                fields["c2ray"],
                remaining=remaining_c2,
                escaped=escaped_c2,
            )
        temperatures = _brightness(data, fields, mode)
        metrics = pair_statistics(
            fields, capacity, meta["box_len"], brightness=temperatures
        )
        arrays = {
            "source": source,
            "capacity": capacity,
            **{f"{name}_xHI": value for name, value in fields.items()},
        }
        if temperatures is not None:
            arrays.update(
                {f"{name}_brightness": value for name, value in temperatures.items()}
            )
        if radiation is not None:
            arrays["analytic_remaining_radiation"] = np.sum(radiation, axis=0)
        if escaped_raw is not None:
            arrays["analytic_escaped_by_direction"] = np.asarray(escaped_raw)
        arrays["analytic_unemitted_source"] = np.asarray(
            analytic.get("unemitted_source", np.zeros_like(source))
        )
        np.savez_compressed(folder / "fields.npz", **arrays)
        row = {
            "seed": meta["seed"],
            "redshift": meta["redshift"],
            "input": meta,
            "budgets": budgets,
            "audits": audits,
            "analytic_metadata": analytic.get("metadata"),
            "boundary_comparison": {
                "analytic": analytic["audit"].get(
                    "boundary", kwargs.get("boundary", "periodic")
                ),
                "script": "closed_periodic_photon_allocation",
                "c2ray": None if c2 is None else c2["audit"].get("boundary"),
                "analytic_vs_script_intended_domain_matched": analytic["audit"].get(
                    "boundary", kwargs.get("boundary", "periodic")
                )
                == "periodic",
                "analytic_vs_c2ray_intended_domain_matched": c2 is not None
                and analytic["audit"].get(
                    "boundary", kwargs.get("boundary", "periodic")
                )
                == "source_periodic_cube",
                "all_three_identical": False
                if c2 is not None
                else kwargs.get("boundary", "periodic") == "periodic",
                "scope": "Intended domain only; C2-Ray subbox loss threshold and numerical photon residual remain separate diagnostics",
            },
            "c2ray_provenance": None if c2 is None else c2.get("provenance"),
            "elapsed_seconds": time.perf_counter() - started,
            **metrics,
        }
        rays = int(config.get("mfp_rays", 4096))
        if rays > 0:
            row["geometric_mfp"] = _mfp(fields, meta["box_len"], rays, threads)
        row["complete"] = bool(
            analytic["audit"].get("complete", False)
            and script_audit["passed"]
            and (
                (c2 is not None and c2["audit"].get("complete", False))
                or not require_c2ray
            )
        )
        row["three_methods_completed"] = bool(
            c2 is not None
            and c2["audit"].get("complete", False)
            and analytic["audit"].get("complete", False)
            and script_audit["passed"]
        )
        _write_json(folder / "metrics.json", row)
        _write_json(
            folder / "receipt.json",
            {
                "identity_sha256": _digest(identity_path),
                "files": {
                    name: _digest(folder / name)
                    for name in ("fields.npz", "metrics.json")
                },
            },
        )
        report["snapshots"].append(row)
        _write_json(output / "comparison_report.json", report)
        if not row["complete"]:
            raise RuntimeError(
                f"A backend did not complete at seed={meta['seed']} z={meta['redshift']}"
            )
    report["complete"] = all(row["complete"] for row in report["snapshots"])
    report["three_methods_completed"] = all(
        row["three_methods_completed"] for row in report["snapshots"]
    )
    _write_json(output / "comparison_report.json", report)
    _plot(output, report["snapshots"])
    if report["complete"]:
        (output / "COMPLETED").write_text(
            "All requested backend calculations completed; convergence is reported separately.\n"
        )
    return report


def main() -> None:
    """Run a JSON-configured comparison from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-missing-c2ray", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(
        json.loads(args.config.read_text()),
        args.output,
        require_c2ray=not args.allow_missing_c2ray,
    )


if __name__ == "__main__":
    main()
