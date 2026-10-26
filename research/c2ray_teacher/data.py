"""Convert restricted native source histories to physical C2-Ray H+He inputs.

The native ultraviolet normalization uses ``N_b0 = No + He_No``: the number
of hydrogen and helium nuclei, not the baryonic mass divided by proton mass.
The input rates already include escape fraction, photon yield, and Eulerian
deposition. Neither density nor escape fraction is applied a second time.

Only the existing prescribed-source, frozen-overdensity histories are accepted.
Their physical gas density dilutes as ``(1 + z)**3`` while the proper cell
volume grows, preserving the number of atoms. This is not an evolving-density
or hydrodynamically coupled calculation.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import struct
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from research.pc_evolution.native_history import (
    array_hash,
    load_history,
    native_time_interval,
    validate_history,
)

ROOT = Path(__file__).resolve().parents[2]
MAGIC = b"C2RTHHE1"


def _integer(value: int, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _constants() -> dict[str, float | str]:
    source = ROOT / "src/py21cmfast/src/Constants.c"
    contents = source.read_text()
    result: dict[str, float | str] = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    for name in ("G", "m_p", "cm_per_Mpc"):
        match = re.search(rf"\.{name}\s*=\s*([0-9.eE+-]+)\s*,", contents)
        if match is None or not np.isfinite(float(match[1])) or float(match[1]) <= 0:
            raise ValueError(f"Cannot read native physical constant {name}")
        result[name] = float(match[1])
    return result


def _validate_physics(metadata: dict[str, Any], dimension: int) -> dict[str, float]:
    expected = {
        "rate_units": "native effective escaped photons per mean-baryon cell per second",
        "source_model": "L-INTEGRAL",
        "source_components": ["ACG"],
        "density_weighting_applied_to_n_ion": False,
        "local_cumulative_n_ion_differenced": False,
        "native_n_ion_is_a_rate": False,
        "dimension": dimension,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Unsupported native history metadata: {key}")
    inputs = metadata.get("inputs", {})
    flags = inputs.get("astro_options", {})
    expected_flags = {
        "USE_MCGS": False,
        "USE_TS_FLUCT": False,
        "USE_REIONIZATION_PHOTOHEATING_FEEDBACK": False,
        "RECOMB_MODEL": "none",
        "PHOTON_CONS_TYPE": "no-photoncons",
        "HALO_SCALING_RELATIONS_MEDIAN": False,
    }
    for key, value in expected_flags.items():
        if flags.get(key) != value:
            raise ValueError(f"Unsupported native source physics: {key}")
    if inputs.get("matter_options", {}).get("SOURCE_MODEL") != "L-INTEGRAL":
        raise ValueError("Unsupported native SOURCE_MODEL")
    try:
        cosmo = {
            key: float(inputs["cosmo_params"][key])
            for key in ("hlittle", "OMm", "OMb", "Y_He", "OMk", "wl", "OMr")
        }
        box_len = float(metadata["box_len_cMpc"])
        t_star = float(metadata["t_STAR"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Missing physical cosmology, box length, or source clock"
        ) from exc
    # Native CosmoParams.OMl is a property, omitted by asdict when derived
    # C-struct parameters are not requested. Match that property exactly.
    cosmo["OMl"] = 1 - cosmo["OMm"]
    if (
        "OMl" in inputs["cosmo_params"]
        and inputs["cosmo_params"]["OMl"] != cosmo["OMl"]
    ):
        raise ValueError("Serialized OMl differs from native derived 1 - OMm")
    if not np.isfinite([*cosmo.values(), box_len, t_star]).all():
        raise ValueError("Physical cosmology and source clock must be finite")
    if not (
        cosmo["hlittle"] > 0
        and 0 < cosmo["OMb"] <= cosmo["OMm"]
        and cosmo["OMl"] >= 0
        and cosmo["OMr"] >= 0
        and 0 < cosmo["Y_He"] < 1
        and cosmo["OMk"] == 0
        and cosmo["wl"] == -1
        and box_len > 0
        and t_star > 0
    ):
        raise ValueError("Unsupported physical cosmology or nonpositive source clock")
    clock = metadata.get("cosmology_for_clocks", {})
    if any(
        clock.get(key) != cosmo[key]
        for key in ("hlittle", "OMm", "OMl", "OMr", "OMk", "wl")
    ):
        raise ValueError("Native clock cosmology differs from physical input cosmology")
    return cosmo


def _age(redshift: float, cosmo: dict[str, float]) -> float:
    h0 = cosmo["hlittle"] * 3.2407e-18
    om, ol = cosmo["OMm"], cosmo["OMl"]
    if ol == 0:
        return float(2 / (3 * h0) / (1 + redshift) ** 1.5)
    prefactor = 2 * np.sqrt(om + ol) / (3 * h0 * np.sqrt(ol))
    return float(prefactor * np.arcsinh(np.sqrt(ol / om) / (1 + redshift) ** 1.5))


def _redshift(age: float, cosmo: dict[str, float]) -> float:
    h0 = cosmo["hlittle"] * 3.2407e-18
    om, ol = cosmo["OMm"], cosmo["OMl"]
    if ol == 0:
        return float((2 / (3 * h0 * age)) ** (2 / 3) - 1)
    prefactor = 2 * np.sqrt(om + ol) / (3 * h0 * np.sqrt(ol))
    return float((np.sqrt(ol / om) / np.sinh(age / prefactor)) ** (2 / 3) - 1)


def _step_edges(times: np.ndarray, stride: int, substeps: int) -> np.ndarray:
    nodes = [*range(0, len(times) - 1, stride), len(times) - 1]
    pieces = [
        np.linspace(times[left], times[right], substeps + 1)[:-1]
        for left, right in itertools.pairwise(nodes)
    ]
    edges = np.r_[np.concatenate(pieces), times[-1]]
    if not np.all(np.diff(edges) > 0):
        raise ValueError("Transport substeps are too small to represent distinctly")
    return edges


def _integrated_rates(
    rates: np.ndarray, times: np.ndarray, start: float, end: float
) -> np.ndarray:
    interior = times[(times > start) & (times < end)]
    nodes = np.r_[start, interior, end]
    result = np.zeros(rates.shape[1], dtype=np.float64)
    for left, right in itertools.pairwise(nodes):
        index = min(int(np.searchsorted(times, left, side="right") - 1), len(times) - 2)
        duration = times[index + 1] - times[index]
        weight = ((left + right) / 2 - times[index]) / duration
        result += (right - left) * (
            (1 - weight) * rates[index] + weight * rates[index + 1]
        )
    return result


def _validate_initial_state(
    temperature: float, xhii: float, xheii: float, xheiii: float
) -> None:
    if not np.isfinite([temperature, xhii, xheii, xheiii]).all() or temperature <= 0:
        raise ValueError(
            "Initial temperature and ion fractions must be finite, with positive temperature"
        )
    if not (0 < xhii < 1 and xheii > 0 and xheiii > 0 and xheii + xheiii < 1):
        raise ValueError(
            "The native driver requires strictly positive H I, H II, He I, He II "
            "and He III initial fractions"
        )


def _write_binary(
    path: Path,
    *,
    header: tuple[int, ...],
    initial: np.ndarray,
    positions: np.ndarray,
    density: np.ndarray,
    steps: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(MAGIC)
            stream.write(struct.pack("<4i", *header))
            stream.write(np.asarray(initial, dtype="<f8").tobytes())
            stream.write(np.asarray(positions, dtype="<i4").tobytes(order="C"))
            stream.write(
                np.asarray(density * (1 + initial[0]) ** 3, dtype="<f8").tobytes(
                    order="F"
                )
            )
            for values, luminosity in steps:
                stream.write(np.asarray(values, dtype="<f8").tobytes())
                stream.write(
                    np.asarray(density * (1 + values[1]) ** 3, dtype="<f8").tobytes(
                        order="F"
                    )
                )
                stream.write(np.asarray(luminosity, dtype="<f8").tobytes())
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_history(
    history: dict[str, Any] | str | Path,
    output: str | Path,
    *,
    refinement: int = 1,
    stride: int = 1,
    time_substeps: int = 1,
    initial_temperature: float = 100.0,
    initial_xhii: float = 1e-6,
    initial_xheii: float = 1e-6,
    initial_xheiii: float = 1e-12,
    thermal: bool = True,
) -> dict[str, Any]:
    """Write a physical prescribed-source history for the native C2-Ray driver.

    Parameters
    ----------
    history : dict or path
        Validated native-history mapping, or its non-pickle NPZ path.
    output : path
        New binary output path. Existing files are never replaced.
    refinement : int
        Odd integer refinement of the absorption mesh. Source centres stay fixed.
    stride : int
        Native intervals combined per transport step. All rate nodes contribute.
    time_substeps : int
        Equal-duration subdivisions of each grouped transport interval.
    initial_temperature : float
        Initial gas temperature in kelvin.
    initial_xhii, initial_xheii, initial_xheiii : float
        Initial species fractions within their respective elements. Every neutral
        and ionized species must initially be present: ``0 < xHII < 1``,
        ``xHeII > 0``, ``xHeIII > 0`` and ``xHeII + xHeIII < 1``.
    thermal : bool
        Must be True. The unchanged upstream isothermal initialization accesses
        unallocated heating tables and is unsupported by this integration.

    Returns
    -------
    dict
        JSON-serializable physical configuration, provenance and per-step budgets.
        The luminosities are actual escaped photons per second; the native driver
        alone converts them to its internal ``S_star`` normalization.
    """
    if (
        isinstance(refinement, (bool, np.bool_))
        or not isinstance(refinement, (int, np.integer))
        or refinement < 1
        or refinement % 2 != 1
    ):
        raise ValueError(
            "refinement must be an odd positive integer to preserve source centres"
        )
    refinement = int(refinement)
    stride = _integer(stride, "stride")
    time_substeps = _integer(time_substeps, "time_substeps")
    _validate_initial_state(
        initial_temperature, initial_xhii, initial_xheii, initial_xheiii
    )
    if not isinstance(thermal, (bool, np.bool_)):
        raise TypeError("thermal must be boolean")
    if not thermal:
        raise ValueError(
            "thermal=True is required: unchanged upstream isothermal initialization "
            "accesses unallocated heating tables"
        )
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if isinstance(history, (str, Path)):
        history = load_history(history)
    metadata = validate_history(history)
    coarse = len(history["capacity"])
    dimension = coarse * refinement
    if dimension < 16 or dimension > np.iinfo(np.int32).max:
        raise ValueError("C2-Ray transport dimension must be at least 16 and fit int32")
    cosmo = _validate_physics(metadata, coarse)
    constants = _constants()
    times = np.asarray(history["times_seconds"])
    redshifts = np.asarray(history["redshifts"])
    for index, (old, new) in enumerate(itertools.pairwise(redshifts)):
        expected, _ = native_time_interval(old, new, cosmo)
        if not np.isclose(
            times[index + 1] - times[index], expected, rtol=2e-10, atol=2.0
        ):
            raise ValueError(
                "History times differ from its declared native cosmological clock"
            )
    h0 = cosmo["hlittle"] * 3.2407e-18
    rho_b = 3 * h0**2 / (8 * np.pi * constants["G"]) * cosmo["OMb"]
    n_h = rho_b * (1 - cosmo["Y_He"]) / constants["m_p"]
    n_he = rho_b * cosmo["Y_He"] / (4 * constants["m_p"])
    n_nuclei = n_h + n_he
    box_cm = metadata["box_len_cMpc"] * constants["cm_per_Mpc"]
    mean_cell_nuclei = n_nuclei * (box_cm / coarse) ** 3
    density = np.array(history["capacity"], dtype=np.float64, copy=True) * n_nuclei
    for axis in range(3):
        density = np.repeat(density, refinement, axis=axis)
    active = np.any(history["rates"] > 0, axis=0)
    dummy = not active.any()
    if dummy:
        active[(coarse // 2,) * 3] = True
    positions = (np.argwhere(active) + 1) * refinement - (refinement - 1) // 2
    rates = np.asarray(history["rates"][:, active], dtype=np.float64)
    edges = _step_edges(times, stride, time_substeps)
    age_initial = _age(float(redshifts[0]), cosmo)

    def at_time(elapsed: float) -> float:
        index = int(np.searchsorted(times, elapsed))
        if index < len(times) and times[index] == elapsed:
            return float(redshifts[index])
        return _redshift(age_initial + elapsed, cosmo)

    steps: list[tuple[np.ndarray, np.ndarray]] = []
    audits = []
    for start, end in itertools.pairwise(edges):
        dt = float(end - start)
        z_mid = _redshift(age_initial + (start + end) / 2, cosmo)
        integrated = (
            _integrated_rates(rates, times, float(start), float(end)) * mean_cell_nuclei
        )
        luminosities = integrated / dt
        if not np.isfinite(luminosities).all() or np.any(luminosities < 0):
            raise ValueError(
                "Physical source conversion overflowed or produced negative rates"
            )
        proper_box = box_cm / (1 + z_mid)
        steps.append((np.array([dt, z_mid, proper_box]), luminosities))
        audits.append(
            {
                "time_start_seconds": float(start),
                "time_end_seconds": float(end),
                "dt_seconds": dt,
                "redshift_start": at_time(float(start)),
                "redshift_end": at_time(float(end)),
                "redshift_density_evaluation": z_mid,
                "proper_box_cm": proper_box,
                "emitted_photons": float(integrated.sum(dtype=np.longdouble)),
                "source_rates_sha256": array_hash(luminosities),
            }
        )
    initial = np.array(
        [
            redshifts[0],
            box_cm / (1 + redshifts[0]),
            initial_temperature,
            initial_xhii,
            initial_xheii,
            initial_xheiii,
        ]
    )
    _write_binary(
        output,
        header=(dimension, len(steps), len(positions), int(thermal)),
        initial=initial,
        positions=positions,
        density=density,
        steps=steps,
    )
    total_nuclei = float(
        np.sum(history["capacity"], dtype=np.longdouble) * mean_cell_nuclei
    )
    with output.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "schema_version": 1,
        "binary_format": MAGIC.decode(),
        "input_sha256": digest,
        "input_bytes": output.stat().st_size,
        "native_array_sha256": dict(metadata["array_sha256"]),
        "native_metadata_sha256": hashlib.sha256(
            json.dumps(metadata, sort_keys=True).encode()
        ).hexdigest(),
        "dimension": dimension,
        "native_dimension": coarse,
        "refinement": refinement,
        "stride": stride,
        "time_substeps": time_substeps,
        "step_count": len(steps),
        "source_count": len(positions),
        "dummy_zero_luminosity_source": dummy,
        "source_positions_sha256": array_hash(positions.astype(np.int32)),
        "source_positions": "fixed original coarse centres; 1-based integer xyz indices",
        "box_len_cMpc": float(metadata["box_len_cMpc"]),
        "density_evolution": "frozen_native_overdensity_with_cosmological_dilution",
        "density_reference_redshift": float(metadata["reference_redshift"]),
        "density_units": "total H plus He nuclei per proper cm^3",
        "source_units": "escaped ionizing photons per second",
        "photon_normalization": "native_N_b0_H_plus_He_nuclei",
        "source_integration": "exact trapezoidal integral of piecewise-linear-in-time native endpoint rates",
        "density_time_sampling": "native cosmic-time midpoint of each transport step",
        "cosmology": cosmo,
        "h": cosmo["hlittle"],
        "omega_m": cosmo["OMm"],
        "omega_b": cosmo["OMb"],
        "omega_lambda": cosmo["OMl"],
        "Y_He": cosmo["Y_He"],
        "helium_number_fraction": n_he / n_nuclei,
        "physical_constants": constants,
        "mean_hydrogen_density_z0_cm3": n_h,
        "mean_helium_density_z0_cm3": n_he,
        "mean_total_nuclei_density_z0_cm3": n_nuclei,
        "mean_native_cell_nuclei": mean_cell_nuclei,
        "initial_total_nuclei": total_nuclei,
        "initial_hydrogen_atoms": total_nuclei * n_h / n_nuclei,
        "initial_helium_atoms": total_nuclei * n_he / n_nuclei,
        "initial_state": {
            "temperature_kelvin": float(initial_temperature),
            "xHII": float(initial_xhii),
            "xHeII": float(initial_xheii),
            "xHeIII": float(initial_xheiii),
        },
        "thermal_evolution": bool(thermal),
        "total_emitted_photons": float(sum(row["emitted_photons"] for row in audits)),
        "steps": audits,
    }
