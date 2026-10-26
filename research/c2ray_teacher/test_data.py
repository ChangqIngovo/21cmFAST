"""Check physical units and binary interchange without invoking the RT solver."""

import copy
import itertools
import struct

import numpy as np
import pytest

from research.c2ray_teacher.data import prepare_history
from research.pc_evolution.native_history import (
    ARRAY_KEYS,
    array_hash,
    native_time_interval,
)


def history_fixture() -> dict:
    """Return an asymmetric, physically timed, restricted native fixture."""
    cosmo = {
        "hlittle": 0.6751,
        "OMm": 0.3121,
        "OMl": 0.6879,
        "OMr": 9e-5,
        "OMk": 0.0,
        "wl": -1.0,
    }
    redshifts = np.array([12.0, 10.0, 8.0])
    durations = [
        native_time_interval(a, b, cosmo)[0] for a, b in itertools.pairwise(redshifts)
    ]
    times = np.r_[0.0, np.cumsum(durations)]
    rates = np.zeros((3, 16, 16, 16), dtype=np.float64)
    rates[:, 2, 5, 9] = np.array([1.0, 3.0, 7.0]) * 1e-17
    rates[:, 11, 4, 1] = np.array([0.0, 2.0, 1.0]) * 1e-17
    capacity = np.ones((16, 16, 16))
    capacity[2, 5, 9] = 4.0
    metadata = {
        "schema_version": 1,
        "density_evolution": "frozen_at_reference_redshift",
        "rates_definition": "native_n_ion_times_H_over_t_STAR",
        "reference_redshift": 8.0,
        "rate_units": "native effective escaped photons per mean-baryon cell per second",
        "density_weighting_applied_to_n_ion": False,
        "local_cumulative_n_ion_differenced": False,
        "native_n_ion_is_a_rate": False,
        "source_model": "L-INTEGRAL",
        "source_components": ["ACG"],
        "t_STAR": 0.5,
        "dimension": 16,
        "box_len_cMpc": 50.0,
        "cosmology_for_clocks": cosmo,
        "inputs": {
            "cosmo_params": {
                **{key: value for key, value in cosmo.items() if key != "OMl"},
                "OMb": 0.049,
                "Y_He": 0.245,
            },
            "matter_options": {"SOURCE_MODEL": "L-INTEGRAL"},
            "astro_options": {
                "USE_MCGS": False,
                "USE_TS_FLUCT": False,
                "USE_REIONIZATION_PHOTOHEATING_FEEDBACK": False,
                "RECOMB_MODEL": "none",
                "PHOTON_CONS_TYPE": "no-photoncons",
                "HALO_SCALING_RELATIONS_MEDIAN": False,
            },
        },
    }
    result = {
        "redshifts": redshifts,
        "times_seconds": times,
        "rates": rates,
        "capacity": capacity,
        "native_xhi": np.ones(rates.shape, dtype=np.float32),
        "metadata": metadata,
    }
    metadata["array_sha256"] = {key: array_hash(result[key]) for key in ARRAY_KEYS}
    return result


def read_binary(path) -> tuple:
    """Independently decode the documented native binary contract."""
    with path.open("rb") as stream:
        assert stream.read(8) == b"C2RTHHE1"
        size, count, sources, thermal = struct.unpack("<4i", stream.read(16))
        initial = np.fromfile(stream, dtype="<f8", count=6)
        positions = np.fromfile(stream, dtype="<i4", count=3 * sources).reshape(
            sources, 3
        )
        density = np.fromfile(stream, dtype="<f8", count=size**3).reshape(
            (size,) * 3, order="F"
        )
        steps = []
        for _ in range(count):
            values = np.fromfile(stream, dtype="<f8", count=3)
            gas = np.fromfile(stream, dtype="<f8", count=size**3).reshape(
                (size,) * 3, order="F"
            )
            luminosities = np.fromfile(stream, dtype="<f8", count=sources)
            steps.append((values, gas, luminosities))
        assert stream.read() == b""
    return size, thermal, initial, positions, density, steps


def test_native_nuclei_units_and_no_duplicate_density_or_escape(tmp_path) -> None:
    """Photon luminosity follows native N_b0, independently of local density."""
    history = history_fixture()
    result = prepare_history(history, tmp_path / "input.bin")
    _, thermal, initial, positions, density, steps = read_binary(tmp_path / "input.bin")
    h0 = 0.6751 * 3.2407e-18
    rho = 3 * h0**2 / (8 * np.pi * 6.6743e-8) * 0.049
    n_h = rho * (1 - 0.245) / 1.67262192369e-24
    n_he = rho * 0.245 / (4 * 1.67262192369e-24)
    atoms_per_cell = (n_h + n_he) * (50 * 3.08567758e24 / 16) ** 3
    np.testing.assert_array_equal(positions, [[3, 6, 10], [12, 5, 2]])
    np.testing.assert_allclose(
        density, history["capacity"] * (n_h + n_he) * 13**3, rtol=2e-15
    )
    np.testing.assert_allclose(
        steps[0][2], np.array([2e-17, 1e-17]) * atoms_per_cell, rtol=2e-15
    )
    assert thermal == 1
    assert initial[2] == 100.0
    assert result["helium_number_fraction"] == pytest.approx(n_he / (n_h + n_he))
    assert result["initial_total_nuclei"] == pytest.approx(
        history["capacity"].sum() * atoms_per_cell
    )
    assert result["photon_normalization"] == "native_N_b0_H_plus_He_nuclei"
    assert result["omega_lambda"] == pytest.approx(0.6879, rel=1e-15)


def test_inconsistent_derived_dark_energy_is_rejected(tmp_path) -> None:
    """Serialized OMl, if supplied, must match the native derived property."""
    history = history_fixture()
    history["metadata"]["inputs"]["cosmo_params"]["OMl"] = 0.5
    with pytest.raises(ValueError, match="OMl"):
        prepare_history(history, tmp_path / "input.bin")


def test_expansion_preserves_atoms_and_step_time_midpoints(tmp_path) -> None:
    """Cosmic dilution and changing proper volume preserve the same gas atoms."""
    history = history_fixture()
    result = prepare_history(history, tmp_path / "input.bin")
    size, _, initial, _, density, steps = read_binary(tmp_path / "input.bin")
    atoms = density.sum() * (initial[1] / size) ** 3
    for index, (values, gas, _) in enumerate(steps):
        dt, z_mid, proper_box = values
        assert gas.sum() * (proper_box / size) ** 3 == pytest.approx(atoms, rel=4e-15)
        z_start, z_end = history["redshifts"][index : index + 2]
        assert z_end < z_mid < z_start
        left = native_time_interval(
            z_start, z_mid, history["metadata"]["cosmology_for_clocks"]
        )[0]
        assert left == pytest.approx(dt / 2, rel=2e-13)
        assert result["steps"][index]["redshift_end"] == z_end


def test_refinement_retains_exact_source_centres_and_budget(tmp_path) -> None:
    """Odd refinement preserves coordinates and expands density, not source counts."""
    history = history_fixture()
    base = prepare_history(history, tmp_path / "base.bin")
    fine = prepare_history(history, tmp_path / "fine.bin", refinement=3)
    size, _, _, positions, density, _ = read_binary(tmp_path / "fine.bin")
    assert size == 48
    np.testing.assert_array_equal(
        (positions - 0.5) / 48, (np.array([[3, 6, 10], [12, 5, 2]]) - 0.5) / 16
    )
    assert fine["source_count"] == base["source_count"]
    assert fine["total_emitted_photons"] == pytest.approx(
        base["total_emitted_photons"], rel=2e-15
    )
    assert fine["initial_total_nuclei"] == pytest.approx(
        base["initial_total_nuclei"], rel=2e-15
    )
    assert density[8, 17, 29] == pytest.approx(density[7, 16, 28])


def test_coarse_time_and_substeps_preserve_piecewise_linear_emission(tmp_path) -> None:
    """Changing transport steps preserves the integral over all native rate nodes."""
    history = history_fixture()
    base = prepare_history(history, tmp_path / "base.bin")
    grouped = prepare_history(history, tmp_path / "group.bin", stride=2)
    fine = prepare_history(history, tmp_path / "fine.bin", stride=2, time_substeps=5)
    assert grouped["step_count"] == 1
    assert fine["step_count"] == 5
    assert grouped["total_emitted_photons"] == pytest.approx(
        base["total_emitted_photons"], rel=3e-15
    )
    assert fine["total_emitted_photons"] == pytest.approx(
        base["total_emitted_photons"], rel=3e-15
    )
    assert sum(row["dt_seconds"] for row in fine["steps"]) == pytest.approx(
        history["times_seconds"][-1]
    )


@pytest.mark.parametrize("refinement", [0, 2, True, 1.5])
def test_unaligned_refinement_is_rejected(tmp_path, refinement) -> None:
    """Do not silently displace point sources when refining the absorption mesh."""
    with pytest.raises(ValueError, match="odd positive integer"):
        prepare_history(
            history_fixture(), tmp_path / "input.bin", refinement=refinement
        )
    assert not (tmp_path / "input.bin").exists()


@pytest.mark.parametrize(
    "flag",
    [
        "USE_MCGS",
        "USE_TS_FLUCT",
        "USE_REIONIZATION_PHOTOHEATING_FEEDBACK",
        "HALO_SCALING_RELATIONS_MEDIAN",
    ],
)
def test_unsupported_source_physics_is_rejected(tmp_path, flag) -> None:
    """Fail rather than silently reuse the restricted native-rate conversion."""
    history = history_fixture()
    history["metadata"]["inputs"]["astro_options"][flag] = True
    with pytest.raises(ValueError, match="Unsupported"):
        prepare_history(history, tmp_path / "input.bin")


def test_provenance_mutation_and_existing_output_are_rejected(tmp_path) -> None:
    """Input hashes and output immutability protect reproducible teacher runs."""
    history = history_fixture()
    original = copy.deepcopy(history)
    output = tmp_path / "input.bin"
    prepare_history(history, output)
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        prepare_history(history, output)
    assert output.read_bytes() == before
    for key in ARRAY_KEYS:
        np.testing.assert_array_equal(history[key], original[key])
    history["rates"][0, 2, 5, 9] *= 2
    with pytest.raises(ValueError, match="hash mismatch"):
        prepare_history(history, tmp_path / "corrupt.bin")


def test_no_photon_history_has_explicit_zero_luminosity_source(tmp_path) -> None:
    """A dark box uses one zero source without creating emitted photons."""
    history = history_fixture()
    history["rates"][:] = 0
    history["metadata"]["array_sha256"]["rates"] = array_hash(history["rates"])
    result = prepare_history(history, tmp_path / "input.bin")
    *_, steps = read_binary(tmp_path / "input.bin")
    assert result["dummy_zero_luminosity_source"] is True
    assert result["source_count"] == 1
    assert result["total_emitted_photons"] == 0
    assert all(np.count_nonzero(row[2]) == 0 for row in steps)


def test_upstream_unsupported_isothermal_initialization_is_rejected(tmp_path) -> None:
    """Reject the upstream isothermal path before writing an unusable input."""
    with pytest.raises(ValueError, match="thermal=True"):
        prepare_history(history_fixture(), tmp_path / "input.bin", thermal=False)
    assert not (tmp_path / "input.bin").exists()


@pytest.mark.parametrize(
    "initial",
    [
        {"initial_xhii": 0.0},
        {"initial_xhii": 1.0},
        {"initial_xheii": 0.0},
        {"initial_xheiii": 0.0},
        {"initial_xheii": 0.7, "initial_xheiii": 0.3},
        {"initial_xheii": 0.8, "initial_xheiii": 0.3},
    ],
)
def test_initial_species_must_all_be_present(
    tmp_path, initial: dict[str, float]
) -> None:
    """Match the unchanged native driver's strict species-state domain."""
    with pytest.raises(ValueError, match="strictly positive"):
        prepare_history(history_fixture(), tmp_path / "input.bin", **initial)
    assert not (tmp_path / "input.bin").exists()
