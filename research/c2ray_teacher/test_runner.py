"""Verify launch resources and validation independently of native RT binaries."""

import json
from pathlib import Path

import numpy as np
import pytest

from research.c2ray_teacher.runner import (
    _launch,
    _native_initialization,
    compare_parallel_outputs,
    parallel_layout,
    validate_config,
    validate_output,
)


@pytest.mark.parametrize(
    ("workers", "expected"), [(1, (1, 1)), (8, (1, 8)), (12, (2, 6)), (32, (4, 8))]
)
def test_layout_uses_all_allocated_cpus(workers: int, expected: tuple) -> None:
    """Use at most eight independent source octants per rank."""
    assert parallel_layout(workers) == expected


@pytest.mark.parametrize("workers", [0, -1, True, 1.5, 33])
def test_invalid_allocation_fails(workers: object) -> None:
    """Reject unsupported CPU requests before launching processes."""
    with pytest.raises(ValueError, match="workers"):
        parallel_layout(workers)


def test_config_rejects_small_grid_and_source_shifting_refinement() -> None:
    """Respect upstream iteration limits and exact source positions."""
    for updates in ({"dimension": 8}, {"refinement": 2}, {"z_start": 4}):
        with pytest.raises(ValueError, match=r"dimension|refinement|Redshifts"):
            validate_config(updates)
    config = validate_config({"dimension": 16, "refinement": 3})
    assert config["refinement"] == 3
    assert config["thermal"] is True


def output_fixture() -> tuple[dict, dict]:
    """Return a synthetic valid field with independently known atomic counts."""
    shape = (1, 16, 16, 16)
    fields = {
        "density_total_cm3": np.ones(shape),
        "xHII": np.full(shape, 0.2),
        "xHeII": np.full(shape, 0.1),
        "xHeIII": np.full(shape, 0.05),
        "temperature_K": np.full(shape, 1e4),
        "xHI_time_average": np.full(shape, 0.9),
        "xHeI_time_average": np.full(shape, 0.9),
        "xHeII_time_average": np.full(shape, 0.08),
        "xHeIII_time_average": np.full(shape, 0.02),
        "Gamma_HI_s": np.full(shape, 0.01),
        "Gamma_HeI_s": np.full(shape, 0.002),
        "Gamma_HeII_s": np.full(shape, 0.001),
        "photoheating_native": np.zeros(shape),
    }
    meta = {
        "time_seconds": 1.0,
        "dt_seconds": 1.0,
        "redshift_mid": 8.0,
        "proper_box_cm": 16.0,
        "emitted_photons": 100.0,
    }
    output = {"dimension": 16, "steps": [meta], "fields": fields}
    manifest = {
        "dimension": 16,
        "step_count": 1,
        "helium_number_fraction": 0.08,
        "initial_total_nuclei": 4096.0,
        "steps": [
            {
                "dt_seconds": 1.0,
                "redshift_end": 7.9,
                "redshift_density_evaluation": 8.0,
                "proper_box_cm": 16.0,
                "emitted_photons": 100.0,
            }
        ],
    }
    return output, manifest


def test_diagnostics_measure_atoms_and_do_not_certify_photon_closure() -> None:
    """A bounded ionization field alone is not a conservation certificate."""
    output, manifest = output_fixture()
    result = validate_output(output, manifest)
    assert result["state_checks_passed"]
    assert result["photon_conservation_certified"] is False
    assert result["history"][0]["xHI_mass"] == pytest.approx(0.8)
    assert result["history"][0]["total_nuclei"] == pytest.approx(4096.0)
    assert result["history"][0]["photoabsorptions_time_average_estimate"] > 0


@pytest.mark.parametrize(
    ("field", "value"),
    [("xHII", 1.01), ("temperature_K", -1.0), ("Gamma_HI_s", np.nan)],
)
def test_invalid_native_states_fail(field: str, value: float) -> None:
    """Do not clip nonphysical or nonfinite states from the native solver."""
    output, manifest = output_fixture()
    output["fields"][field][0, 1, 2, 3] = value
    with pytest.raises(ValueError, match="Native"):
        validate_output(output, manifest)


def test_parallel_comparison_detects_changed_morphology() -> None:
    """Equal mean ionization does not suffice for MPI equivalence."""
    reference, _ = output_fixture()
    candidate, _ = output_fixture()
    candidate["fields"]["xHII"][0, 0, 0, 0] += 0.1
    candidate["fields"]["xHII"][0, 0, 0, 1] -= 0.1
    result = compare_parallel_outputs(reference, candidate)
    assert result["passed"] is False
    assert result["fields"]["xHII"]["maximum_absolute_difference"] == pytest.approx(0.1)


@pytest.mark.parametrize(("difference", "passed"), [(5e-10, True), (2e-9, False)])
def test_parallel_tolerance_allows_only_tiny_absolute_species_roundoff(
    difference: float, passed: bool
) -> None:
    """Resolve near-zero helium fractions without weakening morphology checks."""
    reference, _ = output_fixture()
    candidate, _ = output_fixture()
    field = "xHeIII_time_average"
    reference["fields"][field].fill(1e-12)
    candidate["fields"][field].fill(1e-12)
    candidate["fields"][field][0, 0, 0, 0] += difference
    result = compare_parallel_outputs(reference, candidate)
    assert result["passed"] is passed
    assert result["fields"][field]["absolute_tolerance"] == 1e-9


def test_rate_comparison_has_no_absolute_fraction_tolerance() -> None:
    """Dimensionful photoionization rates retain their relative-scale test."""
    reference, _ = output_fixture()
    candidate, _ = output_fixture()
    field = "Gamma_HeII_s"
    reference["fields"][field].fill(1e-20)
    candidate["fields"][field].fill(2e-20)
    result = compare_parallel_outputs(reference, candidate)
    assert result["passed"] is False
    assert result["fields"][field]["absolute_tolerance"] == 0.0


@pytest.mark.parametrize("name", ["time_seconds", "proper_box_cm", "emitted_photons"])
def test_output_checks_native_clock_geometry_and_budget(name: str) -> None:
    """Catch binary layout, units and source integration changes independently."""
    output, manifest = output_fixture()
    output["steps"][0][name] *= 2
    with pytest.raises(ValueError, match="metadata differs from input"):
        validate_output(output, manifest)


def test_density_dilution_must_preserve_total_nuclei() -> None:
    """Proper volume and physical density must conserve the comoving atom count."""
    output, manifest = output_fixture()
    output["fields"]["density_total_cm3"] *= 1.01
    with pytest.raises(ValueError, match="total nuclei changed"):
        validate_output(output, manifest)


def test_helium_species_share_a_single_population() -> None:
    """Individually bounded helium fractions can still sum to an invalid state."""
    output, manifest = output_fixture()
    output["fields"]["xHeII"].fill(0.8)
    output["fields"]["xHeIII"].fill(0.3)
    with pytest.raises(ValueError, match="helium ionization fractions"):
        validate_output(output, manifest)


@pytest.mark.parametrize(
    "config",
    [
        {"thermal": False},
        {"box_len": np.nan},
        {"intervals": 1.5},
        {"unknown": 2},
        {"initial_xhii": 0},
        {"initial_xhii": 1},
        {"initial_xheii": 0},
        {"initial_xheii": 0.5, "initial_xheiii": 0.5},
    ],
)
def test_configuration_rejects_unvalidated_modes(config: dict) -> None:
    """Reject unsupported solver modes and malformed inputs before allocation."""
    with pytest.raises(
        ValueError, match=r"thermal|box_len|intervals|Unknown|strictly positive"
    ):
        validate_config(config)


def test_zero_exit_native_nonconvergence_is_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject the upstream warning even if its process reports a zero exit status."""
    build = tmp_path / "build"
    (build / "runtime" / "results").mkdir(parents=True)
    executable = build / "c2ray_teacher"
    executable.touch()
    monkeypatch.setattr("research.c2ray_teacher.runner.shutil.which", lambda name: name)

    def fake_run(command: list[str], **kwargs: object) -> None:
        cwd = Path(kwargs["cwd"])
        assert cwd != build / "runtime"
        assert list((cwd / "results").iterdir()) == []
        assert command[command.index("-np") + 1] == "4"
        assert kwargs["env"]["OMP_NUM_THREADS"] == "8"
        (cwd / "results" / "logfile").write_text("Multiple sources not converging\n")

    monkeypatch.setattr("research.c2ray_teacher.runner.subprocess.run", fake_run)
    with pytest.raises(RuntimeError, match="Native solver reported failure"):
        _launch(executable, tmp_path / "input.bin", tmp_path / "science", 32)
    assert not (tmp_path / "COMPLETED").exists()
    assert list((build / "runtime" / "results").iterdir()) == []


def test_native_initialization_retains_measured_spectral_normalization(
    tmp_path: Path,
) -> None:
    """Record the measured source-normalization mismatch without modifying tables."""
    path = tmp_path / "native.log"
    path.write_text(
        'Loading native tables\n{"stage":"native_initialized",'
        '"sed_table_photons_over_nominal":1.0153572221882665}\n'
    )
    assert (
        _native_initialization(path)["sed_table_photons_over_nominal"]
        == 1.0153572221882665
    )


@pytest.mark.parametrize("ratios", [[], [1.0, 1.0], [np.nan], [-1.0], [True]])
def test_native_initialization_rejects_missing_duplicate_or_invalid_records(
    tmp_path: Path, ratios: list
) -> None:
    """Do not manufacture a measured normalization when the native log lacks it."""
    path = tmp_path / "native.log"
    path.write_text(
        "\n".join(
            json.dumps(
                {"stage": "native_initialized", "sed_table_photons_over_nominal": ratio}
            )
            for ratio in ratios
        )
    )
    with pytest.raises(RuntimeError, match=r"exactly one|positive finite"):
        _native_initialization(path)
