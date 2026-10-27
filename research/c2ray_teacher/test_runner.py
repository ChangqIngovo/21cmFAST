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
    read_transport_diagnostics,
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


def test_physical_controls_are_validated_before_native_work():
    """Do not launch or compile an experiment with unphysical control values."""
    config = validate_config({"blackbody_temperature_K": 75000., "clumping_factor": 4.})
    assert config["blackbody_temperature_K"] == 75000.
    assert config["clumping_factor"] == 4.
    for updates in ({"blackbody_temperature_K": 0}, {"clumping_factor": 0.1},
                    {"clumping_factor": float("nan")}):
        with pytest.raises(ValueError, match=r"blackbody|clumping"):
            validate_config(updates)


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
    row = result["history"][0]
    assert row["radiation_ionizations_time_average_estimate"] > 0
    assert "photoabsorptions_time_average_estimate" not in row
    assert result["primary_photon_absorption_count_available"] is False


def test_secondary_inclusive_event_estimate_can_exceed_emitted_photons() -> None:
    """Effective radiation rates count events, not individual absorbed photons."""
    output, manifest = output_fixture()
    output["fields"]["Gamma_HI_s"].fill(0.03)
    result = validate_output(output, manifest)
    estimate = result["history"][0]["radiation_ionizations_time_average_estimate"]
    expected = 4096 * (0.92 * 0.9 * 0.03 + 0.08 * 0.9 * 0.002 + 0.08 * 0.08 * 0.001)
    assert estimate == pytest.approx(expected, rel=1e-12)
    assert estimate > manifest["steps"][0]["emitted_photons"]
    assert result["state_checks_passed"] is True
    assert result["photon_conservation_certified"] is False
    definition = result["diagnostic_definitions"][
        "radiation_ionizations_time_average_estimate"
    ]
    assert definition["includes_secondary_ionizations"] is True
    assert definition["is_primary_photon_absorption_count"] is False
    assert definition["rates_and_fractions_are_iteration_staggered"] is True


def transport_fixture() -> dict:
    """Return separately counted photons and secondary ionization events."""
    return {
        "schema_version": 1,
        "step": 1,
        "time_seconds": 1.0,
        "dt_seconds": 1.0,
        "final_transport_iteration": 2,
        "nominal_emitted_photons": 100.0,
        "table_emitted_photons": 101.0,
        "primary_absorptions_by_species": [80.0, 10.0, 1.0],
        "secondary_ionizations_by_species": [5.0, 2.0, 0.0],
        "primary_absorptions": 91.0,
        "secondary_ionizations": 7.0,
        "boundary_outflow_photons": 10.0,
        "native_boundary_outflow_photons": 9.0,
        "lls_loss_photons": 0.0,
        "transport_rate_product_ionizations": 98.0,
        "postchemistry_rate_product_ionizations": 96.0,
        "transport_fraction_max_abs_change": 0.002,
        "transport_fraction_max_relative_change": 0.01,
        "recorded_cell_visits": 100,
    }


def test_transport_ledger_keeps_secondary_events_out_of_photon_closure(
    tmp_path: Path,
) -> None:
    """Only primary absorption and outward photons balance source photons."""
    output, _ = output_fixture()
    path = tmp_path / "transport.jsonl"
    path.write_text(json.dumps(transport_fixture()) + "\n")
    result = read_transport_diagnostics(path, output, 1.01)
    row = result["steps"][0]
    assert row["photon_residual_over_table_emitted"] == pytest.approx(0)
    assert row["photon_residual_over_nominal_emitted"] == pytest.approx(0.01)
    assert row["secondary_ionizations"] == 7
    assert row["postchemistry_rate_product_difference"] == -2
    assert row["boundary_minus_native_outflow_photons"] == 1
    assert result["cumulative"]["secondary_ionizations"] == 7
    assert result["photon_conservation_certified"] is False
    assert row["transported_emitted_photons"] is None
    assert row["photon_residual_over_transported_emitted"] is None
    assert result["cumulative"]["transported_emitted_photons"] is None


def test_transport_ledger_separates_omitted_spectral_tail(tmp_path: Path) -> None:
    """All active-band photons can escape while omitted table photons remain explicit."""
    output, _ = output_fixture()
    record = transport_fixture()
    for kind in ("primary_absorptions", "secondary_ionizations"):
        record[kind] = 0.0
        record[f"{kind}_by_species"] = [0.0, 0.0, 0.0]
    record.update(
        boundary_outflow_photons=100.0,
        transport_rate_product_ionizations=0.0,
        postchemistry_rate_product_ionizations=0.0,
    )
    path = tmp_path / "transport.jsonl"
    path.write_text(json.dumps(record) + "\n")
    result = read_transport_diagnostics(path, output, 1.01, active_spectral_ratio=1.0)
    for ledger in (result["steps"][0], result["cumulative"]):
        assert ledger["transported_emitted_photons"] == 100.0
        assert ledger["photon_residual_over_transported_emitted"] == 0.0
        assert ledger["photon_residual_over_table_emitted"] == pytest.approx(-1 / 101)
    assert result["photon_conservation_certified"] is False


@pytest.mark.parametrize("ratio", [1.011, 0.0, -1.0, np.nan, True])
def test_transport_ledger_rejects_invalid_active_band_ratio(
    tmp_path: Path, ratio: float
) -> None:
    """The transported band cannot contain more photons than the full table."""
    output, _ = output_fixture()
    with pytest.raises(ValueError, match="active spectral"):
        read_transport_diagnostics(
            tmp_path / "unused.jsonl", output, 1.01, active_spectral_ratio=ratio
        )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("step", 2),
        ("dt_seconds", 2.0),
        ("nominal_emitted_photons", 200.0),
        ("table_emitted_photons", 100.0),
        ("primary_absorptions", 90.0),
        ("transport_rate_product_ionizations", 99.0),
        ("boundary_outflow_photons", -1.0),
        ("secondary_ionizations", np.nan),
    ],
)
def test_transport_diagnostics_reject_misaligned_or_corrupt_records(
    tmp_path: Path, key: str, value: object
) -> None:
    """Do not accept stale iterations, wrong units or incomplete counters."""
    output, _ = output_fixture()
    record = transport_fixture()
    record[key] = value
    path = tmp_path / "transport.jsonl"
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="Transport"):
        read_transport_diagnostics(path, output, 1.01)


def test_transport_ledger_reports_residual_without_rescaling(tmp_path: Path) -> None:
    """A measured photon deficit remains visible instead of being projected away."""
    output, _ = output_fixture()
    record = transport_fixture()
    record["boundary_outflow_photons"] = 5.0
    path = tmp_path / "transport.jsonl"
    path.write_text(json.dumps(record) + "\n")
    result = read_transport_diagnostics(path, output, 1.01)
    assert result["steps"][0]["boundary_outflow_photons"] == 5
    assert result["steps"][0]["photon_residual_over_table_emitted"] == pytest.approx(
        -5 / 101
    )


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


def test_parallel_comparison_checks_independent_counters_and_ignores_raw_native() -> (
    None
):
    """Thread-dependent legacy counters cannot replace independent photon counts."""
    reference, _ = output_fixture()
    candidate, _ = output_fixture()
    reference["transport_diagnostics"] = {"steps": [transport_fixture()]}
    candidate["transport_diagnostics"] = {"steps": [transport_fixture()]}
    candidate["transport_diagnostics"]["steps"][0][
        "native_boundary_outflow_photons"
    ] *= 10
    assert compare_parallel_outputs(reference, candidate)["passed"] is True
    candidate["transport_diagnostics"]["steps"][0]["boundary_outflow_photons"] += 1e-3
    result = compare_parallel_outputs(reference, candidate)
    assert result["passed"] is False
    assert result["transport_counters"]["available"] is True
    assert (
        result["transport_counters"]["fields"]["boundary_outflow_photons"]["passed"]
        is False
    )


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
    assert _native_initialization(path)["sed_transport_photons_over_nominal"] is None
    assert _native_initialization(path)["sed_omitted_tail_photons_over_nominal"] is None


@pytest.mark.parametrize(
    ("active", "tail", "valid"),
    [(1.0, 0.01, True), (1.02, -0.01, False), (1.0, 0.02, False), (1.0, None, False)],
)
def test_native_initialization_checks_transported_and_omitted_bands(
    tmp_path: Path, active: float, tail: float | None, valid: bool
) -> None:
    """Preserve a measured tail and reject contradictory band normalizations."""
    path = tmp_path / "native.log"
    record = {
        "stage": "native_initialized",
        "sed_table_photons_over_nominal": 1.01,
        "sed_transport_photons_over_nominal": active,
        "sed_omitted_tail_photons_over_nominal": tail,
    }
    path.write_text(json.dumps(record) + "\n")
    if valid:
        assert _native_initialization(path) == record
    else:
        with pytest.raises(RuntimeError, match="spectral band"):
            _native_initialization(path)


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


def _initialization_record() -> dict:
    return {
        "stage": "native_initialized",
        "sed_table_photons_over_nominal": 1.01,
        "sed_transport_photons_over_nominal": 1.0,
        "sed_omitted_tail_photons_over_nominal": 0.01,
    }


def test_native_initialization_survives_missing_forwarded_stdout(
    tmp_path: Path,
) -> None:
    """Read actual disk diagnostics without inventing a spectral normalization."""
    log, sidecar = tmp_path / "native.log", tmp_path / "native_initialization.json"
    record = _initialization_record()
    log.write_text("")
    sidecar.write_text(json.dumps(record))
    assert _native_initialization(log, sidecar) == record
    log.write_text(json.dumps(record) + "\n")
    assert _native_initialization(log, sidecar) == record


def test_native_initialization_rejects_disagreeing_copies(tmp_path: Path) -> None:
    """A diagnostic file cannot hide a different stdout record."""
    log, sidecar = tmp_path / "native.log", tmp_path / "native_initialization.json"
    record = _initialization_record()
    sidecar.write_text(json.dumps(record))
    log.write_text(json.dumps(record | {"sed_table_photons_over_nominal": 1.02}))
    with pytest.raises(RuntimeError, match="disagree"):
        _native_initialization(log, sidecar)


@pytest.mark.parametrize(
    "payload",
    [
        "{bad json",
        "[]",
        '{"stage":"other"}',
        '{"stage":"native_initialized","sed_table_photons_over_nominal":-1}',
    ],
)
def test_native_initialization_rejects_malformed_sidecar(
    tmp_path: Path, payload: str
) -> None:
    """Malformed sidecars fail even when stdout appears valid."""
    log, sidecar = tmp_path / "native.log", tmp_path / "native_initialization.json"
    log.write_text(json.dumps(_initialization_record()))
    sidecar.write_text(payload)
    with pytest.raises(RuntimeError, match=r"sidecar|disagree"):
        _native_initialization(log, sidecar)


def test_native_initialization_legacy_fallback_requires_stdout(tmp_path: Path) -> None:
    """Retain old-binary support while refusing missing evidence in both places."""
    log, missing = tmp_path / "native.log", tmp_path / "absent.json"
    record = _initialization_record()
    log.write_text(json.dumps(record))
    assert _native_initialization(log, missing) == record
    log.write_text("")
    with pytest.raises(RuntimeError, match="exactly one"):
        _native_initialization(log, missing)
