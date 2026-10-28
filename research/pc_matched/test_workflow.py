"""Test immutable inputs and restart behavior of the matched pilot."""

import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from research.pc_matched import workflow


def test_small_defaults_and_validation() -> None:
    """Keep the pilot bounded and reject ambiguous parameter selections."""
    config = workflow.validate_config(workflow.DEFAULT_CONFIG)
    assert config["dimension"] == 8
    assert config["base_intervals"] == 10
    assert config["seeds"] == [111, 222]
    assert config["directions"] == [6, 26]
    assert len(workflow.case_settings(config)) == 5
    with pytest.raises(ValueError, match="unique"):
        workflow.validate_config(dict(config, seeds=[111, 111]))
    with pytest.raises(ValueError, match="within"):
        workflow.validate_config(dict(config, reference_z=22.0))
    with pytest.raises(ValueError, match="CFL"):
        workflow.validate_config(dict(config, cfl_values=[1.1]))


def test_opacity_uses_explicit_native_effective_atom_convention() -> None:
    """Keep source units untouched and expose the density conversion."""
    metadata = {"inputs": {"cosmo_params": {"OMb": 0.049, "hlittle": 0.6751, "Y_He": 0.245}}}
    physics = workflow.opacity_parameters(metadata, 18.0)
    expected = 3 * (0.6751 * 3.2407e-18) ** 2 / (8 * np.pi * 6.67430e-8)
    expected *= 0.049 * (1 - 3 * 0.245 / 4) / 1.67262192369e-24
    assert physics["hydrogen_density_z0_cm3"] == pytest.approx(expected, rel=1e-14)
    assert physics["cross_section_cm2"] == pytest.approx(6.3e-18 * (13.6 / 18.0) ** 3)
    with pytest.raises(ValueError, match="cosmology"):
        workflow.opacity_parameters({}, 18.0)


def test_case_matrix_is_one_parameter_at_a_time() -> None:
    """Spatial, angular and CFL changes remain separately identifiable."""
    matrix = workflow.case_settings(workflow.DEFAULT_CONFIG)
    triples = {(row["directions"], row["refinement"], row["cfl"]) for row in matrix}
    assert triples == {(6, 1, 0.4), (26, 1, 0.4), (26, 2, 0.4), (26, 1, 0.2), (26, 1, 0.1)}


def test_convergence_matrix_refines_each_axis_at_joint_fine_settings() -> None:
    """Use three-level numerical axes while preserving the small pilot defaults."""
    config = workflow.configuration("convergence")
    assert config["base_intervals"] == 40
    assert workflow.native_intervals(config) == 160
    matrix = workflow.case_settings(config)
    selections = {(row["directions"], row["refinement"], row["cfl"], row["source_stride"])
                  for row in matrix}
    assert selections == {
        (24, 4, 0.4, 1), (96, 4, 0.4, 1), (384, 4, 0.4, 1),
        (384, 1, 0.4, 1), (384, 2, 0.4, 1),
        (384, 4, 0.2, 1), (384, 4, 0.1, 1),
        (384, 4, 0.1, 4), (384, 4, 0.1, 2),
    }
    estimate = workflow.resource_estimate(config)
    assert estimate["cases_per_seed"] == 9
    assert estimate["total_cases"] == 18
    assert estimate["finest_source_snapshots"] == 161
    assert estimate["max_array_gib"] < config["memory_gb"]
    for row in estimate["estimated_array_gib_by_case"]:
        assert row["snapshots"] == 161
    with pytest.raises(ValueError, match="three directions"):
        workflow.configuration("convergence", directions=[24, 96])
    with pytest.raises(ValueError, match="divide"):
        workflow.configuration("convergence", source_strides=[1, 2, 3])
    with pytest.raises(ValueError, match="nested"):
        workflow.configuration("convergence", refinements=[1, 2, 3])
    config["directions"].append(6)
    assert workflow.CONVERGENCE_CONFIG["directions"] == [24, 96, 384]
    assert workflow.configuration()["directions"] == [6, 26]


def test_source_history_uses_physical_time_knots_on_identical_background() -> None:
    """Change only emissivity sampling, retaining exact gas and all native times."""
    times = np.array([0.0, 0.3, 0.9, 1.5, 3.0, 3.1, 5.0, 7.0, 9.0])
    rates = (times**2 + 0.1)[:, None, None, None] * np.ones((1, 2, 2, 2))
    history = {"capacity": np.full((2, 2, 2), 2.0), "times_seconds": times,
               "redshifts": np.linspace(20, 4, len(times)), "rates": rates, "metadata": {}}
    selected, definition = workflow.source_history(history, 4, "parent-fixture")
    knots = [0, 4, 8]
    assert definition["source_knot_indices"] == knots
    assert definition["native_snapshot_indices"] == list(range(len(times)))
    assert definition["parent_fixture_sha256"] == "parent-fixture"
    assert selected["capacity"] is history["capacity"]
    assert selected["times_seconds"] is history["times_seconds"]
    assert selected["redshifts"] is history["redshifts"]
    np.testing.assert_array_equal(selected["rates"][knots], rates[knots])
    np.testing.assert_allclose(selected["rates"][:, 0, 0, 0],
                               np.interp(times, times[knots], rates[knots, 0, 0, 0]), rtol=1e-15)
    integrated = np.sum(0.5 * (selected["rates"][:-1] + selected["rates"][1:])
                        * np.diff(times)[:, None, None, None], axis=0)
    coarse_integral = np.sum(0.5 * (rates[knots[:-1]] + rates[knots[1:]])
                             * np.diff(times[knots])[:, None, None, None], axis=0)
    np.testing.assert_allclose(integrated, coarse_integral, rtol=1e-15)
    finest, fine_definition = workflow.source_history(history, 1, "parent-fixture")
    np.testing.assert_array_equal(finest["rates"], rates)
    assert fine_definition["source_history_sha256"] != definition["source_history_sha256"]
    assert fine_definition["arrays_sha256"]["capacity"] == definition["arrays_sha256"]["capacity"]
    assert fine_definition["arrays_sha256"]["rates"] != definition["arrays_sha256"]["rates"]
    np.testing.assert_array_equal(history["rates"], rates)
    with pytest.raises(ValueError, match="divide"):
        workflow.source_history(history, 3, "parent-fixture")


def test_repeat_field_preserves_mean_atom_convention() -> None:
    """Refining a piecewise-constant source changes volume, not local density."""
    coarse = np.arange(8, dtype=float).reshape((2, 2, 2))
    fine = workflow.repeat_field(coarse, 2)
    assert fine.shape == (4, 4, 4)
    assert fine.sum() / 8 == coarse.sum()
    np.testing.assert_array_equal(fine.reshape(2, 2, 2, 2, 2, 2).mean((1, 3, 5)), coarse)


def test_identity_rejects_changed_configuration(tmp_path: Path) -> None:
    """A resumed run cannot silently acquire different physical settings."""
    identity = {"configuration": workflow.DEFAULT_CONFIG, "source_sha256": {"a": "b"}}
    workflow.ensure_identity(tmp_path, identity)
    workflow.ensure_identity(tmp_path, identity)
    with pytest.raises(ValueError, match="identity"):
        workflow.ensure_identity(tmp_path, dict(identity, source_sha256={"a": "c"}))


def test_unrecognized_output_is_preserved(tmp_path: Path) -> None:
    """Never adopt an unrelated existing output directory."""
    (tmp_path / "user.txt").write_text("keep")
    with pytest.raises(FileExistsError, match="Unrecognized"):
        workflow.ensure_identity(tmp_path, {"configuration": {}})
    assert (tmp_path / "user.txt").read_text() == "keep"


def test_completion_receipt_rejects_changed_artifacts(tmp_path: Path) -> None:
    """Completed runs are reused only while their artifacts remain unchanged."""
    data = tmp_path / "result.json"
    data.write_text(json.dumps({"ledger": True}))
    workflow.write_completion(tmp_path, [data])
    workflow.verify_completion(tmp_path)
    data.write_text(json.dumps({"ledger": False}))
    with pytest.raises(ValueError, match="changed"):
        workflow.verify_completion(tmp_path)


def test_archiving_preserves_interrupted_matrix(tmp_path: Path) -> None:
    """A restart preserves partial work instead of mixing old/new checkpoints."""
    source = tmp_path / "evolution"
    source.mkdir()
    (source / "progress.json").write_text("partial")
    archived = workflow.archive_partial(tmp_path, source)
    assert archived.parent == tmp_path / "incomplete"
    assert (archived / "progress.json").read_text() == "partial"
    assert not source.exists()


def test_completed_case_reuses_fields_and_rejects_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Case restart checks artifacts before deciding to skip transport."""
    calls = []
    module = ModuleType("research.pc_matched.rt")

    def fake_interval(capacity: np.ndarray, ionized: np.ndarray,
                      photons: np.ndarray, left: np.ndarray, right: np.ndarray,
                      **kwargs: object) -> dict:
        calls.append(kwargs)
        photons = photons.copy()
        photons[0] += 0.5 * (left + right) * kwargs["dt_seconds"]
        absorbed = np.minimum(photons[0], capacity - ionized)
        photons[0] -= absorbed
        return {"ionized": ionized + absorbed, "photons": photons,
                "steps": 1, "ledger": {"passed": True}}

    module.evolve_interval = fake_interval
    monkeypatch.setitem(sys.modules, "research.pc_matched.rt", module)
    history = {
        "capacity": np.ones((2, 2, 2)),
        "redshifts": np.array([20.0, 12.0, 4.0]),
        "times_seconds": np.array([0.0, 1.0, 2.0]),
        "rates": np.full((3, 2, 2, 2), 0.2),
        "metadata": {"inputs": {"cosmo_params": {
            "OMb": 0.049, "hlittle": 0.6751, "Y_He": 0.245,
        }}},
    }
    setting = {"directions": 6, "refinement": 2, "cfl": 0.4}
    fixture = {"seed": 111, "sha256": "input"}
    result = workflow._run_case(tmp_path, history, fixture, setting,
                                workflow.DEFAULT_CONFIG, {"solver": "hash"})
    assert len(calls) == 2
    with np.load(result["path"], allow_pickle=False) as arrays:
        assert arrays["ionized"].shape == (3, 4, 4, 4)
        np.testing.assert_allclose(arrays["ionized"] + arrays["radiation"],
                                   arrays["source_emitted"], rtol=1e-15, atol=0)
    repeated = workflow._run_case(tmp_path, history, fixture, setting,
                                  workflow.DEFAULT_CONFIG, {"solver": "hash"})
    assert repeated == result
    assert len(calls) == 2
    with Path(result["path"]).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="fields changed"):
        workflow._run_case(tmp_path, history, fixture, setting,
                           workflow.DEFAULT_CONFIG, {"solver": "hash"})


def test_interrupted_directional_state_resumes_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restarts preserve radiation, cumulative emission and every saved snapshot."""
    module = ModuleType("research.pc_matched.rt")
    calls = []
    fail_at = [3]

    def fake_interval(capacity: np.ndarray, ionized: np.ndarray,
                      photons: np.ndarray, left: np.ndarray, right: np.ndarray,
                      **kwargs: object) -> dict:
        calls.append(kwargs)
        if len(calls) == fail_at[0]:
            raise RuntimeError("simulated interruption")
        photons = np.roll(photons.copy(), 1, axis=1)
        photons[0] += 0.5 * (left + right) * kwargs["dt_seconds"]
        absorbed = np.minimum(0.4 * photons[0], capacity - ionized)
        photons[0] -= absorbed
        return {"ionized": ionized + absorbed, "photons": photons,
                "steps": 1, "ledger": {"passed": True}}

    module.evolve_interval = fake_interval
    monkeypatch.setitem(sys.modules, "research.pc_matched.rt", module)
    times = np.array([0.0, 0.1, 0.3, 0.6, 1.0])
    rates = (times**2 + 0.1)[:, None, None, None] * np.arange(1, 9).reshape(1, 2, 2, 2)
    history = {"capacity": np.ones((2, 2, 2)), "redshifts": np.linspace(20, 4, 5),
               "times_seconds": times, "rates": rates,
               "metadata": {"inputs": {"cosmo_params": {
                   "OMb": 0.049, "hlittle": 0.6751, "Y_He": 0.245}}}}
    fixture = {"seed": 111, "sha256": "input"}
    setting = {"directions": 6, "refinement": 1, "cfl": 0.4, "source_stride": 2}
    root = tmp_path / "restart"
    with pytest.raises(RuntimeError, match="simulated"):
        workflow._run_case(root, history, fixture, setting, workflow.DEFAULT_CONFIG, {"solver": "hash"})
    folder = root / "cases" / workflow._case_id(111, setting)
    receipt = json.loads((folder / "checkpoint.json").read_text())
    assert receipt["node"] == 1
    assert receipt["identity"]["source_history"]["source_knot_indices"] == [0, 2, 4]
    with pytest.raises(ValueError, match="identity changed"):
        workflow._run_case(root, history, fixture, setting, workflow.DEFAULT_CONFIG, {"solver": "different"})
    fail_at[0] = -1
    resumed = workflow._run_case(root, history, fixture, setting, workflow.DEFAULT_CONFIG, {"solver": "hash"})
    assert len(calls) == 6  # Three original calls, then nodes 2/3/4 from checkpoint 1.
    uninterrupted = workflow._run_case(tmp_path / "fresh", history, fixture, setting,
                                       workflow.DEFAULT_CONFIG, {"solver": "hash"})
    with np.load(resumed["path"], allow_pickle=False) as actual, np.load(
        uninterrupted["path"], allow_pickle=False,
    ) as expected:
        for key in expected.files:
            np.testing.assert_array_equal(actual[key], expected[key])
        metadata = json.loads(str(actual["metadata"]))
        assert metadata["native_snapshot_indices"] == list(range(5))
        assert metadata["source_knot_indices"] == [0, 2, 4]
    assert not (folder / "checkpoint.json").exists()
    assert not list(folder.glob("checkpoint_*.npz"))


def test_checkpoint_corruption_is_not_silently_restarted(tmp_path: Path) -> None:
    """A committed but altered directional checkpoint fails closed."""
    identity = {"setting": {"directions": 6}}
    fields = np.ones((2, 2, 2, 2))
    workflow._save_checkpoint(tmp_path, identity, 1, np.zeros((6, 2, 2, 2)),
                              np.ones((2, 2, 2), dtype=np.longdouble), fields, fields, fields,
                              [{"node": 1}])
    node, arrays, _ = workflow._load_checkpoint(tmp_path, identity, 3, (2, 2, 2))
    assert node == 1
    assert arrays["emitted"].dtype == np.longdouble
    receipt = json.loads((tmp_path / "checkpoint.json").read_text())
    with (tmp_path / receipt["path"]).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="fields changed"):
        workflow._load_checkpoint(tmp_path, identity, 3, (2, 2, 2))


def test_estimate_cli_resolves_selected_profile_defaults(capsys: pytest.CaptureFixture) -> None:
    """A dry resource estimate requires neither a run directory nor native import."""
    workflow.main(["estimate", "--profile", "convergence", "--seeds", "111"])
    result = json.loads(capsys.readouterr().out)
    assert result["configuration"]["base_intervals"] == 40
    assert result["resource_estimate"]["total_cases"] == 9
