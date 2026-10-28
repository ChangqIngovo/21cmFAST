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
