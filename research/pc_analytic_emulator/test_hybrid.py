"""Exercise validation-only routing and independently conserved analytic fallback."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from research.pc_analytic_emulator import hybrid
from research.pc_analytic_emulator.model import Emulator
from research.pc_analytic_emulator.predict import _model_hashes
from research.pc_analytic_emulator.reliability import accuracy_metrics, build_policy
from research.pc_analytic_emulator.test_data import make_campaign
from research.pc_analytic_emulator.test_model import fixture_model


def _fixture(tmp_path: Path, accepted: bool = True) -> tuple[Path, Path]:
    root = make_campaign(
        tmp_path / "native",
        seeds=(1,),
        redshifts=(8.0,),
        n=8,
        native_changes={"cosmo_params": {"Y_He": 0.245}},
    )
    native = root / "seeds/seed1/native/seed1/z8"
    inputs = json.loads((native / "native.json").read_text())["identity"]["inputs"]
    with np.load(native / "source.npz", allow_pickle=False) as saved:
        model = fixture_model(
            saved["source"], saved["capacity"], saved["native_es_xhi"]
        )
        predicted = model.predict(
            saved["source"], saved["capacity"], saved["native_es_xhi"], 8, 256
        )
        truth = predicted["xHI"] if accepted else np.zeros_like(saved["capacity"])
        metrics = accuracy_metrics(predicted["xHI"], truth, saved["capacity"], 256)
    common = copy.deepcopy(inputs)
    common.pop("random_seed")
    common["simulation_options"].pop("N_THREADS")
    model.metadata["teacher_physics"] = common
    settings = {
        "sigma_cm2": 2.7173135802469135e-18,
        "directions": 6,
        "exposure_steps": 2,
        "boundary": "source_periodic_cube",
        "transport_mode": "characteristic",
        "exhaustion_tolerance": 1e-7,
        "batch_tail_tolerance": 1e-4,
        "max_sweeps": 10000,
    }
    model.metadata["teacher_identity"] = {
        "configuration": {
            "sigma_cm2": settings["sigma_cm2"],
            "analytic": {
                key: value for key, value in settings.items() if key != "sigma_cm2"
            },
        },
        "implementation_sha256": hybrid._source_hashes(),
        "native_runtime": {"native_extension_sha256": "unit-test-native-hash"},
    }
    model.save(tmp_path / "model")
    hashes = {"candidate": _model_hashes(tmp_path / "model")}
    records = [
        {
            "split": "validation",
            "candidate": "candidate",
            "geometry": {"DIM": 24, "HII_DIM": 8, "BOX_LEN": 256},
            "redshift": 8,
            "seed": seed,
            "metrics": metrics,
        }
        for seed in (2, 3)
    ]
    policy = build_policy(records, hashes, [2, 3])
    deployment = tmp_path / "deployment"
    hybrid.freeze_deployment(
        {"candidate": tmp_path / "model"}, policy, common, settings, deployment
    )
    return deployment, native


def test_validated_route_preserves_model_prediction(tmp_path: Path) -> None:
    """The approved surrogate runs without reading or producing teacher labels."""
    deployment, native = _fixture(tmp_path)
    output = tmp_path / "output"
    report = hybrid.run(deployment, native, output)
    assert report["route"]["method"] == "emulator"
    assert report["photon_ledger"]["passed"]
    with np.load(native / "source.npz", allow_pickle=False) as data:
        expected = Emulator.load(deployment / "models/candidate").predict(
            data["source"], data["capacity"], data["native_es_xhi"], 8, 256
        )
    with np.load(output / "fields.npz", allow_pickle=False) as result:
        np.testing.assert_array_equal(result["xHI"], expected["xHI"])
    assert (output / "COMPLETED").is_file()


def test_failed_validation_runs_actual_analytic_solver(tmp_path: Path) -> None:
    """Fallback independently transports native photons and audits reservoirs."""
    deployment, native = _fixture(tmp_path, accepted=False)
    output = tmp_path / "output"
    report = hybrid.run(deployment, native, output)
    assert report["route"]["method"] == "teacher"
    assert report["solver_audit"]["complete"]
    assert report["photon_ledger"]["passed"]
    assert report["solver_metadata"]["nH_mean_comoving_cm3"] > 0
    with np.load(output / "fields.npz", allow_pickle=False) as result:
        assert "escaped_by_direction" in result.files
        assert "remaining_photons" in result.files
        residual = (
            result["source"].sum(dtype=np.longdouble)
            - np.sum(
                result["capacity"].astype(np.longdouble) * result["xHII"],
                dtype=np.longdouble,
            )
            - result["remaining_photons"].sum(dtype=np.longdouble)
            - result["escaped_by_direction"].sum(dtype=np.longdouble)
            - result["unemitted_source"].sum(dtype=np.longdouble)
        )
        assert abs(residual) / result["source"].sum() < 1e-10


@pytest.mark.parametrize(
    "artifact", ["policy.json", "deployment.json", "models/candidate/field_model.npz"]
)
def test_deployment_drift_rejected_before_output(tmp_path: Path, artifact: str) -> None:
    """A frozen acceptance decision cannot be applied to changed model bytes."""
    deployment, native = _fixture(tmp_path)
    with (deployment / artifact).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match=r"checksum|changed"):
        hybrid.run(deployment, native, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_native_receipt_drift_rejected(tmp_path: Path) -> None:
    """Inference requires the original source/atom arrays and native metadata."""
    deployment, native = _fixture(tmp_path)
    with (native / "source.npz").open("ab") as stream:
        stream.write(b" changed")
    with pytest.raises(ValueError, match="checksum"):
        hybrid.run(deployment, native, tmp_path / "output")


def test_fallback_preserves_physics_restrictions(tmp_path: Path) -> None:
    """Unvalidated geometry may fall back, changed astrophysics must fail."""
    deployment, _ = _fixture(tmp_path, accepted=False)
    root = make_campaign(
        tmp_path / "changed",
        seeds=(4,),
        redshifts=(8.0,),
        native_changes={
            "astro_params": {"R_BUBBLE_MAX": 49},
            "cosmo_params": {"Y_He": 0.245},
        },
    )
    native = root / "seeds/seed4/native/seed4/z8"
    with pytest.raises(ValueError, match="physical"):
        hybrid.run(deployment, native, tmp_path / "output")


def test_unknown_grid_uses_teacher(tmp_path: Path) -> None:
    """No interpolation of validation evidence across grids is assumed."""
    deployment, _ = _fixture(tmp_path)
    root = make_campaign(
        tmp_path / "newgrid",
        seeds=(4,),
        redshifts=(8.0,),
        n=2,
        box_len=50,
        native_changes={"cosmo_params": {"Y_He": 0.245}},
    )
    report = hybrid.run(
        deployment, root / "seeds/seed4/native/seed4/z8", tmp_path / "output"
    )
    assert report["route"]["method"] == "teacher"
    assert report["photon_ledger"]["passed"]


def test_existing_output_is_not_overwritten(tmp_path: Path) -> None:
    """Never replace an existing completed simulation or frozen deployment."""
    deployment, native = _fixture(tmp_path)
    with pytest.raises((FileExistsError, ValueError)):
        hybrid.run(deployment, native, deployment)


def test_source_change_requires_new_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Identical weights do not authorize a different inference implementation."""
    deployment, native = _fixture(tmp_path)
    hashes = hybrid._source_hashes()
    monkeypatch.setattr(
        hybrid, "_source_hashes", lambda: hashes | {"changed.py": "new"}
    )
    with pytest.raises(ValueError, match="implementation changed"):
        hybrid.run(deployment, native, tmp_path / "output")


def test_in_memory_input_mutation_is_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preserve native array bytes even if a predictor changes its arguments."""
    deployment, native = _fixture(tmp_path)
    original = Emulator.predict

    def changed(
        self: Emulator, source: np.ndarray, *args: object, **kwargs: object
    ) -> dict:
        result = original(self, source, *args, **kwargs)
        source.flat[0] += 1
        return result

    monkeypatch.setattr(Emulator, "predict", changed)
    with pytest.raises(ValueError, match="arrays changed"):
        hybrid.run(deployment, native, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_serialization_corruption_has_no_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A passing in-memory budget does not replace a saved-array audit."""
    deployment, native = _fixture(tmp_path)
    original = np.savez_compressed

    def corrupted(path: Path, **arrays: np.ndarray) -> None:
        arrays["xHII"] = np.zeros_like(arrays["xHII"])
        arrays["xHI"] = np.ones_like(arrays["xHI"])
        original(path, **arrays)

    monkeypatch.setattr(hybrid.np, "savez_compressed", corrupted)
    with pytest.raises(FloatingPointError, match="ledger"):
        hybrid.run(deployment, native, tmp_path / "output")
    assert not (tmp_path / "output/COMPLETED").exists()


@pytest.mark.parametrize(
    "changed", ["sigma_cm2", "directions", "exposure_steps", "max_sweeps"]
)
def test_fallback_settings_must_match_teacher_campaigns(
    tmp_path: Path, changed: str
) -> None:
    """A caller cannot silently choose a different fallback physical target."""
    deployment, _ = _fixture(tmp_path)
    manifest = json.loads((deployment / "deployment.json").read_text())
    policy = json.loads((deployment / "policy.json").read_text())
    settings = manifest["teacher_settings"]
    settings[changed] *= 2
    with pytest.raises(ValueError, match="settings differ"):
        hybrid.freeze_deployment(
            {"candidate": tmp_path / "model"},
            policy,
            manifest["teacher_physics"],
            settings,
            tmp_path / "bad",
        )
    assert not (tmp_path / "bad").exists()


def test_collection_requires_every_teacher_configuration(tmp_path: Path) -> None:
    """Mixed training archives must agree individually, not just at top level."""
    deployment, _ = _fixture(tmp_path)
    manifest = json.loads((deployment / "deployment.json").read_text())
    model = Emulator.load(tmp_path / "model")
    identity = model.metadata["teacher_identity"]
    bad = copy.deepcopy(identity)
    bad["configuration"]["analytic"]["directions"] *= 2
    model.metadata["teacher_identity"] = {
        "campaigns": [{"identity": identity}, {"identity": bad}]
    }
    with pytest.raises(ValueError, match="settings differ"):
        hybrid._teacher_identity(
            model, manifest["teacher_settings"], hybrid._source_hashes()
        )
