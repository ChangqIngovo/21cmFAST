"""Exercise label-free inference, saved photon ledgers and immutable inputs."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from research.pc_analytic_emulator import predict
from research.pc_analytic_emulator.model import Emulator
from research.pc_analytic_emulator.test_model import fixture_model, residual_model


def _fixture(tmp_path: Path, *, residual: bool = False) -> tuple[Path, Path]:
    source = np.linspace(0.1, 0.7, 64).reshape(4, 4, 4)
    capacity = np.linspace(0.7, 1.3, 64).reshape(source.shape)
    native = np.linspace(0.2, 0.9, 64).reshape(source.shape)
    model = fixture_model(source, capacity, native)
    if residual:
        model = residual_model(model)
    directory = tmp_path / "model"
    model.save(directory)
    path = tmp_path / "inputs.npz"
    np.savez(path, source=source, capacity=capacity, native_es_xHI=native)
    return directory, path


@pytest.mark.parametrize("residual", [False, True])
def test_saved_inference_matches_model_and_preserves_inputs(
    tmp_path: Path, residual: bool
) -> None:
    """Publish self-contained fields with independently verified stored counts."""
    model, inputs = _fixture(tmp_path, residual=residual)
    before = hashlib.sha256(inputs.read_bytes()).hexdigest()
    with np.load(inputs, allow_pickle=False) as data:
        expected = Emulator.load(model).predict(
            data["source"], data["capacity"], data["native_es_xHI"], 7.5, 256
        )
    output = tmp_path / "prediction"
    report = predict.run(model, inputs, redshift=7.5, box_len=256, output=output)
    assert report["complete"]
    assert report["saved_field_photon_ledger"]["passed"]
    assert report["input"]["sha256"] == before
    assert hashlib.sha256(inputs.read_bytes()).hexdigest() == before
    assert len(report["model"]["files_sha256"]) == (6 if residual else 3)
    with np.load(output / "fields.npz", allow_pickle=False) as saved:
        for name in ("xHI", "xHII", "raw_xHI", "raw_xHII", "latent_xHII"):
            np.testing.assert_array_equal(saved[name], expected[name])
        emitted = np.sum(saved["source"], dtype=np.longdouble)
        absorbed = np.sum(
            np.multiply(saved["capacity"], saved["xHII"], dtype=np.longdouble)
        )
        unabsorbed = np.longdouble(saved["predicted_unabsorbed_photons"])
        assert abs(emitted - absorbed - unabsorbed) / emitted < 1e-10
    assert json.loads((output / "report.json").read_text()) == report
    assert (output / "COMPLETED").is_file()
    assert (
        report["files_sha256"]["fields.npz"]
        == hashlib.sha256((output / "fields.npz").read_bytes()).hexdigest()
    )


def test_cli_needs_no_teacher_fields(tmp_path: Path) -> None:
    """Run the public module using only the three inference-time arrays."""
    model, inputs = _fixture(tmp_path)
    output = tmp_path / "from_cli"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "research.pc_analytic_emulator.predict",
            "--model",
            str(model),
            "--input",
            str(inputs),
            "--redshift",
            "7.5",
            "--box-len",
            "256",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=os.environ | {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
    )
    assert json.loads(result.stdout)["complete"]
    assert (output / "COMPLETED").is_file()


def test_native_archive_lowercase_neutral_key_is_accepted(tmp_path: Path) -> None:
    """Use native source.npz directly without generating any teacher archive."""
    model, inputs = _fixture(tmp_path)
    with np.load(inputs, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    arrays["native_es_xhi"] = arrays.pop("native_es_xHI")
    np.savez(inputs, **arrays)
    report = predict.run(
        model, inputs, redshift=7.5, box_len=256, output=tmp_path / "out"
    )
    assert report["input"]["arrays_used"] == ["source", "capacity", "native_es_xhi"]
    assert report["saved_field_photon_ledger"]["passed"]
    arrays["native_es_xHI"] = arrays["native_es_xhi"]
    np.savez(inputs, **arrays)
    with pytest.raises(ValueError, match="only one native ES"):
        predict.run(
            model, inputs, redshift=7.5, box_len=256, output=tmp_path / "ambiguous"
        )


@pytest.mark.parametrize("location", ["existing", "model", "model/child", "inputs"])
def test_output_never_overwrites_an_existing_path(
    tmp_path: Path, location: str
) -> None:
    """Reject destructive destinations before changing source/model artifacts."""
    model, inputs = _fixture(tmp_path)
    output = inputs if location == "inputs" else tmp_path / location
    if location == "existing":
        output.mkdir()
        (output / "keep.txt").write_text("untouched")
    before = {
        str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    }
    with pytest.raises((ValueError, FileExistsError)):
        predict.run(model, inputs, redshift=7.5, box_len=256, output=output)
    assert {
        str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    } == before


@pytest.mark.parametrize("fault", ["missing", "complex", "nan", "negative_capacity"])
def test_invalid_input_never_receives_completion(tmp_path: Path, fault: str) -> None:
    """Reject missing, nonphysical and lossy-coercion input arrays."""
    model, inputs = _fixture(tmp_path)
    with np.load(inputs, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    if fault == "missing":
        arrays.pop("native_es_xHI")
    elif fault == "complex":
        arrays["source"] = arrays["source"].astype(complex) + 0.1j
    elif fault == "nan":
        arrays["source"][0, 0, 0] = np.nan
    else:
        arrays["capacity"][0, 0, 0] = -1
    np.savez(inputs, **arrays)
    output = tmp_path / "prediction"
    with pytest.raises((TypeError, ValueError)):
        predict.run(model, inputs, redshift=7.5, box_len=256, output=output)
    assert not (output / "COMPLETED").exists()


def test_extra_teacher_labels_are_not_loaded(tmp_path: Path) -> None:
    """An inaccessible object label cannot enter the inference feature path."""
    model, inputs = _fixture(tmp_path)
    with np.load(inputs, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    np.savez(inputs, **arrays, teacher_xHI=np.array([object()], dtype=object))
    report = predict.run(
        model, inputs, redshift=7.5, box_len=256, output=tmp_path / "prediction"
    )
    assert report["complete"]
    assert report["input"]["arrays_used"] == ["source", "capacity", "native_es_xHI"]


def test_saved_field_corruption_fails_independent_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Do not trust the predictor's passing ledger after serialization."""
    model, inputs = _fixture(tmp_path)
    original = np.savez_compressed

    def corrupted(path: Path, **arrays: np.ndarray) -> None:
        arrays["xHII"] = np.zeros_like(arrays["xHII"])
        arrays["xHI"] = np.ones_like(arrays["xHI"])
        original(path, **arrays)

    monkeypatch.setattr(predict.np, "savez_compressed", corrupted)
    output = tmp_path / "prediction"
    with pytest.raises(FloatingPointError, match=r"saved.*ledger"):
        predict.run(model, inputs, redshift=7.5, box_len=256, output=output)
    assert not (output / "COMPLETED").exists()


def test_input_change_during_prediction_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Require provenance bytes to remain fixed throughout inference."""
    model, inputs = _fixture(tmp_path)
    original = Emulator.predict

    def modified(self: Emulator, *args: object, **kwargs: object) -> dict:
        result = original(self, *args, **kwargs)
        with inputs.open("ab") as stream:
            stream.write(b"changed")
        return result

    monkeypatch.setattr(Emulator, "predict", modified)
    output = tmp_path / "prediction"
    with pytest.raises(ValueError, match="changed during inference"):
        predict.run(model, inputs, redshift=7.5, box_len=256, output=output)
    assert not output.exists()


def _native_inputs(tmp_path: Path, model_path: Path) -> Path:
    from research.pc_analytic.test_campaign import _template

    path = _template(tmp_path / "native_inputs.json")
    inputs = json.loads(path.read_text())
    inputs["simulation_options"].update(HII_DIM=4, DIM=128, BOX_LEN=256)
    path.write_text(json.dumps(inputs))
    common = copy.deepcopy(inputs)
    common.pop("random_seed")
    common["simulation_options"].pop("N_THREADS")
    model = Emulator.load(model_path)
    model.metadata["teacher_physics"] = common
    model.save(model_path)
    return path


def test_trained_model_requires_physical_parameters(tmp_path: Path) -> None:
    """Raw arrays cannot establish compatibility with recorded training physics."""
    model, inputs = _fixture(tmp_path)
    _native_inputs(tmp_path, model)
    with pytest.raises(ValueError, match="native-inputs"):
        predict.run(model, inputs, redshift=7.5, box_len=256, output=tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("fault", [None, "F_STAR10", "grid", "box", "redshift"])
def test_native_parameters_match_training_except_realization_and_threads(
    tmp_path: Path, fault: str | None
) -> None:
    """Reject changed source physics or geometry, while allowing a fresh seed."""
    model, inputs = _fixture(tmp_path)
    path = _native_inputs(tmp_path, model)
    values = json.loads(path.read_text())
    values["random_seed"] = 9001
    values["simulation_options"]["N_THREADS"] = 1
    redshift = 7.5
    if fault == "F_STAR10":
        values["astro_params"][fault] = -1
    elif fault == "grid":
        values["simulation_options"]["HII_DIM"] = 8
    elif fault == "box":
        values["simulation_options"]["BOX_LEN"] = 128
    elif fault == "redshift":
        redshift = 8
    # Accept native.json directly, avoiding a parameter-extraction script.
    path.write_text(json.dumps({"identity": {"inputs": values, "redshift": redshift}}))
    output = tmp_path / "out"
    if fault:
        with pytest.raises(ValueError, match=r"physical|geometry|redshift"):
            predict.run(
                model,
                inputs,
                redshift=7.5,
                box_len=256,
                output=output,
                native_inputs=path,
            )
        assert not output.exists()
    else:
        report = predict.run(
            model,
            inputs,
            redshift=7.5,
            box_len=256,
            output=output,
            native_inputs=path,
        )
        assert report["physical_compatibility"]["training_parameters_matched"]
        assert not report["physical_compatibility"]["native_array_hashes_verified"]


def test_archived_native_metadata_binds_parameter_claims_to_arrays(
    tmp_path: Path,
) -> None:
    """Compare archived native hashes when the supplied sidecar contains them."""
    model, inputs = _fixture(tmp_path)
    path = _native_inputs(tmp_path, model)
    values = json.loads(path.read_text())
    with np.load(inputs, allow_pickle=False) as data:
        hashes = {
            ("native_es_xhi" if name == "native_es_xHI" else name): hashlib.sha256(
                np.ascontiguousarray(data[name]).tobytes()
            ).hexdigest()
            for name in predict.INPUT_ARRAYS
        }
    sidecar = {"identity": {"inputs": values, "redshift": 7.5}, "array_sha256": hashes}
    path.write_text(json.dumps(sidecar))
    report = predict.run(
        model,
        inputs,
        redshift=7.5,
        box_len=256,
        output=tmp_path / "good",
        native_inputs=path,
    )
    assert report["physical_compatibility"]["native_array_hashes_verified"]
    sidecar["array_sha256"]["source"] = "wrong"
    path.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="native array checksum"):
        predict.run(
            model,
            inputs,
            redshift=7.5,
            box_len=256,
            output=tmp_path / "bad",
            native_inputs=path,
        )


def test_saved_weak_ionization_uses_stored_xhii_not_rounded_neutral_complement(
    tmp_path: Path,
) -> None:
    """Retain representable absorbed photons when xHI rounds to exactly one."""
    source = np.full((2, 2, 2), 1e-20)
    inputs = {
        "source": source,
        "capacity": np.ones_like(source),
        "native_es_xHI": np.ones_like(source),
    }
    fields = inputs | {
        "xHI": 1 - source,
        "xHII": source,
        "raw_xHI": 1 - source,
        "raw_xHII": source,
        "latent_xHII": source,
        "predicted_absorbed_photons": np.asarray(8e-20),
        "predicted_unabsorbed_photons": np.asarray(0.0),
    }
    path = tmp_path / "fields.npz"
    np.savez_compressed(path, **fields)
    audit = predict._saved_ledger(path, inputs)
    assert audit["passed"]
    assert audit["field_absorbed_photons"] == pytest.approx(8e-20, rel=1e-15, abs=0)
    assert audit["neutral_field_absorbed_photons"] == 0
    assert audit["neutral_complement_rounding_difference"] == pytest.approx(
        -8e-20, rel=1e-15, abs=0
    )
