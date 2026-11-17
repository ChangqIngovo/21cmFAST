"""Check native cache provenance and genuinely unseen evaluation identities."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from research.pc_analytic_emulator.finalization_20261117 import study
from research.pc_analytic_emulator.test_data import make_campaign


def native_frame(tmp_path: Path) -> tuple[dict, dict]:
    """Make a receipt-backed native frame without requiring teacher inference."""
    root = make_campaign(
        tmp_path / "campaign",
        seeds=(123,),
        redshifts=(8.0,),
        n=8,
        native_changes={"cosmo_params": {"Y_He": 0.245}},
    )
    native = root / "seeds/seed123/native/seed123/z8"
    frame = {
        "native_directory": str(native),
        "seed": 123,
        "redshift": 8.0,
        "dimension": 8,
        "box_len": 256.0,
    }
    with np.load(native / "source.npz", allow_pickle=False) as d:
        arrays = {
            "source": d["source"],
            "capacity": d["capacity"],
            "native_es_xHI": d["native_es_xhi"],
        }
    return frame, arrays


def test_native_cache_audit_reads_only_native_inputs(tmp_path: Path) -> None:
    """Validation geometry and source arrays are backed by native receipts."""
    frame, arrays = native_frame(tmp_path)
    record = study._native_frame(frame, arrays)
    assert record["geometry"] == {"DIM": 24, "HII_DIM": 8, "BOX_LEN": 256.0}
    assert "native.json" in record["files_sha256"]
    assert "teacher_xHI" not in arrays


@pytest.mark.parametrize(
    ("key", "value"),
    [("seed", 124), ("redshift", 7.5), ("dimension", 16), ("box_len", 50.0)],
)
def test_native_frame_identity_must_match(
    tmp_path: Path, key: str, value: float
) -> None:
    """A valid native receipt cannot lend identity to a different cached frame."""
    frame, arrays = native_frame(tmp_path)
    frame[key] = value
    with pytest.raises(ValueError, match="frame identity"):
        study._native_frame(frame, arrays)


@pytest.mark.parametrize("key", ["source", "capacity", "native_es_xHI"])
def test_cached_native_arrays_must_match(tmp_path: Path, key: str) -> None:
    """Native arrays must agree even if a cache has its own valid receipt."""
    frame, arrays = native_frame(tmp_path)
    arrays[key] = arrays[key] + 0.01
    with pytest.raises(ValueError, match="native arrays"):
        study._native_frame(frame, arrays)


def test_native_metadata_checksum_is_required(tmp_path: Path) -> None:
    """Edited geometry metadata is rejected before it can select a route."""
    frame, arrays = native_frame(tmp_path)
    path = Path(frame["native_directory"]) / "native.json"
    value = json.loads(path.read_text())
    value["identity"]["inputs"]["simulation_options"]["BOX_LEN"] = 50.0
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="checksum"):
        study._native_frame(frame, arrays)


def test_development_seed_union_includes_every_candidate() -> None:
    """Training, validation and previously inspected test seeds are excluded."""
    models = {
        "a": SimpleNamespace(
            metadata={"split_seeds": {"train": [1], "validation": [2], "test": [3]}}
        ),
        "b": SimpleNamespace(
            metadata={"split_seeds": {"train": [4], "validation": [5], "test": []}}
        ),
    }
    assert study._development_seeds(models, [6]) == {1, 2, 3, 4, 5, 6}


def test_missing_training_split_cannot_claim_blindness() -> None:
    """Unseen status requires recorded training membership."""
    with pytest.raises(ValueError, match="seed provenance"):
        study._development_seeds({"a": SimpleNamespace(metadata={})}, [])


@pytest.mark.parametrize("artifact", ["features.json", "validation.npz"])
def test_corrupt_cache_rejected_before_prediction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, artifact: str
) -> None:
    """Both cache metadata and cached native/teacher fields require their hashes."""
    frame, arrays = native_frame(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    frame["cache"] = str(cache / "validation.npz")
    np.savez(cache / "validation.npz", **arrays)
    (cache / "features.json").write_text(json.dumps({"validation_snapshots": [frame]}))
    (cache / "receipt.json").write_text(
        json.dumps(
            {
                "files": {
                    name: study.digest(cache / name)
                    for name in ("features.json", "validation.npz")
                }
            }
        )
    )
    with (cache / artifact).open("ab") as stream:
        stream.write(b" ")
    candidates = tmp_path / "candidates.json"
    candidates.write_text(json.dumps({"candidate": "unused"}))
    monkeypatch.setattr(study, "_model_hashes", lambda _: {"model.json": "a" * 64})
    monkeypatch.setattr(study.Emulator, "load", lambda _: SimpleNamespace())
    with pytest.raises(ValueError, match=r"cache .*changed"):
        study.validate(
            cache, candidates, tmp_path / "unused_settings.json", tmp_path / "out"
        )


def test_empty_candidate_set_is_rejected(tmp_path: Path) -> None:
    """A missing candidate has an actionable error before cache access."""
    candidates = tmp_path / "candidates.json"
    candidates.write_text("{}")
    with pytest.raises(ValueError, match="At least one candidate"):
        study.validate(
            tmp_path / "cache", candidates, tmp_path / "settings", tmp_path / "out"
        )
