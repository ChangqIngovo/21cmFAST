"""Check independent teacher labels, immutable inputs and process allocation."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

import numpy as np
import pytest

from research.pc_analytic import campaign


def _template(path: Path) -> Path:
    value = {
        "random_seed": 222,
        "node_redshifts": [],
        "cosmo_params": {},
        "simulation_options": {
            "HII_DIM": 128,
            "DIM": 384,
            "BOX_LEN": 256,
            "N_THREADS": 8,
        },
        "matter_options": {"SOURCE_MODEL": "L-INTEGRAL", "MINIMIZE_MEMORY": True},
        "astro_params": {"R_BUBBLE_MAX": 50},
        "astro_options": {
            "USE_MCGS": False,
            "USE_TS_FLUCT": False,
            "USE_REIONIZATION_PHOTOHEATING_FEEDBACK": False,
            "RECOMB_MODEL": "none",
            "CELL_RECOMB": False,
            "PHOTON_CONS_TYPE": "no-photoncons",
        },
    }
    path.write_text(json.dumps(value))
    return path


def _config(tmp_path: Path, **overrides: object) -> dict:
    return campaign.validate_config(
        {
            "native_template_json": str(_template(tmp_path / "template.json")),
            **overrides,
        }
    )


def test_defaults_are_explicit_and_independent(tmp_path: Path) -> None:
    """Defaults produce independent seeds using the declared small native grid."""
    config = _config(tmp_path)
    assert config["seeds"] == list(range(1001, 1009))
    assert (config["workers"], config["threads"], config["cpus"]) == (4, 4, 16)
    assert config["dimension"] == 32
    assert config["highres_dimension"] == 256
    assert config["box_len"] == 256
    assert config["analytic"]["directions"] == 96
    assert config["analytic"]["exposure_steps"] == 32
    assert "c2ray" not in config


@pytest.mark.parametrize(
    "override",
    [
        {"workers": 5},
        {"threads": 0},
        {"cpus": True},
        {"seeds": [1, 1]},
        {"redshifts": [7, 8]},
        {"sigma_cm2": float("nan")},
        {"sigma_cm2": "invalid"},
        {"redshifts": [8.0, 7.9999999999999]},
        {"analytic": {"directions": 8}},
        {"analytic": {"batch_tail_tolerance": 1e-9}},
        {"native_template_sha256": "wrong"},
    ],
)
def test_bad_allocation_or_identity_is_rejected(tmp_path: Path, override: dict) -> None:
    """Reject oversubscription, invalid sampling and changed native templates."""
    with pytest.raises(
        ValueError,
        match=r"allocation|integer|Seeds|Redshifts|sigma|checksum|analytic|direction",
    ):
        _config(tmp_path, **override)


def test_template_physics_rejected(tmp_path: Path) -> None:
    """Reject physics outside the validated cumulative-budget teacher model."""
    template = _template(tmp_path / "template.json")
    raw = json.loads(template.read_text())
    raw["astro_options"]["RECOMB_MODEL"] = "Sobacchi2014"
    template.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="RECOMB_MODEL"):
        campaign.validate_config({"native_template_json": str(template)})


def _fake_native_data(n: int = 4) -> dict:
    rng = np.random.default_rng(7)
    capacity = rng.uniform(0.8, 1.2, (n, n, n))
    source = rng.uniform(0.1, 0.5, (n, n, n))
    return {
        "source": source,
        "capacity": capacity,
        "native_es_xhi": np.ones_like(source),
        "native_brightness": np.ones_like(source),
        "metadata": {
            "seed": 1001,
            "redshift": 8.0,
            "box_len": 128.0,
            "dimension": n,
            "nH_mean_comoving_cm3": 2e-7,
            "atoms_per_count_unit": 1.0,
        },
    }


def test_teacher_label_has_independent_ledger_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Publish only audited labels and verify their stored bytes before resume."""
    config = _config(tmp_path, brightness="none", workers=1, threads=1, cpus=1)
    data = _fake_native_data()
    calls = []
    monkeypatch.setattr(
        campaign, "load_native_snapshot", lambda *_a, **_kw: copy.deepcopy(data)
    )

    def solve(source: np.ndarray, capacity: np.ndarray, **_kwargs: object) -> dict:
        calls.append(1)
        return {
            "ionized": source.copy(),
            "remaining_radiation": np.zeros((1, *source.shape)),
            "escaped": np.zeros(1),
            "unemitted_source": np.zeros_like(source),
            "audit": {"complete": True, "ledger_passed": True},
            "metadata": {"mock": True},
        }

    monkeypatch.setattr(campaign, "solve", solve)
    native = tmp_path / "native"
    native.mkdir()
    (native / "receipt.json").write_text("{}")
    output = tmp_path / "teacher"
    identity = {"campaign": "same", "kernel": "same"}
    result = campaign._teacher_snapshot(native, output, config, identity)
    assert result["photon_ledger"]["relative_photon_residual"] < 1e-14
    assert result["complete"] is True
    assert (output / "receipt.json").is_file()
    assert campaign._teacher_snapshot(native, output, config, identity) == result
    assert len(calls) == 1
    receipt_path = output / "receipt.json"
    saved_receipt = receipt_path.read_text()
    forged_receipt = json.loads(saved_receipt)
    forged_receipt["record"]["neutral_fraction_mass"] = 1.0
    receipt_path.write_text(json.dumps(forged_receipt))
    with pytest.raises(ValueError, match="differs from saved metrics"):
        campaign._teacher_snapshot(native, output, config, identity)
    receipt_path.write_text(saved_receipt)
    with np.load(output / "fields.npz") as fields:
        np.testing.assert_array_equal(fields["source"], data["source"])
        np.testing.assert_allclose(
            fields["teacher_xHI"], 1 - data["source"] / data["capacity"]
        )
    with (output / "fields.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        campaign._teacher_snapshot(native, output, config, identity)


@pytest.mark.parametrize("fault", ["ledger", "bounds", "source", "complete"])
def test_bad_teacher_is_not_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    """Never publish a receipt when an independent field or budget check fails."""
    config = _config(tmp_path, brightness="none", workers=1, threads=1, cpus=1)
    data = _fake_native_data()
    monkeypatch.setattr(
        campaign, "load_native_snapshot", lambda *_a, **_kw: copy.deepcopy(data)
    )

    def solve(source: np.ndarray, capacity: np.ndarray, **_kwargs: object) -> dict:
        ions = source.copy()
        if fault == "ledger":
            ions *= 0.9
        if fault == "bounds":
            ions[0, 0, 0] = 2 * capacity[0, 0, 0]
        if fault == "source":
            source[:] *= 2
        return {
            "ionized": ions,
            "remaining_radiation": np.zeros((1, *source.shape)),
            "escaped": np.zeros(1),
            "unemitted_source": np.zeros_like(source),
            "audit": {"complete": fault != "complete", "ledger_passed": True},
            "metadata": {},
        }

    monkeypatch.setattr(campaign, "solve", solve)
    native = tmp_path / "native"
    native.mkdir()
    (native / "receipt.json").write_text("{}")
    output = tmp_path / "teacher"
    with pytest.raises((ValueError, RuntimeError)):
        campaign._teacher_snapshot(native, output, config, {"campaign": "same"})
    assert not (output / "receipt.json").exists()


def test_worker_environment_restored() -> None:
    """Spawn inherits bounded thread counts without altering the caller's env."""
    import os

    previous = dict(os.environ)
    with campaign._thread_environment(3):
        assert os.environ["OMP_NUM_THREADS"] == "3"
        assert os.environ["OMP_THREAD_LIMIT"] == "3"
        assert os.environ["OMP_DYNAMIC"] == "FALSE"
        assert os.environ["OMP_WAIT_POLICY"] == "PASSIVE"
        assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
    assert dict(os.environ) == previous


def _fake_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        campaign, "_runtime_identity", lambda: {"native_extension_sha256": "native"}
    )
    monkeypatch.setattr(campaign, "build_identity", lambda: {"kernel": "same"})
    monkeypatch.setattr(
        campaign, "_implementation_hashes", lambda: {"campaign": "same"}
    )


def test_campaign_requires_complete_seed_redshift_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retain successes after a failed seed and reject settings drift on retry."""
    config = _config(
        tmp_path, workers=1, threads=1, cpus=1, seeds=[1001, 1002], redshifts=[8.0, 7.0]
    )
    _fake_runtime(monkeypatch)
    failures = {1002}
    calls = []

    def run_seed(seed: int, _config: dict, _root: str, _identity: dict) -> list[dict]:
        calls.append(seed)
        if seed in failures:
            raise RuntimeError("native simulation failed")
        return [
            {
                "seed": seed,
                "redshift": z,
                "photon_ledger": {"relative_photon_residual": 1e-15},
            }
            for z in _config["redshifts"]
        ]

    monkeypatch.setattr(campaign, "_run_seed", run_seed)
    output = tmp_path / "results"
    with pytest.raises(RuntimeError, match="1 teacher seed"):
        campaign.run(config, output)
    assert not (output / "COMPLETED").exists()
    summary = json.loads((output / "summary.json").read_text())
    assert summary["completed_snapshots"] == 2
    assert summary["failed_seeds"] == [
        {"seed": 1002, "error": "native simulation failed"}
    ]
    failures.clear()
    result = campaign.run(config, output)
    assert result["complete"]
    assert len(result["samples"]) == 4
    assert {(row["seed"], row["redshift"]) for row in result["samples"]} == {
        (1001, 8.0),
        (1001, 7.0),
        (1002, 8.0),
        (1002, 7.0),
    }
    assert (output / "COMPLETED").is_file()
    assert calls == [1001, 1002, 1001, 1002]
    with pytest.raises(ValueError, match="Campaign identity changed"):
        campaign.run(config | {"sigma_cm2": config["sigma_cm2"] * 2}, output)


def test_campaign_rejects_missing_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A seed worker cannot silently omit a requested teacher snapshot."""
    config = _config(
        tmp_path, workers=1, threads=1, cpus=1, seeds=[1001], redshifts=[8.0]
    )
    _fake_runtime(monkeypatch)
    monkeypatch.setattr(campaign, "_run_seed", lambda *_args: [])
    with pytest.raises(RuntimeError, match="incomplete or duplicated"):
        campaign.run(config, tmp_path / "results")
    assert not (tmp_path / "results" / "COMPLETED").exists()


def test_seed_logs_saved_and_verified_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Worker logging identifies seed startup, native readiness and label reuse."""
    config = _config(
        tmp_path, workers=1, threads=1, cpus=1, seeds=[1001], redshifts=[8.0]
    )
    identity = {"native_runtime": {"extension": "same"}}
    monkeypatch.setattr(
        campaign, "_runtime_identity", lambda: identity["native_runtime"]
    )
    monkeypatch.setattr(
        campaign, "native_inputs_for_seed", lambda template, *_args: template
    )
    monkeypatch.setattr(
        campaign,
        "_native_snapshots",
        lambda *_args: [{"redshift": 8.0, "h5_path": "native/z8/native.h5"}],
    )
    configurations = []
    monkeypatch.setattr(
        campaign.logging, "basicConfig", lambda **kwargs: configurations.append(kwargs)
    )

    def label(_native: Path, folder: Path, _config: dict, _identity: dict) -> dict:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "receipt.json").write_text("{}")
        return {
            "seed": 1001,
            "redshift": 8.0,
            "photon_ledger": {"relative_photon_residual": 1e-16},
        }

    monkeypatch.setattr(campaign, "_teacher_snapshot", label)
    caplog.set_level(logging.INFO, logger=campaign.__name__)
    campaign._run_seed(1001, config, str(tmp_path / "results"), identity)
    assert "Teacher seed started: seed=1001 snapshots=1 threads=1" in caplog.text
    assert "Native snapshots ready: seed=1001 count=1" in caplog.text
    assert "Teacher saved: seed=1001 z=8" in caplog.text
    caplog.clear()
    campaign._run_seed(1001, config, str(tmp_path / "results"), identity)
    assert "Teacher resumed and verified: seed=1001 z=8" in caplog.text
    assert "Teacher saved:" not in caplog.text
    assert configurations == [{"level": logging.INFO, "format": "%(message)s"}] * 2
