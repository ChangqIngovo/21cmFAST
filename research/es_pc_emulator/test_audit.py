"""Residual decomposition and initial-history diagnosis use audited real schemas."""
import csv
import json
import sys

import numpy as np
import pytest

from research.es_pc_emulator.audit import (
    DEFAULT_INITIAL_TOLERANCE,
    _all_compatible,
    _any_mismatch,
    _context,
    audit_dataset,
    main,
    residual_metrics,
)
from research.es_pc_emulator.initial_state import DEFAULT_POLICY
from research.es_pc_emulator.test_pairs import export, pair_fixture


def test_constant_history_offset_is_mean_error_not_spatial_error():
    """A uniform offset has no centred spatial variance."""
    rho = np.arange(1, 9).reshape(2, 2, 2)
    es = np.full(rho.shape, 0.4)
    rt = np.full(rho.shape, 0.7)
    metrics = residual_metrics(es, rt, rho)
    assert metrics["MAE"] == pytest.approx(0.3)
    assert metrics["delta_Q_mass"] == pytest.approx(0.3)
    assert metrics["mean_bias_fraction_of_MSE"] == pytest.approx(1)
    assert metrics["mass_mean_bias_fraction_of_MSE"] == pytest.approx(1)
    assert metrics["centred_residual_RMSE"] < 1e-14
    assert metrics["mass_centred_residual_RMSE"] < 1e-14


def test_cancelled_global_error_does_not_hide_spatial_error():
    """Opposite local errors can cancel globally without yielding accurate fields."""
    es = np.array([0.2, 0.8]).reshape(1, 1, 2)
    rt = np.full_like(es, 0.5)
    metrics = residual_metrics(es, rt, np.ones_like(es))
    assert metrics["delta_Q_volume"] == pytest.approx(0)
    assert metrics["MAE"] == pytest.approx(0.3)
    assert metrics["centred_residual_RMSE"] == pytest.approx(0.3)
    assert metrics["mean_bias_fraction_of_MSE"] < 1e-28
    weighted = residual_metrics(es, rt, np.array([1., 3.]).reshape(es.shape))
    assert weighted["delta_Q_mass"] == pytest.approx(-0.15)
    assert weighted["mass_mean_bias_fraction_of_MSE"] == pytest.approx(0.25)
    assert weighted["mass_weighted_RMSE"]**2 == pytest.approx(
        weighted["delta_Q_mass"]**2 + weighted["mass_centred_residual_RMSE"]**2
    )


def test_zero_residual_is_finite_and_has_zero_bias_fraction():
    """Identical fields have well-defined zero diagnostic errors."""
    field = np.ones((2, 2, 2))
    metrics = residual_metrics(field, field, field)
    assert metrics["RMSE"] == 0
    assert metrics["mean_bias_fraction_of_MSE"] == 0
    assert metrics["mass_mean_bias_fraction_of_MSE"] == 0


@pytest.mark.parametrize("bad", [np.nan, -1.0, 1.1])
def test_invalid_fractions_are_rejected(bad):
    """Invalid physical inputs must not produce plausible error metrics."""
    field = np.ones((2, 2, 2))
    with pytest.raises(ValueError, match="Finite matching"):
        residual_metrics(field * bad, field, field)


def test_single_seed_audit_detects_missing_initial_evolution(tmp_path):
    """A neutral RT start disagrees with a native ES field already ionized."""
    fixture = pair_fixture(tmp_path / "seed")
    export(fixture)
    before = fixture["history"].read_bytes()
    output = tmp_path / "audit"
    report = audit_dataset(
        fixture["output"] / "manifest.json", output,
        histories=[fixture["history"]], sidecars=[fixture["es"]],
    )
    assert report["initial_tolerance"] == DEFAULT_POLICY["max_mass_weighted_error"]
    assert report["groups"][0]["initial_state_compatible_at_export"] is False
    assert report["groups"][0]["initial_state_mismatch"] is True
    for row in report["samples"]:
        assert row["initial_ES_Q_volume"] == pytest.approx(0.3, abs=1e-7)
        assert row["initial_teacher_xHII"] == 1e-6
        assert row["initial_delta_Q_mass"] > 0.2
        assert row["density_reference_vs_native_RMSE"] == 0
        assert row["integrated_emitted_over_reference_nuclei"] > 0
        # Fixture deliberately lacks source provenance. Missing is not zero.
        assert row["native_nion_proxy_over_reference_nuclei"] is None
    assert fixture["history"].read_bytes() == before
    saved = json.loads((output / "summary.json").read_text())
    assert saved == report
    with (output / "samples.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 2
    with pytest.raises(FileExistsError):
        audit_dataset(fixture["output"] / "manifest.json", output)


def test_saved_context_is_used_and_tampered_pair_is_rejected(tmp_path):
    """Saved initial metadata works without archives, while hashes remain enforced."""
    fixture = pair_fixture(tmp_path / "seed")
    manifest = export(fixture)
    report = audit_dataset(fixture["output"] / "manifest.json", tmp_path / "audit")
    assert report["initial_tolerance"] == DEFAULT_POLICY["max_mass_weighted_error"]
    assert report["groups"][0]["initial_state_compatible_at_export"] is False
    assert report["groups"][0]["initial_state_mismatch"] is True
    assert report["samples"][0]["history_available"] is False
    path = fixture["output"] / "snapshot_0001.npz"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash changed"):
        audit_dataset(fixture["output"] / "manifest.json", tmp_path / "tampered")
    assert manifest["samples"][0]["seed"] == 1234


def test_old_pair_without_initial_context_remains_unknown():
    """Absent legacy initial conditions cannot be certified as matching."""
    pair = {"metadata": {"history_sha256": "unknown"}}
    result = _context(pair, {}, {}, DEFAULT_INITIAL_TOLERANCE)
    assert result["initial_state_mismatch"] is None
    assert result["initial_mass_weighted_MAE"] is None


def test_source_proxy_and_integrated_rate_are_reported_separately():
    """The native proxy need not equal the integrated teacher emission."""
    shape = (2, 2, 2)
    pair = {"metadata": {"history_sha256": "history", "es_sidecar_sha256": "none",
                          "snapshot_id": 1, "seed": 7, "redshift": 10.,
                          "physics": {"initial_state": {"xHII": 1e-6}}},
            "photons": np.full(shape, 0.2)}
    history = {"metadata": {"random_seed": 7,
                            "source_provenance": [{}, {"native_n_ion_sum": 4.8}]},
               "capacity": np.ones(shape), "redshifts": [12., 10.],
               "native_xhi": np.ones((2, *shape))}
    result = _context(pair, {"history": (None, history)}, {}, DEFAULT_INITIAL_TOLERANCE)
    assert result["native_nion_proxy_over_reference_nuclei"] == pytest.approx(0.6)
    assert result["integrated_over_native_proxy"] == pytest.approx(1 / 3)
    assert result["initial_state_mismatch"] is False


def test_cli_default_matches_exporter_policy(tmp_path, monkeypatch, capsys):
    """CLI and programmatic audits apply the same default screening threshold."""
    fixture = pair_fixture(tmp_path / "seed")
    export(fixture)
    destination = tmp_path / "cli_audit"
    monkeypatch.setattr(sys, "argv", ["audit", "--dataset",
                       str(fixture["output"] / "manifest.json"),
                       "--output", str(destination)])
    main()
    report = json.loads((destination / "summary.json").read_text())
    assert report["initial_tolerance"] == DEFAULT_POLICY["max_mass_weighted_error"]
    assert report["initial_tolerance"] == 0.001
    assert json.loads(capsys.readouterr().out)[0]["initial_state_mismatch"] is True


def test_export_rejection_is_preserved_and_near_threshold_mismatch_detected():
    """A 0.2 percent initial mismatch must fail the shared 0.1 percent threshold."""
    pair = {"metadata": {"history_sha256": "missing", "initial_state_audit": {
        "compatible": False, "mass_weighted_initial_MAE": 0.002}}}
    result = _context(pair, {}, {}, DEFAULT_INITIAL_TOLERANCE)
    assert result["initial_state_compatible_at_export"] is False
    assert result["initial_state_mismatch"] is True
    loose = _context(pair, {}, {}, 0.01)
    assert loose["initial_state_compatible_at_export"] is False
    assert loose["initial_state_mismatch"] is False


def test_known_failure_is_not_hidden_by_missing_context_in_a_group():
    """Unknown samples cannot erase an independently established failure."""
    assert _any_mismatch([True, None]) is True
    assert _any_mismatch([False, None]) is None
    assert _any_mismatch([False, False]) is False
    assert _all_compatible([False, None]) is False
    assert _all_compatible([True, None]) is None
    assert _all_compatible([True, True]) is True
