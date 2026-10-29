"""Independent saved-field checks for the native-budget comparison."""

import hashlib
import json

import numpy as np
import pytest

from research.pc_native_rt.report import audit_case, build_report


def _fixture(root):
    source = np.full((4, 4, 4), .5)
    capacity = np.ones_like(source)
    capacity[0] = 1.2
    ionized = source.copy()
    xhi = 1 - ionized / capacity
    native = {
        "source": source, "capacity": capacity,
        "native_es_xhi": xhi.astype(np.float32),
        "native_brightness": (27 * capacity * xhi).astype(np.float32),
    }
    arrays = native | {
        "ionized": ionized, "radiation": np.zeros_like(source),
        "rt_xhi": xhi, "rt_brightness": native["native_brightness"],
    }
    np.savez(root / "native.npz", **native)
    np.savez(root / "fields.npz", **arrays)
    digest = lambda value: hashlib.sha256(value.tobytes()).hexdigest()
    row = {
        "case_id": "example", "seed": 111, "redshift": 8., "box_len_cMpc": 50.,
        "setting": {"directions": 24, "refinement": 1, "cfl": .4},
        "fields_path": "fields.npz",
        "native": {"fields_path": "native.npz", "source_sha256": digest(source),
                   "capacity_sha256": digest(capacity)},
    }
    return row, arrays


def test_independent_ledger_and_native_identity(tmp_path):
    row, _ = _fixture(tmp_path)
    report = audit_case(tmp_path, row)
    assert report["saved_field_ledger_relative_residual"] == 0
    assert report["native_inputs_identical"]
    assert report["rt_xHI_mass"] == pytest.approx(1 - .5 / 1.05)


def test_saved_ion_error_is_not_hidden_by_claimed_solver_success(tmp_path):
    row, arrays = _fixture(tmp_path)
    arrays["ionized"][0, 0, 0] += .01
    np.savez(tmp_path / "fields.npz", **arrays)
    with pytest.raises(ValueError, match="photon ledger"):
        audit_case(tmp_path, row)


def test_native_budget_changed_without_total_change_rejected(tmp_path):
    row, arrays = _fixture(tmp_path)
    arrays["source"][0, 0, 0] += .01
    arrays["source"][1, 0, 0] -= .01
    np.savez(tmp_path / "fields.npz", **arrays)
    with pytest.raises(ValueError, match="input differs"):
        audit_case(tmp_path, row)


def test_report_never_certifies_convergence_from_global_budget(tmp_path):
    row, _ = _fixture(tmp_path)
    (tmp_path / "summary.json").write_text(json.dumps({
        "cases": [row], "configuration": {"box_len": 50.},
        "all_exhaustion_checks_passed": True,
    }))
    report = build_report(tmp_path)
    assert report["all_native_inputs_identical"]
    assert report["all_saved_field_ledgers_passed"]
    assert not report["convergence_certified"]
    assert (tmp_path / "native_rt_comparison.png").stat().st_size > 1000
