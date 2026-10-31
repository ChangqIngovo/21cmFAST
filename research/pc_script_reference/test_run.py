"""Independent count and archive-safety tests for the reference adapter."""
import hashlib

import numpy as np
import pytest

from research.pc_script_reference.run import archive_path, selected_nodes, summarize, verify_files


def test_extensive_source_not_multiplied_by_density():
    capacity = np.arange(1., 9.).reshape(2, 2, 2)
    source = np.zeros_like(capacity)
    source[0, 0, 0] = 9.
    xhi = np.full_like(capacity, .75)
    metrics = summarize(source, capacity, xhi, np.ones_like(capacity))
    assert metrics["source_per_atom"] == .25
    assert metrics["photon_budget_expected_xHI_mass"] == .75
    assert metrics["script_xHI_mass"] == .75
    assert metrics["native_es_source_minus_ions_per_atom"] == .25


def test_saturation_does_not_imply_negative_neutral_fraction():
    capacity = np.ones((2, 2, 2))
    metrics = summarize(capacity * 2, capacity, np.zeros_like(capacity), capacity * .2)
    assert metrics["photon_budget_expected_xHI_mass"] == 0.
    assert metrics["source_per_atom"] == 2.


def test_checkpoints_always_include_terminal():
    assert selected_nodes(10, 4) == [0, 4, 8, 9]
    with pytest.raises(ValueError):
        selected_nodes(10, 0)


def test_archive_mutation_is_rejected(tmp_path):
    path = tmp_path / "source.npz"
    path.write_bytes(b"original source")
    receipt = {"files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}}
    verify_files(tmp_path, receipt)
    path.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="checksum"):
        verify_files(tmp_path, receipt)


def test_archive_paths_cannot_escape_root(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        archive_path(tmp_path, "../source.npz")


def test_missing_required_receipt_entry_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="checksum"):
        verify_files(tmp_path, {"files": {}}, ("source.npz",))
