"""Contract tests for the untouched native C2-Ray backend."""

import struct
from pathlib import Path

import numpy as np
import pytest

from research.c2ray_teacher.build import (
    FIELD_NAMES,
    META_NAMES,
    _patch_log_format,
    read_output,
    validate_configuration,
)


def test_logging_format_patch_is_guarded_and_only_changes_descriptor(tmp_path):
    """A malformed log FORMAT must not prevent genuine transport execution."""
    code = tmp_path / "code"
    code.mkdir()
    source = code / "radiation_sed_parameters.f90"
    original = "write(logf,'(A,A,A(ES12.5),A//)') label, value\n"
    source.write_text(original)
    record = _patch_log_format(code, tmp_path / "logging.patch")
    assert source.read_text() == original.replace("A(ES12.5)", "A,(ES12.5)")
    assert record["numerical_expressions_changed"] is False
    hashes = record["files"]["code/radiation_sed_parameters.f90"]
    assert hashes["before_sha256"] != hashes["after_sha256"]
    assert "-write" in (tmp_path / "logging.patch").read_text()
    with pytest.raises(ValueError, match="anchor"):
        _patch_log_format(code, tmp_path / "again.patch")


def test_grid_rejects_upstream_zero_convergence_threshold():
    """Reject grids whose native convergence threshold would be zero."""
    with pytest.raises(ValueError, match="16"):
        validate_configuration(8, 0.074, 0.3, 0.048, 0.7)


def test_output_contract_and_truncation(tmp_path: Path):
    """Decode field axes correctly and never accept partial outputs."""
    path = tmp_path / "out.bin"
    with path.open("wb") as stream:
        stream.write(b"C2RTOUT1")
        stream.write(struct.pack("<4i", 2, 1, len(FIELD_NAMES), len(META_NAMES)))
        np.arange(len(META_NAMES), dtype="<f8").tofile(stream)
        for field in range(len(FIELD_NAMES)):
            (np.arange(8, dtype="<f8") + 100 * field).tofile(stream)
    output = read_output(path)
    assert output["dimension"] == 2
    assert output["steps"][0][META_NAMES[3]] == 3
    assert output["fields"][FIELD_NAMES[1]][0, 1, 0, 0] == 101
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="size"):
        read_output(path)


def test_large_output_offsets_do_not_overflow_int32(tmp_path: Path) -> None:
    """A sparse >2 GiB fixture exercises offsets without writing dense data."""
    path = tmp_path / "large.bin"
    n, steps = 272, 2
    expected = 24 + 8 * steps * (len(META_NAMES) + len(FIELD_NAMES) * n**3)
    with path.open("wb") as stream:
        stream.write(b"C2RTOUT1")
        stream.write(struct.pack("<4i", n, steps, len(FIELD_NAMES), len(META_NAMES)))
        stream.truncate(expected)
        stream.seek(expected - 8)
        stream.write(struct.pack("<d", 0.125))
    result = read_output(path)
    assert result["fields"][FIELD_NAMES[-1]][-1, -1, -1, -1] == 0.125
