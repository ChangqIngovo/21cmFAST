"""Contract tests for the untouched native C2-Ray backend."""

import struct
from pathlib import Path

import numpy as np
import pytest

from research.c2ray_teacher.build import (
    FIELD_NAMES,
    META_NAMES,
    read_output,
    validate_configuration,
)


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
