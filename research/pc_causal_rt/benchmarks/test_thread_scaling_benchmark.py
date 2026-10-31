"""Check reproducible pilot inputs and preservation of recorded measurements."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from research.pc_causal_rt.benchmarks.thread_scaling_benchmark import (
    _initial_state,
    benchmark,
    main,
)


def test_default_state_matches_executed_pilot_bytes():
    """Keep the refactored CLI's inputs equal to the independently run pilot."""
    evidence = Path(__file__).parents[1] / "multicore_thread_validation.json"
    expected = json.loads(evidence.read_text())["original_initial_state_sha256"]["32"]
    arrays, parameters = _initial_state(32, 8)
    for name, array in zip(
        ("capacity", "ionized", "photons", "rate"), arrays[:4], strict=True
    ):
        actual = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
        assert actual == expected[name]
    assert parameters["dt_seconds"] == expected["dt_seconds"]


def test_invalid_thread_count_rejected_before_measurement():
    """The benchmark cannot request a thread count unsupported by the solver."""
    with pytest.raises(ValueError, match="thread counts"):
        benchmark(dimensions=(4,), thread_counts=(33,), repeats=1, steps=1)


def test_existing_output_is_not_overwritten(tmp_path, monkeypatch):
    """An existing timing record remains unchanged when the CLI refuses it."""
    output = tmp_path / "existing.json"
    original = '{"old_measurement": 1}\n'
    output.write_text(original)
    monkeypatch.setattr(
        "sys.argv", ["thread_scaling_benchmark", "--output", str(output)]
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert output.read_text() == original
