"""Operational timeout controls leave archive inputs and kernel math unchanged."""

import json
import subprocess
import sys
from types import ModuleType

import numpy as np
import pytest

from research.pc_script_reference import run as runner


@pytest.mark.parametrize("timeout", [0, -1, np.nan, np.inf, -np.inf, True])
def test_invalid_timeout_does_not_create_output(tmp_path, timeout):
    """Reject invalid limits before touching an output directory."""
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="finite and positive"):
        runner.run(tmp_path / "input", output, timeout_seconds=timeout)
    assert not output.exists()


@pytest.mark.parametrize("timeout", [None, 7200.0])
def test_snapshot_receives_timeout_and_failure_has_no_completion(
    tmp_path, monkeypatch, timeout,
):
    """Pass both default and large limits to the kernel without sealing failures."""
    from research.pc_causal_rt import native_history
    from research.pc_native_rt import workflow
    from research.pc_script_reference import kernel

    root = tmp_path / "input"
    history_root = root / "histories" / "seed222"
    history_root.mkdir(parents=True)
    runtime = {"native_extension_sha256": "test-native"}
    (root / "manifest.json").write_text(json.dumps({
        "configuration": {"seeds": [222], "checkpoint_stride": 1},
        "native_runtime": runtime,
    }))
    (history_root / "history_receipt.json").write_text(json.dumps({
        "identity": {}, "files": {"history.npz": "test-history"},
    }))
    (history_root / "history_manifest.json").write_text("{}")
    history = {
        "metadata": {"seed": 222},
        "redshifts": np.array([8.0, 7.0]),
        "times_seconds": np.array([0.0, 1.0]),
        "rates": np.ones((2, 2, 2, 2)),
        "records": [
            {"fields_path": f"node{i}/source.npz", "redshift": z}
            for i, z in enumerate([8.0, 7.0])
        ],
    }
    monkeypatch.setattr(native_history, "load_history", lambda _: history)
    monkeypatch.setattr(workflow, "_runtime_identity", lambda: runtime)
    monkeypatch.setattr(runner, "verify_files", lambda *args, **kwargs: None)
    monkeypatch.setattr(kernel, "provenance", dict)
    fields = {"source": np.ones((2, 2, 2)), "capacity": np.ones((2, 2, 2))}
    monkeypatch.setattr(
        runner, "_native_snapshot", lambda *args: (None, fields, {"files": {}}),
    )
    single_field = ModuleType("py21cmfast.drivers.single_field")
    single_field.brightness_temperature = None
    monkeypatch.setitem(sys.modules, single_field.__name__, single_field)

    expected = 120.0 if timeout is None else timeout
    observed = []

    def timed_kernel(source, capacity, *, build_dir, timeout_seconds):
        assert source is fields["source"]
        assert capacity is fields["capacity"]
        observed.append(timeout_seconds)
        raise subprocess.TimeoutExpired("test-script", timeout_seconds)

    monkeypatch.setattr(kernel, "run", timed_kernel)
    output = tmp_path / "output"
    options = {} if timeout is None else {"timeout_seconds": timeout}
    with pytest.raises(subprocess.TimeoutExpired):
        runner.run(root, output, **options)
    assert observed == [expected]
    assert not (output / "COMPLETED").exists()
    assert not (output / "seed222" / "node00000" / "receipt.json").exists()


def test_cli_passes_large_snapshot_timeout(tmp_path, monkeypatch):
    """The CLI forwards the requested limit to the resumable entry point."""
    received = []
    source, output = tmp_path / "input", tmp_path / "output"
    monkeypatch.setattr(
        runner, "run", lambda *args, **kwargs: received.append((args, kwargs)),
    )
    monkeypatch.setattr(sys, "argv", [
        "run", "--input-run", str(source), "--output", str(output), "--timeout", "7200",
    ])
    runner.main()
    assert received == [((source, output), {"timeout_seconds": 7200.0})]
