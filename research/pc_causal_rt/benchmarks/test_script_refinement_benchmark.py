"""Check the shared-grid diagnostics of the SCRIPT refinement benchmark."""

import numpy as np
import pytest

from research.pc_causal_rt.benchmarks import script_refinement_benchmark as module
from research.pc_causal_rt.benchmarks.script_refinement_benchmark import (
    _field_comparison,
    _pair_metrics,
    _restrict_ionization,
)


def test_restrict_counts_before_ion_fraction():
    """Preserve atoms instead of averaging local neutral fractions."""
    capacity = np.ones((4, 4, 4))
    capacity[:, :, 1::2] = 3.0
    fraction = np.zeros_like(capacity)
    fraction[:, :, 1::2] = 1.0
    atoms, ions, xhi = _restrict_ionization(capacity, fraction, 2)
    np.testing.assert_allclose(atoms, 2.0, rtol=0, atol=1e-15)
    np.testing.assert_allclose(ions, 1.5, rtol=0, atol=1e-15)
    np.testing.assert_allclose(xhi, 0.25, rtol=0, atol=1e-15)
    _, _, empty = _restrict_ionization(np.zeros_like(capacity), fraction, 2)
    np.testing.assert_array_equal(empty, np.ones((2, 2, 2)))


def test_spectrum_denominator_and_unavailable_stage():
    """Use finer-reference power and report missing stages without extrapolation."""
    wave = np.broadcast_to(np.arange(4.0)[:, None, None], (4, 4, 4)).copy()
    fine = {"xhi": np.full_like(wave, 0.9), "brightness": wave}
    coarse = {"xhi": np.full_like(wave, 0.9), "brightness": 2 * wave}
    comparison = _field_comparison(coarse, fine, 50.0)
    np.testing.assert_allclose(
        comparison["power_fractional_error"][0], 3.0, rtol=0, atol=1e-14
    )
    row = {"node": 0, "redshift": 9.0, "xHI_mass": 0.9}
    pair = _pair_metrics([(row, coarse)], [(row, fine)], 50.0)
    assert len(pair["same_time_active"]) == 1
    assert all(not m["available"] for m in pair["matched_mean_xHI"])


def test_parallel_allocations_preserve_order_and_worker_bound(monkeypatch, tmp_path):
    """Dispatch independent allocations concurrently without reordering fields."""
    import threading
    import time

    lock = threading.Lock()
    counts = {"active": 0, "maximum": 0}

    def fake_run(source, capacity, *, build_dir, timeout_seconds):
        with lock:
            counts["active"] += 1
            counts["maximum"] = max(counts["active"], counts["maximum"])
        time.sleep(0.02)
        with lock:
            counts["active"] -= 1
        return source / capacity, {"passed": True}

    monkeypatch.setattr(module, "run", fake_run)
    tasks = [
        {"source": np.full((2, 2, 2), v), "capacity": np.ones((2, 2, 2))}
        for v in (0.1, 0.2, 0.3, 0.4)
    ]
    serial = module._run_allocation_batch(tasks, tmp_path, 1.0, 1)
    parallel = module._run_allocation_batch(tasks, tmp_path, 1.0, 2)
    assert counts["maximum"] == 2
    assert counts["active"] == 0
    for a, b in zip(serial, parallel, strict=True):
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1] == b[1]


def test_failed_parallel_allocation_does_not_return_success(monkeypatch, tmp_path):
    """Propagate a worker failure instead of returning incomplete results."""

    def fail(*args, **kwargs):
        raise RuntimeError("allocation failed")

    monkeypatch.setattr(module, "run", fail)
    task = {"source": np.ones((2, 2, 2)), "capacity": np.ones((2, 2, 2))}
    with pytest.raises(RuntimeError, match="allocation failed"):
        module._run_allocation_batch([task, task], tmp_path, 1.0, 2)


def test_invalid_workers_and_existing_output_rejected(tmp_path):
    """Reject invalid resource settings and never reuse an existing output."""
    with pytest.raises(ValueError, match="positive integer"):
        module.benchmark(tmp_path / "input", tmp_path / "new", 222, (1, 2), workers=0)
    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "preserved.txt"
    marker.write_text("untouched")
    with pytest.raises(ValueError, match="new output directory"):
        module.benchmark(tmp_path / "input", existing, 222, (1, 2), workers=2)
    assert marker.read_text() == "untouched"
