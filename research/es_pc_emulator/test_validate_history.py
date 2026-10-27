"""Check that history-reset controls change only the retained time history."""

import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest

from research.es_pc_emulator.validate_history import (
    CONFIG,
    final_report,
    run,
    truncate_last_interval,
)
from research.pc_evolution.native_history import (
    ARRAY_KEYS,
    _write_fresh_history,
    array_hash,
    load_history,
)


def _history():
    size = 2
    result = {
        "redshifts": np.array([20.0, 14.0, 9.0, 7.9]),
        "times_seconds": np.array([0.0, 10.0, 50.0, 81.0]),
        "rates": np.arange(4 * size**3, dtype=np.float64).reshape(4, size, size, size),
        "capacity": 1.0
        + np.arange(size**3, dtype=np.float64).reshape(size, size, size),
        "native_xhi": np.linspace(1, 0.1, 4 * size**3, dtype=np.float32).reshape(
            4, size, size, size
        ),
        "metadata": {
            "schema_version": 1,
            "density_evolution": "frozen_at_reference_redshift",
            "rates_definition": "native_n_ion_times_H_over_t_STAR",
            "reference_redshift": 8.0,
            "initial_hires_density_sha256": "the-same-IC",
            "random_seed": 333,
            "redshift_start": 20.0,
            "redshift_end": 7.9,
            "intervals": 3,
            "time_interval_quadrature_errors_seconds": [1e-12, 2e-12, 3e-12],
        },
    }
    result["metadata"]["source_provenance"] = [
        {"redshift": float(z), "native_source_sha": str(i)}
        for i, z in enumerate(result["redshifts"])
    ]
    result["metadata"]["array_sha256"] = {
        key: array_hash(result[key]) for key in ARRAY_KEYS
    }
    return result


def test_reset_retains_exact_sources_capacity_interval_and_initial_conditions(tmp_path):
    """Restarting the clock preserves all final-segment physical inputs."""
    parent = _history()
    original = copy.deepcopy(parent)
    source = tmp_path / "parent.npz"
    _write_fresh_history(source, parent)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    reset = truncate_last_interval(parent, digest)
    np.testing.assert_array_equal(reset["times_seconds"], [0.0, 31.0])
    np.testing.assert_array_equal(reset["capacity"], parent["capacity"])
    for name in ("rates", "redshifts", "native_xhi"):
        np.testing.assert_array_equal(reset[name], parent[name][-2:])
    old_emission = 0.5 * (parent["rates"][-2] + parent["rates"][-1]) * 31.0
    new_emission = 0.5 * reset["rates"].sum(axis=0) * np.diff(reset["times_seconds"])[0]
    np.testing.assert_array_equal(old_emission, new_emission)
    meta = reset["metadata"]
    assert meta["controlled_reset_parent_history_sha256"] == digest
    assert meta["controlled_reset_parent_endpoint_indices"] == [2, 3]
    assert meta["controlled_reset_parent_time_origin_seconds"] == 50.0
    assert meta["source_provenance"] == original["metadata"]["source_provenance"][-2:]
    assert (
        meta["initial_hires_density_sha256"]
        == original["metadata"]["initial_hires_density_sha256"]
    )
    assert meta["random_seed"] == 333
    assert meta["intervals"] == 1
    assert meta["redshift_start"] == 9.0
    assert meta["redshift_end"] == 7.9
    assert meta["time_interval_quadrature_errors_seconds"] == [3e-12]
    target = tmp_path / "reset.npz"
    _write_fresh_history(target, reset)
    loaded = load_history(target)
    for name in ARRAY_KEYS:
        np.testing.assert_array_equal(loaded[name], reset[name])
        np.testing.assert_array_equal(parent[name], original[name])
    assert parent["metadata"] == original["metadata"]
    reset["capacity"][0, 0, 0] = 99.0
    reset["metadata"]["source_provenance"][0]["native_source_sha"] = "changed"
    assert parent["capacity"][0, 0, 0] == 1.0
    assert parent["metadata"] == original["metadata"]


@pytest.mark.parametrize("bad_digest", ["", "not-a-hash", "g" * 64])
def test_reset_rejects_invalid_parent_identity(bad_digest):
    """Do not record an unverifiable parent identifier."""
    with pytest.raises(ValueError, match="SHA256"):
        truncate_last_interval(_history(), bad_digest)


@pytest.mark.parametrize("change", ["missing", "reordered"])
def test_reset_rejects_misaligned_source_provenance(change):
    """The retained source records must describe the retained numeric endpoints."""
    history = _history()
    if change == "missing":
        history["metadata"]["source_provenance"].pop()
    else:
        history["metadata"]["source_provenance"].reverse()
    with pytest.raises(ValueError, match="provenance"):
        truncate_last_interval(history, "a" * 64)


def test_unknown_phase_and_nonwarm_resume_fail_before_io(tmp_path):
    """Invalid orchestration cannot leave misleading phase outputs."""
    with pytest.raises(ValueError, match="phase"):
        run("other", tmp_path / "new", Path("upstream"))
    with pytest.raises(ValueError, match="warm"):
        run("reset", tmp_path / "new", Path("upstream"), resume=True)
    assert not (tmp_path / "new").exists()


def test_report_reads_pair_manifest_schema(tmp_path):
    """Report the final saved endpoint using real pair payloads and hashes."""
    from research.es_pc_emulator.test_pairs import export, pair_fixture

    fixture = pair_fixture(tmp_path / "fixture", seed=333, split="test")
    export(fixture)
    report = final_report(fixture["output"] / "manifest.json")
    assert report["sample"]["seed"] == 333
    assert report["sample"]["redshift"] == 8.0
    assert np.isfinite(report["metrics_es_vs_rt"]["MAE"])
    assert 0 <= report["rt_Q_mass"] <= 1
    assert CONFIG["teacher"]["z_start"] == 20.0


def test_late_dataset_fixes_native_threads_and_keeps_independent_splits(
    tmp_path, monkeypatch
):
    """RT worker changes cannot silently change the old-smoke native IC protocol."""
    from research.es_pc_emulator import validate_history as module

    native_calls, transport_calls, exports, merged = [], [], [], []
    monkeypatch.setattr(module, "_provenance", lambda *args: {})

    def fake_native(**kwargs):
        native_calls.append(kwargs)

    def fake_teacher(config, output, upstream, workers):
        transport_calls.append((config, workers))

    def fake_export(history, es, teacher, output, split, **kwargs):
        exports.append((split, kwargs))

    def fake_load(path):
        seed = int(path.parent.name.removeprefix("seed"))
        return {
            "metadata": {
                "native_threads": 1,
                "initial_hires_density_sha256": f"IC-{seed}",
            }
        }

    def fake_combine(manifests, output):
        merged.append((manifests, output))

    monkeypatch.setattr(module, "build_native_history", fake_native)
    monkeypatch.setattr(module, "run_teacher", fake_teacher)
    monkeypatch.setattr(module, "export_pairs", fake_export)
    monkeypatch.setattr(module, "load_history", fake_load)
    monkeypatch.setattr(module, "combine_manifests", fake_combine)
    result = module.run("late-dataset", tmp_path, tmp_path / "upstream", workers=8)
    assert [(row["seed"], row["threads"]) for row in native_calls] == [
        (111, 1),
        (222, 1),
        (333, 1),
    ]
    assert all(
        row["z_start"] == 8.0 and row["z_end"] == 7.9 and row["intervals"] == 1
        for row in native_calls
    )
    assert [workers for _, workers in transport_calls] == [8, 8, 8]
    assert [split for split, _ in exports] == ["train", "validation", "test"]
    assert [options["expected_seed"] for _, options in exports] == [111, 222, 333]
    assert all(
        options["initial_state_config"]["allow_mismatch"] for _, options in exports
    )
    assert len(merged[0][0]) == 3
    assert result["manifest"] == str(tmp_path / "late_dataset_manifest.json")
    assert result["native_threads"] == 1
