"""Check opt-in radius export with a genuine rebuilt native 21cmFAST runtime."""

from pathlib import Path

import numpy as np
import pytest


def test_native_radius_diagnostic_preserves_ionization_and_source_fields(tmp_path):
    """The diagnostic cannot change physics and must record a real full crossing."""
    pytest.importorskip("py21cmfast")
    from py21cmfast import run_coeval
    from py21cmfast.io.caching import OutputCache
    from research.pc_evolution.native_history import require_es_radius_capability
    from research.pc_reconsidered.native_driver import prepare_native

    require_es_radius_capability()
    inputs, _ = prepare_native(
        dimension=8, highres_dimension=24, box_len=24.0, threads=1, random_seed=1234
    )
    assert inputs.matter_options.STORE_ES_RADIUS is False
    diagnostic = inputs.evolve_input_structs(STORE_ES_RADIUS=True)
    original = run_coeval(
        inputs=inputs,
        out_redshifts=5.0,
        regenerate=True,
        write=False,
        cache=OutputCache(Path(tmp_path) / "control"),
    )[0]
    measured = run_coeval(
        inputs=diagnostic,
        out_redshifts=5.0,
        regenerate=True,
        write=False,
        cache=OutputCache(Path(tmp_path) / "diagnostic"),
    )[0]
    for box_name, field in (
        ("ionized_box", "neutral_fraction"),
        ("emissivity_fields", "n_ion"),
        ("perturbed_field", "density"),
    ):
        np.testing.assert_array_equal(
            getattr(original, box_name).get(field),
            getattr(measured, box_name).get(field),
        )
    radius = measured.ionized_box.get("es_crossing_radius")
    neutral = measured.ionized_box.get("neutral_fraction")
    assert radius.shape == (8, 8, 8)
    assert np.isfinite(radius).all()
    assert np.any(radius > 0), "Fixture must exercise an actual ES barrier crossing"
    assert np.all(radius <= diagnostic.astro_params.R_BUBBLE_MAX)
    assert np.all(neutral[radius > 0] == 0)
    assert np.all(radius[neutral > 0] == 0)


def test_native_history_exports_radius_with_separate_hashed_sidecar(tmp_path):
    """Generation keeps the legacy history schema and one shared IC identity."""
    pytest.importorskip("py21cmfast")
    from research.pc_evolution.native_history import (
        ARRAY_KEYS,
        array_hash,
        build_native_history,
    )

    es_path, history_path = tmp_path / "native_es.npz", tmp_path / "native_history.npz"
    result = build_native_history(
        output=history_path,
        es_output=es_path,
        dimension=8,
        highres_dimension=24,
        box_len=24.0,
        threads=1,
        seed=1234,
        z_start=7.0,
        z_end=5.0,
        intervals=2,
        reference_z=6.0,
    )
    with np.load(history_path, allow_pickle=False) as history:
        assert set(history.files) == set(ARRAY_KEYS) | {"metadata"}
    with np.load(es_path, allow_pickle=False) as es:
        import json

        meta = json.loads(str(es["metadata"].item()))
        assert meta["history_array_sha256"] == result["metadata"]["array_sha256"]
        assert (
            meta["initial_hires_density_sha256"]
            == result["metadata"]["initial_hires_density_sha256"]
        )
        assert meta["radius_units"] == "cMpc"
        assert array_hash(es["xhi_es"]) == array_hash(result["native_xhi"])
        assert np.any(es["radius_es_cMpc"] > 0)
        assert np.all(es["radius_es_cMpc"][es["xhi_es"] > 0] == 0)
