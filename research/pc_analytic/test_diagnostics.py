"""Numerical and failure-semantics checks for the same-input diagnostics."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from research.pc_analytic.diagnostics import pair_summary, refinement_plan


def test_plan_varies_one_numerical_axis_at_a_time() -> None:
    """Only the named refinement axis changes in each adjacent pair."""
    cases, pairs = refinement_plan((24, 96, 384), (16, 32, 64))
    assert len(cases) == 5
    assert cases["d96_e32"] == {"directions": 96, "exposure_steps": 32}
    assert [row["axis"] for row in pairs] == ["directions"] * 2 + ["exposure_steps"] * 2
    for pair in pairs:
        coarse, fine = cases[pair["coarse"]], cases[pair["fine"]]
        other = "exposure_steps" if pair["axis"] == "directions" else "directions"
        assert coarse[other] == fine[other]
        assert coarse[pair["axis"]] < fine[pair["axis"]]


@pytest.mark.parametrize(
    ("directions", "doses"),
    [
        ((24, 24, 96), (16, 32, 64)),
        ((24, 96), (16, 32, 64)),
        ((24, 96, 384), (32, 16, 64)),
        ((24, 96, 384), (0, 16, 32)),
        ((24, 96, 384), (True, 16, 32)),
        ((6, 12, 24), (16, 32, 64)),
    ],
)
def test_invalid_refinement_plan(
    directions: tuple[int, ...], doses: tuple[int, ...]
) -> None:
    """Malformed axes fail before native work starts."""
    with pytest.raises(ValueError, match=r"Refinement|Unsupported"):
        refinement_plan(directions, doses)


def test_constant_neutral_fields_keep_power_ratio_undefined() -> None:
    """Vanishing reference fluctuations must not appear as zero relative error."""
    values = np.full((8, 8, 8), 0.5)
    result = pair_summary(values, values, np.ones_like(values), 50.0, redshift=8.0)
    assert result["field"]["MAE"] == 0.0
    assert result["power_below_half_nyquist"]["valid_bins"] == 0
    assert (
        result["power_below_half_nyquist"]["maximum_absolute_ratio_minus_one"] is None
    )


def test_spectrum_ratio_orientation_and_phase_information() -> None:
    """A doubled amplitude has fourfold power; reversed phase has negative r."""
    coordinate = np.arange(16)[:, None, None] / 16
    reference = np.broadcast_to(
        0.5 + 0.1 * np.cos(2 * np.pi * coordinate), (16, 16, 16)
    )
    candidate = 0.5 + 2 * (reference - 0.5)
    result = pair_summary(
        candidate, reference, np.ones_like(reference), 50.0, redshift=8.0
    )
    defined = [x for x in result["spectra_21cm"]["ratio_candidate"] if x is not None]
    np.testing.assert_allclose(defined, 4.0, rtol=1e-12, atol=1e-12)
    assert result["power_below_half_nyquist"][
        "maximum_absolute_ratio_minus_one"
    ] == pytest.approx(3.0)
    anti = pair_summary(
        1 - reference, reference, np.ones_like(reference), 50.0, redshift=8.0
    )
    defined_r = [x for x in anti["spectra_21cm"]["r_candidate"] if x is not None]
    np.testing.assert_allclose(defined_r, -1.0, rtol=1e-12, atol=1e-12)


def test_real_transport_grid_diagnostic_with_uniform_reference_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise receipts and physical count preservation through a grid change."""
    import hashlib
    import json

    from research.pc_analytic.diagnostics import run
    from research.pc_analytic.test_compare import _snapshot
    from research.pc_script_reference import kernel

    def uniform_reference(
        source: np.ndarray, capacity: np.ndarray, **kwargs: Any
    ) -> tuple[np.ndarray, dict[str, Any]]:
        np.testing.assert_allclose(source, source.flat[0], rtol=0, atol=0)
        np.testing.assert_allclose(capacity, 1, rtol=0, atol=0)
        return source / capacity, {"passed": True, "fixture_only": True}

    monkeypatch.setattr(kernel, "run", uniform_reference)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    output = tmp_path / "output"
    report = run(
        _snapshot(snapshot),
        output,
        directions=(6, 24, 26),
        exposures=(1, 2, 3),
        coarse_dimension=2,
        mfp_rays=64,
    )
    assert report["complete"] is True
    grid = report["grid_sensitivity"]
    assert grid["coarse_dimension"] == 2
    assert grid["fine_dimension"] == 4
    assert grid["coarse_count_unit_over_fine_count_unit"] == 8.0
    assert grid["source_physical_total_relative_change"] == 0.0
    assert grid["capacity_physical_total_relative_change"] == 0.0
    assert grid["comparisons"]["script"]["field"]["MAE"] == 0.0
    assert set(grid["comparisons"]) == {
        "source_periodic_cube",
        "closed_periodic",
        "script",
    }
    for name, digest in json.loads((output / "receipt.json").read_text())[
        "files"
    ].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    with pytest.raises(ValueError, match="empty output"):
        run(snapshot, output, directions=(6, 24, 26), exposures=(1, 2, 3))


def test_archived_reference_requires_exact_inputs_and_preserves_unknown_ledger(
    tmp_path: Path,
) -> None:
    """An old C2-Ray field is accepted only with verified matching provenance."""
    import json

    from research.pc_analytic.diagnostics import _archived_c2ray
    from research.pc_analytic.test_grid_compare import _fixture, _rewrite

    archive = _fixture(tmp_path / "archive", 4)

    def add_physics(report: dict[str, Any]) -> None:
        report["common_physics"].update(
            initial_xHII=0.0,
            collisional_ionization=False,
            thermal_evolution=False,
            photoheating_feedback=False,
            sigma_cm2=1.0,
        )

    _rewrite(archive, add_physics)
    folder = archive / "seed222" / "z7p5"
    row = json.loads((folder / "metrics.json").read_text())
    with np.load(folder / "fields.npz") as data:
        source, capacity, expected = (
            np.array(data[name]) for name in ("source", "capacity", "c2ray_xHI")
        )
    neutral, record = _archived_c2ray(archive, source, capacity, row["input"], 1.0)
    np.testing.assert_array_equal(neutral, expected)
    assert record["rerun_in_current_diagnostics"] is False
    assert record["original_budget"]["residual"] is None
    with pytest.raises(ValueError, match="cross section"):
        _archived_c2ray(archive, source, capacity, row["input"], 2.0)
    with pytest.raises(ValueError, match="arrays"):
        _archived_c2ray(archive, source * 1.01, capacity, row["input"], 1.0)
    with (folder / "fields.npz").open("ab") as stream:
        stream.write(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        _archived_c2ray(archive, source, capacity, row["input"], 1.0)
