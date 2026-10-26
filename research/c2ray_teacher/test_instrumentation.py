"""Guard diagnostic insertions against unexpected upstream layouts."""

from pathlib import Path

import pytest

from research.c2ray_teacher.instrumentation import _insert, instrument_sources


def test_missing_or_ambiguous_anchor_stops_instrumentation() -> None:
    """An upstream change must not silently put counters in the wrong place."""
    for source in ("other\n", "anchor\nanchor\n"):
        with pytest.raises(ValueError, match="not unique"):
            _insert(source, "anchor\n", "diagnostic\n")


def test_source_patch_preserves_every_original_statement(tmp_path: Path) -> None:
    """Passive instrumentation only inserts lines and records exact hashes."""
    code = tmp_path / "code"
    (code / "files_for_3D").mkdir(parents=True)
    originals = {
        "radiation_photoionrates.f90": (
            "module radiation_photoionrates\n  implicit none\ncontains\n"
            "    ! Find the heating rates rates by looking up the values in\n"
            "end module radiation_photoionrates\n"
        ),
        "files_for_3D/evolve_point.F90": (
            "module evolve_point\n  implicit none\ncontains\n"
            "          ! Divide the photo-ionization rates by the appropriate neutral density\n"
            "          photon_loss_src_thread(tn)=photon_loss_src_thread(tn) + &\n"
            "               phi%photo_out*vol/vol_ph\nend module evolve_point\n"
        ),
        "files_for_3D/evolve_source.F90": (
            "module evolve_source\n  implicit none\ncontains\n"
            "    photon_loss_src=total_source_flux !-1.0 ! to pass the first while test\n"
            "       photon_loss_src_thread(:) = 0.0 ! reset photon_loss_src to zero\n"
            "    photon_loss(1)=photon_loss(1) + photon_loss_src\n"
            "end module evolve_source\n"
        ),
        "files_for_3D/evolve.F90": (
            "module evolve\n  implicit none\ncontains\n"
            "       call set_rates_to_zero\n"
            "       ! Do a global pass applying all the rates\n"
            "end module evolve\n"
        ),
    }
    for name, text in originals.items():
        (code / name).write_text(text)
    report = instrument_sources(code, tmp_path / "diagnostics.patch")
    assert report["enabled"] is True
    assert len(report["files"]) == 4
    for name, original in originals.items():
        result = (code / name).read_text().splitlines(keepends=True)
        preserved = [
            line
            for line in result
            if "diag_" not in line
            and "transport_diagnostics" not in line
            and line != "  use grid, only: vol\n"
        ]
        assert "".join(preserved) == original
    assert len(report["patch_sha256"]) == 64
    assert len(report["script_sha256"]) == 64
