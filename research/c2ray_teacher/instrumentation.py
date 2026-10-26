"""Insert passive counters into verified build copies of the pinned kernel.

Every original byte remains in order. The inserted calls only read native
values and write adapter-owned diagnostics; they do not change solver states.
"""

from __future__ import annotations

import difflib
import hashlib
from pathlib import Path


def _insert(source: str, anchor: str, addition: str, *, after: bool = False) -> str:
    if source.count(anchor) != 1:
        raise ValueError(f"Native diagnostic anchor is not unique: {anchor!r}")
    replacement = anchor + addition if after else addition + anchor
    return source.replace(anchor, replacement, 1)


def instrument_sources(code: Path, patch_path: Path) -> dict:
    """Instrument a fresh verified build copy and record its exact source patch.

    Parameters
    ----------
    code : Path
        Copied upstream ``code`` directory, never the original checkout.
    patch_path : Path
        Fresh output path for the complete diagnostic-only unified diff.
    """
    edits = {
        "radiation_photoionrates.f90": [
            (
                "  implicit none\n",
                "  use transport_diagnostics, only: diag_capture_primary\n",
                False,
            ),
            (
                "    ! Find the heating rates rates by looking up the values in\n",
                "    call diag_capture_primary(phi%photo_cell_HI,phi%photo_cell_HeI,phi%photo_cell_HeII)\n",
                False,
            ),
        ],
        "files_for_3D/evolve_point.F90": [
            (
                "  implicit none\n",
                "  use transport_diagnostics, only: diag_record_cell,diag_record_boundary\n",
                False,
            ),
            (
                "          ! Divide the photo-ionization rates by the appropriate neutral density\n",
                "          call diag_record_cell(phi%photo_cell_HI,phi%photo_cell_HeI,phi%photo_cell_HeII,vol)\n",
                False,
            ),
            (
                "          photon_loss_src_thread(tn)=photon_loss_src_thread(tn) + &\n",
                "          call diag_record_boundary(phi%photo_out*vol/vol_ph)\n",
                False,
            ),
        ],
        "files_for_3D/evolve_source.F90": [
            (
                "  implicit none\n",
                "  use transport_diagnostics, only: diag_subbox_start,diag_source_complete\n",
                False,
            ),
            (
                "    photon_loss_src=total_source_flux !-1.0 ! to pass the first while test\n",
                "    call diag_subbox_start()\n",
                False,
            ),
            (
                "       photon_loss_src_thread(:) = 0.0 ! reset photon_loss_src to zero\n",
                "       call diag_subbox_start()\n",
                True,
            ),
            (
                "    photon_loss(1)=photon_loss(1) + photon_loss_src\n",
                "    call diag_source_complete()\n",
                True,
            ),
        ],
        "files_for_3D/evolve.F90": [
            (
                "  implicit none\n",
                "  use transport_diagnostics, only: diag_iteration_start,diag_transport_complete\n  use grid, only: vol\n",
                False,
            ),
            (
                "       call set_rates_to_zero\n",
                "       call diag_iteration_start(xh_av,xhe_av,niter)\n",
                True,
            ),
            (
                "       ! Do a global pass applying all the rates\n",
                "       call diag_transport_complete(ndens,phih_grid,phihe_grid,vol)\n",
                False,
            ),
        ],
    }
    report, patches = {}, []
    for name, insertions in edits.items():
        path = code / name
        original = path.read_text()
        updated = original
        for anchor, addition, after in insertions:
            updated = _insert(updated, anchor, addition, after=after)
        # Prove the patch contains insertions only; no native line is deleted.
        diff = list(
            difflib.unified_diff(
                original.splitlines(True),
                updated.splitlines(True),
                fromfile=f"a/code/{name}",
                tofile=f"b/code/{name}",
            )
        )
        if any(line.startswith("-") and not line.startswith("---") for line in diff):
            raise RuntimeError(
                "Diagnostics must not delete or replace native statements"
            )
        path.write_text(updated)
        patches.extend(diff)
        report[f"code/{name}"] = {
            "pristine_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "instrumented_sha256": hashlib.sha256(updated.encode()).hexdigest(),
            "insertions": len(insertions),
        }
    patch_path.write_text("".join(patches))
    return {
        "enabled": True,
        "source_edits": "insertions only; passive counters without numerical feedback",
        "files": report,
        "patch_sha256": hashlib.sha256(patch_path.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
