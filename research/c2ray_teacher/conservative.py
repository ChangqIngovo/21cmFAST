"""Optional transport correction with continuous rays and geometric sky weights.

This changes the transport discretization, not the H/He chemistry or thermal
solver. Native short characteristics remain the default reference. No measured
photon deficit enters the angular weights or source luminosities.
"""

from __future__ import annotations

import difflib
import hashlib
import re
from pathlib import Path

import numpy as np


def angular_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Return cube-face midpoint directions and exact pixel solid angles.

    Parameters
    ----------
    order : int
        Number of pixels along each side of each of six cube faces.
    """
    if type(order) is not int or order < 1:
        raise ValueError("Angular order must be a positive integer")
    directions, weights = [], []

    def primitive(a: float, b: float) -> float:
        return np.arctan2(a * b, np.sqrt(1 + a * a + b * b))

    for axis in range(3):
        others = [i for i in range(3) if i != axis]
        for sign in (-1, 1):
            for i in range(order):
                a0, a1 = -1 + 2 * i / order, -1 + 2 * (i + 1) / order
                for j in range(order):
                    b0, b1 = -1 + 2 * j / order, -1 + 2 * (j + 1) / order
                    d = np.zeros(3)
                    d[axis], d[others[0]], d[others[1]] = (
                        sign,
                        (a0 + a1) / 2,
                        (b0 + b1) / 2,
                    )
                    directions.append(d / np.linalg.norm(d))
                    weights.append(
                        (
                            primitive(a1, b1)
                            - primitive(a0, b1)
                            - primitive(a1, b0)
                            + primitive(a0, b0)
                        )
                        / (4 * np.pi)
                    )
    return np.asarray(directions), np.asarray(weights)


def patch_transport(code: Path, destination: Path) -> dict:
    """Apply guarded transport and stable-average changes to a verified copy.

    Parameters
    ----------
    code : Path
        Copied upstream code directory, after optional passive instrumentation.
    destination : Path
        File recording the complete numerical patch for audit.
    """
    changes = {
        "doric.f90": [
            (
                "  implicit none\n",
                "  use stable_time_average, only: average_exponential\n  implicit none\n",
            )
        ],
        "files_for_3D/evolve_source.F90": [
            (
                "  implicit none\n",
                "  use conservative_transport, only: transport_source\n  implicit none\n",
            ),
            (
                "    ! Pick up source number from the source list\n",
                (
                    "    call transport_source(ns1)\n    sum_nbox=sum_nbox+1\n    return\n\n"
                    "    ! Pick up source number from the source list\n"
                ),
            ),
        ],
        "radiation_photoionrates.f90": [
            (
                (
                    "       if (abs(tau_out_all(i_subband)-tau_in_all(i_subband)) > &\n"
                    "            tau_photo_limit) then\n"
                ),
                (
                    "       ! Continuous rays share exact table endpoints between cells.\n"
                    "       ! Avoid mixing a thin-cell derivative with a table difference.\n"
                    "       if (.true.) then\n"
                ),
            ),
        ],
    }
    patches, files = [], {}
    for name, replacements in changes.items():
        path = code / name
        original = path.read_text()
        updated = original
        for before, after in replacements:
            if updated.count(before) != 1:
                raise ValueError(f"Transport patch anchor is not unique: {name}")
            updated = updated.replace(before, after, 1)
        if name == "doric.f90":
            for i in (1, 2, 3):
                pattern = rf"    if \(abs\(lam{i}dt\)\.lt\.1\.0e-8\) then.*?    endif"
                updated, count = re.subn(
                    pattern,
                    f"    avg_factor_{i}=coef{i}*average_exponential(lam{i}dt)",
                    updated,
                    flags=re.DOTALL,
                )
                if count != 1:
                    raise ValueError("Time-average patch anchor is not unique")
        path.write_text(updated)
        patches.extend(
            difflib.unified_diff(
                original.splitlines(True),
                updated.splitlines(True),
                fromfile=f"a/code/{name}",
                tofile=f"b/code/{name}",
            )
        )
        files[f"code/{name}"] = {
            "before_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "after_sha256": hashlib.sha256(updated.encode()).hexdigest(),
        }
    destination.write_text("".join(patches))
    return {
        "files": files,
        "patch_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "scope": "Continuous cube-face rays with exact solid angles and shared spectral-table endpoints; native H/He equations retained with stable evaluation of analytic eigenmode time averages",
    }
