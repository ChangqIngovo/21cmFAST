"""Opt-in checks of passive counters against the uninstrumented RT solver."""

import json
import os
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from research.c2ray_teacher.build import FIELD_NAMES, read_output


def _input(path: Path, n: int, *, transparent: bool = False) -> None:
    density = 1e-8 if transparent else 1e-3
    sources = np.array([[n // 3, n // 2, n // 2], [2 * n // 3, n // 2, n // 2]])
    with path.open("wb") as stream:
        stream.write(b"C2RTHHE1")
        stream.write(struct.pack("<4i", n, 2, len(sources), 1))
        np.array([9, 5e22, 100, 1e-6, 1e-6, 1e-6], dtype="<f8").tofile(stream)
        sources.astype("<i4").tofile(stream)
        np.full(n**3, density, dtype="<f8").tofile(stream)
        for rates in ([5e48, 5e48], [0, 0]):
            np.array([3e11, 9, 5e22], dtype="<f8").tofile(stream)
            np.full(n**3, density, dtype="<f8").tofile(stream)
            np.array(rates, dtype="<f8").tofile(stream)


def _run(executable: Path, binary: Path, path: Path, threads: int) -> dict:
    (path / "results").mkdir(parents=True)
    subprocess.run(
        [str(executable), str(binary), str(path / "output.bin")],
        cwd=path,
        env={**os.environ, "OMP_NUM_THREADS": str(threads)},
        check=True,
        timeout=120,
        capture_output=True,
    )
    return read_output(path / "output.bin")


def test_diagnostics_preserve_native_fields_and_count_only_final_pass(tmp_path: Path):
    """Keep all native fields identical while independently splitting RT events."""
    instrumented = os.environ.get("C2RAY_DIAGNOSTIC_EXECUTABLE")
    reference = os.environ.get("C2RAY_REFERENCE_EXECUTABLE")
    if not instrumented or not reference:
        pytest.skip("Set diagnostic and reference executable paths for native tests")
    instrumented, reference = Path(instrumented).resolve(), Path(reference).resolve()
    n = json.loads((instrumented.parent / "build_manifest.json").read_text())[
        "dimension"
    ]
    (tmp_path / "tables").symlink_to(
        instrumented.parent / "tables", target_is_directory=True
    )
    binary = tmp_path / "input.bin"
    _input(binary, n, transparent=n >= 32)
    original = _run(reference, binary, tmp_path / "reference", 1)
    serial_diagnostics = None
    for threads in (1, 4):
        case = tmp_path / f"diagnostics_{threads}"
        observed = _run(instrumented, binary, case, threads)
        # Reference OMP=1 and diagnostic OMP=1 are exactly identical; threads
        # may change summation order, so compare those with floating precision.
        for field in FIELD_NAMES:
            if threads == 1:
                np.testing.assert_array_equal(
                    observed["fields"][field], original["fields"][field]
                )
            else:
                np.testing.assert_allclose(
                    observed["fields"][field],
                    original["fields"][field],
                    rtol=1e-10,
                    atol=1e-30,
                )
        rows = [
            json.loads(line)
            for line in (case / "results/transport_diagnostics.jsonl")
            .read_text()
            .splitlines()
        ]
        if threads == 1:
            serial_diagnostics = rows
        assert len(rows) == 2
        for index, row in enumerate(rows):
            assert row["step"] == index + 1
            assert row["final_transport_iteration"] >= 2
            assert row["time_seconds"] == (index + 1) * 3e11
            assert row["dt_seconds"] == 3e11
            assert row["primary_absorptions"] == pytest.approx(
                sum(row["primary_absorptions_by_species"]), rel=1e-13
            )
            assert row["secondary_ionizations"] == pytest.approx(
                sum(row["secondary_ionizations_by_species"]), rel=1e-13, abs=1e-100
            )
            assert row["secondary_ionizations_by_species"][2] == 0
            assert (
                row["native_boundary_outflow_photons"]
                == observed["steps"][index]["native_photon_loss"]
            )
            if threads == 1:
                assert row["boundary_outflow_photons"] == pytest.approx(
                    row["native_boundary_outflow_photons"], rel=3e-12, abs=1e-100
                )
            else:
                assert row["boundary_outflow_photons"] == pytest.approx(
                    serial_diagnostics[index]["boundary_outflow_photons"],
                    rel=3e-12,
                    abs=1e-100,
                )
            events = row["primary_absorptions"] + row["secondary_ionizations"]
            assert row["transport_rate_product_ionizations"] == pytest.approx(
                events, rel=3e-12, abs=1e-100
            )
        assert rows[0]["recorded_cell_visits"] == 2 * n**3
        assert rows[0]["secondary_ionizations"] > 0
        # Dark interval has no transport visits or primary/secondary photons;
        # the native chemistry can still recombine and cool.
        assert rows[1]["recorded_cell_visits"] == 0
        assert rows[1]["primary_absorptions"] == 0
        assert rows[1]["secondary_ionizations"] == 0
    assert not (tmp_path / "reference/results/transport_diagnostics.jsonl").exists()
