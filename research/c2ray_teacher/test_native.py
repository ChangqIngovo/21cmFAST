"""Opt-in tests of the actual unchanged Fortran backend.

Set C2RAY_TEST_EXECUTABLE to a compiled serial/OpenMP backend. These exercise
radiation, chemistry and a subsequent zero-emission interval, not a mock solver.
"""

import os
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from research.c2ray_teacher.build import read_output


def test_native_thermal_dark_interval_and_threads(tmp_path: Path):
    """Evolve finite bounded states and reproduce them across OpenMP counts."""
    value = os.environ.get("C2RAY_TEST_EXECUTABLE")
    if not value:
        pytest.skip("Set C2RAY_TEST_EXECUTABLE for the actual native integration test")
    executable = Path(value).resolve()
    (tmp_path / "tables").symlink_to(
        executable.parent / "tables", target_is_directory=True
    )
    input_path = tmp_path / "input.bin"
    n = 16
    with input_path.open("wb") as stream:
        stream.write(b"C2RTHHE1")
        stream.write(struct.pack("<4i", n, 2, 2, 1))
        np.array([9, 5e22, 100, 1e-6, 1e-6, 1e-6], dtype="<f8").tofile(stream)
        np.array([[5, 8, 8], [11, 8, 8]], dtype="<i4").tofile(stream)
        np.full(n**3, 1e-3, dtype="<f8").tofile(stream)
        for rates in ([5e48, 5e48], [0, 0]):
            np.array([3e11, 9, 5e22], dtype="<f8").tofile(stream)
            np.full(n**3, 1e-3, dtype="<f8").tofile(stream)
            np.array(rates, dtype="<f8").tofile(stream)
    outputs = []
    for threads in (1, 4):
        work = tmp_path / f"threads_{threads}"
        (work / "results").mkdir(parents=True)
        subprocess.run(
            [str(executable), str(input_path), str(work / "output.bin")],
            cwd=work,
            env={**os.environ, "OMP_NUM_THREADS": str(threads)},
            check=True,
            timeout=120,
            capture_output=True,
        )
        output = read_output(work / "output.bin")
        for field in output["fields"].values():
            assert np.isfinite(field).all()
        ionized = output["fields"]["xHII"]
        assert np.all((ionized >= 0) & (ionized <= 1))
        assert ionized[0].mean() > 1e-6
        assert ionized[1].mean() < ionized[0].mean()
        assert output["steps"][1]["emitted_photons"] == 0
        outputs.append(output)
    for name in ("xHII", "xHeII", "xHeIII", "temperature_K"):
        np.testing.assert_allclose(
            outputs[0]["fields"][name],
            outputs[1]["fields"][name],
            rtol=1e-10,
            atol=1e-12,
        )
