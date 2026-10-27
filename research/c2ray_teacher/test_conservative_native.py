"""Opt-in photon, translation and source-partition tests of native ray builds."""

import json
import os
import struct
import subprocess
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from research.c2ray_teacher.build import FIELD_NAMES, read_output
from research.c2ray_teacher.runner import (
    _native_initialization,
    compare_parallel_outputs,
    read_transport_diagnostics,
)


@pytest.fixture
def executable(tmp_path: Path) -> Path:
    """Require an explicit corrected binary and link its verified tables."""
    value = os.environ.get("C2RAY_CONSERVATIVE_EXECUTABLE")
    if not value:
        pytest.skip("Set C2RAY_CONSERVATIVE_EXECUTABLE for native transport tests")
    path = Path(value).resolve()
    manifest = json.loads((path.parent / "build_manifest.json").read_text())
    assert manifest["transport"] == "conservative_rays"
    assert manifest["dimension"] == 16
    assert manifest["numerical_algorithm_unchanged"] is False
    (tmp_path / "tables").symlink_to(path.parent / "tables", target_is_directory=True)
    return path


def run_case(
    executable: Path,
    root: Path,
    *,
    density: float = 1e-3,
    positions: np.ndarray | None = None,
    rates: list[float] | None = None,
    threads: int = 4,
    physical_controls: tuple[float, float] | None = None,
) -> dict:
    """Execute two intervals, including a dark chemistry-only interval."""
    positions = np.array([[5, 8, 8], [10, 8, 8]]) if positions is None else positions
    rates = [5e48, 5e48] if rates is None else rates
    root.mkdir()
    (root / "results").mkdir()
    binary = root / "input.bin"
    with binary.open("xb") as stream:
        stream.write(b"C2RTHHE1" if physical_controls is None else b"C2RTHHE2")
        stream.write(struct.pack("<4i", 16, 2, len(rates), 1))
        np.array([9, 5e22, 100, 1e-6, 1e-6, 1e-6], dtype="<f8").tofile(stream)
        if physical_controls is not None:
            np.array(physical_controls, dtype="<f8").tofile(stream)
        positions.astype("<i4").tofile(stream)
        np.full(16**3, density, dtype="<f8").tofile(stream)
        for source in (rates, np.zeros(len(rates))):
            np.array([3e11, 9, 5e22], dtype="<f8").tofile(stream)
            np.full(16**3, density, dtype="<f8").tofile(stream)
            np.asarray(source, dtype="<f8").tofile(stream)
    log = root / "native.log"
    with log.open("w") as stream:
        subprocess.run(
            [str(executable), str(binary), str(root / "output.bin")],
            cwd=root,
            env=os.environ | {"OMP_NUM_THREADS": str(threads)},
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=180,
        )
    state = read_output(root / "output.bin")
    init = _native_initialization(log, root / "results/native_initialization.json")
    if physical_controls is not None:
        assert init["blackbody_temperature_K"] == physical_controls[0]
        assert init["clumping_factor"] == pytest.approx(physical_controls[1], rel=1e-7)
    counters = read_transport_diagnostics(
        root / "results/transport_diagnostics.jsonl",
        state,
        init["sed_table_photons_over_nominal"],
        active_spectral_ratio=init["sed_transport_photons_over_nominal"],
    )
    assert (
        abs(counters["cumulative"]["photon_residual_over_transported_emitted"]) < 2e-11
    )
    assert counters["steps"][1]["primary_absorptions"] == 0
    for name, values in state["fields"].items():
        assert np.isfinite(values).all(), name
        if name.startswith("x"):
            assert values.min() >= 0
            assert values.max() <= 1
    return state


def test_runtime_spectrum_and_clumping_controls(executable: Path, tmp_path: Path):
    """One executable responds to SED hardness and recombination, not photon rate."""
    base = run_case(executable, tmp_path / "default")
    explicit = run_case(executable, tmp_path / "explicit", physical_controls=(50000., 1.))
    hard = run_case(executable, tmp_path / "hard", physical_controls=(80000., 1.))
    sink = run_case(executable, tmp_path / "sink", physical_controls=(50000., 8.))
    for key in FIELD_NAMES:
        np.testing.assert_array_equal(base["fields"][key], explicit["fields"][key])
    assert base["steps"] == explicit["steps"]
    for changed in (hard, sink):
        assert [s["emitted_photons"] for s in changed["steps"]] == [
            s["emitted_photons"] for s in base["steps"]]
    assert not np.array_equal(hard["fields"]["temperature_K"], base["fields"]["temperature_K"])
    assert sink["fields"]["xHII"][-1].mean() < base["fields"]["xHII"][-1].mean()


@pytest.mark.parametrize("density", [1e-3, 1e-8])
def test_photon_closure_and_threads(
    executable: Path, tmp_path: Path, density: float
) -> None:
    """Photon accounting and gas states are independent of thread count."""
    a = run_case(executable, tmp_path / "one", density=density, threads=1)
    b = run_case(executable, tmp_path / "four", density=density, threads=4)
    # Native chemistry subtracts nearly equal numbers at x~1e-6. Use the
    # same absolute fraction/global-relative tolerance as the MPI benchmark.
    assert compare_parallel_outputs(a, b)["passed"]


def test_periodic_translation(executable: Path, tmp_path: Path) -> None:
    """Moving sources through a periodic boundary translates the solution."""
    shift = np.array([3, 2, -4])
    a = run_case(executable, tmp_path / "base")
    b = run_case(
        executable,
        tmp_path / "translated",
        positions=(np.array([[5, 8, 8], [10, 8, 8]]) - 1 + shift) % 16 + 1,
    )
    shifted = a | {
        "fields": {
            name: np.roll(a["fields"][name], shift, axis=(1, 2, 3))
            for name in FIELD_NAMES
        }
    }
    assert compare_parallel_outputs(shifted, b)["passed"]


def test_source_partition_invariance(executable: Path, tmp_path: Path) -> None:
    """Splitting one emitter into coincident sources preserves its solution."""
    a = run_case(
        executable, tmp_path / "single", positions=np.array([[8, 8, 8]]), rates=[1e49]
    )
    b = run_case(
        executable,
        tmp_path / "split",
        positions=np.array([[8, 8, 8], [8, 8, 8]]),
        rates=[5e48, 5e48],
    )
    assert compare_parallel_outputs(a, b)["passed"]


def test_time_average_matches_high_precision(executable: Path, tmp_path: Path) -> None:
    """Check the compiled cancellation-free formula against decimal arithmetic."""
    build = executable.parent
    manifest = json.loads((build / "build_manifest.json").read_text())
    compiler = manifest["commands"][0][0]
    program = tmp_path / "average.f90"
    program.write_text(
        "program check_average\nuse precision,only:dp\n"
        "use stable_time_average,only:average_exponential\n"
        "implicit none\ninteger::i\nreal(dp)::x\n"
        "do i=1,11\nread(*,*) x\nwrite(*,'(ES25.16E3)') average_exponential(x)\nenddo\nend program\n"
    )
    binary = tmp_path / "average"
    subprocess.run(
        [
            compiler,
            "-I",
            str(build / "objects"),
            str(build / "stable_time_average.f90"),
            str(program),
            "-o",
            str(binary),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    numbers = [
        "-100",
        "-1",
        "-0.1",
        "-0.099",
        "-0.001",
        "-1e-8",
        "-1e-12",
        "0",
        "1e-12",
        "0.001",
        "0.099",
    ]
    observed = np.fromstring(
        subprocess.check_output(
            [str(binary)], input="\n".join(numbers) + "\n", text=True
        ),
        sep=" ",
    )
    with localcontext() as context:
        context.prec = 60
        expected = [
            float((Decimal(x).exp() - 1) / Decimal(x)) if Decimal(x) else 1
            for x in numbers
        ]
    np.testing.assert_allclose(observed, expected, rtol=5e-15, atol=0)


def test_initialization_sidecar_preserves_native_fields(
    executable: Path, tmp_path: Path
) -> None:
    """Compare old/new logging binaries on an identical native physical input."""
    reference = os.environ.get("C2RAY_REFERENCE_LOGGING_EXECUTABLE")
    if not reference:
        pytest.skip(
            "Set C2RAY_REFERENCE_LOGGING_EXECUTABLE for the logging-only comparison"
        )
    old = run_case(Path(reference), tmp_path / "old_logging", threads=1)
    new = run_case(executable, tmp_path / "new_logging", threads=1)
    for name in FIELD_NAMES:
        np.testing.assert_array_equal(old["fields"][name], new["fields"][name])
    assert old["steps"] == new["steps"]
    log = tmp_path / "new_logging/native.log"
    sidecar = tmp_path / "new_logging/results/native_initialization.json"
    reference_record = _native_initialization(log, sidecar)
    log.write_text("")
    assert _native_initialization(log, sidecar) == reference_record
