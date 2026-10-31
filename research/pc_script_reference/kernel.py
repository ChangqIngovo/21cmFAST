"""Run the unmodified public SCRIPT photon allocator in a timed subprocess.

The author kernel and periodic shell ordering are vendored byte-for-byte. This
adapter provides input/output, build provenance, and an external photon ledger.
It does not change the original single-precision arithmetic or clip its field.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np


VENDOR = Path(__file__).with_name("vendor")
PHOTON_TOLERANCE = 2e-5
_FLAGS = ("-O2", "-ffree-line-length-none")
_DRIVER = """program script_reference_driver
  use sort_grid_points, only: sort_grid
  use pc, only: ionized_regions_photon_conserving
  implicit none
  integer :: n, total, unit, ierr
  integer, allocatable :: ix(:), iy(:), iz(:), degeneracy(:), ordering(:)
  real, allocatable :: photons(:,:,:), atoms(:,:,:), qi(:,:,:)
  character(len=4096) :: input_path, output_path
  if (storage_size(n) /= 32 .or. storage_size(1.0) /= 32) error stop 'ABI requires REAL/INTEGER 32'
  if (command_argument_count() /= 2) error stop 'expected input and output paths'
  call get_command_argument(1, input_path)
  call get_command_argument(2, output_path)
  open(newunit=unit, file=trim(input_path), access='stream', form='unformatted', &
       status='old', action='read', iostat=ierr)
  if (ierr /= 0) error stop 'cannot open input'
  read(unit, iostat=ierr) n
  if (ierr /= 0 .or. n < 1 .or. n > 1024) error stop 'invalid dimension'
  total = n**3
  allocate(photons(n,n,n), atoms(n,n,n), qi(n,n,n))
  allocate(ix(total), iy(total), iz(total), degeneracy(total), ordering(total))
  read(unit, iostat=ierr) photons, atoms
  if (ierr /= 0) error stop 'invalid input arrays'
  close(unit)
  call sort_grid(n, ix, iy, iz, degeneracy, ordering, read_write_file=.false.)
  call ionized_regions_photon_conserving(photons, atoms, qi, ix, iy, iz, degeneracy, ordering)
  open(newunit=unit, file=trim(output_path), access='stream', form='unformatted', &
       status='replace', action='write', iostat=ierr)
  if (ierr /= 0) error stop 'cannot open output'
  write(unit) qi
  close(unit)
end program script_reference_driver
"""


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance() -> dict:
    """Return the pinned upstream identity after verifying every vendored file."""
    record = json.loads((VENDOR / "provenance.json").read_text())
    for filename, expected in record["files"].items():
        if _digest(VENDOR / filename) != expected["sha256"]:
            raise ValueError(f"SCRIPT upstream checksum mismatch: {filename}")
    return record


def build(build_dir: str | Path) -> tuple[Path, dict]:
    """Compile the original sources; cached executables are hash-verified."""
    upstream = provenance()
    requested = os.environ.get("SCRIPT_FC", "gfortran")
    found = shutil.which(requested)
    if found is None:
        raise RuntimeError(f"Fortran compiler unavailable: {requested!r}; set SCRIPT_FC")
    compiler = str(Path(found).resolve())
    version = subprocess.run(
        [compiler, "--version"], capture_output=True, text=True, check=True, timeout=20
    ).stdout
    identity = {
        "upstream": upstream,
        "compiler": compiler,
        "compiler_sha256": _digest(Path(compiler)),
        "compiler_version": version,
        "flags": list(_FLAGS),
        "driver_sha256": hashlib.sha256(_DRIVER.encode()).hexdigest(),
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root = Path(build_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    executable = root / f"script-{key}"
    receipt = root / f"script-{key}.json"
    if executable.is_file() and receipt.is_file():
        previous = json.loads(receipt.read_text())
        if previous.get("identity") != identity or previous.get("executable_sha256") != _digest(executable):
            raise ValueError("Cached SCRIPT executable checksum or build identity changed")
        return executable, previous
    with tempfile.TemporaryDirectory(prefix="build-", dir=root) as temporary:
        temporary = Path(temporary)
        driver = temporary / "driver.f90"
        driver.write_text(_DRIVER)
        binary = temporary / "script"
        completed = subprocess.run(
            [compiler, *_FLAGS, str(VENDOR / "sort_grid_points.f90"),
             str(VENDOR / "pc.f90"), str(driver), "-o", str(binary)],
            cwd=temporary, capture_output=True, text=True, timeout=120,
        )
        if completed.returncode:
            raise RuntimeError(f"SCRIPT compilation failed:\n{completed.stderr}")
        record = {"identity": identity, "executable_sha256": _digest(binary),
                  "compiler_stdout": completed.stdout, "compiler_stderr": completed.stderr}
        temporary_receipt = temporary / "receipt.json"
        temporary_receipt.write_text(json.dumps(record, indent=2) + "\n")
        os.replace(binary, executable)
        os.replace(temporary_receipt, receipt)
    return executable, record


def _ledger(source: np.ndarray, capacity: np.ndarray, qi: np.ndarray) -> dict:
    emitted = float(np.sum(source, dtype=np.float64))
    atoms = float(np.sum(capacity, dtype=np.float64))
    absorbed = float(np.sum(qi * capacity, dtype=np.float64))
    # The upstream full-saturation shortcut does not return excess photons.
    # Account for that GLOBAL surplus explicitly without inventing its location.
    surplus = max(emitted - atoms, 0.0)
    residual = emitted - absorbed - surplus
    scale = max(emitted, absorbed + surplus, np.finfo(np.float64).tiny)
    return {"source_photons": emitted, "atom_capacity": atoms,
            "ionized_atoms": absorbed, "surplus_photons": surplus,
            "residual": residual, "relative_residual": abs(residual) / scale}


def run(
    source_counts: np.ndarray,
    capacity_counts: np.ndarray,
    *,
    build_dir: str | Path | None = None,
    timeout_seconds: float = 120,
) -> tuple[np.ndarray, dict]:
    """Return ``(qi, ledger)`` for coeval photon and atom counts in shared units.

    ``qi`` is the original REAL32 result converted to float64 without clipping.
    Counts must be finite; capacity must be strictly positive. Both the original
    input ledger and the actually executed REAL32 ledger are reported. The
    measured residual is never repaired, and an unsuccessful audit has
    ``ledger['passed'] == False``. A timeout raises without returning a field.
    """
    source = np.asarray(source_counts, dtype=np.float64)
    capacity = np.asarray(capacity_counts, dtype=np.float64)
    if source.ndim != 3 or len(set(source.shape)) != 1 or source.shape[0] < 1:
        raise ValueError("SCRIPT requires a nonempty cubic source grid")
    if source.shape != capacity.shape:
        raise ValueError("Source and capacity shapes differ")
    if source.shape[0] > 1024:
        raise ValueError("SCRIPT driver supports at most 1024 cells per side")
    if not np.isfinite(source).all() or np.any(source < 0):
        raise ValueError("Photon counts must be finite and nonnegative")
    if not np.isfinite(capacity).all() or np.any(capacity <= 0):
        raise ValueError("Atom capacities must be finite and strictly positive")
    if not np.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    with np.errstate(over="ignore", under="ignore"):
        source32 = source.astype("=f4")
        capacity32 = capacity.astype("=f4")
    if not np.isfinite(source32).all() or not np.isfinite(capacity32).all() or np.any(capacity32 <= 0):
        raise ValueError("Inputs cannot be represented in upstream REAL32; use common count units")
    if np.any((source > 0) & (source32 == 0)):
        raise ValueError("Positive photons underflow upstream REAL32; use common count units")
    with tempfile.TemporaryDirectory(prefix="script-reference-") as temporary:
        temporary = Path(temporary)
        executable, build_record = build(build_dir if build_dir is not None else temporary / "build")
        input_path, output_path = temporary / "input.bin", temporary / "output.bin"
        with input_path.open("wb") as stream:
            stream.write(np.asarray([source.shape[0]], dtype="=i4").tobytes())
            stream.write(source32.tobytes(order="F"))
            stream.write(capacity32.tobytes(order="F"))
        started = time.perf_counter()
        completed = subprocess.run(
            [str(executable), str(input_path), str(output_path)], cwd=temporary,
            capture_output=True, text=True, timeout=timeout_seconds,
        )
        elapsed = time.perf_counter() - started
        if completed.returncode:
            raise RuntimeError(f"Original SCRIPT failed ({completed.returncode}): {completed.stderr}")
        if not output_path.is_file() or output_path.stat().st_size != source.size * 4:
            raise RuntimeError("Original SCRIPT returned a missing or incomplete field")
        qi = np.fromfile(output_path, dtype="=f4").reshape(source.shape, order="F").astype(np.float64)
    original = _ledger(source, capacity, qi)
    executed = _ledger(source32.astype(np.float64), capacity32.astype(np.float64), qi)
    minimum, maximum = float(np.min(qi)), float(np.max(qi))
    bounded = bool(np.isfinite(qi).all() and minimum >= 0.0 and maximum <= 1.0)
    passed = bounded and original["relative_residual"] <= PHOTON_TOLERANCE and executed["relative_residual"] <= PHOTON_TOLERANCE
    original.update({
        "passed": bool(passed), "tolerance": PHOTON_TOLERANCE,
        "kernel_float32": executed,
        "source_cast_L1_relative_error": float(np.sum(np.abs(source32.astype(np.float64) - source))) / max(original["source_photons"], np.finfo(float).tiny),
        "capacity_cast_L1_relative_error": float(np.sum(np.abs(capacity32.astype(np.float64) - capacity))) / original["atom_capacity"],
        "minimum_qi": minimum, "maximum_qi": maximum, "field_bounds_passed": bounded,
        "elapsed_seconds": elapsed, "upstream_commit": build_record["identity"]["upstream"]["commit"],
        "executable_sha256": build_record["executable_sha256"],
        "build_identity": build_record["identity"],
        "surplus_scope": "global count only; no spatial or directional surplus field is supplied",
    })
    return qi, original
