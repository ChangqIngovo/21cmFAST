"""Build a pinned C2-Ray H+He kernel with optional passive diagnostics.

Grid size, cosmology and abundance *configuration modules* are generated; the
nominal blackbody temperature declaration is exposed as runtime configuration.
Optional guarded insertions count primary and secondary events without feeding
back into native numerical variables. An uninstrumented reference build remains
available for equivalence tests.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

UPSTREAM_COMMIT = "a363656ff1c3b4b4c7334f398578d8c469154515"
HERE = Path(__file__).resolve().parent
FIELD_NAMES = (
    "density_total_cm3",
    "xHII",
    "xHeII",
    "xHeIII",
    "temperature_K",
    "xHI_time_average",
    "xHeI_time_average",
    "xHeII_time_average",
    "Gamma_HI_s",
    "Gamma_HeI_s",
    "Gamma_HeII_s",
    "photoheating_native",
    "xHeIII_time_average",
)
META_NAMES = (
    "time_seconds",
    "dt_seconds",
    "redshift_mid",
    "proper_box_cm",
    "emitted_photons",
    "native_total_ion",
    "native_totrec",
    "native_totcollisions",
    "native_recomions",
    "native_photon_loss",
    "native_LLS_loss",
    "native_dh0",
    "native_dhe0",
    "native_dhe2",
)
SOURCES = (
    "precision.f90",
    "mrgrnk.f90",
    "ctrper.f90",
    "romberg.f90",
    "file_admin.f90",
    "report_memory.f90",
    "mathconstants.f90",
    "cgsconstants.f90",
    "cgsphotoconstants.f90",
    "cgsastroconstants.f90",
    "c2ray_parameters.f90",
    "sed_parameters.f90",
    "atomic.f90",
    "clocks.f90",
    "test.F90",
    "files_for_3D/grid.F90",
    "tped.f90",
    "files_for_3D/mat_ini_test.F90",
    "cosmology.f90",
    "cooling_h.f90",
    "radiation_sizes.f90",
    "radiation_sed_parameters.f90",
    "radiation_tables.f90",
    "files_for_3D/sourceprops_test.F90",
    "radiation_photoionrates.f90",
    "thermal.f90",
    "doric.f90",
    "files_for_3D/photonstatistics.f90",
    "files_for_3D/evolve_data.F90",
    "files_for_3D/column_density.f90",
    "files_for_3D/evolve_point.F90",
    "files_for_3D/evolve_source.F90",
    "files_for_3D/master_slave.F90",
    "files_for_3D/evolve.F90",
)


def sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_configuration(
    n: int, helium: float, omega_m: float, omega_b: float, h: float
) -> None:
    """Reject configurations outside the external adapter's supported scope."""
    if n < 16:
        raise ValueError(
            "C2-Ray's unchanged convergence threshold requires dimension >= 16"
        )
    if not 0 < helium < 1 or not 0 < omega_b < omega_m <= 1 or not 0 < h < 2:
        raise ValueError("Invalid abundance or flat cosmology parameters")


def verify_upstream(upstream: Path) -> dict[str, str]:
    """Verify the pinned code and table bytes, including an archived checkout."""
    lock = json.loads((HERE / "upstream_lock.json").read_text())
    for relative, expected in lock["files"].items():
        path = upstream / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Pinned C2-Ray upstream mismatch: {relative}")
    return lock["files"]


def _patch_log_format(code: Path, destination: Path) -> dict:
    """Fix one malformed upstream FORMAT in corrected build copies only.

    Recent libgfortran rejects the missing comma before ``(ES12.5)``.  This
    changes a log descriptor, not source luminosities or numerical equations.
    The native reference mode deliberately retains its original source bytes.
    """
    relative = "code/radiation_sed_parameters.f90"
    path = code / "radiation_sed_parameters.f90"
    before = "'(A,A,A(ES12.5),A//)'"
    after = "'(A,A,A,(ES12.5),A//)'"
    original = path.read_text()
    if original.count(before) != 1:
        raise ValueError("Native logging FORMAT anchor is not unique")
    updated = original.replace(before, after, 1)
    path.write_text(updated)
    destination.write_text("".join(difflib.unified_diff(
        original.splitlines(True), updated.splitlines(True),
        fromfile=f"a/{relative}", tofile=f"b/{relative}",
    )))
    return {
        "enabled": True,
        "scope": "Logging-only missing-comma FORMAT portability correction",
        "files": {relative: {
            "before_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "after_sha256": hashlib.sha256(updated.encode()).hexdigest(),
        }},
        "patch_sha256": sha256(destination),
        "numerical_expressions_changed": False,
    }


def _patched_hash(patch: dict, key: str, expected: str, label: str) -> str:
    """Follow one audited build-copy edit from its verified input hash."""
    change = patch.get("files", {}).get(key)
    if change is None:
        return expected
    if change["before_sha256"] != expected:
        raise RuntimeError(f"{label} patch started from unverified bytes")
    return change["after_sha256"]


def _runtime_physics_configuration(code: Path, destination: Path) -> dict:
    """Expose nominal blackbody temperature as runtime configuration only.

    Native SED normalization and table calculations remain unchanged. The
    driver assigns this value before calling the native ``rad_ini`` routine.
    """
    relative = "code/sed_parameters.f90"
    path = code / "sed_parameters.f90"
    original = path.read_text()
    before = "real(kind=dp),parameter :: T_eff_nominal=5.0e4"
    after = "real(kind=dp) :: T_eff_nominal=5.0e4"
    if original.count(before) != 1:
        raise ValueError("Native blackbody configuration anchor is not unique")
    updated = original.replace(before, after, 1)
    path.write_text(updated)
    destination.write_text("".join(difflib.unified_diff(
        original.splitlines(True), updated.splitlines(True),
        fromfile=f"a/{relative}", tofile=f"b/{relative}",
    )))
    return {
        "enabled": True,
        "scope": "Runtime blackbody configuration; unchanged SED equations",
        "files": {relative: {
            "before_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "after_sha256": hashlib.sha256(updated.encode()).hexdigest(),
        }},
        "patch_sha256": sha256(destination),
        "numerical_expressions_changed": False,
    }


def _configured_sources(
    upstream: Path,
    destination: Path,
    n: int,
    helium: float,
    omega_m: float,
    omega_b: float,
    h: float,
) -> list[Path]:
    destination.mkdir()
    sizes = f"""module sizes
implicit none
integer, parameter :: Ndim=3, mesh(3)=[{n},{n},{n}]
integer, parameter :: meshx=mesh(1), meshy=mesh(2), meshz=mesh(3)
end module sizes
"""
    (destination / "sizes.f90").write_text(sizes)
    text = (upstream / "code/cosmoparms.f90").read_text()
    for key, value in {"h": h, "Omega0": omega_m, "Omega_B": omega_b}.items():
        text, count = re.subn(
            rf"(parameter :: {key}=)[^!\n]+", rf"\g<1>{value:.17e}_dp ", text
        )
        if count != 1:
            raise ValueError(f"Cosmology configuration declaration not found: {key}")
    (destination / "cosmoparms.f90").write_text(text)
    text = (upstream / "code/abundances.f90").read_text()
    text, count = re.subn(
        r"(^  real\(kind=dp\),public,parameter :: abu_he=)[^!\n]+",
        rf"\g<1>{helium:.17e}_dp ",
        text,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise ValueError("Helium configuration declaration not found")
    (destination / "abundances.f90").write_text(text)
    return sorted(destination.glob("*.f90"))


def _dependency_order(paths: list[Path]) -> list[Path]:
    """Sort module dependencies without rewriting the upstream Fortran."""
    modules = {}
    dependencies = {}
    for path in paths:
        source = path.read_text()
        for name in re.findall(
            r"^\s*module\s+(?!procedure\b)(\w+)", source, re.IGNORECASE | re.MULTILINE
        ):
            modules[name.lower()] = path
        dependencies[path] = {
            name.lower()
            for name in re.findall(
                r"^\s*use\s+(\w+)", source, re.IGNORECASE | re.MULTILINE
            )
        }
    remaining = list(paths)
    ordered = []
    while remaining:
        ready = [
            p
            for p in remaining
            if all(
                modules.get(name) not in remaining or modules.get(name) == p
                for name in dependencies[p]
            )
        ]
        if not ready:
            raise RuntimeError(f"Native module dependency cycle: {remaining}")
        ordered.extend(ready)
        remaining = [p for p in remaining if p not in ready]
    return ordered


def build_backend(
    upstream: Path,
    output_dir: Path,
    *,
    dimension: int,
    helium_number_fraction: float,
    omega_m: float,
    omega_b: float,
    h: float,
    mpi: bool = True,
    compiler: str | None = None,
    diagnostics: bool = True,
    transport: str = "native",
    angular_order: int = 16,
) -> Path:
    """Compile the pinned kernel and return its external-driver executable.

    Parameters
    ----------
    upstream : Path
        Checkout or git archive of the locked upstream revision.
    output_dir : Path
        Fresh build directory. Its ``runtime`` directory is the execution cwd.
    dimension : int
        Number of cells per axis (at least 16).
    helium_number_fraction, omega_m, omega_b, h : float
        Abundance by number and matching native cosmological parameters.
    mpi : bool
        Compile the upstream MPI implementation, with OpenMP enabled in both modes.
    compiler : str, optional
        GNU-compatible Fortran compiler or MPI wrapper executable.
    diagnostics : bool
        Add passive primary/secondary counters to verified build copies. False
        with native transport compiles upstream numerical files byte-for-byte.
    transport : str
        Native short characteristics or opt-in ``conservative_rays`` transport.
        The latter changes spatial transport while retaining native chemistry.
    angular_order : int
        Pixels per side of six angular cube faces for conservative rays.
    """
    validate_configuration(dimension, helium_number_fraction, omega_m, omega_b, h)
    if transport not in ("native", "conservative_rays"):
        raise ValueError("Unknown transport discretization")
    if type(angular_order) is not int or angular_order < 1:
        raise ValueError("angular_order must be a positive integer")
    upstream, output_dir = Path(upstream).resolve(), Path(output_dir).resolve()
    locked = verify_upstream(upstream)
    compiler = compiler or shutil.which("mpifort" if mpi else "gfortran")
    if not compiler:
        raise RuntimeError(
            "GNU Fortran/OpenMPI compiler not found; load gcc and openmpi modules"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    native = output_dir / "upstream"
    shutil.copytree(upstream / "code", native / "code")
    shutil.copytree(upstream / "tables", output_dir / "tables")
    runtime_physics = _runtime_physics_configuration(
        native / "code", output_dir / "runtime_physics_configuration.patch"
    )
    instrumentation = {"enabled": False}
    if diagnostics:
        if __package__:
            from .instrumentation import instrument_sources
        else:
            from instrumentation import instrument_sources

        instrumentation = instrument_sources(
            native / "code", output_dir / "diagnostics.patch"
        )
    numerical_patch = {}
    logging_patch = {"enabled": False}
    if transport == "conservative_rays":
        if __package__:
            from .conservative import patch_transport
        else:
            from conservative import patch_transport
        numerical_patch = patch_transport(
            native / "code", output_dir / "transport.patch"
        )
        logging_patch = _patch_log_format(
            native / "code", output_dir / "logging_format.patch"
        )
    configured = _configured_sources(
        native,
        output_dir / "configuration",
        dimension,
        helium_number_fraction,
        omega_m,
        omega_b,
        h,
    )
    paths = [native / "code" / source for source in SOURCES]
    paths.append(native / "code" / ("mpi.F90" if mpi else "no_mpi.F90"))
    paths += configured
    if diagnostics:
        module = output_dir / "transport_diagnostics.F90"
        shutil.copyfile(HERE / "transport_diagnostics.F90", module)
        paths.append(module)
        instrumentation["module_sha256"] = sha256(module)
    if transport == "conservative_rays":
        module = output_dir / "conservative_transport.F90"
        shutil.copyfile(HERE / "conservative_transport.F90", module)
        paths.append(module)
        numerical_patch["module_sha256"] = sha256(module)
        stable_module = output_dir / "stable_time_average.f90"
        shutil.copyfile(HERE / "stable_time_average.f90", stable_module)
        paths.append(stable_module)
        numerical_patch["time_average_module_sha256"] = sha256(stable_module)
    driver = output_dir / "driver.F90"
    shutil.copyfile(HERE / "driver.F90", driver)
    paths.append(driver)
    flags = [
        "-O2",
        "-cpp",
        "-DGFORT",
        "-fopenmp",
        "-DMY_OPENMP",
        "-ffree-line-length-none",
        "-fallow-argument-mismatch",
    ]
    if mpi:
        flags.append("-DMPI")
    if diagnostics:
        flags.append("-DC2_DIAGNOSTICS")
    if transport == "conservative_rays":
        flags.append(f"-DC2_RAY_ORDER={angular_order}")
    objects = output_dir / "objects"
    objects.mkdir()
    commands = []
    with (output_dir / "build.log").open("w") as log:
        for index, path in enumerate(_dependency_order(paths)):
            command = [compiler, *flags, "-c", str(path), "-o", f"{index:03d}.o"]
            commands.append(command)
            subprocess.run(
                command, cwd=objects, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        executable = output_dir / "c2ray_teacher"
        command = [
            compiler,
            "-fopenmp",
            *map(str, sorted(objects.glob("*.o"))),
            "-o",
            str(executable),
        ]
        commands.append(command)
        subprocess.run(
            command, cwd=objects, stdout=log, stderr=subprocess.STDOUT, check=True
        )
    runtime = output_dir / "runtime"
    (runtime / "results").mkdir(parents=True)
    compiled_hashes = {
        f"code/{source}": sha256(native / "code" / source)
        for source in (*SOURCES, "mpi.F90" if mpi else "no_mpi.F90")
    }
    for key, value in compiled_hashes.items():
        instrumented = instrumentation.get("files", {}).get(key)
        expected = instrumented["instrumented_sha256"] if instrumented else locked[key]
        expected = _patched_hash(runtime_physics, key, expected, "Runtime configuration")
        expected = _patched_hash(logging_patch, key, expected, "Logging")
        expected = _patched_hash(numerical_patch, key, expected, "Numerical")
        if value != expected:
            raise RuntimeError("Unexpected upstream source change during build")
        if instrumented and instrumented["pristine_sha256"] != locked[key]:
            raise RuntimeError("Instrumentation started from unverified native bytes")
    manifest = {
        "upstream_commit": UPSTREAM_COMMIT,
        "dimension": dimension,
        "mpi": mpi,
        "helium_number_fraction": helium_number_fraction,
        "omega_m": omega_m,
        "omega_b": omega_b,
        "h": h,
        "blackbody_temperature_K": 50000,
        "clumping_factor": 1.0,
        "runtime_physics_input_supported": True,
        "runtime_physics_defaults": {"blackbody_temperature_K": 50000.0, "clumping_factor": 1.0},
        "runtime_physics_configuration": runtime_physics,
        "numerical_sources_unchanged": not diagnostics and transport == "native",
        "numerical_algorithm_unchanged": transport == "native",
        "transport": transport,
        "angular_order": angular_order if transport == "conservative_rays" else None,
        "transport_patch": numerical_patch,
        "logging_format_patch": logging_patch,
        "diagnostics_enabled": diagnostics,
        "diagnostics": instrumentation,
        "pristine_upstream_sha256": {key: locked[key] for key in compiled_hashes},
        "compiled_upstream_sha256": compiled_hashes,
        "configuration_sha256": {p.name: sha256(p) for p in configured},
        "driver_sha256": sha256(driver),
        "executable_sha256": sha256(executable),
        "commands": commands,
        "runtime_dir": str(runtime),
        "compiler_version": subprocess.check_output(
            [compiler, "--version"], text=True
        ).splitlines()[0],
    }
    (output_dir / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return executable


def read_output(path: Path) -> dict:
    """Memory-map complete native output and reject truncated/incompatible files."""
    path = Path(path)
    with path.open("rb") as stream:
        if stream.read(8) != b"C2RTOUT1":
            raise ValueError("Unsupported C2-Ray output magic")
        header = np.fromfile(stream, dtype="<i4", count=4)
    if len(header) != 4:
        raise ValueError("Truncated C2-Ray output header")
    # File offsets may exceed 2 GiB even when every header integer is int32.
    n, steps, fields, metadata = map(int, header)
    if fields != len(FIELD_NAMES) or metadata != len(META_NAMES) or n < 1 or steps < 1:
        raise ValueError("Unsupported C2-Ray output header")
    step_values = metadata + fields * n**3
    expected = 24 + 8 * steps * step_values
    if path.stat().st_size != expected:
        raise ValueError(f"C2-Ray output size mismatch: expected {expected}")
    raw = np.memmap(path, dtype="<f8", mode="r", offset=24, shape=(steps, step_values))
    arrays = {}
    for i, name in enumerate(FIELD_NAMES):
        block = raw[:, metadata + i * n**3 : metadata + (i + 1) * n**3]
        arrays[name] = block.reshape((steps, n, n, n)).transpose(0, 3, 2, 1)
    return {
        "dimension": int(n),
        "steps": [
            dict(zip(META_NAMES, map(float, row[:metadata]), strict=True))
            for row in raw
        ],
        "fields": arrays,
    }


def main() -> None:
    """Build the backend using explicit physical configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dimension", type=int, required=True)
    parser.add_argument("--helium-number-fraction", type=float, required=True)
    parser.add_argument("--omega-m", type=float, required=True)
    parser.add_argument("--omega-b", type=float, required=True)
    parser.add_argument("--h", type=float, required=True)
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--compiler")
    parser.add_argument("--no-diagnostics", action="store_true")
    parser.add_argument(
        "--transport", choices=("native", "conservative_rays"), default="native"
    )
    parser.add_argument("--angular-order", type=int, default=16)
    args = vars(parser.parse_args())
    args["mpi"] = not args.pop("serial")
    args["diagnostics"] = not args.pop("no_diagnostics")
    sys.stdout.write(f"{build_backend(**args)}\n")


if __name__ == "__main__":
    main()
