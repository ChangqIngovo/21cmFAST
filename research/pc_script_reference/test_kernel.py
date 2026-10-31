"""Physical and provenance checks of the unmodified public SCRIPT kernel."""

from pathlib import Path
import os
import shutil
import subprocess

import numpy as np
import pytest

from research.pc_script_reference import kernel


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    compiler = os.environ.get("SCRIPT_FC", "gfortran")
    if shutil.which(compiler) is None:
        pytest.skip("original SCRIPT tests require gfortran or SCRIPT_FC")
    return tmp_path_factory.mktemp("script-build")


def test_original_source_hashes():
    record = kernel.provenance()
    assert record["commit"] == "793c33fd7c252f26ab033d4e00c4dfa594f445a4"
    assert record["files"]["pc.f90"]["sha256"] == "36634ea4d62b45738383e88d4bc39f57d6eee821d3f3d18f985465c7e53f22ea"


def test_single_source_partial_shell_and_periodic_wrap(cache):
    photons = np.zeros((5, 5, 5))
    photons[0, 0, 0] = 4.0
    expected = np.zeros_like(photons)
    expected[0, 0, 0] = 1.0
    for axis in range(3):
        for offset in (1, 4):
            index = [0, 0, 0]
            index[axis] = offset
            expected[tuple(index)] = 0.5
    qi, ledger = kernel.run(photons, np.ones_like(photons), build_dir=cache)
    np.testing.assert_array_equal(qi, expected)
    assert ledger["passed"] and ledger["residual"] == 0


def test_heterogeneous_partial_shell_uses_equal_photon_counts(cache):
    photons = np.zeros((5, 5, 5))
    capacity = np.ones_like(photons)
    photons[2, 2, 2] = 5.0
    capacity[2, 2, 2] = 2.0
    neighbours = []
    for axis in range(3):
        for offset in (-1, 1):
            index = [2, 2, 2]
            index[axis] += offset
            neighbours.append(tuple(index))
    for count, index in enumerate(neighbours, 1):
        capacity[index] = count
    qi, ledger = kernel.run(photons, capacity, build_dir=cache)
    assert qi[2, 2, 2] == 1.0
    # Three photons in the first incomplete shell, divided six ways.
    np.testing.assert_allclose([qi[i] * capacity[i] for i in neighbours], 0.5, rtol=1e-7)
    assert np.count_nonzero(qi) == 7
    assert ledger["passed"]


def test_overlap_and_periodic_translation(cache):
    photons = np.zeros((6, 6, 6))
    photons[2, 2, 2] = 12.0
    photons[3, 2, 2] = 8.0
    capacity = np.ones_like(photons)
    qi, ledger = kernel.run(photons, capacity, build_dir=cache)
    translated, other = kernel.run(np.roll(photons, (3, 2, 1), (0, 1, 2)), capacity, build_dir=cache)
    np.testing.assert_allclose(translated, np.roll(qi, (3, 2, 1), (0, 1, 2)), atol=3e-7, rtol=0)
    assert ledger["passed"] and other["passed"]
    assert ledger["surplus_photons"] == 0
    assert ledger["ionized_atoms"] == pytest.approx(20.0, rel=2e-6)
    assert qi.max() <= 1 and qi.min() >= 0


def test_saturation_keeps_global_surplus(cache):
    capacity = np.linspace(0.7, 1.4, 27).reshape((3, 3, 3))
    photons = capacity * 1.2
    qi, ledger = kernel.run(photons, capacity, build_dir=cache)
    np.testing.assert_array_equal(qi, np.ones_like(capacity))
    assert ledger["source_photons"] == photons.sum()
    assert ledger["atom_capacity"] == capacity.sum()
    assert ledger["surplus_photons"] == pytest.approx(photons.sum() - capacity.sum())
    assert ledger["residual"] == 0
    assert ledger["passed"]
    assert ledger["source_cast_L1_relative_error"] > 0


def test_zero_photons(cache):
    photons = np.zeros((3, 3, 3))
    qi, ledger = kernel.run(photons, np.ones_like(photons), build_dir=cache)
    np.testing.assert_array_equal(qi, photons)
    assert ledger["passed"] and ledger["relative_residual"] == 0


@pytest.mark.parametrize("case", ["negative_source", "empty_capacity", "nan", "shape", "float32_overflow", "float32_underflow"])
def test_invalid_inputs_rejected_before_compilation(case):
    source = np.ones((3, 3, 3)) * 0.5
    atoms = np.ones_like(source)
    if case == "negative_source":
        source[0, 0, 0] = -1
    elif case == "empty_capacity":
        atoms[0, 0, 0] = 0
    elif case == "nan":
        source[0, 0, 0] = np.nan
    elif case == "shape":
        source = source[:2]
    elif case == "float32_overflow":
        source[0, 0, 0] = 1e100
    else:
        source[0, 0, 0] = 1e-100
    with pytest.raises(ValueError):
        kernel.run(source, atoms)


def test_upstream_tamper_rejected(tmp_path, monkeypatch):
    destination = tmp_path / "vendor"
    shutil.copytree(kernel.VENDOR, destination)
    with (destination / "pc.f90").open("a") as stream:
        stream.write("! modified\n")
    monkeypatch.setattr(kernel, "VENDOR", destination)
    with pytest.raises(ValueError, match="checksum mismatch"):
        kernel.provenance()


def test_executable_tamper_rejected(tmp_path, cache):
    executable, _ = kernel.build(tmp_path)
    with executable.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        kernel.build(tmp_path)


def test_timeout_does_not_return_a_partial_field(tmp_path, monkeypatch):
    executable = tmp_path / "slow"
    executable.write_text("#!/bin/sh\nexec sleep 10\n")
    executable.chmod(0o755)
    monkeypatch.setattr(kernel, "build", lambda directory: (executable, {}))
    with pytest.raises(subprocess.TimeoutExpired):
        kernel.run(np.ones((2, 2, 2)), np.ones((2, 2, 2)), timeout_seconds=0.02)
