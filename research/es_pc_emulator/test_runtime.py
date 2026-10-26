"""Check runtime identity without breaking compiler wrapper dispatch."""

from types import SimpleNamespace

from research.es_pc_emulator import generate


def test_fingerprint_invokes_mpi_wrapper_by_its_original_name(tmp_path, monkeypatch):
    """OpenMPI symlinks need mpifort argv[0] even when hashing their real target."""
    package = tmp_path / "py21cmfast"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "native.so").write_bytes(b"test-native-identity")
    module = SimpleNamespace(__version__="test", __file__=str(package / "__init__.py"))
    monkeypatch.setattr(generate.importlib, "import_module", lambda name: module)
    target = tmp_path / "opal_wrapper"
    target.write_text('#!/bin/sh\ncase "$0" in *mpifort) echo mpi-version;; *gfortran) echo fortran-version;; *) exit 23;; esac\n')
    target.chmod(0o755)
    for name in ("mpifort", "gfortran"):
        (tmp_path / name).symlink_to(target)
    monkeypatch.setattr(generate.shutil, "which", lambda name: str(tmp_path / name))
    result = generate.runtime_fingerprint()
    assert result["compilers"]["mpifort"]["path"] == str(target)
    assert result["compilers"]["mpifort"]["version"] == "mpi-version\n"


def test_generation_identity_includes_teacher_source_pin(monkeypatch):
    """Changing the pinned teacher source lock must invalidate continuation."""
    monkeypatch.setattr(generate, "runtime_fingerprint", lambda: {"runtime": "fixed"})
    original = generate.generation_identity({})
    key = "research/c2ray_teacher/upstream_lock.json"
    assert key in original["source_sha256"]
    digest = generate.digest
    monkeypatch.setattr(generate, "digest", lambda path: "changed-pin" if path.name == "upstream_lock.json" else digest(path))
    assert original != generate.generation_identity({})
