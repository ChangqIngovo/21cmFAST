"""Test isolated native builds, resumability and source/binary integrity."""

from pathlib import Path

import pytest

from research.pc_matched import native_build


def _build_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a minimal frozen archive with a deterministic compiler substitute."""
    run = tmp_path / "run"
    package = run / "source/src/py21cmfast"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# frozen package\n")
    (package / "physics.c").write_text("/* frozen native physics */\n")
    (run / "source/build_cffi.py").write_text("# frozen compiler entry\n")
    monkeypatch.setattr(native_build, "_base_identity", lambda: {"python": "pinned"})

    def compile_source(source: Path, log: Path) -> Path:
        log.write_text("compiled\n")
        extension = source / "src/py21cmfast/c_21cmfast.so"
        extension.write_bytes(b"new native binary")
        return extension

    def probe(source: Path, extension: Path, log: Path) -> dict:
        return {"package": str(source / "src/py21cmfast"), "extension": str(extension)}

    monkeypatch.setattr(native_build, "_compile", compile_source)
    monkeypatch.setattr(native_build, "_probe", probe)
    return run


def test_build_uses_copy_and_resume_does_not_compile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Build an isolated binary and reuse it only with unchanged inputs."""
    build_run = _build_run(tmp_path, monkeypatch)
    first = native_build.build_native(build_run)
    assert not (build_run / "source/src/py21cmfast/c_21cmfast.so").exists()
    assert first["native_sources"]["physics.c"]
    monkeypatch.setattr(
        native_build, "_compile", lambda *_: pytest.fail("rebuilt complete run")
    )
    assert native_build.build_native(build_run) == first


@pytest.mark.parametrize(
    "relative",
    [
        "source/src/py21cmfast/physics.c",
        "native_build/source/src/py21cmfast/physics.c",
        "native_build/source/src/py21cmfast/c_21cmfast.so",
    ],
)
def test_completed_build_tampering_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: str
) -> None:
    """Never silently rebuild a completed scientific runtime after modification."""
    build_run = _build_run(tmp_path, monkeypatch)
    native_build.build_native(build_run)
    (build_run / relative).write_text("changed\n")
    with pytest.raises(ValueError, match=r"changed|differs"):
        native_build.build_native(build_run)
    assert not list(build_run.glob("native_build.incomplete-*"))


def test_base_runtime_change_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same archive cannot resume under changed dependencies or compiler."""
    build_run = _build_run(tmp_path, monkeypatch)
    native_build.build_native(build_run)
    monkeypatch.setattr(native_build, "_base_identity", lambda: {"python": "changed"})
    with pytest.raises(ValueError, match="base runtime"):
        native_build.build_native(build_run)


def test_interrupted_build_is_preserved_before_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retain incomplete diagnostics and start a fresh source copy on retry."""
    build_run = _build_run(tmp_path, monkeypatch)
    partial = build_run / "native_build"
    partial.mkdir()
    (partial / "build.log").write_text("interrupted compiler diagnostics\n")
    native_build.build_native(build_run)
    archived = list(build_run.glob("native_build.incomplete-*"))
    assert len(archived) == 1
    assert (
        archived[0] / "build.log"
    ).read_text() == "interrupted compiler diagnostics\n"


def test_failed_build_has_no_completed_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed compiler cannot leave a resumable success marker."""
    build_run = _build_run(tmp_path, monkeypatch)

    def fail_compile(*_args: object) -> Path:
        raise RuntimeError("compiler failed")

    monkeypatch.setattr(native_build, "_compile", fail_compile)
    with pytest.raises(RuntimeError, match="compiler failed"):
        native_build.build_native(build_run)
    assert not (build_run / "native_build/build_manifest.json").exists()


def test_archived_binary_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The build must originate from source, without an archived stale binary."""
    build_run = _build_run(tmp_path, monkeypatch)
    (build_run / "source/src/py21cmfast/stale.so").write_bytes(b"stale")
    with pytest.raises(ValueError, match="compiled binary"):
        native_build.build_native(build_run)
