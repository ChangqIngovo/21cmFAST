"""Exercise safe teacher archive relocation and byte-level preservation."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from research.pc_analytic_emulator.refinement_20261117.reproduction import (
    verify_teacher_archive,
)
from research.pc_analytic_emulator.test_data import make_campaign


def _archive(
    root: Path, *, bad_checksum: bool = False, fault: str | None = None
) -> tuple[Path, Path]:
    campaign = make_campaign(root / "teacher", seeds=(1,), redshifts=(8.0,))
    metadata = root / "generation_protocol.json"
    metadata.write_text('{"synthetic_fixture": true}\n')
    path = root / "teacher.tar"
    with tarfile.open(path, "w") as stream:
        hashes = {}
        for file in sorted(campaign.rglob("*")):
            if file.is_file():
                name = f"teacher/{file.relative_to(campaign)}"
                hashes[name] = hashlib.sha256(file.read_bytes()).hexdigest()
                stream.add(file, arcname=name, recursive=False)
        hashes[metadata.name] = hashlib.sha256(metadata.read_bytes()).hexdigest()
        stream.add(metadata, arcname=metadata.name, recursive=False)
        if bad_checksum:
            hashes["teacher/COMPLETED"] = "0" * 64
        payload = json.dumps(hashes).encode()
        entry = tarfile.TarInfo("ARCHIVE_SHA256SUMS.json")
        entry.size = len(payload)
        stream.addfile(entry, io.BytesIO(payload))
        if fault:
            name = {
                "parent": "../outside",
                "absolute": "/outside",
                "link": "teacher/link",
                "duplicate": "teacher/COMPLETED",
            }[fault]
            entry = tarfile.TarInfo(name)
            if fault == "link":
                entry.type, entry.linkname = tarfile.SYMTYPE, "../../outside"
            stream.addfile(entry)
    compressed = root / "teacher.tar.zst"
    subprocess.run(["zstd", "-q", "-o", str(compressed), str(path)], check=True)
    return compressed, campaign


def test_relocated_campaign_matches_checksums_and_original_arrays(
    tmp_path: Path,
) -> None:
    """A fresh extraction remains independently readable with original receipts."""
    archive, original = _archive(tmp_path)
    output = tmp_path / "audit.json"
    report = verify_teacher_archive.run(
        archive,
        tmp_path / "restored",
        output,
        campaign_name="teacher",
        original_root=original,
    )
    assert report["complete"]
    assert report["relocated_dataset_audit_passed"]
    assert report["audited_snapshots"] == 1
    assert report["source_capacity_arrays_match_original"]
    assert report["original_payload_files_unchanged"]
    assert report["archive_unchanged"]
    assert str(tmp_path) not in output.read_text()
    with pytest.raises(FileExistsError):
        verify_teacher_archive.run(
            archive,
            tmp_path / "restored",
            tmp_path / "again.json",
            campaign_name="teacher",
        )


@pytest.mark.parametrize("fault", ["parent", "absolute", "link", "duplicate"])
def test_unsafe_archive_rejected_before_destination_exists(
    tmp_path: Path, fault: str
) -> None:
    """Traversal, special files and repeated names cannot overwrite any path."""
    archive, _ = _archive(tmp_path, fault=fault)
    destination = tmp_path / "restored"
    with pytest.raises(ValueError, match=r"Unsafe|Duplicate|Unsupported"):
        verify_teacher_archive.run(
            archive, destination, tmp_path / "audit.json", campaign_name="teacher"
        )
    assert not destination.exists()
    assert not (tmp_path / "audit.json").exists()


def test_corrupt_payload_checksum_never_receives_success_report(tmp_path: Path) -> None:
    """Extraction alone cannot establish archival integrity."""
    archive, _ = _archive(tmp_path, bad_checksum=True)
    with pytest.raises(ValueError, match="checksum"):
        verify_teacher_archive.run(
            archive,
            tmp_path / "restored",
            tmp_path / "audit.json",
            campaign_name="teacher",
        )
    assert not (tmp_path / "audit.json").exists()
