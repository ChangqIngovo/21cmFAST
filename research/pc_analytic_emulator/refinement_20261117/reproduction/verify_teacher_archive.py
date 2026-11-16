"""Safely relocate and independently audit a checksummed teacher tar.zst archive.

Requires the zstd command. Only regular files and directories under the declared
campaign root and top-level metadata files are accepted. Existing destinations
and reports are never overwritten. This verifies storage and photon ledgers;
it does not evaluate scientific prediction accuracy or use fields for fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tarfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from research.pc_analytic_emulator.data import load_dataset

MANIFEST = "ARCHIVE_SHA256SUMS.json"


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _safe_name(name: str, campaign: str) -> str:
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or not path.parts
        or ".." in path.parts
        or path.as_posix() != name.rstrip("/")
        or (len(path.parts) > 1 and path.parts[0] != campaign)
    ):
        raise ValueError(f"Unsafe archive path: {name}")
    return path.as_posix()


def _original_path(original: Path, name: str, campaign: str) -> Path:
    parts = PurePosixPath(name).parts
    return (
        original.joinpath(*parts[1:])
        if parts[0] == campaign
        else original.parent / name
    )


@contextmanager
def _tar(archive: Path) -> Iterator[tarfile.TarFile]:
    process = subprocess.Popen(
        ["zstd", "-dc", "--", str(archive)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as stream:
            yield stream
        if process.wait() != 0:
            raise ValueError("Cannot decompress teacher archive")
    finally:
        if process.poll() is None:
            process.terminate()
        process.communicate()


def _preflight(archive: Path, campaign: str) -> dict[str, tuple[bool, int]]:
    members = {}
    with _tar(archive) as stream:
        for member in stream:
            name = _safe_name(member.name, campaign)
            if name in members:
                raise ValueError(f"Duplicate archive member: {name}")
            if not member.isfile() and not member.isdir():
                raise ValueError(f"Unsupported archive member type: {name}")
            members[name] = (member.isdir(), member.size)
    if MANIFEST not in members or members[MANIFEST][0]:
        raise ValueError("Archive lacks its embedded checksum manifest")
    for name in members:
        for parent in PurePosixPath(name).parents:
            if parent.as_posix() in members and not members[parent.as_posix()][0]:
                raise ValueError(f"Unsafe archive parent is a regular file: {parent}")
    return members


def _extract(
    archive: Path,
    destination: Path,
    campaign: str,
    members: dict[str, tuple[bool, int]],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    extracted = set()
    with _tar(archive) as stream:
        for member in stream:
            name = _safe_name(member.name, campaign)
            if name in extracted or members.get(name) != (member.isdir(), member.size):
                raise ValueError("Archive changed after preflight")
            if not member.isfile() and not member.isdir():
                raise ValueError("Archive member type changed after preflight")
            extracted.add(name)
            path = destination.joinpath(*PurePosixPath(name).parts)
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as target:
                    shutil.copyfileobj(stream.extractfile(member), target)
    if extracted != set(members):
        raise ValueError("Archive changed during extraction")


def _verify_payload(
    destination: Path, members: dict[str, tuple[bool, int]], campaign: str
) -> dict[str, str]:
    hashes = json.loads((destination / MANIFEST).read_text())
    payload = {
        name
        for name, (directory, _) in members.items()
        if not directory and name != MANIFEST
    }
    if not isinstance(hashes, dict) or set(hashes) != payload:
        raise ValueError("Embedded checksum manifest does not cover every payload file")
    for name, digest in hashes.items():
        _safe_name(name, campaign)
        if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("Invalid embedded checksum")
        if _digest(destination / name) != digest:
            raise ValueError(f"Archived payload checksum failed: {name}")
    return hashes


def run(
    archive: str | Path,
    destination: str | Path,
    output: str | Path,
    *,
    campaign_name: str,
    original_root: str | Path | None = None,
) -> dict[str, Any]:
    """Extract to a fresh directory and verify all archived files and receipts.

    Parameters
    ----------
    archive
        Original read-only tar.zst artifact with an embedded checksum manifest.
    destination
        New extraction directory. Its parent may already exist.
    output
        New sanitized audit JSON outside the extraction and original campaign.
    campaign_name
        Single top-level directory name containing the archived teacher dataset.
    original_root
        Optional original campaign for before/after file checks and exact
        source/capacity array-hash comparison; unnecessary after downloading.

    Returns
    -------
    dict
        Portable integrity evidence, snapshot counts and per-input array hashes.
        No local paths, storage-provider identifiers or scientific metrics.
    """
    archive, destination, output = (
        Path(value).resolve() for value in (archive, destination, output)
    )
    original = Path(original_root).resolve() if original_root is not None else None
    if not re.fullmatch(r"[A-Za-z0-9_-]+", campaign_name):
        raise ValueError("Campaign name must be one safe directory component")
    if destination.exists() or output.exists():
        raise FileExistsError("Extraction destination and audit report must be new")
    if output.is_relative_to(destination) or (
        original is not None
        and (destination.is_relative_to(original) or output.is_relative_to(original))
    ):
        raise ValueError(
            "Audit writes must stay outside original and extracted payloads"
        )
    archive_hash = _digest(archive)
    members = _preflight(archive, campaign_name)
    original_before = {}
    if original is not None:
        for name, (directory, _) in members.items():
            if not directory and name != MANIFEST:
                path = _original_path(original, name, campaign_name)
                original_before[name] = _digest(path)
    _extract(archive, destination, campaign_name, members)
    if _digest(archive) != archive_hash:
        raise ValueError("Archive changed during extraction")
    hashes = _verify_payload(destination, members, campaign_name)
    relocated = load_dataset(destination / campaign_name)
    original_dataset = load_dataset(original) if original is not None else None
    original_samples = (
        {(sample.seed, sample.redshift): sample for sample in original_dataset.samples}
        if original_dataset is not None
        else None
    )
    records = []
    for sample in relocated.samples:
        row = {"seed": sample.seed, "redshift": sample.redshift}
        for field in ("source", "capacity"):
            digest = hashlib.sha256(
                np.ascontiguousarray(sample.arrays[field]).tobytes()
            ).hexdigest()
            row[f"{field}_sha256"] = digest
            if original_samples is not None:
                array = original_samples[(sample.seed, sample.redshift)].arrays[field]
                if (
                    hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
                    != digest
                ):
                    raise ValueError(
                        f"Relocated {field} differs from the original array"
                    )
        records.append(row)
    if original_samples is not None and len(original_samples) != len(records):
        raise ValueError("Original and relocated sample sets differ")
    if original is not None:
        for name, expected in original_before.items():
            path = _original_path(original, name, campaign_name)
            if _digest(path) != expected or hashes[name] != expected:
                raise ValueError("Original payload changed or differs from the archive")
    if _digest(archive) != archive_hash:
        raise ValueError("Archive changed during dataset audit")
    report = {
        "schema": 1,
        "complete": True,
        "archive": archive.name,
        "archive_sha256": archive_hash,
        "archive_bytes": archive.stat().st_size,
        "campaign_directory": campaign_name,
        "fresh_destination": True,
        "preflight_member_count": len(members),
        "verified_payload_files": len(hashes),
        "verified_payload_bytes": sum(members[name][1] for name in hashes),
        "checksum_manifest_sha256": _digest(destination / MANIFEST),
        "all_embedded_checksums_passed": True,
        "relocated_dataset_audit_passed": True,
        "audited_snapshots": len(relocated.samples),
        "all_photon_ledgers_passed": all(
            sample.budget["relative_photon_residual"] <= 1e-10
            for sample in relocated.samples
        ),
        "archive_unchanged": True,
        "original_comparison_available": original is not None,
        "original_payload_files_unchanged": True if original is not None else None,
        "source_capacity_arrays_match_original": True if original is not None else None,
        "input_array_hashes": records,
        "scientific_prediction_metrics_evaluated": False,
        "scope": "Archive integrity, relocation, original receipt audits and photon bookkeeping only; no fitting or scientific prediction comparison",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return report


def main() -> None:
    """Verify a downloaded teacher archive without requiring its original path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--extract-root", required=True, type=Path)
    parser.add_argument("--campaign-name", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--original-root", type=Path)
    args = parser.parse_args()
    report = run(
        args.archive,
        args.extract_root,
        args.output,
        campaign_name=args.campaign_name,
        original_root=args.original_root,
    )
    sys.stdout.write(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "complete",
                    "archive_sha256",
                    "verified_payload_files",
                    "audited_snapshots",
                )
            },
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
