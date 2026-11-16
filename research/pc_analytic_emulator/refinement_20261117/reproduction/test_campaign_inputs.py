"""Mock-runtime checks for portable refinement campaign orchestration."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from research.pc_analytic_emulator.refinement_20261117.reproduction import (
    campaign_inputs as ci,
)


def _save(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def _campaign(
    root: Path, profile: tuple, runtime: dict[str, Any], template_hash: str
) -> None:
    """Make integrity-only fixtures, without scientific arrays or simulation."""
    root.mkdir(parents=True, exist_ok=True)
    seeds, redshifts, dimension, box_len = profile
    _save(
        root / "identity.json",
        {
            **runtime,
            "configuration": {
                "seeds": list(seeds),
                "redshifts": list(redshifts),
                "dimension": dimension,
                "highres_dimension": 96,
                "box_len": box_len,
                "brightness": "native",
                "sigma_cm2": ci.SIGMA_CM2,
                "analytic": ci.ANALYTIC,
                "native_template_sha256": template_hash,
            },
        },
    )
    count = len(seeds) * len(redshifts)
    _save(
        root / "summary.json",
        {
            "complete": True,
            "completed_snapshots": count,
            "requested_snapshots": count,
            "all_photon_ledgers_passed": True,
            "failed_seeds": [],
        },
    )
    _save(root / "dataset_manifest.json", {"complete": True, "synthetic": True})
    _save(
        root / "completion.json",
        {
            "identity_sha256": ci.digest(root / "identity.json"),
            "files": {
                name: ci.digest(root / name)
                for name in ("summary.json", "dataset_manifest.json")
            },
        },
    )


@pytest.fixture
def setup_inputs(  # noqa: PLR0206 - repository property-decorators includes fixtures
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    """Create six reused receipts and replace native/git execution with mocks."""
    source = tmp_path / "repo"
    relative = "research/pc_analytic/campaign.py"
    code = source / relative
    code.parent.mkdir(parents=True)
    code.write_text("# Synthetic teacher implementation.\n")
    runtime = {
        "native_runtime": {
            "native_extension_sha256": "e" * 64,
            "native_package_version": "test",
            "python_version": [3, 12, 0],
        },
        "implementation_sha256": {relative: ci.digest(code)},
    }
    template = tmp_path / "template.json"
    template.write_text('{"random_seed": 222}\n')
    previous, native = tmp_path / "previous", tmp_path / "native"
    names = {
        "original": (native, "teacher_results_verified"),
        "old_external": (native, "teacher_results_external"),
        "old_train": (previous, "teacher_train"),
        "old_validation": (previous, "teacher_validation"),
        "old_resolution": (previous, "teacher_transfer_resolution"),
        "old_volume": (previous, "teacher_transfer_volume"),
    }
    for name, (parent, directory) in names.items():
        _campaign(
            parent / directory, ci.REUSED_PROFILES[name], runtime, ci.digest(template)
        )
    base_python = tmp_path / "base-python"
    base_python.write_text("mock executable; never run\n")
    python = tmp_path / "venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(base_python)
    revisions: dict[Path, str] = {}

    def git(root: Path, *arguments: str) -> str:
        if arguments[:2] == ("worktree", "add"):
            destination = Path(arguments[3])
            (destination / relative).parent.mkdir(parents=True)
            shutil.copyfile(code, destination / relative)
            revisions[destination] = arguments[4]
            return ""
        if arguments[0] == "rev-parse":
            if arguments[1] == "different^{commit}":
                return "b" * 40
            return revisions.get(root, "a" * 40)
        if arguments[0] == "status":
            return ""
        raise AssertionError(arguments)

    monkeypatch.setattr(ci, "_git", git)
    monkeypatch.setattr(ci, "_probe_runtime", lambda *_: runtime)
    return {
        "kwargs": {
            "source": source,
            "previous_campaign_root": previous,
            "native_campaign_root": native,
            "output_root": tmp_path / "output",
            "native_python": python,
            "native_template": template,
        },
        "runtime": runtime,
        "names": names,
    }


def test_prepare_exact_profiles_and_replay(setup_inputs: dict[str, Any]) -> None:
    """All profiles, physical controls and venv paths are frozen reproducibly."""
    kwargs = setup_inputs["kwargs"]
    protocol = ci.prepare_inputs(**kwargs)
    assert ci.prepare_inputs(**kwargs) == protocol
    assert protocol["native_python"] == str(kwargs["native_python"])
    assert protocol["native_python"] != str(kwargs["native_python"].resolve())
    assert protocol["previous_campaign_root"] == str(kwargs["previous_campaign_root"])
    assert protocol["native_campaign_root"] == str(kwargs["native_campaign_root"])
    assert len(protocol["campaign_configs"]) == 8
    assert (
        sum(s["expected_snapshots"] for s in protocol["campaign_configs"].values())
        == 140
    )
    seeds = []
    for name, spec in protocol["campaign_configs"].items():
        config = json.loads(Path(spec["path"]).read_text())
        expected_seeds, z, dimension, box = ci.PROFILES[name]
        assert config["seeds"] == list(expected_seeds)
        assert config["redshifts"] == list(z)
        assert (
            config["dimension"],
            config["highres_dimension"],
            config["box_len"],
        ) == (dimension, 96, box)
        assert (config["workers"], config["threads"], config["cpus"]) == (2, 2, 4)
        assert config["analytic"] == ci.ANALYTIC
        assert config["sigma_cm2"] == ci.SIGMA_CM2
        assert config["native_extension_sha256"] == "e" * 64
        assert ci.digest(spec["path"]) == spec["sha256"]
        seeds.extend(expected_seeds)
    assert len(seeds) == len(set(seeds)) == 20
    assert not set(ci.MIXED_TRAIN_SEEDS) & set(ci.VALIDATION_SEEDS)
    assert not (set(ci.MIXED_TRAIN_SEEDS) | set(ci.VALIDATION_SEEDS)) & set(
        protocol["blind_seeds"]
    )


def test_explicit_regenerated_fallbacks(setup_inputs: dict[str, Any]) -> None:
    """Documented regenerated names work without altering their receipts."""
    for name, fallback in (
        ("original", "teacher_development_original"),
        ("old_external", "teacher_development_external"),
        ("old_train", "teacher_train_new"),
    ):
        parent, canonical = setup_inputs["names"][name]
        (parent / canonical).rename(parent / fallback)
    protocol = ci.prepare_inputs(**setup_inputs["kwargs"])
    assert Path(protocol["reuse_roots"]["old_train"]).name == "teacher_train_new"


def test_ambiguous_reused_campaign_rejected(setup_inputs: dict[str, Any]) -> None:
    """Two matching directories never silently choose an older campaign."""
    parent, name = setup_inputs["names"]["old_train"]
    shutil.copytree(parent / name, parent / "teacher_train_new")
    with pytest.raises(ValueError, match="exactly one"):
        ci.prepare_inputs(**setup_inputs["kwargs"])


@pytest.mark.parametrize(
    "changed", ["template", "commit", "config", "source", "protocol"]
)
def test_frozen_inputs_reject_changes(
    setup_inputs: dict[str, Any], changed: str
) -> None:
    """Resume refuses changed physics, settings, source and protocol records."""
    kwargs = setup_inputs["kwargs"]
    protocol = ci.prepare_inputs(**kwargs)
    if changed == "template":
        kwargs["native_template"].write_text("{}\n")
        with pytest.raises(ValueError, match="template"):
            ci.prepare_inputs(**kwargs)
    elif changed == "commit":
        with pytest.raises(ValueError, match="different commit"):
            ci.prepare_inputs(**kwargs, commit="different")
    else:
        path = {
            "config": Path(protocol["campaign_configs"]["blind_original"]["path"]),
            "source": Path(protocol["source_path"])
            / "research/pc_analytic/campaign.py",
            "protocol": Path(protocol["protocol_path"]),
        }[changed]
        path.write_text("{}\n")
        with pytest.raises(ValueError, match="changed"):
            ci.audit_inputs(protocol)


def test_audit_detects_runtime_change(
    setup_inputs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed native binary cannot generate labels under an old identity."""
    protocol = ci.prepare_inputs(**setup_inputs["kwargs"])
    monkeypatch.setattr(ci, "_probe_runtime", lambda *_: {"changed": True})
    with pytest.raises(ValueError, match="runtime"):
        ci.audit_inputs(protocol)


def test_native_probe_uses_clean_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The probe must resolve campaign imports inside its frozen worktree."""
    monkeypatch.setenv("PYTHONPATH", "/mutable/source")
    monkeypatch.setenv("PYTHONHOME", "/other/python")
    source = tmp_path / "source"
    executable = str(tmp_path / "venv/bin/python")

    def output(command: list[str], **kwargs: Any) -> str:
        assert command[0] == executable
        assert kwargs["cwd"] == source
        assert "PYTHONPATH" not in kwargs["env"]
        assert "PYTHONHOME" not in kwargs["env"]
        assert kwargs["env"]["OMP_NUM_THREADS"] == "2"
        assert kwargs["env"]["OPENBLAS_NUM_THREADS"] == "1"
        return json.dumps(
            {
                "native_runtime": {},
                "implementation_sha256": {},
                "campaign_path": str(source / "research/pc_analytic/campaign.py"),
            }
        )

    monkeypatch.setattr(ci.subprocess, "check_output", output)
    assert ci._probe_runtime(source, executable) == {
        "native_runtime": {},
        "implementation_sha256": {},
    }
    monkeypatch.setattr(
        ci.subprocess,
        "check_output",
        lambda *_, **__: json.dumps(
            {"campaign_path": "/mutable/research/pc_analytic/campaign.py"}
        ),
    )
    with pytest.raises(ValueError, match="outside"):
        ci._probe_runtime(source, executable)


def test_generate_mocked_profiles_and_resume(
    setup_inputs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Generation executes only requested commands and retains integrity status."""
    protocol = ci.prepare_inputs(**setup_inputs["kwargs"])
    calls = []

    def run(command: list[str], **kwargs: Any) -> None:
        config_path = Path(command[command.index("--config") + 1])
        name = config_path.name.removesuffix("_config.json")
        assert kwargs["cwd"] == protocol["source_path"]
        assert kwargs["check"] is True
        assert kwargs["env"]["OMP_THREAD_LIMIT"] == "2"
        root = Path(command[command.index("--output") + 1])
        _campaign(
            root, ci.PROFILES[name], protocol["runtime"], protocol["template_sha256"]
        )
        calls.append(name)

    monkeypatch.setattr(ci.subprocess, "run", run)
    first = ci.generate_inputs(protocol, ["train_resolution"])
    second = ci.generate_inputs(protocol, ["validation_resolution"])
    assert calls == ["train_resolution", "validation_resolution"]
    assert first[0]["completed_snapshots"] == 28
    assert second[0]["completed_snapshots"] == 14
    status = json.loads(
        (Path(protocol["output_root"]) / "generation_status.json").read_text()
    )
    assert not status["complete"]
    assert len(status["campaigns"]) == 2
    assert all("metrics" not in row for row in status["campaigns"])


@pytest.mark.parametrize(
    "profiles", [[], ["unknown"], ["blind_original"] * 2, "blind_original"]
)
def test_unknown_or_duplicate_profiles_rejected(profiles: Any) -> None:
    """Profile validation happens before accessing data or the native runtime."""
    with pytest.raises(ValueError, match="profile"):
        ci.generate_inputs({}, profiles)


def test_failed_ledger_is_not_completed(setup_inputs: dict[str, Any]) -> None:
    """Matching hashes alone cannot mark a failed photon ledger complete."""
    protocol = ci.prepare_inputs(**setup_inputs["kwargs"])
    root = Path(protocol["reuse_roots"]["original"])
    summary = json.loads((root / "summary.json").read_text())
    summary["all_photon_ledgers_passed"] = False
    _save(root / "summary.json", summary)
    receipt = json.loads((root / "completion.json").read_text())
    receipt["files"]["summary.json"] = ci.digest(root / "summary.json")
    _save(root / "completion.json", receipt)
    with pytest.raises(ValueError, match="ledger"):
        ci._completion(root, 56)


def test_collections_exact_counts_no_blind_reads(
    setup_inputs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only four new development profiles and six reused campaigns are read."""
    protocol = ci.prepare_inputs(**setup_inputs["kwargs"])
    for name in (
        "train_resolution",
        "train_volume",
        "validation_resolution",
        "validation_volume",
    ):
        _campaign(
            Path(protocol["campaign_configs"][name]["output"]),
            ci.PROFILES[name],
            protocol["runtime"],
            protocol["template_sha256"],
        )
    read_text = Path.read_text

    def guarded(path: Path, *args: Any, **kwargs: Any) -> str:
        if any(part.startswith("teacher_blind_") for part in path.parts):
            raise AssertionError("Blind outputs must not be inspected")
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    roots = ci.build_collections(protocol)
    assert ci.build_collections(protocol) == roots
    single = json.loads(
        (roots["guarded_single"] / "collection_manifest.json").read_text()
    )
    mixed = json.loads(
        (roots["guarded_mixed"] / "collection_manifest.json").read_text()
    )
    assert (single["training_snapshots"], single["validation_snapshots"]) == (178, 42)
    assert (mixed["training_snapshots"], mixed["validation_snapshots"]) == (250, 42)
    assert len(single["campaigns"]) == 6
    assert len(mixed["campaigns"]) == 10
    assert set(single["train_seeds"]) < set(mixed["train_seeds"])
    assert single["validation_seeds"] == mixed["validation_seeds"]
    assert all("blind" not in row["name"] for row in mixed["campaigns"])
    assert all(row in mixed["campaigns"] for row in single["campaigns"])
    assert single["allow_mixed_geometry"] is True
    root = Path(protocol["reuse_roots"]["original"])
    (root / "dataset_manifest.json").write_text("{}\n")
    with pytest.raises(ValueError, match="receipt changed"):
        ci.build_collections(protocol)


@pytest.fixture
def archived_inputs(  # noqa: PLR0206 - repository property-decorators includes fixtures
    setup_inputs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    """Relocate completed metadata fixtures and forbid all native probing."""
    original = setup_inputs["kwargs"]
    generated = original["output_root"].parent / "generated"
    for name, profile in ci.PROFILES.items():
        _campaign(
            generated / f"teacher_{name}",
            profile,
            setup_inputs["runtime"],
            ci.digest(original["native_template"]),
        )
    relocated = generated.parent / "downloaded"
    relocated.mkdir()
    kwargs = {
        "source": original["source"],
        "output_root": original["output_root"],
    }
    for key, root in (
        ("previous_campaign_root", original["previous_campaign_root"]),
        ("native_campaign_root", original["native_campaign_root"]),
        ("generated_campaign_root", generated),
    ):
        target = relocated / root.name
        shutil.copytree(root, target)
        shutil.rmtree(root)
        kwargs[key] = target
    (original["source"] / "research/pc_analytic/campaign.py").write_text(
        "# New checkout; archived teachers retain historical implementation hashes.\n"
    )
    original["native_python"].unlink()
    original["native_template"].unlink()

    def forbidden(*_: Any, **__: Any) -> Any:
        raise AssertionError("Archived preparation must not probe a native runtime")

    monkeypatch.setattr(ci, "_probe_runtime", forbidden)
    monkeypatch.setattr(ci.subprocess, "run", forbidden)
    return {"kwargs": kwargs, "archive_root": relocated}


def test_archived_relocation_no_runtime_and_exact_collections(
    archived_inputs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Relocated archives train without their native binary, template or source."""
    archive_root = archived_inputs["archive_root"]
    before = {
        str(path): ci.digest(path) for path in archive_root.rglob("*") if path.is_file()
    }
    read_json = ci._json

    def metadata_only(path: Path) -> dict[str, Any]:
        if path.name == "dataset_manifest.json":
            raise AssertionError("Archive manifests must be byte-hashed only")
        return read_json(path)

    monkeypatch.setattr(ci, "_json", metadata_only)
    protocol = ci.prepare_archived_inputs(**archived_inputs["kwargs"])
    assert ci.prepare_archived_inputs(**archived_inputs["kwargs"]) == protocol
    assert protocol["inputs_mode"] == "archived"
    assert protocol["native_python"] is protocol["native_template"] is None
    assert len(protocol["archived_campaigns"]) == 8
    audit = ci.audit_inputs(protocol)
    assert audit["native_probe_performed"] is False
    assert audit["archived_campaign_count"] == 14
    source = Path(protocol["source_path"])
    relative = "research/pc_analytic/campaign.py"
    assert ci.digest(source / relative) != protocol["implementation_sha256"][relative]
    roots = ci.build_collections(protocol)
    assert ci.build_collections(protocol) == roots
    for name, campaigns, count in (
        ("guarded_single", 6, 178),
        ("guarded_mixed", 10, 250),
    ):
        manifest = read_json(roots[name] / "collection_manifest.json")
        assert len(manifest["campaigns"]) == campaigns
        assert manifest["training_snapshots"] == count
        assert manifest["validation_snapshots"] == 42
        assert manifest["validation_seeds"] == list(ci.VALIDATION_SEEDS)
        assert not set(manifest["train_seeds"]) & set(manifest["validation_seeds"])
        assert all("blind" not in item["name"] for item in manifest["campaigns"])
        assert all(
            Path(item["path"]).is_relative_to(archive_root)
            for item in manifest["campaigns"]
        )
    after = {
        str(path): ci.digest(path) for path in archive_root.rglob("*") if path.is_file()
    }
    assert before == after
    with pytest.raises(ValueError, match=r"read-only.*disabled"):
        ci.generate_inputs(protocol)


@pytest.mark.parametrize(
    "field", ["native_runtime", "implementation_sha256", "template", "physics", "seeds"]
)
def test_archived_cross_campaign_identity_mismatch_rejected(
    archived_inputs: dict[str, Any], field: str
) -> None:
    """Matching receipts cannot conceal mixed teacher runtimes or changed physics."""
    root = archived_inputs["kwargs"]["generated_campaign_root"] / "teacher_train_volume"
    identity = ci._json(root / "identity.json")
    if field in ("native_runtime", "implementation_sha256"):
        identity[field] = {"different": "recorded identity"}
    elif field == "template":
        identity["configuration"]["native_template_sha256"] = "d" * 64
    elif field == "physics":
        identity["configuration"]["analytic"]["directions"] = 48
    else:
        identity["configuration"]["seeds"][0] = 9999
    _save(root / "identity.json", identity)
    receipt = ci._json(root / "completion.json")
    receipt["identity_sha256"] = ci.digest(root / "identity.json")
    _save(root / "completion.json", receipt)
    with pytest.raises(ValueError, match="differs"):
        ci.prepare_archived_inputs(**archived_inputs["kwargs"])


@pytest.mark.parametrize(
    "filename",
    ["identity.json", "completion.json", "dataset_manifest.json", "summary.json"],
)
def test_archived_frozen_receipt_changes_rejected(
    archived_inputs: dict[str, Any], filename: str
) -> None:
    """A later archive edit fails identity checks before any training starts."""
    protocol = ci.prepare_archived_inputs(**archived_inputs["kwargs"])
    root = Path(protocol["campaign_configs"]["blind_original"]["output"])
    (root / filename).write_text("{}\n")
    with pytest.raises(ValueError, match="changed"):
        ci.audit_inputs(protocol)


def test_archived_commit_change_rejected(archived_inputs: dict[str, Any]) -> None:
    """Archive mode retains the same immutable source-commit contract."""
    ci.prepare_archived_inputs(**archived_inputs["kwargs"])
    with pytest.raises(ValueError, match="different commit"):
        ci.prepare_archived_inputs(**archived_inputs["kwargs"], commit="different")
