"""Check fixed configurations and the no-refitting/blind-evaluation gates."""

import json
import sys

import pytest

from research.pc_analytic_emulator.refinement_20261117.reproduction import (
    run_campaign as campaign,
)


def test_configurations_preserve_exact_resources_and_shared_validation(tmp_path):
    """Only added training geometries and cache sizes differ between candidates."""
    configs = campaign.configurations(tmp_path)
    single, mixed = configs["guarded_single"], configs["guarded_mixed"]
    difference = {key for key in single if single[key] != mixed[key]}
    assert difference == {"input_root", "train_seeds", "frame_cache_MiB"}
    assert single["frame_cache_MiB"] == 1536
    assert mixed["frame_cache_MiB"] == 4096
    for config in configs.values():
        assert (config["workers"], config["threads"], config["cpus"]) == (2, 1, 2)
        assert config["model_seeds"] == [23, 83]
        assert config["epochs"] == 150
        assert config["budget_epochs"] == 600
        assert config["warmstart_epochs"] == 60
        assert config["test_seeds"] == []
        assert config["external_test_root"] is None
        assert config["validation_seeds"] == [2113, 2114, 2261, 2262, 2271, 2272]
        assert config["feature_set"] == "multiscale_geometry"
        assert config["field_feature_scale_floor"] == 0.01
        assert config["zero_constant_features"] is True
        assert config["include_geometry_budget"] is True
        assert set(config["train_seeds"]).isdisjoint(config["validation_seeds"])
        assert set(config["train_seeds"]).isdisjoint(
            seed for seeds in campaign.BLIND.values() for seed in seeds
        )
    assert set(mixed["train_seeds"]) - set(single["train_seeds"]) == {
        2131,
        2132,
        2141,
        2142,
        *range(2241, 2245),
        *range(2251, 2255),
    }


def test_missing_freeze_record_rejects_before_any_blind_access(tmp_path, monkeypatch):
    """Supplying an absent selection record cannot open scientific test arrays."""
    monkeypatch.setattr(
        campaign,
        "_dataset_identity",
        lambda *args, **kwargs: pytest.fail("Blind input access before freeze"),
    )
    monkeypatch.setattr(
        campaign,
        "verify_protocol",
        lambda *args: pytest.fail("Missing freeze should fail first"),
    )
    with pytest.raises(ValueError, match="existing --freeze-record"):
        campaign.evaluate_frozen(tmp_path, tmp_path / "selection.json")


def test_invalid_selection_rejects_before_any_blind_access(tmp_path, monkeypatch):
    """An existing but invalid selection record cannot bypass verification."""
    (tmp_path / "selection.json").write_text("{}")
    monkeypatch.setattr(campaign, "verify_protocol", lambda *args: {})
    monkeypatch.setattr(
        campaign,
        "_dataset_identity",
        lambda *args, **kwargs: pytest.fail(
            "Blind input access before verified freeze"
        ),
    )

    def invalid(*args):
        raise ValueError("changed selection")

    monkeypatch.setattr(campaign, "verify_selection", invalid)
    with pytest.raises(ValueError, match="changed selection"):
        campaign.evaluate_frozen(tmp_path, tmp_path / "selection.json")
    assert not (tmp_path / "evaluation_started.json").exists()


def test_training_resume_after_selection_never_refits(tmp_path, monkeypatch):
    """The training stage becomes an audit after final selection exists."""
    (tmp_path / "selection.json").write_text("{}")
    monkeypatch.setattr(campaign, "verify_protocol", lambda *args: {})
    monkeypatch.setattr(campaign, "verify_selection", lambda *args: {"frozen": True})
    monkeypatch.setattr(
        campaign.train,
        "run",
        lambda *args: pytest.fail("A selected model cannot be retrained"),
    )
    assert campaign.train_candidates(tmp_path) == {"frozen": True}


def test_immutable_protocol_rejects_changes(tmp_path):
    """Resumption cannot silently mutate a previously declared configuration."""
    path = tmp_path / "record.json"
    campaign.freeze_json(path, {"geometry": "fixed"})
    campaign.freeze_json(path, {"geometry": "fixed"})
    with pytest.raises(ValueError, match="Frozen record changed"):
        campaign.freeze_json(path, {"geometry": "new"})
    assert json.loads(path.read_text()) == {"geometry": "fixed"}


def test_generate_cli_reports_list_result_and_continues(tmp_path, monkeypatch, capsys):
    """Successful generation cannot fail while serializing its list of records."""
    (tmp_path / "generation_protocol.json").write_text("{}")
    monkeypatch.setattr(
        sys, "argv", ["run_campaign", "generate", "--output-root", str(tmp_path)]
    )
    monkeypatch.setattr(campaign, "execute_frozen", lambda *args: False)
    monkeypatch.setattr(
        campaign,
        "generate_inputs",
        lambda *args: [{"campaign": "first"}, {"campaign": "second"}],
    )
    monkeypatch.setattr(campaign, "build_collections", lambda *args: {})
    campaign.main()
    record = json.loads(capsys.readouterr().out)
    assert record["complete"]
    assert record["result_keys"] == ["first", "second"]
