"""Check accuracy contracts and label-free, validation-only deployment routing."""

import copy
import json

import numpy as np
import pytest

from research.pc_analytic_emulator.reliability import (
    accuracy_metrics,
    build_policy,
    select_route,
)

GEOMETRY = {"DIM": 96, "HII_DIM": 16, "BOX_LEN": 50}
HASHES = {"candidate": {"model.json": "a" * 64, "field_model.npz": "b" * 64}}


def fields() -> tuple[np.ndarray, np.ndarray]:
    """Construct a smooth nonconstant teacher and density field."""
    phase = np.arange(16)[:, None, None] * 2 * np.pi / 16
    teacher = np.broadcast_to(0.5 + 0.1 * np.cos(phase), (16, 16, 16)).copy()
    return teacher, np.ones_like(teacher)


def rows() -> list[dict]:
    """Return two independent validation identities with exact predictions."""
    teacher, capacity = fields()
    metrics = accuracy_metrics(teacher, teacher, capacity, 50)
    return [
        {
            "split": "validation",
            "candidate": "candidate",
            "seed": seed,
            "geometry": GEOMETRY,
            "redshift": 7.5,
            "metrics": copy.deepcopy(metrics),
        }
        for seed in (11, 12)
    ]


def test_power_and_residual_have_analytic_single_mode_values() -> None:
    """Amplitude errors square in power and follow Parseval in residual RMS."""
    teacher, capacity = fields()
    prediction = 0.5 + 1.02 * (teacher - 0.5)
    result = accuracy_metrics(prediction, teacher, capacity, 50)
    assert result["mean_relative_power_error"] == pytest.approx(1.02**2 - 1, abs=1e-12)
    assert result["brightness_residual_rms_over_neutral_rms"] == pytest.approx(
        0.002 / np.sqrt(2), abs=1e-12
    )
    assert result["passed"]
    rescaled = accuracy_metrics(prediction, teacher, capacity * 1e30, 50)
    assert rescaled == result


def test_zero_teacher_power_does_not_hide_spurious_emission() -> None:
    """Zero-reference power remains undefined and absolute brightness is tested."""
    teacher, capacity = fields()
    teacher.fill(0)
    predicted = np.maximum(
        0, 0.06 * np.cos(np.arange(16)[:, None, None] * 2 * np.pi / 16)
    )
    predicted = np.broadcast_to(predicted, teacher.shape)
    result = accuracy_metrics(predicted, teacher, capacity, 50)
    assert result["mean_relative_power_error"] is None
    assert result["undefined_power_shells"] > 0
    assert result["brightness_residual_rms_over_neutral_rms"] > 0.01
    assert not result["checks"]["absolute_brightness"]
    assert not result["passed"]


def test_uniform_false_neutral_is_caught_by_budget_despite_zero_power() -> None:
    """A constant offset is a DC error and must fail the mass-budget limit."""
    teacher, capacity = fields()
    teacher.fill(0)
    result = accuracy_metrics(teacher + 0.005, teacher, capacity, 50)
    assert result["brightness_residual_rms_over_neutral_rms"] == 0
    assert not result["checks"]["budget"]
    assert not result["passed"]


def test_late_weak_power_uses_absolute_contract_and_reports_ratio() -> None:
    """Huge relative error near zero signal is retained without dominating acceptance."""
    teacher, capacity = fields()
    teacher *= 1e-5
    result = accuracy_metrics(2 * teacher, teacher, capacity, 50)
    assert result["mean_relative_power_error"] == pytest.approx(3)
    assert result["phase"] == "late"
    assert result["passed"]


def test_policy_roundtrip_routes_without_teacher_labels() -> None:
    """A serialized policy needs only geometry, z and current model identities."""
    policy = json.loads(json.dumps(build_policy(rows(), HASHES, [11, 12])))
    route = select_route(policy, GEOMETRY, 7.5, HASHES)
    assert route["method"] == "emulator"
    assert route["candidate"] == "candidate"
    assert select_route(policy, GEOMETRY, 7.5000000001, HASHES)["method"] == "teacher"
    assert (
        select_route(policy, {**GEOMETRY, "BOX_LEN": 100}, 7.5, HASHES)["method"]
        == "teacher"
    )


@pytest.mark.parametrize("change", ["test_split", "unlisted_seed", "duplicate"])
def test_policy_rejects_test_labels_or_duplicate_realizations(change: str) -> None:
    """Test rows and repeated realizations cannot inflate validation evidence."""
    records = rows()
    if change == "test_split":
        records[0]["split"] = "test"
    elif change == "unlisted_seed":
        records[0]["seed"] = 1008
    else:
        records.append(copy.deepcopy(records[0]))
    with pytest.raises(ValueError, match=r"validation seeds|Duplicate"):
        build_policy(records, HASHES, [11, 12])


def test_policy_requires_every_seed_and_uses_validation_margin() -> None:
    """A mean passing result cannot hide a failing seed or margin failure."""
    records = rows()
    teacher, capacity = fields()
    records[1]["metrics"] = accuracy_metrics(teacher + 0.004, teacher, capacity, 50)
    assert records[1]["metrics"]["passed"]
    full = build_policy(records, HASHES, [11, 12])
    margin = build_policy(records, HASHES, [11, 12], safety_margin=0.75)
    assert select_route(full, GEOMETRY, 7.5, HASHES)["method"] == "emulator"
    assert select_route(margin, GEOMETRY, 7.5, HASHES)["method"] == "teacher"
    records[1]["metrics"] = accuracy_metrics(teacher + 0.006, teacher, capacity, 50)
    failed = build_policy(records, HASHES, [11, 12])
    assert select_route(failed, GEOMETRY, 7.5, HASHES)["method"] == "teacher"
    insufficient = build_policy(rows()[:1], HASHES, [11, 12])
    assert select_route(insufficient, GEOMETRY, 7.5, HASHES)["method"] == "teacher"


def test_candidate_selection_uses_same_seed_coverage_and_smaller_mae() -> None:
    """Unequal candidate validation sets are rejected; passing candidates rank by MAE."""
    records = rows()
    extra = copy.deepcopy(records)
    for row in extra:
        row["candidate"] = "alternative"
    teacher, capacity = fields()
    for row in records:
        row["metrics"] = accuracy_metrics(teacher + 0.001, teacher, capacity, 50)
    hashes = {**HASHES, "alternative": {"model.json": "c" * 64}}
    policy = build_policy(records + extra, hashes, [11, 12])
    assert select_route(policy, GEOMETRY, 7.5, hashes)["candidate"] == "alternative"
    with pytest.raises(ValueError, match="same validation seeds"):
        build_policy(records + extra[:1], hashes, [11, 12])


def test_mutated_model_or_policy_cannot_reuse_validation_evidence() -> None:
    """Hashes prevent serving altered model weights with stale acceptance."""
    policy = build_policy(rows(), HASHES, [11, 12])
    with pytest.raises(ValueError, match="Model artifacts"):
        select_route(policy, GEOMETRY, 7.5, {"candidate": {"model.json": "c" * 64}})
    policy["minimum_seeds"] = 1
    with pytest.raises(ValueError, match="policy changed"):
        select_route(policy, GEOMETRY, 7.5, HASHES)


def test_forged_acceptance_is_rejected() -> None:
    """Passing booleans must agree with the numerical metric contract."""
    records = rows()
    records[0]["metrics"]["MAE"] = 0.1
    with pytest.raises(ValueError, match="measured errors"):
        build_policy(records, HASHES, [11, 12])
