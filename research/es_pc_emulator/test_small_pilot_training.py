"""Split and saved-inference checks without a native run or training sweep."""

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from research.es_pc_emulator import small_pilot_training as training  # noqa: E402
from research.es_pc_emulator.small_pilot_model import FieldNet, build_inputs  # noqa: E402
from research.es_pc_emulator.test_physical_pilot_training import make_case  # noqa: E402


def test_selection_balances_external_seeds_and_rejects_missing_cases(monkeypatch):
    """The external subset includes both IC seeds and requested counts are enforced."""
    def load(record):
        return {"metadata": {"box_len_cMpc": 50., "parameter_names": ["a"], "theta": [1.], **record},
                "xhi_es": np.zeros((3, 8, 8, 8)), "redshifts": np.array([10., 9., 8.])}

    monkeypatch.setattr(training, "load_case", load)
    records = [{"case_id": split, "split": split, "seed": 111}
               for split in ("train", "validation", "test")]
    records += [{"case_id": f"external_{j:03d}", "split": "external_test", "seed": 222 if j < 4 else 333}
                for j in range(8)]
    limits = {"train": 1, "validation": 1, "test": 1, "external_test": 4}
    selected = training.select_cases({"cases": records}, limits)
    assert [c["metadata"]["seed"] for c in selected[-4:]] == [222, 333, 222, 333]
    with pytest.raises(ValueError, match="Requested 2 train"):
        training.select_cases({"cases": records}, limits | {"train": 2})


def test_saved_field_inference_needs_no_teacher_labels(tmp_path):
    """A saved field model predicts without RT labels and rejects changed geometry."""
    case = make_case()
    case["photons"] = np.cumsum(case["photons"], axis=0)
    _, names = build_inputs(case, 0, history_steps=2)
    model = FieldNet(len(names), width=2, wide=True, auxiliary=True)
    checkpoint = {"kind": "field", "feature_names": names, "history_steps": 2,
                  "width": 2, "wide": True, "auxiliary": True, "state_dict": model.state_dict(),
                  "grid_dimension": 4, "box_len_cMpc": 50.,
                  "input_mean": torch.zeros((1, len(names), 1, 1, 1)),
                  "input_scale": torch.ones((1, len(names), 1, 1, 1))}
    path = tmp_path / "model.pt"
    torch.save(checkpoint, path)
    for name in ("xhi_rt", "temperature_rt", "xheii_rt", "xheiii_rt"):
        case.pop(name)
    predicted = training.predict_checkpoint(path, case)
    np.testing.assert_allclose(predicted["xhi"], case["xhi_es"], atol=1e-7)
    assert predicted["auxiliary"].shape == (2, 4, 4, 4, 4)
    changed = copy.deepcopy(case)
    changed["metadata"]["box_len_cMpc"] = 100.
    with pytest.raises(ValueError, match="training grid and box"):
        training.predict_checkpoint(path, changed)


def test_direct_checkpoint_is_label_free_and_respects_output_axes():
    """Direct predictions preserve their redshift/shell axes without field labels."""
    case = make_case()
    model = training.direct_model(6, 4, 4)
    for parameter in model.parameters():
        parameter.data.zero_()
    checkpoint = {"kind": "direct_p21", "state_dict": model.state_dict(), "width": 4,
                  "n_parameters": 6, "n_outputs": 4, "output_shape": [2, 2],
                  "input_mean": torch.zeros(6), "input_scale": torch.ones(6),
                  "output_mean": torch.full((4,), float(np.log(3.))), "output_scale": torch.ones(4),
                  "parameter_names": case["metadata"]["parameter_names"], "power_floor": 1e-8,
                  "redshifts": case["redshifts"].tolist(), "box_len_cMpc": case["metadata"]["box_len_cMpc"]}
    for name in list(case):
        if name not in ("metadata", "redshifts"):
            case.pop(name)
    np.testing.assert_allclose(training.predict_checkpoint(checkpoint, case)["P21"], np.full((2, 2), 3.))
