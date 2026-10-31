"""Archive reuse must not change the native physical experiment."""
import pytest

from research.pc_causal_rt.archive_run import archive_configuration
from research.pc_causal_rt.workflow import configuration


def test_archive_defaults_only_change_source_contract_and_numerics():
    original = configuration({"source_mode": "instantaneous"})
    corrected = archive_configuration(original)
    assert corrected["source_mode"] == "native_cumulative"
    assert len(corrected["settings"]) == 1
    assert corrected["settings"][0]["refinement"] == 1
    for key in set(original) - {"source_mode", "settings"}:
        assert corrected[key] == original[key]


@pytest.mark.parametrize("override", [{"box_len": 80.}, {"threads": 1},
                                     {"dimension": 16}, {"photon_energy_ev": 20.},
                                     {"source_mode": "instantaneous"}])
def test_archive_rejects_physical_changes(override):
    with pytest.raises(ValueError, match="cannot be changed"):
        archive_configuration(configuration(), override)


def test_archive_accepts_only_requested_numerical_changes():
    corrected = archive_configuration(configuration(), {"checkpoint_stride": 2,
        "settings": [dict(directions=24, refinement=1, cfl=.4, time_substeps=2, source_stride=1)]})
    assert corrected["checkpoint_stride"] == 2
    assert corrected["settings"][0]["directions"] == 24
