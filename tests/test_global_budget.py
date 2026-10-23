"""Independent contracts for the native global UV-budget mode."""

import attrs
import numpy as np
import pytest

from py21cmfast.wrapper.inputs import InputParameters
from py21cmfast.wrapper.outputs import IonizedBox

pytestmark = pytest.mark.filterwarnings(
    "ignore:^USE_MCGS needs a non-trivial V_CB_MODEL:UserWarning"
)


@pytest.fixture
def qin20_inputs():
    return InputParameters.from_template(
        "Qin20",
        random_seed=1234,
        HII_DIM=8,
        DIM=24,
        HIRES_TO_LOWRES_FACTOR=None,
        BOX_LEN=200,
        N_THREADS=4,
        PHOTON_CONS_TYPE="no-photoncons",
    )


def test_global_budget_preserves_all_qin20_parameters(qin20_inputs):
    calibrated = qin20_inputs.evolve_input_structs(PHOTON_CONS_TYPE="global-budget")
    before = qin20_inputs.asdict(
        include_cosmo_tables="never", only_cstruct_params=False
    )
    after = calibrated.asdict(include_cosmo_tables="never", only_cstruct_params=False)
    assert after["astro_options"].pop("PHOTON_CONS_TYPE") == "global-budget"
    assert before["astro_options"].pop("PHOTON_CONS_TYPE") == "no-photoncons"
    assert before == after
    assert calibrated.matter_options.SOURCE_MODEL == "E-INTEGRAL"
    assert calibrated.astro_options.HII_FILTER == "sharp-k"
    assert calibrated.astro_options.USE_MCGS
    assert calibrated.astro_options.USE_TS_FLUCT
    assert calibrated.astro_options.RECOMB_MODEL == "inhomogeneous"
    assert calibrated.astro_options.USE_REIONIZATION_PHOTOHEATING_FEEDBACK
    assert calibrated.astro_params.R_BUBBLE_MAX == 50


def test_global_budget_is_a_distinct_cached_configuration(qin20_inputs):
    calibrated = qin20_inputs.evolve_input_structs(PHOTON_CONS_TYPE="global-budget")
    assert calibrated.astro_options != qin20_inputs.astro_options
    # Parameter identity must distinguish PC output from the native ES cache.
    assert calibrated != qin20_inputs
    assert hash(calibrated) != hash(qin20_inputs)
    assert calibrated.random_seed == qin20_inputs.random_seed
    assert calibrated.node_redshifts == qin20_inputs.node_redshifts


def test_global_budget_audit_fields_are_explicit(qin20_inputs):
    out = IonizedBox.new(inputs=qin20_inputs, redshift=8)
    required = {
        "pc_photons",
        "pc_ionized_atoms",
        "pc_recombinations",
        "pc_recombination_previous",
        "pc_remaining_photons",
        "pc_budget_residual",
        "pc_amplitude_low",
        "pc_amplitude_high",
        "pc_boundary_weight",
        "pc_count_tolerance",
        "pc_evaluations",
        "pc_status",
        "pc_ionized_atoms_total",
        "pc_xray_ionized_atoms",
        "pc_xray_baseline_roundoff",
    }
    registered = {field.name for field in attrs.fields(IonizedBox)}
    assert required <= registered
    assert out.pc_status == 0
    assert out.pc_evaluations == 0
    assert all(np.isfinite(getattr(out, name)) for name in required)


@pytest.mark.parametrize("escape_log10", [-400, -10, 0])
def test_native_global_budget_source_limits(qin20_inputs, escape_log10):
    """Exercise exact-zero, float32-small, and surplus budgets in native C.

    These isolated source-boundary tests deliberately disable thermal/recombination
    history; the separate paired validator retains the full Qin20 template.
    """
    from py21cmfast import run_coeval

    inputs = qin20_inputs.evolve_input_structs(
        PHOTON_CONS_TYPE="global-budget",
        USE_MCGS=False,
        USE_TS_FLUCT=False,
        RECOMB_MODEL="none",
        USE_REIONIZATION_PHOTOHEATING_FEEDBACK=False,
        F_ESC10_ACG=escape_log10,
        F_STAR10_ACG=0,
    ).clone(node_redshifts=())
    coeval = run_coeval(inputs=inputs, out_redshifts=8, write=False, regenerate=True)[0]
    box = coeval.ionized_box
    rho = 1 + coeval.perturbed_field.get("density").astype(np.longdouble)
    xhi = box.get("neutral_fraction").astype(np.longdouble)
    q = xhi.size * np.longdouble(box.nion_unconditional_acg)
    uv = np.sum(rho * (1 - xhi), dtype=np.longdouble)
    pending = np.longdouble(box.pc_remaining_photons)
    assert np.isfinite(xhi).all()
    assert np.all((xhi >= 0) & (xhi <= 1))
    assert pending >= 0
    assert abs(q - uv - pending) <= box.pc_count_tolerance + 1e-11
    if escape_log10 == -400:
        # The linear escape fraction underflows to exact zero, a valid source limit.
        assert q == 0
        assert uv == 0
        assert pending == 0
        assert np.all(xhi == 1)
        assert np.all(box.get("ionisation_rate_G12") == 0)
        assert np.all(box.get("z_reion") == -1)
    elif escape_log10 == -10:
        assert 0 < q / xhi.size < np.finfo(np.float32).eps
        assert box.pc_status == 3
        assert pending > box.pc_count_tolerance
    else:
        assert np.all(xhi == 0)
        assert box.pc_status == 2
        assert pending > 0
        assert np.all(box.get("ionisation_rate_G12") > 0)
        assert np.all(box.get("z_reion") == 8)
