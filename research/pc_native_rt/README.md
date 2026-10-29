# Direct RT with identical native snapshot inputs

For continuous native source rates, conservative evolving gas, persistent ions
and radiation, see [`pc_causal_rt`](../pc_causal_rt/README.md). The experiment
below remains an independent cumulative-budget snapshot diagnostic.

This experiment compares the **original 21cmFAST ES output** with direct
finite-volume radiation transfer using exactly the same snapshot's additive
`EmissivityFields.n_ion` and evolved density. It addresses the source-budget and
density mismatch in `research/pc_matched`. It does not fit an ionization history
or rescale a field to obtain agreement with ES.

## What is preserved

Both calculations use the same native initial conditions, cosmology, galaxy
scaling relations, escaped cumulative photon field, effective atom-count
convention and density at the requested redshift. Source counts are neither
multiplied by density nor converted to a rate with `H/t_STAR`. The native ES
result is retained without modification. The original native
`brightness_temperature` function computes both 21-cm fields, using the same
native density, velocity and active brightness settings.

The default native configuration is the **same restricted configuration used
in the previous experiment**: L-INTEGRAL, atomic-cooling galaxies, no
recombinations, no spin-temperature fluctuations and no photoheating feedback.
It is not the unrestricted Qin20 template. These settings are saved with the
results. Unsupported input configurations are rejected, not silently simplified.

## What this RT experiment means

Each native snapshot is a separate initial-value radiation-transfer problem.
At its fixed redshift, the default initialization places that snapshot's entire
cumulative photon inventory isotropically at the native source locations:

\[
q_m(\tau=0,\mathbf{x})=w_m S_{\rm native}(\mathbf{x},z),
\qquad I(\tau=0,\mathbf{x})=0.
\]

The direct discrete-ordinate finite-volume solver then evolves

\[
\partial_\tau q_m+(c/a)\mathbf n_m\cdot\nabla q_m
=-c\sigma n_0a^{-3}(C-I)q_m,
\qquad
\partial_\tau I=c\sigma n_0a^{-3}(C-I)\sum_mq_m.
\]

Here `C=1+delta(z)`, and `a` and the gas are held fixed **only during this
snapshot's transport calculation**. The next snapshot obtains its own density
from native 21cmFAST. The solver has no ES threshold and no global ionization
rescaling. Periodic fluxes, coupled analytic absorption and explicit residual
radiation preserve

\[
\sum S_{\rm injected}=\sum I+\sum_mq_m.
\]

Radiation propagation uses the physical speed of light. However, **tau is the
elapsed time of this frozen-snapshot pulse experiment, not cosmological time**.
Snapshots are not chained together. An optional finite injection duration
allows a sensitivity test of the source ordering assumed by this experiment;
it does not supply a missing cosmological source history.

The native cumulative photon proxy does not uniquely specify a causal emission
history. Consequently this calculation must not be described as a complete
time-dependent replacement of 21cmFAST, or as preserving all optional native
physics. A causal replacement additionally requires consistent Lagrangian
source production, gas/ion advection, and native history-dependent processes.
Differencing Eulerian source grids and clipping negative increments, resetting
ion counts during density updates, or forcing global agreement are not valid
substitutes.

The transfer closure still uses the explicitly recorded 18-eV H opacity per
native effective absorber. Native ES does not specify that monochromatic
opacity or this pulse ordering. These are RT assumptions requiring sensitivity
tests. The ES filtering radius is retained in the native run; it is not silently
converted into an absorbing screen or photon destruction term in RT.

## Stopping, resolution and interpretation

Photons propagate until the remaining absorbable inventory
`min(radiation, neutral atoms)` is below the configured exhaustion tolerance,
or the explicit work limit is reached. Photons in excess of the total atom
capacity remain radiation. A limited calculation is recorded as unfinished;
its remaining photons are never converted into ions to make it pass.

Photon-ledger closure, exhaustion and field convergence are separate checks.
The first-order spatial transport and finite direction quadrature need
resolution studies. The default is a small pipeline check, not a certified
one-percent training teacher. Do not interpret agreement of the global budget
as agreement of morphology or the 21-cm spectrum.

Spatial subdivision repeats the native source and atom densities. Fine-grid
counts are averaged back to the native grid, preserving its count units. It
does not add unresolved cosmological structure.

## Gadi

From a clean committed checkout:

```bash
bash research/pc_native_rt/gadi_prepare.sh --submit
```

The launcher reuses an existing dependency runtime, freezes the commit, and
builds the corresponding native extension on the compute node. It
checks input and executable identities and resumes completed snapshot cases.
Use the printed `resubmit.sh` for the same frozen run after interruption.

Inspect `results/summary.json` for the experiment scope, native-input identity,
photon ledgers, exhaustion status and native ES/RT comparisons. Saved fields
retain native source and atom grids, the untouched ES field, the RT field and
both native brightness outputs. Numerical refinement settings can be supplied
through the launcher; run identity changes require a fresh output directory.

For example, `PC_DIRECTIONS=96,384 PC_REFINEMENTS=2,4 PC_CFL_VALUES=0.2`
selects four numerical configurations for every native snapshot. All four use
the same archived native fields. Keep the native thread count fixed across
separate experiments: native initial-condition generation can depend on that
count even at the same random seed. The actual initial-condition hash is saved.

## Validation

Measured local results are in [`local_validation.json`](local_validation.json).
The 69 new tests and 82 existing `pc_matched` tests pass. Twelve actual native
snapshots (two seeds, z=10,9,8,7,6,5) passed exact input-identity, independent
photon-ledger and exhaustion checks, with maximum ledger residual 2.32e-16.
Another 18 snapshot/configuration calculations exercised spatial, angular and
CFL refinement; maximum ledger residual was 1.34e-15. A frozen preparation and
PBS-body execution, including isolated native compilation and completed-run
reuse, passed locally. No Gadi scheduler submission was made from this runtime.

These results do not certify a one-percent teacher. For seed 111 at z=7,
16-cubed to 32-cubed RT changed a primary power bin by 2.38%. On identical
source/density arrays, changing pulse injection to constant injection over one
box crossing changed a primary power bin by 1.65%. This explicitly measures a
source-ordering assumption that the native cumulative budget does not specify.

```bash
python -m pytest research/pc_native_rt -q --confcutdir=research/pc_native_rt --import-mode=importlib
python -m pytest research/pc_matched -q --confcutdir=research/pc_matched --import-mode=importlib
```
