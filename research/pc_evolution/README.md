# Continuous refined-front evolution with native 21cmFAST sources

This experiment advances the current refined photon-conserving front through
an actual redshift-dependent 21cmFAST source history. Fine ionization geometry
is carried between intervals. It tests temporal allocation, source quadrature,
and a source-matched ES control. It is a prescribed-source calculation in gas
frozen at a reference redshift, rather than a fully coupled Qin20 history.

## Source and state update

Each source endpoint uses the same native initial conditions and the restricted
deterministic L-INTEGRAL, ACG-only prescription from `pc_reconsidered`. Its native
stellar-population identity gives

\[
\dot S_i(z)=n_{{\rm ion},i}(z)H(z)/t_\star.
\]

The C implementation sets the escape-weighted SFR density to
`n_ion * rho_b / sfr_timescale`, with `sfr_timescale=t_STAR/H(z)`. Thus the
expression is an instantaneous effective photon rate. No extra density,
escape-fraction or photon-yield factor is applied. The cumulative native
`n_ion` proxy is never locally differenced or used directly as an increment.
The native SFR Hubble clock includes radiation; elapsed time uses the native
matter–Lambda `dtdz` expression, integrated over each interval. Both definitions
and the native source code hashes are recorded.

The finest source nodes provide positive trapezoidal photon increments.
Coarser transport intervals sum those same increments, so all temporal arms
receive the same photons from each source at every common output time.
`refined.solve(..., initial_state=previous_state)` updates the sparse fine
geometry. Its `absorbed` output is an increment; spectra and ionization histories
use the cumulative `state['ionized']` instead.

The capacity is `1 + density` at the explicit reference redshift (default 8).
The source positions remain the original coarse-cell centres. Native source
fields change with redshift, but this experiment adds no gas advection,
recombination, photoheating feedback, spin-temperature evolution, or finite
light-travel time. The initial gas is neutral at the first redshift (default
25); emission before that redshift is not included. An evolving-density PC
history requires a conservative ion-advection/remapping update before this
fixed-capacity assumption can be removed.

## Comparison matrix

Default parameters are four seeds, a 200 cMpc box, a 32³ source mesh, native
initial mesh 96³, and 321 real source endpoints from z=25 to z=5. Gas subdivisions
are f=2 and f=4 (eight and 64 children per source cell). Uniform direction counts
384 and 768 are fixed across all temporal arms. There are 80, 160 and 320 transport
intervals, with common checkpoints every four finest intervals: 48 PC histories
and four matched ES histories in total. The late-time outputs are separated by
0.25 in redshift, covering partial ionization through overlap for the tested
source prescription.

The source-matched ES comparator reconstructs the native-window, sharp-k,
centre-cell barrier from the **same cumulative rate-integrated source counts**
and fixed gas. Its source amplitude is one, without ionization-fraction matching.
The native independent coeval ES neutral fraction is also retained as context;
it uses a different cumulative source proxy and evolving density and is not
the controlled PC/non-PC comparison.

Time-step diagnostics require both successive refinements to change target-band
21-cm power, original-cell count L1 and resolved-subcell count L1 by at most 1%,
at common epochs with both mass ionized fractions in [0.1,0.95]. Missing eligible
epochs never pass. Cumulative source equality, photon ledgers, local bounds and
monotonic ionization are checked independently. Source quadrature is measured
separately by comparing real endpoint trapezoids with the common finer-node
integral at every common checkpoint in the same ionization window; late post-overlap photons cannot dilute earlier cumulative integration errors.
Grouping a fixed set of increments cannot establish its accuracy.

The spectral band runs from the fundamental mode through the lesser of four
times that mode and Nyquist. Its physical bounds and mode count are saved.
Fractional HI spectra are undefined at complete ionization and are omitted
there. Brightness uses saturated spin temperature without RSD, as in the
existing large-scale statistic. All reported time-convergence findings are
conditional on the tested angular and gas resolution; two direction counts
provide an angular sensitivity check, not two successive angular convergence
steps.

Escaped and finite-range retained photons stay in cumulative separate ledgers;
they are not restarted at their sources. The default numerical horizon is eight
box lengths. Retained radiation while neutral gas remains is unresolved in this
model and is included in the convergence checks. Excess emission after complete
ionization remains in the radiation ledger.

## Gadi

From the published `Photon_conserving` checkout:

```bash
export PC_THREADS=32 PC_MEM=320GB PC_WALLTIME=12:00:00
bash research/pc_evolution/gadi_prepare.sh --submit
```

The preparation selects the latest completed compatible run only for its
runtime environment. It archives the selected Git commit, verifies installed
native sources and the compiled extension, and creates a fresh `evolution_pc_*`
run. Actual initial conditions and source histories are generated inside the
PBS job. Supply a runtime explicitly with `PC_RUNTIME_PARENT` or the first
positional argument. `PC_PYTHON` selects another installed interpreter only if
the native implementation matches the archived source.

Native generation uses the allocated OpenMP threads, one seed at a time.
Independent PC histories then run in separate single-thread worker processes,
up to the CPU and memory planning limits. A single history's transport kernel
is serial. The scheduler records its allocation estimate separately from each
worker's measured peak RSS. The requested 12 hours is a PBS limit, not a runtime
prediction.

Configuration overrides: `PC_DIMENSION`, `PC_BOX_LEN` (cMpc), `PC_SEEDS`,
`PC_Z_START`, `PC_Z_END`, `PC_BASE_INTERVALS`, `PC_REFERENCE_Z`, `PC_ANGLES`,
`PC_MEMORY_GB` (worker scheduling budget in GiB), `PC_THREADS`, `PC_MEM`,
`PC_QUEUE`, `PC_WALLTIME`, and `PC_EVOLUTION_RUN_ROOT`.

## Outputs and validation

- `results/summary.json`: histories, PC/ES comparisons, temporal and angular
  changes, source quadrature diagnostics, and separate convergence results.
- `results/execution.json`: timings, worker resource plan, measured peak RSS,
  task logs and failures.
- `results/evolution_validation.png` and `.pdf`: history and temporal diagnostics
  for the first seed; all seeds remain in the JSON.
- `results/fixtures/seed*.npz`: native endpoint source rates and their provenance.
- `results/cases/*/checkpoints/*.npz`: cumulative source ledgers and sparse fine
  states, with hashes in the case reports and receipts.

`COMPLETED` means calculations, independent photon audits and shared source
budgets passed. Inspect the separate temporal, source-quadrature and angular
results before interpreting precision. Checkpoint replay, changing source
order, zero emission, finite-range retention, fully ionized spectra, and an
independent dense front reference are exercised by the tests:

```bash
python -m pytest research/pc_evolution -q --confcutdir=research
```

For a small local native integration test:

```bash
python research/pc_evolution/workflow.py run --root /fresh/output \
  --threads 4 --dimension 8 --box-len 200 --seeds 1234 \
  --base-intervals 4 --angles 384 768
```

The checked local integration (`local_validation.json`) used one native16³
seed, 321 source endpoints, and twelve PC histories through complete ionization.
All cumulative photon audits and shared-source checks passed. The largest
finest-time-pair power change was 0.000889%, and the corresponding maximum fine
L1/photon change was 0.000594%. Two full time-diagnostic gates remained false
because retained photons reached 1.40e-8 versus the declared 1e-8 limit; their
field and power tests passed. The angular fine-field difference still reached
1.58%, so temporal stability does not establish angular convergence. This local
integration does not replace the four-seed 32³ Gadi run.
