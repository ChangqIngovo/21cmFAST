# Continuous native-source radiative transfer

This path carries the ionized gas and directional radiation through cosmic
time. Native 21cmFAST supplies the same initial conditions, galaxy source model,
cosmology and 2LPT displacements throughout the calculation. It addresses the
snapshot resets and fixed-density limitations of the earlier experiments.

## Equations and source convention

In effective native mean-atom-cell units the radiation and ions obey

\[
\partial_t q_m + (c/a)\,\mathbf n_m\cdot\nabla q_m
=w_m j-c\sigma n_0a^{-3}(C-I)q_m,
\]
\[
\partial_t I+\nabla\cdot(I\mathbf u/a)
=c\sigma n_0a^{-3}(C-I)\sum_mq_m.
\]

The prescribed native matter flow also conserves the total atom density C.
Radiation uses physical c. Ions are never reset at a new redshift, and photons
remaining in flight are never converted into ions to reach a target history.
The independent global ledgers are

\[
\sum C(t)=\sum C(t_0),\qquad
\sum I(t)+\sum_mq_m(t)=\sum_p Q_p(t_0)+\int_{t_0}^t\sum j(t')\,dt'.
\]

The default `source_mode=native_cumulative` retains the native coeval photon
budget. Let Q_p(t) be native `n_ion` on a persistent Lagrangian source cell p.
The source is the positive proper-time increment of Q_p, deposited at that
source's original 2LPT position. The unchanged native emissivity function exports
Q_p using a temporary zero-displacement copy of the initial conditions; restoring
the original displacements must reproduce every archived Eulerian source field
within native float32 CIC rounding (L1 relative tolerance 1e-6). Source packets
use the native low-resolution source grid; they are distinct from the high-
resolution gas mass packets. No native physical parameter or native C routine
is changed.

Cumulative Q_p is interpolated linearly in proper time. Each interval emits
(Q_p(t_new)-Q_p(t_old))/dt at midpoint source positions in each matter substep.
Negative packet increments are rejected, never clipped. A translating constant
source budget emits no new photons. The first-node Q_p is deposited once as an
isotropic initial radiation inventory, with neutral initial gas; this is an
explicit boundary condition, not a reconstruction of the pre-start radiation
field. Subsequent steps inject only increments. The global ledger is therefore
initial Q_p plus subsequent emission = ions + radiation, and its total at every
native node agrees with that node's Q_p sum. At z_start=25 the initial stellar
budget is tiny; starting later requires checking this boundary condition.

The historical emission-location map and the current-location cumulative native
source map are saved separately. They generally differ because sources move.
The static additive ES diagnostic receives the current cumulative source map.
Native ES and the original SCRIPT reference now use the same cumulative budget
up to native float32 deposition rounding. Finite-c RT retains photons in flight;
its ionization history is not forced to follow either coeval allocation model.

`source_mode=instantaneous` preserves the former experimental convention,
`j = n_ion * H(z) / t_STAR`, for explicit reproduction of older runs. Its time
integral is not the native coeval cumulative budget. It is no longer the default
for matching 21cmFAST photon counts.

The native matter-Lambda clock is inverted analytically for intermediate
redshifts; native growth is evaluated there. Cumulative source endpoints interpolate in
proper time. The legacy instantaneous option retains its original native Hubble
convention and zero initial radiation. For the default cumulative option, the
first-node native photon inventory is included explicitly as described above.

## Gas motion without changing native smoothing

Initial atom masses and 2LPT trajectories are extracted from the actual native
initial-condition fields and use native nearest-node displacement lookup.
A uniform mass cloud one native cell wide reproduces native CIC deposition
exactly through finite-volume overlaps. On a grid refined by f, its width is f
fine cells, not one fine cell. Summing its fine cell masses reproduces the same
native coarse density, so resolution changes do not change the gas smoothing
or resolve a new point-particle density field.

Ion advection integrates the overlap of each translated material cloud with
the old and new cells. The positive mass-transfer matrix has the correct old
and new capacities as its marginals. It preserves ion counts, local bounds and
the identity map at zero displacement. It uses piecewise-constant ion fractions
and is first order in space; its errors are included in the spatial/time study.
Native float32 deposition/FFT roundoff is kept separate from the conserved
double-precision atoms and audited at every native endpoint (absolute density
tolerance 3e-6). Density filtering, unsupported perturbation modes, negative
initial masses and native clipping that breaches this audit are rejected.

Each matter step moves gas halfway, evolves radiation/absorption through the
full interval on that midpoint density, then completes the gas motion.
`time_substeps` independently refines this splitting; the radiation CFL controls
its inner substeps. Continuous source injection and absorption are now solved
together in each reaction half step, rather than leaving a final half-step
source injection unabsorbed. The constant-source local solve is analytic and
conservative, including optically thick cells. Generic linear source endpoints
use the half-step mean rate, preserving the exact emitted budget and second-order
time accuracy; the native-cumulative source is constant within each matter step.
Radiation streaming uses MUSCL with monotonized-central slopes and SSPRK2,
with CFL <= 0.5 to preserve positive cell averages. No field is clipped or
globally normalized to repair a photon budget.

The source-free absorption update also computes its surviving radiation directly,
avoiding a subtractive roundoff failure in very opaque cells. Measured changes,
limitations and rejected propagation candidates are documented in
[`NUMERICAL_IMPROVEMENTS.md`](NUMERICAL_IMPROVEMENTS.md).

The common local ion-capacity tolerance is2e-11 relative to cell capacity.
This admits floating-point accumulation during repeated conservative remaps;
the unmodified ion counts are retained and the excess is reported. It is not
a clipped reservoir or a photon-loss allowance. Larger violations fail.

## Physical scope

This is the existing restricted native ACG/L-INTEGRAL configuration: no
recombinations, thermal or spin-temperature evolution, mini-halo sources, or
photoheating feedback. Unsupported native input configurations fail explicitly.
The original native brightness function evaluates every comparison field with
the same native density and velocity, including the active brightness settings.

The opacity remains the explicitly recorded single-energy, effective-atom
closure: default18eV, native H+He effective nucleus count, H-like cross section.
It is not separate H/He chemistry. Photon-frequency redshifting, diffuse
recombination radiation and thermal feedback are not supplied. Native ES's
filtering radius is retained in its reference, not silently turned into an RT
absorber. Thus the present validation applies to these recorded equations and
the currently active baseline; it does not certify all optional 21cmFAST physics.

## Numerical study and outputs

The default is two seeds,50cMpc,8^3 native cells (24^3 initial mass packets),
160 source intervals from z=25 to5, and11 settings per seed. All numerical
branches use identical native sources and atom trajectories. Five independent
three-level ladders meet at a common finest reference:

| Axis | Levels |
|---|---|
| Directions |24,96,384|
| RT refinement |1,2,4|
| Radiation CFL |0.4,0.2,0.1|
| Matter substeps per native interval |1,2,4|
| Source-knot stride |4,2,1|

Keep the native thread count fixed when comparing separate runs: native initial
conditions can depend on that count even for the same seed. Actual initial
conditions, source arrays and trajectories are archived and hashed.

The original strict certification profile requires each axis to pass both
adjacent comparisons at the finest settings of every other axis. Its default
checks use the same cosmic timestamps, active mass-neutral fractions 0.05--0.9,
neutral-field MAE <= 0.01 and every defined target power bin within 1%. This legacy
profile and its recorded `tolerance` and certification results remain unchanged.
The primary band runs from the fundamental to the lesser of four fundamental
modes and half the native-grid Nyquist. At 8^3 in 50 cMpc its bounds are
0.12566--0.25133 cMpc^-1 and it contains only one shell, centred at
0.16036 cMpc^-1. It does not certify accuracy on all 21cm scales.

The current working goal is spatial-refinement stability comparable to original
SCRIPT using the same archived 21cmFAST inputs, refinement grids, output
averaging and spectral bands. A 5% large-scale power difference is a provisional
reference, not a universal cutoff or evidence that this goal has been reached.
Every assessment must state its actual k range and active epochs. Neutral-field
MAE remains a diagnostic without a new arbitrary threshold; higher-k changes
must be reported alongside SCRIPT's measured refinement changes. Neither
RT--SCRIPT field agreement nor a per-bin requirement that RT change less than
SCRIPT establishes a true-error bound. The first same-input benchmark is recorded
in `script_refinement_validation.json`, with details and a reproduction command
in `NUMERICAL_IMPROVEMENTS.md`. It shows mixed, comparable spatial sensitivities
at matched volume-neutral fraction 0.4 for one local seed222 history; it does not
establish convergence across all epochs, seeds or numerical axes.
Photon, atom and source-budget audits remain strict;
angular, temporal and source-sampling sensitivity still need to be assessed.
Higher native resolution and larger volumes remain necessary for a production
teacher.

`COMPLETED` means the histories and independent photon/atom audits succeeded.
`summary.json` and `causal_rt_report.json` contain the separate original-profile
numerical certification, not a SCRIPT-level assessment. Missing comparison
steps, active epochs or defined power bins cannot pass that certification.
The report also saves contextual ES and same-budget static ES
comparisons; their difference from RT is never used as a numerical error target.

Native archives and source histories are hashed. Every saved checkpoint carries
fine directional radiation, fine ions and cumulative source counts for restart;
coarse diagnostic fields are archived separately. The latest restart state is
published atomically. Changed sources, input files, native binary or output
artifacts are rejected. The launcher freezes the selected commit and builds its
native extension inside the compute job.

## Gadi

For the first corrected-source check, reuse the existing native history and
native binary, without rebuilding 21cmFAST or repeating native source histories:

```bash
PC_INPUT_RUN=/g/data/qp00/sz4640/21cmfast_runs/causal_rt_5c1b6d8719_20261029T121626_GyH1 \
PC_MEM=8GB PC_WALLTIME=01:00:00 \
bash research/pc_causal_rt/gadi_budget_prepare.sh --submit
```

This freezes a fresh `causal_budget_*` run and uses D96, f1, CFL0.4, four matter
substeps, and every source knot. Old RT checkpoints are not resumed because
those used a different source budget. The old native archives are read-only.
The script additionally runs the original SCRIPT allocation as an external
comparison, using unchanged native source/density and brightness functions.
Results include `causal_rt_report.json`, `budget_script_comparison.json` and
`budget_script_comparison.png`; the comparison includes same-redshift and
matched-volume-neutral-fraction fields and spectra. A one-hour allocation is
a limit, not a Gadi runtime measurement. Use the printed `resubmit.sh` for an
interrupted corrected run. This small check cannot certify convergence.

The archive launcher permits numerical overrides only; it rejects changes to
native physical inputs. All optional physics that was already disabled in the
old archive remains outside this experiment's validation scope. SCRIPT is a
reference, not a replacement for 21cmFAST physical prescriptions.

For the complete numerical ladders with newly generated native histories,
submit each seed separately:

```bash
for seed in 111 222; do
    PC_SEEDS="$seed" PC_THREADS=8 PC_MEM=32GB PC_MEMORY_GB=24 \
    PC_WALLTIME=48:00:00 \
    bash research/pc_causal_rt/gadi_prepare.sh --submit
done
```

This is a requested walltime limit, not a measured Gadi duration. Use each
printed `resubmit.sh` to continue that exact frozen calculation if interrupted.
The two jobs independently report each seed's convergence; both must pass.

For a short pipeline validation, explicitly override the matrix:

```bash
PC_SEEDS=111 PC_DIMENSION=4 PC_SOURCE_INTERVALS=8 \
PC_Z_START=10 PC_Z_END=7 PC_DIRECTIONS=24 PC_REFINEMENTS=1 \
PC_CFL_VALUES=0.4 PC_TIME_SUBSTEPS=2 PC_SOURCE_STRIDES=1 \
PC_THREADS=2 PC_MEM=16GB PC_MEMORY_GB=12 PC_WALLTIME=01:00:00 \
bash research/pc_causal_rt/gadi_prepare.sh --submit
```

`PC_CONFIG_JSON` accepts the complete workflow dictionary, with explicit PC_*
environment values overriding it. Output is written to a fresh `causal_rt_*`
run under the standard21cmFAST run directory. Plot: `results/causal_rt_comparison.png`.

## RT--native ES hypothesis check

`hypothesis_config.json` selects eight cases without a Cartesian product:
RT grids at native refinement factors1,2,3; angular directions24,96,384
on the native grid; CFL0.4,0.2,0.1 on that grid; and matter substeps4,8.
Every source stride is1. This diagnoses numerical sensitivity of the current
comparison, with no change to the archived native physical inputs. It is not
the full common-anchor convergence matrix above. Saving every native node
reduces the interpolation interval in matched-neutral-fraction comparisons.

```bash
cd /g/data/qp00/sz4640/21cmFAST
unset PC_DIRECTIONS PC_REFINEMENTS PC_CFL_VALUES PC_TIME_SUBSTEPS PC_SOURCE_STRIDES
unset PC_MAX_STEPS PC_CHECKPOINT_STRIDE PC_TOLERANCE
PC_INPUT_RUN=/g/data/qp00/sz4640/21cmfast_runs/causal_rt_5c1b6d8719_20261029T121626_GyH1 \
PC_CONFIG_JSON=/g/data/qp00/sz4640/21cmFAST/research/pc_causal_rt/hypothesis_config.json \
PC_MEM=8GB PC_WALLTIME=01:00:00 \
bash research/pc_causal_rt/gadi_budget_prepare.sh --submit
```

The budget PBS workflow also writes `hypothesis/hypothesis_report.json`,
outside the completed RT input directory. It compares RT directly with the
original native ES fields, at identical redshifts and at volume-neutral
fractions0.8,0.6,0.4,0.2. Fields are interpolated before computing spectra;
the bracket nodes, redshifts and weights are saved. SCRIPT remains a separate
comparison. For each numerical pair the report records the RT--RT change,
both RT--ES differences, all spectral shells and cross correlations. Ratios
whose reference-model difference vanishes are undefined, never set to zero.

The same-time mass-neutral history admits the exact no-recombination identity

```
xHI_RT - xHI_ES = radiation/atoms - (emitted - native_ES_ions)/atoms.
```

This separates the RT radiation inventory from the native ES allocation
deficit. The inventory can depend on numerical resolution; this identity
alone does not establish a converged finite-light-speed effect. Persistent
field differences and shrinking refinement changes support a difference in
modeling assumptions. A single refinement difference is not an error bound.

Already completed corrected runs can be compared without rerunning RT:

```bash
python -m research.pc_causal_rt.hypothesis \
  --input-roots /path/to/baseline/results /path/to/refined/results \
  --output-dir /path/to/separate/hypothesis
```

The analysis checks completion and case receipts, native histories, cumulative
source-budget hashes, native binaries and scientific implementation hashes.
Different seeds may be present in different input roots; comparisons for a
given seed require the identical archived history. The photon-energy and
other physical configuration must agree. Comparison cases must save the same
checkpoint nodes; the supplied configuration reruns its baseline at the same
dense cadence as the refinements. The analysis does not alter inputs.

## Tests

```bash
python -m pytest research/pc_causal_rt -q --confcutdir=research/pc_causal_rt --import-mode=importlib
```

Tests cover analytic radiation propagation/absorption, exact gas translation,
same-density refinement, atom/ion/photon closure, actual native source extraction,
time/growth functions, interrupted continuous evolution, identical resumed and
uninterrupted outputs, immutable archives and incomplete convergence matrices.
Measured results are recorded in `local_validation.json` after integration tests.

The earlier instantaneous-source implementation passed126 causal and151 existing tests. Six actual
continuous histories (two seeds, three numerical settings,162 native source
snapshots) evolved from z=25 to5. Independent source reconstruction and atom/
photon audits passed; maximum saved photon and atom residuals were5.83e-15 and
6.23e-17. Native versus conserved gas density differed by at most1.88e-7.
Interrupted and uninterrupted state evolution agree exactly for the tested
frozen source history. A frozen PBS body, isolated native build and completed
reuse passed locally. Gadi submission was not made from this environment.

These are execution and consistency results. The preliminary8^3 to16^3
transport comparison still changed the target power bin by1.66% and1.31% in
the two seeds. The full five-axis Gadi matrix remains necessary; no1% teacher
certification has been issued.

## Native cumulative budget validation

The corrected source interface passes164 causal RT tests, including moving
sources with zero emission, source growth, initial inventory, refinement and
bitwise restart equivalence. Two actual archived8³ histories (seeds111/222,
81 native nodes each) were evolved with D96/f1/CFL0.4/t4/s1. All native source
audits passed; the maximum saved photon-ledger residual was4.16e-15 and the
maximum cumulative packet-budget residual was8.22e-17. Source totals agree with
the original deposited float32 native grids within native roundoff.

The local RT cases took20.09s and19.78s, excluding source export and comparison.
For local seed222 atz7, mass-neutral fractions were0.37017 (RT),0.36242
(SCRIPT) and0.39749 (native ES). Radiation in flight was0.007743 photons/atom.
These are local validation data, not Gadi timings or a convergence certificate.
The full frozen PBS workflow and receipt-preserving resume also passed locally.
Exact measurements are saved in `native_budget_validation.json`.

## Local hypothesis validation

`hypothesis_validation.json` records the fixed-source refinement experiment
using the frozen6f90f73c solver and two local80-interval native histories.
For seed222 at matched volume-neutral fraction0.4, RT--native ES field MAE
is0.07274,0.07296,0.07417 for RT grids8^3,16^3,24^3. The corresponding
successive RT--RT differences decrease from0.01689 to0.00687. Seed111 also
retains its RT--ES difference under8^3 to16^3 refinement (0.07877 to0.07858).
Matched fields use the recorded saved-node interpolation brackets; these
local runs saved every fourth native node, whereas the Gadi hypothesis
configuration above saves every node.

The spatial comparisons do not establish1% power convergence: the worst
active first-shell changes for seed222 are1.706% and1.665%. Higher-k changes
are also recorded. Angular refinement24 to96 to384 passes both pair checks
on the native RT grid. Radiation CFL0.4 to0.2 changes its first-shell power
by up to1.084%;0.2 to0.1 reduces that change to0.418%. Thus the persistent
neutral-field difference supports a difference in ionization modeling,
while numerical contributions to power remain measurable. These are local
fixed-source results, not a rerun of the user's Gadi native realization.
