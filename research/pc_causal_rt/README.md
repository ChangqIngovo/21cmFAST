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
\sum I(t)+\sum_mq_m(t)=\int_{t_0}^t\sum j(t')\,dt'.
\]

The source is the native instantaneous escaped stellar photon rate,
`j = n_ion * H(z) / t_STAR`, using its native SFR convention. Native `n_ion`
is not itself a rate. Its independent coeval cumulative proxy generally does
not equal this time integral. Neither quantity is rescaled to force agreement.
Original native ES is therefore a contextual comparison. A separate static ES
control receives the integrated emission-location map; this shares the emitted
budget but does not become a causal radiation-transfer solution.

The native matter-Lambda clock is inverted analytically for intermediate
redshifts; native growth is evaluated there. Source endpoints interpolate in
proper time. The source's native Hubble convention includes radiation, as in
the unchanged source code. Emission before the specified initial redshift is
not supplied: initially gas is neutral and radiation is zero. Extending that
initial condition earlier is a separate physical-history sensitivity test.

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
its inner substeps. The reaction update is analytically coupled and conservative.
Radiation streaming uses MUSCL with monotonized-central slopes and SSPRK2,
with CFL <= 0.5 to preserve positive cell averages. No field is clipped or
globally normalized to repair a photon budget.

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

Each axis must pass both adjacent comparisons at the finest settings of every
other axis. Checks use the same cosmic timestamps, active mass-neutral fractions
0.05--0.9, neutral-field MAE<=0.01 and every defined target power bin within1%.
The primary band is explicitly limited to the lesser of four fundamental modes
and half the native-grid Nyquist. At8^3 this includes only one shell. It is a
small numerical validation, not a claim of1% accuracy on all21cm scales. Higher
native resolution and larger volumes remain necessary for a production teacher.

`COMPLETED` means the histories and independent photon/atom audits succeeded.
`summary.json` and `causal_rt_report.json` contain the separate numerical
certification. Missing comparison steps, active epochs or defined power bins
cannot pass. The report also saves contextual ES and same-budget static ES
comparisons; their difference from RT is never used as a numerical error target.

Native archives and source histories are hashed. Every saved checkpoint carries
fine directional radiation, fine ions and cumulative source counts for restart;
coarse diagnostic fields are archived separately. The latest restart state is
published atomically. Changed sources, input files, native binary or output
artifacts are rejected. The launcher freezes the selected commit and builds its
native extension inside the compute job.

## Gadi

For the complete numerical ladders, submit each seed separately:

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

## Tests

```bash
python -m pytest research/pc_causal_rt -q --confcutdir=research/pc_causal_rt --import-mode=importlib
```

Tests cover analytic radiation propagation/absorption, exact gas translation,
same-density refinement, atom/ion/photon closure, actual native source extraction,
time/growth functions, interrupted continuous evolution, identical resumed and
uninterrupted outputs, immutable archives and incomplete convergence matrices.
Measured results are recorded in `local_validation.json` after integration tests.

The checked implementation passes126 new and151 existing tests. Six actual
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
