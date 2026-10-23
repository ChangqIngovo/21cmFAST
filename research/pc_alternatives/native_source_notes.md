# Native Qin20 source audit: the missing additive interface

The existing `global-budget` implementation is unchanged. This diagnostic reads
native saved fields and optionally repeats one frozen native ionization step.
It does not supply a production replacement.

## Result

**A single positive cell source grid cannot exactly reproduce the native Qin20
E-INTEGRAL source fields at all excursion-set radii using the native filter.**
This is not merely an expected consequence of the source code: the saved native
fields violate a necessary Fourier support condition for such a representation.

A positive source grid can be *defined* from the cell-scale native source
estimate, preserving the ACG and MCG global photon budgets. That is a useful
interface for a local allocation algorithm, but it is a new spatial source
closure, and must be validated as such. Keeping all Qin20 stellar and feedback
parameters does not make this closure exactly equivalent to the native
multiscale conditional-HMF construction.

## Reproducible inputs and limits

`native_source_audit.py` reads the frozen fixture
`/workspace/scratch/40b418003e4a/qin20_validation/final_guard_coarse/frozen_native_step`.
The main audit uses `previous_ionized.h5` and `previous_density.h5`, both at z=8.
SHA256 fingerprints, full option metadata, every radius, and diagnostic values
are in `native_source_results.json`.

This is an 8^3 ionization grid, 24^3 initial grid, 200 cMpc periodic box, seed
1234, and coarse node sequence [35, 20, 12, 9, 8, 7.5]. It preserves the Qin20
astrophysical parameters and source/thermal flags, but is not a converged
spatial or temporal science run. The saved metadata has `POWER_SPECTRUM=EH` and
`V_CB_MODEL=NONE`. An explicit comparison against
`InputParameters.from_template("Qin20", random_seed=1234)` confirms that these
are the installed template's settings, not experiment overrides. All
astrophysical parameters, matter options, and cosmological parameters match
that template. The only astrophysical-option difference is
`PHOTON_CONS_TYPE=global-budget`; numerical differences are HII_DIM, DIM,
BOX_LEN, N_THREADS, and the explicitly changed node chronology. The native
wrapper still warns that the grid is too coarse and that MCG evolution needs
a nontrivial streaming-velocity model; the latter warning also occurs for the
unmodified installed template. Coarse resolution and chronology prevent calling
this a converged production validation.

The fixture follows the `global-budget` history. Both PC and no-PC modes use
the same source evaluation functions, although their feedback histories can
diverge. The counterexample therefore establishes that the native source
representation is not generally additive; it is not a claim that every
history has the numerical error percentages measured here.

Run from the repository root:

```bash
PYTHONPATH=src \
  /workspace/scratch/40b418003e4a/native_es_research_20261023/21cmfast_runtime/bin/python \
  research/pc_alternatives/native_source_audit.py --next-step
```

Omit `--next-step` to use only numpy/h5py and never call the native extension.
No production C changes or source-field instrumentation are needed.

## What the native calculation actually stores

Primary evidence is the repository source at retained production commit
`f35833db4470fce8742332ee003773a343c4b1c6`:

- `src/py21cmfast/src/IonisationBox.c`: `copy_filter_transform`,
  `calculate_conditional_nion`, `find_ionised_regions`, `setup_radii`,
  `set_recombination_rates`, and `pc_calibrate`.
- `src/py21cmfast/src/filtering.c`: `sharp_k_filter` and `filter_box`.
- `src/py21cmfast/src/hmf.c`: `evaluate_nion_conditional_acg/mcg` and
  `nion_unconditional_acg/mcg`.
- `src/py21cmfast/src/recombinations.c`: `recombination_rate`.

In Qin20 `SOURCE_MODEL=E-INTEGRAL`, the inputs filtered at radius R include
density, previous density, and feedback turnover masses. The code evaluates
the nonlinear conditional-HMF integral anew at that R. The radius changes the
conditional mass limit and variance as well as the smoothed inputs. With
MCGs enabled, it separately stores a source history at each radius:

\[
g_{a,R}(t)=g_{a,R}(t_-) + f_{a,R}(t;M_{\rm turn}(t))
                       - f_{a,R}(t_-;M_{\rm turn}(t)),
\]

where a denotes ACG or MCG. Thus even the history correction is not obtained
by filtering a single cumulative cell photon map.

Immediately before applying the ES barrier, the code rescales each species
at each R by a separate **volume-mean** correction:

\[
q_{a,R}(i)=g_{a,R}(i)
\frac{\bar q_a^{\rm uncond}}{\langle g_{a,R}\rangle_V}.
\]

The audit uses these stored-source estimates before the small high-redshift
source floors in `find_ionised_regions`. It is an additivity audit of the source
history arrays, not an exact reconstruction of every barrier comparison.

For E-INTEGRAL, `q_R` enters the barrier directly, alongside the ionization
and recombination cost per baryon in the region. It is not divided by density.
The L-INTEGRAL/halo-grid branch does explicitly divide its filtered source-count
grid by density. Consequently treating E-INTEGRAL's conditional arrays as
already additive photon counts would mix different conventions.

## Necessary-condition test

The native Qin20 sharp-k kernel is

\[
\widetilde W_R(k)=\mathbf 1[0.413566994\,kR\le 1].
\]

The smallest index is a special unfiltered cell step. If a source field S
existed such that \(F_R=W_R*S\), then every Fourier coefficient of F_R
outside the retained support would be exactly zero. This condition holds
even if S were allowed to be negative, so failure rules out a positive S too.

The audit tests two interpretations independently:

1. `F_R=q_R`, treating the conditional efficiency itself as the filtered field.
2. `F_R=rho_R*q_R`, the natural count interpretation, with
   `rho_R=1+W_R*delta` and no density clipping needed in this fixture.

Both fail. The maximum forbidden fraction of fluctuation power is **0.5486%**
for q_R and **0.7271%** for rho_R*q_R. For the latter, the forbidden-mode RMS
is about **119,500 times float32 epsilon times the field mean**. A true
convolution control passes with roundoff-level rejected power. All discrete
modes are safely away from the sharp-k cutoff; float-vs-double cutoff decisions
do not explain this result. The failed support condition alone is sufficient
for the nonexistence statement; no nonlinear inverse fit is necessary.

There is also a DC inconsistency: the density-weighted source count changes
with R even though each q_R has the same volume mean. Modes retained at both
the cell scale and R also disagree, by up to 6.42% in relative L2 norm for the
count interpretation.

## Size of the closure change in this fixture

Define q_cell as the sum of the separately normalized ACG and MCG cell fields.
Two simple predictions are `W_R*q_cell` and `W_R*(rho*q_cell)/rho_R`.
Their maximum RMS discrepancies are respectively **8.29%** and **13.36%**
of the native q_R fluctuation standard deviation. These are not percentages
of the mean and not power-spectrum suppression predictions.

The native global budget is

\[
Q=N_{\rm cell}(\bar q_{\rm ACG}^{\rm uncond}
              +\bar q_{\rm MCG}^{\rm uncond})=314.9296648795.
\]

The natural unnormalized local count interpretation sums to
`sum(rho*q_cell)=317.2827357707`, 0.747% above Q. Multiplying the combined
grid by 0.9925836781 forces agreement. Refiltering that normalized grid gives
up to 13.25% RMS error relative to the native fluctuation standard deviation.

Prefer preserving each species' budget independently:

\[
S_{a,i}=Q_a\frac{\rho_i q_{a,\mathrm{cell}}(i)}
                   {\sum_j\rho_j q_{a,\mathrm{cell}}(j)}.
\]

| Species | Native global Q_a | Unnormalized local sum | Required factor |
|---|---:|---:|---:|
| ACG | 183.9832161 | 185.9705185 | 0.9893138846 |
| MCG | 130.9464488 | 131.3122173 | 0.9972145130 |

This species-preserving version gives a maximum 13.19% fluctuation-RMS mismatch
after refiltering. It is positive in this fixture and exactly preserves the
declared species budgets. The normalization is a modeling choice about spatial
emissivity; the numbers are not evidence that the original code emitted too
many physical photons.

| R (cMpc) | Native sum(rho_R*q_R)/Q | Volume-grid prediction error/std | Species-preserving packet error/std |
|---|---:|---:|---|
| 15.509, unfiltered cell | 1.0074717 | 0 | 0.04814 |
| 24.977 | 1.0065900 | 0.06330 | 0.10637 |
| 36.569 | 1.0051928 | 0.08219 | 0.13166 |
| 48.673 | 1.0039971 | 0.08294 | 0.13160 |

The sharp-k real-space kernel also has negative weights. On this small discrete
grid the maximum sum of the magnitudes of its negative weights is 3.2581, while
the signed total is one. It is unsuitable as a nonnegative photon transfer
probability. Using its ES scale ranking is possible; using its signed entries
as transfer/debit weights is not.

### Independent moderate-grid cost fixture

The later `native_benchmark.py prepare` run created a 32^3 ionization, 96^3
initial-grid fixture with the same 200 cMpc box, seed 1234, 4 threads and exact
installed Qin20 astrophysical/matter/cosmological settings. Its numerical
chronology is [35, 12, 8, 7.5]. Repeating this source audit on its saved z=8
history gives `native_source_grid32.json`:

- Maximum forbidden fluctuation power: 4.2341% for q_R and **5.7080%** for
  rho_R*q_R.
- Filtering q_cell gives a maximum **37.98%** error relative to native
  fluctuation standard deviation.
- Refiltering the species-budget-normalized count closure gives a maximum
  **56.22%** error relative to native fluctuation standard deviation.
- The combined cell-count normalization factor is **0.9227381**, corresponding
  to an unnormalized count sum about 8.37% above the declared global budget.

The chronology and prior PC/no-PC history differ between these two fixtures,
so this is **not a resolution-convergence measurement**. It does show that the
cell-source closure discrepancy is not universally small. A cheap allocation
method does not resolve this independent modeling uncertainty.

## Units, history, and recombinations

The existing global PC ledger uses **effective native photon-count units**:
one mean-density cell has atom capacity one, a cell has capacity rho_i, and
the photon budget is Ncell times the unconditional cumulative source estimate.
For local neutral capacity, X-ray preionization must be removed:
`C_i=rho_i*(1-xe_i)` before UV ionizations. A physical hydrogen count conversion
would multiply all of these consistently by mean H atoms per cell. This audit
does not prove an independent emitted-photon/helium/secondary-ionization budget.

For recombinations, the ES barrier labels the stored field as recombinations
per mean baryon, then divides it by rho_R. The rate helper describes its output
as a rate per baryon integrated over a density PDF; its caller evaluates it at
`1+z_eff=(1+z)*rho^(1/3)` and accumulates it without an explicit extra rho.
Therefore **follow and document the native ledger convention** when comparing
algorithms; do not add another rho factor merely from the helper's wording.
A derivation of physical sink units independent of this implementation is still
needed before claiming exact radiative-transfer photon conservation.

The optional frozen native step z=8 -> 7.5 confirms that the input prior history
is unchanged. The normalized combined local cumulative packet grid increases
in every cell here: minimum increment 0.093746455 and global increment
88.84070467. This single positive example does not prove that differences of
Eulerian cumulative grids always define nonnegative emissivity. Advection,
feedback, and changing normalization can move or lower source history locally.
Silently clipping negative future increments would violate the budget.

## Clean integration boundary and outstanding work

1. Retain Qin20 native density, source parameters, stellar efficiency, MCG,
   X-ray, and feedback calculations. Keep the retained `global-budget` mode
   as an independent baseline.
2. After native cell source history is available, define one explicit positive
   additive UV source field with separately declared ACG/MCG budgets. The above
   species-normalized S is an inexpensive provisional closure: O(Ncell) work
   beyond quantities already available with Qin20. The native multiscale q_R
   fields may remain priorities/priors; they cannot be simultaneously imposed
   as exact additive inventories.
3. Run a separately selected allocation/transport operator on that source
   field and on positive cell capacities. Debit each source once; retain
   unabsorbed/unreachable photons explicitly. Transfer kernels must be
   nonnegative, even if the ES predictor keeps sharp-k filtering.
4. To evolve causally, export nonnegative instantaneous emissivity or transport
   cumulative histories consistently before forming increments. The present
   saved cumulative source fields alone do not establish that interface.
5. The returned ionization map must be paired with a declared UV-background,
   mean-free-path, and current-step recombination closure. Reusing an ES Gamma
   field after arbitrarily changing the ionization topology is an additional
   approximation. Feedback and recombinations must be evolved separately for
   PC and no-PC arms, not applied only as final-map postprocessing.
6. Validate resolution, chronology, isolated sources, overlapping sources,
   source permutation, shadows, and directional transport against a reference.
   Photon allocation does not by itself establish correct morphology or
   radiative transfer. Switching to native L-INTEGRAL or discrete halos gives
   a more natural additive source representation, but changes the Qin20
   source model and must be a separate experiment.

This exposes a concrete choice rather than an implementation obstacle that can
be removed by a faster allocator: preserve the exact native multiscale source
statistics as heuristic information, or adopt a consistent additive source
closure. An exact equivalence claim is ruled out by the saved counterexample.
This is a property of independently conditioned source estimates, not evidence
that the native conditional-HMF model is intrinsically physically incorrect.

## Runtime evidence

The previously saved native paired run through z=8 used 72 default evolution
nodes per arm and reports ES 425.86 s and global-budget PC 372.94 s. Those are
whole-arm elapsed times from `qin20_validation/native_paired_z8/report.json`.
Execution order, warm state, and noisy runtime make them unsuitable for claiming
a PC speedup. The optional step in the current results takes about 6 s; a prior
repeat took about 12 s. Neither is a full-model cost benchmark. Additional O(N)
packet construction is likely inexpensive, but full paired Qin20 timings and
production-grid memory measurements are still required for any candidate.

### Scheduled 32^3 native benchmark

`native_benchmark.py` now supplies an actual native ionization-cost denominator,
with results in `native_benchmark.json`. Preparation evolved the exact installed
Qin20 physical settings at grid32/DIM96, BOX200, seed1234, 4 threads along the
declared coarse [35,12,8,7.5] chronology. All four physical parameter groups
matched the installed template. Preparation completed in 90.22 s after initial
conditions; this timing was not used as a paired performance benchmark.

Measurement loaded the final frozen z=8 -> 7.5 inputs, copied identical fields
outside timed regions, used 4 native threads, warmed both modes once, and then
timed ES/PC followed by PC/ES while other research benchmarks were paused.

| Mode | Warmup (s) | Timed samples (s) | Median (s) |
|---|---:|---|---:|
| no-photoncons | 12.166 | 9.606, 9.708 | 9.657 |
| global-budget | 10.838 | 9.584, 7.394 | 8.489 |

The actual measured native ionization scale is about 9–10 s. The two-sample
PC/ES median ratio 0.879 **does not establish a speedup**, given the large PC
sample spread and shared-host timing noise. Both modes produced identical
unconditional ACG/MCG sources and preserved the frozen prior history.

These times include native conditional source integrations and history at all
ES radii, filtering, ES thresholding, UV-background and recombination updates,
and the global-budget solve where enabled. They exclude initial conditions,
density/velocity perturbation, spin-temperature and X-ray evolution,
brightness-temperature computation, fixture I/O, independent input copies,
and extraction of the proposed packet source closure. They are not full coeval
timings. The frozen prior history is the no-PC history for both calls, as
appropriate for an operator-cost comparison; it is not a paired science
evolution of two self-consistent histories.

The corresponding packet/capacity diagnostic input is
`/workspace/scratch/40b418003e4a/qin20_native_cost_grid32/source_capacity.npz`.
It contains source, density, X-ray baseline, previous recombinations, and the
native xHI map separately. `capacity` includes remaining atom capacity but
**not recombination costs**. A candidate allocation on that pair alone cannot
be compared physically to the native current-step ionization solution.

## Event-front actual-grid cost and angular comparison

`event_native_benchmark.py` and `event_native_optimized_benchmark.json` apply
the optimized continuous-rate `event_front` prototype to the same 32^3
diagnostic arrays. Compilation is excluded, independent workers measure peak
process RSS, and all three quadratures
use periodic boundaries, duration one, and a maximum straight-ray distance
32*sqrt(3) cell lengths. This reinterprets the saved cumulative photon inventory
as a constant rate over an arbitrary unit interval. It is **not native
chronological emission** and excludes all recombinations.

| Angular directions | One-thread solve (s) | Peak process RSS (MiB) | Explicit rays | Retained photons |
|---:|---:|---:|---:|---:|
| 48 | 0.7662 | 113.2 | 805,488 | 0.177771 |
| 192 | 3.1697 | 262.2 | 3,223,296 | 0.0822913 |
| 768 | 16.9369 | 851.4 | 12,879,360 | 0.167939 |

Total injected photons are 25,702.9382. Periodic escape is zero; finite-range
retained photons are accounted separately and are not counted as absorbed.
Global, source, angular, and independent absorption ledgers pass to floating
point precision, with no negative absorption or capacity violation. The indexed
event heap peaks at 32,768 entries at every K, removing stale entries, and blocked
ray allocation avoids vector-growth memory peaks. C++ SHA256 is
`aca74136b8383b7202903a4570cc78473feb4ae4f487a832b758cb2db9426c19`.
The old report and maps remain preserved; an independent comparison confirms
optimized K48 and K192 absorption and xHI maps are **bitwise identical** to
their original outputs. The final K768 run checked 5.06 GB effective available
memory before launch and used explicitly recorded 40 s / 1200 MiB projection
guards. Its measured runtime and memory are below 20 s and 1 GiB.

Relative to the four-thread native ionization median, this additional
single-thread kernel costs 7.93% at K48, 32.82% at K192, and 175.38% at K768.
These are **additional kernel costs**, not ratios of two complete
interchangeable native simulations. Each optimized point is one timed run;
large-grid scaling and full native end-to-end costs remain unvalidated.
At K48 vs K192 the mean absolute xHI difference is 0.004545, RMS difference
0.014522, maximum cell difference 0.22725, and map correlation 0.998034.
The mode-weighted absolute difference of shell-mean neutral-fraction power is
0.181% of the reference weighted power; the lowest six-mode shell differs by
1.08%. K192 vs K768 further reduces mean absolute xHI difference to 0.001290,
RMS to 0.004138, maximum cell difference to 0.05831, with map correlation
0.999840. The corresponding weighted shell-power difference is 0.07428%, with
0.329% in the lowest six-mode shell. These changes support angular
self-consistency on this fixture, not physical accuracy or full grid/angular
convergence. Directly comparing K48 to K768 gives mean absolute xHI difference
0.004360, maximum cell difference 0.21939, and weighted shell-power difference
0.22382%. Finite-range retained photons vary nonmonotonically with K.

## A future instantaneous source-rate interface

A short source inspection identifies a better chronological input than
differences of cumulative E-INTEGRAL histories:

- `EmissivityFields.sfrd_acg` and `sfrd_mcg` are already calculated for Qin20
  spin-temperature evolution. `EmissivityFields.c:get_integral_properties`
  multiplies Eulerian SFRD by rho; `hmf.c:sfrd_conditional_*` obtains the SFRD
  mass integral with escape fractions removed. Native `sfr_timescale` is
  `t_STAR*t_hubble(z)`, and `cosmology.c` explicitly defines t_hubble in seconds.
- Qin20 has ALPHA_ESC=0. Thus its escaped UV rate can use the existing species
  SFRD grids with constant physical escape fractions and the POP2/POP3 yields.
  In the native effective baryon convention,
  \[
  \dot e_i=\frac{
   f_{\rm esc,ACG}N_{\gamma,ACG}\,\mathrm{SFRD}_{ACG,i}
   +f_{\rm esc,MCG}N_{\gamma,MCG}\,\mathrm{SFRD}_{MCG,i}}
   {\bar\rho_b}.
  \]
  The cell-volume factor cancels when counts are normalized to mean baryons
  per cell. Use the code's transformed physical escape fractions, not the
  logarithmic Python parameter values. Do not insert a years-to-seconds factor
  twice. Conversion to independently defined physical hydrogen-photon counts
  still requires an explicit hydrogen/helium/yield-unit audit.
- For a general mass-dependent escape fraction, the correct integrand is
  `fesc(M)*N_gamma*SFR(M)*dn/dM`; multiplying an unweighted SFRD map by an
  average escape fraction is not equivalent. The existing
  `fesc_weighted_sfrd` name is misleadingly incomplete: its C construction
  **already includes the photon yield**. It is currently allocated only for
  Lagrangian source models and cannot simply be requested as an E-INTEGRAL
  field without extending the interface.
- `drivers/single_field.py:compute_ionization_field` currently substitutes a
  dummy EmissivityFields object for E-INTEGRAL even if a field is supplied.
  A future explicitly selected rate-based mode must change that boundary and
  retain/export the instantaneous rates. No such production change was made
  in this study.

Accumulate newly emitted photons by positive time quadrature, for example
\(\Delta E_i=\Delta t(\dot e_i(t_n)+\dot e_i(t_{n+1}))/2\), with feedback and
temporal resolution checked. Carry ionization, pending radiation, and cumulative
emission/absorption/sink ledgers forward. Inject only this increment. Never
reinject the cumulative nion map each step, and never repair a negative
cumulative-map difference by clipping it.

Two modeling qualifications remain. First, native SFRD mean correction uses
the average of SFRD/rho in Eulerian mode. Its summed emissivity therefore need
not equal the unconditional SFRD normalization times volume. The global budget
must explicitly be the integral of the chosen positive rate field, or a
separately declared positive species normalization. Second, the time integral
of SFRD-weighted UV emission is not generally equal to the legacy nion history
update. A rate-based chronological model preserves Qin20 stellar parameters but
introduces a temporal source closure; it must be compared against the legacy
history rather than silently asserted to match it.

The present event prototype also returns only separate source and direction
*marginals* for its finite-range retained photons. Those totals alone do not
encode the joint source-direction-position state needed to resume transport.
Before time evolution, either complete the infinite-speed traversal or retain
resumable ray state. Recombination, changing density, evolving rates and opacity,
and feedback introduce additional state transitions beyond this monotone-front
cost experiment.
