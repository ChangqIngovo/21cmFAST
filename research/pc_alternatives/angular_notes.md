# Angular transport as a 21cmFAST photon-conservation alternative

These are research probes. Production code and its existing `global-budget`
mode are unchanged. They transport supplied photon increments through toy
hydrogen capacities; they do not implement Qin20 source evolution, recombinations,
X-rays, thermal evolution or a new public 21cmFAST mode.

## What is worth keeping

An angular radiation field can remove the number-of-sources factor without
throwing away direction. For fixed global directions `n_a` and quadrature weights
`w_a`, store `P[j,a]`, the number of photons in each cell and angular bin. Sources
inject `S[j] w_a`. Positive streaming and proportional local absorption preserve
each channel's count. This avoids the `number of sources × number of target
cells` graph used by direct source-to-cell allocation.

However, conservation does not establish accuracy. The first-order prototype
passes ledgers, a full opaque wall, translations and thread reproducibility,
while it broadens a diagonal beam and leaks around a finite screen. At low
angular and spatial resolution it is a poor replacement for physical transport.
It is a consistent discretization of transport with finite-resolution errors,
not a proof that the underlying transport equations are inconsistent.

`angular_high_order.py` adds positive MC-limited MUSCL spatial reconstruction
and SSPRK2 streaming. The separate high-order results quantify whether the
extra work buys sufficient accuracy. This remains a transport solver replacing
the ES ionization decision, while reusing 21cmFAST inputs and other physics.

## Discrete equations and ledger

For remaining atom capacity `C_j-A_j`, first compute

`T_j = sum_a P[j,a]`, `dA_j = min(C_j-A_j, T_j)`.

Subtract `P[j,a] dA_j/T_j` from every angular bin. This makes `0 <= A <= C`
and preserves `A + sum_a P_a`. The capacity model is an infinitely opaque
sharp-front limit until a cell's atom budget is exhausted, with no recombinations.
It is not a frequency-dependent finite-opacity chemistry solver.

First-order streaming uses

`P'[j,a] = (1-sum_i nu_ai) P[j,a] + sum_i nu_ai P[upwind_i(j),a]`,

where `nu_ai = c dt |n_ai|/dx` and `sum_i nu_ai <= 1`. All coefficients
are nonnegative and their sum is one. Periodic streaming preserves each angular
sum; open boundaries explicitly accumulate escaped photons. At every final
time, for every angular bin,

`w_a sum_j S_j = absorbed_a + pending_a + escaped_a`.

The code retains pending photons. It never calls unabsorbed radiation ionization.
Although no explicit source labels are stored, the positive linear streaming
and common cell absorption multiplier allow a nonnegative conceptual source
decomposition conditional on the solved radiation field. This does not create
an exact continuum angular or straight-characteristic solution at finite K.

For MUSCL, the monotonized-central reconstruction has positive face values
with sum-direction CFL at most 1/2. Each forward-Euler streaming stage is
conservative and positive. Absorption is applied after both SSPRK2 stages;
the final convex combination averages **both radiation and ionized counts**.
Absorption and escape counters receive the same one-half stage weights. A
final local projection removes any coexistence of radiation and unfilled atom
capacity introduced by convex averaging. Thus an unexhausted opaque cell cannot
transmit during an unabsorbed intermediate RK stage. Second-order accuracy is
claimed for smooth streaming, not across discontinuous ionization fronts.
The higher-order verification also fills a 64-atom periodic box from 1000
photons, leaving exactly 936 photons in radiation, and reproduces the final
one-dimensional 8.4-photon front to a maximum cell error of 3.3e-15.

## Negative probes that must be retained

The low-cost donor-cell implementation at four threads gives:

- Nonuniform-grid total residual about `4e-14` photons for 226 injected photons;
  per-angle residual below `8e-15`; zero translation error; identical physical
  arrays on one and four threads.
- A full axis-aligned opaque wall admits exactly zero photons beyond it.
- For an isotropic 1000-photon source two cells before an open-box plane,
  the wall receives about 44% at K24–96, below the 50% hemispheric maximum.
  The plane has finite transverse extent, so 50% is a bound, not its exact value.
- A diagonal collimated pulse has transverse variance 9.6 cell-squared after
  traveling 13.576 cells, versus zero for exact transport. This exactly matches
  the stencil prediction `variance = Nsteps*dt/sqrt(2)` for this orientation.
- A finite oblique screen fully blocking the exact central ray leaks 23.787%
  of the pulse around its edges. Passing a plane-wall test alone therefore
  misses a serious finite-resolution error.
- The isolated R6 sphere has radial-ranking shape L1 error about 0.49–0.50 of
  the photon budget at K24–96. A separate K384 probe still gives 0.5018,
  showing that angular refinement alone does not fix spatial diffusion.

The reference sphere is an exact-count radial ranking of cells, not a voxel
integral of a continuous sphere. It is a morphology diagnostic; its discrepancy
contains cell sampling error in addition to transport error.

The two equal spheres at separation 1.5R require a distinct geometric test:
directions leaving the first sphere through its overlap with the second obey
`n dot axis > 0.75`, hence occupy exactly 1/8 of 4pi. This is **not** the
second sphere's full apparent solid angle, whose fraction is 0.127322.
Across 1000 sampled orientations, equal-weight antipodal Fibonacci bins give
RMS absolute errors 0.02979, 0.01769, 0.01003 and 0.003446 for K24,48,96,384.
The corresponding sampled maximum errors are 0.08333, 0.04167, 0.03125 and
0.01042. Exact angular-bin budgets alone do not guarantee an accurate 1/8 cone.

## Cost and comparison scope

Dense angular storage costs approximately `16 K Ncell` bytes for donor-cell
and `24 K Ncell` bytes for the high-order prototype, plus scalar fields.
Dense work is `O(K Ncell Ntime)` with no explicit source-count factor.
First-order N32/K24 and K48 use about 13 and 26 MB; N64 uses about 105 and
206 MB. These numbers include the two angular work arrays but omit small Python
and runtime overheads.

The first timing pass compared with one forward and 12 inverse scalar FFTs.
It found large kernel-only ratios. These are **not whole-21cmFAST overheads**:
Qin20's conditional source integration and other physics can dominate the ES
step. The root experiment subsequently measured a full frozen N32 Qin20 ES
ionization call around 9.657 seconds on the same four-thread setup. High-order
results must be interpreted against that full-call measurement, with explicit
qualifications about source statistics, elapsed transport time and grid scaling.
Timing repetitions made while other research ran are preliminary.

The active-region improvement is exact: an angular streaming cell is skipped
only when it and all adjacent possible inflow cells have exactly zero radiation.
MC reconstruction cannot create inflow from a second neighbour across an empty
upwind neighbour because its slope then vanishes. Absorption in empty cells is
also skipped. There is no intensity cutoff or dropped photon tail. Physical
arrays agree bit-for-bit with the dense computation in the nonuniform test.
The scalar activity scan remains `O(Ncell)` per stage; angular work scales with
the currently active radiation region, and angular storage is still dense.

`angular_high_order_results.json` preserves dense measurements.
`angular_high_order_active_results.json` measures the same equations with exact
activity masks. At N32, R6 and a 24-cell light travel distance, the active
four-thread K24/48/96/384 runs took roughly 0.25/0.42/0.75/2.83 seconds.
Their sphere/voxel L1 errors were 0.416/0.371/0.356/0.354 of the photon budget.
The diagonal-beam screen leakage fell from 23.787% to 2.615%, and transverse
variance from 14.4 to 3.091 cell-squared at the same flight distance. This is
substantial progress, but 2.6% shadow leakage is not exact ray transport.

`angular_resolution_results.json` gives the fixed-K96 spatial study: R4/N24,
R8/N40 and R12/N64 have sphere/voxel L1 errors 0.433, 0.300 and 0.285 and
times 0.140, 1.67 and 6.77 seconds. The continuous sphere is integrated with
24-by-24 midpoint quadrature in x/y and exact z intersections; its total-volume
error is below 4.4e-5. Refining the R12 case to K384 lowers the L1 error to
0.220 and removes spurious open-boundary escape, but costs 45.10 seconds and
2.42 GB. These tests show improvement with resolution, not completed numerical
convergence or a uniformly small cost.

A saved Qin20-derived N32 source/capacity fixture was also injected as a
**synthetic simultaneous flash**, without interpreting its saved cumulative
source values as a physical redshift increment. K48 transport over 16 cell
light-crossing lengths took 4.83 seconds for Q/C=0.824; 0.541% of photons were
still explicit radiation at the endpoint. Scaling only the source to Q/C=0.0824
left seven locally supercritical source cells and cost 0.057 seconds. This
illustrates the early sparse / late dense workload range. Neither is a Qin20
reionization run, a feedback test, nor a validated physical ionization history.
It would be misleading to multiply these isolated-call ratios by a whole run
without checking source increments, persistent radiation, time steps and
recombination chemistry.

The current high-order implementation is therefore a **candidate to retain**,
not a demonstrated solution to all accuracy/cost requirements. It offers a
source-count-independent alternative if characteristic packets become expensive,
and a clear continuum refinement path, but the tested sphere errors remain
substantial. Sparse block allocation could reduce memory to the active region;
it has not been implemented. At large late-time filling factors the radiation
can occupy most of the volume, so neither mask speedups nor sparse memory have
a uniform guarantee. A production integration must retain radiation between
redshift steps, derive physically valid source increments, include sinks, and
compare the resulting morphology/power with a resolved reference.

## A defensible hybrid, and what it cannot skip

An ES map may propose refinement regions, a front guess or a preconditioner.
It cannot irrevocably ionize the interior and leave angular transport only the
residual budget: if ES placed absorption behind an opaque obstacle, a residual
correction cannot reclaim those photons or repair the incorrect atom allocation.
Every retained bulk ionization needs a valid previous transport ledger or a new
allocation certificate; otherwise the full proposed map must remain revisable.

A promising hybrid keeps explicit angular packets or characteristic fluxes at
interfaces, resolves absorption near neutral fronts, and traverses certified
transparent interiors in long jumps. A block may be skipped exactly only when
its opacity and new sinks are zero throughout the jump, it has no unaccounted
emission, the packet's trajectory and flight time are retained, and the block's
exit is found along that same characteristic. A cell flagged fully ionized by
ES is not such a certificate, especially with recombinations or residual HI.

If a path segment instead has a certified optical-depth bound `tau <= epsilon`,
neglecting its absorption changes that packet's number by at most
`P (1-exp(-epsilon)) <= P epsilon`. Accumulated path bounds can control this
approximation, but its missed absorption must be acknowledged. Source-free
transparent blocks can be organized in an octree or distance transform; this
reduces traversals only where geometry admits large empty regions. It gives no
worst-case promise that arbitrary opaque networks are as cheap as ES filtering.

Merging packets with only matching cell and direction is insufficient for exact
transport: different subcell impact parameters can hit opposite sides of a
small obstacle. Exact merges require matching trajectory/phase and emission
time information, or a certified uniform response for the whole packet bundle.
Approximate merging can still be useful with angular/spatial convergence tests.

Implicit or short-characteristic sweeps can remove a light-crossing CFL limit
for a static-opacity steady problem. The unknown saturable absorber field still
requires a coupled nonlinear solve; periodic boxes add cyclic dependencies.
Recomputing an equilibrium flow must not be confused with a time-dependent
front when different source histories compete for the same atom capacity.

## Prior work relevant to this direction

[TRAPHIC (Pawlik & Schaye 2008)](https://arxiv.org/html/0802.1715v2)
uses reception cones to merge photon packets and cap outgoing packets per
particle. Its paper gives source-number-independent transport scaling at fixed
spatial/angular resolution, tracks packet clocks and opacity, and demonstrates
shadow tests. Its SPH algorithm is not the finite-volume prototype here.

[ARTIST (Molaro et al. 2019, full v2)](https://arxiv.org/html/1901.03340v2)
is a directional shell-transport extension of SimFast21, with explicit photon
counts and finite propagation speed. It is not a native ES correction.
Section 5 reports 12,000 versus 20 single-core minutes for its comparison with
21CMMC, with 160 versus 80 redshift samples: 600 times total and 300 times per
sample in that published setup. That is evidence against assuming directional
shell RT is automatically a small ES overhead, not a prediction for this code.

## Reproduction

Run `angular_tests.py` for the donor-cell invariant, geometry and microbenchmark
suite; run `angular_high_order_tests.py` for matched-time beam and sphere
comparisons. Both compile small C/OpenMP kernels into a temporary directory.
JSON output contains measured errors, residual radiation and timing scope.
No production source file is modified or loaded by these solvers.
