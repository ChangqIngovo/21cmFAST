# Fixed-direction front transport needs an absorption clock

Research audit, 2026-10-23. This note specifies a restricted temporal model for
the experimental first-hit characteristic solver. It is not a claim that a
complete radiative-transfer or Qin20 coupling problem has been solved.
`event_front_audit.py` is an independent small-graph verifier: it uses exhaustive
event scans, not the candidate's C++ heap or spatial DDA. Its six test groups
include a directly embeddable three-dimensional six-axis fixture and 100 seeded
graph comparisons. All tests pass; results are in `event_front_audit_results.json`.

## 1. Model and approximation boundary

For the interval `[0,T]`, assume:

- Fixed source positions, fixed nonnegative emission rates, and a fixed angular
  quadrature. Source `s`, direction `b` has rate `r_sb=q_s w_sb`, with
  `w_sb>=0` and `sum_b w_sb=1`.
- Static nonnegative cell capacities. One absorbed photon pays one atom.
  Recombination, spectral hardening, scattering and hydrodynamic changes are
  disabled in this verification model.
- A cell remains completely opaque until its remaining capacity is exhausted,
  then becomes completely transparent. Subcell ionization structure is not
  resolved. Every beam retains its source, direction and geometric path.
- Infinite propagation speed through transparent cells. There is no in-flight
  radiation reservoir inside the represented domain. A photon stops at the
  first opaque cell on its own direction. Domain boundaries or a finite support
  cutoff have explicitly accounted outgoing/unused flux.

This is a legitimate **quasistatic, discrete-angle, opaque-cell sharp-front
approximation**. Infinite propagation speed removes the flight-time clock;
it does not remove the time required to supply enough photons to ionize a
foreground cell. Fixed snapshot photon increments can define piecewise-constant
rates `q_s=P_s/T`, but that temporal interpolation must be stated. Arbitrarily
clipped differences of cumulative, nonadditive source estimators do not become
physical emissivity merely by division by `T`.

Straight directions prevent a specific numerical error: bending a photon's
path or reassigning its angular budget. They do not remove angular quadrature
aliasing, cell-averaged chemistry errors or opaque-until-full closure errors.
"No artificial diffusion" is consequently too broad a claim for the entire
finite-grid ionization field. Angular and spatial convergence remain required.

Adaptive angular ray construction is an established transport approach [1].
This note derives the timing and bookkeeping of the present restricted model;
it makes no novelty claim for rays, event queues or front propagation.

## 2. Why full-budget synchronous hit rounds are not enough

Take beam A with ordered path `P -> C -> X`, and beam B with path `C -> Y`.
Both emit at rate 1 for duration 2. Capacities are `P=1`, `C=2`, and large
capacities at X and Y.

Full-budget hit rounds assign A's first photon to P and B's two photons to C.
In the next round A's leftover photon passes through the already full C to X.
The final absorption is `(P,C,X,Y)=(1,2,1,0)`.

With continuous emission, P fills at `t=1`; by then B has paid one atom in C.
C subsequently receives both rates and fills at `t=1.5`. The final half unit
of time sends 0.5 photons to X and 0.5 to Y. The correct discrete-model answer
is `(1,2,0.5,0.5)`. Two batches of duration 1 happen to recover it, whereas one
batch does not. Both batch answers obey every source, target and angular ledger.

An additional fixture with `P=0.91`, `C=1.07` shows that batch refinement error
need not decrease monotonically: two steps are better than three. Event timing
eliminates this artificial dependence for constant rates; merely adding more
synchronous hit rounds inside one photon batch does not.

The spatial fixture uses two voxel-center sources at `(0,0,0)` and `(1,0,0)`,
each with six axis directions of rate 1, total rate 6, for duration 2. Their
own-cell capacities are 6 and 12. The first opens at `t=1`; its +x beam then
joins the second source's local rate 6, so the second opens at `13/7`.
The common downstream cell `(2,0,0)` receives `2/7` photons, not the batch
algorithm's 1. This fixture tests spatial geometry and deferred source-ray
creation together without relying on a contrived non-geometric graph.

## 3. Continuous-rate event equations

For each beam `b`, let `j_b(t)` be its first still-opaque cell. A beam with no
such cell contributes to the declared outgoing/unused category. With residual
capacity `C_j(t)`, define

\[
R_j(t)=\sum_{b:j_b(t)=j}r_b.
\]

Between changes of the first-hit assignments, all `R_j` are constant and

\[
\dot C_j=-R_j\quad(C_j>0),\qquad
t_{\rm next}=t+\min_{j:R_j>0}\frac{C_j(t)}{R_j}.
\]

Advance to the earlier of `t_next` and the end of the interval. At a filling
event, mark the exhausted cells transparent and advance only the beams that
were waiting at those cells. Each finds its next opaque cell along the same
direction. The new target's rate changes at the same physical time. This
definition supplies a unique piecewise-linear solution; no arbitrary source
ordering is part of the model.

No geometric flight-time event is needed for the declared infinite-c limit.
If finite propagation speed is later required, delayed flux-arrival events
and an explicit in-flight photon reservoir are necessary. Adding an ad hoc
delay to cell filling while omitting stored photons is not the same model.

## 4. Conservation proofs

Let `A_bj(t)` denote the photons beam `b` has paid to cell `j`. Over a period
`[u,v]` during which `j_b=j`, set

\[
A_{bj}(v)-A_{bj}(u)=r_b(v-u).
\]

The beam has exactly one first absorber, or is in the outgoing category, at
almost every instant. Its disjoint residence intervals partition `[0,T]`.
Consequently

\[
\sum_j A_{bj}(T)+E_b(T)=r_bT.
\]

This proves every fixed angular channel's ledger. Summing directions yields
each source ledger `q_s T`; summing sources yields the global ledger. `E_b`
must be labeled escape only at a physical open boundary. At an imposed
allocation-radius cutoff it is unused/out-of-support budget, not automatically
physical escape. A periodic box similarly needs a defined path/cutoff policy;
leaving one periodic image is not physical escape.

For each cell,

\[
C_j(t)=C_j(0)-\sum_b A_{bj}(t).
\]

The next-event time prevents an update past zero capacity. Thus `C_j>=0` and
`0<=sum_b A_bj<=C_j(0)`. At filling, exact arithmetic gives equality. Changes
of target occur at instants of zero time measure and create or destroy no
photons. An individual beam may cross already transparent cells but never
pays an opaque downstream cell before its own foreground path is cleared.

These are exact statements for the finite ray/cell model. The angular channels
are quadrature representatives. Exact conservation in those channels does
not prove exact integration over every continuum angular cone.

## 5. Temporal composition, permutation and monotonicity

For unchanged capacities, source positions, angular directions and constant
rates, the event equations define an autonomous evolution. Pausing at any
time preserves the residual capacities and the same first-hit paths. Resuming
therefore gives the same solution as one uninterrupted update:

\[
\Phi_{t_2}(\Phi_{t_1}(C))=\Phi_{t_1+t_2}(C).
\]

The emission *rate* must be unchanged: dividing the photon increment and its
duration together is legitimate; reusing a whole-step increment in each
substep is not. Redrawing angular directions between substeps changes the
model, as do recombination, source motion and a changed emissivity history.
Reconstructing the exact same first opaque cells from the source is logically
possible, but preserving cursors avoids repeated traversal. Historical source
absorption records must still be retained if source-attribution diagnostics
are desired.

Permuting beams or sources leaves the rate sums and mathematical solution
unchanged. Floating-point sums may differ at roundoff level, so numerical
permutation tests need an explicit tolerance.

There is also a comparison property. Hold all paths and capacities fixed and
increase one or more nonnegative beam rates. Ionized atom counts cannot
decrease at any cell. To see this, suppose a first downward crossing of the
lower-rate solution occurs at a not-yet-full cell. Every beam reaching that
cell in the lower-rate solution has all its predecessors full. Those
predecessors are also full in the higher-rate solution before a first
crossing. The same beam therefore reaches this cell, at an equal or larger
rate, contradicting a downward crossing. Saturated cells stay saturated.

This property is restricted to the fixed-gas, no-recombination, fixed-path
model. It is not a theorem about coupled radiative feedback or changed source
histories. The executable tests temporal composition and rate monotonicity on
100 seeded small graphs in addition to the analytic fixtures.

## 6. Heap implementation with lazy state updates

Each active cell can store its residual capacity at time `t_last`, its current
incoming rate `R`, and a scheduled filling time. To inspect or change its rate
at global time `t`, first materialize

\[
C(t)=C(t_{\rm last})-R(t-t_{\rm last}).
\]

Then change `R`, set `t_last=t`, and reschedule `t+C/R`. Other cells do not need
to be updated at every event. A minimum heap of filling times provides the
next event; stale records must be discarded using cell generation/version
numbers or avoided using a true decrease-key structure.

For source/angle diagnostics, an individual constant-rate ray needs only its
current target-entry time. When that target fills, the ray's absorbed amount
there is `r_b*(t_fill-t_entry)`. An unrelated ray entering the same cell does
not change this ray's own rate, so all existing rays need not be touched on
every rate change. At the final time, unfinished residence intervals contribute
the corresponding partial absorption. This also avoids a dense source-by-cell
attribution matrix, which the tiny independent verifier uses only for clarity.

At a mathematical tie, all cells scheduled at the same `t` are exhausted before
the tied rays are advanced. Newly encountered zero-capacity cells are skipped,
and all zero-time consequences are resolved before global time increases.
Processing one tied event must not let a ray pay a cell that was already
exhausted at that same time.

In floating point, a broad time tolerance must not silently declare a later
event full early. Its positive remaining capacity would otherwise be unpaid.
Prefer exact minimum-time processing plus residual checks and zero-time
completion. If numerical tolerances skip or fill a tiny residual, expose and
bound that atom-count discrepancy; do not simultaneously claim exact real
arithmetic equality. Include exact ties, near ties, tiny capacities, disparate
rates, stale heap entries and source permutations in implementation tests.

## 7. Deferred creation of weak-source rays is exact in this model

Before an opaque source cell opens, every direction emitted inside it has
that same cell as its first absorber. All its directions can therefore be
replaced by one local scalar rate `q_s`. External beams contribute separately
to the same cell's event rate. If the cell opens at time `tau_s`, its source
has paid `q_s*tau_s` locally; its angular attribution is implicitly
`w_sb*q_s*tau_s`.

Only then instantiate the angular rays, with their original source position,
fixed directions and rates `q_s*w_sb`, and advance them out of the now
transparent source cell. This gives exactly the explicit-ray model's future
fluxes. It is a compression of identical first absorbers, not an approximation
that isotropizes radiation from different positions or moves the source.
An initially transparent source cell needs immediate ray creation.

The executable confirms this for weights `(0.1,0.2,0.7)`, source rate 1 and
source-cell capacity 0.4: activation is at 0.4; local angular absorption is
`0.4*w`, and downstream absorption by time 1 is `0.6*w`.

This removes early allocation of `N_source*K` ray records when most emitting
cells remain opaque. It does not remove that worst-case memory cost after
many source cells have opened. Late-time source activation and peak memory
must be measured, not inferred from a fast mostly neutral snapshot.

## 8. Work and unresolved accuracy

With static capacities and no recombination, each cell fills at most once.
The number of distinct nonzero-capacity filling events is therefore at most
the number of initially unfilled cells, although one event time can fill
several cells. This bounds the events, not the number of interacting rays.

Let `M` be all geometric ray traversal operations, `H` all newly encountered
opaque ray/target incidences, and `N_f` the number of active front cells. With
preserved cursors and local target updates, a useful complexity accounting is

\[
O(N_{\rm cell}+N_{\rm source}+N_{\rm rays}+M+H\log N_f).
\]

Loose worst-case estimates can write `H` as `N_source*K*L_hit`. Transparent
block skipping can reduce `M` if a block is **certifiably** paid/transparent;
it does not alter the event equations. A cell-level opaque clump must prevent
such skipping. A lazy heap may hold `O(H)` historical records unless stale
entries are periodically removed; a decrease-key heap can restrict the active
queue to `O(N_f)`. Ray storage remains `O(N_active_source*K)` after materialization.

Remaining physical/numerical work includes angular convergence, subcell front
error, source-cell discretization, domain/support conventions, recombination
and spectral absorption, rates from native 21cmFAST fields, thermal/Gamma
coupling, and end-to-end runtime. The exact event-clock result does not prove
that a finite angular resolution is sufficient, or that total cost will remain
within two or three times an ES run.

## Reference

[1] Abel & Wandelt (2002), *Adaptive Ray Tracing for Radiative Transfer around
Point Sources*, https://arxiv.org/abs/astro-ph/0111033 .

The conservation, temporal composition and comparison arguments here concern
the explicitly defined discrete model. The independent graph code verifies
their finite fixtures; it is not a substitute for testing the geometric
characteristic implementation against those fixtures.
