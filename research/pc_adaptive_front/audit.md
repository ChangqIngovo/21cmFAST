# Independent audit: exact lazy eight-subcell refinement

## Scope and useful distinction

The proposed adaptive solver is a different representation of the existing
uniform eight-subcell model, with the same photon sources, directions, gas
capacities, finite path horizon and infinite propagation speed. It can avoid
creating child state until it is needed. This changes neither the prescribed
source budget nor the physical opacity rule. Equality to the full eight-subcell
solver is the appropriate implementation oracle; continuum sphere comparisons
separately measure spatial-model error.

The result below assumes static equal-density children, nonnegative prescribed
constant local source rates, irreversible cell exhaustion, fixed straight-ray
quadratures, and no recombination or gas advection. It is a model-relative state
compression, not a continuum radiation-transport theorem.

## When a parent may remain coarse

A parent has total initial remaining neutral capacity C and a central source
of rate q. Its eight children have capacities C/8 and receive q/8 from the local
isotropic source. This requires the discrete angular rule to preserve exact
equal octant budgets, as the current octant-balanced rule does; continuum
isotropy alone is insufficient for a finite unbalanced direction set. Before
the first materialized ray reaches any child, their
states are identical:

\[
A_\alpha(t)=\frac{\min(qt,C)}8,
\qquad
C_\alpha(t)=\frac{\max(C-qt,0)}8.
\]

Therefore the scalar parent can evolve analytically until the earlier of:

1. its local exhaustion time C/q, or
2. the first transported ray arrival at any child.

If q=0 there is no local exhaustion event. If C=0 the parent is transparent from
the start. An initially uniform partial ionization is handled by defining C as
the *remaining* capacity, not the total atom count.

If local exhaustion occurs first, all eight children clear simultaneously. One
parent event can replace their eight identical-time events, and all source
angular octants launch from their original common physical point.

If a ray arrives first at t*, synchronize local physical absorption to t*, then
initialize eight children with their appropriate initial capacities, local rates
q/8, absorption min(q t*,C)/8, and last-update times t*. Attach the arriving ray
only to the geometrically entered child. Refinement must occur before adding
that directional rate; averaging it across the parent changes the model.

A materialized ray from the same source after periodic wrapping is also such an
arrival. Source identity alone must never bypass refinement.

## Equivalence argument

Between events, the unrefined parent and the eight uniform children have exactly
the same linear trajectories after summation. At promotion, each reconstructed
child state equals its full-grid counterpart. From then onward the adaptive
solver uses the same child-level rate additions, capacity exhaustion events and
ray paths. An event induction therefore preserves the same absorption map and
source/angle escape and retention ledgers in exact arithmetic.

The local source budget must be attributed once. One valid implementation keeps
physical absorption and source attribution separate: transfer physical A/8 to
children on promotion, but credit each child's local source q/8 times its final
clearing time (or duration if uncleared) once, as the full fine solver does.
Only parents that clear without promotion credit the scalar q times clearing
time. Crediting q t* on promotion *and* q/8 times the full later clearing time
would double-count the pre-promotion local photons.

Equal-time arrivals and clearings carry no finite photon amount at the instant
of the event. They may be processed in a deterministic order, provided zero-time
forwarding is completed and no child with positive remaining capacity is skipped.
Floating-point event order can differ from the uniform solver, so numerical
agreement should be measured with stringent relative/local tolerances rather
than promised as bitwise identity without tests.

## When child state is necessary

An initially nonuniform child-ionization state must be expanded before evolution.
A mean parent ionized fraction does not identify which subvolumes are transparent.
Equal total absorption is insufficient to reconstruct the path of the next ray.

A partially ionized parent may remain scalar only while all child states are
known to be identical and no directional incoming rate distinguishes them. Once
promoted, preserve the individual child states across time steps unless all
children are transparent. Replacing a nonuniform child state by eight copies of
the parent average changes front geometry even though total photons are unchanged.

Uniformly partial, never-promoted source-only parents need no persistent child
array. This is especially useful when many faint source cells ionize locally
without sending photons outward.

## Safe transparent-parent coarsening

When all eight remaining capacities are zero, future transport sees the parent
as transparent in this monotone-opacity model. A coarse parent crossing can
replace child crossings without changing absorption. Its ray origin, direction
and absolute travelled distance must stay unchanged.

A transparent-parent DDA must retain exact face ordering. Computing a rounded
endpoint and nudging it through a nearby second face can skip a thin opaque
neighbor. At a parent exit, use explicit long-double face-time comparisons and
midpoint comparisons for non-crossed axes, including negative unwrapped periodic
coordinates. Preserve the existing convention that a cell first entered exactly
at the horizon receives no positive-length interval.

State reclamation has additional software conditions:

- Advance every ray formerly attached to the exhausted children.
- Release or invalidate all pending child events before reusing their IDs.
- Preserve per-source and per-angle ledger ownership independently of child IDs.
- Do not reuse a child block while the current event handler still holds a
  reference into it; defer reclamation until that handler has completed.

Report both currently active children and peak allocated storage. If cleared
blocks remain allocated forever, the method has lazy allocation but has not
achieved active-front-only storage.

## Required counterexample/regression tests

1. **Uniform local sources only:** partial dose leaves every child identical;
   source-only exhaustion should need one parent event instead of eight.
2. **Incoming illumination before local exhaustion:** compare all child states,
   not merely parent sums; include q=0 and weak positive q.
3. **Simultaneous local and incoming events:** place equal sources symmetrically;
   confirm source, angle, cell and global ledgers and ray paths.
4. **Initial mixed child state:** a clear child beside an opaque child must pass
   and absorb different incoming rays despite an identical parent mean.
5. **Stateful time subdivision:** one constant-rate step versus multiple steps
   carrying sparse child states and retaining fixed angular directions.
6. **Periodic source return:** own-source rays that wrap back are normal rays;
   translation should preserve the result.
7. **Edges, corners and thin intersections:** nearly tied face crossings must not
   skip a positive-length opaque child or neighboring parent.
8. **Finite horizon and open boundary:** separate escaped and retained photons;
   test entry exactly at the horizon.
9. **Tiny positive capacity and count-unit rescaling:** no absolute opacity
   threshold may erase a small neutral capacity; local tolerance scales with
   local atom counts, not the global photon budget.
10. **Repeated refine/clear/reuse:** stress memory-pool reuse and event IDs with
    many small advancing fronts; active storage and peak allocation must agree
    with the reported memory model.

The adaptive refinement criterion is an actual ray arrival, rather than a guess
from a previous coarse front. This avoids delayed-refinement errors and means
no user-chosen front-width buffer is needed for equivalence to the full fine grid.
Its speedup still depends on how many parents receive nonlocal illumination and
how much work is spent tracing rays; equality does not by itself ensure speed.

## Independent state-retention counterexample

`audit_state_check.py` uses two sources (120 and 45 photons), a uniform 24³ gas
field and 1536 fixed directions per source. It compares one full emission step
with two half steps using the existing uniform eight-subcell solver:

| Continuation state | Final coarse-map L1/Q versus one step | Maximum coarse xHII error |
|---|---:|---:|
| Preserve all child ionizations | 2.30e-17 | numerical roundoff |
| Replace children by parent mean | 1.979% | 0.08203 |

All three runs absorb exactly the same total photons and retain none. The first
half creates 115 nonuniform parents. Thus photon closure alone cannot validate
coarsening a partial front. A sparse adaptive implementation must retain those
mixed child states; it can still represent every uniform parent by a scalar.
The experiment is a fixed-gas model check, not cosmological time convergence.

## Independent implementation review and randomized checks

The implemented event loop keeps a child block alive until every opaque child
has processed its event and forwarded its attached rays. It then removes all
child events before releasing the slot. Child blocks have stable addresses while
other parents are promoted. Initial transparent source children emit only their
own octants, and promoted-child local source attribution is credited once from
the interval origin. No remaining correctness issue was found in these paths.

One state-continuation defect was found and fixed during review. If a parent is
almost fully ionized, `C - sum(child_ionized)` and
`sum(C/8 - child_ionized)` can differ appreciably *relative to the tiny remaining
capacity*, although their absolute difference is ordinary floating roundoff.
A valid example had coarse remainder 3.6004533e-13 and child sum 3.6000369e-13;
the native consistency check rejected it. After checking that parent and child
counts agree within the local atom tolerance, the wrapper now uses the child
remaining-capacity sum as the authoritative parent remainder. The regression
includes a zero-source continuation so no physical evolution can hide the issue.

`test_adversarial.py` contains 15 additional tests: this near-transparent state,
initial clear source octants, and randomized mixed child geometries. All passed.
`audit_fuzz.py` ran 48 further deterministic realizations with uniform or sparse
mixed initial parents, almost-transparent states, open/periodic boundaries,
short and multiple-box horizons, 24/96 directions, and 0.1–3 photons per atom.
The maximum fine absorption-map L1/Q against the full eight-subcell solver was
1.82e-18; maximum coarse ionized-fraction difference was 1.85e-16. Inputs and code
hashes are retained in `audit_fuzz.json`.

Persistent ionization values still have floating-point resolution. For example,
adding 1e-20 photons per cell to a stored ionized count of 0.5 is below one double
ulp: the separately returned absorbed increment is retained, but ordinary
`I_old + absorbed` cannot change the stored scalar. This limitation is shared
with the full fine-grid implementation. Exact arbitrary weak-tail accumulation
onto nonzero counts would need a persistent compensation field; it is distinct
from the resolved spatial-state loss demonstrated above.

## Final default-kernel and wrapper review

The delivered kernel adds compact integer ray records, a four-child event heap,
and local DDA traversal inside an already refined parent. The local traversal
retains absolute long-double face times, updates only axes that crossed, and
returns parent exits to the original open/periodic boundary logic. It introduces
no reciprocal-step accumulation or change to near-tied face ordering. Child-slot
reclamation and local-source attribution retain the audited semantics above.

The final wrapper uses remaining-neutral density for angle prediction by default;
controlled full-fine equivalence comparisons hold direction counts fixed. Its
fully transparent periodic shortcut is exact: there are no absorbers or open
boundaries, so every new photon remains in the separately reported radiation
ledger. The returned state carries gas ionization only, not a hidden re-emission
of those retained photons.

After the default kernel was frozen, all 15 independent adversarial tests and
48 randomized cases were rerun. Their maximum map differences remained
1.82e-18 in fine L1/Q and 1.85e-16 in coarse ionized fraction. The refreshed JSON
records the final source hashes. No practical blocker remains in this review.
