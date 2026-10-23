# Independent verification of the event-front implementation

Validation performed 2026-10-23. The repaired implementation passes the new
independent finite-model comparisons. The audit found two classes of numerical
defect before repair; it did not merely repeat the earlier accounting tests.

Verified code hashes:

| File | SHA-256 |
|---|---|
| `event_front.cpp` | `d97be4a8bd20132d8ed6da7314e1bef1d148c0346435d91d49fb2a36aa628fd8` |
| `event_front.py` | `1fea8965df8f480d06857a01b15b08e8ad03741420552daa7eda5dac6e374688` |
| `characteristics.py` | `15b87ad85050658fe5dda0ebc3fc2615c4bf430af4b0320320c96dd7c3c8dc26` |

Hashes were checked before and after the final reference run and did not change.
Results are recorded in `validation_event_reference_results.json`.

## An independent reference

`validation_event_reference.py` does not call the candidate's clearance,
traversal, event queue, weak-source compression or absorption reconstruction.
It first constructs every fixed ray's full ordered path with ordinary DDA:
integer plane indices, absolute plane-crossing distances, no position nudges,
no clearance jumps, and simultaneous advancement only for equal crossing
parameters. The source cells are explicit members of each path. Periodic
paths keep unwrapped indices and may revisit the same physical cell.

The temporal reference then scans all rays and all cells at every exhaustion
event. It creates every source-angle channel immediately, computes aggregate
incident rates explicitly, and accumulates actual absorbed photons directly.
It uses NumPy long-double arithmetic for its event bookkeeping and geometric
parameters. On this runtime `np.finfo(longdouble).nmant=63`; it remains a
finite-precision numerical reference, not exact symbolic arithmetic.

Separate hand-derived tests establish its axis paths, negative and positive
corner crossings, periodic repetitions, and the analytic two-source six-axis
fixture with source-cell opening time `13/7`. These checks reduce the risk of
comparing two implementations that share the same unnoticed convention error.

Every candidate ionization cell is compared with the reference. Source
absorption, escape and retained-budget arrays are also compared. Each source's
error and ledger are normalized by its own emitted budget, so a bright source
cannot hide a dim source's error.

## Defects found and repaired

The initial core hash was
`aca74136b8383b7202903a4570cc78473feb4ae4f487a832b758cb2db9426c19`;
the initial Python wrapper hash was
`c2c3817b9795e84ba645f181acfc75957d6fbd3ae493ccf150c0314abfab0902`.
This audit supplied the following minimal reproductions to the designated
core editor. The reference and tests were maintained separately from that edit.

1. **Weak absorption disappeared from the returned cell field.** A one-cell
   box with emission `0.001`, capacity `1e12`, and duration 1 returned absorption
   `0.0009765625`, while its source ledger correctly recorded `0.001`. The
   discrepancy was 2.34375% of all emitted photons. At capacity `1e20`, the cell
   field lost the complete `0.001` increment. Computing absorption as initial
   capacity minus remaining capacity caused cancellation. The repair records
   absorbed increments directly with compensated summation and retains a
   compensated remaining capacity. Regression tolerances were not loosened.
2. **The geometric `1e-10` position nudge changed first-hit decisions.** A
   transparent one-cell box has a +x exit at distance 0.5. With horizon
   `0.5+5e-11`, the old core nevertheless retained all photons because its
   artificial overshoot exceeded the horizon. More seriously, in a five-cell
   box a source at `(1,1,1)` with direction proportional to
   `(1,1+1e-10,0)` first crosses the positive-length opaque cell `(1,2,1)`.
   The old nudge skipped this thin intersection and paid `(2,2,1)` instead.
   Both global and source ledgers still closed. Offsets `1e-12` and `1e-14`
   reproduced that opacity error. The repair uses integer unwrapped voxel
   cursors and long-double absolute crossing parameters, with no fixed nudge
   or broad near-corner tie tolerance.

The repaired clearance jumps remain conservative: a chessboard clearance of
`d` guarantees the current cell's surrounding `d-1` layers are transparent.
A path advance of `d-1` cannot cross an unverified voxel first. Open-box padding
also prevents a periodic wrap from being mistaken for a transparent shortcut.
The endpoint remains on the original ray, and the final ordinary DDA crossing
uses the strict integer-plane rule. The independent ordinary paths verify the
result without using this optimization.

The strict support convention is now explicit: an opaque target needs
`entry_distance < horizon`, giving a positive path interval inside it. Entry
exactly at the cutoff is retained budget. Open-box escape occurs when the
exit distance is `<= horizon`. Crossing a periodic image boundary is never
counted as physical escape. Equality at a sharp cutoff is a convention;
the old failures with a positive margin fail either reasonable equality rule.

## Final results

| Independent comparison group | Passed | Largest voxel error / total emitted photons | Largest source error / that source's budget |
|---|---:|---:|---:|
| Strict horizons, weak absorption, corner slivers, periodic loops, ray-storage boundary | 42/42 | `5.78e-16` | `8.50e-15` |
| Random small-grid source/capacity configurations | 100/100 | `1.16e-16` | `7.71e-16` |
| Exact/near event ties, unit scaling and mixed capacities | 24/24 | `3.27e-17` | `2.03e-16` |

The 100 random configurations use grid lengths 2--6, one to four sources,
five or fifteen directions, random capacities and transparent cells, open
and periodic boundaries, varying duration and varying support radius. Fixed
axis and diagonal directions are mixed with random directions. The other
fixtures include:

- Capacity up to `1e20` with source emission `0.001`.
- Global source/capacity unit scaling by `1e-120` and `1e120`.
- Local capacities spanning `exp(-25)` to `exp(25)`.
- Exact and near simultaneous cell exhaustion.
- Horizons just before, at and just after cell/box faces, including offsets
  smaller than the removed `1e-10` nudge.
- Positive and negative near-corner rays that intersect a very thin opaque
  voxel before the diagonal neighbor.
- Empty periodic paths that repeatedly traverse the box.
- 4,617 explicit rays, crossing the implementation's 4,096-record storage
  block boundary while preserving target lists and source accounting.

The comparison acceptance tolerance was `2e-11` of the emitted budget, with
an additional `2e-11` per-source relative check. The observed errors above
were much smaller. These floating-point errors describe agreement with the
same finite ray/voxel model; they are not continuum physical accuracy estimates.

All **12 pytest tests pass**, including the hand-derived reference checks and
the additional horizon-bound checks below.
Ruff checks pass for the two new Python files.

## Source histories and temporal composition

Five history comparisons pass: a constant-rate interval, the same interval
split into four unequal durations, A then B, B then A, and an on/off/on history
with changed rates. Both implementations advance the same remaining capacities
between intervals, using the same physical source locations and directions.

For unchanged rates, one interval versus four unequal intervals differs by
at most `1.11e-16` per cell. Each changed-rate history independently agrees
with the full-scan reference. The largest history-comparison cell discrepancy
is `1.11e-16`; the largest source absorption discrepancy is `3.55e-15` photons
in these fixtures.

Reversing A/B emission order changes the atom map by L1 `5/3`, despite equal
integrated source photons. This is expected: different temporal histories can
pay a shared front with different sources and send later photons in different
directions. The correct invariance is splitting the **same** history, not
discarding its ordering. The current wrapper recreates fixed rays between
intervals; that is equivalent here because propagation is instantaneous and
the same rays' first remaining opaque cells determine future absorption.

## A useful error certificate for the numerical horizon

The finite horizon can be controlled in this frozen, no-recombination model
without first implementing a general radiation reservoir. Suppose every short
ray path is a prefix of its longer counterpart, with identical source rates,
directions, initial capacities and open/periodic boundary convention. Let
`A_short,j` and `A_long,j` denote absorbed atoms, `R` retained out-of-support
photons, and `E` physical open-boundary escape.

Then, in exact arithmetic,

\[
A_{\rm long,j}\ge A_{\rm short,j},\qquad
\sum_j|A_{\rm long,j}-A_{\rm short,j}|
\le R_{\rm short}.
\]

For the first inequality, suppose there were a first downward crossing at a
not-full cell. Every beam reaching that cell in the short solution has already
paid all its predecessors along the common prefix. Before a first crossing,
those predecessors are also paid in the long solution. The beam therefore
reaches this same not-full cell at the same rate there; added downstream paths
can contribute additional rates but cannot remove the common ones. A downward
crossing is impossible. Fully paid cells remain paid.

The global ledgers now give

\[
\sum_j(A_{\rm long,j}-A_{\rm short,j})
=R_{\rm short}-R_{\rm long}-(E_{\rm long}-E_{\rm short}).
\]

For periodic paths `E=0`. For open paths, a beam that escapes in the short
solution has the same path to the physical boundary in the long solution,
whose intervening cells can only clear earlier. Thus
`E_long>=E_short`, proving the same upper bound. Cellwise monotonicity makes
the signed total change equal the atom-weighted L1 change. Because absorption
is bounded by the finite cell capacities, the bound also applies to the
monotone infinite-horizon absorption limit, without requiring a numerical
traversal of an actually infinite periodic ray.

A slightly tighter bound is

\[
\|A_{\rm long}-A_{\rm short}\|_1
\le\min\left(R_{\rm short},\sum_j(C_{\rm initial,j}-A_{\rm short,j})\right).
\]

This handles a fully paid periodic box: retained photons can remain large,
but there is no further absorption error. In a partially neutral frozen step,
`R_short/Q < epsilon` is therefore a sufficient certificate that extending
the horizon changes the atom map by less than `epsilon Q` in L1. Increasing
the horizon until that criterion holds is a legitimate controlled
approximation for this particular missing piece. A large retained fraction
is a conservative uncertainty bound; it need not imply a large actual error.

The executable adds **25** comparisons: one analytic chain saturating the
bound and 24 random open/periodic cases, each checked against the independent
path/event reference. All pass. No negative cell change and no bound excess
were observed; the largest additional reference discrepancy was
`3.07e-17` of emitted photons. The analytic chain retains half the short-run
photons, and the longer run adds exactly that half to absorption.

This certificate concerns atom counts, not an unweighted volume-average
ionization fraction or a relative power spectrum. Converting to those metrics
requires density/field bounds or direct comparison. It also does not justify
discarding photons in a later recombining or feedback-coupled history: changes
in capacity, rates, opacity or source locations invalidate this comparison.
Source-by-source absorbed counts need not increase even when every cell's
total absorption does; the proof only asserts the stated cell and global bounds.

## What has and has not been verified

**Passed:** independent cell-by-cell implementation agreement for the tested
finite model; open/periodic boundary classification; strict finite horizons;
first-opaque-cell ordering including near-corner intersections; weak absorption
precision; same-history temporal composition; specified changed histories;
source attribution and budget accounting in these fixtures; and the stated
retained-budget bound under nested horizon extension.

**Found, repaired, and covered by regressions:** cancellation of weak cell
absorption; horizon misclassification; and missed opaque sliver cells caused
by the same geometric nudge. No unresolved mismatch remains in this suite at
the recorded hashes. This is a finite test set, not a proof of every possible
floating-point configuration or extreme grid size.

**Model assumptions remain:** centered point sources; fixed finite angular
quadrature; opaque-until-full cells; static gas; infinite propagation speed;
constant rates within each supplied history interval; no recombination,
scattering, multifrequency absorption or evolving thermal feedback. A source
history must supply physical nonnegative emission rates; the test does not
construct them from cumulative 21cmFAST estimators.

**Remaining physical work is not closed by this audit:** angular and spatial
convergence against an appropriate transport reference; recombination and
thermal/Gamma coupling; native Qin20 source and snapshot-state integration;
and full PC/non-PC runtime, memory and cosmological-output comparisons. Passing
the discrete solver comparison does not establish that a particular `K`, grid
size, timestep, or opacity closure is scientifically accurate or sufficiently
fast for Qin20. Those are separate validation stages.

## Reproduction

From the repository root with NumPy, SciPy, pytest and a C++17 compiler:

```bash
python research/pc_alternatives/validation_event_reference.py --random-cases 100
python -m pytest -q research/pc_alternatives/test_validation_event_reference.py
```

The result writer records source hashes and fails if the code changes during
the run or any comparison fails. It preserves minimal original failure values
alongside the final passing results.
