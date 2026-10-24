# A bounded angular approximation for faint sources

## Established statement and assumptions

Consider a fixed finite voxel domain with nonnegative initial neutral capacities
C_i. Each photon direction follows a fixed straight path, ending at the chosen
open boundary or finite propagation horizon. A voxel remains opaque until its
capacity is exhausted and then stays transparent. Propagation is instantaneous.
Sources emit nonnegative prescribed rates; the implemented tests use constant
rates over a unit interval. Source positions, initial capacities, time interval,
and all ray paths of a selected strong-source set are identical between runs.

Let A_i be the final absorbed atom count in cell i. Separate sources into strong
S and weak W, with weak emitted budget Q_W. Two calculations may use arbitrary
different angular quadratures for W, provided each retains its original total
source rate and the strong quadratures remain unchanged. Then

\[
\boxed{\sum_i|A_i^{(1)}-A_i^{(2)}|\le 2Q_W.}
\]

This bounds **photon-count-weighted map error in this discrete sharp-front
model**. It does not bound a volume-weighted ionization-fraction error, a power
spectrum, or error relative to continuum radiative transfer. It also does not
apply when recombinations reopen cells, gas moves, feedback changes strong
source rates, or a changed geometry recomputes the strong angular plans.

## Proof

First compare the strong-only calculation, with absorption A_i^0, to a run with
any additional nonnegative weak emission.

**1. No cell exhausts later after adding photons.** Consider the earliest
strong-only exhaustion event at which this ordering could fail. Any strong ray
that contributes a positive amount to that event has traversed a prefix of cells
that exhausted strictly earlier. By induction those prefix cells are already
transparent at equally early or earlier times in the augmented calculation.
The unchanged strong ray therefore reaches the cell no later. Its integrated
contribution up to the old exhaustion time is at least as large, and additional
weak contributions are nonnegative. The augmented cell must already have
exhausted, a contradiction. Initially transparent cells supply the induction
base; simultaneous events deliver zero extra photons at the instant of arrival
and do not alter the argument.

**2. Final absorption is pointwise monotone.** If a cell exhausts in the augmented
run, its absorption is its full capacity and cannot be smaller than in the base
run. If it does not exhaust, every contributing old strong ray arrives no later
and is absorbed there for no shorter an interval. Thus A_i >= A_i^0 in all cells.

**3. The already-present strong photons cannot have greater total absorption.**
For a particular strong ray with constant rate q, list the cells along its finite
path, including its source cell. Let M be the largest exhaustion time among
these cells, with infinity if one never exhausts. Until M, every emitted photon
on that ray is absorbed somewhere on the path; after M it leaves the modeled
support. Its total absorption by time T is therefore

\[
A_{\rm ray}=q\min(T,M).
\]

Because every cell exhausts no later, M cannot increase and neither can this
strong ray's total absorption. The source-cell local phase is simply the first
cell in this path; it is not an exception to the argument. Periodic revisits do
not create additional capacity: retain only the first occurrence of each cell
when applying the formula. For prescribed time-varying rates, replace q min(T,M)
by the integral of the same nonnegative rate up to min(T,M).

Consequently, the net increase in total absorption comes from at most the weak
emitted photons:

\[
0\le\sum_i(A_i-A_i^0)\le Q_W.
\]

Both alternative weak quadratures obey this inequality relative to the same
strong-only calculation. The triangle inequality then gives

\[
\sum_i|A_i^{(1)}-A_i^{(2)}|
\le\sum_i(A_i^{(1)}-A_i^0)+\sum_i(A_i^{(2)}-A_i^0)
\le2Q_W.
\]

The factor two is sharp: an isolated weak source in a transparent cavity can put
its budget Q_W onto either of two disjoint opaque targets under two one-ray
quadratures. The absorption maps differ by exactly 2Q_W while each preserves its
source budget.

## Verification and actual savings

`audit_theory.py` invokes the unchanged event core with explicit strong/weak
angular banks. Forty-eight randomized geometries passed all three intermediate
inequalities and the final bound. They include partial and transparent cells,
open and periodic boundaries, finite horizons, one/six/twelve weak directions,
and weak budgets scaled down by 1e-4. A separate regression reaches the exact
2Q_W bound. Four theorem-specific tests pass.

On the existing frozen native 32^3 source field, reducing selected faint sources
from 96 to 6 directions gave:

| Weak budget fraction | Selected sources | Materialized ray reduction | Measured map L1/Q | Guaranteed bound |
|---|---:|---:|---:|---:|
| 0.10% | 154 | 0.11% | 0.00098% | 0.20% |
| 0.50% | 656 | 0.46% | 0.00388% | 1.00% |
| 1.00% | 1208 | 0.76% | 0.00585% | 2.00% |
| 5.00% | 4727 | 3.35% | 0.02609% | 10.00% |

The final row took about 1.50 s versus 1.41 s in one timing trial. The analogous
second-half experiment saved similarly few rays. Most faint source cells never
launch rays, so coarsening their planned quadrature does not remove much actual
work. The theorem is useful as a certificate but does not establish a worthwhile
speedup on this source field. No production default is changed.

## A tighter outgoing-budget proposal, not implemented or validated here

A potentially useful common calculation would retain each weak source's local
ionizations but truncate its rays when they leave the source cell. The two full
angular solutions should both dominate this truncated calculation. Repeating
the argument suggests replacing Q_W by the weak photons released beyond their
origins in that common run.

A full run clears weak origins no later than the truncated run. Its measured
outgoing budget

\[
B_W=\sum_{s\in W}Q_s\,[1-t_{{\rm launch},s}/T]_+
\]

could therefore provide the conservative certificate 2B_W. This is potentially
much smaller than 2Q_W for late-launched sources. A future implementation could
record launch times or debit an error allowance when choosing a cheaper angular
rule at activation. Even if the subset W is selected dynamically, a post hoc
comparison can condition on its realized membership, provided the full reference
and the coarse run retain identical preassigned angular plans for every strong
source. Recomputing strong quadratures from the changed evolving geometry would
invalidate that premise.

This tighter construction has **not** been implemented or tested in this turn.
It is a next-step proposal, not a certified option in the delivered solver.

## Reproduction

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research/pc_refinement/audit_theory.py \
  --native /path/to/source_capacity.npz
python -m pytest research/pc_refinement/test_audit_theory.py -q \
  --confcutdir=research/pc_refinement
```
