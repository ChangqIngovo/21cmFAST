# Independent acceptance audit for the inexpensive-accuracy update

The existing event solver already has separately checked source budgets and a
coherent sharp-front event clock. The useful new acceptance gates concern
angular symmetry, unresolved front geometry, cavity sampling, and the weighting
of reported errors. This audit adds independent analytic probes rather than
repeating the original conservation suite.

## Findings

**A plane aligned with the Fibonacci polar axis is an unusually favorable
angular test.** The analytic intercepted fraction for a plane at distance 3.5
and a propagation range of 12 is 17/48 = 35.4167%, independent of orientation.
Across 133 normals, 24 Fibonacci directions produce 29.1667–41.6667%; at 768
and 3072 directions the worst absolute errors are 0.651 and 0.260 percentage
points. The old x-normal plane test alone cannot establish orientation accuracy.
These tests integrate the analytic spherical cap directly, so they do not mix
angular quadrature error with an oblique voxelized wall.

**Cubic rotation covariance remains a measurable defect of the default
quadrature.** Rotate the density and two unequal sources together by a cyclic
coordinate permutation, then undo the output rotation. The physical voxel
problem is unchanged. Normalized absorption differences are:

| Quadrature setting | L1 difference / emitted photons |
|---|---:|
| Fixed 24 | 122.67% |
| Fixed 96 | 24.39% |
| Fixed 384 | 6.65% |
| Existing adaptive, maximum 768 in this case | 4.16% |

Every source ledger still passes. A cubically symmetric angular rule is a
plausible cost-neutral remedy, but exact first and second angular moments alone
are insufficient: six axis directions satisfy those moments exactly while
having more than 10 percentage points of spherical-cap error. The quadrature
agent has been given both the symmetry and the multi-orientation cap gates.

**A small first-hit probe set is not a reliable upper bound on front extent.**
A transparent radius-5 core is extended to radius 17 inside a cone centered in
the largest angular hole of the probe set. With 24 probes, a continuum cone
covering 4.27% of the sky is completely missed; the resulting estimate of the
mean squared front radius is 31.1% too small. Actual voxel tracing confirms the
issue: the probes reach only 5.68 cells while a refined quadrature reaches 17.67
cells. With 12 probes the missed continuum fraction is 8.43% and the squared
radius underestimate is 47.1%. These are deliberately adverse geometries, not
statements about their frequency in cosmological fields. The sampling agent
has the reproducer to quantify the cost of rotated or progressive probe banks.

**Photon-weighted map L1 does not bound volume/fraction error.** A conservative
transfer in a grid with one low-density cell gives photon L1/Q = 1.11e-7 while
that cell changes by 0.7 in xHII and the volume-mean absolute fraction error is
1.09%. Report both count-weighted and fraction-weighted diagnostics. This is not
a reason to drop density from conservation; it is a reason to keep the two
weightings separate when assessing morphology.

**The independent comparison ledger has an overly permissive local tolerance.**
Its per-cell capacity allowance scales with total box luminosity. A 1e-8-capacity
cell can receive 2e-8 atoms and still pass when a distant source has budget 100.
The updated comparison gate and `strict_ledger` instead normalize the capacity excess by the local total atom
capacity, while retaining a separate global count test. This fixes a validation
loophole; the audit has not found that the event solver itself produces this
example. The criterion is invariant under changes in photon-count units.

## Review of the front-refinement prototype

The prototype's physical source position is preserved: the eight fine source
indices share the original coarse-cell centre as their geometric origin, and
each emits only the corresponding angular octant. The source weights and gas
capacities are divided by eight before reassembling coarse-cell absorption.
This avoids the half-fine-cell source displacement produced by naive nearest-cell
injection. Its matched coarse/fine comparisons must use the same angular rule.
An eight-sign/cyclic-permutation rule is not generally invariant under all cube
rotations for an arbitrary seed; the special repeated-component 24-point orbit
and a full signed-permutation rule do provide the relevant symmetry.

Global refinement still incurs eight times as many grid entries. A possible
subsequent acceleration is to instantiate fine cells only when incoming photons
reach them or their own source clears them. An untouched weak source cell has
analytically local absorption Q*t, so it need not be expanded immediately. This
is an implementation proposal, not an already measured speedup.

## Reproduction and acceptance

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research/pc_refinement/audit_accuracy.py
python -m pytest research/pc_refinement/test_audit.py -q --confcutdir=research/pc_refinement
```

Six independent tests pass. `audit_accuracy_results.json` records the angular
moments, cap integrations, missed-cone geometries, rotation test, and weighting
counterexample. Model selection should require improvements against independent
continuum/geometry oracles as well as high-angle references, and should measure
angular-reference uncertainty, local fraction errors, setup time, materialized
ray memory and the complete native coeval cost separately.
