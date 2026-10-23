# Local radial redistribution and hierarchical accounting

Research only. The production global-budget method is unchanged. Inputs here
are photon **increments** and residual atom capacities, not 21cmFAST's
independently conditioned collapsed-fraction estimates at different radii.

## What was implemented

`hierarchy.py` contains two independent count-conserving closures.

1. `radial_redistribute`: every source independently fills increasing-distance
   shells. Complete shells receive their hydrogen demand. In the terminal
   shell, the default gives equal photons to each tied-distance cell, following
   the rule seen in the public SCRIPT source. Provisional cell claims can
   exceed capacity. Every overclaimed cell then simultaneously redistributes
   its excess to currently neutral cells in increasing-distance order; a
   terminal correction shell receives fractions proportional to remaining
   capacity. All correction sources see the same frozen state in each round.
   A capacity-proportional first-shell variant is retained for comparison.
2. `dyadic_pool`: nonoverlapping cubic blocks successively double in side
   length. Each block exchanges only the minimum of its residual photon and
   atom totals. Debits and absorptions are proportional to their positive
   residual counts. This deliberately changes spherical geometry.

The first is a transparent Python reconstruction of the broad modern SCRIPT
algorithm. It is **not a run of the public optimized Fortran implementation**.
The exact numerical counterexamples below belong to this reconstruction.
It adds an explicit unspent reservoir, floating-point stopping tolerance,
iteration cap and per-hop support radius; these differ from the public code.
Both methods return capacities, absorbed photons and remaining photons.
Every call independently audits their nonnegative global ledger.

For radial redistribution, provisional claims satisfy a conserved sum.
Clipping claims to capacity stores the excess, and each relay divides rather
than duplicates that excess. Thus the sum of absorbed and remaining photons
equals the input. A tracer could attribute relayed photons back to sources,
but the prototype does not retain those labels. The relay position and
distance are not the original emitter position and distance. Neither count
closure supplies a directional photon budget or an optical-depth calculation.

The tree proof is local: within each disjoint node, equal photon and atom
amounts are removed. Nested levels therefore cannot double-spend a photon or
atom. This proves accounting; it does not prove spherical morphology.

## Measured results

The complete machine-readable record is `hierarchy_results.json`, reproduced
with `hierarchy_tests.py`. All listed photon ledgers close to floating-point
roundoff and all absorbed counts remain within cell capacity.

| Probe | SCRIPT-like radial closure | Dyadic closure |
|---|---:|---:|
| Isolated 257-photon source, unit density | Exact lattice radius-4 sphere | Cubic layers; centroid displaced 3.2165 cells along every axis |
| Equal and unequal overlapping sources | Ledger closes | Ledger closes |
| Source permutation | Identical in tested case | Source order absent |
| Translation and undo | Identical in tested case | L1 change / photons = 1.6123 |
| 90-degree lattice rotation and undo | Differences below 4e-16 per cell | Differences below 5e-16 per cell |
| Opaque plane, maximum direct angular share 50% | Receives 74.1193% | Receives 99.1998% |
| Preionized two-sphere 1/8 cone | Receives 0% instead of 12.5% | Receives 0% instead of 12.5% |
| Uniform-density source monotonicity fixture | Fails: one cell 1 → 0.0504493 | No failure in the separate 120 random tests |

The opaque plane is two cells from an isotropic source emitting 1000 photons.
Plane capacity is 10^6 per cell, the surrounding capacity is one. Within the
nonwrapping region, at most half the emitted directions can hit the plane.
The SCRIPT-like result has no measurable behind-plane leakage yet violates
this angular bound. The capacity-proportional first-shell variant deposits
97.2995% into the plane. Absence of a visible shadow leak is therefore an
insufficient transport check.

For the 1/8 probe, two already ionized spheres have R=12, d=18 in a 64³ box.
Add 100 photons at the first center; inside the union the available atom
capacity is zero. The cone cos(theta) >= d/(2R)=0.75 must carry 12.5% of an
isotropic increment to the second sphere's far exterior. The radial closure
instead spends every photon on nearer neutral surfaces outside that cone.
Its incremental absorption centroid moves 1.31 cells **away** from the
second sphere. The continuum infinitesimal front-flux centroid is +1.30903
cells toward it, obtained by integrating the union's ray-exit radius.
The dyadic result also develops transverse centroid shifts of 9.526 cells.
This counterexample uses transparent preionized gas, not an opaque absorber.

The simple uniform-density monotonicity fixture is a 16³ grid with sources
at (6,8,8) and (9,8,8), each emitting 160 photons. Increasing only the first
source to 240 photons changes absorption at (7,4,7) from 1 to 0.0504493464.
Across all cells, 13.8094 previously allocated photons move elsewhere. Fixed
gas, no recombinations and no radiative feedback make this a meaningful
comparison. Solving the same 320 photons once versus in two equal increments
changes the final map by L1=72.9665 atoms, 22.802% of the photon count, with a
maximum per-cell difference of one. This is numerical time partitioning of
the same source proportions, not two different emission histories.

A separate support probe uses sources (7,8,8),(9,8,8), 32 photons each, in a
16³ box. Limiting every relay to radius 2.5 still deposits 0.5 photons beyond
radius 2.5 of **both original sources**, reaching distance sqrt(10). A per-hop
radius must not be interpreted as an emitter mean-free-path bound.

## Cost and what it does not establish

The Python radial implementation scans cached displacement arrays for each
source or relay. Its cost is roughly O((S+E) M), with M the supported
displacement count; the current transparent implementation also copies a
capacity array per initial source. Geometry construction sorts offsets.
The public SCRIPT code walks shells only until the packet is exhausted and
is substantially more optimized. Tree pooling costs O(N log n) operations
and O(N) working memory, with n cells per side and N=n³.

Preliminary shared-host medians for eight sparse sources (100 photons each),
unit density, warm geometry, three repetitions:

| Grid | Python radial, 1 worker | Dyadic, 1 worker | FFT filter comparator, 4 workers |
|---|---:|---:|---:|
| 32³ | 0.00370 s | 0.00510 s | 0.00393 s |
| 64³ | 0.02935 s | 0.05706 s | 0.06804 s |
| 128³ | 0.23597 s | 0.54937 s | 0.35112 s |

These are deliberately labeled toy timings. The comparator only filters two
fixed fields through several top-hat scales; it is not a complete native
21cmFAST step. The source pattern remains sparse as grid size increases,
and shared-host contention was possible. They neither establish realistic
Qin20 overhead nor justify rejecting optimized SCRIPT as expensive. Published
modern SCRIPT timings are a separate, more relevant indication that a
conservative scalar redistribution baseline can be inexpensive.

## Decision

Keep SCRIPT-like redistribution as a useful conservative statistical baseline
and a candidate to benchmark on identical 21cmFAST source fields. Do not call
it a strict directional solution, even when opaque screens are excluded:
the two-sphere incremental test and uniform-gas monotonicity test still fail.
Do not promote the fixed dyadic tree to a morphology model; its isolated
sphere and translation failures already disqualify it for that purpose.
Averaging shifted trees would reduce an origin artifact but would produce an
ensemble-mean ionization field, with additional cost and state tracking; it
would not restore angular budgets.

Any next fast/accurate hybrid needs an explicit **acceptance certificate**
for its cheap spherical branch, with directional treatment when the
certificate fails. Neither connected-component membership nor scalar local
capacity is that certificate.

## A positive certificate, and its measured limited coverage

`certified_hybrid.py` implements an actual all-direction density certificate
for one isolated isotropic source in static hydrogen, with a sharp front,
no recombinations, no scattering and no competing sources. It is a
classifier and test harness, not a completed hybrid transport solver.

For each radial shell it finds every intersecting density cell using the
minimum and maximum distance from the source to that cell's closed cube.
The minimum and maximum density over these cells are rigorous bounds for
every direction in that shell. Their integrals with weight
`(r_outer³-r_inner³)/3` bound the directional atom demand M(Ω). If these
bounds lie within `(1±epsilon) Q/(4π)`, a spherical trial front differs from
the exact direction-dependent front by at most `epsilon Q` atoms in the
atom-weighted L1 norm. Every angular cone has the same relative budget-error
bound. This uses cell geometry and extrema, not a finite sampled-ray claim.
An exact photon ledger for the spherical trial is a **separate** acceptance
condition. The function refuses PC-update eligibility when it is missing.

The toy tests certify uniform density exactly and a reflection-odd 2%
density perturbation with an atom-L1/Q bound of 1.9379%. Its exact total
spherical demand is known by reflection symmetry. An 80% perturbation and
an opaque screen are rejected. Source count greater than one also rejects
this single-source certificate. The shell extrema are deliberately
conservative; rejection does not prove the actual error is large.

For the analytic union of two equal spheres, connectedness is much weaker
than all-source visibility. A point on B's exposed surface with normal
component `n_x > -d/(2R)` has source-A outward flux numerator `R+d n_x`.
For `sqrt(2) R < d < 2R` there is a nonempty exposed band where this is
negative: a ray from A crosses neutral space before entering B. At R=12,
d=18 the stored witness has numerator -0.75 and a neutral gap of 0.727922
cells. In this exact two-ball family, `d <= sqrt(2) R` is the common
star-shape threshold for both center sources. This formula must not be
generalized to arbitrary components.

The classifier was also measured on the frozen **native Qin20 32³ fields**
provided by the root experiment, with its input SHA256 stored in
`certified_hybrid_results.json`. Every one of the 32,768 source cells is
positive; 5,657 exceed its local capacity. A scalar local allocation
`min(q,c)` would account for 83.905% of photons, leaving 16.095% requiring
nonlocal allocation. This is an accounting opportunity, not a proof that
all physical photons would be locally absorbed.

The native fully ionized mask has 7,373 cells in 78 periodic six-connected
components. Periodic BFS unwraps each component and detects winding. Its
largest component contains 7,146 cells and winds around the box. For every
nonwinding component the code tests all positive source centers against all
inward boundary-face half-spaces, using source-coordinate extrema. This
certifies common visibility for the actual axis-aligned voxel polyhedron.
It accepts 58 small components, but these contain just **0.2684% of all
source photons**, or **0.07019% when single-cell components are excluded**.
Thus a strict common-front shortcut has negligible demonstrated coverage
on this particular percolated state.

This geometry measurement does not make native ES morphology a physical
reference. Staircase voxel boundaries are more restrictive than smooth
fronts, and partial cells are outside this component test. Even accepted
visibility is insufficient by itself for conservative finite-volume front
evolution or competition between components. Measuring broader acceptance
would require a justified smooth-front/source-group approximation with its
own error budget. It cannot be asserted from connectedness alone.

## Primary references and inspected source

- Choudhury & Paranjape (2018), independent bubbles and overlap correction:
  https://academic.oup.com/mnras/article/481/3/3821/5104410
- Choudhury, Paranjape & Bosman (2021), modern SCRIPT implementation and
  timings: https://academic.oup.com/mnras/article/501/4/5782/6074258
  and https://arxiv.org/abs/2003.08958
- Maity & Choudhury (2022), synchronous overlap correction described in
  footnote 2: https://doi.org/10.1093/mnras/stac182
- Public SCRIPT source inspected at commit
  793c33fd7c252f26ab033d4e00c4dfa594f445a4, `fortran/pc.f90`:
  https://api.bitbucket.org/2.0/repositories/rctirthankar/script/src/793c33fd7c252f26ab033d4e00c4dfa594f445a4/fortran/pc.f90

No public SCRIPT source was copied into the production repository. Its
published implementation was inspected to distinguish first-pass equal
photon shells from correction-pass capacity-proportional shells.
