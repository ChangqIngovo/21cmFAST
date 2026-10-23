# What a fast 21cmFAST photon-conserving approximation must preserve

Research audit, 2026-10-23. Production `global-budget` remains unchanged.
The accompanying `counterexamples.py` executes nine mathematical witnesses and
an exactly solvable approximation check. Results are in
`counterexamples_results.json`. These are finite verification problems, not
Qin20 simulations or evidence that a candidate is production-ready.

## 1. Three different guarantees

Let source budgets be nonnegative photon counts `q_s`, target capacities be
nonnegative atom counts `c_j`, and `A_sj` be photons absorbed at target `j`
attributed to source `s`. A static, pure-H, no-recombination allocation obeys

\[
A_{sj}\ge0,\qquad \sum_j A_{sj}\le q_s,\qquad
\sum_s A_{sj}\le c_j.
\]

Global accounting only requires a total identity. Source/target allocation
requires all the inequalities above. Physical transport additionally requires
the allowed paths, opacity, angular emission and time history to be consistent.
The first two are necessary but do not imply the third.

For time-dependent chemistry the corresponding photon equation contains
newly ionized atoms, recombinations, stored radiation, escaped radiation and
other absorbers. A leftover numerical budget is not an ionized atom. A UV
correction cannot count X-ray preionization twice. In Qin20 the source and sink
definitions must be established from the actual native fields before replacing
them with a grid of additive photons; this is the separate source audit's task.

The distinction is consistent with the literature: photon conservation can
fail in ES constructions [1], and small global count errors can accompany much
larger errors in spatial statistics [2]. Neither result says that imposing a
global equality determines the correct morphology.

## 2. Independent spherical barriers are insufficient

An explicit radius-limited example embeds a line in three dimensions:

| Object | Position x | Photons or capacity |
|---|---:|---:|
| Source 1 | 0 | 1.25 photons |
| Source 2 | 10 | 1.75 photons |
| Target A | -1 | 1 atom |
| Target B | 1 | 1 atom |
| Target C | 10 | 1 atom |

All other source/capacity entries vanish. The allowed source-to-target distance
is 1. A radius-one ball about A contains 1.25 photons and 1 atom; the same is
true of B. C sees 1.75 photons and 1 atom. Every independent barrier strictly
passes, and total photons equal total atoms: 3.

Nevertheless A and B can jointly receive at most 1.25 photons. C can absorb at
most 1. A linear-program verifier finds a maximum absorption of 2.25, leaving
0.75 atoms unfilled and 0.75 photons outside the permitted receiving support.
Changing equality to a strict barrier does not repair the problem.

For any target subset `T`, a necessary allocation condition is

\[
\sum_{j\in T} d_j\le\sum_{s\in N(T)}q_s,
\]

where `d_j` is the requested absorption and `N(T)` contains the sources allowed
to pay those targets. Proof: sum nonnegative `A_sj` over `T`, restrict to its
neighbors, and apply each source budget. In the example `T={A,B}` violates the
condition by 0.75. Individual-cell tests and the global test miss this subset.
This is a fractional allocation condition; a flow optimizer verifying it is
not a radiative-transfer solver.

The precise conclusion is limited: one cannot preserve all these independent
binary decisions, the same allowed support, and exclusive photon spending.
Changing the selection, support or allocation rule can avoid this example.
It is not an impossibility theorem for every ES-inspired or FFT algorithm.

## 3. Exact allocation still permits unphysical transport

### Radial precedence

Send one photon down one beam through a foreground capacity of 1 and then a
background capacity of 1. In the opaque, sharp-front approximation the answer
is `(1,0)`. Simultaneous capacity-proportional matching gives `(0.5,0.5)`.
Every source and cell inequality passes, yet the background is ionized before
the foreground is paid. A finite-opacity transfer model can transmit through
the foreground, but must then use its optical depth; it does not justify the
opaque model's capacity split.

### Angular quotas

Let an isotropic source emit 2 photons into two equal-solid-angle sectors, with
first-intercepting capacities `(1,100)` at the same radius. Each sector receives
1 photon. Capacity-proportional shell allocation instead gives
`(0.0198019802,1.9801980198)`. It conserves total photons and cell capacities but
exceeds the second sector's emission quota by 0.9801980198 photons.

Taking thinner radial shells repairs neither this example nor the missing
angular information. Conversely, fixing each sector's total quota without
radial ordering still permits the foreground/background failure. Both
constraints are needed. Finite sectors also mix unresolved directions; their
shadow error must converge as angular resolution increases.

### Low angular moments

Two opposing unit beams along x and two opposing unit beams along y have
identical radiation number density `E=2` and flux vector `F=0`. Their incident
flux through a +x face is respectively 1 and 0. A local state containing only
`E,F` cannot distinguish these boundary responses without a closure. This does
not disqualify moment methods: it identifies their angular approximation. For
example, RAMSES-RT explicitly uses an M1 Eddington-tensor closure and an
evolving radiation field [3]. A zero flux vector must not be mistaken for zero
radiation density or zero photoionization.

### What radial statistics and FFTs do and do not retain

Move an opaque clump from `(1,0,0)` to `(0,1,0)` relative to a source. Every
source-centered spherical cumulative capacity is unchanged, but a target on
the +x axis changes from shadowed to directly exposed. The executable verifies
this on a 9-cubed grid. This only proves loss of information in those radial
aggregates. The full spatial fields and their full FFTs distinguish the cases.
FFT is a computational representation, not a physical approximation by itself.

A sharp-k Fourier filter has another, separate limitation: its real-space
kernel is signed. On the executable 32-cubed fixture, the normalized kernel
sums to approximately 1 but has minimum `-0.0007467335` and integrated negative
weight `3.0994046`. It cannot directly be interpreted as probabilities or
nonnegative photon transfers. A positive real-space ball/cone kernel is possible,
but changes the filter estimator and still needs transport constraints.

## 4. A conservative diffusion obstacle is useful but has a different closure

Consider the proposed elliptic obstacle problem

\[
\min_{u\ge0}\frac12u^TLu+(c-s)^Tu,\qquad a=s-Lu,
\]

with a connected periodic local graph Laplacian `L`, nonnegative edge weights, and
`sum(s)<sum(c)`. Its KKT conditions give `a<=c` and `u_j(c_j-a_j)=0`.
Nonnegativity of `a` also follows: at `u_j>0`, `a_j=c_j`; at `u_j=0`, the
M-matrix stencil gives `(Lu)_j<=0`, hence `a_j>=s_j>=0`. The row-sum identity
gives `sum(a)=sum(s)`. Antisymmetric edge currents give local accounting.

These are strong algebraic properties. They describe conservative diffusion
down gradients of `u`; no directional source label or straight-line visibility
has been supplied. Local conservation therefore does not establish an
appropriate closure for freely streaming ionizing radiation. This is a useful
fast comparison model, and could have a physical interpretation in a separately
specified diffusive radiation regime, but should not inherit a free-streaming
claim.

The saturated cases require explicit treatment. If `sum(s)>sum(c)`, the
objective is unbounded along positive constant shifts of `u`. If the totals
are equal, constant shifts produce nonunique `u` while `a=c` can still be fixed.
A reservoir or boundary sink must account for excess photons.
For a disconnected graph, the corresponding supply condition is required in
each connected component; a global inequality alone is not sufficient.

## 5. The two-bubble geometry has three distinct numbers

For equal spheres of radius `R`, center separation `d`, and a source at the
first center, a direction with cosine `mu` toward the second center intersects
the first surface inside the second sphere exactly when `mu>=d/(2R)`. Thus

\[
f_{\text{A surface inside B}}=\frac{1-d/(2R)}2=\frac18
\quad\text{at }d=1.5R.
\]

The entire apparent solid angle of B instead has fraction

\[
f_{\text{any intersection with B}}
=\frac{1-\sqrt{1-(R/d)^2}}2=0.127322003750035.
\]

The latter includes a small set of rays that leave A before entering B. It is
not the requested 12.5% contact-cap benchmark. One million deterministic
equal-area directions reproduce `0.125` and `0.127322` respectively.

Overlap volume is different again:

\[
\frac{V_{\cap}}{V_R}
=\frac{(4R+d)(2R-d)^2}{16R^3}=\frac{11}{128};
\qquad \frac{V_{\cap}}{2V_R}=\frac{11}{256}.
\]

Neither overlap volume fraction is an angular photon fraction. These are
geometrical identities for specified spheres, not a complete solution for
their evolving fronts or for how photons are reassigned after overlap.

## 6. Positive result: a certified nearly spherical approximation

There is a useful regime in which an ES sphere is a controlled physical
approximation. Assume one isolated isotropic source, static gas, no
recombinations/scattering, an opaque sharp front, an initially spherical paid
region, and no escape or light-travel-time restriction. For each direction
define the incremental atom demand per steradian

\[
M_\Omega(r)=\int_{r_0}^{r}n(r',\Omega)r'^2\,dr'.
\]

For emitted photons `Q`, the exact directional front satisfies
`M_Omega(R_Omega)=Q/(4pi)`. Let a trial ES sphere `R` have exactly `Q` atoms in
its new shell. If a **uniform** certificate establishes

\[
\sup_\Omega\left|\frac{M_\Omega(R)}{Q/(4\pi)}-1\right|\le\epsilon,
\]

then its mass-weighted morphology error obeys

\[
\int n|\chi_{\rm trial}-\chi_{\rm exact}|\,dV
=\int|M_\Omega(R)-Q/(4\pi)|\,d\Omega\le\epsilon Q.
\]

The identity follows because both fronts intersect each source ray once and
radial demand is monotone. Every angular subset's absorption also differs from
its isotropic quota by at most `epsilon` times that quota. The approximate
front has exact total counts and controlled, rather than exact, angular counts.

This certificate does not require solving for all exact directional radii.
For example, certified shellwise density bounds `n_-(r)<=n(r,Omega)<=n_+(r)`
give radial integral bounds on every `M_Omega(R)`. Bounds from voxel/block
extrema can be conservative; in a strongly inhomogeneous region they will
reject rather than falsely certify an ES sphere. Sampling a few directions
alone does not certify the unsampled ones.

The executable check uses four equal-angle sectors with densities
`(0.9,1,1.1,1)`. A unit-radius sphere has exact total counts, a uniform
directional-demand bound of 10%, and an exactly computed atom morphology error
of 5%. This validates the bound on a solvable anisotropic example; it does not
show that cosmological cells satisfy a 10% bound.

Disjoint isolated bubbles can use this result independently until certified
outer front bounds touch. Sources outside each bubble must be shown not to
contribute during the accepted interval. Merely labeling a bubble isolated in
an ES map is insufficient. Continuous or densely clustered source fields may
leave little domain eligible for this shortcut.

## 7. A restricted multi-source route without tracing explicit rays

The earlier V17 arrival-field derivation remains valid in its stated regime:
a transparent cavity with a smooth boundary strictly star-shaped about every
source, static gas, constant source rates, no recombinations or scattering,
infinite light speed, and a sufficiently short evolution interval. Define

\[
F_i(x)=\frac{q_i(x-s_i)}{4\pi|x-s_i|^3},\qquad F=\sum_i F_i.
\]

An arrival field solves `F dot grad(T)=n`, with `T=0` on the old front. If
`F_i dot grad(T)>0` for every source throughout the swept region, the shares
`alpha_i=F_i dot grad(T)/n` are nonnegative and sum to one. Since
`div(T F_i)=n alpha_i` outside sources and the source flux is tangent to its
cone walls, the divergence theorem gives exact source and every-cone counts.
This is an Eulerian transport equation; it is not the original scalar ES test.

A useful inexpensive sufficient positivity test is available for a compact
source group. If every source lies in a ball `B(c,a)`, then

\[
(x-c)\cdot\nabla T>a|\nabla T|
\quad\Longrightarrow\quad
(x-s_i)\cdot\nabla T>0\quad\text{for every }i.
\]

This follows directly from Cauchy-Schwarz. It removes a source-by-front-cell
loop from this particular certificate; it does not establish the other
conditions or eliminate the need to compute a sufficiently accurate `F`.
Isolated-patch, zero-padded convolutions can accelerate unattenuated fluxes.
A periodic Poisson field with its mean source removed instead has a distributed
negative divergence and cannot silently be substituted into the source-free
divergence proof.

For an ES-derived trial arrival field `T_hat`, the earlier sufficient conditions
were correct boundary values, common visibility, no extra level-set branches,
all source-direction derivatives positive, and a uniform residual bound

\[
|F\cdot\nabla\widehat T/n-1|\le\delta<1.
\]

After exact total-atom normalization, the proved atom morphology error is at
most `2 delta` of the increment, and the source-cone relative error is at most
`2 delta/(1-delta)`. These bounds hold only while the stated regular common
cavity exists. At nonzero residual, exact global normalization does **not**
make every source budget exact. Correcting source shares afterwards needs a
fresh positivity, directional and residual analysis.

The arrival method can accept cheap trial fronts where this certificate is
available and fall back elsewhere. A practical uniform certificate needs
interpolation/derivative error bounds, not just values at selected grid nodes.
Disconnected cavities, first contacts, internal opaque clumps and partially
visible sources are precisely the cases where automatic acceptance fails.
In particular, connected is weaker than commonly visible. For the union of
the user's two `R` spheres at `d=1.5R`, an exposed B-surface point has normal
component `n_x>-0.75`, whereas its source-A normal flux is negative whenever
`n_x<-2/3`. The nonempty band `-0.75<n_x<-2/3` therefore violates the
common-visibility condition. A component-label shortcut must reject this band;
the common-cavity proof cannot be applied to that entire overlapping union.
No global theorem or full Qin20 implementation of this hybrid is supplied here.

## 8. Defensible acceleration options

1. **Certified bulk plus conservative angular fallback.** Accept spherical or
   common-cavity updates only under the corresponding error certificate. Carry
   photon and angular ledgers into fallback. Rejecting a trial must happen
   before irreversible ionization or double expenditure; transporting only
   whatever photons remain after uncertified ES painting cannot repair the
   earlier assignments.
2. **Skip transparent blocks.** A previously paid, sufficiently transparent
   block can be crossed without visiting each cell when a bound controls its
   absorption and propagation error. Current recombinations/opacity must be
   included. A block with unresolved neutral clumps is not automatically
   transparent because its average ionized fraction is high.
3. **Merge distant sources with a visibility test.** Replacing a compact source
   group by its luminosity centroid cancels the first geometric multipole and
   gives a far-field flux error of order `(group size/distance)^2` away from
   singularities. An opaque edge can make visibility discontinuous, so the
   geometric opening criterion alone is insufficient; split bundles when their
   opacity/visibility differs. Adaptive angular rays are an established way to
   concentrate angular work where grid resolution requires it [4].
4. **Use moment or diffusion closures as explicit approximations.** They can be
   inexpensive and source-count independent in appropriate formulations [3],
   but must pass the target morphology/statistics tests. Photon conservation
   should not hide their closure errors.

These are conditional research paths, not a proven universally cheap algorithm.
The fraction requiring angular fallback is an output to measure. It can become
large near complicated overlap, opaque structure, or a densely populated source
field. No general bound of a few times ES runtime has been established.

## 9. Acceptance criteria and cost accounting

The following are proposed engineering gates, not user-imposed tolerances or
scientifically universal thresholds:

| Gate | Proposed measurement |
|---|---|
| Ledger | Source, cell, recombination, escape/reservoir residuals; floating-point tolerance scaled to problem size and output precision |
| Direction | Shadow leakage, cone overspend and foreground precedence on analytically specified fixtures |
| Convergence | Refine angle, grid, radial discretization and time separately; no fixed-resolution pass substitutes for convergence |
| Stability | Permute sources and rotate/translate configurations; refine schedule with the same physical source history |
| Cosmological accuracy | Compare matched source histories and matched ionized fractions to a converged transport reference; report xHI, 21-cm power, cross correlation, morphology and Gamma/sink changes |
| Initial screening target | For example <=1% absolute ionization-history difference and <=5% power change across a declared k range under further refinement; revise to science needs, do not call these established accuracy |
| Runtime target | Investigate total walltime <=2--3 times the same Qin20 non-PC run, same grid/threads and hardware; this is a suggested search budget, not a guaranteed bound |
| Memory and tails | Peak memory, source/active-cell counts, FFTs/iterations/interactions, slowest snapshots and fallback fraction |

Pointwise source-brightness monotonicity is a useful diagnostic at fixed gas,
spectrum and source schedule with feedback disabled: adding photons should
not undo already possible ionization in the sharp-front model. With feedback,
different source histories, or gas dynamics, a pointwise decrease is not by
itself unphysical. Capacity changes require care: more atoms can mean more
absorbed photons but a lower ionized *fraction*, so a single undifferentiated
"capacity monotonicity" assertion is inappropriate.

Photon conservation does not predict suppression of the power spectrum. The
four-cell executable example has two ionized atoms in both maps but fundamental
unnormalized powers 2 and 0. A relative power error can be large near a spectral
cancellation even when an atom-L1 error is small. Cosmological statistics must
therefore be tested directly, with both fixed-redshift and fixed-ionization
comparisons and clear mode-count/box-size limitations.

## 10. Prior failures that must not be silently reintroduced

The prior `native_es_research_20261023/METHOD.md` rejected an exclusive-donor
spherical ES algorithm despite exact source/cell accounting: increasing source
emission reduced some cell ionization, and splitting the same exposure altered
the map. A convex objective did not automatically cure those issues. The
laminar alternative gained fixed-tree stability by changing spherical geometry
and introducing grid-origin dependence. These are relevant baselines, not
discarded tests.

The V21 exact cone/Cartesian-face geometry verified flux partitioning and free
streaming to floating-point precision. Its own report explicitly retained
finite-bin shadow diffusion as a transport approximation. Its axial-contact
calculation showed a first-order, nonmonotonic grid error from voxelized initial
fronts and shared midpoint capacity; excellent angular bookkeeping alone did
not remove that spatial error. Neither result should be advertised as exact
finite-grid morphology.

Internal antecedents inspected:

- `native_es_research_20261023/METHOD.md`
- `front_rule_research/v17_repair/arrival_theory.txt`
- `front_rule_research/v21_angular/angular_faces_report.txt`
- `front_rule_research/v21_angular/axial_contact_theory.txt`

## Primary references

[1] Paranjape, Choudhury & Padmanabhan (2016), *Photon Number Conserving Models
of H II Bubbles during Reionization*, https://arxiv.org/abs/1512.01345 .

[2] Choudhury & Paranjape (2018), *Photon number conservation and the large-scale
21 cm power spectrum in semi-numerical models of reionization*,
https://arxiv.org/abs/1807.00836 .

[3] Rosdahl et al. (2013), *RAMSES-RT: Radiation hydrodynamics in the cosmological
context*, https://arxiv.org/abs/1304.7126 .

[4] Abel & Wandelt (2002), *Adaptive Ray Tracing for Radiative Transfer around
Point Sources*, https://arxiv.org/abs/astro-ph/0111033 .

The finite counterexamples, obstacle audit, single-source error bound and
source-ball positivity test above are derivations in this research note; no
claim of publication novelty is made.
