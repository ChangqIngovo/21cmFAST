# Positive follow-up: certified common-visible cavity flux

This is a promising restricted acceleration route. The new implementation
`hybrid_flux_probe.py` computes the exact frozen-boundary vacuum photon flux
of cell-centred point sources, using three face-integrated convolution
kernels. Its local divergence and closed-cavity budgets are tested to
floating-point precision. A complete evolving-front hybrid and Qin20
integration have **not** been implemented or benchmarked here.

The positive mathematics complements `theory.md`, sections 6–7: use a cheap
flux evaluator only where common visibility can be certified, and retain a
directional first-absorber fallback where the certificate fails. This preserves
the existing production method during research.

## 1. When the total vector flux is enough

Assume a transparent ionized cavity, isotropic point sources with rates q_s,
pure hydrogen, an opaque sharp front, no internal absorbers or scattering,
constant rates over the step, and the infinite-light-speed approximation. Let

\[
F_s(x)=\frac{q_s(x-s)}{4\pi|x-s|^3},\qquad F=\sum_sF_s.
\]

If every boundary point sees every included source and every source ray exits
the cavity only once, every incident contribution is outward. The physical
instantaneous normal speed is

\[
v_n=\frac{F\cdot n}{n_H}.
\]

There is no angular closure in this formula: F was calculated from the full
source geometry, and the common-visibility assumption removes the distinction
between net and incident outward flux. It is not the general claim that the
first moment alone determines radiation in arbitrary partially opaque media.
At frozen geometry, each source's boundary contribution integrates to q_s by
the divergence theorem; any specified source cone has its correct solid-angle
quota as well.

This gives an instantaneous physical flux, not permission to extrapolate the
same speed across a large new layer. The evolving region must keep the
visibility property. Front reconstruction, atom integration, finite-step error
and changes in geometry still need a controlled moving-boundary or arrival-time
update. `F dot grad(T)=n_H` is the arrival equation under the conditions stated
in `theory.md`. A conservative finite-volume implementation is a plausible
next step, but was not tested here.

## 2. An actual no-ray visibility certificate

Represent the cavity by a bounded, embedded, watertight polyhedral boundary with
consistent outward normals, and establish that every source is inside it. Let
all sources lie in a known bounding set G. For each boundary face with point
x_f and outward unit normal n_f, require

\[
x_f\cdot n_f > h_G(n_f),\qquad
h_G(n)=\sup_{s\in G}s\cdot n.
\]

The face is planar, so checking one point is uniform across the entire face.
This inequality means `(x-s) dot n > 0` for every source and every face.
Any ray starting at a source can therefore cross the boundary only outward.
Since the cavity is bounded and the source is inside, it has exactly one exit,
apart from measure-zero edge/vertex directions handled by continuity. This
proves common visibility; convexity is sufficient but not necessary.

For a source bounding ball B(c,a), the check is
`(x_f-c) dot n_f > a`. For an axis-aligned bounding box centred at c with
half-widths a_i, use the often tighter test

\[
(x_f-c)\cdot n_f>\sum_i a_i |n_{f,i}|.
\]

After constructing source bounds, cost is O(number of boundary faces), with no
source-by-face ray loop. The implementation includes the box test on a voxel
boundary. The topology assumption is not inferred or repaired by this function;
the numerical validation uses ordinary closed cubes where it is known.
An ES connected-component label by itself is not a visibility certificate.

Voxel staircases can conservatively reject otherwise smooth-looking fronts.
A reconstructed smooth/polyhedral front may admit more certified regions,
but its geometry must match the mass accounting; smoothing a label and
forgetting the changed atom volume is not acceptable. Internal neutral holes,
separated opaque components and most first-contact necks trigger rejection or
require separate handling. Geometry must be recertified through any accepted
advance, not only at the initial snapshot.

## 3. Exact face kernels preserve a local ledger

Point-sampling `r/r^3` on a grid and applying an unrelated finite-difference
divergence generally does not preserve its delta-function source identity.
Instead integrate it across each cell face exactly. For a unit rectangular
face with positive normal coordinate z, tangential limits u_1,u_2 and v_1,v_2,
the signed solid angle is the corner sum

\[
\Omega=\sum_{a,b\in\{1,2\}}(-1)^{a+b}
\arctan\frac{u_av_b}{z\sqrt{u_a^2+v_b^2+z^2}}.
\]

The implementation uses the corresponding four-corner signs and z is always
a nonzero half-integer. The face transfer from a source is q_s Omega/(4pi).
Translation invariance makes each of the three face orientations a convolution
on a staggered grid. These kernels are signed **vector normal fluxes**; they
are not nonnegative target-allocation weights and must not be clipped.

The six oriented solid angles of a cell sum to 4pi if the source is inside and
zero otherwise. Thus the shared face fluxes obey, in exact arithmetic,

\[
\sum_{f\in\partial cell}F_f=q_{cell}.
\]

Summing cells cancels each internal face exactly. A closed certified cavity
therefore receives exactly the rates of its internal sources, without a global
rescaling. FFT roundoff creates the small reported residuals. Each input source
is a point at its cell centre; assigning extended source emission to one point
is a spatial model choice, not an error eliminated by this proof.

The convolutions use a 2x zero-padded extent in every dimension, with the full
source-target offset kernel for an isolated patch. This avoids the unphysical
mean-subtracted periodic Poisson identity `div(F)=S-mean(S)`. In a periodic
cosmological box, a compact component crossing a box seam requires consistent
unwrapping and source images. A percolating component is not automatically a
bounded common-visible Euclidean cavity and requires a separate treatment.

## 4. What was actually verified

Run `python -m unittest -v test_hybrid_flux.py` and
`python hybrid_flux_probe.py --workers 1` in the research environment.
Five independent tests pass:

- Rectangular face solid angles agree with 64x64 Gauss–Legendre integration of
  the vector kernel, including a negative normal offset.
- A unit cell-centred source sends exactly 1/6 through each of its six cell
  faces, to roundoff; cellwise divergence recovers its input field.
- Many sources in a closed convex cube give positive outward face flux,
  exact total source flux and cubic-axis rotational covariance.
- A rigorous centroid-monopole bound exceeds every tested direct-sum error.
- The visibility certificate rejects an internal neutral hole and an included
  source outside the proposed cavity.

The saved 16³ fixture uses a cube with 600 boundary faces and 512 nonzero source
cells inside it. The source-box visibility margin is 1.5 cells, so it is
certified rather than merely assumed. Total input is
`526.7337985754784`; outward boundary flux is `526.7337985754785`;
the maximum cellwise divergence discrepancy is `2.22e-15`.
The minimum boundary-face flux is positive, `0.3950538178`.
For one source, maximum divergence discrepancy is `8.33e-17`.
These are frozen-geometry tests, not a completed front-evolution test.

A separate audit includes an outside source. Its field contributes zero **net**
flux through the cavity but changes individual boundary fluxes, even when the
combined boundary flux remains positive everywhere. Thus checking positivity
and total counts on an all-source convolution does not certify that the right
sources were included. Source membership and transparency remain necessary.

## 5. Cost and component handling

`IsolatedFaceFlux(shape, workers=4)` constructs and caches three kernel spectra.
`evaluator(source)` then uses one shared source FFT and three inverse FFTs:
four transforms per evaluation, plus three forward kernel transforms at setup.
No explicit rays, angular fields, or source-by-target matrix are stored.

The isolated padding contains 8N cells for an N-cell patch. Cost is
`O(4 * 8N log(8N))`; using four padded transforms is not equivalent in work to
four unpadded ES transforms. Cached kernel spectra occupy about 24 original
float64 grid equivalents. Peak evaluator arrays are roughly 56 original grid
equivalents, plus three cropped face arrays, divergence, Python and FFT
workspace: about 0.9–1.1 GiB at a 128³ original patch. This is a liveness
estimate, not a measured peak RSS. Kernels may be streamed at extra FFT cost.

The 16³ frozen fixture takes approximately 2–4 ms with one worker on this host,
but this tiny diagnostic is not an end-to-end speed claim. The evaluation
does not include component finding, certification, front advancement,
recombinations, or fallback transport.

Separate opaque components require source-separated evaluation. Multiplying an
all-source F by component labels does not remove the unwanted source terms.
If each component uses its own bounding patch, work scales as
`sum_c N_patch,c log(N_patch,c)`. Well-packed disjoint small cavities can be
cheap; overlapping bounding boxes or complicated components can make that sum
much larger than the global volume. The cosmological certified fraction and
patch-volume overhead have not been measured, so no bounded full-run overhead
is established.

## 6. A rigorous compact-group multipole shortcut

For sources in B(c,a), choose c as the luminosity centroid. Put
`M2=sum_s q_s |s-c|^2`, Q=sum q_s, and d=|x-c|>a. The monopole is
`F0=Q(x-c)/(4pi d^3)`. Direct differentiation gives

\[
\|D^2(r/|r|^3)[h,h]\|\le 6|h|^2/|r|^4.
\]

The first-order terms cancel at the centroid, so Taylor's theorem establishes
the **uniform** bound

\[
|F-F_0|\le E(x)=\frac{3M_2}{4\pi(d-a)^4}
\le\frac{3Qa^2}{4\pi(d-a)^4}.
\]

For a normal flux, the same absolute bound applies. If `F0 dot n > E`, true
normal flux is positive and its relative approximation error is at most
`E/(F0 dot n-E)`. Use a distance lower bound over an entire face/patch for a
uniform certificate, rather than sampling a few points. Group error bounds
add by the triangle inequality. Geometry/visibility certification is separate;
the multipole bound is for vacuum, not attenuation through neutral structures.

The test uses 64 sources and 2048 evaluation directions. Actual maximum errors
relative to monopole magnitude are compared with the rigorous bounds:

| a/d | Measured maximum | Uniform analytic bound |
| --- | --- | --- |
| 0.05 | 0.0309% | 0.2486% |
| 0.10 | 0.1244% | 1.2343% |
| 0.20 | 0.5110% | 7.9084% |
| 0.40 | 2.2876% | 99.9780% |

The analytic inequality supplies the certificate; testing directions only
checks its implementation. Conservative bounds reject broad near-front source
groups. A source tree could split rejected groups and accept compact distant
groups, reducing typical boundary-evaluation cost; no general subquadratic
worst-case or performance result is claimed here.

There is an additional ledger subtlety. A single fixed source-group partition
used over the entire closed cavity boundary preserves each group's total Q:
each replacement monopole is still inside the certified cavity. Exact face
solid angles then retain the divergence identity for that declared grouped
source field. If different targets use different group centres/partitions,
the resulting piecewise approximate F is not automatically one globally
divergence-consistent field. Standard adaptive evaluation needs an explicit
error/ledger treatment; the error bound alone is not exact photon accounting.
For grouped sources, individual source attribution also requires a stated
approximation: aggregate second-order flux accuracy does not by itself prove
every original source's cone allocation is equally accurate.

## 7. Concrete next step and unresolved work

A defensible hybrid would construct a front geometry with atom-consistent
cut-cell volumes, certify common visibility through the proposed advance,
evaluate its true source-separated face flux, and advance a conservative
arrival/front model while recertifying. Noncertified components or intervals
would use the separately developed directional first-hit solver. Emitted and
absorbed photon reservoirs must be transferred exactly once between the two
branches; one cannot keep both branches' allocations in an overlapping region.

This investigation supplies the source-separated frozen-flux evaluator, its
local ledger proof, a cheap geometric certificate and a controlled multipole
bound. It does not yet supply the geometry-consistent finite-step updater,
the branch handoff, native Qin20 source/sink closure, or evidence that the
certified fraction is large enough to keep runtime close to ordinary ES.
