# Geometry-aware angular sampling of an existing ionized cavity

The useful update is a conditional incremental profile: 48 first-wall probes,
RMS predicted front radius, surface sampling factor 16, positive remaining-capacity
reference density, and a 1%-of-total-source-budget guard. On the frozen native
32³ second-half experiment, it reduces the map difference from a K3072 reference
by 63% and runs about 6.5 times faster than the original adaptive allocation.
This improves angular allocation within the same approximate transport model;
it is not a certificate of radiative-transfer accuracy.

## Method

The original predictor adds a connected transparent component's bounding radius
to the source's isolated ionization radius. A winding cavity can consequently
receive many more rays than its nearby walls require. Conversely, using total
atom density to predict an incremental radius in a partially ionized medium can
allocate too few directions.

`sampling_solver.solve` changes individual source angle counts. Source locations,
photon budgets, constant emission rates, simultaneous clearing events, straight
rays, and the Fibonacci quadrature family remain those of the baseline. The
optional `backend` argument accepts the original or compatible compact event
kernel; measurements below use the original kernel. Sources are never grouped.

For sources next to initially transparent cells, 48 probe directions measure the
first initially opaque-cell distance `l(Ω)`. The source's own cell is skipped,
because outgoing rays only launch after it clears. Predict

\[
R_f(\Omega)=\left[l(\Omega)^3+\frac{3Q_s}{4\pi\rho_{\rm ref}}\right]^{1/3},
\qquad K_s\propto 16\,4\pi\langle R_f^2\rangle_\Omega.
\]

The radial volume relation is exact for a central source in a continuum spherical
cavity surrounded by uniform remaining neutral density. It is a heuristic for
inhomogeneous or nonspherical media. Radii are capped at the transport horizon;
angle counts round up to dyadic levels, capped at the configured maximum
(normally 12,288). With `effective_density='remaining'`, the reference density is
the tenth percentile of strictly positive remaining capacity; zeros are excluded.

The raw sampler retains total-density/surface-factor-4 defaults for controlled
comparisons. The improved incremental profile sets the new options explicitly.
Initially neutral inputs at the original factor 4 reproduce original counts and
maps exactly. Factor 16 deliberately spends more directions on those inputs.

For sources contributing at least 1% of total emission, an optional guard retains
at least their original conservative angle count, evaluated with the historical
**total-capacity** density convention. The main probe predictor still uses
remaining capacity. Only these dominant sources enter the component query.
The threshold includes floating-point equality, tested with 100 equal sources.
This fixes the observed isolated-cavity regression without becoming an angular
error bound. In native32 the largest source contributes 0.10398%, so the guard
selects zero sources and leaves the unguarded profile's counts and maps unchanged.

## Native 32³ second-half comparison

Use the frozen Qin20 source/capacity fixture. Evolve half its emission with the
original adaptive solver, then hold that initial state fixed for all second-half
comparisons. Source counts, capacities, and the 16-cell horizon are identical.
The original adaptive solve **was executed**: 39.300 s and 1,882,456,064 bytes of
allocated ray storage. K3072 took 42.66 s and allocated 3,263,692,800 bytes.
K1536→K3072 changes the absorption map by 0.06457% of emitted photons, so the
reference itself retains measurable angular error.

`L1/Q` means `sum(abs(A - A_reference))/Q_emitted`, where `A` is newly absorbed
photon count. The xHII metrics divide by common hydrogen density before taking
volume mean absolute or maximum absolute errors.

| Allocation | Seconds | L1/Q versus K3072 | Mean absolute ΔxHII | Maximum absolute ΔxHII |
|---|---:|---:|---:|---:|
| Original component predictor | 39.300 | 0.56956% | 0.0024471 | 0.085323 |
| RMS, factor 4, total density | 1.465 | 0.67908% | 0.0029252 | 0.118724 |
| RMS, factor 8, total density | 2.741 | 0.44122% | 0.0018998 | 0.069664 |
| RMS, factor 16, total density | 5.463 | 0.26531% | 0.0011409 | 0.043527 |
| RMS, factor 4, remaining density | 1.547 | 0.58007% | 0.0024950 | 0.101708 |
| RMS, factor 8, remaining density | 3.225 | 0.35087% | 0.0015107 | 0.071124 |
| **RMS, factor 16, remaining density** | **6.075** | **0.21028%** | **0.0009063** | **0.036971** |

The final profile materializes 4,929,480 rays and allocates 315,621,376 bytes of
ray storage. These are single warm timings on a shared host, with compilation
excluded. They establish improvement for this frozen incremental case, not an
end-to-end cosmological speedup or convergence result. The optional compact
backend is measured separately by the integrated benchmark.

The earlier factor-4/8 total-density scan also tried the 90th percentile and
maximum probe distance. Those variants cost more without improving the RMS
tradeoff on this fixture. `sampling_benchmark.json` retains that initial scan
against K768; it must not be confused with the later K3072 comparison above.

## Partial ionization and geometry checks

In a uniform partially ionized medium, incremental radius scales as
`(Q / remaining_density)^(1/3)`. With xHII=0.99 and an intended continuum
incremental radius of six cells, total density selects K96 and differs from
K65536 by 83.58% of the increment. Remaining density selects K3072 and differs by
1.75%, for 1.79 ms versus 3.09 ms. This uses surface factor 4, isolating the density
correction. A separate scale-invariance test checks the equivalent rescaled
neutral-capacity problem.

The saved high-angle toy references use the same grid/event model. Results for
factor 16, remaining density, with the 1% dominant-source guard are:

| Case | Original L1/Q | Guarded profile L1/Q |
|---|---:|---:|
| All partial, no outward transport | 0 | 0 |
| Isolated source | 2.0555% | 0.6635% |
| Equal overlapping sources | 2.0313% | 0.6367% |
| Unequal overlapping sources | 2.1854% | 0.7161% |
| Tiny increment in overlapping transparent bubbles | 1.9620% | 1.9620% |
| Opaque plane | 2.4736% | 2.4736% |
| Dense correlated sources | 0.6464% | 0.1934% |

Without the guard, the tiny-increment case drops K12288→K6144 and regresses to
3.7702%; the guard restores the original count and map. An earlier R7 cavity
experiment similarly regressed from 3.56% at original K12288 to 10.34%/6.49% at
probe factors 4/8. These negative results motivate the guard and higher sampling
factor; no universal improvement is claimed. The dense-correlated toy uses
roughly four times the original runtime because factor 16 adds directions.
The native incremental speedup cannot be generalized to neutral inputs.

Thirty-two sampling/faint-source tests passed: periodic translation, tiny positive
origin residuals, saturated periodic retention, uniform remaining-density scaling,
the dominant-source guard (including its historical-density and equality edge
cases), invalid parameters on the transparent shortcut, native integer bounds,
subnormal per-ray budgets, and source/angle ledgers. Global, per-source, and
per-angle accounting use relative tolerance 5e-10. Cell bounds use each cell's
own remaining-capacity scale, avoiding a global tolerance that hides tiny-cell
errors. Geometry JSON also includes volume and maximum xHII errors and shell
power comparisons.

## Certified optional faint-source coarsening

`faint_sources.py` is separate from the recommended cavity profile. If all
strong-source directions and counts remain fixed, changing only the quadratures
of weak sources W obeys the model-relative bound

\[
\|A_{\rm changed}-A_{\rm original}\|_1\le2\sum_{s\in W}Q_s.
\]

The selector spends at most half a requested global L1 budget on those sources,
preserves their emission, and coarsens their angles. It selects whole equal-score
groups for translation covariance. The greedy selection is not an optimal
knapsack solver; predicted savings need not become runtime savings because some
sources never launch outward rays.

On native32, allowed L1/Q changes of 0.1%, 0.3%, and 1% yielded measured changes
of 0.00924%, 0.02116%, and 0.04965%, with runtimes 37.95, 36.83, and 34.75 s versus
39.30 s. This valid certificate gives only modest speedup here. A tighter bound
using the remaining outgoing budget at source launch is derived in the separate
audit notes, but is not implemented in the core.

## Limits

Sparse probes can miss narrow transparent cones. `audit_accuracy.py` constructs
a roughly 4.3%-solid-angle opening between 24 directions, all of which miss the
far wall. Extra or rotated probes reduce this risk without supplying an absolute
error bound. The dominant-source guard does not solve this for many individually
faint sources. Initial geometry also cannot predict every cavity cleared
collectively during the interval. The density percentile can miss very empty
channels, and an angular cap can prevent further accuracy gains.

Accordingly, the sampler reports `angular_accuracy_certified=False`. Conservation
and ledger checks do not certify map accuracy. The model assumes static density,
irreversible opaque-cell clearing, constant source emission per step, infinite
light speed, no recombination evolution, and a finite transport horizon. These
experiments do not validate full Qin20 histories, 21-cm observables, finite-light-
speed physics, or spatial-resolution convergence.

## Reproduction

Run from the repository root with NumPy/SciPy/pytest and a C++ compiler. Replace
fixture and reference paths as appropriate:

```bash
python -m pytest research/pc_refinement/test_sampling.py \
  research/pc_refinement/test_faint_sources.py -q --confcutdir=research/pc_refinement
python research/pc_refinement/sampling_native_compare.py \
  --native /path/to/source_capacity.npz --fields /path/to/second_half_fields.npz
python research/pc_refinement/sampling_convergence.py \
  --fields /path/to/second_half_fields.npz --angles 3072 \
  --reference-output /path/to/reference3072.npz --output /path/to/convergence3072.json
python research/pc_refinement/sampling_density.py \
  --fields /path/to/second_half_fields.npz --reference /path/to/reference3072.npz \
  --output /path/to/density.json
python research/pc_refinement/sampling_geometry.py \
  --reference-dir /path/to/comparison_fields --output /path/to/geometry.json
```

`sampling_native_compare.json`, `sampling_convergence3072.json`,
`sampling_factor16.json`, `sampling_density.json`, `sampling_geometry.json`, and
`sampling_faint_benchmark.json` retain numerical audits. Geometry reproduction
fails explicitly when any required reference file is absent. These standalone
experiments change no production physics flags.
