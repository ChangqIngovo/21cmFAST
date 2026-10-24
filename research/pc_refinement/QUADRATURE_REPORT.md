# Equal-cost angular quadrature comparison

**Decision: retain the original Fibonacci direction rule.** Seven replacements
were tested with the same sources, cell capacities, source-specific angle counts,
opaque-voxel event equations, and distance limits. None improves accuracy across
both native fixtures and the geometric tests. These experiments isolate angular
quadrature; they do not remove the baseline model's coarse-cell front bias.

## Rules implemented

| Rule | Construction |
|---|---|
| Fibonacci control | Original equal-area spiral and equal photon weights |
| Antipodal Fibonacci | Hemisphere spiral plus its negatives; exactly zero dipole |
| Moment design | Equal-weight 24-point spherical 5-design; 48-point spherical 7-design; rotated unions at larger counts |
| Rotated Fibonacci | One deterministic rigid rotation of the original nodes |
| Low-order design | Moment designs at 24/48 directions; original Fibonacci at 96 and above |
| Cubic equal-area | Full signed-permutation orbits of equal-area nodes in the spherical fundamental triangle; 24-point design at the lowest level |
| Voronoi weights | Original Fibonacci nodes weighted by their spherical Voronoi areas |
| Moment weights | Positive weights closest to uniform subject to exact monopole, dipole, and quadrupole constraints |

The 24-point design uses permutations and signs of `(sqrt(p),sqrt(q),sqrt(q))`,
where `p=(1+sqrt(1.6))/3` and `q=(1-p)/2`. For the 48-point design the squared
coordinates are the roots of `t^3-t^2+t/5-1/105=0`; signed permutations reproduce
all sphere moments through degree seven. These facts are checked against every
monomial of the corresponding degree, rather than only a few moments.

The cubic rule maps equal-area samples into `x>=y>=z>=0`. With
`n=(cos(theta),sin(theta)cos(phi),sin(theta)sin(phi))`, its azimuth CDF is
`phi-asin(sin(phi)/sqrt(2))` over area `pi/12`; conditional `cos(theta)` is uniform.
The full symmetry group is explicitly tested. This achieves exact Cartesian
rotation/reflection symmetry, but its coherent sampling errors can still be
larger than those of the original spiral.

Nonuniform weights use a separate copy of the event core. Only the angular rate
assignment and local-absorption angular ledger differ. Every weight is positive,
each source's weights sum to one, and per-source, per-angle, and cell-capacity
audits remain enforced. The baseline core is unchanged.

## Main measurements

Normalized discrepancies are `sum(abs(absorption-reference))/emitted_photons`.
The native 32³ fixture is the density-weighted/species-normalized frozen Qin20
E-INTEGRAL source field from the previous study. The native 64³ fixture is the
restricted L-INTEGRAL coeval. Their source physics and ionization stages differ;
these are two independent tests, not a resolution-convergence pair.

| Rule | Native 32³ L1/Q | Native 64³ L1/Q | Isolated R4 L1/Q |
|---|---:|---:|---:|
| Original Fibonacci | **0.3884%** | 0.5639% | **2.0555%** |
| Antipodal | 0.4657% | **0.5461%** | 2.1821% |
| Moment design | 0.6039% | 0.6992% | 7.4538% |
| Rotated Fibonacci | 0.3953% | 0.6087% | 2.4319% |
| Low-order design | 0.5140% | 0.6127% | 2.0555% |
| Cubic equal-area | 0.6245% | 0.8039% | 4.7917% |
| Voronoi weights | 0.3889% | 0.5648% | 2.0555% |
| Moment weights | 0.3904% | 0.5674% | 2.0555% |

Original Fibonacci and both weighted variants are essentially indistinguishable
on the high-angle isolated/overlap tests; their native differences are also too
small to establish a useful accuracy gain. Antipodal sampling improves the 64³
fixture slightly but worsens the 32³ fixture. Choosing it only for the favorable
fixture would not be a validated general rule.

The large-angle rotated-design union is substantially worse in all geometries.
Its exact smooth angular moments therefore do not guarantee accurate integration
of discontinuous first-absorber maps. Likewise, enforcing cubic symmetry does
not establish better angular integration.

Sparse references are the previously saved 262144-angle results, dense-toy
reference 12288, native 32³ reference 3072, and native 64³ reference 768. The
32³ reference refinement difference was 0.0546% and the 64³ difference 0.2934%.
The R8 sphere additionally uses a fresh 65536-angle reference and a separately
integrated continuum sphere. All geometric and native result rows, including
continuum comparisons and resolved powers, are in `quadrature_results.json`.

Kernel times remain comparable because each source requests the same angle
count; realized ray activation can still change with the front. Three sequential repetitions are recorded; geometry preparation is
cached and compilation excluded. Timings are exploratory shared-host wall times.
There is no accuracy advantage sufficient to justify changing the default.

## Validation and reproduction

Seventeen tests pass, including exact moment tests, all signed coordinate
permutations, fixed-rate time subdivision, positive weights, independent angular
ledgers, uniform-weight equivalence to the original C++ core, and exact baseline
reproduction without patching the baseline module.

```bash
python -m pytest research/pc_refinement/test_quadrature.py -q --confcutdir=research/pc_refinement
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research/pc_refinement/bench_quadrature.py \
  --native32 /path/to/native32/source_capacity.npz \
  --native64 /path/to/native64/native_fields.npz
```

Saved reference cubes are reused when present after checking input hashes. In a
fresh checkout the script regenerates the reference at the recorded angle count;
this is deliberately more expensive than the candidate calculation. No source
or photon normalization is fitted to the reference.
