# Origin of spatial-resolution dependence

The production late-stage results rule out photon loss, insufficient transport
distance and output projection as the main explanation of the measured late
5--13% bias-power differences. This study separates the spectral mechanism
from a causal intervention on the transport discretization.

## Spectral mechanism in the production box

Let Q be the ionized mass fraction, and let delta_I denote the contrast in
ionized-hydrogen **mass density**, rather than in the ionization fraction.
Atom counting gives the exact identities

    delta_HI = (delta_m - Q delta_I) / (1-Q)
    b_HI,cross = (1 - Q b_I,cross) / (1-Q)

where b_X,cross=P_X,m/P_m. The reported bias power is b_HI^2=P_HI/P_m;
it is not generally the square of the cross-bias. At fixed Q, a change in the
ionized-density cross-bias is amplified by Q/(1-Q) in the neutral-density
cross-bias.

At Q=0.9 and k around 0.052 h/cMpc, the measured four-seed mean b_I,cross is
3.26556, 3.32267, 3.40842 for grids 100, 50, 25. Changes of 1.75% and 4.37% in this
allocation--density bias accompany 5.45% and 13.11% fine/coarse differences in
HI bias squared. The corresponding signed HI cross-biases are
-19.3900,-19.9041,-20.6758.

The exact band decomposition

    P_HI = b_HI,cross^2 P_m + P_perpendicular

attributes about 80% and 78% of the difference in b_HI^2 to the correlated term
at Q=.9. The perpendicular component includes nonlinear physical structure and
discretization effects; it is not identified as numerical noise. These are
spectral decompositions relative to each grid's own matter field.

The difference varies with wavenumber. At Q=.9 the 100/50 and 100/25 deviations
are -0.93%/-2.60% in the fundamental shell, growing to -5.45%/-13.11% near 0.052.
The fundamental has six full modes per seed, so the strict k->0 limit remains
unmeasured. Photon conservation fixes the zero mode; finite-k allocation and
source-to-absorption displacement moments remain unconstrained.

`analyze_large.py` reads the supplied spectra only; its reproducible output,
with input hashes and paired-seed statistics, is `large_spectral_results.json`.

## Causal intervention: hold source positions and gas density fixed

`point_front.py/.cpp` is an independent diagnostic implementation of the same
static, simultaneous-source, opaque-cell model. Source positions and budgets
are explicit arrays and are independent of the gas mesh. Refining the gas grid
splits the same piecewise-constant atom counts, without introducing new density
information or moving the sources. It eagerly allocates its rays, so this
reference is not the proposed production optimization.

It reproduces the production adaptive eight-subcell solution on native inputs:
the coarse absorbed-field L1/emitted difference is below 4.6e-17 and the maximum
subcell fraction difference is below 1.3e-15 (`native_bridge.json`). Unit tests
also compare the existing one-cell and eight-subcell models, periodic/open
boundaries, exact-horizon escape, translation/reflection, coincident sources and
source permutation.

The native32 fixture is conservatively reduced to a 16^3 base source/density
grid. The causal factorial separately changes density smoothing, source
amplitudes, source relocation/aggregation, and gas-front cell size. Every
spectrum is measured on the same 8^3 grid (18 modes). Sources and densities in
the principal front-only comparison both originate from the same 8^3 fields.

At Q=.9, with 12,288 directions and the same source coordinates, source budgets,
piecewise gas density and output grid:

| Quantity | Gas32^3 | Gas16^3 |
|---|---:|---:|
| HI bias squared |535.432|608.039|
| Signed HI--matter cross-bias |-22.6948|-24.2479|
| Ionized mass fraction |0.900000000|0.899999968|
| Ionized volume fraction |0.892268678|0.890670127|
| Cov(rho,x_HII)=Q_mass-Q_volume |0.00773132|0.00932984|

Front coarsening alone raises the measured power by 13.56%, strengthens the
cross-bias magnitude by 6.84%, and raises ionization--density covariance by 20.68%.
Because the density is constant within every measurement cell in this pair,
the reported volume fractions and covariance are preserved by projection.

The 6144->12288 angle change alters the two powers by -0.179%/+0.081%; the final
retained fractions are 0 and 3.55e-8. The front effect is much larger than the
measured angular change, with negligible undelivered photons. An independent
comparison retaining the original 16^3 source coordinates and density, but
refining gas 32^3->64^3, reduces power by 2.706% at 1536 directions, with zero
retained photons.

The full low-angle factorial shows interactions and cancellation among input
smoothing and source aggregation. In particular, aggregated bright sources
need more angles. Those contrasts cannot be added as independent percentages
or used to assign the production 13.11% difference a precise causal fraction.
`factorial_summary.json` selects the controlled results; the separate raw
factorial JSONs retain lower-angle results and all convergence checks.

## Interpretation and next model change

The current opaque cell absorbs all its incident ray rates into one neutral
capacity and becomes transparent only after that capacity is exhausted. A
larger cell combines spatially distinct illumination and neutral regions.
Subdividing it resolves transparent paths while other parts remain neutral.
The intervention demonstrates that this discretization changes the
ionization--density association even when total ionization, source locations
and the underlying density are held fixed.

The evidence supports testing additional front subdivision at fixed source
coordinates, concentrating new cells on nonuniform partially ionized fronts.
The current eight children all touch the central source; a 64-child extension
must emit only into the central eight initially and explicitly transport to
the rest. Uniformly dividing the source across all 64 would move the emission
and invalidate this diagnosis.

These experiments establish a causal mechanism in one small native fixture.
Quantifying its share of the production-volume discrepancy still requires the
same independent-source/gas experiment on the large fixtures. No production
ionization rule is changed in this revision.

## Reproduce

The new diagnostic suite passes 17 tests. Across the 66 stored factorial
solves, the largest independent relative global count residual is 6.10e-19
and the largest relative per-source residual is 3.30e-16. These checks certify
the stated discrete accounting and implementation comparisons, not continuum
radiative-transfer accuracy.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest \
  research/pc_scale_origin -q --confcutdir=research/pc_scale_origin
python research/pc_scale_origin/analyze_large.py \
  --summary /path/to/late/summary.json \
  --projection /path/to/late/parent_coarsening.json \
  --output research/pc_scale_origin/large_spectral_results.json
python research/pc_scale_origin/factorial.py \
  --fixture /path/to/native_fields32 --output /path/to/front_check.json \
  --doses .9 --angles 6144 --horizon-boxes 16 \
  --variants D_both_coarse_inputs_gas2 E_coarse_front_gas1
python research/pc_scale_origin/plot_diagnosis.py
```

The high-angle reference requires substantially more rays than the adaptive
production solver. The diagnostic's cost is not a production runtime estimate.
