# Spatial transport correction

## Isolating the error

The original opaque two-source test showed primary photon deficits of about
2.17%, 9.38%, 9.35% and 6.74% at 16, 48, 80 and 144 cells per axis. Source
positions, density, physical box and elapsed time were held fixed.

`geometry_audit.py` removes chemistry and column interpolation entirely. In a
uniform monochromatic medium with cell optical depth one, analytic incoming
columns plus the native geometry give total absorption 0.90504 rather than
unity. The neglected exterior tail is below 1e-8. This isolates a geometric
mechanism without source competition or a nonlinear chemistry solve.

The source cell uses a half-cell path in every direction; external cells use
the spherical-shell weight `V / (4 pi r^2 path)`. These independent weights
do not define continuous Cartesian flux accounting. Integrating only the
source cell over its actual cube reduces the deficit to 3.30%, but leaves
an external-cell geometric error. This experiment does not attribute every
error in every inhomogeneous field to geometry.

## Correction

The optional `conservative_rays` mode tiles the sky using six cube faces.
Each pixel has its exact solid-angle weight `w = Omega / (4 pi)`. A midpoint
ray crosses actual Cartesian faces with DDA. No measured deficit enters its
weight or source luminosity.

For source s, ray a and crossed cell i, primary deposition is

\[
A_{sai}=w_a[\Phi_s(\mathbf C_i)-\Phi_s(\mathbf C_{i+1})],\qquad
\mathbf C_{i+1}=\mathbf C_i+\mathbf n_i\ell_i.
\]

The vector C contains HI, HeI and HeII columns; Phi uses the existing spectral
tables. Outgoing columns become the next cell's incoming columns. Hence

\[
\sum_i A_{sai}+E_{sa}=w_a\Phi_s(0),\qquad
\sum_{a,i}A_{sai}+\sum_aE_{sa}=\Phi_s(0).
\]

The source cell has its direction-dependent path to the cube boundary.
Deposits use the actual receiving volume. H/He absorption partition,
secondary ionization, recombination, collisional ionization and thermal
equations are retained. Secondary events are excluded from primary closure.

Two numerical changes accompany this transport:

* Primary absorption uses the same spectral table at both endpoints. Mixing
  a thin-cell derivative approximation with a downstream table value would
  break the telescoping identity.
* DORIC averages contain `(exp(lambda dt)-1)/(lambda dt)`. Subtraction near
  zero amplified tiny summation differences into approximately 4e-8 changes
  in average fractions. A convergent series evaluates the same expression
  for `abs(lambda dt)<0.1`. The reaction equations are unchanged.

The finite source-centred tracing domain and periodic density mapping remain;
outflow is recorded without reinjection. LLS is rejected in this new mode.
The default native control is preserved. Corrected builds record their exact
patch and explicitly set `numerical_algorithm_unchanged=false`.

## Convergence definition

The fixed probe uses two 5e48 photons/s sources, a 5e22 cm proper box, uniform
H+He nuclei density 1e-3 cm^-3, an illuminated interval of 3e11 seconds and
an equally long dark interval. Odd refinements preserve the physical source
positions exactly.

`probe_compare.py` conservatively projects to a declared, fixed target grid
by exact cell-volume overlaps. It reports ionization L1 differences relative
to the newly ionized amount, total ionization changes, temperature L1 relative
to the thermal change, and integrated neutral-fraction power below half the
target Nyquist frequency. It separately reports the unfiltered RT-grid
difference. A front only a few fine cells wide need not be resolved even when
its target-grid cell averages agree.

Target tolerances are 1% for ionization L1, 0.5% for total ionization, 5% for
thermal L1 and 1% for integrated low-k power. Require successive spatial
refinements and a separate angular refinement at the highest resolution.
These are fixture-specific research tolerances. Cosmological histories need
their own checks, including time resolution and the frozen-density assumption.

## Measured results (2026-10-26)

The corrected executable, including stable time averages, was rerun with
24,576 rays per source (angular order 64). Both the illuminated and dark
intervals pass the fixed 16^3 target diagnostics. The illuminated results are:

| RT meshes | Ionization L1 on fixed 16^3 target | Unfiltered ionization L1 on coarser RT mesh |
| --- | ---: | ---: |
| 80^3 to 144^3 | 0.580% | 12.77% |
| 144^3 to 208^3 | 0.269% | 9.65% |
| 208^3 to 272^3 | 0.0595% | 5.89% |

The two columns use the same newly ionized amount for normalization but
different spatial projections. These are successive-resolution differences,
not errors against an exact solution. Unfiltered differences are evaluated
on different coarser meshes; they do not alone establish a convergence order.

At 272^3, increasing angular order from 64 to 128 (98,304 rays per source)
changes target-16 ionization by 0.00265%, target-64 ionization by 0.0270%, and
the unfiltered field by 1.47%. Thus angular accuracy also depends on the
spatial scale being tested. Primary absorption plus boundary escape closes
to approximately 1e-13 of active-band emission; this is a transport identity,
not a certificate of the chemistry time integral or spatial morphology.

The higher-resolution target check is intentionally retained: 208^3 to 272^3
differs by 0.0655% on target 32^3, but by **1.0663% on target 64^3**, failing
the declared 1% L1 threshold. No tolerance was relaxed. The tiny ionized
regions span only about four finest cells in radius. Resolving their sharp
fronts to 1% needs further spatial and angular refinement. This work repairs
the geometric conservation defect and demonstrates convergence of the coarse
cell averages; it does **not** certify the fine front or a cosmological box.

Machine-readable [results](spatial_results/) include input/output hashes,
numerical patch hashes, both intervals, all thresholds, the failed target-64
test, the independent geometry experiment and parallel checks. The native
1/8/32-worker configurations agree under the existing tolerance; these were
correctness tests on an 8-CPU local host, not a 32-core speed benchmark.
Targeted tests also cover opaque/transparent photon closure, a dark interval,
periodic source translation, coincident-source splitting and the compiled
stable average against 60-digit arithmetic.

## Reproduction

```bash
python research/c2ray_teacher/geometry_audit.py --output /fresh/geometry.json

python research/c2ray_teacher/spatial_probe.py \
  --upstream /path/to/pinned/C2-Ray3Dm1D_Helium \
  --output /fresh/spatial64 --transport conservative_rays \
  --refinements 5 9 13 17 --angular-order 64 --threads 32

python research/c2ray_teacher/spatial_probe.py \
  --upstream /path/to/pinned/C2-Ray3Dm1D_Helium \
  --output /fresh/angular128 --transport conservative_rays \
  --refinements 17 --angular-order 128 --threads 32

python research/c2ray_teacher/probe_compare.py \
  --reference /fresh/spatial64/q13 --candidate /fresh/spatial64/q17 \
  --target-dimension 16 --output /fresh/spatial_comparison.json

python research/c2ray_teacher/probe_compare.py \
  --reference /fresh/spatial64/q17 --candidate /fresh/angular128/q17 \
  --target-dimension 16 --output /fresh/angular_comparison.json
```

The cosmological runner accepts `"transport": "conservative_rays"` and
`"angular_order": 64`. Gadi preparation accepts `C2_TRANSPORT=conservative_rays
C2_ANGULAR_ORDER=64`. There are `6 * angular_order^2` rays per source; cost
depends on source count and path length. This is an RT teacher implementation,
not a demonstrated low-cost field emulator.

Large output handling uses Python integer offsets and streaming hashes. The
272^3 two-step output exposed an int32 file-size overflow in the old reader.
A sparse-file regression test covers it without writing dense test data.
