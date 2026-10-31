# Numerical improvements

Physical inputs remain the archived 21cmFAST sources, cumulative photon budgets,
gas trajectories, physical light speed and recorded effective-atom opacity.
Numerical changes are tested against independent solutions and then against
the same native histories. Original SCRIPT supplies an external benchmark for
spatial-refinement stability on matched inputs; its field is not a known exact
solution. Case-to-case changes measure numerical sensitivity, not a true-error
bound.

## 1. Continuous source--absorption coupling

The earlier sequence injected photons after its final absorption half step.
In optically thick cells this left an artificial endpoint inventory of about
`q * dt / 2`. It also contaminated a spatial refinement test: a smaller cell
reduces the CFL timestep and hence this inventory, even for a homogeneous field
with no spatial gradients.

The new local half step evolves neutral atoms `N`, directional photons `P_d`
and a constant isotropic emission rate `q` together:

```text
dN/dt   = -kappa * N * sum(P_d)
dP_d/dt = weights_d * q - kappa * N * P_d
```

Let `D = sum(P_d_initial) - N_initial` and

```text
F(s) = exp[-kappa * (D*s + q*s*s/2)]
J    = integral_0^h F(s) ds
K    = integral_0^h (h-s)*F(s) ds
T    = 1 / (1 + kappa*N_initial*J).
```

The exact constant-source solution is

```text
N_final   = N_initial * F(h) * T
P_d_final = T*P_d_initial + weights_d*q*T*(h + kappa*N_initial*K).
```

The directional incoming radiation keeps its own attenuation factor; it is
not isotropized. The code evaluates positive Gaussian moments after scaling
out their maximum exponent. Long-double intermediates and large-argument
moment expansions avoid overflow and loss of a weak source by subtraction.
The ion increment is evaluated as a positive moment expression, so sources
far below neutral-count roundoff still contribute their photons.

For a generic linearly changing source, each reaction half step uses its exact
mean rate. The total injected count remains the exact trapezoid, and the
time-dependent solution converges at second order. The native-cumulative
workflow already has a constant source rate within each matter substep.
Midpoint expansion factors and photon streaming are unchanged in this step.

### Independent verification

A homogeneous test at redshift 7, physical 18 eV opacity and 0.4 emitted photons
per atom has reference ionization 0.3997386834672251 and radiation inventory
0.0002613165327749 photons per atom. The old CFL 0.4 result was 0.3800 and 0.0200.
The coupled update agrees with the independent high-precision solution to
double-precision roundoff at CFL 0.4, 0.2 and 0.1. For linear sources, halving the
timestep reduces the error by approximately four.

Seventeen new tests check weak and stiff absorption, front completion,
anisotropic incoming radiation, varying emission, a 1e-30 source, and transparent
or fully ionized limits. The full causal RT suite passes 199 tests after this
change. Numerical refinement of the native fields is reported separately;
homogeneous agreement alone does not establish spatial convergence.

### Native-history result

Four complete local seed222 histories reuse the exact earlier source and gas
archives. All audits pass, with maximum photon residual 1.08e-14. The change
in the first power shell under CFL 0.4 to 0.2 is 1.107% (previously 1.084%);
CFL 0.2 to 0.1 gives 0.408% (previously 0.418%). Spatial 8^3 to 16^3 refinement
gives 1.686% (previously 1.706%). Thus this fixes an identified local integration
error but does not resolve the native-field convergence problem. At z=7 the
mass-neutral fraction changes 0.37016808 to 0.37000505, with the source and atom
inventories unchanged.

Measured RT times are 24.65, 33.96 and 56.24 s for the three native-grid CFL cases,
and 187.13 s for 16^3/CFL 0.4. These are local case timings under shared load,
not controlled speedup measurements or Gadi estimates. Exact benchmark and
native refinement results are in `source_absorption_validation.json`.

## 2. Positive source-free absorption tail

The independent travelling-front benchmark exposed a roundoff failure in the
source-free reaction. The previous expression `1 - absorbed / photons` could
become slightly negative when almost all photons were absorbed. For a valid
front-cell state with 0.1999934282541434 photons and 0.5999801257685806 neutral
atoms, this produced a survival fraction of about -1.08e-19 and aborted the run.

The update now evaluates the surviving inventory directly. With initial
photons `P`, neutral atoms `N`, and the integrated absorption coefficient `C`,

```text
m = min(P, N)
d = abs(P - N)
phi = (1 - exp(-C*d)) / (C*d), with phi(0) = 1
remaining = m * exp(-C*d) / (1 + m*C*phi) + max(P - N, 0)
survival = remaining / P.
```

Every term is nonnegative. The same analytic absorption equation and directional
attenuation are retained, with no clipping of fields or global normalization.
This is a robustness correction at the opaque limit; it is not evidence of
improved native-field spatial convergence. A regression test uses the exact
previously failing state through the public solver interface.

The causal RT suite passes 202 tests. A complete seed222 D96/f1/CFL0.4 native
history was repeated: all 21 saved checkpoint fields and the final full angular
photon state are bitwise identical to the coupled-source baseline. The maximum
photon residual is 4.03e-15. This confirms no regression on that trajectory;
the new analytic test covers the previously failing corner. Exact comparison
and input hashes are recorded in `source_free_validation.json`.

## 3. Acceptance-goal update: SCRIPT-comparable refinement stability

This is a change to the working acceptance goal, not a solver improvement.
The aim is now spatial-refinement stability comparable to original SCRIPT on
the same archived 21cmFAST inputs, refinement grids, output averaging and
spectral bands. Changing the goal does not by itself establish a new pass or
accuracy claim.

A 5% large-scale power difference is a provisional literature reference only
([Maity & Choudhury 2023, Appendix C](https://academic.oup.com/mnras/article/521/3/4140/7079143)),
not a measured error for these inputs or a replacement universal cutoff. Reports must
identify the tested k range and epochs, compare higher-k refinement changes
with SCRIPT explicitly, and retain neutral-field MAE as a diagnostic without
inventing a new absolute threshold. The native 8^3, 50 cMpc primary band contains
only the shell at k=0.16036 cMpc^-1, within bounds 0.12566--0.25133 cMpc^-1; it
cannot establish a result for larger-scale modes absent from the box.

The original strict profile (default MAE <= 0.01 and primary power changes <= 1%)
and all historical reports retain their original meaning. Physics, solver,
configuration defaults, photon/atom/source-budget audits and saved results are
unchanged. Other numerical sensitivities remain part of the assessment; meeting
the new spatial goal alone would not certify all numerical axes or a true-error
bound.

### Same-input SCRIPT spatial benchmark

Unmodified original SCRIPT was run at all 81 archived nodes of the local
seed222 history, on 8^3, 16^3 and 32^3 allocation grids. Source and gas clouds
retain their physical width when the grid changes. The factor cubed conversion
of packet counts only changes cell-volume units; restricted source and atom
fields agree to roundoff. Ion and atom counts are restricted conservatively
before neutral fractions and native 21cmFAST brightness are computed. All 243
SCRIPT photon ledgers pass their original audit, with maximum relative residual
7.59e-9. This does not alter the RT audit tolerance.

Using the same 21 saved-node cadence as the existing RT comparison, at matched
volume-neutral fraction 0.4, the 8^3 to 16^3 changes are:

| Diagnostic | RT | Original SCRIPT |
| --- | ---: | ---: |
| Neutral-field MAE | 0.01688 | 0.03292 |
| Power change, k=0.1604 cMpc^-1 | 1.77% | 0.40% |
| Power change, k=0.2803 cMpc^-1 | 0.27% | 4.00% |
| Power change, k=0.3939 cMpc^-1 | 5.80% | 14.63% |
| Power change, k=0.4660 cMpc^-1 | 12.39% | 14.62% |

Power changes are absolute ratios relative to the finer result. These are each
method's refinement changes, not RT--SCRIPT field differences or errors relative
to a known solution. The result is mixed: RT has the larger first-shell change
but the smaller MAE and changes in the other three shells. This is evidence of
comparable spatial sensitivity in this example, not an overall precision ranking.
The RT data retain their recorded coupled-source kernel identity; the later
positive-tail correction separately reproduced the native-grid trajectory
bitwise, as documented above.

SCRIPT 16^3 to 32^3 at the same matched fraction gives MAE 0.01771 and power
changes of 0.35%, 3.92%, 0.06% and 4.95%. Finer allocation therefore still changes
SCRIPT's field. Its full 81-node output also exposes a sampling limitation:
at matched fraction 0.2, 21-node interpolation gives MAE 0.01646 for 8^3 to 16^3,
whereas 81 nodes give 0.03219. At fraction 0.4 the corresponding MAEs are 0.03292
and 0.03270. Late-stage matched-fraction comparisons need denser RT output;
the sparse result must not be treated as a convergence certificate.

The reproducible driver is `benchmarks/script_refinement_benchmark.py`; input,
implementation and result hashes, all tested stages and both output cadences
are recorded in `script_refinement_validation.json`. This benchmark leaves
the native physical inputs, RT equations and historical result files unchanged.

From the repository root, with the archive's native extension and a Fortran
compiler available, select a completed native-cumulative RT archive and a new
output directory:

```bash
uv run --no-sync python -m research.pc_causal_rt.benchmarks.script_refinement_benchmark \
  --rt-root /path/to/completed/rt/results \
  --output /path/to/new/script_refinement_results \
  --seed 222 --factors 1 2 4
```

## Remaining front-resolution error

An independent source-free plane wave provides a known solution of the actual
transport and absorption equations: the ion fraction is a travelling logistic
profile, photons are proportional to that fraction, and the front speed is
`c_comoving * P0 / (1 + P0)`. The benchmark uses exact cell averages and compares
only the causal window unaffected by the periodic trailing boundary. Its
cross-section is varied to test resolved and unresolved front widths; this
does not change the production opacity or native inputs.

With the final retained solver at CFL 0.2:

| Grid | Smooth-front ion-fraction MAE | Thin-front ion-fraction MAE |
| --- | ---: | ---: |
| 16^3 | 0.00417884 | 0.0264113 |
| 32^3 | 0.00110877 | 0.0125529 |
| 64^3 | 0.000265895 | 0.00612180 |

The smooth front approaches second-order spatial convergence; the unresolved
thin front remains approximately first order. On 32^3 cells, reducing CFL from
0.2 to 0.1 to 0.05 changes the thin-front MAE only from 0.0125529 to 0.0122640 to
0.0120934. Thus smaller timesteps alone do not remove this spatial limitation.
These are analytic front tests, not cosmological-field power-spectrum errors.
The reproducible driver is `benchmarks/front_benchmark.py`, with results and implementation
hashes in `front_convergence_validation.json`.

From the repository root, write a fresh report without overwriting the recorded
validation data:

```bash
uv run --no-sync python research/pc_causal_rt/benchmarks/front_benchmark.py \
  --repo . --transport-module research/pc_causal_rt/transport.py \
  --output /tmp/front_validation.json
```

## Propagation candidates that were not adopted

The retained propagation method remains MUSCL-MC with SSPRK2. Several isolated
prototypes improved selected benchmarks but failed broader checks:

| Candidate | Reason it was not adopted |
| --- | --- |
| Conservative characteristic remap | Fixed-grid temporal convergence fell to first order. |
| Parabolic reconstruction with smooth-extremum switching | A sharp oblique pulse developed a 16.8% overshoot. |
| WENO3-JS with SSPRK3 | Smooth and discontinuous transport errors were larger than the retained method. |
| Published MP5 with SSPRK3 | The tested multidimensional implementation produced a 16.0% sharp-pulse overshoot. |
| MP5 with a global intensity bound | A distant bright pulse allowed a weaker front to overshoot by 15.7%, even though global positivity, maximum and photon budgets passed. |

The last candidate reduced the N32 single-pulse L1 error by about 22%, but
that result did not justify the nonlocal limiter behavior. Two new regression
tests now cover both a single sharp pulse and separated pulses of unequal
brightness. Candidate evidence is recorded separately in
`rejected_streaming_candidates.json`; none of these experimental propagation
kernels is used by the production solver.
