# Autonomous photon-conservation and emulator validation — 2026-11-16

This run checks the analytic coeval teacher and its emulator separately.
The PC solver conserves independently recorded photon inventories. A new
native simulation campaign tests whether the emulator also predicts the
teacher's absorption and spatial field. Matching a ledger is not used as a
substitute for those field comparisons.

## Solver checks and repairs

The Python finite-exposure absorption expression previously multiplied two
small optical-depth factors before dividing by their coefficient. The
intermediate product could underflow even when the final absorbed count was
representable: `P=H=1e100, coefficient=1e-300` returned zero instead of approximately
`1e-100`. Factoring the expression restores that weak-absorption limit. Complex
physical arrays are now rejected instead of discarding their imaginary parts.
The production C++ transport kernel and physical parameters are unchanged.

Independent checks cover the weak/opaque limits, pre-ionized inventories,
zero sources, finite bounds, count-unit rescaling, direction/rotation inputs,
and open, source-periodic-cube and recycling-periodic boundaries. Twenty-four
heterogeneous cases were run with both one and two threads; all paired fields
were bitwise identical. Their worst independent ledger residual was
`3.36e-15`. All sixteen open/finite-domain cases completed. Five of eight
recycling-periodic cases reached the deliberately finite work limit; their
unemitted and remaining photons were retained and completion remained false.
Causes include restricted angular path coverage and slowly decaying tails.

See [the inventory stress report](validation_20261116/solver_inventory_stress.json)
and the added solver/absorption regression tests.

## Same-input comparisons with SCRIPT and C²-Ray

Original native seed 222 arrays were checksum-verified, then rerun through the
current analytic solver and unmodified upstream SCRIPT Fortran. The latter's
outputs reproduce the archived SCRIPT arrays bitwise with the newly recovered
compiler. C²-Ray values below come from the previously completed, independently
verified native execution; C²-Ray was **not rerun in this diagnostic session**.
The adapter verifies identical source/atom arrays, native parameters and
cross-section before including that reference.

Geometry is `L=50 cMpc`, `N=16`, so cells are `3.125 cMpc`. The default analytic
case uses 96 directions and 32 exposure subdivisions. The lower-k column below
contains the three available shells below half Nyquist, not a certified band.
The common saturated-temperature brightness coefficient cancels in power ratios.

| Measurement | z=8 | z=7.5 |
| --- | ---: | ---: |
| Analytic vs SCRIPT xHI MAE | 0.01784 | 0.04037 |
| Analytic vs SCRIPT maximum lower-k power difference | 3.04% | 7.25% |
| Closed-periodic analytic vs SCRIPT maximum lower-k power difference | 3.06% | 6.64% |
| Analytic vs archived C²-Ray xHI MAE | 0.01637 | 0.02423 |
| Analytic vs archived C²-Ray maximum lower-k power difference | 11.66% | 8.44% |
| 96→384 directions: maximum lower-k power change | 1.34% | 0.83% |
| 32→64 exposure subdivisions: maximum lower-k power change | 0.043% | 0.089% |

Analytic ledger residuals are at most `9.85e-17`. SCRIPT's original-input ledger
residuals are `1.05e-10` and `5.41e-10`; original REAL32 input conversion and
allocation precision are retained. C²-Ray lacks independently separated stored
and escaped photon counters in this reference, so its strict ledger remains
undefined rather than being inferred from missing photons.

Geometric MFP uses common random draws, `xHI=0.5` phase boundaries and a one-box
distance cap. Analytic–SCRIPT phase CDF differences are about 1.0–1.3 percentage
points. Capped rays remain in the restricted means, and their fractions are
reported. This statistic describes geometry rather than physical opacity.

Conservatively restricting the same inputs from N16 to N8 gives near-grid power
changes of approximately 42–60% for PC/SCRIPT. The common lower-k band then has
only one shell. These tests do not certify spatial convergence or establish that
every mode below half Nyquist is accurate.

Full shell spectra, cross-correlations, MFP and grid/boundary results are in
[reference_summary.json](validation_20261116/reference_summary.json).

## Physical interpretation

The native cumulative source and density prescriptions are shared. The source
array is not multiplied by density again or integrated as an emission rate.
The native effective H+He atom count convention is preserved, with one hydrogen
opacity per effective atom. This remains a frozen coeval calculation without
recombination, thermal evolution, photoheating feedback or independent helium
chemistry.

Native ES applies `R_BUBBLE_MAX` to its maximum filtering scale. Analytic PC
uses resolved neutral opacity and the selected boundary; it does not apply
that ES MFP prescription. Reports now expose both definitions explicitly under
`physical_mapping`. The same source inputs therefore do not imply identical
propagation physics or identical ionization morphology.

## Fresh native simulations and initial emulator diagnosis

Eight genuinely new native realizations, seeds 2101–2108, were generated at
z=10,9,8,7.5,7,6.5,6. The output grid is N16 in a 50-cMpc box, while the native
density grid is N96. All 56 teacher snapshots completed; their maximum
independent photon residual is `9.35e-17`.

The repeated native campaign reproduced 55/56 snapshots bitwise. One native
float32 source array differed by at most `2.38e-7`; its teacher difference was
at most `1.60e-7`. Running the final solver on that case's unchanged original
arrays reproduced every transport array bitwise. The difference arose in native
regeneration, not in the PC fixes.

The initial v3 experiment used 35 training snapshots (seeds 2101–2105), 14 validation
snapshots (2106–2107) and 7 previously unseen test snapshots (2108). Four network
initializations ran with two workers and two threads each. Training took 95.16 s,
excluding the initial teacher audit. Validation selected residual initialization 47;
its objective was 3.83% below the same-run best legacy model.

An independent evaluation recomputed fields, full-complex-FFT spectra and
voxel-traced MFP without the training/evaluation helper functions. On the six
active test snapshots (`0.05 < teacher mass-weighted xHI < 0.95`):

| Predictor | xHI MAE | xHI RMSE | Mean lower-k power-ratio error |
| --- | ---: | ---: | ---: |
| Native ES | 0.04954 | 0.12516 | 19.41% |
| Same-run legacy model47 | 0.03284 | 0.07279 | 13.22% |
| Validation-selected v3 residual47 | 0.03384 | 0.07345 | 15.28% |

Thus the residual's validation gain did not generalize to this seed. At z=6,
both learned models predicted complete ionization whereas the teacher retained
mass-weighted `xHI=0.00520`. Including that snapshot gives the selected model a
mean power-ratio error of 27.38%, compared with 25.27% for native ES. The emulator
still closes its predicted global photon ledger to `2.95e-17`.

The budget parameterization explains an avoidable failure mode: the old scalar
head uses a residual about eta=1. Its positive tanh branch has no available upward
capacity, so it predicts eta=1 with zero local gradient. This occurred even for
a training snapshot with nonzero teacher neutral fraction. The training data
already cover late neutral fractions from 0.000765 to 0.02999; missing epoch
coverage is not the explanation for that example.

The first results, paired legacy comparison and exact endpoint diagnosis are
preserved in [first_independent_test_summary.json](validation_20261116/first_independent_test_summary.json),
[first_paired_legacy_comparison.json](validation_20261116/first_paired_legacy_comparison.json)
and [budget_endpoint_diagnosis.json](validation_20261116/budget_endpoint_diagnosis.json).
Seed 2108 becomes a regression test for subsequent development.

## Budget candidate and four fresh external tests

A direct-sigmoid efficiency network was added to the v3 candidate set. Its
initialization and normalization use training samples only. The residual MLP,
tables and constant estimator remain eligible, and the complete deployed
field/budget/projection combination is still selected on the original validation
seeds. Tests verify finite gradients away from saturation, saved-model reload,
and identical budget targets in fitting and inference.

The additional candidate did **not** win validation. The deployed model remains
residual initialization 47 with the original fitted budget; its predictions are
unchanged. This code change therefore does not establish an accuracy improvement.
The final training workflow took 94.05 s, excluding the initial archive audit.

Four additional native realizations, seeds 2109–2112, supplied 28 fresh snapshots
after the candidate change. Their labels were used only for post-selection
evaluation. Combined with the first campaign, all **84 native teacher snapshots**
completed. Independent recomputation gives a maximum teacher photon residual of
`1.02e-16` and a maximum external emulator residual of `2.66e-17`.

The table averages the 24 active external snapshots, with equal snapshot weights.
Power uses individual Fourier modes satisfying `0 < k < 0.5 k_Nyquist`, four
shells in this box. This is a measured comparison band, not a convergence claim.

| Predictor | xHI MAE | xHI RMSE | Mean lower-k power-ratio error |
| --- | ---: | ---: | ---: |
| Native ES | 0.04779 | 0.12352 | 17.28% |
| Validation-best legacy model 47 | 0.03175 | 0.06735 | 14.01% |
| Validation-selected v3 residual 47 | 0.03219 | 0.06687 | 15.91% |

The emulator reduces mean field MAE by 32.6% relative to native ES. The residual
model has slightly lower RMSE than legacy but higher MAE and power error. There
is no consistent improvement over the simpler legacy model on fresh seeds.

Independent geometric MFP uses 32,768 rays per field and phase. Across the active
snapshots, mean absolute restricted-mean errors are 6.24% for the ionized phase
and 3.48% for the neutral phase (native ES: 13.07% and 8.71%). Mean maximum CDF
differences are 0.0343 and 0.0194. These measure thresholded geometry at the same
redshift; they do not validate physical absorption lengths.

Late-stage errors remain material. At z=6 the selected model's mean lower-k
power-ratio error is 254.8%; including these four snapshots increases the overall
28-snapshot average to 50.0% (native ES: 46.2%). Small reference power increases
relative errors, but the neutral fraction is also mispredicted:

| Seed, z=6 | Teacher mass-weighted xHI | Emulator mass-weighted xHI |
| --- | ---: | ---: |
| 2109 | 0.002783 | 0.000000 |
| 2110 | 0.002788 | 0.007077 |
| 2111 | 0.013240 | 0.014963 |
| 2112 | 0.004098 | 0.018768 |

At this epoch, seed 2109 has no predicted neutral phase, so its neutral MFP is
undefined. The other three neutral-phase restricted-mean errors average 26.56%.
Ionized rays frequently reach the box-distance cap; near-box-length means cannot
establish accurate late morphology.

Thus global conservation passes while the absorption estimator and late neutral
islands remain inaccurate. These endpoint results are retained separately and
were not used to select a replacement model. See
[the final training summary](validation_20261116/final_training_summary.json) and
[the independent external audit](validation_20261116/external_independent_compact.json).

## Reproducibility and standalone prediction

The training implementation identity now includes `warmstart.py` and the shared
physics/photon-audit module; changing either invalidates recovery of old work.
The original completed training resumed with all 212 file contents and modification
times unchanged. The standalone inference command checks native parameters and
array hashes and reproduces all seven initial held-out predictions and all 28
external predictions bitwise. The final saved model also survives a separate
save/reload round trip. Final resume verification leaves all 275 output files
unchanged; the completion receipt independently covers 272 hashed artifacts.
See [the inference verification](validation_20261116/external_inference_verification.json)
and [the final resume verification](validation_20261116/final_resume_verification.json).

The restored native runtime was built from the recorded source parent. The fork
has no release tags, so default source-version inference produced 0.0.1 and exposed
the native HDF5 compatibility layer's negative-major-version error. The build
used the explicitly recorded `SETUPTOOLS_SCM_PRETEND_VERSION=4.3.2.dev0+g5d29a9c2`;
this changes package metadata, not native physical source code. Compiler/runtime
identities and the original failed attempts were retained.

All simulations and training described here ran locally. Gadi was not contacted
successfully because its hostname could not be resolved from this environment.
The existing frozen Gadi launchers remain available.

The targeted suite completed with **618 passed, 1 skipped, and 10 subtests passed**.
The skip requires the separately pinned pyC2Ray source/compiler combination.
Ruff lint/format, six shell launcher syntax checks and `git diff --check` passed.
The new physical-parameter and native-array checks in standalone prediction were
also exercised on real native inputs, including the saved direct-sigmoid candidate.

The complete native campaign and training configurations are in
[reproduction/](validation_20261116/reproduction/). From the repository root:

```bash
python research/pc_analytic/validation_20261116/reproduction/reproduce.py \
    --source "$PWD" --output /absolute/new_validation_run \
    --python /absolute/native_runtime/bin/python
```

This generates the 56 fitting/regression teachers, the 28 external teachers and
the final seed-separated training run. The native runtime needs the recorded
GSL/FFTW/compiler dependencies. Configurations preserve the numerical and
physical settings used here. Full fields, models, plots and detailed audit
arrays are saved separately; compact machine-readable evidence is committed
beside this report.
