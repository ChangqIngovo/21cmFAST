# Completed emulator improvement and independent validation

The selected emulator reduces neutral-field and 21-cm power errors on new
realizations of the training geometry. It does **not** transfer reliably to a
different grid or box. Its supported evaluation domain is the recorded native
physics at **L=50 cMpc, HII_DIM=16, DIM=96**, with the tested redshifts 10–6.
The teacher transport, source convention and physical switches are unchanged.

## Data and model selection

The campaign contains 236 unique native/teacher snapshots from 32 initial-condition
seeds. This work generated 152 new snapshots and reused 84 audited development
snapshots. An additional independent rerun of the 28 validation snapshots reproduced
all 280 saved arrays bit for bit.

| Role | Seeds | Snapshots | Geometry |
| --- | --- | ---: | --- |
| Training | 2101–2112, 2121–2128 | 164 | L50, N16 |
| Validation | 2113–2116 | 28 | L50, N16 |
| Fresh blind test | 2117–2120 | 28 | L50, N16 |
| Grid transfer | 2131–2132 | 8 | L50, N32 |
| Box/grid transfer | 2141–2142 | 8 | L256, N32 |

The additional training realizations include z=6.75, 6.25 and 5.75 to sample the
late absorption deficit. All redshifts of a seed remain in one split. Previously
inspected seeds 2101–2112 are development data. No blind/transfer labels entered
fitting, checkpoint selection or the following configuration comparison.

Both training configurations used the same 164/28 split, four model seeds
(11, 23, 47, 83), budgets, projection choices and projected field/Fourier objective.
An unchanged previously deployed model was evaluated on the same validation data.

| Candidate | Validation objective | Validation field MSE |
| --- | ---: | ---: |
| Unchanged previous model | 0.272473 | 0.00723225 |
| Expanded training, original 24 features | 0.140976 | 0.00600408 |
| Expanded training, 88 features | **0.100520** | **0.00354594** |

The selected model uses seed **83**, `multiscale_local` features, a **tabulated
absorption budget**, `capacity_l2` projection and a learned residual around its
frozen base. The base also uses 88 features and the tabulated budget. Selection
was frozen before any blind/transfer evaluation. The saved selection SHA256 is
`493a9cdc3ba74c05131e550173eaaacf356ca3a9451e62bf05935a2668297efe`.

## Changes that were tested

1. **Local spatial information.** The 24 original native-only features are
   supplemented with 64 bounded features: face/edge/corner moments, local
   contrasts, scalar gradient alignments, and Gaussian moments/covariances at
   one and two cell widths. Inputs remain native source, atom and ES fields;
   no teacher fields are used during inference. The paired validation objective
   improves by 28.7% relative to expanded training with the original features.
2. **Absorption deficit.** A direct sigmoid model of `1-eta` was tested with
   mass-fraction and late neutral-fraction losses. The common scalar validation
   loss was 0.005115 versus 0.001411 for the table. A separate fixed 24-candidate
   kernel study reached 0.003097. Neither replaced the selected table. These
   rejected alternatives are retained as reproducible experiments.
3. **Auditing and execution.** Multi-campaign loading verifies original receipts,
   physical settings and seed separation. A development-only fit can finish
   without reading test labels. The orchestration freezes models before testing,
   checks resume identity and retains the old model as a validation-scored fallback.
   Budget parameterization is included in frozen-inference checksums.
4. **Training cache.** The expanded feature/base cache occupies about 564 MiB per
   worker. Replaying the actual access sequence with a 512 MiB cap caused 88,
   83 and 96 reloads in three epochs; 1024 MiB caused none after warmup.
   `frame_cache_MiB` is configurable without changing numerical results. Default
   512 MiB preserves existing runs. Cache-on/off weights, histories and predictions
   were bitwise identical in the controlled test.

The actual two training workflows took 903 s and 1591 s respectively, each using
two processes with two numerical threads. They ran concurrently. These times
used the original 512 MiB cache; the cache replay is not an end-to-end speedup
measurement. The frozen training source predates the final cache and metadata
guard additions; its file identities are retained. Those additions do not alter
the numerical predictor.

## Independent blind results

All rows below compare the same new seed realizations against the analytic
teacher. “Active” means `0.05 < teacher mass-weighted xHI < 0.95`: 24 of 28
snapshots. Each snapshot has equal weight. Power error is the mean absolute
ratio error over defined shells, after selecting individual Fourier modes with
`0 < k < 0.5*k_Nyquist` (here k<0.50265 cMpc^-1). Brightness uses the common
saturated-spin-temperature, real-space prescription without RSD.

| Active-snapshot metric | Previous | Selected | Native ES |
| --- | ---: | ---: | ---: |
| xHI MAE | 0.03437 | **0.02282** | 0.05355 |
| Mean per-snapshot xHI RMSE | 0.07244 | **0.04856** | 0.13912 |
| Mean low-k power error | 17.01% | **9.89%** | 17.97% |
| Mean absolute mass-ionized-fraction error | **0.002327** | 0.003574 | 0.021465 |
| Ionized geometric-MFP mean absolute error [cMpc] | 1.186 | **0.930** | 1.099 |
| Neutral geometric-MFP mean absolute error [cMpc] | 0.355 | **0.314** | 1.244 |

Active-field MAE falls by 33.6% and mean power error by 41.9%. Improvement is not
uniform: the worst active snapshot's power error rises from 36.3% to 48.1%.
At z=6, mean MAE rises from 0.002654 to 0.004138, while mean defined power error
falls from 194.8% to 60.0%. Only three of four z=6 teacher spectra are nonzero;
the remaining ratio is undefined, not counted as agreement. The larger mean
mass-fraction error is retained in the result rather than hidden by the exact
photon ledger.

Across all 28 snapshots, MAE changes from 0.02984 to 0.02015 and mean defined
power error from 36.77% to 15.46%. Mean model inference takes 0.02185 s for the
previous model and 0.04009 s for the selected model, including features and
projection but excluding I/O and diagnostics. Native ES runtime is not measured
by this benchmark.

MFP is a geometric first-exit statistic of the thresholded xHI=0.5 phases, using
4096 common-random rays per phase. It is capped at one periodic box length;
capped rays enter the restricted mean and empty phases remain undefined. It is
not an independently solved frequency-dependent photon mean free path.

## Remaining error and failed geometry transfer

The frozen selected weights were then applied without tuning to two additional
geometries. The L256 arm changes box size, cell width and native physical density
resolution together; it is not an isolated volume experiment.

| Active snapshots | Previous MAE | Selected MAE | Selected low-k power error |
| --- | ---: | ---: | ---: |
| L50, N16: 24 | 0.03437 | 0.02282 | 9.89% |
| L50, N32: 6 | 0.08271 | **0.15307** | **42.14%** |
| L256, N32: 5 | 0.13372 | **0.20961** | **198.28%** |

These weights should not be used as a validated N32 or 256 cMpc emulator.
At L50/N32 the mean mass-fraction error improves to 0.002374 while the spatial
error grows, directly separating photon-total accuracy from morphology.

Fixed-weight oracle interventions replace predicted absorbed counts with the
true teacher counts at every model layer. Active MAE becomes 0.02182 on the
blind L50/N16 data, 0.15365 on L50/N32, and 0.20555 on L256/N32. Thus correcting
the photon totals does not repair the transfer failure. The dominant remaining
problem there is the learned spatial mapping. Single-geometry training and the
change in physical length represented by local cell features are plausible
contributors; these experiments do not isolate their separate causal effects.
Oracle interventions are diagnostic controls, not additive error fractions or
deployable predictions. No model was retuned after these tests. The inspected
test seeds must be treated as regression data in future development.

Within the training geometry, late errors retain a budget contribution. At
z=6.5, replacing the final budget reduces low-k power error from 37.1% to 22.0%;
replacing frozen-base budgets as well reduces it to 16.2%. This additional change
demonstrates propagation of base-budget errors into the residual correction.
At z=6 it changes from 60.0% to 28.3% for the final-only oracle and 28.6%
for the recursive oracle, with three defined teacher spectra. The [fixed-weight controls](completion_20261116/oracle_evidence.json)
retain the full distinction between final-only and recursive interventions.

## Conservation, physical scope and verification

Independent audits recompute ionized atoms from saved fields and combine them
with separately stored photon reservoirs. The maximum teacher residual over
the campaign is 1.47e-16; the maximum benchmark residual is 1.41e-16. The
emulator constrains its predicted global absorbed/unabsorbed split. Accuracy of
that split is measured separately against the teacher.

The selected model passes 55 rotation/reflection/translation/count-unit checks:
maximum field discrepancy 2.78e-8. Serialization reproduces predictions exactly.
Standalone inference on early, intermediate and late validation snapshots
reproduces the model outputs and passes saved-field ledger checks. Native input
physics is checked before inference; benchmark geometry changes are recorded.

The final affected suite reports **688 passed, 1 skipped**. The skip requires
the pinned C2-Ray source and Fortran compiler. Ruff, formatting and shell syntax
checks pass. Independent slab/FFT/DDA metric self-checks pass. No teacher
absorption/transport or native physical kernel was changed in this work.

The physical scope remains frozen coeval cumulative native photons, initially
neutral gas, no recombinations, thermal evolution, photoheating feedback or
independent helium chemistry. Native ES uses its recorded filtering-radius
prescription; the teacher uses its recorded optical-depth/domain-boundary
prescription. The new field-fit results do not establish convergence to full RT.

## Code, artifacts and execution

- [Complete photon-conservation source map](../PHOTON_CONSERVATION.md)
- [Actual development evidence](completion_20261116/development_evidence.json)
- [Blind and transfer numbers](completion_20261116/benchmark_evidence.json)
- [Generate native teachers, fit, freeze and test](completion_20261116/reproduction/README.md)
- [Gadi launcher and resource settings](completion_20261116/reproduction/GADI.md)

The executed runs were local; the Gadi submission path was tested with a mocked
scheduler and was not submitted to Gadi. Reproduction scripts regenerate native
inputs rather than depending on this machine's collection paths. Training and
inference weights are distributed with the full diagnostics archive. Archive
filenames and checksums are recorded in `completion_20261116/artifacts.json`;
downloads are supplied in the private conversation. Public evidence replaces
machine-specific paths with `${CAMPAIGN_ROOT}`, `${PREVIOUS_CAMPAIGN_ROOT}` and
`${REPOSITORY_ROOT}`. Its recorded checksums refer to the original archived files.
