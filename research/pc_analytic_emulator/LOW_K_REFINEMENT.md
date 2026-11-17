# Training the deployed pure emulator at low k

The revised heads reduce the worst per-snapshot mean low-k 21-cm power error
from 1.471% to 0.879% on 42 fresh snapshots. All 35 non-late snapshots meet a
1% mean-power target. The inference graph remains the same pure exposure
emulator; it neither calls the teacher nor substitutes a teacher field.

## Cause and correction

On the previously inspected seeds 2501–2522, the four >1% cases occurred at
z=6.5. Their predicted ionized mass fractions were high by 0.00068–0.00095.
Keeping the latent spatial field fixed and inserting the teacher absorption
count **only in an offline diagnostic** reduced the worst mean low-k error
from 1.830% to 0.856%. The power discrepancy was primarily coherent amplitude
bias, not random field noise. Restoring the old feature recipe still left all
four errors above 1%, so the feature mismatch alone did not explain them.

The old spatial objective fitted the clipped raw field before photon-budget
projection. It also normalized a mode-weighted aggregate Fourier residual,
although evaluation averages fractional power errors equally across shells.
A small absorption-count error could therefore bias the final low-k power
without being adequately penalized during training.

`exposure_refine.py` now prepares descriptors with the frozen learned onset
head and the exact deployed depth. `exposure_spectral_fit.py` differentiates
through the final bounded photon-budget projection. Training combines a
field loss with equal-shell, phase-sensitive complex brightness residuals;
a supplementary power loss constrains amplitude. All coefficients are fitted
on training seeds. Validation chooses among frozen/refitted reservoir heads
and four spectral weights. Projection and local absorption bounds remain
unchanged. Fully ionized and weak late signals use absolute brightness error.

The selected L50 models retain their original reservoir coefficients and
improve the spatial head **under that predicted budget**. L256 selects a
refitted reservoir head. Consequently the largest mass-fraction error is
unchanged; this update fixes the measured low-k bias rather than claiming
that the absorbed-count predictor has become exact.

## Independent evidence

The spatial-head fit used 124 snapshots from 31 existing development seeds.
The previously reported 250x cohort was explicitly reused as development
validation. All coefficients were frozen before evaluating six new seeds
2801, 2802, 2811, 2812, 2821, and 2822 at z=10,9,8,7.5,7,6.5,6.

For each snapshot the statistic is the arithmetic mean of
`abs(P_emulator/P_teacher - 1)` over spherical shells, excluding DC and
selecting individual modes with `0 < k < k_Nyquist/2`. No positive/negative
power errors cancel in this average. The 21-cm field is proportional to
`density * xHI` in the shared saturated-spin, real-space setup.

| BOX_LEN [cMpc] | HII_DIM | low-k upper bound [cMpc^-1] | Worst snapshot mean, previous | Revised |
| ---: | ---: | ---: | ---: | ---: |
| 50 | 16 | 0.50265 | 1.471% | 0.879% |
| 50 | 32 | 1.00531 | 1.460% | 0.758% |
| 256 | 32 | 0.19635 | 0.530% | 0.310% |

Across the 35 non-late snapshots, the mean of the snapshot means falls from
0.539% to 0.205%. The worst individual low-k shell is 1.100%, so the 1%
result applies to the **per-snapshot average**, not every shell.

Across all 42 snapshots, mean xHI MAE falls from 0.001001 to 0.000637, and
maximum MAE from 0.003461 to 0.002199. The largest mass-fraction error remains
0.001124. The largest independently read-back photon residual is 2.70e-17.
No prediction calls the teacher. Every original field/budget/late-signal
gate passes; all 35 non-late snapshots additionally pass the 1% power gate.

The late low-k absolute brightness residual increases slightly from 0.362%
to 0.396% of the fully neutral RMS, both below the unchanged 1% limit.
Geometric MFP CDF maxima are 0.02295 (ionized) and 0.02905 (neutral), using
4096 shared rays per phase. The earlier single empty-neutral-phase mismatch
is absent. High-k brightness shells are diagnostic: their maximum error
falls from 2.598% to 1.072%. These checks establish improved emulation of
the same-grid teacher; they do not certify the teacher's grid convergence.

The native executable was rebuilt from the same source. Its binary hash
changed; native inputs and teacher arrays for three old control snapshots,
one per geometry, reproduced bit for bit. Every other physical/numerical
signature remained identical. The paired audit rechecks these control arrays
before accepting that runtime difference.

## Reproduce the fit

The released low-k data archive includes `deployed_cache/`, `final_model/`,
all new teacher campaigns, frozen evaluation protocols and saved predictions.
Use the previous pure exposure release as `base_model`. The cache binds its
original manifest hash, onset hash and depth; it can be relocated.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m research.pc_analytic_emulator.exposure_refine fit \
  --model /path/to/base_model \
  --cache /path/to/deployed_cache \
  --output /path/to/refitted_model
```

To prepare descriptors from new supervised arrays:

```bash
python -m research.pc_analytic_emulator.exposure_refine prepare \
  --model /path/to/base_model --manifest /path/to/inputs.json \
  --workers 4 --output /path/to/deployed_cache
```

The manifest is `{ "schema": 1, "records": [...] }`. Each record has
`split` (`train` or `validation`), `seed`, `redshift`, `dim`, `box_len`,
`native_archive` and `native_sha256`. Paths are relative to the manifest.
Each numeric NPZ contains `source`, `capacity`, `native_es_xHI`, and
`teacher_xHI`. The teacher field is used only for the supervised objective.
The preparer rejects repeated snapshots, overlapping seeds and unsupported
geometries. Training cannot consume validation or test-labelled records.

Inference uses the unchanged entry point:

```bash
python -m research.pc_analytic_emulator.exposure_emulator \
  --model /path/to/final_model --snapshot /path/to/native_snapshot \
  --redshift 7.5 --box-len 50 --output /path/to/prediction
```

Support remains DIM=96 and the three geometries above, with the original
native physical parameter set. The teacher is frozen coeval hydrogen with
no recombinations, thermal evolution, photoheating feedback, or independent
helium chemistry. No additional physics was introduced by this refinement.
