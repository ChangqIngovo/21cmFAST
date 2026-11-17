# Low-k emulator review, 18 November 2026

The review found no error that invalidates the reported low-k improvement.
The released weights and inference implementation are unchanged. This review
adds checks to the training data interface and one spectral-loss parameter.

## Findings and corrections

1. **Training provenance was incomplete.** The refinement preparer checked a
   compact NPZ checksum and geometry but did not independently verify its
   physical parameters, solver definition or declared seed. A relabelled copy
   could consequently enter a different data split. New preparation uses
   audited teacher campaigns and binds their actual sample identities and
   physical signatures to the feature cache. Historical compact caches require
   an explicit whole-manifest checksum to reproduce an existing experiment;
   this does not retrospectively certify missing provenance.
2. **A zero spectral variance floor was accepted.** With zero teacher power,
   this setting could overflow the relative power loss. The floor must now be
   strictly positive. The released fit used the unchanged default, `1e-8`.

The released 166-record refinement manifest has no repeated native archive
hash across its training and validation splits. Its final six test seeds are
disjoint from all recorded development seeds. The provenance finding concerns
what the training interface would accept, rather than evidence that the
released evaluation mixed its data splits.

## Independent checks

- Re-read all 84 saved predictions from the old and revised models, verified
  native arrays and frozen model hashes, and independently recomputed the
  field, budget and spectral metrics. The maximum metric discrepancy was
  `6.66e-16`.
- Re-ran actual native-input inference at `z=6.5` in all three supported
  geometries. Predicted and raw fields, absorbed and unabsorbed counts, and
  absorption efficiencies reproduced the saved values bit for bit.
- Finite-differenced the complete projected objective on nine real training
  snapshots, covering all three geometries and `z=8,6.5,6`. The largest absolute
  directional-derivative discrepancy was `2.93e-11`.
- Prepared six development snapshots with the new audited input pipeline on
  two workers, checking 315 archived source files. Cached and direct inference
  agree to at most `4.44e-16` in xHI across all three geometries.
- All 161 focused regression tests pass, covering projection, spectral fitting,
  training provenance, native-data audit, inference and accuracy metrics.

Machine-readable evidence is in
[`evaluation_audit.json`](validation/lowk_recheck_20261118/evaluation_audit.json)
and [`spectral_gradients.json`](validation/lowk_recheck_20261118/spectral_gradients.json).
Model and input checksums are recorded in those files.
The input-pipeline check is recorded in
[`prepare_integration.json`](validation/lowk_recheck_20261118/prepare_integration.json).

## What remains approximate

| Quantity | Current independent result |
| --- | ---: |
| Worst non-late snapshot mean low-k power error | 0.879% |
| Largest individual low-k shell error | 1.100% |
| Individual low-k shells above 1% | 4 / 232 |
| Largest absolute mass-ionized fraction bias | 0.001124 |
| Largest late brightness residual / fully neutral RMS | 0.396% |
| Independent global photon-ledger residual | approximately `2.7e-17` |

The average power criterion is satisfied; it does not require every shell to
be below 1%. Global ledger closure concerns predicted absorption plus the
predicted unabsorbed reservoir. The nonzero mass-ionized bias measures the
remaining disagreement with the teacher's absorbed count.

There are six independent test realizations with seven epochs each, not 42
independent realizations. Evidence covers the three declared geometries and
the original physical parameter set. It validates the emulator against its
same-grid teacher, not the teacher's continuum limit or omitted recombination,
thermal and photoheating processes. The late brightness and ionized-phase MFP
maxima regress slightly relative to the previous model; those changes remain
visible in the original report.
