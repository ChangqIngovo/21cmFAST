# Pure coeval photon-conserving emulator

The [low-k refinement](LOW_K_REFINEMENT.md) trains through the deployed
photon-budget projection and reduces the worst independent snapshot mean
low-k power error to 0.879%. Use its refined model and training recipe.

The model takes the same 21cmFAST cumulative source, atom-capacity and
excursion-set arrays used by the analytic teacher. It predicts one coeval
neutral field and a global unabsorbed reservoir. All snapshots follow the
same inference graph, including late overlap and fully ionized boxes.

## Why the previous field was wrong

Changing only the predicted total photon absorption did little to repair the
worst spatial errors. On the previous confirmation set, replacing the learned
counts throughout the old network with revised predictions reduced its maximum
mass-fraction bias from 0.04966 to 0.00455, yet only 29 of 42 snapshots passed
all gates. Supplying the true absorbed counts to the old field network changed
the worst L50/N32 case's MAE from 0.07908 to 0.07059. In a different late snapshot
the old network spread neutral gas over 84.7% of cells at xHI > 0.001, versus 5.63%
in the teacher.

These results motivate modeling when dense regions become transparent. In
the optically thick limit a donor's contribution along a given voxel prefix
starts when the last upstream cell has ionized. Its integrated contribution
is proportional to the donor rate times the fraction of exposure remaining
after that onset. Multiplying source excess by an isotropic scalar loses
that ordering. Applying this prefix relation with observed onset labels
passed 98/98 archived diagnostic snapshots, with field MAE about 0.0011.
Those labels were extracted from a temporary offline replay whose final
field and photon reservoirs were bitwise identical to each archived teacher.

## Inference and validation

A supervised tree ensemble estimates saturation exposures from the native
arrays. Sixteen shared exposure layers pool native inputs along fixed,
donor-centred voxel prefixes. These layers operate on latent onset grids
rather than photon trajectories or evolved photon inventories. A fitted
spatial head corrects the resulting finite-exposure field error. A separate
fitted reservoir head uses terminal exposure and field summaries to estimate
the global unabsorbed count. Projection enforces the resulting absorbed count
within the local absorption bounds. There is no teacher call in the
inference entry point.

Both the training and evaluation use frozen coeval density, initially neutral
hydrogen, native cumulative ionizing photon counts and the recorded finite
periodic source domain. The support tested here is the original physical
parameter set, DIM=96, and (BOX_LEN, HII_DIM) of (50,16), (50,32) or (256,32).
The coordinate origin is periodic. Even grids preserve the teacher's
signed half-box range [-N/2, N/2-1] at the terminal boundary.

On 42 held-out snapshots from six independent seeds, covering seven redshifts
from 6 to 10, the pure emulator met the previously fixed field, budget and
power criteria throughout. The previous emulator passed 27/42 on the same
labels. The mean field MAE fell from 0.01870 to 0.001050. The largest active
per-snapshot mean shell power error below half Nyquist was 1.83%. The largest
late low-k complex brightness residual was 0.682% of the fully neutral RMS.
The maximum independently audited global photon residual was 4.06e-17.
Thresholded geometric first-exit CDF discrepancies were at most 0.0262
(ionized) and 0.0279 (neutral), when both phases were present. At z=6
in seed2501 the teacher retained one neutral voxel at threshold 0.5 and
the emulator predicted none.

## Use the released model

From the repository root, with NumPy, SciPy and a C++17 compiler:

```bash
python -m research.pc_analytic_emulator.exposure_emulator \
    --model /path/to/frozen_model \
    --snapshot /path/to/native/seed2501/z7p5 \
    --redshift 7.5 --box-len 50 \
    --output /path/to/new_prediction
```

The input folder contains receipt-backed `native.json` and `source.npz`.
The saved `fields.npz` includes both fractions and the predicted reservoir.
The report audits its photon inventory after reading back the written arrays.
The model binds the numerical implementation and checks native parameter
compatibility with its training data.

## Train and test

Generate archived teacher campaigns with `research.pc_analytic.campaign`.
The following recipe uses the distributed, independently audited feature
cache and the portable onset training manifest. The cached features can
also be generated with `exposure_train prepare --help` from the original
teacher campaigns.

```bash
python -m research.pc_analytic_emulator.exposure_train fit \
    --cache /path/to/feature_cache26 --output /path/to/field_fit
python -m research.pc_analytic_emulator.exposure_onset_fit \
    --manifest /path/to/onset_training/training_manifest.json \
    --sample-cells 3072 --steps 200 --output /path/to/onset.npz
python -m research.pc_analytic_emulator.exposure_assemble \
    --base /path/to/field_fit --onset /path/to/onset.npz --depth 16 \
    --output /path/to/frozen_model
python -m research.pc_analytic_emulator.exposure_evaluate \
    --model /path/to/frozen_model \
    --teacher /path/to/original /path/to/resolution /path/to/volume \
    --workers 4 --mfp-rays 4096 --output /path/to/independent_holdout
```

The onset fit requires scikit-learn. Its serialized inference uses numeric
arrays. Training uses 89 observed onset histories and 35 offline
continuous-exposure targets from 31 development seeds. The spatial and
reservoir heads use the historical training set plus 60 new overlap
snapshots. Use separate realization seeds in every validation and test
cohort. The evaluator checks those splits and produces a report for each
snapshot before aggregating fixed-threshold field, power, budget and
geometric morphology metrics.
